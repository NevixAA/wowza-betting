"""ONE first-sight freeze for every prospective log (2026-10-25 platform brief, step a2).

    from src.research.first_sight import FirstSightStore
    store = FirstSightStore(path, key=[...], columns=[...], observed_col="prediction_ts",
                            kickoff_col="kickoff_utc")
    store.append(rows)          # -> number of rows appended; existing bytes are never rewritten
    store.integrity()           # duplicate keys, post-kickoff rows, frozen-column digest
    store.verify()              # sidecar digest check (only for stores created with digest=True)

THE RULES, in the order append applies them
  1. the batch is reindexed to `columns` (a missing column becomes NaN, an extra one is dropped)
  2. rows whose key (compared as strings, exactly as stored) is already in the file are dropped:
     EARLIEST ROW WINS, a later run can never revise a logged opinion
  3. rows observed at/after kickoff are REFUSED (when both `observed_col` and `kickoff_col` are set
     and both values parse). A row with no known kickoff is kept, as every existing writer does.
  4. within the batch the first row per key wins
  5. the survivors are APPENDED (pandas to_csv mode="a"); a header is written only to an empty file.
     Bytes already in the file are never touched.
  6. digest=True only: before appending, the frozen-column digest of the rows already in the file is
     checked against the sidecar <file>.first_sight.json; on a mismatch NOTHING is appended (fail
     closed) and the store reports TAMPERED. After appending, the sidecar is updated.

DIGEST. `frozen_digest` hashes the frozen columns (default: every column) of the first n rows, read
as stored strings. Settlement that fills grading-only columns leaves it unchanged; editing a frozen
value changes it. output/system_health.json records (rows, digest) per store every day, so the next
day's check proves the first `rows` rows are unchanged without needing git history.

CONCURRENCY ASSUMPTIONS (documented, not enforced by a lock)
  * ONE WRITER PER FILE AT A TIME. Every current writer runs inside a serialised GitHub Actions
    concurrency group (predict: `group: predict, cancel-in-progress: false`; daily_summary has its own).
    append() is read-keys-then-append, which is not atomic across processes: two simultaneous writers
    could both append the same key.
  * Even then the file stays readable and the freeze stays honest: readers dedupe on the key keeping
    the FIRST row (shadow_experiments.evaluate, promotion_gate, analytics.first_sight in Pro), and
    `integrity()` reports duplicate keys so the breach is visible rather than silent.
  * Partial writes: a crash mid-append can leave a truncated last line. pandas writes the whole batch
    in one call; integrity() would report it as a parse failure.

The store never raises out of append(): a failure logs a warning and returns 0, because every caller
is a recorder that must not be able to stop predict.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".first_sight.json"


def _ts(s: pd.Series) -> pd.Series:
    s = s.where(~(s.isna() | s.astype(str).str.strip().isin(["", "nan", "NaT", "None", "<NA>"])), None)
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, errors="coerce", utc=True)


def frozen_digest(df: pd.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    """sha256 over the frozen columns of the first n rows, as stored strings. Column order fixed by
    `cols` (absent columns hash as an empty marker, so adding a later column cannot fake a match)."""
    d = df if n is None else df.iloc[:n]
    cols = list(cols) if cols is not None else list(d.columns)
    h = hashlib.sha256()
    h.update(str(len(d)).encode())
    for c in cols:
        h.update(b"\x1e" + c.encode("utf-8") + b"\x1d")
        if c in d.columns:
            h.update("\x1f".join(d[c].astype(str).tolist()).encode("utf-8"))
        else:
            h.update(b"<absent>")
    return h.hexdigest()


class FirstSightStore:
    def __init__(self, path, key, columns, observed_col: str | None = None,
                 kickoff_col: str | None = None, frozen_cols=None, digest: bool = False,
                 lineterminator: str | None = "\n"):
        self.path = Path(path)
        self.key = list(key)
        self.columns = list(columns)
        self.observed_col = observed_col
        self.kickoff_col = kickoff_col
        self.frozen_cols = list(frozen_cols) if frozen_cols is not None else list(columns)
        self.digest = digest
        self.lineterminator = lineterminator
        missing = [k for k in self.key if k not in self.columns]
        if missing:
            raise ValueError(f"key column(s) {missing} not in columns")

    # ── read side ────────────────────────────────────────────────────────────────────────────
    @property
    def sidecar(self) -> Path:
        return self.path.with_name(self.path.name + SIDECAR_SUFFIX)

    def _has_rows(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def read(self) -> pd.DataFrame:
        if not self._has_rows():
            return pd.DataFrame(columns=self.columns)
        return pd.read_csv(self.path, dtype=str, keep_default_na=False)

    def seen_keys(self) -> set:
        if not self._has_rows():
            return set()
        old = pd.read_csv(self.path, dtype=str, keep_default_na=False, usecols=self.key)
        return set(map(tuple, old[self.key].values))

    def post_kickoff_mask(self, d: pd.DataFrame) -> pd.Series:
        if not (self.observed_col and self.kickoff_col) or d.empty \
                or self.observed_col not in d.columns or self.kickoff_col not in d.columns:
            return pd.Series(False, index=d.index)
        o, k = _ts(d[self.observed_col]), _ts(d[self.kickoff_col])
        return o.notna() & k.notna() & (o >= k)

    def integrity(self, prior: dict | None = None) -> dict:
        """Health of the stored file. `prior` = {"rows": n, "frozen_digest": h} from an earlier check:
        the first n rows must still hash to h (append-only for the frozen columns)."""
        out = {"path": self.path.as_posix(), "present": self.path.exists(), "rows": 0,
               "duplicate_keys": 0, "post_kickoff_rows": 0, "frozen_digest": None,
               "prior_rows": None, "prior_match": None, "status": "OK", "reasons": []}
        if not self._has_rows():
            out["status"] = "EMPTY"
            return out
        try:
            d = self.read()
        except Exception as e:                                         # noqa: BLE001
            out.update(status="UNREADABLE", reasons=[f"{type(e).__name__}: {e}"[:200]])
            return out
        out["rows"] = int(len(d))
        miss = [k for k in self.key if k not in d.columns]
        if miss:
            out["reasons"].append(f"key column(s) missing: {miss}")
        else:
            out["duplicate_keys"] = int(d[self.key].astype(str).duplicated(keep="first").sum())
            if out["duplicate_keys"]:
                out["reasons"].append(f"{out['duplicate_keys']} duplicate key row(s): earliest-wins breached")
        out["post_kickoff_rows"] = int(self.post_kickoff_mask(d).sum())
        if out["post_kickoff_rows"]:
            out["reasons"].append(f"{out['post_kickoff_rows']} row(s) observed at/after kickoff")
        out["frozen_digest"] = frozen_digest(d, self.frozen_cols)
        if prior and prior.get("rows") is not None and prior.get("frozen_digest"):
            n = int(prior["rows"])
            out["prior_rows"] = n
            out["prior_match"] = (len(d) >= n and frozen_digest(d, self.frozen_cols, n) == prior["frozen_digest"])
            if not out["prior_match"]:
                out["reasons"].append(f"first {n} row(s) no longer match the recorded frozen digest "
                                      f"(rows {n} -> {len(d)}): a frozen value was rewritten or rows removed")
        if out["reasons"]:
            out["status"] = "TAMPERED" if out["prior_match"] is False else "WARN"
        return out

    def verify(self) -> dict:
        """Check the sidecar (digest=True stores). {'ok': bool|None, 'reason': str}."""
        if not self.sidecar.exists():
            return {"ok": None, "reason": "no sidecar"}
        try:
            sc = json.loads(self.sidecar.read_text(encoding="utf-8"))
            d = self.read()
            n = int(sc["rows"])
            ok = len(d) >= n and frozen_digest(d, sc.get("frozen_cols", self.frozen_cols), n) == sc["frozen_digest"]
            return {"ok": bool(ok), "reason": "" if ok else f"first {n} rows changed since the last append"}
        except Exception as e:                                         # noqa: BLE001
            return {"ok": False, "reason": f"sidecar unreadable: {type(e).__name__}: {e}"[:200]}

    def _write_sidecar(self) -> None:
        d = self.read()
        self.sidecar.write_text(json.dumps({
            "file": self.path.name, "key": self.key, "frozen_cols": self.frozen_cols, "rows": int(len(d)),
            "frozen_digest": frozen_digest(d, self.frozen_cols),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=1) + "\n",
            encoding="utf-8")

    # ── write side ───────────────────────────────────────────────────────────────────────────
    def select_new(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Rules 1-4: the rows append() would write. Pure w.r.t. the file (reads keys only)."""
        new = rows.reindex(columns=self.columns)
        if self._has_rows():
            seen = self.seen_keys()
            new = new[[tuple(map(str, k)) not in seen for k in new[self.key].astype(str).values]]
        late = self.post_kickoff_mask(new)
        if late.any():
            log.warning(f"[first_sight] {self.path.name}: refused {int(late.sum())} row(s) observed at/after kickoff")
            new = new[~late]
        return new.drop_duplicates(self.key, keep="first")

    def append_rows(self, rows: pd.DataFrame | None) -> pd.DataFrame:
        """Append and return exactly the rows written (empty frame when nothing was). Never raises."""
        empty = pd.DataFrame(columns=self.columns)
        try:
            if rows is None or rows.empty:
                return empty
            if self.digest and self.sidecar.exists():
                v = self.verify()
                if v["ok"] is False:
                    log.error(f"[first_sight] {self.path.name}: TAMPERED ({v['reason']}) - nothing appended")
                    return empty
            new = self.select_new(rows)
            if new.empty:
                return empty
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = not self._has_rows()
            kw = {"lineterminator": self.lineterminator} if self.lineterminator is not None else {}
            new.to_csv(self.path, mode="a", header=header, index=False, **kw)
            if self.digest:
                self._write_sidecar()
            return new
        except Exception as e:                                         # noqa: BLE001 - recorder
            log.warning(f"[first_sight] {self.path.name}: append failed ({type(e).__name__}: {e})")
            return empty

    def append(self, rows: pd.DataFrame | None) -> int:
        return int(len(self.append_rows(rows)))
