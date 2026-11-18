"""One-off (2026-11-18): restore the player_history rows the 2026-11-17 and 2026-11-18 collects
rewrote, and keep what they wrote BESIDE the originals.

Evidence (row-level diff of consecutive committed versions of
player_history.parquet/month_2026-11.parquet on main):
  * 7696840eb (11-16 12:14) is the last version before the two rewrites; origin/data-truth-100's
    copy of the part (synced 11-16 20:27) is byte-for-byte the same rows.
  * 018ff0ebe (11-17) added 25 rows and rewrote 91 `referee` cells on 3 Brazil fixtures
    (1492440, 1492447, 1492449: "Ramon Abatti Abel, Brazil" -> "R. Abatti", ...) and derived
    features (referee_yellows_pg, referee_strictness, card_clash_index, opp_strength_index, ...)
    on rows of 110+ fixtures; fb1cbefc9 (11-18) rewrote team_corners_pg / defender_sot_role_index.

The restore, under keep-first (history_integrity.merge_history):
  * every row that existed at BASE takes its BASE values back, exactly;
  * every row added later keeps the values it was FIRST committed with;
  * every cell the restore changes is written to output/corrections/ with the value being
    reverted and the commit that wrote it, so nothing is lost.

    python scripts/oneoff_restore_player_history_2026_11.py            # dry run: counts only
    python scripts/oneoff_restore_player_history_2026_11.py --apply

Reads git objects only (no API call).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import history_integrity as HI  # noqa: E402
from player_model import pipeline as P  # noqa: E402

PART = "player_history.parquet/month_2026-11.parquet"
BASE = "7696840eb"                       # last version before the rewrites
REWRITES = ["018ff0ebe", "fb1cbefc9"]    # the collects that rewrote committed rows, in order
OUT_DIR = ROOT / "output" / "corrections"
OUT = OUT_DIR / "player_history_rewrites_reverted_2026-11.parquet"
AUDIT = OUT_DIR / "player_history_rewrites_reverted_2026-11_audit.json"
KEY = HI.KEY


def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], capture_output=True, check=True, cwd=ROOT)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def _part(rev: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_git("show", f"{rev}:{PART}", binary=True)))


def _ne(a: pd.Series, b: pd.Series) -> pd.Series:
    return ~((a == b) | (a.isna() & b.isna()))


def first_committed(base: pd.DataFrame, versions: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.Series]:
    """Each key as FIRST committed: BASE rows, then keys added by each later version."""
    first = base.set_index(KEY)
    src = pd.Series(BASE, index=first.index)
    for rev, v in versions:
        v = v.set_index(KEY)
        add = v.index.difference(first.index)
        if len(add):
            first = pd.concat([first, v.loc[add]])
            src = pd.concat([src, pd.Series(rev, index=add)])
    return first, src


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    base = _part(BASE)
    versions = [(r, _part(r)) for r in REWRITES]
    current = pd.read_parquet(ROOT / PART)
    cur = current.set_index(KEY)
    first, src = first_committed(base, versions)
    assert set(cur.index) == set(first.index), "the current part has keys outside BASE + rewrites"
    assert list(current.columns) == list(base.columns)

    # the cells to revert, with the commit that wrote each reverted value
    want = first.loc[cur.index]
    recs = []
    for c in [c for c in current.columns if c not in KEY]:
        ne = _ne(cur[c], want[c])
        if not ne.any():
            continue
        for k in cur.index[ne.to_numpy()]:
            writer = None
            for rev, v in versions:
                vv = v.set_index(KEY)
                if k in vv.index and not ((vv.at[k, c] == want.at[k, c])
                                          or (pd.isna(vv.at[k, c]) and pd.isna(want.at[k, c]))):
                    writer = rev
                    break
            recs.append({"fixture_id": int(k[0]), "player_id": int(k[1]), "column": c,
                         "restored_value": str(want.at[k, c]), "reverted_value": str(cur.at[k, c]),
                         "restored_from": str(src.at[k]), "reverted_written_by": writer or ""})
    rev_df = pd.DataFrame(recs)
    raw = rev_df[rev_df["column"].isin(HI.RAW_FACT_COLUMNS)] if len(rev_df) else rev_df
    audit = {
        "part": PART, "base": BASE, "rewrites": REWRITES,
        "rows_current": int(len(cur)), "rows_base": int(len(base)),
        "rows_added_after_base_kept": int((src != BASE).sum()),
        "cells_reverted": int(len(rev_df)),
        "rows_touched": int(rev_df[["fixture_id", "player_id"]].drop_duplicates().shape[0]) if len(rev_df) else 0,
        "fixtures_touched": int(rev_df["fixture_id"].nunique()) if len(rev_df) else 0,
        "raw_fact_cells_reverted": raw.groupby("column").size().to_dict() if len(raw) else {},
        "raw_fact_fixtures": sorted(int(x) for x in raw["fixture_id"].unique()) if len(raw) else [],
        "by_column": rev_df.groupby("column").size().sort_values(ascending=False).to_dict() if len(rev_df) else {},
    }
    print(json.dumps(audit, indent=1, default=int))
    if not a.apply:
        print("(dry run; --apply to write)")
        return 0

    # restored part: the current row order, each row as first committed
    restored = want.reset_index()[current.columns]
    for c in current.columns:
        restored[c] = restored[c].astype(current[c].dtype)
    full = pd.read_parquet(ROOT / "player_history.parquet")
    pid = full.set_index(KEY).index
    in_part = pid.isin(cur.index)
    rest = full[~in_part]
    merged = pd.concat([rest, restored], ignore_index=True)
    st = P._save_history(merged, ROOT / "player_history.parquet")
    assert st["written"] == [Path(PART).name], st["written"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rev_df.to_parquet(OUT, index=False)
    AUDIT.write_text(json.dumps(audit, indent=1, default=int) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {PART}, {OUT.relative_to(ROOT)}, {AUDIT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
