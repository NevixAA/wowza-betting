"""
collect_player_profiles.py - API-Football /players?id&season for every (player_id, season) in
player_history.parquet without a stored profile (owner decision C/D, 2026-11-02).

    python scripts/collect_player_profiles.py [--limit N] [--dry-run]

OUTPUT output/collect/player_profiles.parquet, one row per (player_id, season):
  status            'ok' (the provider returned the player) or 'empty' (it returned nothing: a real
                    answer, recorded so it is not asked again)
  height_cm, weight_kg, birth_date   player.height / weight / birth.date as the provider gives
                    them; unparseable or absent -> NaN / None (never a placeholder)
  penalty_scored, penalty_missed     the season's statistics[].penalty, summed over the entries
                    (competitions / clubs) where BOTH are reported; NaN when no entry reports both
  pen_entries       how many entries carried both (0 -> the season's penalty record is unknown)
The parquet is the journal: a stored key is never fetched again; a failed request stores nothing.

INVALID KEYS ARE NEVER SENT (owner 2026-11-04, fix round 9). Run-day 3 of the collection left 1,522
pairs the provider rejects ('The Season field cannot be empty' / 'The Id field cannot be 0'); a
rejection stores nothing, so they were asked again every run. They came from our own rows:
  * season blank: 3,423 player_history rows of national-team football (Friendlies, World Cup
    qualifiers, Africa Cup of Nations; 139 fixtures, 2025-10-10..2026-06-10) written by
    `player_model.pipeline --mode collect-wc` (data_fetcher.collect_national_team_history), whose
    row meta had no season, so the parser wrote "". Fixed at the source (data_fetcher.
    national_row_season);
  * player_id 0: 256 rows of 153 players written by the league collect before 2026-11-02, when
    data_fetcher._stable_player_id still accepted the provider's id 0 (fixed then, fa3c758d).
A key is checked BEFORE any call (`key_problem`): the player id must be a provider id > 0, and the
season is the row's stored season or, when that is blank, src.season_resolver on the row's league and
match date. A key that fails is skipped and counted by reason - never sent, never retried with a
substitute value. `--invalid-keys-out` lists the history rows behind them (beside the history,
never in it).
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import (COLLECT_DIR, FLUSH_EVERY, AFClient, BudgetStop,  # noqa: E402
                            append_parquet, history_frame, read_parquet)

OUT = COLLECT_DIR / "player_profiles.parquet"
COLS = ["player_id", "season", "status", "height_cm", "weight_kg", "birth_date",
        "penalty_scored", "penalty_missed", "pen_entries", "fetched_at"]
_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def _measure(v) -> float:
    """'185 cm' / '185' / 185 -> 185.0; None / '' / no number -> NaN."""
    if v is None:
        return float("nan")
    m = _NUM.search(str(v))
    return float(m.group(1)) if m else float("nan")


def parse(body: dict | None, player_id: int, season: str) -> dict:
    resp = (body or {}).get("response") or []
    row = {"player_id": int(player_id), "season": str(season), "status": "empty",
           "height_cm": float("nan"), "weight_kg": float("nan"), "birth_date": None,
           "penalty_scored": float("nan"), "penalty_missed": float("nan"), "pen_entries": 0,
           "fetched_at": datetime.now(timezone.utc).isoformat()}
    if not resp:
        return row
    p = resp[0].get("player") or {}
    row["status"] = "ok"
    row["height_cm"] = _measure(p.get("height"))
    row["weight_kg"] = _measure(p.get("weight"))
    bd = (p.get("birth") or {}).get("date")
    row["birth_date"] = str(bd)[:10] if bd else None
    sc = ms = 0
    n = 0
    for st in resp[0].get("statistics") or []:
        pen = st.get("penalty") or {}
        s, m = pen.get("scored"), pen.get("missed")
        if isinstance(s, (int, float)) and isinstance(m, (int, float)):
            sc += s
            ms += m
            n += 1
    if n:
        row["penalty_scored"], row["penalty_missed"] = float(sc), float(ms)
    row["pen_entries"] = n
    return row


PROBLEM_ID, PROBLEM_SEASON = "player_id_invalid", "season_unresolved"


def _season_text(v) -> str | None:
    """A stored season label as the provider's season id ('2025'), or None when blank / not a year."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s.isdigit() else None


def key_frame(h: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per history row: player_id, the season to ask for (`season_key`) and `problem`
    (None | PROBLEM_ID | PROBLEM_SEASON). The season is the stored one, else src.season_resolver on
    the row's league and match date (one resolution per distinct league and day); never a guess."""
    from src.season_resolver import is_known, resolve_api_season
    cols = ["fixture_id", "player_id", "season", "league", "date"]
    h = history_frame(cols) if h is None else h.copy()
    if h.empty:
        return pd.DataFrame(columns=cols + ["season_key", "problem"])
    for c in cols:
        if c not in h.columns:
            h[c] = None
    pid = pd.to_numeric(h["player_id"], errors="coerce")
    stored = h["season"].map(_season_text)
    day = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    need = stored.isna()
    memo: dict = {}
    resolved = pd.Series(None, index=h.index, dtype=object)
    for i, lg, d in zip(h.index[need], h.loc[need, "league"], day[need]):
        k = (lg, d)
        if k not in memo:
            s = resolve_api_season(lg, d) if isinstance(d, str) else None
            memo[k] = s if is_known(s) else None
        resolved[i] = memo[k]
    h["season_key"] = stored.where(stored.notna(), resolved)
    h["problem"] = None
    h.loc[h["season_key"].isna(), "problem"] = PROBLEM_SEASON
    h.loc[~(pid > 0).to_numpy(), "problem"] = PROBLEM_ID          # an invented id outranks the season
    h["player_id"] = pid
    return h


def key_problem(player_id, season) -> str | None:
    """Why (player_id, season) must not be sent to /players, or None when it may be."""
    try:
        ok_id = player_id is not None and int(float(player_id)) > 0
    except (TypeError, ValueError):
        ok_id = False
    if not ok_id:
        return PROBLEM_ID
    return None if _season_text(season) else PROBLEM_SEASON


def plan_keys(h: pd.DataFrame | None = None, have: pd.DataFrame | None = None) -> dict:
    """{'todo': valid keys without a stored profile (newest season first), 'skipped': {reason:
    pairs}, 'skipped_rows': {reason: rows}, 'keys': key_frame()}."""
    kf = key_frame(h)
    have = read_parquet(OUT) if have is None else have
    held = ({(int(p), str(s)) for p, s in zip(have["player_id"], have["season"])} if len(have) else set())
    out = {"skipped": {}, "skipped_rows": {}, "keys": kf}
    bad = kf[kf["problem"].notna()]
    for r, g in bad.groupby("problem"):
        out["skipped_rows"][r] = int(len(g))
        out["skipped"][r] = int(len(g.assign(_s=g["season_key"].fillna(g["season"].astype(str)))
                                      [["player_id", "_s"]].drop_duplicates()))
    ok = kf[kf["problem"].isna()]
    keys = {(int(p), str(s)) for p, s in zip(ok["player_id"], ok["season_key"])} - held
    # newest seasons first: they feed the live features
    out["todo"] = sorted(keys, key=lambda k: (k[1], k[0]), reverse=True)
    return out


def todo_keys() -> list[tuple[int, str]]:
    return plan_keys()["todo"]


def invalid_key_rows(kf: pd.DataFrame) -> pd.DataFrame:
    """The history rows behind the skipped keys (league, date, season as stored, reason)."""
    cols = ["fixture_id", "player_id", "league", "date", "season", "problem"]
    return kf.loc[kf["problem"].notna(), cols].sort_values(["problem", "league", "date"]).reset_index(drop=True)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        invalid_out: Path | None = None) -> dict:
    p = plan_keys()
    keys = p["todo"]
    print(f"[profiles] {len(keys):,} valid (player, season) pair(s) without a stored profile; "
          f"skipped, never sent: {p['skipped']} pair(s) from {p['skipped_rows']} history row(s)")
    if invalid_out is not None:
        rows = invalid_key_rows(p["keys"])
        Path(invalid_out).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(invalid_out, index=False)
        print(f"[profiles] {len(rows):,} history row(s) behind the skipped keys -> {invalid_out}")
    if dry_run or not keys:
        return {"todo": len(keys), "fetched": 0, "skipped": p["skipped"]}
    client = client or AFClient()
    buf, fetched = [], 0
    try:
        for pid, season in keys[: limit or None]:
            if key_problem(pid, season):                  # plan_keys already removed these
                continue
            body = client.get("players", {"id": pid, "season": season})
            if body is None:
                continue
            buf.append(parse(body, pid, season))
            fetched += 1
            if len(buf) >= FLUSH_EVERY:
                append_parquet(OUT, buf, ["player_id", "season"], COLS)
                buf = []
    except BudgetStop as e:
        print(f"[profiles] budget stop: {e} (resumable)")
    finally:
        if buf:
            append_parquet(OUT, buf, ["player_id", "season"], COLS)
        client.flush()
    print(f"[profiles] fetched {fetched:,}; {len(keys) - fetched:,} left")
    return {"todo": len(keys), "fetched": fetched, "skipped": p["skipped"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football player profiles per season")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--invalid-keys-out", type=Path, default=None,
                    help="write the history rows behind the skipped keys to this CSV")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run, invalid_out=a.invalid_keys_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
