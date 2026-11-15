"""AU2 - serving-time HISTORY LAG on the audited board (Phase 2B pre-kickoff availability). READ-ONLY.

Training features for a fixture see EVERY earlier match. At serving time the history is whatever
football-data had published when predict ran. This measures, for the audited board
(output/predictions.csv, 2026-11-01T10:59:39Z), how many Standard board teams had already PLAYED a
league match that is absent from the history the board was scored on.

Evidence of "already played": the fixture appeared on an earlier committed predictions board with a
kickoff before the audited board's generation time (boards since 2026-10-20), and its date is after
that league's last history date.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, AUDITED_COMMIT, load_history_offline  # noqa: E402

import pandas as pd  # noqa: E402

import config  # noqa: E402


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def main() -> int:
    board = pd.read_csv(ROOT / "output/predictions.csv", low_memory=False)
    gen = pd.Timestamp(board["generated_at"].iloc[0])
    std = board[board["model_type"] == "standard"]
    raw = load_history_offline()
    raw = raw[raw["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    hmax = raw.groupby("league")["date"].max()
    seen = []
    for line in filter(None, git("log", "--since=2026-10-20", "--format=%H", AUDITED_COMMIT, "--",
                                 "output/predictions.csv").split("\n")):
        try:
            p = pd.read_csv(io.StringIO(git("show", f"{line}:output/predictions.csv")),
                            usecols=["league", "date", "kickoff_utc", "home_team", "away_team", "model_type"])
            seen.append(p[p["model_type"] == "standard"])
        except Exception:                                              # noqa: BLE001
            continue
    fx = pd.concat(seen).drop_duplicates(["league", "date", "home_team", "away_team"])
    fx["ko"] = pd.to_datetime(fx["kickoff_utc"], utc=True, errors="coerce")
    played = fx[(fx["ko"] < gen) & (pd.to_datetime(fx["date"]) > fx["league"].map(hmax))]
    teams_missing = {}
    for r in played.itertuples():
        for t in (r.home_team, r.away_team):
            teams_missing.setdefault((r.league, t), []).append(str(r.date))
    rows = []
    for r in std.itertuples():
        mh = teams_missing.get((r.league, r.home_team), [])
        ma = teams_missing.get((r.league, r.away_team), [])
        rows.append({"league": r.league, "date": r.date, "home": r.home_team, "away": r.away_team,
                     "history_max": str(hmax.get(r.league))[:10], "home_unrecorded_matches": len(mh),
                     "away_unrecorded_matches": len(ma)})
    df = pd.DataFrame(rows)
    out = {"audited_commit": AUDITED_COMMIT, "board_generated_at": str(gen),
           "standard_board_rows": int(len(df)),
           "history_max_date_by_league": {k: str(v)[:10] for k, v in hmax.items()},
           "rows_with_any_team_missing_a_played_match": int(((df["home_unrecorded_matches"] > 0) | (df["away_unrecorded_matches"] > 0)).sum()),
           "rows_both_teams_missing": int(((df["home_unrecorded_matches"] > 0) & (df["away_unrecorded_matches"] > 0)).sum()),
           "played_fixtures_absent_from_history": int(len(played)),
           "by_league": df.assign(any=(df["home_unrecorded_matches"] > 0) | (df["away_unrecorded_matches"] > 0))
                          .groupby("league")["any"].agg(["sum", "size"]).to_dict("index")}
    df.to_csv(DATA / "history_lag_board.csv", index=False)
    (DATA / "history_lag.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
