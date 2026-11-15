"""Coverage audit of the team-information block (src/team_info_features.py) per track, league and
season: the share of Phase 2 frame rows whose value is available LEAK-FREE at the early tip
(kickoff - 153 h), per source, both clubs. Offline, committed data only.

    python scripts/team_info_coverage.py --out DIR [--frame FEATURE_FRAME.parquet]

Writes DIR/team_info_coverage.csv (one row per track x league x season) and
DIR/team_info_coverage_summary.json (per track and overall). Without --frame the committed frame is
built exactly as the harness builds it (retrain_harness.protocol.load_committed_frame, a few minutes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import team_info_features as TI  # noqa: E402

#: source -> the column whose presence on BOTH sides means the source was available
SOURCES = {"injuries (absence, lineup-weighted)": "info_xi_absent_w",
           "lineups (previous XIs)": "info_xi_continuity",
           "player goals (goal share of absentees)": "info_goal_share_absent",
           "xG (rolling, per team)": "info_xg_for5",
           "coach tenure": "info_coach_tenure_days",
           "transfers": "info_transfers_in_120d",
           "fixture congestion": "info_matches_14d"}


def coverage(feat: pd.DataFrame, blk: pd.DataFrame) -> pd.DataFrame:
    from src.model import track_of
    d = feat[["date", "league"]].copy()
    d["track"] = track_of(feat["league"]).to_numpy()
    d["season"] = feat["season"].astype(str) if "season" in feat.columns else pd.to_datetime(feat["date"]).dt.year.astype(str)
    for name, col in SOURCES.items():
        d[name] = (blk[f"home_{col}"].notna() & blk[f"away_{col}"].notna()).to_numpy()
    g = d.groupby(["track", "league", "season"])
    out = g[list(SOURCES)].mean().round(4)
    out.insert(0, "rows", g.size())
    return out.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", default=None)
    a = ap.parse_args(argv)
    if a.frame:
        feat = pd.read_parquet(a.frame)
    else:
        from src.retrain_harness import protocol as P
        feat, _ = P.load_committed_frame(P.load_config())
    store = TI.load_store(feat)
    blk = TI.info_features(feat, store)
    cov = coverage(feat, blk)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out / "team_info_coverage.csv", index=False)
    summ = {"version": TI.INFO_VERSION, "rows": int(len(feat)), "team_ids": store.info.get("team_ids"),
            "by_track": {}}
    from src.model import track_of
    tr = track_of(feat["league"]).to_numpy()
    for t in ("standard", "new_format", "all"):
        m = np.ones(len(feat), bool) if t == "all" else (tr == t)
        summ["by_track"][t] = {"rows": int(m.sum())} | {
            name: round(float((blk[f"home_{c}"].notna() & blk[f"away_{c}"].notna()).to_numpy()[m].mean()), 4)
            for name, c in SOURCES.items()}
    (out / "team_info_coverage_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
