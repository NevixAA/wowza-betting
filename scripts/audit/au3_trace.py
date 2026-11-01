"""AU3 Phase 4 step 1: join every first-sight O/U 2.5 ledger row to the predictions.csv row committed in
the SAME commit (the predict run that first logged it). Gives, per tip, the model probability, BOTH
sides' prices, the computed edges, the chosen side and the serving recipe exactly as production saw
them - the row trace match -> features -> probability -> price -> side -> ledger.

Writes reports/au3_data/trace_ou25.csv. Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au3_common import DATA, git_show, heavy_lock, read_csv_text  # noqa: E402

KEEP = ["league", "date", "kickoff_utc", "home_team", "away_team", "p_over25", "odds_over25", "odds_under25",
        "best_over25", "best_under25", "implied_prob_over", "implied_prob_under", "bookmaker_overround",
        "impl_prob_over", "impl_prob_under", "edge_over", "edge_under", "best_edge", "best_side", "bet",
        "signal_tier", "tier_over", "tier_under", "model_type", "serving_recipe", "serving_no_form", "no_form_data",
        "home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5",
        "drift_signal", "both_losing", "min_odds_blocked", "max_odds_blocked", "generated_at", "model_version",
        "model_sha", "odds_snapshots", "api_overround", "p_over25_poisson_dc", "league_avg_goals"]


def main():
    fs = pd.read_csv(DATA / "settled_ou25.csv", dtype=str, low_memory=False)
    fs = fs[fs["side"].isin(["OVER", "UNDER"])]
    rows = []
    commits = fs["fs_commit"].unique()
    for i, sha in enumerate(commits):
        txt = git_show(sha, "output/predictions.csv")
        bt = git_show(sha, "output/bets.csv")
        sub = fs[fs["fs_commit"] == sha]
        P = read_csv_text(txt) if txt else pd.DataFrame()
        B = read_csv_text(bt) if bt else pd.DataFrame()
        for r in sub.itertuples(index=False):
            rec = {"match_date": r.match_date, "home_team": r.home_team, "away_team": r.away_team, "side": r.side,
                   "pred_found": False, "bets_found": False}
            for src, F, tag in (("pred", P, "p_"), ("bets", B, "b_")):
                if F.empty or "home_team" not in F.columns:
                    continue
                m = F[(F["home_team"] == r.home_team) & (F["away_team"] == r.away_team)
                      & (F.get("date", pd.Series("", index=F.index)).astype(str).str[:10] == r.match_date)]
                if m.empty:
                    continue
                rec[f"{src}_found"] = True
                x = m.iloc[-1]
                for c in KEEP:
                    if c in F.columns and c not in ("home_team", "away_team"):
                        rec[tag + c] = x[c]
            rows.append(rec)
        if i % 100 == 0:
            print(i, len(commits), flush=True)
    tr = pd.DataFrame(rows)
    out = fs.merge(tr, on=["match_date", "home_team", "away_team", "side"], how="left")
    out.to_csv(DATA / "trace_ou25.csv", index=False)
    print("pred found", out["pred_found"].mean(), "bets found", out["bets_found"].mean())


if __name__ == "__main__":
    with heavy_lock("AU3 trace predictions at first-sight commits"):
        main()
