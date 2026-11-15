"""AU3: all-fixture FIRST-SIGHT panel from the git history of output/predictions.csv.

For every fixture (date, home, away) the panel records the first committed predict row, and separately
the first row in which each market carried a price (O/U 2.5 two-sided; BTTS yes; O1.5; O3.5). This is
the AVAILABILITY universe: what production saw for every fixture, tipped or not, at the moment a price
first existed. Used for blind baselines (back every priced fixture) and availability-bias checks.

Writes reports/au3_data/pred_panel.csv. Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au3_common import DATA, git_versions, heavy_lock, read_csv_text  # noqa: E402

COLS = ["league", "kickoff_utc", "model_type", "p_over25", "odds_over25", "odds_under25", "p_btts", "odds_btts",
        "odds_btts_no", "btts_books", "btts_overround", "p_over15", "odds_over15", "p_over35", "odds_over35",
        "best_side", "best_edge", "signal_tier", "serving_recipe", "model_version", "generated_at"]
MARKETS = {"ou25": ["odds_over25", "odds_under25"], "btts": ["odds_btts"], "over15": ["odds_over15"],
           "over35": ["odds_over35"]}


def main():
    first, firstm = {}, {m: {} for m in MARKETS}
    n = 0
    for sha, ct, txt in git_versions("output/predictions.csv"):
        n += 1
        try:
            d = read_csv_text(txt)
        except Exception:
            continue
        if not {"date", "home_team", "away_team"} <= set(d.columns):
            continue
        d["date"] = d["date"].astype(str).str[:10]
        for c in COLS:
            if c not in d.columns:
                d[c] = None
        for r in d[["date", "home_team", "away_team"] + COLS].to_dict("records"):
            k = (r["date"], r["home_team"], r["away_team"])
            if k not in first:
                first[k] = {**r, "ct": ct, "sha": sha}
            for m, oc in MARKETS.items():
                if k in firstm[m]:
                    continue
                if all(pd.notna(r[c]) and str(r[c]) not in ("", "nan") for c in oc):
                    firstm[m][k] = {f"{m}_{c}": r[c] for c in COLS if c.startswith(("p_", "odds_", "btts_"))} | \
                                   {f"{m}_ct": ct, f"{m}_sha": sha, f"{m}_kickoff": r["kickoff_utc"]}
    P = pd.DataFrame([{**v} for v in first.values()])
    for m in MARKETS:
        M = pd.DataFrame([{"date": k[0], "home_team": k[1], "away_team": k[2], **v} for k, v in firstm[m].items()])
        if len(M):
            P = P.merge(M, on=["date", "home_team", "away_team"], how="left")
    P.to_csv(DATA / "pred_panel.csv", index=False)
    print(n, "versions;", len(P), "fixtures;", {m: len(v) for m, v in firstm.items()})


if __name__ == "__main__":
    with heavy_lock("AU3 predictions.csv first-sight panel"):
        main()
