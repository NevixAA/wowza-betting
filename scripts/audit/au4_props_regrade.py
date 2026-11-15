"""AU4 Phase 8 - independent re-grade of settled prop CLV records and the props ledger.

player_model/clv_tracker.settle_results grades clv_records by (player_id, date) then (name, date)
and computes `won` from goals/assists/SOT/cards with `x or 0`. It never reads `minutes`, so a
player who is in player_history with minutes == 0 (unused substitute) is graded LOSS, while a
bookmaker voids "anytime scorer"/"shots on target"/"to be carded" for a non-participant. A NaN
stat (`nan or 0` -> nan) also grades LOSS.

This script re-grades every clv_records row with result in {WIN, LOSS} from player_history:
  * matched by normalised name + date (+-0 days), and the stats row's home/away must be the
    record's fixture (team-token match) - a namesake in another fixture is NOT accepted;
  * minutes == 0 or NaN -> VOID (non-participant), stats NaN -> UNRESOLVED;
  * otherwise WIN/LOSS by the market threshold.
Writes reports/au4_data/props_regrade.csv and props_regrade.json.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au4_common import ROOT, OUT, DATA, block_bootstrap, heavy_lock, save_csv, save_json  # noqa: E402
from au4_clv_forensic import pnorm, same_fixture  # noqa: E402

THRESH = {"goals": ("goals", 1), "goals2": ("goals", 2), "assists": ("assists", 1),
          "sot": ("shots_on_target", 1), "sot2": ("shots_on_target", 2),
          "sot3": ("shots_on_target", 3), "sot4": ("shots_on_target", 4), "cards": ("cards", 1)}


def load_history(dates) -> pd.DataFrame:
    months = sorted({d[:7] for d in dates})
    cols = ["fixture_id", "player_id", "player_name", "date", "home_team", "away_team", "team",
            "minutes", "goals", "assists", "shots_on_target", "yellow_cards", "red_cards"]
    parts = []
    for m in months:
        p = ROOT / "player_history.parquet" / f"month_{m}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=cols))
    h = pd.concat(parts, ignore_index=True)
    h["dkey"] = h["date"].astype(str).str[:10]
    h["pk"] = h["player_name"].map(pnorm)
    h["cards"] = h["yellow_cards"].fillna(0) + h["red_cards"].fillna(0)
    h.loc[h["yellow_cards"].isna() & h["red_cards"].isna(), "cards"] = np.nan
    return h


def regrade(rec: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    idx = {k: g for k, g in h.groupby(["pk", "dkey"])}
    out = []
    for r in rec.itertuples():
        g = idx.get((pnorm(r.player), r.d))
        st, why = "UNRESOLVED", "no_stats_row"
        n_rows = 0 if g is None else len(g)
        if g is not None:
            m = f"{'' if False else ''}"
            gg = g[[same_fixture(r.match, f"{a} vs {b}") for a, b in zip(g["home_team"], g["away_team"])]]
            if gg.empty:
                st, why = "UNRESOLVED", "stats_row_other_fixture"
            else:
                s = gg.iloc[0]
                col, k = THRESH.get(r.market, (None, None))
                if col is None:
                    why = "unknown_market"
                elif pd.isna(s["minutes"]) or float(s["minutes"]) <= 0:
                    st, why = "VOID", "minutes_0_or_nan"
                elif pd.isna(s[col]):
                    st, why = "UNRESOLVED", "stat_nan"
                else:
                    st, why = ("WIN" if float(s[col]) >= k else "LOSS"), "graded"
        out.append({"bet_id": r.bet_id, "au4_result": st, "au4_reason": why, "n_name_date_rows": n_rows})
    return rec.merge(pd.DataFrame(out), on="bet_id", how="left")


def main() -> None:
    c = pd.read_csv(OUT / "clv_records.csv", low_memory=False)
    c["d"] = c["bet_id"].astype(str).str.split("|", n=3).str[1].str[:10]
    st = c[c["result"].isin(["WIN", "LOSS"])].copy()
    h = load_history(st["d"].unique())
    g = regrade(st, h)
    g["odds_bet"] = pd.to_numeric(g["odds_bet"], errors="coerce")
    g["pnl"] = pd.to_numeric(g["pnl"], errors="coerce")
    g["au4_pnl"] = np.where(g["au4_result"] == "WIN", g["odds_bet"] - 1,
                            np.where(g["au4_result"] == "LOSS", -1.0, np.nan))
    g["fixture"] = g["match"].astype(str) + "|" + g["d"]
    save_csv(g, "props_regrade.csv")
    xt = pd.crosstab(g["result"], g["au4_result"])
    graded = g[g["au4_result"].isin(["WIN", "LOSS"])]
    res = {"n_settled_production": len(g), "crosstab": xt.to_dict(),
           "reasons": g["au4_reason"].value_counts().to_dict(),
           "production_roi": block_bootstrap(g["pnl"], g["fixture"]),
           "au4_roi_graded_only": block_bootstrap(graded["au4_pnl"], graded["fixture"]),
           "au4_hit_rate": float((graded["au4_result"] == "WIN").mean()),
           "au4_mean_implied": float((1 / graded["odds_bet"]).mean()),
           "production_hit_rate": float((g["result"] == "WIN").mean()),
           "by_tier": {t: {"n": len(x), "roi": x["au4_pnl"].mean(),
                           "ci": block_bootstrap(x["au4_pnl"], x["fixture"], n_boot=1000)[1:3]}
                       for t, x in graded.groupby(graded["notes"].astype(str))},
           "by_market": {t: {"n": len(x), "roi": x["au4_pnl"].mean(), "hit": (x["au4_result"] == "WIN").mean(),
                             "implied": (1 / x["odds_bet"]).mean()}
                         for t, x in graded.groupby("market")}}
    save_json(res, "props_regrade.json")
    print(xt); print({k: v for k, v in res.items() if k not in ("crosstab",)})


if __name__ == "__main__":
    with heavy_lock("AU4 props_regrade"):
        main()
