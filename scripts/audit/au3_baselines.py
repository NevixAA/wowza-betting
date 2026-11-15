"""AU3: blind baselines on the all-fixture first-sight panel (reports/au3_data/pred_panel.csv).

* O/U 2.5: back OVER (or UNDER) on every fixture at its first two-sided price; tipped vs not tipped.
* NF BTTS: league-stratified comparison tipped vs not tipped (does model selection add inside a league?).
Writes reports/au3_data/baselines.json. Discovery-only numbers; nothing here is a proposed filter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au3_common import DATA, LIVE_FROM, roi_ci  # noqa: E402
from au3_settle import Resolver, load_results  # noqa: E402


def r_(p, dt):
    r = roi_ci(p, dt)
    return {"n": r["n"], "pnl": r["pnl"], "roi_pct": None if not r["n"] else round(100 * r["roi"], 2),
            "ci_day": [None if np.isnan(v) else round(100 * v, 1) for v in r["ci_day"]]}


def main():
    out = {}
    P = pd.read_csv(DATA / "pred_panel.csv", low_memory=False)
    P = P[P.ou25_odds_over25.notna() & P.ou25_odds_under25.notna()].copy()
    R = Resolver(load_results())
    tot = []
    for r in P.itertuples(index=False):
        sc, _ = R.find(r.league, r.date, r.home_team, r.away_team)
        if sc:
            hg, ag = sc["af" if "af" in sc else "fd"]; tot.append(hg + ag)
        else:
            tot.append(np.nan)
    P["tot"] = tot
    P = P[P.tot.notna()]
    oo = pd.to_numeric(P.ou25_odds_over25, errors="coerce"); ou = pd.to_numeric(P.ou25_odds_under25, errors="coerce")
    P["pnl_over"] = np.where(P.tot >= 3, oo - 1, -1.0)
    P["pnl_under"] = np.where(P.tot <= 2, ou - 1, -1.0)
    P["p_over"] = pd.to_numeric(P.ou25_p_over25, errors="coerce")
    P["fair_over"] = (1 / oo) / (1 / oo + 1 / ou)
    P["md"] = pd.to_datetime(P.date)
    T = pd.read_csv(DATA / "phase4_rows.csv", low_memory=False)
    tk = set(zip(T.match_date, T.home_team, T.away_team, T.side))
    P["tip_over"] = [(a, b, c, "OVER") in tk for a, b, c in zip(P.date, P.home_team, P.away_team)]
    P["tip_under"] = [(a, b, c, "UNDER") in tk for a, b, c in zip(P.date, P.home_team, P.away_team)]
    res = []
    for w, x in (("all_live", P[P.md >= "2026-04-26"]), ("since_0901", P[P.md >= LIVE_FROM])):
        for mt, y in list(x.groupby("model_type")) + [("ALL", x)]:
            for side in ("over", "under"):
                tip = y[f"tip_{side}"]
                res.append({"window": w, "track": mt, "side": side.upper(),
                            "blind_all": r_(y[f"pnl_{side}"].values, y.date.values),
                            "tipped": r_(y[tip][f"pnl_{side}"].values, y[tip].date.values),
                            "not_tipped": r_(y[~tip][f"pnl_{side}"].values, y[~tip].date.values)})
            yy = y[y.p_over.notna()]
            res.append({"window": w, "track": mt, "side": "CALIBRATION", "n": int(len(yy)),
                        "mean_p_over_model": round(float(yy.p_over.mean()), 4),
                        "mean_fair_over_market": round(float(yy.fair_over.mean()), 4),
                        "realised_over": round(float((yy.tot >= 3).mean()), 4),
                        "brier_model": round(float(((yy.p_over - (yy.tot >= 3)) ** 2).mean()), 4),
                        "brier_market": round(float(((yy.fair_over - (yy.tot >= 3)) ** 2).mean()), 4)})
    out["ou25_blind"] = res
    # NF BTTS league-stratified tipped vs not tipped
    U = pd.read_csv(DATA / "phase9_nf_btts_universe.csv", low_memory=False)
    U = U[U.pnl.notna()]
    st = []
    for w, x in (("all_live", U), ("since_0901", U[pd.to_datetime(U.date) >= LIVE_FROM])):
        diffs, wts = [], []
        for lg, g in x.groupby("league"):
            a, b = g[g.tipped], g[~g.tipped]
            st.append({"window": w, "league": lg, "tipped": r_(a.pnl.values, a.date.values),
                       "not_tipped": r_(b.pnl.values, b.date.values)})
            if len(a) >= 5 and len(b) >= 5:
                diffs.append(a.pnl.mean() - b.pnl.mean()); wts.append(min(len(a), len(b)))
        st.append({"window": w, "league": "STRATIFIED_DIFF_tipped_minus_not (leagues with >=5 each)",
                   "weighted_diff_pct": round(100 * float(np.average(diffs, weights=wts)), 2) if diffs else None,
                   "leagues": len(diffs)})
    out["nf_btts_stratified"] = st
    json.dump(out, open(DATA / "baselines.json", "w"), indent=1, default=str)
    f = lambda r: f"n {r['n']} {r['pnl']:+.2f}u {r['roi_pct']}% {r['ci_day']}" if r["n"] else "n0"
    for r in res:
        if r["side"] == "CALIBRATION":
            print(r)
        else:
            print(r["window"], r["track"], r["side"], "| blind", f(r["blind_all"]), "| tipped", f(r["tipped"]), "| not", f(r["not_tipped"]))
    for r in st:
        print(r if "weighted_diff_pct" in r else (r["window"], r["league"], f(r["tipped"]), f(r["not_tipped"])))


if __name__ == "__main__":
    main()
