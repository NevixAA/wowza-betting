"""AU3 Phase 9: positive-edge candidates (NF BTTS, Brazil and other positive leagues).

Inputs (all produced by the au3_* scripts from origin/main @ AUDIT_REF):
    reports/au3_data/settled_side.csv, settled_ou25.csv / phase4_rows.csv, pred_panel.csv
Writes reports/au3_data/phase9_*.csv + phase9_numbers.json.

Rules applied: flat 1u; first-sight price unless a column says otherwise; independently settled rows
only; never a missing outcome as a loss; every subset reported with its denominator; no new filter is
proposed from these numbers (discovery only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au3_common import DATA, LIVE_FROM, git_show, read_csv_text, roi_ci  # noqa: E402
from au3_settle import Resolver, load_results, outcome  # noqa: E402

TRACE_COLS = ["odds_btts", "odds_btts_no", "btts_books", "btts_overround", "btts_fair_yes", "p_btts", "kickoff_utc",
              "generated_at", "odds_over15", "odds_over35", "p_over15", "p_over35", "model_version_btts"]


def r_(x, dates):
    r = roi_ci(x, dates)
    return {"n": r["n"], "pnl": r["pnl"], "roi_pct": None if not r["n"] else round(100 * r["roi"], 1),
            "ci_day": [None if np.isnan(v) else round(100 * v, 1) for v in r["ci_day"]],
            "ci_iid": [None if np.isnan(v) else round(100 * v, 1) for v in r["ci_iid"]]}


def trace_side(d):
    rows = []
    for sha in d.fs_commit.unique():
        txt = git_show(sha, "output/predictions.csv")
        if not txt:
            continue
        P = read_csv_text(txt)
        P["date"] = P["date"].astype(str).str[:10]
        sub = d[d.fs_commit == sha]
        for r in sub.itertuples(index=False):
            m = P[(P.home_team == r.home_team) & (P.away_team == r.away_team) & (P.date == r.match_date)]
            rec = {"match_date": r.match_date, "home_team": r.home_team, "away_team": r.away_team, "market": r.market,
                   "tr_found": not m.empty}
            if not m.empty:
                x = m.iloc[-1]
                rec.update({"tr_" + c: x[c] for c in TRACE_COLS if c in P.columns})
            rows.append(rec)
    return d.merge(pd.DataFrame(rows), on=["match_date", "home_team", "away_team", "market"], how="left")


def bh(pvals, q=0.10):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(qv, 1)
    return out


def p_one_sided(pnl, n_boot=4000, seed=11):
    """one-sided bootstrap p for mean > 0."""
    pnl = np.asarray(pnl, float)
    if len(pnl) < 5:
        return 1.0
    rng = np.random.default_rng(seed)
    b = np.array([pnl[rng.integers(0, len(pnl), len(pnl))].mean() for _ in range(n_boot)])
    return float((b <= 0).mean())


def main():
    nums = {}
    d = pd.read_csv(DATA / "settled_side.csv", low_memory=False)
    d = d[d.source == "live"].copy()
    d = trace_side(d)
    d["md"] = pd.to_datetime(d.match_date)
    d["tier_fs"] = d.signal_tier; d["tier_ledger"] = d.final_signal_tier
    d["settled"] = d.ind_result.isin(["WIN", "LOSS"])
    d["pnl_fs"] = d.ind_pnl_fs; d["pnl_ledger_price"] = d.ind_pnl_final
    d["edge_fs"] = pd.to_numeric(d.edge_pct, errors="coerce") / 100
    d.to_csv(DATA / "phase9_side_rows.csv", index=False)
    nb = d[(d.market == "btts") & (d.model_type == "new_format")]
    # ---- the claim reproduced on the committed ledger ---------------------------------------------
    led = pd.read_csv(Path(DATA).parents[1] / "output" / "side_bets_ledger.csv", low_memory=False)
    led = led[(led.source == "live") & (led.market == "btts") & (led.model_type == "new_format") & led.result.isin(["WIN", "LOSS"])]
    claim = {}
    for lab, x in (("ledger_tier_S+M_all", led[led.signal_tier.isin(["SNIPER", "MARKSMAN"])]),
                   ("ledger_all_tiers_all", led),
                   ("ledger_tier_S+M_since0901", led[led.signal_tier.isin(["SNIPER", "MARKSMAN"]) & (led.match_date >= LIVE_FROM)]),
                   ("ledger_tier_S+M_to_1031", led[led.signal_tier.isin(["SNIPER", "MARKSMAN"]) & (led.match_date <= "2026-10-31")])):
        claim[lab] = r_(x.pnl.values, x.match_date.values)
    nums["claim_reproduction_committed_ledger"] = claim
    # ---- first-sight / independent variants -------------------------------------------------------
    var = {}
    s = nb[nb.settled]
    for w, x in (("all_live", s), ("since_0901", s[s.md >= LIVE_FROM]), ("pre_0901", s[s.md < LIVE_FROM])):
        for tl, y in (("S+M_ledger_tier", x[x.tier_ledger.isin(["SNIPER", "MARKSMAN"])]),
                      ("S+M_first_sight_tier", x[x.tier_fs.isin(["SNIPER", "MARKSMAN"])]),
                      ("all_tiers", x), ("VALUABLE_first_sight", x[x.tier_fs == "VALUABLE"])):
            var[f"{w}|{tl}|first_sight_price"] = r_(y.pnl_fs.values, y.match_date.values)
            var[f"{w}|{tl}|ledger_price"] = r_(y.pnl_ledger_price.values, y.match_date.values)
    nums["nf_btts_variants_independent_settlement"] = var
    nums["nf_btts_rows"] = {"live_rows": int(len(nb)), "independently_settled": int(nb.settled.sum()),
                            "ledger_settled": int(nb.final_result.isin(["WIN", "LOSS"]).sum()),
                            "unresolved_but_ledger_settled": int((~nb.settled & nb.final_result.isin(["WIN", "LOSS"])).sum()),
                            "odds_changed_after_first_sight": int((nb.odds_f != nb.final_odds_f).sum()),
                            "tier_upgraded_after_first_sight": int((nb.tier_fs != nb.tier_ledger).sum())}
    # ---- concentration -------------------------------------------------------------------------------
    sa = s[s.md >= LIVE_FROM]
    conc = []
    for lab, x in (("all_live", s), ("since_0901", sa)):
        for lg, g in x.groupby("league"):
            conc.append({"window": lab, "league": lg, **r_(g.pnl_fs.values, g.match_date.values),
                         "share_of_bets": round(len(g) / len(x), 3)})
        ex = x[x.league != "Argentina Primera Division"]
        conc.append({"window": lab, "league": "EX-ARGENTINA", **r_(ex.pnl_fs.values, ex.match_date.values),
                     "share_of_bets": round(len(ex) / len(x), 3)})
        exb = x[~x.league.isin(["Argentina Primera Division", "Brazil Serie A"])]
        conc.append({"window": lab, "league": "EX-ARGENTINA-EX-BRAZIL", **r_(exb.pnl_fs.values, exb.match_date.values),
                     "share_of_bets": round(len(exb) / len(x), 3)})
        # top-day dependence
        day = x.groupby("match_date").pnl_fs.sum().sort_values(ascending=False)
        conc.append({"window": lab, "league": "TOP5_DAYS_PROFIT_u", "n": int(len(day)), "pnl": round(float(day.head(5).sum()), 2),
                     "roi_pct": round(100 * (x.pnl_fs.sum() - day.head(5).sum()) / max(1, len(x) - x.match_date.isin(day.head(5).index).sum()), 1),
                     "ci_day": None, "ci_iid": None, "share_of_bets": round(x.match_date.isin(day.head(5).index).mean(), 3)})
    pd.DataFrame(conc).to_csv(DATA / "phase9_concentration.csv", index=False)
    nums["concentration"] = conc
    # odds bands, months, tiers, model version, regime (first sight, independent settlement)
    def split(x, col):
        return [{"key": str(k), **r_(g.pnl_fs.values, g.match_date.values),
                 "hit": round(float((g.ind_result == "WIN").mean()), 3), "mean_odds": round(float(g.odds_f.mean()), 2)}
                for k, g in x.groupby(col, observed=True)]
    s2 = s.assign(odds_band=pd.cut(s.odds_f, [1, 1.8, 2.0, 2.2, 2.5, 99], right=False),
                  month=s.md.dt.to_period("M").astype(str),
                  week=s.md.dt.to_period("W").astype(str),
                  mv=s.final_model_version.fillna("UNKNOWN_LEGACY"),
                  books=pd.to_numeric(s.tr_btts_books, errors="coerce").fillna(-1).astype(int).astype(str).replace("-1", "NA"),
                  two_sided=pd.to_numeric(s.tr_odds_btts_no, errors="coerce").notna(),
                  edge_band=pd.cut(s.edge_fs, [-1, 0.04, 0.06, 0.08, 0.10, 0.15, 9], right=False))
    nums["splits_all_live"] = {c: split(s2, c) for c in ("odds_band", "month", "tier_fs", "tier_ledger", "mv", "books",
                                                         "two_sided", "edge_band")}
    s3 = s2[s2.md >= LIVE_FROM]
    nums["splits_since_0901"] = {c: split(s3, c) for c in ("odds_band", "month", "tier_fs", "mv", "books", "two_sided", "edge_band")}
    nums["weekly_all_live"] = split(s2, "week")
    # price provenance at first sight
    tr = nb[nb.tr_found == True]  # noqa: E712
    ob = pd.to_numeric(tr.tr_odds_btts, errors="coerce")
    nums["price_provenance"] = {
        "traced": int(len(tr)), "of": int(len(nb)),
        "ledger_odds_eq_pred_odds_btts": int(np.isclose(tr.odds_f, ob, atol=0.005).sum()),
        "two_sided_no_price_present": int(pd.to_numeric(tr.tr_odds_btts_no, errors="coerce").notna().sum()),
        "books_field_present": int(pd.to_numeric(tr.tr_btts_books, errors="coerce").notna().sum()),
        "books_distribution": pd.to_numeric(tr.tr_btts_books, errors="coerce").value_counts().sort_index().to_dict(),
        "margin_median (btts_overround is stored as 1/y+1/n-1)": float(pd.to_numeric(tr.tr_btts_overround, errors="coerce").median()),
        "margin_below_0": int((pd.to_numeric(tr.tr_btts_overround, errors="coerce") < 0).sum()),
        "odds_exactly_1.85_fallback": int((nb.odds_f == 1.85).sum()),
    }
    # duplicates / correlated positions
    k = ["match_date", "home_team", "away_team"]
    allside = d.copy()
    nums["same_fixture_multi_side_market"] = int(allside.groupby(k).market.nunique().gt(1).sum())
    ou = pd.read_csv(DATA / "settled_ou25.csv", low_memory=False)
    ou = ou[ou.side.isin(["OVER", "UNDER"])]
    j = nb.merge(ou[k + ["side"]], on=k, how="left")
    nums["nf_btts_with_ou25_tip_same_fixture"] = j.groupby("side", dropna=False).size().rename(lambda x: str(x)).to_dict()
    nums["nf_btts_dup_keys"] = int(nb.duplicated(k).sum())
    # ---- availability universe & blind baseline (pred_panel) -----------------------------------------
    P = pd.read_csv(DATA / "pred_panel.csv", low_memory=False)
    P = P[(P.model_type == "new_format") & P.btts_odds_btts.notna()].copy()
    R = Resolver(load_results())
    res = []
    for r in P.itertuples(index=False):
        sc, why = R.find(r.league, r.date, r.home_team, r.away_team)
        if sc:
            hg, ag = sc["af" if "af" in sc else "fd"]
            res.append(outcome("btts", "YES", hg, ag))
        else:
            res.append("UNRESOLVED")
    P["res"] = res
    P["o"] = pd.to_numeric(P.btts_odds_btts, errors="coerce")
    P["pnl"] = np.where(P.res == "WIN", P.o - 1, np.where(P.res == "LOSS", -1.0, np.nan))
    P["tipped"] = P.set_index(["date", "home_team", "away_team"]).index.isin(nb.set_index(k).index)
    P["md"] = pd.to_datetime(P.date)
    P.to_csv(DATA / "phase9_nf_btts_universe.csv", index=False)
    base = []
    for w, x in (("all_live", P[P.md >= "2026-08-09"]), ("since_0901", P[P.md >= LIVE_FROM])):
        x = x[x.pnl.notna()]
        for lab, y in (("BLIND_YES_all_priced", x), ("tipped", x[x.tipped]), ("not_tipped", x[~x.tipped])):
            base.append({"window": w, "population": lab, **r_(y.pnl.values, y.date.values)})
        for lg, g in x.groupby("league"):
            base.append({"window": w, "population": f"BLIND_YES|{lg}", **r_(g.pnl.values, g.date.values),
                         "tipped_share": round(float(g.tipped.mean()), 3)})
    nums["availability_and_blind_baseline"] = base
    nums["availability"] = {"nf_fixtures_in_panel": int((pd.read_csv(DATA / "pred_panel.csv", low_memory=False).model_type == "new_format").sum()),
                            "nf_fixtures_btts_priced": int(len(P))}
    # ---- per-league positive cells across all markets, since 0901, with BH ------------------------------
    o4 = pd.read_csv(DATA / "phase4_rows.csv", low_memory=False)
    o4 = o4[o4.settled & (pd.to_datetime(o4.match_date) >= LIVE_FROM)]
    cells = []
    for (lg, mt, sd), g in o4.groupby(["league", "model_type", "side"]):
        cells.append(("ou25_" + sd.lower(), mt, lg, g.pnl.values, g.match_date.values))
    for (lg, mt, mk), g in sa.groupby(["league", "model_type", "market"]):
        cells.append((mk, mt, lg, g.pnl_fs.values, g.match_date.values))
    sall = d[d.settled & (d.md >= LIVE_FROM)]
    for (lg, mt, mk), g in sall[sall.market != "btts"].groupby(["league", "model_type", "market"]):
        cells.append((mk, mt, lg, g.pnl_fs.values, g.match_date.values))
    for (lg, mt), g in sall[(sall.market == "btts") & (sall.model_type == "standard")].groupby(["league", "model_type"]):
        cells.append(("btts", mt, lg, g.pnl_fs.values, g.match_date.values))
    C = pd.DataFrame([{"market": a, "track": b, "league": c, **r_(p, dt), "p_one_sided": p_one_sided(p)}
                      for a, b, c, p, dt in cells])
    C = C.drop_duplicates(["market", "track", "league"])
    C["bh_q"] = bh(C.p_one_sided.values)
    C = C.sort_values("p_one_sided")
    C.to_csv(DATA / "phase9_league_cells_since0901.csv", index=False)
    nums["league_cells_tested"] = int(len(C))
    nums["league_cells_positive_point"] = int((C.roi_pct > 0).sum())
    nums["league_cells_bh_q_le_0.10"] = C[C.bh_q <= 0.10][["market", "track", "league", "n", "roi_pct", "bh_q"]].to_dict("records")
    nums["top_cells"] = C.head(12)[["market", "track", "league", "n", "pnl", "roi_pct", "ci_day", "p_one_sided", "bh_q"]].to_dict("records")
    json.dump(nums, open(DATA / "phase9_numbers.json", "w"), indent=1, default=str)
    print(json.dumps({k: nums[k] for k in ("claim_reproduction_committed_ledger", "nf_btts_rows", "price_provenance",
                                           "league_cells_tested", "league_cells_bh_q_le_0.10")}, indent=1, default=str))


if __name__ == "__main__":
    main()
