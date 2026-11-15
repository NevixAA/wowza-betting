"""AU4 Phase 8 - props: model-only vs market-only vs model+market (out-of-time).

Input: output/player_ledger.csv rows with a price (market_odds), a model_prob and a settled
result in {WIN, LOSS} (VOID / DNP / POSTPONED / unsettled excluded - missing outcomes are not
losses). Entry prices are additionally checked against the same-fixture price history
(reports/au4_data/clv_records_reconstructed.csv, from au4_clv_forensic.py): rows whose entry
price is NOT present in that fixture's history are reported separately (namesake / borrowed
prices, fixed in production 2026-10-25 bb385e0e, older rows never rewritten).

Prices are one-sided (no "No"/"Under" for SOT, cards, assists; 146 goals "No" quotes), so a
two-sided de-vig is NOT available. The market probability is therefore the raw implied 1/odds,
and every comparison uses a logistic RECALIBRATION fitted only on earlier months:
    MARKET : y ~ logit(1/odds)            (+ market fixed effects)
    MODEL  : y ~ logit(model_prob)        (+ market fixed effects)
    BOTH   : y ~ logit(1/odds) + logit(model_prob) (+ market fixed effects)
Expanding-window by month (train <= month m-1, test month m), first test month = 3rd month with
data. Metrics on pooled test rows: log-loss, Brier, AUC; delta BOTH-MARKET with fixture-block
bootstrap CI (negative LL delta = the model adds information beyond the price).
Also: conditional calibration (model vs realised inside market-prob bins), tail calibration in
the top model-edge decile, and flat-stake ROI of rows the production tier marked as bets.
Writes reports/au4_data/props_mvm.json and props_mvm_calibration.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au4_common import OUT, DATA, block_bootstrap, heavy_lock, save_csv, save_json  # noqa: E402
from au4_clv_forensic import pnorm  # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def design(d, kind, markets):
    cols = []
    if kind in ("MARKET", "BOTH"):
        cols.append(logit(1 / d["market_odds"].to_numpy()))
    if kind in ("MODEL", "BOTH"):
        cols.append(logit(d["model_prob"].to_numpy()))
    X = np.column_stack(cols) if cols else np.zeros((len(d), 0))
    fe = np.column_stack([(d["market"] == m).to_numpy().astype(float) for m in markets[1:]]) if len(markets) > 1 else np.zeros((len(d), 0))
    return np.column_stack([X, fe]) if X.size or fe.size else np.zeros((len(d), 1))


def walk_forward(d: pd.DataFrame) -> pd.DataFrame:
    months = sorted(d["month"].unique())
    markets = sorted(d["market"].unique())
    preds = []
    for i, m in enumerate(months):
        if i < 2:
            continue
        tr, te = d[d["month"] < m], d[d["month"] == m]
        if tr["y"].nunique() < 2 or te.empty:
            continue
        row = te[["key", "fixture", "y", "market", "model_prob", "market_odds", "tier", "entry_ok"]].copy()
        for kind in ("BASE", "MARKET", "MODEL", "BOTH"):
            Xtr = design(tr, kind, markets) if kind != "BASE" else np.column_stack([(tr["market"] == mk).astype(float) for mk in markets])
            Xte = design(te, kind, markets) if kind != "BASE" else np.column_stack([(te["market"] == mk).astype(float) for mk in markets])
            lr = LogisticRegression(C=1e4, max_iter=2000, fit_intercept=(kind != "BASE"))
            lr.fit(Xtr, tr["y"])
            row[f"p_{kind}"] = lr.predict_proba(Xte)[:, 1]
        preds.append(row)
    return pd.concat(preds, ignore_index=True)


def score(p: pd.DataFrame) -> dict:
    y = p["y"].to_numpy()
    out = {"n": len(p), "fixtures": p["fixture"].nunique(), "base_rate": float(y.mean())}
    for k in ("BASE", "MARKET", "MODEL", "BOTH"):
        q = p[f"p_{k}"].to_numpy()
        out[k] = {"logloss": float(ll(y, q).mean()), "brier": float(((q - y) ** 2).mean()),
                  "auc": float(roc_auc_score(y, q)) if len(np.unique(y)) > 1 else None}
    # raw, uncalibrated
    out["RAW_model_prob"] = {"logloss": float(ll(y, p["model_prob"].to_numpy()).mean()),
                             "brier": float(((p["model_prob"] - y) ** 2).mean()),
                             "auc": float(roc_auc_score(y, p["model_prob"]))}
    out["RAW_implied"] = {"logloss": float(ll(y, 1 / p["market_odds"].to_numpy()).mean()),
                          "brier": float(((1 / p["market_odds"] - y) ** 2).mean()),
                          "auc": float(roc_auc_score(y, 1 / p["market_odds"]))}
    dll = ll(y, p["p_BOTH"].to_numpy()) - ll(y, p["p_MARKET"].to_numpy())
    dbr = (p["p_BOTH"] - y) ** 2 - (p["p_MARKET"] - y) ** 2
    out["delta_BOTH_minus_MARKET_ll"] = block_bootstrap(dll, p["fixture"])
    out["delta_BOTH_minus_MARKET_brier"] = block_bootstrap(dbr.to_numpy(), p["fixture"])
    dmm = ll(y, p["p_MODEL"].to_numpy()) - ll(y, p["p_MARKET"].to_numpy())
    out["delta_MODEL_minus_MARKET_ll"] = block_bootstrap(dmm, p["fixture"])
    return out


def main() -> None:
    l = pd.read_csv(OUT / "player_ledger.csv", low_memory=False)
    l["market_odds"] = pd.to_numeric(l["market_odds"], errors="coerce")
    l["model_prob"] = pd.to_numeric(l["model_prob"], errors="coerce")
    d = l[l["market_odds"].gt(1.0) & l["model_prob"].between(0, 1) & l["result"].isin(["WIN", "LOSS"])].copy()
    n_priced = int(l["market_odds"].gt(1.0).sum())
    d["y"] = (d["result"] == "WIN").astype(int)
    d["month"] = d["match_date"].astype(str).str[:7]
    d["fixture"] = d["home_team"].astype(str) + " vs " + d["away_team"].astype(str) + "|" + d["match_date"].astype(str)
    d["key"] = "PLAYER|" + d["match_date"].astype(str).str[:10] + "|" + d["player_name"].astype(str) + "|" + d["market"].astype(str)
    # entry-price verification from the CLV reconstruction (same date/player/market)
    rc = pd.read_csv(DATA / "clv_records_reconstructed.csv", usecols=["bet_id", "odds_bet", "entry_seen_ever", "hist_status"])
    rc["k2"] = rc["bet_id"].map(lambda s: "|".join([s.split("|")[1], pnorm(s.split("|")[2]), s.split("|")[3]]))
    d["k2"] = d["match_date"].astype(str).str[:10] + "|" + d["player_name"].map(pnorm) + "|" + d["market"].astype(str)
    m = rc.drop_duplicates("k2").set_index("k2")
    d["entry_seen_ever"] = d["k2"].map(m["entry_seen_ever"])
    d["entry_ok"] = d["entry_seen_ever"].map({True: "VERIFIED", False: "NOT_IN_FIXTURE_HISTORY"}).fillna("UNVERIFIABLE")
    d["pre_fix"] = d["match_date"].astype(str) < "2026-10-26"
    res = {"n_priced_rows": n_priced, "n_settled_priced_model": len(d),
           "entry_ok_counts": d["entry_ok"].value_counts().to_dict(),
           "markets": d["market"].value_counts().to_dict(), "months": d["month"].value_counts().sort_index().to_dict()}
    p = walk_forward(d)
    res["oot_all"] = score(p)
    # UPPER BOUND (not chronological): 5-fold cross-fit grouped by fixture, per market.
    from sklearn.model_selection import GroupKFold
    cv = {}
    for mk, g in d.groupby("market"):
        if len(g) < 150 or g["y"].nunique() < 2:
            continue
        g = g.reset_index(drop=True)
        pr = {k: np.zeros(len(g)) for k in ("MARKET", "MODEL", "BOTH")}
        for tr, te in GroupKFold(5).split(g, groups=g["fixture"]):
            for k in pr:
                X = design(g, k, [mk])
                lr = LogisticRegression(C=1e4, max_iter=2000).fit(X[tr], g["y"].iloc[tr])
                pr[k][te] = lr.predict_proba(X[te])[:, 1]
        y = g["y"].to_numpy()
        cv[mk] = {"n": len(g), **{k: float(ll(y, v).mean()) for k, v in pr.items()},
                  "dLL_BOTH_MARKET": block_bootstrap(ll(y, pr["BOTH"]) - ll(y, pr["MARKET"]), g["fixture"]),
                  "auc": {k: float(roc_auc_score(y, v)) for k, v in pr.items()}}
    res["crossfit_grouped_by_market"] = cv
    print("crossfit", {k: (v["n"], round(v["MARKET"], 4), round(v["BOTH"], 4), [round(x, 4) for x in v["dLL_BOTH_MARKET"][:3]]) for k, v in cv.items()})
    res["oot_verified_entry"] = score(p[p["entry_ok"] == "VERIFIED"])
    res["oot_by_market"] = {mk: score(g) for mk, g in p.groupby("market") if len(g) >= 150 and g["y"].nunique() > 1}
    # conditional calibration: model vs realised within market-prob bins (test rows)
    p["imp"] = 1 / p["market_odds"]
    p["imp_bin"] = pd.qcut(p["imp"], 10, duplicates="drop")
    cal = p.groupby("imp_bin", observed=True).agg(n=("y", "size"), realised=("y", "mean"), implied=("imp", "mean"),
                                                   model=("model_prob", "mean"), p_market_recal=("p_MARKET", "mean"),
                                                   p_both=("p_BOTH", "mean")).reset_index()
    cal["imp_bin"] = cal["imp_bin"].astype(str)
    save_csv(cal, "props_mvm_calibration.csv")
    # tail calibration: top decile of model edge (model_prob - implied)
    p["edge"] = p["model_prob"] - p["imp"]
    top = p[p["edge"] >= p["edge"].quantile(0.9)]
    res["tail_top_edge_decile"] = {"n": len(top), "model_prob": top["model_prob"].mean(), "implied": top["imp"].mean(),
                                   "realised": top["y"].mean(), "p_market_recal": top["p_MARKET"].mean(),
                                   "p_both": top["p_BOTH"].mean(),
                                   "flat_roi": float((np.where(top["y"] == 1, top["market_odds"] - 1, -1)).mean()),
                                   "roi_ci": block_bootstrap(np.where(top["y"] == 1, top["market_odds"] - 1, -1), top["fixture"])}
    topm = p[p["model_prob"] >= p["model_prob"].quantile(0.9)]
    res["tail_top_model_prob_decile"] = {"n": len(topm), "model_prob": topm["model_prob"].mean(), "realised": topm["y"].mean(),
                                         "implied": topm["imp"].mean()}
    # executable ROI by production tier (all settled priced rows, not just OOT) - split by entry check
    d["pnl_flat"] = np.where(d["y"] == 1, d["market_odds"] - 1, -1.0)
    roi = {}
    for (t, e), g in d.groupby(["tier", "entry_ok"]):
        roi[f"{t}|{e}"] = {"n": len(g), "roi": g["pnl_flat"].mean(), "ci": block_bootstrap(g["pnl_flat"], g["fixture"], n_boot=1000)[1:3]}
    bets = d[d["tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"])]
    roi["BET_TIERS_ALL"] = {"n": len(bets), "roi": bets["pnl_flat"].mean(), "ci": block_bootstrap(bets["pnl_flat"], bets["fixture"])[1:3]}
    bv = bets[bets["entry_ok"] == "VERIFIED"]
    roi["BET_TIERS_VERIFIED"] = {"n": len(bv), "roi": bv["pnl_flat"].mean(), "ci": block_bootstrap(bv["pnl_flat"], bv["fixture"])[1:3]}
    roi["ALL_PRICED"] = {"n": len(d), "roi": d["pnl_flat"].mean(), "ci": block_bootstrap(d["pnl_flat"], d["fixture"])[1:3]}
    res["roi"] = roi
    save_csv(p, "props_mvm_oot_predictions.csv")
    save_json(res, "props_mvm.json")
    import json
    print(json.dumps({k: res[k] for k in ("n_priced_rows", "n_settled_priced_model", "entry_ok_counts", "months")}, default=str))
    for k in ("oot_all", "oot_verified_entry"):
        r = res[k]
        print(k, r["n"], {m: r[m] for m in ("BASE", "MARKET", "MODEL", "BOTH", "RAW_model_prob", "RAW_implied")})
        print("   dLL both-market", r["delta_BOTH_minus_MARKET_ll"], "dBrier", r["delta_BOTH_minus_MARKET_brier"], "model-market", r["delta_MODEL_minus_MARKET_ll"])
    for mk, r in res["oot_by_market"].items():
        print(mk, r["n"], "dLL", [round(x, 5) for x in r["delta_BOTH_minus_MARKET_ll"][:3]], "AUC mkt/model/both", r["MARKET"]["auc"], r["MODEL"]["auc"], r["BOTH"]["auc"])
    print("tail", res["tail_top_edge_decile"]); print("roi", {k: (v["n"], round(v["roi"], 4), v["ci"]) for k, v in roi.items()})


if __name__ == "__main__":
    with heavy_lock("AU4 props_mvm"):
        main()
