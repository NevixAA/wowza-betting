"""AU3 Phase 4: UNDER 2.5 forensic investigation on immutable first-sight O/U 2.5 tips.

Input: reports/au3_data/trace_ou25.csv (au3_first_sight -> au3_settle -> au3_trace).
Output: reports/au3_data/phase4_*.csv and phase4_numbers.json, printed summary.

ROI convention: flat 1u at the first-sight price; WIN = odds-1, LOSS = -1; only rows settled by the
INDEPENDENT result (au3_settle) count; UNRESOLVED rows are excluded, never counted as losses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au3_common import DATA, LIVE_FROM, fmt_roi, roi_ci  # noqa: E402

# Threshold regimes (git history of predict.yml env and models/best_params_*.json, origin/main):
REGIMES = [("T0_config_defaults", "2000-01-01"), ("T1_league_opt_0726", "2026-07-26 11:24"),
           ("T2_bt_0802", "2026-08-02 12:12"), ("T3_env_widened_0821", "2026-08-21 16:06"),
           ("T4_bt_0901", "2026-09-01 09:43"), ("T5_bt_1001", "2026-10-01 11:34"), ("T6_bt_1006", "2026-10-06 11:26")]
SERVING_SWITCH = pd.Timestamp("2026-10-28 12:38")   # predict.yml f2761eba7 (14:38 +02:00)


def regime(ts):
    out = REGIMES[0][0]
    for name, start in REGIMES:
        if ts >= pd.Timestamp(start):
            out = name
    return out


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load():
    d = pd.read_csv(DATA / "trace_ou25.csv", low_memory=False)
    d["fs_time"] = pd.to_datetime(d["fs_commit_ct"], unit="s")
    d["md"] = pd.to_datetime(d["match_date"])
    d["odds"] = num(d["odds"])
    d["edge_fs"] = num(d["edge_pct"]) / 100
    d["edge_final"] = num(d["final_edge_pct"]) / 100
    d["tier_fs"] = d["signal_tier"].replace({"VALUE": "VALUABLE"})
    d["tier_final"] = d["final_signal_tier"].replace({"VALUE": "VALUABLE"})
    d["p_over"] = num(d["p_p_over25"])
    d["oo"] = num(d["p_odds_over25"]); d["ou"] = num(d["p_odds_under25"])
    d["p_side"] = np.where(d.side == "OVER", d.p_over, 1 - d.p_over)
    d["odds_pred_side"] = np.where(d.side == "OVER", d.oo, d.ou)
    d["odds_pred_other"] = np.where(d.side == "OVER", d.ou, d.oo)
    d["overround"] = 1 / d.oo + 1 / d.ou
    d["p_mkt_fair_side"] = (1 / d.odds_pred_side) / d.overround
    d["edge_recomputed"] = d.p_side - 1 / d.odds
    d["pred_edge_side"] = np.where(d.side == "OVER", num(d["p_edge_over"]), num(d["p_edge_under"]))
    d["won"] = (d.ind_result == "WIN").astype(float)
    d["settled"] = d.ind_result.isin(["WIN", "LOSS"])
    d["pnl"] = np.where(d.ind_result == "WIN", d.odds - 1, np.where(d.ind_result == "LOSS", -1.0, np.nan))
    d["regime"] = d.fs_time.map(regime)
    d["serving"] = np.where((d.model_type == "standard") & (d.fs_time >= SERVING_SWITCH), "std_v9_repaired",
                            np.where(d.model_type == "standard", "std_legacy", d.model_type.astype(str) + "_legacy"))
    d["serving_recorded"] = d.get("p_serving_recipe", pd.Series(np.nan, index=d.index))
    d["mv"] = d["final_model_version"].fillna("UNKNOWN_LEGACY") if "final_model_version" in d else "UNKNOWN_LEGACY"
    d["odds_band"] = pd.cut(d.odds, [1.0, 2.0, 2.5, 3.0, 99], right=False, labels=["<2.0", "2.0-2.5", "2.5-3.0", ">=3.0"])
    d["edge_band"] = pd.cut(d.edge_fs, [-1, 0.03, 0.05, 0.08, 0.12, 0.2, 9], right=False,
                            labels=["<3%", "3-5%", "5-8%", "8-12%", "12-20%", ">=20%"])
    d["window"] = np.where(d.md >= LIVE_FROM, "since_2026-09-01", "pre_2026-09-01")
    return d


def table(d, by, label):
    rows = []
    for k, g in d.groupby(by, observed=True):
        s = g[g.settled]
        if not len(s):
            continue
        r = roi_ci(s.pnl.values, s.match_date.values)
        rows.append({"split": label, "key": k if not isinstance(k, tuple) else " | ".join(map(str, k)),
                     "n": r["n"], "pnl_u": r["pnl"], "roi_pct": round(100 * r["roi"], 1),
                     "ci_lo": round(100 * r["ci_day"][0], 1), "ci_hi": round(100 * r["ci_day"][1], 1),
                     "hit": round(s.won.mean(), 3), "mean_p_model": round(s.p_side.mean(), 3),
                     "mean_p_mkt_fair": round(s.p_mkt_fair_side.mean(), 3), "mean_odds": round(s.odds.mean(), 2),
                     "mean_edge_fs": round(s.edge_fs.mean(), 3)})
    return rows


def main():
    d = load()
    nums = {}
    # ---- integrity checks (causes 1, 2, 3, 4, 5, 6) -------------------------------------------------
    t = d[d.pred_found == True]  # noqa: E712
    nums["trace_coverage"] = {"rows": int(len(d)), "pred_found": int(d.pred_found.sum()), "bets_found": int(d.bets_found.sum())}
    nums["c3_ledger_odds_eq_pred_side_odds"] = int(np.isclose(t.odds, t.odds_pred_side, atol=0.005).sum())
    nums["c3_ledger_odds_eq_pred_OTHER_side_odds_only"] = int((~np.isclose(t.odds, t.odds_pred_side, atol=0.005)
                                                            & np.isclose(t.odds, t.odds_pred_other, atol=0.005)).sum())
    nums["c3_ledger_odds_neither"] = int((~np.isclose(t.odds, t.odds_pred_side, atol=0.005)
                                       & ~np.isclose(t.odds, t.odds_pred_other, atol=0.005)).sum())
    pe = t[t.pred_edge_side.notna()]
    nums["c1_pred_edge_equals_p_side_minus_1_over_odds"] = {
        "n": int(len(pe)),
        "match_within_0.001": int(np.isclose(pe.pred_edge_side, pe.p_side - 1 / pe.odds_pred_side, atol=1e-3).sum())}
    le = t[t.edge_fs.notna() & t.p_side.notna()]
    nums["c1_ledger_edge_equals_recomputed"] = {"n": int(len(le)),
                                                "match_within_0.0006": int(np.isclose(le.edge_fs, le.edge_recomputed, atol=6e-4).sum())}
    # side choice: chosen side must be the larger-edge side
    both = t[num(t.p_edge_over).notna() & num(t.p_edge_under).notna()]
    larger = np.where(num(both.p_edge_over) >= num(both.p_edge_under), "OVER", "UNDER")
    nums["c3_side_is_larger_edge_side"] = {"n": int(len(both)), "agree": int((larger == both.side.values).sum())}
    pb = t[t["p_bet"].notna()] if "p_bet" in t else t.iloc[0:0]
    nums["c3_ledger_side_eq_pred_bet"] = {"n": int(len(pb)), "agree": int((pb["p_bet"] == pb.side).sum())}
    s = d[d.settled]
    nums["c4_c6_settlement_mismatch_vs_independent"] = int(((d.ledger_result.isin(["WIN", "LOSS"])) &
                                                            (d.ind_result.isin(["WIN", "LOSS"])) &
                                                            (d.ledger_result != d.ind_result)).sum())
    nums["c4_independently_settled"] = int(len(s))
    nums["c5_wrong_line_check"] = "UNDER wins iff total<=2 in independent settlement; ledger agrees row by row (see c4)"
    ov = t.overround.dropna()
    nums["overround_at_first_sight"] = {"n": int(len(ov)), "median": round(float(ov.median()), 4),
                                        "p05": round(float(ov.quantile(.05)), 4), "p95": round(float(ov.quantile(.95)), 4),
                                        "share_below_1": round(float((ov < 1).mean()), 4)}
    for side in ("OVER", "UNDER"):
        x = t[(t.side == side)].overround.dropna()
        nums[f"overround_{side}"] = {"median": round(float(x.median()), 4), "share_below_1": round(float((x < 1).mean()), 4)}
    # ---- headline ROI by side, both windows --------------------------------------------------------------
    rows = []
    for w, dd in (("all_live", d), ("since_2026-09-01", d[d.md >= LIVE_FROM]), ("pre_2026-09-01", d[d.md < LIVE_FROM])):
        for side in ("OVER", "UNDER"):
            x = dd[(dd.side == side) & dd.settled]
            r = roi_ci(x.pnl.values, x.match_date.values)
            rows.append({"window": w, "side": side, **{k: r[k] for k in ("n", "pnl")}, "roi_pct": round(100 * r["roi"], 2),
                         "ci_day": [round(100 * v, 1) for v in r["ci_day"]],
                         "hit": round(float(x.won.mean()), 4) if len(x) else None,
                         "mean_p_model": round(float(x.p_side.mean()), 4) if len(x) else None,
                         "mean_p_mkt_fair": round(float(x.p_mkt_fair_side.mean()), 4) if len(x) else None,
                         "breakeven_hit": round(float((1 / x.odds).mean()), 4) if len(x) else None})
    nums["headline"] = rows
    # ---- splits ---------------------------------------------------------------------------------------
    out = []
    L = d[d.md >= LIVE_FROM]
    for lab, dd in (("since_0901", L), ("all_live", d)):
        out += table(dd, ["side"], f"{lab}:side")
        out += table(dd, ["model_type", "side"], f"{lab}:track x side")
        out += table(dd, ["side", "odds_band"], f"{lab}:side x odds band")
        out += table(dd, ["side", "edge_band"], f"{lab}:side x first-sight edge band")
        out += table(dd, ["side", "tier_fs"], f"{lab}:side x first-sight tier")
        out += table(dd, ["side", "tier_final"], f"{lab}:side x LEDGER (upgraded) tier")
        out += table(dd, ["side", "regime"], f"{lab}:side x threshold regime")
        out += table(dd, ["side", "serving"], f"{lab}:side x serving model")
        out += table(dd, ["side", "mv"], f"{lab}:side x model_version")
        out += table(dd, ["side", "league"], f"{lab}:side x league")
        out += table(dd, ["side", "model_type", "odds_band"], f"{lab}:side x track x odds band")
    T = pd.DataFrame(out)
    T.to_csv(DATA / "phase4_splits.csv", index=False)
    # ---- calibration on bet rows (cause 7) and model vs market --------------------------------------
    cal = []
    for w, dd in (("since_0901", L), ("all_live", d)):
        for side in ("OVER", "UNDER"):
            x = dd[(dd.side == side) & dd.settled & dd.p_side.notna() & dd.p_mkt_fair_side.notna()]
            if not len(x):
                continue
            bm = float(((x.p_side - x.won) ** 2).mean()); bk = float(((x.p_mkt_fair_side - x.won) ** 2).mean())
            se = float(np.sqrt(x.won.var() / len(x)))
            cal.append({"window": w, "side": side, "n": int(len(x)), "claimed_p_model": round(float(x.p_side.mean()), 4),
                        "market_fair_p": round(float(x.p_mkt_fair_side.mean()), 4), "realised": round(float(x.won.mean()), 4),
                        "overconfidence_pp": round(100 * float(x.p_side.mean() - x.won.mean()), 2),
                        "z_model_vs_realised": round(float((x.p_side.mean() - x.won.mean()) / se), 2) if se > 0 else None,
                        "brier_model": round(bm, 4), "brier_market_devig": round(bk, 4)})
    nums["calibration_on_bets"] = cal
    # cause 12 larger-edge selection bias: does larger claimed edge realise more?
    lb = []
    for side in ("OVER", "UNDER"):
        x = L[(L.side == side) & L.settled & L.edge_fs.notna()]
        if len(x) > 10:
            rho = pd.Series(x.edge_fs.values).corr(pd.Series(x.pnl.values), method="spearman")
            lb.append({"side": side, "n": int(len(x)), "spearman_edge_vs_pnl": round(float(rho), 3)})
    nums["c12_edge_vs_pnl_since_0901"] = lb
    # cause 10: legacy/repaired mixing - rows logged after the switch per serving recipe recorded on the row
    nums["c10_serving_recipe_recorded"] = d.groupby(["serving", d["serving_recorded"].fillna("NA")]).size().reset_index(
        ).rename(columns={0: "n"}).to_dict("records")
    # tier upgrade effect on the headline
    nums["tier_upgraded_rows"] = int((d.tier_fs != d.tier_final).sum())
    # side per fixture: both sides on one fixture (flip)
    fx = d.groupby(["match_date", "home_team", "away_team"]).side.nunique()
    nums["fixtures_with_both_sides"] = int((fx > 1).sum())
    json.dump(nums, open(DATA / "phase4_numbers.json", "w"), indent=1, default=str)
    d.to_csv(DATA / "phase4_rows.csv", index=False)
    print(json.dumps(nums, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
