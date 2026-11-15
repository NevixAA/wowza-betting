"""Phase 2 / P1_UNDER25_RELEASE_GATE - why did new-format O/U 2.5 UNDER tips underperform?

READ-ONLY research script. It reads committed files only, writes one JSON of numbers, and changes
no ledger, policy, threshold, stake or track state.

    python scripts/p2_under_analysis.py --data <checkout whose output/ holds the ledgers> \
        --corrections <checkout whose output/corrections holds the sidecars> \
        --out reports/p2_under_data/under_analysis.json

Inputs (recorded in the JSON with the commit SHAs passed on the command line):
  output/bets_ledger.csv        every O/U 2.5 tip, first sight per fixture, settled in place
  output/v91_shadow.csv         first-sight model probability (p_v9 = P(over 2.5)) for EVERY scored
                                NF fixture, tipped or not, with the market probability and the score
  output/corrections/clv_first_half_contaminated.csv   rows whose closing price is a first-half
                                price (excluded from CLV, per the sidecar's own action column)

Method: flat 1 unit per tip; Wilson 95% CI for hit rates; percentile bootstrap (10,000 resamples,
seed 20261103) for ROI, mean CLV, bias and log-loss differences. No value is imputed: a row missing
what a metric needs is left out of that metric and the count is reported.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261103
B = 10_000
CUTOFF = "2026-08-10"                      # config.PERFORMANCE_CUTOFF_DATE
FIX_P0_1_UTC = "2026-11-02T08:34:00Z"      # 820bb6cd: first main commit containing 1ea46192 (P0-1)
FIX_2026_11_01_UTC = "2026-11-01T17:00:00Z"  # src/nf_calibration.SINCE: first board with 11-01 serving fixes
PAPER_SINCE = "2026-11-02"                 # owner: new_format|OU25_UNDER -> PAPER


def wilson(k: int, n: int) -> list:
    if n == 0:
        return [None, None]
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def boot_mean(x, rng) -> list:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [None, None]
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def tip_block(d: pd.DataFrame, rng) -> dict:
    """Flat-1u record of settled tips: n, hit, Wilson CI, break-even, model p, ROI + bootstrap CI."""
    s = d[d["result"].isin(["WIN", "LOSS"])]
    n = int(len(s))
    if n == 0:
        return {"n": 0}
    win = (s["result"] == "WIN").to_numpy()
    odds = s["odds"].to_numpy(float)
    ret = np.where(win, odds - 1.0, -1.0)
    k = int(win.sum())
    out = {"n": n, "wins": k, "hit": round(k / n, 4), "hit_ci95": wilson(k, n),
           "breakeven_hit": round(float(np.mean(1.0 / odds)), 4),
           "model_p_mean": round(float(np.nanmean(s["model_p"])), 4) if s["model_p"].notna().any() else None,
           "mean_odds": round(float(odds.mean()), 3),
           "units": round(float(ret.sum()), 2), "roi": round(float(ret.mean()), 4),
           "roi_ci95": boot_mean(ret, rng)}
    return out


def load_ledger(data: Path) -> pd.DataFrame:
    b = pd.read_csv(data / "output" / "bets_ledger.csv", low_memory=False)
    b["odds"] = pd.to_numeric(b["odds"], errors="coerce")
    b["edge"] = pd.to_numeric(b["edge_pct"], errors="coerce") / 100.0
    # The ledger stores edge = model P(side) - 1/odds (pipeline edge on the tipped side), so the model's
    # probability for the tipped side is 1/odds + edge. Not imputed: NaN when either is missing.
    b["model_p"] = 1.0 / b["odds"] + b["edge"]
    b["gen"] = pd.to_datetime(b["generated_at"], errors="coerce", utc=True)
    b["md"] = pd.to_datetime(b["match_date"], errors="coerce")
    b["result"] = b["result"].astype(str).str.upper().replace({"NAN": np.nan})
    return b


def ledger_analysis(b: pd.DataFrame, cont: pd.DataFrame, rng) -> dict:
    live = b[b["source"].astype(str).eq("live")]
    nf = live[(live["model_type"].astype(str) == "new_format") & (live["md"] >= CUTOFF)]
    u = nf[nf["side"].astype(str).str.upper() == "UNDER"].copy()
    o = nf[nf["side"].astype(str).str.upper() == "OVER"].copy()
    res: dict = {}
    res["counts"] = {
        "nf_live_since_cutoff_rows": int(len(nf)),
        "nf_under_rows": int(len(u)), "nf_over_rows": int(len(o)),
        "nf_avoid_rows": int((nf["side"].astype(str).str.upper() == "AVOID").sum()),
        "nf_under_settled_win_loss": int(u["result"].isin(["WIN", "LOSS"]).sum()),
        "nf_under_void": int((u["result"] == "VOID").sum()),
        "nf_under_pending": int(u["result"].isna().sum()),
        "nf_under_duplicate_fixture_rows": int(u.duplicated(["match_date", "home_team", "away_team"]).sum()),
    }
    staked = ("SNIPER", "MARKSMAN")
    res["under_all_tiers"] = tip_block(u, rng)
    res["under_staked_tiers"] = tip_block(u[u["signal_tier"].isin(staked)], rng)
    res["under_valuable"] = tip_block(u[u["signal_tier"] == "VALUABLE"], rng)
    res["over_all_tiers"] = tip_block(o, rng)
    res["over_staked_tiers"] = tip_block(o[o["signal_tier"].isin(staked)], rng)
    # THE staked basis (dashboard_data/today_view): paper leagues out. USA MLS has been a paper league
    # since 2026-10-07 (config.PAPER_LEAGUES), so it is shown separately rather than mixed in.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config                                                      # noqa: E402
    paper = u["league"].map(config.is_paper_league).fillna(False).astype(bool)
    res["paper_leagues"] = sorted(config.PAPER_LEAGUES)
    res["under_staked_tiers_ex_paper_leagues"] = tip_block(u[u["signal_tier"].isin(staked) & ~paper], rng)
    res["under_all_tiers_ex_paper_leagues"] = tip_block(u[~paper], rng)
    res["under_staked_tiers_paper_leagues_only"] = tip_block(u[u["signal_tier"].isin(staked) & paper], rng)
    res["under_by_tier"] = {t: tip_block(u[u["signal_tier"] == t], rng) for t in ("SNIPER", "MARKSMAN", "VALUABLE")}
    res["under_by_tier_ex_paper_leagues"] = {t: tip_block(u[(u["signal_tier"] == t) & ~paper], rng)
                                             for t in ("SNIPER", "MARKSMAN", "VALUABLE")}
    bands = [(0, 1.8), (1.8, 2.0), (2.0, 2.2), (2.2, 2.5), (2.5, 99)]
    res["under_by_odds_band"] = {f"{lo:.2f}-{hi:.2f}": tip_block(u[(u["odds"] >= lo) & (u["odds"] < hi)], rng)
                                 for lo, hi in bands}
    ebands = [(0, 0.04), (0.04, 0.08), (0.08, 0.12), (0.12, 9)]
    res["under_by_edge_band"] = {f"{lo:.2f}-{hi:.2f}": tip_block(u[(u["edge"] >= lo) & (u["edge"] < hi)], rng)
                                 for lo, hi in ebands}
    lg = u[u["result"].isin(["WIN", "LOSS"])]["league"].value_counts()
    res["under_by_league_n_ge_20"] = {k: tip_block(u[u["league"] == k], rng) for k in lg[lg >= 20].index}
    res["under_leagues_lt_20_settled"] = {"leagues": int((lg < 20).sum()), "tips": int(lg[lg < 20].sum())}
    u["month"] = u["md"].dt.strftime("%Y-%m")
    res["under_by_month"] = {m: tip_block(g, rng) for m, g in u.groupby("month")}
    res["under_by_model_version"] = {str(m): tip_block(g, rng) for m, g in u.groupby(u["model_version"].fillna("(blank)"))}
    # Edge vs outcome: does a bigger claimed edge win more often?
    s = u[u["result"].isin(["WIN", "LOSS"]) & u["edge"].notna()]
    if len(s) > 10:
        r = pd.Series((s["result"] == "WIN").astype(float).to_numpy()).corr(
            pd.Series(s["edge"].to_numpy()), method="spearman")
        res["under_edge_vs_win_spearman"] = round(float(r), 4)
    # Model probability bias on the tipped set: mean model P(under) vs realised under rate.
    s = u[u["result"].isin(["WIN", "LOSS"]) & u["model_p"].notna()]
    gap = (s["model_p"] - (s["result"] == "WIN").astype(float)).to_numpy()
    res["under_model_p_minus_hit"] = {"n": int(len(s)), "mean": round(float(gap.mean()), 4),
                                      "ci95": boot_mean(gap, rng)}
    # CLV on settled tips with a closing price, excluding the first-half-contaminated sidecar rows.
    key = ["generated_at", "match_date", "home_team", "away_team", "side"]
    bad = set(map(tuple, cont[key].astype(str).values)) if len(cont) else set()
    c = u[pd.to_numeric(u["clv_pct"], errors="coerce").notna()].copy()
    c["_bad"] = [tuple(x) in bad for x in c[key].astype(str).values]
    clean = pd.to_numeric(c.loc[~c["_bad"], "clv_pct"], errors="coerce").to_numpy() / 100.0
    in_scope = cont[(cont.get("model_type", pd.Series(dtype=str)).astype(str) == "new_format")
                    & (cont.get("side", pd.Series(dtype=str)).astype(str) == "UNDER")
                    & (pd.to_datetime(cont.get("match_date"), errors="coerce") >= CUTOFF)] if len(cont) else cont
    res["under_clv"] = {"rows_with_clv": int(len(c)), "excluded_first_half_sidecar": int(c["_bad"].sum()),
                        "sidecar_nf_under_rows_since_cutoff": int(len(in_scope)),
                        "n": int(len(clean)), "mean_clv": round(float(clean.mean()), 4) if len(clean) else None,
                        "ci95": boot_mean(clean, rng),
                        "share_positive": round(float((clean > 0).mean()), 4) if len(clean) else None}
    # P0-1 fix boundary and the 2026-11-01 serving fixes.
    t1, t2 = pd.Timestamp(FIX_2026_11_01_UTC), pd.Timestamp(FIX_P0_1_UTC)
    gen = u["gen"].dt.tz_convert("UTC") if u["gen"].dt.tz is not None else u["gen"]
    res["under_by_serving_period"] = {
        "before_2026-11-01T17Z": tip_block(u[gen < t1], rng),
        "2026-11-01T17Z_to_P0-1": tip_block(u[(gen >= t1) & (gen < t2)], rng),
        "after_P0-1_2026-11-02T08:34Z": tip_block(u[gen >= t2], rng),
        "after_P0-1_rows_any_status": int((gen >= t2).sum()),
    }
    res["under_since_paper_2026-11-02_rows"] = int((u["md"] >= PAPER_SINCE).sum())
    # Fixture-data checks on the NF Under rows.
    s = u[u["result"].isin(["WIN", "LOSS"])]
    exp_pnl = np.where(s["result"] == "WIN", s["odds"] - 1.0, -1.0)
    pnl = pd.to_numeric(s["pnl"], errors="coerce").to_numpy()
    res["data_checks"] = {
        "settled_rows_pnl_missing": int(np.isnan(pnl).sum()),
        "settled_rows_pnl_not_flat_1u": int((np.abs(pnl - exp_pnl) > 0.011).sum()),
        "ledger_units_as_booked": round(float(np.nansum(pnl)), 2),
        "ledger_units_recomputed": round(float(exp_pnl.sum()), 2),
        "settled_with_future_match_date": int((s["md"] > pd.Timestamp.now().normalize()).sum()),
        "odds_missing_or_le_1": int((~(u["odds"] > 1.0)).sum()),
        "fixtures_with_both_sides_in_ledger": int(len(
            set(map(tuple, u[["match_date", "home_team", "away_team"]].astype(str).values))
            & set(map(tuple, o[["match_date", "home_team", "away_team"]].astype(str).values)))),
    }
    return res


def shadow_analysis(data: Path, b: pd.DataFrame, rng) -> dict:
    s = pd.read_csv(data / "output" / "v91_shadow.csv", low_memory=False)
    s = s[(s["model_type"].astype(str) == "new_format") & (s["market"].astype(str) == "ou25")].copy()
    s["t"] = pd.to_datetime(s["logged_at"], errors="coerce", utc=True)
    s = s.sort_values("t", kind="mergesort").drop_duplicates("fixture_id", keep="first")
    hg, ag = pd.to_numeric(s["home_goals"], errors="coerce"), pd.to_numeric(s["away_goals"], errors="coerce")
    s["p"] = pd.to_numeric(s["p_v9"], errors="coerce")
    s["pm"] = pd.to_numeric(s["p_market"], errors="coerce")
    s["y"] = np.where(hg.notna() & ag.notna(), ((hg + ag) > 2.5).astype(float), np.nan)
    out: dict = {"nf_ou25_fixtures_logged": int(len(s))}
    st = s[s["y"].notna() & s["p"].notna()]
    out["settled"] = int(len(st))

    def blk(x: pd.DataFrame) -> dict:
        if len(x) < 2:
            return {"n": int(len(x))}
        y, p = x["y"].to_numpy(), np.clip(x["p"].to_numpy(), 1e-6, 1 - 1e-6)
        r = {"n": int(len(x)), "over_rate": round(float(y.mean()), 4),
             "over_rate_ci95": wilson(int(y.sum()), len(x)),
             "model_p_over_mean": round(float(p.mean()), 4),
             "bias_model_minus_actual": round(float(p.mean() - y.mean()), 4),
             "bias_ci95": boot_mean(p - y, rng)}
        m = x["pm"].notna()
        if m.sum() >= 2:
            pm = np.clip(x.loc[m, "pm"].to_numpy(), 1e-6, 1 - 1e-6)
            ym, pp = x.loc[m, "y"].to_numpy(), np.clip(x.loc[m, "p"].to_numpy(), 1e-6, 1 - 1e-6)
            ll_mod = -(ym * np.log(pp) + (1 - ym) * np.log(1 - pp))
            ll_mkt = -(ym * np.log(pm) + (1 - ym) * np.log(1 - pm))
            r.update({"n_with_market": int(m.sum()), "market_p_over_mean": round(float(pm.mean()), 4),
                      "logloss_model": round(float(ll_mod.mean()), 4), "logloss_market": round(float(ll_mkt.mean()), 4),
                      "logloss_model_minus_market_ci95": boot_mean(ll_mod - ll_mkt, rng),
                      "brier_model": round(float(((pp - ym) ** 2).mean()), 4),
                      "brier_market": round(float(((pm - ym) ** 2).mean()), 4)})
        return r

    out["all_settled"] = blk(st)
    t1, t2 = pd.Timestamp(FIX_2026_11_01_UTC), pd.Timestamp(FIX_P0_1_UTC)
    out["by_serving_period"] = {"before_2026-11-01T17Z": blk(st[st["t"] < t1]),
                                "2026-11-01T17Z_to_P0-1": blk(st[(st["t"] >= t1) & (st["t"] < t2)]),
                                "after_P0-1": blk(st[st["t"] >= t2]),
                                "after_P0-1_logged_any_status": int((s["t"] >= t2).sum())}

    # Model minus market P(over) needs no result, so it can be read on fixtures not yet played.
    def gap(x: pd.DataFrame) -> dict:
        m = x["p"].notna() & x["pm"].notna()
        g = (x.loc[m, "p"] - x.loc[m, "pm"]).to_numpy()
        return {"n": int(m.sum()), "mean_model_minus_market": round(float(g.mean()), 4) if len(g) else None,
                "ci95": boot_mean(g, rng)}
    out["model_minus_market_all_logged"] = {"before_2026-11-01T17Z": gap(s[s["t"] < t1]),
                                            "2026-11-01T17Z_to_P0-1": gap(s[(s["t"] >= t1) & (s["t"] < t2)]),
                                            "after_P0-1": gap(s[s["t"] >= t2])}
    out["logged_at_max"] = str(s["t"].max())
    edges = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
    st = st.assign(bin=pd.cut(st["p"], edges, include_lowest=True))
    out["calibration_bins_p_over"] = {str(k): blk(g) for k, g in st.groupby("bin", observed=True)}
    lg = st["league"].value_counts()
    out["by_league_n_ge_20"] = {k: blk(st[st["league"] == k]) for k in lg[lg >= 20].index}
    # How often does the model's side disagree with the market favourite on totals?
    v = st[st["pm"].notna()]
    out["model_side_under_share"] = round(float((v["p"] < 0.5).mean()), 4) if len(v) else None
    out["market_side_under_share"] = round(float((v["pm"] < 0.5).mean()), 4) if len(v) else None
    # Cross-source settlement check: ledger NF Under result vs the shadow log's score for the same fixture.
    u = b[(b["source"].astype(str) == "live") & (b["model_type"].astype(str) == "new_format")
          & (b["side"].astype(str).str.upper() == "UNDER") & b["result"].isin(["WIN", "LOSS"])]
    j = u.merge(s[["match_date", "home_team", "away_team", "y"]].dropna(subset=["y"])
                .drop_duplicates(["match_date", "home_team", "away_team"]),
                on=["match_date", "home_team", "away_team"], how="inner")
    exp = np.where(j["y"] == 0, "WIN", "LOSS")
    out["ledger_vs_shadow_settlement"] = {"joined": int(len(j)), "disagree": int((j["result"].to_numpy() != exp).sum()),
                                          "examples": j.loc[j["result"].to_numpy() != exp,
                                                            ["match_date", "league", "home_team", "away_team", "result", "y"]]
                                          .head(10).astype(str).to_dict("records")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-sha", required=True)
    ap.add_argument("--corrections", required=True)
    ap.add_argument("--corrections-sha", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    data, corr = Path(a.data), Path(a.corrections)
    b = load_ledger(data)
    cf = corr / "output" / "corrections" / "clv_first_half_contaminated.csv"
    cont = pd.read_csv(cf) if cf.exists() else pd.DataFrame()
    res = {"inputs": {"ledgers_and_shadow_from": a.data_sha, "corrections_from": a.corrections_sha,
                      "seed": SEED, "bootstrap": B, "cutoff": CUTOFF,
                      "p0_1_on_main_utc": FIX_P0_1_UTC, "serving_fixes_2026_11_01_utc": FIX_2026_11_01_UTC},
           "ledger": ledger_analysis(b, cont, rng),
           "shadow": shadow_analysis(data, b, rng)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
