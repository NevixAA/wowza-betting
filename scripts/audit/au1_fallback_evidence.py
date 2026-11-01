"""AU1 Phase 1 - defaults / fallback evidence on committed data (READ-ONLY).

Sections (all written to reports/au1_data/, nothing else is touched):
  A. synthetic side-price fills in the committed side-market backtests (BTTS 1.85 / O1.5 1.40 / O3.5 2.60)
     + what the CURRENT loader join would fill from the committed standard_sidemarket_odds_history.csv
  B. live side-market ledger: price provenance classes, timestamp validity against an observed kickoff,
     and flat-1u performance on genuine timestamp-valid prices only (matchday block bootstrap)
  C. legacy batch-median imputation on the current board: per served legacy payload, the NaN share of
     every declared feature in the scored batch; all-NaN -> 0.0; side models' mixed std+NF batch
  D. O/U 2.5 ledger price completeness (missing / non-positive odds) for context

Usage: python scripts/audit/au1_fallback_evidence.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "au1_data"
O = ROOT / "output"
FILL = {"btts": 1.85, "over15": 1.40, "over35": 2.60}
STD = ["La Liga 2", "Championship", "League Two", "League One", "Serie B", "Bundesliga 2", "Ligue 2"]
RNG = np.random.default_rng(20261101)


def block_ci(units: np.ndarray, blocks: np.ndarray, n_boot: int = 4000) -> tuple:
    """95% CI of the mean flat-1u return, resampling whole matchdays (blocks) with replacement."""
    if len(units) < 2:
        return (np.nan, np.nan)
    ub = pd.Series(units).groupby(blocks)
    sums, cnts = ub.sum().to_numpy(), ub.size().to_numpy()
    k = len(sums)
    idx = RNG.integers(0, k, size=(n_boot, k))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def perf(g: pd.DataFrame) -> dict:
    u = g["unit"].to_numpy(float)
    lo, hi = block_ci(u, g["match_date"].astype(str).to_numpy())
    return {"n": int(len(g)), "matchdays": int(g["match_date"].nunique()), "wins": int((u > 0).sum()),
            "units": round(float(u.sum()), 2), "roi": round(float(u.mean()), 4) if len(u) else np.nan,
            "ci_lo": round(lo, 4) if lo == lo else np.nan, "ci_hi": round(hi, 4) if hi == hi else np.nan,
            "mean_odds": round(float(g["odds"].mean()), 3) if len(g) else np.nan}


# ── A ────────────────────────────────────────────────────────────────────────────────────────
def section_a() -> dict:
    res = {}
    for t, c in FILL.items():
        d = pd.read_csv(O / f"backtest_results_{t}.csv", low_memory=False,
                        usecols=["date", "league", "season", "home_team", "away_team", "price_synthetic",
                                 "signal_tier", "bet", "bet_stake", "pnl", f"odds_{t}"])
        syn = d["price_synthetic"].astype(bool)
        real = d[~syn]
        bets = real[real["bet"].eq("BET")]
        sm = bets[bets["signal_tier"].isin(["SNIPER", "MARKSMAN"])]
        res[t] = {
            "rows": int(len(d)), "synthetic_flag": int(syn.sum()),
            "price_eq_fill_constant": int(d[f"odds_{t}"].round(4).eq(c).sum()),
            "synthetic_but_not_constant": int((syn & ~d[f"odds_{t}"].round(4).eq(c)).sum()),
            "real_rows": int(len(real)), "real_date_min": str(real["date"].min()), "real_date_max": str(real["date"].max()),
            "real_bets_all_tiers": int(len(bets)),
            "real_bets_all_tiers_stake": round(float(bets["bet_stake"].sum()), 2),
            "real_bets_all_tiers_pnl": round(float(bets["pnl"].sum()), 2),
            "real_bets_all_tiers_roi_per_stake": round(float(bets["pnl"].sum() / bets["bet_stake"].sum()), 4) if len(bets) else None,
            "real_sniper_marksman": int(len(sm)),
            "real_sm_roi_per_stake": round(float(sm["pnl"].sum() / sm["bet_stake"].sum()), 4) if len(sm) else None,
            "synthetic_rows_with_bet": int((syn & d["bet"].eq("BET")).sum()),
            "synthetic_rows_with_nonzero_pnl": int((syn & pd.to_numeric(d["pnl"], errors="coerce").fillna(0).ne(0)).sum()),
            "date_min": str(d["date"].min()), "date_max": str(d["date"].max()),
        }
    # what the CURRENT legacy loader join fills from the committed side-market history
    import src.data_loader as DL
    DL._IJ.audit_league_free_lookup = lambda *a, **k: None          # never write join health from an audit
    d = pd.read_csv(O / "backtest_results_btts.csv", low_memory=False,
                    usecols=["date", "league", "home_team", "away_team"])
    for col in ("odds_btts", "odds_over15", "odds_over35"):
        d[col] = np.nan
    d["date"] = pd.to_datetime(d["date"])
    y = DL._enrich_with_standard_sidemarket_odds(d.reset_index(drop=True))
    y["season_month"] = y["date"].dt.strftime("%Y-%m")
    res["current_loader_rejoin"] = {
        "rows": int(len(y)),
        **{f"{c}_filled": int(y[c].notna().sum()) for c in ("odds_btts", "odds_over15", "odds_over35")},
        "btts_filled_2024_11_to_2026_10": int(y.loc[y["date"].between("2024-11-01", "2026-10-31"), "odds_btts"].notna().sum()),
    }
    hist = pd.read_csv(O / "standard_sidemarket_odds_history.csv")
    hist["ts_kind"] = np.where(hist["snapshot_ts"].astype(str).str.endswith("T00:00:00Z"), "midnight_placeholder", "timed_capture")
    hist["has_kickoff"] = hist["kickoff_utc"].notna()
    res["sidemarket_history_provenance"] = hist.groupby(["ts_kind", "has_kickoff"]).size().rename("rows").reset_index().to_dict("records")
    return res


# ── B ────────────────────────────────────────────────────────────────────────────────────────
def kickoff_map() -> pd.DataFrame:
    parts = []
    files = [O / "book_odds_snapshots.csv", *sorted(glob.glob(str(O / "book_odds_snapshots_archive" / "*.csv"))),
             O / "standard_odds_history.csv", O / "newformat_odds_history.csv", O / "standard_sidemarket_odds_history.csv"]
    for f in files:
        try:
            x = pd.read_csv(f, usecols=["match_date", "kickoff_utc", "match"], low_memory=False)
        except Exception:                                   # noqa: BLE001 - optional sources
            continue
        parts.append(x.dropna(subset=["kickoff_utc"]))
    k = pd.concat(parts, ignore_index=True)
    k["match_date"] = k["match_date"].astype(str).str[:10]
    k["kickoff_utc"] = pd.to_datetime(k["kickoff_utc"], utc=True, errors="coerce")
    k = k.dropna(subset=["kickoff_utc"])
    g = k.groupby(["match_date", "match"])["kickoff_utc"]
    out = g.agg(["min", "max"]).reset_index()
    out["kickoff_conflict"] = out["max"] - out["min"] > pd.Timedelta(hours=1)
    return out.rename(columns={"min": "kickoff_utc"})[["match_date", "match", "kickoff_utc", "kickoff_conflict"]]


def section_b() -> dict:
    s = pd.read_csv(O / "side_bets_ledger.csv", low_memory=False)
    s["match_date"] = s["match_date"].astype(str).str[:10]
    s["odds"] = pd.to_numeric(s["odds"], errors="coerce")
    s["match"] = s["home_team"].astype(str) + " vs " + s["away_team"].astype(str)
    km = kickoff_map()
    s = s.merge(km, on=["match_date", "match"], how="left")
    s["gen_utc"] = pd.to_datetime(s["generated_at"], utc=True, errors="coerce")
    fill = s.apply(lambda r: FILL.get(str(r["market"])), axis=1)
    notes = s["notes"].astype(str)
    cls = np.select(
        [s["odds"].isna() | (s["odds"] <= 1.0),
         s["odds"].round(4).eq(fill.astype(float)),
         notes.str.contains("fair_odds|no_market_price", na=False),
         s["kickoff_utc"].isna(),
         s["gen_utc"] >= s["kickoff_utc"]],
        ["NO_PRICE", "EQUALS_FILL_CONSTANT", "SETTLED_AT_FAIR_ODDS", "KICKOFF_UNOBSERVED", "POST_KICKOFF"],
        default="GENUINE_TIMESTAMP_VALID")
    s["price_class"] = cls
    s["settled"] = s["result"].astype(str).str.upper().isin(["WIN", "LOSS"])
    s["unit"] = np.where(s["result"].astype(str).str.upper().eq("WIN"), s["odds"] - 1.0,
                         np.where(s["result"].astype(str).str.upper().eq("LOSS"), -1.0, np.nan))
    pnl = pd.to_numeric(s["pnl"], errors="coerce")
    s["pnl_mismatch"] = s["settled"] & (pnl - s["unit"]).abs().gt(0.011)
    res = {"rows": int(len(s)), "class_counts": s["price_class"].value_counts().to_dict(),
           "class_x_market": pd.crosstab(s["price_class"], s["market"]).to_dict(),
           "kickoff_conflicts": int(s["kickoff_conflict"].fillna(False).sum()),
           "settled": int(s["settled"].sum()), "unsettled_or_void": int((~s["settled"]).sum()),
           "result_values": s["result"].fillna("<blank>").astype(str).value_counts().to_dict(),
           "ledger_pnl_vs_recomputed_mismatch": int(s["pnl_mismatch"].sum()),
           "lead_minutes_median": float(((s["kickoff_utc"] - s["gen_utc"]).dt.total_seconds() / 60).median())}
    rows = []
    for start in ("2026-08-10", "2026-09-01"):
        x = s[s["settled"] & (s["match_date"] >= start)]
        for scope, g0 in (("ALL_SETTLED_ANY_PRICE", x),
                          ("GENUINE_TIMESTAMP_VALID", x[x["price_class"].eq("GENUINE_TIMESTAMP_VALID")])):
            for (mkt, mt), g in g0.groupby(["market", "model_type"]):
                rows.append({"from": start, "scope": scope, "market": mkt, "model_type": mt, **perf(g)})
            for tier, g in g0.groupby("signal_tier"):
                rows.append({"from": start, "scope": scope, "market": "ALL", "model_type": f"tier={tier}", **perf(g)})
    pd.DataFrame(rows).to_csv(OUT / "side_ledger_genuine_price_performance.csv", index=False)
    s[["generated_at", "match_date", "league", "match", "market", "model_type", "signal_tier", "odds", "kickoff_utc",
       "price_class", "result", "pnl", "unit"]].to_csv(OUT / "side_ledger_price_classes.csv", index=False)
    return res


# ── C ────────────────────────────────────────────────────────────────────────────────────────
def section_c() -> dict:
    p = pd.read_csv(O / "predictions.csv", low_memory=False)
    import config
    mt = p["model_type"].astype(str)
    scopes = {"newformat (legacy, served)": (config.MODEL_FILE_NEWFORMAT, mt.eq("new_format")),
              "standard legacy (NOT served since 10-28; feature_health monitors it)": (config.MODEL_FILE_STANDARD, mt.eq("standard"))}
    for t, f in config.SIDE_MARKETS.items():
        scopes[f"side {t} (legacy, served std+NF in ONE batch)"] = (f, mt.isin(["standard", "new_format"]))
    scopes["ht_over05 (legacy, std only)"] = (config.HT_MODEL_FILE_05, mt.eq("standard"))
    scopes["ht_over15 (legacy, std only)"] = (config.HT_MODEL_FILE_15, mt.eq("standard"))
    out, detail = {}, []
    for name, (f, mask) in scopes.items():
        pl = pickle.load(open(f, "rb"))
        cols = list(pl["feature_cols"])
        b = p.loc[mask]
        missing_cols = [c for c in cols if c not in b.columns]
        present = [c for c in cols if c in b.columns]
        X = b[present].apply(pd.to_numeric, errors="coerce")
        nan_share = X.isna().mean()
        all_nan = [c for c in present if nan_share[c] == 1.0]
        partial = {c: round(float(nan_share[c]), 3) for c in present if 0 < nan_share[c] < 1}
        const = [c for c in present if X[c].nunique(dropna=True) == 1 and nan_share[c] < 1]
        rec = {"artifact": Path(f).name, "recipe": pl.get("recipe", "legacy(no recipe key)"),
               "has_persisted_imputer": bool(pl.get("imputer")), "n_features": len(cols), "batch_rows": int(len(b)),
               "declared_but_absent_from_frame": missing_cols, "all_nan_in_batch_filled_0.0": all_nan,
               "partially_nan_batch_median_filled": partial, "constant_in_batch": const,
               "cells_imputed_share": round(float(X.isna().to_numpy().mean()), 4) if len(present) else None}
        if "side" in name:
            nf = X[mt.loc[mask].eq("new_format").to_numpy()]
            sd = X[mt.loc[mask].eq("standard").to_numpy()]
            rec["nf_rows_nan_where_std_rows_observed"] = [c for c in present if nf[c].isna().mean() > 0.5 and sd[c].notna().mean() > 0.5]
            rec["nf_cells_imputed_share"] = round(float(nf.isna().to_numpy().mean()), 4) if len(nf) else None
            rec["std_cells_imputed_share"] = round(float(sd.isna().to_numpy().mean()), 4) if len(sd) else None
        out[name] = rec
        for c in present:
            detail.append({"scope": name, "feature": c, "nan_share": round(float(nan_share[c]), 4),
                           "batch_median": X[c].median()})
    pd.DataFrame(detail).to_csv(OUT / "board_feature_imputation.csv", index=False)
    # placeholder-looking constants written to predictions.csv (invariant 9 check)
    sus = {}
    for c in p.columns:
        v = pd.to_numeric(p[c], errors="coerce")
        if v.notna().mean() > 0.9 and v.nunique() == 1:
            sus[c] = float(v.dropna().iloc[0])
    out["constant_numeric_columns_on_board"] = sus
    out["board_generated_at"] = str(p["generated_at"].iloc[0]) if "generated_at" in p else None
    out["board_git_sha"] = str(p["git_sha"].iloc[0]) if "git_sha" in p else None
    return out


# ── D ────────────────────────────────────────────────────────────────────────────────────────
def section_d() -> dict:
    b = pd.read_csv(O / "bets_ledger.csv", low_memory=False)
    odds = pd.to_numeric(b["odds"], errors="coerce")
    return {"rows": int(len(b)), "odds_missing": int(odds.isna().sum()), "odds_le_1": int((odds <= 1).sum()),
            "model_version_missing": int(b["model_version"].isna().sum()),
            "model_type_values": b["model_type"].fillna("<blank>").value_counts().to_dict(),
            "side_values": b["side"].fillna("<blank>").value_counts().to_dict()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"A_synthetic_side_prices": section_a(), "B_side_ledger": section_b(),
           "C_board_imputation": section_c(), "D_ou_ledger_prices": section_d(), "E_rest_days": section_e()}
    (OUT / "fallback_evidence.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:12000])



# ── E: serving-path rest days (src/feature_engineering.py build_upcoming_features) ─────────────
def section_e() -> dict:
    """Board self-consistency: a club that appears twice on the board cannot have rested MORE days
    before its later fixture than the gap since its earlier board fixture; NaN there means the
    earlier board fixture itself was not seen. Then model sensitivity of the served legacy payloads."""
    import warnings
    warnings.simplefilter("ignore")
    p = pd.read_csv(O / "predictions.csv", low_memory=False)
    p["d"] = pd.to_datetime(p["date"])
    long = pd.concat([
        p[["league", "d", "home_team", "home_rest_days"]].rename(columns={"home_team": "team", "home_rest_days": "rest"}),
        p[["league", "d", "away_team", "away_rest_days"]].rename(columns={"away_team": "team", "away_rest_days": "rest"})])
    long = long.sort_values(["team", "d"])
    long["prev_d"] = long.groupby("team")["d"].shift()
    later = long[long["prev_d"].notna()].copy()
    later["gap"] = (later["d"] - later["prev_d"]).dt.days
    res = {"board_rows": int(len(p)), "board_date_sorted": bool(p["d"].is_monotonic_increasing),
           "rest_nan_share_board": round(float(long["rest"].isna().mean()), 3),
           "repeat_team_fixtures": int(len(later)),
           "repeat_rest_nan": int(later["rest"].isna().sum()),
           "repeat_rest_eq_gap": int((later["rest"] == later["gap"]).sum()),
           "repeat_rest_gt_gap_IMPOSSIBLE": int((later["rest"] > later["gap"]).sum()),
           "repeat_rest_lt_gap": int((later["rest"] < later["gap"]).sum())}
    later[["league", "team", "prev_d", "d", "gap", "rest"]].to_csv(OUT / "board_rest_days_consistency.csv", index=False)
    # sensitivity: served legacy NF payload, rest days as served vs both rest-days set to 7
    import config
    from src.model import predict_proba
    nf = p[p["model_type"].eq("new_format")].copy()
    pl = pickle.load(open(config.MODEL_FILE_NEWFORMAT, "rb"))
    base = predict_proba(nf, payload=pl).to_numpy()
    alt = nf.copy()
    alt[["home_rest_days", "away_rest_days"]] = 7.0
    p7 = predict_proba(alt, payload=pl).to_numpy()
    # batch dependence: each row scored ALONE (legacy _prep fills its NaN with 0.0, not the batch median)
    solo = np.array([predict_proba(nf.iloc[[i]], payload=pl).iloc[0] for i in range(len(nf))])
    has_nan = nf[[c for c in pl["feature_cols"] if c in nf.columns]].isna().any(axis=1).to_numpy()
    res["nf_rest7_abs_dp_median"] = round(float(np.median(np.abs(p7 - base))), 4)
    res["nf_rest7_abs_dp_max"] = round(float(np.max(np.abs(p7 - base))), 4)
    res["nf_served_vs_stored_p_over25_maxdiff"] = round(float(np.nanmax(np.abs(base - nf["p_over25"].to_numpy()))), 6)
    res["nf_rows_with_any_nan_feature"] = int(has_nan.sum())
    res["nf_solo_vs_batch_abs_dp_median_on_nan_rows"] = round(float(np.median(np.abs(solo - base)[has_nan])), 4) if has_nan.any() else None
    res["nf_solo_vs_batch_abs_dp_max_on_nan_rows"] = round(float(np.max(np.abs(solo - base)[has_nan])), 4) if has_nan.any() else None
    res["nf_solo_vs_batch_abs_dp_max_on_complete_rows"] = round(float(np.max(np.abs(solo - base)[~has_nan])), 6) if (~has_nan).any() else None
    return res


if __name__ == "__main__":
    main()
