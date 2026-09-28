"""V9 adapter — what production is predicting, how it performed, how healthy it is."""
from __future__ import annotations

import pandas as pd

from dashboard_data.core import (V9_DIR, Freshness, freshness_from_timestamp,
                                 newest_date, read_csv, read_json)

OUT = V9_DIR / "output"


def predictions() -> pd.DataFrame:
    return read_csv(OUT / "predictions.csv")


def ledger() -> pd.DataFrame:
    """Settled + open bets. `model_type` separates the two tracks and must never be pooled."""
    d = read_csv(OUT / "bets_ledger.csv")
    if not d.empty:
        for c in ("generated_at", "match_date"):
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce", utc=True)
    return d


def retrain_log() -> dict:
    return read_json(OUT / "retrain_log.json")


def training_coverage() -> dict:
    return read_json(OUT / "training_coverage.json")


def performance(days: int | None = None) -> dict:
    """P&L split by track and tier. NEVER one blended total across unrelated markets.

    VALUABLE is reported separately and excluded from the staked figure: it is a half-stake
    monitor tier, and folding it into a headline ROI mixes two different decisions.
    Sample size travels with every number, because a percentage without an n invites exactly
    the celebration of noise the brief warns about.
    """
    d = ledger()
    if d.empty or "result" not in d.columns:
        return {"available": False, "why": "no settled bets in bets_ledger.csv"}
    s = d[d["result"].isin(["WIN", "LOSS"])].copy()
    if days and "generated_at" in s.columns:
        cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        s = s[s["generated_at"] >= cut]
    if s.empty:
        return {"available": False, "why": f"no settled bets in the last {days} days"}
    out = {"available": True, "window_days": days, "by_track": {}}
    for track, g in s.groupby(s.get("model_type", "unknown")):
        tiers = {}
        for tier, t in g.groupby(s.loc[g.index].get("signal_tier", "unknown")):
            pnl = pd.to_numeric(t.get("pnl"), errors="coerce").fillna(0)
            tiers[str(tier)] = {"n": int(len(t)), "pnl": round(float(pnl.sum()), 2),
                                "roi": round(float(pnl.sum() / len(t)), 4) if len(t) else None,
                                "hit": round(float((t["result"] == "WIN").mean()), 4)}
        pnl = pd.to_numeric(g.get("pnl"), errors="coerce").fillna(0)
        staked = g[g.get("signal_tier", "").isin(["SNIPER", "MARKSMAN"])]
        spnl = pd.to_numeric(staked.get("pnl"), errors="coerce").fillna(0)
        out["by_track"][str(track)] = {
            "n": int(len(g)), "pnl": round(float(pnl.sum()), 2),
            "roi": round(float(pnl.sum() / len(g)), 4) if len(g) else None,
            "hit": round(float((g["result"] == "WIN").mean()), 4),
            "staked_only": {"n": int(len(staked)), "pnl": round(float(spnl.sum()), 2),
                            "roi": round(float(spnl.sum() / len(staked)), 4) if len(staked) else None},
            "by_tier": tiers,
        }
    return out


def by_league(days: int | None = 30, min_n: int = 5) -> pd.DataFrame:
    """Per-league performance. Rows under `min_n` are KEPT but flagged, never hidden or greened."""
    d = ledger()
    if d.empty or "result" not in d.columns:
        return pd.DataFrame()
    s = d[d["result"].isin(["WIN", "LOSS"])].copy()
    if days and "generated_at" in s.columns:
        s = s[s["generated_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
    if s.empty:
        return pd.DataFrame()
    s["pnl"] = pd.to_numeric(s.get("pnl"), errors="coerce").fillna(0)
    g = s.groupby(["league", s.get("model_type", "unknown")]).agg(
        n=("result", "size"), pnl=("pnl", "sum"),
        hit=("result", lambda x: (x == "WIN").mean())).reset_index()
    g["roi"] = (g["pnl"] / g["n"]).round(4)
    g["reliable"] = g["n"] >= min_n
    return g.sort_values("n", ascending=False).reset_index(drop=True)


def health() -> dict:
    """Freshness derived from CONTENT — the newest row in each artifact, never file mtime."""
    preds, led = predictions(), ledger()
    cov = training_coverage()
    rl = retrain_log()
    latest_run = rl.get("latest_run")
    runs = rl.get("runs", {})
    last = runs.get(latest_run, {}) if latest_run else {}
    return {
        "predictions": {
            "rows": int(len(preds)),
            "freshness": freshness_from_timestamp(
                newest_date(preds, "kickoff_utc", "date"), "fast", "predictions.csv")},
        "ledger": {
            "rows": int(len(led)),
            "settled": int((led.get("result", pd.Series(dtype=object))
                            .isin(["WIN", "LOSS"])).sum()) if not led.empty else 0,
            "freshness": freshness_from_timestamp(
                newest_date(led, "generated_at"), "fast", "bets_ledger.csv")},
        "retrain": {
            "latest_run": latest_run,
            "models": {k: {"promoted": v.get("promoted"), "why": v.get("why"),
                           "basis": v.get("comparison_basis"),
                           "logloss_old": v.get("logloss_old"),
                           "logloss_new": v.get("logloss_new"),
                           "canary": v.get("canary")}
                       for k, v in last.items()},
            "freshness": freshness_from_timestamp(latest_run, "daily", "retrain_log.json")},
        "training_coverage": {"leagues": len(cov) if isinstance(cov, dict) else 0},
    }


def calibration(bins: int = 8, min_bin: int = 20) -> dict:
    """Claimed probability against what actually happened.

    HOW THE CLAIM IS RECOVERED. v9's edge is `model_prob − 1/odds`, so the model's probability is
    `edge_pct/100 + 1/odds` — the ledger stores both, which means no re-scoring is needed and no
    model has to be loaded. The realised rate is simply how often those bets won.

    WHY IT MATTERS MORE THAN ANY OTHER CHART HERE. This is the estate's load-bearing negative
    result: claimed 0.5430 against realised 0.4066, overconfident by +13.64pp on n=792, z=7.81.
    It survived an adversarial selection-bias attack, and the gap is the same size on staked bets
    (+16.80pp) as on never-staked VALUABLE (+10.78pp). That last fact is the finding — if the
    tiers carried information the gaps would differ. The bookmaker's own *vigged* price beats the
    model on Brier (0.2368 vs 0.2583) on the model's own selected bets.

    Bins below `min_bin` are returned flagged, never dropped: a bin of 3 that sits on the diagonal
    is not evidence of calibration.
    """
    import numpy as np
    d = ledger()
    if d.empty or "result" not in d.columns:
        return {"available": False, "why": "no settled bets"}
    s = d[d["result"].isin(["WIN", "LOSS"])].copy()
    odds = pd.to_numeric(s.get("odds"), errors="coerce")
    edge = pd.to_numeric(s.get("edge_pct"), errors="coerce")
    if odds.isna().all() or edge.isna().all():
        return {"available": False, "why": "ledger has no odds/edge_pct to recover a claim from"}
    # edge_pct is stored as a percentage in the ledger, so it is divided by 100 here.
    s["claimed"] = (edge / 100.0) + (1.0 / odds)
    s["won"] = (s["result"] == "WIN").astype(float)
    s = s[s["claimed"].between(0, 1)].dropna(subset=["claimed"])
    if len(s) < min_bin:
        return {"available": False, "why": f"only {len(s)} usable settled bets"}
    s["bin"] = pd.cut(s["claimed"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = (s.groupby("bin", observed=True)
           .agg(n=("won", "size"), claimed=("claimed", "mean"), realised=("won", "mean"))
           .reset_index().dropna(subset=["claimed"]))
    g["reliable"] = g["n"] >= min_bin
    g["gap_pp"] = ((g["claimed"] - g["realised"]) * 100).round(2)
    g["bin"] = g["bin"].astype(str)
    overall_claim = float(s["claimed"].mean())
    overall_real = float(s["won"].mean())
    n = int(len(s))
    # A one-sample z on the difference between the claimed mean and the realised rate.
    se = float(np.sqrt(max(overall_real * (1 - overall_real), 1e-9) / n))
    by_tier = {}
    if "signal_tier" in s.columns:
        for tier, t in s.groupby("signal_tier"):
            if len(t) < min_bin:
                continue
            by_tier[str(tier)] = {"n": int(len(t)),
                                  "claimed": round(float(t["claimed"].mean()), 4),
                                  "realised": round(float(t["won"].mean()), 4),
                                  "gap_pp": round(float((t["claimed"].mean()
                                                         - t["won"].mean()) * 100), 2)}
    return {"available": True, "n": n, "bins": g,
            "claimed": round(overall_claim, 4), "realised": round(overall_real, 4),
            "gap_pp": round((overall_claim - overall_real) * 100, 2),
            "z": round((overall_claim - overall_real) / se, 2) if se else None,
            "by_tier": by_tier}
