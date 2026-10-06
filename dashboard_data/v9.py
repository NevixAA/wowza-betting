"""V9 adapter — what production is predicting, how it performed, how healthy it is."""
from __future__ import annotations

import pandas as pd

import config

from dashboard_data.core import (V9_DIR, Freshness, freshness_from_timestamp,
                                 newest_date, read_csv, read_json)

OUT = V9_DIR / "output"


def predictions() -> pd.DataFrame:
    return read_csv(OUT / "predictions.csv")


def ledger(include_side_markets: bool = True,
           since_cutoff: bool = True) -> pd.DataFrame:
    """Settled + open bets. `model_type` separates the two tracks and must never be pooled.

    TWO DEFECTS FIXED HERE, 2026-10-06. Together they inverted the headline: the dashboard
    reported a staked P/L of -35.66u when the honest figure is +10.71u.

    1. SIDE MARKETS WERE MISSING ENTIRELY. This read `bets_ledger.csv` only, i.e. main O/U, so
       BTTS / Over 1.5 / Over 3.5 never reached any dashboard number. They are not a rounding
       error — staked side markets are +33.73u since the cutoff and are the profitable part of
       the book. A "Staked P/L" that omits the winning markets is not a P/L.

    2. PRE-CUTOFF BETS WERE COUNTED. 216 staked bets going back to 2025-08-09, worth -12.64u,
       were being mixed into the headline. CLAUDE.md is explicit: performance is "counted only
       from PERFORMANCE_CUTOFF_DATE onward so pre-fix tips are excluded from win/P&L while
       still contributing to CLV". Those are tips from before the fixes that produced the
       current system.

    Both default to the corrected behaviour; pass False to see the raw history.
    """
    frames = []
    main = read_csv(OUT / "bets_ledger.csv")
    if not main.empty:
        main = main.copy()
        main["market"] = "ou25"
        frames.append(main)
    if include_side_markets:
        side = read_csv(OUT / "side_bets_ledger.csv")
        if not side.empty:
            frames.append(side)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    for c in ("generated_at", "match_date"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce", utc=True)
    if since_cutoff and "match_date" in d.columns:
        cut = pd.Timestamp(str(getattr(config, "PERFORMANCE_CUTOFF_DATE", "2026-08-10")),
                           tz="UTC")
        d = d[d["match_date"].isna() | (d["match_date"] >= cut)]
    return d.reset_index(drop=True)


def retrain_log() -> dict:
    return read_json(OUT / "retrain_log.json")


def training_coverage() -> dict:
    return read_json(OUT / "training_coverage.json")


def _break_even(frame) -> float | None:
    """Break-even hit rate implied by the prices actually taken: mean(1/odds).

    NOT 1/mean(odds). They differ by Jensen's inequality, and the wrong one previously
    flattered the SNIPER tier by 1.5 percentage points on this estate.
    """
    import numpy as np
    if frame is None or not len(frame) or "odds" not in frame.columns:
        return None
    o = pd.to_numeric(frame["odds"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(o) & (o > 1)
    return round(float(np.mean(1.0 / o[ok])), 4) if ok.any() else None


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
            # BREAK-EVEN TRAVELS WITH THE HIT RATE. A 45% hit rate is excellent at 2.40 and a
            # disaster at 1.70, so a hit rate shown without the bar it has to clear invites
            # exactly the wrong read. mean(1/odds), never 1/mean(odds).
            "staked_only": {
                "n": int(len(staked)), "pnl": round(float(spnl.sum()), 2),
                "roi": round(float(spnl.sum() / len(staked)), 4) if len(staked) else None,
                "hit": (round(float((staked["result"] == "WIN").mean()), 4)
                        if len(staked) else None),
                "break_even": _break_even(staked)},
            "by_tier": tiers,
        }
    return out


def by_league(days: int | None = 30, min_n: int = 5,
              staked_only: bool = True) -> pd.DataFrame:
    """Per-league performance. Rows under `min_n` are KEPT but flagged, never hidden or greened.

    STAKED TIERS ONLY BY DEFAULT. This pooled VALUABLE with SNIPER and MARKSMAN, so a chart
    titled "where the money went" was showing a paper tier as money. The distortion is not
    cosmetic: on 2026-10-06 the standard track read -65.11u across 553 tips while the staked
    loss was -9.17u across 133 — seven times larger. Pass staked_only=False to see every tip.
    """
    d = ledger()
    if d.empty or "result" not in d.columns:
        return pd.DataFrame()
    s = d[d["result"].isin(["WIN", "LOSS"])].copy()
    if staked_only and "signal_tier" in s.columns:
        s = s[s["signal_tier"].astype(str).str.upper().isin(("SNIPER", "MARKSMAN"))]
    if days and "generated_at" in s.columns:
        s = s[s["generated_at"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
    if s.empty:
        return pd.DataFrame()
    s["pnl"] = pd.to_numeric(s.get("pnl"), errors="coerce").fillna(0)
    s["_inv"] = 1.0 / pd.to_numeric(s.get("odds"), errors="coerce").where(
        lambda x: x > 1)
    g = s.groupby(["league", s.get("model_type", "unknown")]).agg(
        n=("result", "size"), pnl=("pnl", "sum"),
        hit=("result", lambda x: (x == "WIN").mean()),
        # Each league's OWN bar. A 45% hit rate is excellent at 2.40 and a disaster at 1.70,
        # so comparing every league against one estate-wide mean compares them against a line
        # none of them actually has to clear.
        break_even=("_inv", "mean")).reset_index()
    g["break_even"] = g["break_even"].round(4)
    g["excess_hit"] = (g["hit"] - g["break_even"]).round(4)
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
