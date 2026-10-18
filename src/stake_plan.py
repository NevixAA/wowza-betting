"""Stake plan — the approved bankroll policy applied to each predict run's tips. OUTPUT ONLY.

WIRED 2026-10-14 (owner OK). src/bankroll_policy.py (the single sizing implementation, also used by
Pro's simulator) was built 2026-10-13 but called by nothing. This module feeds it the tips a predict
run produced and writes what the policy says to stake:

    output/stake_plan.csv    the CURRENT plan: one row per tip, final fraction, units, and the reason
                             for every zero (PAPER track, odds cap, one position per fixture, ...)
    output/stake_audit.csv   append-only, FIRST SIGHT per (policy_version, fixture, track): a stake can
                             be reproduced later from the policy version and the inputs

WHAT IT DOES NOT DO. It never decides whether a tip exists, never changes a tier, never sends a
message and never places a bet (Cloudbet / wowza-exec are untouched). Stakes are recommendations.

BANKROLL BASIS. There is no committed real-bankroll ledger, so stakes are sized on a NOTIONAL bankroll
(STAKE_PLAN_BANKROLL, default 100 units: 1.25 units = 1.25% of bankroll) with peak = bankroll, i.e. the
drawdown throttle is NOT applied. Stated in every row (`bankroll_basis`) rather than implied.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

import config

log = logging.getLogger(__name__)

PLAN = config.OUTPUT_DIR / "stake_plan.csv"
AUDIT = config.OUTPUT_DIR / "stake_audit.csv"
SIDE_MARKET = {"btts": "BTTS", "over15": "OVER15", "over35": "OVER35"}
BASIS = "notional bankroll, drawdown throttle not applied (no real-bankroll ledger)"


PRO_RATINGS = "output/studies/league_ratings.json"


def load_league_ratings(max_age_days: float = 8.0) -> dict:
    """Pro's league ratings 1-10 (src/studies/league_results.py). Fail closed: missing, unreadable or
    older than max_age_days -> {} (every league gets the minimum scale). Recorded timestamp, never mtime."""
    import json
    try:
        j = None
        for p in (os.getenv("PRO_LOCAL"), "../pro", "../wowzaV9-Pro"):
            if p and os.path.exists(os.path.join(p, PRO_RATINGS)):
                j = json.loads(open(os.path.join(p, PRO_RATINGS), encoding="utf-8").read())
                break
        if j is None:
            import requests
            r = requests.get("https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main/" + PRO_RATINGS, timeout=20)
            j = r.json() if r.status_code == 200 else None
        if not j:
            return {}
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["generated_at"])).total_seconds() / 86400
        if not (0 <= age <= max_age_days):
            return {}
        return {str(k): int(v["rating"]) for k, v in (j.get("ratings") or {}).items()
                if isinstance(v, dict) and isinstance(v.get("rating"), int) and 1 <= v["rating"] <= 10}
    except Exception as e:                                           # noqa: BLE001 - fail closed
        log.warning(f"league ratings unavailable ({type(e).__name__}) - minimum scale for every league")
        return {}


def selections(bets: pd.DataFrame | None, side_bets: pd.DataFrame | None) -> pd.DataFrame:
    """Tips -> the reader's input: fixture, league, model_type, market, signal_tier, odds."""
    rows = []
    if bets is not None and not bets.empty:
        b = bets[bets["signal_tier"].isin(("SNIPER", "MARKSMAN", "VALUABLE"))]
        for r in b.itertuples(index=False):
            side = str(getattr(r, "best_side", "")).upper()
            odds = getattr(r, "odds_over25" if side == "OVER" else "odds_under25", None)
            rows.append({"date": str(r.date)[:10], "fixture": f"{str(r.date)[:10]}|{r.home_team}|{r.away_team}",
                         "league": r.league, "model_type": getattr(r, "model_type", "unknown"),
                         "market": "OU25", "side": side, "signal_tier": r.signal_tier, "odds": odds})
    if side_bets is not None and not side_bets.empty:
        s = side_bets[side_bets["signal_tier"].isin(("SNIPER", "MARKSMAN", "VALUABLE"))]
        for r in s.itertuples(index=False):
            m = SIDE_MARKET.get(str(r.market).lower())
            if not m:
                continue
            d = str(getattr(r, "date", getattr(r, "match_date", "")))[:10]
            rows.append({"date": d, "fixture": f"{d}|{r.home_team}|{r.away_team}", "league": r.league,
                         "model_type": getattr(r, "model_type", "unknown"), "market": m,
                         "side": str(getattr(r, "side", "") or "YES/OVER"), "signal_tier": r.signal_tier,
                         "odds": getattr(r, "odds", None)})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    return d


def write(bets: pd.DataFrame | None, side_bets: pd.DataFrame | None) -> pd.DataFrame:
    """Size this run's tips under the approved policy and write plan + audit. Never raises."""
    try:
        from src.bankroll_policy import load_policy, size_day
        sel = selections(bets, side_bets)
        bankroll = float(os.getenv("STAKE_PLAN_BANKROLL", "100"))
        policy = load_policy()
        # size_day sizes ONE day (its daily and league caps are per day), and a predict run holds up to
        # a week of fixtures, so each match date is sized on its own.
        ratings = load_league_ratings() if (policy.get("league_rating_sizing") or {}).get("enabled") else None
        days = [size_day(g, policy, bankroll=bankroll, peak=bankroll, league_ratings=ratings)
                for _, g in sel.groupby("date")] if len(sel) else []
        plan = pd.concat(days, ignore_index=True) if days else size_day(sel, policy, bankroll=bankroll, peak=bankroll)
        plan["bankroll_basis"] = BASIS
        plan["generated_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
        plan.to_csv(PLAN, index=False)
        if not plan.empty:
            key = ["policy_version", "fixture", "track"]
            new = plan.drop_duplicates(key)
            if AUDIT.exists():
                prev = pd.read_csv(AUDIT, dtype=str, usecols=lambda c: c in key)
                seen = set(map(tuple, prev[key].astype(str).values)) if set(key) <= set(prev.columns) else set()
                new = new[[tuple(x) not in seen for x in new[key].astype(str).values]]
                # Append under the EXISTING header: a new column (league_rating, 2026-10-18) must never
                # shift every later value into the wrong column of an append-only file.
                head = list(pd.read_csv(AUDIT, nrows=0).columns)
                new = new.reindex(columns=head)
            if not new.empty:
                new.to_csv(AUDIT, mode="a", header=not AUDIT.exists(), index=False)
        staked = plan[plan["final_fraction"] > 0] if not plan.empty else plan
        log.info(f"Stake plan → {PLAN.name}: {len(plan)} tip(s), {len(staked)} with a stake, "
                 f"total {100 * float(staked['final_fraction'].sum()) if len(staked) else 0:.2f}% of bankroll")
        return plan
    except Exception as e:                                           # noqa: BLE001 - never fail predict
        log.warning(f"stake plan skipped: {e}")
        return pd.DataFrame()
