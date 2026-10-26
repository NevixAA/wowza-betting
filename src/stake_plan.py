"""Stake plan — the approved bankroll policy applied to each predict run's tips. OUTPUT ONLY.

WIRED 2026-10-14 (owner OK). src/bankroll_policy.py (the single sizing implementation, also used by
Pro's simulator) was built 2026-10-13 but called by nothing. This module feeds it the tips a predict
run produced and writes what the policy says to stake:

    output/stake_plan.csv    the CURRENT plan: one row per tip, final fraction, units, and the reason
                             for every zero (PAPER track, odds cap, one position per fixture, ...)
    output/stake_audit.csv   append-only, FIRST SIGHT per (policy_version, fixture, track): a stake can
                             be reproduced later from the policy version and the inputs

Every row carries `side` and its `track`; O/U 2.5 tracks are per side since 2026-10-26
(new_format|OU25_OVER ..., src/bankroll_policy.track_key). Audit rows written before the split carry
the parent key (new_format|OU25) under the same policy_version, whose settings each side copied.

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

# THE SIDE-SELECTION CONTRACT (P0 2026-10-22). Defined ONCE here; pipeline._generate_side_bets
# emits exactly these columns (plus legacy `market_odds`) and selections() reads only these. Before
# this, the producer dropped model_type and side and named the price `market_odds`, while this
# reader did getattr(r, "model_type", "unknown") / getattr(r, "odds", None): every BTTS / OVER15 tip
# became track `unknown|BTTS` -> UNSUPPORTED -> zero stake, and its odds NaN, silently. The unit test
# built inputs that already had the right names, so it never saw the mismatch. No getattr defaults:
# a missing column now raises SelectionSchemaError instead of degrading to "unknown".
#   market   producer's lowercase target ("btts" / "over15" / "over35"), mapped via SIDE_MARKET
#   odds     the decimal price the tip was generated at
#   model_type  the originating prediction's tag: standard | new_format | unknown (genuine only)
SIDE_SELECTION_COLUMNS = ("date", "league", "home_team", "away_team", "model_type", "market", "side",
                          "signal_tier", "odds", "model_prob", "fair_prob", "edge", "ev")
TIPPED = ("SNIPER", "MARKSMAN", "VALUABLE")


class SelectionSchemaError(ValueError):
    """Side selections do not satisfy SIDE_SELECTION_COLUMNS (missing column, model_type or odds)."""


def check_side_selections(side_bets: pd.DataFrame) -> None:
    """Raise SelectionSchemaError unless every tipped row carries the full contract with a real
    model_type and a numeric odds value."""
    missing = [c for c in SIDE_SELECTION_COLUMNS if c not in side_bets.columns]
    if missing:
        raise SelectionSchemaError(f"side selections missing column(s) {missing}")
    t = side_bets[side_bets["signal_tier"].isin(TIPPED)]
    mt = t["model_type"].astype(str).str.strip()
    bad_mt = int((t["model_type"].isna() | mt.isin(("", "nan", "None"))).sum())
    bad_odds = int(pd.to_numeric(t["odds"], errors="coerce").isna().sum())
    if bad_mt or bad_odds:
        raise SelectionSchemaError(f"side selections lost model_type on {bad_mt} row(s) / odds on "
                                   f"{bad_odds} row(s) of {len(t)}")


def health_check(plan: pd.DataFrame, policy: dict) -> list[str]:
    """Problems that mean the plan is silently wrong rather than legitimately zero. Empty = healthy.

    1. A market that has a real-money (REAL_STATES) track in the policy, has tips in the plan, and
       100% of those tips map to an UNSUPPORTED track - the 2026-10-22 failure shape.
    2. Any tip whose odds are missing.
    3. Any O/U 2.5 tip routed to no side track (track ..._UNKNOWN_SIDE, owner split 2026-10-26).
    """
    from src.bankroll_policy import REAL_STATES, UNKNOWN_SIDE, split_track
    if plan is None or plan.empty:
        return []
    probs = []
    # market of a side-split track is its base market: new_format|OU25_OVER -> OU25
    active_markets = {split_track(k)[1] for k, v in (policy.get("tracks") or {}).items()
                      if "|" in k and v.get("state") in REAL_STATES and float(v.get("base_fraction", 0) or 0) > 0}
    for m in sorted(active_markets):
        g = plan[plan["market"] == m]
        if len(g) and (g["deployment_state"] == "UNSUPPORTED").all():
            probs.append(f"{m}: {len(g)} tip(s), ALL on UNSUPPORTED tracks {sorted(set(g['track']))} "
                         f"although the policy has an active {m} track")
    n_odds = int(pd.to_numeric(plan["odds"], errors="coerce").isna().sum())
    if n_odds:
        probs.append(f"{n_odds} tip(s) with no odds")
    n_side = int(plan["track"].astype(str).str.endswith(f"_{UNKNOWN_SIDE}").sum()) if "track" in plan else 0
    if n_side:
        probs.append(f"{n_side} O/U tip(s) with no usable side (track *_{UNKNOWN_SIDE}, no stake)")
    return probs


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
        check_side_selections(side_bets)                    # the contract, or a loud failure
        s = side_bets[side_bets["signal_tier"].isin(TIPPED)]
        for r in s.itertuples(index=False):
            m = SIDE_MARKET.get(str(r.market).lower())
            if not m:
                continue
            d = str(r.date)[:10]
            rows.append({"date": d, "fixture": f"{d}|{r.home_team}|{r.away_team}", "league": r.league,
                         "model_type": str(r.model_type), "market": m, "side": str(r.side),
                         "signal_tier": r.signal_tier, "odds": r.odds})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    return d


#: SHADOW columns (market_edge_v2, 2026-10-25). Appended AFTER sizing as trailing columns of
#: stake_plan.csv only: they never reach size_day, so no fraction, reason or cap can depend on them,
#: and stake_audit.csv keeps its existing header (write() reindexes the audit to it).
SHADOW_PLAN_COLS = ("selection_engine_version", "decision_state", "engine_version", "p_market_fair",
                    "p_decision", "p_lower_bound", "ev_lower_bound", "market_source",
                    "clv_evidence_state")
_SHADOW_JOIN = ["fixture", "market", "side"]


def attach_shadow(plan: pd.DataFrame, shadow: pd.DataFrame | None) -> pd.DataFrame:
    """Plan + trailing shadow columns. Existing columns, their order and every value are untouched;
    a plan row with no shadow match gets NaN (never an invented number). Never raises."""
    try:
        if plan is None or plan.empty:
            return plan
        out = plan.copy()
        out["selection_engine_version"] = "legacy_raw_edge"         # the engine that MADE these tips
        cols = [c for c in SHADOW_PLAN_COLS if c != "selection_engine_version"]
        if shadow is not None and len(shadow) and set(_SHADOW_JOIN) <= set(shadow.columns):
            s = (shadow.drop_duplicates(_SHADOW_JOIN, keep="first")
                 .set_index(_SHADOW_JOIN).reindex(columns=cols))
            idx = pd.MultiIndex.from_arrays([out[k].astype(str) for k in _SHADOW_JOIN])
            s.index = pd.MultiIndex.from_arrays([s.index.get_level_values(i).astype(str)
                                                 for i in range(len(_SHADOW_JOIN))])
            m = s.reindex(idx)
            for c in cols:
                out[c] = m[c].to_numpy()
        else:
            for c in cols:
                out[c] = pd.NA
        return out
    except Exception as e:                                           # noqa: BLE001 - shadow only
        log.warning(f"stake plan shadow columns skipped ({type(e).__name__}: {e})")
        return plan


def _contract_shadow(frame: str, df: pd.DataFrame, schema: str) -> None:
    """SHADOW contract check (src/contracts, 2026-10-25) on a frame that is ALREADY written. Records
    violations in output/contract_violations.json; never changes, blocks or raises."""
    try:
        from src.contracts.shadow import observe
        observe(frame, df, schema)
    except Exception:                                                # noqa: BLE001 - observation only
        pass


def prior_directions(fixtures=None) -> dict:
    """{fixture: "OVER"|"UNDER"} of the FIRST position already STAKED (final_fraction > 0) on each
    fixture by an earlier run, from stake_audit.csv (append-only, first sight). Feeds bankroll_policy
    .size_day's one-direction-per-fixture guard (owner rule 2026-10-25) so it also holds across runs.
    Missing/unreadable audit -> {} (the guard then works within the day only). Never raises."""
    from src.bankroll_policy import goal_direction
    try:
        if not AUDIT.exists():
            return {}
        a = pd.read_csv(AUDIT, dtype=str, usecols=lambda c: c in ("fixture", "market", "side",
                                                                 "final_fraction", "generated_at"))
        if not {"fixture", "market", "final_fraction"} <= set(a.columns):
            return {}
        a = a[pd.to_numeric(a["final_fraction"], errors="coerce").fillna(0) > 0]
        if fixtures is not None:
            a = a[a["fixture"].isin(set(fixtures))]
        if a.empty:
            return {}
        sides = a["side"] if "side" in a.columns else pd.Series("", index=a.index)
        a = a.assign(_dir=[goal_direction(m, s) for m, s in zip(a["market"], sides)])
        a = a[a["_dir"].notna()]
        if "generated_at" in a.columns:
            a = a.sort_values("generated_at", kind="mergesort")
        a = a.drop_duplicates("fixture", keep="first")
        return dict(zip(a["fixture"].astype(str), a["_dir"]))
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"stake plan: prior directions unavailable ({type(e).__name__}) - within-day guard only")
        return {}


def write(bets: pd.DataFrame | None, side_bets: pd.DataFrame | None,
          shadow: pd.DataFrame | None = None) -> pd.DataFrame:
    """Size this run's tips under the approved policy and write plan + audit. Never raises.

    `shadow` (src/edge_engine_shadow.run) only adds trailing SHADOW_PLAN_COLS to stake_plan.csv."""
    try:
        from src.bankroll_policy import load_policy, size_day
        sel = selections(bets, side_bets)
        bankroll = float(os.getenv("STAKE_PLAN_BANKROLL", "100"))
        policy = load_policy()
        # size_day sizes ONE day (its daily and league caps are per day), and a predict run holds up to
        # a week of fixtures, so each match date is sized on its own.
        ratings = load_league_ratings() if (policy.get("league_rating_sizing") or {}).get("enabled") else None
        prior = prior_directions(sel["fixture"]) if len(sel) else {}
        days = [size_day(g, policy, bankroll=bankroll, peak=bankroll, league_ratings=ratings,
                         prior_directions=prior)
                for _, g in sel.groupby("date")] if len(sel) else []
        plan = pd.concat(days, ignore_index=True) if days else size_day(sel, policy, bankroll=bankroll, peak=bankroll)
        for problem in health_check(plan, policy):
            log.error(f"STAKE PLAN HEALTH: {problem}")
        plan["bankroll_basis"] = BASIS
        plan["generated_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
        written = attach_shadow(plan, shadow)
        written.to_csv(PLAN, index=False)
        _contract_shadow("stake_plan.csv", written, "STAKE_PLAN_ROW")
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
                _contract_shadow("stake_audit.csv (appended rows)", new, "STAKE_AUDIT_ROW")
        staked = plan[plan["final_fraction"] > 0] if not plan.empty else plan
        log.info(f"Stake plan → {PLAN.name}: {len(plan)} tip(s), {len(staked)} with a stake, "
                 f"total {100 * float(staked['final_fraction'].sum()) if len(staked) else 0:.2f}% of bankroll")
        return plan
    except SelectionSchemaError as e:                                # contract broken: loud, not silent
        log.error(f"STAKE PLAN NOT WRITTEN - selection contract broken: {e}")
        return pd.DataFrame()
    except Exception as e:                                           # noqa: BLE001 - never fail predict
        log.warning(f"stake plan skipped: {e}")
        return pd.DataFrame()
