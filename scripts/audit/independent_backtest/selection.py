"""Independent selection rules, written from the documented policy - NOT imported from production.

Policy as documented (CLAUDE.md, src/betting.py docstrings, owner decisions), re-implemented:
  O/U 2.5   p_under = 1 - p_over ; edge_side = p_side - 1/odds_side (raw implied, no de-vig)
            a side QUALIFIES when edge >= VALUABLE_THRESHOLD (deployed env) and odds >= MIN_*_ODDS
            and (new-format, from 2026-10-22) odds <= NF_MAX_OU_ODDS. Both qualify -> larger edge.
            Blind-fixture guard: any of the 4 core rolling-form columns missing (or serving_no_form)
            -> no bet, when the guard is active (code present AND (REQUIRE_FORM_DATA on OR date >= UNTIL)).
            Tier: SNIPER >= per-league approved sniper_th, else min(calibrated league sniper, cap),
            else side global; edge-ceiling -> MARKSMAN for leagues without a per-league threshold;
            MARKSMAN >= approved marksman_th / league marksman (capped) / global; else VALUABLE.
  Side mkts p = p_{market} (one-sided: btts=YES, over15/over35=OVER); fair = (1/odds)/1.08;
            edge = p - fair; VALUABLE >= 0.04; SNIPER/MARKSMAN = league's own thresholds only when
            `approved` (older files without the key: own unless drop), else 0.10 / 0.08.
Independent integrity gates (stricter than production; a failure is RECORDED, never defaulted):
  real finite quote > 1.0 on the selected side, a probability strictly inside (0,1),
  decision time strictly before kickoff when kickoff is known.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .regimes import Regime, code_regime_active

CORE_FORM = ["home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5"]
SIDE_MARKETS = {"btts": "YES", "over15": "OVER", "over35": "OVER"}
SIDE_OVERROUND = 1.08


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _truthy(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def blind_guard_active(reg: Regime, when: pd.Timestamp) -> bool:
    if not code_regime_active("blind_guard_code", when):
        return False
    flag = str(reg.env.get("REQUIRE_FORM_DATA", "1")).strip().lower()
    if flag not in ("0", "false", "no"):
        return True
    until = reg.env.get("REQUIRE_FORM_DATA_UNTIL", "2026-09-15")
    return when.date().isoformat() >= str(until)


def ou25_tier(edge: float, side: str, league: str, reg: Regime) -> str:
    cap = reg.get("LEAGUE_SNIPER_CAP", 1.0) or 1.0
    opt = reg.std_params.get(league) if isinstance(reg.std_params, dict) else None
    opt = opt if isinstance(opt, dict) else None
    if opt and opt.get("approved") and opt.get("sniper_th") is not None:
        per_league, sn = True, float(opt["sniper_th"])
    elif league in reg.values.get("league_sniper", {}):
        per_league, sn = True, min(reg.values["league_sniper"][league], cap)
    else:
        per_league = False
        sn = reg.get(f"SNIPER_THRESHOLD_{side}", reg.get("SNIPER_THRESHOLD", 0.12))
    if edge >= sn:
        return "SNIPER"
    if not per_league:
        ceil = reg.values.get("league_edge_ceiling", {}).get(league, reg.get("EDGE_CEILING", 0.19))
        if edge > ceil:
            return "MARKSMAN"
    if opt and opt.get("approved") and opt.get("marksman_th") is not None:
        mm = float(opt["marksman_th"])
    elif league in reg.values.get("league_marksman", {}):
        mm = min(reg.values["league_marksman"][league], cap)
    else:
        mm = reg.get("MARKSMAN_THRESHOLD", 0.14)
    if edge >= mm:
        return "MARKSMAN"
    return "VALUABLE"


def decide_ou25(row: dict, reg: Regime, when: pd.Timestamp) -> dict:
    """One fixture, one board. Returns decision dict with an explicit `reason` when no bet."""
    out = {"market": "ou25", "selected": False, "side": None, "odds": math.nan, "edge": math.nan,
           "tier": None, "reason": ""}
    p = _f(row.get("p_over25"))
    oo, ou = _f(row.get("odds_over25")), _f(row.get("odds_under25"))
    if not (0.0 < p < 1.0):
        out["reason"] = "no_model_probability"
        return out
    ko = row.get("kickoff_utc")
    if ko is not None and str(ko) not in ("", "nan", "None", "<NA>"):
        kot = pd.Timestamp(str(ko))
        kot = kot.tz_localize("UTC") if kot.tzinfo is None else kot.tz_convert("UTC")
        if when >= kot:
            out["reason"] = "post_kickoff_quote"
            return out
    vt = reg.get("VALUABLE_THRESHOLD", 0.04)
    mo, mu = reg.get("MIN_OVER_ODDS", 1.75), reg.get("MIN_UNDER_ODDS", 1.75)
    nf = str(row.get("model_type", "")) == "new_format"
    cap = reg.get("NF_MAX_OU_ODDS", 0) if code_regime_active("nf_price_cap", when) else 0
    cands = []
    for side, odds, prob, mn in (("OVER", oo, p, mo), ("UNDER", ou, 1.0 - p, mu)):
        if not (odds > 1.0) or math.isnan(odds):
            continue
        e = prob - 1.0 / odds
        if e < vt:
            continue
        if odds < mn:
            continue
        if nf and cap and odds > cap:
            continue
        cands.append((e, side, odds))
    if not cands:
        if math.isnan(oo) and math.isnan(ou):
            out["reason"] = "unpriced"
        elif any((s_odds > 1.0) and (pr - 1.0 / s_odds >= vt) for s_odds, pr in ((oo, p), (ou, 1 - p))
                 if not math.isnan(s_odds)):
            out["reason"] = "price_bound_blocked"
        else:
            out["reason"] = "below_threshold"
        return out
    cands.sort(key=lambda c: (-c[0], c[1] != "OVER"))
    e, side, odds = cands[0]
    blind = any(math.isnan(_f(row.get(c))) for c in CORE_FORM if c in row) or _truthy(row.get("serving_no_form"))
    if blind and blind_guard_active(reg, when):
        out.update(reason="blind_fixture_guard", side=side, odds=odds, edge=e)
        return out
    out.update(selected=True, side=side, odds=odds, edge=e, tier=ou25_tier(e, side, str(row.get("league")), reg),
               blind=blind)
    return out


def side_tier(edge: float, market: str, league: str, reg: Regime) -> str:
    lp = (reg.side_params.get(market, {}) or {}).get(league, {}) if isinstance(reg.side_params, dict) else {}
    lp = lp if isinstance(lp, dict) else {}
    approved = lp.get("approved")
    own = approved if approved is not None else not lp.get("drop", False)
    sn, mm = (lp.get("sniper_th", 0.10), lp.get("marksman_th", 0.08)) if own else (0.10, 0.08)
    if edge >= sn:
        return "SNIPER"
    if edge >= mm:
        return "MARKSMAN"
    return "VALUABLE"


def decide_side(row: dict, market: str, reg: Regime, when: pd.Timestamp) -> dict:
    out = {"market": market, "selected": False, "side": SIDE_MARKETS[market], "odds": math.nan,
           "edge": math.nan, "tier": None, "reason": ""}
    p, odds = _f(row.get(f"p_{market}")), _f(row.get(f"odds_{market}"))
    if not (0.0 < p < 1.0):
        out["reason"] = "no_model_probability"
        return out
    if not (odds > 1.0):
        out["reason"] = "unpriced"
        return out
    ko = row.get("kickoff_utc")
    if ko is not None and str(ko) not in ("", "nan", "None", "<NA>"):
        kot = pd.Timestamp(str(ko))
        kot = kot.tz_localize("UTC") if kot.tzinfo is None else kot.tz_convert("UTC")
        if when >= kot:
            out["reason"] = "post_kickoff_quote"
            return out
    edge = p - (1.0 / odds) / SIDE_OVERROUND
    if edge < 0.04:
        out["reason"] = "below_threshold"
        return out
    out.update(selected=True, odds=odds, edge=edge, tier=side_tier(edge, market, str(row.get("league")), reg))
    return out


def goal_direction(market: str, side: str) -> str | None:
    if market in ("ou25", "over15", "over35"):
        return "OVER" if str(side).upper() == "OVER" else "UNDER"
    return None


def snapshot_time(row: dict) -> pd.Timestamp:
    g = row.get("generated_at")
    if g is not None and str(g) not in ("", "nan", "None", "<NA>"):
        t = pd.Timestamp(str(g))
    else:
        t = pd.Timestamp(str(row["snap_commit_ts"]))
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
