"""side_thresholds.py - THE side-market tier bars and where they came from (fix9/defaults 2026-11-04).

    sniper, marksman, valuable, source = side_bars(league_params)

WHY ONE FUNCTION. The choice "this league's own approved threshold, or the global bar" was written
three times (pipeline._generate_side_bets, src/integrity/side_edge_devig.thresholds and
src/research/edge_replay.legacy_side_tier), each with the global bar typed as a literal default
(`lp.get("sniper_th", 0.10)`). A tip tiered on that typed bar was indistinguishable from a tip tiered
on its league's own, approved threshold: same SNIPER word in the ledger, the alert and the dashboard.

Now the bar is the declared policy config.SIDE_GLOBAL_BARS (values unchanged - owner rule: no
threshold changes) and every caller learns WHICH bar tiered the tip: source "LEAGUE" only when both
of the league's own bars were used; "GLOBAL" whenever a global bar was (the league is unapproved,
dropped, absent from the params file, or lacks a bar). The rule itself is the one pipeline used since
2026-09-22: `approved` decides; an older params file without `approved` falls back to `not drop`.
"""
from __future__ import annotations

import config

LEAGUE = config.THRESHOLD_SOURCE_LEAGUE
GLOBAL = config.THRESHOLD_SOURCE_GLOBAL


def global_bars() -> tuple[float, float, float]:
    """(sniper, marksman, valuable) of the declared global policy."""
    g = config.SIDE_GLOBAL_BARS
    return float(g["sniper"]), float(g["marksman"]), float(g["valuable"])


def use_own(lp: dict | None) -> bool:
    """True when the league's own threshold is trusted: approved out of sample (or, for a params
    file written before `approved` existed, not dropped)."""
    lp = lp or {}
    approved = lp.get("approved")
    if approved is not None:
        return bool(approved)
    return not lp.get("drop", False)


def side_bars(lp: dict | None) -> tuple[float, float, float, str]:
    """(sniper_th, marksman_th, valuable_th, threshold_source) for one league's params entry
    (`{}` / None for a league the params file does not list)."""
    lp = lp or {}
    g_sn, g_mm, g_val = global_bars()
    if not use_own(lp):
        return g_sn, g_mm, g_val, GLOBAL
    own_sn, own_mm = lp.get("sniper_th"), lp.get("marksman_th")
    sn = float(own_sn) if own_sn is not None else g_sn
    mm = float(own_mm) if own_mm is not None else g_mm
    return sn, mm, g_val, (LEAGUE if own_sn is not None and own_mm is not None else GLOBAL)


def tier_for(edge: float, bars: tuple) -> str:
    """SNIPER / MARKSMAN / VALUABLE / AVOID for an edge against (sniper, marksman, valuable[, src])."""
    sn, mm, val = bars[0], bars[1], bars[2]
    if edge >= sn:
        return "SNIPER"
    if edge >= mm:
        return "MARKSMAN"
    if edge >= val:
        return "VALUABLE"
    return "AVOID"
