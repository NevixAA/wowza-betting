"""side_thresholds.py - THE tier bars production serves: the league's OWN, owner-applied bars, or none.

    bars, reason = own_bars(lp, market, league)       # ((sniper, marksman, valuable) | None, reason)
    tier_for(edge, bars)                              # SNIPER / MARKSMAN / VALUABLE / AVOID

OWNER RULE 2026-11-09 ("100% real data, no fakes, no bugs, no defaults, no global vars"). Until then a
league whose own threshold was not approved was tiered on a GLOBAL bar: config.SIDE_GLOBAL_BARS
(0.10 / 0.08 / 0.04) for BTTS / Over 1.5 / Over 3.5, and config.SNIPER_THRESHOLD_OVER/_UNDER,
MARKSMAN_THRESHOLD, VALUABLE_THRESHOLD and EDGE_CEILING (plus the hand-set LEAGUE_SNIPER_THRESHOLDS
table) for O/U 2.5. VALUABLE had no per-league value at all. Since 2026-11-09 serving has NO global
stand-in anywhere: a league x market tiers ONLY on its own three bars, else it gets no tier (AVOID,
reason NO_LEAGUE_BAR) - no tip, no stake, no message.

A LEAGUE'S OWN BARS are a params entry (models/best_params_standard.json for ou25, models/
best_params_side_markets.json[market] for the side markets) that
  * carries sniper_th, marksman_th AND valuable_th, all finite, ordered VALUABLE <= MARKSMAN <= SNIPER;
  * is not dropped (drop: true = that league x market is switched OFF); and
  * was written by src/threshold_apply, i.e. its entry id "<market>|<league>" is in the history of
    config/threshold_applied.json - the OWNER approved it (monthly proposals, owner approves, apply writes).
The entry's own `approved` flag (= statistically PROVEN under the strict rule) does NOT gate tiering:
UNPROVEN bars the owner applied are still that league's own real bars (owner 2026-11-08: every league has
its own bars). It is carried for display and staking only.

The old global constants stay in config.py as HISTORICAL values (what served before 2026-11-09) for the
research replays (src/research/edge_replay) and the audits; data truth rule 6 fails any serving module
that reads them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import config

LEAGUE = config.THRESHOLD_SOURCE_LEAGUE
#: tier reason of a league x market with no own, owner-applied bars (recorded, never a stand-in)
NO_LEAGUE_BAR = "NO_LEAGUE_BAR"
#: written by src/threshold_apply only (its history lists every applied entry id)
APPLIED_LEDGER = Path(config.BASE_DIR) / "config" / "threshold_applied.json"

_APPLIED_CACHE: dict = {}


def applied_entries(path: Path | None = None) -> frozenset:
    """Entry ids ("<market>|<league>") src/threshold_apply has applied, cached per file mtime.
    A missing or unreadable ledger = nothing applied (fail closed: no bars, never a stand-in)."""
    p = Path(path) if path is not None else APPLIED_LEDGER
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return frozenset()
    hit = _APPLIED_CACHE.get(str(p))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        hist = json.loads(p.read_text(encoding="utf-8")).get("history") or []
    except Exception:                                                 # noqa: BLE001 - unreadable = none applied
        hist = []
    ids = frozenset(str(e) for h in hist if isinstance(h, dict) for e in (h.get("entries") or []))
    _APPLIED_CACHE[str(p)] = (mtime, ids)
    return ids


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def own_bars(lp: dict | None, market: str, league: str,
             applied: frozenset | None = None) -> tuple[tuple[float, float, float] | None, str]:
    """((sniper, marksman, valuable), "") when `league` x `market` has its own owner-applied bars,
    else (None, reason detail). `lp` = that league's params entry ({} / None when the file lacks it)."""
    lp = lp or {}
    if not lp:
        return None, f"{NO_LEAGUE_BAR}: no {market} params entry for {league}"
    if lp.get("drop") is True:
        return None, f"{NO_LEAGUE_BAR}: {market} switched OFF for {league} (drop)"
    sn, mm, val = _num(lp.get("sniper_th")), _num(lp.get("marksman_th")), _num(lp.get("valuable_th"))
    missing = [k for k, v in (("sniper_th", sn), ("marksman_th", mm), ("valuable_th", val)) if v is None]
    if missing:
        return None, f"{NO_LEAGUE_BAR}: {market} {league} lacks {', '.join(missing)}"
    if not (val <= mm <= sn):
        return None, f"{NO_LEAGUE_BAR}: {market} {league} bars not ordered (valuable<=marksman<=sniper)"
    ids = applied if applied is not None else applied_entries()
    if f"{market}|{league}" not in ids:
        return None, f"{NO_LEAGUE_BAR}: {market}|{league} not applied by src/threshold_apply (owner)"
    return (sn, mm, val), ""


def side_bars(lp: dict | None, market: str, league: str):
    """(sniper, marksman, valuable, "LEAGUE") of the league's own bars, or None (no tier)."""
    bars, _ = own_bars(lp, market, league)
    return None if bars is None else (*bars, LEAGUE)


def tier_for(edge: float, bars) -> str:
    """SNIPER / MARKSMAN / VALUABLE / AVOID for an edge against (sniper, marksman, valuable[, src]).
    No bars (None) or no edge = AVOID."""
    if bars is None:
        return "AVOID"
    try:
        e = float(edge)
    except (TypeError, ValueError):
        return "AVOID"
    if not math.isfinite(e):
        return "AVOID"
    sn, mm, val = bars[0], bars[1], bars[2]
    if e >= sn:
        return "SNIPER"
    if e >= mm:
        return "MARKSMAN"
    if e >= val:
        return "VALUABLE"
    return "AVOID"
