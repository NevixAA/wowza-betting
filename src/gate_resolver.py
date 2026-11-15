"""§2 — the ONE place that answers "what gate is actually running for this cell?".

    from src.gate_resolver import resolve_effective_tier_gates
    g = resolve_effective_tier_gates("Championship", "ou25", "standard")
    g.own_bars                                # True = the league's own owner-applied bars serve
    g.sniper, g.marksman, g.valuable          # those bars (None when the cell has none)
    g.sniper_source                           # and where they came from (or the NO_LEAGUE_BAR reason)

WHY THIS EXISTS. A research script that reconstructs production thresholds from `config.py`
gets the wrong answer, and did: the per-league threshold study published on 2026-10-05 reported
Championship's SNIPER gate as 0.15 when production actually ran 0.07. Every caller that needs the
serving gates (staking, the research lock, the threshold proposals) reads them here.

This module mirrors `src/betting.py::_base_tier` / `evaluate_value` and
`pipeline.py::_generate_side_bets` exactly. It is READ-ONLY: it never writes a threshold, never
changes a tier, and importing it cannot affect production.

ONE RULE FOR EVERY MARKET SINCE 2026-11-09 (owner: "100% real data, no fakes, no bugs, no defaults, no
global vars"). A league x market (ou25, btts, over15, over35) is tiered ONLY on its own owner-applied
sniper / marksman / valuable bars (src/side_thresholds.own_bars). Without them the cell has NO gate:
own_bars False, the three bars None, the sources carry the NO_LEAGUE_BAR reason, and production gives
every fixture in it AVOID. Before that date the main O/U path fell back to the hand-set
LEAGUE_SNIPER_THRESHOLDS table and the global SNIPER/MARKSMAN/VALUABLE thresholds (with the
EDGE_CEILING demotion), and the side markets to a fixed 0.10 / 0.08 / 0.04 global bar. Those values
are historical (config.py) and no longer read here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config
from src.side_thresholds import own_bars

log = logging.getLogger(__name__)

MAIN_MARKET = "ou25"
SIDE_MARKETS = ("btts", "over15", "over35")


@dataclass
class Gates:
    league: str
    market: str
    model_type: str
    valuable: float | None
    marksman: float | None
    sniper: float | None
    valuable_source: str
    marksman_source: str
    sniper_source: str
    #: True = the league's own owner-applied bars serve this cell; False = NO_LEAGUE_BAR (no tier)
    own_bars: bool = False
    workflow_override_applied: bool = False
    #: kept for the readers of the pre-2026-11-09 schema: True exactly when own_bars is True
    approved_optimizer_applied: bool = False
    league_cap_applied: bool = False
    edge_ceiling: float | None = None
    edge_ceiling_active: bool = False
    regime_id: str = ""
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"{path.name} unreadable ({e}) — treating as absent, which is what "
                    f"production does too")
        return {}


def _standard_params() -> dict:
    return _read(config.MODELS_DIR / "best_params_standard.json")


def _side_params() -> dict:
    return _read(config.MODELS_DIR / "best_params_side_markets.json")


def _league_params(league: str, market: str) -> dict:
    if market == MAIN_MARKET:
        return _standard_params().get(league) or {}
    return (_side_params().get(market) or {}).get(league) or {}


def resolve_effective_tier_gates(league: str, market: str = MAIN_MARKET,
                                 model_type: str | None = None,
                                 side: str = "") -> Gates:
    """The gates production would actually apply to this cell, with provenance.

    `side` is kept for the callers (the research lock resolves OU25 OVER and UNDER separately); both
    sides of O/U 2.5 use the league's O/U 2.5 bars."""
    league = str(league)
    market = str(market).lower()
    model_type = model_type or config.model_type_for_league(league)
    notes: list[str] = []
    lp = _league_params(league, market)
    bars, reason = own_bars(lp, market, league)
    if bars is not None:
        sniper, marksman, valuable = bars
        pf = "best_params_standard" if market == MAIN_MARKET else f"best_params_side_markets[{market}]"
        src = f"{pf}[{league}] (owner-applied, threshold_apply)"
        status = lp.get("status") or ("PROVEN" if lp.get("approved") is True else "UNPROVEN")
        notes.append(f"bar status {status}")
    else:
        sniper = marksman = valuable = None
        src = reason
        if lp:
            notes.append("a params entry exists for this cell but is not usable as its own bars: " + reason)
    return Gates(
        league=league, market=market, model_type=model_type,
        valuable=valuable, marksman=marksman, sniper=sniper,
        valuable_source=src, marksman_source=src, sniper_source=src,
        own_bars=bars is not None,
        approved_optimizer_applied=bars is not None,
        workflow_override_applied=False, league_cap_applied=False,
        edge_ceiling=None, edge_ceiling_active=False,
        regime_id=regime_id(), notes=notes)


def regime_id() -> str:
    """A short hash of everything that determines a gate: the two params files and the owner-applied
    entry ids. Two results computed under different regimes are not comparable, and this is what
    lets a stored result say which regime produced it instead of being silently re-interpreted later."""
    import hashlib
    from src.side_thresholds import applied_entries
    payload = {"standard": _standard_params(), "side": _side_params(),
               "applied": sorted(applied_entries())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


def all_gates(markets=(MAIN_MARKET,) + SIDE_MARKETS) -> list[Gates]:
    """Every league x market cell production can produce, with its effective gates."""
    leagues = set(config.ENABLED_LEAGUES) | set(_standard_params())
    for m in SIDE_MARKETS:
        leagues |= set(_side_params().get(m) or {})
    return [resolve_effective_tier_gates(lg, m) for lg in sorted(leagues) for m in markets]
