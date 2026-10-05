"""§2 — the ONE place that answers "what gate is actually running for this cell?".

    from src.gate_resolver import resolve_effective_tier_gates
    g = resolve_effective_tier_gates("Championship", "ou25", "standard")
    g.sniper, g.marksman, g.valuable          # the numbers production really uses
    g.sniper_source                           # and where each one came from

WHY THIS EXISTS. A research script that reconstructs production thresholds from `config.py`
gets the wrong answer, and did: the per-league threshold study published on 2026-10-05 reported
Championship's SNIPER gate as 0.15 when production actually runs 0.07, because
`models/best_params_standard.json` carries an APPROVED auto-optimised value that
`src/betting.py` prefers over the hand-set table. Every number in that study's "current" column
was therefore suspect for any approved league.

This module mirrors `src/betting.py::_base_tier` and `pipeline.py::_generate_side_bets` exactly.
It is READ-ONLY: it never writes a threshold, never changes a tier, and importing it cannot
affect production. If production's logic changes, this must change with it — which is why the
tests assert against the real artifacts rather than against hard-coded expectations.

THE TWO PATHS ARE GENUINELY DIFFERENT, and conflating them is a measurement error:

    main O/U (ou25)     betting.py::_base_tier
                        approved best_params_standard -> LEAGUE_SNIPER_THRESHOLDS (capped)
                        -> side-specific global. EDGE_CEILING applies ONLY when no per-league
                        value was found. VALUABLE = config.VALUABLE_THRESHOLD.

    side markets        pipeline.py::_generate_side_bets
                        approved best_params_side_markets[target] -> fixed 0.10 / 0.08.
                        VALUABLE is HARD-CODED 0.04 and does NOT follow
                        config.VALUABLE_THRESHOLD.

So on a production run with VALUABLE_THRESHOLD=0.03, a main O/U bet becomes VALUABLE at 3% while
a BTTS bet on the same fixture needs 4%. That is not a bug to fix here — it is a fact any study
comparing markets has to respect.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config

log = logging.getLogger(__name__)

MAIN_MARKET = "ou25"
SIDE_MARKETS = ("btts", "over15", "over35")

#: Hard-coded in pipeline.py::_generate_side_bets. Not read from config, deliberately recorded.
SIDE_GLOBAL_SNIPER = 0.10
SIDE_GLOBAL_MARKSMAN = 0.08
SIDE_VALUABLE_HARDCODED = 0.04


@dataclass
class Gates:
    league: str
    market: str
    model_type: str
    valuable: float
    marksman: float
    sniper: float
    valuable_source: str
    marksman_source: str
    sniper_source: str
    workflow_override_applied: bool = False
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


def _is_approved(lp: dict) -> bool:
    """Production's own rule, reproduced including its back-compat branch.

    `approved` present -> use it. Absent (an older artifact) -> fall back to `not drop`. This
    matters: an older file with drop=False but no approved flag IS used by production, and a
    study that required `approved is True` would mis-report those leagues as un-optimised.
    """
    if not isinstance(lp, dict):
        return False
    a = lp.get("approved")
    return bool(a) if a is not None else (not lp.get("drop", False))


def resolve_effective_tier_gates(league: str, market: str = MAIN_MARKET,
                                 model_type: str | None = None,
                                 side: str = "") -> Gates:
    """The gates production would actually apply to this cell, with provenance for each.

    `side` only matters on the main market and only when no per-league value exists, because
    that is the one branch where SNIPER_THRESHOLD_OVER / _UNDER are consulted.
    """
    league = str(league)
    model_type = model_type or config.model_type_for_league(league)
    notes: list[str] = []

    if market in SIDE_MARKETS:
        lp = (_side_params().get(market) or {}).get(league) or {}
        approved = _is_approved(lp) and lp.get("sniper_th") is not None
        if approved:
            sniper = float(lp["sniper_th"])
            marksman = float(lp.get("marksman_th", SIDE_GLOBAL_MARKSMAN))
            src = f"best_params_side_markets[{market}][{league}] (approved)"
            s_src = m_src = src
        else:
            sniper, marksman = SIDE_GLOBAL_SNIPER, SIDE_GLOBAL_MARKSMAN
            s_src = m_src = "pipeline.py fixed side-market global"
            if lp:
                notes.append("an optimiser value exists for this cell but is NOT approved, so "
                             "production ignores it")
        return Gates(
            league=league, market=market, model_type=model_type,
            # NOT config.VALUABLE_THRESHOLD. Hard-coded in _generate_side_bets.
            valuable=SIDE_VALUABLE_HARDCODED, marksman=marksman, sniper=sniper,
            valuable_source="pipeline.py hard-coded 0.04 (does NOT follow VALUABLE_THRESHOLD)",
            marksman_source=m_src, sniper_source=s_src,
            approved_optimizer_applied=approved,
            workflow_override_applied=False,
            regime_id=regime_id(), notes=notes)

    # ── main O/U ──────────────────────────────────────────────────────────────────────────────
    lp = _standard_params().get(league) or {}
    approved = _is_approved(lp) and lp.get("sniper_th") is not None
    cap = float(__import__("os").getenv("LEAGUE_SNIPER_CAP", "1.0"))
    cap_applied = False
    has_per_league = False

    if approved:
        sniper = float(lp["sniper_th"])
        s_src = f"best_params_standard[{league}] (approved, OOS-positive)"
        has_per_league = True
    elif league in config.LEAGUE_SNIPER_THRESHOLDS:
        calibrated = config._LEAGUE_SNIPER_CALIBRATED.get(league)
        sniper = config.LEAGUE_SNIPER_THRESHOLDS[league]
        has_per_league = True
        cap_applied = calibrated is not None and sniper < calibrated
        s_src = "config.LEAGUE_SNIPER_THRESHOLDS" + (
            f" (calibrated {calibrated:.2f}, CAPPED to {sniper:.2f} by LEAGUE_SNIPER_CAP={cap})"
            if cap_applied else "")
        if lp and not approved:
            notes.append("an optimiser value exists for this league but is NOT approved "
                         "(OOS-negative), so production uses the hand-set value")
    elif side.upper() == "OVER":
        sniper, s_src = config.SNIPER_THRESHOLD_OVER, "config.SNIPER_THRESHOLD_OVER (global)"
    elif side.upper() == "UNDER":
        sniper, s_src = config.SNIPER_THRESHOLD_UNDER, "config.SNIPER_THRESHOLD_UNDER (global)"
    else:
        sniper, s_src = config.SNIPER_THRESHOLD, "config.SNIPER_THRESHOLD (global)"

    if approved and lp.get("marksman_th") is not None:
        marksman = float(lp["marksman_th"])
        m_src = f"best_params_standard[{league}] (approved)"
    elif getattr(config, "LEAGUE_MARKSMAN_THRESHOLDS", {}).get(league) is not None:
        marksman = float(config.LEAGUE_MARKSMAN_THRESHOLDS[league])
        m_src = "config.LEAGUE_MARKSMAN_THRESHOLDS"
    else:
        marksman = float(config.MARKSMAN_THRESHOLD)
        m_src = "config.MARKSMAN_THRESHOLD (global, env-overridable)"

    # The ceiling is NOT a tier bound — it DEMOTES an implausibly large edge to MARKSMAN, and
    # only for leagues with no per-league calibration. Recorded because a study that ignores it
    # will mis-label the top of the edge distribution in exactly those leagues.
    ceiling = (getattr(config, "LEAGUE_EDGE_CEILING", {}).get(league, config.EDGE_CEILING)
               if hasattr(config, "LEAGUE_EDGE_CEILING") else config.EDGE_CEILING)

    return Gates(
        league=league, market=market, model_type=model_type,
        valuable=float(config.VALUABLE_THRESHOLD), marksman=marksman, sniper=sniper,
        valuable_source="config.VALUABLE_THRESHOLD (global, env-overridable)",
        marksman_source=m_src, sniper_source=s_src,
        approved_optimizer_applied=approved,
        workflow_override_applied=_env_overridden(),
        league_cap_applied=cap_applied,
        edge_ceiling=float(ceiling) if ceiling is not None else None,
        edge_ceiling_active=not has_per_league,
        regime_id=regime_id(), notes=notes)


def _env_overridden() -> bool:
    import os
    return any(os.getenv(k) is not None for k in
               ("LEAGUE_SNIPER_CAP", "MARKSMAN_THRESHOLD", "VALUABLE_THRESHOLD",
                "SNIPER_THRESHOLD"))


def regime_id() -> str:
    """A short hash of everything that determines a gate.

    Two results computed under different regimes are not comparable, and this is what lets a
    stored result say which regime produced it instead of being silently re-interpreted later.
    """
    import hashlib
    import os
    payload = {
        "env": {k: os.getenv(k) for k in
                ("LEAGUE_SNIPER_CAP", "MARKSMAN_THRESHOLD", "VALUABLE_THRESHOLD",
                 "SNIPER_THRESHOLD", "EDGE_CEILING")},
        "standard": _standard_params(),
        "side": _side_params(),
        "league_sniper": dict(config.LEAGUE_SNIPER_THRESHOLDS),
        "league_marksman": dict(getattr(config, "LEAGUE_MARKSMAN_THRESHOLDS", {})),
        "valuable": float(config.VALUABLE_THRESHOLD),
        "marksman": float(config.MARKSMAN_THRESHOLD),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


def all_gates(markets=(MAIN_MARKET,) + SIDE_MARKETS) -> list[Gates]:
    """Every league x market cell production can produce, with its effective gates."""
    leagues = sorted(set(config.ENABLED_LEAGUES) | set(config.LEAGUE_SNIPER_THRESHOLDS))
    return [resolve_effective_tier_gates(lg, m) for lg in leagues for m in markets]
