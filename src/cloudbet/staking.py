"""How much to stake. Owner's setting: 4.5% of the CURRENT bankroll, flat.

    from src.cloudbet.staking import stake_for, StakingRules

WHAT 4.5% IS, MEASURED ON THIS ESTATE'S OWN RESULTS (output/staking_study.json):

    optimal flat fraction          2.2%   (bootstrap median 2.1%)
    log-growth turns negative at   4.5%   <- the setting
    log-growth at 4.5%             -0.00002   i.e. exactly the break-even line
    log-growth at 7.5%             -0.00129   the originally proposed figure

So 4.5% is the largest fraction at which the bankroll neither compounds nor decays. Above it the
MEDIAN path shrinks even when the arithmetic ROI is positive, because a 4.5% loss needs more
than a 4.5% win to undo. This is recorded here rather than argued: the number is the owner's
decision, and the code implements it.

FLOORS AND CEILINGS ARE NOT OPINIONS, THEY ARE FAILURE MODES:

  - stake is computed from the CURRENT bankroll, so a losing run automatically reduces the
    stake. Fixed-dollar staking does the opposite and is what produced a 26% ruin probability in
    the study.
  - `max_fraction_per_day` caps total daily exposure. Twenty bets at 4.5% is 90% of the bankroll
    at risk in one day, which no staking plan survives if a slate correlates — and slates do
    correlate, which is why the study bootstrapped by matchday rather than by bet.
  - a stake below `min_stake` is skipped rather than rounded up, because rounding up is how a
    small bankroll quietly starts betting a larger fraction than configured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class StakingRules:
    #: Owner's setting, 2026-10-05. Measured break-even fraction for this bet set.
    fraction: float = 0.045
    #: Total fraction of bankroll that may be at risk across one day. 20 bets x 4.5% = 90%
    #: unconstrained; bets on one slate are correlated, so the cap is what stops a single bad
    #: matchday from being a bankroll event.
    max_fraction_per_day: float = 0.30
    min_stake: float = 1.0
    max_stake: float | None = None
    #: Props are longer-odds and higher-variance; this scales their stake relative to a team
    #: bet. 1.0 means treat them identically.
    prop_multiplier: float = 1.0


def stake_for(bankroll: float, rules: StakingRules | None = None,
              staked_today: float = 0.0, is_prop: bool = False) -> tuple[float, str]:
    """Stake for one bet, and the reason if it is zero.

    Returns (stake, reason). A zero stake is never an error — it is the daily cap or the floor
    doing its job, and the caller logs the reason rather than silently placing nothing.
    """
    r = rules or StakingRules()
    if bankroll <= 0:
        return 0.0, "bankroll is zero or negative"

    stake = bankroll * r.fraction * (r.prop_multiplier if is_prop else 1.0)

    room = bankroll * r.max_fraction_per_day - staked_today
    if room <= 0:
        return 0.0, (f"daily exposure cap reached "
                     f"({r.max_fraction_per_day:.0%} of bankroll already staked)")
    if stake > room:
        # Trim rather than skip: a partial stake on the best remaining edge beats nothing.
        log.info(f"stake trimmed {stake:.2f} -> {room:.2f} by the daily exposure cap")
        stake = room

    if r.max_stake is not None:
        stake = min(stake, r.max_stake)
    if stake < r.min_stake:
        # NOT rounded up. Rounding up is how a shrinking bankroll quietly starts betting a
        # larger fraction than configured.
        return 0.0, f"stake {stake:.2f} is below the {r.min_stake:.2f} floor"

    return round(stake, 2), ""


def describe(bankroll: float, rules: StakingRules | None = None) -> str:
    r = rules or StakingRules()
    per = bankroll * r.fraction
    return (f"bankroll {bankroll:,.2f} | {r.fraction:.1%} per bet = {per:,.2f} | "
            f"daily exposure cap {r.max_fraction_per_day:.0%} = {bankroll * r.max_fraction_per_day:,.2f} "
            f"({int(bankroll * r.max_fraction_per_day // per) if per else 0} bets at full size)")
