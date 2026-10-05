"""Which bets the Cloudbet bot takes, in which order, and how many.

    from src.cloudbet.selection import select, SelectionRules

THE OWNER'S RULES, as given:

    tier         SNIPER or MARKSMAN
    edge         above 5%
    order        DESCENDING by edge — best edge placed first
    daily cap    20 bets, hard
    markets      ou25, btts, over15, over35
    singles      one stake per selection

WHY ORDER MATTERS AND IS NOT COSMETIC. With a hard daily cap, ordering IS selection: the 21st
best bet is not placed. Sorting by edge descending means the cap removes the weakest candidates
rather than whatever happened to be scored last. It also means that if the bot dies halfway
through a run, what got placed is the best half, not a random half.

THREE GUARDS THAT ARE NOT IN THE OWNER'S SPEC BUT ARE REQUIRED FOR AN AUTOMATED PLACER:

1. ONE SIDE PER FIXTURE, FIRST TIP WINS. Measured on the real ledger: 14 fixtures since the
   August cutoff carry tips on BOTH sides, 1-6 days apart, because the model changes its mind as
   the price moves. Two of those had both sides at staked tiers. A human sees the second tip and
   ignores it; a bot would place both, pay the spread twice and guarantee a loss on the pair.
   First tip wins, matching the shadow log's freeze-at-first-sight discipline.

2. ONE BET PER SELECTION, EVER. Enforced locally against the placed ledger as well as by
   Cloudbet's referenceId dedup. Two independent layers because the failure is expensive and
   silent: a 5-minute workflow that re-runs after a timeout would otherwise re-stake.

3. PRICED AT CLOUDBET. A selection with no Cloudbet price is DROPPED, not placed at our price.
   Today edge is computed against Bet365 and a cross-book consensus, and Cloudbet is in neither;
   an edge against a price nobody will fill is not an edge.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

STAKED_TIERS = ("SNIPER", "MARKSMAN")
DEFAULT_MARKETS = ("ou25", "btts", "over15", "over35")
#: Player props, enabled by the owner on 2026-10-05 with a different bar: any POSITIVE edge,
#: not the 5% the team markets must clear. They compete for the same daily cap, so a prop only
#: displaces a team bet when its edge is larger.
PROP_MARKETS = ("goals", "assists", "sot", "sot2", "sot3", "cards")


@dataclass
class SelectionRules:
    min_edge: float = 0.05            # "only above 5% edge" — team markets
    #: Props clear a POSITIVE edge instead. Strictly greater than zero: an edge of exactly 0.0
    #: is the market's own price, which after vig is a losing bet.
    prop_min_edge: float = 0.0
    tiers: tuple = STAKED_TIERS
    markets: tuple = DEFAULT_MARKETS
    prop_markets: tuple = PROP_MARKETS
    #: Props are not tier-gated. The team tier ladder (SNIPER/MARKSMAN) is not computed for
    #: props — their `tier` column carries PAPER/VALUABLE from a different scheme entirely, so
    #: applying the team filter to them would reject every prop for the wrong reason.
    apply_tier_filter_to_props: bool = False
    max_bets_per_day: int = 20        # "up to 20 bets a day (no more)"
    one_side_per_fixture: bool = True
    #: How many bets may land on ONE match across different markets.
    #:
    #: BTTS YES, Over 1.5 and Over 2.5 on the same fixture are close to the same bet: if the
    #: match finishes 0-0 all three lose together. Observed on the 2026-10-05 board, two
    #: fixtures each attracted three markets, which at 4.5% a bet is 13.5% of bankroll on one
    #: 90-minute event. The daily exposure cap does not catch this, because it limits the DAY
    #: and this is concentration within it. None means no limit.
    max_bets_per_fixture: int | None = 2
    require_cloudbet_price: bool = True


@dataclass
class Rejected:
    """Why a candidate did not make it. Kept because the counts are the diagnostic.

    A bot that places 3 bets when you expected 20 is not obviously broken — without this you
    cannot tell "the board was thin" from "the market mapping silently matched nothing", and
    those need opposite fixes.
    """
    reasons: dict = field(default_factory=dict)

    def add(self, reason: str, n: int = 1):
        self.reasons[reason] = self.reasons.get(reason, 0) + n

    def __str__(self):
        return ", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items())) or "none"


def select(candidates: pd.DataFrame, rules: SelectionRules | None = None,
           already_placed: set[str] | None = None,
           placed_fixtures: set[str] | None = None) -> tuple[pd.DataFrame, Rejected]:
    """Apply the rules in order and return what to place, best edge first.

    `candidates` needs at minimum: fixture_id, market, side, edge, signal_tier, and
    cloudbet_price (NaN when the market was not found at Cloudbet).

    `already_placed` holds reference ids from the persisted ledger; `placed_fixtures` holds
    fixture ids already bet on any side today. Both are passed in rather than read here so the
    caller owns the durable state and this stays a pure function — which is what makes it
    testable without a filesystem.
    """
    r = rules or SelectionRules()
    rej = Rejected()
    already_placed = already_placed or set()
    placed_fixtures = placed_fixtures or set()

    d = candidates.copy()
    if d.empty:
        return d, rej

    is_prop = d["market"].astype(str).isin(r.prop_markets)

    # TIER: team markets only. Props carry a different tier vocabulary (PAPER/VALUABLE) from a
    # separate scheme, so applying the team ladder to them rejects every prop for the wrong
    # reason — it would read as "the model said no" when the model was never asked this question.
    n0 = len(d)
    tier_ok = d["signal_tier"].astype(str).str.upper().isin([t.upper() for t in r.tiers])
    d = d[tier_ok | (is_prop if not r.apply_tier_filter_to_props else False)]
    rej.add("tier", n0 - len(d))

    n0 = len(d)
    enabled = tuple(r.markets) + tuple(r.prop_markets)
    d = d[d["market"].astype(str).isin(enabled)]
    rej.add("market_not_enabled", n0 - len(d))

    # EDGE: two bars. Team markets must clear min_edge; props need only be positive.
    n0 = len(d)
    e = pd.to_numeric(d["edge"], errors="coerce")
    prop_row = d["market"].astype(str).isin(r.prop_markets)
    d = d[(prop_row & (e > r.prop_min_edge)) | (~prop_row & (e > r.min_edge))]
    rej.add(f"edge_below_bar(team>{r.min_edge:.0%},prop>{r.prop_min_edge:.0%})", n0 - len(d))

    if r.require_cloudbet_price:
        n0 = len(d)
        px = pd.to_numeric(d.get("cloudbet_price"), errors="coerce")
        d = d[px.notna() & (px > 1.0)]
        rej.add("no_cloudbet_price", n0 - len(d))

    if d.empty:
        return d, rej

    # SORT BEFORE the caps, so every cap below removes the WEAKEST candidates.
    d = d.sort_values("edge", ascending=False, kind="mergesort").reset_index(drop=True)

    # Already placed — reference-id level. Belt to Cloudbet's braces.
    if already_placed and "reference_id" in d.columns:
        n0 = len(d)
        d = d[~d["reference_id"].isin(already_placed)]
        rej.add("already_placed", n0 - len(d))

    # ONE SIDE PER FIXTURE. Within this run, the first (highest-edge) row for a fixture wins;
    # across runs, a fixture already bet is skipped entirely regardless of side.
    if r.one_side_per_fixture:
        n0 = len(d)
        d = d[~d["fixture_id"].astype(str).isin({str(f) for f in placed_fixtures})]
        rej.add("fixture_already_bet", n0 - len(d))
        # One row per fixture_id. Props are keyed per (player, market, date) rather than per
        # match, so several props on one fixture all survive — only a second SIDE of the same
        # team market is dropped. Keying props by match would have silently kept one prop per
        # game.
        n0 = len(d)
        d = d.drop_duplicates(subset=["fixture_id"], keep="first")
        rej.add("second_side_same_fixture", n0 - len(d))

    # CORRELATION CAP: at most N markets on one match, keeping the highest-edge ones. Applied
    # before the daily cap so the day's 20 slots are not spent on a handful of matches.
    if r.max_bets_per_fixture is not None and "match_key" in d.columns:
        n0 = len(d)
        d = d.groupby("match_key", sort=False).head(r.max_bets_per_fixture)
        rej.add("over_fixture_cap", n0 - len(d))

    # THE HARD DAILY CAP, applied last so it trims the weakest survivors.
    n0 = len(d)
    d = d.head(r.max_bets_per_day)
    rej.add("over_daily_cap", n0 - len(d))

    return d.reset_index(drop=True), rej


def remaining_today(placed_today: int, rules: SelectionRules | None = None) -> int:
    """How many more bets the cap allows. Never negative."""
    r = rules or SelectionRules()
    return max(0, r.max_bets_per_day - int(placed_today))
