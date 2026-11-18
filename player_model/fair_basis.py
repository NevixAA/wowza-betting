"""Where a priced props tip's fair probability came from (owner decision 2026-11-05).

A tip's `fair_basis` says how `fair_implied` was obtained:

    two_sided_devig     a REAL two-sided price: both sides of the bet (Yes/No, or Over/Under at the
                        tip's line) from the same bookmaker in the same answer, power de-vigged
                        (src.market_edge.power_devig), the median over books that quoted both.
    assumed_overround   only the tip's own side is quoted, so the fair probability is
                        (1 / odds) / an ASSUMED overround (player_model.config.OVERROUND_BY_ODDS).

Kept dependency-free so the Telegram senders and the dashboard can label a row without importing
the prediction stack. Props stay PAPER-only; this module changes no tier and no stake.
"""
from __future__ import annotations

ASSUMED_OVERROUND = "assumed_overround"
TWO_SIDED_DEVIG = "two_sided_devig"

#: Short, reader-facing labels (Telegram, the props tables, the dashboard).
LABEL = {TWO_SIDED_DEVIG: "two-sided price", ASSUMED_OVERROUND: "margin assumed"}


def label(basis) -> str | None:
    """The short label of a fair_basis value; None when the row has no (known) basis, e.g. unpriced."""
    if basis is None or not isinstance(basis, str):
        return None
    return LABEL.get(basis.strip())
