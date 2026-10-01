"""§11 — the market baseline, and an honest answer when there isn't one.

    from src.market_anchor import anchor
    a = anchor(over_quotes={"bet365": 1.91, "pinnacle": 1.95}, under_quotes={...})
    a.probability   # de-vigged P(over), or None
    a.status        # EXCHANGE_FAIR | CROSS_BOOK_MEDIAN | SINGLE_BOOK | INSUFFICIENT_MARKET_DATA

THE HIERARCHY, best first:

    1. EXCHANGE_FAIR          an exchange price, commission-adjusted — closest to a true market
    2. CROSS_BOOK_MEDIAN      median de-vigged probability across >= 3 books
    3. SINGLE_BOOK            one book's two-sided market, de-vigged
    4. INSUFFICIENT_MARKET_DATA   anything else

WHY THE FOURTH STATUS EXISTS AND MUST NEVER BE SKIPPED. A one-sided price cannot be de-vigged:
with only the over quote you cannot separate the bookmaker's margin from their opinion, so any
"fair probability" derived from it is really `1/odds` with the vig silently left in — biased
high by roughly half the overround, every time, in the same direction. An edge computed against
that is an artifact of the margin.

The estate has already paid for this exact mistake once in a different form. `over15` was
certified at +13.5% ROI on 12,186 of 12,187 rows priced at a CONSTANT 1.40 default, which made
"edge" a bare model-probability threshold with no market in it at all; those numbers went on to
authorise real markets in league_roi_config.json. A fabricated anchor is worse than no anchor,
because no anchor is visible and a fabricated one is not.

So a one-sided market returns `probability=None` and status INSUFFICIENT_MARKET_DATA. Callers
must treat that as OBSERVE — log it, never price against it.

DE-VIG METHOD. Power by default, proportional available. Proportional divides out the overround
evenly, which systematically over-prices favourites and under-prices longshots; power solves for
the exponent k where sum(p_i^k) = 1, which matches observed bookmaker margin shape better. On a
two-outcome market the difference is small but it is one-directional, and this estate bets a lot
of short prices where a one-directional bias accumulates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

EXCHANGE_FAIR = "EXCHANGE_FAIR"
CROSS_BOOK_MEDIAN = "CROSS_BOOK_MEDIAN"
SINGLE_BOOK = "SINGLE_BOOK"
INSUFFICIENT = "INSUFFICIENT_MARKET_DATA"

#: Books needed before a median is preferred to a single quote.
MIN_BOOKS_FOR_MEDIAN = 3
#: Overround outside this band means the pair is stale, mismatched or mis-parsed. A 1.40/1.40
#: pair implies a 43% margin and is not a market; a sub-1.0 pair is arbitrage or a bad parse.
MIN_OVERROUND, MAX_OVERROUND = 1.00, 1.25
#: Typical exchange commission, used to turn a traded price into a fair one.
EXCHANGE_COMMISSION = 0.02


@dataclass(frozen=True)
class Anchor:
    probability: float | None
    status: str
    n_books: int
    overround: float | None
    method: str
    detail: str = ""

    @property
    def usable(self) -> bool:
        """True only when a real two-sided market produced this. Check it before pricing."""
        return self.probability is not None and self.status != INSUFFICIENT


def power_devig(odds: list[float], tol: float = 1e-10, max_iter: int = 100) -> list[float]:
    """Solve sum((1/o_i)^k) = 1 for k. Returns fair probabilities summing to 1.

    Power rather than proportional because the bookmaker's margin is not spread evenly across
    outcomes — it is heavier on longshots. Proportional de-vig therefore leaves favourites too
    cheap and longshots too dear, in the same direction every time.
    """
    raw = np.array([1.0 / o for o in odds if o and o > 1.0], dtype=float)
    if len(raw) < 2:
        return []
    lo, hi = 0.2, 3.0
    for _ in range(max_iter):
        k = (lo + hi) / 2
        s = float(np.sum(raw ** k))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k            # need to shrink more
        else:
            hi = k
    return list(raw ** k / np.sum(raw ** k))


def proportional_devig(odds: list[float]) -> list[float]:
    raw = np.array([1.0 / o for o in odds if o and o > 1.0], dtype=float)
    return list(raw / raw.sum()) if len(raw) >= 2 else []


def anchor(over_quotes: dict[str, float] | None = None,
           under_quotes: dict[str, float] | None = None,
           exchange_over: float | None = None,
           exchange_under: float | None = None,
           method: str = "power") -> Anchor:
    """Best available market probability for the OVER side, or an honest refusal.

    `over_quotes` / `under_quotes` map bookmaker -> decimal odds. A book contributes only when
    it quotes BOTH sides: a market is two-sided or it is not a market.
    """
    over_quotes = {k: v for k, v in (over_quotes or {}).items() if v and v > 1.0}
    under_quotes = {k: v for k, v in (under_quotes or {}).items() if v and v > 1.0}
    devig = power_devig if method == "power" else proportional_devig

    # 1. exchange, commission-adjusted
    if exchange_over and exchange_under and exchange_over > 1.0 and exchange_under > 1.0:
        eo = 1.0 + (exchange_over - 1.0) * (1 - EXCHANGE_COMMISSION)
        eu = 1.0 + (exchange_under - 1.0) * (1 - EXCHANGE_COMMISSION)
        ov = 1 / eo + 1 / eu
        if MIN_OVERROUND <= ov <= MAX_OVERROUND:
            p = devig([eo, eu])
            return Anchor(round(p[0], 6), EXCHANGE_FAIR, 1, round(ov, 5), method,
                          "exchange price, commission-adjusted")

    # a book counts only if it prices BOTH sides
    both = sorted(set(over_quotes) & set(under_quotes))
    pairs = []
    for b in both:
        ov = 1 / over_quotes[b] + 1 / under_quotes[b]
        if MIN_OVERROUND <= ov <= MAX_OVERROUND:
            pairs.append((b, over_quotes[b], under_quotes[b], ov))

    if not pairs:
        one_sided = len(over_quotes) > 0 or len(under_quotes) > 0
        return Anchor(None, INSUFFICIENT, 0, None, method,
                      "one-sided quotes only — a fair probability cannot be separated from the "
                      "margin, so none is returned" if one_sided else "no usable quotes")

    # 2. cross-book median of per-book de-vigged probabilities
    if len(pairs) >= MIN_BOOKS_FOR_MEDIAN:
        ps = [devig([o, u])[0] for _, o, u, _ in pairs]
        return Anchor(round(float(np.median(ps)), 6), CROSS_BOOK_MEDIAN, len(pairs),
                      round(float(np.mean([p[3] for p in pairs])), 5), method,
                      f"median of {len(pairs)} two-sided books")

    # 3. single book
    b, o, u, ov = pairs[0]
    return Anchor(round(devig([o, u])[0], 6), SINGLE_BOOK, len(pairs), round(ov, 5), method,
                  f"single two-sided book ({b})")
