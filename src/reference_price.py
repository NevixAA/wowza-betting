"""O/U 2.5 tip REFERENCE PRICE: a multi-book, de-vigged, two-sided consensus (owner 2026-11-24).
==============================================================================================

Owner direction 2026-11-24: "right now I'm not betting anything ... as many bookmakers as we can have". The board's
O/U 2.5 reference price is no longer one book's quote (before: the first book OddsAPI happened to list). It is:

  * EVERY book in the response that quotes BOTH sides at the exact 2.5 line, each price > 1.0, one unambiguous
    price per side, overround inside market_edge_v2's [overround_min, overround_max] (config/market_edge_v2.json)
  * de-vigged per book with src.market_edge.power_devig (the engine's own helper, reused - not forked)
  * the MEDIAN of the per-book fair P(Over) across those books
  * only when at least MIN_BOOKS (3) books pass; fewer -> no reference price (missing stays missing, never a
    fallback to one book or an invented price)

Recorded beside it: n_books, the BEST available price per side + its bookmaker (executable), and the FIRST-LISTED
book's pair (comparison only, never a basis). The consensus fair price is 1 / median fair probability: a market
estimate, not a bookmaker's offer; its label is "consensus:power_median:n=<k>", never a book name.
No served model reads odds_over25 / odds_under25 as a feature (tests/test_reference_price.py checks every payload),
so model probabilities do not move; only edges and therefore which rows clear a bar.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import NamedTuple

from src import market_edge as ME

ROOT = Path(__file__).resolve().parents[1]
MIN_BOOKS = 3
METHOD = "power_median"


class Consensus(NamedTuple):
    over: float | None            # consensus fair price, Over 2.5 (1 / median fair P)
    under: float | None
    label: str | None             # "consensus:power_median:n=<k>" or None
    n_books: int                  # valid two-sided books (also when below MIN_BOOKS)
    best_over: float | None
    best_over_book: str | None
    best_under: float | None
    best_under_book: str | None
    first_over: float | None      # first-listed book's pair - comparison only
    first_under: float | None
    first_book: str | None


def _overround_bounds() -> tuple[float, float]:
    try:
        p = json.loads((ROOT / "config" / "market_edge_v2.json").read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        p = {}
    return (float(p.get("overround_min", ME.DEFAULTS["overround_min"])),
            float(p.get("overround_max", ME.DEFAULTS["overround_max"])))


def book_pairs(bookmakers: list, line: float = 2.5, market: str = "totals") -> list:
    """[(book, over, under)] in the provider's listing order, every book with one unambiguous real price per side."""
    out = []
    for bm in bookmakers or []:
        key = bm.get("key")
        if not key:
            continue
        seen: dict = {"Over": set(), "Under": set()}
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market:
                continue
            for o in mkt.get("outcomes", []):
                if o.get("point") == line and o.get("name") in seen and ME._valid_price(o.get("price")):
                    seen[o["name"]].add(float(o["price"]))
        if len(seen["Over"]) == 1 and len(seen["Under"]) == 1:
            out.append((key, next(iter(seen["Over"])), next(iter(seen["Under"]))))
    return out


def consensus(bookmakers: list, line: float = 2.5, min_books: int = MIN_BOOKS) -> Consensus:
    pairs = book_pairs(bookmakers, line)
    first = pairs[0] if pairs else (None, None, None)
    lo, hi = _overround_bounds()
    valid = [(b, o, u) for b, o, u in pairs
             if (ovr := ME.overround(o, u)) is not None and lo <= ovr <= hi and ME.power_devig(o, u) is not None]
    n = len(valid)
    bo = max(valid, key=lambda t: t[1]) if valid else None
    bu = max(valid, key=lambda t: t[2]) if valid else None
    common = dict(n_books=n, best_over=bo[1] if bo else None, best_over_book=bo[0] if bo else None,
                  best_under=bu[2] if bu else None, best_under_book=bu[0] if bu else None,
                  first_over=first[1], first_under=first[2], first_book=first[0])
    if n < min_books:
        return Consensus(None, None, None, **common)
    p = statistics.median(ME.power_devig(o, u) for _, o, u in valid)
    return Consensus(1.0 / p, 1.0 / (1.0 - p), f"consensus:{METHOD}:n={n}", **common)
