"""O/U 2.5 bet price: the MEDIAN REAL PRICE across bookmakers (owner 2026-11-24 decision on #104).
=================================================================================================

Owner 2026-11-24: "as many bookmakers as we can have"; then, deciding #104 - "Median real price": tips are recorded,
scored and qualified at the MEDIAN of the real (vigged) prices across all books, minimum 3 two-sided books.

  * VALID BOOK: quotes BOTH sides at the exact 2.5 line, each price > 1.0, one unambiguous price per side, overround
    inside market_edge_v2's [overround_min, overround_max] (config/market_edge_v2.json) and de-viggable.
  * BET PRICE (odds_over25 / odds_under25): per side, the median of the valid books' real prices - only with
    >= MIN_BOOKS (3) books. Fewer -> NO price (the match stays on the board and in the learning ledger with an empty
    price; never tipped or staked; no fallback). Label (odds_over25_book) = src.btts_odds.book_label: "median:<books>".
  * FAIR REFERENCE (fair_over25 / fair_under25): the de-vigged consensus - per book src.market_edge.power_devig
    (reused, not forked), median fair P(Over), 1 / P. Shown beside the bet price; never used to qualify or score.
  * COMPARISON ONLY: the best price per side + its book, and the first-listed book's pair.
No served model reads odds_over25 / odds_under25 as a feature (tests/test_reference_price.py checks every payload).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import NamedTuple

from src import market_edge as ME

ROOT = Path(__file__).resolve().parents[1]
MIN_BOOKS = 3


class Reference(NamedTuple):
    over: float | None            # BET price: median real Over 2.5 price (>= MIN_BOOKS books) else None
    under: float | None
    label: str | None             # "median:<sorted books>" or None
    n_books: int                  # valid two-sided books (also when below MIN_BOOKS)
    fair_over: float | None       # de-vigged power-median consensus (reference only)
    fair_under: float | None
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


def reference(bookmakers: list, line: float = 2.5, min_books: int = MIN_BOOKS) -> Reference:
    from src.btts_odds import book_label
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
        return Reference(None, None, None, fair_over=None, fair_under=None, **common)
    p = statistics.median(ME.power_devig(o, u) for _, o, u in valid)
    return Reference(float(statistics.median(o for _, o, _ in valid)), float(statistics.median(u for *_, u in valid)),
                     book_label([b for b, _, _ in valid]), fair_over=1.0 / p, fair_under=1.0 / (1.0 - p), **common)
