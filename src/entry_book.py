"""Entry book for the O/U 2.5 board price: a FIXED, NAMED bookmaker order (config/entry_book_order.json).
=======================================================================================================

Before this module src/predict._fetch_odds_api kept the first Over 2.5 and the first Under 2.5 that OddsAPI happened
to list (often LeoVegas / Unibet / Nordicbet - books no bar was measured on). The owner's bars rule is one book's
two-sided pair from a named order: Bet365, then Pinnacle, then William Hill, 1xBet, Ladbrokes only as fallbacks.

pick_pair(bookmakers, line, track) -> Pair(over, under, book):
  * ONE BOOK'S PAIR: both sides from the same book (as the bars price, src.bars_opt.oddsapi_prices).
  * EXACT LINE: only the provider's 2.5 Over AND Under; no quarter or other line stands in.
  * NO PRICE <= 1.0 (a suspended-market quote is no price), and a book quoting two different prices for the same
    outcome is ambiguous and skipped.
  * FIRST BOOK IN THE ORDER that has a valid pair wins. No listed book -> Pair(None, None, None): no board price.
Only the price source changes: no served model reads odds_over25 / odds_under25 as a feature (checked in
tests/test_entry_book.py against every payload in models/).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
ORDER_PATH = ROOT / "config" / "entry_book_order.json"


class Pair(NamedTuple):
    over: float | None
    under: float | None
    book: str | None


@lru_cache(maxsize=4)
def _cfg(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def order(track: str | None = None, path: Path | None = None) -> tuple:
    c = _cfg(str(path or ORDER_PATH))
    return tuple((c.get("order_by_track") or {}).get(track) or c["order"])


def _price(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 1.0 else None


def book_pairs(bookmakers: list, line: float = 2.5, market: str = "totals") -> dict:
    """{book key: (over, under)} for every book quoting a valid, unambiguous two-sided pair at `line`."""
    out = {}
    for bm in bookmakers or []:
        key = bm.get("key")
        if not key:
            continue
        seen: dict = {"Over": set(), "Under": set()}
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market:
                continue
            for o in mkt.get("outcomes", []):
                if o.get("point") == line and o.get("name") in seen:
                    p = _price(o.get("price"))
                    if p is not None:
                        seen[o["name"]].add(p)
        if len(seen["Over"]) == 1 and len(seen["Under"]) == 1:
            out[key] = (next(iter(seen["Over"])), next(iter(seen["Under"])))
    return out


def pick_pair(bookmakers: list, line: float = 2.5, track: str | None = None, path: Path | None = None) -> Pair:
    pairs = book_pairs(bookmakers, line)
    for b in order(track, path):
        if b in pairs:
            return Pair(pairs[b][0], pairs[b][1], b)
    return Pair(None, None, None)
