"""O/U 2.5 entry book: a fixed, named order (config/entry_book_order.json), never the first book OddsAPI lists."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from src import entry_book as E

ROOT = Path(__file__).resolve().parents[1]


def _b(key, over=None, under=None, line=2.5, extra=()):
    outs = [{"name": "Over", "point": line, "price": over}] if over is not None else []
    outs += [{"name": "Under", "point": line, "price": under}] if under is not None else []
    outs += list(extra)
    return {"key": key, "markets": [{"key": "totals", "outcomes": outs}]}


def test_order_is_the_owner_rule():
    assert E.order() == ("bet365", "pinnacle", "williamhill", "onexbet", "ladbrokes_uk")


def test_first_listed_unapproved_book_is_never_the_entry_price():
    books = [_b("leovegas_se", 2.2, 1.7), _b("unibet_nl", 2.15, 1.72), _b("williamhill", 2.0, 1.8),
             _b("pinnacle", 2.05, 1.85)]
    assert E.pick_pair(books) == E.Pair(2.05, 1.85, "pinnacle")
    assert E.pick_pair(books[:3]) == E.Pair(2.0, 1.8, "williamhill")
    assert E.pick_pair([_b("pinnacle", 2.0, 1.8), _b("bet365", 1.95, 1.9)]) == E.Pair(1.95, 1.9, "bet365")


def test_no_listed_book_means_no_price():
    assert E.pick_pair([_b("leovegas_se", 2.2, 1.7), _b("betsson", 2.1, 1.75)]) == E.Pair(None, None, None)
    assert E.pick_pair([]) == E.Pair(None, None, None)


def test_same_book_pair_exact_line_and_real_prices_only():
    # one side only / a suspended 1.0 quote / another line / an ambiguous double quote: that book is skipped
    assert E.pick_pair([_b("pinnacle", 2.0), _b("williamhill", 2.1, 1.8)]).book == "williamhill"
    assert E.pick_pair([_b("pinnacle", 1.0, 1.9), _b("onexbet", 2.1, 1.8)]).book == "onexbet"
    assert E.pick_pair([_b("pinnacle", 2.0, 1.8, line=2.25), _b("onexbet", 2.1, 1.8)]).book == "onexbet"
    amb = _b("pinnacle", 2.0, 1.8, extra=[{"name": "Over", "point": 2.5, "price": 2.3}])
    assert E.pick_pair([amb, _b("ladbrokes_uk", 2.1, 1.8)]).book == "ladbrokes_uk"


def test_order_by_track_override(tmp_path):
    E._cfg.cache_clear()
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"order": ["bet365", "pinnacle", "williamhill"],
                             "order_by_track": {"standard": ["bet365", "pinnacle"]}}), encoding="utf-8")
    books = [_b("williamhill", 2.0, 1.8)]
    assert E.pick_pair(books, track="standard", path=p) == E.Pair(None, None, None)
    assert E.pick_pair(books, track="new_format", path=p).book == "williamhill"


def test_no_served_model_reads_the_entry_price():
    """The change moves only the PRICE: no serving payload has a price-derived feature, so model probabilities are
    identical whichever book prices the row."""
    joblib = pytest.importorskip("joblib")
    files = sorted(glob.glob(str(ROOT / "models" / "model_v9_*.pkl")) + glob.glob(str(ROOT / "models" / "model_ht_*.pkl")))
    assert files

    def cols(p):
        if isinstance(p, dict):
            yield from (p.get("feature_cols") or [])
            for v in (p.get("tracks") or {}).values():
                yield from cols(v)
    for f in files:
        feats = set(cols(joblib.load(f)))
        hit = sorted(c for c in feats if "odds" in c or "implied" in c or "overround" in c)
        assert not hit, (Path(f).name, hit)
