"""O/U 2.5 reference price: multi-book de-vigged consensus, >= 3 books, never one first-listed book (owner 2026-11-24)."""
from __future__ import annotations

import glob
import statistics
from pathlib import Path

import pytest

from src import market_edge as ME
from src import reference_price as R

ROOT = Path(__file__).resolve().parents[1]


def _b(key, over=None, under=None, line=2.5, extra=()):
    outs = [{"name": "Over", "point": line, "price": over}] if over is not None else []
    outs += [{"name": "Under", "point": line, "price": under}] if under is not None else []
    outs += list(extra)
    return {"key": key, "markets": [{"key": "totals", "outcomes": outs}]}


def test_consensus_is_the_median_power_devig_of_every_valid_book():
    books = [_b("leovegas_se", 2.2, 1.7), _b("unibet_nl", 2.0, 1.85), _b("williamhill", 2.1, 1.75),
             _b("pinnacle", 2.05, 1.83)]
    c = R.consensus(books)
    p = statistics.median(ME.power_devig(o, u) for o, u in [(2.2, 1.7), (2.0, 1.85), (2.1, 1.75), (2.05, 1.83)])
    assert c.n_books == 4 and c.label == "consensus:power_median:n=4"
    assert abs(c.over - 1 / p) < 1e-12 and abs(c.under - 1 / (1 - p)) < 1e-12
    assert (c.best_over, c.best_over_book) == (2.2, "leovegas_se")
    assert (c.best_under, c.best_under_book) == (1.85, "unibet_nl")
    assert (c.first_over, c.first_under, c.first_book) == (2.2, 1.7, "leovegas_se")   # comparison only


def test_fewer_than_three_books_means_no_reference_price():
    c = R.consensus([_b("williamhill", 2.0, 1.8), _b("pinnacle", 2.05, 1.83)])
    assert (c.over, c.under, c.label, c.n_books) == (None, None, None, 2)
    assert c.first_book == "williamhill"                      # recorded, never used as a fallback
    assert R.consensus([]).n_books == 0 and R.consensus([]).over is None


def test_only_real_unambiguous_two_sided_pairs_at_the_exact_line_count():
    good = [_b("a", 2.0, 1.85), _b("b", 2.05, 1.8), _b("c", 2.1, 1.78)]
    noise = [_b("one_sided", 2.0), _b("suspended", 1.0, 1.9), _b("other_line", 2.0, 1.8, line=2.25),
             _b("ambiguous", 2.0, 1.8, extra=[{"name": "Over", "point": 2.5, "price": 2.3}]),
             _b("huge_margin", 1.5, 1.5)]                       # overround 0.33 > market_edge_v2 overround_max
    assert R.consensus(noise + good).n_books == 3
    assert R.consensus(noise + good[:2]).over is None


def test_no_served_model_reads_the_reference_price():
    """Only the PRICE moves: no serving payload has a price-derived feature, so model probabilities are identical."""
    joblib = pytest.importorskip("joblib")
    files = sorted(glob.glob(str(ROOT / "models" / "model_v9_*.pkl")) + glob.glob(str(ROOT / "models" / "model_ht_*.pkl")))
    assert files

    def cols(p):
        if isinstance(p, dict):
            yield from (p.get("feature_cols") or [])
            for v in (p.get("tracks") or {}).values():
                yield from cols(v)
    for f in files:
        hit = sorted(c for c in set(cols(joblib.load(f))) if "odds" in c or "implied" in c or "overround" in c)
        assert not hit, (Path(f).name, hit)
