"""O/U 2.5 bet price = median REAL price of >= 3 two-sided books; de-vigged consensus beside it as the fair reference;
fewer books -> the match stays unpriced, never tipped (owner 2026-11-24 decision on #104)."""
from __future__ import annotations

import glob
import statistics
from pathlib import Path

import pandas as pd
import pytest

import config
from src import market_edge as ME
from src import reference_price as R

ROOT = Path(__file__).resolve().parents[1]


def _b(key, over=None, under=None, line=2.5, extra=()):
    outs = [{"name": "Over", "point": line, "price": over}] if over is not None else []
    outs += [{"name": "Under", "point": line, "price": under}] if under is not None else []
    outs += list(extra)
    return {"key": key, "markets": [{"key": "totals", "outcomes": outs}]}


PAIRS = [("leovegas_se", 2.2, 1.7), ("unibet_nl", 2.0, 1.85), ("williamhill", 2.1, 1.75), ("pinnacle", 2.05, 1.83)]


def test_bet_price_is_the_median_real_price_and_fair_is_the_devigged_consensus():
    c = R.reference([_b(*t) for t in PAIRS])
    assert c.n_books == 4 and c.label == "median:leovegas_se;pinnacle;unibet_nl;williamhill"
    assert c.over == statistics.median([2.2, 2.0, 2.1, 2.05]) and c.under == statistics.median([1.7, 1.85, 1.75, 1.83])
    p = statistics.median(ME.power_devig(o, u) for _, o, u in PAIRS)
    assert abs(c.fair_over - 1 / p) < 1e-12 and abs(c.fair_under - 1 / (1 - p)) < 1e-12
    assert (c.best_over, c.best_over_book, c.best_under, c.best_under_book) == (2.2, "leovegas_se", 1.85, "unibet_nl")
    assert (c.first_over, c.first_under, c.first_book) == (2.2, 1.7, "leovegas_se")    # comparison only


def test_fewer_than_three_books_means_no_price_and_no_fallback():
    c = R.reference([_b("williamhill", 2.0, 1.8), _b("pinnacle", 2.05, 1.83)])
    assert (c.over, c.under, c.label, c.fair_over, c.n_books) == (None, None, None, None, 2)
    assert c.first_book == "williamhill"
    assert R.reference([]).n_books == 0 and R.reference([]).over is None


def test_only_real_unambiguous_two_sided_pairs_at_the_exact_line_count():
    good = [_b("a", 2.0, 1.85), _b("b", 2.05, 1.8), _b("c", 2.1, 1.78)]
    noise = [_b("one_sided", 2.0), _b("suspended", 1.0, 1.9), _b("other_line", 2.0, 1.8, line=2.25),
             _b("ambiguous", 2.0, 1.8, extra=[{"name": "Over", "point": 2.5, "price": 2.3}]),
             _b("huge_margin", 1.5, 1.5)]                       # overround 0.33 > market_edge_v2 overround_max
    assert R.reference(noise + good).n_books == 3
    assert R.reference(noise + good[:2]).over is None


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, p):
        self._p = p

    def json(self):
        return self._p


def test_board_keeps_a_below_three_books_match_unpriced_and_never_tips_it(tmp_path, monkeypatch):
    import requests

    import src.btts_odds as B
    import src.predict as P
    import src.rolling_store as RS
    from src import betting
    league = sorted(set(config.ODDS_API_SPORT_KEYS) & set(config.ENABLED_LEAGUES))[0]
    monkeypatch.setattr(config, "ODDS_API_SPORT_KEYS", {league: config.ODDS_API_SPORT_KEYS[league]})
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(B, "fetch", lambda rows: rows)
    monkeypatch.setattr(RS, "roll", lambda *a, **k: {})
    ko = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev = [{"commence_time": ko, "home_team": "H", "away_team": "A", "bookmakers": [_b("a", 2.0, 1.8), _b("b", 2.1, 1.7)]}]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(ev))
    out = P._fetch_odds_api()
    assert len(out) == 1                                          # the match stays on the board
    r = out.iloc[0]
    assert pd.isna(r["odds_over25"]) and pd.isna(r["odds_under25"]) and pd.isna(r["odds_over25_book"])
    assert r["odds_ou25_n_books"] == 2 and pd.isna(r["odds_over25_captured_at"])
    v = betting.evaluate_value(out.assign(p_over25=0.9))
    assert v["bet"].iloc[0] == "AVOID"                            # never tipped without a price


def test_no_served_model_reads_the_bet_price():
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
