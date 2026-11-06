"""AU4 audit tests - price integrity and prop CLV.

Two kinds of test:
  * ADVERSARIAL tests of the audit's own independent helpers (must PASS).
  * REGRESSION tests for production defects found by AU4. They encode the CORRECT behaviour and are
    marked xfail(strict=True): they fail today (documenting the ACTIVE defect) and will XPASS -
    turning the suite red - the day the defect is fixed, so the marker must then be removed.
Nothing here writes to output/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from au4_common import block_bootstrap, summarise  # noqa: E402
from au4_clv_forensic import same_fixture  # noqa: E402
import au4_price_integrity as pi  # noqa: E402
from au4_props_regrade import regrade  # noqa: E402

DAY = "2026-10-25"


# ───────────────────────── adversarial tests of the audit helpers ─────────────────────────
def test_cross_book_sides_are_never_paired():
    d = pd.DataFrame([
        {"fixture_id": 1, "bookmaker": "A", "market": "ou", "line": 2.5, "snapshot_ts": "t", "side": "over", "odds": 2.10},
        {"fixture_id": 1, "bookmaker": "B", "market": "ou", "line": 2.5, "snapshot_ts": "t", "side": "under", "odds": 2.10},
    ])
    r = pi.pair_stats(d, ["fixture_id", "bookmaker", "market", "line", "snapshot_ts"], "side", "over", "under")
    assert r["pairs"] == 0                       # Over from A + Under from B is NOT a pair
    x = pi.best_cross_book(d, ["fixture_id", "market", "line", "snapshot_ts"], "side", "over", "under")
    assert x["lt_1.00"] == 1                     # and mixing them manufactures an "arbitrage" (0.952)


def test_same_book_different_time_is_not_a_pair():
    d = pd.DataFrame([
        {"k": 1, "bookmaker": "A", "snapshot_ts": "t1", "side": "over", "odds": 1.9},
        {"k": 1, "bookmaker": "A", "snapshot_ts": "t2", "side": "under", "odds": 1.9},
    ])
    assert pi.pair_stats(d, ["k", "bookmaker", "snapshot_ts"], "side", "over", "under")["pairs"] == 0


def test_line_mapping_inversion_detected():
    d = pd.DataFrame([{"g": 1, "_lk": 1.5, "odds": 2.0}, {"g": 1, "_lk": 2.5, "odds": 1.8},
                      {"g": 2, "_lk": 1.5, "odds": 1.3}, {"g": 2, "_lk": 2.5, "odds": 1.9}])
    r = pi.monotone(d, ["g"], [1.5, 2.5], True)
    assert r == {"groups_compared": 2, "violations": 1, "violation_rate": 0.5}


def test_post_kickoff_quote_flagged():
    r = pi.post_ko(pd.Series(["2026-10-25T14:59:59Z", "2026-10-25T15:00:00Z", "2026-10-25T16:00:00Z"]),
                   pd.Series(["2026-10-25T15:00:00Z"] * 3))
    assert r["post_kickoff_rows"] == 2           # AT kickoff is already post-kickoff


def test_namesake_in_other_fixture_is_not_same_fixture():
    assert not same_fixture("Vitoria vs Atletico Paranaense", "Vasco da Gama vs Corinthians")
    assert same_fixture("Castellon vs Sporting Gijon", "Sporting Gijon vs Castellón")   # swapped listing


def test_regrade_dnp_is_void_and_missing_stat_is_not_a_loss():
    rec = pd.DataFrame([
        {"bet_id": "b1", "player": "Ana Lima", "d": DAY, "match": "X vs Y", "market": "goals"},
        {"bet_id": "b2", "player": "Bo Kim", "d": DAY, "match": "X vs Y", "market": "sot"},
        {"bet_id": "b3", "player": "Cy Diaz", "d": DAY, "match": "X vs Y", "market": "goals"},
        {"bet_id": "b4", "player": "Di Eze", "d": DAY, "match": "X vs Y", "market": "goals"},
    ])
    h = pd.DataFrame([
        {"pk": "ana lima", "dkey": DAY, "home_team": "X", "away_team": "Y", "minutes": 0, "goals": 0, "shots_on_target": 0, "cards": 0, "assists": 0},
        {"pk": "bo kim", "dkey": DAY, "home_team": "X", "away_team": "Y", "minutes": 90, "goals": 0, "shots_on_target": np.nan, "cards": 0, "assists": 0},
        {"pk": "cy diaz", "dkey": DAY, "home_team": "P", "away_team": "Q", "minutes": 90, "goals": 2, "shots_on_target": 3, "cards": 0, "assists": 0},
        {"pk": "di eze", "dkey": DAY, "home_team": "X", "away_team": "Y", "minutes": 90, "goals": 1, "shots_on_target": 1, "cards": 0, "assists": 0},
    ])
    g = regrade(rec, h).set_index("bet_id")["au4_result"]
    assert g["b1"] == "VOID"            # non-participant
    assert g["b2"] == "UNRESOLVED"      # missing stat is not a loss
    assert g["b3"] == "UNRESOLVED"      # namesake in another fixture never settles the bet
    assert g["b4"] == "WIN"


def test_odds_ratio_clv_is_odds_weighted():
    """clv_pct = odds_bet/odds_close - 1 = (p_close - p_bet) / p_bet: every probability move is
    weighted by the entry ODDS. Most short-priced records drifting against us plus a few longshots
    shortening gives a POSITIVE mean clv_pct while the market moved against us on average."""
    p0 = np.r_[np.full(95, 0.60), np.full(5, 0.05)]          # 95 short prices, 5 longshots
    p1 = np.r_[np.full(95, 0.58), np.full(5, 0.09)]          # shorts drift out, longshots come in
    ratio = p1 / p0 - 1
    assert (p1 - p0).mean() < 0                               # market moved against us on average
    assert np.log1p(ratio).mean() < 0
    assert (ratio > 0).mean() == 0.05                         # "beat close" 5%
    assert ratio.mean() > 0                                   # ...yet the production verdict is POSITIVE
    assert summarise(np.log1p(ratio))["mean"] < ratio.mean()


def test_block_bootstrap_ci_contains_estimate():
    v = np.r_[np.ones(50), -np.ones(50)]
    est, lo, hi, g = block_bootstrap(v, np.repeat(np.arange(20), 5), n_boot=300)
    assert lo <= est <= hi and g == 20


# ───────────────────────── regression tests for ACTIVE production defects ─────────────────────────
def _close_out_with(monkeypatch, rec, hist):
    from player_model import clv_tracker
    import src.rolling_store as rs
    closes = {}
    monkeypatch.setattr(clv_tracker.clv_capture, "_load", lambda: [rec])
    monkeypatch.setattr(clv_tracker.clv_capture, "capture_close", lambda bid, o, u, *a: closes.__setitem__(bid, o))
    monkeypatch.setattr(clv_tracker.clv_capture, "mark_status", lambda *a: None)
    monkeypatch.setattr(clv_tracker, "_player_ids", lambda: {})
    monkeypatch.setattr(rs, "read_all", lambda path, dates=None: hist.copy())
    clv_tracker.close_out()
    return closes


# AU4-P8-03 FIXED 2026-11-06 (Phase 2 part 7): the close is taken from quotes strictly before the
# recorded kickoff (player_model.clv_integrity.decide_close).
def test_prop_close_is_strictly_pre_kickoff(monkeypatch):
    # kickoff 15:00; the 15:05 row is an in-play price and must never be the close
    rec = {"bet_id": f"PLAYER|{DAY}|Ana Lima|goals", "match": "X vs Y", "odds_close": "",
           "odds_bet": "3.0", "ts_bet": f"{DAY}T14:40:00+00:00", "kickoff_utc": f"{DAY}T15:00:00Z"}
    hist = pd.DataFrame([
        {"snapshot_ts": f"{DAY}T14:30:00Z", "match_date": DAY, "match": "X vs Y", "player": "ana lima", "market": "goals", "odds": 3.0},
        {"snapshot_ts": f"{DAY}T15:05:00Z", "match_date": DAY, "match": "X vs Y", "player": "ana lima", "market": "goals", "odds": 1.4},
    ])
    closes = _close_out_with(monkeypatch, rec, hist)
    assert closes.get(rec["bet_id"]) == 3.0


# AU4-P8-04 FIXED 2026-11-02: the close is best-of-books (each book's last price, best of them),
# the same basis as the best-of-books entry (player_model.clv_tracker.close_best_of_books).
def test_prop_close_is_same_book_as_entry(monkeypatch):
    rec = {"bet_id": f"PLAYER|{DAY}|Ana Lima|sot", "match": "X vs Y", "odds_close": "", "odds_bet": "2.0",
           "ts_bet": f"{DAY}T09:30:00+00:00", "kickoff_utc": f"{DAY}T18:00:00Z"}
    hist = pd.DataFrame([
        {"snapshot_ts": f"{DAY}T09:00:00Z", "match_date": DAY, "match": "X vs Y", "player": "ana lima", "market": "sot", "odds": 2.0, "bookmaker": "Bet365"},
        {"snapshot_ts": f"{DAY}T13:00:00Z", "match_date": DAY, "match": "X vs Y", "player": "ana lima", "market": "sot", "odds": 1.6, "bookmaker": "Betano"},
    ])
    closes = _close_out_with(monkeypatch, rec, hist)
    assert closes.get(rec["bet_id"]) == 2.0      # Bet365 never moved; Betano's price is not this bet's close


# AU4-P8-01 FIXED 2026-11-06 (Phase 2 part 7): clv_report describes, it never judges an edge.
def test_clv_verdict_not_positive_when_market_moved_against(monkeypatch, tmp_path):
    from src import clv_capture
    rows = []
    # 9 records shorten slightly against us, 1 longshot drifts hugely: mean ratio > 0, mean log < 0,
    # implied-probability move < 0, beat-close 10%
    for i in range(9):
        rows.append({"bet_id": f"b{i}", "clv_pct": round(2.0 / 2.1 - 1, 4), "clv_prob": ""})
    rows.append({"bet_id": "b9", "clv_pct": 0.8, "clv_prob": ""})
    monkeypatch.setattr(clv_capture, "_load", lambda: rows)
    rep = clv_capture.clv_report()
    assert not rep["verdict"].startswith("POSITIVE")


@pytest.mark.xfail(strict=True, reason="AU4-P5-01 HISTORICAL-DATA/ACTIVE-CODE: _closing_for_market reads mis-mapped NF parser-window rows")
def test_ledger_close_ignores_nf_parser_window_rows(monkeypatch):
    import update_results as ur
    d = pd.DataFrame([
        {"snapshot_date": "2026-07-20", "snapshot_ts": "2026-07-20T18:00:00Z", "match_date": "2026-07-22",
         "league": "USA MLS", "match": "Nashville SC vs CF Montreal", "market": "under25", "odds": 1.25},
    ])
    d["_hk"] = d["match"].map(lambda m: ur._norm(m.split(" vs ")[0]))
    d["_ak"] = d["match"].map(lambda m: ur._norm(m.split(" vs ")[1]))
    d["_dk"] = d["match_date"]
    monkeypatch.setattr(ur, "_CSV_ARCHIVES", [d])
    # the only row is a forward newformat_odds_history O/U row from before the 2026-08-10 parser fix
    assert np.isnan(ur._closing_for_market("Nashville SC", "CF Montreal", "2026-07-22", "under25"))
