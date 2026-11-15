"""Phase 2 part 7: props CLV integrity (player_model.clv_integrity, the close_out writer, the audit).

A CLV value is written only for a record whose entry price was quoted for the same player / market /
fixture by the tip time and whose close was quoted before the recorded kickoff; every other record gets
its status and no CLV. Synthetic data only; every file lives under tmp_path."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

from player_model import clv_integrity as CI
from player_model import clv_tracker
from src import clv_capture
from src.contracts import schemas, schema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import props_clv_integrity as audit  # noqa: E402

DAY = "2026-10-20"
MATCH = "Alpha FC vs Beta United"
OTHER = "Gamma City vs Delta Town"
KO = f"{DAY}T18:00:00Z"
BID = f"PLAYER|{DAY}|Ana Lima|sot"


def _row(ts, odds, book="Bet365", source="api_football", match=MATCH, player="ana lima", market="sot"):
    return {"snapshot_ts": f"{DAY}T{ts}:00Z", "match_date": DAY, "match": match, "player": player,
            "market": market, "odds": odds, "source": source, "bookmaker": book}


def _hist(*rows):
    return CI.index_history(CI.prepare_history(pd.DataFrame(list(rows))))


def _rec(**kw):
    r = {"bet_id": BID, "match": MATCH, "odds_bet": "2.0", "ts_bet": f"{DAY}T09:30:00+00:00",
         "odds_close": "", "kickoff_utc": KO}
    r.update(kw)
    return r


def _m(idx, match=MATCH):
    return CI.identity_rows(idx, DAY, "Ana Lima", "sot", match)


# ── the writer's decision ─────────────────────────────────────────────────────────────────────
def test_same_book_close_strictly_before_kickoff():
    idx = _hist(_row("09:00", 2.0), _row("15:00", 1.8), _row("18:10", 1.3))     # 18:10 is in-play
    d = CI.decide_close(_rec(), _m(idx), KO)
    assert (d["status"], d["close"]) == (CI.VALID_SAME_BOOK, 1.8)


def test_best_of_books_close_is_cross_book():
    idx = _hist(_row("09:00", 2.0), _row("15:00", 1.8), _row("10:00", 1.9, "Betano"))
    d = CI.decide_close(_rec(), _m(idx), KO)
    assert d["status"] == CI.VALID_CROSS_BOOK and d["close"] == 1.9    # Betano's price, not the entry book


def test_oddsapi_rows_never_prove_a_same_book_close():
    idx = _hist(_row("09:00", 2.0, None, "oddsapi"), _row("15:00", 1.8, None, "oddsapi"))
    assert CI.decide_close(_rec(), _m(idx), KO)["status"] == CI.VALID_CROSS_BOOK


def test_entry_price_from_another_fixture_is_identity_unproven():
    # the 2.0 was a namesake's price in another match: never this record's entry
    idx = _hist(_row("09:00", 2.0, match=OTHER), _row("09:00", 1.5), _row("15:00", 1.4))
    d = CI.decide_close(_rec(), _m(idx), KO)
    assert d["status"] == CI.IDENTITY_UNPROVEN and d["close"] is None


def test_entry_price_quoted_only_after_the_tip_is_identity_unproven():
    idx = _hist(_row("11:00", 2.0), _row("15:00", 1.8))
    d = CI.decide_close(_rec(), _m(idx), KO)
    assert d["status"] == CI.IDENTITY_UNPROVEN and "only after" in d["reason"]


def test_another_line_is_another_market():
    idx = _hist(_row("09:00", 2.0, market="sot2"), _row("15:00", 1.8, market="sot2"))
    assert CI.decide_close(_rec(), _m(idx), KO)["status"] == CI.IDENTITY_UNPROVEN


def test_no_kickoff_no_clv():
    idx = _hist(_row("09:00", 2.0), _row("15:00", 1.8))
    assert CI.decide_close(_rec(), _m(idx), None)["status"] == CI.TIMING_UNPROVEN


def test_entry_after_kickoff():
    idx = _hist(_row("09:00", 2.0), _row("15:00", 1.8))
    d = CI.decide_close(_rec(ts_bet=f"{DAY}T18:30:00+00:00"), _m(idx), KO)
    assert d["status"] == CI.ENTRY_AFTER_KICKOFF


def test_namesake_ids_block_the_record():
    idx = _hist(_row("09:00", 2.0), _row("15:00", 1.8))
    assert CI.decide_close(_rec(), _m(idx), KO, namesake=True)["status"] == CI.IDENTITY_UNPROVEN


def test_price_row_with_another_player_id_is_not_this_player():
    h = pd.DataFrame([{**_row("09:00", 2.0), "player_id": 111}, {**_row("15:00", 1.8), "player_id": 222}])
    idx = CI.index_history(CI.prepare_history(h))
    m = CI.identity_rows(idx, DAY, "Ana Lima", "sot", MATCH, player_id="111")
    assert list(m["_odds"]) == [2.0]


# ── the writer itself (close_out on a temp clv file) ──────────────────────────────────────────
@pytest.fixture
def clv_file(tmp_path, monkeypatch):
    f = tmp_path / "clv_records.csv"
    monkeypatch.setattr(clv_capture, "CLV_FILE", f)
    monkeypatch.setattr(clv_tracker, "_player_ids", lambda: {})
    monkeypatch.setattr(clv_tracker, "_tip_kickoffs", lambda: {})
    return f


def _run(monkeypatch, hist):
    import src.rolling_store as rs
    monkeypatch.setattr(rs, "read_all", lambda path, dates=None: pd.DataFrame(hist))
    return clv_tracker.close_out()


def _rows(f):
    with open(f, newline="", encoding="utf-8") as fh:
        return {r["bet_id"]: r for r in csv.DictReader(fh)}


def test_writer_records_status_and_no_clv_for_a_failed_record(clv_file, monkeypatch):
    clv_capture.log_bet(BID, "sot", "Ana Lima", MATCH, "over", 2.0, None,
                        ts_bet=f"{DAY}T09:30:00+00:00", kickoff_utc=KO)
    bid2 = f"PLAYER|{DAY}|Bo Kim|sot"
    clv_capture.log_bet(bid2, "sot", "Bo Kim", MATCH, "over", 3.0, None,
                        ts_bet=f"{DAY}T09:30:00+00:00", kickoff_utc=KO)
    hist = [_row("09:00", 2.0), _row("15:00", 1.8), _row("18:20", 1.1),
            _row("09:00", 2.5, player="bo kim"), _row("15:00", 2.2, player="bo kim")]  # 3.0 never quoted
    assert _run(monkeypatch, hist) == 1
    r = _rows(clv_file)
    assert r[BID]["clv_status"] == CI.VALID_SAME_BOOK and float(r[BID]["odds_close"]) == 1.8
    assert r[BID]["clv_pct"] != ""
    assert r[bid2]["clv_status"] == CI.IDENTITY_UNPROVEN
    assert r[bid2]["odds_close"] == "" and r[bid2]["clv_pct"] == ""
    # a final status is not revisited
    assert _run(monkeypatch, hist) == 0
    # the file meets the contract, including "a CLV value only when VALID"
    v = schema.validate(pd.read_csv(clv_file), schemas.CLV_RECORD)
    assert not [x for x in v if x["kind"] == "rule"]


def test_writer_waits_for_kickoff_and_retries_timing_unproven(clv_file, monkeypatch):
    clv_capture.log_bet(BID, "sot", "Ana Lima", MATCH, "over", 2.0, None, ts_bet=f"{DAY}T09:30:00+00:00")
    hist = [_row("09:00", 2.0), _row("15:00", 1.8)]
    assert _run(monkeypatch, hist) == 0
    assert _rows(clv_file)[BID]["clv_status"] == CI.TIMING_UNPROVEN
    monkeypatch.setattr(clv_tracker, "_tip_kickoffs", lambda: {(DAY, MATCH): KO})
    assert _run(monkeypatch, hist) == 1                                      # retried once a kickoff exists
    assert _rows(clv_file)[BID]["clv_status"] == CI.VALID_SAME_BOOK


def test_contract_flags_a_clv_value_on_a_failed_record():
    d = pd.DataFrame([{"bet_id": "b", "ts_bet": "2026-10-20T09:00:00Z", "market": "sot", "match": MATCH,
                       "side": "over", "odds_bet": 2.0, "odds_close": 1.8, "clv_pct": 0.11,
                       "clv_status": CI.CLOSE_AFTER_KICKOFF}])
    assert "clv_only_when_valid" in {x["column"] for x in schema.validate(d, schemas.CLV_RECORD)
                                     if x["kind"] == "rule"}


def test_contract_status_vocabulary_matches_the_module():
    assert schemas.CLV_STATUSES == CI.STATUSES and schemas.CLV_VALID == CI.VALID


def test_report_never_judges_an_edge(clv_file):
    clv_capture.log_bet(BID, "sot", "Ana Lima", MATCH, "over", 2.0, None)
    clv_capture.capture_close(BID, 1.5, None, CI.VALID_SAME_BOOK)
    rep = clv_capture.clv_report()
    assert rep["verdict"] == clv_capture.REPORT_VERDICT and "edge signal" not in rep["verdict"]
    assert rep["by_status"] == {CI.VALID_SAME_BOOK: 1}


# ── the audit of stored records ───────────────────────────────────────────────────────────────
def _stored(idx, ko=KO, quotes=None, **kw):
    rec = _rec(odds_close="1.8", **kw)
    return CI.classify_stored(rec, _m(idx), CI.utc(ko) if ko else pd.NaT, quotes=quotes)


def test_stored_close_first_quoted_in_play_is_close_after_kickoff():
    idx = _hist(_row("09:00", 2.0), _row("18:20", 1.8))
    assert _stored(idx)["status"] == CI.CLOSE_AFTER_KICKOFF


def test_stored_close_never_quoted_for_this_fixture_is_identity_unproven():
    idx = _hist(_row("09:00", 2.0), _row("15:00", 1.8, match=OTHER))
    assert _stored(idx)["status"] == CI.IDENTITY_UNPROVEN


def test_stored_record_without_close_is_no_close():
    idx = _hist(_row("09:00", 2.0))
    assert CI.classify_stored(_rec(), _m(idx), CI.utc(KO))["status"] == CI.NO_CLOSE


def test_same_book_needs_the_per_book_quotes():
    idx = _hist(_row("09:00", 2.0), _row("15:00", 1.8))
    q = CI.prepare_history(pd.DataFrame([
        {**_row("09:00", 2.0, "Paddy Power", "oddsapi"), "side": "yes"},
        {**_row("15:00", 1.8, "Paddy Power", "oddsapi"), "side": "yes"},
        {**_row("15:00", 9.0, "Paddy Power", "oddsapi"), "side": "no"}]), per_book=True)
    assert _stored(idx, quotes=q)["status"] == CI.VALID_SAME_BOOK
    other = CI.prepare_history(pd.DataFrame([
        {**_row("09:00", 2.0, "Paddy Power", "oddsapi"), "side": "yes"},
        {**_row("15:00", 1.8, "Betfair", "oddsapi"), "side": "yes"}]), per_book=True)
    assert _stored(idx, quotes=other)["status"] == CI.VALID_CROSS_BOOK
    assert _stored(idx, quotes=CI.prepare_history(pd.DataFrame(), per_book=True))["status"] == CI.VALID_CROSS_BOOK


def test_audit_counts_statuses_and_refuses_two_kickoffs():
    clv = pd.DataFrame([_rec(odds_close="1.8", kickoff_utc=""),
                        _rec(bet_id=f"PLAYER|{DAY}|Bo Kim|sot", odds_close="", kickoff_utc="")])
    hist = pd.DataFrame([_row("09:00", 2.0), _row("15:00", 1.8)])
    index = pd.DataFrame([{"date": DAY, "match": MATCH, "kickoff_utc": KO, "n_distinct_kickoff": 2}])
    ko_map, bad = audit.kickoffs(index, None)
    assert (DAY, MATCH) in bad and not ko_map
    st = audit.classify(clv, hist, ko_map, bad, None, pd.DataFrame())
    assert list(st["status"]) == [CI.TIMING_UNPROVEN, CI.NO_CLOSE]
    by, summ = audit.tables(st)
    assert summ["by_status"][CI.TIMING_UNPROVEN] == 1 and summ["records"] == 2
    assert "mean" not in str(sorted(summ)) and "roi" not in str(sorted(summ))
