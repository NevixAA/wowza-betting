"""Tests for per-book capture and the kickoff ladder."""
from __future__ import annotations
import numpy as np, pandas as pd, pytest
import src.book_quotes as bq


def test_ladder_bands_are_assigned_tightest_first():
    assert bq.ladder_band(5) == "T-10m"
    assert bq.ladder_band(10) == "T-10m"
    assert bq.ladder_band(25) == "T-30m"
    assert bq.ladder_band(59) == "T-1h"
    assert bq.ladder_band(120) == "T-3h"
    assert bq.ladder_band(300) == "T-6h"
    assert bq.ladder_band(1000) == "FAR"


def test_a_post_kickoff_quote_is_never_a_near_band():
    """An in-play price treated as a close would manufacture huge fake CLV."""
    assert bq.ladder_band(-1) == "POST"
    assert bq.ladder_band(-500) == "POST"


def test_unknown_kickoff_is_its_own_band():
    assert bq.ladder_band(None) == "UNKNOWN"
    assert bq.ladder_band(float("nan")) == "UNKNOWN"


def _payload():
    return {"response": [{"bookmakers": [
        {"id": 8, "name": "Bet365", "bets": [
            {"id": 5, "name": "Goals Over/Under", "values": [
                {"value": "Over 2.5", "odd": "1.91"}, {"value": "Under 2.5", "odd": "1.95"},
                {"value": "Over 3.5", "odd": "3.40"}]},
            {"id": 8, "name": "Both Teams Score", "values": [
                {"value": "Yes", "odd": "1.80"}, {"value": "No", "odd": "2.00"}]}]},
        {"id": 6, "name": "Bwin", "bets": [
            {"id": 5, "name": "Goals Over/Under", "values": [
                {"value": "Over 2.5", "odd": "1.88"}, {"value": "Under 2.5", "odd": "1.98"}]}]},
    ]}]}


def test_every_bookmaker_is_kept_not_collapsed():
    rows = bq.parse_books(_payload(), 1, "2026-10-05T19:00:00Z", "L", "H", "A",
                          "2026-10-05T18:50:00Z")
    books = {r["bookmaker"] for r in rows}
    assert books == {"Bet365", "Bwin"}, "consensus must not replace the individual quotes"


def test_line_is_separate_from_market():
    rows = bq.parse_books(_payload(), 1, "2026-10-05T19:00:00Z", "L", "H", "A",
                          "2026-10-05T18:50:00Z")
    ou = [r for r in rows if r["market"] == "ou" and r["bookmaker"] == "Bet365"]
    assert {r["line"] for r in ou} == {2.5, 3.5}
    assert {r["side"] for r in ou} == {"over", "under"}


def test_schema_has_every_field_section_10_requires():
    rows = bq.parse_books(_payload(), 7, "2026-10-05T19:00:00Z", "L1", "H", "A",
                          "2026-10-05T18:50:00Z", model_type="standard")
    for f in ("fixture_id", "bookmaker", "market", "side", "line", "odds",
              "snapshot_ts", "kickoff_utc", "source", "model_type", "league"):
        assert f in rows[0], f"missing required field {f}"
    assert rows[0]["minutes_to_kickoff"] == 10.0
    assert rows[0]["ladder_band"] == "T-10m"


def test_only_full_match_goals_count_as_ou():
    """Live data carries SEVEN bets whose name contains 'over/under'. Only id 5 is ours.

    A name-based classifier swallowed second-half goals (26), corners (57/58) and time windows
    (197/198), producing 'O/U 2.5' quotes from 1.25 to 3.50 and an anchor of p=0.254 where the
    real market was ~0.63.
    """
    p = {"response": [{"bookmakers": [{"id": 1, "name": "X", "bets": [
        {"id": 5, "name": "Goals Over/Under", "values": [{"value": "Over 2.5", "odd": "1.90"}]},
        {"id": 26, "name": "Goals Over/Under - Second Half",
         "values": [{"value": "Over 2.5", "odd": "3.50"}]},
        {"id": 57, "name": "Home Corners Over/Under",
         "values": [{"value": "Over 2.5", "odd": "1.30"}]},
        {"id": 197, "name": "Over/Under 15m-30m", "values": [{"value": "Over 0.5", "odd": "2.5"}]},
        {"id": 33, "name": "Asian Handicap", "values": [{"value": "Home -1.5", "odd": "2.0"}]},
    ]}]}]}
    rows = bq.parse_books(p, 1, "2026-10-05T19:00:00Z", "L", "H", "A", "2026-10-05T18:00:00Z")
    ou = [r for r in rows if r["market"] == "ou"]
    assert len(ou) == 1, f"only bet id 5 is full-match goals, got {len(ou)}"
    assert ou[0]["odds"] == 1.90


def test_first_half_btts_is_not_stored_as_btts():
    """The capture already records 4.3% of btts_yes rows being first-half prices."""
    p = {"response": [{"bookmakers": [{"id": 1, "name": "X", "bets": [
        {"id": 34, "name": "Both Teams To Score - First Half",
         "values": [{"value": "Yes", "odd": "5.5"}]}]}]}]}
    rows = bq.parse_books(p, 1, "2026-10-05T19:00:00Z", "L", "H", "A", "2026-10-05T18:00:00Z")
    assert not any(r["market"] == "btts" for r in rows)


# ── closing line ──────────────────────────────────────────────────────────────────────────────
def _frame(mins):
    return pd.DataFrame([{"fixture_id": 1, "market": "ou", "side": "over", "line": 2.5,
                          "odds": 1.9 + i / 100, "bookmaker": "B", "ladder_band": bq.ladder_band(m),
                          "minutes_to_kickoff": m} for i, m in enumerate(mins)])


def test_closing_quote_is_the_last_one_inside_the_window():
    c = bq.closing_quote(_frame([360, 60, 25, 8]), 1, "ou", "over", 2.5)
    assert c["minutes_to_kickoff"] == 8.0


def test_no_close_is_returned_when_nothing_got_near_kickoff():
    """A six-hour-old price is not a close, and must not be substituted for one."""
    assert bq.closing_quote(_frame([360, 300, 240]), 1, "ou", "over", 2.5) is None


def test_post_kickoff_quotes_are_never_used_as_the_close():
    assert bq.closing_quote(_frame([-5, -30]), 1, "ou", "over", 2.5) is None


def test_append_keeps_only_changed_prices(tmp_path):
    p = tmp_path / "q.csv"
    rows = bq.parse_books(_payload(), 1, "2026-10-05T19:00:00Z", "L", "H", "A",
                          "2026-10-05T18:50:00Z")
    assert bq.append_quotes(rows, p) == len(rows)
    assert bq.append_quotes(rows, p) == 0, "an unchanged price is not new information"


# ── monthly partitioning ──────────────────────────────────────────────────────────────────────
def _q(ts, odds, fid=1, book=8):
    return {"snapshot_ts": ts, "fixture_id": fid, "kickoff_utc": "2026-10-04T18:00:00Z",
            "minutes_to_kickoff": 30.0, "ladder_band": "T-30m", "league": "L", "model_type": "std",
            "home_team": "A", "away_team": "B", "bookmaker_id": book, "bookmaker": "Bet365",
            "market": "ou", "side": "over", "line": 2.5, "odds": odds, "source": "t"}


def test_rows_are_filed_by_their_own_day_not_todays(tmp_path, monkeypatch):
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_q("2026-09-30T23:50:00Z", 1.90), _q("2026-10-01T00:10:00Z", 1.95)])
    assert {f.name for f in tmp_path.glob("*.csv")} == {"2026-09-30.csv", "2026-10-01.csv"}


def test_dedup_spans_the_part_boundary(tmp_path, monkeypatch):
    """A price unchanged across midnight must not be rewritten into the new part."""
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_q("2026-09-30T23:50:00Z", 1.90)])
    assert bq.append_quotes([_q("2026-10-01T00:10:00Z", 1.90)]) == 0
    assert bq.append_quotes([_q("2026-10-01T00:20:00Z", 1.95)]) == 1


def test_a_finished_part_is_never_rewritten(tmp_path, monkeypatch):
    """Git stores a changed file in full, so an immutable finished month is the whole point."""
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_q("2026-09-30T23:50:00Z", 1.90)])
    sept = tmp_path / "2026-09-30.csv"
    before = sept.read_bytes()
    bq.append_quotes([_q("2026-10-01T00:20:00Z", 1.95), _q("2026-10-02T00:20:00Z", 2.00)])
    assert sept.read_bytes() == before


def test_load_quotes_reads_every_part(tmp_path, monkeypatch):
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    monkeypatch.setattr(bq, "LEGACY_FILE", tmp_path / "nonexistent.csv")
    bq.append_quotes([_q("2026-09-30T23:50:00Z", 1.90), _q("2026-10-01T00:10:00Z", 1.95)])
    assert len(bq.load_quotes()) == 2


# ── §6 observation heartbeat ──────────────────────────────────────────────────────────────────
def _qb(ts, odds, band, mins, fid=1, book=8):
    return {"snapshot_ts": ts, "fixture_id": fid, "kickoff_utc": "2026-10-04T18:00:00Z",
            "minutes_to_kickoff": mins, "ladder_band": band, "league": "L", "model_type": "std",
            "home_team": "A", "away_team": "B", "bookmaker_id": book, "bookmaker": "Bet365",
            "market": "ou", "side": "over", "line": 2.5, "odds": odds, "source": "t"}


def test_the_briefs_exact_scenario_an_unchanged_price_is_still_provably_observed(tmp_path,
                                                                                monkeypatch):
    """T-3h 1.90, T-1h 1.90, T-30m 1.90, T-10m 1.90.

    Under change-only storage the archive keeps ONE row at T-3h and cannot show the price was
    ever seen near kickoff. The near rungs must each leave a record.
    """
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_qb("2026-10-04T15:00:00Z", 1.90, "T-3h", 180.0)])
    for ts, band, mins in [("2026-10-04T17:00:00Z", "T-1h", 60.0),
                           ("2026-10-04T17:30:00Z", "T-30m", 30.0),
                           ("2026-10-04T17:50:00Z", "T-10m", 10.0)]:
        assert bq.append_quotes([_qb(ts, 1.90, band, mins)]) == 1, f"{band} left no record"
    d = bq.load_quotes()
    assert set(d["ladder_band"]) == {"T-3h", "T-1h", "T-30m", "T-10m"}
    assert set(d[d.ladder_band != "T-3h"]["obs_reason"]) == {"heartbeat"}
    assert bq.closing_quote(d, 1, "ou", "over", 2.5)["odds"] == 1.90


def test_resampling_one_rung_does_not_explode_storage(tmp_path, monkeypatch):
    """The NEAR loop samples the same rung repeatedly to catch MOVEMENT. An unchanged price
    must leave one heartbeat for that rung, not one per pass."""
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    assert bq.append_quotes([_qb("2026-10-04T17:30:00Z", 1.90, "T-30m", 30.0)]) == 1
    for i in range(5):
        assert bq.append_quotes([_qb(f"2026-10-04T17:3{i+1}:00Z", 1.90, "T-30m", 29.0 - i)]) == 0
    assert len(bq.load_quotes()) == 1


def test_a_real_move_inside_a_rung_is_always_kept(tmp_path, monkeypatch):
    """The heartbeat must not suppress movement -- that is the other half of the file's job."""
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_qb("2026-10-04T17:30:00Z", 1.90, "T-30m", 30.0)])
    assert bq.append_quotes([_qb("2026-10-04T17:35:00Z", 1.95, "T-30m", 25.0)]) == 1
    d = bq.load_quotes()
    assert list(d["obs_reason"]) == ["change", "change"]


def test_far_rungs_stay_change_only(tmp_path, monkeypatch):
    """Nothing is certified against a T-6h price, and a heartbeat on every band would push a
    monthly part past the size limit the partitioning exists to avoid."""
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_qb("2026-10-04T06:00:00Z", 1.90, "FAR", 720.0)])
    assert bq.append_quotes([_qb("2026-10-04T12:00:00Z", 1.90, "T-6h", 360.0)]) == 0
    assert bq.append_quotes([_qb("2026-10-04T15:00:00Z", 1.90, "T-3h", 180.0)]) == 0


def test_a_post_kickoff_heartbeat_can_never_become_a_close(tmp_path, monkeypatch):
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_qb("2026-10-04T18:05:00Z", 1.90, "POST", -5.0)])
    assert bq.closing_quote(bq.load_quotes(), 1, "ou", "over", 2.5) is None


def test_a_missing_t30_close_stays_missing(tmp_path, monkeypatch):
    """Never substitute an earlier quote. The honest answer is that there is no close."""
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    bq.append_quotes([_qb("2026-10-04T12:00:00Z", 1.90, "T-6h", 360.0)])
    assert bq.closing_quote(bq.load_quotes(), 1, "ou", "over", 2.5) is None


# ── volume control (measured on the first real CI runs, 2026-10-04) ───────────────────────────
def test_unmodelled_ou_lines_are_dropped():
    """The API returns THIRTY O/U lines; we model three. Keeping the rest made 72% of the
    archive markets nothing reads, at 40 MB/day against a 100 MB per-file limit."""
    assert bq._classify(5, "Goals Over/Under", "Over 2.5") == ("ou", "over", 2.5)
    for bad in ("Over 4.5", "Over 2.75", "Under 0.5", "Over 6.5"):
        assert bq._classify(5, "Goals Over/Under", bad) is None, bad


def test_h2h_is_no_longer_captured():
    """1X2 is a research track with no bet and no consumer — 9% of rows for nothing."""
    assert bq._classify(1, "Match Winner", "Home") is None
    assert 1 not in bq.BET_IDS


def test_far_is_kept_once_as_an_opening_price_not_change_tracked(tmp_path, monkeypatch):
    """97.8% of captured rows were FAR and nothing is ever certified against one. What FAR is
    for is the OPENING line — one row, not a week of slow drift."""
    monkeypatch.setattr(bq, "QUOTES_DIR", tmp_path)
    assert bq.append_quotes([_qb("2026-10-01T06:00:00Z", 1.90, "FAR", 4000.0)]) == 1
    assert bq.load_quotes().iloc[0]["obs_reason"] == "open"
    # a genuine FAR price move is NOT stored — it is drift nobody reads
    assert bq.append_quotes([_qb("2026-10-02T06:00:00Z", 2.10, "FAR", 3000.0)]) == 0
    # but the near rungs still capture everything
    assert bq.append_quotes([_qb("2026-10-04T17:30:00Z", 2.10, "T-30m", 30.0)]) == 1
