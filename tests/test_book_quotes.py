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
