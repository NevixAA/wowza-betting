"""Owner 2026-11-11 "Paper until re-run": odds_over25 / odds_under25 are NEVER a closing price.

* the loader keeps the PRE-MATCH pair (B365, else Pinnacle, else Avg - one source for both sides, with provenance)
  apart from the CLOSING pair (B365C / PC / AvgC / MaxC -> odds_*_close, CLV only);
* banked rows (closing-preferred before the fix) move their value to the closing columns and get their pre-match
  pair only through the CERTIFIED one-to-one football-data join (src/fd_prematch.py);
* the join is name-resolved, one-to-one, score-CHECKED (never score-matched), and uses the proven crosswalk;
* Data truth 8 turns RED when any closing column is read as a bet price;
* protocol v5 refuses a Standard universe not built through the certified join.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import data_loader as DL
from src import fd_prematch as F
from src.bars_opt import fresh_holdout as FH
from src.integrity import data_truth as DT


# ── the loader: pre-match and closing apart ───────────────────────────────────────────────────────

def _fd(**cols):
    return pd.DataFrame({k: [v] if not isinstance(v, list) else v for k, v in cols.items()})


def test_prematch_pair_never_takes_a_closing_column():
    raw = _fd(**{"B365>2.5": 1.90, "B365<2.5": 1.95, "AvgC>2.5": 2.10, "AvgC<2.5": 1.75,
                 "B365C>2.5": 2.05, "B365C<2.5": 1.80})
    p = DL.ou25_prices(raw)
    assert (p["odds_over25"].iloc[0], p["odds_under25"].iloc[0], p["odds_ou25_source"].iloc[0]) == (1.90, 1.95, "B365")
    assert (p["odds_over25_close"].iloc[0], p["odds_under25_close"].iloc[0]) == (2.05, 1.80)
    assert p["odds_ou25_close_source"].iloc[0] == "B365C"


def test_a_row_with_only_closing_prices_has_no_prematch_pair():
    raw = _fd(**{"AvgC>2.5": 2.10, "AvgC<2.5": 1.75, "MaxC>2.5": 2.20, "MaxC<2.5": 1.80})
    p = DL.ou25_prices(raw)
    assert np.isnan(p["odds_over25"].iloc[0]) and np.isnan(p["odds_under25"].iloc[0])
    assert p["odds_ou25_source"].iloc[0] == ""
    assert p["odds_over25_close"].iloc[0] == 2.10 and p["odds_ou25_close_source"].iloc[0] == "AVGC"


def test_one_source_for_both_sides_per_row():
    # Bet365 has only the Over side on row 0 -> the pair comes from Pinnacle, never B365 Over + P Under
    raw = _fd(**{"B365>2.5": [1.90, 1.80], "B365<2.5": [np.nan, 2.00], "P>2.5": [1.93, 1.85],
                 "P<2.5": [1.97, 2.02]})
    p = DL.ou25_prices(raw)
    assert p["odds_ou25_source"].tolist() == ["P", "B365"]
    assert p["odds_over25"].tolist() == [1.93, 1.80] and p["odds_under25"].tolist() == [1.97, 2.00]


def test_no_prematch_source_is_a_closing_column():
    for _, oc, uc in DL._OU25_PREMATCH + F.PREMATCH:
        assert not DT._CLOSING_RE.search(oc) and not DT._CLOSING_RE.search(uc)
    for _, oc, uc in DL._OU25_CLOSE + F.CLOSE:
        assert DT._CLOSING_RE.search(oc) and DT._CLOSING_RE.search(uc)


def _bank_rows():
    return pd.DataFrame({
        "league": ["Championship", "Championship", "Argentina Primera Division"],
        "date": pd.to_datetime(["2024-08-10", "2024-08-17", "2024-08-10"]),
        "home_team": ["Alpha United", "Bravo City", "Club Uno"], "away_team": ["Bravo City", "Alpha United", "Club Dos"],
        "home_goals": [2, 0, 1], "away_goals": [1, 0, 1],
        "odds_over25": [2.10, 1.70, 1.88], "odds_under25": [1.75, 2.15, 1.92]})


def test_banked_closing_preferred_value_moves_to_closing_and_prematch_comes_from_the_certified_join(monkeypatch):
    fd = pd.DataFrame({"league": ["Championship"], "date": pd.to_datetime(["2024-08-10"]),
                       "fd_home": ["Alpha United"], "fd_away": ["Bravo City"], "fd_hg": [2], "fd_ag": [1],
                       "pm_over": [1.95], "pm_under": [1.90], "pm_source": ["B365"],
                       "close_over": [2.10], "close_under": [1.75], "close_source": ["B365C"], "file": ["E1_2425.csv"]})
    monkeypatch.setattr(F, "load_fd_raw", lambda d: fd)
    monkeypatch.setattr(F, "load_crosswalk", lambda *a, **k: {})
    d = DL._ou25_provenance(_bank_rows())
    # row 0: joined -> pre-match Bet365; its old (closing-preferred) value sits in the closing columns
    assert (d.loc[0, "odds_over25"], d.loc[0, "odds_under25"], d.loc[0, "odds_ou25_source"]) == (1.95, 1.90, "FD_CERTIFIED_B365")
    assert (d.loc[0, "odds_over25_close"], d.loc[0, "odds_ou25_close_source"]) == (2.10, "LEGACY_CLOSING_PREFERRED")
    # row 1: no football-data row -> NO pre-match price, never the closing one
    assert np.isnan(d.loc[1, "odds_over25"]) and d.loc[1, "odds_ou25_source"] == "LEGACY_NO_PREMATCH"
    assert d.loc[1, "odds_over25_close"] == 1.70
    # row 2: new-format snapshot history keeps its pair, labelled
    assert d.loc[2, "odds_over25"] == 1.88 and d.loc[2, "odds_ou25_source"] == "NF_SNAPSHOT_HISTORY"


def test_a_failed_join_never_falls_back_to_the_closing_price(monkeypatch):
    def boom(d):
        raise OSError("fd_raw unreadable")
    monkeypatch.setattr(F, "load_fd_raw", boom)
    d = DL._ou25_provenance(_bank_rows())
    assert d.loc[[0, 1], "odds_over25"].isna().all() and d.loc[[0, 1], "odds_under25"].isna().all()
    assert d.loc[0, "odds_over25_close"] == 2.10


def test_rows_parsed_with_provenance_are_left_alone(monkeypatch):
    monkeypatch.setattr(F, "load_fd_raw", lambda d: pytest.fail("no banked row: the join must not run"))
    b = _bank_rows().iloc[:1].copy()
    b["odds_ou25_source"] = "B365"
    d = DL._ou25_provenance(b)
    assert d.loc[0, "odds_over25"] == 2.10 and d.loc[0, "odds_ou25_source"] == "B365"


# ── the certified join ────────────────────────────────────────────────────────────────────────────

def _fdrow(date, h, a, hg, ag, over=1.9, league="Championship"):
    return {"league": league, "date": pd.Timestamp(date), "fd_home": h, "fd_away": a, "fd_hg": hg, "fd_ag": ag,
            "pm_over": over, "pm_under": 1.9, "pm_source": "B365", "close_over": 2.0, "close_under": 1.8,
            "close_source": "B365C", "file": "x.csv"}


def _served(rows):
    return pd.DataFrame([{"league": "Championship", "date": pd.Timestamp(d), "home_team": h, "away_team": a,
                          "home_goals": hg, "away_goals": ag} for d, h, a, hg, ag in rows])


def test_join_is_name_resolved_one_to_one_and_score_checked():
    sv = _served([("2024-08-10", "Alpha United", "Bravo City", 2, 1),
                  ("2024-08-17", "Bravo City", "Charlie Town", 0, 0),
                  ("2024-08-24", "Charlie Town", "Alpha United", 1, 1),
                  ("2024-08-31", "Alpha United", "Charlie Town", 3, 0)])
    fd = pd.DataFrame([_fdrow("2024-08-10", "Alpha United", "Bravo City", 2, 1, over=1.80),   # joins
                       _fdrow("2024-08-17", "Bravo City", "Charlie Town", 1, 0),             # score differs
                       _fdrow("2024-08-24", "Zulu Rovers", "Yankee Athletic", 1, 1),         # names unknown: no
                       _fdrow("2024-08-31", "Alpha United", "Charlie Town", 3, 0),           # two fd rows
                       _fdrow("2024-09-01", "Alpha United", "Charlie Town", 3, 0)])          # claim one served
    j, rep = F.join(fd, sv)
    assert j.loc[0, "pm_over"] == 1.80 and j.loc[0, "pm_source"] == "B365"
    assert j.loc[1:, "pm_over"].isna().all()                       # no score-only pairing, no double claim
    r = rep[0]
    assert r["joined"] == 1 and r["score_check_failed"] == 1 and r["fd_unresolved_club"] == 1
    assert r["served_claimed_twice"] == 1


def test_join_accepts_a_unique_one_day_shift_and_uses_the_crosswalk():
    sv = _served([("2024-08-10", "Alpha United", "Bravo City", 2, 1)])
    fd = pd.DataFrame([_fdrow("2024-08-11", "Alpha Utd FC Old", "Bravo City", 2, 1)])
    j0, _ = F.join(fd, sv)
    assert j0["pm_over"].isna().all()                               # unknown spelling: not joined
    j, rep = F.join(fd, sv, crosswalk={"Championship": {"Alpha Utd FC Old": "Alpha United"}})
    assert j.loc[0, "pm_over"] == 1.9 and rep[0]["joined"] == 1


def test_committed_crosswalk_is_proven_and_pinned_by_protocol_v5():
    cw = json.loads(F.CROSSWALK.read_text(encoding="utf-8"))
    for lg, m in cw["map"].items():
        for fdn, served in m.items():
            ev = cw["evidence"][lg][fdn]
            assert ev["served"] == served and ev["joined_rows"] >= F.MIN_DATES
    assert FH.standard_ou25_source() == {"over25": F.CERTIFIED, "crosswalk_sha256": F.crosswalk_sha256()}


# ── Data truth 8 and the protocol ─────────────────────────────────────────────────────────────────

def test_data_truth_8_is_red_when_a_closing_column_is_a_bet_price(monkeypatch):
    assert DT.closing_as_bet_price() == []
    monkeypatch.setattr(DL, "_OU25_PREMATCH", DL._OU25_PREMATCH + (("AVGC", "AvgC>2.5", "AvgC<2.5"),))
    hits = DT.closing_as_bet_price()
    assert hits and hits[0]["where"] == "src/data_loader._OU25_PREMATCH"
    c = DT.bet_time_evidence()
    assert c["status"] == DT.RED and "closing-as-bet" in c["value"]


def test_data_truth_8_flags_a_phase2_benchmark_on_closing_prices(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "phase2_retrain.json").write_text(json.dumps(
        {"markets": {"over25": {"price": {"yes": "odds_over25_close", "no": "odds_under25"}}}}), encoding="utf-8")
    hits = DT.closing_as_bet_price(tmp_path)
    assert [h["where"] for h in hits] == ["config/phase2_retrain.json markets.over25.price.yes"]


def test_protocol_v5_refuses_a_universe_not_built_through_the_certified_join(tmp_path):
    ok, why = FH.check_universe_source(tmp_path)
    assert not ok and "missing" in why
    (tmp_path / "universe_standard_source.json").write_text(json.dumps(F.universe_source(False)), encoding="utf-8")
    ok, why = FH.check_universe_source(tmp_path)
    assert not ok and "FD_CERTIFIED_JOIN" in why
    (tmp_path / "universe_standard_source.json").write_text(json.dumps(F.universe_source(True)), encoding="utf-8")
    assert FH.check_universe_source(tmp_path) == (True, "")
    assert FH.price_stamp()["standard_ou25"]["over25"] == F.CERTIFIED
