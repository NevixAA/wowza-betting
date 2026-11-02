"""team_frame findings (CLAUDE_PROMPT_2 P0/P1 #1 and P2 'v9 training odds / stats', 2026-11-02).

Each test FAILS on data-truth-100 HEAD 84e06800:
  * the same match twice in the team training frame (same day under two spellings, mojibake, or
    1-3 days apart) - and a row whose last-5 window holds its own match;
  * impossible shot counts kept as counts;
  * a negative-margin O/U pair read as an overround of exactly 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as DL


def _row(date, lg, h, a, hg, ag, **kw):
    return {"date": pd.Timestamp(date), "league": lg, "season": "2024", "home_team": h,
            "away_team": a, "home_goals": float(hg), "away_goals": float(ag), **kw}


def _frame(rows):
    return pd.DataFrame(rows)


# ── one match, one row ───────────────────────────────────────────────────────────────────────────
def test_one_side_same_day_alias_is_proven_by_evidence_not_spelling():
    """'U. Cluj' / 'Universitatea Cluj': team_names cannot spell-match them, but on 3 dates the SAME
    opponent played them on the SAME day at the SAME score - one club. HEAD required a spelling proof
    too, so the pair (93 dates in the real history) stayed two teams with every match twice."""
    rows = []
    for d, opp, s in (("2024-03-01", "Rapid", (1, 0)), ("2024-03-08", "Farul", (2, 2)),
                      ("2024-03-15", "Sepsi", (0, 1))):
        rows += [_row(d, "Romanian Superliga", "U. Cluj", opp, *s),
                 _row(d, "Romanian Superliga", "Universitatea Cluj", opp, *s)]
    rows.append(_row("2024-04-01", "Romanian Superliga", "Universitatea Cluj", "Rapid", 1, 1))
    m = DL._one_side_aliases(_frame(rows))
    assert m == {"Romanian Superliga": {"U. Cluj": "Universitatea Cluj"}}
    assert DL._EVIDENCE_ONLY_ALIASES["Romanian Superliga"]["U. Cluj"][0] == "Universitatea Cluj"


def test_one_side_alias_never_merges_two_clubs_that_met():
    rows = []
    for d in ("2024-03-01", "2024-03-08", "2024-03-15"):
        rows += [_row(d, "Romanian Superliga", "U. Cluj", "Rapid", 1, 0),
                 _row(d, "Romanian Superliga", "Universitatea Cluj", "Rapid", 1, 0)]
    rows.append(_row("2024-05-01", "Romanian Superliga", "U. Cluj", "Universitatea Cluj", 2, 0))
    assert DL._one_side_aliases(_frame(rows)) == {}


def test_near_date_copy_is_removed_and_the_richer_row_survives():
    """MLS in fd_history twice: local date with shots (2023-08-30), UTC date without (08-31)."""
    rich = _row("2023-08-30", "USA MLS", "New England Revolution", "New York Red Bulls", 1, 0,
                home_shots=15.0, away_shots=10.0, odds_over25=np.nan, row_source="fd_cache")
    thin = _row("2023-08-31", "USA MLS", "New England Revolution", "New York Red Bulls", 1, 0,
                home_shots=np.nan, away_shots=np.nan, odds_over25=2.05, row_source="fd_cache")
    other = _row("2023-09-02", "USA MLS", "Orlando City", "New York Red Bulls", 2, 2,
                 home_shots=9.0, away_shots=8.0, odds_over25=1.9, row_source="fd_cache")
    out = DL._one_match_one_row(_frame([rich, thin, other]))
    assert len(out) == 2
    ne = out[out["home_team"] == "New England Revolution"].iloc[0]
    assert str(ne["date"])[:10] == "2023-08-30" and ne["home_shots"] == 15.0
    assert ne["odds_over25"] == 2.05                  # real value of the SAME match, from its copy
    assert out.attrs["one_match_report"]["near_date_pairs"] == 1


def test_source_rule_prefers_football_data_over_the_archive():
    a = _row("2024-01-06", "Championship", "Hull", "Stoke", 2, 1, home_shots=12.0, away_shots=7.0,
             home_corners=5.0, row_source="backtest_archive")
    f = _row("2024-01-07", "Championship", "Hull", "Stoke", 2, 1, home_shots=np.nan,
             away_shots=np.nan, home_corners=np.nan, row_source="football_data")
    out = DL._one_match_one_row(_frame([a, f]))
    assert len(out) == 1
    r = out.iloc[0]
    assert r["row_source"] == "football_data" and str(r["date"])[:10] == "2024-01-07"
    assert r["home_shots"] == 12.0 and r["home_corners"] == 5.0


def test_real_matches_are_never_merged():
    rows = [_row("2024-01-06", "Championship", "Hull", "Stoke", 2, 1),
            _row("2024-01-10", "Championship", "Hull", "Stoke", 2, 1),     # 4 days: not a copy
            _row("2024-01-07", "Championship", "Stoke", "Hull", 2, 1),     # reversed fixture
            _row("2024-01-08", "Championship", "Hull", "Stoke", 0, 0)]     # other score
    assert len(DL._one_match_one_row(_frame(rows))) == 4


def test_one_side_same_day_copy_is_one_match_without_renaming():
    rows = [_row("2024-08-04", "Austrian Bundesliga", "Rapid Vienna", "Tirol", 3, 1,
                 home_shots=14.0, row_source="fd_cache"),
            _row("2024-08-04", "Austrian Bundesliga", "Rapid Vienna", "WSG Tirol FC Wattens", 3, 1,
                 home_shots=np.nan, row_source="fd_cache"),
            _row("2024-08-04", "Austrian Bundesliga", "Sturm Graz", "LASK", 3, 1)]
    out = DL._one_match_one_row(_frame(rows))
    assert len(out) == 2
    assert set(out["away_team"]) == {"Tirol", "LASK"}


def test_mojibake_name_is_repaired_and_its_copy_removed():
    rows = [_row("2024-08-04", "Bundesliga 2", "Greuther Furth", "Preußen Münster", 3, 1,
                 row_source="fd_cache"),
            _row("2024-08-04", "Bundesliga 2", "Greuther Furth", "PreuÃ\x9fen MÃ¼nster", 3, 1,
                 row_source="fd_cache"),
            _row("2024-08-11", "Bundesliga 2", "PreuÃ\x9fen MÃ¼nster", "Hannover", 0, 0,
                 row_source="fd_cache")]
    out = DL._one_match_one_row(DL._repair_club_text(_frame(rows)))
    assert len(out) == 2
    assert set(out["home_team"]) | set(out["away_team"]) == {"Greuther Furth", "Preußen Münster",
                                                            "Hannover"}


def test_download_bytes_are_decoded_not_guessed():
    utf8 = "HomeTeam,AwayTeam\nPreußen Münster,Köln\n".encode("utf-8")
    latin = "HomeTeam,AwayTeam\nPreußen Münster,Köln\n".encode("latin-1")
    bom = b"\xef\xbb\xbf" + utf8
    for b in (utf8, latin, bom):
        assert DL._decode_csv(b) == "HomeTeam,AwayTeam\nPreußen Münster,Köln\n"


def test_no_row_holds_its_own_match_in_its_last5_window():
    from src.integrity.duplicate_matches import duplicate_pairs, self_containing_rows
    rows = [_row("2023-06-03", "USA MLS", "Philadelphia Union", "CF Montreal", 3, 0),
            _row("2023-06-04", "USA MLS", "Philadelphia Union", "CF Montreal", 3, 0),
            _row("2023-06-10", "USA MLS", "Philadelphia Union", "Orlando City", 1, 1)]
    df = _frame(rows)
    before = self_containing_rows(df, duplicate_pairs(df))
    assert int(before.sum()) == 1
    clean = DL._one_match_one_row(df)
    assert int(self_containing_rows(clean, duplicate_pairs(clean)).sum()) == 0


# ── impossible stats ────────────────────────────────────────────────────────────────────────────
def test_impossible_shot_counts_become_missing_with_a_count():
    rows = [_row("2024-01-01", "League One", "A", "B", 1, 0, home_shots=3.0, away_shots=5.0,
                 home_sot=4.0, away_sot=2.0),                         # home shots < SOT
            _row("2024-01-02", "League One", "C", "D", 0, 3, home_shots=6.0, away_shots=2.0,
                 home_sot=2.0, away_sot=2.0),                         # away shots < goals
            _row("2024-01-03", "League One", "E", "F", 1, 1, home_shots=0.0, away_shots=0.0,
                 home_sot=0.0, away_sot=0.0),                         # all-zero placeholder
            _row("2024-01-04", "League One", "G", "H", 2, 0, home_shots=11.0, away_shots=4.0,
                 home_sot=5.0, away_sot=1.0)]                         # a true row, untouched
    out = DL._impossible_stats_to_missing(_frame(rows))
    assert out.loc[0, ["home_shots", "home_sot"]].isna().all() and out.loc[0, "away_shots"] == 5.0
    assert out.loc[1, ["away_shots", "away_sot"]].isna().all() and out.loc[1, "home_shots"] == 6.0
    assert out.loc[2, ["home_shots", "away_shots", "home_sot", "away_sot"]].isna().all()
    assert out.loc[3, "home_shots"] == 11.0 and out.loc[3, "home_sot_ratio"] == pytest.approx(5 / 11)
    assert pd.isna(out.loc[0, "home_sot_ratio"])
    assert out.attrs["impossible_stats"] == {"home_shots_lt_sot": 1, "home_shots_lt_goals": 0,
                                             "away_shots_lt_sot": 0, "away_shots_lt_goals": 1,
                                             "all_zero_shot_rows": 1}


# ── negative-margin O/U pairs ───────────────────────────────────────────────────────────────────
def test_negative_margin_pair_is_not_a_price_and_never_an_overround_of_zero():
    from src.integrity.price_provenance import overround_or_missing
    o = pd.Series([2.10, 1.95, np.nan, 0.0])
    u = pd.Series([2.05, 1.85, 1.90, 1.90])
    m = overround_or_missing(o, u)
    assert pd.isna(m[0])                                  # 1/2.10 + 1/2.05 < 1: two books' best
    assert m[1] == pytest.approx(1 / 1.95 + 1 / 1.85 - 1)
    assert pd.isna(m[2]) and pd.isna(m[3])
    assert pd.isna(overround_or_missing(2.10, 2.05))
    f = DL._flag_ou25_pair(pd.DataFrame({"odds_over25": o, "odds_under25": u}))
    assert f["ou25_pair_is_price"].tolist()[:2] == [False, True]
    assert f["ou25_pair_is_price"].isna().tolist()[2:] == [True, True]
    assert f["odds_over25"].iloc[0] == 2.10               # the quotes themselves are kept
