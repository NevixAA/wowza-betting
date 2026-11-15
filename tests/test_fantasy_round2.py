"""Fantasy Data Truth round 2 (2026-11-02): FPL <-> API-Football id crosswalk, team-level goals
conceded for the clean-sheet model, and the profile placeholders read as missing.

    python -m pytest -p no:cacheprovider tests/test_fantasy_round2.py -q
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from player_model import fantasy as F
from player_model import fantasy_crosswalk as CW

ROOT = Path(__file__).resolve().parents[1]
REF = date(2026, 11, 1)


def _fpl(rows):
    return pd.DataFrame([{"known_name": "", "web_name": "", **r} for r in rows])


def _af(rows):
    return pd.DataFrame([{"af_alt_name": None, "af_age": np.nan, "af_birth_date": None, **r} for r in rows])


# ── (a) the crosswalk ────────────────────────────────────────────────────────────────────────────
def test_crosswalk_pairs_on_name_club_and_birth_date_with_refusals():
    fpl = _fpl([
        {"fpl_id": 1, "fpl_name": "Malo Gusto", "fpl_team": "Chelsea", "fpl_birth_date": "2003-05-19"},
        {"fpl_id": 2, "fpl_name": "Conor Bradley", "web_name": "Bradley", "fpl_team": "Liverpool",
         "fpl_birth_date": "2003-07-09"},
        {"fpl_id": 3, "fpl_name": "Kenny Tete", "web_name": "Tete", "fpl_team": "Fulham",
         "fpl_birth_date": "1995-10-09"},
        {"fpl_id": 4, "fpl_name": "James Smith", "fpl_team": "Everton", "fpl_birth_date": "2000-01-01"},
        {"fpl_id": 5, "fpl_name": "Ola Aina", "fpl_team": "Nott'm Forest", "fpl_birth_date": "1996-10-08"},
        {"fpl_id": 6, "fpl_name": "Kalvin Phillips", "fpl_team": "Man City", "fpl_birth_date": "1995-12-02"},
    ])
    af = _af([
        # initial + surname, same club, API-Football age one year behind (measured lag): paired
        {"af_player_id": 101, "af_name": "M. Gusto", "af_team": "Chelsea", "af_age": 22},
        # exact name outranks Bradley Barcola, whose first name equals Conor Bradley's web_name
        {"af_player_id": 102, "af_name": "Conor Bradley", "af_team": "Liverpool", "af_age": 22},
        {"af_player_id": 103, "af_name": "Bradley Barcola", "af_team": "Liverpool", "af_age": 23},
        # a different person who shares a short name, at another club: never paired
        {"af_player_id": 104, "af_name": "Tetê", "af_team": "Leicester", "af_age": 25},
        # two compatible candidates in one club: refused
        {"af_player_id": 105, "af_name": "J. Smith", "af_team": "Everton", "af_age": 26},
        {"af_player_id": 106, "af_name": "J. Smith", "af_team": "Everton", "af_age": 25},
        # provider birth date CONTRADICTS FPL's: refused although name and club agree
        {"af_player_id": 107, "af_name": "Ola Aina", "af_team": "Nottingham Forest",
         "af_birth_date": "1999-01-01"},
        # moved club, exact name and equal provider birth date: paired off-club
        {"af_player_id": 108, "af_name": "Kalvin Phillips", "af_team": "Ipswich",
         "af_birth_date": "1995-12-02"},
    ])
    pairs, refused = CW.build(fpl, af, REF)
    got = dict(zip(pairs["fpl_id"], pairs["af_player_id"]))
    assert got == {1: 101, 2: 102, 6: 108}
    meth = dict(zip(pairs["fpl_id"], pairs["method"]))
    assert meth[1] == "name+club+age" and meth[6] == "name+dob"
    why = dict(zip(refused["fpl_id"], refused["reason"]))
    assert why[4] == "ambiguous" and why[3] == "club unresolved" and why[5] == "no proven candidate"


def test_crosswalk_refuses_an_api_football_id_claimed_twice():
    fpl = _fpl([{"fpl_id": 1, "fpl_name": "Joe Rodon", "fpl_team": "Leeds", "fpl_birth_date": "1997-10-22"},
                {"fpl_id": 2, "fpl_name": "Joe Rodon", "fpl_team": "Leeds", "fpl_birth_date": "1997-10-22"}])
    af = _af([{"af_player_id": 9, "af_name": "Joe Rodon", "af_team": "Leeds", "af_age": 28}])
    pairs, refused = CW.build(fpl, af, REF)
    assert pairs.empty and set(refused["reason"]) == {"api-football id claimed twice"}


def test_no_birth_date_and_no_age_needs_an_exact_name():
    fpl = _fpl([{"fpl_id": 1, "fpl_name": "Robert Lynch Sánchez", "fpl_team": "Chelsea",
                 "fpl_birth_date": "1997-11-18"},
                {"fpl_id": 2, "fpl_name": "Lucas Digne", "fpl_team": "Aston Villa",
                 "fpl_birth_date": "1993-07-20"}])
    af = _af([{"af_player_id": 1, "af_name": "Robert Sánchez", "af_team": "Chelsea"},
              {"af_player_id": 2, "af_name": "Lucas Digne", "af_team": "Aston Villa"}])
    pairs, _ = CW.build(fpl, af, REF)
    assert pairs[["fpl_id", "af_player_id", "method"]].values.tolist() == [[2, 2, "name+club"]]


def test_names_unescape_html_entities():
    assert CW.names_compatible("D. O&apos;Shea", "Dara O'Shea")


def test_fpl_crosswalk_joins_on_ids_not_names():
    # Leeds' Rodrigo (API-Football id 1) used to be given Rodri's FPL id (web_name 'Rodrigo').
    fpl = pd.DataFrame({"fpl_id": [402, 500], "match_key": ["rodrigorodrihernandezcascante", "x"],
                        "web_key": ["rodrigo", "x"]})
    pl = pd.DataFrame({"player_id": [1, 2, 3], "player_name": ["Rodrigo", "Someone", "Left"]})
    assert F.fpl_crosswalk(pl, fpl, crosswalk=CW.load().iloc[0:0]).isna().all()
    cw = pd.DataFrame({"fpl_id": [500, 999], "af_player_id": [2, 3]})     # 999 not in live FPL set
    ids = F.fpl_crosswalk(pl, fpl, crosswalk=cw)
    assert math.isnan(ids.iloc[0]) and ids.iloc[1] == 500 and math.isnan(ids.iloc[2])


def test_stored_crosswalk_is_one_to_one_with_evidence():
    cw = CW.load()
    assert len(cw) > 0
    assert not cw["fpl_id"].duplicated().any() and not cw["af_player_id"].duplicated().any()
    assert set(cw["method"]) <= {"name+club+dob", "name+club+age", "name+club", "name+dob"}


# ── (b) team-level goals conceded ───────────────────────────────────────────────────────────────
def _hist():
    rows = []
    for fid, d in ((1, "2026-09-01"), (2, "2026-09-08")):
        for pos in ("G", "D", "M", "F"):
            rows.append({"team": "Arsenal", "fixture_id": fid, "date": pd.Timestamp(d),
                         "position": pos, "goals_conceded": 0})          # keeper row says 0 too
    rows.append({"team": "Arsenal", "fixture_id": 3, "date": pd.Timestamp("2026-09-15"),
                 "position": "D", "goals_conceded": 0})                   # no stored result
    return pd.DataFrame(rows)


def test_clean_sheet_input_is_the_team_score_not_a_player_row():
    res = pd.DataFrame({"fixture_id": [1, 2], "home_name": ["Arsenal", "Chelsea"],
                        "away_name": ["Fulham", "Arsenal"], "ft_home": [1, 3], "ft_away": [0, 1]})
    gc = F._team_goals_conceded(_hist(), res)
    assert dict(zip(gc["fixture_id"], gc["gc"])) == {1: 0.0, 2: 3.0}     # fixture 3: unknown
    assert F._team_conceded_map(_hist(), results=res) == {"Arsenal": 1.5}


def test_unresolved_club_gives_no_figure():
    res = pd.DataFrame({"fixture_id": [1, 2], "home_name": ["Leeds", "Chelsea"],
                        "away_name": ["Fulham", "Everton"], "ft_home": [1, 3], "ft_away": [0, 1]})
    assert F._team_conceded_map(_hist(), results=res) == {}


# ── (c) profile placeholders ───────────────────────────────────────────────────────────────────
def test_profile_placeholders_are_missing_in_fantasy():
    d = pd.DataFrame({"age": [25.0, 31.0], "height_cm": [180.0, 191.0], "age_peak_delta": [2.0, 4.0],
                      "height_advantage_score": [0.0, 1.2], "opp_def_mean_height": [180.0, 183.0]})
    m = F._mask_profile_placeholders(d)
    assert m["age"].isna().tolist() == [True, False] and m["age_peak_delta"].isna().tolist() == [True, False]
    assert m["height_cm"].isna().tolist() == [True, False]
    assert m["height_advantage_score"].isna().tolist() == [True, False]
    assert m["opp_def_mean_height"].isna().tolist() == [True, False]
    assert d["age"].tolist() == [25.0, 31.0]                              # input left untouched


def test_placeholder_comment_states_the_measured_share():
    src = (ROOT / "player_model" / "fantasy.py").read_text(encoding="utf-8")
    assert "~84% placeholder" not in src and "100% of rows carry the placeholders" in src
