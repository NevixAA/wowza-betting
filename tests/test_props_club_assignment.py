"""Player props must name a player at a club he plays for, in a match that club is in.

Two defects fixed on 2026-10-08, both reported by the owner from Telegram:

1. LEAGUE CORROBORATION LEAKED THROUGH TRANSFERS. `_team_leagues` was keyed on each player's
   CURRENT club, so a club inherited every competition its signings had played in elsewhere.
   Sporting CP looked like a La Liga 2 club and was accepted as "Sporting Gijón": Luis Suárez,
   Maxi Araújo, Issa Doumbia and Irankunda were tipped in Cádiz v Sporting Gijón.
2. STALE CLUBS. A player who moved to a league we do not collect kept his last collected club:
   Benzema tipped for Real Madrid (last appearance 2023), Verratti for PSG, Brozović for Inter —
   357 of 2,581 board rows.
"""
from __future__ import annotations

import inspect

from player_model import api_football
from player_model import config as pm_config
from player_model import predict
from src.team_names import resolve


def test_a_senior_side_never_matches_its_reserve_side():
    # Club identity is src.team_names.resolve only (api_football._same_club / _club_name_subset
    # were deleted 2026-11-02 once the notifier stopped using them).
    assert resolve("Real Sociedad", ["Real Sociedad II"]) is None
    assert resolve("Barcelona", ["Barcelona B"]) is None
    # the legitimate short-name cases still pass
    assert resolve("Plymouth", ["Plymouth Argyle"]) == "Plymouth Argyle"
    assert resolve("West Ham", ["West Ham United"]) == "West Ham United"


def test_sporting_cp_is_not_sporting_gijon_by_name():
    assert resolve("Sporting CP", ["Sporting Gijón"]) is None


def test_the_private_token_matchers_are_gone():
    assert not hasattr(api_football, "_same_club")
    assert not hasattr(api_football, "_club_name_subset")
    assert not hasattr(api_football, "_team_match")


def test_league_corroboration_uses_the_clubs_own_rows():
    src = inspect.getsource(predict.run_player_predictions)
    assert 'history_df.assign(_t=history_df["team"]' in src, \
        "a club's competitions must come from rows where IT played, not from its current players"
    assert "history_df.assign(_t=_club)" not in src


def test_stale_club_guard_is_wired():
    assert pm_config.STALE_DAYS == 120
    src = inspect.getsource(predict.run_player_predictions)
    assert "_eligible(p, c)" in src and "_squad_clubs" in src
