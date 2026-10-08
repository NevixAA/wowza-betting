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

from player_model import config as pm_config
from player_model import predict
from player_model.api_football import _club_name_subset, _same_club


def test_a_senior_side_never_subset_matches_its_reserve_side():
    assert not _club_name_subset("Real Sociedad", "Real Sociedad II")
    assert not _club_name_subset("Barcelona", "Barcelona B")
    # the legitimate short-name cases still pass
    assert _club_name_subset("Plymouth", "Plymouth Argyle")
    assert _club_name_subset("West Ham", "West Ham United")


def test_sporting_cp_is_not_sporting_gijon_by_name():
    assert not _same_club("Sporting CP", "Sporting Gijón")


def test_league_corroboration_uses_the_clubs_own_rows():
    src = inspect.getsource(predict.run_player_predictions)
    assert 'history_df.assign(_t=history_df["team"]' in src, \
        "a club's competitions must come from rows where IT played, not from its current players"
    assert "history_df.assign(_t=_club)" not in src


def test_stale_club_guard_is_wired():
    assert pm_config.STALE_DAYS == 120
    src = inspect.getsource(predict.run_player_predictions)
    assert "_eligible(p, c)" in src and "_squad_clubs" in src
