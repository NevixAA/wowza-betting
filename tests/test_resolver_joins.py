"""Enrichment joins and API-Football fixture lookups match clubs through src.team_names.resolve
within the league (2026-11-02, owner rule). The old api_football_ou._norm_name key (suffix
stripping + its own alias dict) is gone; a club no rule proves is skipped, never guessed.

    python -m pytest tests/test_resolver_joins.py -q
"""
import numpy as np
import pandas as pd

from src import api_football_ou as AF
from src import data_loader as DL


def _hist():
    return pd.DataFrame({
        "league": ["England League Two"] * 2 + ["Argentina Primera Division"],
        "date": pd.to_datetime(["2026-10-01", "2026-10-08", "2026-10-02"]),
        "home_team": ["Colchester", "Oxford United", "Boca Juniors"],
        "away_team": ["Oxford United", "Colchester", "River Plate"],
    })


def test_source_names_resolve_onto_the_history_spelling_within_the_league():
    right = pd.DataFrame({
        "league": ["England League Two", "Argentina Primera Division", "Unknown League"],
        "home_team": ["Colchester United", "Platense", "Colchester United"],
        "away_team": ["Oxford", "Boca Juniors", "Oxford"],
    })
    out = DL._resolve_onto_history(right, _hist(), "home_team", "away_team", "t")
    # Colchester United / Oxford are alias-store spellings of the history clubs; Platense is NOT
    # River Plate (old stem tier) and the unknown league has no pool: both rows skipped.
    assert out[["_h", "_a"]].values.tolist() == [["Colchester", "Oxford United"]]


def test_resolve_onto_history_without_league_returns_nothing():
    right = pd.DataFrame({"home_team": ["Colchester"], "away_team": ["Oxford United"]})
    out = DL._resolve_onto_history(right, _hist(), "home_team", "away_team", "t")
    assert out.empty and {"_h", "_a"} <= set(out.columns)


def test_norm_name_identity_key_is_gone():
    assert not hasattr(AF, "_norm_name") and not hasattr(AF, "_POST_NORM_ALIASES")


def test_board_row_fixture_id_resolves_clubs_within_league_and_date():
    ids = {
        ("Brazil Serie A", "2026-11-05", "Vasco DA Gama", "Atletico-MG"): 11,
        ("Brazil Serie A", "2026-11-05", "Flamengo", "Atletico GO"): 12,
        ("Brazil Serie B", "2026-11-05", "Vasco DA Gama", "Atletico-MG"): 13,
    }
    row = {"league": "Brazil Serie A", "date": "2026-11-05",
           "home_team": "Vasco", "away_team": "Atletico-MG"}
    assert AF._row_fixture_id(row, ids) == 11
    # the old key stripped the state tag, so 'Atletico GO' and 'Atletico-MG' were one key
    row2 = dict(row, home_team="Flamengo", away_team="Atletico-MG")
    assert AF._row_fixture_id(row2, ids) is None
    assert AF._row_fixture_id(dict(row, date="2026-11-06"), ids) is None


def test_injury_features_resolve_board_names_onto_the_api_fixture():
    ctx = {"fixture_ids": {("Vasco DA Gama", "Santos", "2026-11-05"): 7},
           "top_scorers": {"Vasco DA Gama": [101], "Santos": [202]},
           "injuries": {7: {"Vasco DA Gama": [101]}}}
    out = AF.injury_features_from_context("Vasco", "Santos", "2026-11-05", ctx)
    assert out["key_attacker_absent_home"] == 1.0 and out["key_attacker_absent_away"] == 0.0
    miss = AF.injury_features_from_context("Platense", "Santos", "2026-11-05", ctx)
    assert all(np.isnan(v) for v in miss.values())
