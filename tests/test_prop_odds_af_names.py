"""API-Football prop values: "Under"/"Over" are WORDS, never substrings of a player's name (2026-10-20)."""
from player_model.prop_odds_af import _clean_player, _is_under


def test_names_containing_under_are_not_under_quotes():
    for name in ("Noah Saunders", "Harvey Saunders", "Isaac Ogundere", "Jostein Gundersen"):
        assert not _is_under(name)
        assert _clean_player(name) == name


def test_names_containing_over_are_not_mangled():
    assert _clean_player("Grover Smith") == "Grover Smith"
    assert _clean_player("Overmars") == "Overmars"


def test_real_side_words_still_parse():
    assert _is_under("Harry Kane - Under 1.5") and _clean_player("Harry Kane - Under 1.5") == "Harry Kane"
    assert not _is_under("Harry Kane Over 1.5") and _clean_player("Harry Kane Over 1.5") == "Harry Kane"
    assert _clean_player("Liberato Cacace - 1+") == "Liberato Cacace"
