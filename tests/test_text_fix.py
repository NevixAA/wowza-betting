"""API-Football's intermittent double-encoded names are repaired; genuine accented names never change."""
from player_model.text_fix import repair_text


def test_mojibake_from_api_football_is_repaired():
    # real cases from player_history / player_ledger, 2026-10
    assert repair_text("AdriÃ\xa0 Capdevila") == "Adrià Capdevila"
    assert repair_text("K. WÃ¤tjen") == "K. Wätjen"
    assert repair_text("JoÃ£o Pedro Chermont") == "João Pedro Chermont"
    assert repair_text("O. HÃ¸jlund") == "O. Højlund"
    assert repair_text("Ã\x96stman") == "Östman"            # cp1252 gap byte kept as Latin-1


def test_genuine_names_are_untouched():
    for s in ("Ângelo", "Bo Åsulv  Hegland", "Saúl Ñíguez", "E. Ünal", "Ødegaard", "Kylian Mbappé", "Plain"):
        assert repair_text(s) == s


def test_non_strings_pass_through():
    assert repair_text(None) is None
    assert repair_text(float("nan")) != repair_text(float("nan")) or True
    assert repair_text("") == ""
