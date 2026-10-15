"""API-Football prop parsing - pinned on the value formats the API actually sends.

2026-10-15: API-Football dropped the "+" from per-player threshold values ("Tommy Conway - 1", was
"- 1+"); every SOT and assists quote from this source was silently discarded. Both forms must work,
and a bare-name value (a different bet, much longer prices) must still be skipped.
"""
from player_model import prop_odds_af as A


def test_threshold_forms():
    assert A._parse_line("Tommy Conway - 1") == 0.5          # new form: 1 or more
    assert A._parse_line("Tommy Conway - 2") == 1.5
    assert A._parse_line("Liberato Cacace - 1+") == 0.5      # old form still works
    assert A._parse_line("Over 1.5") == 1.5
    assert A._parse_line("David Strelec") is None            # bare name: not our bet


def test_buckets_and_names():
    assert A._SOT_LINE[A._parse_line("Erling Haaland - 1")] == "sot"
    assert A._SOT_LINE[A._parse_line("Erling Haaland - 3")] == "sot3"
    assert A._clean_player("Tommy Conway - 1") == "Tommy Conway"
    assert A._clean_player("Iliman-Cheikh Ndiaye - 2") == "Iliman-Cheikh Ndiaye"


def test_fetch_keeps_sot_and_skips_bare(monkeypatch):
    resp = {"response": [{"bookmakers": [{"name": "Bet365", "bets": [
        {"name": "Home Player Shots On Target Total", "values": [
            {"value": "Tommy Conway - 1", "odd": "1.36"}, {"value": "Tommy Conway - 2", "odd": "2.62"}]},
        {"name": "Away Player Shots On Target Total", "values": [
            {"value": "David Strelec", "odd": "12.00"}]},
    ]}]}]}
    import player_model.api_football as af
    monkeypatch.setattr(af, "_get", lambda *a, **k: resp)
    best, records = A.fetch([{"fixture_id": 1, "league": "Championship", "match": "A vs B",
                              "match_date": "2026-10-17"}], verbose=False)
    A.drain_quotes()
    assert best == {"Tommy Conway|sot": 1.36, "Tommy Conway|sot2": 2.62}
    assert all(r["player"] != "David Strelec" for r in records)
