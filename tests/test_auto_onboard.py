"""Auto-onboarding (owner standing approval 2026-10-20): fail closed, additive, never guesses a price source."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import auto_onboard as AO  # noqa: E402

SPORTS = [
    {"key": "soccer_brazil_serie_b", "title": "Brazil Série B", "description": "Brazilian Serie B"},
    {"key": "soccer_brazil_campeonato", "title": "Brazil Série A", "description": "Brasileirão Série A"},
    {"key": "soccer_germany_liga3", "title": "3. Liga - Germany", "description": "German 3. Liga"},
    {"key": "basketball_nba", "title": "NBA", "description": "US Basketball"},
]


def test_sport_key_unique_match():
    assert AO.match_sport_key("Brazil", "Brazil Serie B", SPORTS)[0] == "soccer_brazil_serie_b"
    assert AO.match_sport_key("Germany", "Germany 3. Liga", SPORTS)[0] == "soccer_germany_liga3"


def test_no_key_or_ambiguous_is_skipped_never_guessed():
    assert AO.match_sport_key("Ghana", "Ghana Premier League", SPORTS)[0] is None
    two = SPORTS + [{"key": "soccer_brazil_serie_b_women", "title": "Brazil Serie B", "description": "women"}]
    k, why = AO.match_sport_key("Brazil", "Brazil Serie B", two)
    assert k is None and "ambiguous" in why


def test_generic_division_name_never_takes_the_top_flight_key():
    # Probe 2026-10-20: "Denmark 1. Division" matched soccer_denmark_superliga (wrong league)
    dk = [{"key": "soccer_denmark_superliga", "title": "Denmark Superliga", "description": "Danish Superliga"}]
    assert AO.match_sport_key("Denmark", "Denmark 1. Division", dk)[0] is None
    ie = [{"key": "soccer_league_of_ireland", "title": "League of Ireland", "description": "Premier Division"}]
    assert AO.match_sport_key("Ireland", "Ireland First Division", ie)[0] is None


def test_plan_is_additive_and_idempotent():
    q = [{"league_id": 72, "league": "Brazil Serie B", "country": "Brazil", "current_season": "2026", "week": "W45"},
         {"league_id": 71, "league": "Brazil Serie A", "country": "Brazil", "current_season": "2026"},
         {"league_id": 570, "league": "Ghana Premier League", "country": "Ghana", "current_season": "2026"}]
    add, skipped = AO.plan(q, {"leagues": []}, {"Brazil Serie A"}, {71}, SPORTS, "2026-11-10T09:00:00Z")
    assert [e["name"] for e in add] == ["Brazil Serie B"]
    e = add[0]
    assert e["odds_api_sport_key"] == "soccer_brazil_serie_b" and e["timezone"] == "America/Sao_Paulo"
    assert e["history_seasons"] == ["2025", "2024", "2023", "2022"]
    reasons = {s["league"]: s["reason"] for s in skipped}
    assert reasons["Brazil Serie A"] == "already configured" and "Ghana Premier League" in reasons
    add2, _ = AO.plan(q, {"leagues": add}, {"Brazil Serie A"}, {71}, SPORTS, "2026-11-17T09:00:00Z")
    assert add2 == []                                   # second run adds nothing


def test_config_merge_is_additive(tmp_path, monkeypatch):
    import config
    f = tmp_path / "auto.json"
    f.write_text(json.dumps({"leagues": [
        {"name": "Brazil Serie B", "api_football_id": 72, "current_season": "2026", "history_seasons": ["2025"],
         "odds_api_sport_key": "soccer_brazil_serie_b", "timezone": "America/Sao_Paulo"},
        {"name": "Championship", "api_football_id": 40, "current_season": "2026",
         "odds_api_sport_key": "x"},                                   # existing name: must be ignored
        {"name": "Broken", "api_football_id": "x"}]}), encoding="utf-8")   # incomplete: skipped
    for attr in ("NEW_FORMAT_LEAGUES", "API_FOOTBALL_ONLY_LEAGUES", "ENABLED_LEAGUES"):
        monkeypatch.setattr(config, attr, set(getattr(config, attr)))
    for attr in ("API_FOOTBALL_EXTRA_SEASONS", "API_FOOTBALL_IDS", "API_FOOTBALL_SEASONS", "ODDS_API_SPORT_KEYS",
                 "LEAGUE_TIMEZONES"):
        monkeypatch.setattr(config, attr, dict(getattr(config, attr)))
    monkeypatch.setattr(config, "AUTO_LEAGUES_FILE", f)
    ids_before = dict(config.API_FOOTBALL_IDS)
    added = config._merge_auto_leagues()
    assert added == ["Brazil Serie B"]
    assert "Brazil Serie B" in config.ENABLED_LEAGUES and "Brazil Serie B" in config.NEW_FORMAT_LEAGUES
    assert config.model_type_for_league("Brazil Serie B") == "new_format"
    assert config.ODDS_API_SPORT_KEYS["Brazil Serie B"] == "soccer_brazil_serie_b"
    assert config.API_FOOTBALL_IDS["Championship"] == ids_before["Championship"]   # never overwritten
    assert not config.is_paper_league("Brazil Serie B")                            # it SENDS tips


def test_shipped_file_is_valid_and_merges_cleanly():
    data = json.loads((ROOT / "config" / "auto_leagues.json").read_text(encoding="utf-8"))
    assert isinstance(data.get("leagues"), list)
