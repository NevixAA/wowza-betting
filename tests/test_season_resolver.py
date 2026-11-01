"""R2-A finding 6: league-aware API-Football season resolution; no generic "2025".

Covers the rollovers (European July start, calendar-year January, J1's 2026 switch to an autumn-spring
calendar keyed by END year), SEASON_UNKNOWN for anything unregistered / an unknown provider / an
unparseable date, aliases, the auto-onboarding overlay, and that no code path still reads
config.API_SEASON or falls back to a literal year.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import season_resolver as SR  # noqa: E402
from src.season_resolver import SEASON_UNKNOWN, resolve_api_season  # noqa: E402


@pytest.mark.parametrize("league,date,season", [
    # European start-year leagues roll on 1 July
    ("Championship", "2026-06-30", "2025"), ("Championship", "2026-07-01", "2026"),
    ("Championship", "2026-12-31", "2026"), ("Championship", "2027-01-01", "2026"),
    ("Denmark Superliga", "2026-07-18", "2026"), ("Denmark Superliga", "2027-05-25", "2026"),
    ("Mexico Liga MX", "2027-04-10", "2026"), ("Saudi Pro League", "2026-08-20", "2026"),
    ("Scottish League Two", "2026-11-01", "2026"),
    # calendar-year leagues roll on 1 January
    ("Brazil Serie A", "2026-12-06", "2026"), ("Brazil Serie A", "2027-01-01", "2027"),
    ("USA MLS", "2026-12-06", "2026"), ("K-League 1", "2026-03-01", "2026"),
    ("Argentina Primera Division", "2027-02-01", "2027"),
    # J1: calendar year until the 2026 switch, then END-year keyed (verified on the API 2026-10-22)
    ("Japan J-League", "2025-11-01", "2025"), ("Japan J-League", "2026-03-01", "2026"),
    ("Japan J-League", "2026-06-30", "2026"), ("Japan J-League", "2026-08-07", "2027"),
    ("Japan J-League", "2027-05-30", "2027"), ("Japan J-League", "2027-08-10", "2028"),
    ("World Cup 2026", "2026-06-20", "2026"),
])
def test_rollovers(league, date, season):
    assert resolve_api_season(league, date) == season


def test_accepts_date_types():
    for d in ("2026-11-01", pd.Timestamp("2026-11-01 15:00"), pd.Timestamp("2026-11-01", tz="UTC"),
              pd.Timestamp("2026-11-01").date(), pd.Timestamp("2026-11-01").to_pydatetime()):
        assert resolve_api_season("Japan J-League", d) == "2027"


def test_default_date_is_today_and_matches_config_table():
    for lg, s in config.API_FOOTBALL_SEASONS.items():
        assert resolve_api_season(lg) == s


@pytest.mark.parametrize("league", ["Austria Bundesliga II", "championship", "Narnia Premier", "", None,
                                    "Brazil Serie B"])
def test_unknown_league_is_season_unknown_never_a_year(league):
    r = SR.resolve(league, "2026-11-01")
    assert r.season == SEASON_UNKNOWN and r.reason
    assert not SR.is_known(r.season)


def test_alias_resolves_to_canonical_rule():
    assert resolve_api_season("Danish Superliga", "2026-11-01") == "2026"
    assert SR.resolve("Austria Bundesliga", "2026-11-01").canonical == "Austrian Bundesliga"


def test_unknown_provider_and_bad_date():
    assert resolve_api_season("Championship", "2026-11-01", provider="oddsapi") == SEASON_UNKNOWN
    assert resolve_api_season("Championship", "not a date") == SEASON_UNKNOWN
    assert resolve_api_season("Championship", pd.NaT) == SEASON_UNKNOWN


def test_auto_onboarded_league_uses_its_static_season(tmp_path, monkeypatch):
    f = tmp_path / "auto.json"
    f.write_text(json.dumps({"leagues": [{"name": "Brazil Serie B", "current_season": "2026"},
                                         {"name": "Ruled League", "season_rule": {"type": "calendar_year"}}]}),
                 encoding="utf-8")
    monkeypatch.setattr(SR, "AUTO_LEAGUES_FILE", f)
    r = SR.resolve("Brazil Serie B", "2027-03-01")
    assert r.season == "2026" and "static" in r.rule and r.source.endswith("auto_leagues.json")
    assert resolve_api_season("Ruled League", "2027-03-01") == "2027"
    h = SR.health("2026-11-01")
    row = next(x for x in h["leagues"] if x["league"] == "Brazil Serie B")
    assert row["status"] == "WARN"


def test_health_lists_every_active_league_known(tmp_path):
    h = SR.write_health(tmp_path / "s.json", today="2026-11-01")
    d = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    active = {r["league"]: r for r in d["leagues"] if r["active"]}
    assert set(active) == set(config.ENABLED_LEAGUES) - set(config.AUTO_LEAGUES)
    assert all(SR.is_known(r["season"]) for r in active.values())
    assert d["status"] in ("PASS", "WARN") and h["resolver_version"] == SR.RESOLVER_VERSION
    assert active["Japan J-League"]["season"] == "2027"


def test_api_season_is_deprecated_sentinel():
    assert config.API_SEASON == SEASON_UNKNOWN
    from player_model import config as pcfg
    assert pcfg.API_SEASON == SEASON_UNKNOWN


def test_no_code_reads_api_season_or_falls_back_to_a_literal_year():
    """The audit, as a test: no call site may read config.API_SEASON or default a season lookup to a
    literal year / the current calendar year."""
    bad = []
    pat = [re.compile(r"\bAPI_SEASON\b(?!_)"),
           re.compile(r"API_FOOTBALL_SEASONS\.get\([^)]*,\s*(\"20\d\d\"|str\(now|str\(now_year|_cfg\.API_SEASON|config\.API_SEASON)")]
    for p in list(ROOT.glob("*.py")) + list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) \
            + list((ROOT / "player_model").rglob("*.py")):
        if p.name in ("config.py",) or "_archive" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            code = line.split("#", 1)[0]
            if any(rx.search(code) for rx in pat) and "R2 finding 6" not in line:
                bad.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    bad = [b for b in bad if not b.startswith(("src" + "\\season_resolver.py", "src/season_resolver.py"))
           and "_API_SEASON_CACHE" not in b]
    assert not bad, "\n".join(bad)


def test_api_football_ou_skips_on_unknown_season(monkeypatch):
    import src.api_football_ou as AF
    calls = []
    monkeypatch.setattr(AF, "_KEY", "x")
    monkeypatch.setattr(AF, "_get", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(AF, "_resolve_team_id", lambda *a, **k: calls.append(("team",) + a) or None)
    monkeypatch.setitem(config.API_FOOTBALL_IDS, "Narnia Premier", 999999)
    before = len(SR.degradations())
    assert AF.fetch_team_season_stats("Team", "Narnia Premier") == {}
    assert calls == []                                            # never asked the API with a guessed year
    assert len(SR.degradations()) == before + 1
    # a known league resolves its own season, explicitly passed through
    AF.fetch_team_season_stats("Team", "Japan J-League", kickoff_date="2026-11-01")
    assert calls and calls[-1][-1] == "2027"
