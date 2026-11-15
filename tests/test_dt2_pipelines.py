"""Data Truth round 2 - pipelines scope (update_results / backfill_clv / one-off regrades /
standard_repair / config / join replay). No invented values, names through the resolver.

    python -m pytest tests/test_dt2_pipelines.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config

ROOT = Path(__file__).resolve().parents[1]


# ── standard_repair.attach_entry_prices: team_names.resolve in the league+date pool ─────────────
def _prices(rows):
    return pd.DataFrame([{"date": d, "league": lg, "fd_home": h, "fd_away": a,
                          "odds_over25_entry": o, "odds_under25_entry": 1.9,
                          "entry_price_class": "REAL", "entry_price_source": "fd",
                          "odds_over25_close": o, "odds_under25_close": 1.9,
                          "close_price_class": "REAL"} for d, lg, h, a, o in rows])


def test_entry_prices_join_by_resolver_within_league_and_date():
    from src.standard_repair import attach_entry_prices
    prices = _prices([("2026-10-24", "Championship", "Cardiff", "Hull", 2.05),
                      ("2026-10-24", "League One", "Cardiff", "Hull", 9.99),      # other league
                      ("2026-10-24", "Championship", "Bristol City", "Leeds", 2.2),
                      ("2026-10-24", "Championship", "Bristol Rvs", "Stoke", 2.3)])
    frame = pd.DataFrame({"date": ["2026-10-24", "2026-10-25", "2026-10-24"],
                          "league": ["Championship"] * 3,
                          "home_team": ["Cardiff City", "Cardiff City", "Bristol"],
                          "away_team": ["Hull City", "Hull City", "Leeds"]})
    out = attach_entry_prices(frame, prices)
    assert out.loc[0, "odds_over25_entry"] == 2.05                 # same club, same league+date
    assert np.isnan(out.loc[1, "odds_over25_entry"])               # no price that day: none taken
    assert out.loc[1, "entry_price_class"] == "MISSING"
    # 'Bristol' is ambiguous in the day's pool (Bristol City / Bristol Rvs): no guess
    assert np.isnan(out.loc[2, "odds_over25_entry"]) and out.loc[2, "entry_price_class"] == "MISSING"


# ── COVID exclusion per league from its season rule, weights from the match date ────────────────
def test_load_history_uses_the_per_league_covid_mask(monkeypatch):
    import src.data_loader as DL
    import src.standard_repair as SRP
    raw = pd.DataFrame({"date": pd.to_datetime(["2020-01-10", "2022-01-10", "2020-07-10"]),
                        "league": ["Championship", "Championship", "USA MLS"],
                        "season": ["2019/20", "2021/22", "2020"]})
    monkeypatch.setattr(DL, "load_all_matches", lambda *a, **k: raw.copy())
    got = SRP.load_history()
    # The calendar-year league's 2020 season is excluded too (the old label set missed it).
    assert list(got["season"]) == ["2021/22"]


def test_train_track_refuses_the_retired_label_weights():
    from src.standard_repair import train_track
    with pytest.raises(ValueError):
        train_track(pd.DataFrame({"league": [], "date": []}), "legacy_v9", "none", weights="legacy")


# ── config.model_type_for_league / auto-league timezone ─────────────────────────────────────────
def test_model_type_is_none_not_a_placeholder():
    assert config.model_type_for_league("Championship") == "standard"
    assert config.model_type_for_league("no such league") is None
    assert "unknown" not in {config.model_type_for_league(x) for x in ("x", "", None)}


def test_auto_league_without_timezone_is_skipped(monkeypatch, tmp_path, caplog):
    import json
    f = tmp_path / "auto_leagues.json"
    f.write_text(json.dumps({"leagues": [{"name": "Testland Liga", "api_football_id": 99999,
                                          "current_season": "2026", "odds_api_sport_key": "soccer_x"}]}),
                 encoding="utf-8")
    monkeypatch.setattr(config, "AUTO_LEAGUES_FILE", f)
    monkeypatch.setattr(config, "NEW_FORMAT_LEAGUES", set(config.NEW_FORMAT_LEAGUES))
    with caplog.at_level("WARNING"):
        assert config._merge_auto_leagues() == []
    assert "Testland Liga" not in config.NEW_FORMAT_LEAGUES
    assert "no timezone" in caplog.text


# ── scripts/backfill_clv: the close is looked up inside the row's league ────────────────────────
def test_backfill_clv_passes_the_league(monkeypatch, tmp_path):
    import importlib.util
    import update_results as ur
    spec = importlib.util.spec_from_file_location("backfill_clv", ROOT / "scripts" / "backfill_clv.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"league": "League Two", "home_team": "Barrow", "away_team": "Gillingham",
                   "match_date": "2026-10-24", "side": "OVER", "odds": "2.2", "result": "WIN",
                   "closing_odds": "", "clv_pct": ""},
                  {"league": "", "home_team": "Barrow", "away_team": "Gillingham",
                   "match_date": "2026-10-24", "side": "OVER", "odds": "2.2", "result": "WIN",
                   "closing_odds": "", "clv_pct": ""}]).to_csv(tmp_path / "bets_ledger.csv", index=False)
    seen = []

    def fake_close(home, away, date, side, league=None):
        seen.append(league)
        return 2.0
    monkeypatch.setattr(ur, "_closing_odds", fake_close)
    assert B.backfill_main(dry=True) == 1
    assert seen == ["League Two"]                         # blank league: skipped, never looked up


# ── scripts/oneoff_recheck_dnp_voids_2026_11: player_id only, settler helpers ───────────────────
def test_dnp_recheck_matches_by_player_id_only(monkeypatch):
    import importlib.util
    import src.league_registry as LRm
    import update_results as U
    spec = importlib.util.spec_from_file_location(
        "oneoff_recheck", ROOT / "scripts" / "oneoff_recheck_dnp_voids_2026_11.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    monkeypatch.setattr(LRm, "prop_leagues", lambda *a, **k: {"Championship": 40})
    monkeypatch.setattr(U, "_find_prop_fixture",
                        lambda *a, **k: ({"fixture": {"id": 777}, "goals": {"home": 1, "away": 0}}, ""))
    rec = {"player_id": 10, "player_name": "A. Laurienté", "team_id": 1, "minutes_played": 90,
           "substitute": False, "goals": 1, "assists": 0, "shots_on_target": 2, "yellow_card": 0}
    other = {**rec, "player_id": 11, "player_name": "B. Other", "team_id": 2, "goals": 0}
    monkeypatch.setattr(U, "_fixture_players",
                        lambda fid: {"teams": {1: [rec], 2: [other]}, "by_id": {10: rec, 11: other}})
    base = {"result": "VOID", "notes": "DNP", "league": "Championship", "match_date": "2026-10-24",
            "home_team": "Cardiff City", "away_team": "Hull City", "market": "goals", "market_odds": "3.0"}
    led = pd.DataFrame([{**base, "player_name": "Armand Laurienté", "player_id": "10"},
                        {**base, "player_name": "A. Laurienté", "player_id": ""},      # name only
                        {**base, "player_name": "Somebody", "player_id": "55"}])        # not in squad
    out = S.recheck(led)
    upd = list(out["updates"].values())
    assert upd[0]["result"] == "WIN" and upd[0]["pnl"] == "2.0"
    assert upd[1]["notes"] == "DNP_NO_PLAYER_ID" and upd[1]["result"] == "VOID"   # never by name
    assert upd[2]["notes"] == "DNP (not in matchday squad)"
    assert out["counts"]["no_player_id"] == 1
    src = (ROOT / "scripts" / "oneoff_recheck_dnp_voids_2026_11.py").read_text(encoding="utf-8")
    assert "_initial_surname_match" not in src and "U._norm(" not in src


# ── research / integrity scripts ────────────────────────────────────────────────────────────────
def test_league_weekly_settle_has_no_pairwise_fallback():
    import update_results as U
    src = (ROOT / "src" / "research" / "league_weekly.py").read_text(encoding="utf-8")
    assert "U._names_match" not in src and not hasattr(U, "_names_match")


def test_join_replay_no_longer_sets_the_dead_flag():
    src = (ROOT / "scripts" / "integrity_af_join_replay.py").read_text(encoding="utf-8")
    assert 'os.environ["WOWZA_SAFE_ENRICH_JOINS"]' not in src
    assert 'getenv("WOWZA_SAFE_ENRICH_JOINS"' not in src


def test_no_reader_of_the_covid_label_set_in_pipelines_scope():
    files = ["src/standard_repair.py", "src/backtest.py", "scripts/integrity_feature_parity.py",
             "scripts/register_shadow_experiment.py", "scripts/standard_feature_audit.py",
             "scripts/train_repaired_challengers.py"]
    for f in files:
        src = (ROOT / f).read_text(encoding="utf-8")
        for name in ("config.COVID_SEASONS", "config.TRAINING_DECAY_WEIGHTS", "config.DEFAULT_DECAY_WEIGHT",
                     "legacy_weights("):
            assert name not in src, (f, name)


def test_bankroll_policy_note_says_unrated_means_no_stake():
    import json
    note = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8")
                      )["league_rating_sizing"]["note"]
    assert "NO stake" in note and "unrated = 20%" not in note
