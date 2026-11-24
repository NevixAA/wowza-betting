"""Contracts AMBER 2026-11-22: no decimal price can be <= 1.0, and v91_shadow keeps p_v9_served.

Four new CONTRACTS-1 violations on main: two OddsAPI 1.0 quotes captured by src/predict.py
(book_odds_snapshots.csv), one 1.0 close taken by update_results (bets_ledger.csv), p_v9_served dropped
by src/shadow_compare.COLUMNS (v91_shadow.csv), and two pre-rule opposite-direction stake_audit rows.
These tests pin the three writer fixes; the historical rows are acknowledged in config/contract_baseline.json
with their corrections sidecars.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config


def test_shadow_columns_hold_every_required_contract_column():
    from src.contracts.schemas import V91_SHADOW_ROW
    from src.shadow_compare import COLUMNS
    missing = [c for c in V91_SHADOW_ROW.required_columns if c not in COLUMNS]
    assert missing == []


def test_append_shadow_keeps_p_v9_served(tmp_path, monkeypatch):
    import src.shadow_compare as SC
    monkeypatch.setattr(SC, "SHADOW_FILE", tmp_path / "v91_shadow.csv")
    old = pd.DataFrame({c: [""] for c in SC.COLUMNS if c != "p_v9_served"})
    old.loc[0, ["fixture_id", "market", "match_date"]] = ["a", "ou25", "2026-11-20"]
    old.to_csv(SC.SHADOW_FILE, index=False)                       # a file written before the fix
    n = SC.append_shadow([{"fixture_id": "b", "market": "ou25", "match_date": "2026-11-23",
                           "p_v9": 0.5, "p_v9_served": 0.55}])
    assert n == 1
    out = pd.read_csv(SC.SHADOW_FILE, dtype=str)
    assert "p_v9_served" in out.columns
    assert out.loc[out.fixture_id == "b", "p_v9_served"].iloc[0] == "0.55"
    assert out.loc[out.fixture_id == "a", "p_v9_served"].isna().all()   # missing stays missing


def test_json_close_skips_a_suspended_quote(tmp_path, monkeypatch):
    import update_results as U
    p = tmp_path / "odds_history_v9.json"
    p.write_text(json.dumps({"A vs B | 2026-11-21": [{"over": 2.0, "under": 1.9},
                                                    {"over": 1.0, "under": 1.8}],
                             "C vs D | 2026-11-21": [{"over": 1.0, "under": 1.0}]}), encoding="utf-8")
    monkeypatch.setattr(config, "ODDS_HISTORY_JSON", p)
    assert U._closing_odds_json("A", "B", "2026-11-21", "OVER") == 2.0
    assert U._closing_odds_json("A", "B", "2026-11-21", "UNDER") == 1.8
    assert np.isnan(U._closing_odds_json("C", "D", "2026-11-21", "OVER"))


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def test_predict_never_captures_or_selects_a_quote_at_or_below_one(tmp_path, monkeypatch):
    import requests

    import src.btts_odds as B
    import src.predict as P
    import src.rolling_store as RS
    league = sorted(set(config.ODDS_API_SPORT_KEYS) & set(config.ENABLED_LEAGUES))[0]
    monkeypatch.setattr(config, "ODDS_API_SPORT_KEYS", {league: config.ODDS_API_SPORT_KEYS[league]})
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(B, "fetch", lambda rows: rows)
    monkeypatch.setattr(RS, "roll", lambda *a, **k: {})
    ko = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def book(key, over, under):
        return {"key": key, "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "point": 2.5, "price": over}, {"name": "Under", "point": 2.5, "price": under}]}]}

    ev = [{"commence_time": ko, "home_team": "H", "away_team": "A",
           "bookmakers": [book("pinnacle", 1.0, 1.9), book("williamhill", 2.05, 1.8)]}]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(ev))
    out = P._fetch_odds_api()
    assert len(out) == 1
    assert out["odds_over25"].iloc[0] == 2.05                       # the 1.0 quote is no price
    snap = pd.read_csv(tmp_path / "book_odds_snapshots.csv")
    assert (snap["odds"] > 1.0).all()
    assert len(snap) == 3
