"""Owner 2026-11-09: "100% real data, no fakes, no bugs, no defaults, no global vars".

No serving path may tier, gate or stake a tip on a global / pooled / hand-set bar. A league x market
tiers ONLY on its own owner-applied sniper / marksman / valuable bars; without them: AVOID, reason
NO_LEAGUE_BAR, no stake, no tip. Data truth rule 6 fails any serving module that reads a global bar.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config
import src.betting as B
import src.side_thresholds as ST
from src.integrity import data_truth as DT
from src.integrity import side_edge_devig as SE

OWN = {"sniper_th": 0.12, "marksman_th": 0.09, "valuable_th": 0.05, "approved": False, "status": "UNPROVEN"}


@pytest.fixture
def bars_env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    ledger = tmp_path / "threshold_applied.json"
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(ST, "APPLIED_LEDGER", ledger)

    def write(std=None, side=None, applied=()):
        (models / "best_params_standard.json").write_text(json.dumps(std or {}), encoding="utf-8")
        (models / "best_params_side_markets.json").write_text(json.dumps(side or {}), encoding="utf-8")
        ledger.write_text(json.dumps({"history": [{"entries": list(applied)}]}), encoding="utf-8")
        ST._APPLIED_CACHE.clear()
        B._STANDARD_TH_CACHE = None
    yield write
    ST._APPLIED_CACHE.clear()
    B._STANDARD_TH_CACHE = None


def _board(leagues, edge):
    """O/U 2.5 rows whose OVER edge is `edge` against a 2.00 price."""
    p = 0.5 + edge
    return pd.DataFrame({"league": leagues, "p_over25": [p] * len(leagues),
                         "odds_over25": [2.0] * len(leagues), "odds_under25": [2.0] * len(leagues),
                         "home_scored_last5": 1.0, "away_scored_last5": 1.0,
                         "home_conceded_last5": 1.0, "away_conceded_last5": 1.0})


# ── the resolver ──────────────────────────────────────────────────────────────────────────────
def test_own_bars_require_all_three_ordered_and_owner_applied(bars_env):
    bars_env()
    ok = ST.own_bars(OWN, "ou25", "L", applied=frozenset({"ou25|L"}))
    assert ok == ((0.12, 0.09, 0.05), "")
    assert ST.own_bars(OWN, "ou25", "L", applied=frozenset())[0] is None
    assert ST.own_bars({k: v for k, v in OWN.items() if k != "valuable_th"}, "ou25", "L",
                       applied=frozenset({"ou25|L"}))[0] is None
    assert ST.own_bars({}, "btts", "L", applied=frozenset({"btts|L"}))[1].startswith(ST.NO_LEAGUE_BAR)


def test_tier_for_without_bars_is_avoid():
    assert ST.tier_for(0.99, None) == "AVOID"
    assert ST.tier_for(float("nan"), (0.1, 0.08, 0.04)) == "AVOID"


def test_the_side_thresholds_module_has_no_global_bar():
    assert not hasattr(ST, "global_bars") and not hasattr(ST, "GLOBAL")
    assert not hasattr(config, "SIDE_GLOBAL_BARS")          # renamed HISTORICAL_SIDE_GLOBAL_BARS


# ── O/U 2.5 serving (src/betting.evaluate_value) ──────────────────────────────────────────────
def test_unapproved_league_is_avoid_with_no_league_bar_and_no_stake(bars_env):
    bars_env(std={"Own League": OWN}, applied=["ou25|Own League"])
    out = B.evaluate_value(_board(["Own League", "No Bar League"], 0.20))
    own, nob = out.iloc[0], out.iloc[1]
    assert own["signal_tier"] == "SNIPER" and own["threshold_source"] == config.THRESHOLD_SOURCE_LEAGUE
    assert nob["signal_tier"] == "AVOID" and nob["tier_reason"] == ST.NO_LEAGUE_BAR
    assert nob["bet"] == "AVOID" and nob["bet_stake"] == 0 and nob["stake_over"] == 0
    assert set(out["threshold_source"]) <= {config.THRESHOLD_SOURCE_LEAGUE, ST.NO_LEAGUE_BAR}


def test_ou25_uses_the_leagues_own_three_bars(bars_env):
    bars_env(std={"L": OWN}, applied=["ou25|L"])
    tiers = [B.evaluate_value(_board(["L"], e)).iloc[0]["signal_tier"] for e in (0.13, 0.10, 0.06, 0.04)]
    assert tiers == ["SNIPER", "MARKSMAN", "VALUABLE", "AVOID"]


def test_league_valuable_bar_replaces_the_global_floor(bars_env):
    """A league whose own VALUABLE bar is below the old global 0.04 tips at its own bar."""
    bars_env(std={"L": {**OWN, "valuable_th": 0.02}}, applied=["ou25|L"])
    r = B.evaluate_value(_board(["L"], 0.03)).iloc[0]
    assert r["signal_tier"] == "VALUABLE" and r["bet"] == "OVER"


def test_global_env_thresholds_no_longer_move_a_tier(bars_env, monkeypatch):
    bars_env(std={"L": OWN}, applied=["ou25|L"])
    a = B.evaluate_value(_board(["L", "M"], 0.10))[["signal_tier", "tier_reason"]].values.tolist()
    for k, v in {"VALUABLE_THRESHOLD": 0.0, "VALUE_THRESHOLD": 0.0, "MARKSMAN_THRESHOLD": 0.0,
                 "SNIPER_THRESHOLD_OVER": 0.0, "SNIPER_THRESHOLD": 0.0}.items():
        monkeypatch.setattr(config, k, v)
    b = B.evaluate_value(_board(["L", "M"], 0.10))[["signal_tier", "tier_reason"]].values.tolist()
    assert a == b == [["MARKSMAN", ""], ["AVOID", ST.NO_LEAGUE_BAR]]


def test_stake_evidence_follows_own_bars(bars_env):
    bars_env(std={"L": OWN}, applied=["ou25|L"])
    assert B.league_threshold_evidence("L", "ou25") is True
    assert B.league_threshold_evidence("M", "ou25") is False


# ── side markets (shadow mirror shares the rule) ──────────────────────────────────────────────
def test_side_market_bars_and_shadow_never_global(bars_env):
    side = {"btts": {"L": OWN}}
    bars_env(side=side, applied=["btts|L"])
    assert SE.thresholds(side, "btts", "L") == (0.12, 0.09, 0.05)
    assert SE.thresholds(side, "btts", "M") is None and SE.tier(0.5, None) == "AVOID"
    assert ST.side_bars({}, "over15", "M") is None


# ── data truth rule 6 ─────────────────────────────────────────────────────────────────────────
def test_rule6_is_green_on_this_tree_and_gated():
    c = DT.no_global_bars()
    assert c["status"] == DT.GREEN, c["detail"]
    assert "Data truth 6" in DT.GATED


@pytest.mark.parametrize("line", ["x = config.VALUE_THRESHOLD", "b = config.SIDE_GLOBAL_BARS['sniper']",
                                  "t = getattr(config, 'MARKSMAN_THRESHOLD', 0.1)", "g = global_bars()",
                                  "s = app_config.THRESHOLD_SOURCE_GLOBAL",
                                  "c = config.LEAGUE_EDGE_CEILING.get(lg)"])
def test_rule6_catches_a_global_bar_read(tmp_path, line):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "betting.py").write_text(
        f'"""doc mentions config.VALUE_THRESHOLD harmlessly"""\n# config.EDGE_CEILING in a comment\n{line}\n',
        encoding="utf-8")
    hits = DT.scan_global_bars(["src/betting.py"], root=tmp_path)
    assert len(hits) == 1 and hits[0]["line"] == 3


def test_rule6_red_blocks_the_gate():
    red = [DT._c("Data truth 6: no global bars in serving", DT.RED, "1", "x")]
    assert any(f.startswith("Data truth 6") for f in DT.gate(checks=red))
