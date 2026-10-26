"""Drift tier adjustment REMOVED (owner decision 2026-10-26).

Every tip keeps the tier its edge earns; drift_signal stays as information; the tier the removed rule
WOULD have produced is kept as drift_shadow_tier and logged once per tip at first sight
(output/drift_shadow.csv); the switch is a config.py constant that no environment variable can flip.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import config
from src import drift as D
from src.betting import evaluate_value

ROOT = Path(__file__).resolve().parents[1]
LG = "Testland League"                       # not in any table -> global thresholds
DAY = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
SIGNALS = ["Confirmed", "Conflicted", "Neutral", "New"]


def _row(p_over, home="A", away="B", **kw):
    r = {"league": LG, "date": DAY, "kickoff_utc": f"{DAY} 15:00:00", "home_team": home, "away_team": away,
         "p_over25": p_over, "odds_over25": 2.0, "odds_under25": 1.9,
         "home_scored_last5": 7, "away_scored_last5": 6, "home_conceded_last5": 5, "away_conceded_last5": 6}
    r.update(kw)
    return r


# base tiers under predict.yml's deployed bars (VALUABLE 0.03, MARKSMAN 0.08; global SNIPER 0.12):
# edge 0.05 VALUABLE, 0.16 SNIPER, 0.11 MARKSMAN
BOARD = [(0.55, "VALUABLE"), (0.66, "SNIPER"), (0.61, "MARKSMAN")]


@pytest.fixture(autouse=True)
def _deployed_bars(monkeypatch):
    monkeypatch.setattr(config, "VALUABLE_THRESHOLD", 0.03)
    monkeypatch.setattr(config, "MARKSMAN_THRESHOLD", 0.08)


def _base():
    return evaluate_value(pd.DataFrame([_row(p, home=f"H{i}") for i, (p, _) in enumerate(BOARD)]))


def test_preconditions_cover_every_tier_the_old_rule_touched():
    assert list(_base()["signal_tier"]) == [t for _, t in BOARD]


def test_flag_is_off_by_owner_decision():
    assert config.DRIFT_TIER_ADJUSTMENT is False


@pytest.mark.parametrize("signal", SIGNALS)
def test_tier_is_identical_to_base_for_every_drift_signal(signal):
    base = _base()
    df = evaluate_value(pd.DataFrame([_row(p, home=f"H{i}", drift_signal=signal)
                                      for i, (p, _) in enumerate(BOARD)]))
    assert list(df["signal_tier"]) == list(base["signal_tier"])
    for c in ("bet", "best_side", "best_edge", "bet_stake", "edge_over", "edge_under"):
        assert list(df[c]) == list(base[c]), c
    assert list(df["drift_signal"]) == [signal] * len(BOARD)          # still recorded


def test_shadow_tier_is_what_the_removed_rule_would_have_produced():
    exp = {"Confirmed": ["MARKSMAN", "SNIPER", "SNIPER"],     # VALUABLE->MARKSMAN, MARKSMAN->SNIPER (edge>=0.10)
           "Conflicted": ["VALUABLE", "MARKSMAN", "MARKSMAN"],  # SNIPER->MARKSMAN
           "Neutral": ["VALUABLE", "SNIPER", "MARKSMAN"], "New": ["VALUABLE", "SNIPER", "MARKSMAN"]}
    for sig, want in exp.items():
        df = evaluate_value(pd.DataFrame([_row(p, home=f"H{i}", drift_signal=sig)
                                          for i, (p, _) in enumerate(BOARD)]))
        assert list(df["drift_shadow_tier"]) == want, sig
        assert list(df["signal_tier"]) == [t for _, t in BOARD], sig


def test_avoid_stays_avoid_in_the_shadow():
    df = evaluate_value(pd.DataFrame([_row(0.50, drift_signal="Confirmed")]))
    assert df.iloc[0]["signal_tier"] == "AVOID" and df.iloc[0]["drift_shadow_tier"] == "AVOID"


def test_only_a_code_change_restores_the_old_rule(monkeypatch):
    monkeypatch.setattr(config, "DRIFT_TIER_ADJUSTMENT", True)
    df = evaluate_value(pd.DataFrame([_row(p, home=f"H{i}", drift_signal="Confirmed")
                                      for i, (p, _) in enumerate(BOARD)]))
    assert list(df["signal_tier"]) == ["MARKSMAN", "SNIPER", "SNIPER"]


@pytest.mark.parametrize("val", ["1", "true", "True", "yes"])
def test_env_var_cannot_flip_it(monkeypatch, caplog, val):
    monkeypatch.setenv("DRIFT_TIER_ADJUSTMENT", val)
    with caplog.at_level(logging.WARNING, logger="src.betting"):
        df = evaluate_value(pd.DataFrame([_row(0.55, drift_signal="Confirmed")]))
    assert df.iloc[0]["signal_tier"] == "VALUABLE"
    assert any("IGNORED" in r.getMessage() for r in caplog.records)


def test_config_is_a_constant_not_an_env_read():
    src = (ROOT / "config.py").read_text(encoding="utf-8")
    assert 'getenv("DRIFT_TIER_ADJUSTMENT"' not in src and "getenv('DRIFT_TIER_ADJUSTMENT'" not in src
    env = dict(os.environ, DRIFT_TIER_ADJUSTMENT="1", DRIFT_UPGRADE_EDGE="0.0")
    out = subprocess.run([sys.executable, "-c", "import config; print(config.DRIFT_TIER_ADJUSTMENT, "
                          "config.DRIFT_UPGRADE_EDGE)"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.stdout.strip().splitlines()[-1] == "False 0.1"


def test_no_workflow_sets_the_flag():
    import re
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        assert not re.search(r"\bDRIFT_TIER_ADJUSTMENT\s*[:=]", wf.read_text(encoding="utf-8")), wf.name


# ── first-sight shadow log ───────────────────────────────────────────────────────────────────────
def test_shadow_is_logged_on_first_sight_and_never_revised(tmp_path):
    p = tmp_path / "drift_shadow.csv"
    preds = evaluate_value(pd.DataFrame([_row(0.55, drift_signal="Confirmed"),
                                         _row(0.50, home="Z", drift_signal="Confirmed")]))  # 2nd is AVOID
    assert D.log_shadow(preds, p) == 1
    got = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert list(got.columns) == D.SHADOW_COLS
    r = got.iloc[0]
    assert (r["signal_tier"], r["drift_shadow_tier"], r["drift_signal"], r["drift_would_change"],
            r["side"], r["drift_tier_adjustment_enabled"]) == \
           ("VALUABLE", "MARKSMAN", "Confirmed", "True", "OVER", "False")
    before = p.read_bytes()
    later = evaluate_value(pd.DataFrame([_row(0.55, drift_signal="Conflicted")]))
    assert D.log_shadow(later, p) == 0 and p.read_bytes() == before        # earliest row wins


def test_shadow_refuses_post_kickoff_rows_and_never_raises(tmp_path):
    p = tmp_path / "drift_shadow.csv"
    past = evaluate_value(pd.DataFrame([_row(0.55, drift_signal="Neutral", kickoff_utc="2020-01-01 15:00:00")]))
    assert D.log_shadow(past, p) == 0
    assert D.log_shadow(pd.DataFrame({"x": [1]}), p) == 0
    assert D.log_shadow(None, p) == 0


def test_mode_predict_writes_the_shadow_log(tmp_path, monkeypatch):
    import pipeline
    import src.predict as P
    import src.stake_plan as SP
    from src import edge_engine_shadow, shadow_experiments
    from src import ledger as L
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    board = pd.DataFrame([_row(p, home=f"H{i}", drift_signal="Confirmed", model_type="new_format")
                          for i, (p, _) in enumerate(BOARD)])
    monkeypatch.setattr(P, "predict_upcoming", lambda *a, **k: (evaluate_value(board), []))
    monkeypatch.setattr(pipeline, "load_models", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_log_shadow", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_log_h2h", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "append_ht_tips", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "append_ht_observations", lambda *a, **k: None)
    monkeypatch.setattr(shadow_experiments, "score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub")))
    monkeypatch.setattr(edge_engine_shadow, "run", lambda *a, **k: None)
    monkeypatch.setattr(SP, "PLAN", tmp_path / "stake_plan.csv")
    monkeypatch.setattr(SP, "AUDIT", tmp_path / "stake_audit.csv")
    monkeypatch.setattr(SP, "load_league_ratings", lambda *a, **k: {})
    pipeline.mode_predict(historical=pd.DataFrame({"league": [LG]}))
    bets = pd.read_csv(tmp_path / "bets.csv")
    assert sorted(bets["signal_tier"]) == sorted(t for _, t in BOARD)          # served = base tiers
    led = pd.read_csv(tmp_path / "bets_ledger.csv")
    assert sorted(led["signal_tier"]) == sorted(t for _, t in BOARD)
    sh = pd.read_csv(tmp_path / "drift_shadow.csv")
    assert len(sh) == 3 and int(sh["drift_would_change"].sum()) == 2


# ── Telegram ─────────────────────────────────────────────────────────────────────────────────────
def test_tip_message_shows_drift_as_information_not_as_an_upgrade(tmp_path, monkeypatch):
    from telegram_bot import notifier as N
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", tmp_path)
    evaluate_value(pd.DataFrame([_row(0.55, drift_signal="Confirmed", model_type="standard")])) \
        .assign(best_side="OVER").to_csv(tmp_path / "bets.csv", index=False)
    sent = []
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_load_notified", lambda *a, **k: set())
    monkeypatch.setattr(N, "_save_notified", lambda *a, **k: None)
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    monkeypatch.setattr(N, "_send", lambda token, chat, text: sent.append(text) or True)
    assert N.notify_new_snipers() == 1
    msg = sent[0]
    from src.tier_labels import tier_label                                  # display words, 2026-10-26
    assert f"{tier_label('VALUABLE').upper()} TIP" in msg                  # not upgraded by Confirmed
    assert tier_label("MARKSMAN").upper() not in msg.upper()
    assert "Drift: Confirmed" in msg and "(info only)" in msg
    assert "upgrad" not in msg.lower()
