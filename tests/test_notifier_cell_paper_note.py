"""Telegram stake labels on ORDINARY tips.

History: go-live rehearsal defect 2 (2026-11-15) made the PAPER line per cell. Owner 2026-11-17 ("Show stake on
TOP-3"): an ordinary prediction carries NO stake label at all - only TOP-3 picks show "0.5% stake"
(src/top3_league.message) - and the word "paper" never reaches a user. Whatever the policy says, the note is ''."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bankroll_policy as BP  # noqa: E402
from telegram_bot import notifier as N  # noqa: E402

ARG = "Argentina Primera Division"
BRA = "Brazil Serie A"

POLICY = {
    "tracks": {"new_format|BTTS": {"state": "ACTIVE", "base_fraction": 0.0125},
               "new_format|OU25_OVER": {"state": "ACTIVE", "base_fraction": 0.01}},
    "cell_staking": {"enabled": True, "cells": {
        f"new_format|BTTS|{ARG}": {"state": "ACTIVE", "base_fraction": 0.0125}}},
}


def test_no_stake_label_on_an_ordinary_tip_whatever_the_policy(monkeypatch):
    for pol in (POLICY, dict(POLICY, cell_staking={"enabled": False})):
        monkeypatch.setattr(BP, "load_policy", lambda path=None, _p=pol: _p)
        for args in (("new_format", "BTTS", "YES", ARG), ("new_format", "BTTS", "YES", BRA),
                     ("new_format", "OU25", "OVER", BRA), ("standard", "BTTS", "YES", BRA)):
            assert N._paper_track_note(*args) == ""


def test_side_bets_messages_carry_no_stake_label(monkeypatch, tmp_path):
    monkeypatch.setattr(BP, "load_policy", lambda path=None: POLICY)
    day = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    base = {"date": day, "market": "btts", "side": "YES", "signal_tier": "QUALIFIED", "edge": 0.05,
            "ev": 0.05, "market_odds": 1.9, "odds": 1.9, "model_prob": 0.55, "model_type": "new_format",
            "cell_status": "PROMISING", "edge_calibrated": 0.02, "threshold_source": ""}
    pd.DataFrame([dict(base, league=ARG, home_team="Boca", away_team="River"),
                  dict(base, league=BRA, home_team="Santos", away_team="Gremio")]
                 ).to_csv(tmp_path / "side_bets.csv", index=False)
    sent: list[str] = []
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "0"})
    monkeypatch.setattr(N, "_send", lambda token, chat_id, text: sent.append(text) or True)
    monkeypatch.setattr(N, "_load_notified", lambda path=None: set())
    monkeypatch.setattr(N, "_save_notified", lambda keys, path=None: None)
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    assert N.notify_side_bets() == 2
    arg = next(m for m in sent if "Boca" in m)
    bra = next(m for m in sent if "Santos" in m)
    for m in (arg, bra):
        assert "paper" not in m.lower() and "not staked" not in m and "stake" not in m.lower()
