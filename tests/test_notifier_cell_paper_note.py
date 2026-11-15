"""Go-live rehearsal defect 2 (2026-11-15): the PAPER line of a Telegram tip read the TRACK state, so with
per-cell staking on a tip in a cell without its own staking entry (PAPER) read as staked whenever its
track was ACTIVE; and notify_side_bets never drew a PAPER line at all. Both now use
src/bankroll_policy.stake_entry (the per-cell verdict the stake plan uses)."""
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


def test_note_is_per_cell_when_cell_staking_is_on(monkeypatch):
    monkeypatch.setattr(BP, "load_policy", lambda path=None: POLICY)
    assert N._paper_track_note("new_format", "BTTS", "YES", ARG) == ""
    note = N._paper_track_note("new_format", "BTTS", "YES", BRA)          # ACTIVE track, no cell entry
    assert "PAPER" in note and f"new_format|BTTS|{BRA}" in note
    assert "PAPER" in N._paper_track_note("new_format", "OU25", "OVER", BRA)


def test_note_reads_the_track_when_cell_staking_is_off(monkeypatch):
    off = dict(POLICY, cell_staking={"enabled": False})
    monkeypatch.setattr(BP, "load_policy", lambda path=None: off)
    assert N._paper_track_note("new_format", "BTTS", "YES", BRA) == ""
    assert "PAPER" in N._paper_track_note("standard", "BTTS", "YES", BRA)


def test_side_bets_message_carries_the_cell_paper_line(monkeypatch, tmp_path):
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
    assert "PAPER" not in arg
    assert "PAPER" in bra and f"new_format|BTTS|{BRA}" in bra
