"""Owner reports of 2026-10-11, pinned so they cannot come back:

1. Opposite tips on the same match (MARKSMAN UNDER 2.5 then VALUABLE OVER 2.5).
2. "Yesterday's results" left out BTTS / O/U 1.5 / O/U 3.5.
3. Today's tips as a readable table image.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from src import ledger as L

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
YDAY = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


def _bet(side, tier="VALUABLE", date=TODAY, edge=0.05):
    return {"date": date, "league": "Japan J-League", "home_team": "Mito", "away_team": "Shimizu",
            "bet": side, "best_side": side, "odds_over25": 2.45, "odds_under25": 1.89,
            "best_edge": edge, "signal_tier": tier, "drift_signal": "", "model_type": "new_format"}


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    return tmp_path


# ── 1. flip guard ─────────────────────────────────────────────────────────────────────────────
def test_an_opposite_side_is_not_a_new_tip(out):
    L.append_tips(pd.DataFrame([_bet("UNDER", "MARKSMAN")]))
    L.append_tips(pd.DataFrame([_bet("OVER", "VALUABLE")]))
    led = pd.read_csv(out / "bets_ledger.csv")
    assert len(led) == 1 and led.iloc[0]["side"] == "UNDER" and led.iloc[0]["signal_tier"] == "MARKSMAN"
    assert "FLIP_SEEN OVER" in str(led.iloc[0]["notes"])


def test_the_flip_is_noted_once_not_on_every_run(out):
    L.append_tips(pd.DataFrame([_bet("UNDER", "MARKSMAN")]))
    for _ in range(3):
        L.append_tips(pd.DataFrame([_bet("OVER")]))
    note = str(pd.read_csv(out / "bets_ledger.csv").iloc[0]["notes"])
    assert note.count("FLIP_SEEN OVER") == 1


def test_same_side_upgrades_still_work(out):
    L.append_tips(pd.DataFrame([_bet("UNDER", "VALUABLE")]))
    L.append_tips(pd.DataFrame([_bet("UNDER", "SNIPER", edge=0.15)]))
    led = pd.read_csv(out / "bets_ledger.csv")
    assert len(led) == 1 and led.iloc[0]["signal_tier"] == "SNIPER"


def test_a_flipped_tip_is_dropped_from_the_send_source(out):
    L.append_tips(pd.DataFrame([_bet("UNDER", "MARKSMAN")]))
    board = pd.DataFrame([_bet("OVER"), {**_bet("OVER"), "home_team": "Other", "away_team": "Club"}])
    sent = L.drop_flipped(board)
    assert list(sent["home_team"]) == ["Other"]


def test_pipeline_filters_the_send_source():
    src = open("pipeline.py", encoding="utf-8").read()
    i = src.index("_send = config.drop_paper_leagues(bets)")
    assert "drop_flipped(_send)" in src[i:i + 400]


# ── 2 + 3. digest ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def digest(out, monkeypatch):
    from telegram_bot import notifier, tips_table
    sent, photos = [], []
    monkeypatch.setattr(notifier, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(notifier, "_load_notified", lambda *a, **k: set())
    monkeypatch.setattr(notifier, "_save_notified", lambda *a, **k: None)
    monkeypatch.setattr(notifier, "_send", lambda token, chat, text: sent.append(text) or True)
    monkeypatch.setattr(tips_table, "send_photo", lambda token, chat, png, caption="": photos.append(png) or True)
    gen = f"{config.PERFORMANCE_CUTOFF_DATE} 09:00"
    pd.DataFrame([
        {"source": "live", "generated_at": gen, "match_date": TODAY, "league": "Japan J-League",
         "home_team": "Mito", "away_team": "Shimizu", "side": "UNDER", "odds": 1.89, "edge_pct": 4.0,
         "signal_tier": "MARKSMAN", "model_type": "new_format", "result": "", "pnl": ""},
        {"source": "live", "generated_at": gen.replace("09", "10"), "match_date": TODAY, "league": "Japan J-League",
         "home_team": "Mito", "away_team": "Shimizu", "side": "OVER", "odds": 2.45, "edge_pct": 4.6,
         "signal_tier": "VALUABLE", "model_type": "new_format", "result": "", "pnl": ""},
    ]).to_csv(out / "bets_ledger.csv", index=False)
    pd.DataFrame([
        {"source": "live", "generated_at": gen, "match_date": YDAY, "league": "Argentina Primera Division",
         "home_team": "Boca", "away_team": "River", "market": "btts", "odds": 2.1, "edge_pct": 10,
         "signal_tier": "SNIPER", "model_type": "new_format", "result": "WIN", "pnl": 1.1},
        {"source": "live", "generated_at": gen, "match_date": YDAY, "league": "Argentina Primera Division",
         "home_team": "Lanus", "away_team": "Tigre", "market": "over15", "odds": 1.5, "edge_pct": 9,
         "signal_tier": "VALUABLE", "model_type": "new_format", "result": "LOSS", "pnl": -1.0},
    ]).to_csv(out / "side_bets_ledger.csv", index=False)
    notifier.notify_daily_digest()
    return sent, photos


def test_yesterday_results_include_side_markets(digest):
    sent, _ = digest
    results = [m for m in sent if "DAILY RESULTS" in m][0]
    y = results.split("Ledger")[0]
    assert "BTTS" in y and "staked 1W/0L" in y
    assert "Over 1.5" in y and "data only" in y


def test_digest_shows_one_side_per_fixture(digest):
    sent, _ = digest
    brief = [m for m in sent if "DAILY BRIEFING" in m][0]
    assert "UNDER 2.5" in brief and "OVER 2.5" not in brief


def test_digest_sends_a_table_image(digest):
    _, photos = digest
    assert photos and photos[0][:8] == b"\x89PNG\r\n\x1a\n"
