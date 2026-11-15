"""Owner 2026-11-11 "Show calibrated edge": every future tip shows its CALIBRATED edge (the cell's own frozen
calibration, the live TOP-3 rule); a cell without a statistic shows "edge: not calibrated"; a raw edge is never
displayed as the edge."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src import calibrated_edge as CE
from src.research import top3_ranking as TR

LG = "Argentina Primera Division"


def _stats(lam=0.6, lam_se=0.1, mside="btts_yes", track="new_format", league=LG):
    return pd.DataFrame([{"track": track, "league": league, "mside": mside, "n": 0, "roi": None, "se": None,
                          "lcb": None, "eb": None, "lam": lam, "lam_se": lam_se, "lam_rows": 120}])


def _row(p_used=0.62, fair=0.50, odds=1.9, mside="btts_yes", league=LG):
    return pd.DataFrame([{"track": "new_format", "league": league, "mside": mside, "p_used": p_used, "fair": fair,
                          "odds": odds}])


def test_calibrated_edge_is_the_cells_own_cautious_calibration():
    e = CE.calibrated(_row(), _stats()).iloc[0]
    lam_lo = min(max(0.6 - TR.Z95 * 0.1, 0.0), 1.0)                     # LIVE_RULE R3: lam - 1.96 se, clipped
    assert e == pytest.approx(lam_lo * (0.62 - 0.50))
    assert e != pytest.approx(0.62 - 0.50)                               # never the raw edge


def test_no_statistic_is_not_calibrated_never_a_raw_number():
    assert math.isnan(CE.calibrated(_row(league="Other"), _stats()).iloc[0])
    assert math.isnan(CE.calibrated(_row(), None).iloc[0])
    assert CE.edge_text(float("nan")) == "edge: not calibrated"
    assert CE.edge_text(None) == "edge: not calibrated"
    assert CE.edge_text(3.24) == "+3.2 pp (calibrated)"


def test_display_edge_of_a_future_tip_is_only_the_calibrated_one():
    assert math.isnan(CE.display_edge_pct("QUALIFIED", None, 12.0))     # raw 12.0 is never shown
    assert CE.display_edge_pct("QUALIFIED", 4.5, 12.0) == 4.5
    assert CE.display_edge_pct("SNIPER", None, 12.0) == 12.0            # a historical row keeps its record


def test_telegram_tip_shows_the_calibrated_edge_not_the_raw(tmp_path, monkeypatch):
    import config
    from telegram_bot import notifier as N
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"date": "2099-01-01", "league": "Championship", "home_team": "H", "away_team": "A",
                   "signal_tier": "QUALIFIED", "bet": "OVER", "best_side": "OVER", "odds_over25": 2.0,
                   "odds_under25": 1.9, "best_edge": 0.15, "edge_calibrated": 0.031, "cell_status": "UNPROVEN",
                   "model_type": "standard", "drift_signal": "Neutral"}]).to_csv(tmp_path / "bets.csv", index=False)
    sent = []
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_load_notified", lambda *a, **k: set())
    monkeypatch.setattr(N, "_save_notified", lambda *a, **k: None)
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    monkeypatch.setattr(N, "_send", lambda token, chat, text: sent.append(text) or True)
    assert N.notify_new_snipers() == 1
    assert "+3.1 pp (calibrated)" in sent[0] and "15.0%" not in sent[0]


def test_telegram_tip_without_calibration_says_so(tmp_path, monkeypatch):
    import config
    from telegram_bot import notifier as N
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"date": "2099-01-01", "league": "Championship", "home_team": "H", "away_team": "A",
                   "signal_tier": "QUALIFIED", "bet": "OVER", "best_side": "OVER", "odds_over25": 2.0,
                   "odds_under25": 1.9, "best_edge": 0.15, "edge_calibrated": None, "cell_status": "UNPROVEN",
                   "model_type": "standard", "drift_signal": "Neutral"}]).to_csv(tmp_path / "bets.csv", index=False)
    sent = []
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_load_notified", lambda *a, **k: set())
    monkeypatch.setattr(N, "_save_notified", lambda *a, **k: None)
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    monkeypatch.setattr(N, "_send", lambda token, chat, text: sent.append(text) or True)
    assert N.notify_new_snipers() == 1
    assert "edge: not calibrated" in sent[0] and "15.0%" not in sent[0]


def test_top3_message_shows_calibrated_edge():
    from src import top3_league as T
    p = pd.DataFrame([{"league": "L", "league_week_id": "L|2026-W48", "top3_rank": 1, "home_team": "H",
                       "away_team": "A", "market": "OU25", "side": "OVER", "odds": 2.0, "slot": "TOP",
                       "cell_status": "UNPROVEN", "edge": 0.20, "edge_calibrated": 0.042,
                       "selection_mode": T.SELECTION_MODE}])
    msg = T.message(p)
    assert "+4.2 pp (calibrated)" in msg and "20.0" not in msg
