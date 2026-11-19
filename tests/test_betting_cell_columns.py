"""Go-live rehearsal defect 1 (2026-11-15): generate_bets dropped the cell qualification columns, so
bets.csv / bets_ledger O/U 2.5 rows had blank cell_status / qualifying_bar / edge_calibrated_pct and the
Telegram O/U tips read "UNPROVEN" and "edge: not calibrated" even for a PROMISING cell."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import ledger as LG  # noqa: E402
from src.betting import generate_bets  # noqa: E402
from src.tier_labels import QUALIFIED  # noqa: E402


def _preds() -> pd.DataFrame:
    return pd.DataFrame([{
        "date": "2026-11-22", "kickoff_utc": "2026-11-22T15:00:00Z", "league": "Championship",
        "home_team": "Leeds", "away_team": "Hull", "best_side": "OVER",
        "odds_over25": 1.95, "odds_under25": 1.90, "edge_over": 0.06, "edge_under": -0.04,
        "best_edge": 0.06, "bet": "OVER", "bet_stake": 0.0, "signal_tier": QUALIFIED,
        "model_type": "standard", "fixture_id": 1234567, "oddsapi_event_id": "abc123",
        "cell_status": "PROMISING", "qualifying_bar": 0.04, "cell_p_used": 0.55, "edge_calibrated": 0.031,
    }, {
        "date": "2026-11-22", "kickoff_utc": "2026-11-22T15:00:00Z", "league": "Championship",
        "home_team": "Stoke", "away_team": "Derby", "best_side": "UNDER",
        "odds_over25": 2.10, "odds_under25": 1.75, "edge_over": -0.05, "edge_under": 0.01,
        "best_edge": 0.01, "bet": "AVOID", "bet_stake": 0.0, "signal_tier": "AVOID",
        "model_type": "standard", "fixture_id": 7654321, "oddsapi_event_id": "zzz",
        "cell_status": "", "qualifying_bar": np.nan, "cell_p_used": np.nan, "edge_calibrated": np.nan,
    }])


def test_generate_bets_carries_identity_and_cell_columns():
    bets = generate_bets(_preds())
    assert len(bets) == 1
    r = bets.iloc[0]
    for c in ("fixture_id", "oddsapi_event_id", "cell_status", "qualifying_bar",
              "cell_p_used", "edge_calibrated"):
        assert c in bets.columns, c
    assert r["cell_status"] == "PROMISING" and r["qualifying_bar"] == 0.04
    assert r["cell_p_used"] == 0.55 and r["edge_calibrated"] == 0.031


def test_bets_ledger_row_is_qualified_with_calibrated_edge(tmp_path, monkeypatch):
    monkeypatch.setattr(LG, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(config, "ODDS_HISTORY_JSON", tmp_path / "no_history.json")
    LG.append_tips(generate_bets(_preds()))
    led = pd.read_csv(tmp_path / "bets_ledger.csv", dtype=str)
    assert len(led) == 1
    r = led.iloc[0]
    assert r["cell_status"] == "PROMISING"
    assert float(r["qualifying_bar"]) == 0.04
    assert float(r["edge_calibrated_pct"]) == 3.1
    assert float(r["cell_p_used"]) == 0.55
    assert str(r["fixture_id"]) == "1234567"
