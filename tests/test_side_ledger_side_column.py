"""The side ledger records which side was tipped (Data Truth audit findings 26, 28).

    python -m pytest tests/test_side_ledger_side_column.py -q
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import ledger as LG

ROOT = Path(__file__).resolve().parents[1]


def test_new_side_rows_carry_their_side(tmp_path, monkeypatch):
    monkeypatch.setattr(LG, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    tips = pd.DataFrame([
        {"date": "2026-11-05", "league": "Brazil Serie A", "home_team": "A", "away_team": "B",
         "market": "btts", "side": "YES", "signal_tier": "VALUABLE", "market_odds": 1.9,
         "edge": 0.05, "ev": 0.04, "model_prob": 0.56, "model_type": "new_format"},
        {"date": "2026-11-05", "league": "Brazil Serie A", "home_team": "C", "away_team": "D",
         "market": "over15", "signal_tier": "VALUABLE", "market_odds": 1.3,
         "edge": 0.05, "ev": 0.04, "model_prob": 0.82, "model_type": "new_format"}])
    LG.append_side_market_tips(tips)
    out = pd.read_csv(LG.SIDE_LEDGER_FILE, dtype=str)
    assert out["side"].tolist() == ["YES", "OVER"]


def test_committed_side_ledger_has_no_empty_side():
    p = ROOT / "output" / "side_bets_ledger.csv"
    if p.exists():
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        assert (d["side"].str.strip() != "").all()


def test_missing_price_edge_ev_and_prob_stay_blank(tmp_path, monkeypatch):
    # They were written as 0 / 0.0, which reads as a real zero edge, EV and probability.
    monkeypatch.setattr(LG, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    LG.append_side_market_tips(pd.DataFrame([
        {"date": "2026-11-05", "league": "Brazil Serie A", "home_team": "A", "away_team": "B",
         "market": "btts", "side": "YES", "signal_tier": "VALUABLE", "model_type": "new_format"}]))
    row = pd.read_csv(LG.SIDE_LEDGER_FILE, dtype=str).iloc[0]
    assert all(pd.isna(row[c]) for c in ("odds", "edge_pct", "ev_pct", "model_prob"))


def test_unknown_ou_model_key_is_not_called_standard(monkeypatch):
    from src import model_registry as reg

    def boom(*a, **k):
        raise KeyError("no such model")
    monkeypatch.setattr(reg, "ou_model_for_row", boom)
    assert LG._ou_key("new_format") is None
    assert LG._version_fields({}, None)["model_version"] == "UNKNOWN_LEGACY"
