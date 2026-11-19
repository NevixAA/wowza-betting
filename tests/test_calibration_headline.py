"""Model Performance calibration headline (dashboard_data.v9.calibration, pages/9_🎓_Model_Performance.py).

Estate scan 2026-11-02: headline 8.97pp, corrected 7.44pp. Cause: the claim was rebuilt as
edge_pct/100 + 1/odds on EVERY ledger row, but side-market edge is measured against the de-vigged
price, so side claims carried the bookmaker margin. Side rows record model_prob; that is the claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ledger():
    rows = []
    # O/U 2.5: model_prob not recorded; claim = edge/100 + 1/odds = 0.05 + 0.5 = 0.55
    for i in range(20):
        rows.append(dict(market="ou25", signal_tier="SNIPER", result="WIN" if i < 11 else "LOSS",
                         odds=2.0, edge_pct=5.0, model_prob=float("nan"), source="live"))
    # side: de-vigged edge 5pp at odds 2.0 (fair 0.48) -> recorded model_prob 0.53;
    # edge + 1/odds would claim 0.55 instead.
    for i in range(20):
        rows.append(dict(market="btts", signal_tier="SNIPER", result="WIN" if i < 10 else "LOSS",
                         odds=2.0, edge_pct=5.0, model_prob=0.53, source="live"))
    return pd.DataFrame(rows)


def test_side_market_claim_is_the_recorded_model_prob(monkeypatch):
    from dashboard_data import v9
    monkeypatch.setattr(v9, "ledger", lambda *a, **k: _ledger())
    c = v9.calibration(min_bin=5)
    assert c["available"] and c["n"] == 40
    # claimed = (20*0.55 + 20*0.53)/40 = 0.54 ; realised = 21/40 = 0.525 ; gap = 1.5pp
    assert c["claimed"] == 0.54 and c["realised"] == 0.525 and c["gap_pp"] == 1.5
    assert c["side_rows_without_model_prob"] == 0


def test_side_row_without_model_prob_is_left_out_not_rebuilt(monkeypatch):
    from dashboard_data import v9
    d = _ledger()
    d.loc[d["market"] == "btts", "model_prob"] = float("nan")
    monkeypatch.setattr(v9, "ledger", lambda *a, **k: d)
    c = v9.calibration(min_bin=5)
    assert c["n"] == 20 and c["claimed"] == 0.55 and c["side_rows_without_model_prob"] == 20
