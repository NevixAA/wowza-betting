"""ops6 item 1: the 6 quarantined side_bets_ledger rows (first-half BTTS closes, listed in
output/corrections/side_clv_first_half_contaminated.csv) must leave the side CLV panels."""
from pathlib import Path

import pandas as pd

from dashboard_data import evidence_panels as ep

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "output" / "corrections" / "side_clv_first_half_contaminated.csv"


def _side_frame():
    listed = pd.read_csv(SIDECAR)
    s = listed.drop(columns=["archive_evidence", "reason", "action"]).copy()
    s["signal_tier"] = "SNIPER"
    clean = s.iloc[[0]].copy()
    clean["home_team"], clean["away_team"] = "Clean Home FC", "Clean Away FC"
    clean["closing_odds"], clean["clv_pct"] = 2.4, 4.0
    return pd.concat([s, clean], ignore_index=True), len(listed)


def test_quarantined_side_rows_carry_no_clv():
    side, n_listed = _side_frame()
    assert n_listed == 6
    d = ep.load_bets(bets=pd.DataFrame(), side_bets=side)
    assert len(d) == n_listed + 1                       # P/L rows all kept
    assert d["clv"].notna().sum() == 1                  # only the clean row keeps a CLV
    assert d["clv"].dropna().iloc[0] == 0.04


def test_side_ledger_frame_not_mutated():
    side, _ = _side_frame()
    before = side["clv_pct"].copy()
    ep.load_bets(bets=pd.DataFrame(), side_bets=side)
    pd.testing.assert_series_equal(side["clv_pct"], before)
