"""Side-market tips fall back to API-Football like O/U 2.5 tips do (2026-10-25).

Without the fallback a BTTS/O1.5/O3.5 tip whose fixture the primary score source missed stayed
pending forever, while the O/U tip on the same match settled: 38 rows since 2026-10-10.
"""
import pandas as pd

import update_results as U

COLS = ["source", "generated_at", "match_date", "league", "home_team", "away_team", "market", "odds",
        "signal_tier", "result", "pnl", "closing_odds", "clv_pct"]


def _ledger(tmp_path, rows):
    p = tmp_path / "side_bets_ledger.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return p


def _run(monkeypatch, tmp_path, primary, af):
    p = _ledger(tmp_path, [
        ["live", "2026-10-24 06:00", "2026-10-24", "League Two", "Gillingham", "Grimsby Town", "btts", "1.80",
         "MARKSMAN", "", "", "", ""],
        ["live", "2026-10-24 06:00", "2026-10-24", "League Two", "Gillingham", "Grimsby Town", "over35", "3.10",
         "SNIPER", "", "", "", ""]])
    calls = []
    monkeypatch.setattr(U, "SIDE_LEDGER_FILE", p)
    monkeypatch.setattr(U, "_FD_SOURCES", {"League Two": "E3"})
    monkeypatch.setattr(U, "fetch_scores_fd", lambda league: primary)
    monkeypatch.setattr(U, "fetch_scores_apifootball", lambda league: calls.append(league) or af)
    monkeypatch.setattr(U, "_closing_for_market", lambda *a, **k: float("nan"))
    U.update_side_market_results(days=3)
    return pd.read_csv(p, dtype=str), calls


GAME = {"home_team": "Gillingham", "away_team": "Grimsby Town", "home_score": 2, "away_score": 1,
        "date_str": "2026-10-24"}


def test_primary_miss_settles_from_api_football(monkeypatch, tmp_path):
    out, calls = _run(monkeypatch, tmp_path, primary=[], af=[GAME])
    assert calls                                          # fallback was used
    assert out["result"].tolist() == ["WIN", "LOSS"]     # 2-1: BTTS yes, under 3.5
    assert float(out["pnl"][0]) == 0.8 and float(out["pnl"][1]) == -1.0


def test_primary_hit_makes_no_fallback_call(monkeypatch, tmp_path):
    out, calls = _run(monkeypatch, tmp_path, primary=[GAME], af=[])
    assert not calls
    assert out["result"].tolist() == ["WIN", "LOSS"]


def test_found_nowhere_stays_pending(monkeypatch, tmp_path):
    out, _ = _run(monkeypatch, tmp_path, primary=[], af=[])
    assert out["result"].isna().all()
