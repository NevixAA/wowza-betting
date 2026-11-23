"""Half-time feature parity (owner P0 deep-repair item 4, 2026-11-22: "Repair half-time feature parity").

home/away_has_ht_history = 1.0 when the team's form window (its last 5 matches in the same league,
strictly before the match) holds at least one real half-time score, else 0.0. Checked three ways on
hostile synthetic histories:

  expected  an independent re-implementation below (plain loop, no src.feature_engineering code)
  training  src.feature_engineering.build_features (the frame the Phase 2 harness fits on)
  serving   build_upcoming_features + src.ht_features.attach_ht_features (what src/predict.py runs)

and the flag must equal "the HT form value is present" on both sides (an imputed HT form value next
to has_ht_history=1 would be a missingness-marker bug). The served Standard HT payloads must not read
the degenerate home_has_ht_history (Data truth 3; reports/P0_HT_FEATURE_PARITY.md).

    python -m pytest tests/test_ht_feature_parity.py -q
"""
from __future__ import annotations

import pickle
from collections import deque

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering as FE

L1, L2 = "England Premier League", "England Championship"
N = 5


def _row(date, league, home, away, hht=1.0, aht=0.0, hg=2, ag=1):
    return {"date": pd.Timestamp(date), "league": league, "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag, "total_goals": hg + ag, "over25": float(hg + ag > 2.5),
            "home_shots": 10.0, "away_shots": 8.0, "home_sot": 4.0, "away_sot": 3.0,
            "home_corners": 5.0, "away_corners": 4.0, "home_fouls": 11.0, "away_fouls": 12.0,
            "ht_home_goals": hht, "ht_away_goals": aht, "odds_over25": 1.9, "odds_under25": 1.95}


def expected_flag(hist: pd.DataFrame, league: str, team: str, date) -> float:
    """Independent: the team's last N league matches dated before `date`; any with both HT goals."""
    h = hist[(hist["league"] == league) & (pd.to_datetime(hist["date"]) < pd.Timestamp(date))
             & ((hist["home_team"] == team) | (hist["away_team"] == team))]
    w = deque(maxlen=N)
    for _, r in h.sort_values("date", kind="mergesort").iterrows():
        w.append(pd.notna(r["ht_home_goals"]) and pd.notna(r["ht_away_goals"]))
    return float(any(w))


def _training(hist, nxt):
    out = FE.build_features(pd.concat([hist, pd.DataFrame([nxt])], ignore_index=True))
    hit = out[(out["home_team"] == nxt["home_team"]) & (out["away_team"] == nxt["away_team"])
              & (pd.to_datetime(out["date"]) == nxt["date"])]
    assert len(hit) == 1
    return hit.iloc[0]


def _serving(hist, nxt):
    from src.ht_features import HT_FEATURE_COLS, attach_ht_features
    up = pd.DataFrame([{k: nxt[k] for k in ("date", "league", "home_team", "away_team", "odds_over25", "odds_under25")}])
    sv = FE.build_upcoming_features(up, hist)
    ht = attach_ht_features(hist, sv)
    assert len(ht) == len(sv)
    for c in HT_FEATURE_COLS:
        if c in ht.columns:
            sv[c] = ht[c].to_numpy()
    return sv.iloc[0]


def _no_ht(rows):
    return [{**r, "ht_home_goals": np.nan, "ht_away_goals": np.nan} for r in rows]


def _games(team, league, start, k, prefix, **kw):
    d0 = pd.Timestamp(start)
    return [_row(d0 + pd.Timedelta(days=7 * i), league, *((team, f"{prefix}{i}") if i % 2 == 0 else (f"{prefix}{i}", team)), **kw)
            for i in range(k)]


CASES = {
    # name: (history rows, next fixture)
    "full_ht_history": (_games("A", L1, "2026-01-03", 8, "X") + _games("B", L1, "2026-01-04", 8, "Y"),
                        _row("2026-06-01", L1, "A", "B")),
    "no_earlier_match": (_games("B", L1, "2026-01-04", 6, "Y"), _row("2026-06-01", L1, "NEW", "B")),
    "window_all_missing_ht": (_no_ht(_games("A", L1, "2026-01-03", 7, "X")) + _games("B", L1, "2026-01-04", 6, "Y"),
                              _row("2026-06-01", L1, "A", "B")),
    # HT only in matches older than the window (the 2021/22 backtest_archive pattern): flag 0
    "ht_only_outside_window": (_games("A", L1, "2025-08-02", 4, "P") + _no_ht(_games("A", L1, "2026-01-03", 5, "X"))
                               + _games("B", L1, "2026-01-04", 6, "Y"), _row("2026-06-01", L1, "A", "B")),
    # one HT score inside the window is enough
    "one_ht_in_window": (_no_ht(_games("A", L1, "2026-01-03", 4, "X")) + _games("A", L1, "2026-03-01", 1, "Q")
                         + _games("B", L1, "2026-01-04", 6, "Y"), _row("2026-06-01", L1, "A", "B")),
    # half an HT score is not a real HT score
    "half_ht_score_only": ([{**r, "ht_away_goals": np.nan} for r in _games("A", L1, "2026-01-03", 5, "X")]
                           + _games("B", L1, "2026-01-04", 6, "Y"), _row("2026-06-01", L1, "A", "B")),
    # HT history in another league does not count (promoted club)
    "other_league_history": (_games("A", L2, "2026-01-03", 6, "X") + _no_ht(_games("A", L1, "2026-04-01", 2, "Z"))
                             + _games("B", L1, "2026-01-04", 6, "Y"), _row("2026-06-01", L1, "A", "B")),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_has_ht_history_training_serving_and_independent_agree(name):
    rows, nxt = CASES[name]
    hist = pd.DataFrame(rows)
    tr, sv = _training(hist, nxt), _serving(hist, nxt)
    for side, team in (("home", nxt["home_team"]), ("away", nxt["away_team"])):
        col = f"{side}_has_ht_history"
        exp = expected_flag(hist, nxt["league"], team, nxt["date"])
        assert float(tr[col]) == exp, f"{name}: training {col}={tr[col]} expected {exp}"
        assert float(sv[col]) == exp, f"{name}: serving {col}={sv[col]} expected {exp}"
        # the flag is the missingness marker of the HT form block, on both sides
        assert float(tr[col]) == float(pd.notna(tr[f"{side}_ht_scored_last5"])), f"{name}: training marker"
        assert float(sv[col]) == float(pd.notna(sv[f"{side}_ht_scored_last5"])), f"{name}: serving marker"


def test_expected_values_of_the_hostile_cases():
    want = {"full_ht_history": 1.0, "no_earlier_match": 0.0, "window_all_missing_ht": 0.0,
            "ht_only_outside_window": 0.0, "one_ht_in_window": 1.0, "half_ht_score_only": 0.0,
            "other_league_history": 0.0}
    for name, v in want.items():
        rows, nxt = CASES[name]
        assert expected_flag(pd.DataFrame(rows), nxt["league"], nxt["home_team"], nxt["date"]) == v, name


def test_served_standard_ht_models_do_not_read_the_degenerate_flag():
    """Owner 2026-11-22: the Standard HT models drop home_has_ht_history (1.0 on every training row
    after PR #59). away_has_ht_history is not degenerate (min 0, max 1) and stays."""
    import config
    from src.integrity.data_truth import team_payload_spread
    for f in (config.HT_MODEL_FILE_05, config.HT_MODEL_FILE_15):
        with open(f, "rb") as fh:
            p = pickle.load(fh)
        assert "home_has_ht_history" not in p["feature_cols"], f.name
        assert "away_has_ht_history" in p["feature_cols"], f.name
        assert team_payload_spread(p) == {}, f"{f.name}: a served input holds one value on the training rows"
