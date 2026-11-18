"""AU1 audit regression tests (READ-ONLY: they read committed artifacts and import production code,
never write). Each defect is encoded as the CORRECT behaviour marked xfail(strict=True): today the
test fails (defect present -> XFAIL, suite green); when the defect is fixed the test XPASSes and the
strict flag turns that into a failure so the marker gets removed deliberately.
Guards that already hold are plain tests.

Run: python -m pytest tests/audit/test_au1_findings.py -q
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
O = ROOT / "output"
PROD_ENV = {"LEAGUE_SNIPER_CAP": "0.12", "MARKSMAN_THRESHOLD": "0.08", "VALUABLE_THRESHOLD": "0.03"}


def _tier_under_env(env: dict, edge: float, side: str, league: str) -> str:
    """betting._base_tier evaluated in a child process whose env is a predict.yml regime."""
    e = {k: v for k, v in os.environ.items() if k not in ("LEAGUE_SNIPER_CAP", "MARKSMAN_THRESHOLD",
                                                          "VALUABLE_THRESHOLD", "EDGE_CEILING")}
    e.update(env)
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); from src import betting as B; "
            f"print(B._base_tier({edge}, {side!r}, {league!r}))")
    return subprocess.run([sys.executable, "-c", code], env=e, cwd=str(ROOT), capture_output=True,
                          text=True, check=True).stdout.strip().splitlines()[-1]


# ── AU1-B01: EDGE_CEILING / LEAGUE_EDGE_CEILING can never fire ───────────────────────────────
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="AU1-B01: ceiling check sits AFTER the SNIPER return and every "
                   "ceiling (0.15-0.19) exceeds the global SNIPER floor (0.12) -> dead code")
def test_edge_above_league_ceiling_is_downgraded():
    # config: Brazil Serie A ceiling 0.17, 'above this the model is overconfident ... downgraded to MARKSMAN'
    assert _tier_under_env(PROD_ENV, 0.30, "OVER", "Brazil Serie A") == "MARKSMAN"


# ── AU1-B02: LEAGUE_SNIPER_CAP does not cap approved best_params thresholds ──────────────────
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="AU1-B02: predict.yml documents 'SNIPER per-league -> capped at "
                   "0.12' but models/best_params_standard.json (approved League Two 0.13) bypasses the cap")
def test_league_sniper_cap_applies_to_every_league():
    assert _tier_under_env(PROD_ENV, 0.125, "OVER", "League Two") == "SNIPER"


# ── AU1-B03: serving-path rest days misaligned / mostly missing ──────────────────────────────
def _replica_serving_rest_days(hist: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim logic of src/feature_engineering.py build_upcoming_features (rest-days block):
         combined = pd.concat([hist[[...]].tail(1000), df[[...]]]).sort_values("date").reset_index(drop=True)
         combined_rd = _add_rest_days(combined); tail = combined_rd.tail(len(df)).reset_index(drop=True)
         df["home_rest_days"] = tail["home_rest_days"].values   (positional!)"""
    from src.feature_engineering import _add_rest_days
    cols = ["date", "league", "home_team", "away_team"]   # league: _add_rest_days keys (league, team)
    combined = pd.concat([hist[cols].tail(1000), df[cols]]).sort_values("date").reset_index(drop=True)
    tail = _add_rest_days(combined).tail(len(df)).reset_index(drop=True)
    out = df.copy()
    out["home_rest_days"] = tail["home_rest_days"].values
    out["away_rest_days"] = tail["away_rest_days"].values
    return out


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="AU1-B03: positional assignment of date-sorted rest days onto a "
                   "board that is ordered by league, not date")
def test_serving_rest_days_follow_their_fixture():
    # league: _add_rest_days keys (league, team) since the Data Truth 2026-11-01 rewrite
    hist = pd.DataFrame({"date": pd.to_datetime(["2026-10-20", "2026-10-25"]), "league": ["L1", "L2"],
                         "home_team": ["A", "C"], "away_team": ["B", "D"]})
    board = pd.DataFrame({"date": pd.to_datetime(["2026-11-07", "2026-11-01"]), "league": ["L1", "L2"],  # league order, not date
                          "home_team": ["A", "C"], "away_team": ["X", "Y"]})
    out = _replica_serving_rest_days(hist, board)
    assert out.loc[0, "home_rest_days"] == 18      # A: 10-20 -> 11-07
    assert out.loc[1, "home_rest_days"] == 7       # C: 10-25 -> 11-01


# AU1-B03 on the committed board: FIXED (data-truth-100, rest days keyed per fixture); xfail removed
# 2026-11-02 when it XPASSed - now a plain guard.
def test_committed_board_rest_days_physically_possible():
    p = pd.read_csv(O / "predictions.csv", low_memory=False)
    p["d"] = pd.to_datetime(p["date"])
    long = pd.concat([p[["d", "home_team", "home_rest_days"]].set_axis(["d", "team", "rest"], axis=1),
                      p[["d", "away_team", "away_rest_days"]].set_axis(["d", "team", "rest"], axis=1)])
    long = long.sort_values(["team", "d"])
    long["gap"] = (long["d"] - long.groupby("team")["d"].shift()).dt.days
    later = long[long["gap"].notna()]
    assert int((later["rest"] > later["gap"]).sum()) == 0


# ── AU1-B04: committed side-market backtests are stale against the current loader ────────────
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="AU1-B04: the 2026-10-06 side backtest flags 100% of 2024-11..2026-10 "
                   "standard rows synthetic, but the committed history + current loader join price ~72% of them")
def test_side_backtest_real_price_coverage_matches_loader(monkeypatch):
    import src.data_loader as DL
    # never write join health from an audit; monkeypatch so the stub cannot leak into other tests
    monkeypatch.setattr(DL._IJ, "audit_league_free_lookup", lambda *a, **k: None)
    d = pd.read_csv(O / "backtest_results_btts.csv", low_memory=False,
                    usecols=["date", "league", "home_team", "away_team", "price_synthetic"])
    art_real = int((~d["price_synthetic"].astype(bool)).sum())
    x = d[["date", "league", "home_team", "away_team"]].copy()
    for c in ("odds_btts", "odds_over15", "odds_over35"):
        x[c] = np.nan
    x["date"] = pd.to_datetime(x["date"])
    y = DL._enrich_with_standard_sidemarket_odds(x.reset_index(drop=True))
    assert abs(int(y["odds_btts"].notna().sum()) - art_real) <= 0.05 * art_real


# ── guards that hold today ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("t", ["btts", "over15", "over35"])
def test_synthetic_side_prices_never_bet(t):
    d = pd.read_csv(O / f"backtest_results_{t}.csv", low_memory=False,
                    usecols=["price_synthetic", "bet", "pnl", "signal_tier"])
    syn = d["price_synthetic"].astype(bool)
    assert not (syn & d["bet"].eq("BET")).any()
    assert not (syn & pd.to_numeric(d["pnl"], errors="coerce").fillna(0).ne(0)).any()
    assert set(d.loc[syn, "signal_tier"]) <= {"NO_PRICE"}


def test_side_ledger_has_no_missing_or_nonpositive_prices():
    s = pd.read_csv(O / "side_bets_ledger.csv", low_memory=False)
    odds = pd.to_numeric(s["odds"], errors="coerce")
    assert odds.notna().all() and (odds > 1.0).all()


# ── AU1-B05: legacy serving payloads were batch-dependent (known P0 bug 1, quantified) ───────
# RESOLVED 2026-11-09 by the clean-data switch: the New-Format champion is the clean-retrain recipe
# with a persisted train-only imputer, so a fixture's probability no longer depends on the other
# fixtures scored with it. The test now guards that (it was a strict xfail recording the bug).
def test_newformat_probability_independent_of_batch():
    import config
    from src.model import predict_proba
    p = pd.read_csv(O / "predictions.csv", low_memory=False)
    nf = p[p["model_type"].eq("new_format")]
    pl = pickle.load(open(config.MODEL_FILE_NEWFORMAT, "rb"))
    nan_rows = nf[nf[[c for c in pl["feature_cols"] if c in nf]].isna().any(axis=1)]
    assert len(nan_rows) > 0
    batch = predict_proba(nf, payload=pl).loc[nan_rows.index]
    solo = pd.Series([predict_proba(nf.loc[[i]], payload=pl).iloc[0] for i in nan_rows.index], index=nan_rows.index)
    assert float((batch - solo).abs().max()) < 1e-6


# ── AU1-B06: feature health monitors the payload that no longer serves Standard O/U 2.5 ──────
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="AU1-B06: record_feature_health is handed the LEGACY standard "
                   "payload's 42 columns while v9_repaired (41 columns, own feature build) serves")
def test_feature_health_standard_matches_served_payload():
    import config
    fh = json.loads((O / "feature_health.json").read_text(encoding="utf-8"))
    rep = pickle.load(open(config.MODEL_FILE_STANDARD_REPAIRED, "rb"))
    assert fh["models"]["standard"]["n_features"] == len(rep["feature_cols"])
