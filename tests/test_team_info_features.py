"""The team-information block (src/team_info_features.py), its harness wiring (config/phase2_info.json,
protocol variants, bet-time market candidates) and the leak rules, on synthetic stores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import team_info_features as TI
from src.retrain_harness import bet_prices as BP
from src.retrain_harness import candidates as C
from src.retrain_harness import protocol as P

ROOT = Path(__file__).resolve().parents[1]
UTC = "UTC"


def _store(n_weeks: int = 20, seed: int = 0, injuries_after_kickoff: bool = True) -> TI.InfoStore:
    """Two clubs (ids 10, 20) of league 'L' (league_id 99, season '2025') playing each other weekly
    plus each a club (30 / 40) in between; 14-man squads, random XIs, random injury lists."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-04 15:00", tz=UTC)
    fx, lu, inj, xg, goals = [], [], [], [], []
    fid = 1000
    for w in range(n_weeks):
        for (h, a, off) in ((10, 30, 0), (40, 20, 0), (20, 10, 3)):
            k = start + pd.Timedelta(days=7 * w + off)
            fid += 1
            fx.append((fid, "L", 99, "2025", k, h, a))
            xg.append((fid, float(rng.uniform(0.3, 2.5)), float(rng.uniform(0.3, 2.5))))
            for t in (h, a):
                squad = [t * 100 + i for i in range(14)]
                xi = sorted(rng.choice(squad, 11, replace=False).tolist())
                for i, p in enumerate(xi):
                    lu.append((fid, t, p, "G" if p % 100 == 0 else ("D" if p % 100 < 5 else "M")))
                    goals.append((fid, t, p, float(rng.integers(0, 2))))
                for p in rng.choice(squad, 2, replace=False).tolist():
                    inj.append((fid, t, p))
    s = TI.InfoStore(
        fixtures=pd.DataFrame(fx, columns=["fixture_id", "league", "league_id", "season", "kickoff", "home_id", "away_id"]),
        lineups=pd.DataFrame(lu, columns=["fixture_id", "team_id", "player_id", "pos"]),
        injuries=pd.DataFrame(inj, columns=["fixture_id", "team_id", "player_id"]),
        injury_cover={(99, "2025")},
        xg=pd.DataFrame(xg, columns=["fixture_id", "hxg", "axg"]),
        goals=pd.DataFrame(goals, columns=["fixture_id", "team_id", "player_id", "goals"]),
        coaches=pd.DataFrame({"team_id": [10, 20], "start": TI.coach_known_from(pd.Series(pd.to_datetime(["2024-07-01", "2025-02-01"], utc=True))),
                              "end": pd.to_datetime([None, None], utc=True)}),
        transfers=pd.DataFrame({"team_in_id": [10, 10, 20], "date": pd.to_datetime(["2025-01-10", "2025-03-01", "2025-02-15"], utc=True)}),
        team_ids={("L", "Ten"): 10, ("L", "Twenty"): 20},
        transfer_cover={10, 20})
    return s


def _rows(s: TI.InfoStore) -> pd.DataFrame:
    f = s.fixtures[(s.fixtures["home_id"] == 20) & (s.fixtures["away_id"] == 10)]
    return pd.DataFrame({"date": f["kickoff"].dt.tz_convert(None).dt.normalize().to_numpy(), "league": "L",
                         "home_team": "Twenty", "away_team": "Ten"})


def test_block_columns_and_nan_for_unknown_club():
    s = _store()
    r = pd.DataFrame({"date": [pd.Timestamp("2025-03-10")], "league": ["L"], "home_team": ["Nobody"],
                      "away_team": ["Ten"]})
    out = TI.info_features(r, s)
    assert list(out.columns) == TI.INFO_FEATURE_COLS
    assert out.filter(like="home_").isna().all(axis=None)           # unmapped club: every value NaN
    assert out["away_info_xi_continuity"].notna().all()


def test_truncation_invariance_is_the_leak_test():
    """A store holding only what had kicked off before each row's tip time gives the same numbers."""
    s = _store(seed=3)
    rows = _rows(s)
    kick = TI.kickoff_of(rows, s)
    full = TI.info_features(rows, s, kickoffs=kick)
    assert full.notna().any().all()                                  # every feature exercised
    for i in range(len(rows)):
        T = pd.Timestamp(kick[i] - np.timedelta64(TI.EARLY_TIP_MAX_HOURS, "h"), tz=UTC)
        part = TI.info_features(rows.iloc[[i]], s.truncated(T), kickoffs=kick[[i]])
        a, b = full.iloc[[i]].to_numpy(), part.to_numpy()
        assert ((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-12)).all(), (i, full.iloc[i], part.iloc[0])


def test_own_and_recent_injury_lists_are_never_read():
    """(a) the fixture's own list, and a list of a fixture that kicked off < margin before T, never count."""
    s = _store(seed=5)
    rows = _rows(s).iloc[[10]]
    kick = TI.kickoff_of(rows, s)
    T = kick[0] - np.timedelta64(TI.EARLY_TIP_MAX_HOURS, "h")
    base = TI.info_features(rows, s, kickoffs=kick)
    late = s.fixtures[(s.fixtures["kickoff"].dt.tz_convert(None).to_numpy() > T - np.timedelta64(TI.INJURY_AMEND_MARGIN_H, "h"))]
    ext = []
    for fid in late["fixture_id"]:
        for t in (10, 20):
            ext += [(fid, t, t * 100 + i) for i in range(14)]           # everybody injured, too late to know
    s.injuries = pd.concat([s.injuries, pd.DataFrame(ext, columns=s.injuries.columns)], ignore_index=True)
    after = TI.info_features(rows, s, kickoffs=kick)
    pd.testing.assert_frame_equal(base, after)


def test_uncovered_injuries_are_nan_and_covered_empty_list_is_zero():
    s = _store(seed=7)
    rows = _rows(s).iloc[[8]]
    s.injuries = s.injuries.iloc[0:0]
    out = TI.info_features(rows, s)
    assert out["home_info_xi_absent_w"].iat[0] == 0.0 and out["home_info_xi_strength_ratio"].iat[0] == 1.0
    s.injury_cover = set()
    out = TI.info_features(rows, s)
    assert np.isnan(out["home_info_xi_absent_w"].iat[0]) and np.isnan(out["home_info_gk_absent"].iat[0])


def test_coach_month_rule_and_transfer_window():
    s = _store()
    r = pd.DataFrame({"date": [pd.Timestamp("2025-02-20")], "league": ["L"], "home_team": ["Twenty"],
                      "away_team": ["Ten"]})
    out = TI.info_features(r, s)
    # Twenty's coach started 'in February' -> known from 1 March; T is mid-February -> no coach known
    assert np.isnan(out["home_info_coach_tenure_days"].iat[0])
    assert out["away_info_coach_tenure_days"].iat[0] > 0
    assert out["away_info_transfers_in_120d"].iat[0] == 1.0          # 2025-01-10 only; 2025-03-01 is after T
    assert out["home_info_transfers_in_120d"].iat[0] == 0.0          # covered club, nothing in the window


# ── harness wiring ──────────────────────────────────────────────────────────────────────────

def test_info_config_validates_and_variant_lists():
    cfg = P.load_config(ROOT / "config" / "phase2_info.json")
    base = P.load_config()
    for k in ("holdout", "walk_forward", "comparison"):
        assert cfg[k] == base[k], f"{k} must be the Nov 9/10 protocol"
    assert set(cfg["markets"]) == {"over15", "over25", "over35", "btts"}
    cols = TI.INFO_FEATURE_COLS + [c for g in cfg["features"]["groups"].values() for c in g["columns"]]
    for track in cfg["tracks"]:
        sets = P.feature_sets(cfg, track, cols)
        assert not set(TI.INFO_FEATURE_COLS) & set(sets[""][0])
        assert set(TI.INFO_FEATURE_COLS) <= set(sets["+info"][0])
        assert set(sets[""][0]) <= set(sets["+info"][0])
        labels = [lbl for lbl, _b, _s in P.candidate_runs(cfg, track, ["champion_recipe", "market_bet", "model_market_bet"])]
        assert labels == ["champion_recipe", "champion_recipe+info", "market_bet", "model_market_bet",
                          "model_market_bet+info"]
    assert P.variant_blockers(cfg, cols) == []
    assert P.variant_blockers(cfg, [c for c in cols if c != "home_info_xg_for5"])
    assert P.feature_sets(base, "standard", cols).keys() == {""}       # the clean-retrain config is unchanged


def test_bet_prices_private_role_and_two_labels(tmp_path):
    with pytest.raises(PermissionError):
        BP.load_bet_prices(ROOT / "config" / "x.parquet")
    d = pd.DataFrame({"fixture_id": ["2025-01-01|L|A|B", "2025-01-02|L|C|D"], "market": ["over25", "over25"],
                      "median_yes": [1.9, 2.0], "median_no": [1.95, 1.8], "n_labels": [1, 2]})
    d.to_parquet(tmp_path / "p.parquet")
    pr = BP.load_bet_prices(tmp_path / "p.parquet")
    feat = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "league": "L",
                         "home_team": ["A", "C"], "away_team": ["B", "D"]})
    cfg = P.load_config(ROOT / "config" / "phase2_info.json")
    out, info = BP.attach_bet_prices(feat, pr, cfg)
    assert out["bet_over25_yes"].tolist()[0] == 1.9 and np.isnan(out["bet_over25_yes"].iat[1])
    assert info["rows_with_two_sided_bet_price"]["over25"] == 1
    d.assign(role="CLOSE").to_parquet(tmp_path / "c.parquet")
    with pytest.raises(ValueError):
        BP.load_bet_prices(tmp_path / "c.parquet")
    p = C.MarketBet({}, cfg["markets"]["over25"], [], "standard", "over25").fit({}).predict(out)
    assert 0 < p[0] < 1 and np.isnan(p[1])


def test_model_market_bet_not_fitted_without_priced_validation():
    cfg = P.load_config(ROOT / "config" / "phase2_info.json")
    c = C.build("model_market_bet", cfg, "standard", "over25", ["x"])
    val = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"] * 3), "over25": [0, 1, 0], "x": [1.0, 2.0, 3.0],
                        "bet_over25_yes": [np.nan] * 3, "bet_over25_no": [np.nan] * 3})
    with pytest.raises(C.NotFitted):
        c.fit({"fit": val, "validation": val, "train": val, "test": val})
