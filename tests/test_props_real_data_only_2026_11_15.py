"""P0 "REAL DATA ONLY" (owner 2026-11-15): props preprocessing, unsupported features, artifacts.

C  Every props payload scores the raw matrix: no batch statistic, no typed fill, the same row gives
   the same probability alone, in any batch and in any order; a missing REQUIRED_FAIL_ROW input gives
   no probability; a payload listing an unsupported (fabricated) feature is refused.
E  The served artifacts (models/model_player_*.pkl) on the production path with missing inputs: no
   invented observation reaches a model.
F  Adversarial regression scans (context-aware): silent `df[x] = 0.0` column creation, fillna with an
   invented average, a live batch median, schema / payload mismatch, unsupported features re-entering,
   point-in-time violations.
"""
from __future__ import annotations

import ast
import io
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config as C  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model import model as M  # noqa: E402
from player_model import props_contract as PC  # noqa: E402
from player_model import props_gate as PPG  # noqa: E402

SP9 = ("sp_taker_score", "sp_receiver_score", "set_piece_threat_score", "set_piece_corner_matchup",
       "defender_sot_role_index", "defender_sp_edge", "sp_threat_vs_weak_sp_defense", "sp_taker_assist_edge",
       "opp_sp_goals_conceded_pg")


def _load(path: Path) -> dict:
    return PPG._HarnessUnpickler(io.BytesIO(path.read_bytes())).load()


def _served() -> dict:
    return {m: p for m, p in C.MODEL_FILES.items() if p.exists()}


def _rows(feats, n=60, seed=3) -> pd.DataFrame:
    """Plausible synthetic inputs for every feature (real values do not matter for invariance)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.gamma(2.0, 0.5, size=(n, len(feats))), columns=list(feats))
    for c in ("pos_forward", "pos_midfielder", "pos_defender", "is_home", "return_from_injury_flag"):
        if c in df.columns:
            df[c] = rng.integers(0, 2, n).astype(float)
    if "n_prev_games" in df.columns:
        df["n_prev_games"] = rng.integers(1, 40, n).astype(float)
    return df


# ── A: the set-piece features ───────────────────────────────────────────────────────────────────

def test_set_piece_features_are_excluded_everywhere_active():
    assert set(SP9) <= C.SETPIECE_PROXIES_REMOVED <= M.UNSUPPORTED_FEATURES
    assert not set(SP9) & set(C.PLAYER_FEATURE_COLS)
    assert not set(SP9) & set(PC.load_contract()["features"])        # never IMPUTABLE: not in the schema
    assert set(SP9) <= set(FE.REMOVED_SETPIECE_FEATURES)
    df = FE._add_composites(_rows(FE._COMPOSITE_INPUTS, 5))
    assert not set(SP9) & set(df.columns)                            # never built, not even NaN
    for f in ("fantasy.py", "fantasy_features.py", "fantasy_v2.py", "predict.py"):
        src = (ROOT / "player_model" / f).read_text(encoding="utf-8")
        assert not [x for x in SP9 if x in src], f


@pytest.mark.parametrize("market", sorted(C.MODEL_FILES))
def test_a_served_payload_with_an_unsupported_feature_gets_no_probability(market):
    path = C.MODEL_FILES[market]
    if not path.exists():
        pytest.skip("absent")
    payload = _load(path)
    bad = M.unsupported_features(payload)
    out = M.predict_with_diagnostics(_rows(payload["feature_cols"], 8), payload)
    if bad:
        assert market in C.HIDDEN_PENDING_CLEAN_RETRAIN, (market, bad)   # only hidden markets list them
        assert out["p"].isna().all() and (out["imputer_version"] == M.REFUSED_UNSUPPORTED).all()
    else:
        assert out["p"].notna().any()


# ── C: preprocessing ───────────────────────────────────────────────────────────────────────────

def _scorable():
    out = {}
    for m, p in _served().items():
        pl = _load(p)
        if not M.unsupported_features(pl):
            out[m] = pl
    if not out:
        pytest.skip("no scorable served payload")
    return out


def test_predictions_do_not_depend_on_batch_composition_or_order():
    for market, payload in _scorable().items():
        df = _rows(payload["feature_cols"], 60)
        rng = np.random.default_rng(1)
        for c in payload["feature_cols"][:40]:
            df.loc[rng.random(len(df)) < 0.2, c] = np.nan             # incomplete rows
        full = M.predict_proba(df, payload)
        rev = M.predict_proba(df.iloc[::-1], payload).loc[df.index]
        alone = pd.concat([M.predict_proba(df.iloc[[i]], payload) for i in range(len(df))])
        junk = _rows(payload["feature_cols"], 30, seed=9) * 50 + 7
        junk.index = junk.index + 10_000
        mixed = M.predict_proba(pd.concat([junk, df]), payload).loc[df.index]
        for other in (rev, alone, mixed):
            np.testing.assert_allclose(other.to_numpy(), full.to_numpy(), rtol=0, atol=1e-12,
                                       err_msg=market)


def test_a_missing_required_input_fails_explicitly():
    for market, payload in _scorable().items():
        req = M.required_features(payload)
        assert req, market
        df = _rows(payload["feature_cols"], 6)
        df.loc[df.index[:3], req[0]] = np.nan
        out = M.predict_with_diagnostics(df, payload)
        assert out["p"].iloc[:3].isna().all() and out["required_fail"].iloc[:3].all(), market
        assert out["p"].iloc[3:].notna().all(), market
        absent = df.drop(columns=[req[0]])                            # an ABSENT required column
        assert M.predict_with_diagnostics(absent, payload)["p"].isna().all(), market


def test_prep_fills_nothing_and_never_creates_a_zero_column():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    X, cols = M._prep(df, ["a", "b"])
    assert X["a"].isna().sum() == 1 and X["b"].isna().all() and cols == ["a", "b"]


def test_no_statistic_of_holdout_or_live_rows_reaches_a_fitted_imputer():
    """The clean harness's recipe_retrained medians come from its fit block only, and serving uses the
    persisted values: perturbing the scoring rows cannot move them."""
    from player_model.props_harness import RecipeRetrained
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    X.loc[rng.random(400) < 0.3, "a"] = np.nan
    y = (rng.random(400) < 0.4).astype(int)
    m = RecipeRetrained().fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    assert m.fill_["a"] == pytest.approx(float(X["a"].iloc[:300].median()))
    live = X.iloc[300:].copy()
    p1 = m.predict(live)
    live2 = pd.concat([live, pd.DataFrame({"a": [1e6] * 50, "b": [1e6] * 50, "c": [1e6] * 50})])
    p2 = m.predict(live2)[: len(live)]
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-12)


def test_legacy_training_cannot_run():
    with pytest.raises(RuntimeError, match="retired"):
        M.train(_rows(C.PLAYER_FEATURE_COLS, 300).assign(target_goals=1), "goals")


# ── E: production path with missing inputs ──────────────────────────────────────────────────────

def test_production_path_with_missing_inputs_never_invents_an_observation():
    for market, payload in _scorable().items():
        df = _rows(payload["feature_cols"], 10)
        df.iloc[:, : len(df.columns) // 2] = np.nan                    # half the inputs missing
        out = M.predict_with_diagnostics(df, payload)
        feats = list(payload["feature_cols"])
        if payload.get("candidate") == "hgb_nan_native":
            # the harness model receives NaN: a different row with the cell FILLED gives another value
            X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in feats})
            assert X.isna().any().any()
            np.testing.assert_allclose(np.asarray(payload["model"].predict(X), float)[~out["required_fail"]],
                                       out["p"][~out["required_fail"]].to_numpy(), rtol=0, atol=0)
        elif payload.get("candidate") == "recipe_retrained":
            # the only fill is the PERSISTED fit-block median of the artifact
            fill = payload["state"]["fill_values"]
            assert set(fill) <= set(feats)
        assert out.loc[out["required_fail"], "p"].isna().all()


# ── F: adversarial regression scans ─────────────────────────────────────────────────────────────

FEATURE_CODE = ("player_model/feature_engineering.py", "player_model/injury_features.py",
                "player_model/lineup_started.py", "player_model/league_quality.py",
                "player_model/history_refetch.py", "player_model/model.py", "player_model/props_harness.py")


def _calls(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return tree, [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def test_no_fillna_with_an_invented_value_in_feature_code():
    """fillna(<literal>) is allowed only for 0 / False / "" on a COUNTER (a prefix sum's empty start,
    a boolean mask) - never a typed average (1.3, 0.5, 0.72, 6.0 ...)."""
    bad = []
    for f in FEATURE_CODE:
        _, calls = _calls(f)
        for c in calls:
            if isinstance(c.func, ast.Attribute) and c.func.attr == "fillna" and c.args:
                a = c.args[0]
                # a numeric literal other than 0 is an invented value; a string label is display text
                if (isinstance(a, ast.Constant) and isinstance(a.value, (int, float))
                        and not isinstance(a.value, bool) and a.value != 0):
                    bad.append(f"{f}:{c.lineno} fillna({a.value!r})")
    assert not bad, bad


def test_no_live_batch_median_in_the_scoring_path():
    """No .median() / .fillna() call inside the scoring functions (docstrings aside)."""
    tree = ast.parse((ROOT / "player_model/model.py").read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("_prep", "predict_with_diagnostics", "_predict_recipe", "predict_proba"):
        attrs = {c.func.attr for c in ast.walk(funcs[name])
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
        assert not attrs & {"median", "fillna"}, (name, attrs & {"median", "fillna"})


def test_no_silent_zero_column_creation():
    """`df[x] = 0` / `= 0.0` creating a feature column is refused in the feature and scoring code."""
    # a diagnostics COUNTER column (n_*) may start at 0; a feature column may not
    pat = re.compile(r"""^\s*\w+\[["'](?!n_)[\w]+["']\]\s*=\s*0(\.0)?\s*(#.*)?$""")
    bad = [f"{f}:{i}" for f in FEATURE_CODE
           for i, line in enumerate((ROOT / f).read_text(encoding="utf-8").splitlines(), 1) if pat.match(line)]
    assert not bad, bad


def test_schema_and_payload_must_match():
    """A repaired payload whose feature_cols differ from its persisted imputer is refused; the contract
    covers exactly the training feature list."""
    c = PC.load_contract()
    assert set(c["features"]) == set(C.PLAYER_FEATURE_COLS)
    df = _rows(C.PLAYER_FEATURE_COLS, 400)
    df["target_goals"] = (np.random.default_rng(0).random(400) < 0.3).astype(int)
    df["date"] = pd.date_range("2025-01-01", periods=400, freq="D")
    payload = M.build_payload(M.train(df, "goals", recipe=PC.RECIPE_REPAIRED), "goals")
    payload["feature_cols"] = list(payload["feature_cols"])[:-1]
    with pytest.raises(ValueError, match="differ"):
        M.predict_with_diagnostics(df.head(3), payload)


def test_unsupported_features_cannot_re_enter():
    for name in ("SETPIECE_PROXIES_REMOVED", "PROFILE_CONSTANT_FEATURES_WITHHELD", "SHOTS_DEFINITION_WITHHELD",
                 "TYPED_FORMULA_FEATURES_WITHHELD", "LEGACY_QUALITY_FEATURES_REMOVED"):
        assert not set(getattr(C, name)) & set(C.PLAYER_FEATURE_COLS), name
    adversarial = {"market": "x", "models": {}, "feature_cols": ["goals_pg", "sp_taker_score"], "metrics": {}}
    out = M.predict_with_diagnostics(pd.DataFrame({"goals_pg": [0.3]}), adversarial)
    assert out["p"].isna().all() and out["imputer_version"].iloc[0] == M.REFUSED_UNSUPPORTED


def test_point_in_time_a_row_never_reads_its_own_match():
    """Adversarial: change the row's OWN result - every pre-match feature of that row must not move."""
    rows = []
    for k in range(8):
        for pid, team, ih in ((1, "A", 1), (2, "B", 0)):
            rows.append({"fixture_id": 100 + k, "player_id": pid, "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * k),
                         "league": "Premier League", "home_team": "A", "away_team": "B", "team": team, "is_home": ih,
                         "season": "2025", "position": "F", "minutes": 90, "started": 1.0, "goals": 1, "assists": 0,
                         "shots_total": 3, "shots_on_target": 1, "yellow_cards": 0, "key_passes": 1, "rating": 7.0,
                         "duels_total": 5, "duels_won": 2, "fouls_committed": 1, "fouls_drawn": 1, "passes_total": 20})
    base = FE.build_features(rows)
    rows2 = [dict(r) for r in rows]
    for r in rows2:
        if r["fixture_id"] == 107:
            r.update(goals=5, shots_total=12, shots_on_target=9, started=0.0, minutes=20, yellow_cards=1)
    alt = FE.build_features(rows2)
    feats = [f for f in C.PLAYER_FEATURE_COLS if f in base.columns]
    a = base[base["fixture_id"] == 107].set_index("player_id")[feats]
    b = alt[alt["fixture_id"] == 107].set_index("player_id")[feats]
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def _invented_fills(src: str) -> list[int]:
    out = []
    for c in ast.walk(ast.parse(src)):
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "fillna" and c.args
                and isinstance(c.args[0], ast.Constant) and isinstance(c.args[0].value, (int, float))
                and not isinstance(c.args[0].value, bool) and c.args[0].value != 0):
            out.append(c.lineno)
    return out


ZERO_COL = re.compile(r"""^\s*\w+\[["'](?!n_)[\w]+["']\]\s*=\s*0(\.0)?\s*(#.*)?$""")


def test_the_scans_catch_adversarial_code():
    bad = '''
df["rest_days"] = df["rest_days"].fillna(6.0)
df["opp_goals_conceded_pg"] = df["opp_goals_conceded_pg"].fillna(1.3)
df["sp_taker_score"] = 0.0
X = X.fillna(X.median())
'''
    good = '''
prev = prev.where(prev.notna(), 0.0)
vz = v.fillna(0.0)
out["n_missing"] = 0
label = s.fillna("(none)")
'''
    assert _invented_fills(bad) == [2, 3] and _invented_fills(good) == []
    assert [i for i, l in enumerate(bad.splitlines()) if ZERO_COL.match(l)] == [3]
    assert not [l for l in good.splitlines() if ZERO_COL.match(l)]
    fn = ast.parse(bad)
    assert {c.func.attr for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)} \
        >= {"median", "fillna"}                                          # the batch-median line is seen


def test_candidate_ablation_drops_only_the_named_group(monkeypatch):
    from player_model import props_harness as H
    frame = pd.DataFrame(columns=list(C.PLAYER_FEATURE_COLS))
    full = H.harness_features(frame)
    monkeypatch.setenv(H.ABLATION_ENV, "injury")
    abl = H.harness_features(frame)
    assert set(full) - set(abl) == set(H.ABLATION_GROUPS["injury"])
    monkeypatch.setenv(H.ABLATION_ENV, "nonsense")
    with pytest.raises(ValueError):
        H.harness_features(frame)
