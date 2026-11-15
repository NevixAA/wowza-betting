"""Props profile features on REAL values only, and the served-feature spread check (2026-11-10).

Finding of 2026-11-10: player_history stores age 25 / height 180 / weight 75 / age_peak_delta 0 on
every row, and the served props payloads list age / height features. These tests pin that (a) the
feature code never reads a stored placeholder as a value, (b) age is the age on the match date, (c)
the props feature list carries the profile features and no NaN-by-construction column, and (d) data
truth rule 3 turns RED when a served feature holds one value (a placeholder, a constant or NaN) on
its training rows - which it does on today's committed payloads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model import league_quality as LQ  # noqa: E402
from src.integrity import data_truth as DT  # noqa: E402


def _rows():
    return pd.DataFrame({"player_id": [1, 1, 2, 3], "date": pd.to_datetime(
        ["2020-01-01", "2024-07-01", "2024-07-01", "2024-07-01"]),
        "age": [25.0] * 4, "height_cm": [180.0] * 4, "weight_kg": [75.0] * 4})


def _profiles():
    return pd.DataFrame({"player_id": [1, 2], "height_cm": [191.0, np.nan], "weight_kg": [84.0, 70.0],
                         "birth_date": ["2000-01-01", None]})


# ── the feature path ────────────────────────────────────────────────────────────────────────────

def test_stored_placeholders_are_missing_without_a_profile(monkeypatch):
    monkeypatch.setattr(FE, "collected_profiles", lambda: pd.DataFrame())
    out = FE._attach_profiles(_rows())
    assert out[["age", "height_cm", "weight_kg"]].isna().all().all()


def test_profile_values_and_age_on_the_match_date(monkeypatch):
    monkeypatch.setattr(FE, "collected_profiles", _profiles)
    out = FE._attach_profiles(_rows())
    assert list(out["height_cm"].iloc[:2]) == [191.0, 191.0]
    assert out["weight_kg"].iloc[2] == 70.0
    # born 2000-01-01: 20.0 on 2020-01-01 and 24.5 on 2024-07-01 - the age THEN, not today's age
    assert out["age"].iloc[0] == pytest.approx(20.0, abs=0.01)
    assert out["age"].iloc[1] == pytest.approx(24.5, abs=0.01)
    # no birth date / no height / no profile at all: NaN, never 25 or 180
    assert np.isnan(out["age"].iloc[2]) and np.isnan(out["height_cm"].iloc[2])
    assert out.iloc[3][["age", "height_cm", "weight_kg"]].isna().all()


def test_defender_heights_count_only_known_heights(monkeypatch):
    monkeypatch.setattr(FE, "collected_profiles", _profiles)
    h = FE._height_series(_rows())
    assert list(h.iloc[:2]) == [191.0, 191.0]
    assert h.iloc[2:].isna().all()                      # 180 stored placeholder -> missing


def test_profile_composites_stay_nan_on_a_missing_input():
    f = pd.DataFrame({"height_cm": [190.0, np.nan], "opp_def_mean_height": [185.0, 185.0],
                      "aerial_won_rate": [0.5, 0.5], "age": [30.0, np.nan]})
    FE.profile_composites(f)
    assert f["height_diff_vs_opp_def"].iloc[0] == 5.0 and np.isnan(f["height_diff_vs_opp_def"].iloc[1])
    assert f["age_peak_delta"].iloc[0] == 3.0 and np.isnan(f["age_peak_delta"].iloc[1])
    assert np.isnan(f["height_aerial_interaction"].iloc[1])


def test_feature_list_has_the_profiles_and_no_nan_by_construction_column():
    assert set(config.PROFILE_FEATURES) <= set(config.PLAYER_FEATURE_COLS)
    assert set(config.LEGACY_QUALITY_FEATURES_REMOVED) == set(LQ.LEGACY_COLUMNS)
    assert not set(LQ.LEGACY_COLUMNS) & set(config.PLAYER_FEATURE_COLS)


def test_no_placeholder_writer_in_the_feature_code():
    lines = (ROOT / "player_model" / "feature_engineering.py").read_text(encoding="utf-8").splitlines()
    src = "\n".join(ln.split("#", 1)[0] for ln in lines)          # code, not the history comments
    for bad in ('["height_cm"] = 180', '["age"] = 25', '["weight_kg"] = 75', "fillna(180", "fillna(25"):
        assert bad not in src, bad


# ── data truth rule 3: served features carry data ───────────────────────────────────────────────

class _Prof:
    def __init__(self, prof):
        self.training_profile_ = prof


def test_training_profile_placeholder_turns_red():
    pl = {"feature_cols": ["age", "goals_pg"], "model": _Prof({
        "age": {"top_value": 25.0, "top_share": 1.0}, "goals_pg": {"top_value": 0.0, "top_share": 0.41}})}
    sv = DT.props_payload_spread(pl)
    assert set(sv) == {"age"} and sv["age"]["placeholder"] and sv["age"]["value"] == 25.0


def test_single_value_share_above_the_line_turns_red():
    pl = {"feature_cols": ["a", "b"], "model": _Prof({
        "a": {"top_value": 1.0, "top_share": 0.9995}, "b": {"top_value": None, "top_share": 0.999}})}
    assert set(DT.props_payload_spread(pl)) == {"a"}                 # 99.9% exactly is not above


def test_fitted_estimators_prove_a_constant():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from player_model.model import _UsableColumns

    X = pd.DataFrame({"real": np.arange(50.0), "ph": np.full(50, 180.0), "dead": np.full(50, np.nan)})

    class _Cal:
        def __init__(self, pipe):
            self._pipe = pipe
    usable = _Cal(Pipeline([("usable", _UsableColumns().fit(X))]))
    scaler = _Cal(Pipeline([("scaler", StandardScaler().fit(X[["real", "ph"]]))]))

    class _HGB:
        feature_cols_ = ["real", "ph", "dead"]
        model_ = usable
    sv = DT.props_payload_spread({"feature_cols": ["real", "ph", "dead"], "model": _HGB()})
    assert set(sv) == {"ph", "dead"}
    legacy = DT.props_payload_spread({"feature_cols": ["real", "height_cm"], "models": {"lr": scaler}})
    assert set(legacy) == {"height_cm"} and legacy["height_cm"]["value"] == 180.0
    assert legacy["height_cm"]["placeholder"]


def test_team_imputer_record():
    pl = {"feature_cols": ["a", "b", "c"], "imputer": {"features": {
        "a": {"missing_rate": 0.0, "min": 0.0, "max": 3.0},
        "b": {"missing_rate": 0.0, "min": 2.0, "max": 2.0},
        "c": {"missing_rate": 1.0, "min": None, "max": None}}}}
    assert set(DT.team_payload_spread(pl)) == {"b", "c"}


def test_rule_3_red_on_a_spread_finding_even_without_a_board(tmp_path, monkeypatch):
    monkeypatch.setattr(DT, "OUT", tmp_path)
    spread = {"props:x.pkl": {"status": DT.RED, "note": "", "single_value": {
        "age": {"value": 25.0, "share": 1.0, "how": "training_profile", "placeholder": True}}}}
    c = DT.missing(spread=spread)
    assert c["status"] == DT.RED and "PLACEHOLDER: age" in c["value"]
    assert DT.missing(spread={})["status"] == DT.GREY


def test_todays_committed_payloads():
    """Today (2026-11-10): RED - four served props payloads list the six legacy league-quality
    columns, NaN on every training row. No served payload holds a profile placeholder (the
    2026-11-09 props models learned real ages and heights); the hidden legacy payloads do."""
    sp = DT.served_feature_spread()
    served = {k: v for k, v in sp.items() if v["status"] != DT.NOT_SERVED}
    for k, v in served.items():
        assert not [c for c, f in v["single_value"].items() if f["placeholder"]], k
    red = {k for k, v in served.items() if v["status"] == DT.RED}
    if any(set(LQ.LEGACY_COLUMNS) <= set(v["single_value"]) for v in served.values()):
        assert red                                     # the 2026-11-09 payloads are still served
    hidden = {k: v for k, v in sp.items() if v["status"] == DT.NOT_SERVED}
    assert all(k.startswith("props:") for k in hidden)
