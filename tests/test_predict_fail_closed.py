"""R2-A finding 8: predict FAILS CLOSED for unknown leagues and for a missing New Format artifact.

Before: `unknown_mask` fixtures were scored by the STANDARD model ("Fallback (unknown league)"), and
New Format fixtures were scored by the STANDARD model when the NF artifact was missing. Both now get
NO probability -> no edge -> AVOID -> no tip, plus a routing record for the health file.
Routing is by EXACT canonical name: near-name variants (alias spellings, case, whitespace, typos,
registry leagues with no model) are unknown, never "close enough".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

KNOWN = [("Championship", "A0", "A1"), ("League One", "B0", "B1"), ("Brazil Serie A", "N0", "N1")]
NEAR_NAMES = ["Austria Bundesliga", "Danish Superliga", "championship", "CHAMPIONSHIP", "Serie B ",
              " League One", "La Liga2", "Bundesliga II", "Brazil Serie B", "Premier League",
              "Champions League", "World Cup", "Ligue  2", ""]
P_STUB = 0.62


def _board(leagues):
    day = pd.Timestamp("2026-11-07")
    rows = []
    for i, (lg, h, a) in enumerate(leagues):
        rows.append({"league": lg, "date": day, "kickoff_utc": day + pd.Timedelta(hours=15),
                     "home_team": h or f"H{i}", "away_team": a or f"A{i}",
                     "odds_over25": 2.10, "odds_under25": 1.80, "odds_over15": 1.3, "odds_over35": 3.1,
                     "odds_btts": 1.85})
    return pd.DataFrame(rows)


def _fake_features(upcoming, historical):
    f = upcoming.copy().reset_index(drop=True)
    for c in ("home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5"):
        f[c] = 1.4
    return f


@pytest.fixture()
def run(monkeypatch, tmp_path):
    import src.drift as D
    import src.feature_engineering as FE
    import src.health_check as H
    import src.model as M
    import src.predict as P
    monkeypatch.setattr(P, "_detect_postponed", lambda df: [])
    monkeypatch.setattr(FE, "build_upcoming_features", _fake_features)
    monkeypatch.setattr(D, "snapshot", lambda *a, **k: None)
    monkeypatch.setattr(D, "purge_old", lambda *a, **k: None)
    monkeypatch.setattr(D, "enrich", lambda df: df)
    monkeypatch.setattr(H, "record_fetch", lambda *a, **k: None)
    monkeypatch.setattr(H, "record_feature_health", lambda *a, **k: None)
    monkeypatch.setattr(M, "predict_proba", lambda X, payload=None, **k: pd.Series(P_STUB, index=X.index))
    monkeypatch.setattr(config, "HT_MODEL_FILE_05", tmp_path / "no_ht.pkl")
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE", "legacy_v9")
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE_RAW", "legacy_v9")

    def _run(leagues, nf_payload=True, side=True):
        board = _board(leagues)
        monkeypatch.setattr(P, "_fetch_odds_api", lambda days_ahead=7: board.copy())
        out, _ = P.predict_upcoming(pd.DataFrame(), {"feature_cols": []},
                                    payload_newformat={"feature_cols": []} if nf_payload else None,
                                    side_payloads={"btts": {"feature_cols": []}} if side else None)
        return out, dict(P._LAST_ROUTING)
    return _run


def test_known_leagues_still_scored(run):
    out, routing = run(KNOWN)
    assert (out["p_over25"] == P_STUB).all()
    assert list(out["model_type"]) == ["standard", "standard", "new_format"]
    assert routing["status"] == "PASS" and routing["unknown_league_fixtures"] == []


@pytest.mark.parametrize("name", NEAR_NAMES)
def test_unknown_or_near_name_league_gets_no_probability_and_no_bet(run, name):
    out, routing = run(KNOWN + [(name, "X1", "X2")])
    unk = out[out["home_team"] == "X1"].iloc[0]
    assert unk["model_type"] == "unknown"
    assert pd.isna(unk["p_over25"]), f"{name!r} was scored - silent routing"
    assert pd.isna(unk.get("p_btts", np.nan))
    assert unk["bet"] == "AVOID" and unk["signal_tier"] == "AVOID" and float(unk["bet_stake"]) == 0.0
    assert routing["status"] == "WARN"
    assert [f["league"] for f in routing["unknown_league_fixtures"]] == [name]
    known = out[out["home_team"] != "X1"]
    assert (known["p_over25"] == P_STUB).all()                      # neighbours unaffected


def test_unknown_league_never_reaches_tips(run):
    from src.betting import generate_bets
    out, _ = run(KNOWN + [("Austria Bundesliga", "X1", "X2")])
    # P_STUB=0.62 at 2.10 is a ~14pp edge on every scored row: ONLY the missing probability stops it
    b = generate_bets(out)
    assert not b.empty and "Austria Bundesliga" not in set(b["league"])


def test_missing_nf_artifact_leaves_nf_unscored_not_standard_scored(run):
    out, routing = run(KNOWN, nf_payload=False)
    nf = out[out["league"] == "Brazil Serie A"]
    assert nf["p_over25"].isna().all() and (nf["model_type"] == "new_format").all()
    assert set(nf["bet"]) == {"AVOID"}
    assert (out[out["model_type"] == "standard"]["p_over25"] == P_STUB).all()
    assert routing["new_format_artifact_missing"] is True and routing["new_format_unscored_fixtures"] == 1
    assert routing["status"] == "FAIL"


def test_standard_fallback_code_is_gone():
    src = (ROOT / "src" / "predict.py").read_text(encoding="utf-8")
    assert "payload_newformat if payload_newformat is not None else payload" not in src
    assert "p_unk = predict_proba" not in src


def test_routing_is_exact_canonical_only():
    assert config.model_type_for_league("Austria Bundesliga") == "unknown"     # alias is NOT a model route
    assert config.model_type_for_league("championship") == "unknown"
    assert config.model_type_for_league("Premier League") == "unknown"         # registry, collection only
    assert config.model_type_for_league("Austrian Bundesliga") == "new_format"


def test_integrity_health_written_by_pipeline_hook(run, monkeypatch, tmp_path):
    from src import league_registry as LR, season_resolver as SR, standard_serving as SS
    import src.predict as P
    monkeypatch.setattr(LR, "HEALTH_FILE", tmp_path / "lr.json")
    monkeypatch.setattr(SR, "HEALTH_FILE", tmp_path / "sr.json")
    monkeypatch.setattr(SS, "HEALTH_FILE", tmp_path / "ss.json")
    run(KNOWN + [("Danish Superliga", "X1", "X2")])
    P.write_integrity_health()
    lr = json.loads((tmp_path / "lr.json").read_text(encoding="utf-8"))
    assert lr["last_predict_routing"]["unknown_league_fixtures"][0]["league"] == "Danish Superliga"
    assert lr["status"] in ("WARN", "FAIL")
    assert json.loads((tmp_path / "sr.json").read_text(encoding="utf-8"))["leagues"]
    assert json.loads((tmp_path / "ss.json").read_text(encoding="utf-8"))["serving_recipe"] == "legacy_v9"
