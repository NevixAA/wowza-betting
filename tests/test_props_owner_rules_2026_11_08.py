"""Owner decisions 2026-11-08 on props (synthetic data only):

  * promotion of the clean-retrain models ("u can upload the new props ofcourse"), through
    props_gate.promote_clean_retrain: the harness model itself serves, the model in place is archived
    for rollback, the artifact hash is checked;
  * "No GK attacking tips": no goalkeeper tip on SOT / goals / assists, cards allowed, an unknown
    position is kept and labelled, never guessed;
  * "Retrain them too": goals2 / goals3 / sot3 / sot4 hidden from tips and the board until their clean
    retrain, and added to the props clean-retrain markets;
  * the CLV placeholder fix: a quote at odds <= 1.01 is not a price.
"""
from __future__ import annotations

import gzip
import io
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config as pconfig  # noqa: E402
from player_model import props_gate as PPG  # noqa: E402
from player_model import props_harness as H  # noqa: E402
from player_model import props_harness_clv as CLV  # noqa: E402
from player_model import props_contract as PC  # noqa: E402
from player_model.model import predict_proba, predict_with_diagnostics  # noqa: E402
from player_model.serving_rules import position_status, serving_skip_reason  # noqa: E402

T0 = datetime(2026, 11, 8, 22, 0, tzinfo=timezone.utc)


# ── serving rules ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("market", ["sot", "sot2", "goals", "assists"])
def test_goalkeeper_is_never_tipped_on_an_attacking_market(market):
    assert serving_skip_reason(market, position_status("G")) == pconfig.SKIP_GK_ATTACKING
    assert serving_skip_reason(market, position_status("Goalkeeper")) == pconfig.SKIP_GK_ATTACKING


def test_goalkeeper_cards_stay_allowed_and_outfield_players_pass():
    assert serving_skip_reason("cards", "G") is None
    for pos in ("D", "M", "F"):
        for m in ("sot", "sot2", "goals", "assists", "cards"):
            assert serving_skip_reason(m, pos) is None


@pytest.mark.parametrize("pos", [None, float("nan"), "", "nan", "  ", "X"])
def test_unknown_position_is_labelled_not_guessed(pos):
    s = position_status(pos)
    assert s == pconfig.POSITION_UNKNOWN
    assert serving_skip_reason("sot", s) is None            # kept (and labelled by predict)


@pytest.mark.parametrize("market", sorted(pconfig.HIDDEN_PENDING_CLEAN_RETRAIN))
def test_placeholder_markets_are_hidden_for_every_player(market):
    for pos in ("G", "D", "M", "F", pconfig.POSITION_UNKNOWN):
        assert serving_skip_reason(market, pos) == pconfig.SKIP_HIDDEN_MARKET


def test_hidden_set_and_gk_set_are_the_owner_lists():
    assert pconfig.HIDDEN_PENDING_CLEAN_RETRAIN == {"goals2", "goals3", "sot3", "sot4"}
    assert pconfig.GK_BLOCKED_MARKETS == {"sot", "sot2", "sot3", "sot4", "goals", "goals2", "goals3", "assists"}
    assert "cards" not in pconfig.GK_BLOCKED_MARKETS


def test_predict_applies_the_rule_with_a_reason_code_and_the_unknown_label():
    src = (ROOT / "player_model" / "predict.py").read_text(encoding="utf-8")
    assert "serving_skip_reason(market, _pos_status)" in src
    assert "_skip_reasons[_skip] += 1" in src
    assert '"position_status"' in src
    assert "if _is_gk:" not in src                          # the old silent skip is gone


def test_board_drops_hidden_and_gk_rows_and_counts_them():
    from dashboard_data import props_view as V
    tips = pd.DataFrame([
        {"market": "sot", "position": "G"}, {"market": "cards", "position": "G"},
        {"market": "sot3", "position": "F"}, {"market": "goals2", "position": "M"},
        {"market": "goals", "position": "F"}, {"market": "sot", "position": None},
    ])
    kept, removed = V.board_rows(tips)
    assert list(kept.index) == [1, 4, 5]
    assert removed == {pconfig.SKIP_HIDDEN_MARKET: 2, pconfig.SKIP_GK_ATTACKING: 1}
    note = V.hidden_markets_note()
    assert pconfig.SKIP_HIDDEN_MARKET in note and "placeholder" in note


# ── CLV placeholder quotes ───────────────────────────────────────────────────────────────────────

def _quote(ts, book, odds, side="yes", player="Djene Dakonam", day="2026-10-19"):
    return {"snapshot_ts": ts, "match_date": day, "league": "La Liga", "match": "Getafe vs Rayo Vallecano",
            "player": player, "market": "goals", "side": side, "point": None, "bookmaker": book,
            "odds": odds, "source": "oddsapi"}


def _dakonam_quotes():
    """The exact shape found on the holdout: a 56.0 opening price, later prices, and a Betfair
    Exchange 1.01 placeholder as the LAST answer before the match day."""
    return pd.DataFrame([
        _quote("2026-10-17T12:16:55Z", "William Hill", 36.0),
        _quote("2026-10-17T12:16:55Z", "Paddy Power", 56.0),
        _quote("2026-10-17T18:34:53Z", "Paddy Power", 61.0),
        _quote("2026-10-17T20:46:00Z", "Betfair", 20.0),
        _quote("2026-10-18T12:40:58Z", "Betfair", 1.01),
    ])


def _scored():
    return pd.DataFrame([{"market": "goals", "date": "2026-10-19", "player_name": "D. Dakonam",
                          "home_team": "Getafe", "away_team": "Rayo Vallecano", "y": 0.0, "p": 0.045}])


def test_a_1_01_quote_is_never_the_close():
    yp = CLV.yes_prices(_dakonam_quotes())
    assert len(yp) == 1
    assert yp["price_open"].iloc[0] == 56.0
    assert yp["price_close"].iloc[0] == 20.0               # not the 1.01 placeholder
    rows = CLV.clv_rows(_scored(), _dakonam_quotes(), "p")
    if len(rows):
        assert rows["clv"].max() < 51                      # the bug gave +51.3
        assert (rows["p_fair_close"] < 0.9).all()


def test_placeholder_counts_report_what_was_dropped():
    pc = CLV.placeholder_counts(_dakonam_quotes())
    assert pc["total"] == 1 and pc["of_quotes"] == 5 and pc["by_bookmaker"] == {"Betfair": 1}
    assert CLV.PLACEHOLDER_MAX_ODDS == 1.01
    assert CLV.placeholder_counts(pd.DataFrame())["total"] == 0


def test_only_placeholders_give_no_price_at_all():
    q = pd.DataFrame([_quote("2026-10-18T12:40:58Z", "Betfair", 1.01)])
    assert CLV.yes_prices(q).empty


# ── harness: the four old markets ───────────────────────────────────────────────────────────────

def test_harness_supports_the_four_old_markets():
    for m in ("goals2", "goals3", "sot3", "sot4"):
        assert m in H.MARKETS_IN_SCOPE and m in H.TARGETS and m in H.LABEL_RULE
    assert H.TARGETS["goals3"] == "target_goals3" and H.LABEL_RULE["goals3"] == ("goals", 3)
    assert "goals3" not in pconfig.MARKET_TARGETS          # no serving model entry: harness-only target
    H.validate_decisions({"holdout_start": "2026-09-01", "holdout_end": "2026-10-31",
                          "min_verified_share": 0.95, "min_verified_share_holdout": 0.85,
                          "markets": list(H.MARKETS_IN_SCOPE), "feature_rule": "all_prior_verified"})


def test_goals3_label_follows_the_null_rule():
    df = pd.DataFrame({"goals": [0, 2, 3, 4, np.nan]})
    lab = H.label(df, "goals3")
    assert lab.iloc[:4].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert np.isnan(lab.iloc[4])                           # a missing count is a missing label


def test_market_verified_share_is_per_market():
    f = pd.DataFrame({"date": pd.to_datetime(["2026-08-01"] * 4 + ["2026-09-10"] * 4),
                      "target_sot": [1, 0, 1, 0, 1, 0, 1, 0],
                      "target_sot4": [1, np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan]})
    ok = np.array([True, False, True, True, True, False, True, True])
    split = H.split_of(f["date"], "2026-09-01", "2026-10-31")
    a = H.market_verified_share(f, ok, split, "sot")
    b = H.market_verified_share(f, ok, split, "sot4")
    assert (a["train"], a["holdout"]) == (0.75, 0.75)
    assert (b["train"], b["holdout"], b["train_rows"], b["holdout_rows"]) == (1.0, 0.5, 1, 2)


def _synthetic_frame():
    rng = np.random.default_rng(3)
    rows, fid = [], 1000
    for d in pd.date_range("2026-01-01", periods=90, freq="D"):
        for p in range(40):
            if rng.random() < 0.5:
                continue
            fid += 1
            x = rng.normal()
            rows.append({"date": d, "player_id": p + 1, "fixture_id": fid, "f1": x,
                         "target_sot": float(rng.random() < 1 / (1 + np.exp(-x))),
                         "target_sot4": float(rng.random() < 0.3), "label_verified": True,
                         "window_verified": True, "all_prior_verified": True,
                         "player_name": f"P{p}", "home_team": "Home FC", "away_team": "Away FC"})
    return pd.DataFrame(rows)


def test_a_market_under_its_own_floor_is_not_fitted_while_the_others_are(monkeypatch, tmp_path):
    monkeypatch.setenv(H.ENABLE_ENV, "true")
    f = _synthetic_frame()
    hold = pd.to_datetime(f["date"]) >= pd.Timestamp("2026-03-01")
    # sot4's holdout labels exist only on unverified rows -> its holdout share is 0
    f.loc[hold & (np.arange(len(f)) % 2 == 0), "label_verified"] = False
    f.loc[hold & (np.arange(len(f)) % 2 == 1), "target_sot4"] = np.nan
    monkeypatch.setattr(H, "build_frame", lambda raw: f)
    monkeypatch.setattr(H, "harness_features", lambda frame: ["f1"])
    plan = json.loads(json.dumps(H.load_plan()))
    plan["owner_decisions"] = {"holdout_start": "2026-03-01", "holdout_end": "2026-03-31",
                               "min_verified_share": 0.4, "min_verified_share_holdout": 0.4,
                               "markets": ["sot", "sot4"], "feature_rule": "label_only"}
    plan["walk_forward"] = {"step_matchdays": 5, "n_folds": 2}
    rep = H.run(plan, raw=pd.DataFrame(), out_dir=tmp_path / "o", model_dir=tmp_path / "m",
                quotes=pd.DataFrame())
    assert rep["status"] == "OK"
    assert rep["markets"]["sot4"]["status"].startswith("STOPPED") and "holdout share" in rep["markets"]["sot4"]["status"]
    assert "artifacts" not in rep["markets"]["sot4"]
    assert "artifacts" in rep["markets"]["sot"] and rep["markets"]["sot"]["verified_share"]["holdout"] >= 0.4
    assert not list((tmp_path / "m").glob("props_clean_sot4_*"))


def test_harness_main_pickles_through_the_imported_module():
    src = (ROOT / "player_model" / "props_harness.py").read_text(encoding="utf-8")
    tail = src[src.rindex('if __name__ == "__main__":'):]
    assert "from player_model import props_harness as _module" in tail and "_module.main()" in tail


# ── promotion of a clean-retrain model ──────────────────────────────────────────────────────────

class _Legacy:
    def predict_proba(self, X):
        p = np.full(len(X), 0.2)
        return np.column_stack([1 - p, p])


@pytest.fixture()
def ptmp(tmp_path, monkeypatch):
    models = tmp_path / "models"
    lin = tmp_path / "out" / "model_lineage"
    files = {m: models / f"model_player_{m}.pkl" for m in pconfig.MARKETS}
    monkeypatch.setattr(pconfig, "MODEL_FILES", files)
    monkeypatch.setattr(pconfig, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(PPG, "REGISTRY_DIR", models / "registry" / "props")
    monkeypatch.setattr(PPG, "PROPS_POINTER_FILE", models / "registry" / "props" / "props_champions.json")
    monkeypatch.setattr(PPG, "LINEAGE_DIR", lin)
    monkeypatch.setattr(PPG, "DECISIONS_FILE", lin / "props_promotion_decisions.csv")
    monkeypatch.setattr(PPG, "CHAMPION_HISTORY", lin / "props_champion_history.csv")
    models.mkdir(parents=True)
    legacy = pickle.dumps({"models": {"m": _Legacy()}, "feature_cols": ["f1", "f2"], "market": "sot"})
    files["sot"].write_bytes(legacy)
    yield tmp_path, files, legacy


def _artifact(market="sot", candidate="hgb_nan_native"):
    f = _synthetic_frame()
    f["f2"] = np.where(np.arange(len(f)) % 5 == 0, np.nan, f["f1"] * 0.5)
    cut = pd.Timestamp("2026-03-01")
    tr, cal = f[f["date"] < cut], f[f["date"] >= cut]
    cls = H.HGBNaNNative if candidate == "hgb_nan_native" else H.RecipeRetrained
    m = cls().fit(tr[["f1", "f2"]], tr["target_sot"], cal[["f1", "f2"]], cal["target_sot"])
    art = {"market": market, "candidate": candidate, "harness_version": H.HARNESS_VERSION,
           "hand_adjustment": "none", "model": m, "state": m.state(),
           "window": {"fit_dates": ["2026-01-01", "2026-02-28"], "calibration_dates": ["2026-03-01", "2026-03-31"],
                      "fit_rows": len(tr), "calibration_rows": len(cal)},
           "owner_decisions": {}}
    return pickle.dumps(art), m, f


def _promote(raw, sha=None, market="sot"):
    import hashlib
    return PPG.promote_clean_retrain(market, raw, expected_sha256=sha or hashlib.sha256(raw).hexdigest(),
                                     run_id="42513445187", code_sha="271632d0",
                                     holdout_metrics={"n": 100, "events": 20, "ll_delta_vs_recipe": -0.001,
                                                      "ci95_lo": -0.002, "ci95_hi": -0.0005},
                                     approval="u can upload the new props ofcourse", approved_on="2026-11-08",
                                     now=T0)


@pytest.mark.parametrize("candidate", ["hgb_nan_native", "recipe_retrained"])
def test_promoted_model_serves_exactly_the_harness_probability(ptmp, candidate):
    tmp, files, legacy = ptmp
    raw, model, f = _artifact(candidate=candidate)
    rec = _promote(raw)
    assert rec["verdict"] == PPG.PROMOTE and rec["promoted"]
    payload = pickle.loads(files["sot"].read_bytes())
    assert payload["recipe"] == PC.RECIPE_CLEAN_HARNESS and payload["candidate"] == candidate
    X = f[["f1", "f2"]].tail(200).copy()
    X["unrelated"] = 1.0
    served = predict_proba(X, payload).to_numpy()
    np.testing.assert_array_equal(served, model.predict(X))
    # an absent feature column reaches the model as NaN, never as a typed value
    X2 = X.drop(columns=["f2"])
    np.testing.assert_array_equal(predict_proba(X2, payload).to_numpy(),
                                  model.predict(X2.assign(f2=np.nan)))
    diag = predict_with_diagnostics(X2, payload)
    assert (diag["n_missing"] == 1).all() and (diag["recipe"] == PC.RECIPE_CLEAN_HARNESS).all()


def test_promotion_archives_the_model_in_place_and_records_everything(ptmp):
    tmp, files, legacy = ptmp
    raw, _, _ = _artifact()
    rec = _promote(raw)
    ptr = json.loads(PPG.PROPS_POINTER_FILE.read_text(encoding="utf-8"))
    assert ptr["sot"]["promotion_source"] == PPG.CLEAN_RETRAIN_SOURCE
    assert ptr["sot"]["model_version"] == rec["model_version"]
    assert ptr["sot"]["training_cutoff"] == "2026-03-31"
    prior = rec["champion_version"]
    assert prior.startswith("props_sot_legacy_")
    stored = gzip.decompress(PPG._champion_bytes_path("sot", prior).read_bytes())
    assert stored == legacy                                 # the old bytes are kept for rollback
    dec = pd.read_csv(PPG.DECISIONS_FILE)
    last = dec.iloc[-1]
    assert last["stage"] == PPG.CLEAN_RETRAIN_SOURCE and bool(last["promoted"])
    assert "u can upload the new props" in last["reason"]
    man = PPG.read_manifest("sot", rec["model_version"])
    assert man["clean_retrain"]["run_id"] == "42513445187" and man["clean_retrain"]["code_sha"] == "271632d0"
    PPG.rollback_props("sot", prior, reason="test")
    assert files["sot"].read_bytes() == legacy


def test_promotion_refuses_a_wrong_hash_a_wrong_market_and_a_hidden_market(ptmp):
    tmp, files, legacy = ptmp
    raw, _, _ = _artifact()
    with pytest.raises(ValueError, match="sha256"):
        _promote(raw, sha="0" * 64)
    with pytest.raises(ValueError, match="not goals"):
        _promote(raw, market="goals")
    with pytest.raises(PermissionError, match="hidden"):
        _promote(raw, market="sot3")
    assert files["sot"].read_bytes() == legacy             # nothing changed


def test_main_pickled_artifacts_load_through_the_gate(monkeypatch):
    """The first run pickled classes as __main__.<Class>; the gate's unpickler maps them."""
    import hashlib
    _, model, _ = _artifact()
    main = sys.modules["__main__"]
    monkeypatch.setattr(H.HGBNaNNative, "__module__", "__main__")
    monkeypatch.setattr(main, "HGBNaNNative", H.HGBNaNNative, raising=False)
    raw_main = pickle.dumps({"market": "sot", "candidate": "hgb_nan_native", "harness_version": "x",
                             "model": model, "state": {}, "window": {}})
    monkeypatch.undo()
    assert b"__main__" in raw_main
    art = PPG.load_clean_artifact(raw_main, hashlib.sha256(raw_main).hexdigest())
    assert isinstance(art["model"], H.HGBNaNNative)


def test_committed_serving_models_are_the_clean_ones_where_promoted():
    """After the promotion commit: sot/sot2/goals/assists serve hgb_nan_native and cards serves
    recipe_retrained, from run 42855107089 (2026-11-10 retrain on real profiles, 128 features; it replaced run
    42640972400 of 2026-11-09); the four hidden markets keep their old files."""
    ptr = json.loads((ROOT / "models" / "registry" / "props" / "props_champions.json").read_text(encoding="utf-8"))
    want = {"sot": "hgb_nan_native", "sot2": "hgb_nan_native", "goals": "hgb_nan_native",
            "assists": "hgb_nan_native", "cards": "recipe_retrained"}
    for m, cand in want.items():
        assert ptr[m]["promotion_source"] == PPG.CLEAN_RETRAIN_SOURCE, m
        payload = pickle.loads((ROOT / "models" / f"model_player_{m}.pkl").read_bytes())
        assert payload["recipe"] == PC.RECIPE_CLEAN_HARNESS and payload["candidate"] == cand
        assert payload["provenance"]["run_id"] == "42855107089"
    for m in ("goals2", "sot3", "sot4"):
        assert ptr[m]["promotion_source"] != PPG.CLEAN_RETRAIN_SOURCE, m
