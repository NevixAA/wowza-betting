"""The ONE production promotion gate (owner requirement 2026-10-25).

    python -m pytest tests/test_promotion_gate.py -q

Every test runs against a temporary registry — nothing here touches models/ or output/.
"""
from __future__ import annotations

import gzip
import json
import pickle
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import model_registry as reg  # noqa: E402
from src import promotion_gate as PG  # noqa: E402

STD = sorted(config.STANDARD_FORMAT_LEAGUES)[:2]
NF = sorted(config.NEW_FORMAT_LEAGUES)[:2]
T0 = datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc)


@pytest.fixture()
def tmpreg(tmp_path, monkeypatch):
    files = {mt: tmp_path / "models" / f"model_{mt}.pkl" for mt in reg.MODEL_FILES}
    monkeypatch.setattr(reg, "MODEL_FILES", files)
    monkeypatch.setattr(reg, "REGISTRY_DIR", tmp_path / "models" / "registry")
    monkeypatch.setattr(reg, "POINTER_FILE", tmp_path / "models" / "registry" / "cc.json")
    lin = tmp_path / "out" / "model_lineage"
    monkeypatch.setattr(reg, "LINEAGE_DIR", lin)
    monkeypatch.setattr(reg, "CHAMPION_HISTORY", lin / "champion_history.csv")
    monkeypatch.setattr(reg, "RETRAIN_BATCHES", lin / "retrain_batches.csv")
    monkeypatch.setattr(reg, "HIGH_WATER", lin / "high_water_mark.json")
    monkeypatch.setattr(PG, "OUTCOMES_FILE", tmp_path / "out" / "v91_shadow.csv")
    monkeypatch.setattr(PG.config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.delenv(PG.OWNER_OVERRIDE_ENV, raising=False)
    reg._version_cache.clear()
    yield tmp_path
    reg._version_cache.clear()


# ── synthetic data ────────────────────────────────────────────────────────────────────────────
def _data(n=4000, seed=1, leagues=None):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < truth).astype(float)
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(n) * 6, unit="h")
    leagues = leagues if leagues is not None else np.array([STD[i % 2] for i in range(n)])
    return y, truth, dates, leagues


def _noisy(truth, sd, seed=2):
    return np.clip(truth + np.random.default_rng(seed).normal(0, sd, len(truth)), 0.02, 0.98)


def _manifest(mt, raw, t=datetime(2026, 10, 18, 8, 0, tzinfo=timezone.utc)):
    sha = reg.sha256_bytes(raw)
    return {"model_type": mt, "model_version": reg.make_version(mt, t, sha),
            "artifact_sha256": sha, "retrain_batch_id": "rb_test", "training_cutoff": ""}


# ── config ────────────────────────────────────────────────────────────────────────────────────
def test_thresholds_are_declared_in_one_versioned_dated_file():
    cfg = json.loads((ROOT / "config" / "promotion_gate.json").read_text(encoding="utf-8"))
    assert cfg["gate_version"] == PG.GATE_VERSION and cfg["declared_on"] == "2026-10-25"
    h = cfg["holdout"]
    assert h["min_n_per_track"] == 1000 and h["material_ll_improvement"] == 0.001
    assert h["bootstrap_unit"] == "match_date"
    assert cfg["prospective"]["min_n_per_track"] == 1000
    assert PG.MIN_EFFECT_LOG_LOSS == h["material_ll_improvement"]       # one source


# ── period 1: the holdout ─────────────────────────────────────────────────────────────────────
def test_better_candidate_qualifies_on_holdout():
    y, truth, d, lg = _data()
    ev = PG.evaluate_holdout(y, _noisy(truth, 0.06), truth, dates=d, leagues=lg)
    assert ev["qualified"], ev["reason"]
    t = ev["tracks"]["standard"]
    assert t["ll_delta"] <= -PG.MATERIAL_LL and t["ci"][1] < 0 and t["n_dates"] >= 20


def test_candidate_within_old_degradation_tolerance_does_not_qualify():
    """+0.002 log loss sat comfortably inside the old 0.030 tolerance and promoted."""
    y, truth, d, lg = _data()
    worse = _noisy(truth, 0.05, seed=3)
    ev = PG.evaluate_holdout(y, truth, worse, dates=d, leagues=lg)
    t = ev["tracks"]["standard"]
    assert 0 < t["ll_delta"] < 0.030
    assert not ev["qualified"] and not t["checks"]["material_improvement"]


def test_immaterial_but_significant_gain_does_not_qualify():
    y, truth, d, lg = _data(n=20000, seed=5)
    champ = _noisy(truth, 0.012, seed=6)
    ev = PG.evaluate_holdout(y, champ, truth, dates=d, leagues=lg)
    t = ev["tracks"]["standard"]
    assert -PG.MATERIAL_LL < t["ll_delta"] < 0
    assert not t["checks"]["material_improvement"] and not ev["qualified"]


def test_league_regression_blocks():
    n = 4000
    lg = np.array([STD[0]] * 3400 + [STD[1]] * 600)
    y, truth, d, _ = _data(n=n, leagues=lg)
    champ = _noisy(truth, 0.06)
    cand = truth.copy()
    cand[3400:] = np.clip(1 - truth[3400:], 0.02, 0.98)        # destroys the second league
    ev = PG.evaluate_holdout(y, champ, cand, dates=d, leagues=lg)
    t = ev["tracks"]["standard"]
    assert not t["checks"]["no_league_regression"] and not ev["qualified"]
    assert t["league_regressions"][0]["league"] == STD[1]


def test_canary_blocks_mean_shift_and_spread_change():
    y, truth, d, lg = _data()
    champ = _noisy(truth, 0.06)
    shifted = np.clip(truth + 0.03, 0.02, 0.98)
    ev = PG.evaluate_holdout(y, champ, shifted, dates=d, leagues=lg)
    assert not ev["tracks"]["standard"]["checks"]["prediction_canary"] and not ev["qualified"]
    ok, det = PG.prediction_canary(champ, np.clip(0.5 + (champ - 0.5) * 0.4, 0.02, 0.98))
    assert not ok and any("sd ratio" in f for f in det["failures"])
    ok2, det2 = PG.prediction_canary(champ, np.where(champ > 0.5, 0.99, 0.01))
    assert not ok2


def test_brier_and_calibration_are_reported_per_track():
    y, truth, d, lg = _data()
    ev = PG.evaluate_holdout(y, _noisy(truth, 0.06), truth, dates=d, leagues=lg)
    t = ev["tracks"]["standard"]
    for k in ("brier_non_inferior", "ece_not_worse", "slope_not_worse"):
        assert k in t["checks"]
    assert t["candidate"]["cal_slope"] is not None


def test_tracks_are_never_pooled():
    """Standard improves a lot, new-format gets worse: a pooled number would pass; per track it
    must not."""
    h = 6000
    lg = np.array([STD[i % 2] for i in range(h)] + [NF[i % 2] for i in range(h)])
    y, truth, d, _ = _data(n=2 * h, leagues=lg)
    champ = truth.copy()
    champ[:h] = _noisy(truth[:h], 0.08)                       # standard champion is poor
    cand = truth.copy()
    cand[h:] = _noisy(truth[h:], 0.04, seed=4)                # NF slightly worse
    ev = PG.evaluate_holdout(y, champ, cand, dates=d, leagues=lg)
    assert set(ev["evaluable_tracks"]) == {"standard", "new_format"}
    assert ev["tracks"]["standard"]["passed"] and not ev["tracks"]["new_format"]["passed"]
    pooled = float((PG._ll(y, cand) - PG._ll(y, champ)).mean())
    assert pooled < -PG.MATERIAL_LL            # the pool would have looked like a clear win
    assert not ev["qualified"]


def test_small_track_is_reported_but_cannot_decide():
    lg = np.array([STD[i % 2] for i in range(3000)] + [NF[0]] * 400)
    y, truth, d, _ = _data(n=3400, leagues=lg)
    ev = PG.evaluate_holdout(y, _noisy(truth, 0.06), truth, dates=d, leagues=lg)
    assert ev["evaluable_tracks"] == ["standard"]
    assert ev["tracks"]["new_format"]["evaluable"] is False


def test_fail_closed_on_missing_dates_leagues_or_few_matchdays():
    y, truth, d, lg = _data()
    assert not PG.evaluate_holdout(y, truth, truth, dates=None, leagues=lg)["qualified"]
    assert not PG.evaluate_holdout(y, truth, truth, dates=d, leagues=None)["qualified"]
    few = pd.to_datetime(["2026-01-01"] * 2000 + ["2026-01-02"] * 2000)
    ev = PG.evaluate_holdout(y, _noisy(truth, 0.06), truth, dates=few, leagues=lg)
    assert not ev["qualified"] and not ev["tracks"]["standard"]["checks"]["ci_excludes_zero"]


def test_holdout_seen_by_champion_is_refused():
    y, truth, d, lg = _data()
    ev = PG.evaluate_holdout(y, _noisy(truth, 0.06), truth, dates=d, leagues=lg,
                             champion_cutoff="2026-06-01")
    assert ev["untouched"] == "VIOLATED" and not ev["qualified"]


# ── submit_candidate: retrains still save candidates, never promote ────────────────────────
def _submit(tmpreg, monkeypatch, p_new, n=4000, leagues=None, at=T0, seed_champion=True):
    y, truth, d, lg = _data(n=n, leagues=leagues)
    test_df = pd.DataFrame({"date": d, "league": lg, "over25": y})
    old = b"old-champion"
    if seed_champion:
        PG._install_champion("standard", _manifest("standard", old), old, reason="seed")
    champ = _noisy(truth, 0.06)
    import src.model as M
    monkeypatch.setattr(M, "predict_proba",
                        lambda df, payload=None: champ if payload == "OLD" else
                        (truth if p_new is None else p_new(truth)))
    results = {"lr": {"model": f"m{at.isoformat()}", "feature_cols": ["f"],
                      "metrics": {"log_loss": 0.69}}}
    return PG.submit_candidate("standard", reg.MODEL_FILES["standard"],
                               "OLD" if seed_champion else None, results, test_df, "over25",
                               "same_holdout" if seed_champion else "no_incumbent",
                               test_df, batch="rb_t", now=at), old


def test_qualified_candidate_is_parked_not_promoted(tmpreg, monkeypatch):
    (gate, extra), old = _submit(tmpreg, monkeypatch, None)
    assert gate["qualified"] and gate["verdict"] == PG.SHADOW and gate["parked"]
    assert reg.MODEL_FILES["standard"].read_bytes() == old           # champion untouched
    assert reg.challenger_path("standard").exists()
    dec = pd.read_csv(PG.decisions_file())
    assert dec.iloc[-1]["gate_version"] == PG.GATE_VERSION
    assert dec.iloc[-1]["stage"] == "HOLDOUT" and not dec.iloc[-1]["promoted"]
    rb = pd.read_csv(reg.RETRAIN_BATCHES)
    assert rb.iloc[-1]["verdict"] == PG.SHADOW and not rb.iloc[-1]["promoted"]


def test_worse_candidate_keeps_champion_and_is_recorded(tmpreg, monkeypatch):
    (gate, extra), old = _submit(tmpreg, monkeypatch, lambda t: np.clip(t * 0.9 + 0.05 +
                                                                       0.06, 0.02, 0.98))
    assert gate["verdict"] == PG.KEEP and not gate["parked"]
    assert reg.MODEL_FILES["standard"].read_bytes() == old
    assert reg.read_manifest("standard", extra["challenger_version"])["model_role"] == "REJECTED"


def test_unscorable_incumbent_fails_closed(tmpreg, monkeypatch):
    old = b"old"
    PG._install_champion("standard", _manifest("standard", old), old, reason="seed")
    y, truth, d, lg = _data()
    df = pd.DataFrame({"date": d, "league": lg, "over25": y})
    results = {"lr": {"model": "m", "feature_cols": ["f"], "metrics": {}}}
    gate, _ = PG.submit_candidate("standard", reg.MODEL_FILES["standard"], None, results, df,
                                  "over25", "incumbent_unscorable", df, now=T0)
    assert gate["verdict"] == PG.KEEP and "EVALUATION_FAILED" in gate["reason"]
    assert reg.MODEL_FILES["standard"].read_bytes() == old


def test_bootstrap_is_explicit_and_only_without_any_champion_record(tmpreg, monkeypatch):
    (gate, extra), _ = _submit(tmpreg, monkeypatch, None, seed_champion=False)
    assert gate["verdict"] == PG.BOOTSTRAP
    assert reg.read_pointer()["standard"]["promotion_source"] == "BOOTSTRAP"
    assert pd.read_csv(PG.decisions_file()).iloc[-1]["stage"] == "BOOTSTRAP"


def test_missing_serving_file_is_restored_not_bootstrapped(tmpreg, monkeypatch):
    old = b"registered-champion"
    PG._install_champion("standard", _manifest("standard", old), old, reason="seed")
    reg.MODEL_FILES["standard"].unlink()
    y, truth, d, lg = _data()
    df = pd.DataFrame({"date": d, "league": lg, "over25": y})
    results = {"lr": {"model": "new", "feature_cols": ["f"], "metrics": {}}}
    gate, _ = PG.submit_candidate("standard", reg.MODEL_FILES["standard"], None, results, df,
                                  "over25", "no_incumbent", df, now=T0)
    assert gate["verdict"] == PG.KEEP
    assert reg.MODEL_FILES["standard"].read_bytes() == old        # the champion, not the candidate


def test_qualified_challenger_in_slot_is_not_displaced(tmpreg, monkeypatch):
    (g1, e1), _ = _submit(tmpreg, monkeypatch, None, at=T0)
    assert g1["parked"]
    # a second qualified candidate a week later, same champion (predictions still stubbed)
    y, truth, d, lg = _data()
    df = pd.DataFrame({"date": d, "league": lg, "over25": y})
    results = {"lr": {"model": "second", "feature_cols": ["f"], "metrics": {}}}
    g2, e2 = PG.submit_candidate("standard", reg.MODEL_FILES["standard"], "OLD", results, df,
                                 "over25", "same_holdout", df, now=T0 + timedelta(days=7))
    assert g2["qualified"] and g2["verdict"] == PG.SHADOW and not g2["parked"]
    assert reg.load_challenger("standard")[1]["model_version"] == e1["challenger_version"]


def test_retrain_path_still_trains_and_saves_a_candidate(tmpreg, monkeypatch):
    """pipeline._train_one end to end with stubbed fitting: the retrain completes, the candidate
    gets a manifest + decision rows, retrain_log is written, and the champion does not change."""
    import pipeline as P
    import src.model as M
    old = b"serving-champion"
    PG._install_champion("standard", _manifest("standard", old), old, reason="seed")
    y, truth, d, lg = _data(n=5000)
    valid = pd.DataFrame({"date": d, "league": lg, "over25": y})
    champ = _noisy(truth, 0.06)

    def fake_predict(df, payload=None):
        idx = pd.Index(d).get_indexer(df["date"])         # split resets the index; dates are unique
        return champ[idx] if payload == "OLD" else truth[idx]
    monkeypatch.setattr(M, "predict_proba", fake_predict)
    monkeypatch.setattr(P, "model_predict_proba", fake_predict)
    monkeypatch.setattr(P, "load_models", lambda model_file=None: "OLD")
    monkeypatch.setattr(P, "get_feature_importances", lambda payload: pd.DataFrame())
    monkeypatch.setattr(P, "train_model", lambda v, **k: {
        "lr": {"model": "fitted", "feature_cols": ["f"], "metrics": {"log_loss": 0.69}}})
    monkeypatch.setattr(P, "TRAIN_LOG", tmpreg / "out" / "retrain_log.json")
    P._RETRAIN_BATCH["id"] = "rb_smoke"
    P._train_one(valid, "standard", reg.MODEL_FILES["standard"])
    assert reg.MODEL_FILES["standard"].read_bytes() == old
    entry = next(iter(json.loads((tmpreg / "out" / "retrain_log.json").read_text())["runs"]
                      .values()))["standard"]
    assert entry["promoted"] is False and entry["gate_version"] == PG.GATE_VERSION
    assert entry["verdict"] == PG.SHADOW and entry["qualified"] and entry["parked"]
    assert reg.read_manifest("standard", entry["challenger_version"]) is not None
    assert len(pd.read_csv(reg.RETRAIN_BATCHES)) == 1


# ── period 2: the prospective period ──────────────────────────────────────────────────────────
def _prospective(meta, champ_v, n_per_track, better=True, tracks=("standard",), seed=11,
                 start=T0 + timedelta(days=1)):
    """h2h first-sight rows logged after the freeze and before kickoff, plus their outcomes."""
    rng = np.random.default_rng(seed)
    h, o = [], []
    for tr in tracks:
        lgs = STD if tr == "standard" else NF
        for i in range(n_per_track):
            day = start + timedelta(days=i // 15)
            truth = rng.uniform(0.3, 0.7)
            goals = (3, 1) if rng.uniform() < truth else (1, 0)
            pc = float(np.clip(truth + rng.normal(0, 0.06), 0.02, 0.98))
            pn = truth if better else float(np.clip(truth + rng.normal(0, 0.12), 0.02, 0.98))
            home, away = f"H{tr}{i}", f"A{tr}{i}"
            md = (day + timedelta(days=2)).strftime("%Y-%m-%d")
            h.append({"logged_at": day.isoformat(), "match_date": md,
                      "kickoff_utc": f"{md} 15:00:00", "league": lgs[i % 2], "home_team": home,
                      "away_team": away, "model_type": "standard", "market": "over25",
                      "champion_version": champ_v, "p_served": pc, "p_champion_raw": pc,
                      "challenger_version": meta["model_version"], "p_challenger": pn})
            o.append({"match_date": md + " 00:00:00", "home_team": home, "away_team": away,
                      "home_goals": goals[0], "away_goals": goals[1]})
    return (pd.DataFrame(h).astype(str), pd.DataFrame(o).astype(str))


def test_no_prospective_period_means_no_promotion(tmpreg, monkeypatch):
    (gate, extra), old = _submit(tmpreg, monkeypatch, None)
    rec = PG.review_parked("standard", now=T0 + timedelta(days=7),
                           h2h=pd.DataFrame(), outcomes=pd.DataFrame())
    assert rec["verdict"] == PG.SHADOW and not rec["promoted"]
    assert reg.MODEL_FILES["standard"].read_bytes() == old
    # and an insufficient prospective sample is still not enough
    meta = reg.load_challenger("standard")[1]
    h, o = _prospective(meta, reg.read_pointer()["standard"]["model_version"], 200)
    rec2 = PG.review_parked("standard", now=T0 + timedelta(days=20), h2h=h, outcomes=o)
    assert rec2["verdict"] == PG.SHADOW and rec2["prospective_status"] == "PENDING"
    assert reg.MODEL_FILES["standard"].read_bytes() == old


def test_passing_all_criteria_promotes(tmpreg, monkeypatch):
    (gate, extra), old = _submit(tmpreg, monkeypatch, None)
    champ_v = reg.read_pointer()["standard"]["model_version"]
    meta = reg.load_challenger("standard")[1]
    h, o = _prospective(meta, champ_v, 1100)
    rec = PG.review_parked("standard", now=T0 + timedelta(days=90), h2h=h, outcomes=o,
                           batch="rb_review")
    assert rec["verdict"] == PG.PROMOTE and rec["promoted"], rec["reason"]
    ptr = reg.read_pointer()["standard"]
    assert ptr["model_version"] == extra["challenger_version"]
    assert ptr["gate_version"] == PG.GATE_VERSION
    raw = gzip.decompress(reg.champion_artifact_path("standard", ptr["model_version"]).read_bytes())
    assert reg.MODEL_FILES["standard"].read_bytes() == raw != old
    assert not reg.challenger_path("standard").exists()
    dec = pd.read_csv(PG.decisions_file())
    last = dec.iloc[-1]
    assert last["stage"] == "PROSPECTIVE_REVIEW" and last["verdict"] == PG.PROMOTE
    assert int(last["prospective_n"]) >= 1000


def test_prospective_wrong_rows_do_not_count(tmpreg, monkeypatch):
    """Rows logged after kickoff, before the freeze, or against a different champion are ignored."""
    (gate, extra), old = _submit(tmpreg, monkeypatch, None)
    champ_v = reg.read_pointer()["standard"]["model_version"]
    meta = reg.load_challenger("standard")[1]
    h, o = _prospective(meta, champ_v, 1100, start=T0 - timedelta(days=200))     # before freeze
    assert PG.prospective_rows("standard", meta["model_version"], champ_v,
                               meta["shadow_since"], h2h=h, outcomes=o).empty
    h2, o2 = _prospective(meta, "some_other_champion", 1100)
    assert PG.prospective_rows("standard", meta["model_version"], champ_v,
                               meta["shadow_since"], h2h=h2, outcomes=o2).empty
    h3, o3 = _prospective(meta, champ_v, 50)
    h3["kickoff_utc"] = h3["match_date"].str[:10] + " 00:00:00"
    h3["logged_at"] = (pd.to_datetime(h3["kickoff_utc"]) + pd.Timedelta(hours=1)).astype(str)
    assert PG.prospective_rows("standard", meta["model_version"], champ_v,
                               meta["shadow_since"], h2h=h3, outcomes=o3).empty


def test_prospective_failure_retires_the_challenger(tmpreg, monkeypatch):
    (gate, extra), old = _submit(tmpreg, monkeypatch, None)
    champ_v = reg.read_pointer()["standard"]["model_version"]
    meta = reg.load_challenger("standard")[1]
    h, o = _prospective(meta, champ_v, 1100, better=False)
    rec = PG.review_parked("standard", now=T0 + timedelta(days=90), h2h=h, outcomes=o)
    assert rec["verdict"] == PG.KEEP and not rec["promoted"]
    assert reg.MODEL_FILES["standard"].read_bytes() == old
    assert not reg.challenger_path("standard").exists()


def test_unqualified_shadow_can_never_promote(tmpreg, monkeypatch):
    old = b"old"
    PG._install_champion("standard", _manifest("standard", old), old, reason="seed")
    raw = pickle.dumps({"models": {}, "feature_cols": ["f"], "metrics": {}})
    m = {**_manifest("standard", raw, T0), "parent_champion_version":
         reg.read_pointer()["standard"]["model_version"],
         "gate": {"gate_version": PG.GATE_VERSION, "qualified": False,
                  "evaluable_tracks": ["standard"]}}
    reg.set_challenger("standard", m, raw)
    h, o = _prospective(reg.load_challenger("standard")[1],
                        reg.read_pointer()["standard"]["model_version"], 1100)
    rec = PG.review_parked("standard", now=T0 + timedelta(days=30), h2h=h, outcomes=o)
    assert rec["verdict"] != PG.PROMOTE and reg.MODEL_FILES["standard"].read_bytes() == old


def test_owner_override_names_an_exact_parked_version(tmpreg, monkeypatch):
    (gate, extra), old = _submit(tmpreg, monkeypatch, None)
    monkeypatch.setenv(PG.OWNER_OVERRIDE_ENV, "standard_20990101T000000Z_deadbeef")
    rec = PG.review_parked("standard", now=T0 + timedelta(days=1), h2h=pd.DataFrame(),
                           outcomes=pd.DataFrame())
    assert rec["verdict"] == PG.SHADOW                                # wrong version: no effect
    monkeypatch.setenv(PG.OWNER_OVERRIDE_ENV, extra["challenger_version"])
    rec = PG.review_parked("standard", now=T0 + timedelta(days=1), h2h=pd.DataFrame(),
                           outcomes=pd.DataFrame())
    assert rec["verdict"] == PG.PROMOTE and rec["stage"] == "OWNER_OVERRIDE"
    assert reg.read_pointer()["standard"]["promotion_source"] == "OWNER_OVERRIDE"


def test_rollback_goes_through_the_gate_and_is_logged(tmpreg):
    a, b = b"model-a", b"model-b"
    ma = _manifest("btts", a, datetime(2026, 10, 1, tzinfo=timezone.utc))
    PG._install_champion("btts", ma, a, reason="first")
    PG._install_champion("btts", _manifest("btts", b), b, reason="second")
    PG.rollback("btts", ma["model_version"], reason="test")
    assert reg.MODEL_FILES["btts"].read_bytes() == a
    assert pd.read_csv(PG.decisions_file()).iloc[-1]["verdict"] == PG.ROLLBACK


def test_save_models_refuses_serving_files():
    from src.model import save_models
    with pytest.raises(PermissionError):
        save_models({"lr": {"model": None, "feature_cols": [], "metrics": {}}},
                    model_file=config.MODEL_FILE_STANDARD)


# ── static: nothing else may change what serves ───────────────────────────────────────────────
def _py_files():
    skip = {"tests", "_archive", ".venv", "venv", ".claude", ".git", "node_modules"}
    for p in ROOT.rglob("*.py"):
        if not (set(p.relative_to(ROOT).parts) & skip):
            yield p


def test_only_the_gate_writes_the_pointer_or_a_serving_file():
    gate = (ROOT / "src" / "promotion_gate.py").resolve()
    registry = (ROOT / "src" / "model_registry.py").resolve()
    model_py = (ROOT / "src" / "model.py").resolve()
    write_ptr = re.compile(r"_atomic_write_\w+\(\s*(reg\.)?POINTER_FILE")
    write_serving = re.compile(r"_atomic_write_bytes\(\s*(reg\.)?MODEL_FILES\[")
    install = re.compile(r"(_install_champion|set_champion|set_challenger|clear_challenger)\(")
    open_serving = re.compile(r"open\(\s*config\.(HT_)?MODEL_FILE\w*\s*,\s*['\"]w")
    offenders = []
    for p in _py_files():
        rp = p.resolve()
        txt = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(txt.splitlines(), 1):
            code = line.split("#", 1)[0]
            where = f"{p.relative_to(ROOT)}:{i}"
            if write_ptr.search(code) and rp != gate:
                offenders.append(where + " writes the champion pointer")
            if write_serving.search(code) and rp != gate:
                offenders.append(where + " writes a serving model file")
            if open_serving.search(code):
                offenders.append(where + " opens a serving model file for writing")
            if install.search(code) and rp not in (gate, registry) \
                    and not code.lstrip().startswith("def "):
                offenders.append(where + " changes the champion/challenger outside the gate")
            if re.search(r"(?<!`)\bsave_models\(", code) and rp != model_py \
                    and not code.lstrip().startswith("def "):
                offenders.append(where + " calls save_models")
            if "current_champions.json" in code and rp != registry and re.search(
                    r"write_text|write_bytes|open\(.*['\"]w|json\.dump\(", code):
                offenders.append(where + " writes current_champions.json")
    assert not offenders, "\n".join(offenders)
    # the registry module itself must not write the pointer any more
    assert not write_ptr.search((ROOT / "src" / "model_registry.py").read_text(encoding="utf-8"))
    assert not hasattr(reg, "set_champion") and not hasattr(reg, "rollback")
    # and no workflow writes it by hand
    for y in (ROOT / ".github" / "workflows").glob("*.yml"):
        for line in y.read_text(encoding="utf-8").splitlines():
            if "current_champions.json" in line:
                assert not re.search(r">|cp |mv |tee ", line), f"{y.name}: {line}"


def test_only_one_definition_of_promoted_remains():
    assert not (ROOT / "src" / "champion_gate.py").exists()
    src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert "TRAIN_MAX_LOGLOSS_RISE" not in src and "_shadow_gate(" not in src
    assert "PG.review_parked(" in src and "PG.submit_candidate(" in src
    assert src.index("PG.review_parked(") < src.index("PG.submit_candidate(")
    rt = (ROOT / "retrain.py").read_text(encoding="utf-8")
    assert "PG.submit_candidate(" in rt and "save_models(" not in rt.replace("`save_models()`", "")
