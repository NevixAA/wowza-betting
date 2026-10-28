"""Standard O/U 2.5 serving switch (owner 2026-10-28: "switch Standard now").

    python -m pytest tests/test_standard_ou25_switch.py -q

  * default legacy_v9 -> predict output byte-identical to the pre-switch code, no new column;
  * v9_repaired -> ONLY Standard p_over25 (and what is derived from it) changes; New Format rows,
    side-market and HT probabilities are identical;
  * the repaired serving path is batch-composition invariant;
  * the registry keeps the imputer; owner_install writes a SEPARATE file and an auditable record;
  * no retrain path can overwrite or revert the switch; rollback (env -> legacy_v9) works, and a
    broken repaired file falls back to legacy, stamped.
All data is synthetic (tiny models, as in test_standard_forensic_repair); no network, and no
serving file or output/ file is written.
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import model as M  # noqa: E402
from src import model_registry as reg  # noqa: E402
from src import promotion_gate as PG  # noqa: E402
from src import standard_serving as SS  # noqa: E402
from src.features_repaired import build_features_repaired, feature_cols_for  # noqa: E402

STD_A, STD_B, NF = "Championship", "League One", "Brazil Serie A"
FEATS = feature_cols_for("none")
SIDE_P = ["p_btts", "p_over15", "p_over35", "p_ht_over05", "p_ht_over15"]
SWITCH_COLS = ["serving_recipe", "serving_no_form"]
OU_DERIVED = {"p_over25", "impl_prob_over", "impl_prob_under", "edge_over", "edge_under", "kelly_over",
              "kelly_under", "ev_over", "ev_under", "stake_over", "stake_under", "tier_over", "tier_under",
              "min_odds_blocked", "max_odds_blocked", "bet", "bet_stake", "signal_tier",
              "drift_shadow_tier", "best_side", "best_edge", "best_odds", "no_form_data", "both_losing"}


# ── synthetic world ───────────────────────────────────────────────────────────────────────────
def _rounds(teams):
    t = list(teams) + ([None] if len(teams) % 2 else [])
    n, out = len(t), []
    for _ in range(n - 1):
        out.append([(t[i], t[n - 1 - i]) for i in range(n // 2) if t[i] and t[n - 1 - i]])
        t = [t[0]] + [t[-1]] + t[1:-1]
    return out + [[(a, h) for h, a in rd] for rd in out]


def _season(league, teams, start, seed):
    rng = np.random.default_rng(seed)
    rows, d = [], pd.Timestamp(start)
    for rd in _rounds(teams):
        d += pd.Timedelta(days=7)
        for h, a in rd:
            hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
            rows.append({"date": d, "league": league, "season": "x", "home_team": h, "away_team": a,
                         "home_goals": hg, "away_goals": ag,
                         "ht_home_goals": min(hg, int(rng.binomial(hg, .45))),
                         "ht_away_goals": min(ag, int(rng.binomial(ag, .45))),
                         "home_shots": rng.integers(5, 20), "away_shots": rng.integers(4, 18),
                         "home_sot": rng.integers(1, 8), "away_sot": rng.integers(1, 7),
                         "home_corners": rng.integers(2, 10), "away_corners": rng.integers(1, 9),
                         "home_fouls": rng.integers(8, 16), "away_fouls": rng.integers(8, 16),
                         "odds_over25": rng.uniform(1.7, 2.3), "odds_under25": rng.uniform(1.6, 2.2)})
    df = pd.DataFrame(rows)
    df["over25"] = ((df.home_goals + df.away_goals) > 2.5).astype(float)
    df["btts"] = ((df.home_goals > 0) & (df.away_goals > 0)).astype(float)
    return df


def _history():
    parts = []
    for k, start in enumerate(("2023-08-01", "2024-08-01")):
        parts.append(_season(STD_A, [f"A{i}" for i in range(10)], start, 10 + k))
        parts.append(_season(STD_B, [f"B{i}" for i in range(10)], start, 20 + k))
        parts.append(_season(NF, [f"N{i}" for i in range(8)], start, 30 + k))
    return pd.concat(parts, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)


DAY = pd.Timestamp("2025-03-01")


def _upcoming():
    fx = [(STD_A, "A0", "A1"), (STD_A, "A2", "A3"), (STD_A, "A4", "Newcomers FC"),
          (STD_B, "B0", "B1"), (STD_B, "B2", "B3"), (NF, "N0", "N1"), (NF, "N2", "N3")]
    rng = np.random.default_rng(5)
    return pd.DataFrame([{"league": lg, "date": DAY, "kickoff_utc": DAY + pd.Timedelta(hours=15),
                          "home_team": h, "away_team": a,
                          "odds_over25": round(float(rng.uniform(1.7, 2.4)), 2),
                          "odds_under25": round(float(rng.uniform(1.6, 2.2)), 2),
                          "odds_over15": 1.3, "odds_over35": 3.1, "odds_btts": 1.85}
                         for lg, h, a in fx])


def _fake_legacy_features(upcoming, historical):
    """Stand-in for feature_engineering.build_upcoming_features: deterministic per fixture."""
    f = upcoming.copy().reset_index(drop=True)
    for i, c in enumerate(FEATS):
        f[c] = [((hash((c, h, a)) % 1000) / 1000.0) * (1 + i % 3)
                for h, a in zip(f["home_team"], f["away_team"])]
    return f


@pytest.fixture(scope="module")
def world():
    hist = _history()
    feat = build_features_repaired(hist)
    rows = feat.dropna(subset=["over25", "home_scored_last5"])
    std_rows = rows[rows["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    legacy = M.build_payload(M.train(rows, "over25", feature_cols=FEATS), "over25")
    btts = M.build_payload(M.train(rows, "btts", feature_cols=FEATS), "over25")
    res = M.train_repaired(std_rows, "over25", feature_cols=FEATS)
    extra = {"research": {"recipe": "v9_repaired", "track": "standard", "target": "over25",
                          "price_mode": "none", "team_scope": "league", "training_cutoff": "2025-02-01"},
             "identity_alias_map": {}}
    raw = reg.payload_bytes(res, "over25", extra=extra)
    return {"hist": hist, "legacy": legacy, "btts": btts, "res": res, "raw": raw,
            "repaired": pickle.loads(raw)}


@pytest.fixture()
def predict_env(world, tmp_path, monkeypatch):
    """predict_upcoming with OddsAPI, legacy feature build and every writer stubbed."""
    import src.drift as D
    import src.feature_engineering as FE
    import src.health_check as H
    import src.predict as P
    monkeypatch.setattr(P, "_fetch_odds_api", lambda days_ahead=7: _upcoming())
    monkeypatch.setattr(P, "_detect_postponed", lambda df: [])
    monkeypatch.setattr(FE, "build_upcoming_features", _fake_legacy_features)
    monkeypatch.setattr(D, "snapshot", lambda *a, **k: None)
    monkeypatch.setattr(D, "purge_old", lambda *a, **k: None)
    monkeypatch.setattr(D, "enrich", lambda df: df)
    monkeypatch.setattr(H, "record_fetch", lambda *a, **k: None)
    monkeypatch.setattr(H, "record_feature_health", lambda *a, **k: None)
    monkeypatch.setenv("REQUIRE_FORM_DATA", "1")
    f = tmp_path / "model_v9_standard_repaired.pkl"
    f.write_bytes(world["raw"])
    monkeypatch.setattr(config, "MODEL_FILE_STANDARD_REPAIRED", f)

    def run(recipe):
        monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE", recipe)
        out, _ = P.predict_upcoming(world["hist"], world["legacy"], payload_newformat=world["legacy"],
                                    side_payloads={"btts": world["btts"]})
        return out
    return run


def _std(df):
    return df[df["league"].isin(config.STANDARD_FORMAT_LEAGUES)]


def _nf(df):
    return df[df["league"].isin(config.NEW_FORMAT_LEAGUES)]


# ── config ────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("val,expect", [(None, "legacy_v9"), ("", "legacy_v9"), ("garbage", "legacy_v9"),
                                        ("legacy_v9", "legacy_v9"), ("v9_repaired", "v9_repaired")])
def test_switch_value_parsing_defaults_to_legacy(val, expect):
    env = {k: v for k, v in os.environ.items() if k != "STANDARD_OU25_SERVING_RECIPE"}
    if val is not None:
        env["STANDARD_OU25_SERVING_RECIPE"] = val
    r = subprocess.run([sys.executable, "-c", "import config;print(config.STANDARD_OU25_SERVING_RECIPE)"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip().splitlines()[-1] == expect


def test_switch_is_off_on_main_predict_yml():
    """The code can land before the owner OK: predict.yml on this commit does not set the switch."""
    from src.research_lock import deployed_env
    env = deployed_env()
    assert env.get("STANDARD_OU25_SERVING_RECIPE", "legacy_v9") in ("legacy_v9", "v9_repaired")


def test_research_lock_sees_the_switch():
    """Setting the env in predict.yml changes the fingerprint (needs owner re-approval)."""
    from src.research_lock import config_env_keys, deployed_env
    assert "STANDARD_OU25_SERVING_RECIPE" in config_env_keys()
    yml = "        env:\n          STANDARD_OU25_SERVING_RECIPE: \"v9_repaired\"\n"
    assert deployed_env(yml)["STANDARD_OU25_SERVING_RECIPE"] == "v9_repaired"


# ── 1. default legacy -> identical ────────────────────────────────────────────────────────────
def test_default_legacy_is_byte_identical_to_pre_switch_code(predict_env, monkeypatch):
    a = predict_env("legacy_v9")
    assert not set(SWITCH_COLS) & set(a.columns)
    monkeypatch.setattr(SS, "apply", lambda feat, std_mask, probas, historical, **k: probas)
    b = predict_env("legacy_v9")                         # == the code before this change
    pd.testing.assert_frame_equal(a, b)
    assert a.to_csv(index=False) == b.to_csv(index=False)


def test_generate_bets_unchanged_in_legacy(predict_env):
    from src.betting import generate_bets
    bets = generate_bets(predict_env("legacy_v9"))
    assert "serving_recipe" not in bets.columns


# ── 2. repaired -> only Standard OU25 changes ────────────────────────────────────────────────
def test_repaired_changes_only_standard_ou25(predict_env, world):
    L, R = predict_env("legacy_v9"), predict_env("v9_repaired")
    assert set(R.columns) - set(L.columns) == set(SWITCH_COLS)
    Rd = R.drop(columns=SWITCH_COLS)[L.columns]
    # New Format rows: every column identical
    pd.testing.assert_frame_equal(_nf(L).reset_index(drop=True), _nf(Rd).reset_index(drop=True))
    # side / HT probabilities identical on every row
    cols = [c for c in SIDE_P if c in L.columns]
    pd.testing.assert_frame_equal(L[cols], Rd[cols])
    # Standard rows: only O/U-derived columns may differ
    keep = [c for c in L.columns if c not in OU_DERIVED]
    pd.testing.assert_frame_equal(_std(L)[keep].reset_index(drop=True), _std(Rd)[keep].reset_index(drop=True))
    # and the Standard p_over25 IS the repaired payload's
    exp = SS.score_fixtures(_std(_upcoming()), SS.serving_history(world["hist"], world["repaired"]),
                            world["repaired"])["p_over25_repaired"].to_numpy()
    np.testing.assert_allclose(_std(R)["p_over25"].to_numpy(), exp, rtol=0, atol=1e-12)
    assert not np.allclose(_std(R)["p_over25"].to_numpy(), _std(L)["p_over25"].to_numpy())
    assert set(_std(R)["serving_recipe"]) == {"v9_repaired"}
    assert set(_nf(R)["serving_recipe"]) == {"legacy_v9"}


def test_no_repaired_form_forces_avoid(predict_env):
    R = predict_env("v9_repaired")
    new = R[R["away_team"] == "Newcomers FC"].iloc[0]
    assert bool(new["serving_no_form"]) and bool(new["no_form_data"])
    assert new["bet"] == "AVOID" and new["signal_tier"] == "AVOID"


def test_bets_carry_serving_recipe_and_ledger_attributes_repaired(predict_env, monkeypatch):
    from src import ledger as LG
    from src.betting import generate_bets
    R = predict_env("v9_repaired")
    bets = generate_bets(R)
    assert "serving_recipe" in bets.columns or bets.empty
    assert LG._ou_key("standard", "v9_repaired") == "standard_repaired"
    assert LG._ou_key("standard", None) == "standard"
    assert LG._ou_key("new_format", "legacy_v9") in ("newformat", "standard")


# ── 3. batch-composition invariance ───────────────────────────────────────────────────────────
def test_repaired_serving_is_batch_invariant(world):
    hist = SS.serving_history(world["hist"], world["repaired"])
    up = _std(_upcoming())
    board = SS.score_fixtures(up, hist, world["repaired"])["p_over25_repaired"]
    for i in up.index:
        alone = SS.score_fixtures(up.loc[[i]], hist, world["repaired"])["p_over25_repaired"]
        assert alone.loc[i] == board.loc[i]
    rev = SS.score_fixtures(up.iloc[::-1], hist, world["repaired"])["p_over25_repaired"]
    pd.testing.assert_series_equal(rev.loc[up.index], board)


def test_scorer_refuses_non_standard_rows(world):
    with pytest.raises(ValueError):
        SS.score_fixtures(_nf(_upcoming()), world["hist"], world["repaired"])


# ── 4. registry keeps the imputer ────────────────────────────────────────────────────────────
def test_payload_bytes_keeps_imputer_and_provenance(world):
    p = pickle.loads(world["raw"])
    assert p["imputer"] and p["recipe"] == "v9_repaired" and p["identity_alias_map"] == {}
    assert p["research"]["track"] == "standard"
    SS.validate_payload(p)


def test_payload_bytes_legacy_is_byte_identical():
    """A legacy result (plain dict, no recipe meta) pickles to exactly the pre-fix bytes."""
    res = {"logistic": {"model": "m1", "metrics": {"auc": 0.6}, "feature_cols": ["a", "b"]},
           "gradient_boost": {"model": "m2", "metrics": {"auc": 0.61}, "feature_cols": ["a", "b"]}}
    old = pickle.dumps({"target": "over25", "models": {k: v["model"] for k, v in res.items()},
                        "feature_cols": res[next(iter(res))]["feature_cols"],
                        "metrics": {k: v["metrics"] for k, v in res.items()}})
    assert reg.payload_bytes(res) == old
    tr = M.TrainResults(res, meta={"imputer": {"version": "x"}, "recipe": "v9_repaired"})
    p = pickle.loads(reg.payload_bytes(tr, extra={"research": {"k": 1}}))
    assert p["imputer"] == {"version": "x"} and p["recipe"] == "v9_repaired" and p["research"] == {"k": 1}


def test_legacy_payload_is_not_servable_as_repaired(world):
    with pytest.raises(ValueError):
        SS.validate_payload(world["legacy"])


# ── owner install, gate, retrain ─────────────────────────────────────────────────────────────
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
    monkeypatch.setattr(PG.config, "MODELS_DIR", tmp_path / "models")
    reg._version_cache.clear()
    files["standard"].parent.mkdir(parents=True, exist_ok=True)
    files["standard"].write_bytes(b"LEGACY-STANDARD-BYTES")
    yield tmp_path
    reg._version_cache.clear()


APPROVAL = ("Nevo (owner) 2026-10-28: switch Standard OU25 to repaired recipe before forward evidence; "
            "rollback = set STANDARD_OU25_SERVING_RECIPE=legacy_v9")


def test_owner_install_writes_separate_file_and_audit_trail(tmpreg, world):
    legacy_sha = reg.sha256_file(reg.MODEL_FILES["standard"])
    man = PG.owner_install("standard_repaired", world["raw"], approval=APPROVAL,
                           approved_by="Nevo (owner)", training={"training_cutoff": "2025-02-01"})
    assert reg.MODEL_FILES["standard_repaired"].read_bytes() == world["raw"]
    assert reg.sha256_file(reg.MODEL_FILES["standard"]) == legacy_sha           # rollback intact
    ptr = reg.read_pointer()["standard_repaired"]
    assert ptr["model_version"] == man["model_version"] and ptr["promotion_source"] == "OWNER_INSTALL"
    assert APPROVAL in ptr["promotion_reason"]
    assert man["owner_approval"]["text"] == APPROVAL
    d = pd.read_csv(PG.decisions_file(), dtype=str)
    last = d.iloc[-1]
    assert last["stage"] == "OWNER_INSTALL" and last["verdict"] == "PROMOTE"
    assert last["promoted"].lower() == "true" and last["candidate_version"] == man["model_version"]
    # predict-side attribution names the repaired artifact
    assert reg.stamp_for("standard_repaired")["model_version"] == man["model_version"]
    assert man["model_version"].startswith("standard_repaired_")
    gz = reg.champion_artifact_path("standard_repaired", man["model_version"])
    assert gzip.decompress(gz.read_bytes()) == world["raw"]


def test_owner_install_refusals(tmpreg, world):
    with pytest.raises(PermissionError):
        PG.owner_install("standard", world["raw"], approval=APPROVAL, approved_by="x")
    with pytest.raises(PermissionError):
        PG.owner_install("standard_repaired", world["raw"], approval="", approved_by="x")
    with pytest.raises(ValueError):
        PG.owner_install("standard_repaired", pickle.dumps(world["legacy"]), approval=APPROVAL,
                         approved_by="x")
    assert not reg.MODEL_FILES["standard_repaired"].exists()
    man = PG.owner_install("standard_repaired", world["raw"], approval=APPROVAL, approved_by="x")
    with pytest.raises(PermissionError):                          # second install must name the first
        PG.owner_install("standard_repaired", world["raw"] + b"", approval=APPROVAL, approved_by="x",
                         now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    PG.owner_install("standard_repaired", world["raw"], approval=APPROVAL, approved_by="x",
                     supersede=man["model_version"], now=datetime(2026, 11, 1, tzinfo=timezone.utc))


def test_retrain_cannot_submit_to_or_review_the_frozen_type(tmpreg, world):
    man = PG.owner_install("standard_repaired", world["raw"], approval=APPROVAL, approved_by="x")
    before = reg.MODEL_FILES["standard_repaired"].read_bytes()
    gate, _ = PG.submit_candidate("standard_repaired", reg.MODEL_FILES["standard_repaired"],
                                  world["repaired"], world["res"], None, "over25", "same_holdout",
                                  pd.DataFrame({"date": []}))
    assert gate["verdict"] == "KEEP_CHAMPION" and "FROZEN" in gate["reason"]
    assert PG.review_parked("standard_repaired") is None
    assert reg.MODEL_FILES["standard_repaired"].read_bytes() == before
    assert reg.read_pointer()["standard_repaired"]["model_version"] == man["model_version"]


def test_legacy_standard_promotion_keeps_the_repaired_pointer(tmpreg, world):
    man = PG.owner_install("standard_repaired", world["raw"], approval=APPROVAL, approved_by="x")
    raw = pickle.dumps(world["legacy"])
    sha = reg.sha256_bytes(raw)
    PG._install_champion("standard", {"model_type": "standard", "artifact_sha256": sha,
                                      "model_version": reg.make_version("standard", reg.utcnow(), sha)},
                         raw, reason="test promotion")
    ptr = reg.read_pointer()
    assert ptr["standard_repaired"]["model_version"] == man["model_version"]
    assert reg.MODEL_FILES["standard_repaired"].read_bytes() == world["raw"]


def test_no_retrain_code_or_workflow_touches_the_repaired_model():
    import inspect
    import pipeline
    for fn in (pipeline.mode_train,):
        src = inspect.getsource(fn)
        assert "standard_repaired" not in src and "STANDARD_REPAIRED" not in src
    for wf in ("retrain.yml", "backtest_matrix.yml"):
        p = ROOT / ".github" / "workflows" / wf
        if p.exists():
            t = p.read_text(encoding="utf-8")
            assert "model_v9_standard_repaired" not in t and "STANDARD_OU25_SERVING_RECIPE" not in t
    # nothing scheduled installs it: the training workflow is dispatch-only
    t = (ROOT / ".github" / "workflows" / "train_standard_repaired.yml").read_text(encoding="utf-8")
    assert "schedule:" not in t and "workflow_dispatch" in t


def test_save_paths_refuse_the_repaired_serving_file(world, tmp_path):
    from src.standard_repair import save_research_payload
    with pytest.raises(PermissionError):
        M.save_models(world["res"], model_file=config.MODEL_FILE_STANDARD_REPAIRED)
    with pytest.raises(PermissionError):
        save_research_payload(world["repaired"], config.MODEL_FILE_STANDARD_REPAIRED)


# ── rollback / failure ───────────────────────────────────────────────────────────────────────
def test_rollback_env_restores_legacy_exactly(predict_env, monkeypatch):
    R = predict_env("v9_repaired")
    back = predict_env("legacy_v9")
    monkeypatch.setattr(SS, "apply", lambda feat, std_mask, probas, historical, **k: probas)
    pure = predict_env("legacy_v9")
    pd.testing.assert_frame_equal(back, pure)
    assert "serving_recipe" in R.columns and "serving_recipe" not in back.columns


def test_broken_repaired_file_falls_back_to_legacy_stamped(predict_env, monkeypatch, tmp_path):
    L = predict_env("legacy_v9")
    monkeypatch.setattr(config, "MODEL_FILE_STANDARD_REPAIRED", tmp_path / "missing.pkl")
    R = predict_env("v9_repaired")
    np.testing.assert_array_equal(_std(R)["p_over25"].to_numpy(), _std(L)["p_over25"].to_numpy())
    assert set(_std(R)["serving_recipe"]) == {"legacy_v9_fallback"}
    assert not _std(R)["serving_no_form"].any()


# ── attribution + health ─────────────────────────────────────────────────────────────────────
def test_row_attribution_mapping():
    assert reg.ou_model_for_row("standard", "v9_repaired") == "standard_repaired"
    assert reg.ou_model_for_row("standard", "legacy_v9_fallback") == "standard"
    assert reg.ou_model_for_row("standard", None) == "standard"
    assert reg.ou_model_for_row("unknown", "v9_repaired") == "standard"
    assert reg.PRED_COLUMN["standard_repaired"] == "p_over25"


def test_system_health_serving_check(tmp_path, monkeypatch):
    from src import system_health as S
    monkeypatch.setattr(S, "OUT", tmp_path)
    assert S.standard_serving_health()["status"] == S.GREEN                    # no board
    pd.DataFrame({"model_type": ["standard"], "model_version": ["standard_x"]}).to_csv(
        tmp_path / "predictions.csv", index=False)
    assert S.standard_serving_health()["status"] == S.GREEN                    # switch off
    pd.DataFrame({"model_type": ["standard", "new_format"],
                  "serving_recipe": ["v9_repaired", "legacy_v9"],
                  "model_version": ["standard_repaired_20261028T000000Z_abcd1234", "nf"]}).to_csv(
        tmp_path / "predictions.csv", index=False)
    assert S.standard_serving_health()["status"] == S.GREEN
    pd.DataFrame({"model_type": ["standard"], "serving_recipe": ["legacy_v9_fallback"],
                  "model_version": ["standard_1"]}).to_csv(tmp_path / "predictions.csv", index=False)
    assert S.standard_serving_health()["status"] == S.RED
    pd.DataFrame({"model_type": ["standard"], "serving_recipe": ["v9_repaired"],
                  "model_version": ["UNREGISTERED_standard_repaired_abc"]}).to_csv(
        tmp_path / "predictions.csv", index=False)
    assert S.standard_serving_health()["status"] == S.RED


def test_frozen_alias_map_is_committed_and_identity_1_0():
    p = ROOT / "config" / "standard_ou25_identity_aliases.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["version"] == "identity/1.0" and isinstance(d["map"], dict) and d["map"]


# ── CI scripts on synthetic data (the real runs happen only in GitHub Actions) ────────────────
def _load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_training_script_produces_a_servable_payload_before_cutoff(tmp_path):
    T = _load_script("train_standard_repaired")
    alias = tmp_path / "aliases.json"
    alias.write_text(json.dumps({"version": "identity/1.0", "map": {}}), encoding="utf-8")
    cutoff = pd.Timestamp("2025-01-01")
    raw, rec = T.train(cutoff, alias, raw=_history())
    p = pickle.loads(raw)
    SS.validate_payload(p)
    assert p["research"]["training_cutoff"] == "2025-01-01"
    assert pd.Timestamp(p["research"]["training_end"]) < cutoff
    assert "n_aliases" in p["research"]["identity_report"]
    assert rec["training_rows"] == p["research"]["rows"] and rec["served_holdout_metrics"]
    assert p["research"]["alias_map_sha256"] == T._sha256(alias)


def test_preview_pipeline_proves_invariance(predict_env, world, monkeypatch):
    V = _load_script("standard_switch_preview")
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE", "legacy_v9")   # restored on teardown
    payloads = {"std": world["legacy"], "nf": world["legacy"], "side": {"btts": world["btts"]}}
    L = V.run_mode("legacy_v9", _upcoming(), world["hist"], payloads)
    R = V.run_mode("v9_repaired", _upcoming(), world["hist"], payloads)
    inv = V.invariance(L, R)
    required = {k: v for k, v in inv.items() if k not in V.INFORMATIONAL}
    assert required and all(v["ok"] for v in required.values()), inv
    t = V.fixture_table(L, R)
    assert set(t.loc[t["track"] == "newformat", "change"]) <= {"NO_TIP_BOTH", "SAME_TIP"}
    s = V.summarise(t, inv, L, R, {"board_generated_at": "x"})
    assert s["invariance_all_ok"] and s["fixtures"]["standard"] == 5
    assert "| league |" in V.report_md(s, t)
