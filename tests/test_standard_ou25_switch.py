"""Standard O/U 2.5 serving switch (owner 2026-10-28: "switch Standard now"; v9_repaired RETIRED 2026-11-15).

RETIREMENT (owner 2026-11-15, "fix what you need"): v9_repaired / models/model_v9_standard_repaired.pkl
filled xG and inside-box shots with 0.0 on every training row. STANDARD_OU25_SERVING_RECIPE accepts
only legacy_v9 (v9_repaired is refused like a typo), owner_install refuses the retired type, the
registry pointer and the decision log record the retirement, the file stays for audit. The tests of
the retired serving path became tests of its refusal; the scorer-level tests (score_fixtures, kept
for the audit scripts) stay. History of the switch:

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
OU_DERIVED = {"p_over25", "p_over25_raw", "impl_prob_over", "impl_prob_under", "edge_over", "edge_under", "kelly_over",
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


def _fake_legacy_features(upcoming, historical, **_kw):    # predict passes identity_aliases (fix12)
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
    # Real-data serving guard (fix/real-data-provenance 2026-11-15): live scoring refuses a payload
    # without a persisted train-only imputer and a 1.0 imputer that fills a feature with no training
    # values (xG / inside-box are absent from this synthetic history), so the served stand-ins carry
    # an imputer and only the features the synthetic history holds. The refusal itself is tested in
    # tests/test_real_data_pipeline_trace.py.
    feats = [c for c in FEATS if pd.to_numeric(std_rows[c], errors="coerce").notna().any()]
    legacy = M.build_payload(M.train_repaired(rows, "over25", feature_cols=feats), "over25")
    btts = M.build_payload(M.train_repaired(rows, "btts", feature_cols=feats), "over25")
    res = M.train_repaired(std_rows, "over25", feature_cols=feats)
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
# R2-A (2026-11-01, finding 17): an invalid value is no longer laundered into legacy_v9. config keeps
# loading (dashboard / notifier / results must not die) but marks it INVALID; predict refuses to start
# (see test_invalid_recipe_is_a_configuration_error).
@pytest.mark.parametrize("val,expect", [(None, "legacy_v9"), ("", "legacy_v9"), ("garbage", "INVALID"),
                                        ("v9_repaird", "INVALID"),
                                        ("legacy_v9", "legacy_v9"), ("v9_repaired", "INVALID")])
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
    assert env.get("STANDARD_OU25_SERVING_RECIPE", "legacy_v9") == "legacy_v9"     # v9_repaired RETIRED


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


# ── 2. RETIRED (owner 2026-11-15): v9_repaired is refused, never served ──────────────────────
def test_retired_recipe_is_refused_before_anything_is_served(predict_env):
    with pytest.raises(SS.ServingConfigError, match="RETIRED"):
        predict_env("v9_repaired")


def test_retired_recipe_in_the_env_is_refused(monkeypatch):
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE_RAW", "v9_repaired")
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE", "INVALID")
    with pytest.raises(SS.ServingConfigError, match="RETIRED"):
        SS.check_config()
    assert config.STANDARD_OU25_SERVING_RECIPES == ("legacy_v9",)
    r = config.RETIRED_STANDARD_OU25_SERVING_RECIPES["v9_repaired"]
    assert r["retired_on"] == "2026-11-15" and r["owner_words"] == "fix what you need"


def test_apply_never_rescores_standard_rows(world):
    std = pd.Series([True, False])
    probas = np.array([0.4, 0.6])
    feat = pd.DataFrame({"x": [1, 2]})
    out = SS.apply(feat, std, probas, world["hist"], payload=world["repaired"])
    assert out is probas and list(feat.columns) == ["x"]
    assert SS.last_health()["status"] == "PASS" and SS.last_health()["served_by_repaired"] == 0


def test_retired_file_is_kept_for_audit_and_the_registry_says_retired():
    assert config.MODEL_FILE_STANDARD_REPAIRED.exists()                  # kept, never deleted
    ent = reg.read_pointer()["standard_repaired"]
    assert ent["status"] == "RETIRED" and ent["retired_on"] == "2026-11-15"
    assert ent["retirement_owner_words"] == "fix what you need"
    assert ent["artifact_sha256"] == reg.sha256_file(config.MODEL_FILE_STANDARD_REPAIRED)
    d = pd.read_csv(PG.decisions_file(), dtype=str, keep_default_na=False)
    r = d[(d["stage"] == "RETIRE") & (d["model_type"] == "standard_repaired")]
    assert len(r) == 1 and r.iloc[0]["verdict"] == "RETIRED" and "fix what you need" in r.iloc[0]["reason"]


def test_ledger_still_attributes_old_repaired_rows():
    from src import ledger as LG
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
        # a no-form fixture has NO probability (serving guard), alone or on the board
        assert alone.loc[i] == board.loc[i] or (np.isnan(alone.loc[i]) and np.isnan(board.loc[i]))
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


def test_owner_install_refuses_the_retired_type(tmpreg, world):
    with pytest.raises(PermissionError, match="RETIRED"):
        PG.owner_install("standard_repaired", world["raw"], approval=APPROVAL, approved_by="Nevo (owner)")
    assert not reg.MODEL_FILES["standard_repaired"].exists()
    with pytest.raises(PermissionError):
        PG.owner_install("standard", world["raw"], approval=APPROVAL, approved_by="x")


def test_retire_model_type_is_logged_once_and_keeps_the_bytes(tmpreg, world):
    reg.MODEL_FILES["standard_repaired"].write_bytes(world["raw"])
    ptr = reg.read_pointer()
    ptr["standard_repaired"] = {"model_version": "standard_repaired_x", "artifact_sha256": "abc"}
    reg._atomic_write_json(reg.POINTER_FILE, ptr)
    assert PG.retire_model_type("standard_repaired") is True
    assert PG.retire_model_type("standard_repaired") is False             # idempotent
    ent = reg.read_pointer()["standard_repaired"]
    assert ent["status"] == "RETIRED" and ent["model_version"] == "standard_repaired_x"
    assert reg.MODEL_FILES["standard_repaired"].read_bytes() == world["raw"]
    d = pd.read_csv(PG.decisions_file(), dtype=str, keep_default_na=False)
    assert (d["stage"] == "RETIRE").sum() == 1
    with pytest.raises(PermissionError):
        PG.retire_model_type("standard")


def test_retrain_cannot_submit_to_or_review_the_frozen_type(tmpreg, world):
    gate, _ = PG.submit_candidate("standard_repaired", reg.MODEL_FILES["standard_repaired"],
                                  world["repaired"], world["res"], None, "over25", "same_holdout",
                                  pd.DataFrame({"date": []}))
    assert gate["verdict"] == "KEEP_CHAMPION" and "FROZEN" in gate["reason"]
    assert PG.review_parked("standard_repaired") is None
    assert not reg.MODEL_FILES["standard_repaired"].exists()


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


# ── no rollback recipe any more ──────────────────────────────────────────────────────────────
def test_the_repaired_training_workflow_refuses_to_run():
    t = (ROOT / ".github" / "workflows" / "train_standard_repaired.yml").read_text(encoding="utf-8")
    assert t.count("RETIRED - this workflow no longer trains or installs anything") == 2   # both jobs
    assert "schedule:" not in t


def test_invalid_recipe_is_a_configuration_error(predict_env, monkeypatch):
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE_RAW", "v9_repaird")
    with pytest.raises(SS.ServingConfigError):
        predict_env("legacy_v9")
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE_RAW", "v9_repaired")
    with pytest.raises(SS.ServingConfigError):
        predict_env("INVALID")                          # what config.py stores for a bad env value


def test_invalid_recipe_raises_before_odds_are_fetched(monkeypatch):
    from src import predict as P
    calls = []
    monkeypatch.setattr(P, "_fetch_odds_api", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE_RAW", "garbage")
    with pytest.raises(SS.ServingConfigError):
        P.predict_upcoming(pd.DataFrame(), {}, payload_newformat=None)
    assert calls == []


# ── attribution + health ─────────────────────────────────────────────────────────────────────
def test_row_attribution_mapping():
    assert reg.ou_model_for_row("standard", "v9_repaired") == "standard_repaired"
    assert reg.ou_model_for_row("standard", "legacy_v9_fallback") == "standard"     # pre-R2 boards
    assert reg.ou_model_for_row("standard", "v9_repaired_failed") == reg.NO_MODEL   # fail closed
    assert reg.ou_model_for_row("standard", None) == "standard"
    assert reg.ou_model_for_row("unknown", "v9_repaired") == reg.NO_MODEL           # R2-A: no fallback
    assert reg.PRED_COLUMN["standard_repaired"] == "p_over25"


def test_system_health_serving_check(tmp_path, monkeypatch):
    from src import system_health as S
    monkeypatch.setattr(S, "OUT", tmp_path)
    # no board -> no evidence: a missing input is never GREEN (CLAUDE_PROMPT_2, 2026-11-02)
    assert S.standard_serving_health()["status"] == S.GREY
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
    pd.DataFrame({"model_type": ["standard"], "serving_recipe": ["v9_repaired_failed"],
                  "model_version": ["NONE_FAIL_CLOSED"]}).to_csv(tmp_path / "predictions.csv", index=False)
    assert S.standard_serving_health()["status"] == S.RED                      # failed closed
    pd.DataFrame({"model_type": ["standard"], "serving_recipe": ["v9_repaired"],
                  "model_version": ["UNREGISTERED_standard_repaired_abc"]}).to_csv(
        tmp_path / "predictions.csv", index=False)
    assert S.standard_serving_health()["status"] == S.RED


def test_frozen_alias_map_is_committed_and_identity_1_1():
    # identity/1.1 (2026-11-02): + Ireland Bray Wanderers -> Bray (owner-accepted evidence merge)
    p = ROOT / "config" / "standard_ou25_identity_aliases.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["version"] == "identity/1.1" and isinstance(d["map"], dict) and d["map"]


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


def test_preview_of_the_retired_recipe_is_refused(predict_env, world, monkeypatch):
    """scripts/standard_switch_preview compared legacy_v9 with v9_repaired; the retired recipe no
    longer scores anything, so its preview mode stops at the configuration check."""
    V = _load_script("standard_switch_preview")
    monkeypatch.setattr(config, "STANDARD_OU25_SERVING_RECIPE", "legacy_v9")   # restored on teardown
    payloads = {"std": world["legacy"], "nf": world["legacy"], "side": {"btts": world["btts"]}}
    L = V.run_mode("legacy_v9", _upcoming(), world["hist"], payloads)
    assert len(L["preds"]) and "serving_recipe" not in L["preds"].columns
    with pytest.raises(SS.ServingConfigError):
        V.run_mode("v9_repaired", _upcoming(), world["hist"], payloads)
