"""Props production certification (owner P0 deep-repair list 2026-11-22, item 5).

    python -m src.integrity.props_certification                       # print the verdict
    python -m src.integrity.props_certification --write               # + output/props_real_data_provenance.json
    python -m src.integrity.props_certification --extract-run DIR     # write the run evidence file from a
                                                                      # downloaded props-clean-retrain artifact
    python -m src.integrity.props_certification --write --run-artifact DIR [--holdout-frame F.parquet]
                                                                      # + offline identity / holdout replay

WHAT IT CERTIFIES. Every INSTALLED props artifact, read from the real serialized bytes:
  production   the champion of each market props tips are made on (pointer models/registry/props/
               props_champions.json, serving file config.MODEL_FILES, market not hidden)
  rare_served  the champion of each hidden rare market (config.HIDDEN_PENDING_CLEAN_RETRAIN): served as
               player_base_rate since 2026-11-18 for the rare research picks, never a normal tip
  research     the parked challenger of a market (models/registry/props/<m>/challenger.pkl.gz): scored
               first-sight beside the champion, never a tip
Per artifact: exact sha256 vs the registry, model version, recipe, raw input provenance (owner feature
rule + verified-share floors, the run that trained it and that run's own artifact sha256), feature
schema (no withheld / proxy / removed feature, schema hash = the run's), training window and cutoff,
historical coverage, chronological validation, missing-data handling, train-fitted preprocessing (no
batch statistic: a RUNTIME batch-invariance probe), player identity, train/serve recipe parity and
production vs research permission. Plus the global evidence: the legacy batch-median `_prep` is not
reachable in production (call graph + runtime probe), point-in-time / identity tests, the committed
serving-vs-training parity report of the run.

VERDICT. VERIFIED only when every check of every artifact and every global check passes; otherwise
FAILED with the exact failing checks. src.integrity.real_data_provenance.props_status reads the file and
re-hashes the installed artifacts: a file whose recorded hashes no longer match is STALE, never
VERIFIED. No price is read or written here.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import pickle
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "props_real_data_provenance.json"
REG = ROOT / "models" / "registry" / "props"
POINTER = REG / "props_champions.json"
EVIDENCE_DIR = ROOT / "reports" / "p0_props_cert"
PARITY_REPORT = ROOT / "reports" / "props_feature_bugs_parity_2026-11-15_run43035218594_sample6316.json"
VERSION = "PROPS-CERT-1.0"
VERIFIED, FAILED = "VERIFIED", "FAILED"
PRODUCTION, RARE, RESEARCH = "production", "rare_served", "research"
HARNESS_VERSION = "PROPS-HARNESS-1.0"
RECIPE = "props_clean_harness"
#: candidate -> the imputer record clean_serving_payload writes (props_gate.clean_serving_payload)
IMPUTER = {"hgb_nan_native": "NONE_NAN_NATIVE", "recipe_retrained": "FIT_BLOCK_MEDIAN_PERSISTED",
           "player_base_rate": "NONE_PLAYER_ID_RATE"}
#: owner feature rule and floors (owner 2026-11-08); the harness refuses to fit below them
FEATURE_RULE, MIN_TRAIN_SHARE, MIN_HOLDOUT_SHARE = "all_prior_verified", 0.95, 0.85
#: batch-invariance probe sizes (owner item 5: alone, with 10 others, with 1,000)
BATCH_SIZES = (10, 1000)
FLOAT_TOL = 1e-12                        # BLAS dot-product rounding, never a statistic
SINGLE_VALUE_MAX_SHARE = 0.999           # = props_harness.SINGLE_VALUE_MAX_SHARE / data_truth rule 3
#: tests whose presence the global evidence names (file, function); each runs in the suite
PIT_TESTS = (("tests/test_props_train_serve_parity_2026_11_09.py", "test_serving_rows_equal_the_training_rows_exactly"),
             ("tests/test_props_train_serve_parity_2026_11_09.py", "test_position_is_pre_match_evidence_only"),
             ("tests/test_props_train_serve_parity_2026_11_09.py", "test_penalty_prior_does_not_read_the_rows_own_count"),
             ("tests/test_props_train_serve_parity_2026_11_09.py", "test_same_day_rows_have_one_order"),
             ("tests/test_props_data_truth.py", "test_serving_uses_the_training_definitions"),
             ("tests/test_props_real_profiles.py", None))
IDENTITY_TESTS = (("tests/test_props_truth.py", "test_stable_player_id_rejects_zero"),
                  ("tests/test_props_truth.py", "test_fixture_side_by_team_id_then_identity"),
                  ("tests/test_props_truth.py", "test_club_conflict_tip_time"),
                  ("tests/test_props_train_serve_parity_2026_11_09.py", "test_identity_conflict_needs_both_conditions"))


# ── loading ──────────────────────────────────────────────────────────────────────────────────────
class _Unpickler(pickle.Unpickler):
    """Harness classes pickled under __main__ resolve to player_model.props_harness (as data_truth)."""
    def find_class(self, module, name):
        return super().find_class("player_model.props_harness" if module == "__main__" else module, name)


def read_bytes(path: Path) -> bytes:
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if str(path).endswith(".gz") else raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_raw(raw: bytes) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _Unpickler(io.BytesIO(raw)).load()


def pickled_sklearn_versions(raw: bytes) -> list[str]:
    """The scikit-learn version(s) recorded in the pickle (every estimator stores _sklearn_version)."""
    return sorted({v.decode() for v in re.findall(rb"_sklearn_version\x94\x8c\x05([0-9]+\.[0-9]+\.[0-9]+)", raw)}
                  | {v.decode() for v in re.findall(rb"_sklearn_version.{1,6}?([0-9]+\.[0-9]+\.[0-9]+)", raw)})


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def installed_artifacts() -> list[dict]:
    """[{key, market, role, version, path, manifest}] - every installed props artifact."""
    from player_model import config as pc
    ptr = _json(POINTER) or {}
    out = []
    for market, path in pc.MODEL_FILES.items():
        ent = ptr.get(market) or {}
        v = ent.get("model_version")
        out.append({"key": f"{market}:champion", "market": market,
                    "role": RARE if market in pc.HIDDEN_PENDING_CLEAN_RETRAIN else PRODUCTION,
                    "version": v, "path": Path(path), "pointer": ent,
                    "manifest": _json(REG / market / "versions" / f"{v}.json") if v else None})
    for meta_p in sorted(REG.glob("*/challenger.json")):
        meta = _json(meta_p) or {}
        m, v = meta_p.parent.name, meta.get("model_version")
        out.append({"key": f"{m}:challenger", "market": m, "role": RESEARCH, "version": v,
                    "path": meta_p.parent / "challenger.pkl.gz", "pointer": meta,
                    "manifest": _json(REG / m / "versions" / f"{v}.json") if v else None})
    return out


def installed_hashes() -> dict:
    """{key: sha256 of the decompressed bytes} - what props_status re-checks for staleness."""
    return {a["key"]: (sha256(read_bytes(a["path"])) if a["path"].exists() else None)
            for a in installed_artifacts()}


# ── run evidence (the props clean-retrain run that trained every installed artifact) ─────────────
def extract_run_evidence(run_dir: Path, run_id: str) -> dict:
    """The run's own record, from its downloaded artifact (props_harness_report.json + the harness
    pickles): sha256 of every harness artifact, verified shares, feature-schema hash, windows,
    walk-forward folds and holdout counts. Probabilities, metrics on prices and CLV are left out."""
    run_dir = Path(run_dir)
    rep = json.loads((run_dir / "output" / "props_harness" / "props_harness_report.json").read_text(encoding="utf-8"))
    markets = {}
    for mk, m in rep["markets"].items():
        arts = {}
        for cand, a in (m.get("artifacts") or {}).items():
            p = run_dir / a["path"]
            arts[cand] = {"path": a["path"], "sha256_reported": a["sha256"],
                          "sha256_downloaded": sha256(p.read_bytes()) if p.exists() else None}
        ho = m.get("holdout") or {}
        markets[mk] = {
            "target": m.get("target"), "rows": m.get("rows"), "verified_share": m.get("verified_share"),
            "holdout": {"dates": ho.get("dates"), "train_rows": ho.get("train_rows"),
                        "test_rows": ho.get("test_rows"), "status": ho.get("status"), "window": ho.get("window"),
                        "events": {c: (v or {}).get("events") for c, v in (ho.get("metrics") or {}).items()},
                        "unscored": {c: (v or {}).get("unscored") for c, v in (ho.get("metrics") or {}).items()}},
            "walk_forward": [{"fold": f.get("fold"), "test_dates": f.get("test_dates"), "status": f.get("status"),
                              "train_rows": f.get("train_rows"), "test_rows": f.get("test_rows"),
                              "window": f.get("window")} for f in (m.get("walk_forward") or [])],
            "artifacts": arts}
    return {"run_id": str(run_id), "harness_version": rep.get("harness_version"),
            "plan_version": rep.get("plan_version"), "generated_at": rep.get("generated_at"),
            "code_sha": rep.get("code_sha"), "evidence_class": rep.get("evidence_class"),
            "hand_adjustment": rep.get("hand_adjustment"), "owner_decisions": rep.get("owner_decisions"),
            "status": rep.get("status"), "verified_share_train": rep.get("verified_share_train"),
            "verified_share_holdout": rep.get("verified_share_holdout"), "rows_in_scope": rep.get("rows_in_scope"),
            "features": {k: v for k, v in (rep.get("features") or {}).items() if k in ("n", "sha256", "profile_coverage")},
            "markets": markets,
            "source": "GitHub Actions artifact 'props-clean-retrain' of run " + str(run_id)}


def run_evidence(run_id: str) -> dict | None:
    return _json(EVIDENCE_DIR / f"run{run_id}_evidence.json")


# ── probe rows ───────────────────────────────────────────────────────────────────────────────────
def probe_frame(n: int = 1200, seed: int = 11) -> pd.DataFrame:
    """REAL rows: the newest month of the committed player_history (features as stored, player ids).
    Used only as model INPUT for invariance / identity probes - never as a result."""
    from player_model import config as pc
    parts = sorted((ROOT / "player_history.parquet").glob("month_*.parquet"))
    if not parts:
        raise FileNotFoundError("player_history.parquet has no month part")
    cols = list(dict.fromkeys(list(pc.PLAYER_FEATURE_COLS) + ["player_id", "fixture_id", "date"]))
    frames, have = [], 0
    for p in reversed(parts):
        import pyarrow.parquet as pq
        names = set(pq.read_schema(p).names)
        d = pd.read_parquet(p, columns=[c for c in cols if c in names])
        frames.append(d)
        have += len(d)
        if have >= n:
            break
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[pd.to_numeric(df["player_id"], errors="coerce") > 0]
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def _score(payload: dict, df: pd.DataFrame) -> np.ndarray:
    from player_model import model as M
    return M.predict_proba(df, payload).to_numpy(dtype=float)


def batch_invariance(payload: dict, rows: pd.DataFrame, n_targets: int = 25, seed: int = 3) -> dict:
    """One OpenMP thread (threadpoolctl): many tiny predict calls are ~20x slower with thread start-up
    per call. Tree traversal is per row; threads only split the rows, so the probabilities are the same."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _batch_invariance(payload, rows, n_targets, seed)


def _batch_invariance(payload: dict, rows: pd.DataFrame, n_targets: int, seed: int) -> dict:
    """Each target row scored (a) alone, (b) with 10 others, (c) with 1,000 others, (d) in a shuffled
    batch, (e) beside unrelated rows whose inputs are partly missing (same column dtypes). Identical =
    |diff| <= FLOAT_TOL (1e-12: float rounding of a BLAS dot product, never a statistic); bit-identical
    rows are counted apart. (f) MIXED DTYPE, informational: the unrelated rows as float64 while the
    targets are float32 (the stored player_history dtype) - a column dtype that depends on the batch."""
    rng = np.random.default_rng(seed)
    rows = rows.reset_index(drop=True)
    targets = rows.iloc[:n_targets].reset_index(drop=True)
    others = rows.iloc[n_targets:].reset_index(drop=True)
    alone = np.array([_score(payload, targets.iloc[[i]])[0] for i in range(len(targets))])
    out = {}

    def _cmp(name, p, into=out):
        d = np.abs(p - alone)
        both_nan = np.isnan(p) & np.isnan(alone)
        d = np.where(both_nan, 0.0, d)
        d = np.where(np.isnan(d), np.inf, d)
        into[name] = {"max_abs_diff": float(d.max()) if len(d) else 0.0, "rows_not_bit_identical": int((d > 0).sum()),
                      "rows_differing": int((d > FLOAT_TOL).sum())}

    for k in BATCH_SIZES:
        add = others.iloc[:k]
        batch = pd.concat([targets, add], ignore_index=True)
        _cmp(f"with_{len(add)}_others", _score(payload, batch)[:len(targets)])
    big = pd.concat([targets, others.iloc[:BATCH_SIZES[-1]]], ignore_index=True)
    perm = rng.permutation(len(big))
    shuffled = _score(payload, big.iloc[perm].reset_index(drop=True))
    back = np.empty(len(big))
    back[perm] = shuffled
    _cmp("shuffled", back[:len(targets)])
    junk = others.iloc[:200].copy()
    feats = [c for c in junk.columns if c not in ("player_id", "fixture_id", "date")
             and pd.api.types.is_float_dtype(junk[c])]
    mask = rng.random((len(junk), len(feats))) < 0.4
    for j, c in enumerate(feats):
        junk[c] = junk[c].where(~mask[:, j])                        # NaN, dtype kept
    _cmp("with_unrelated_missing_rows", _score(payload, pd.concat([junk, targets], ignore_index=True))[len(junk):])
    out["targets"] = int(len(targets))
    out["targets_scored"] = int(np.isfinite(alone).sum())
    out["invariant"] = all(v["rows_differing"] == 0 for k, v in out.items() if isinstance(v, dict))
    mixed = junk.copy()
    for c in feats:
        mixed[c] = mixed[c].astype("float64")
    info = {}
    _cmp("mixed_dtype_float64_neighbours", _score(payload, pd.concat([mixed, targets], ignore_index=True))[len(mixed):], info)
    out["informational"] = info
    return out


# ── per-artifact checks ──────────────────────────────────────────────────────────────────────────
def _forbidden_features() -> set:
    from player_model import config as pc
    s = set(pc.SETPIECE_PROXIES_REMOVED) | set(pc.TYPED_FORMULA_FEATURES_WITHHELD) | \
        set(pc.LEGACY_QUALITY_FEATURES_REMOVED)
    for name in ("_DEAD_GOAL_TYPE_FEATURES", "_DATA_GAP_FEATURES"):
        s |= set(getattr(pc, name, set()))
    return s


def _num(v) -> float:
    """A number, or NaN when absent / not numeric: every comparison with NaN is False (fails closed)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _chk(ok: bool, detail="") -> dict:
    return {"ok": bool(ok), "detail": detail}


def certify_artifact(a: dict, rows: pd.DataFrame | None, ev: dict | None, runtime_sklearn: str) -> dict:
    from player_model import config as pc
    from player_model import props_contract as PC
    rec = {"key": a["key"], "market": a["market"], "role": a["role"], "model_version": a["version"],
           "path": str(a["path"].relative_to(ROOT)).replace("\\", "/"), "checks": {}}
    c = rec["checks"]
    if not a["path"].exists():
        c["file_present"] = _chk(False, "artifact file missing")
        rec["status"] = FAILED
        return rec
    raw = read_bytes(a["path"])
    sha = sha256(raw)
    rec["sha256"] = sha
    man = a.get("manifest") or {}
    ptr = a.get("pointer") or {}
    c["file_present"] = _chk(True)
    c["sha256_matches_registry"] = _chk(
        sha == man.get("artifact_sha256") == ptr.get("artifact_sha256") and bool(a["version"])
        and str(a["version"]).endswith(sha[:8]),
        f"bytes {sha[:12]} | manifest {str(man.get('artifact_sha256'))[:12]} | "
        f"{'pointer' if a['role'] != RESEARCH else 'challenger.json'} {str(ptr.get('artifact_sha256'))[:12]}")
    pl = load_raw(raw)
    cand = pl.get("candidate")
    prov = pl.get("provenance") or {}
    dec = pl.get("owner_decisions") or {}
    win = pl.get("window") or {}
    m = pl.get("model")
    rec.update({"candidate": cand, "recipe": pl.get("recipe"), "harness_version": pl.get("harness_version"),
                "provider": "API-Football player match statistics (player_history + cup rows, re-fetched counts), "
                            "built by player_model.props_harness.build_frame",
                "run_id": prov.get("run_id"), "code_sha": prov.get("code_sha"),
                "harness_artifact_sha256": prov.get("artifact_sha256"),
                "training_cutoff": man.get("training_cutoff"), "window": win,
                "holdout": {k: (prov.get("holdout") or {}).get(k) for k in ("n", "events", "holdout_dates", "rule")},
                "approval": {"approved_on": prov.get("approved_on"), "approval": prov.get("approval")},
                "pickled_sklearn": pickled_sklearn_versions(raw), "runtime_sklearn": runtime_sklearn})
    c["recipe_clean_harness"] = _chk(
        pl.get("recipe") == RECIPE == PC.RECIPE_CLEAN_HARNESS and pl.get("harness_version") == HARNESS_VERSION
        and pl.get("hand_adjustment") == "none" and cand in IMPUTER and cand == man.get("candidate"),
        f"recipe {pl.get('recipe')}, harness {pl.get('harness_version')}, candidate {cand}, "
        f"hand_adjustment {pl.get('hand_adjustment')}")
    # raw input provenance: owner feature rule + floors, the run, the run's own artifact sha
    floors_ok = (dec.get("feature_rule") == FEATURE_RULE and _num(dec.get("min_verified_share")) >= MIN_TRAIN_SHARE
                 and _num(dec.get("min_verified_share_holdout")) >= MIN_HOLDOUT_SHARE)
    run_ok, run_detail = False, "no committed run evidence"
    if ev:
        e_m = (ev.get("markets") or {}).get(a["market"]) or {}
        e_a = (e_m.get("artifacts") or {}).get(cand) or {}
        vs = e_m.get("verified_share") or {}
        run_ok = (str(ev.get("run_id")) == str(prov.get("run_id")) and ev.get("status") == "OK"
                  and ev.get("code_sha") == prov.get("code_sha")
                  and prov.get("artifact_sha256") == e_a.get("sha256_reported")
                  and (e_a.get("sha256_downloaded") in (None, e_a.get("sha256_reported")))
                  and _num(vs.get("train")) >= _num(dec.get("min_verified_share"))
                  and _num(vs.get("holdout")) >= _num(dec.get("min_verified_share_holdout"))
                  and ev.get("owner_decisions") == dec)
        run_detail = (f"run {ev.get('run_id')} status {ev.get('status')}; harness artifact "
                      f"{str(prov.get('artifact_sha256'))[:12]} vs run {str(e_a.get('sha256_reported'))[:12]}"
                      f"{' (downloaded bytes re-hashed)' if e_a.get('sha256_downloaded') else ''}; verified share "
                      f"train {vs.get('train')} / holdout {vs.get('holdout')}")
    c["raw_input_provenance"] = _chk(floors_ok and run_ok and bool(prov.get("run_id")),
                                     f"feature_rule {dec.get('feature_rule')}, floors "
                                     f"{dec.get('min_verified_share')}/{dec.get('min_verified_share_holdout')}; {run_detail}")
    # feature schema
    fcols = list(pl.get("feature_cols") or [])
    forbidden = sorted(set(fcols) & _forbidden_features())
    contract = set(PC.load_contract()["features"])
    if cand == "player_base_rate":
        schema_ok = fcols == [] and list(pl.get("inputs") or []) == ["player_id"]
        detail = f"no feature (inputs {pl.get('inputs')})"
    else:
        sch = hashlib.sha256(json.dumps(fcols).encode()).hexdigest()
        want = ((ev or {}).get("features") or {}).get("sha256")
        schema_ok = (not forbidden and set(fcols) <= contract and set(fcols) == set(pc.PLAYER_FEATURE_COLS)
                     and list(getattr(m, "feature_cols_", fcols)) == fcols and (want is None or sch == want))
        detail = (f"{len(fcols)} features, withheld/proxy/removed {len(forbidden)}, outside contract "
                  f"{len(set(fcols) - contract)}, = config.PLAYER_FEATURE_COLS {set(fcols) == set(pc.PLAYER_FEATURE_COLS)}, "
                  f"schema sha {sch[:12]} {'= run' if sch == want else '(no run evidence)' if want is None else '!= run ' + str(want)[:12]}")
    rec["features"] = len(fcols)
    rec["withheld_or_proxy_features"] = forbidden
    c["feature_schema"] = _chk(schema_ok, detail)
    prof = getattr(m, "training_profile_", None) or {}
    degen = sorted(f for f, e in prof.items() if f in fcols and _num(e.get("top_share")) > SINGLE_VALUE_MAX_SHARE)
    c["no_single_value_feature"] = _chk(not degen, f"{len(degen)} served feature(s) one value on > 99.9% of "
                                                   f"training rows{': ' + ', '.join(degen[:6]) if degen else ''}")
    # training window, cutoff, chronological validation
    fd, cd = win.get("fit_dates") or [None, None], win.get("calibration_dates") or [None, None]
    hd = (prov.get("holdout") or {}).get("holdout_dates") or [None, None]
    chrono = (all(fd + cd + hd) and fd[1] < cd[0] and cd[1] == man.get("training_cutoff") == ptr.get("training_cutoff")
              and cd[1] < hd[0] == dec.get("holdout_start") and hd[1] == dec.get("holdout_end"))
    wf_ok, wf_detail = False, "no run evidence"
    if ev:
        e_m = (ev.get("markets") or {}).get(a["market"]) or {}
        e_h = e_m.get("holdout") or {}
        folds = e_m.get("walk_forward") or []
        ok_folds = [f for f in folds if f.get("status") == "OK"
                    and (f.get("window") or {}).get("calibration_dates", [None, None])[1] < (f.get("test_dates") or [""])[0]]
        wf_ok = (e_h.get("window") == win and e_h.get("status") == "OK" and len(folds) > 0 and len(ok_folds) == len(folds))
        wf_detail = f"{len(ok_folds)}/{len(folds)} walk-forward folds trained strictly before their test week; holdout window = run"
    c["training_window_and_cutoff"] = _chk(chrono, f"fit {fd}, calibration {cd}, cutoff {man.get('training_cutoff')}, holdout {hd}")
    c["chronological_validation"] = _chk(chrono and wf_ok, wf_detail)
    tr = man.get("training_rows") or {}
    cov_ok = (win.get("fit_rows") and win.get("calibration_rows") and tr.get("fit") == win.get("fit_rows")
              and tr.get("calibration") == win.get("calibration_rows") and (prov.get("holdout") or {}).get("n"))
    rec["historical_coverage"] = {"fit_rows": win.get("fit_rows"), "calibration_rows": win.get("calibration_rows"),
                                  "first_training_date": fd[0], "holdout_rows": (prov.get("holdout") or {}).get("n")}
    c["historical_coverage"] = _chk(bool(cov_ok), f"{win.get('fit_rows')} fit + {win.get('calibration_rows')} calibration rows "
                                                  f"from {fd[0]}; holdout n {(prov.get('holdout') or {}).get('n')}")
    # missing data handling + train-fitted preprocessing
    from sklearn.ensemble import HistGradientBoostingClassifier
    if cand == "hgb_nan_native":
        final = m.model_._pipe.steps[-1][1]
        md_ok = isinstance(final, HistGradientBoostingClassifier) and pl.get("imputer_version") == IMPUTER[cand]
        md = "NaN-native HistGradientBoosting, no value filled anywhere"
    elif cand == "recipe_retrained":
        fills = getattr(m, "fill_", {})
        md_ok = (list(fills) == list(m.feature_cols_) == fcols and all(np.isfinite(list(fills.values())))
                 and not getattr(m, "dropped_unfittable_", []) and pl.get("imputer_version") == IMPUTER[cand])
        md = f"persisted FIT-block medians for {len(fills)} features (computed at fit, never from the scoring batch)"
    else:
        rates = getattr(m, "rates_", {})
        md_ok = (pl.get("imputer_version") == IMPUTER[cand] and 0 < float(m.market_rate_) < 1 and len(rates) > 0
                 and float(m.predict(None, players=np.array([-1.0]))[0]) == float(m.market_rate_))
        md = (f"player's training-window rate ({len(rates)} players); a player never seen in training gets the "
              f"training window's market rate (fitted, persisted), never a typed value")
    c["missing_data_handling"] = _chk(md_ok, md)
    # identity
    if cand == "player_base_rate":
        keys = list(getattr(m, "rates_", {}).keys())
        id_ok = all(isinstance(k, (int, np.integer)) and int(k) > 0 for k in keys)
        c["player_identity"] = _chk(id_ok, f"{len(keys)} API-Football player ids, all positive integers")
    else:
        c["player_identity"] = _chk(True, "scores the API-Football player_id x fixture_id rows built by "
                                          "build_upcoming_features (identity tests: global evidence)")
    # batch invariance (no batch statistic)
    if rows is not None:
        bi = batch_invariance(pl, rows)
        rec["batch_invariance"] = bi
        c["no_batch_statistics"] = _chk(bi["invariant"] and bi["targets_scored"] == bi["targets"],
                                        f"{bi['targets']} players alone vs with 10 / 1,000 others, shuffled, beside "
                                        f"unrelated rows with missing inputs: max |diff| "
                                        f"{max(v['max_abs_diff'] for k, v in bi.items() if isinstance(v, dict) and k != 'informational')}"
                                        f" (tolerance {FLOAT_TOL}); mixed-dtype probe (informational) "
                                        f"{bi['informational']['mixed_dtype_float64_neighbours']['max_abs_diff']}")
    else:
        c["no_batch_statistics"] = _chk(False, "no probe rows (player_history.parquet unreadable)")
    # train/serve recipe parity: serving calls the harness model's own predict(); pickled = runtime sklearn
    from player_model import model as M
    src = Path(M.__file__).read_text(encoding="utf-8")
    calls_own = 'payload["model"].predict(raw, players=players)' in src
    skl = rec["pickled_sklearn"]
    skl_ok = (cand == "player_base_rate" and not skl) or skl == [runtime_sklearn]
    c["train_serve_recipe_parity"] = _chk(calls_own and skl_ok,
                                          f"serving calls the harness model's own predict() {calls_own}; pickled sklearn "
                                          f"{skl or 'none (no estimator)'} vs runtime {runtime_sklearn}")
    # permission
    if a["role"] == RESEARCH:
        serving_sha = sha256(read_bytes(pc.MODEL_FILES[a["market"]])) if a["market"] in pc.MODEL_FILES else None
        perm_ok = (man.get("model_role") == "SHADOW" and man.get("promotion_status") == "SHADOW_RESEARCH"
                   and not (man.get("gate") or {}).get("qualified") and serving_sha != sha)
        rec["permission"] = "RESEARCH_ONLY (challenger, first-sight scoring, never a tip)"
    elif a["role"] == RARE:
        perm_ok = (man.get("model_role") == "CHAMPION" and cand == "player_base_rate"
                   and a["market"] in pc.BASE_RATE_SERVED_MARKETS and ptr.get("promotion_source") == "OWNER_APPROVED_CLEAN_RETRAIN")
        rec["permission"] = "RARE_RESEARCH_PICKS (hidden market: no normal tip; owner 2026-11-18)"
    else:
        perm_ok = (man.get("model_role") == "CHAMPION" and ptr.get("promotion_source") == "OWNER_APPROVED_CLEAN_RETRAIN"
                   and a["market"] in pc.MARKETS)
        rec["permission"] = "PRODUCTION (owner-approved clean retrain champion)"
    c["permission"] = _chk(perm_ok, rec["permission"])
    rec["status"] = VERIFIED if all(v["ok"] for v in c.values()) else FAILED
    return rec


# ── global evidence ──────────────────────────────────────────────────────────────────────────────
def _test_present(path: str, fn: str | None) -> bool:
    p = ROOT / path
    if not p.exists():
        return False
    return fn is None or re.search(rf"^def {re.escape(fn)}\(", p.read_text(encoding="utf-8"), re.M) is not None


IMPORTS_PROPS_MODEL = re.compile(r"from player_model\.model import|from player_model import [^\n]*\bmodel\b|"
                                 r"import player_model\.model|from \.model import|from \. import [^\n]*\bmodel\b")
R2_SHADOW_DIR = ROOT / "models" / "challengers" / "props_r2"


def prep_reachability(payloads: dict) -> dict:
    """Is the legacy batch-median player_model.model._prep reachable in production?

    Call graph (static, over every non-test module that imports player_model.model): _prep is called
    only inside player_model/model.py - model.train(recipe='legacy') and the legacy branch of
    predict_with_diagnostics / predict_proba, taken only for a payload WITHOUT a clean or repaired
    recipe. model.train's callers are research scripts (no serving write: save_model raises); the weekly
    props retrain (pipeline.mode_train) is retired. Runtime: _prep replaced by a function that raises,
    every installed serving payload AND every R2 shadow payload predict.py scores is run through the
    production entry points (model.predict_proba / predict_with_diagnostics)."""
    from player_model import model as M
    from player_model import pipeline as PL
    from player_model import props_contract as PC
    prep_calls, train_calls = [], []
    for p in sorted(ROOT.rglob("*.py")):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if rel.startswith(("tests/", "_archive/", ".")) or "/." in rel or rel == "src/integrity/props_certification.py":
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if rel != "player_model/model.py" and not (IMPORTS_PROPS_MODEL.search(txt) and
                                                   (rel.startswith("player_model/") or "player_model" in txt)):
            continue
        try:
            tree = ast.parse(txt)
        except SyntaxError:
            continue
        for node in ast.walk(tree):                                    # CALLS only (AST, not text)
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name == "_prep":
                prep_calls.append(f"{rel}:{node.lineno}")
            elif name == "train" and rel != "player_model/model.py":
                train_calls.append(f"{rel}:{node.lineno}")
    outside = [c for c in prep_calls if not c.startswith(("player_model/model.py:", "player_model/props_harness_clv.py:"))]
    research_train = [c for c in train_calls if c.startswith("scripts/")]
    production_train = [c for c in train_calls if not c.startswith("scripts/")]
    retired = "nothing trained, scored or parked" in Path(PL.__file__).read_text(encoding="utf-8")
    shadows = {}
    for f in sorted(R2_SHADOW_DIR.glob("props_r2_*.pkl")):
        shadows["r2_shadow:" + f.stem.replace("props_r2_", "")] = load_raw(f.read_bytes())
    recipes = {m: M.payload_recipe(p) for m, p in {**payloads, **shadows}.items()}
    legacy = sorted(m for m, r in recipes.items() if r not in (RECIPE, PC.RECIPE_REPAIRED))
    orig, hits = M._prep, {"n": 0}

    def _boom(*a, **k):
        hits["n"] += 1
        raise AssertionError("legacy batch-median _prep reached")
    runtime_ok, err = True, ""
    try:
        M._prep = _boom
        rows = probe_frame(200)
        for m, p in payloads.items():
            M.predict_proba(rows, p)
        for m, p in shadows.items():
            M.predict_with_diagnostics(rows, p)
    except AssertionError as e:
        runtime_ok, err = False, str(e)
    except Exception as e:                                             # noqa: BLE001
        runtime_ok, err = False, f"{type(e).__name__}: {e}"
    finally:
        M._prep = orig
    ok = not legacy and retired and runtime_ok and hits["n"] == 0 and not outside and not production_train
    return {"ok": ok, "reachable_in_production": not ok,
            "prep_call_sites": [c for c in prep_calls if c.startswith("player_model/model.py:")],
            "prep_callers_outside_model_py": outside,
            "note_props_harness_clv": "player_model/props_harness_clv.py defines its own quote _prep (a different function)",
            "train_callers_production": production_train, "train_callers_research_scripts": research_train,
            "legacy_weekly_train_retired": retired, "payload_recipes": recipes, "legacy_payloads_scored": legacy,
            "runtime_probe": {"payloads_scored": len(payloads) + len(shadows), "prep_calls": hits["n"],
                              "ok": runtime_ok, "error": err},
            "residual_risk": "props_gate.rollback_props can re-install a legacy registry version (operator action); "
                             "the certification then reports that payload FAILED (recipe), props_status STALE, and "
                             "the legacy branch would run again - no code guard refuses it today"}


def global_evidence(run_id: str | None, payloads: dict) -> dict:
    g = {}
    g["legacy_prep_unreachable"] = prep_reachability(payloads)
    pit = {f"{p}::{f or '*'}": _test_present(p, f) for p, f in PIT_TESTS}
    g["point_in_time_tests"] = {"ok": all(pit.values()), "tests": pit}
    ids = {f"{p}::{f}": _test_present(p, f) for p, f in IDENTITY_TESTS}
    g["identity_tests"] = {"ok": all(ids.values()), "tests": ids}
    par = _json(PARITY_REPORT)
    from player_model import config as pc
    ml_markets = [m for m, p in payloads.items() if p.get("candidate") != "player_base_rate"]
    if par:
        probs = par.get("probabilities") or {}
        ok = (par.get("features_mismatched") == 0 and not par.get("features_absent")
              and len(par.get("features") or {}) == len(pc.PLAYER_FEATURE_COLS)
              and all((probs.get(m) or {}).get("rows_not_identical") == 0 for m in ml_markets)
              and run_id is not None and str(run_id) in PARITY_REPORT.name)
        g["serving_vs_training_parity"] = {
            "ok": ok, "report": str(PARITY_REPORT.relative_to(ROOT)).replace("\\", "/"),
            "paired_rows": par.get("paired_rows"), "fixtures": par.get("fixtures"), "sample": par.get("sample"),
            "features_mismatched": par.get("features_mismatched"),
            "probabilities_not_identical": {m: (probs.get(m) or {}).get("rows_not_identical") for m in probs},
            "markets_required": ml_markets,
            "note": "player_base_rate markets read only player_id (no feature to mismatch)"}
    else:
        g["serving_vs_training_parity"] = {"ok": False, "report": None}
    return g


# ── offline evidence (needs the downloaded run artifact) ─────────────────────────────────────────
def offline_identity(run_dir: Path, arts: list[dict], rows: pd.DataFrame) -> dict:
    """Installed model == the run's own harness model: identical probabilities on the probe rows."""
    out = {}
    for a in arts:
        if not a["path"].exists():
            continue
        pl = load_raw(read_bytes(a["path"]))
        prov = pl.get("provenance") or {}
        hp = run_dir / "models" / "challengers" / "props_clean" / f"props_clean_{a['market']}_{pl.get('candidate')}.pkl"
        if not hp.exists():
            out[a["key"]] = {"ok": False, "detail": "harness artifact not in the run download"}
            continue
        hraw = hp.read_bytes()
        art = load_raw(hraw)
        feats = list(pl.get("feature_cols") or [])
        X = pd.DataFrame({c: pd.to_numeric(rows[c], errors="coerce") if c in rows.columns else np.nan for c in feats},
                         index=rows.index)
        players = pd.to_numeric(rows["player_id"], errors="coerce").to_numpy()
        p1 = np.asarray(pl["model"].predict(X, players=players), float)
        p2 = np.asarray(art["model"].predict(X, players=players), float)
        d = float(np.nanmax(np.abs(p1 - p2))) if len(p1) else 0.0
        out[a["key"]] = {"ok": sha256(hraw) == prov.get("artifact_sha256") and d == 0.0
                         and bool(np.isfinite(p1).all()),
                         "harness_sha256": sha256(hraw), "rows": int(len(rows)), "max_abs_diff": d}
    return out


def offline_holdout_replay(run_dir: Path, frame_path: Path, arts: list[dict]) -> dict:
    """The run's scored holdout (props_harness_holdout_scored.csv.gz) re-scored by the INSTALLED artifacts
    on the holdout frame rebuilt today by props_harness.build_frame (same code path as training)."""
    from player_model import model as M
    h = pd.read_csv(run_dir / "output" / "props_harness" / "props_harness_holdout_scored.csv.gz", low_memory=False)
    fr = pd.read_parquet(frame_path)
    key = ["player_id", "fixture_id"]
    for c in key:
        fr[c] = pd.to_numeric(fr[c], errors="coerce")
        h[c] = pd.to_numeric(h[c], errors="coerce")
    fr = fr.drop_duplicates(key)
    out = {}
    for a in arts:
        if not a["path"].exists():
            continue
        pl = load_raw(read_bytes(a["path"]))
        cand = pl.get("candidate")
        g = h[h["market"] == a["market"]]
        j = g.merge(fr, on=key, how="inner", suffixes=("", "_f"))
        p = M.predict_proba(j, pl).to_numpy(float)
        ref = pd.to_numeric(j[f"p_{cand}"], errors="coerce").to_numpy(float)
        d = np.abs(p - ref)
        out[a["key"]] = {"holdout_rows": int(len(g)), "matched": int(len(j)),
                         "identical": int((d == 0).sum()), "within_1e-9": int((d <= 1e-9).sum()),
                         "max_abs_diff": float(np.nanmax(d)) if len(d) else None,
                         "share_within_1e-9": float((d <= 1e-9).mean()) if len(d) else None}
    return out


# ── build ────────────────────────────────────────────────────────────────────────────────────────
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                  # noqa: BLE001
        return "unknown"


def build(run_artifact: Path | None = None, holdout_frame: Path | None = None) -> dict:
    import sklearn
    arts = installed_artifacts()
    try:
        rows = probe_frame()
    except Exception:                                                  # noqa: BLE001
        rows = None
    run_ids = set()
    for a in arts:
        if a["path"].exists():
            run_ids.add(str((load_raw(read_bytes(a["path"])).get("provenance") or {}).get("run_id")))
    run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
    ev = run_evidence(run_id) if run_id else None
    recs = [certify_artifact(a, rows, ev, sklearn.__version__) for a in arts]
    serving = {a["market"]: load_raw(read_bytes(a["path"])) for a in arts if a["role"] != RESEARCH and a["path"].exists()}
    g = global_evidence(run_id, serving)
    g["single_training_run"] = {"ok": run_id is not None and ev is not None, "run_ids": sorted(run_ids),
                                "evidence_file": f"reports/p0_props_cert/run{run_id}_evidence.json" if ev else None}
    offline = {}
    if run_artifact:
        offline["identity_vs_run_artifact"] = offline_identity(Path(run_artifact), arts, rows) if rows is not None else {}
        if holdout_frame:
            offline["holdout_replay"] = offline_holdout_replay(Path(run_artifact), Path(holdout_frame), arts)
    failing = [f"{r['key']}: {k} - {v['detail']}" for r in recs for k, v in r["checks"].items() if not v["ok"]]
    failing += [f"global: {k}" for k, v in g.items() if not v.get("ok")]
    failing += [f"offline identity: {k}" for k, v in (offline.get("identity_vs_run_artifact") or {}).items() if not v.get("ok")]
    verdict = VERIFIED if not failing else FAILED
    return {"artifact": "props_real_data_provenance", "version": VERSION,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "code_sha": _git_sha(),
            "verdict": verdict, "failing": failing,
            "counts": {"artifacts": len(recs), "verified": sum(r["status"] == VERIFIED for r in recs),
                       "by_role": {r: sum(x["role"] == r for x in recs) for r in (PRODUCTION, RARE, RESEARCH)}},
            "installed_hashes": {r["key"]: r.get("sha256") for r in recs},
            "artifacts": recs, "global": g, "offline": offline,
            "rule": "VERIFIED only when every check of every installed props artifact and every global check passes; "
                    "props_status re-hashes the installed artifacts and reports STALE on any difference. No price is "
                    "read or written."}


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--extract-run", type=Path)
    ap.add_argument("--run-id")
    ap.add_argument("--run-artifact", type=Path)
    ap.add_argument("--holdout-frame", type=Path)
    a = ap.parse_args(argv)
    if a.extract_run:
        if not a.run_id:
            ap.error("--extract-run needs --run-id")
        ev = extract_run_evidence(a.extract_run, a.run_id)
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / f"run{a.run_id}_evidence.json").write_text(_dump(ev), encoding="utf-8")
        print(f"wrote reports/p0_props_cert/run{a.run_id}_evidence.json")
        return 0
    res = build(a.run_artifact, a.holdout_frame)
    print(f"props certification: {res['verdict']} - {res['counts']['verified']}/{res['counts']['artifacts']} artifacts")
    for f in res["failing"]:
        print("  FAIL", f)
    if a.write:
        OUT.write_text(_dump(res), encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if res["verdict"] == VERIFIED else 1


if __name__ == "__main__":
    sys.exit(main())
