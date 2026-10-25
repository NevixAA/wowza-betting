"""Player-prop models promote only through the canonical promotion gate (owner decision 2026-10-25).

    python -m pytest tests/test_props_promotion_gate.py -q

Every test runs against a temporary registry and temporary serving files — nothing here touches
models/ or output/.
"""
from __future__ import annotations

import json
import math
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

from player_model import config as pconfig  # noqa: E402
from player_model import props_gate as PPG  # noqa: E402
from src import model_registry as reg  # noqa: E402

T0 = datetime(2026, 10, 25, 4, 0, tzinfo=timezone.utc)


class ColumnModel:
    """A 'model' whose probability is one feature column. Picklable, deterministic."""

    def predict_proba(self, X):
        p = np.clip(np.asarray(X, float)[:, 0], 1e-4, 1 - 1e-4)
        return np.column_stack([1 - p, p])


def _results(col: str) -> dict:
    return {"m": {"model": ColumnModel(), "feature_cols": [col], "metrics": {"log_loss": 0.1}}}


def _payload(col: str, market: str) -> bytes:
    from player_model.model import payload_bytes
    return payload_bytes(_results(col), market)


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
    monkeypatch.setattr(PPG, "H2H_FILE", lin / "props_shadow_h2h.csv")
    monkeypatch.setattr(PPG, "HISTORY_FILE", tmp_path / "player_history.parquet")
    monkeypatch.delenv(PPG.OWNER_OVERRIDE_ENV, raising=False)
    models.mkdir(parents=True)
    yield tmp_path


def _holdout(market="goals", n=6000, seed=1, champ_sd=0.0, cand_sd=0.0, base=None):
    """Synthetic holdout: true probability, outcomes, and champion/candidate noisy versions of it."""
    rng = np.random.default_rng(seed)
    b = base if base is not None else PPG.market_cfg(market)["base_rate"]
    truth = np.clip(rng.beta(2, 2 / b - 2, n), 0.002, 0.95)
    y = (rng.uniform(size=n) < truth).astype(int)
    noise = lambda sd, s: np.clip(truth * np.exp(np.random.default_rng(s).normal(0, sd, n)), 1e-4, 0.99)  # noqa: E731
    dates = pd.Timestamp("2026-03-01") + pd.to_timedelta(np.arange(n) // max(1, n // 120), unit="D")
    return pd.DataFrame({"date": dates, "league": np.where(np.arange(n) % 2, "La Liga", "Serie A"),
                         "fixture_id": np.arange(n) // 20 + 1000, "player_id": np.arange(n) % 997 + 1,
                         pconfig.MARKET_TARGETS[market]: y, "truth": truth,
                         "pc": noise(champ_sd, seed + 10) if champ_sd else truth,
                         "pn": noise(cand_sd, seed + 20) if cand_sd else truth})


def _serve(market, col="pc"):
    raw = _payload(col, market)
    pconfig.MODEL_FILES[market].write_bytes(raw)
    return raw


# ── config ────────────────────────────────────────────────────────────────────────────────────
def test_props_section_is_versioned_declared_and_covers_every_market():
    cfg = json.loads((ROOT / "config" / "promotion_gate.json").read_text(encoding="utf-8"))
    p = cfg["props"]
    assert p["gate_version"] == PPG.GATE_VERSION == "PG-PROPS-1.0"
    assert p["declared_on"] == "2026-10-25" and p["canonical_gate_version"] == cfg["gate_version"]
    assert set(pconfig.MARKETS) <= set(p["markets"])
    assert p["holdout"]["bootstrap_unit"] == "match_date"
    assert p["holdout"]["min_n_per_market"] == p["prospective"]["min_n_per_market"] == 1000
    # the canonical file's team values are untouched
    assert cfg["holdout"]["material_ll_improvement"] == 0.001


def test_market_thresholds_follow_the_declared_scaling_rule():
    """Every rescaled value = the PG-1.0 value x a factor computed from the LABEL base rate."""
    canon = json.loads((ROOT / "config" / "promotion_gate.json").read_text(encoding="utf-8"))
    for m, t in PPG.CFG["markets"].items():
        b = t["base_rate"]
        h = -(b * math.log(b) + (1 - b) * math.log(1 - b)) / math.log(2)
        v, k = b * (1 - b) / 0.25, b / 0.5
        expect = {"material_ll_improvement": canon["holdout"]["material_ll_improvement"] * h,
                  "max_league_ll_regression": canon["holdout"]["max_league_ll_regression"] * h,
                  "max_brier_worsening": canon["holdout"]["max_brier_worsening"] * v,
                  "max_ece_worsening": canon["holdout"]["max_ece_worsening"] * k,
                  "max_mean_shift": canon["canary"]["max_mean_shift"] * k,
                  "max_quantile_shift": canon["canary"]["max_quantile_shift"] * k,
                  "moved_pp": canon["canary"]["moved_pp"] * k}
        for key, e in expect.items():
            assert abs(t[key] - e) / e < 0.05, (m, key, t[key], e)
    # rare markets get a smaller absolute floor than common ones, never the same number
    assert PPG.market_cfg("sot4")["material_ll_improvement"] < PPG.market_cfg("goals")["material_ll_improvement"]


# ── period 1 ──────────────────────────────────────────────────────────────────────────────────
def test_better_candidate_qualifies_on_holdout():
    d = _holdout("goals", n=20000, champ_sd=0.15)
    ev = PPG.evaluate_holdout("goals", d["target_goals"], d["pc"], d["pn"], dates=d["date"],
                              leagues=d["league"])
    assert ev["qualified"], ev["reason"]
    assert ev["ll_delta"] <= -PPG.market_cfg("goals")["material_ll_improvement"] and ev["ci"][1] < 0


def test_tolerance_only_candidate_does_not_qualify():
    """Worse by a little — comfortably inside the old PROPS_MAX_LOGLOSS_RISE 0.030 — used to promote."""
    d = _holdout("goals", cand_sd=0.25)
    ev = PPG.evaluate_holdout("goals", d["target_goals"], d["pc"], d["pn"], dates=d["date"],
                              leagues=d["league"])
    assert 0 < ev["ll_delta"] < 0.030
    assert not ev["qualified"] and not ev["checks"]["material_improvement"]


def test_immaterial_gain_does_not_qualify():
    """The candidate is the truth, the champion 3% over-stated: really better, but immaterially."""
    d = _holdout("goals", n=40000, seed=4)
    champ = np.clip(d["truth"] * 1.03, 1e-4, 0.99)
    ev = PPG.evaluate_holdout("goals", d["target_goals"], champ, d["truth"], dates=d["date"],
                              leagues=d["league"])
    assert -PPG.market_cfg("goals")["material_ll_improvement"] < ev["ll_delta"] < 0
    assert not ev["checks"]["material_improvement"] and not ev["qualified"]


def test_too_few_events_cannot_qualify():
    d = _holdout("sot4", n=3000, champ_sd=0.6)                      # ~8 events
    ev = PPG.evaluate_holdout("sot4", d["target_sot4"], d["pc"], d["pn"], dates=d["date"])
    assert ev["events"] < 100 and not ev["evaluable"] and not ev["qualified"]


def test_tolerance_only_candidate_is_not_promoted_and_serving_file_untouched(ptmp):
    old = _serve("goals")
    d = _holdout("goals", cand_sd=0.25)
    gate = PPG.submit_candidate("goals", _results("pn"), d, now=T0)
    assert gate["verdict"] == PPG.KEEP and not gate["promoted"] and not gate["parked"]
    assert pconfig.MODEL_FILES["goals"].read_bytes() == old
    log = pd.read_csv(PPG.DECISIONS_FILE)
    assert set(log["gate_version"]) == {"PG-PROPS-1.0"}
    assert list(log["stage"]) == ["REGISTER", "HOLDOUT"]       # legacy champion registered, not changed


# ── period 2: no prospective period -> never PROMOTE ──────────────────────────────────────────
def _qualify(market="goals"):
    old = _serve(market)
    d = _holdout(market, n=20000, champ_sd=0.15)
    gate = PPG.submit_candidate(market, _results("pn"), d, now=T0)
    assert gate["verdict"] == PPG.SHADOW and gate["qualified"] and gate["parked"], gate["reason"]
    return old, gate


def test_qualified_candidate_is_parked_not_promoted(ptmp):
    old, gate = _qualify()
    assert pconfig.MODEL_FILES["goals"].read_bytes() == old
    assert not gate["promoted"] and PPG.challenger_path("goals").exists()


def test_no_prospective_period_means_shadow_more(ptmp):
    old, gate = _qualify()
    for later in (T0 + timedelta(days=7), T0 + timedelta(days=60)):
        rec = PPG.review_parked("goals", now=later, h2h=pd.DataFrame(), outcomes=pd.DataFrame())
        assert rec["verdict"] == PPG.SHADOW and not rec["promoted"]
    assert pconfig.MODEL_FILES["goals"].read_bytes() == old


def _h2h(market, gate, d, logged, col_ch="pn", n=None):
    d = d.iloc[: n or len(d)]
    return pd.DataFrame({
        "logged_at": logged, "market": market, "challenger_version": gate["candidate_version"],
        "champion_version": gate["champion_version"], "fixture_id": d["fixture_id"].astype(str),
        "player_id": d["player_id"].astype(str), "league": d["league"],
        "match_date": "2026-11-01", "kickoff_utc": "2026-11-01T15:00:00Z",
        "p_champion_raw": d["pc"], "p_challenger": d[col_ch]}).drop_duplicates(
        ["fixture_id", "player_id"])


def _outcomes(market, d):
    return pd.DataFrame({"fixture_id": d["fixture_id"].astype(str), "player_id": d["player_id"].astype(str),
                         "y": d[pconfig.MARKET_TARGETS[market]].astype(float)})


def test_too_little_prospective_data_is_shadow_more(ptmp):
    old, gate = _qualify()
    fut = _holdout("goals", n=20000, champ_sd=0.15, seed=9)
    fut["date"] = pd.Timestamp("2026-11-01")
    h = _h2h("goals", gate, fut, "2026-10-30T10:00:00Z", n=400)
    rec = PPG.review_parked("goals", now=T0 + timedelta(days=8), h2h=h, outcomes=_outcomes("goals", fut))
    assert rec["verdict"] == PPG.SHADOW and rec["prospective_status"] == "PENDING"
    assert pconfig.MODEL_FILES["goals"].read_bytes() == old


def test_rows_logged_before_the_freeze_or_after_kickoff_do_not_count(ptmp):
    old, gate = _qualify()
    fut = _holdout("goals", n=20000, champ_sd=0.15, seed=9)
    before = _h2h("goals", gate, fut, "2026-10-20T10:00:00Z")            # before the freeze (T0)
    late = _h2h("goals", gate, fut, "2026-11-01T16:00:00Z")              # after kickoff
    for h in (before, late):
        rec = PPG.review_parked("goals", now=T0 + timedelta(days=8), h2h=h,
                                outcomes=_outcomes("goals", fut))
        assert rec["verdict"] == PPG.SHADOW and rec["prospective_n"] == 0


def test_both_periods_passed_promotes_through_the_gate_only(ptmp):
    """Positive control: period 2 CAN pass, and then the gate — only the gate — installs it."""
    old, gate = _qualify()
    fut = _holdout("goals", n=20000, champ_sd=0.15, seed=9)
    h = _h2h("goals", gate, fut, "2026-10-30T10:00:00Z")
    rec = PPG.review_parked("goals", now=T0 + timedelta(days=8), h2h=h, outcomes=_outcomes("goals", fut))
    assert rec["verdict"] == PPG.PROMOTE and rec["promoted"], rec["reason"]
    assert pconfig.MODEL_FILES["goals"].read_bytes() == _payload("pn", "goals") != old
    assert PPG.read_pointer()["goals"]["model_version"] == gate["candidate_version"]
    assert not PPG.challenger_path("goals").exists()
    # and the old champion can be restored exactly
    legacy = gate["champion_version"]
    PPG.rollback_props("goals", legacy, reason="test")
    assert pconfig.MODEL_FILES["goals"].read_bytes() == old
    assert pd.read_csv(PPG.DECISIONS_FILE)["verdict"].iloc[-1] == PPG.ROLLBACK


def test_prospective_worse_challenger_is_retired(ptmp):
    old, gate = _qualify()
    fut = _holdout("goals", n=20000, cand_sd=0.6, seed=9)                        # now the challenger is worse
    h = _h2h("goals", gate, fut, "2026-10-30T10:00:00Z")
    rec = PPG.review_parked("goals", now=T0 + timedelta(days=8), h2h=h, outcomes=_outcomes("goals", fut))
    assert rec["verdict"] == PPG.KEEP and not PPG.challenger_path("goals").exists()
    assert pconfig.MODEL_FILES["goals"].read_bytes() == old


def test_challenger_frozen_against_a_different_champion_is_retired(ptmp):
    old, gate = _qualify()
    pconfig.MODEL_FILES["goals"].write_bytes(_payload("truth", "goals"))   # serving changed by hand
    rec = PPG.review_parked("goals", now=T0 + timedelta(days=8), h2h=pd.DataFrame(),
                            outcomes=pd.DataFrame())
    assert rec["verdict"] == PPG.KEEP and "retired" in rec["reason"]
    assert pd.read_csv(PPG.DECISIONS_FILE)["stage"].tolist().count("REGISTER") == 2   # drift logged


# ── per-market separation ─────────────────────────────────────────────────────────────────────
def test_markets_are_judged_and_stored_separately(ptmp):
    sot_old = _serve("sot")
    old, gate = _qualify("goals")
    assert not PPG.challenger_path("sot").exists()
    assert pconfig.MODEL_FILES["sot"].read_bytes() == sot_old
    # goals' first-sight rows can never count for sot
    fut = _holdout("goals", n=20000, champ_sd=0.15, seed=9)
    h = _h2h("goals", gate, fut, "2026-10-30T10:00:00Z")
    assert PPG.prospective_rows("sot", gate["candidate_version"], gate["champion_version"], T0,
                                h2h=h, outcomes=_outcomes("goals", fut)).empty
    PPG.review_parked("goals", now=T0 + timedelta(days=8), h2h=h, outcomes=_outcomes("goals", fut))
    assert pconfig.MODEL_FILES["sot"].read_bytes() == sot_old
    assert set(PPG.read_pointer()) == {"goals"}          # sot's registry entry never touched
    assert pd.read_csv(PPG.DECISIONS_FILE)["market"].eq("goals").all()


def test_bootstrap_only_without_any_serving_file_or_record(ptmp):
    d = _holdout("cards", n=20000, champ_sd=0.15)
    gate = PPG.submit_candidate("cards", _results("pn"), d, now=T0)
    assert gate["verdict"] == PPG.BOOTSTRAP and pconfig.MODEL_FILES["cards"].exists()
    # a known champion whose file disappears is RESTORED, never bootstrapped over
    good = pconfig.MODEL_FILES["cards"].read_bytes()
    pconfig.MODEL_FILES["cards"].unlink()
    g2 = PPG.submit_candidate("cards", _results("pc"), d, now=T0 + timedelta(days=7))
    assert g2["verdict"] == PPG.KEEP and pconfig.MODEL_FILES["cards"].read_bytes() == good


# ── predict side: first-sight logging ─────────────────────────────────────────────────────────
def test_first_sight_recorder_logs_once_and_refuses_after_kickoff(ptmp):
    old, gate = _qualify()
    feat = pd.DataFrame({"player_id": [11, 12, 13], "pc": [0.1, 0.2, 0.3], "pn": [0.15, 0.25, 0.35]})
    r = PPG.FirstSightRecorder(now=T0 + timedelta(days=1))
    assert r.add("goals", feat, feat["pc"], league="La Liga", match_date="2026-10-27",
                 kickoff_utc="2026-10-27T19:00:00Z", fixture_id=555) == 3
    assert r.add("sot", feat, feat["pc"], league="La Liga", match_date="2026-10-27",
                 kickoff_utc="2026-10-27T19:00:00Z", fixture_id=555) == 0      # nothing parked
    assert r.add("goals", feat, feat["pc"], league="x", match_date="2026-10-27",
                 kickoff_utc="", fixture_id=float("nan")) == 0                  # ungradable
    assert r.flush() == 3
    h = pd.read_csv(PPG.H2H_FILE, dtype=str)
    assert h["p_challenger"].astype(float).tolist() == [0.15, 0.25, 0.35]
    assert set(h["champion_version"]) == {gate["champion_version"]}
    first = PPG.H2H_FILE.read_bytes()
    r2 = PPG.FirstSightRecorder(now=T0 + timedelta(days=1, hours=3))
    r2.add("goals", feat.assign(pn=0.9), feat["pc"], league="La Liga", match_date="2026-10-27",
           kickoff_utc="2026-10-27T19:00:00Z", fixture_id=555)
    assert r2.flush() == 0 and PPG.H2H_FILE.read_bytes() == first           # earliest wins
    r3 = PPG.FirstSightRecorder(now=datetime(2026, 10, 27, 19, 5, tzinfo=timezone.utc))
    r3.add("goals", feat, feat["pc"], league="La Liga", match_date="2026-10-27",
           kickoff_utc="2026-10-27T19:00:00Z", fixture_id=556)
    assert r3.flush() == 0                                                   # after kickoff


def test_recorder_never_writes_the_registry(ptmp):
    _serve("goals")
    r = PPG.FirstSightRecorder(now=T0)
    feat = pd.DataFrame({"player_id": [1], "pc": [0.1]})
    r.add("goals", feat, feat["pc"], league="x", match_date="2026-10-27", kickoff_utc="",
          fixture_id=1)
    assert not PPG.PROPS_POINTER_FILE.exists() and not PPG.DECISIONS_FILE.exists()


# ── static: only the gate writes a props serving file ─────────────────────────────────────────
def _py_files():
    skip = {"tests", "_archive", ".venv", "venv", ".claude", ".git", "node_modules"}
    for p in ROOT.rglob("*.py"):
        if not (set(p.relative_to(ROOT).parts) & skip):
            yield p


def test_only_the_props_gate_writes_props_serving_files():
    gate = (ROOT / "player_model" / "props_gate.py").resolve()
    model_py = (ROOT / "player_model" / "model.py").resolve()
    serving_ref = re.compile(r"MODEL_FILES\[|model_player_|serving_file\(")
    writes = re.compile(r"_atomic_write_bytes\(|write_bytes\(|pickle\.dump\(|joblib\.dump\(|"
                        r"open\([^)]*['\"][wa]b?['\"]|shutil\.(copy|move)|os\.replace\(")
    offenders = []
    for p in _py_files():
        rp = p.resolve()
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            code = line.split("#", 1)[0]
            where = f"{p.relative_to(ROOT)}:{i}"
            if re.search(r"\bsave_model\(\s*[A-Za-z_]", code) and not code.lstrip().startswith("def ") \
                    and rp != model_py:
                offenders.append(where + " calls player_model save_model")
            if "player_model" in p.parts or "model_player_" in code:
                if serving_ref.search(code) and writes.search(code) and rp != gate:
                    offenders.append(where + " writes a props serving model file")
    assert not offenders, "\n".join(offenders)
    # save_model itself refuses
    from player_model.model import save_model
    with pytest.raises(PermissionError):
        save_model(_results("pc"), "goals")
    # the old tolerance gate is gone from the train path
    src = (ROOT / "player_model" / "pipeline.py").read_text(encoding="utf-8")
    assert 'os.getenv("PROPS_MAX_LOGLOSS_RISE"' not in src and "PROPS_FORCE_PROMOTE" not in src
    assert "PPG.review_parked(" in src and "PPG.submit_candidate(" in src
    assert src.index("PPG.review_parked(") < src.index("PPG.submit_candidate(")
    # no workflow copies or moves a props model into place by hand
    for y in (ROOT / ".github" / "workflows").glob("*.yml"):
        for line in y.read_text(encoding="utf-8").splitlines():
            if "model_player_" in line:
                assert not re.search(r"(^|\s)(cp|mv|tee)\s|>\s*models/", line), f"{y.name}: {line}"


# ── workflows stage what the gate writes ──────────────────────────────────────────────────────
def test_workflows_stage_the_gate_records():
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    train = (ROOT / ".github" / "workflows" / "player_props_train.yml").read_text(encoding="utf-8")
    for p in (PPG.DECISIONS_FILE, PPG.CHAMPION_HISTORY):
        assert rel(p) in train
    assert f"d={rel(PPG.REGISTRY_DIR)}" in train and 'git add -A -f "$d"' in train
    assert "output/props_retrain_log.json" in train and "c.MODEL_FILES.values()" in train
    assert "python -m player_model.pipeline --mode train" in train
    assert 'get("latest_run")' in train                           # the weekly guard still reads it
    props = (ROOT / ".github" / "workflows" / "player_props.yml").read_text(encoding="utf-8")
    assert rel(PPG.H2H_FILE) in props
    # every registry write lands under the staged directory
    for p in (PPG.PROPS_POINTER_FILE, PPG.challenger_path("goals"), PPG.challenger_meta_path("goals")):
        assert rel(p).startswith(rel(PPG.REGISTRY_DIR) + "/")
    # nothing under the props registry is matched by the team registry's manifest glob
    assert rel(PPG._manifest_path("goals", "v")).count("/") > rel(reg.REGISTRY_DIR).count("/") + 2


# ── end to end: the retrain job still completes ───────────────────────────────────────────────
def _synthetic_history(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    feats = [c for c in pconfig.PLAYER_FEATURE_COLS][:6]
    X = rng.normal(size=(n, len(feats)))
    d = pd.DataFrame(X, columns=feats)
    d["n_prev_games"] = 5
    d["date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) // 20, unit="D")
    d["league"] = np.where(np.arange(n) % 2, "La Liga", "Serie A")
    d["fixture_id"] = np.arange(n) // 20 + 1
    d["player_id"] = np.arange(n) % 300 + 1
    for m, t in pconfig.MARKET_TARGETS.items():
        z = X[:, 0] - 1.2 + rng.normal(0, 1, n)
        d[t] = (z > 0.3).astype(int)                                      # ~15% positives
    return d


def test_props_retrain_job_completes_and_changes_no_serving_file(ptmp, monkeypatch):
    from player_model import pipeline as PP
    hist = ptmp / "player_history.parquet"
    _synthetic_history().to_parquet(hist)
    monkeypatch.setattr(PP, "HISTORY_CACHE", hist)
    monkeypatch.setattr(PP.config, "OUTPUT_DIR", ptmp / "out")
    for m in pconfig.MARKETS:
        _serve(m, col=pconfig.PLAYER_FEATURE_COLS[0])
    before = {m: f.read_bytes() for m, f in pconfig.MODEL_FILES.items()}
    PP.mode_train()
    blob = json.loads((ptmp / "out" / "props_retrain_log.json").read_text(encoding="utf-8"))
    run = blob["runs"][blob["latest_run"]]
    assert set(run) == set(pconfig.MARKETS)
    assert all(v["gate_version"] == "PG-PROPS-1.0" for v in run.values())
    assert not any(v["promoted"] for v in run.values())                  # no prospective period
    assert all(v["verdict"] in (PPG.KEEP, PPG.SHADOW) for v in run.values()), run
    # the comparison really ran: champion and candidate scored on the same >= 1000 holdout rows
    assert all(v["holdout_n"] >= 1000 and v["ll_delta"] is not None for v in run.values()), run
    assert {m: f.read_bytes() for m, f in pconfig.MODEL_FILES.items()} == before
    log = pd.read_csv(PPG.DECISIONS_FILE)
    assert (log["stage"] == "HOLDOUT").sum() == len(pconfig.MARKETS)
    assert (log["stage"] == "REGISTER").sum() == len(pconfig.MARKETS)
    # a second run the same day gets its own key instead of overwriting the first
    PP.mode_train()
    blob = json.loads((ptmp / "out" / "props_retrain_log.json").read_text(encoding="utf-8"))
    assert len(blob["runs"]) == 2 and {m: f.read_bytes() for m, f in pconfig.MODEL_FILES.items()} == before
