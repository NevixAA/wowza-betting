"""Champion high-water mark (owner 2026-10-18): identity, gate, lineage, attribution.

Every test runs against a temporary registry — nothing here touches models/ or output/.
"""
import csv
import gzip
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import model_registry as reg  # noqa: E402
from src import champion_gate as cg  # noqa: E402


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
    reg._version_cache.clear()
    yield tmp_path
    reg._version_cache.clear()


def _manifest(mt, raw, t=datetime(2026, 10, 18, 8, 0, tzinfo=timezone.utc)):
    sha = reg.sha256_bytes(raw)
    return {"model_type": mt, "model_version": reg.make_version(mt, t, sha),
            "artifact_sha256": sha, "retrain_batch_id": "rb_test", "training_cutoff": "2026-10-01"}


# ── identity ──────────────────────────────────────────────────────────────────────────────────
def test_version_ids_unique_and_deterministic():
    t = datetime(2026, 10, 18, 8, 4, 23, tzinfo=timezone.utc)
    a = reg.make_version("standard", t, "a" * 64)
    assert a == reg.make_version("standard", t, "a" * 64) == "standard_20261018T080423Z_aaaaaaaa"
    assert a != reg.make_version("standard", t, "b" * 64)
    assert a != reg.make_version("newformat", t, "a" * 64)


def test_manifest_is_immutable(tmpreg):
    m = _manifest("btts", b"one")
    reg.write_manifest(m)
    reg.write_manifest(dict(m))                      # identical rewrite is a no-op
    with pytest.raises(ValueError):
        reg.write_manifest({**m, "artifact_sha256": reg.sha256_bytes(b"two")})


# ── champion pointer, artifacts, history ──────────────────────────────────────────────────────
def test_promotion_is_atomic_bundle(tmpreg):
    raw = pickle.dumps({"models": {}, "feature_cols": ["x"]})
    m = _manifest("standard", raw)
    reg.set_champion("standard", m, raw, reason="test")
    assert reg.MODEL_FILES["standard"].read_bytes() == raw
    assert reg.read_pointer()["standard"]["model_version"] == m["model_version"]
    stored = reg.champion_artifact_path("standard", m["model_version"])
    assert gzip.decompress(stored.read_bytes()) == raw
    assert reg.read_manifest("standard", m["model_version"])["model_role"] == "CHAMPION"
    assert reg.stamp_for("standard")["model_version"] == m["model_version"]


def test_challenger_never_touches_serving_file(tmpreg):
    champ = b"champion-bytes"
    reg.set_champion("over15", _manifest("over15", champ), champ, reason="t")
    chal = b"challenger-bytes"
    reg.set_challenger("over15", _manifest("over15", chal,
                                           datetime(2026, 10, 25, tzinfo=timezone.utc)), chal)
    assert reg.MODEL_FILES["over15"].read_bytes() == champ
    assert reg.read_pointer()["over15"]["artifact_sha256"] == reg.sha256_bytes(champ)


def test_history_append_only_and_rollback(tmpreg):
    a, b = b"model-a", b"model-b"
    ma = _manifest("btts", a, datetime(2026, 10, 1, tzinfo=timezone.utc))
    mb = _manifest("btts", b, datetime(2026, 10, 8, tzinfo=timezone.utc))
    reg.set_champion("btts", ma, a, reason="first")
    first_line = reg.CHAMPION_HISTORY.read_text(encoding="utf-8").splitlines()[1]
    reg.set_champion("btts", mb, b, reason="second")
    lines = reg.CHAMPION_HISTORY.read_text(encoding="utf-8").splitlines()
    assert lines[1] == first_line and len(lines) == 3          # header + 2, first untouched
    # rollback restores A's exact bytes without retraining
    reg.rollback("btts", ma["model_version"])
    assert reg.MODEL_FILES["btts"].read_bytes() == a
    assert reg.read_pointer()["btts"]["model_version"] == ma["model_version"]
    rows = list(csv.DictReader(open(reg.CHAMPION_HISTORY, encoding="utf-8")))
    assert rows[-1]["promotion_source"] == "ROLLBACK" and len(rows) == 3


def test_no_model_mixing(tmpreg):
    s, n = b"std", b"bt1"
    reg.set_champion("standard", _manifest("standard", s), s, reason="t")
    v_std = reg.read_pointer()["standard"]["model_version"]
    reg.set_champion("btts", _manifest("btts", n), n, reason="t")
    assert reg.read_pointer()["standard"]["model_version"] == v_std
    assert reg.MODEL_FILES["standard"].read_bytes() == s


def test_unregistered_file_gets_real_hash_not_invented_version(tmpreg):
    f = reg.MODEL_FILES["over35"]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"mystery")
    v = reg.stamp_for("over35")["model_version"]
    assert v == f"UNREGISTERED_over35_{reg.sha256_bytes(b'mystery')[:12]}"


def test_ou_routing_mirrors_predict_fallbacks(tmpreg):
    assert reg.ou_model_for("standard") == "standard"
    assert reg.ou_model_for("unknown") == "standard"            # predict.py:339 fallback
    assert reg.ou_model_for("new_format") == "standard"         # NF file missing -> standard
    reg.MODEL_FILES["newformat"].parent.mkdir(parents=True, exist_ok=True)
    reg.MODEL_FILES["newformat"].write_bytes(b"nf")
    assert reg.ou_model_for("new_format") == "newformat"


# ── the gate ──────────────────────────────────────────────────────────────────────────────────
def _data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < truth).astype(float)
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    return y, truth, dates


def test_worse_challenger_keeps_champion():
    y, truth, d = _data()
    good = truth
    worse = np.clip(truth + np.random.default_rng(2).normal(0, 0.08, len(y)), 0.02, 0.98)
    v = cg.decide(y, good, worse, dates=d)
    assert v["verdict"] == cg.KEEP and v["ll_delta"] > 0


def test_slightly_worse_inside_old_tolerance_still_keeps():
    """The 2026-09-24 newformat case: +0.00119 used to promote inside the tolerance."""
    y, truth, d = _data()
    worse = np.clip(truth * 0.985 + 0.0075, 0.02, 0.98)
    v = cg.decide(y, truth, worse, dates=d)
    assert v["verdict"] == cg.KEEP


def test_tied_challenger_not_promoted():
    y, truth, d = _data()
    tiny = np.clip(truth + 1e-5, 0.02, 0.98)
    v = cg.decide(y, truth, tiny, dates=d)
    assert v["verdict"] != cg.PROMOTE
    v2 = cg.decide(y, truth, truth.copy(), dates=d)
    assert v2["verdict"] == cg.KEEP


def test_better_challenger_promotes():
    y, truth, d = _data(n=6000)
    blurred = np.clip(0.5 + (truth - 0.5) * 0.3, 0.02, 0.98)   # champion knows less
    v = cg.decide(y, blurred, truth, dates=d)
    assert v["verdict"] == cg.PROMOTE, v["reason"]
    assert v["ll_ci"][1] < 0 and v["ll_delta"] <= -cg.MATERIAL_LL


def test_unproven_gain_is_shadowed():
    y, truth, d = _data(n=1500, seed=4)
    slightly = np.clip(0.5 + (truth - 0.5) * 0.85, 0.02, 0.98)
    v = cg.decide(y, slightly, truth, dates=d)
    assert v["verdict"] in (cg.SHADOW, cg.PROMOTE)
    # force the unproven case: a gain whose CI must cross zero on a small sample
    y2, t2, d2 = _data(n=1100, seed=9)
    ch = np.clip(t2 + np.random.default_rng(3).normal(0, 0.03, len(y2)), 0.02, 0.98)
    v2 = cg.decide(y2, ch, t2, dates=d2)
    if v2["ll_ci"][1] >= 0 and v2["ll_delta"] < 0 and v2["ll_ci"][0] <= -cg.MATERIAL_LL:
        assert v2["verdict"] == cg.SHADOW


def test_league_regression_blocks_promotion():
    y, truth, d = _data(n=6000)
    blurred = np.clip(0.5 + (truth - 0.5) * 0.3, 0.02, 0.98)
    lg = np.array(["A"] * 5000 + ["B"] * 1000)
    ch = truth.copy()
    ch[5000:] = np.clip(1 - truth[5000:], 0.02, 0.98)           # destroys league B
    v = cg.decide(y, blurred, ch, leagues=lg, dates=d)
    assert v["verdict"] != cg.PROMOTE


def test_comparison_requires_same_observations():
    y, truth, d = _data()
    v = cg.decide(y, truth, truth[:-5], dates=d)
    assert v["verdict"] == cg.KEEP and "EVALUATION_FAILED" in v["reason"]
    v2 = cg.decide(y, truth, truth, same_rows=False)
    assert v2["verdict"] == cg.KEEP and "EVALUATION_FAILED" in v2["reason"]


def test_failed_evaluation_keeps_champion():
    y, truth, d = _data(n=200)
    assert cg.decide(y, truth, truth * 0.5, dates=d)["verdict"] == cg.KEEP     # n too small
    v = cg.decide(["a", "b"], [0.1, 0.2], [0.3, 0.4])                       # garbage
    assert v["verdict"] == cg.KEEP and "EVALUATION_FAILED" in v["reason"]


def test_bootstrap_promotes_only_when_no_champion_exists():
    assert cg.decide([], [], [], incumbent_exists=False)["verdict"] == cg.PROMOTE


# ── attribution ───────────────────────────────────────────────────────────────────────────────
def test_ledger_rows_carry_model_version(tmpreg, tmp_path, monkeypatch):
    from src import ledger as L
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    raw = b"std-model"
    reg.set_champion("standard", _manifest("standard", raw), raw, reason="t")
    bets = pd.DataFrame([{"date": "2026-10-20", "home_team": "A", "away_team": "B",
                          "bet": "OVER", "odds_over25": 2.0, "odds_under25": 1.8,
                          "best_edge": 0.05, "signal_tier": "VALUABLE", "league": "X",
                          "model_type": "standard"}])
    L.append_tips(bets)
    led = pd.read_csv(tmp_path / "bets_ledger.csv")
    assert led.loc[0, "model_version"] == reg.read_pointer()["standard"]["model_version"]
    assert led.loc[0, "prediction_role"] == "CHAMPION"


def test_legacy_ledger_rows_stay_readable(tmpreg, tmp_path, monkeypatch):
    from src import ledger as L
    f = tmp_path / "bets_ledger.csv"
    pd.DataFrame([{c: "" for c in L.LEDGER_COLS[:20]}]).to_csv(f, index=False)  # old schema
    monkeypatch.setattr(L, "LEDGER_FILE", f)
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    bets = pd.DataFrame([{"date": "2026-10-21", "home_team": "C", "away_team": "D",
                          "bet": "UNDER", "odds_over25": 2.0, "odds_under25": 1.8,
                          "best_edge": 0.04, "signal_tier": "VALUABLE", "league": "X",
                          "model_type": "standard"}])
    L.append_tips(bets)
    led = pd.read_csv(f)
    assert len(led) == 2 and pd.isna(led.loc[0, "model_version"])     # legacy stays blank
    assert str(led.loc[1, "model_version"]).startswith(("UNKNOWN", "UNREGISTERED"))


def test_h2h_shadow_preserves_challenger_version(tmpreg, tmp_path, monkeypatch):
    import pipeline as P
    champ, chal = b"c", b"ch"
    reg.set_champion("btts", _manifest("btts", champ), champ, reason="t")
    mch = _manifest("btts", chal, datetime(2026, 10, 25, tzinfo=timezone.utc))
    monkeypatch.setattr(reg, "load_challenger",
                        lambda mt: ({"x": 1}, mch) if mt == "btts" else None)
    monkeypatch.setattr(P, "load_models", lambda model_file=None: {"x": 0})
    monkeypatch.setattr(P, "model_predict_proba",
                        lambda df, payload=None: np.full(len(df), 0.6 if payload == {"x": 1}
                                                         else 0.5))
    monkeypatch.setattr(P, "H2H_FILE", tmp_path / "h2h.csv")
    preds = pd.DataFrame([{"date": "2026-10-26", "league": "X", "home_team": "A",
                           "away_team": "B", "model_type": "standard", "p_btts": 0.52}])
    P._log_h2h(preds)
    P._log_h2h(preds)                                         # second run: first sight holds
    h = pd.read_csv(tmp_path / "h2h.csv")
    row = h[h.model_type == "btts"]
    assert len(row) == 1
    assert row.iloc[0]["challenger_version"] == mch["model_version"]
    assert row.iloc[0]["p_challenger"] == 0.6 and row.iloc[0]["p_champion_raw"] == 0.5
    assert row.iloc[0]["champion_version"] == reg.read_pointer()["btts"]["model_version"]


def test_champion_decision_writes_lineage_and_only_promotes_on_evidence(tmpreg, monkeypatch):
    """End to end through pipeline._champion_decision with stubbed predictions."""
    import pipeline as P
    y, truth, d = _data(n=3000)
    test_df = pd.DataFrame({"date": d, "league": "L", "over25": y})
    valid = test_df.copy()
    old = b"old-champion"
    reg.set_champion("standard", _manifest("standard", old), old, reason="seed")
    results = {"lr": {"model": "m", "feature_cols": ["f"], "metrics": {"log_loss": 0.69}}}

    def run(p_new):
        monkeypatch.setattr(P, "model_predict_proba",
                            lambda df, payload=None: truth if payload == "OLD" else p_new)
        return P._champion_decision("standard", reg.MODEL_FILES["standard"], "OLD", results,
                                    test_df, "over25", "same_holdout", valid)

    worse = np.clip(truth + 0.05, 0.02, 0.98)
    gate, extra = run(worse)
    assert gate["verdict"] == cg.KEEP
    assert reg.MODEL_FILES["standard"].read_bytes() == old           # champion untouched
    assert reg.read_manifest("standard", extra["challenger_version"])["model_role"] == "REJECTED"
    rb = pd.read_csv(reg.RETRAIN_BATCHES)
    assert rb.iloc[-1]["verdict"] == cg.KEEP and not rb.iloc[-1]["promoted"]

    # incumbent cannot be scored -> fail closed
    gate2, _ = P._champion_decision("standard", reg.MODEL_FILES["standard"], None, results,
                                    test_df, "over25", "stored_metrics_FALLBACK", valid)
    assert gate2["verdict"] == cg.KEEP and "EVALUATION_FAILED" in gate2["reason"]
    assert reg.MODEL_FILES["standard"].read_bytes() == old
