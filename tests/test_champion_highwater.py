"""Champion high-water mark (owner 2026-10-18): identity, lineage, attribution.

The gate itself is tested in tests/test_promotion_gate.py (owner 2026-10-25).

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
from src import promotion_gate as PG  # noqa: E402


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
    PG._install_champion("standard", m, raw, reason="test")
    assert reg.MODEL_FILES["standard"].read_bytes() == raw
    assert reg.read_pointer()["standard"]["model_version"] == m["model_version"]
    stored = reg.champion_artifact_path("standard", m["model_version"])
    assert gzip.decompress(stored.read_bytes()) == raw
    assert reg.read_manifest("standard", m["model_version"])["model_role"] == "CHAMPION"
    assert reg.stamp_for("standard")["model_version"] == m["model_version"]


def test_challenger_never_touches_serving_file(tmpreg):
    champ = b"champion-bytes"
    PG._install_champion("over15", _manifest("over15", champ), champ, reason="t")
    chal = b"challenger-bytes"
    reg.set_challenger("over15", _manifest("over15", chal,
                                           datetime(2026, 10, 25, tzinfo=timezone.utc)), chal)
    assert reg.MODEL_FILES["over15"].read_bytes() == champ
    assert reg.read_pointer()["over15"]["artifact_sha256"] == reg.sha256_bytes(champ)


def test_history_append_only_and_rollback(tmpreg):
    a, b = b"model-a", b"model-b"
    ma = _manifest("btts", a, datetime(2026, 10, 1, tzinfo=timezone.utc))
    mb = _manifest("btts", b, datetime(2026, 10, 8, tzinfo=timezone.utc))
    PG._install_champion("btts", ma, a, reason="first")
    first_line = reg.CHAMPION_HISTORY.read_text(encoding="utf-8").splitlines()[1]
    PG._install_champion("btts", mb, b, reason="second")
    lines = reg.CHAMPION_HISTORY.read_text(encoding="utf-8").splitlines()
    assert lines[1] == first_line and len(lines) == 3          # header + 2, first untouched
    # rollback restores A's exact bytes without retraining
    PG.rollback("btts", ma["model_version"])
    assert reg.MODEL_FILES["btts"].read_bytes() == a
    assert reg.read_pointer()["btts"]["model_version"] == ma["model_version"]
    rows = list(csv.DictReader(open(reg.CHAMPION_HISTORY, encoding="utf-8")))
    assert rows[-1]["promotion_source"] == "ROLLBACK" and len(rows) == 3


def test_no_model_mixing(tmpreg):
    s, n = b"std", b"bt1"
    PG._install_champion("standard", _manifest("standard", s), s, reason="t")
    v_std = reg.read_pointer()["standard"]["model_version"]
    PG._install_champion("btts", _manifest("btts", n), n, reason="t")
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


# ── attribution ───────────────────────────────────────────────────────────────────────────────
def test_ledger_rows_carry_model_version(tmpreg, tmp_path, monkeypatch):
    from src import ledger as L
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    raw = b"std-model"
    PG._install_champion("standard", _manifest("standard", raw), raw, reason="t")
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
    PG._install_champion("btts", _manifest("btts", champ), champ, reason="t")
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
