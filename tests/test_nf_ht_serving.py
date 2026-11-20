"""New-Format half-time O/U models connected to live predictions (owner 2026-11-20: "only this and that's it").

Pins that
  * predict scores New-Format rows with the New-Format HT payloads (models/model_v9_ht_over05/15_nf.pkl) and
    Standard rows with the Standard HT payloads, each track with ITS OWN file only (a missing NF file leaves
    the NF HT probability empty - never the Standard model), and Standard HT is unchanged;
  * the board's model_version_ht_over05/15 names the New-Format HT model on New-Format rows and the Standard
    HT model on Standard rows; the head-to-head log attributes each row to its own model type;
  * New-Format HT is PREDICTION LEDGER ONLY: no ht_ledger tip row, no ht_observations row, no Telegram HT tip
    without a captured price AND an approved New-Format HT bar (none exists), no TOP-3 market, no stake;
  * the two NF HT model types are registered (serving files only the promotion gate writes).
Synthetic boards and tmp files only; no committed output is touched.
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

STD_LG, NF_LG = "Championship", "Brazil Serie A"
P = {"std05": 0.71, "std15": 0.33, "nf05": 0.64, "nf15": 0.27, "ou": 0.55}


def _board():
    day = pd.Timestamp("2026-11-22")
    rows = [(STD_LG, "Norwich City", "Cardiff City"), (NF_LG, "Flamengo", "Palmeiras")]
    return pd.DataFrame([{"league": lg, "date": day, "kickoff_utc": day + pd.Timedelta(hours=15),
                          "home_team": h, "away_team": a, "odds_over25": 2.0, "odds_under25": 1.85}
                         for lg, h, a in rows])


@pytest.fixture()
def serve(monkeypatch, tmp_path):
    import src.feature_engineering as FE
    import src.ht_features as HTF
    import src.model as M
    import src.predict as PR

    files = {"HT_MODEL_FILE_05": "std05", "HT_MODEL_FILE_15": "std15",
             "HT_MODEL_FILE_05_NF": "nf05", "HT_MODEL_FILE_15_NF": "nf15"}
    by_path = {}
    for attr, tag in files.items():
        f = tmp_path / f"{tag}.pkl"
        f.write_bytes(b"x")
        monkeypatch.setattr(config, attr, f)
        by_path[str(f)] = {"feature_cols": [], "tag": tag}

    def _feat(upcoming, historical, **_kw):
        f = upcoming.copy().reset_index(drop=True)
        for c in ("home_scored_last5", "away_scored_last5"):
            f[c] = 1.4
        return f

    def _ht(historical, feat, **_kw):
        out = feat[["home_team", "away_team"]].copy()
        for c in HTF.HT_FEATURE_COLS:
            out[c] = 0.5
        return out

    monkeypatch.setattr(FE, "build_upcoming_features", _feat)
    monkeypatch.setattr(HTF, "attach_ht_features", _ht)
    monkeypatch.setattr(M, "load_models", lambda model_file=None, **k: by_path[str(model_file)])
    monkeypatch.setattr(M, "predict_proba",
                        lambda X, payload=None, **k: pd.Series(P.get(payload.get("tag"), P["ou"]), index=X.index))

    def _run():
        return PR.serve_fixtures(_board(), pd.DataFrame(), {"feature_cols": [], "tag": "ou"},
                                 {"feature_cols": [], "tag": "ou"}, None, record_health=False)
    return _run, tmp_path, monkeypatch


def test_each_track_is_scored_by_its_own_ht_model(serve):
    run, _, _ = serve
    out = run().set_index("league")
    assert out.at[STD_LG, "model_type"] == "standard" and out.at[NF_LG, "model_type"] == "new_format"
    assert out.at[STD_LG, "p_ht_over05"] == P["std05"] and out.at[STD_LG, "p_ht_over15"] == P["std15"]
    assert out.at[NF_LG, "p_ht_over05"] == P["nf05"] and out.at[NF_LG, "p_ht_over15"] == P["nf15"]


def test_missing_nf_file_leaves_nf_ht_empty_and_standard_unchanged(serve):
    run, tmp, mp = serve
    mp.setattr(config, "HT_MODEL_FILE_05_NF", tmp / "absent.pkl")
    out = run().set_index("league")
    assert pd.isna(out.at[NF_LG, "p_ht_over05"]) and pd.isna(out.at[NF_LG, "p_ht_over15"])
    assert out.at[STD_LG, "p_ht_over05"] == P["std05"] and out.at[STD_LG, "p_ht_over15"] == P["std15"]


def test_missing_standard_file_never_borrows_the_nf_model(serve):
    run, tmp, mp = serve
    mp.setattr(config, "HT_MODEL_FILE_05", tmp / "absent.pkl")
    out = run().set_index("league")
    assert pd.isna(out.at[STD_LG, "p_ht_over05"]) and pd.isna(out.at[STD_LG, "p_ht_over15"])
    assert out.at[NF_LG, "p_ht_over05"] == P["nf05"]


# ── attribution ────────────────────────────────────────────────────────────────────────────────────

def _fake_stamp(m, path=None):
    return {"model_version": f"V_{m}", "retrain_batch_id": "", "training_cutoff": "", "prediction_role": "CHAMPION"}


def test_board_version_columns_name_the_track_model(monkeypatch):
    from src import model_registry as reg
    from src import provenance
    monkeypatch.setattr(reg, "stamp_for", _fake_stamp)
    df = provenance.stamp_versions(pd.DataFrame({"model_type": ["standard", "new_format", "unknown"]}))
    assert df["model_version_ht_over05"].tolist() == ["V_ht_over05", "V_ht_over05_nf", "V_ht_over05"]
    assert df["model_version_ht_over15"].tolist() == ["V_ht_over15", "V_ht_over15_nf", "V_ht_over15"]
    assert set(df["model_version_btts"]) == {"V_btts"}                       # other markets unchanged
    std_only = provenance.stamp_versions(pd.DataFrame({"model_type": ["standard"]}))
    assert std_only["model_version_ht_over05"].tolist() == ["V_ht_over05"]


def test_registry_knows_the_nf_ht_model_types():
    from src import model as M
    from src import model_registry as reg
    for mt, f, tgt in (("ht_over05_nf", config.HT_MODEL_FILE_05_NF, "ht_over05"),
                       ("ht_over15_nf", config.HT_MODEL_FILE_15_NF, "ht_over15")):
        assert reg.MODEL_FILES[mt] == f and reg.TARGET[mt] == tgt
        assert reg.SERVES_TRACK[mt] == "new_format"
        with pytest.raises(PermissionError):                                  # serving file: gate only
            M.save_models({}, model_file=f)
    assert reg.SERVES_TRACK["ht_over05"] == reg.SERVES_TRACK["ht_over15"] == "standard"


def test_h2h_logs_each_ht_row_under_its_own_model(monkeypatch, tmp_path):
    import pipeline
    from src import model_registry as reg
    monkeypatch.setattr(pipeline, "H2H_FILE", tmp_path / "h2h.csv")
    monkeypatch.setattr(reg, "stamp_for", _fake_stamp)
    monkeypatch.setattr(reg, "load_challenger", lambda mt: None)
    preds = pd.DataFrame({"date": ["2026-11-22", "2026-11-22"], "league": [STD_LG, NF_LG],
                          "home_team": ["A", "B"], "away_team": ["C", "D"],
                          "model_type": ["standard", "new_format"],
                          "p_ht_over05": [0.7, 0.6], "p_ht_over15": [0.3, 0.2]})
    pipeline._log_h2h(preds)
    d = pd.read_csv(tmp_path / "h2h.csv", dtype=str)
    got = {(r.model_type, r.league, r.champion_version) for r in d.itertuples()}
    assert got == {("ht_over05", STD_LG, "V_ht_over05"), ("ht_over15", STD_LG, "V_ht_over15"),
                   ("ht_over05_nf", NF_LG, "V_ht_over05_nf"), ("ht_over15_nf", NF_LG, "V_ht_over15_nf")}


# ── ledger only: no tip, no Telegram, no TOP-3, no stake ───────────────────────────────────────────

def _preds():
    return pd.DataFrame({"date": ["2026-11-22", "2026-11-22"], "league": [STD_LG, NF_LG],
                         "home_team": ["A", "B"], "away_team": ["C", "D"],
                         "model_type": ["standard", "new_format"],
                         "p_ht_over05": [0.9, 0.95], "p_ht_over15": [0.7, 0.8]})


def test_ht_tip_and_observation_logs_stay_standard_only(monkeypatch, tmp_path):
    import src.ledger as L
    monkeypatch.setattr(L, "HT_LEDGER_FILE", tmp_path / "ht_ledger.csv")
    monkeypatch.setattr(L, "HT_OBS_FILE", tmp_path / "ht_observations.csv")
    monkeypatch.setattr(L, "_ht_price_lookup", lambda: None)
    L.append_ht_tips(_preds())
    L.append_ht_observations(_preds())
    assert pd.read_csv(tmp_path / "ht_ledger.csv")["league"].tolist() == [STD_LG]
    assert pd.read_csv(tmp_path / "ht_observations.csv")["league"].tolist() == [STD_LG]
    nf_only = _preds().iloc[[1]]
    (tmp_path / "ht_ledger.csv").unlink()
    L.append_ht_tips(nf_only)
    assert not (tmp_path / "ht_ledger.csv").exists()


def test_no_approved_nf_ht_bar_exists():
    from telegram_bot import notifier as N
    side = json.loads((config.MODELS_DIR / "best_params_side_markets.json").read_text(encoding="utf-8"))
    for key in N.HT_PARAMS_KEY.values():
        assert not set((side.get(key) or {})) & set(config.NEW_FORMAT_LEAGUES), key
    for lg in config.NEW_FORMAT_LEAGUES:
        for line in ("0.5", "1.5"):
            for sd in ("OVER", "UNDER"):
                cell, why = N.ht_bar(lg, line, sd)
                assert cell is None and why


def test_nf_ht_is_never_a_sendable_telegram_candidate():
    from telegram_bot import notifier as N
    df = _preds().iloc[[1]]
    # with a price on every market-side, the real (absent) New-Format HT bars block every candidate
    priced = N.ht_tip_candidates(df, price_fn=lambda *a: 1.9)
    assert len(priced) == 4 and all(c["reason"].startswith(N.HT_SKIP_NO_BAR) for c in priced)
    # with no captured price nothing reaches the bar check
    unpriced = N.ht_tip_candidates(df, price_fn=lambda *a: None)
    assert len(unpriced) == 4 and all(c["reason"] == N.HT_SKIP_NO_PRICE for c in unpriced)
    # an approved bar alone is not enough either: the price is required
    cell = {"stub": True}
    bar_only = N.ht_tip_candidates(df, price_fn=lambda *a: None, bar_fn=lambda *a: (cell, ""))
    assert all(c["reason"] == N.HT_SKIP_NO_PRICE for c in bar_only)


def test_ht_is_no_top3_market_and_never_staked():
    from src import staked, top3_league
    assert not [m for m in top3_league.MARKETS if "HT" in m.upper()]
    assert not [v for v in top3_league.MARKETS.values() if any("ht" in x for x in v)]
    assert staked.LEDGER_MARKET["ht_ledger.csv"] == (None, None)            # no bankroll-policy market


def test_nf_ht_never_enters_the_bets_frames():
    # bets.csv (O/U 2.5) and side_bets (BTTS / O1.5 / O3.5) are built from their own probability columns
    src = (ROOT / "src" / "betting.py").read_text(encoding="utf-8")
    assert "p_ht_over" not in src
    pipe = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    side = pipe[pipe.index("def _generate_side_bets"):]
    side = side[:side.index("\ndef ", 10)]
    assert "p_ht_over" not in side
