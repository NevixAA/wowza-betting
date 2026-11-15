"""Phase 2 / P0 train-serve parity - CURRENT-BOARD RESCORE (not a historical replay).

What it measures: the effect of the rolling-form serving fix (main 1ea46192, carried on
data-truth-100) on ONE committed board - output/predictions.csv - scored with the committed models.

Two arms, everything identical except the rolling-form window:
  NEW  = data-truth-100 build_upcoming_features (serving form = training's window)
  OLD  = the same output with every rolling-form column (and the columns computed only from them)
         replaced by what build_upcoming_features at 1ea46192^ (the pre-fix code, read with
         `git show`) computes on the SAME history and the SAME fixtures.
Both arms then go through src.predict.predict_upcoming unchanged (routing, Standard v9_repaired
serving, NF calibration, side markets, HT, evaluate_value, drift), with the committed models.

Offline by construction: the network is blocked (socket.connect raises), the football-data
download is skipped (the committed output/fd_history.parquet cache is the history), the odds
fetch returns the committed board's own fetch columns, and the odds-snapshot / purge / fetch-health
writers are stubbed. Any tracked file the run touches is restored by the caller (git checkout).

    python scripts/p2_parity_board_rescore.py <out_dir>
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GIT = r"C:\Program Files\Git\cmd\git.exe"
OLD_REF = "1ea46192^"


def _no_net(*a, **k):
    raise OSError("network blocked by p2_parity_board_rescore (offline rescore)")


socket.socket.connect = _no_net            # nothing below may reach the network
socket.create_connection = _no_net

# The predict step's env in .github/workflows/predict.yml (data-truth-100), so serving matches production.
import os                                   # noqa: E402
for _k, _v in {"REQUIRE_FORM_DATA": "0", "REQUIRE_FORM_DATA_UNTIL": "2026-09-15",
               "STANDARD_OU25_SERVING_RECIPE": "v9_repaired", "LEAGUE_SNIPER_CAP": "0.12",
               "MARKSMAN_THRESHOLD": "0.08", "VALUABLE_THRESHOLD": "0.03"}.items():
    os.environ[_k] = _v

import config                               # noqa: E402
from src import feature_engineering as FE  # noqa: E402
from src import data_loader as DL          # noqa: E402
from src import predict as P               # noqa: E402

FETCH_COLS = ["league", "date", "kickoff_utc", "home_team", "away_team", "odds_over25", "odds_under25",
              "best_over25", "best_under25", "odds_over15", "odds_over35", "odds_btts", "_oa_event_id",
              "_oa_sport_key", "odds_btts_no", "btts_books", "btts_overround", "btts_fair_yes"]
FORM_COLS = [c for pair in FE._FORM_FEATURES.values() for c in pair]
DERIVED = ["home_attack_str", "away_attack_str", "home_defense_str", "away_defense_str",
           "combined_ht_goals_avg", "home_ht_attack_str", "away_ht_attack_str", "home_ht_defense_str",
           "away_ht_defense_str", "home_sot_ratio_last5", "away_sot_ratio_last5", "combined_sot_ratio",
           "p_over25_poisson_dc"]
OUT_COLS = ["p_over25", "p_over25_raw", "p_btts", "p_over15", "p_over35", "p_ht_over05", "p_ht_over15",
            "edge_over", "edge_under", "tier_over", "tier_under", "bet", "best_side", "signal_tier",
            "stake_over", "stake_under", "bet_stake"]


def _old_module():
    src = subprocess.run([GIT, "-C", str(ROOT), "show", f"{OLD_REF}:src/feature_engineering.py"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    sha = subprocess.run([GIT, "-C", str(ROOT), "rev-parse", OLD_REF], capture_output=True, text=True,
                         check=True).stdout.strip()
    tmp = Path(sys.argv[1]) / "fe_pre_fix.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("fe_pre_fix", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    board = pd.read_csv(ROOT / "output" / "predictions.csv")
    fetch = board[FETCH_COLS].copy()
    fetch["date"] = pd.to_datetime(fetch["date"])        # the live fetch hands Timestamps, the CSV text

    DL._ci_download_all = lambda have=None: []          # committed cache only, no download
    hist = DL.load_all_matches(force=True)

    old_mod, old_sha = _old_module()
    new_build = FE.build_upcoming_features
    captured = {}

    def _old_arm(upcoming, historical, *a, **k):
        new = new_build(upcoming, historical, *a, **k)
        old = old_mod.build_upcoming_features(upcoming, historical, *a, **k)
        swap = [c for c in FORM_COLS + DERIVED if c in old.columns]
        new[swap] = old[swap].to_numpy()
        captured["old_feat"], captured["swapped"] = new.copy(), swap
        return new

    def _new_arm(upcoming, historical, *a, **k):
        new = new_build(upcoming, historical, *a, **k)
        captured["new_feat"] = new.copy()
        return new

    P._fetch_odds_api = lambda days_ahead=7: fetch.copy()
    P._detect_postponed = lambda up: []
    import src.drift as _drift
    _drift.snapshot = lambda up: None
    _drift.purge_old = lambda cutoff_days=10: None
    P.attach_fixture_ids = lambda df, fixtures_for=None: df

    from src.model import load_models
    pay_std = load_models(model_file=config.MODEL_FILE_STANDARD)
    pay_nf = load_models(model_file=config.MODEL_FILE_NEWFORMAT)
    side = {t: load_models(model_file=f) for t, f in config.SIDE_MARKETS.items() if f.exists()}

    res = {}
    for arm, fn in (("new", _new_arm), ("old", _old_arm)):
        FE.build_upcoming_features = fn
        preds, _ = P.predict_upcoming(hist, pay_std, payload_newformat=pay_nf, side_payloads=side, days_ahead=7)
        res[arm] = preds.reset_index(drop=True)
    FE.build_upcoming_features = new_build

    new, old = res["new"], res["old"]
    assert len(new) == len(old) == len(board)
    assert (new["home_team"].to_numpy() == old["home_team"].to_numpy()).all()
    key = new[["league", "date", "home_team", "away_team", "model_type"]].copy()

    rows = []
    for c in FORM_COLS + DERIVED + OUT_COLS:
        if c not in new.columns:
            continue
        a, b = old[c], new[c]
        num = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
        if num:
            both_nan = a.isna() & b.isna()
            diff = (a - b).abs()
            changed = ~both_nan & ((a.isna() != b.isna()) | (diff > 1e-9))
        else:
            changed = a.astype(str) != b.astype(str)
        for mt in ("new_format", "standard"):
            m = key["model_type"] == mt
            r = {"column": c, "track": mt, "fixtures": int(m.sum()), "changed": int((changed & m).sum())}
            if num:
                d = diff[m & changed & a.notna() & b.notna()]
                r.update({"nan_flips": int((m & (a.isna() != b.isna())).sum()),
                          "mean_abs_change": round(float(d.mean()), 4) if len(d) else 0.0,
                          "max_abs_change": round(float(d.max()), 4) if len(d) else 0.0})
            rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "column_summary.csv", index=False)

    per = key.copy()
    for c in ("home_scored_last5", "away_scored_last5", "p_over25", "p_btts", "p_over15", "p_over35",
              "edge_over", "edge_under", "tier_over", "tier_under", "best_side", "bet", "signal_tier"):
        if c in new.columns:
            per[f"{c}_old"], per[f"{c}_new"] = old[c].to_numpy(), new[c].to_numpy()
    per.to_csv(out / "per_fixture.csv", index=False)

    # Does the OLD arm reproduce what production served? (committed board = pre-fix code)
    repro = {}
    for c in FORM_COLS:
        if c in board.columns and c in old.columns:
            a, b = pd.to_numeric(board[c], errors="coerce"), pd.to_numeric(old[c], errors="coerce")
            same = (a.isna() & b.isna()) | ((a - b).abs() <= 1e-6)
            repro[c] = int(same.sum())
    meta = {"board_rows": int(len(board)), "board_generated_at": str(board["generated_at"].iloc[0]),
            "board_git_sha": str(board["git_sha"].iloc[0]),
            "old_ref": OLD_REF, "old_sha": old_sha, "history_rows": int(len(hist)),
            "history_max_date": str(pd.to_datetime(hist["date"]).max().date()),
            "swapped_columns": captured.get("swapped", []),
            "old_arm_equals_committed_board_form": repro,
            "fixtures_any_form_change": int(
                ((old[FORM_COLS] - new[FORM_COLS]).abs().fillna(0).gt(1e-9)
                 | (old[FORM_COLS].isna() != new[FORM_COLS].isna())).any(axis=1).sum())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
