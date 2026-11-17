"""AU2 forensic audit - LIVE BOARD forensics (Phase 2B/2C). READ-ONLY.

Board = output/predictions.csv on the audited commit (generated 2026-11-01T10:59:39Z). History =
v9's offline loader on the same commit (committed caches; production downloads football-data at
run time, so history-dependent values are reproduced approximately, and the script says how well).

  1. REPRODUCIBILITY of served probabilities from the persisted feature row: every legacy-frame
     model (NF OU25, BTTS, O1.5, O3.5, HT 0.5, HT 1.5) is re-scored on the persisted columns with
     the SAME batch composition production used (NF batch / Standard+NF batch / Standard batch).
     The repaired Standard OU25 probability cannot be checked this way: its 41 inputs are built
     inside src/standard_serving and are NOT persisted.
  2. REST DAYS (legacy builder): missingness on the board and its causes - the legacy code
     (tail(1000) + concat + sort + POSITIONAL tail) is re-run, then the same quantity is computed
     by fixture key, with and without league-scoped name resolution.
  3. HT FEATURES: attach_ht_features(history, board) and src/predict.py's POSITIONAL assignment vs
     key-aligned assignment, compared with the persisted ht_* columns.
  4. REPAIRED Standard features for the same board via src.standard_serving (frozen alias map):
     rest-day / form missingness the repaired model actually sees.

    python scripts/audit/au2_live_board.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, AUDITED_COMMIT, heavy_lock, load_history_offline  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

warnings.filterwarnings("ignore")
IDENT = ["date", "league", "home_team", "away_team"]


def _load(p):
    import pickle
    with open(p, "rb") as f:
        return pickle.load(f)


def reproduce(board: pd.DataFrame) -> dict:
    from src.model import predict_proba
    std = board["model_type"] == "standard"
    nf = board["model_type"] == "new_format"
    out = {}
    jobs = [("NF_OU25", config.MODEL_FILE_NEWFORMAT, nf, "p_over25", nf),
            ("BTTS", config.MODEL_FILE_BTTS, std | nf, "p_btts", std | nf),
            ("O15", config.MODEL_FILE_OVER15, std | nf, "p_over15", std | nf),
            ("O35", config.MODEL_FILE_OVER35, std | nf, "p_over35", std | nf),
            ("HT05", config.HT_MODEL_FILE_05, std, "p_ht_over05", std),
            ("HT15", config.HT_MODEL_FILE_15, std, "p_ht_over15", std)]
    for name, f, batch, col, _ in jobs:
        p = predict_proba(board[batch], payload=_load(f)).to_numpy()
        served = pd.to_numeric(board.loc[batch, col], errors="coerce").to_numpy()
        d = np.abs(p - served)
        out[name] = {"n": int(batch.sum()), "max_abs_diff": float(np.nanmax(d)),
                     "reproduced_within_1e-6": int((d <= 1e-6).sum())}
        if name in ("BTTS", "O15", "O35"):
            alone = predict_proba(board[std], payload=_load(f)).to_numpy()
            dd = np.abs(alone - pd.to_numeric(board.loc[std, col], errors="coerce").to_numpy())
            out[name]["std_rows_rescored_in_std_only_batch_max_diff_pp"] = round(float(dd.max()) * 100, 3)
            out[name]["std_rows_rescored_in_std_only_batch_rows_gt_0.5pp"] = int((dd > 0.005).sum())
    return out


def rest_days(board: pd.DataFrame, hist: pd.DataFrame) -> dict:
    from src.feature_engineering import _add_rest_days
    from src.team_names import resolve
    b = board[IDENT].copy()
    b["date"] = pd.to_datetime(b["date"])
    h = hist.sort_values("date")
    # (a) production code path, verbatim logic of build_upcoming_features
    comb = pd.concat([h[["date", "home_team", "away_team"]].tail(1000), b[["date", "home_team", "away_team"]]]
                     ).sort_values("date").reset_index(drop=True)
    tail = _add_rest_days(comb).tail(len(b)).reset_index(drop=True)
    prod_h, prod_a = tail["home_rest_days"].to_numpy(), tail["away_rest_days"].to_numpy()
    tail_names = tail[["home_team", "away_team"]].to_numpy()
    misaligned = int((tail_names[:, 0] != b["home_team"].to_numpy()).sum())
    # (b) by key, raw names, full history (not just tail 1000)
    last = {}
    for r in h.itertuples():
        for t in (r.home_team, r.away_team):
            last[(r.league, t)] = r.date
    pools = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in h.groupby("league")}

    def days(lg, t, d, res):
        name = t
        if res and t not in pools.get(lg, []):
            name = resolve(t, pools.get(lg, [])) or t
        ld = last.get((lg, name))
        return (d - ld).days if ld is not None else np.nan
    key_raw_h = np.array([days(r.league, r.home_team, r.date, False) for r in b.itertuples()], float)
    key_res_h = np.array([days(r.league, r.home_team, r.date, True) for r in b.itertuples()], float)
    persisted_h = pd.to_numeric(board["home_rest_days"], errors="coerce").to_numpy()
    persisted_a = pd.to_numeric(board["away_rest_days"], errors="coerce").to_numpy()
    out = {}
    for lbl, m in (("standard", board["model_type"] == "standard"), ("new_format", board["model_type"] == "new_format")):
        m = m.to_numpy()
        out[lbl] = {
            "rows": int(m.sum()),
            "persisted_home_missing": round(float(np.isnan(persisted_h[m]).mean()), 3),
            "persisted_away_missing": round(float(np.isnan(persisted_a[m]).mean()), 3),
            "reproduced_prod_home_missing": round(float(np.isnan(prod_h[m]).mean()), 3),
            "reproduced_prod_agrees_with_persisted_missingness": round(float((np.isnan(prod_h[m]) == np.isnan(persisted_h[m])).mean()), 3),
            "by_key_raw_name_home_missing": round(float(np.isnan(key_raw_h[m]).mean()), 3),
            "by_key_resolved_name_home_missing": round(float(np.isnan(key_res_h[m]).mean()), 3),
        }
    out["rows_whose_positional_tail_row_is_a_different_fixture"] = misaligned
    out["board_rows"] = int(len(b))
    out["board_is_date_sorted"] = bool(b["date"].is_monotonic_increasing)
    return out


def ht_alignment(board: pd.DataFrame, hist: pd.DataFrame) -> dict:
    from src.ht_features import attach_ht_features, HT_FEATURE_COLS
    std = board[board["model_type"] == "standard"].copy()
    feat = board.copy()       # production attaches over the WHOLE board (std + nf), then scores std rows
    ht = attach_ht_features(hist, feat)
    pos = pd.DataFrame({c: ht[c].to_numpy() for c in HT_FEATURE_COLS})            # what predict does
    htk = ht.copy()
    htk["date"] = pd.to_datetime(htk["date"]).dt.strftime("%Y-%m-%d")
    f2 = feat[IDENT].copy()
    f2["date"] = pd.to_datetime(f2["date"]).dt.strftime("%Y-%m-%d")
    aligned = f2.merge(htk[IDENT + list(HT_FEATURE_COLS)], on=IDENT, how="left")
    persisted = board[list(HT_FEATURE_COLS)].apply(pd.to_numeric, errors="coerce")
    res = {}
    for c in ("ht_home_n", "ht_away_n", "ht_lambda_total", "ht_home_attack"):
        p = persisted[c].to_numpy()
        res[c] = {"persisted_eq_positional": round(float(np.isclose(p, pos[c].to_numpy(), atol=1e-6, equal_nan=True).mean()), 3),
                  "persisted_eq_key_aligned": round(float(np.isclose(p, aligned[c].to_numpy(), atol=1e-6, equal_nan=True).mean()), 3),
                  "positional_eq_key_aligned": round(float(np.isclose(pos[c].to_numpy(), aligned[c].to_numpy(), atol=1e-6, equal_nan=True).mean()), 3)}
    res["attach_returns_rows"] = int(len(ht))
    res["rows_on_board"] = int(len(feat))
    # probability effect on Standard rows: score HT payload with positional vs key-aligned features
    from src.model import predict_proba
    sm = (board["model_type"] == "standard").to_numpy()
    for fname, col in ((config.HT_MODEL_FILE_05, "ht05"), (config.HT_MODEL_FILE_15, "ht15")):
        pl = _load(fname)
        a = board.copy(); b2 = board.copy()
        for c in HT_FEATURE_COLS:
            a[c] = pos[c].to_numpy(); b2[c] = aligned[c].to_numpy()
        pa = predict_proba(a[sm], payload=pl).to_numpy(); pb = predict_proba(b2[sm], payload=pl).to_numpy()
        d = np.abs(pa - pb)
        res[f"{col}_prob_positional_vs_aligned"] = {"n": int(sm.sum()), "median_pp": round(float(np.median(d)) * 100, 3),
                                                    "max_pp": round(float(d.max()) * 100, 3),
                                                    "rows_gt_1pp": int((d > 0.01).sum())}
    return res


def repaired_board(board: pd.DataFrame, raw: pd.DataFrame) -> dict:
    from src import standard_serving as S
    from src.features_repaired import build_upcoming_features_repaired
    pay = S.load_payload()
    std = board[board["model_type"] == "standard"].copy()
    hist = S.serving_history(raw, pay)
    ident = std[IDENT].reset_index(drop=True)
    f = build_upcoming_features_repaired(ident, hist, price_mode="none", team_scope="league")
    out = {"rows": int(len(f))}
    for c in ("home_rest_days", "away_rest_days", "home_scored_last5", "away_scored_last5",
              "home_ht_over05_rate", "home_season_goals_h", "home_xg_last5"):
        out[f"{c}_missing"] = round(float(f[c].isna().mean()), 3)
    out["home_rest_days_median"] = float(f["home_rest_days"].median())
    out["home_rest_days_gt_14"] = round(float((f["home_rest_days"] > 14).mean()), 3)
    unresolved = (f["history_team_home"] == std["home_team"].to_numpy()) & ~std["home_team"].isin(set(hist["home_team"]) | set(hist["away_team"])).to_numpy()
    out["home_names_not_in_history_after_resolution"] = int(unresolved.sum())
    from src.model import predict_proba
    p = predict_proba(f, payload=pay).to_numpy()
    served = pd.to_numeric(std["p_over25"], errors="coerce").to_numpy()
    d = np.abs(p - served)
    out["p_rebuilt_vs_served"] = {"median_abs_pp": round(float(np.median(d)) * 100, 3),
                                  "max_abs_pp": round(float(d.max()) * 100, 3),
                                  "within_0.5pp": round(float((d <= 0.005).mean()), 3),
                                  "note": "history differs from production's (offline committed cache vs live download), so exact equality is not expected"}
    out["history_max_date_by_league"] = {lg: str(g["date"].max())[:10] for lg, g in hist.groupby("league")}
    return out


def main() -> int:
    board = pd.read_csv(ROOT / "output/predictions.csv", low_memory=False)
    raw = load_history_offline()
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    out = {"audited_commit": AUDITED_COMMIT, "board_generated_at": str(board["generated_at"].iloc[0]),
           "board_rows": int(len(board)), "model_types": board["model_type"].value_counts().to_dict()}
    out["reproduce_served_from_persisted_features"] = reproduce(board)
    out["rest_days_legacy"] = rest_days(board, raw)
    out["ht_alignment"] = ht_alignment(board, raw)
    out["repaired_standard_board"] = repaired_board(board, raw)
    (DATA / "live_board_forensics.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    with heavy_lock("AU2 live board forensics"):
        raise SystemExit(main())
