"""AU2 forensic audit - CANARY train-vs-serve feature test (Phase 2B). READ-ONLY.

Question: given IDENTICAL history, does the feature builder used for TRAINING produce the same
numbers as the builder used for SERVING, for the same fixture? And does the model then produce the
same probability?

Canary fixtures = every Standard / New Format fixture PLAYED in [CANARY_START, CANARY_END] that is in
v9's own history on the audited commit (offline: committed caches only, no network). Because they are
in history we know the truth for both builders:

  TRAIN  legacy   src.feature_engineering.build_features(history COVID-filtered)  -> the fixture's row
  SERVE  legacy   src.feature_engineering.build_upcoming_features(fixture, history[date < match day])
  TRAIN  repaired src.features_repaired.build_features_repaired(canonical history)  -> fixture row
  SERVE  repaired src.features_repaired.build_upcoming_features_repaired(fixture, canonical history,
                  cutoff = match day)   (exactly what src.standard_serving.score_fixtures calls)
  TRAIN  HT       src.ht_features.build_ht_features(valid)   (pipeline.mode_train: on `valid`)
  SERVE  HT       src.ht_features.attach_ht_features(history, board) then POSITIONAL assignment, as
                  src.predict.predict_upcoming does (`feat[c] = _ht[c].to_numpy()`).

The serve builder receives the fixture with the SAME club spelling the history uses, so name
resolution is held fixed; name effects are measured separately on the live board (au2_live_board.py).
Serving receives the same COVID-filtered history as training here ("identical history").

Tolerances (documented, applied per feature per fixture):
  EXACT    |train - serve| <= 1e-9                       (both NaN counts as agreement)
  CLOSE    |train - serve| <= 1e-6 * max(1, |train|)
  MISSING_MISMATCH  exactly one side NaN
  Probability agreement: |p_train - p_serve| <= 0.005 (0.5pp) = PASS, else FAIL.

Outputs (reports/au2_data/): canary_feature_agreement.csv (per model family x feature),
canary_rows_<family>.csv (per fixture probabilities), canary_summary.json.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, AUDITED_COMMIT, heavy_lock, load_history_offline  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("au2_canary")

CANARY_START = pd.Timestamp("2026-10-17")
CANARY_END = pd.Timestamp("2026-10-30")
EXACT_TOL, CLOSE_REL, P_TOL = 1e-9, 1e-6, 0.005
IDENT = ["date", "league", "home_team", "away_team"]


def _load(p):
    import pickle
    with open(p, "rb") as f:
        return pickle.load(f)


def compare(train: pd.DataFrame, serve: pd.DataFrame, cols: list, family: str) -> pd.DataFrame:
    """Row-aligned comparison (both frames indexed 0..n-1 in the same fixture order)."""
    out = []
    for c in cols:
        a = pd.to_numeric(train[c], errors="coerce") if c in train.columns else pd.Series(np.nan, index=train.index)
        b = pd.to_numeric(serve[c], errors="coerce") if c in serve.columns else pd.Series(np.nan, index=serve.index)
        a, b = a.to_numpy(float), b.to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        one_nan = np.isnan(a) ^ np.isnan(b)
        d = np.abs(a - b)
        exact = both_nan | (~one_nan & (d <= EXACT_TOL))
        close = both_nan | (~one_nan & (d <= CLOSE_REL * np.maximum(1, np.abs(a))))
        out.append({
            "family": family, "feature": c, "n": len(a),
            "in_train_frame": c in train.columns, "in_serve_frame": c in serve.columns,
            "train_missing": float(np.isnan(a).mean()), "serve_missing": float(np.isnan(b).mean()),
            "exact_rate": float(exact.mean()), "close_rate": float(close.mean()),
            "missing_mismatch_rate": float(one_nan.mean()),
            "median_abs_diff": float(np.nanmedian(np.where(one_nan | both_nan, np.nan, d))) if (~(one_nan | both_nan)).any() else np.nan,
            "max_abs_diff": float(np.nanmax(np.where(one_nan | both_nan, np.nan, d))) if (~(one_nan | both_nan)).any() else np.nan,
            "train_median": float(np.nanmedian(a)) if (~np.isnan(a)).any() else np.nan,
            "serve_median": float(np.nanmedian(b)) if (~np.isnan(b)).any() else np.nan,
            "train_sd": float(np.nanstd(a)) if (~np.isnan(a)).any() else np.nan,
            "serve_sd": float(np.nanstd(b)) if (~np.isnan(b)).any() else np.nan,
        })
    r = pd.DataFrame(out)
    r["verdict"] = np.where(r["close_rate"] >= 0.999, "AGREE",
                            np.where(r["close_rate"] >= 0.95, "MOSTLY_AGREE", "DISAGREE"))
    return r


def prob_compare(payload, tr: pd.DataFrame, sv: pd.DataFrame, family: str) -> tuple[pd.DataFrame, dict]:
    from src.model import predict_proba
    pt = predict_proba(tr.reset_index(drop=True), payload=payload).to_numpy()
    ps = predict_proba(sv.reset_index(drop=True), payload=payload).to_numpy()
    d = np.abs(pt - ps)
    rows = tr[IDENT].reset_index(drop=True).copy()
    rows["p_train_builder"], rows["p_serve_builder"], rows["abs_diff"] = pt, ps, d
    s = {"family": family, "n": int(len(d)), "pass_rate_0.5pp": float((d <= P_TOL).mean()),
         "median_abs_diff_pp": round(float(np.median(d)) * 100, 3),
         "p90_abs_diff_pp": round(float(np.quantile(d, 0.9)) * 100, 3),
         "max_abs_diff_pp": round(float(d.max()) * 100, 3),
         "mean_p_train": round(float(pt.mean()), 4), "mean_p_serve": round(float(ps.mean()), 4)}
    return rows, s


def legacy_serve(fixtures: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """build_upcoming_features per match day, history strictly before that day. Input order kept."""
    from src.feature_engineering import build_upcoming_features
    parts = []
    for d, g in fixtures.groupby("date", sort=True):
        up = g.copy()
        up["best_over25"], up["best_under25"] = up["odds_over25"], up["odds_under25"]
        f = build_upcoming_features(up.drop(columns=[c for c in up.columns if c.startswith("_")] , errors="ignore"),
                                    hist[hist["date"] < d])
        f["_k"] = g["_k"].to_numpy()
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    from src.feature_engineering import build_features
    from src.features_repaired import build_features_repaired, build_upcoming_features_repaired
    from src import standard_serving
    from src.ht_features import build_ht_features, attach_ht_features, HT_FEATURE_COLS

    raw = load_history_offline()
    hist = raw[~raw["season"].isin(config.COVID_SEASONS)].copy() if config.EXCLUDE_COVID_SEASONS else raw.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    summary = {"audited_commit": AUDITED_COMMIT, "canary_window": [str(CANARY_START.date()), str(CANARY_END.date())],
               "history_rows": int(len(hist)), "history_max_date": str(hist["date"].max().date()),
               "tolerances": {"exact": EXACT_TOL, "close_rel": CLOSE_REL, "prob_pp": P_TOL * 100},
               "families": {}}
    agree = []

    # ── LEGACY: train builder over all history (as mode_train) ────────────────────────────
    tr_all = build_features(hist)
    tr_all["date"] = pd.to_datetime(tr_all["date"]).dt.normalize()
    win = tr_all[(tr_all["date"] >= CANARY_START) & (tr_all["date"] <= CANARY_END)
                 & tr_all["home_goals"].notna()].copy()
    win = win.sort_values(IDENT, kind="mergesort").reset_index(drop=True)
    win["_k"] = np.arange(len(win))
    win = win[win["league"].isin(set(config.STANDARD_FORMAT_LEAGUES) | set(config.NEW_FORMAT_LEAGUES))]
    win = win.reset_index(drop=True)
    win["_k"] = np.arange(len(win))
    fx = win[IDENT + ["odds_over25", "odds_under25", "_k"]].copy()
    sv_all = legacy_serve(fx, hist).sort_values("_k").reset_index(drop=True)
    assert (sv_all["home_team"].to_numpy() == win["home_team"].to_numpy()).all()

    pay = {k: _load(p) for k, p in {
        "standard_legacy": config.MODEL_FILE_STANDARD, "newformat": config.MODEL_FILE_NEWFORMAT,
        "btts": config.MODEL_FILE_BTTS, "over15": config.MODEL_FILE_OVER15,
        "over35": config.MODEL_FILE_OVER35, "standard_repaired": config.MODEL_FILE_STANDARD_REPAIRED,
        "ht_over05": config.HT_MODEL_FILE_05, "ht_over15": config.HT_MODEL_FILE_15}.items()}
    std = win["league"].isin(config.STANDARD_FORMAT_LEAGUES).to_numpy()
    nf = ~std
    for fam, payname, mask in (("STD_OU25_legacy", "standard_legacy", std), ("NF_OU25", "newformat", nf),
                               ("BTTS_std", "btts", std), ("BTTS_nf", "btts", nf),
                               ("O15_std", "over15", std), ("O15_nf", "over15", nf),
                               ("O35_std", "over35", std), ("O35_nf", "over35", nf)):
        cols = pay[payname]["feature_cols"]
        tr, sv = win[mask].reset_index(drop=True), sv_all[mask].reset_index(drop=True)
        agree.append(compare(tr, sv, cols, fam))
        rows, s = prob_compare(pay[payname], tr, sv, fam)
        rows.to_csv(DATA / f"canary_rows_{fam}.csv", index=False)
        summary["families"][fam] = s
        print(fam, s, flush=True)

    # ── HT: train = build_ht_features(valid); serve = attach_ht_features(raw history, board) ──
    valid = tr_all.dropna(subset=["over25", "home_scored_last5"])
    tr_ht = build_ht_features(valid)
    tr_ht["date"] = pd.to_datetime(tr_ht["date"]).dt.normalize()
    key = lambda d: d["date"].astype(str) + "|" + d["league"] + "|" + d["home_team"] + "|" + d["away_team"]  # noqa: E731
    tr_ht = tr_ht.assign(_key=key(tr_ht)).drop_duplicates("_key").set_index("_key")
    ws = win[std].reset_index(drop=True)
    sv_ht_parts, sv_ht_pos_parts = [], []
    for d, g in ws.groupby("date", sort=True):
        board = sv_all[std].reset_index(drop=True).loc[g.index].copy()
        # serving board in a NON-sorted order (league descending) to show what positional assignment does
        board = board.sort_values(["league", "home_team"], ascending=[False, True]).reset_index(drop=True)
        h = hist[hist["date"] < d]
        ht = attach_ht_features(h, board)
        aligned = board[IDENT].merge(ht.assign(date=pd.to_datetime(ht["date"]).dt.normalize())[IDENT + list(HT_FEATURE_COLS)],
                                     on=IDENT, how="left")
        positional = board[IDENT].copy()
        for c in HT_FEATURE_COLS:
            positional[c] = ht[c].to_numpy()            # what src/predict.py does
        sv_ht_parts.append(aligned)
        sv_ht_pos_parts.append(positional)
    sv_ht = pd.concat(sv_ht_parts, ignore_index=True)
    sv_ht_pos = pd.concat(sv_ht_pos_parts, ignore_index=True)
    trh = tr_ht.reindex(key(sv_ht.assign(date=pd.to_datetime(sv_ht["date"])))).reset_index(drop=True)
    agree.append(compare(trh, sv_ht, list(HT_FEATURE_COLS), "HT_features_key_aligned"))
    agree.append(compare(sv_ht, sv_ht_pos, list(HT_FEATURE_COLS), "HT_positional_vs_key_aligned"))

    # ── REPAIRED Standard OU25: frozen alias map, train builder vs serve builder ────────────
    rp = pay["standard_repaired"]
    canon = standard_serving.serving_history(raw, rp)           # standard leagues, COVID-filtered, frozen map
    canon = canon.copy()
    canon["date"] = pd.to_datetime(canon["date"]).dt.normalize()
    tr_r = build_features_repaired(canon, price_mode="none", team_scope="league")
    tr_r = tr_r[(tr_r["date"] >= CANARY_START) & (tr_r["date"] <= CANARY_END) & tr_r["home_goals"].notna()]
    tr_r = tr_r.sort_values(IDENT, kind="mergesort").reset_index(drop=True)
    parts = []
    for d, g in tr_r.groupby("date", sort=True):
        f = build_upcoming_features_repaired(g[IDENT].reset_index(drop=True), canon, price_mode="none",
                                             cutoff=d, team_scope="league")
        f["_i"] = g.index.to_numpy()
        parts.append(f)
    sv_r = pd.concat(parts, ignore_index=True).sort_values("_i").reset_index(drop=True)
    assert (sv_r["home_team"].to_numpy() == tr_r["home_team"].to_numpy()).all()
    agree.append(compare(tr_r, sv_r, rp["feature_cols"], "STD_OU25_repaired"))
    rows, s = prob_compare(rp, tr_r, sv_r, "STD_OU25_repaired")
    rows.to_csv(DATA / "canary_rows_STD_OU25_repaired.csv", index=False)
    summary["families"]["STD_OU25_repaired"] = s
    print("STD_OU25_repaired", s, flush=True)

    # ── Batch invariance: legacy batch-median imputation vs repaired persisted imputer ──────
    from src.model import predict_proba
    bi = {}
    for fam, payname, frame in (("STD_OU25_legacy", "standard_legacy", sv_all[std].reset_index(drop=True)),
                                ("STD_OU25_repaired", "standard_repaired", sv_r)):
        full = predict_proba(frame, payload=pay[payname]).to_numpy()
        half = np.concatenate([predict_proba(frame.iloc[:len(frame) // 2], payload=pay[payname]).to_numpy(),
                               predict_proba(frame.iloc[len(frame) // 2:], payload=pay[payname]).to_numpy()])
        d = np.abs(full - half)
        bi[fam] = {"n": int(len(d)), "rows_changed_gt_0.1pp": int((d > 0.001).sum()),
                   "max_change_pp": round(float(d.max()) * 100, 3), "median_change_pp": round(float(np.median(d)) * 100, 4)}
    # side models: Standard rows scored in a Standard-only batch vs the production mixed (std+nf) batch
    for payname in ("btts", "over15", "over35"):
        mixed = predict_proba(sv_all, payload=pay[payname]).to_numpy()[std]
        alone = predict_proba(sv_all[std].reset_index(drop=True), payload=pay[payname]).to_numpy()
        d = np.abs(mixed - alone)
        bi[f"{payname}_std_rows_mixed_vs_std_only_batch"] = {
            "n": int(len(d)), "rows_changed_gt_0.1pp": int((d > 0.001).sum()),
            "max_change_pp": round(float(d.max()) * 100, 3), "median_change_pp": round(float(np.median(d)) * 100, 4)}
    summary["batch_invariance"] = bi
    print(json.dumps(bi, indent=1), flush=True)

    agr = pd.concat(agree, ignore_index=True)
    agr.to_csv(DATA / "canary_feature_agreement.csv", index=False)
    summary["feature_verdicts"] = {f: g["verdict"].value_counts().to_dict() for f, g in agr.groupby("family")}
    (DATA / "canary_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary["feature_verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    with heavy_lock("AU2 canary train-vs-serve"):
        raise SystemExit(main())
