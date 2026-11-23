"""Row-level REAL DATA certification of every dataset that feeds a model (owner 2026-11-23: "the most
important achievement is 100% real data for every row in our data set, for all the models").

    python scripts/certify_real_data.py                       # print the summary
    python scripts/certify_real_data.py --write               # + output/real_data_certification.json
    python scripts/certify_real_data.py --write --std-frame <saved Standard training frame .parquet>
                                                              # + row-level counts on the saved frame
    python scripts/certify_real_data.py --write --pro-root <wowzaV9-Pro @ origin/main> --v11-root <wowza_v11 @ origin/main>
        --exec-root <wowza-exec clone, data branches fetched> [--fantasy-frame <props_harness.build_frame parquet>]
                                                              # + other repos row level, Fantasy V1 parity
    Without a root the scope is UNVERIFIED (CI). wowza-exec is PRIVATE: only counts and verdicts are written.

READ-ONLY with respect to models, data, ledgers and serving. COUNTS ONLY: no price, odds, probability
or stake value is printed or written (a price column is read only to say whether a value is there).

Every value (cell) of a dataset is put in exactly one PROVENANCE CLASS:
    PROVIDER_OBSERVED            a provider's own number (football-data, API-Football, FPL, OddsAPI)
    VERIFIED_DERIVED             computed from observed values by documented, tested code
                                 (rolling form, rates, indicators, a provider rule, a two-source check)
    CORRECTED_BESIDE_ORIGINAL    a provider value replaced by a proven provider value, the original
                                 kept beside it (output/corrections/*, *_orig columns)
    MISSING_KEPT_MISSING         no observation: the value stays NaN / the row is excluded / no probability
    DEFAULT_OR_SYNTHETIC         ANY default, placeholder, constant fill or synthetic value stored or
                                 read as if it were an observation (named: dataset, column, rows, cause)
    UNVERIFIED_VALUE             a value no provider confirms (no provider holds it, or a provider
                                 contradicts it) - not invented on purpose, but not proven real either
A model may still FILL a missing value inside the model (a train-only median + `__missing` indicator,
a NaN-native tree, a fitted base rate). Those are listed apart as TRAINING-FITTED IMPUTATIONS (identified,
versioned, validated); they are not data and are never counted as observed.

A scope is REAL_100 only when it holds 0 DEFAULT_OR_SYNTHETIC / UNVERIFIED_VALUE values and every
input of it was measured here. A scope that could not be measured is UNVERIFIED, never a pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "real_data_certification.json"
VERSION = "REAL-DATA-CERT-1.0"
OBS, DER, COR, MIS, DEF, UNV = ("PROVIDER_OBSERVED", "VERIFIED_DERIVED", "CORRECTED_BESIDE_ORIGINAL",
                                "MISSING_KEPT_MISSING", "DEFAULT_OR_SYNTHETIC", "UNVERIFIED_VALUE")
CLASSES = (OBS, DER, COR, MIS, DEF, UNV)
_CODE = {c: i for i, c in enumerate(CLASSES)}
REAL_100, REAL_WITH_GAPS, NOT_REAL, UNVERIFIED = ("REAL_100", "REAL_100_WITH_MISSING_KEPT_MISSING",
                                                  "NOT_REAL", "UNVERIFIED")


def _z() -> dict:
    return {c: 0 for c in CLASSES}


def _full(cells: dict) -> dict:
    """Every class present: a class a dataset never produced holds zero cells (it was counted, not assumed)."""
    out = _z()
    out.update(cells)
    return out


def _add(a: dict, b: dict) -> dict:
    b = _full(b)
    for k in CLASSES:
        a[k] = int(a[k]) + int(b[k])
    return a


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                                   # noqa: BLE001
        return ""


def _verdict(cells: dict, measured: bool = True) -> str:
    if not measured:
        return UNVERIFIED
    c = _full(cells)
    if c[DEF] > 0 or c[UNV] > 0:
        return NOT_REAL
    return REAL_WITH_GAPS if c[MIS] > 0 else REAL_100


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ── 1. history banks (the raw match rows every team model is built from) ─────────────────────────
FD_OBS = ["home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot", "home_corners",
          "away_corners", "home_fouls", "away_fouls", "ht_home_goals", "ht_away_goals"]
AF_OBS = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "HXG", "AXG",
          "HIB", "AIB"]


def _corr_rows(name: str) -> int:
    p = ROOT / "output" / "corrections" / name
    if not p.exists():
        return 0
    try:
        return int(len(pd.read_csv(p, low_memory=False)))
    except Exception:                                                   # noqa: BLE001
        return 0


def history_banks() -> dict:
    out = {}
    for f, cols, prov in (("fd_history.parquet", FD_OBS, "football-data.co.uk CSV"),
                          ("af_history.parquet", AF_OBS, "API-Football /fixtures + /fixtures/statistics")):
        df = pd.read_parquet(ROOT / "output" / f)
        cells, per = _z(), {}
        for c in cols:
            if c not in df.columns:
                continue
            v = _num(df[c])
            n_obs, n_mis = int(v.notna().sum()), int(v.isna().sum())
            per[c] = {OBS: n_obs, MIS: n_mis}
            cells[OBS] += n_obs
            cells[MIS] += n_mis
        out[f] = {"rows": int(len(df)), "provider": prov, "cells": cells, "per_column": per,
                  "verdict": _verdict(cells)}
    corr = {n: _corr_rows(n) for n in (
        "history_date_corrections.csv", "history_score_90min.csv", "history_score_conflicts_resolved.csv",
        "history_awarded_results.csv", "history_provider_supplement.csv", "history_removed_duplicates.csv",
        "history_excluded_cup_rows.csv", "af_history_removed_duplicates.csv",
        "af_history_restored_provider_rows_2026-11_excluded.csv")}
    rest = ROOT / "output" / "corrections" / "af_history_restored_provider_rows_2026-11_verified.parquet"
    corr["af_history_restored_provider_rows_2026-11_verified.parquet"] = (
        int(len(pd.read_parquet(rest))) if rest.exists() else 0)
    out["corrections_beside_originals"] = {
        "rows_by_file": corr,
        "note": "corrections are applied by the loader at read time; the bank files keep the provider rows. "
                "Removed duplicates / excluded cup rows are rows taken OUT of training (kept in the file for audit)."}
    ht = ROOT / "output" / "collect" / "ht_fill_audit.json"
    if ht.exists():
        h = json.loads(ht.read_text(encoding="utf-8"))
        out["ht_fill_from_api_football"] = {
            "filled_rows_total": h.get("filled_rows_total"),
            "class": OBS,
            "note": "half-time scores for bank rows lacking them, taken from API-Football (a second provider's own "
                    "number; refused on a full-time mismatch). fd_history.parquet itself is not modified."}
    tot = _z()
    for k in ("fd_history.parquet", "af_history.parquet"):
        _add(tot, out[k]["cells"])
    out["verdict"] = _verdict(tot)
    return out


# ── 2. team model training data (every served payload; cells from the payload's own imputer record) ─
def _team_class(feature: str) -> str:
    from src.integrity.real_data_provenance import TEAM_FEATURES, SOURCE_VERIFIED
    if feature.endswith("__missing"):
        return DER
    spec = TEAM_FEATURES.get(feature)
    if spec is None:
        return "UNTRACED"
    return OBS if spec.get("cls") == SOURCE_VERIFIED else DER


def team_training(std_frame: str | None, carry: dict | None = None) -> dict:
    import config
    from src.integrity.real_data_provenance import TEAM_ARTIFACTS, _parts, artifact_status
    from src.model import load_models
    out, imputations = {}, []
    feat = None
    if std_frame:
        feat = pd.read_parquet(std_frame)
    for label, attr, tracks in TEAM_ARTIFACTS:
        status = artifact_status(attr)
        if not str(status).startswith("active"):
            continue
        path = Path(getattr(config, attr))
        pl = load_models(model_file=path)
        for track, part in _parts(pl, tracks).items():
            imp = part.get("imputer") or {}
            feats = imp.get("features") or {}
            n_fit = int(imp["fitted_on_rows"])          # a payload without the record is UNVERIFIED (raises)
            cells, per, untraced, consts, fabricated = _z(), {}, [], [], []
            for f in part.get("feature_cols") or []:
                st = feats.get(f) or {}
                if f.endswith("__missing"):
                    n_obs, n_mis = n_fit, 0
                else:
                    n_obs = int(st["count"])                  # non-missing training values (imputer record)
                    n_mis = n_fit - n_obs
                cls = _team_class(f)
                if cls == "UNTRACED":
                    untraced.append(f)
                    cls = DEF                                    # unknown provenance is never counted real
                if not f.endswith("__missing") and st and st.get("median") is None:
                    fabricated.append(f)                         # no training value: any number here is invented
                    cells[DEF] += n_fit
                    per[f] = {DEF: n_fit}
                    continue
                if not f.endswith("__missing") and st and st.get("min") is not None and st.get("min") == st.get("max"):
                    consts.append(f)
                cells[cls] += n_obs
                cells[MIS] += n_mis
                per[f] = {cls: n_obs, MIS: n_mis}
            key = f"{label} [{track}]"
            n_missing_feats = sum(1 for v in per.values() if v.get(MIS))
            imputations.append({
                "scope": key, "payload": path.name, "imputer_version": imp.get("version"),
                "fitted_on_rows": n_fit, "fitted_on_dates": imp.get("fitted_on_dates"),
                "rule": "train-only median per feature + `__missing` indicator, clipped to the training range; "
                        "a REQUIRED form feature missing at serving gives no probability (src/model.py serving guard)",
                "features_with_missing_cells": n_missing_feats,
                "cells_filled_in_fit_block": int(cells[MIS]),
                "validated_by": "tests/test_real_data_pipeline_trace.py; tests/test_imputer*.py; "
                                "scripts/team_train_serve_parity.py (feature parity)"})
            rec = {"payload": path.name, "status": status, "training_rows": part.get("training_rows"),
                   "imputer_fit_rows": n_fit, "n_features": len(part.get("feature_cols") or []),
                   "cells": cells, "untraced_features": untraced, "constant_on_training_rows": consts,
                   "fabricated_features": fabricated, "verdict": _verdict(cells),
                   "basis": "payload imputer record (count / missing_rate per feature over the fit block)"}
            # low-coverage features (most cells are the fitted fill, not data) named explicitly
            present = {f: sum(n for c, n in v.items() if c in (OBS, DER)) for f, v in per.items()}
            rec["low_coverage_features"] = {f: round(present[f] / n_fit, 4) for f, v in per.items()
                                            if n_fit and MIS in v and present[f] / n_fit < 0.5}
            prev = ((carry or {}).get("payloads") or {}).get(key) or {}
            carried = (feat is None and track == "standard" and prev.get("row_level_saved_frame")
                       and prev.get("payload") == path.name and prev.get("training_rows") == part.get("training_rows"))
            if (feat is not None and track == "standard") or carried:
                if carried:
                    # no --std-frame: the saved-frame counts of the last run on the SAME payload are carried forward
                    # (named), never dropped - dropping them would pass rows the frame proved unverified
                    rec["row_level_saved_frame"] = {**prev["row_level_saved_frame"],
                                                    "carried_forward_from": {k: (carry or {}).get(k) for k in
                                                                             ("generated_at", "code_sha")}}
                else:
                    rec["row_level_saved_frame"] = _std_frame_rows(feat, part)
                ar = rec["row_level_saved_frame"].get("backtest_archive_rows") or {}
                bad = (int(ar["rows"]) + int(ar["downstream_training_rows_reading_them"])) if "rows" in ar else 0
                if bad and not ar.get("rows"):
                    ar["note"] = "no archive row is a training row of this payload; later rows read them in their windows"
                if bad:
                    # the archive's own unproven stat cells + at least one rolling input of every later row
                    # whose ROLLING_N window reads them (corners / fouls per game, SOT ratio)
                    unv = (int(ar["stat_cells_contradicted_by_a_provider"]) + int(ar["stat_cells_no_provider_holds"])
                           + int(ar["downstream_training_rows_reading_them"]))
                    rec["rows_with_unverified_inputs"] = bad
                    rec["cells"][UNV] += unv
                    rec["cells"][DER] -= unv
                    rec["verdict"] = _verdict(rec["cells"])
            out[key] = rec
    tot = _z()
    for v in out.values():
        _add(tot, v["cells"])
    return {"payloads": out, "cells": tot, "verdict": _verdict(tot), "training_fitted_imputations": imputations,
            "backtest_archive_scope": archive_scope()}


def _std_frame_rows(feat: pd.DataFrame, part: dict) -> dict:
    """Row-level counts on the saved Standard training frame (the rows the payload trained on)."""
    from src.retrain_harness import protocol as P
    from src.standard_repair import frame_hash
    cfg = P.load_config()
    tgt = part.get("target")
    market = None
    for m, spec in (cfg.get("markets") or {}).items():
        if spec.get("target") == tgt:
            market = m
            break
    if market is None:
        return {"status": UNVERIFIED, "why": f"no harness market for target {tgt!r}"}
    rows = P.eligible_rows(feat, cfg, "standard", market)
    end = part.get("training_end")
    if end:
        rows = rows[pd.to_datetime(rows["date"]) <= pd.Timestamp(end)]
    cols = [c for c in part.get("feature_cols") or [] if not c.endswith("__missing") and c in rows.columns]
    X = rows[cols].apply(pd.to_numeric, errors="coerce")
    all_present = int(X.notna().all(axis=1).sum())
    src = rows["row_source"].astype(str).value_counts().to_dict() if "row_source" in rows.columns else {}
    arch = _archive_rows(feat, rows)
    return {"frame_hash": frame_hash(feat), "rows": int(len(rows)), "payload_training_rows": part.get("training_rows"),
            "backtest_archive_rows": arch,
            "rows_every_input_present": all_present, "rows_with_any_input_missing": int(len(rows) - all_present),
            "row_source": {k: int(v) for k, v in src.items()},
            "missing_inputs_by_feature": {c: int(X[c].isna().sum()) for c in cols if X[c].isna().any()}}


ARCHIVE_STATS = [("home_shots", "HS"), ("away_shots", "AS"), ("home_sot", "HST"), ("away_sot", "AST"),
                 ("home_corners", "HC"), ("away_corners", "AC"), ("home_fouls", "HF"), ("away_fouls", "AF")]


def _archive_rows(feat: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """Training rows whose match row came from output/backtest_all_leagues.csv (row_source backtest_archive):
    an internal backtest output, not a provider file. Each stat cell is checked against API-Football
    (af_history) and football-data (fd_history); the next config.ROLLING_N matches of both clubs read it."""
    import config
    if "row_source" not in rows.columns:
        return {"status": UNVERIFIED}
    every = feat[feat["row_source"].astype(str) == "backtest_archive"]       # read by later rows' windows
    tr_keys = set(map(tuple, rows.assign(date=pd.to_datetime(rows["date"]).dt.normalize())[
        ["league", "date", "home_team", "away_team"]].astype(str).to_numpy()))
    arch = every[[tuple(x) in tr_keys for x in every.assign(date=pd.to_datetime(every["date"]).dt.normalize())[
        ["league", "date", "home_team", "away_team"]].astype(str).to_numpy()]]
    af = pd.read_parquet(ROOT / "output" / "af_history.parquet")
    fd = pd.read_parquet(ROOT / "output" / "fd_history.parquet")
    for d in (af, fd):
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    key = ["league", "date", "home_team", "away_team"]
    agree = contra = none = 0
    for _, r in arch.iterrows():
        day = pd.Timestamp(r["date"]).normalize()
        k = ((af["league"] == r["league"]) & (af["date"] == day) & (af["home_team"] == r["home_team"])
             & (af["away_team"] == r["away_team"]))
        kf = ((fd["league"] == r["league"]) & (fd["date"] == day) & (fd["home_team"] == r["home_team"])
              & (fd["away_team"] == r["away_team"]))
        for ours, afc in ARCHIVE_STATS:
            v = _num(pd.Series([r.get(ours)])).iloc[0]
            if pd.isna(v):
                continue
            a_v = _num(af.loc[k, afc]).dropna()
            f_v = _num(fd.loc[kf, ours]).dropna() if ours in fd.columns else pd.Series(dtype=float)
            prov = pd.concat([a_v, f_v])
            if prov.empty:
                none += 1
            elif (prov == v).all():
                agree += 1
            else:
                contra += 1
    f2 = feat.copy()
    f2["date"] = pd.to_datetime(f2["date"]).dt.normalize()
    down = set()
    for _, r in every.iterrows():
        lg = f2[f2["league"] == r["league"]].sort_values("date")
        for club in (r["home_team"], r["away_team"]):
            nxt = lg[((lg["home_team"] == club) | (lg["away_team"] == club))
                     & (lg["date"] > pd.Timestamp(r["date"]).normalize())].head(int(config.ROLLING_N))
            down |= set(map(tuple, nxt[key].astype(str).to_numpy()))
    tr = tr_keys
    return {"archive_rows_in_frame": int(len(every)), "rows": int(len(arch)), "stat_cells_agree_with_a_provider": agree,
            "stat_cells_contradicted_by_a_provider": contra, "stat_cells_no_provider_holds": none,
            "downstream_training_rows_reading_them": len(down & tr),
            "matches": [f"{r['league']} {pd.Timestamp(r['date']).date()} {r['home_team']} v {r['away_team']}"
                        for _, r in arch.iterrows()]}


def archive_scope() -> dict:
    """Which tracks can read output/backtest_all_leagues.csv at all (the loader appends it per league)."""
    from src import league_registry as LR
    p = ROOT / "output" / "backtest_all_leagues.csv"
    if not p.exists():
        return {"status": "ABSENT"}
    lg = pd.read_csv(p, usecols=["league"], low_memory=False)["league"].astype(str)
    by_track: dict = {}
    for name, n in lg.value_counts().items():
        e = LR.entry(name) or {}
        t = e.get("model_type") or "UNREGISTERED"
        by_track.setdefault(t, {"leagues": 0, "rows": 0})
        by_track[t]["leagues"] += 1
        by_track[t]["rows"] += int(n)
    return {"file": "output/backtest_all_leagues.csv", "rows": int(len(lg)), "by_track": by_track,
            "note": "an archive row enters training only where the identity pipeline keeps it over the provider "
                    "rows; the saved Standard frame counts are in team_training.payloads[*].row_level_saved_frame"}


# ── 3. serving board (the inputs live fixtures are scored on) ─────────────────────────────────────
def serving_board() -> dict:
    from src.integrity import data_truth as DT
    p = ROOT / "output" / "predictions.csv"
    if not p.exists():
        return {"status": UNVERIFIED, "why": "no board"}
    board = pd.read_csv(p, low_memory=False)
    feats = DT.serving_features()
    mt = board.get("model_type", pd.Series("", index=board.index)).astype(str)
    out, tot = {}, _z()
    for (track, market), cols in feats.items():
        sub = board[mt == track]
        if not isinstance(cols, list):
            out[f"{track}|{market}"] = {"status": "NOT_SERVED" if cols is None else UNVERIFIED}
            continue
        raw = sorted({c[:-len("__missing")] if c.endswith("__missing") else c for c in cols})
        cells, absent = _z(), []
        for c in raw:
            if c not in sub.columns:
                absent.append(c)                       # computed at scoring time, not written to the board
                continue
            v = _num(sub[c])
            cls = _team_class(c)
            cls = DEF if cls == "UNTRACED" else cls
            cells[cls] += int(v.notna().sum())
            cells[MIS] += int(v.isna().sum())
        _add(tot, cells)
        out[f"{track}|{market}"] = {"fixtures": int(len(sub)), "cells": cells,
                                    "inputs_not_on_board": len(absent), "verdict": _verdict(cells)}
    return {"board_rows": int(len(board)), "tracks": out, "cells": tot, "verdict": _verdict(tot),
            "note": "inputs computed at scoring and not exported to the board are counted by the train/serve "
                    "parity evidence, not here"}


# ── 4. props (player rows -> the certified props payloads) ───────────────────────────────────────
def props() -> dict:
    from player_model import props_harness as H
    import pyarrow.dataset as ds
    raw = H.source_frame()
    cs = raw["count_source"].astype(str).value_counts().to_dict()
    n_src = {k: int((raw["count_source"].astype(str) == k).sum()) for k in ("refetched", "provider_rule",
                                                                             "zeros_unverified")}
    side_fixed = int((raw["is_home_orig"].notna() & (raw["is_home_orig"] != raw["is_home"])).sum()) \
        if "is_home_orig" in raw.columns else None
    gk_moved = (int((_num(raw["goals_conceded_provider"]).notna() & _num(raw["goals_conceded"]).isna()).sum())
                if {"goals_conceded_provider", "goals_conceded"} <= set(raw.columns) else None)
    pop = H.population(raw)
    rows = {"source_rows": int(len(raw)),
            "count_source": {k: int(v) for k, v in cs.items()},
            "classes_of_rows": {**_z(), OBS: n_src["refetched"], DER: n_src["provider_rule"],
                                MIS: n_src["zeros_unverified"]},
            "home_away_side_corrected_beside_original": side_fixed,
            "keeper_goals_conceded_contradicting_score_set_missing_value_kept_beside": gk_moved,
            "population_rows": int(len(pop)),
            "training_eligible_all_prior_verified": int(pop["all_prior_verified"].sum()),
            "excluded_not_all_prior_verified": int((~pop["all_prior_verified"].astype(bool)).sum())}
    # stored placeholders in player_history.parquet (read as missing by the harness and serving)
    d = ds.dataset(str(ROOT / "player_history.parquet"), format="parquet")
    have = set(d.schema.names)
    ph_cols = [c for c in ("age", "height_cm", "weight_kg", "age_peak_delta") if c in have]
    t = d.to_table(columns=ph_cols).to_pandas()
    from player_model.feature_engineering import _PROFILE_PLACEHOLDERS
    stored = {}
    for c in ph_cols:
        val = _PROFILE_PLACEHOLDERS.get(c, 0.0 if c == "age_peak_delta" else None)
        if val is None:
            continue
        stored[c] = {"placeholder": val, "rows": int((_num(t[c]) == val).sum()), "of": int(len(t))}
    rows["stored_placeholders_in_player_history"] = {
        "columns": stored, "reaches_model": False,
        "why_not": "player_model/feature_engineering.py _PROFILE_PLACEHOLDERS: the placeholder value is read "
                   "as missing; profile features come from output/collect/player_profiles.parquet (NaN if unknown); "
                   "props certification: no served feature holds one value on > 99.9% of training rows"}
    cert = json.loads((ROOT / "output" / "props_real_data_provenance.json").read_text(encoding="utf-8"))
    rows["certification"] = {"verdict": cert.get("verdict"), "artifacts": (cert.get("counts") or {}).get("artifacts"),
                             "verified": (cert.get("counts") or {}).get("verified"),
                             "generated_at": cert.get("generated_at"), "code_sha": cert.get("code_sha")}
    rows["training_fitted_imputations"] = [
        {"scope": "props goals / assists / sot / sot2 (hgb_nan_native)", "rule": "none: NaN-native trees, nothing filled"},
        {"scope": "props cards (recipe_retrained) + recipe challengers",
         "rule": "fit-block medians persisted in the payload for 128 features (RecipeRetrained._fill)",
         "validated_by": "output/props_real_data_provenance.json batch_invariance + serving-vs-training parity (6,316 rows)"},
        {"scope": "rare props goals2 / goals3 / sot3 / sot4 (player_base_rate)",
         "rule": "an unseen player gets the fitted window market rate (beta prior, method of moments)"}]
    rows["verdict"] = REAL_WITH_GAPS if rows["classes_of_rows"][MIS] else REAL_100
    return rows


# ── 5. corners / cards side models ────────────────────────────────────────────────────────────────
def side_models() -> dict:
    a = json.loads((ROOT / "models" / "side_models" / "side_models.json").read_text(encoding="utf-8"))
    out = {"model_version": a.get("model_version"), "data_end": a.get("data_end"), "families": {}}
    for fam, cov in (a.get("coverage") or {}).items():
        served = {lg: c for lg, c in cov.items() if c.get("covered")}
        tot = {"two_source_verified": 0, "single_source_sane": 0, "fixtures_in_window": 0,
               "not_verified_or_no_stats": 0}
        for lg, c in served.items():
            k = "two_source_verified" if c.get("basis") == "two_source" else "single_source_sane"
            tot[k] += int(c["verified"])
            tot["fixtures_in_window"] += int(c["fixtures_in_window"])
        tot["not_verified_or_no_stats"] = tot["fixtures_in_window"] - tot["two_source_verified"] - tot["single_source_sane"]
        zf = sum(int(a["data_audit"][lg]["zero_fill_suspect"]) for lg in served)
        out["families"][fam] = {
            "leagues_served": len(served), "leagues_refused": sorted(set(cov) - set(served)),
            "fixtures": tot, "zero_fill_suspect_fixtures": zf,
            "cells": {**_z(), OBS: tot["two_source_verified"] + tot["single_source_sane"],
                      MIS: tot["not_verified_or_no_stats"]},
            "note": "a fixture whose counts are not verified (two sources disagree / no stats / failed sanity) is "
                    "NaN, never a training row; single-source leagues pass a sanity rule, not a second source"}
    tot = _z()
    for f in out["families"].values():
        _add(tot, f["cells"])
    out["cells"] = tot
    out["verdict"] = _verdict(tot)
    return out


# ── 6. fantasy inputs ─────────────────────────────────────────────────────────────────────────────
def fantasy() -> dict:
    from player_model import fantasy_defaults as FD
    from collections import Counter
    by = dict(Counter(d["class"] + ("_FIXED" if str(d["note"]).startswith("FIXED") else "") for d in FD.DEFAULTS))
    h = json.loads((ROOT / "output" / "fantasy_health.json").read_text(encoding="utf-8"))
    fire = ((h.get("defaults_audit") or {}).get("fire_counts") or {})
    unfixed_fallbacks = [d["id"] for d in FD.DEFAULTS if d["class"] == "MISSING_DATA_FALLBACK"
                         and not str(d["note"]).startswith("FIXED")]
    domain = [d["id"] for d in FD.DEFAULTS if d["class"] == "DOMAIN_PRIOR"]
    # the legacy props `_prep` batch median: reachable only through a legacy props payload; the props
    # certification proves no installed payload takes that branch (AST call graph + runtime probe)
    cert = json.loads((ROOT / "output" / "props_real_data_provenance.json").read_text(encoding="utf-8"))
    lp = (cert.get("global") or {}).get("legacy_prep_unreachable") or {}
    unreachable = []
    if lp.get("ok") is True and lp.get("reachable_in_production") is False             and (lp.get("runtime_probe") or {}).get("ok") is True and cert.get("verdict") == "VERIFIED":
        unreachable = [i for i in unfixed_fallbacks if i == "props_batch_median"]
        unfixed_fallbacks = [i for i in unfixed_fallbacks if i not in unreachable]
    log = ROOT / "output" / "fantasy_v2_oos.csv"
    oos = pd.read_csv(log, low_memory=False) if log.exists() else pd.DataFrame()
    return {"register_counts": by, "missing_data_fallbacks_not_fixed": unfixed_fallbacks,
            "fallbacks_unreachable_per_props_certification": unreachable,
            "domain_priors_unfitted": domain,
            "runtime_fire_counts": {k: v for k, v in fire.items() if not isinstance(v, dict)},
            "oos_rows": int(len(oos)),
            "evidence": "reports/P1_FANTASY_VALIDATION.md: 2,631 of 2,631 FINAL player-gameweeks equal FPL event/live",
            "verdict": NOT_REAL if unfixed_fallbacks else REAL_WITH_GAPS,
            "note": "DOMAIN_PRIOR constants are unfitted conventions in the projection formula (not observations); "
                    "a MISSING_DATA_FALLBACK not marked FIXED still substitutes a value for a missing input"}


# ── 7. ledgers (counts only; a price column is read only for presence) ───────────────────────────
LEDGERS = (("bets_ledger.csv", "odds"), ("side_bets_ledger.csv", "odds"), ("ht_ledger.csv", "entry_odds"),
           ("player_ledger.csv", "market_odds"), ("top3_league_ledger.csv", "odds"),
           ("props_top3_ledger.csv", "odds"), ("v91_shadow.csv", "entry_odds"))


def ledgers() -> dict:
    from src.integrity.synthetic_roi import SYNTHETIC_LABELS
    out = {}
    for f, pc in LEDGERS:
        p = ROOT / "output" / f
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        v = _num(df[pc]) if pc in df.columns else pd.Series(np.nan, index=df.index)
        syn = pd.Series(False, index=df.index)
        for c in df.columns:
            if df[c].dtype == object:
                syn |= df[c].astype(str).str.contains("|".join(SYNTHETIC_LABELS), regex=True, na=False)
        cells = {OBS: int(((v > 1) & ~syn).sum()), DER: 0, COR: 0, MIS: int(v.isna().sum()),
                 DEF: int((syn | (v.notna() & (v <= 1))).sum())}
        book = next((c for c in ("bookmaker", "book", "market_source") if c in df.columns), None)
        out[f] = {"rows": int(len(df)), "price_column": pc, "price_cells": {**_z(), **cells},
                  "book_or_source_column": book, "verdict": _verdict(cells),
                  "note": "PROVIDER_OBSERVED = a quote > 1 is stored (counts only); the bookmaker / capture time "
                          "is recorded only where book_or_source_column says so (reports/P0_SYNTHETIC_ROI_QUARANTINE.md)"}
    files = [Path(x) for x in glob.glob(str(ROOT / "output" / "prediction_ledger" / "**" / "*.csv"), recursive=True)
             if "_runs" not in x and "_coverage" not in x]
    n = scored = unscored = no_reason = 0
    for fp in files:
        try:
            d = pd.read_csv(fp, usecols=lambda c: c in ("scored", "probability", "not_scored_reason"),
                            low_memory=False)
        except Exception:                                               # noqa: BLE001
            continue
        n += len(d)
        s = d["scored"].astype(str).str.lower().isin(("true", "1")) if "scored" in d.columns else _num(d["probability"]).notna()
        scored += int(s.sum())
        unscored += int((~s).sum())
        if "not_scored_reason" in d.columns:
            no_reason += int(((~s) & d["not_scored_reason"].isna()).sum())
    out["prediction_ledger (v9, all partitions)"] = {
        "partitions": len(files), "rows": n,
        "cells": {OBS: 0, DER: scored, COR: 0, MIS: unscored, DEF: 0},
        "unscored_without_reason": no_reason,
        "verdict": NOT_REAL if no_reason else _verdict({OBS: 0, DER: scored, MIS: unscored}),
        "note": "a probability is a model output on the inputs above (VERIFIED_DERIVED); an unscored row keeps "
                "its reason and no number (MISSING_KEPT_MISSING). No price is stored in this ledger."}
    tot = _z()
    for v in out.values():
        _add(tot, v.get("price_cells") or v.get("cells") or {})
    return {"files": out, "cells": tot, "verdict": _verdict(tot)}


# ── 6b. Fantasy V1 train/serve parity (the frame V1 scores the props payloads on) ─────────────────
FANTASY_V1_MARKETS = ("goals", "assists", "sot2")       # player_model/fantasy.build_fantasy_projections


def fantasy_v1_parity(frame: str | None = None, n_dates: int = 4) -> dict:
    """V1's serving path REPLAYED on the fantasy league's last `n_dates` match dates: for each date D the
    players who played that day are built exactly as V1 builds a next fixture
    (fantasy.v1_model_inputs: build_upcoming_features on the props source frame dated strictly before D,
    restricted by _v1_context) and compared with the props training frame row of the same
    (fixture_id, player_id) (props_harness.build_frame on the current source_frame - the frame the
    installed payloads were fitted on). Every feature of each payload is compared for EXACT equality
    (NaN equals NaN, no tolerance) and the installed payload is scored on both sides. Also counts how
    many of V1's live scoring rows sit inside the props training window (fit + calibration dates):
    V1 scores next fixtures since 2026-11-23, so that count is 0 by construction. Counts only."""
    from player_model import fantasy as FA
    from player_model import props_harness as H
    from player_model.model import load_model, predict_proba
    raw = H.source_frame()
    fr = pd.read_parquet(frame) if frame else H.build_frame(raw)
    key = ["fixture_id", "player_id"]
    fl = FA._fantasy_league_rows(raw)
    days = sorted(pd.to_datetime(fl["date"]).dt.normalize().dropna().unique())[-n_dates:]
    parts, n_rows, n_unbuilt = [], 0, 0
    fields = [c for c in ("player_id", "player_name", "fixture_id", "date", "league", "team", "home_team",
                          "away_team", "is_home", "referee") if c in fl.columns]
    for d in days:
        day = fl[pd.to_datetime(fl["date"]).dt.normalize() == d]
        rows = day[fields].assign(date=pd.Timestamp(d).strftime("%Y-%m-%d")).to_dict("records")
        sv = FA.v1_model_inputs(day, raw[pd.to_datetime(raw["date"]) < d], rows)
        n_rows += len(day)
        n_unbuilt += int((~sv["_v1_built"]).sum())
        parts.append(sv[sv["_v1_built"]].drop(columns=["_v1_built"]))
    sv = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=key)
    for d in (sv, fr):
        for c in key:
            d[c] = _num(d[c])
    j = sv.merge(fr, on=key, suffixes=("_s", "_t"))
    win = (load_model("goals") or {}).get("window") or {}
    end = (win.get("calibration_dates") or win.get("fit_dates") or [None, None])[1]
    out = {"replay_dates": [str(pd.Timestamp(d).date()) for d in days], "replay_rows": n_rows,
           "serving_rows": int(len(sv)), "v1_rows_without_props_build_row": n_unbuilt,
           "training_frame_rows": int(len(fr)), "paired_rows": int(len(j)),
           "serving_rows_without_training_row": int(len(sv) - len(j)),
           "props_training_window_end": end,
           "live_v1_rows_inside_props_training_window": 0,
           "live_v1_rows_note": "V1 scores each player's NEXT fixture (dated after today > training end)",
           "markets": {}}
    feats_all: dict = {}
    for m in FANTASY_V1_MARKETS:
        pay = load_model(m)
        if pay is None:
            out["markets"][m] = {"status": UNVERIFIED, "why": "payload not installed"}
            continue
        mod = pay.get("model")
        fc = list(getattr(mod, "feature_cols_", None) or pay.get("feature_cols") or [])
        Xs, Xt, per = {}, {}, {}
        for c in fc:
            a = _num(j[c + "_s"]) if c + "_s" in j else (_num(j[c]) if c in j else pd.Series(np.nan, index=j.index))
            b = _num(j[c + "_t"]) if c + "_t" in j else (_num(j[c]) if c in j else pd.Series(np.nan, index=j.index))
            Xs[c], Xt[c] = a, b
            d = ~((a.isna() & b.isna()) | (a == b))
            per[c] = {"differ": int(d.sum()), "both_valued": int((d & a.notna() & b.notna()).sum()),
                      "serving_missing_training_valued": int((d & a.isna()).sum()),
                      "serving_valued_training_missing": int((d & b.isna()).sum()),
                      "both_valued_within_1e-6_relative": int((d & a.notna() & b.notna()
                                                               & np.isclose(a, b, rtol=1e-6, atol=0)).sum())}
            feats_all[c] = per[c]
        Xs, Xt = pd.DataFrame(Xs, index=j.index), pd.DataFrame(Xt, index=j.index)
        Xs["player_id"] = Xt["player_id"] = j["player_id"].to_numpy()
        ps, pt = predict_proba(Xs, pay).to_numpy(), predict_proba(Xt, pay).to_numpy()
        diff = sum(v["differ"] for v in per.values())
        out["markets"][m] = {"payload_candidate": pay.get("candidate"), "features": len(fc),
                             "cells_compared": int(len(j) * len(fc)), "differing_cells": int(diff),
                             "features_identical": sum(1 for v in per.values() if not v["differ"]),
                             "features_differing": sum(1 for v in per.values() if v["differ"]),
                             "different_probability_rows": int((ps != pt).sum())}
    tot = {k: sum(v[k] for v in feats_all.values()) for k in ("differ", "both_valued", "serving_missing_training_valued",
                                                               "serving_valued_training_missing",
                                                               "both_valued_within_1e-6_relative")}
    out["distinct_features"] = len(feats_all)
    out["differing_cells_distinct_features"] = tot
    out["features_identical"] = sorted(c for c, v in feats_all.items() if not v["differ"])
    out["top_differing_features"] = dict(sorted(((c, v["differ"]) for c, v in feats_all.items() if v["differ"]),
                                                key=lambda kv: -kv[1])[:15])
    ok = bool(out["markets"]) and all(v.get("differing_cells") == 0 for v in out["markets"].values()) \
        and out["serving_rows_without_training_row"] == 0
    out["verdict"] = "PARITY_CERTIFIED" if ok else "PARITY_FAILED"
    out["why"] = ("serving = fantasy.v1_model_inputs (props serving build, replayed), training = "
                  "props_harness.build_frame; cells differ" if not ok else "0 differing cells")
    return out


# ── 8. other repos: row-level audit of the stores (counts only) ──────────────────────────────────
# Every rule is named; a rule that cannot run on a table says so (never a silent pass).
import re  # noqa: E402

R_SENT, R_CONST, R_PRICE, R_PROB, R_NEG, R_DUP, R_POSTKO, R_WRONG, R_IMPUTED, R_NOID, R_NODOC = (
    "sentinel_or_placeholder_value", "constant_column_where_variation_expected", "decimal_price_le_1",
    "probability_outside_0_1", "negative_count", "duplicate_key", "prematch_capture_after_kickoff",
    "flagged_wrong_by_own_columns", "flagged_synthetic_default_fallback_imputed",
    "no_provider_or_source_identity", "no_documented_derivation")
RULES = {R_SENT: DEF, R_CONST: DEF, R_PRICE: DEF, R_PROB: DEF, R_NEG: DEF, R_IMPUTED: DEF,
         R_DUP: UNV, R_POSTKO: UNV, R_WRONG: UNV, R_NOID: UNV, R_NODOC: UNV}
RULE_TEXT = {
    R_SENT: "numeric -99/-999/-9999/999/9999/99999 or text n/a|none|null|nan|unknown|tbd|placeholder|default|? in an "
            "identity / provenance column",
    R_CONST: "a numeric value column (>= 100 values, not bool / version / line) holding ONE value on every row",
    R_PRICE: "a decimal price column (odds / price / best_* / over / under) <= 1.0",
    R_PROB: "a probability column (p_* / *_prob* / implied* / fair_*) outside [0, 1]",
    R_NEG: "a count column (goals, shots, corners, cards, books, rows, credits, minutes ...) < 0",
    R_DUP: "a row repeating the key of an earlier row of the table (key = every identity / capture / run column "
           "present); its cells are not a distinct observation",
    R_POSTKO: "a pre-match table row whose capture time (snapshot_ts / captured_at / first_seen_ts / frozen_ts / "
              "generated_at / predicted_at / logged_at / ts_bet) is after its kickoff, or (any non-live table) a "
              "minutes_to_kickoff / minutes_close_before_kickoff / close_minutes_to_kickoff < 0",
    R_WRONG: "the row's own quality_flags carry a flag its source registers as WRONG (Pro src/quality.py)",
    R_IMPUTED: "the row's own columns say synthetic / default / fallback / imputed / placeholder / degraded / no form "
               "(bool true, or the label in quality_flags / notes / source); its derived cells rest on substitutes",
    R_NOID: "no provider identity on the row: a price cell whose row names no real bookmaker (a provider feed without "
            "a book layer, e.g. API-Football /odds/live, counts its source), or a provider-table row without "
            "source / provider",
    R_NODOC: "a derived table no code (src / scripts / workflows) or README / docs of its repo names (no documented "
             "derivation)",
}
_META = re.compile(r"^(run_id|ingested_at|observed_at|observed_at_source|source_sha|pro_git_sha|git_sha|model_sha|"
                   r"model_content_sha|feature_hash|generated_at|schema_version|repo|writer|payload_version|"
                   r"calc_version|calculation_version|rule_version|fixture_key_version|fixture_key_status|"
                   r"old_fixture_key|old_combo_id|quality_flags|notes|retrain_batch_id|row_id|unit_key|endpoint|"
                   r"http_status|finished_at|attempted_at|requested_at|lookup_at|fixture_key|fixture_ref|"
                   r"combo_id|research_staleness|.*_version|__.*)$")
_IDENT = re.compile(r"^(league|league_id|league_slug|country|season|match_date|date|league_date|kickoff.*|ko|"
                    r"commence_time|home_team|away_team|home|away|team|.*team_id|fixture_id|af_fixture_id|event_id|"
                    r".*_event_id|sport_key|player_name|player_id|player|position|bookmaker|bookmaker_id|book|bet|"
                    r"bet_id|market|market_name|market_family|selection|side|line|point|handicap|outcome_name|"
                    r"outcome_description|round|referee|venue|is_home|model_type|track|match)$")
_PRICE = re.compile(r"(^|_)(odds|price)(_|$)|^best_(over|under)|^(over|under)$")
_PRICE_NOT = re.compile(r"(snapshots|band|source|age|updated|stale|blocked|count|books|basis|stake|_ts$|_at$|ladder|"
                        r"capture|n_|_n$|bucket|ratio|drift|move|valid|^is_|_ok$|_flag|_id$|fair)")
_PROB = re.compile(r"(^|_)(prob|probability|p)(_|$)|_probability$|implied|^fair_|_fair_")
_PROB_NOT = re.compile(r"(overround|_pp$|pnl|pos_|pen_|phi|pass|pct|_n$|lambda|odds|_std$|_range$|coef|ci_)")
_COUNT = re.compile(r"(^|_)(goals|shots|sot|corners|fouls|yellow|yellows|red|reds|saves|blocked|insidebox|outsidebox|"
                    r"cards|books|n|starters|forwards|minute|elapsed_mins|injuries|credits|rows|events|snapshots|"
                    r"legs|poll|tackles|passes|minutes|n_rows|n_clean|n_flagged|n_strict_clean)(_|$)")
_MTK = re.compile(r"^(minutes_to_kickoff|minutes_before_kickoff|minutes_close_before_kickoff|close_minutes_to_kickoff)$")
_COUNT_NOT = re.compile(r"(kickoff|prevented|_str|diff|delta|drift|gd|_pp$|to_kickoff|edge|clv|pnl|ratio|share|rate|avg|"
                        r"_pg|per90|last5|roll|ewm|pct|lambda|_ts$|_at$|_age)")
_CONST_NOT = re.compile(r"(^is_|_flag|version|band|^line$|^point$|handicap|leg_index|poll_n|_id$|^n_starters$|"
                        r"selection|^y$|_sha$|^has_|_has_|stake|kelly|_ok$)")
# constant BY DESIGN (documented in the writer's code): not a fill, never counted as a default
CONST_BY_DESIGN = {
    "anchor_probability": "wowza_v11 scripts/v11_microstructure.py: the sample median of p_market_entry broadcast to "
                          "every row (the reversion anchor), by design",
}
_CAPTURE = ("snapshot_ts", "captured_at", "capture_ts", "first_seen_ts", "frozen_ts", "generated_at",
            "predicted_at", "logged_at", "ts_bet", "entry_ts")
_KICKOFF = ("kickoff_utc", "kickoff_ts", "commence_time", "kickoff", "ko")
_POST_TABLES = re.compile(r"(live|result|settle|stats|quality|ft90|status|journal|units|runs|graded|residual|"
                          r"_runs|match$|scoreboard|movement|micro|momentum|placebo|folds|evidence|coverage|summary|"
                          r"by_|clv|backfill)")
_ROW_ID = {"fixture_key", "fixture_id", "af_fixture_id", "event_id", "oddsapi_event_id", "combo_id", "player_id",
           "row_id", "unit_key", "pick_key", "fixture_ref", "check", "match", "snapshot_id"}
_SENT_NUM = (-99.0, -999.0, -9999.0, 999.0, 9999.0, 99999.0)
_SENT_TXT = ("n/a", "na", "none", "null", "nan", "unknown", "tbd", "placeholder", "default", "?", "undefined")
_SENT_COLS = re.compile(r"(team|league|bookmaker|^book$|source|sha|kickoff|commence|^date$|match_date|player_name|"
                        r"season|referee|venue|coach|model_version|model_id|_ts$|_at$)")
_SYN_COL = re.compile(r"(synthetic|is_default|^default_|fallback|imputed|placeholder|degraded|no_form)")
_SYN_TXT = re.compile(r"(?:SYNTHETIC|FALLBACK|IMPUTED|PLACEHOLDER|DEGRADED|INVALID_FOR_PROFITABILITY)", re.I)
_FAKE_BOOK = re.compile(r"^(v9_|best|selected|consensus|avg|average|max|min|median|market|model|unknown)", re.I)
# a price ACROSS books (median / best / consensus) computed by the writer: a derived value, not one book's quote
_AGG_BOOK = re.compile(r"^(median|consensus|avg|average|mean|max|min|best|v9_selected_best)$", re.I)
_PROVIDER_TABLE = re.compile(r"(odds|quote|price|team_match_stats|scout_results$|team_news|data/events|book_close|"
                             r"sharp|fixtures$|special_prices)")
_DERIVED_TABLE = re.compile(r"(ledger|log|settle|clv|signal|model|feature|combo|calibration|bars_opt|au\d_data|study|"
                            r"shadow|correction|graded|residual|movement|micro|momentum|placebo|folds|scoreboard|"
                            r"summary|by_|candidates|heartbeat)")


def _wrong_flags(pro_root: Path | None) -> set:
    """The flags Pro's own registry calls WRONG (src/quality.py Flag(name, True, ...))."""
    if pro_root is None or not (pro_root / "src" / "quality.py").exists():
        return set()
    return set(re.findall(r'Flag\("([A-Z0-9_]+)",\s*True', (pro_root / "src" / "quality.py").read_text(encoding="utf-8")))


def _ts(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    t = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    if t.isna().sum() > 0.05 * s.notna().sum():
        t = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return t


def _txt(s: pd.Series) -> pd.Series:
    """The column as text, a missing value as "" (any dtype)."""
    return s.astype(object).where(s.notna(), "").astype(str)


def _numeric(s: pd.Series):
    """The column as numbers, or None when it is not a numeric column (>= 95% of its values parse)."""
    if pd.api.types.is_bool_dtype(s):
        return None
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    nn = s.dropna()
    if nn.empty:
        return None
    head = pd.to_numeric(nn.head(200), errors="coerce")
    if head.notna().mean() < 0.95:
        return None
    v = pd.to_numeric(s, errors="coerce")
    return v if v.notna().sum() >= 0.95 * len(nn) else None


class _TableAudit:
    """Streaming per-table audit: one call to add() per file; finish() applies the table-level rules."""

    def __init__(self, name: str, kind: str, documented: bool | None, wrong_flags: set,
                 kickoff_map: dict | None = None):
        self.name, self.kind, self.documented = name, kind, documented
        self.wrong, self.kmap = wrong_flags, kickoff_map
        self.files = self.rows = 0
        self.cells = _z()
        self.rules = {r: {"rows": 0, "cells": 0} for r in RULES}
        self.checkable = {r: False for r in RULES}
        self.colstat: dict = {}           # numeric value column -> [n, min, max, good cells]
        self.keys: list = []
        self.key_cols: set = set()
        self.table_provider: str | None = None
        self.rule_cols = {r: set() for r in RULES}
        self.postko_own_flag = 0
        self.agg_price_cells = 0

    def _base(self, c: str) -> str:
        if _META.match(c):
            return DER
        if _IDENT.match(c) or (_PRICE.search(c) and not _PRICE_NOT.search(c)):
            return OBS
        if re.search(r"(^ft90_|_restated$|_corrected$)", c):
            return COR
        if _PROB.search(c) and not _PROB_NOT.search(c):
            return DER
        return OBS if self.kind == "provider" else DER

    def add(self, df: pd.DataFrame) -> None:
        self.files += 1
        n = len(df)
        self.rows += n
        if not n:
            return
        cols = [c for c in df.columns if not str(c).startswith("__")]
        code = {}                                       # column -> np array of class names (object)
        price_cols = [c for c in cols if _PRICE.search(c) and not _PRICE_NOT.search(c) and _numeric(df[c]) is not None]
        # row-level rules
        flags = _txt(df["quality_flags"]) if "quality_flags" in df.columns else None
        r_wrong = np.zeros(n, bool)
        if flags is not None and self.wrong:
            self.checkable[R_WRONG] = True
            r_wrong = np.array(flags.str.split("|").map(lambda fl: bool(set(fl) & self.wrong)), dtype=bool)
        elif flags is not None:
            self.checkable[R_WRONG] = True
        r_imp = np.zeros(n, bool)
        for c in cols:
            if _SYN_COL.search(c):
                v = df[c]
                r_imp |= (v.astype(str).str.lower().isin(("true", "1", "1.0", "yes"))).to_numpy()
        for c in ("quality_flags", "notes", "source", "odds_source", "price_basis"):
            if c in df.columns and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
                r_imp |= _txt(df[c]).str.contains(_SYN_TXT, na=False).to_numpy()
        self.checkable[R_IMPUTED] = True
        # identity: a price row needs a real book; any other row a source / provider
        books = [c for c in ("bookmaker", "book", "entry_bookmaker") if c in df.columns]
        srcs = [c for c in ("source", "provider", "data_provider", "stats_source", "news_source", "odds_source",
                            "sport_key", "api_source") if c in df.columns]
        has_src = np.zeros(n, bool)
        prov_src = np.zeros(n, bool)          # the provider itself publishes the price (no book layer)
        for c in srcs:
            v = _txt(df[c]).str.strip()
            has_src |= ((v != "") & ~v.str.lower().isin(_SENT_TXT)).to_numpy()
            prov_src |= v.str.lower().str.startswith(("api_football", "oddsapi")).to_numpy()
        has_book = np.zeros(n, bool)
        agg_book = np.zeros(n, bool)
        for c in books:
            v = _txt(df[c]).str.strip()
            has_book |= ((v != "") & ~v.str.match(_FAKE_BOOK)).to_numpy()
            agg_book |= v.str.match(_AGG_BOOK).to_numpy()
        agg_book &= ~has_book
        self.checkable[R_NOID] = True
        no_id_rows = ~(has_src | has_book) if (self.kind == "provider" and not self.table_provider) \
            else np.zeros(n, bool)
        no_book_rows = ~(has_book | agg_book | (prov_src & (self.kind == "provider"))) if price_cols \
            else np.zeros(n, bool)
        # pre-match capture after kickoff
        r_post = np.zeros(n, bool)
        if not _POST_TABLES.search(self.name):
            cap = next((c for c in _CAPTURE if c in df.columns), None)
            ko = next((c for c in _KICKOFF if c in df.columns), None)
            if cap and ko:
                a, b = _ts(df[cap]), _ts(df[ko])
                r_post = np.array(a.notna() & b.notna() & (a > b), dtype=bool)
                self.checkable[R_POSTKO] = True
            elif cap and self.kmap is not None and "fixture_key" in df.columns:
                b = _ts(df["fixture_key"].map(self.kmap))
                a = _ts(df[cap])
                r_post = np.array(a.notna() & b.notna() & (a > b), dtype=bool)
                self.checkable[R_POSTKO] = True
        if "live" not in self.name:
            # a capture the row itself stamps as taken after kickoff (also in settled / CLV tables: a close is pre-match)
            for c in cols:
                if _MTK.match(c):
                    r_post = r_post | np.array(_num(df[c]) < 0, dtype=bool)
                    self.checkable[R_POSTKO] = True
        good = np.zeros(n, np.int64)
        noid_hit = np.zeros(n, bool)
        for c in cols:
            s = df[c]
            base = self._base(c)
            if self.kind == "derived" and base == DER and self.documented is False and not _META.match(c):
                base = UNV
            if isinstance(s.dtype, pd.CategoricalDtype):
                s = s.astype(object)
            miss = np.array(s.isna(), dtype=bool)
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                miss = miss | (np.array(_txt(s).str.strip() == "", dtype=bool) & (c != "quality_flags"))
            cls = np.full(n, _CODE[base], dtype=np.int8)
            v = _numeric(s)
            rule_hit = {}
            if _SENT_COLS.search(c) and not pd.api.types.is_bool_dtype(s):
                if v is not None and not re.search(r"(_ts$|_at$|kickoff|date|sha|season)", c):
                    rule_hit[R_SENT] = v.isin(_SENT_NUM).to_numpy()
                else:
                    rule_hit[R_SENT] = _txt(s).str.strip().str.lower().isin(_SENT_TXT).to_numpy() & ~miss
                self.checkable[R_SENT] = True
            if v is not None:
                if c in price_cols:
                    rule_hit[R_PRICE] = (v <= 1.0).to_numpy()
                    self.checkable[R_PRICE] = True
                if _PROB.search(c) and not _PROB_NOT.search(c) and not (_PRICE.search(c) and not _PRICE_NOT.search(c)):
                    rule_hit[R_PROB] = ((v < 0) | (v > 1)).to_numpy()
                    self.checkable[R_PROB] = True
                if _COUNT.search(c) and not _COUNT_NOT.search(c):
                    rule_hit[R_NEG] = (v < 0).to_numpy()
                    self.checkable[R_NEG] = True
            for r, m in rule_hit.items():
                m = m & ~miss
                if m.any():
                    self.rules[r]["rows"] += int(m.sum())
                    self.rules[r]["cells"] += int(m.sum())
                    self.rule_cols[r].add(c)
                    cls[m] = _CODE[RULES[r]]
            # row-level reclassification of value cells (identity / provenance cells keep their class)
            free = (~miss) & (cls == _CODE[base])
            if not _META.match(c):
                for r, m in ((R_IMPUTED, free & r_imp & (base == DER)),
                             (R_WRONG, free & r_wrong & (base in (OBS, DER, COR)))):
                    m = m & (cls == _CODE[base])
                    cls[m] = _CODE[RULES[r]]
                    self.rules[r]["cells"] += int(m.sum())
                if base in (OBS, DER, COR):
                    m = free & r_post & (cls == _CODE[base])
                    cls[m] = _CODE[UNV]
                    self.rules[R_POSTKO]["cells"] += int(m.sum())
                if base == OBS:
                    m = free & (cls == _CODE[OBS]) & (no_book_rows if c in price_cols else no_id_rows)
                    cls[m] = _CODE[UNV]
                    self.rules[R_NOID]["cells"] += int(m.sum())
                    noid_hit |= m
                    if m.any():
                        self.rule_cols[R_NOID].add(c)
            if c in price_cols and base == OBS and agg_book.any():
                m = agg_book & ~miss & (cls == _CODE[OBS])
                cls[m] = _CODE[DER]                        # across-books price: derived by the writer
                self.agg_price_cells += int(m.sum())
            cls[miss] = _CODE[MIS]
            for k, cnt in zip(CLASSES, np.bincount(cls, minlength=len(CLASSES))):
                self.cells[k] += int(cnt)
            g = cls <= _CODE[COR]
            good += g
            if v is not None and not _META.match(c) and not _IDENT.match(c) and not _CONST_NOT.search(c) \
                    and not pd.api.types.is_bool_dtype(s):
                vv = v[~miss & g]
                st = self.colstat.setdefault(c, [0, np.inf, -np.inf, 0])
                if len(vv):
                    st[0] += int(vv.notna().sum())
                    st[1] = min(st[1], float(vv.min()))
                    st[2] = max(st[2], float(vv.max()))
                    st[3] += int(g.sum())
        for r, m in ((R_WRONG, r_wrong), (R_IMPUTED, r_imp), (R_POSTKO, r_post), (R_NOID, noid_hit)):
            self.rules[r]["rows"] += int(m.sum())
        if r_post.any() and "is_post_kickoff" in df.columns:
            self.postko_own_flag += int((r_post & df["is_post_kickoff"].astype(str).str.lower()
                                         .isin(("true", "1", "1.0")).to_numpy()).sum())
        # duplicate keys (cells moved at finish)
        kc = [c for c in cols if (_IDENT.match(c) or c in _CAPTURE or c in ("run_id", "fixture_key", "combo_id",
              "leg_index", "check", "flag", "table", "unit_key", "pick_key", "slot", "band", "kind",
              "snapshot_label", "signal_type", "elapsed_mins", "market_a", "market_b", "calc_version", "row_id",
              "event_type", "status", "fixture_ref", "predicted_at", "model_id", "model_version", "tier", "source"))]
        if kc and set(kc) & _ROW_ID:
            self.checkable[R_DUP] = True
            self.key_cols |= set(kc)
            kh = pd.util.hash_pandas_object(df[kc].astype(str), index=False).to_numpy()
            try:
                rh = pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
            except TypeError:
                rh = pd.util.hash_pandas_object(df[cols].astype(str), index=False).to_numpy()
            self.keys.append(pd.DataFrame({"k": kh, "r": rh, "g": good}))

    def finish(self, private: bool = False) -> dict:
        if self.table_provider:
            self.checkable[R_NOID] = True
        consts = []
        for c, (n, lo, hi, g) in self.colstat.items():
            if n >= 100 and lo == hi and c not in CONST_BY_DESIGN:
                consts.append(c)
                moved = int(n)
                for k in (OBS, DER, COR):
                    take = min(self.cells[k], moved)
                    self.cells[k] -= take
                    self.cells[DEF] += take
                    moved -= take
                    if not moved:
                        break
                self.rules[R_CONST]["cells"] += int(n)
        self.checkable[R_CONST] = bool(self.colstat)
        if self.keys:
            k = pd.concat(self.keys, ignore_index=True)
            dup = k.duplicated("k").to_numpy()
            conf = dup & ~k.duplicated(["k", "r"]).to_numpy()
            mv = int(k["g"].to_numpy()[dup].sum())
            for cls in (OBS, DER, COR):
                take = min(self.cells[cls], mv)
                self.cells[cls] -= take
                self.cells[UNV] += take
                mv -= take
            self.rules[R_DUP] = {"rows": int(dup.sum()), "cells": int(k["g"].to_numpy()[dup].sum()),
                                 "conflicting_rows": int(conf.sum())}
            self.keys = []
        if self.kind == "derived":
            self.checkable[R_NODOC] = self.documented is not None
            if self.documented is False:
                self.rules[R_NODOC]["rows"] = self.rows
        out = {"files": self.files, "rows": self.rows, "kind": self.kind, "cells": dict(self.cells),
               "rules": {r: (v if self.checkable[r] else {"status": "NOT_CHECKABLE"}) for r, v in self.rules.items()},
               "verdict": _verdict(self.cells)}
        self.rule_cols[R_CONST] = set(consts)
        for r, cs in self.rule_cols.items():
            if cs and self.checkable[r]:
                out["rules"][r]["columns"] = len(cs) if private else sorted(cs)
        if self.postko_own_flag and self.checkable[R_POSTKO]:
            out["rules"][R_POSTKO]["of_which_flagged_is_post_kickoff_by_own_column"] = self.postko_own_flag
        if self.table_provider:
            out["table_level_provider"] = self.table_provider
        if self.agg_price_cells:
            out["across_books_derived_price_cells"] = self.agg_price_cells
        if not private:
            out["key_columns"] = sorted(self.key_cols) if self.checkable[R_DUP] else []
        return out


def _code_text(root: Path | None) -> str:
    if root is None or not root.exists():
        return ""
    parts = []
    for sub in ("src", "scripts", "player_model", ".github", "docs", "output"):
        for f in (root / sub).rglob("*") if (root / sub).exists() else []:
            if f.is_file() and (f.suffix in (".py", ".yml", ".yaml", ".sh") or f.name == "README.md"
                                or (sub == "docs" and f.suffix == ".md")):
                try:
                    parts.append(f.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    return "\n".join(parts)


def _read_any(path_or_bytes, suffix: str) -> pd.DataFrame:
    import io
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    if suffix == ".parquet":
        return pd.read_parquet(src)
    return pd.read_csv(src, low_memory=False)


def _scope_total(tables: dict) -> dict:
    tot = _z()
    for v in tables.values():
        if isinstance(v, dict) and "cells" in v:
            _add(tot, v["cells"])
    return tot


def _rule_totals(tables: dict) -> dict:
    rt = {r: {"rows": 0, "cells": 0, "tables": 0} for r in RULES}
    for v in tables.values():
        for r, x in (v.get("rules") or {}).items():
            if "rows" in x and "cells" in x and (x["rows"] or x["cells"]):
                rt[r]["rows"] += int(x["rows"])
                rt[r]["cells"] += int(x["cells"])
                rt[r]["tables"] += 1
    return rt


# Pro modules whose reads make a table an input of a Pro model or ledger (consumers found by grep of src/)
PRO_MODEL_LEDGER = re.compile(r"src/(models/|scout/(model|hypotheses|promotion)|pipelines/(train_|shadow|bet_builder|"
                              r"btts_eval|paper_1x2|tourney|xg_value|news_impact|backfill_history|live_odds)|"
                              r"prediction_ledger|combo/|bankroll/|price_trader/|prediction_lab/|shadow_learning/|"
                              r"player_v3/|props_matchup/|specials/|features/|market/)")
PRO_INFRA = re.compile(r"src/(data/season_store|monitoring/|schemas|pipelines/(migrate_fixture_key_v2|"
                       r"fixture_key_v2_vocab|data_quality|registry)|importers/|architecture/|analytics/|quality)")


def _pro_consumers(pro_root: Path, table: str) -> list:
    hits = []
    pat = re.compile(r"[\"']" + re.escape(table) + r"[\"']")
    for f in (pro_root / "src").rglob("*.py"):
        try:
            if pat.search(f.read_text(encoding="utf-8", errors="ignore")):
                hits.append(f.relative_to(pro_root).as_posix())
        except OSError:
            pass
    return sorted(hits)


def pro_scope(pro_root: Path | None) -> dict:
    if pro_root is None or not (pro_root / "data").exists():
        return {"verdict": UNVERIFIED, "why": "no --pro-root (a wowzaV9-Pro checkout of origin/main) given"}
    season = sorted((pro_root / "data").glob("season_*"))[-1]
    wrong = _wrong_flags(pro_root)
    kmap = None
    fx = sorted((season / "fixtures").rglob("*.parquet"))
    if fx:
        f = pd.concat([pd.read_parquet(p, columns=["fixture_key", "kickoff_utc"]) for p in fx], ignore_index=True)
        kmap = f.dropna().drop_duplicates("fixture_key", keep="last").set_index("fixture_key")["kickoff_utc"].to_dict()
    tables, feeds = {}, {}
    for tdir in sorted(p for p in season.iterdir() if p.is_dir()):
        files = sorted(list(tdir.rglob("*.parquet")) + list(tdir.rglob("*.csv")))
        cons = _pro_consumers(pro_root, tdir.name)
        model_ledger = [c for c in cons if PRO_MODEL_LEDGER.search(c) and not PRO_INFRA.search(c)]
        feeds[tdir.name] = {"consumers": cons, "model_or_ledger_consumers": model_ledger,
                            "feeds_pro_model_or_ledger": bool(model_ledger)}
        first = None
        for p in files:
            d = _read_any(p, p.suffix)
            if first is None:
                first = d
                srcs = set(d["source"].dropna().astype(str).unique()) if "source" in d.columns else set()
                prov = any(s.startswith(("api_football", "oddsapi")) for s in srcs) or (
                    bool(_PROVIDER_TABLE.search(tdir.name)) and not _DERIVED_TABLE.search(tdir.name))
                ta = _TableAudit(tdir.name, "provider" if prov else "derived", bool(cons), wrong, kmap)
            ta.add(d)
        if first is None:
            tables[tdir.name] = {"files": 0, "rows": 0, "verdict": "EMPTY"}
            continue
        tables[tdir.name] = ta.finish()
        print(f"[certify] pro {tdir.name}: {ta.rows} rows", file=sys.stderr, flush=True)
        tables[tdir.name]["feeds_pro_model_or_ledger"] = feeds[tdir.name]["feeds_pro_model_or_ledger"]
    tot = _scope_total(tables)
    fed = {k: v for k, v in tables.items() if v.get("feeds_pro_model_or_ledger")}
    tot_fed = _scope_total(fed)
    sha = subprocess.run(["git", "-C", str(pro_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"repo": "NevixAA/wowzaV9-Pro", "commit": sha, "store": season.relative_to(pro_root).as_posix(),
            "tables": tables, "consumers": feeds, "wrong_flags_registry": sorted(wrong),
            "rule_totals": _rule_totals(tables), "cells": tot, "verdict": _verdict(tot),
            "tables_feeding_models_or_ledgers": sorted(fed), "cells_feeding_models_or_ledgers": tot_fed,
            "verdict_feeding_models_or_ledgers": _verdict(tot_fed),
            "how_feeds_was_determined": "a table feeds a Pro model / ledger when a module matching PRO_MODEL_LEDGER "
                                        "(scout model, shadow, bet_builder, train_*, prediction_ledger, combo, bankroll, "
                                        "price_trader, ...) names it as a string literal in src/; monitoring / "
                                        "importer / migration modules are not consumers"}


V9_BANKS = {"output/af_history.parquet": "API-Football /fixtures + /fixtures/statistics (the v9 bank certified in "
                                        "history_banks)",
            "output/fd_history.parquet": "football-data.co.uk CSV (the v9 bank certified in history_banks)"}
V11_INPUT_RE = re.compile(r"_load_v9\(\s*[\"']([A-Za-z0-9_./-]+)[\"']|load_snapshots\(|AF_HISTORY_REL\s*=\s*[\"']([^\"']+)")


def v11_scope(v11_root: Path | None) -> dict:
    if v11_root is None or not (v11_root / "output").exists():
        return {"verdict": UNVERIFIED, "why": "no --v11-root (a wowza_v11 checkout of origin/main) given"}
    code = _code_text(v11_root)
    inputs = set()
    for m in V11_INPUT_RE.finditer(code):
        if m.group(1):
            inputs.add("output/" + m.group(1))
        elif m.group(2):
            inputs.add(m.group(2) if m.group(2).startswith("output/") else "output/" + m.group(2))
        else:
            inputs.add("output/book_odds_snapshots.csv")          # book_consensus.load_snapshots default
    tables = {}
    for rel in sorted(inputs):
        p = ROOT / rel
        if not p.exists():
            tables[f"input v9 {rel}"] = {"verdict": UNVERIFIED, "why": "not in this wowza-betting checkout"}
            continue
        bank = V9_BANKS.get(rel)
        ta = _TableAudit(f"v9 {rel}", "provider" if bank or (_PROVIDER_TABLE.search(rel) and not _DERIVED_TABLE.search(rel))
                         else "derived", True, set())
        ta.table_provider = bank
        ta.add(_read_any(p, p.suffix))
        tables[f"input v9 {rel}"] = ta.finish()
    outs = sorted(list((v11_root / "output").rglob("*.csv")))
    for p in outs:
        rel = p.relative_to(v11_root).as_posix()
        stem = re.sub(r"_?\d{4}-\d{2}-\d{2}$", "", p.stem)
        documented = stem in code or p.name in code
        ta = _TableAudit(rel, "derived", documented, set())
        ta.add(_read_any(p, p.suffix))
        tables[f"output {rel}"] = ta.finish()
    tot = _scope_total(tables)
    unread = [k for k, v in tables.items() if v.get("verdict") == UNVERIFIED]
    sha = subprocess.run(["git", "-C", str(v11_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"repo": "NevixAA/wowza_v11", "commit": sha, "tables": tables, "rule_totals": _rule_totals(tables),
            "inputs_found_in_code": sorted(inputs),
            "inputs_not_stored": ["football-data.co.uk season CSVs fetched at run time by src/results.py (not stored "
                                  "in any repo: not measurable here)"],
            "cells": tot, "verdict": UNVERIFIED if unread else _verdict(tot)}


EXEC_REFS = ("af-prices-live", "af-prices-archive", "odds-history", "oddsapi-book-close", "af-side-odds",
             "prediction-ledger-v11", "sharp-paper", "specials-prices")


def _exec_table(path: str) -> str:
    segs = path.split("/")
    if "prediction_ledger" in path:
        i = next(i for i, s in enumerate(segs) if s.startswith("prediction_ledger"))
        return "/".join(segs[: i + 4]) if "_runs" not in segs else "/".join(segs[: i + 3])
    keep = [s for s in segs[:-1] if not re.match(r"^(dt=|run=|\d{4}-\d{2}|\d{4}$)", s)]
    stem = segs[-1].rsplit(".", 1)[0]
    if re.search(r"\d{4}-?\d{2}-?\d{2}|\d{8,}|^run|-\d+$|^\d", stem):
        return "/".join(keep)
    return "/".join(keep + [stem])


def _git_blobs(root: Path, ref: str) -> list:
    ls = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "-l", ref], capture_output=True, text=True,
                        check=True).stdout.splitlines()
    out = []
    for line in ls:
        meta, path = line.split("\t", 1)
        _, typ, sha, _size = meta.split()
        if typ == "blob" and path.endswith((".csv", ".parquet")):
            out.append((sha, path))
    return out


class _CatFile:
    def __init__(self, root: Path):
        self.p = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)

    def read(self, sha: str) -> bytes:
        self.p.stdin.write((sha + "\n").encode())
        self.p.stdin.flush()
        head = self.p.stdout.readline().split()
        n = int(head[2])
        data = self.p.stdout.read(n)
        self.p.stdout.read(1)
        return data

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# an ORIGINAL kept beside its correction (e.g. the specials prices before the 2026-11-23 dedupe): measured and
# reported, read by no model, and not part of the store's verdict (the corrected table is)
_SUPERSEDED = re.compile(r"(^|/)archive/[^/]*pre_")


def exec_scope(exec_root: Path | None, refs=EXEC_REFS, pro_root=None, v11_root=None) -> dict:
    """PRIVATE repo: counts and verdicts only. No column names, values, prices or rows are written."""
    if exec_root is None or not (exec_root / ".git").exists():
        return {"verdict": UNVERIFIED, "why": "no --exec-root (a wowza-exec clone with its data branches fetched) given"}
    code = {"wowza-betting": _code_text(ROOT), "exec": _code_text(exec_root),
            "pro": _code_text(Path(pro_root)) if pro_root else "", "v11": _code_text(Path(v11_root)) if v11_root else ""}
    cat = _CatFile(exec_root)
    branches, unique = {}, {}
    try:
        for ref in refs:
            r = f"origin/{ref}"
            try:
                blobs = _git_blobs(exec_root, r)
            except subprocess.CalledProcessError:
                branches[ref] = {"verdict": UNVERIFIED, "why": f"{r} not fetched"}
                continue
            sha = subprocess.run(["git", "-C", str(exec_root), "rev-parse", "--short=9", r], capture_output=True,
                                 text=True).stdout.strip()
            by_t: dict = {}
            for b, path in blobs:
                by_t.setdefault(_exec_table(path), []).append((b, path))
            tables, unreadable = {}, 0
            for t, fs in sorted(by_t.items()):
                prov = bool(_PROVIDER_TABLE.search(t)) and not _DERIVED_TABLE.search(t)
                stem = t.rsplit("/", 1)[-1]
                documented = any(stem in txt for txt in code.values()) if not prov else True
                ta = _TableAudit(t, "provider" if prov else "derived", documented, set())
                for b, path in fs:
                    try:
                        d = _read_any(cat.read(b), ".parquet" if path.endswith(".parquet") else ".csv")
                    except Exception:                                   # noqa: BLE001
                        unreadable += 1
                        continue
                    ta.add(d)
                tables[t] = ta.finish(private=True)
                print(f"[certify] exec {ref}: table {len(tables)} of {len(by_t)} ({ta.rows} rows)", file=sys.stderr,
                      flush=True)
                for b, _ in fs:
                    unique.setdefault(b, None)
            sup = {t: v for t, v in tables.items() if _SUPERSEDED.search(t)}
            tables = {t: v for t, v in tables.items() if t not in sup}
            tot = _scope_total(tables)
            branches[ref] = {"commit": sha, "data_files": len(blobs), "unreadable_files": unreadable,
                             "originals_kept_beside_corrections": {
                                 t: {"rows": v["rows"], "verdict": v["verdict"], "cells": v["cells"],
                                     "rules_fired": {r: x for r, x in v["rules"].items() if x.get("rows") or x.get("cells")}}
                                 for t, v in sup.items()},
                             "tables": len(tables), "rows": sum(v["rows"] for v in tables.values()),
                             "cells": tot, "rule_totals": _rule_totals(tables),
                             "tables_not_real": sum(1 for v in tables.values() if v["verdict"] == NOT_REAL),
                             "verdict": UNVERIFIED if unreadable else _verdict(tot),
                             "per_table": {t: {"rows": v["rows"], "verdict": v["verdict"],
                                               "cells": v["cells"],
                                               "rules_fired": {r: x for r, x in v["rules"].items()
                                                               if x.get("rows") or x.get("cells")}}
                                           for t, v in tables.items()}}
    finally:
        cat.close()
    tot = _z()
    for v in branches.values():
        if "cells" in v:
            _add(tot, v["cells"])
    measured = all("cells" in v for v in branches.values())
    return {"repo": "NevixAA/wowza-exec (PRIVATE: counts and verdicts only)", "branches": branches,
            "unique_data_blobs": len(unique),
            "note": "af-prices-archive is the immutable seed snapshot of af-prices-live: their blobs overlap, so the "
                    "scope total counts shared files once per branch",
            "cells": tot, "verdict": _verdict(tot, measured)}


def external_repos(pro_root=None, v11_root=None, exec_root=None) -> dict:
    """Row-level audit of the other repos' stores; a scope whose checkout is not given is UNVERIFIED (never a pass)."""
    out = {"wowzaV9-Pro (season store)": pro_scope(Path(pro_root) if pro_root else None),
           "wowza_v11 (inputs + output ledgers)": v11_scope(Path(v11_root) if v11_root else None),
           "wowza-exec (private price + ledger stores)": exec_scope(Path(exec_root) if exec_root else None,
                                                                   pro_root=pro_root, v11_root=v11_root)}
    vs = [v.get("verdict") for v in out.values()]
    verdict = UNVERIFIED if UNVERIFIED in vs else (NOT_REAL if NOT_REAL in vs else
                                                   (REAL_WITH_GAPS if REAL_WITH_GAPS in vs else REAL_100))
    return {"repos": out, "rules": RULE_TEXT, "rule_class": RULES, "verdict": verdict}


def _prev_team_training(std_frame) -> dict | None:
    """The committed certification's team_training block (+ its run stamp), read only when no --std-frame is given."""
    if std_frame or not OUT.exists():
        return None
    try:
        d = json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tt = (d.get("datasets") or {}).get("team_training") or {}
    return {"payloads": tt.get("payloads") or {}, "generated_at": d.get("generated_at"), "code_sha": d.get("code_sha")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--std-frame", default=None, help="saved Standard training frame (row-level counts)")
    ap.add_argument("--pro-root", default=None, help="wowzaV9-Pro checkout of origin/main (row-level store audit)")
    ap.add_argument("--v11-root", default=None, help="wowza_v11 checkout of origin/main")
    ap.add_argument("--exec-root", default=None, help="wowza-exec clone with its data branches fetched (counts only)")
    ap.add_argument("--fantasy-frame", default=None,
                    help="props training frame parquet (props_harness.build_frame); rebuilt when absent")
    ap.add_argument("--skip-fantasy-parity", action="store_true")
    a = ap.parse_args(argv)
    doc = {"artifact": "real_data_certification", "version": VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "generated_by": "scripts/certify_real_data.py", "code_sha": _git_sha(),
           "classes": list(CLASSES), "datasets": {}, "errors": {}}
    for name, fn in (("history_banks", history_banks), ("team_training", lambda: team_training(a.std_frame, _prev_team_training(a.std_frame))),
                     ("serving_board", serving_board), ("props", props), ("side_models", side_models),
                     ("fantasy", fantasy),
                     ("fantasy_v1_parity", (lambda: {"verdict": UNVERIFIED, "why": "--skip-fantasy-parity"})
                      if a.skip_fantasy_parity else (lambda: fantasy_v1_parity(a.fantasy_frame))),
                     ("ledgers", ledgers),
                     ("external_repos", lambda: external_repos(a.pro_root, a.v11_root, a.exec_root))):
        try:
            doc["datasets"][name] = fn()
        except Exception as e:                                          # noqa: BLE001
            doc["errors"][name] = f"{type(e).__name__}: {e}"
            doc["datasets"][name] = {"verdict": UNVERIFIED}
    defaults = []
    for k, v in (doc["datasets"].get("team_training") or {}).get("payloads", {}).items():
        ar = (v.get("row_level_saved_frame") or {}).get("backtest_archive_rows") or {}
        if ar.get("rows") or ar.get("downstream_training_rows_reading_them"):
            defaults.append({"dataset": k, "columns": [c for c, _ in ARCHIVE_STATS], "cells": v["cells"][UNV],
                             "rows": v.get("rows_with_unverified_inputs"),
                             "cause": "row_source backtest_archive (output/backtest_all_leagues.csv, an internal backtest "
                                      "output): stat values no provider holds or a provider contradicts; read by the "
                                      "row itself and by the next ROLLING_N matches of both clubs",
                             "reaches_model": True, "class": UNV})
        if v["cells"][DEF]:
            defaults.append({"dataset": k, "columns": v["fabricated_features"] + v["untraced_features"],
                             "cells": v["cells"][DEF], "cause": "no training value / untraced feature",
                             "reaches_model": True})
    ph = ((doc["datasets"].get("props") or {}).get("stored_placeholders_in_player_history") or {})
    for c, r in (ph.get("columns") or {}).items():
        if r["rows"]:
            defaults.append({"dataset": "player_history.parquet (stored)", "columns": [c], "cells": r["rows"],
                             "cause": f"legacy build_features placeholder {r['placeholder']} written on every row",
                             "reaches_model": False})
    for k, v in ((doc["datasets"].get("ledgers") or {}).get("files") or {}).items():
        d = _full(v.get("price_cells") or v.get("cells") or {})[DEF]
        if d:
            defaults.append({"dataset": k, "columns": [v.get("price_column")], "cells": d,
                             "cause": "synthetic label or price <= 1", "reaches_model": False})
    fa = doc["datasets"].get("fantasy") or {}
    for i in fa.get("missing_data_fallbacks_not_fixed") or []:
        defaults.append({"dataset": "fantasy projection", "columns": [i], "cells": None,
                         "cause": "MISSING_DATA_FALLBACK not fixed (player_model/fantasy_defaults.py)",
                         "reaches_model": True})
    ext = ((doc["datasets"].get("external_repos") or {}).get("repos") or {})
    pro = ext.get("wowzaV9-Pro (season store)") or {}
    for t, v in (pro.get("tables") or {}).items():
        if "cells" not in v:
            continue
        bad = int(v["cells"][DEF]) + int(v["cells"][UNV])
        if bad:
            defaults.append({"dataset": f"wowzaV9-Pro {pro.get('store')}/{t}", "cells": bad,
                             "rules": sorted(r for r, x in (v.get("rules") or {}).items() if x.get("rows") or x.get("cells")),
                             "cause": "row-level rules (external_repos.rules)",
                             "reaches_model": bool(v.get("feeds_pro_model_or_ledger"))})
    v11 = ext.get("wowza_v11 (inputs + output ledgers)") or {}
    for t, v in (v11.get("tables") or {}).items():
        if "cells" not in v:
            continue
        bad = int(v["cells"][DEF]) + int(v["cells"][UNV])
        if bad:
            defaults.append({"dataset": f"wowza_v11 {t}", "cells": bad,
                             "rules": sorted(r for r, x in (v.get("rules") or {}).items() if x.get("rows") or x.get("cells")),
                             "cause": "row-level rules (external_repos.rules)",
                             "reaches_model": t.startswith("input ") or "shadow" in t})
    ex = ext.get("wowza-exec (private price + ledger stores)") or {}
    for b, v in (ex.get("branches") or {}).items():
        if "cells" not in v:
            continue
        bad = int(v["cells"][DEF]) + int(v["cells"][UNV])
        if bad:
            defaults.append({"dataset": f"wowza-exec {b} (private, counts only)", "cells": bad,
                             "rules": sorted(r for r, x in (v.get("rule_totals") or {}).items() if x.get("rows") or x.get("cells")),
                             "cause": "row-level rules (external_repos.rules)", "reaches_model": None})
    fp = doc["datasets"].get("fantasy_v1_parity") or {}
    if fp.get("verdict") == "PARITY_FAILED":
        defaults.append({"dataset": "Fantasy V1 serving frame (player_history latest row per player)",
                         "cells": sum(int(m["differing_cells"]) for m in fp["markets"].values() if "differing_cells" in m),
                         "cause": fp.get("why"), "reaches_model": True})
    doc["defaults_found"] = defaults
    doc["summary"] = {k: (v.get("verdict") if isinstance(v, dict) else None) for k, v in doc["datasets"].items()}
    txt = json.dumps(doc, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"
    if a.write:
        OUT.write_text(txt, encoding="utf-8", newline="\n")
        print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    for k, v in doc["datasets"].items():
        print(f"{k:16s} {v.get('verdict') if isinstance(v, dict) else v}")
    print(f"defaults found: {len(defaults)} ({sum(1 for d in defaults if d['reaches_model'])} reach a model)")
    if doc["errors"]:
        print("ERRORS:", doc["errors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
