"""Row-level REAL DATA certification of every dataset that feeds a model (owner 2026-11-23: "the most
important achievement is 100% real data for every row in our data set, for all the models").

    python scripts/certify_real_data.py                       # print the summary
    python scripts/certify_real_data.py --write               # + output/real_data_certification.json
    python scripts/certify_real_data.py --write --std-frame <saved Standard training frame .parquet>
                                                              # + row-level counts on the saved frame

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


def team_training(std_frame: str | None) -> dict:
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
            if feat is not None and track == "standard":
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


# ── 8. other repos (Pro grid inputs, v11 inputs, exec stores): not measured row by row here ───────
EXTERNAL = (
    ("wowzaV9-Pro (grid / research inputs)", "reports/P1_PRO_RECONCILIATION.md",
     "registry reconciled as-of its own timestamp (26 tables); calculation versions deduped; no row-level "
     "provenance class per value"),
    ("wowza_v11 (research inputs)", "reports/P1_V11_RECONCILIATION.md",
     "ledger identity / schema / chronology reconciled with v9 and Pro (0 duplicate keys, 0 post-kickoff rows); "
     "no row-level provenance class per value"),
    ("wowza-exec (private price + ledger stores)", "reports/P1_PRICE_CLV_COVERAGE.md",
     "dual-write stores; certified-close coverage measured; no row-level provenance class per value"),
)


def external_repos() -> dict:
    """Recorded, never counted as a pass: a scope this script does not measure row by row is UNVERIFIED."""
    out = {}
    for name, evidence, what in EXTERNAL:
        out[name] = {"verdict": UNVERIFIED, "evidence": evidence,
                     "evidence_present": (ROOT / evidence).exists(), "what_is_proven": what}
    return {"repos": out, "verdict": UNVERIFIED}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--std-frame", default=None, help="saved Standard training frame (row-level counts)")
    a = ap.parse_args(argv)
    doc = {"artifact": "real_data_certification", "version": VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "generated_by": "scripts/certify_real_data.py", "code_sha": _git_sha(),
           "classes": list(CLASSES), "datasets": {}, "errors": {}}
    for name, fn in (("history_banks", history_banks), ("team_training", lambda: team_training(a.std_frame)),
                     ("serving_board", serving_board), ("props", props), ("side_models", side_models),
                     ("fantasy", fantasy), ("ledgers", ledgers), ("external_repos", external_repos)):
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
