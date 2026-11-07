"""Walk-forward and holdout stages. Everything lands in ONE run folder:

    <out>/<run_id>/manifest.json         config + sha256, code sha, frame hash, stages, errors
    <out>/<run_id>/predictions.parquet   stage, fold, track, market, candidate, row_id, date, league, y, p
    <out>/<run_id>/metrics.json          per stage x track x market x candidate: scores, calibration, CLV
    <out>/<run_id>/per_league.csv        the same per league (leagues under min_league_n marked)
    <out>/<run_id>/comparison.csv        the pre-registered comparison, Holm-corrected, with verdicts
    <out>/<run_id>/models/...            holdout-stage models (fitted on the walk-forward region only)
    <out>/<run_id>/holdout_opened.json   written BEFORE the holdout is scored; a second holdout stage
                                         in the same run folder is refused

Nothing here writes outside the run folder, touches models/, the registry or the promotion gate.

With owner_decisions.new_format_ht_variant on, every new_format candidate that reads features also
runs as '<name>+ht' (the HT block restored, protocol.feature_sets); both are in the Holm family.
If the variant cannot run, the run is refused before anything is written.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrain_harness import candidates as C
from src.retrain_harness import evaluate as E
from src.retrain_harness import protocol as P

log = logging.getLogger(__name__)

STAGES = ("walk_forward", "holdout")


def _code_sha() -> str:
    sha = os.getenv("GITHUB_SHA", "").strip()
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=P.ROOT, timeout=20).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _row_ids(df: pd.DataFrame) -> pd.Series:
    return (pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") + "|" + df["league"].astype(str) + "|"
            + df["home_team"].astype(str) + "|" + df["away_team"].astype(str))


def enabled_candidates(cfg: dict, only=None) -> list[str]:
    names = [n for n in P.CANDIDATES if n in cfg["candidates"] and cfg["candidates"][n].get("enabled")]
    if only:
        names = [n for n in names if n in set(only)]
    ref = cfg["comparison"]["reference"]
    if ref not in names:
        raise ValueError(f"the reference candidate {ref!r} must be enabled")
    return names


def _score(runs, cfg, track, market, fsets, frames, stage, fold, errors, models_dir=None):
    """`runs`: (label, base candidate, feature-set suffix) from protocol.candidate_runs;
    `fsets`: {suffix: (features, excluded)} from protocol.feature_sets."""
    out = []
    test = frames["test"]
    if not len(test):
        return out
    y = pd.to_numeric(test[cfg["markets"][market]["target"]], errors="coerce").to_numpy(float)
    ids = _row_ids(test).to_numpy()
    for name, base, suffix in runs:
        feats = fsets[suffix][0]
        cand = C.build(base, cfg, track, market, feats)
        cand.name = name
        try:
            cand.fit(frames)
            p = np.asarray(cand.predict(test), float)
        except Exception as e:                                         # noqa: BLE001
            errors.append({"stage": stage, "fold": fold, "track": track, "market": market,
                           "candidate": name, "error": f"{type(e).__name__}: {e}"})
            log.warning(f"  [{stage} f{fold}] {track}/{market}/{name} failed: {e}")
            continue
        if models_dir is not None:
            d = models_dir / track / market
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{name}.pkl", "wb") as fh:
                pickle.dump({"candidate": cand, "meta": cand.meta(), "features": feats,
                             "train_dates": [str(pd.to_datetime(frames["train"]["date"]).min().date()),
                                             str(pd.to_datetime(frames["train"]["date"]).max().date())]}, fh)
        out.append(pd.DataFrame({"stage": stage, "fold": fold, "track": track, "market": market,
                                 "candidate": name, "row_id": ids, "date": test["date"].to_numpy(),
                                 "league": test["league"].astype(str).to_numpy(), "y": y, "p": p,
                                 "deployable": cand.deployable}))
        if cand.notes:
            errors.append({"stage": stage, "fold": fold, "track": track, "market": market,
                           "candidate": name, "note": "; ".join(cand.notes)})
    return out


def run(feat: pd.DataFrame, cfg: dict, out_dir, *, stages=STAGES, only=None, tracks=None, env=None,
        data_info: dict | None = None, run_id: str | None = None) -> Path:
    """Fit and score. Refuses unless protocol.fit_blockers is empty. `tracks` narrows the run to
    some of the configured tracks (one workflow job per track); it never adds one."""
    P.assert_fit_allowed(cfg, env)
    tracks = list(tracks or cfg["tracks"])
    if set(tracks) - set(cfg["tracks"]):
        raise ValueError(f"tracks {sorted(set(tracks) - set(cfg['tracks']))} are not in the config")
    if P.HT_VARIANT_TRACK in tracks:
        b = P.ht_variant_blockers(cfg, feat.columns)        # the frame check: never skip silently
        if b:
            raise PermissionError("fitting refused: " + "; ".join(b))
    for s in stages:
        if s not in STAGES:
            raise ValueError(f"unknown stage {s!r}")
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    folder = P.assert_run_folder(Path(out_dir) / run_id)
    folder.mkdir(parents=True, exist_ok=True)
    if "holdout" in stages and (folder / "holdout_opened.json").exists():
        raise PermissionError(f"the holdout was already scored in {folder}; it is scored once per run")
    names = enabled_candidates(cfg, only)
    manifest = {"harness_version": cfg.get("harness_version"), "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config_sha256": P.config_sha(cfg), "config": cfg, "code_sha": _code_sha(),
                "data": data_info or {}, "stages": list(stages), "tracks": tracks, "candidates": names,
                "promotion": "none - the owner decides; this run wrote no serving file"}
    preds, errors, features, runs_by_track = [], [], {}, {}
    for track in tracks:
        fsets = P.feature_sets(cfg, track, feat.columns)
        runs = P.candidate_runs(cfg, track, names)
        runs_by_track[track] = [r[0] for r in runs]
        features[track] = {"features": fsets[""][0], "excluded": fsets[""][1],
                           "variants": {s: {"features": f, "excluded": ex} for s, (f, ex) in fsets.items() if s}}
        for market in cfg["markets"]:
            rows = P.eligible_rows(feat, cfg, track, market)
            pre, hold, _after = P.split_holdout(rows, cfg)
            if "walk_forward" in stages:
                for fold in P.walk_forward_folds(pre, cfg):
                    fr = P.fold_frames(pre, fold)
                    P.assert_disjoint(fr)
                    preds += _score(runs, cfg, track, market, fsets, fr, "walk_forward", fold["fold"], errors)
            if "holdout" in stages and len(hold):
                (folder / "holdout_opened.json").write_text(json.dumps(
                    {"opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "holdout": cfg["holdout"]}, indent=2), encoding="utf-8")
                fr = P.final_frames(pre, hold, cfg)
                P.assert_disjoint(fr)
                preds += _score(runs, cfg, track, market, fsets, fr, "holdout", -1, errors,
                                models_dir=folder / "models")
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(
        columns=["stage", "fold", "track", "market", "candidate", "row_id", "date", "league", "y", "p"])
    pred.to_parquet(folder / "predictions.parquet", index=False)
    write_reports(pred, feat, cfg, folder)
    manifest.update(features=features, errors=errors, candidates_by_track=runs_by_track,
                    finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return folder


def write_reports(pred: pd.DataFrame, feat: pd.DataFrame, cfg: dict, folder: Path) -> None:
    bins = int(cfg["comparison"]["ece_bins"])
    out = {}
    lookup = None
    if len(pred):
        lookup = feat.assign(row_id=_row_ids(feat)).drop_duplicates("row_id").set_index("row_id")
    for key, g in pred.groupby(["stage", "track", "market", "candidate"], sort=True):
        stage, track, market, cand = key
        m = E.metrics(g["y"], g["p"], bins)
        m["calibration_table"] = E.calibration_table(g["y"], g["p"], bins) if m["n"] else []
        m["deployable"] = bool(g["deployable"].iloc[0]) if "deployable" in g else None
        rows = lookup.reindex(g["row_id"])
        m["clv"] = E.clv(rows.reset_index(drop=True), g["p"].to_numpy(float), cfg["markets"][market], market)
        out.setdefault(stage, {}).setdefault(track, {}).setdefault(market, {})[cand] = m
    (folder / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if len(pred):
        E.per_league(pred, cfg).to_csv(folder / "per_league.csv", index=False)
    cmp_ = E.compare(pred, cfg) if len(pred) else pd.DataFrame()
    cmp_.to_csv(folder / "comparison.csv", index=False)
