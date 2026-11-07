"""The pre-registered protocol: config, guards, eligible rows, features, folds, final holdout.

Nothing in this module fits a model. `build_plan` only counts rows and lists columns, which is what
`--dry-run` prints from the committed data.

SPLITS. Per (track, market) the eligible rows are cut on whole calendar dates:

    |---- walk-forward region (dates < holdout.start) ----|-- HOLDOUT (start..end) --|-- unused --|
     [ min_train_matchdays ][ block 1 ][ block 2 ] ...

  * fold k: TRAIN = every eligible row dated before the block (expanding window); its last
    `validation_matchdays` dates are the VALIDATION block (model+market blend weights only);
    TEST = the block's `block_matchdays` dates.
  * HOLDOUT: rows dated holdout.start..holdout.end inclusive. No fold ever contains one, and the
    holdout stage scores them once, with models fitted on the walk-forward region only.
  * rows dated after holdout.end are not used at all (counted in the plan).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "config" / "phase2_retrain.json"
ENABLE_VAR = "PHASE2_RETRAIN_ENABLED"
TRACKS = ("standard", "new_format")
CANDIDATES = ("champion_recipe", "logistic", "hgb", "dixon_coles", "market_only", "model_market")
#: Folders a run may never write into (serving models, registry, committed outputs).
FORBIDDEN_RUN_ROOTS = ("models", "output", "config", "src", "registry")
#: Candidates that read the feature list (dixon_coles reads goals, market_only reads prices).
FEATURE_CANDIDATES = ("champion_recipe", "logistic", "hgb", "model_market")
#: owner_decisions.new_format_ht_variant (owner 2026-11-07): the track that gets the HT variant,
#: the suffix of the variant's candidate names, and the HT history indicators of the HT block.
HT_VARIANT_TRACK = "new_format"
HT_VARIANT_SUFFIX = "+ht"
HT_HISTORY_COLS = ("home_has_ht_history", "away_has_ht_history")


# ── config ──────────────────────────────────────────────────────────────────────────────────

def load_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    cfg = json.loads(p.read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def config_sha(cfg: dict) -> str:
    """sha256 of the config as canonical JSON - recorded in every run manifest."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


def validate_config(cfg: dict) -> None:
    """Structural checks. A missing required value raises; nothing is filled in."""
    for key in ("holdout", "walk_forward", "tracks", "markets", "features", "candidates",
                "comparison", "data"):
        if key not in cfg:
            raise ValueError(f"phase2 retrain config has no {key!r} block")
    hs, he = holdout_window(cfg)
    if he < hs:
        raise ValueError(f"holdout end {he.date()} is before its start {hs.date()}")
    unknown = set(cfg["tracks"]) - set(TRACKS)
    if unknown:
        raise ValueError(f"unknown tracks {sorted(unknown)}")
    unknown = set(cfg["candidates"]) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates {sorted(unknown)}")
    wf = cfg["walk_forward"]
    for k in ("min_train_matchdays", "block_matchdays", "validation_matchdays"):
        if not isinstance(wf.get(k), int) or wf[k] <= 0:
            raise ValueError(f"walk_forward.{k} must be a positive integer")
    if wf["validation_matchdays"] >= wf["min_train_matchdays"]:
        raise ValueError("walk_forward.validation_matchdays must be smaller than min_train_matchdays")
    for name, m in cfg["markets"].items():
        for k in ("target", "kind", "goals"):
            if k not in m:
                raise ValueError(f"market {name!r} has no {k!r}")
    cmp_ = cfg["comparison"]
    if cmp_.get("reference") not in cfg["candidates"]:
        raise ValueError("comparison.reference must be one of the candidates")
    if cmp_.get("correction") != "holm":
        raise ValueError("comparison.correction must be 'holm' (pre-registered)")
    od = cfg.get("owner_decisions") or {}
    if "new_format_ht_variant" in od and not isinstance(od["new_format_ht_variant"], bool):
        raise ValueError("owner_decisions.new_format_ht_variant must be true or false")


def holdout_window(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    h = cfg["holdout"]
    if not h.get("start") or not h.get("end"):
        raise ValueError("holdout.start and holdout.end must both be set in the config")
    return pd.Timestamp(h["start"]).normalize(), pd.Timestamp(h["end"]).normalize()


def holdout_approved(cfg: dict) -> bool:
    h = cfg["holdout"]
    return bool(str(h.get("approved_by") or "").strip()) and bool(str(h.get("approved_at") or "").strip())


def fit_blockers(cfg: dict, env=None) -> list[str]:
    """Every reason fitting must not start. Empty list = allowed."""
    env = os.environ if env is None else env
    out = []
    if not holdout_approved(cfg):
        out.append("holdout dates not approved by the owner (config holdout.approved_by / approved_at)")
    if str(env.get(ENABLE_VAR, "")).strip().lower() != "true":
        out.append(f"{ENABLE_VAR} is not 'true' (repository variable; set only after owner approval)")
    out += ht_variant_blockers(cfg)
    return out


def assert_fit_allowed(cfg: dict, env=None) -> None:
    b = fit_blockers(cfg, env)
    if b:
        raise PermissionError("fitting refused: " + "; ".join(b))


def assert_run_folder(path: Path | str, root: Path | None = None) -> Path:
    """The run folder may not be (or be inside) a production folder of the repository."""
    root = (root or ROOT).resolve()
    p = Path(path).resolve()
    for name in FORBIDDEN_RUN_ROOTS:
        bad = (root / name).resolve()
        if p == bad or bad in p.parents:
            raise PermissionError(f"run folder {p} is inside {name}/ - runs write to their own folder only")
    if p == root:
        raise PermissionError("run folder may not be the repository root")
    return p


# ── features ────────────────────────────────────────────────────────────────────────────────

def feature_list(cfg: dict, track: str, columns, restore=()) -> tuple[list[str], dict]:
    """(features, excluded{column: reason}) for a track, given the frame's columns.

    A configured column that the frame does not have is EXCLUDED and listed - never created.
    `restore` lifts those columns out of the track's exclusion list (the HT variant); it never
    adds a column that is not in an included group or that is price-derived."""
    fcfg = cfg["features"]
    cols = set(columns)
    never = set(fcfg.get("never_model_inputs") or ())
    track_ex = set((fcfg.get("track_exclude") or {}).get(track) or ()) - set(restore)
    feats, excluded = [], {}
    for gname, g in fcfg["groups"].items():
        for c in g["columns"]:
            if not g.get("include"):
                excluded[c] = f"group {gname} not included"
            elif c in never:
                excluded[c] = "price-derived (never a model input)"
            elif c in track_ex:
                excluded[c] = f"excluded for track {track}"
            elif c not in cols:
                excluded[c] = "not in the feature frame"
            elif c not in feats:
                feats.append(c)
    return feats, excluded


# ── the new_format HT variant (owner_decisions.new_format_ht_variant) ───────────────────────────

def ht_variant_on(cfg: dict) -> bool:
    return (cfg.get("owner_decisions") or {}).get("new_format_ht_variant") is True


def ht_block(cfg: dict) -> list[str]:
    """The HT block: the HT columns of the new_format exclusion list (every name with '_ht_'; xG and
    inside-box are not HT and stay excluded) plus the HT history indicators."""
    ex = (cfg["features"].get("track_exclude") or {}).get(HT_VARIANT_TRACK) or ()
    out = [c for c in ex if "_ht_" in c]
    return out + [c for c in HT_HISTORY_COLS if c not in out]


def ht_variant_blockers(cfg: dict, columns=None) -> list[str]:
    """Every reason the HT variant cannot run while the owner flag is on. Empty = it can run, or
    the flag is off. With `columns` (the feature frame's columns) the frame is checked too."""
    if not ht_variant_on(cfg):
        return []
    why = []
    if HT_VARIANT_TRACK not in cfg["tracks"]:
        why.append(f"track {HT_VARIANT_TRACK} is not configured")
    block = ht_block(cfg)
    ex = set((cfg["features"].get("track_exclude") or {}).get(HT_VARIANT_TRACK) or ())
    if not [c for c in block if c in ex]:
        why.append(f"features.track_exclude.{HT_VARIANT_TRACK} holds no HT column, so the variant "
                   "would equal the champion list")
    grouped = {c for g in cfg["features"]["groups"].values() if g.get("include") for c in g["columns"]}
    never = set(cfg["features"].get("never_model_inputs") or ())
    off = [c for c in block if c not in grouped or c in never]
    if off:
        why.append(f"HT block column(s) not in an included feature group: {off}")
    if not [n for n in FEATURE_CANDIDATES if (cfg["candidates"].get(n) or {}).get("enabled")]:
        why.append("no enabled candidate uses features")
    if columns is not None:
        cols = set(columns)
        missing = [c for c in block if c not in cols]
        if missing:
            why.append(f"HT block column(s) not in the feature frame: {missing}")
    return [f"new_format HT variant (owner_decisions.new_format_ht_variant) cannot run: {w}" for w in why]


def feature_sets(cfg: dict, track: str, columns) -> dict:
    """{suffix: (features, excluded)} for a track: '' is the champion exclusion list; '+ht' (new_format,
    flag on) is the same list with the HT block restored."""
    out = {"": feature_list(cfg, track, columns)}
    if ht_variant_on(cfg) and track == HT_VARIANT_TRACK:
        out[HT_VARIANT_SUFFIX] = feature_list(cfg, track, columns, restore=ht_block(cfg))
    return out


def candidate_runs(cfg: dict, track: str, names) -> list[tuple[str, str, str]]:
    """(label, base candidate, feature-set suffix) for every candidate a track trains and scores.
    With the HT variant on, every new_format candidate that uses features appears twice
    ('hgb' and 'hgb+ht'); market_only and dixon_coles read no features and appear once."""
    out = []
    for n in names:
        out.append((n, n, ""))
        if ht_variant_on(cfg) and track == HT_VARIANT_TRACK and n in FEATURE_CANDIDATES:
            out.append((n + HT_VARIANT_SUFFIX, n, HT_VARIANT_SUFFIX))
    return out


# ── rows and splits ─────────────────────────────────────────────────────────────────────────

def track_series(leagues: pd.Series) -> pd.Series:
    """'standard' | 'new_format' | None per row - the ONE routing rule (src.model.track_of)."""
    from src.model import track_of
    return pd.Series(track_of(leagues).to_numpy(), index=leagues.index)


def eligible_rows(feat: pd.DataFrame, cfg: dict, track: str, market: str) -> pd.DataFrame:
    """Rows of one track with the market's label and the required form present, date-sorted.
    Ties are broken by (league, home, away) so the order never depends on the input order."""
    m = cfg["markets"][market]
    tgt = m["target"]
    if tgt not in feat.columns:
        return feat.iloc[0:0].copy()
    d = feat[track_series(feat["league"]) == track]
    d = d[pd.to_numeric(d[tgt], errors="coerce").notna()]
    for c in cfg["data"].get("require_present") or ():
        if c not in d.columns:
            return d.iloc[0:0].copy()
        d = d[pd.to_numeric(d[c], errors="coerce").notna()]
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    tie = [c for c in ("league", "home_team", "away_team") if c in d.columns]
    return d.sort_values(["date"] + tie, kind="mergesort").reset_index(drop=True)


def split_holdout(rows: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(walk-forward region, holdout, after-holdout) - disjoint, whole dates."""
    hs, he = holdout_window(cfg)
    d = pd.to_datetime(rows["date"]).dt.normalize()
    pre = rows[d < hs]
    hold = rows[(d >= hs) & (d <= he)]
    after = rows[d > he]
    return pre, hold, after


def walk_forward_folds(pre: pd.DataFrame, cfg: dict) -> list[dict]:
    """Fold boundaries on the walk-forward region's matchdays. No fitting, no labels read."""
    wf = cfg["walk_forward"]
    dates = np.array(sorted(pd.to_datetime(pre["date"]).dt.normalize().unique()))
    n0, blk, nval = wf["min_train_matchdays"], wf["block_matchdays"], wf["validation_matchdays"]
    folds = []
    k, start = 0, n0
    while start < len(dates):
        end = min(start + blk, len(dates))
        folds.append({"fold": k,
                      "train_start": pd.Timestamp(dates[0]),
                      "validation_start": pd.Timestamp(dates[start - nval]),
                      "train_end": pd.Timestamp(dates[start - 1]),
                      "test_start": pd.Timestamp(dates[start]),
                      "test_end": pd.Timestamp(dates[end - 1])})
        k, start = k + 1, end
    return folds


def fold_frames(pre: pd.DataFrame, fold: dict) -> dict:
    """{'train', 'fit', 'validation', 'test'} for one fold. fit = train minus validation."""
    d = pd.to_datetime(pre["date"]).dt.normalize()
    train = pre[d <= fold["train_end"]]
    dt = pd.to_datetime(train["date"]).dt.normalize()
    return {"train": train,
            "fit": train[dt < fold["validation_start"]],
            "validation": train[dt >= fold["validation_start"]],
            "test": pre[(d >= fold["test_start"]) & (d <= fold["test_end"])]}


def final_frames(pre: pd.DataFrame, hold: pd.DataFrame, cfg: dict) -> dict:
    """The holdout stage: train on the whole walk-forward region, validation = its last block."""
    dates = sorted(pd.to_datetime(pre["date"]).dt.normalize().unique())
    nval = cfg["walk_forward"]["validation_matchdays"]
    if len(dates) <= nval:
        raise ValueError(f"walk-forward region has {len(dates)} matchdays; need more than {nval}")
    vstart = pd.Timestamp(dates[-nval])
    d = pd.to_datetime(pre["date"]).dt.normalize()
    return {"train": pre, "fit": pre[d < vstart], "validation": pre[d >= vstart], "test": hold}


def assert_disjoint(frames: dict) -> None:
    """fit < validation < test in time, and train never reaches the test block."""
    def _rng(f):
        d = pd.to_datetime(f["date"])
        return d.min(), d.max()
    if len(frames["fit"]) and len(frames["validation"]):
        if _rng(frames["fit"])[1] >= _rng(frames["validation"])[0]:
            raise AssertionError("fit block overlaps the validation block")
    if len(frames["train"]) and len(frames["test"]):
        if _rng(frames["train"])[1] >= _rng(frames["test"])[0]:
            raise AssertionError("training window overlaps the test block")


# ── the plan (dry run) ──────────────────────────────────────────────────────────────────────

def price_coverage(rows: pd.DataFrame, mcfg: dict) -> dict:
    """Rows with a usable two-sided price / with any price - counted, nothing computed."""
    pr = mcfg.get("price")
    if not pr:
        return {"two_sided": 0, "any": 0, "note": "no price in the frame for this market"}
    yes = pd.to_numeric(rows[pr["yes"]], errors="coerce") if pr.get("yes") in rows.columns else None
    no = pd.to_numeric(rows[pr["no"]], errors="coerce") if pr.get("no") and pr["no"] in rows.columns else None
    any_ = int((yes > 1).sum()) if yes is not None else 0
    if yes is None or no is None:
        return {"two_sided": 0, "any": any_, "note": "one-sided price only: no de-vig possible"}
    ok = (yes > 1) & (no > 1)
    flag = pr.get("pair_valid_flag")
    if flag and flag in rows.columns:
        ok &= rows[flag].astype("boolean").fillna(False).astype(bool)
    return {"two_sided": int(ok.sum()), "any": any_}


def _rows_complete(df: pd.DataFrame, cols: list) -> int:
    """Rows on which every column in `cols` holds a value (0 when there are no rows or columns)."""
    if not len(df) or not cols:
        return 0
    return int(df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).sum())


def build_plan(feat: pd.DataFrame, cfg: dict) -> dict:
    """Everything --dry-run prints. Counts rows and lists columns; fits nothing."""
    plan = {"harness_version": cfg.get("harness_version"), "config_sha256": config_sha(cfg),
            "holdout": {"start": str(holdout_window(cfg)[0].date()), "end": str(holdout_window(cfg)[1].date()),
                        "approved": holdout_approved(cfg)},
            "frame": {"rows": int(len(feat)),
                      "date_min": str(pd.to_datetime(feat["date"]).min().date()) if len(feat) else None,
                      "date_max": str(pd.to_datetime(feat["date"]).max().date()) if len(feat) else None},
            "ht_variant": {"on": ht_variant_on(cfg), "block": ht_block(cfg) if ht_variant_on(cfg) else [],
                           "blockers": ht_variant_blockers(cfg, feat.columns)},
            "tracks": {}}
    min_league = cfg["comparison"].get("min_league_n")
    enabled = [n for n in CANDIDATES if (cfg["candidates"].get(n) or {}).get("enabled")]
    for track in cfg["tracks"]:
        sets = feature_sets(cfg, track, feat.columns)
        feats, excluded = sets[""]
        t = {"features": feats, "n_features": len(feats), "excluded_features": excluded,
             "candidates": [lbl for lbl, _b, _s in candidate_runs(cfg, track, enabled)], "markets": {},
             "variants": {}}
        for suf, (vf, vex) in sets.items():
            if suf:
                t["variants"][suf] = {"features": vf, "n_features": len(vf), "excluded_features": vex,
                                      "restored": [c for c in vf if c not in feats],
                                      "candidates": [lbl for lbl, _b, s in candidate_runs(cfg, track, enabled)
                                                     if s == suf],
                                      "markets": {}}
        for market, mcfg in cfg["markets"].items():
            rows = eligible_rows(feat, cfg, track, market)
            pre, hold, after = split_holdout(rows, cfg)
            folds = walk_forward_folds(pre, cfg) if len(pre) else []
            fold_rows = []
            for f in folds:
                fr = fold_frames(pre, f)
                fold_rows.append({"fold": f["fold"], "test_start": str(f["test_start"].date()),
                                  "test_end": str(f["test_end"].date()), "n_train": int(len(fr["train"])),
                                  "n_validation": int(len(fr["validation"])), "n_test": int(len(fr["test"]))})
            per_league = {}
            for lg in sorted(set(rows["league"].astype(str))) if len(rows) else []:
                n_pre = int((pre["league"].astype(str) == lg).sum())
                n_hold = int((hold["league"].astype(str) == lg).sum())
                per_league[lg] = {"walk_forward_region": n_pre, "holdout": n_hold,
                                  "after_holdout_unused": int((after["league"].astype(str) == lg).sum()),
                                  "holdout_meets_min_league_n": (n_hold >= min_league) if min_league else None}
            t["markets"][market] = {
                "eligible_rows": int(len(rows)), "walk_forward_region": int(len(pre)),
                "holdout": int(len(hold)), "after_holdout_unused": int(len(after)),
                "walk_forward_matchdays": int(pd.to_datetime(pre["date"]).nunique()) if len(pre) else 0,
                "holdout_matchdays": int(pd.to_datetime(hold["date"]).nunique()) if len(hold) else 0,
                "folds": fold_rows, "n_folds": len(fold_rows),
                "rows_scored_by_walk_forward": int(sum(r["n_test"] for r in fold_rows)),
                "price": {"walk_forward_region": price_coverage(pre, mcfg) if len(pre) else None,
                          "holdout": price_coverage(hold, mcfg) if len(hold) else None},
                "per_league": per_league}
            for v in t["variants"].values():
                # The variant trains and scores the SAME rows (eligibility never reads a feature);
                # ht_block_complete counts the rows on which every restored HT value is present.
                v["markets"][market] = {"eligible_rows": int(len(rows)), "walk_forward_region": int(len(pre)),
                                        "holdout": int(len(hold)),
                                        "ht_block_complete": {
                                            "walk_forward_region": _rows_complete(pre, v["restored"]),
                                            "holdout": _rows_complete(hold, v["restored"])}}
        plan["tracks"][track] = t
    return plan


# ── data (committed frame, offline) ─────────────────────────────────────────────────────────

def _block_network() -> None:
    import socket

    def _no_net(*a, **k):
        raise OSError("network blocked (phase2 retrain harness reads committed data only)")
    socket.socket.connect = _no_net
    socket.create_connection = _no_net


def load_committed_frame(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """The loader's served frame from the committed data, then src.feature_engineering.build_features.

    Offline: the football-data download is switched off and the network is blocked, so every row is
    a committed row with every loader correction applied. The loader's training-coverage report
    (a committed output file) is not rewritten. Returns (feature frame, info)."""
    if cfg["data"].get("source") != "committed":
        raise ValueError("only data.source='committed' is implemented")
    _block_network()
    import config
    from src import data_loader as DL
    from src.feature_engineering import build_features
    DL._ci_download_all = lambda have=None: []
    DL._report_training_coverage = lambda out: None
    raw = DL.load_all_matches(force=True)
    info = {"loaded_rows": int(len(raw)),
            "af_written_back_not_served": (raw.attrs.get("af_written_back") or {}).get("total"),
            "ou25_pairs_flagged_not_a_price": int((raw["ou25_pair_is_price"] == False).sum())  # noqa: E712
            if "ou25_pair_is_price" in raw.columns else None}
    if cfg["data"].get("exclude_covid_seasons"):
        m = config.covid_season_mask(raw)
        info["covid_rows_excluded"] = int(m.sum())
        raw = raw[~m]
    feat = build_features(raw)
    feat["date"] = pd.to_datetime(feat["date"])
    info["feature_rows"] = int(len(feat))
    from src.standard_repair import frame_hash
    info["frame_hash"] = frame_hash(feat)
    return feat, info
