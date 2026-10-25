"""Prospective champion-vs-challenger experiments, frozen at FIRST SIGHT. P0 2026-10-22.

    from src import shadow_experiments as SE
    rows = SE.score(preds, champion_payloads)   # predict: one row per fixture x active experiment
    SE.append_first_sight(rows)                 # append-only, earliest opinion wins
    SE.settle_file()                            # update_results: adds RESULTS only
    SE.evaluate()                               # paired log loss / Brier on settled rows
    python -m src.shadow_experiments guard      # exit 1 when a registered experiment logs nothing

WHY THIS EXISTS. Two prospective logs were meant to carry a challenger and neither ever did:

* output/v91_shadow.csv wrote `p_pro_hgb` / `p_v91` as hard-coded `pd.NA` (pipeline._log_shadow),
  so 669/669 rows were empty by construction.
* output/model_lineage/shadow_h2h.csv scored a challenger only when the RETRAIN challenger slot
  (models/registry/<type>/challenger.pkl.gz) was filled, and the slot is filled only by a
  SHADOW_MORE retrain verdict. No retrain has issued one since the high-water mark went live, so
  `p_champion_raw`, `challenger_version` and `p_challenger` were empty on 1,388/1,388 rows.
* The HGB Over 2.5 challenger behind H-MODEL-01 never existed as an artifact at all: it was fitted
  in memory inside src/model_validation.evaluate_candidates and thrown away.

So the preregistered prospective test had a clock and no data. This module gives an experiment a
FROZEN challenger artifact (models/registry/experiments/<id>/challenger.pkl.gz, sha-verified),
scores it at predict time beside the champion on the identical feature rows, and freezes the
result the first time a fixture is seen.

WHAT IT MAY NEVER DO. Nothing here reaches a served probability, a tier, a tip, a stake, a
threshold or Telegram. `score` reads `preds` and returns a NEW frame; it never writes to `preds`.
Every predict-side entry point is wrapped by the caller so it cannot raise into predict.

FIRST SIGHT. predict runs every 5-15 minutes; keeping the latest opinion would turn a prospective
log into a retrospective one. The ledger is append-only (`mode="a"`, existing bytes never
rewritten) and deduped on (experiment_id, fixture_id, match_date), keeping the EARLIEST row.
Settlement fills only the RESULT columns, and refuses to write if any first-sight byte changes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = config.MODELS_DIR / "registry" / "experiments"
EXPERIMENTS_FILE = config.MODELS_DIR / "registry" / "experiments.json"
LEDGER_FILE = config.OUTPUT_DIR / "model_lineage" / "shadow_experiments.csv"
EVAL_FILE = config.OUTPUT_DIR / "model_lineage" / "shadow_experiments_eval.json"
V91_SHADOW_FILE = config.OUTPUT_DIR / "v91_shadow.csv"

#: Written once, at first sight, and never revised.
FIRST_SIGHT_COLS = [
    "experiment_id", "hypothesis_id", "fixture_id", "match_date", "kickoff_utc", "league",
    "home_team", "away_team", "model_type", "market", "prediction_ts",
    "champion_version", "champion_sha256", "challenger_version", "challenger_sha256",
    "p_served", "p_champion_raw", "p_challenger", "p_market",
    "dataset_id", "code_commit", "challenger_training_cutoff", "feature_schema",
]
#: Filled by settlement only. Knowable only after the match: may grade, never select.
RESULT_COLS = ["home_goals", "away_goals", "y", "result_source", "settled_at"]
LEDGER_COLS = FIRST_SIGHT_COLS + RESULT_COLS
KEY = ["experiment_id", "fixture_id", "match_date"]

#: A registered experiment that has had eligible fixtures for this long and logged ZERO challenger
#: probabilities is broken, not slow. The guard fails loudly after it.
GUARD_GRACE_HOURS = 48
#: ...and the check is ROLLING over this window, so a path that worked once and then died fails too.
GUARD_WINDOW_HOURS = 72


# ── registry ──────────────────────────────────────────────────────────────────────────────────
def load_registry(path: Path | None = None) -> list[dict]:
    """Active experiments. Never raises: an unreadable registry means 'no experiments'."""
    p = Path(path) if path else EXPERIMENTS_FILE
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"[experiments] registry unreadable ({e}) — no experiments scored")
        return []
    return [e for e in d.get("experiments", []) if str(e.get("status", "")).upper() == "ACTIVE"]


def artifact_path(exp: dict) -> Path:
    return config.BASE_DIR / exp["artifact"]


_payload_cache: dict[str, dict] = {}


def load_challenger(exp: dict) -> dict | None:
    """The frozen challenger payload, verified against the registered sha256. None on mismatch."""
    eid = exp["experiment_id"]
    if eid in _payload_cache:
        return _payload_cache[eid]
    raw = gzip.decompress(artifact_path(exp).read_bytes())
    if hashlib.sha256(raw).hexdigest() != exp.get("artifact_sha256"):
        log.warning(f"[experiments] {eid}: artifact bytes do not match the registry — not scored")
        return None
    payload = pickle.loads(raw)
    _payload_cache[eid] = payload
    return payload


def register(exp: dict, payload: dict, path: Path | None = None) -> dict:
    """Write a challenger artifact + manifest and add it to the registry (replacing the same id).

    Used once, by scripts/register_shadow_experiment.py. A REGISTERED artifact is never rewritten
    in place: a different model is a different challenger_version, and the version embeds the sha.
    """
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sha = hashlib.sha256(raw).hexdigest()
    eid = exp["experiment_id"]
    d = EXPERIMENTS_DIR / eid
    d.mkdir(parents=True, exist_ok=True)
    (d / "challenger.pkl.gz").write_bytes(gzip.compress(raw, 6, mtime=0))
    exp = {**exp, "artifact": (d / "challenger.pkl.gz").relative_to(config.BASE_DIR).as_posix(),
           "artifact_sha256": sha,
           "challenger_version": f"{exp['challenger_family']}_{exp['model_type']}_"
                                 f"{exp['trained_at'].replace('-', '').replace(':', '')}_{sha[:8]}"}
    (d / "manifest.json").write_text(json.dumps(exp, indent=2, default=str) + "\n",
                                     encoding="utf-8", newline="\n")
    p = Path(path) if path else EXPERIMENTS_FILE
    try:
        reg = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        reg = {"experiments": []}
    reg["experiments"] = [e for e in reg.get("experiments", []) if e.get("experiment_id") != eid]
    reg["experiments"].append(exp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    _payload_cache.pop(eid, None)
    return exp


# ── identity ──────────────────────────────────────────────────────────────────────────────────
def fixture_ids(d: pd.DataFrame) -> pd.Series:
    """Exactly the id pipeline._log_shadow writes: the OddsAPI event id, else date|home|away."""
    fid = d["_oa_event_id"] if "_oa_event_id" in d.columns else pd.Series(index=d.index,
                                                                         dtype=object)
    natural = (d["date"].astype(str) + "|" + d["home_team"].astype(str) + "|"
               + d["away_team"].astype(str))
    return fid.where(fid.notna() & (fid.astype(str) != ""), natural)


def _file_sha(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:                                                 # noqa: BLE001
        return ""


# ── predict side ──────────────────────────────────────────────────────────────────────────────
def score(preds: pd.DataFrame, champion_payloads: dict | None = None,
          experiments: list[dict] | None = None, now: datetime | None = None) -> pd.DataFrame:
    """One first-sight row per (fixture, active experiment). READ-ONLY on `preds`.

    The champion is re-scored RAW on exactly the rows the challenger sees, so the pair is
    like-for-like even if predict ever post-processes the served number; `p_served` records what
    production actually served. Rows whose kickoff has already passed are skipped: a prediction
    written after kickoff is not prospective.
    """
    from src import model_registry as reg
    from src.model import load_models, predict_proba

    cols = LEDGER_COLS
    exps = load_registry() if experiments is None else experiments
    if preds is None or preds.empty or not exps:
        return pd.DataFrame(columns=cols)
    now = now or datetime.now(timezone.utc)
    champion_payloads = dict(champion_payloads or {})
    tag = (preds["model_type"].astype(str).map(reg.ou_model_for)
           if "model_type" in preds.columns else pd.Series("standard", index=preds.index))
    # Only rows whose OWN model_type is this track. `ou_model_for` sends unknown leagues to the
    # standard model as a fallback; those are not the standard track and are not eligible.
    raw_tag = preds["model_type"].astype(str) if "model_type" in preds.columns else tag
    fids = fixture_ids(preds)
    ko = pd.to_datetime(preds.get("kickoff_utc", pd.Series(pd.NaT, index=preds.index)),
                        errors="coerce", utc=True)
    future = ko.isna() | (ko > pd.Timestamp(now))
    frames = []
    for exp in exps:
        mt, col = exp["model_type"], exp.get("pred_column", "p_over25")
        want_tag = "new_format" if mt == "newformat" else mt
        if col not in preds.columns:
            continue
        m = ((tag == mt) & (raw_tag == want_tag) & future
             & pd.to_numeric(preds[col], errors="coerce").notna())
        sub = preds[m]
        if sub.empty:
            continue
        try:
            ch = load_challenger(exp)
            if ch is None:
                continue
            champ = champion_payloads.get(mt)
            if champ is None:
                champ = load_models(model_file=reg.MODEL_FILES[mt])
                champion_payloads[mt] = champ
            p_raw = np.asarray(predict_proba(sub, payload=champ), float)
            p_ch = np.asarray(predict_proba(sub, payload=ch), float)
        except Exception as e:                                         # noqa: BLE001
            log.warning(f"[experiments] {exp['experiment_id']} not scored ({e})")
            continue
        stamp = reg.stamp_for(mt)
        p_mkt = sub.get("impl_prob_over", sub.get("implied_prob_over",
                                                  pd.Series(np.nan, index=sub.index)))
        frames.append(pd.DataFrame({
            "experiment_id": exp["experiment_id"], "hypothesis_id": exp.get("hypothesis_id", ""),
            "fixture_id": fids[m].astype(str).values,
            "match_date": sub["date"].astype(str).str[:10].values,
            "kickoff_utc": (sub["kickoff_utc"].astype(str).values
                            if "kickoff_utc" in sub.columns else ""),
            "league": sub["league"].values, "home_team": sub["home_team"].values,
            "away_team": sub["away_team"].values, "model_type": mt,
            "market": exp.get("market", "over25"), "prediction_ts": now.isoformat(),
            "champion_version": stamp.get("model_version", ""),
            "champion_sha256": _file_sha(reg.MODEL_FILES[mt]),
            "challenger_version": exp["challenger_version"],
            "challenger_sha256": exp["artifact_sha256"],
            "p_served": pd.to_numeric(sub[col], errors="coerce").round(6).values,
            "p_champion_raw": np.round(p_raw, 6), "p_challenger": np.round(p_ch, 6),
            "p_market": pd.to_numeric(p_mkt, errors="coerce").round(6).values,
            "dataset_id": (sub["git_sha"].astype(str).values if "git_sha" in sub.columns
                           else ""),
            "code_commit": os.getenv("GITHUB_SHA", "")[:12] or "local",
            "challenger_training_cutoff": exp.get("training_cutoff", ""),
            "feature_schema": exp.get("feature_schema", ""),
        }))
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    for c in RESULT_COLS:
        out[c] = np.nan
    return out[cols]


def _read_ledger(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def first_sight_store(path: Path | None = None):
    """The shared first-sight store (src/research/first_sight.py) for the experiment ledger. Only the
    FIRST_SIGHT_COLS are frozen; settlement fills RESULT_COLS later."""
    from src.research.first_sight import FirstSightStore
    return FirstSightStore(Path(path) if path else LEDGER_FILE, key=KEY, columns=LEDGER_COLS,
                           observed_col="prediction_ts", kickoff_col="kickoff_utc",
                           frozen_cols=FIRST_SIGHT_COLS)


def append_first_sight(rows: pd.DataFrame, path: Path | None = None) -> int:
    """Append rows whose key has never been logged. Existing bytes are never rewritten.
    Migrated 2026-10-25 onto the shared FirstSightStore; byte-identical to the previous inline
    implementation on committed inputs (score() already keeps only rows with kickoff > now and
    stamps prediction_ts = now, so the store's kickoff refusal is a no-op)."""
    if rows is None or rows.empty:
        return 0
    new = first_sight_store(path).append_rows(rows)
    if new.empty:
        return 0
    try:                                         # SHADOW contract check (src/contracts); observe only
        from src.contracts.shadow import observe
        observe("model_lineage/shadow_experiments.csv (appended rows)", new, "SHADOW_EXPERIMENT_ROW")
    except Exception:                                                 # noqa: BLE001
        pass
    return int(len(new))


# ── settlement (results only) ─────────────────────────────────────────────────────────────────
def first_sight_digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for c in FIRST_SIGHT_COLS:
        if c in df.columns:
            h.update(c.encode())
            h.update("\x1f".join(df[c].astype(str).tolist()).encode("utf-8"))
    return h.hexdigest()


def _blank(v) -> bool:
    return str(v).strip() in ("", "nan", "NaN", "None", "<NA>")


def settle(ledger: pd.DataFrame, outcomes: pd.DataFrame,
           now: datetime | None = None) -> tuple[pd.DataFrame, int]:
    """Fill RESULT_COLS for unsettled rows from settled outcomes. PURE.

    `outcomes` carries fixture_id, match_date, home_goals, away_goals, result_source — in
    production the settled rows of v91_shadow.csv, which update_results settles from the same
    results hierarchy as every ledger. A row is settled only when its prediction was written
    BEFORE kickoff. First-sight columns are hashed before and after; a change raises.
    """
    out = ledger.copy()
    if out.empty or outcomes is None or outcomes.empty:
        return out, 0
    now = now or datetime.now(timezone.utc)
    for c in RESULT_COLS:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].astype(object)
    before = first_sight_digest(out)
    o = outcomes.copy()
    o = o[~o["home_goals"].map(_blank) & ~o["away_goals"].map(_blank)]
    idx = {(str(f), str(d)[:10]): r for f, d, r in
           zip(o["fixture_id"], o["match_date"], o.to_dict("records"))}
    n = 0
    for i, r in out.iterrows():
        if not _blank(r.get("y", "")):
            continue
        hit = idx.get((str(r["fixture_id"]), str(r["match_date"])[:10]))
        if hit is None:
            continue
        ko = pd.to_datetime(r.get("kickoff_utc", ""), errors="coerce", utc=True)
        ts = pd.to_datetime(r.get("prediction_ts", ""), errors="coerce", utc=True)
        if pd.notna(ko) and (pd.isna(ts) or ts >= ko):
            continue                                   # not prospective -> never graded
        try:
            hg, ag = int(float(hit["home_goals"])), int(float(hit["away_goals"]))
        except (TypeError, ValueError):
            continue
        line = {"over25": 2.5, "over15": 1.5, "over35": 3.5}.get(str(r.get("market")), 2.5)
        out.at[i, "home_goals"] = str(hg)
        out.at[i, "away_goals"] = str(ag)
        out.at[i, "y"] = "1" if hg + ag > line else "0"
        out.at[i, "result_source"] = str(hit.get("result_source", "") or "v91_shadow")
        out.at[i, "settled_at"] = now.isoformat()
        n += 1
    if first_sight_digest(out) != before:
        raise RuntimeError("settlement would change a first-sight value — refused")
    return out, n


def settle_file(path: Path | None = None, outcomes_path: Path | None = None) -> int:
    p = Path(path) if path else LEDGER_FILE
    op = Path(outcomes_path) if outcomes_path else V91_SHADOW_FILE
    if not (p.exists() and op.exists()):
        return 0
    led = _read_ledger(p)
    oc = pd.read_csv(op, dtype=str, keep_default_na=False)
    out, n = settle(led, oc)
    if n:
        out[LEDGER_COLS].to_csv(p, index=False, lineterminator="\n")
    log.info(f"[experiments] settled {n} row(s) -> {p.name}")
    return n


# ── evaluation ────────────────────────────────────────────────────────────────────────────────
def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate(ledger: pd.DataFrame | None = None, n_boot: int = 2000, seed: int = 7) -> dict:
    """Paired champion-raw vs challenger on SETTLED first-sight rows, per experiment x model_type.

    Paired: both probabilities are for the same fixture, written at the same moment, so the
    per-row delta is the unit. CI is a block bootstrap over match dates (fixtures on one day share
    conditions). Tracks are never pooled (invariant 1). Descriptive only — the decision rule is
    the preregistered one (n_min, material floor) and is not re-decided here.
    """
    if ledger is None and not LEDGER_FILE.exists():
        return {}
    d = _read_ledger(LEDGER_FILE) if ledger is None else ledger.astype(str)
    out = {}
    if d.empty:
        return out
    for (eid, mt), g in d.groupby(["experiment_id", "model_type"]):
        y = pd.to_numeric(g["y"], errors="coerce")
        pc = pd.to_numeric(g["p_champion_raw"], errors="coerce")
        px = pd.to_numeric(g["p_challenger"], errors="coerce")
        ps = pd.to_numeric(g["p_served"], errors="coerce")
        ok = y.notna() & pc.notna() & px.notna()
        rec = {"rows": int(len(g)), "settled_paired": int(ok.sum()),
               "challenger_logged": int(px.notna().sum())}
        if ok.sum() >= 1:
            yy, a, b, s = (y[ok].to_numpy(float), pc[ok].to_numpy(float),
                           px[ok].to_numpy(float), ps[ok].to_numpy(float))
            dl = _ll(yy, b) - _ll(yy, a)
            db = (b - yy) ** 2 - (a - yy) ** 2
            rec.update(ll_champion_raw=float(_ll(yy, a).mean()),
                       ll_challenger=float(_ll(yy, b).mean()),
                       ll_served=float(np.nanmean(_ll(yy, s))),
                       brier_champion_raw=float(((a - yy) ** 2).mean()),
                       brier_challenger=float(((b - yy) ** 2).mean()),
                       delta_log_loss=float(dl.mean()), delta_brier=float(db.mean()))
            dates = g.loc[ok, "match_date"].to_numpy()
            ud = np.unique(dates)
            if len(ud) >= 5:
                rng = np.random.default_rng(seed)
                by = {k: dl[dates == k] for k in ud}
                bs = [np.concatenate([by[k] for k in rng.choice(ud, len(ud))]).mean()
                      for _ in range(n_boot)]
                rec["delta_log_loss_ci95"] = [float(np.percentile(bs, 2.5)),
                                              float(np.percentile(bs, 97.5))]
        out[f"{eid}|{mt}"] = rec
    return out


# ── guard ─────────────────────────────────────────────────────────────────────────────────────
def guard(now: datetime | None = None, experiments: list[dict] | None = None,
          ledger: pd.DataFrame | None = None, h2h: pd.DataFrame | None = None,
          grace_hours: float = GUARD_GRACE_HOURS,
          window_hours: float = GUARD_WINDOW_HOURS) -> list[str]:
    """Failures, one string each. Empty list = healthy.

    For every ACTIVE experiment registered longer ago than `grace_hours`, over the last
    `window_hours` (never earlier than registration):

    * FAIL when the champion log (shadow_h2h.csv) shows eligible fixtures of the experiment's
      model type but the experiment ledger holds ZERO non-null challenger probabilities — exactly
      the silent state H-MODEL-01 sat in for 18 days. Rolling, so a path that worked once and
      then died is caught too.
    * FAIL when nothing at all of that model type has been logged since registration: the
      predict-side recorder is not running.
    A quiet window with no new fixtures of either kind is not a failure.
    """
    now = now or datetime.now(timezone.utc)
    exps = load_registry() if experiments is None else experiments
    if ledger is None:
        ledger = _read_ledger(LEDGER_FILE) if LEDGER_FILE.exists() else pd.DataFrame()
    if h2h is None:
        h2h_p = config.OUTPUT_DIR / "model_lineage" / "shadow_h2h.csv"
        h2h = (pd.read_csv(h2h_p, dtype=str, keep_default_na=False) if h2h_p.exists()
               else pd.DataFrame())
    now_ts = pd.Timestamp(now)
    fails = []
    for e in exps:
        reg_at = pd.to_datetime(e.get("registered_at"), errors="coerce", utc=True)
        if pd.isna(reg_at) or (now_ts - reg_at) < pd.Timedelta(hours=grace_hours):
            continue
        mt, eid = e["model_type"], e["experiment_id"]
        start = max(reg_at, now_ts - pd.Timedelta(hours=window_hours))

        def _count_h2h(since):
            if h2h is None or h2h.empty:
                return 0
            t = pd.to_datetime(h2h["logged_at"], errors="coerce", utc=True)
            return int(((h2h["model_type"].astype(str) == mt) & (t >= since)).sum())

        def _count_logged(since):
            if ledger.empty:
                return 0
            g = ledger[ledger["experiment_id"].astype(str) == eid]
            t = pd.to_datetime(g["prediction_ts"], errors="coerce", utc=True)
            return int((pd.to_numeric(g["p_challenger"], errors="coerce").notna()
                        & (t >= since)).sum())

        elig, logged = _count_h2h(start), _count_logged(start)
        if logged == 0 and elig > 0:
            fails.append(f"{eid}: {elig} eligible {mt} fixture(s) logged by the champion since "
                         f"{start:%Y-%m-%d %H:%M}Z and ZERO challenger probabilities")
        elif _count_h2h(reg_at) == 0 and _count_logged(reg_at) == 0:
            fails.append(f"{eid}: registered {reg_at:%Y-%m-%d %H:%M}Z and neither the champion "
                         f"nor the challenger has logged a {mt} fixture since - predict-side "
                         f"logging is not running")
    return fails


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["settle", "evaluate", "guard"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "settle":
        settle_file()
        return 0
    if a.cmd == "evaluate":
        res = evaluate()
        EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        EVAL_FILE.write_text(json.dumps({"computed_at": datetime.now(timezone.utc).isoformat(),
                                         "results": res}, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
        print(json.dumps(res, indent=2))
        return 0
    fails = guard()
    for f in fails:
        print(f"::error title=shadow experiment not logging::{f}")
    if fails:
        return 1
    print(f"shadow experiments OK ({len(load_registry())} active)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
