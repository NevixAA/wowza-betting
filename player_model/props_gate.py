"""THE promotion gate for player-prop models. The only code allowed to change which props model serves.

    from player_model import props_gate as PPG
    PPG.review_parked("goals", outcomes=df)               # retrain start: period 2 for the parked one
    gate = PPG.submit_candidate("goals", results, hold)   # retrain: period 1, record, maybe park
    rec  = PPG.FirstSightRecorder(); rec.add(...); rec.flush()   # predict: prospective evidence
    PPG.rollback_props("goals", "<version>", reason="...")       # operational, logged

Owner decision 2026-10-25 ("yes"): player-prop models are promoted only through the canonical
promotion gate, like the team models. Before this, player_model/pipeline.mode_train called
save_model() whenever the candidate's mean log loss had not risen by more than
PROPS_MAX_LOGLOSS_RISE (0.030) — and the two numbers compared were each model's OWN holdout, i.e.
different rows. Every serving props model was replaced at every retrain: 39 serving-file
replacements in five retrains between 2026-10-12 and 2026-10-25, most of them by candidates whose
own-holdout log loss was WORSE than the incumbent's.

────────────────────────────────────────────────────────────────────────────────────────────
SAME DEFINITION OF "PROMOTED" AS src/promotion_gate.py (PG-1.0), PER MARKET
────────────────────────────────────────────────────────────────────────────────────────────
A props model is PROMOTED when, and only when, this module installs it as the serving model of its
market. Two independent periods, both against the model actually serving, both per MARKET (goals,
goals2, assists, sot, sot2, sot3, sot4, cards — never pooled; markets play the role tracks play in
PG-1.0):

  PERIOD 1 — the untouched chronological holdout (retrain time, `submit_candidate`)
     the last 20% of rows by date, exactly as train() split them (model.split_bounds); served
     (ensemble) probabilities of champion and candidate on the SAME rows:
       n >= 1000 and events >= 100
       log-loss improvement >= the market's material floor
       match-date block-bootstrap 95% CI upper < 0
       Brier non-inferior, ECE not materially worse, calibration slope not grossly worse
       prediction-distribution canary, no major league regression
     Passing makes the candidate QUALIFIED -> parked as challenger (SHADOW_MORE). Never PROMOTE.

  PERIOD 2 — the independent prospective period (later retrains, `review_parked`)
     first-sight rows of output/model_lineage/props_shadow_h2h.csv written by props predict
     (FirstSightRecorder): champion and parked challenger scored on the same upcoming player-fixture
     at the same moment, logged before kickoff and after the freeze, graded from
     player_history.parquet by (fixture_id, player_id). Per market: n >= 1000 and events >= 100,
     log-loss delta < 0, Brier non-inferior, canary, league regression. PASS -> PROMOTE.
     FAIL / champion significantly ahead at n >= 300 / 180 days -> retired. Else SHADOW_MORE.
     With no prospective data the verdict is never PROMOTE.

Verdicts PROMOTE | KEEP_CHAMPION | SHADOW_MORE, plus BOOTSTRAP (no serving file and no record of any
champion), REGISTER (an already-serving legacy file is given an identity — its bytes are not
touched) and ROLLBACK (explicit, logged). Every decision is appended to
output/model_lineage/props_promotion_decisions.csv with the gate version.

────────────────────────────────────────────────────────────────────────────────────────────
THRESHOLDS — config/promotion_gate.json section "props", PG-PROPS-1.0, declared 2026-10-25
────────────────────────────────────────────────────────────────────────────────────────────
Wherever a PG-1.0 value is dimensionless it is reused unchanged (bootstrap design, slope rule, sd
ratio band, share moved, extreme-share increase, min_league_n, early fail, 180-day expiry, n >= 1000).

PG-1.0's probability- and score-scale numbers were set for over/under 2.5 (base rate ~0.5). Props
base rates run from 0.26% (sot4) to 22% (sot); an absolute 0.001 log-loss floor is 0.35% of goals'
log loss but 5.5% of sot4's, so reusing it would make rare markets effectively unpromotable and
common ones too easy. Each such number is therefore RESCALED by a factor computed from the market's
LABEL base rate only (player_history.parquet, 2026-10-25) — never from any model's output:
  * log-loss quantities x H(base)/ln2      (the same share of the market's irreducible uncertainty)
  * Brier quantities    x base(1-base)/0.25 (Brier's own scale is the outcome variance)
  * probability units   x base/0.5          (ECE, mean shift, quantile shift, moved_pp)

New in PG-PROPS-1.0:
  * min_events_per_market 100 in BOTH periods. With a 0.26% base rate 1000 rows hold ~3 goals'
    worth of positives; log loss, ECE and slope on rare events are driven by the positives. 100
    events (and, implicitly, >> 100 non-events) is the standard minimum for validating a binary
    prediction model (Vergouwe et al. 2005; Collins et al. 2016).
  * min_league_events 30: a league's delta is only allowed to veto when it holds >= 30 events.
  * extreme_lo 0.001, the props Platt clip floor (model._PlattCalibratedModel), instead of 0.05 —
    most props probabilities legitimately sit below 0.05; pinned-at-the-floor is the degenerate case.

PAPER-ONLY IS UNCHANGED. This module decides which model produces props probabilities. It does not
touch tiers, staking or Telegram; props stay blocked from real staking (root invariant 2).

Known cost, recorded: props predict scores ~a few thousand player-fixtures a week and only players
who then appear in player_history are graded, so sot4 / goals2 need ~38k / ~14k graded rows for 100
events. Their promotions will take many months. That is the intended rate.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from player_model import config as pconfig  # noqa: E402
from src import model_registry as reg  # noqa: E402  (atomic writes, append-only CSV, hashing)
from src import promotion_gate as PG  # noqa: E402   (the canonical scoring primitives)
from src.research.first_sight import FirstSightStore  # noqa: E402

log = logging.getLogger(__name__)

CONFIG_FILE = _ROOT / "config" / "promotion_gate.json"

PROMOTE, KEEP, SHADOW = PG.PROMOTE, PG.KEEP, PG.SHADOW
BOOTSTRAP, ROLLBACK, REGISTER = PG.BOOTSTRAP, PG.ROLLBACK, "REGISTER"

#: Exact parked challenger versions the owner approved by hand (comma-separated). Can only name a
#: challenger that already exists, so it cannot pre-approve a future retrain. Canary still applies.
OWNER_OVERRIDE_ENV = "PROPS_GATE_OWNER_OVERRIDE"

DECISION_COLS = [
    "decided_at", "gate_version", "canonical_gate_version", "stage", "market",
    "candidate_version", "champion_version", "verdict", "promoted", "qualified", "parked",
    "retrain_batch_id", "holdout_n", "holdout_events", "holdout_ll_delta", "holdout_ci_lo",
    "holdout_ci_hi", "prospective_status", "prospective_n", "prospective_events",
    "prospective_ll_delta", "checks", "metrics", "reason",
]
CHAMPION_HISTORY_COLS = [
    "market", "champion_version", "start_at", "prior_champion", "source", "reason",
    "artifact_sha256", "retrain_batch_id", "gate_version",
]
H2H_COLS = [
    "logged_at", "market", "challenger_version", "champion_version", "fixture_id", "player_id",
    "league", "match_date", "kickoff_utc", "p_champion_raw", "p_challenger",
]
H2H_KEY = ["market", "challenger_version", "champion_version", "fixture_id", "player_id"]


# ── configuration ─────────────────────────────────────────────────────────────────────────────
def load_config(path: Path | None = None) -> dict:
    blob = json.loads(Path(path or CONFIG_FILE).read_text(encoding="utf-8"))
    cfg = blob.get("props")
    if not cfg:
        raise ValueError("promotion gate config has no 'props' section")
    for k in ("gate_version", "declared_on", "holdout", "canary", "prospective", "markets"):
        if k not in cfg:
            raise ValueError(f"props gate config missing '{k}'")
    miss = [m for m in pconfig.MARKETS if m not in cfg["markets"]]
    if miss:
        raise ValueError(f"props gate config declares no thresholds for market(s) {miss}")
    return cfg


CFG = load_config()
GATE_VERSION = CFG["gate_version"]
CANONICAL_VERSION = CFG.get("canonical_gate_version", PG.GATE_VERSION)

# Paths are module attributes so tests can point them at a temporary tree.
REGISTRY_DIR = pconfig.MODELS_DIR / "registry" / "props"
PROPS_POINTER_FILE = REGISTRY_DIR / "props_champions.json"
LINEAGE_DIR = pconfig.OUTPUT_DIR / "model_lineage"
DECISIONS_FILE = LINEAGE_DIR / "props_promotion_decisions.csv"
CHAMPION_HISTORY = LINEAGE_DIR / "props_champion_history.csv"
H2H_FILE = LINEAGE_DIR / "props_shadow_h2h.csv"
HISTORY_FILE = pconfig.BASE_DIR / "player_history.parquet"


def market_cfg(market: str) -> dict:
    return CFG["markets"][market]


def canary_cfg(market: str) -> dict:
    """PG-1.0's canary dict shape, filled with this market's thresholds."""
    c, m = CFG["canary"], market_cfg(market)
    return {"max_mean_shift": m["max_mean_shift"], "sd_ratio_min": c["sd_ratio_min"],
            "sd_ratio_max": c["sd_ratio_max"], "quantiles": c["quantiles"],
            "max_quantile_shift": m["max_quantile_shift"], "extreme_lo": c["extreme_lo"],
            "extreme_hi": c["extreme_hi"],
            "max_extreme_share_increase": c["max_extreme_share_increase"],
            "moved_pp": m["moved_pp"], "max_frac_moved": c["max_frac_moved"]}


def serving_file(market: str) -> Path:
    return pconfig.MODEL_FILES[market]


def _mdir(market: str) -> Path:
    return REGISTRY_DIR / market


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def batch_id(now: datetime | None = None) -> str:
    return "props_" + reg.batch_id(now or utcnow())


def make_version(market: str, at: datetime, sha: str) -> str:
    return f"props_{market}_{at.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{sha[:8]}"


# ── decision log ──────────────────────────────────────────────────────────────────────────────
def record_decision(row: dict) -> None:
    """APPEND-ONLY, one row per gate decision, with the gate version."""
    def _j(v):
        return v if isinstance(v, str) else json.dumps(v, default=str, sort_keys=True)
    r = {"decided_at": reg.iso(utcnow()), "gate_version": GATE_VERSION,
         "canonical_gate_version": CANONICAL_VERSION, **row}
    for k in ("checks", "metrics"):
        if k in r and r[k] is not None:
            r[k] = _j(r[k])
    reg._append_csv(DECISIONS_FILE, DECISION_COLS, r)


# ── registry: pointer, manifests, champion bytes, challenger slot ─────────────────────────────
def read_pointer() -> dict:
    try:
        return json.loads(PROPS_POINTER_FILE.read_text(encoding="utf-8")) if PROPS_POINTER_FILE.exists() else {}
    except Exception:                                                  # noqa: BLE001
        return {}


def _manifest_path(market: str, version: str) -> Path:
    return _mdir(market) / "versions" / f"{version}.json"


def _write_manifest(man: dict) -> None:
    p = _manifest_path(man["market"], man["model_version"])
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("artifact_sha256") != man.get("artifact_sha256"):
            raise ValueError(f"manifest {p.name} exists with a different artifact")
        return
    reg._atomic_write_json(p, man)


def read_manifest(market: str, version: str) -> dict | None:
    try:
        return json.loads(_manifest_path(market, version).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return None


def _champion_bytes_path(market: str, version: str) -> Path:
    return _mdir(market) / "champions" / f"{version}.pkl.gz"


def _store_champion_bytes(market: str, version: str, raw: bytes) -> None:
    p = _champion_bytes_path(market, version)
    if not p.exists():
        reg._atomic_write_bytes(p, gzip.compress(raw, mtime=0))


def challenger_path(market: str) -> Path:
    return _mdir(market) / "challenger.pkl.gz"


def challenger_meta_path(market: str) -> Path:
    return _mdir(market) / "challenger.json"


def _slot_meta(market: str) -> dict | None:
    m = challenger_meta_path(market)
    try:
        return json.loads(m.read_text(encoding="utf-8")) if m.exists() else None
    except Exception:                                                  # noqa: BLE001
        return None


def load_parked(market: str) -> tuple[dict, dict] | None:
    """(payload, meta) of the parked challenger, or None. Never raises."""
    try:
        meta = _slot_meta(market)
        p = challenger_path(market)
        if not meta or not p.exists():
            return None
        raw = gzip.decompress(p.read_bytes())
        if reg.sha256_bytes(raw) != meta.get("artifact_sha256"):
            log.warning(f"[props_gate] {market}: parked bytes do not match their manifest - ignored")
            return None
        return pickle.loads(raw), meta
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[props_gate] {market}: could not load parked challenger ({e})")
        return None


def _park(market: str, meta: dict, raw: bytes) -> None:
    reg._atomic_write_bytes(challenger_path(market), gzip.compress(raw, mtime=0))
    reg._atomic_write_json(challenger_meta_path(market), meta)


def _vacate_slot(market: str) -> None:
    for p in (challenger_path(market), challenger_meta_path(market)):
        if p.exists():
            p.unlink()


def _install_props_champion(market: str, manifest: dict, raw: bytes, *, reason: str,
                            source: str, batch: str = "", at: datetime | None = None) -> None:
    """THE ONLY WRITER OF A SERVING PROPS MODEL FILE. Order keeps every crash point recoverable:
    immutable champion copy -> manifest -> serving file (write-then-rename) -> pointer -> history."""
    at = at or utcnow()
    version = manifest["model_version"]
    _store_champion_bytes(market, version, raw)
    _write_manifest({**manifest, "model_role": "CHAMPION"})
    reg._atomic_write_bytes(serving_file(market), raw)
    ptr = read_pointer()
    prior = (ptr.get(market) or {}).get("model_version", "")
    ptr[market] = {"model_version": version, "champion_since": reg.iso(at),
                   "artifact_sha256": manifest["artifact_sha256"],
                   "training_cutoff": manifest.get("training_cutoff") or "",
                   "promotion_source": source, "promotion_reason": reason[:400],
                   "gate_version": GATE_VERSION}
    reg._atomic_write_json(PROPS_POINTER_FILE, ptr)
    reg._append_csv(CHAMPION_HISTORY, CHAMPION_HISTORY_COLS, {
        "market": market, "champion_version": version, "start_at": reg.iso(at),
        "prior_champion": prior, "source": source, "reason": reason[:400],
        "artifact_sha256": manifest["artifact_sha256"], "retrain_batch_id": batch,
        "gate_version": GATE_VERSION})


def champion_identity(market: str, *, register: bool = True, now: datetime | None = None) -> str:
    """Version of the model ACTUALLY serving this market ('' when there is no serving file).

    A serving file the registry does not know (every props model before PG-PROPS-1.0, or a file
    changed outside the gate) is REGISTERED: given an identity from its own bytes, its bytes copied
    for rollback, logged. Registration never changes the serving file. register=False (predict)
    returns UNREGISTERED_<sha12> instead of writing anything."""
    f = serving_file(market)
    if not f.exists():
        return ""
    raw = f.read_bytes()
    sha = reg.sha256_bytes(raw)
    ent = read_pointer().get(market) or {}
    if ent.get("artifact_sha256") == sha and ent.get("model_version"):
        return ent["model_version"]
    if not register:
        return f"UNREGISTERED_{sha[:12]}"
    now = now or utcnow()
    version = f"props_{market}_legacy_{sha[:8]}"
    drift = bool(ent)
    try:
        payload = pickle.loads(raw)
        n_train = {k: (v or {}).get("n_train") for k, v in (payload.get("metrics") or {}).items()}
    except Exception:                                                  # noqa: BLE001
        n_train = {}
    _store_champion_bytes(market, version, raw)
    _write_manifest({"market": market, "model_version": version, "artifact_sha256": sha,
                     "registered_at": reg.iso(now), "training_cutoff": None,
                     "training_rows": n_train, "model_role": "CHAMPION",
                     "note": "registered as found serving; training cutoff unknown (legacy)"})
    ptr = read_pointer()
    ptr[market] = {"model_version": version, "champion_since": reg.iso(now),
                   "artifact_sha256": sha, "training_cutoff": "",
                   "promotion_source": "REGISTERED_DRIFT" if drift else "REGISTERED_LEGACY",
                   "promotion_reason": "serving file registered as found; bytes unchanged",
                   "gate_version": GATE_VERSION}
    reg._atomic_write_json(PROPS_POINTER_FILE, ptr)
    record_decision({"stage": REGISTER, "market": market, "candidate_version": version,
                     "champion_version": ent.get("model_version", ""), "verdict": REGISTER,
                     "promoted": False,
                     "reason": ("serving file differs from the registered champion "
                                f"{ent.get('model_version')} - registered as found, NOT changed"
                                if drift else "legacy serving file registered as found; bytes unchanged")})
    return version


def rollback_props(market: str, to_version: str, reason: str = "operational rollback") -> None:
    """Restore an earlier champion's exact bytes. Operational, owner-initiated, logged."""
    p = _champion_bytes_path(market, to_version)
    man = read_manifest(market, to_version)
    if not p.exists() or man is None:
        raise FileNotFoundError(f"no stored champion bytes/manifest for {to_version}")
    raw = gzip.decompress(p.read_bytes())
    if reg.sha256_bytes(raw) != man["artifact_sha256"]:
        raise ValueError(f"{to_version}: stored bytes do not match the manifest hash")
    prior = (read_pointer().get(market) or {}).get("model_version", "")
    _install_props_champion(market, man, raw, reason=f"ROLLBACK: {reason}", source="ROLLBACK")
    record_decision({"stage": ROLLBACK, "market": market, "candidate_version": to_version,
                     "champion_version": prior, "verdict": ROLLBACK, "promoted": True,
                     "reason": reason})


# ── PERIOD 1: the untouched holdout, per market ───────────────────────────────────────────────
def evaluate_holdout(market: str, y, p_champion, p_candidate, *, dates, leagues=None,
                     champion_cutoff=None) -> dict:
    """Period 1 for ONE market. NEVER RAISES — any failure returns qualified=False with the reason."""
    h, m = CFG["holdout"], market_cfg(market)
    out = {"market": market, "qualified": False, "possible_gain": False, "evaluable": False,
           "n": 0, "events": 0, "checks": {}, "reason": "", "untouched": "UNVERIFIED"}
    try:
        y = np.asarray(y, float)
        pc, pn = np.asarray(p_champion, float), np.asarray(p_candidate, float)
        if not (len(y) == len(pc) == len(pn)) or dates is None or len(dates) != len(y):
            out["reason"] = "EVALUATION_FAILED: labels, predictions and dates differ in length"
            return out
        dts = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        ok = np.isfinite(y) & np.isfinite(pc) & np.isfinite(pn) & dts.notna().to_numpy()
        y, pc, pn = y[ok], pc[ok], pn[ok]
        dts = dts[ok].reset_index(drop=True)
        lg = None if leagues is None else pd.Series(list(leagues)).astype(str)[ok].to_numpy()
        n, ev = int(len(y)), int(y.sum()) if len(y) else 0
        out.update(n=n, events=ev, dropped_rows=int((~ok).sum()))
        if champion_cutoff:
            cut = pd.to_datetime(champion_cutoff, errors="coerce")
            if pd.notna(cut) and n:
                if dts.min() <= cut:
                    out["untouched"] = "VIOLATED"
                    out["reason"] = (f"EVALUATION_FAILED: holdout starts {dts.min().date()} but the "
                                     f"champion trained through {cut.date()} - not untouched")
                    return out
                out["untouched"] = "VERIFIED"
        if n < int(h["min_n_per_market"]) or ev < int(h["min_events_per_market"]):
            out["reason"] = (f"n={n}, events={ev} below min n {h['min_n_per_market']} / events "
                             f"{h['min_events_per_market']} - nothing can be decided")
            return out
        out["evaluable"] = True
        cm, nm = PG.metrics(y, pc), PG.metrics(y, pn)
        per_row = PG._ll(y, pn) - PG._ll(y, pc)
        d = float(per_row.mean())
        days = dts.dt.strftime("%Y-%m-%d").to_numpy()
        lo, hi, nd = PG.day_block_ci(per_row, days, n_boot=int(h["n_boot"]),
                                     seed=int(h["bootstrap_seed"]), level=float(h["ci_level"]))
        if nd < int(h["min_bootstrap_blocks"]):
            lo = hi = float("nan")
        canary_ok, canary = PG.prediction_canary(pc, pn, canary_cfg(market))
        regs = league_regressions(per_row, y, lg, market)
        s_c, s_n, se_n = cm["cal_slope"], nm["cal_slope"], nm["cal_slope_se"]
        if s_c is None or s_n is None or se_n is None:
            slope_ok, slope_det = True, None
        else:
            slope_det = abs(s_n - 1) - abs(s_c - 1)
            slope_ok = not (slope_det > float(h["max_slope_deterioration"])
                            and slope_det > float(h["slope_se_multiple"]) * se_n)
        floor = float(m["material_ll_improvement"])
        checks = {
            "material_improvement": d <= -floor,
            "ci_excludes_zero": bool(hi == hi and hi < 0),
            "brier_non_inferior": (nm["brier"] - cm["brier"]) <= float(m["max_brier_worsening"]),
            "ece_not_worse": (nm["ece"] - cm["ece"]) <= float(m["max_ece_worsening"]),
            "slope_not_worse": bool(slope_ok),
            "prediction_canary": bool(canary_ok),
            "no_league_regression": not regs,
        }
        out.update(champion=cm, candidate=nm, ll_delta=round(d, 7),
                   brier_delta=round(nm["brier"] - cm["brier"], 7),
                   ece_delta=round(nm["ece"] - cm["ece"], 7),
                   slope_deterioration=None if slope_det is None else round(slope_det, 4),
                   ci=[None if lo != lo else round(lo, 7), None if hi != hi else round(hi, 7)],
                   n_dates=nd, canary=canary, league_regressions=regs,
                   checks={k: bool(v) for k, v in checks.items()},
                   qualified=all(checks.values()),
                   possible_gain=bool(lo == lo and lo <= -floor and d < 0 and canary_ok))
        out["reason"] = ("all holdout criteria met" if out["qualified"] else
                         "failed: " + ", ".join(k for k, v in checks.items() if not v))
        return out
    except Exception as e:                                             # noqa: BLE001
        out.update(qualified=False, reason=f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}")
        return out


def league_regressions(per_row_delta, y, leagues, market: str) -> list[dict]:
    """Leagues with n >= min_league_n AND events >= min_league_events whose delta exceeds the
    market's bound. Below either minimum a league's number is noise and may not veto."""
    if leagues is None:
        return []
    h, m = CFG["holdout"], market_cfg(market)
    df = pd.DataFrame({"d": np.asarray(per_row_delta, float), "y": np.asarray(y, float),
                       "lg": np.asarray(leagues).astype(str)})
    out = []
    for name, g in df.groupby("lg"):
        if len(g) >= int(h["min_league_n"]) and g["y"].sum() >= int(h["min_league_events"]) \
                and float(g["d"].mean()) > float(m["max_league_ll_regression"]):
            out.append({"league": name, "n": int(len(g)), "events": int(g["y"].sum()),
                        "ll_delta": round(float(g["d"].mean()), 6)})
    return out


# ── PERIOD 2: the independent prospective period ──────────────────────────────────────────────
def _key(v) -> str:
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return ""


def load_outcomes(market: str, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """(fixture_id, player_id, y) for every labelled player-match. From the frame mode_train already
    holds, or player_history.parquet with column projection."""
    t = pconfig.MARKET_TARGETS[market]
    cols = ["fixture_id", "player_id", t]
    if history is None:
        if not Path(HISTORY_FILE).exists():
            return pd.DataFrame(columns=["fixture_id", "player_id", "y"])
        history = pd.read_parquet(HISTORY_FILE, columns=cols)
    if not set(cols) <= set(history.columns):
        return pd.DataFrame(columns=["fixture_id", "player_id", "y"])
    o = history[cols].dropna()
    return pd.DataFrame({"fixture_id": o["fixture_id"].map(_key), "player_id": o["player_id"].map(_key),
                         "y": o[t].astype(float).to_numpy()}).drop_duplicates(
        ["fixture_id", "player_id"], keep="last")


def prospective_rows(market: str, challenger_version: str, champion_version: str, frozen_at,
                     h2h: pd.DataFrame | None = None,
                     outcomes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Graded first-sight rows for (challenger, champion) in one market. Only rows logged BEFORE
    kickoff and AFTER the freeze count; the first row per key wins."""
    cols = ["y", "p_champion", "p_challenger", "match_date", "league"]
    if h2h is None:
        h2h = (pd.read_csv(H2H_FILE, dtype=str, keep_default_na=False)
               if H2H_FILE.exists() else pd.DataFrame())
    if outcomes is None:
        outcomes = load_outcomes(market)
    if h2h.empty or outcomes is None or outcomes.empty or not set(H2H_COLS) <= set(h2h.columns):
        return pd.DataFrame(columns=cols)
    h = h2h[(h2h["market"].astype(str) == market)
            & (h2h["challenger_version"].astype(str) == challenger_version)
            & (h2h["champion_version"].astype(str) == champion_version)].copy()
    if h.empty:
        return pd.DataFrame(columns=cols)
    h = h.drop_duplicates(H2H_KEY, keep="first")
    logged = pd.to_datetime(h["logged_at"].astype(str), errors="coerce", utc=True, format="mixed")
    ko = pd.to_datetime(h["kickoff_utc"].astype(str), errors="coerce", utc=True, format="mixed")
    md = pd.to_datetime(h["match_date"].astype(str).str[:10], errors="coerce", utc=True)
    before_ko = pd.Series(np.where(ko.notna(), logged < ko, logged < md), index=h.index)
    frozen = pd.to_datetime(frozen_at, errors="coerce", utc=True)
    after_freeze = (logged >= frozen) if pd.notna(frozen) else pd.Series(False, index=h.index)
    h = h[logged.notna() & before_ko & after_freeze]
    if h.empty:
        return pd.DataFrame(columns=cols)
    h = h.assign(fixture_id=h["fixture_id"].map(_key), player_id=h["player_id"].map(_key))
    g = h.merge(outcomes, on=["fixture_id", "player_id"], how="inner")
    a = pd.to_numeric(g["p_champion_raw"], errors="coerce")
    b = pd.to_numeric(g["p_challenger"], errors="coerce")
    g = g.assign(p_champion=a, p_challenger=b)
    g = g[np.isfinite(g["p_champion"]) & np.isfinite(g["p_challenger"])]
    return pd.DataFrame({"y": g["y"].astype(float), "p_champion": g["p_champion"].astype(float),
                         "p_challenger": g["p_challenger"].astype(float),
                         "match_date": g["match_date"].astype(str).str[:10],
                         "league": g["league"].astype(str)}, columns=cols).reset_index(drop=True)


def evaluate_prospective(market: str, rows: pd.DataFrame, *, frozen_at,
                         now: datetime | None = None) -> dict:
    """Period 2 verdict for one market: PASS | FAIL | PENDING. NEVER RAISES."""
    p, h, m = CFG["prospective"], CFG["holdout"], market_cfg(market)
    now = now or utcnow()
    out = {"status": "PENDING", "n": 0, "events": 0, "ll_delta": None, "reason": "", "checks": {}}
    try:
        rows = rows if rows is not None else pd.DataFrame()
        n = int(len(rows))
        out["n"] = n
        frozen = pd.to_datetime(frozen_at, errors="coerce", utc=True)
        if n:
            y, a, b = (rows["y"].to_numpy(float), rows["p_champion"].to_numpy(float),
                       rows["p_challenger"].to_numpy(float))
            ev = int(y.sum())
            dl = PG._ll(y, b) - PG._ll(y, a)
            lo, hi, nd = PG.day_block_ci(dl, rows["match_date"].to_numpy(), n_boot=int(h["n_boot"]),
                                         seed=int(h["bootstrap_seed"]), level=float(h["ci_level"]))
            ok_c, can = PG.prediction_canary(a, b, canary_cfg(market))
            regs = league_regressions(dl, y, rows["league"].to_numpy(), market)
            br = float(np.mean((b - y) ** 2) - np.mean((a - y) ** 2))
            checks = {"min_n": n >= int(p["min_n_per_market"]),
                      "min_events": ev >= int(p["min_events_per_market"]),
                      "ll_improves": float(dl.mean()) < float(p["max_ll_delta_exclusive"]),
                      "brier_non_inferior": br <= float(m["max_brier_worsening"]),
                      "prediction_canary": bool(ok_c), "no_league_regression": not regs}
            out.update(events=ev, ll_delta=round(float(dl.mean()), 7), brier_delta=round(br, 7),
                       ci=[None if lo != lo else round(lo, 7), None if hi != hi else round(hi, 7)],
                       canary=can, league_regressions=regs, checks=checks)
            if n >= int(p["early_fail_min_n"]) and lo == lo and lo > 0:
                out.update(status="FAIL", reason=f"champion significantly ahead (CI [{lo:+.6f}, {hi:+.6f}])")
            elif not (checks["min_n"] and checks["min_events"]):
                out.update(status="PENDING", reason=f"n={n}, events={ev} < {p['min_n_per_market']} / "
                                                    f"{p['min_events_per_market']}")
            elif all(checks.values()):
                out.update(status="PASS", reason=f"delta {dl.mean():+.6f} on n={n}, events={ev}")
            else:
                out.update(status="FAIL", reason="failed: " + ", ".join(k for k, v in checks.items() if not v))
        else:
            out["reason"] = "no graded prospective rows yet"
        if out["status"] == "PENDING":
            age = (now - frozen.to_pydatetime()) if pd.notna(frozen) else None
            if age is None or age > timedelta(days=int(p["max_age_days"])):
                out.update(status="FAIL", expired=True,
                           reason=f"EXPIRED: no prospective verdict within {p['max_age_days']} days; "
                                  + out["reason"])
        return out
    except Exception as e:                                             # noqa: BLE001
        out.update(status="PENDING", reason=f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}")
        return out


# ── RETRAIN ENTRY POINT 1: judge the parked challenger ────────────────────────────────────────
def _slot_held_by_qualified(market: str, now: datetime) -> str:
    meta = _slot_meta(market)
    if not meta:
        return ""
    g = meta.get("gate") or {}
    if not (g.get("qualified") and g.get("gate_version") == GATE_VERSION):
        return ""
    frozen = pd.to_datetime(meta.get("shadow_since"), errors="coerce", utc=True)
    if pd.isna(frozen) or now - frozen.to_pydatetime() > timedelta(
            days=int(CFG["prospective"]["max_age_days"])):
        return ""
    return meta.get("model_version", "")


def _promote_parked(market: str, meta: dict, *, batch: str, now: datetime, reason: str,
                    source: str) -> None:
    raw = gzip.decompress(challenger_path(market).read_bytes())
    if reg.sha256_bytes(raw) != meta.get("artifact_sha256"):
        raise ValueError("parked challenger bytes do not match its manifest")
    man = {k: v for k, v in meta.items() if k not in ("model_role", "shadow_since", "gate")}
    _install_props_champion(market, man, raw, reason=reason, source=source, batch=batch, at=now)
    _vacate_slot(market)


def review_parked(market: str, *, now: datetime | None = None, batch: str = "",
                  h2h: pd.DataFrame | None = None,
                  outcomes: pd.DataFrame | None = None) -> dict | None:
    """Period 2 for the challenger in this market's slot — the ONLY path by which a retrained props
    model can be PROMOTED. Returns the decision record, or None when the slot is empty. NEVER RAISES."""
    now = now or utcnow()
    try:
        meta = _slot_meta(market)
        if not meta or not challenger_path(market).exists():
            return None
        version = meta.get("model_version", "")
        champ_v = champion_identity(market, now=now)
        g = meta.get("gate") or {}
        base = {"stage": "PROSPECTIVE_REVIEW", "market": market, "candidate_version": version,
                "champion_version": champ_v, "retrain_batch_id": batch,
                "qualified": bool(g.get("qualified")), "holdout_n": g.get("holdout_n"),
                "holdout_events": g.get("holdout_events"),
                "holdout_ll_delta": g.get("holdout_ll_delta"),
                "holdout_ci_lo": g.get("holdout_ci_lo"), "holdout_ci_hi": g.get("holdout_ci_hi")}

        if meta.get("parent_champion_version", "") != champ_v:
            _vacate_slot(market)
            rec = {**base, "verdict": KEEP, "promoted": False, "parked": False,
                   "reason": f"retired: frozen against {meta.get('parent_champion_version')}, "
                             f"champion is now {champ_v}"}
            record_decision(rec)
            return rec

        approved = {s.strip() for s in os.getenv(OWNER_OVERRIDE_ENV, "").split(",") if s.strip()}
        if version in approved:
            if g.get("canary_ok") is False:
                rec = {**base, "verdict": SHADOW, "promoted": False, "parked": True,
                       "reason": "OWNER_OVERRIDE refused: challenger failed the holdout canary"}
            else:
                _promote_parked(market, meta, batch=batch, now=now, source="OWNER_OVERRIDE",
                                reason=f"OWNER_OVERRIDE via {OWNER_OVERRIDE_ENV}")
                rec = {**base, "stage": "OWNER_OVERRIDE", "verdict": PROMOTE, "promoted": True,
                       "parked": False, "reason": f"owner approved {version} by exact version"}
            record_decision(rec)
            return rec

        frozen = pd.to_datetime(meta.get("shadow_since"), errors="coerce", utc=True)
        if not g.get("qualified") or g.get("gate_version") != GATE_VERSION:
            expired = pd.isna(frozen) or (now - frozen.to_pydatetime() > timedelta(
                days=int(CFG["prospective"]["max_age_days"])))
            if expired:
                _vacate_slot(market)
            rec = {**base, "verdict": KEEP if expired else SHADOW, "promoted": False,
                   "parked": not expired,
                   "reason": ("not promotable: " + ("did not qualify on its holdout"
                              if g.get("gate_version") == GATE_VERSION else
                              f"qualified under gate {g.get('gate_version') or 'none'}, not {GATE_VERSION}")
                              + ("; expired, slot cleared" if expired else "; kept as research shadow"))}
            record_decision(rec)
            return rec

        rows = prospective_rows(market, version, champ_v, meta.get("shadow_since"),
                                h2h=h2h, outcomes=outcomes)
        pv = evaluate_prospective(market, rows, frozen_at=meta.get("shadow_since"), now=now)
        base.update(prospective_status=pv["status"], prospective_n=pv["n"],
                    prospective_events=pv.get("events"), prospective_ll_delta=pv["ll_delta"],
                    checks=pv.get("checks"), metrics={"prospective": pv})
        if pv["status"] == "PASS":
            _promote_parked(market, meta, batch=batch, now=now, source="PROMOTION_GATE",
                            reason=f"{GATE_VERSION}: holdout qualified + prospective PASS - {pv['reason']}")
            rec = {**base, "verdict": PROMOTE, "promoted": True, "parked": False,
                   "reason": f"both periods passed: {pv['reason']}"}
        elif pv["status"] == "FAIL":
            _vacate_slot(market)
            rec = {**base, "verdict": KEEP, "promoted": False, "parked": False,
                   "reason": f"prospective FAIL, challenger retired: {pv['reason']}"}
        else:
            rec = {**base, "verdict": SHADOW, "promoted": False, "parked": True,
                   "reason": f"prospective pending: {pv['reason']}"}
        record_decision(rec)
        return rec
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[props_gate] {market}: prospective review failed ({e}) - champion unchanged")
        return {"verdict": KEEP, "promoted": False,
                "reason": f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}"}


# ── RETRAIN ENTRY POINT 2: score, record and (maybe) park a new candidate ─────────────────────
def submit_candidate(market: str, results: dict, holdout: pd.DataFrame | None, *,
                     training_cutoff: str | None = None, training_rows: int | None = None,
                     batch: str = "", now: datetime | None = None) -> dict:
    """Period 1 for a freshly trained candidate. NEVER returns PROMOTE (no prospective period at
    birth). FAIL CLOSED: anything that stops the comparison is KEEP_CHAMPION and the serving file
    is untouched. The only serving-file write here is BOOTSTRAP: no serving file AND no champion
    record of any kind for this market."""
    from player_model.model import payload_bytes, predict_proba

    now = now or utcnow()
    batch = batch or batch_id(now)
    gate = {"market": market, "verdict": KEEP, "promoted": False, "qualified": False,
            "parked": False, "reason": "", "gate_version": GATE_VERSION, "holdout": {},
            "candidate_version": "", "champion_version": "", "retrain_batch_id": batch}
    try:
        raw = payload_bytes(results, market)
        sha = reg.sha256_bytes(raw)
        version = make_version(market, now, sha)
        gate["candidate_version"] = version
    except Exception as e:                                             # noqa: BLE001
        gate["reason"] = f"EVALUATION_FAILED: candidate could not be serialised ({e})"
        record_decision({"stage": "HOLDOUT", "market": market, "verdict": KEEP, "promoted": False,
                         "retrain_batch_id": batch, "reason": gate["reason"]})
        return gate

    champ_v = ""
    try:
        champ_v = champion_identity(market, now=now)
    except Exception as e:                                             # noqa: BLE001
        gate["reason"] = f"EVALUATION_FAILED: champion identity unavailable ({e})"
    gate["champion_version"] = champ_v
    bootstrap = False
    ev: dict = {}

    if not serving_file(market).exists():
        ent = read_pointer().get(market) or {}
        if ent.get("model_version"):
            p = _champion_bytes_path(market, ent["model_version"])
            if p.exists() and reg.sha256_bytes(gzip.decompress(p.read_bytes())) == ent.get("artifact_sha256"):
                reg._atomic_write_bytes(serving_file(market), gzip.decompress(p.read_bytes()))
                gate["reason"] = ("serving file was missing; the registered champion's bytes were "
                                  "restored. Candidate not compared this run.")
                champ_v = gate["champion_version"] = ent["model_version"]
            else:
                gate["reason"] = ("EVALUATION_FAILED: serving file missing and the registered "
                                  "champion's bytes cannot be restored - NOT bootstrapping over a "
                                  "known champion")
        else:
            bootstrap = True
            gate.update(verdict=BOOTSTRAP, promoted=True,
                        reason="BOOTSTRAP: no serving file and no champion record of any kind for "
                               "this market - first model installed, logged")
    elif not gate["reason"]:
        if holdout is None or holdout.empty:
            gate["reason"] = "EVALUATION_FAILED: no holdout rows to score the champion on"
        else:
            try:
                with open(serving_file(market), "rb") as f:
                    old_payload = pickle.load(f)
                new_payload = pickle.loads(raw)
                y = pd.to_numeric(holdout[pconfig.MARKET_TARGETS[market]], errors="coerce")
                p_old = np.asarray(predict_proba(holdout, old_payload), float)
                p_new = np.asarray(predict_proba(holdout, new_payload), float)
                man = read_manifest(market, champ_v) or {}
                ev = evaluate_holdout(market, y.to_numpy(float), p_old, p_new,
                                      dates=holdout["date"] if "date" in holdout.columns else None,
                                      leagues=holdout["league"] if "league" in holdout.columns else None,
                                      champion_cutoff=man.get("training_cutoff") or None)
                gate["holdout"] = ev
                gate["qualified"] = bool(ev.get("qualified"))
                if ev.get("qualified"):
                    gate.update(verdict=SHADOW, reason=f"QUALIFIED on the holdout ({ev['reason']}); "
                                "no prospective period yet - frozen for period 2, not promoted")
                elif ev.get("possible_gain"):
                    gate.update(verdict=SHADOW, reason="not qualified but a material gain is inside "
                                f"the CI - research shadow, never promotable: {ev['reason']}")
                else:
                    gate.update(verdict=KEEP, reason=ev.get("reason") or "not qualified")
            except Exception as e:                                     # noqa: BLE001
                gate.update(verdict=KEEP, qualified=False,
                            reason=f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}")

    # identity, artifacts, lineage — wrapped so bookkeeping can never flip a verdict
    try:
        fcols = list(results[next(iter(results))]["feature_cols"])
        role = {SHADOW: "SHADOW", BOOTSTRAP: "CHAMPION"}.get(gate["verdict"], "REJECTED")
        manifest = {
            "market": market, "model_version": version, "retrain_batch_id": batch,
            "trained_at": reg.iso(now), "training_cutoff": training_cutoff,
            "training_rows": training_rows, "target": pconfig.MARKET_TARGETS[market],
            "feature_schema_version": hashlib.sha256("|".join(sorted(map(str, fcols))).encode()).hexdigest()[:12],
            "n_features": len(fcols), "code_commit_sha": os.getenv("GITHUB_SHA", "")[:12] or "local",
            "artifact_sha256": sha, "parent_champion_version": champ_v,
            "validation": {k: ev.get(k) for k in ("n", "events", "ll_delta", "ci", "brier_delta",
                                                   "ece_delta", "qualified", "checks")},
            "promotion_status": gate["verdict"], "model_role": role, "gate_version": GATE_VERSION,
        }
        if bootstrap:
            _install_props_champion(market, manifest, raw, reason=gate["reason"], source="BOOTSTRAP",
                                    batch=batch, at=now)
            gate["champion_version"] = version
        else:
            _write_manifest(manifest)
            if gate["verdict"] == SHADOW:
                held = _slot_held_by_qualified(market, now)
                if held and held != version:
                    gate["reason"] += f" | slot held by qualified {held}; not parked"
                else:
                    _park(market, {**manifest, "shadow_since": reg.iso(now), "gate": {
                        "gate_version": GATE_VERSION, "qualified": bool(gate["qualified"]),
                        "canary_ok": bool((ev.get("checks") or {}).get("prediction_canary", False)),
                        "holdout_n": ev.get("n"), "holdout_events": ev.get("events"),
                        "holdout_ll_delta": ev.get("ll_delta"),
                        "holdout_ci_lo": (ev.get("ci") or [None, None])[0],
                        "holdout_ci_hi": (ev.get("ci") or [None, None])[1]}}, raw)
                    gate["parked"] = True
        ci = ev.get("ci") or [None, None]
        record_decision({
            "stage": "BOOTSTRAP" if bootstrap else "HOLDOUT", "market": market,
            "candidate_version": version, "champion_version": champ_v, "verdict": gate["verdict"],
            "promoted": bootstrap, "qualified": gate["qualified"], "parked": gate["parked"],
            "retrain_batch_id": batch, "holdout_n": ev.get("n"), "holdout_events": ev.get("events"),
            "holdout_ll_delta": ev.get("ll_delta"), "holdout_ci_lo": ci[0], "holdout_ci_hi": ci[1],
            "prospective_status": "NONE", "checks": ev.get("checks") or {},
            "metrics": {k: ev.get(k) for k in ("champion", "candidate", "canary", "league_regressions",
                                               "n_dates", "untouched", "slope_deterioration")},
            "reason": gate["reason"][:600]})
    except Exception as e:                                             # noqa: BLE001
        log.error(f"[props_gate] {market}: lineage step failed ({e})")
        if bootstrap and not (serving_file(market).exists()
                              and reg.sha256_file(serving_file(market)) == sha):
            gate.update(verdict=KEEP, promoted=False,
                        reason=f"EVALUATION_FAILED: bootstrap install failed ({e}); nothing changed")
    return gate


# ── PREDICT SIDE: first-sight logging of the parked challenger ────────────────────────────────
def h2h_store(path: Path | None = None) -> FirstSightStore:
    return FirstSightStore(Path(path) if path else H2H_FILE, key=H2H_KEY, columns=H2H_COLS,
                           observed_col="logged_at", kickoff_col="kickoff_utc")


class FirstSightRecorder:
    """Collects (champion, parked challenger) probabilities for the same player-fixture during one
    props predict run and appends them once, first-sight, at the end. A recorder: it never raises
    and never changes what predict returns. Markets with no parked challenger cost nothing."""

    def __init__(self, now: datetime | None = None):
        self.now = now
        self.rows: list[dict] = []
        self._parked: dict = {}
        self._champ: dict = {}

    def _challenger(self, market: str):
        if market not in self._parked:
            self._parked[market] = load_parked(market)
            if self._parked[market] is not None:
                self._champ[market] = champion_identity(market, register=False)
        return self._parked[market]

    def add(self, market: str, feat_df: pd.DataFrame, p_champion, *, league: str,
            match_date: str, kickoff_utc: str, fixture_id) -> int:
        try:
            ch = self._challenger(market)
            if ch is None or feat_df is None or feat_df.empty or not _key(fixture_id):
                return 0
            payload, meta = ch
            from player_model.model import predict_proba
            p_ch = np.asarray(predict_proba(feat_df, payload), float)
            p_ch_ = np.asarray(p_champion, float)
            ts = reg.iso(self.now or utcnow())
            pids = feat_df["player_id"].tolist() if "player_id" in feat_df.columns else [None] * len(feat_df)
            n = 0
            for pid, a, b in zip(pids, p_ch_, p_ch):
                if not _key(pid):
                    continue
                self.rows.append({"logged_at": ts, "market": market,
                                  "challenger_version": meta.get("model_version", ""),
                                  "champion_version": self._champ.get(market, ""),
                                  "fixture_id": _key(fixture_id), "player_id": _key(pid),
                                  "league": league, "match_date": str(match_date)[:10],
                                  "kickoff_utc": kickoff_utc or "",
                                  "p_champion_raw": round(float(a), 6),
                                  "p_challenger": round(float(b), 6)})
                n += 1
            return n
        except Exception as e:                                         # noqa: BLE001
            log.warning(f"[props_gate] first-sight add failed for {market} ({e})")
            return 0

    def flush(self, path: Path | None = None) -> int:
        try:
            if not self.rows:
                return 0
            n = h2h_store(path).append(pd.DataFrame(self.rows, columns=H2H_COLS))
            self.rows = []
            if n:
                print(f"[props_gate] first-sight: {n} champion-vs-challenger row(s) -> {H2H_FILE.name}")
            return n
        except Exception as e:                                         # noqa: BLE001
            log.warning(f"[props_gate] first-sight flush failed ({e})")
            return 0
