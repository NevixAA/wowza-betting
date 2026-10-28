"""THE production promotion gate. The only code allowed to change which model serves.

    from src import promotion_gate as PG
    PG.review_parked("standard")                       # retrain start: judge the parked challenger
    gate, extra = PG.submit_candidate("standard", ...)  # retrain: score + record + maybe park
    PG.rollback("btts", "<version>", reason="...")      # operational, logged

Owner requirement, 2026-10-25: "A serving champion must NOT change because a candidate merely falls
within a loose degradation tolerance. One canonical production promotion gate ... Only that gate may
update models/registry/current_champions.json. Research and production must not have contradictory
definitions of 'promoted'. Retraining may be frequent. Promotion must be rare."

────────────────────────────────────────────────────────────────────────────────────────────
WHAT "PROMOTED" MEANS (the one definition, in v9 and in Pro)
────────────────────────────────────────────────────────────────────────────────────────────
A model is PROMOTED when, and only when, this module installs it as the serving champion. That
takes TWO independent periods, both against the champion actually serving, both per track:

  PERIOD 1 — the untouched chronological holdout (at retrain time, `submit_candidate`)
     same rows, same labels, served probabilities of both models; per track:
       n >= min_n_per_track                          (1000)
       log-loss improvement >= material floor        (0.001)
       match-day block-bootstrap 95% CI upper < 0    (improvement distinguishable from noise)
       Brier non-inferior                            (worsening <= 0.0001)
       calibration not materially worse              (ECE +0.01; slope, see below)
       prediction-distribution canary                (mean, sd ratio, quantiles, extremes, moves)
       no major league regression                    (league n >= 150: delta <= +0.02)
     A candidate that clears all of it is QUALIFIED. A qualified candidate is NOT promoted — it is
     frozen into the challenger slot (SHADOW_MORE), because nothing has yet been measured on data
     that did not exist when it was trained.

  PERIOD 2 — the independent prospective period (at later retrains, `review_parked`)
     first-sight rows of output/model_lineage/shadow_h2h.csv: what the frozen challenger and the
     serving champion each said about the same upcoming fixture at the same moment, logged before
     kickoff and after the challenger was frozen, graded from output/v91_shadow.csv. Per track:
       n >= 1000 settled rows, log-loss delta < 0, Brier non-inferior, canary, league regression.
     PASS -> PROMOTE.  FAIL (or champion significantly ahead at n >= 300, or 180 days) -> retired.
     Anything else -> SHADOW_MORE. With no prospective period the verdict is never PROMOTE.

Verdicts: PROMOTE | KEEP_CHAMPION | SHADOW_MORE (+ BOOTSTRAP / ROLLBACK, both explicit and logged).
OWNER OVERRIDE (2026-10-28): `owner_install` installs a FROZEN model type (reg.FROZEN_MODEL_TYPES,
today only "standard_repaired") on the owner's recorded approval, without either period — logged
as stage OWNER_INSTALL / PROMOTE. Retrains never submit to a frozen type (submit_candidate refuses).
Every decision, with every metric and the gate version, is appended to
output/model_lineage/promotion_decisions.csv (and, for retrain decisions, retrain_batches.csv).

────────────────────────────────────────────────────────────────────────────────────────────
THRESHOLDS — config/promotion_gate.json, gate_version PG-1.0, declared 2026-10-25
────────────────────────────────────────────────────────────────────────────────────────────
Declared before any candidate was scored under this gate. Wherever the estate had ALREADY declared
a value before 2026-10-25, that value is reused rather than re-chosen, so it cannot have been fitted
to the candidates it now judges.

* min_n_per_track 1000 (holdout). The number both earlier v9 gates and Pro's registry already used.
  At p~0.5 the per-row log-loss sd is ~0.69; with paired deltas between two related models the sd of
  the delta is ~0.02-0.05, so n=1000 gives a standard error near the 0.001 floor — below it the CI
  cannot resolve the floor at all.
* material_ll_improvement 0.001. Declared 2026-10-01 (MIN_EFFECT_LOG_LOSS). It sits inside the
  measured meta-stack leak bias (0.0006-0.0014) and at about the half-width of a 4,000-12,000 row
  holdout's CI, i.e. the smallest change this estate can see at all. Significance alone is not
  enough: with enough rows a 1e-7 gain has a CI that excludes zero.
* CI by MATCH-DATE blocks, not rows. Fixtures on one day share weather, a round's scheduling and a
  goal environment, so rows are not independent; resampling days is what src/shadow_experiments and
  Pro's model_lineage already do. Fewer than 20 dates -> no CI -> fail closed.
* Brier non-inferiority 0.0001 (MAX_BRIER_WORSE, 2026-10-18). Brier is the second proper score: a
  log-loss gain bought by a Brier loss means the gain lives in the tails.
* ECE +0.01 (both earlier gates). Calibration SLOPE is new in PG-1.0: deterioration in |slope-1| must
  exceed BOTH 0.25 and 2 standard errors of the candidate's slope to block. With the spread these
  models have (sd of p ~0.04-0.08) the slope's standard error on 4,000 rows is ~0.2, so a tighter
  bound would block on noise — the check exists for gross over/under-confidence, the estate's
  measured defect (+13.64pp).
* Canary (new numbers except where noted): mean shift <= 0.02 (2pp moves every edge on the board by
  2pp, i.e. most of a VALUABLE threshold); sd ratio within [1/1.5, 1.5] and moved>10pp share <= 25%
  (pipeline._behaviour_canary, set from the 2026-09-23 incident); max quantile shift (5/25/50/75/95)
  <= 0.05; share of p outside [0.05, 0.95] may rise by at most 2pp (pinned probabilities manufacture
  fake edges). Each is computed on the identical rows for both models, so it measures the model, not
  the data. A wholesale change of shape is an architecture change: owner override, not this gate.
* League regression: league n >= 150, delta <= +0.02 (both earlier gates). Below 150 rows a league's
  number is noise and letting it veto makes the gate superstitious.
* Prospective min_n_per_track 1000: the owner's own figure for prospective model comparisons
  (H-MODEL-01, models/registry/experiments.json, 2026-10-25). Sign replication (delta < 0) rather than
  a second significance test: period 1 already established significance on an untouched set; period 2
  asks whether it is still true on data that did not exist at freeze time. early_fail at n >= 300
  when the champion is significantly ahead, and a 180-day expiry, keep a dead challenger from holding
  the slot forever.

────────────────────────────────────────────────────────────────────────────────────────────
KNOWN COSTS, RECORDED RATHER THAN HIDDEN
────────────────────────────────────────────────────────────────────────────────────────────
* A 2026-10-04 replay of 1,568 simulated comparisons found no promotion rule beat having no gate
  (project_shadow_learning). The owner chose stability over that small average edge on 2026-10-18
  and again, explicitly, on 2026-10-25. Pro's model_lineage report measures what it costs.
* Half-time models cannot be graded prospectively from full-time scores, so ht_over05 / ht_over15
  can be QUALIFIED but never PROMOTED by this gate until a half-time outcome source exists. They are
  a measured null (project_ht_market_edge), so this costs nothing today.
* Prospective volume is ~100-200 settled rows per track per week, so a promotion takes months.
  That is the intended rate.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import model_registry as reg  # noqa: E402

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "promotion_gate.json"

PROMOTE, KEEP, SHADOW = "PROMOTE", "KEEP_CHAMPION", "SHADOW_MORE"
BOOTSTRAP, ROLLBACK = "BOOTSTRAP", "ROLLBACK"
VERDICTS = (PROMOTE, KEEP, SHADOW, BOOTSTRAP, ROLLBACK)

#: Env var holding EXACT parked challenger versions the owner has approved by hand (comma-separated).
#: It can only name a challenger that already exists, so it cannot pre-approve a future retrain.
OWNER_OVERRIDE_ENV = "PROMOTION_GATE_OWNER_OVERRIDE"

DECISION_COLS = [
    "decided_at", "gate_version", "stage", "model_type", "candidate_version", "champion_version",
    "verdict", "promoted", "qualified", "parked", "retrain_batch_id", "evaluable_tracks",
    "holdout_n", "holdout_ll_delta", "holdout_ci_lo", "holdout_ci_hi", "prospective_status",
    "prospective_n", "prospective_ll_delta", "checks", "metrics", "reason",
]


# ── configuration ─────────────────────────────────────────────────────────────────────────────
def load_config(path: Path | None = None) -> dict:
    cfg = json.loads(Path(path or CONFIG_FILE).read_text(encoding="utf-8"))
    for k in ("gate_version", "declared_on", "holdout", "canary", "prospective"):
        if k not in cfg:
            raise ValueError(f"promotion gate config missing '{k}'")
    return cfg


CFG = load_config()
GATE_VERSION = CFG["gate_version"]
#: Back-compat name used by scripts/replay_promotions.py. Same number, one source.
MIN_EFFECT_LOG_LOSS = float(CFG["holdout"]["material_ll_improvement"])
MATERIAL_LL = MIN_EFFECT_LOG_LOSS


def decisions_file() -> Path:
    return reg.LINEAGE_DIR / "promotion_decisions.csv"


def h2h_file() -> Path:
    return reg.LINEAGE_DIR / "shadow_h2h.csv"


#: Settled outcomes for prospective grading. Module-level so tests can point it elsewhere.
OUTCOMES_FILE = config.OUTPUT_DIR / "v91_shadow.csv"


# ── scoring primitives ────────────────────────────────────────────────────────────────────────
def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def calibration_slope(y, p) -> tuple[float, float]:
    """Slope (and its standard error) of logit(P(y=1)) on logit(p), by IRLS. 1.0 = calibrated
    spread; < 1 = overconfident. Returns (nan, nan) when it cannot be estimated."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    if len(y) < 30 or np.std(x) < 1e-9:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    try:
        for _ in range(50):
            eta = X @ beta
            mu = 1 / (1 + np.exp(-eta))
            w = np.clip(mu * (1 - mu), 1e-9, None)
            H = X.T @ (X * w[:, None])
            step = np.linalg.solve(H, X.T @ (y - mu))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-8:
                break
        cov = np.linalg.inv(H)
        return float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0)))
    except Exception:                                                  # noqa: BLE001
        return float("nan"), float("nan")


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    s, se = calibration_slope(y, p)
    return {"n": int(len(y)), "log_loss": round(float(_ll(y, p).mean()), 6),
            "brier": round(float(np.mean((p - y) ** 2)), 6),
            "ece": round(_ece(y, p, int(CFG["holdout"]["ece_bins"])), 6),
            "cal_slope": None if s != s else round(s, 4),
            "cal_slope_se": None if se != se else round(se, 4),
            "mean_p": round(float(p.mean()), 5), "base_rate": round(float(y.mean()), 5)}


def day_block_ci(delta, days, *, n_boot: int | None = None, seed: int | None = None,
                 level: float | None = None) -> tuple[float, float, int]:
    """CI on the mean of `delta`, resampling whole MATCH DATES with replacement.

    Returns (lo, hi, n_dates); (nan, nan, n_dates) when there are fewer dates than the declared
    minimum, which callers must treat as a failed check (fail closed)."""
    h = CFG["holdout"]
    n_boot = int(n_boot or h["n_boot"])
    seed = int(h["bootstrap_seed"] if seed is None else seed)
    level = float(level or h["ci_level"])
    by = pd.DataFrame({"d": pd.Series(days).astype(str).str[:10].to_numpy(),
                       "x": np.asarray(delta, float)}).groupby("d")["x"].agg(["sum", "size"])
    nd = int(len(by))
    if nd < int(h["min_bootstrap_blocks"]):
        return float("nan"), float("nan"), nd
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nd, size=(n_boot, nd))
    means = by["sum"].to_numpy()[idx].sum(1) / by["size"].to_numpy()[idx].sum(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), nd


def block_bootstrap_ci(per_row_delta: np.ndarray, block: int = 50, n_boot: int = 2000,
                       seed: int = 20261001) -> tuple[float, float]:
    """Fixed-length ROW-block bootstrap. Kept for research scripts (scripts/movement_segments.py);
    the gate itself uses `day_block_ci`."""
    rng = np.random.default_rng(seed)
    n = len(per_row_delta)
    if n < block * 3:
        block = max(1, n // 10)
    starts = np.arange(0, max(n - block, 1))
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng.choice(starts, k, replace=True)])[:n]
        means[i] = per_row_delta[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def prediction_canary(p_champion, p_candidate, cfg: dict | None = None) -> tuple[bool, dict]:
    """Has the candidate's probability DISTRIBUTION moved more than a retrain should move it?

    Same rows for both models, so every number here is a property of the model, not of the data.
    Returns (ok, details) — details lists each failure by name."""
    c = cfg or CFG["canary"]
    pc, pn = np.asarray(p_champion, float), np.asarray(p_candidate, float)
    sc, sn = float(np.std(pc)), float(np.std(pn))
    ratio = (sn / sc) if sc > 1e-9 else (1.0 if sn <= 1e-9 else float("inf"))
    qs = [float(q) for q in c["quantiles"]]
    qshift = float(np.max(np.abs(np.quantile(pn, qs) - np.quantile(pc, qs))))
    ext = lambda p: float(np.mean((p < c["extreme_lo"]) | (p > c["extreme_hi"])))  # noqa: E731
    ext_inc = ext(pn) - ext(pc)
    moved = float(np.mean(np.abs(pn - pc) > c["moved_pp"]))
    d = {"mean_champion": round(float(pc.mean()), 5), "mean_candidate": round(float(pn.mean()), 5),
         "mean_shift": round(abs(float(pn.mean() - pc.mean())), 5),
         "sd_champion": round(sc, 5), "sd_candidate": round(sn, 5),
         "sd_ratio": round(ratio, 4) if np.isfinite(ratio) else None,
         "max_quantile_shift": round(qshift, 5), "extreme_share_increase": round(ext_inc, 5),
         "frac_moved": round(moved, 5)}
    fails = []
    if d["mean_shift"] > c["max_mean_shift"]:
        fails.append(f"mean shift {d['mean_shift']:.4f} > {c['max_mean_shift']}")
    if not (np.isfinite(ratio) and c["sd_ratio_min"] <= ratio <= c["sd_ratio_max"]):
        fails.append(f"sd ratio {ratio:.3f} outside [{c['sd_ratio_min']}, {c['sd_ratio_max']}]")
    if qshift > c["max_quantile_shift"]:
        fails.append(f"quantile shift {qshift:.4f} > {c['max_quantile_shift']}")
    if ext_inc > c["max_extreme_share_increase"]:
        fails.append(f"extreme-probability share +{ext_inc:.3f} > {c['max_extreme_share_increase']}")
    if moved > c["max_frac_moved"]:
        fails.append(f"{moved:.1%} of rows moved > {c['moved_pp']:.0%}")
    d["failures"] = fails
    return not fails, d


def league_regressions(per_row_delta, leagues, min_n: int, max_reg: float) -> list[dict]:
    out = []
    if leagues is None:
        return out
    s = pd.Series(np.asarray(per_row_delta, float)).groupby(
        pd.Series(leagues).astype(str).to_numpy())
    for name, g in s:
        if len(g) >= min_n and float(g.mean()) > max_reg:
            out.append({"league": name, "n": int(len(g)), "ll_delta": round(float(g.mean()), 5)})
    return out


def track_of(league) -> str:
    """standard | new_format | unknown. The canonical tag; tracks are never pooled."""
    return config.model_type_for_league(league)


# ── PERIOD 1: the untouched holdout ───────────────────────────────────────────────────────────
def _evaluate_track(y, pc, pn, days, leagues) -> dict:
    h = CFG["holdout"]
    n = int(len(y))
    rec = {"n": n, "evaluable": n >= int(h["min_n_per_track"]), "checks": {}, "passed": False}
    if not rec["evaluable"]:
        rec["reason"] = f"n={n} < {h['min_n_per_track']}: reported, excluded from the decision"
        return rec
    cm, nm = metrics(y, pc), metrics(y, pn)
    per_row = _ll(y, pn) - _ll(y, pc)
    d = float(per_row.mean())
    lo, hi, nd = day_block_ci(per_row, days)
    canary_ok, canary = prediction_canary(pc, pn)
    regs = league_regressions(per_row, leagues, int(h["min_league_n"]),
                              float(h["max_league_ll_regression"]))
    s_c, s_n, se_n = cm["cal_slope"], nm["cal_slope"], nm["cal_slope_se"]
    if s_c is None or s_n is None or se_n is None:
        slope_ok, slope_det = True, None            # unestimable slope cannot veto; ECE still does
    else:
        slope_det = abs(s_n - 1) - abs(s_c - 1)
        slope_ok = not (slope_det > float(h["max_slope_deterioration"])
                        and slope_det > float(h["slope_se_multiple"]) * se_n)
    checks = {
        "material_improvement": d <= -float(h["material_ll_improvement"]),
        "ci_excludes_zero": bool(hi == hi and hi < 0),
        "brier_non_inferior": (nm["brier"] - cm["brier"]) <= float(h["max_brier_worsening"]),
        "ece_not_worse": (nm["ece"] - cm["ece"]) <= float(h["max_ece_worsening"]),
        "slope_not_worse": bool(slope_ok),
        "prediction_canary": bool(canary_ok),
        "no_league_regression": not regs,
    }
    rec.update(champion=cm, candidate=nm, ll_delta=round(d, 6),
               brier_delta=round(nm["brier"] - cm["brier"], 6),
               ece_delta=round(nm["ece"] - cm["ece"], 6),
               slope_deterioration=None if slope_det is None else round(slope_det, 4),
               ci=[None if lo != lo else round(lo, 6), None if hi != hi else round(hi, 6)],
               n_dates=nd, canary=canary, league_regressions=regs,
               checks={k: bool(v) for k, v in checks.items()},
               passed=all(checks.values()),
               possible_gain=bool(lo == lo and lo <= -float(h["material_ll_improvement"])
                                  and d < 0 and canary_ok))
    rec["reason"] = ("all holdout criteria met" if rec["passed"] else
                     "failed: " + ", ".join(k for k, v in checks.items() if not v))
    return rec


def evaluate_holdout(y, p_champion, p_candidate, *, dates, leagues=None, tracks=None,
                     champion_cutoff=None) -> dict:
    """Period 1. NEVER RAISES — any failure returns qualified=False with the reason.

    `tracks` defaults to track_of(league) per row. `champion_cutoff`, when known, must precede the
    holdout (the champion may not have trained on it). Unknown cutoffs (legacy champions) are
    recorded as UNVERIFIED and do not block: contamination of the CHAMPION can only flatter the
    champion, never the candidate, so it biases this gate against promotion, not toward it.
    """
    out = {"qualified": False, "possible_gain": False, "tracks": {}, "evaluable_tracks": [],
           "reason": "", "holdout_n": 0, "untouched": "UNVERIFIED"}
    try:
        y = np.asarray(y, float)
        pc = np.asarray(p_champion, float)
        pn = np.asarray(p_candidate, float)
        if not (len(y) == len(pc) == len(pn)) or dates is None or len(dates) != len(y):
            out["reason"] = "EVALUATION_FAILED: labels, predictions and dates differ in length"
            return out
        if tracks is None:
            if leagues is None:
                out["reason"] = "EVALUATION_FAILED: no league labels — tracks cannot be separated"
                return out
            tracks = [track_of(lg) for lg in leagues]
        dts = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        lg = pd.Series(list(leagues)).astype(str) if leagues is not None else None
        tr = pd.Series(list(tracks)).astype(str)
        m = np.isfinite(y) & np.isfinite(pc) & np.isfinite(pn) & dts.notna().to_numpy()
        y, pc, pn = y[m], pc[m], pn[m]
        dts, tr = dts[m].reset_index(drop=True), tr[m].reset_index(drop=True)
        lg = None if lg is None else lg[m].reset_index(drop=True)
        out["holdout_n"] = int(len(y))
        out["dropped_rows"] = int((~m).sum())
        if len(y) == 0:
            out["reason"] = "EVALUATION_FAILED: no usable holdout rows"
            return out
        if champion_cutoff:
            cut = pd.to_datetime(champion_cutoff, errors="coerce")
            if pd.notna(cut):
                if dts.min() <= cut:
                    out["untouched"] = "VIOLATED"
                    out["reason"] = (f"EVALUATION_FAILED: holdout starts {dts.min().date()} but the "
                                     f"champion trained through {cut.date()} — not untouched")
                    return out
                out["untouched"] = "VERIFIED"
        days = dts.dt.strftime("%Y-%m-%d").to_numpy()
        for t in sorted(tr.unique()):
            ii = np.flatnonzero((tr == t).to_numpy())
            out["tracks"][t] = _evaluate_track(y[ii], pc[ii], pn[ii], days[ii],
                                               None if lg is None else lg.iloc[ii].to_numpy())
        ev = [t for t, r in out["tracks"].items() if r["evaluable"]]
        out["evaluable_tracks"] = ev
        if not ev:
            out["reason"] = (f"no track reaches n >= {CFG['holdout']['min_n_per_track']} — "
                             f"nothing can be decided")
            return out
        out["qualified"] = all(out["tracks"][t]["passed"] for t in ev)
        out["possible_gain"] = any(out["tracks"][t].get("possible_gain") for t in ev)
        out["reason"] = "; ".join(f"[{t}] {out['tracks'][t]['reason']}" for t in ev)
        return out
    except Exception as e:                                             # noqa: BLE001
        out.update(qualified=False, reason=f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}")
        return out


# ── PERIOD 2: the independent prospective period ──────────────────────────────────────────────
def _label(market: str, hg: int, ag: int) -> float | None:
    m = str(market).lower()
    if m == "btts":
        return float(hg > 0 and ag > 0)
    line = {"over25": 2.5, "over15": 1.5, "over35": 3.5}.get(m)
    return None if line is None else float(hg + ag > line)


def prospective_rows(model_type: str, challenger_version: str, champion_version: str,
                     frozen_at, h2h: pd.DataFrame | None = None,
                     outcomes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Graded first-sight rows for (challenger, champion). Columns: y, p_champion, p_challenger,
    match_date, league, track. Only rows logged BEFORE kickoff and AFTER the freeze count."""
    cols = ["y", "p_champion", "p_challenger", "match_date", "league", "track"]
    if h2h is None:
        f = h2h_file()
        h2h = pd.read_csv(f, dtype=str, keep_default_na=False) if f.exists() else pd.DataFrame()
    if outcomes is None:
        outcomes = (pd.read_csv(OUTCOMES_FILE, dtype=str, keep_default_na=False)
                    if Path(OUTCOMES_FILE).exists() else pd.DataFrame())
    need_h = {"model_type", "challenger_version", "champion_version", "logged_at", "match_date",
              "home_team", "away_team", "league", "p_champion_raw", "p_challenger"}
    need_o = {"match_date", "home_team", "away_team", "home_goals", "away_goals"}
    if h2h.empty or outcomes.empty or not need_h <= set(h2h.columns) \
            or not need_o <= set(outcomes.columns):
        return pd.DataFrame(columns=cols)
    if "kickoff_utc" not in h2h.columns:
        h2h = h2h.assign(kickoff_utc="")
    if "market" not in h2h.columns:
        h2h = h2h.assign(market=reg.TARGET.get(model_type, ""))
    h = h2h[(h2h["model_type"].astype(str) == model_type)
            & (h2h["challenger_version"].astype(str) == challenger_version)
            & (h2h["champion_version"].astype(str) == champion_version)].copy()
    if h.empty:
        return pd.DataFrame(columns=cols)
    logged = pd.to_datetime(h["logged_at"].astype(str), errors="coerce", utc=True, format="mixed")
    ko = pd.to_datetime(h["kickoff_utc"].astype(str), errors="coerce", utc=True, format="mixed")
    md = pd.to_datetime(h["match_date"].astype(str).str[:10], errors="coerce", utc=True)
    before_ko = pd.Series(np.where(ko.notna(), logged < ko, logged < md), index=h.index)
    frozen = pd.to_datetime(frozen_at, errors="coerce", utc=True)
    after_freeze = (logged >= frozen) if pd.notna(frozen) else pd.Series(False, index=h.index)
    h = h[logged.notna() & before_ko & after_freeze]
    o = outcomes[(outcomes["home_goals"].astype(str).str.strip() != "")
                 & (outcomes["away_goals"].astype(str).str.strip() != "")]
    goals = {}
    for r in o.itertuples(index=False):
        try:
            goals[(str(r.match_date)[:10], str(r.home_team), str(r.away_team))] = (
                int(float(r.home_goals)), int(float(r.away_goals)))
        except (TypeError, ValueError):
            continue
    rows = []
    for r in h.itertuples(index=False):
        g = goals.get((str(r.match_date)[:10], str(r.home_team), str(r.away_team)))
        if g is None:
            continue
        yv = _label(getattr(r, "market", reg.TARGET.get(model_type, "")), *g)
        try:
            a, b = float(r.p_champion_raw), float(r.p_challenger)
        except (TypeError, ValueError):
            continue
        if yv is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        rows.append({"y": yv, "p_champion": a, "p_challenger": b,
                     "match_date": str(r.match_date)[:10], "league": str(r.league),
                     "track": track_of(r.league)})
    return pd.DataFrame(rows, columns=cols)


def evaluate_prospective(rows: pd.DataFrame, required_tracks: list[str], *, frozen_at,
                         now: datetime | None = None) -> dict:
    """Period 2 verdict: PASS | FAIL | PENDING, per required track and overall. NEVER RAISES."""
    p = CFG["prospective"]
    h = CFG["holdout"]
    now = now or datetime.now(timezone.utc)
    out = {"status": "PENDING", "tracks": {}, "reason": "", "n": 0, "ll_delta": None}
    try:
        frozen = pd.to_datetime(frozen_at, errors="coerce", utc=True)
        if not required_tracks:
            out.update(status="FAIL", reason="no evaluable holdout track recorded at freeze")
            return out
        rows = rows if rows is not None else pd.DataFrame()
        out["n"] = int(len(rows))
        if len(rows):
            dl_all = _ll(rows["y"], rows["p_challenger"]) - _ll(rows["y"], rows["p_champion"])
            out["ll_delta"] = round(float(dl_all.mean()), 6)
        for t in required_tracks:
            g = rows[rows["track"] == t] if len(rows) else rows
            n = int(len(g))
            r = {"n": n}
            if n:
                y, a, b = (g["y"].to_numpy(float), g["p_champion"].to_numpy(float),
                           g["p_challenger"].to_numpy(float))
                dl = _ll(y, b) - _ll(y, a)
                lo, hi, nd = day_block_ci(dl, g["match_date"].to_numpy())
                ok_c, can = prediction_canary(a, b)
                regs = league_regressions(dl, g["league"].to_numpy(), int(h["min_league_n"]),
                                          float(h["max_league_ll_regression"]))
                br = float(np.mean((b - y) ** 2) - np.mean((a - y) ** 2))
                checks = {"min_n": n >= int(p["min_n_per_track"]),
                          "ll_improves": float(dl.mean()) < float(p["max_ll_delta_exclusive"]),
                          "brier_non_inferior": br <= float(p["max_brier_worsening"]),
                          "prediction_canary": bool(ok_c), "no_league_regression": not regs}
                r.update(ll_delta=round(float(dl.mean()), 6), brier_delta=round(br, 6),
                         ci=[None if lo != lo else round(lo, 6),
                             None if hi != hi else round(hi, 6)],
                         canary=can, league_regressions=regs, checks=checks)
                if n >= int(p["early_fail_min_n"]) and lo == lo and lo > 0:
                    r["status"] = "FAIL"
                    r["reason"] = f"champion significantly ahead (CI [{lo:+.5f}, {hi:+.5f}])"
                elif not checks["min_n"]:
                    r["status"] = "PENDING"
                    r["reason"] = f"n={n} < {p['min_n_per_track']}"
                elif all(checks.values()):
                    r["status"] = "PASS"
                    r["reason"] = f"delta {dl.mean():+.5f} on n={n}"
                else:
                    r["status"] = "FAIL"
                    r["reason"] = "failed: " + ", ".join(k for k, v in checks.items() if not v)
            else:
                r.update(status="PENDING", reason="no graded prospective rows yet")
            out["tracks"][t] = r
        st = [r["status"] for r in out["tracks"].values()]
        age = (now - frozen.to_pydatetime()) if pd.notna(frozen) else None
        if "FAIL" in st:
            out["status"] = "FAIL"
        elif all(s == "PASS" for s in st):
            out["status"] = "PASS"
        elif age is None or age > timedelta(days=int(p["max_age_days"])):
            out["status"] = "FAIL"
            out["expired"] = True
        out["reason"] = "; ".join(f"[{t}] {r['status']}: {r['reason']}"
                                  for t, r in out["tracks"].items())
        if out.get("expired"):
            out["reason"] = (f"EXPIRED: no prospective verdict within {p['max_age_days']} days; "
                             + out["reason"])
        return out
    except Exception as e:                                             # noqa: BLE001
        out.update(status="PENDING", reason=f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}")
        return out


# ── the ONLY writers of the champion pointer and the serving file ─────────────────────────────
def _install_champion(model_type: str, manifest: dict, raw: bytes, *, reason: str,
                      batch: str = "", source: str = "PROMOTION_GATE",
                      holdout: dict | None = None, at: datetime | None = None) -> None:
    """ATOMIC INSTALL. Order is chosen so every crash point is recoverable:

    1. immutable copy of the bytes into champions/   (rollback target exists before anything moves)
    2. manifest                                       (identity exists)
    3. serving file, write-then-rename                (predict sees old bytes or new, never half)
    4. pointer, write-then-rename
    5. history row opening the new era

    A crash between 3 and 4 leaves the pointer naming the previous champion while the file holds
    the new one; predict stamps rows with the file's real hash (from the manifest in step 2).
    """
    at = at or reg.utcnow()
    version = manifest["model_version"]
    reg.store_champion_artifact(model_type, version, raw)
    reg.write_manifest({**manifest, "model_role": "CHAMPION"})
    reg._atomic_write_bytes(reg.MODEL_FILES[model_type], raw)
    ptr = reg.read_pointer()
    prior = (ptr.get(model_type) or {}).get("model_version", "")
    ptr[model_type] = {
        "model_version": version, "champion_since": reg.iso(at),
        "retrain_batch_id": manifest.get("retrain_batch_id", ""),
        "training_cutoff": manifest.get("training_cutoff", ""),
        "artifact_sha256": manifest["artifact_sha256"],
        "promotion_source": source, "promotion_reason": reason[:400],
        "gate_version": GATE_VERSION,
    }
    reg._atomic_write_json(reg.POINTER_FILE, ptr)
    hd = holdout or {}
    reg._append_csv(reg.CHAMPION_HISTORY, reg.CHAMPION_HISTORY_COLS, {
        "model_type": model_type, "champion_version": version, "start_at": reg.iso(at),
        "end_at": "", "promotion_retrain_batch": batch, "prior_champion": prior,
        "holdout_n": hd.get("n"), "holdout_logloss": hd.get("log_loss"),
        "holdout_brier": hd.get("brier"), "holdout_ece": hd.get("ece"),
        "fresh_oos_logloss": hd.get("fresh_oos_logloss"), "promotion_source": source,
        "promotion_reason": reason[:400], "artifact_sha256": manifest["artifact_sha256"],
        "git_commit": manifest.get("code_commit_sha", ""), "reconstructed": False,
    })
    reg._version_cache.clear()


def rollback(model_type: str, to_version: str, reason: str = "operational rollback") -> None:
    """Restore an earlier champion's exact bytes. Operational, owner-initiated, logged."""
    p = reg.champion_artifact_path(model_type, to_version)
    man = reg.read_manifest(model_type, to_version)
    if not p.exists() or man is None:
        raise FileNotFoundError(f"no stored champion artifact/manifest for {to_version}")
    raw = gzip.decompress(p.read_bytes())
    if reg.sha256_bytes(raw) != man["artifact_sha256"]:
        raise ValueError(f"{to_version}: stored bytes do not match the manifest hash")
    prior = (reg.read_pointer().get(model_type) or {}).get("model_version", "")
    _install_champion(model_type, man, raw, reason=f"ROLLBACK: {reason}", source="ROLLBACK")
    record_decision({"stage": "ROLLBACK", "model_type": model_type, "candidate_version": to_version,
                     "champion_version": prior, "verdict": ROLLBACK, "promoted": True,
                     "reason": reason})


OWNER_INSTALL = "OWNER_INSTALL"


def owner_install(model_type: str, raw: bytes, *, approval: str, approved_by: str,
                  training: dict | None = None, supersede: str = "",
                  now: datetime | None = None) -> dict:
    """EXPLICIT OWNER OVERRIDE of PG-1.0 for a FROZEN model type (owner 2026-10-28).

    PG-1.0 promotes a model only after a holdout qualification AND a prospective period against
    the serving champion. The repaired Standard OU25 model is being switched in BEFORE any forward
    evidence, by owner decision — that is not something the gate can decide, so it is not routed
    through `submit_candidate` (which would correctly refuse it) and it is not hidden either:

      * only model types in reg.FROZEN_MODEL_TYPES (a separate serving file and champion; the legacy
        `standard` champion and its bytes are never touched, so the rollback stays one env value);
      * the payload must be a valid repaired Standard OU25 payload (src/standard_serving);
      * `approval` (the owner's words, verbatim) and `approved_by` are mandatory and are written to
        the manifest, the champion pointer, champion_history.csv and promotion_decisions.csv
        (stage OWNER_INSTALL, verdict PROMOTE) — so system_health's "pointer == last promoted
        decision" check holds and the act is auditable;
      * an existing champion of the type is replaced only when `supersede` names it EXACTLY.

    Written through `_install_champion`, i.e. this module stays the only writer of the serving
    file and of current_champions.json. Raises on any refusal; nothing is written in that case.
    """
    import pickle
    from src import standard_serving
    now = now or reg.utcnow()
    training = dict(training or {})
    if model_type not in reg.FROZEN_MODEL_TYPES:
        raise PermissionError(f"owner_install is only for frozen model types "
                              f"{sorted(reg.FROZEN_MODEL_TYPES)}; {model_type} goes through the gate")
    if not str(approval or "").strip() or not str(approved_by or "").strip():
        raise PermissionError("owner_install needs the owner's approval text and approver name")
    standard_serving.validate_payload(pickle.loads(raw))
    prior = (reg.read_pointer().get(model_type) or {}).get("model_version", "")
    if prior and supersede != prior:
        raise PermissionError(f"{model_type} already has champion {prior}; pass supersede={prior!r} "
                              f"to replace it deliberately")
    sha = reg.sha256_bytes(raw)
    trained_at = pd.to_datetime(training.get("trained_at") or reg.iso(now), utc=True).to_pydatetime()
    version = reg.make_version(model_type, trained_at, sha)
    payload = pickle.loads(raw)
    fcols = list(payload.get("feature_cols") or [])
    import hashlib
    held = payload.get("served_holdout_metrics") or {}
    manifest = {
        "model_type": model_type, "model_version": version,
        "retrain_batch_id": training.get("retrain_batch_id") or f"owner_{now:%Y%m%dT%H%M%SZ}",
        "trained_at": reg.iso(trained_at),
        "training_start": training.get("training_start"), "training_end": training.get("training_end"),
        "training_cutoff": training.get("training_cutoff"), "training_rows": training.get("training_rows"),
        "target": "over25", "recipe": payload.get("recipe"),
        "model_recipe_version": payload.get("model_recipe_version"),
        "imputer_version": payload.get("imputer_version"),
        "feature_schema_version": hashlib.sha256("|".join(sorted(map(str, fcols))).encode()).hexdigest()[:12],
        "n_features": len(fcols),
        "code_commit_sha": os.getenv("GITHUB_SHA", "")[:12] or training.get("code_commit_sha") or "local",
        "artifact_sha256": sha, "parent_champion_version": prior,
        "validation_metrics": {"served_holdout": held, "blocks": payload.get("blocks")},
        "promotion_status": OWNER_INSTALL, "model_role": "CHAMPION",
        "owner_approval": {"text": approval, "approved_by": approved_by, "installed_at": reg.iso(now),
                           "gate_version_overridden": GATE_VERSION},
        "research": payload.get("research"),
    }
    reason = f"{OWNER_INSTALL} by {approved_by}: {approval}"
    _install_champion(model_type, manifest, raw, reason=reason, batch=manifest["retrain_batch_id"],
                      source=OWNER_INSTALL, holdout=held, at=now)
    stem = reg.MODEL_FILES[model_type].stem
    reg._atomic_write_bytes(config.MODELS_DIR / f"metrics_{stem}.json",
                            json.dumps(payload.get("metrics") or {}, indent=2, default=str).encode("utf-8"))
    record_decision({"stage": OWNER_INSTALL, "model_type": model_type, "candidate_version": version,
                     "champion_version": prior, "verdict": PROMOTE, "promoted": True,
                     "qualified": False, "parked": False,
                     "retrain_batch_id": manifest["retrain_batch_id"], "prospective_status": "NONE",
                     "metrics": {"served_holdout": held}, "reason": reason[:600]})
    reg.update_high_water(model_type, {"champion_version": version,
                                       "historical": {**held, "promoted_at": reg.iso(now),
                                                      "gate_version": GATE_VERSION,
                                                      "promotion_source": OWNER_INSTALL}})
    log.warning(f"[{model_type}] OWNER_INSTALL {version} (prior: {prior or 'none'}) — {reason[:200]}")
    return manifest


def reconstruct_pointer_if_missing(pointer: dict, source: str = "LEGACY_GIT") -> bool:
    """One-off lineage reconstruction (scripts/backfill_model_lineage.py). Writes the pointer only
    when NONE exists — it can never replace a champion — and logs the act. Returns True if written."""
    if reg.POINTER_FILE.exists():
        return False
    reg._atomic_write_json(reg.POINTER_FILE, pointer)
    for mt, p in pointer.items():
        record_decision({"stage": "RECONSTRUCT", "model_type": mt,
                         "candidate_version": p.get("model_version", ""), "verdict": BOOTSTRAP,
                         "promoted": False, "reason": f"pointer reconstructed ({source})"})
    return True


def _restore_serving_file(model_type: str) -> bool:
    """The serving file is missing but the registry knows the champion: put ITS bytes back.
    Not a promotion — the champion does not change."""
    ent = reg.read_pointer().get(model_type) or {}
    v = ent.get("model_version", "")
    p = reg.champion_artifact_path(model_type, v) if v else None
    if not (p and p.exists()):
        return False
    raw = gzip.decompress(p.read_bytes())
    if reg.sha256_bytes(raw) != ent.get("artifact_sha256"):
        return False
    reg._atomic_write_bytes(reg.MODEL_FILES[model_type], raw)
    reg._version_cache.pop(model_type, None)
    return True


# ── decision log ──────────────────────────────────────────────────────────────────────────────
def record_decision(row: dict) -> None:
    """APPEND-ONLY, one row per gate decision, with the gate version."""
    def _j(v):
        return v if isinstance(v, str) else json.dumps(v, default=str, sort_keys=True)
    r = {"decided_at": reg.iso(reg.utcnow()), "gate_version": GATE_VERSION, **row}
    for k in ("checks", "metrics", "evaluable_tracks"):
        if k in r and r[k] is not None:
            r[k] = _j(r[k])
    reg._append_csv(decisions_file(), DECISION_COLS, r)


def _summary_track(gate: dict) -> dict:
    """The track the retrain_batches row reports: the largest evaluable one (never a pool)."""
    tr = gate.get("tracks") or {}
    ev = [t for t in gate.get("evaluable_tracks") or [] if t in tr]
    if not ev:
        return {}
    return tr[max(ev, key=lambda t: tr[t]["n"])]


# ── challenger slot policy ────────────────────────────────────────────────────────────────────
def _slot_meta(model_type: str) -> dict | None:
    m = reg.challenger_meta_path(model_type)
    try:
        return json.loads(m.read_text(encoding="utf-8")) if m.exists() else None
    except Exception:                                                  # noqa: BLE001
        return None


def _slot_held_by_qualified(model_type: str, now: datetime) -> str:
    """Version of a QUALIFIED parked challenger still inside its prospective window, else ''."""
    meta = _slot_meta(model_type)
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


# ── RETRAIN ENTRY POINT 1: judge the parked challenger ────────────────────────────────────────
def review_parked(model_type: str, *, now: datetime | None = None, batch: str = "",
                  h2h: pd.DataFrame | None = None,
                  outcomes: pd.DataFrame | None = None) -> dict | None:
    """Period 2 for the challenger in the slot. The ONLY path by which a retrain challenger can be
    PROMOTED. Returns the decision record, or None when the slot is empty. NEVER RAISES."""
    now = now or reg.utcnow()
    if model_type in reg.FROZEN_MODEL_TYPES:
        return None                       # frozen: no challenger slot is ever reviewed (2026-10-28)
    try:
        ch = reg.load_challenger(model_type)
        if ch is None:
            return None
        payload, meta = ch
        version = meta.get("model_version", "")
        champ_v = (reg.read_pointer().get(model_type) or {}).get("model_version", "")
        g = meta.get("gate") or {}
        base = {"stage": "PROSPECTIVE_REVIEW", "model_type": model_type,
                "candidate_version": version, "champion_version": champ_v,
                "retrain_batch_id": batch, "qualified": bool(g.get("qualified")),
                "evaluable_tracks": g.get("evaluable_tracks") or [],
                "holdout_n": g.get("holdout_n"), "holdout_ll_delta": g.get("holdout_ll_delta"),
                "holdout_ci_lo": g.get("holdout_ci_lo"), "holdout_ci_hi": g.get("holdout_ci_hi")}

        # Its holdout evidence was measured against a champion that no longer serves.
        if meta.get("parent_champion_version", "") != champ_v:
            reg.clear_challenger(model_type)
            rec = {**base, "verdict": KEEP, "promoted": False, "parked": False,
                   "reason": f"retired: frozen against {meta.get('parent_champion_version')}, "
                             f"champion is now {champ_v}"}
            record_decision(rec)
            return rec

        # Owner override: an EXACT, already-parked version, canary still required.
        approved = {s.strip() for s in os.getenv(OWNER_OVERRIDE_ENV, "").split(",") if s.strip()}
        if version in approved:
            if g.get("canary_ok") is False:
                rec = {**base, "verdict": SHADOW, "promoted": False, "parked": True,
                       "reason": "OWNER_OVERRIDE refused: challenger failed the holdout canary"}
                record_decision(rec)
                return rec
            _promote_parked(model_type, payload, meta, batch, now,
                            reason=f"OWNER_OVERRIDE via {OWNER_OVERRIDE_ENV}", source="OWNER_OVERRIDE")
            rec = {**base, "stage": "OWNER_OVERRIDE", "verdict": PROMOTE, "promoted": True,
                   "parked": False, "reason": f"owner approved {version} by exact version"}
            record_decision(rec)
            return rec

        if not g.get("qualified") or g.get("gate_version") != GATE_VERSION:
            frozen = pd.to_datetime(meta.get("shadow_since"), errors="coerce", utc=True)
            expired = pd.isna(frozen) or (now - frozen.to_pydatetime() > timedelta(
                days=int(CFG["prospective"]["max_age_days"])))
            if expired:
                reg.clear_challenger(model_type)
            rec = {**base, "verdict": KEEP if expired else SHADOW, "promoted": False,
                   "parked": not expired,
                   "reason": ("not promotable: " + ("did not qualify on its holdout"
                              if g.get("gate_version") == GATE_VERSION else
                              f"qualified under gate {g.get('gate_version') or 'none'}, not "
                              f"{GATE_VERSION}") + ("; expired, slot cleared" if expired else
                                                    "; kept as research shadow"))}
            record_decision(rec)
            return rec

        rows = prospective_rows(model_type, version, champ_v, meta.get("shadow_since"),
                                h2h=h2h, outcomes=outcomes)
        pv = evaluate_prospective(rows, list(g.get("evaluable_tracks") or []),
                                  frozen_at=meta.get("shadow_since"), now=now)
        base.update(prospective_status=pv["status"], prospective_n=pv["n"],
                    prospective_ll_delta=pv["ll_delta"],
                    checks={t: r.get("checks") for t, r in pv["tracks"].items()},
                    metrics={"prospective": pv["tracks"]})
        if pv["status"] == "PASS":
            _promote_parked(model_type, payload, meta, batch, now,
                            reason=f"{GATE_VERSION}: holdout qualified + prospective PASS — "
                                   f"{pv['reason']}"[:400], source="PROMOTION_GATE")
            rec = {**base, "verdict": PROMOTE, "promoted": True, "parked": False,
                   "reason": f"both periods passed: {pv['reason']}"}
        elif pv["status"] == "FAIL":
            reg.clear_challenger(model_type)
            rec = {**base, "verdict": KEEP, "promoted": False, "parked": False,
                   "reason": f"prospective FAIL, challenger retired: {pv['reason']}"}
        else:
            rec = {**base, "verdict": SHADOW, "promoted": False, "parked": True,
                   "reason": f"prospective pending: {pv['reason']}"}
        record_decision(rec)
        if rec["verdict"] == PROMOTE:
            log.info(f"  [{model_type}] PROMOTED parked challenger {version} — {rec['reason']}")
        else:
            log.info(f"  [{model_type}] parked challenger {version}: {rec['verdict']} — "
                     f"{rec['reason'][:200]}")
        return rec
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"  [{model_type}] prospective review failed ({e}) — champion unchanged")
        return {"verdict": KEEP, "promoted": False,
                "reason": f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}"}


def _promote_parked(model_type, payload, meta, batch, now, *, reason, source) -> None:
    import pickle
    raw = gzip.decompress(reg.challenger_path(model_type).read_bytes())
    if reg.sha256_bytes(raw) != meta.get("artifact_sha256"):
        raise ValueError("parked challenger bytes do not match its manifest")
    man = {k: v for k, v in meta.items() if k not in ("model_role", "shadow_since", "gate")}
    hold = ((meta.get("gate") or {}).get("summary_candidate_metrics") or {})
    _install_champion(model_type, man, raw, reason=reason, batch=batch, source=source,
                      holdout=hold, at=now)
    met = (payload or pickle.loads(raw)).get("metrics") or {}
    stem = reg.MODEL_FILES[model_type].stem
    reg._atomic_write_bytes(config.MODELS_DIR / f"metrics_{stem}.json",
                            json.dumps(met, indent=2, default=str).encode("utf-8"))
    reg.clear_challenger(model_type)
    reg.update_high_water(model_type, {"champion_version": meta["model_version"],
                                       "historical": {**hold, "promoted_at": reg.iso(now),
                                                      "gate_version": GATE_VERSION,
                                                      "retrain_batch_id": batch}})


# ── RETRAIN ENTRY POINT 2: score, record and (maybe) park a new candidate ─────────────────────
def submit_candidate(model_type: str, model_file, old_payload, results: dict, test_df,
                     target: str, basis: str, valid, *, batch: str = "",
                     now: datetime | None = None) -> tuple[dict, dict]:
    """Period 1 for a freshly trained candidate. NEVER returns PROMOTE: a candidate has no
    prospective period at birth. Returns (gate, extra); gate["verdict"] in KEEP/SHADOW/BOOTSTRAP.

    FAIL CLOSED: anything that stops the comparison returns KEEP_CHAMPION, and the serving file is
    untouched. The lineage write is wrapped separately so bookkeeping cannot flip a verdict.
    """
    import hashlib
    from src.model import predict_proba as model_predict_proba

    mt = model_type
    now = now or reg.utcnow()
    batch = batch or reg.batch_id(now)
    extra = {"retrain_batch_id": batch}
    if mt in reg.FROZEN_MODEL_TYPES:
        # A frozen model type (owner 2026-10-28) is never touched by a retrain: no comparison, no
        # bootstrap over a missing file, no manifest, no slot. Logged so the refusal is visible.
        reason = (f"FROZEN: {mt} changes only by owner_install (owner decision 2026-10-28); "
                  f"retrain candidates are not accepted")
        record_decision({"stage": "HOLDOUT", "model_type": mt, "verdict": KEEP, "promoted": False,
                         "retrain_batch_id": batch, "reason": reason,
                         "champion_version": (reg.read_pointer().get(mt) or {}).get("model_version", "")})
        return ({"verdict": KEEP, "reason": reason, "qualified": False, "possible_gain": False,
                 "tracks": {}, "evaluable_tracks": [], "holdout_n": 0, "parked": False,
                 "ll_champion": None, "ll_challenger": None, "ll_ci": None}, extra)
    gate = {"verdict": KEEP, "reason": "", "qualified": False, "possible_gain": False,
            "tracks": {}, "evaluable_tracks": [], "holdout_n": 0, "parked": False,
            "ll_champion": None, "ll_challenger": None, "ll_ci": None}
    serving_exists = Path(model_file).exists()
    bootstrap = False

    # ── 0. a missing serving file is restored or bootstrapped, explicitly ──────────────────
    if not serving_exists:
        ptr_has = bool((reg.read_pointer().get(mt) or {}).get("model_version"))
        if ptr_has and _restore_serving_file(mt):
            gate["reason"] = ("serving file was missing; the registered champion's bytes were "
                              "restored. Candidate not compared this run.")
            record_decision({"stage": "RESTORE", "model_type": mt, "verdict": KEEP,
                             "promoted": False, "retrain_batch_id": batch,
                             "champion_version": reg.read_pointer()[mt]["model_version"],
                             "reason": gate["reason"]})
        elif ptr_has:
            gate["reason"] = ("EVALUATION_FAILED: serving file missing and the registered "
                              "champion's bytes cannot be restored — NOT bootstrapping over a "
                              "known champion; fix the registry by hand")
            log.error(f"  [{mt}] {gate['reason']}")
        else:
            bootstrap = True
            gate.update(verdict=BOOTSTRAP, reason="BOOTSTRAP: no serving file and no champion "
                        "record of any kind for this model type — first model installed, logged")
            log.warning(f"  [{mt}] {gate['reason']}")
    # ── 1. the comparison ───────────────────────────────────────────────────────────────────
    elif basis != "same_holdout" or old_payload is None or test_df is None:
        gate["reason"] = (f"EVALUATION_FAILED: champion could not be scored on the candidate's "
                          f"holdout (basis={basis})")
    else:
        try:
            # build_payload == the old four-key dict for legacy results; for a v9_repaired result
            # it keeps the persisted imputer, so the candidate is scored as it would serve.
            from src.model import build_payload as _build_payload
            payload_new = _build_payload(results, target)
            p_old = np.asarray(model_predict_proba(test_df, payload=old_payload), dtype=float)
            p_new = np.asarray(model_predict_proba(test_df, payload=payload_new), dtype=float)
            y = pd.to_numeric(test_df[target], errors="coerce").to_numpy(dtype=float)
            champ_man = reg.read_manifest(mt, (reg.read_pointer().get(mt) or {}).get(
                "model_version", "")) or {}
            ev = evaluate_holdout(
                y, p_old, p_new,
                dates=test_df["date"] if "date" in test_df.columns else None,
                leagues=test_df["league"] if "league" in test_df.columns else None,
                champion_cutoff=champ_man.get("training_cutoff") or None)
            gate.update(ev)
            s = _summary_track(ev)
            if s.get("champion"):
                gate.update(ll_champion=s["champion"]["log_loss"],
                            ll_challenger=s["candidate"]["log_loss"], ll_ci=s.get("ci"))
            if ev["qualified"]:
                gate.update(verdict=SHADOW, reason=f"QUALIFIED on the holdout ({ev['reason']}); "
                            f"no prospective period yet — frozen for period 2, not promoted")
            elif ev["possible_gain"]:
                gate.update(verdict=SHADOW, reason=f"not qualified but a material gain is "
                            f"inside the CI — research shadow, never promotable: {ev['reason']}")
            else:
                gate.update(verdict=KEEP, reason=ev["reason"] or "not qualified")
        except Exception as e:                                         # noqa: BLE001
            gate.update(verdict=KEEP, qualified=False,
                        reason=f"EVALUATION_FAILED: {type(e).__name__}: {str(e)[:160]}")

    # ── 2. identity, artifacts and lineage ─────────────────────────────────────────────────
    try:
        raw = reg.payload_bytes(results)          # byte-identical to the pre-gate serving format
        sha = reg.sha256_bytes(raw)
        version = reg.make_version(mt, now, sha)
        champ_v = (reg.read_pointer().get(mt) or {}).get("model_version", "")
        extra.update(challenger_version=version, champion_version=champ_v)
        dates = pd.to_datetime(valid["date"], errors="coerce") if "date" in valid.columns else None
        t0 = pd.to_datetime(test_df["date"], errors="coerce").min() if (
            test_df is not None and "date" in test_df.columns) else None
        seen = dates[dates < t0] if (dates is not None and t0 is not None) else dates
        fcols = list(results[next(iter(results))]["feature_cols"])
        s = _summary_track(gate)
        role = {SHADOW: "SHADOW", BOOTSTRAP: "CHAMPION"}.get(gate["verdict"], "REJECTED")
        manifest = {
            "model_type": mt, "model_version": version, "retrain_batch_id": batch,
            "trained_at": reg.iso(now),
            "training_start": None if seen is None or seen.empty else str(seen.min())[:10],
            "training_end": None if seen is None or seen.empty else str(seen.max())[:10],
            "training_cutoff": None if seen is None or seen.empty else str(seen.max())[:10],
            "training_rows": int(len(valid)), "target": target,
            "feature_schema_version": hashlib.sha256(
                "|".join(sorted(map(str, fcols))).encode()).hexdigest()[:12],
            "n_features": len(fcols),
            "code_commit_sha": os.getenv("GITHUB_SHA", "")[:12] or "local",
            "artifact_sha256": sha, "parent_champion_version": champ_v,
            "validation_metrics": {"gate_version": GATE_VERSION,
                                   "tracks": {t: {k: r.get(k) for k in (
                                       "n", "ll_delta", "ci", "brier_delta", "ece_delta",
                                       "passed")} for t, r in (gate.get("tracks") or {}).items()}},
            "promotion_status": gate["verdict"], "model_role": role,
        }
        if bootstrap:
            _install_champion(mt, manifest, raw, reason=gate["reason"], batch=batch,
                              source="BOOTSTRAP", at=now)
            reg.write_metrics_file(mt, results)
        else:
            reg.write_manifest(manifest)
            if gate["verdict"] == SHADOW:
                held = _slot_held_by_qualified(mt, now)
                if held and held != version:
                    gate["parked"] = False
                    gate["reason"] += f" | slot held by qualified {held}; not parked"
                else:
                    canary_ok = all((r.get("checks") or {}).get("prediction_canary", False)
                                    for t, r in (gate.get("tracks") or {}).items()
                                    if t in gate.get("evaluable_tracks", []))
                    reg.set_challenger(mt, {**manifest, "gate": {
                        "gate_version": GATE_VERSION, "qualified": bool(gate["qualified"]),
                        "evaluable_tracks": gate.get("evaluable_tracks") or [],
                        "canary_ok": bool(canary_ok), "holdout_n": gate.get("holdout_n"),
                        "holdout_ll_delta": s.get("ll_delta"),
                        "holdout_ci_lo": (s.get("ci") or [None, None])[0],
                        "holdout_ci_hi": (s.get("ci") or [None, None])[1],
                        "summary_candidate_metrics": s.get("candidate") or {}}}, raw)
                    gate["parked"] = True
        ci = gate.get("ll_ci") or [None, None]
        checks = {t: r.get("checks") for t, r in (gate.get("tracks") or {}).items()}
        record_decision({
            "stage": "BOOTSTRAP" if bootstrap else "HOLDOUT", "model_type": mt,
            "candidate_version": version, "champion_version": champ_v,
            "verdict": gate["verdict"], "promoted": bootstrap, "qualified": gate["qualified"],
            "parked": gate["parked"], "retrain_batch_id": batch,
            "evaluable_tracks": gate.get("evaluable_tracks") or [],
            "holdout_n": gate.get("holdout_n"), "holdout_ll_delta": s.get("ll_delta"),
            "holdout_ci_lo": ci[0], "holdout_ci_hi": ci[1], "prospective_status": "NONE",
            "checks": checks, "metrics": {"tracks": gate.get("tracks"),
                                          "untouched": gate.get("untouched")},
            "reason": gate["reason"][:600]})
        reg.record_decision({
            "retrain_batch_id": batch, "decided_at": reg.iso(now), "model_type": mt,
            "challenger_version": version, "champion_at_training_time": champ_v,
            "verdict": gate["verdict"], "promoted": bootstrap,
            "champion_after": version if bootstrap else champ_v,
            "training_rows": len(valid), "training_start": manifest["training_start"],
            "training_end": manifest["training_end"],
            "training_cutoff": manifest["training_cutoff"], "holdout_n": gate.get("holdout_n"),
            "ll_champion": gate.get("ll_champion"), "ll_challenger": gate.get("ll_challenger"),
            "ll_delta": s.get("ll_delta"), "ll_ci_lo": ci[0], "ll_ci_hi": ci[1],
            "brier_delta": s.get("brier_delta"), "ece_delta": s.get("ece_delta"),
            "later_half_ll_delta": "",
            "league_regressions": json.dumps(s.get("league_regressions") or []),
            "safe_to_run": (s.get("checks") or {}).get("prediction_canary"),
            "better_than_champion": bool(gate["qualified"]),
            "comparison_basis": basis, "reason": f"{GATE_VERSION}: {gate['reason']}"[:400],
            "legacy_rule_would_promote": "", "source": f"PROMOTION_GATE_{GATE_VERSION}",
        })
        cm = s.get("champion") or {}
        reg.update_high_water(mt, {
            "champion_version": version if bootstrap else champ_v,
            "current_regime": {"holdout_n": gate.get("holdout_n"), "log_loss": cm.get("log_loss"),
                               "brier": cm.get("brier"), "ece": cm.get("ece"),
                               "measured_at": reg.iso(now), "retrain_batch_id": batch},
            "last_challenger": {"version": version, "verdict": gate["verdict"],
                                "qualified": bool(gate["qualified"]), "parked": gate["parked"],
                                "ll_delta": s.get("ll_delta"), "ll_ci": gate.get("ll_ci"),
                                "gate_version": GATE_VERSION}})
    except Exception as e:                                             # noqa: BLE001
        log.error(f"  [{mt}] lineage/registry step failed ({e})")
        if bootstrap:
            try:
                ok = reg.sha256_file(Path(model_file)) == reg.sha256_bytes(
                    reg.payload_bytes(results))
            except Exception:                                          # noqa: BLE001
                ok = False
            if not ok:
                gate.update(verdict=KEEP, reason=f"EVALUATION_FAILED: bootstrap install failed "
                                                 f"({e}); nothing serving was changed")
    return gate, extra
