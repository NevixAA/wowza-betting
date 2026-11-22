"""Fantasy paired walk-forward: every arm scored on the SAME player-gameweeks, in time order.

    python -m player_model.fantasy_paired_eval        # reads output/fantasy_v2_oos.csv
                                                      # writes output/fantasy_v2_paired_walkforward.json

WHY THIS EXISTS (P1 Fantasy validation, 2026-11-22). `fantasy_residual.score_arms` scores each arm
on the rows that arm happens to cover. On the ACTIVE population FPL and HGB cover every row (2,631
FINAL rows on 2026-11-22) while WOWZA, LINEAR and RIDGE cover only the 968 rows where the old Wowza
projection existed (LINEAR and RIDGE fit and predict on complete rows, and `wowza_gap` is one of
their inputs). A pooled table that puts those numbers side by side compares different players.
This module scores only the rows on which EVERY compared arm has a forecast, so a difference
between two arms is a difference in forecasts, not in populations.

CHRONOLOGY. A row is accepted only if every gameweek in its `train_gws` is earlier than its own
gameweek (the walk-forward contract of `fantasy_residual.walk_forward`). A violation is reported
and the row is excluded; it is never silently scored.

WHAT IS REPORTED, per arm, pooled and per gameweek: n, MAE, RMSE, bias (forecast - actual),
Spearman rank correlation, top-10 overlap with the actual top 10, mean actual points of the
forecast top 10, and captaincy regret (best actual score in the gameweek minus the actual score of
the arm's top-forecast player, in raw FPL points; a captain doubles both, so captain-point regret
is twice this). Paired differences against FPL carry a gameweek-cluster bootstrap 90% interval.

This module DECIDES NOTHING. It never reads or writes FANTASY_V2_RESIDUAL_MODE; the residual stays
shadow-only until the owner says otherwise.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
OOS = OUT / "fantasy_v2_oos.csv"
REPORT = OUT / "fantasy_v2_paired_walkforward.json"
VERSION = "fantasy_paired_walkforward-1.0"

ARMS = ["FPL", "WOWZA", "LINEAR", "RIDGE", "HGB"]
#: arms that cover every FINAL row (HGB is NaN-native; FPL is the provider value)
FULL_COVERAGE_ARMS = ["FPL", "HGB"]
ARM_LABEL = {
    "FPL": "FPL baseline (ep_next)",
    "WOWZA": "Wowza V1 (standalone projection x P(start))",
    "LINEAR": "V2 LINEAR (FPL + OLS residual, shadow)",
    "RIDGE": "V2 RIDGE (FPL + ridge residual, shadow)",
    "HGB": "V2 HGB (FPL + gradient-boosting residual, shadow)",
}
TOP_K = 10
BOOT_N = 2000
BOOT_SEED = 20261122


def _arm_series(d: pd.DataFrame, arm: str) -> pd.Series:
    """The forecast column for an arm. FPL is `ep_next` in the OOS file (the FPL column is not
    persisted there)."""
    col = arm if arm in d.columns else ("ep_next" if arm == "FPL" else arm)
    return pd.to_numeric(d[col], errors="coerce")


def chronology_violations(oos: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where the row's training window is empty or holds a gameweek >= its own."""
    def bad(gw, tg) -> bool:
        if tg is None or (isinstance(tg, float) and math.isnan(tg)) or str(tg).strip() == "":
            return True
        try:
            gws = [int(float(x)) for x in str(tg).split(",") if str(x).strip() != ""]
        except ValueError:
            return True
        return (not gws) or max(gws) >= int(gw)
    return pd.Series([bad(g, t) for g, t in zip(oos["gw"], oos["train_gws"])], index=oos.index)


def paired_population(oos: pd.DataFrame, arms=ARMS, final_only: bool = True) -> tuple[pd.DataFrame, dict]:
    """Rows on which every arm in `arms` has a forecast and the actual is known.

    Returns (frame with one forecast column per arm + actual_points, exclusion counts)."""
    d = oos.copy()
    excl = {"rows_in": int(len(d))}
    if final_only:
        d = d[d["status"] == "FINAL"]
        excl["not_final"] = excl["rows_in"] - int(len(d))
    dup = d.duplicated(subset=["gw", "element"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated (gw, element) rows in the OOS frame")
    viol = chronology_violations(d)
    excl["chronology_violation"] = int(viol.sum())
    d = d[~viol]
    d = d.assign(actual_points=pd.to_numeric(d["actual_points"], errors="coerce"))
    excl["actual_missing"] = int(d["actual_points"].isna().sum())
    d = d[d["actual_points"].notna()]
    for arm in arms:
        d[arm] = _arm_series(d, arm)
        excl[f"missing_{arm}"] = int(d[arm].isna().sum())
    keep = d[list(arms)].notna().all(axis=1)
    excl["dropped_any_arm_missing"] = int((~keep).sum())
    d = d[keep].copy()
    excl["rows_scored"] = int(len(d))
    cols = ["gw", "element", "player_name", "team", "position", "actual_points"] + list(arms)
    return d[[c for c in cols if c in d.columns]].reset_index(drop=True), excl


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _topk(g: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    """Deterministic top-k: forecast descending, ties broken by FPL element id."""
    return g.sort_values([col, "element"], ascending=[False, True]).head(k)


def gw_metrics(g: pd.DataFrame, arm: str, k: int = TOP_K) -> dict:
    y, f = g["actual_points"].to_numpy(float), g[arm].to_numpy(float)
    e = f - y
    top_pred = _topk(g, arm, k)
    top_act = _topk(g, "actual_points", k)
    cap = top_pred.iloc[0]
    best = float(y.max())
    rec = {
        "n": int(len(g)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(np.mean(e)),
        "spearman": _spearman(f, y),
        "top10_overlap": int(len(set(top_pred["element"]) & set(top_act["element"]))),
        "top10_actual_mean": float(top_pred["actual_points"].mean()),
        "captain_element": int(cap["element"]),
        "captain_name": str(cap.get("player_name", "")),
        "captain_forecast": float(cap[arm]),
        "captain_actual": float(cap["actual_points"]),
        "best_actual": best,
        "captain_regret": float(best - cap["actual_points"]),
    }
    return rec


def _round(o, nd: int = 4):
    if isinstance(o, float):
        return None if math.isnan(o) else round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def _pooled(d: pd.DataFrame, arm: str, per_gw: list[dict]) -> dict:
    y, f = d["actual_points"].to_numpy(float), d[arm].to_numpy(float)
    e = f - y
    regs = [r[arm]["captain_regret"] for r in per_gw]
    sp = [r[arm]["spearman"] for r in per_gw if r[arm]["spearman"] is not None]
    return {"n": int(len(d)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)), "spearman_pooled": _spearman(f, y),
            "spearman_mean_per_gw": float(np.mean(sp)) if sp else None,
            "top10_overlap_mean": float(np.mean([r[arm]["top10_overlap"] for r in per_gw])),
            "top10_actual_mean": float(np.mean([r[arm]["top10_actual_mean"] for r in per_gw])),
            "captain_regret_total": float(np.sum(regs)),
            "captain_regret_mean": float(np.mean(regs))}


def paired_delta(d: pd.DataFrame, arm: str, base: str = "FPL", n: int = BOOT_N,
                 seed: int = BOOT_SEED) -> dict:
    """arm minus base on identical rows, with a gameweek-cluster bootstrap 90% interval.

    With few gameweeks the interval is wide by construction; it is reported, not tuned."""
    gws = sorted(d["gw"].unique())
    y = d["actual_points"].to_numpy(float)
    ea, eb = d[arm].to_numpy(float) - y, d[base].to_numpy(float) - y
    gi = d["gw"].map({g: i for i, g in enumerate(gws)}).to_numpy()
    # per-gameweek sufficient statistics: rows, sum|e|, sum e^2, sum e (arm, base)
    cnt = np.bincount(gi, minlength=len(gws)).astype(float)
    S = {k: np.bincount(gi, weights=w, minlength=len(gws)) for k, w in (
        ("abs_a", np.abs(ea)), ("abs_b", np.abs(eb)), ("sq_a", ea ** 2), ("sq_b", eb ** 2),
        ("e_a", ea), ("e_b", eb))}
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(gws), len(gws)), minlength=len(gws))
                  for _ in range(n)]).astype(float)                  # n x G resample weights
    N = W @ cnt

    def stat(kind: str, w: np.ndarray, nn):
        if kind == "mae":
            return (w @ S["abs_a"] - w @ S["abs_b"]) / nn
        if kind == "rmse":
            return np.sqrt(w @ S["sq_a"] / nn) - np.sqrt(w @ S["sq_b"] / nn)
        return np.abs(w @ S["e_a"] / nn) - np.abs(w @ S["e_b"] / nn)

    out = {}
    one = np.ones(len(gws))
    for kind in ("mae", "rmse", "bias"):
        point = float(stat(kind, one, cnt.sum()))
        lo, hi = np.quantile(stat(kind, W, N), [0.05, 0.95])
        key = "abs_bias" if kind == "bias" else kind
        out[f"{key}_minus_fpl"] = point
        out[f"{key}_ci90"] = [float(lo), float(hi)]
    return out


def score(pop: pd.DataFrame, arms=ARMS) -> dict:
    gws = sorted(int(g) for g in pop["gw"].unique())
    per_gw = []
    for gw in gws:
        g = pop[pop["gw"] == gw]
        rec = {"gw": gw, "n": int(len(g))}
        for arm in arms:
            rec[arm] = gw_metrics(g, arm)
        per_gw.append(rec)
    pooled = {arm: _pooled(pop, arm, per_gw) for arm in arms}
    vs_fpl = {}
    for arm in arms:
        if arm == "FPL":
            continue
        dlt = paired_delta(pop, arm)
        dlt["gws_mae_better_than_fpl"] = int(sum(r[arm]["mae"] < r["FPL"]["mae"] for r in per_gw))
        dlt["gws_regret_lower_than_fpl"] = int(sum(r[arm]["captain_regret"] < r["FPL"]["captain_regret"]
                                                   for r in per_gw))
        dlt["gws"] = len(per_gw)
        vs_fpl[arm] = dlt
    return {"gameweeks": gws, "rows": int(len(pop)), "pooled": pooled, "vs_fpl": vs_fpl,
            "by_gw": per_gw}


def evaluate(oos: pd.DataFrame) -> dict:
    """Primary: all five arms on identical FINAL rows. Secondary: FPL vs HGB on every FINAL row
    (both cover it), so the restriction the primary needs is visible."""
    prim, prim_x = paired_population(oos, ARMS)
    sec, sec_x = paired_population(oos, FULL_COVERAGE_ARMS)
    res = {
        "calculation_version": VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "output/fantasy_v2_oos.csv",
        "residual_mode_note": "shadow only; this report changes nothing that is served",
        "arm_labels": ARM_LABEL,
        "regret_unit": "raw FPL points (captain points = 2x)",
        "primary": {"population": "ALL_ARMS_PAIRED",
                    "definition": "FINAL gameweeks; rows where FPL, WOWZA, LINEAR, RIDGE and HGB all "
                                  "have a forecast and the actual is known; chronology checked",
                    "exclusions": prim_x, **(score(prim, ARMS) if len(prim) else {"rows": 0})},
        "secondary": {"population": "FPL_HGB_ALL_FINAL",
                      "definition": "FINAL gameweeks; every row with an FPL and an HGB forecast",
                      "exclusions": sec_x,
                      **(score(sec, FULL_COVERAGE_ARMS) if len(sec) else {"rows": 0})},
    }
    return _round(res)


def main() -> int:
    if not OOS.exists():
        print(f"no {OOS.name}; nothing to score")
        return 0
    res = evaluate(pd.read_csv(OOS, low_memory=False))
    REPORT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    p = res["primary"]
    print(f"paired walk-forward: {p.get('rows')} rows, GWs {p.get('gameweeks')}")
    for arm, m in (p.get("pooled") or {}).items():
        print(f"  {arm:6s} MAE {m['mae']:.4f} RMSE {m['rmse']:.4f} bias {m['bias']:+.4f} "
              f"rho {m['spearman_pooled']} regret {m['captain_regret_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
