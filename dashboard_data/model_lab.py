"""Read-only views for the Model Lab page: champions, challenger slots, promotion decisions, the H-MODEL-01
prospective comparison and the latest retrain. Small JSON/CSV files only (largest: retrain_log.json ~110 KB).

Nothing here scores, trains, loads a model artifact or decides anything. The ONE code allowed to change
which model serves is src/promotion_gate.py (team, PG-1.0) / player_model/props_gate.py (props,
PG-PROPS-1.0); this module shows what they recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "models" / "registry"
LIN = ROOT / "output" / "model_lineage"
CHAMPIONS = REG / "current_champions.json"
EXPERIMENTS = REG / "experiments.json"
GATE_CFG = ROOT / "config" / "promotion_gate.json"
DECISIONS = LIN / "promotion_decisions.csv"
PROPS_DECISIONS = LIN / "props_promotion_decisions.csv"
SHADOW_EVAL = LIN / "shadow_experiments_eval.json"
RETRAIN_LOG = ROOT / "output" / "retrain_log.json"

DECISION_COLS = ["decided_at", "gate_version", "stage", "model_type", "market", "candidate_version",
                 "champion_version", "verdict", "promoted", "holdout_n", "holdout_ll_delta", "holdout_ci_lo",
                 "holdout_ci_hi", "prospective_status", "prospective_n", "prospective_ll_delta", "reason"]


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def champions(path: Path | None = None) -> pd.DataFrame:
    d = _json(path or CHAMPIONS)
    rows = [{"model": k, "model_version": v.get("model_version"), "champion_since": v.get("champion_since"),
             "promotion_source": v.get("promotion_source"), "gate_version": v.get("gate_version") or "",
             "promotion_reason": v.get("promotion_reason")} for k, v in d.items() if isinstance(v, dict)]
    return pd.DataFrame(rows)


def challenger_slots(reg: Path | None = None) -> pd.DataFrame:
    """One row per models/registry/<type>/challenger.json (metadata only; the model file is never opened)."""
    rows = []
    for f in sorted((reg or REG).glob("*/challenger.json")):
        c = _json(f)
        vm = c.get("validation_metrics") or {}
        g = c.get("gate") or {}
        ci = vm.get("ll_ci") or [g.get("holdout_ci_lo"), g.get("holdout_ci_hi")]
        rows.append({"model": c.get("model_type", f.parent.name), "challenger_version": c.get("model_version"),
                     "status": c.get("promotion_status"), "role": c.get("model_role"),
                     "shadow_since": c.get("shadow_since"), "trained_at": c.get("trained_at"),
                     "holdout_n": (vm.get("holdout") or {}).get("n"),
                     "holdout_ll_delta": vm.get("ll_delta", g.get("holdout_ll_delta")),
                     "ll_ci": f"[{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else "",
                     "qualified_pg": g.get("qualified"), "gate_version": g.get("gate_version") or ""})
    return pd.DataFrame(rows)


def decisions(path: Path | None = None) -> pd.DataFrame:
    p = path or DECISIONS
    if not Path(p).exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(p, usecols=lambda c: c in DECISION_COLS)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame()
    return d.reindex(columns=[c for c in DECISION_COLS if c in d.columns]).sort_values(
        "decided_at", ascending=False, kind="mergesort").reset_index(drop=True) if "decided_at" in d else d


def gate_config(path: Path | None = None) -> dict:
    g = _json(path or GATE_CFG)
    return {"team": g.get("gate_version"), "props": (g.get("props") or {}).get("gate_version"),
            "holdout_min_n": (g.get("holdout") or {}).get("min_n_per_track"),
            "prospective_min_n": (g.get("prospective") or {}).get("min_n_per_track"),
            "material_ll": (g.get("holdout") or {}).get("material_ll_improvement"),
            "max_age_days": (g.get("prospective") or {}).get("max_age_days")}


def prospective(eval_path: Path | None = None, exp_path: Path | None = None) -> tuple[pd.DataFrame, str | None]:
    """H-MODEL-01 paired comparison per track, with n against the registered per-track minimum."""
    ev = _json(eval_path or SHADOW_EVAL)
    exps = {e.get("experiment_id"): e for e in (_json(exp_path or EXPERIMENTS).get("experiments") or [])}
    rows = []
    for key, r in (ev.get("results") or {}).items():
        exp_id, _, track = key.partition("|")
        e = exps.get(exp_id, {})
        need = int(e.get("n_min_per_track") or 1000)
        n = int(r.get("settled_paired") or 0)
        rows.append({"experiment": exp_id, "track": track, "status": e.get("status", ""),
                     "settled_paired_n": n, "required_n": need, "progress": min(1.0, n / need) if need else 0.0,
                     "logged_rows": r.get("rows"),
                     "delta_log_loss": r.get("delta_log_loss"), "delta_brier": r.get("delta_brier"),
                     "ll_champion": r.get("ll_champion_raw"), "ll_challenger": r.get("ll_challenger"),
                     "evaluable": n >= need,
                     "registered_at": e.get("registered_at"), "challenger_version": e.get("challenger_version")})
    return pd.DataFrame(rows), ev.get("computed_at")


def latest_retrain(path: Path | None = None) -> tuple[str | None, pd.DataFrame]:
    d = _json(path or RETRAIN_LOG)
    runs = d.get("runs") or {}
    key = d.get("latest_run") or (sorted(runs)[-1] if runs else None)
    run = runs.get(key) or {}
    rows = [{"model": m, "verdict_logged": v.get("verdict"), "promoted_logged": v.get("promoted"),
             "basis": v.get("comparison_basis"), "rows": v.get("rows"), "at": v.get("at"),
             "challenger_version": v.get("challenger_version"), "why": v.get("why")}
            for m, v in run.items() if isinstance(v, dict)]
    return key, pd.DataFrame(rows)
