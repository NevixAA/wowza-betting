"""Read-only views for the Production page: today's stake plan and the canonical track registry.

  * output/stake_plan.csv      src/stake_plan.py, written by predict.yml: every current tip with the
                               approved bankroll policy applied (deployment_state, final stake, reason)
  * config/track_registry.json scripts/build_track_registry.py: production state per league x market x
                               side x track, research regime (PROTECTED / REPAIR / LEARNING), owner lock

Display only. Nothing here sizes a stake, changes a state or re-applies the policy; it shows what the
writers recorded. Both files are small (stake_plan ~45 KB, registry ~210 KB) and column-projected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAKE_PLAN = ROOT / "output" / "stake_plan.csv"
REGISTRY = ROOT / "config" / "track_registry.json"

PLAN_COLS = ["date", "fixture", "league", "track", "market", "side", "signal_tier", "odds", "deployment_state",
             "final_fraction", "stake_amount", "reason", "decision_state", "ev_lower_bound", "policy_version",
             "generated_at"]
REG_COLS = ["key", "league", "market", "side", "model_type", "research_regime", "owner_locked",
            "production_state", "track_state", "real_money", "paper_league", "sent", "evidence_status",
            "evidence_n", "independent_matchdays", "evidence_cutoff", "evidence_reason"]


def stake_plan(path: Path | None = None, today: str | None = None) -> pd.DataFrame:
    p = path or STAKE_PLAN
    if not Path(p).exists():
        return pd.DataFrame(columns=PLAN_COLS)
    try:
        d = pd.read_csv(p, usecols=lambda c: c in PLAN_COLS).reindex(columns=PLAN_COLS)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame(columns=PLAN_COLS)
    d["final_fraction"] = pd.to_numeric(d["final_fraction"], errors="coerce").fillna(0.0)
    d["date"] = d["date"].astype(str).str[:10]
    if today:
        d = d[d["date"] >= today]
    d["stake_pct"] = (d["final_fraction"] * 100).round(3)
    return d.sort_values(["date", "stake_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def plan_summary(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame(columns=["deployment_state", "tips", "staked", "stake_pct_total"])
    return (plan.assign(staked=plan["final_fraction"] > 0)
            .groupby("deployment_state", dropna=False)
            .agg(tips=("fixture", "size"), staked=("staked", "sum"), stake_pct_total=("stake_pct", "sum"))
            .reset_index().sort_values("stake_pct_total", ascending=False))


def registry(path: Path | None = None) -> tuple[dict, pd.DataFrame]:
    try:
        r = json.loads(Path(path or REGISTRY).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}, pd.DataFrame(columns=REG_COLS)
    meta = {k: r.get(k) for k in ("registry_version", "generated_for", "policy_version", "selection_engine_version",
                                  "global", "production_state_counts", "n_entries")}
    d = pd.DataFrame(r.get("entries") or []).reindex(columns=REG_COLS)
    return meta, d


def protected(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return reg
    m = reg["research_regime"].eq("PROTECTED") | reg["owner_locked"].fillna(False).astype(bool)
    return reg[m].reset_index(drop=True)
