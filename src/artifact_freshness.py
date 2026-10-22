"""Canonical artifact-freshness manifest (P1-3, 2026-10-22) -> output/artifact_manifest.json.

    python -m src.artifact_freshness            # writes the manifest; exits 0 (it reports, it never blocks)

WHY. Decision-facing artifacts drifted apart silently: track_evidence.json said 2026-10-04 for 18 days
(its date was hard-coded), allocation_registry and v91_readiness were run once by hand and never again,
while ledgers and shadow health were current. A reader (the dashboard, an agent, the owner) could reach
different conclusions depending on which file it opened. This manifest states, per artifact, when it was
generated, how old it may be, and whether it is valid for CURRENT decisions.

RULES
* Freshness comes from a timestamp the artifact RECORDS about itself (a JSON field or a CSV column),
  never from file mtime - a CI checkout resets mtime on every run (CLAUDE.md).
* No recorded timestamp = UNKNOWN, treated as STALE (an unknown age is infinitely old).
* lifecycle OPERATIONAL -> must be younger than max_age_hours to be valid_for_decisions.
  STUDY -> a dated analysis: never "current"; valid only as "evidence as of <date>".
  HISTORICAL -> a frozen record kept for provenance; never valid for current decisions.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "artifact_manifest.json"
PRO_LOCAL = Path(os.getenv("PRO_LOCAL") or (ROOT.parent / "pro"))
CALC_VERSION = "1.0.0"

H, D = 1, 24
#: path, repo, owner, how to read its own timestamp, lifecycle, max age (hours), decision-facing,
#: superseded_by. ts = ("json", [keys]) | ("csv", column).
ARTIFACTS = [
    ("output/shadow_settlement_health.json", "v9", "update_results.yml", ("json", ["generated_at"]),
     "OPERATIONAL", 30 * H, True, None),
    ("output/stake_plan.csv", "v9", "predict.yml (pipeline -> src/stake_plan.py)", ("csv", "generated_at"),
     "OPERATIONAL", 12 * H, True, None),
    ("output/track_evidence.json", "v9", "evidence_refresh.yml (weekly)", ("json", ["generated_for"]),
     "OPERATIONAL", 8 * D, True, None),
    ("output/allocation_registry.json", "v9", "evidence_refresh.yml (weekly)", ("json", ["generated_at_utc"]),
     "OPERATIONAL", 8 * D, True, None),
    ("output/v91_readiness.json", "v9", "evidence_refresh.yml (weekly)", ("json", ["generated_at"]),
     "OPERATIONAL", 8 * D, True, None),
    ("output/clv_collection_health.json", "v9", "odds capture (P1-1)", ("json", ["generated_at"]),
     "OPERATIONAL", 30 * H, True, None),
    ("output/fantasy_v2_summary.json", "v9", "fantasy_refresh.yml", ("json", ["generated_at", "as_of"]),
     "OPERATIONAL", 3 * D, True, None),
    ("output/league_market_gate_registry.json", "v9", "scripts/gate_study.py (one-off, 2026-10-06)",
     ("json", ["generated_at_utc"]), "HISTORICAL", None, False,
     "Pro registry/candidate_forward_tests.json (frozen forward tests) + Pro output/edge_reassessment"),
    ("output/bankroll/current_policy_recommendation.json", "pro", "monthly_bankroll_policy.yml",
     ("json", ["generated_at"]), "OPERATIONAL", 35 * D, True, None),
    ("output/bankroll/track_evidence.json", "pro", "monthly_bankroll_policy.yml", ("json", ["generated_at"]),
     "OPERATIONAL", 35 * D, True, None),
    ("output/research_index.json", "pro", "pro_research_index.yml", ("json", ["generated_at"]),
     "OPERATIONAL", 3 * D, False, None),
    ("output/studies/learned_tiers/learned_tiers.json", "pro", "pro_learned_tiers.yml (weekly)",
     ("json", ["generated_at"]), "OPERATIONAL", 8 * D, True, None),
    ("output/valuable_deep_dive/valuable_results.json", "pro", "src/studies/valuable (2026-10-20)",
     ("json", ["generated_at"]), "STUDY", None, True, None),
    ("output/edge_reassessment/edge_reassessment.json", "pro", "src/studies/edge_reassessment (2026-10-20)",
     ("json", ["generated_at"]), "STUDY", None, True, None),
    ("output/scout/league_status.json", "pro", "pro_scout.yml", ("json", ["generated_at"]),
     "OPERATIONAL", 2 * D, False, None),
]


def _parse_ts(v) -> pd.Timestamp | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        t = pd.Timestamp(str(v).replace(" UTC", ""))
    except (ValueError, TypeError):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _recorded_ts(path: Path, how) -> tuple[pd.Timestamp | None, str]:
    kind, key = how
    if not path.exists():
        return None, "MISSING"
    try:
        if kind == "json":
            d = json.loads(path.read_text(encoding="utf-8"))
            for k in key:
                if isinstance(d, dict) and d.get(k):
                    return _parse_ts(d[k]), f"json:{k}"
            return None, "NO_RECORDED_TIMESTAMP"
        col = pd.read_csv(path, usecols=lambda c: c == key, low_memory=False)
        if key not in col.columns or col.empty:
            return None, "NO_RECORDED_TIMESTAMP"
        return _parse_ts(col[key].dropna().astype(str).max()), f"csv:{key}"
    except Exception as e:                                             # noqa: BLE001 - unreadable = unknown age
        return None, f"UNREADABLE ({type(e).__name__})"


def evaluate(now: datetime | None = None, pro_root: Path | None = None) -> dict:
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts
    pro = pro_root or PRO_LOCAL
    rows = []
    for rel, repo, owner, how, lifecycle, max_age, decision, superseded in ARTIFACTS:
        base = ROOT if repo == "v9" else pro
        ts, src = _recorded_ts(base / rel, how)
        age_h = round(float((now_ts - ts).total_seconds() / 3600), 1) if ts is not None else None
        if src == "MISSING":
            fresh, valid, reason = "MISSING", False, "artifact not found"
        elif lifecycle == "HISTORICAL":
            fresh, valid, reason = "HISTORICAL", False, f"frozen record; superseded by {superseded}"
        elif lifecycle == "STUDY":
            fresh, valid = "DATED_STUDY", decision
            reason = f"analysis as of {ts.date() if ts is not None else '?'} - valid only as evidence of that date"
        elif ts is None:
            fresh, valid, reason = "UNKNOWN", False, f"no recorded timestamp ({src}) - treated as stale"
        elif age_h > max_age:
            fresh, valid = "STALE", False
            reason = f"{age_h:.0f} h old > {max_age} h limit - NOT VALID FOR CURRENT DECISIONS"
        else:
            fresh, valid, reason = "FRESH", decision, ""
        rows.append({"artifact": rel, "owner_repo": {"v9": "NevixAA/wowza-betting", "pro": "NevixAA/wowzaV9-Pro"}[repo],
                     "owner": owner, "generated_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts is not None else None,
                     "timestamp_source": src, "lifecycle": lifecycle, "max_age_hours": max_age,
                     "age_hours": age_h, "freshness": fresh, "decision_facing": decision,
                     "valid_for_decisions": bool(valid), "reason": reason, "superseded_by": superseded,
                     "calculation_version": CALC_VERSION})
    df = pd.DataFrame(rows)
    stale_df = df[df["decision_facing"] & df["freshness"].isin(["STALE", "UNKNOWN", "MISSING"])]
    return {"generated_at": now_ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "calculation_version": CALC_VERSION,
            "n_artifacts": len(rows), "n_stale_decision_facing": int(len(stale_df)),
            "stale_decision_facing": stale_df["artifact"].tolist(), "artifacts": rows}


def status_of(rel: str, manifest: dict | None = None) -> dict:
    """The manifest row for one artifact (dashboard use). Unknown artifact -> UNKNOWN, never FRESH."""
    m = manifest or (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"artifacts": []})
    for r in m.get("artifacts", []):
        if r["artifact"] == rel:
            return r
    return {"artifact": rel, "freshness": "UNKNOWN", "valid_for_decisions": False,
            "reason": "not in the artifact manifest - treated as stale"}


def main() -> int:
    m = evaluate()
    OUT.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    for r in m["artifacts"]:
        print(f"{r['freshness']:12s} {str(r['age_hours']):>7s} h  {r['artifact']}  {r['reason']}")
    print(f"[freshness] {m['n_stale_decision_facing']} decision-facing artifact(s) stale/unknown/missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
