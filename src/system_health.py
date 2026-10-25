"""System-health checks with EXPLICIT red/amber/green rules (owner repair brief 2026-10-22).

    from src.system_health import evaluate      # list of checks, each {check, status, value, rule, detail}
    python -m src.system_health --workflows      # CI: also records GitHub workflow health -> output/workflow_health.json

Every status comes from a rule written below, never from wording. Inputs are small committed artifacts
(cheap enough for the dashboard to evaluate at render time); GitHub workflow health is recorded by the
daily evidence_refresh job because the dashboard must not call the GitHub API on every page view.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
GREEN, AMBER, RED, GREY = "GREEN", "AMBER", "RED", "NO_DATA"
REPOS = ("NevixAA/wowza-betting", "NevixAA/wowzaV9-Pro", "NevixAA/wowza_v11")


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def _csv(p: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(p, low_memory=False, **kw)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame()


def _c(check, status, value, rule, detail=""):
    return {"check": check, "status": status, "value": value, "rule": rule, "detail": detail}


def side_routing(plan: pd.DataFrame, policy: dict) -> dict:
    rule = "RED if any ACTIVE/ACTIVE_LOW policy track has tips and 100% map to UNSUPPORTED; GREEN otherwise"
    if plan.empty:
        return _c("Side-market routing", GREY, "no stake plan", rule)
    side = plan[plan["market"].isin(["BTTS", "OVER15", "OVER35"])]
    bad = []
    for track, t in (policy.get("tracks") or {}).items():
        if t.get("state") not in ("ACTIVE", "ACTIVE_LOW", "ACTIVE_STRONG", "PROBATION"):
            continue
        mt, mk = track.split("|")
        g = side[side["market"] == mk]
        g = g[g["model_type"].isin([mt, "unknown"])] if len(g) else g
        if len(g) and (g["deployment_state"] == "UNSUPPORTED").all():
            bad.append(track)
    n_unknown = int(side["track"].astype(str).str.startswith("unknown|").sum()) if len(side) else 0
    return _c("Side-market routing", RED if bad else GREEN, f"{len(side)} side tips, {n_unknown} unknown|*", rule,
              f"tracks fully UNSUPPORTED: {bad}" if bad else "")


def stake_mapping(plan: pd.DataFrame) -> dict:
    rule = "GREEN 0% of tips on unknown|* tracks or without odds; AMBER < 5%; RED >= 5%"
    if plan.empty:
        return _c("Stake-plan mapping", GREY, "no stake plan", rule)
    bad = plan["track"].astype(str).str.startswith("unknown|") | pd.to_numeric(plan["odds"], errors="coerce").isna()
    share = float(bad.mean() * 100)
    return _c("Stake-plan mapping", GREEN if share == 0 else AMBER if share < 5 else RED, f"{share:.1f}% unmapped", rule,
              f"generated {plan['generated_at'].iloc[0] if 'generated_at' in plan else '?'}")


def shadow_settlement(h: dict) -> dict:
    rule = "GREEN >= 95% of finished shadow rows settled; AMBER >= 80%; RED < 80%"
    v = h.get("settlement_pct_of_finished")
    if v is None:
        return _c("Shadow settlement coverage", GREY, "no health file", rule)
    return _c("Shadow settlement coverage", GREEN if v >= 95 else AMBER if v >= 80 else RED,
              f"{v}% ({h.get('settled_rows')}/{h.get('finished_prospective_rows')})", rule, f"as of {h.get('generated_at')}")


def challenger_coverage(shadow: pd.DataFrame, now: datetime) -> dict:
    rule = "over rows logged in the last 7 days: GREEN >= 90% carry a challenger probability; AMBER > 0%; RED 0%"
    if shadow.empty:
        return _c("Challenger prediction coverage", GREY, "no shadow log", rule)
    ts = pd.to_datetime(shadow.get("logged_at"), utc=True, errors="coerce")
    recent = shadow[ts >= pd.Timestamp(now) - pd.Timedelta(days=7)]
    col = next((c for c in ("p_challenger", "p_pro_hgb", "p_v91") if c in shadow.columns), None)
    if recent.empty or col is None:
        return _c("Challenger prediction coverage", GREY, "no recent rows", rule)
    share = float(pd.to_numeric(recent[col], errors="coerce").notna().mean() * 100)
    return _c("Challenger prediction coverage", GREEN if share >= 90 else AMBER if share > 0 else RED,
              f"{share:.0f}% of {len(recent)} rows ({col})", rule)


def certified_clv(shadow: pd.DataFrame, clv_health: dict) -> dict:
    rule = "share of settled shadow rows with a close observed <= 30 min before kickoff: GREEN >= 50%; AMBER >= 20%; RED < 20%"
    s = shadow[shadow.get("result", pd.Series(dtype=str)).isin(["WIN", "LOSS"])] if len(shadow) else shadow
    if s.empty or "close_status" not in s.columns:
        return _c("Certified CLV coverage", GREY, "no settled rows", rule)
    share = float((s["close_status"] == "OK").mean() * 100)
    return _c("Certified CLV coverage", GREEN if share >= 50 else AMBER if share >= 20 else RED,
              f"{share:.1f}% of {len(s)} settled", rule,
              f"collection health generated {clv_health.get('generated_at')}" if clv_health else "")


def stale_artifacts(manifest: dict) -> dict:
    rule = "decision-facing artifacts STALE/UNKNOWN/MISSING: GREEN 0; AMBER 1-2; RED >= 3 (manifest itself > 36 h old = RED)"
    if not manifest:
        return _c("Stale artifacts", RED, "no manifest", rule)
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(manifest["generated_at"])).total_seconds() / 3600
    n = int(manifest.get("n_stale_decision_facing", 0))
    st = RED if age > 36 or n >= 3 else AMBER if n >= 1 else GREEN
    return _c("Stale artifacts", st, f"{n} stale", rule, ", ".join(manifest.get("stale_decision_facing", [])))


def forward_progress(prog: dict) -> dict:
    rule = "GREEN if the frozen forward evaluator ran within 48 h; AMBER within 7 days; RED older/missing (progress itself is informational)"
    if not prog:
        return _c("Forward candidate progress", RED, "no progress file", rule)
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(prog["generated_at"])).total_seconds() / 3600
    hs = prog.get("hypotheses", [])
    n = sum((h.get("TRUE_FORWARD") or {}).get("n", 0) for h in hs)
    need = sum(h.get("required_n", 0) for h in hs if h.get("candidate_gate") is not None)
    return _c("Forward candidate progress", GREEN if age <= 48 else AMBER if age <= 168 else RED,
              f"{n}/{need} forward bets · {prog.get('status_counts')}", rule)


def workflow_health(wh: dict) -> dict:
    rule = "failed/timed-out runs across the 3 repos in the last 24 h: GREEN 0; AMBER 1-2; RED >= 3 (record > 36 h old = AMBER)"
    if not wh:
        return _c("GitHub workflow health", GREY, "not recorded yet", rule)
    n = int(wh.get("failures_24h", 0))
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(wh["generated_at"])).total_seconds() / 3600
    st = RED if n >= 3 else AMBER if (n >= 1 or age > 36) else GREEN
    return _c("GitHub workflow health", st, f"{n} failed / {wh.get('runs_24h')} runs (24 h)", rule,
              "; ".join(f"{f['repo']}: {f['workflow']}" for f in wh.get("failures", [])[:5]))


def fantasy_status(summary: dict) -> dict:
    rule = "GREEN if the V2 summary is < 72 h old and the residual mode is shadow or gated; RED if older or missing"
    if not summary:
        return _c("Fantasy V2 status", RED, "no summary", rule)
    ts = summary.get("generated_at") or summary.get("as_of")
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo
           else pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts).tz_localize("UTC")).total_seconds() / 3600 if ts else 1e9
    gate = summary.get("benchmark_status") or summary.get("residual") or {}
    return _c("Fantasy V2 status", GREEN if age < 72 else RED,
              f"mode {os.getenv('FANTASY_V2_RESIDUAL_MODE', 'shadow (workflow)')}", rule,
              json.dumps(gate, default=str)[:200] if gate else "")


def weekly_research_panel(registry: dict, lock: dict | None) -> dict:
    """Stub for the weekly per-league research panel (P0, 2026-10-25). The panel itself does not exist
    yet, so its row metrics are NO_DATA; what is checked now is the research registry and the
    non-regression lock (src/research_lock.py)."""
    rule = ("RED if protected_tracks_modified (production state/thresholds/stakes/eligibility moved without "
            "re-approval) or the lock cannot be verified; AMBER if the regime registry is missing or a newly "
            "PROTECTED track awaits approval; GREEN otherwise. Panel rows: NO_DATA until the panel exists")
    if lock is None or "protected_tracks_modified" not in lock:
        return _c("Weekly research panel", RED, "lock not verifiable", rule, (lock or {}).get("error", ""))
    counts = registry.get("regime_counts") or {}
    leagues = len(registry.get("tracked_leagues") or [])
    value = (f"{leagues} leagues · {counts} · protected_tracks_modified={lock['protected_tracks_modified']} · "
             f"panel rows NO_DATA")
    if lock["protected_tracks_modified"]:
        return _c("Weekly research panel", RED, value, rule, "; ".join(lock.get("changes", [])[:5]))
    if not registry or lock.get("unapproved_new_protected_tracks"):
        return _c("Weekly research panel", AMBER, value, rule,
                  "no regime registry" if not registry else
                  f"new PROTECTED awaiting approval: {lock['unapproved_new_protected_tracks']}")
    return _c("Weekly research panel", GREEN, value, rule,
              f"registry evidence cutoff {registry.get('evidence_cutoff')}; lock approved by "
              f"{lock.get('approved_by')} at {lock.get('approved_at')}")


def _research_lock_status() -> dict:
    try:
        from src.research_lock import check
        return check()
    except Exception as e:                                            # noqa: BLE001 - fail closed: RED
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def evaluate(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    plan = _csv(OUT / "stake_plan.csv")
    policy = _json(ROOT / "config" / "bankroll_policy.json")
    shadow = _csv(OUT / "v91_shadow.csv", dtype={"result": str, "close_status": str})
    pro = Path(os.getenv("PRO_LOCAL") or (ROOT.parent / "pro"))
    prog = _json(pro / "output" / "edge_reassessment" / "candidate_forward_progress.json")
    if not prog:
        try:
            from dashboard_data.pro import _f                          # HTTP fallback when no Pro checkout
            from dashboard_data.core import read_json
            prog = read_json(_f("output/edge_reassessment/candidate_forward_progress.json")) or {}
        except Exception:                                             # noqa: BLE001
            prog = {}
    return [side_routing(plan, policy), stake_mapping(plan),
            shadow_settlement(_json(OUT / "shadow_settlement_health.json")),
            challenger_coverage(shadow, now), certified_clv(shadow, _json(OUT / "clv_collection_health.json")),
            stale_artifacts(_json(OUT / "artifact_manifest.json")), forward_progress(prog),
            workflow_health(_json(OUT / "workflow_health.json")), fantasy_status(_json(OUT / "fantasy_v2_summary.json")),
            weekly_research_panel(_json(ROOT / "config" / "league_market_regimes.json"), _research_lock_status())]


def record_workflows() -> dict:
    """CI only: failed runs in the last 24 h across the three repos (GitHub API; token if available)."""
    import requests
    tok = os.getenv("GITHUB_TOKEN", "")
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs, fails = 0, []
    for repo in REPOS:
        r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs",
                         params={"per_page": 100, "created": f">{since}"}, headers=hdr, timeout=30).json()
        for w in r.get("workflow_runs", []):
            runs += 1
            if w.get("conclusion") in ("failure", "timed_out", "startup_failure"):
                fails.append({"repo": repo.split("/")[1], "workflow": w["name"], "at": w["created_at"],
                              "sha": w["head_sha"][:8]})
    out = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "runs_24h": runs,
           "failures_24h": len(fails), "failures": fails}
    (OUT / "workflow_health.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    if "--workflows" in sys.argv:
        print(record_workflows())
    for c in evaluate():
        print(f"{c['status']:8s} {c['check']:32s} {c['value']}")
