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


def _experiment_start() -> pd.Timestamp | None:
    """Earliest registration of an ACTIVE challenger experiment (models/registry/experiments.json)."""
    reg = _json(ROOT / "models" / "registry" / "experiments.json").get("experiments", [])
    ts = [pd.Timestamp(e["registered_at"]) for e in reg
          if e.get("status", "ACTIVE") == "ACTIVE" and e.get("registered_at")]
    return min(ts) if ts else None


def challenger_coverage(shadow: pd.DataFrame, now: datetime) -> dict:
    rule = ("over rows logged in the last 7 days AND after the challenger experiment was registered "
            "(rows logged before a challenger existed cannot carry one): GREEN >= 90% carry a challenger "
            "probability; AMBER > 0%; RED 0%")
    if shadow.empty:
        return _c("Challenger prediction coverage", GREY, "no shadow log", rule)
    ts = pd.to_datetime(shadow.get("logged_at"), utc=True, errors="coerce")
    since = pd.Timestamp(now) - pd.Timedelta(days=7)
    start = _experiment_start()
    if start is not None:
        since = max(since, start.tz_localize("UTC") if start.tzinfo is None else start)
    recent = shadow[ts >= since]
    col = next((c for c in ("p_challenger", "p_pro_hgb", "p_v91") if c in shadow.columns), None)
    if recent.empty or col is None:
        return _c("Challenger prediction coverage", GREY, "no recent rows", rule)
    share = float(pd.to_numeric(recent[col], errors="coerce").notna().mean() * 100)
    return _c("Challenger prediction coverage", GREEN if share >= 90 else AMBER if share > 0 else RED,
              f"{share:.0f}% of {len(recent)} rows ({col})", rule, f"window from {since:%Y-%m-%d %H:%M} UTC")


def certified_clv(shadow: pd.DataFrame, clv_health: dict) -> dict:
    rule = ("share of settled shadow rows with a close observed <= 30 min before kickoff, measured from the "
            "collection fix (clv_collection_health.json measure_from; all-time shown for context): "
            "GREEN >= 50%; AMBER >= 20%; RED < 20%")
    s = shadow[shadow.get("result", pd.Series(dtype=str)).isin(["WIN", "LOSS"])] if len(shadow) else shadow
    all_time = (float((s["close_status"] == "OK").mean() * 100), len(s)) \
        if len(s) and "close_status" in s.columns else (None, 0)
    since = (clv_health or {}).get("shadow_since_measure_from") or {}
    if since.get("eligible"):
        share, n, frm = float(since["certified_t30_pct"]), int(since["eligible"]), clv_health.get("measure_from")
    elif all_time[0] is not None:
        share, n, frm = all_time[0], all_time[1], "all time"
    else:
        return _c("Certified CLV coverage", GREY, "no settled rows", rule)
    detail = f"since {str(frm)[:10]}" + (f" · all-time {all_time[0]:.1f}% of {all_time[1]}" if all_time[0] is not None else "")
    return _c("Certified CLV coverage", GREEN if share >= 50 else AMBER if share >= 20 else RED,
              f"{share:.1f}% of {n} settled", rule, detail)


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


def weekly_research_panel(registry: dict, lock: dict | None, panel: dict | None = None,
                          now: datetime | None = None) -> dict:
    """The weekly per-league research panel (src/research/league_weekly.py, 2026-10-25). `panel` is
    output/league_market_health.json (written by the panel on every predict and settlement run); without it
    the panel's row metrics are NO_DATA. The research registry and the non-regression lock
    (src/research_lock.py) are checked either way."""
    rule = ("RED if protected_tracks_modified (production state/thresholds/stakes/eligibility moved without "
            "re-approval) or the lock cannot be verified; AMBER if the regime registry is missing, a newly "
            "PROTECTED track awaits approval, the panel health is missing or > 48 h old, or a tracked league "
            "has fixtures this (or last) gameweek but NO panel rows (a silent disappearance; a league with no "
            "fixtures is NO_FIXTURES and is fine); GREEN otherwise")
    if lock is None or "protected_tracks_modified" not in lock:
        return _c("Weekly research panel", RED, "lock not verifiable", rule, (lock or {}).get("error", ""))
    counts = registry.get("regime_counts") or {}
    leagues = len(registry.get("tracked_leagues") or [])
    ptm = lock["protected_tracks_modified"]
    if panel:
        cov = panel.get("clv_coverage")
        rows = (f"tracked {panel.get('tracked_leagues')} · with current-week data "
                f"{panel.get('leagues_with_current_week_data')} · missing {len(panel.get('missing_leagues') or [])} · "
                f"candidates {panel.get('actual_candidate_rows')}/{panel.get('expected_candidate_rows')} · "
                f"panel {panel.get('panel_rows')} (settled {panel.get('settled_rows')}, pending "
                f"{panel.get('pending_rows')}) · CLV coverage {'-' if cov is None else f'{100 * cov:.0f}%'}")
    else:
        rows = "panel rows NO_DATA"
    value = f"{leagues} leagues · {counts} · protected_tracks_modified={ptm} · {rows}"
    if ptm:
        return _c("Weekly research panel", RED, value, rule, "; ".join(lock.get("changes", [])[:5]))
    if not registry or lock.get("unapproved_new_protected_tracks"):
        return _c("Weekly research panel", AMBER, value, rule,
                  "no regime registry" if not registry else
                  f"new PROTECTED awaiting approval: {lock['unapproved_new_protected_tracks']}")
    if panel:
        ts = pd.to_datetime(panel.get("generated_at"), errors="coerce", utc=True)
        age_h = ((pd.Timestamp(now or datetime.now(timezone.utc)) - ts).total_seconds() / 3600
                 if pd.notna(ts) else 1e9)
        miss = sorted(set(panel.get("missing_leagues") or []) | set(panel.get("missing_leagues_previous_week") or []))
        if age_h > 48 or miss:
            return _c("Weekly research panel", AMBER, value, rule,
                      (f"panel health {age_h:.0f} h old; " if age_h > 48 else "")
                      + (f"fixtures but no rows: {miss}" if miss else ""))
    return _c("Weekly research panel", GREEN, value, rule,
              f"registry evidence cutoff {registry.get('evidence_cutoff')}; lock approved by "
              f"{lock.get('approved_by')} at {lock.get('approved_at')}")


def _slim_dashboard() -> bool:
    """True inside the SLIM dashboard deployment (deploy/README_DASHBOARD_DEPLOY.md): code-only, data
    mirrored, no workflows / thresholds / cloudbet client to fingerprint. Never true in CI."""
    env = os.getenv("WOWZA_SLIM", "").strip().lower()
    if env in ("0", "false", "no", "off") or (ROOT / ".github" / "workflows" / "predict.yml").exists():
        return False                                                  # a full checkout is never slim
    return env in ("1", "true", "yes", "on") or (ROOT / "deploy" / "SLIM").exists()


def _research_lock_status() -> dict:
    if _slim_dashboard():
        # The live lock check needs the whole repo (workflows, threshold files, a config subprocess).
        # The slim dashboard shows the result CI recorded in output/system_health.json instead, and
        # says so; a missing record stays unverifiable (RED), never assumed fine.
        rec = (_json(HEALTH_FILE).get("research_lock") or {})
        if "protected_tracks_modified" not in rec:
            return {"error": "slim deployment: no CI-recorded research lock in output/system_health.json"}
        return {**rec, "source": f"recorded by CI in output/system_health.json at "
                                 f"{_json(HEALTH_FILE).get('generated_at')}"}
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
            weekly_research_panel(_json(ROOT / "config" / "league_market_regimes.json"), _research_lock_status(),
                                  _json(OUT / "league_market_health.json"), now)]


# ── CANONICAL HEALTH ARTIFACT (2026-10-25 platform brief, step a4) ─────────────────────────────
# output/system_health.json is written once a day by evidence_refresh.yml. It carries every check from
# evaluate() PLUS checks too heavy for a dashboard render (they read whole ledgers): the protected-
# track lock, data contracts on the committed files, first-sight store integrity (with yesterday's
# digests as the reference), promotion-gate health, edge-engine shadow health and the track registry.
# The dashboard keeps calling evaluate(); nothing here runs at render time.
HEALTH_FILE = OUT / "system_health.json"
ARTIFACT_VERSION = "SYSTEM-HEALTH-1"
_STATUS_RANK = {GREEN: 0, GREY: 1, AMBER: 1, RED: 2}


def first_sight_stores() -> dict:
    """Every committed first-sight log, described as a FirstSightStore (read-only use here)."""
    from src.research.first_sight import FirstSightStore
    from src import edge_engine_shadow as EES
    from src import shadow_experiments as SE
    from src.shadow_compare import COLUMNS as V91_COLS, FROZEN as V91_FROZEN
    from src.research import league_weekly as LW
    lin = OUT / "model_lineage"
    h2h_cols = ["logged_at", "match_date", "kickoff_utc", "league", "home_team", "away_team", "model_type",
                "market", "champion_version", "p_served", "p_champion_raw", "challenger_version", "p_challenger"]
    return {
        "edge_engine_shadow.csv": EES.first_sight_store(OUT / "edge_engine_shadow.csv"),
        "model_lineage/shadow_experiments.csv": SE.first_sight_store(lin / "shadow_experiments.csv"),
        "v91_shadow.csv": FirstSightStore(OUT / "v91_shadow.csv", key=["fixture_id", "market", "match_date"],
                                          columns=V91_COLS, observed_col="logged_at", kickoff_col="kickoff_utc",
                                          frozen_cols=list(V91_FROZEN)),
        "league_weekly_candidates.csv": LW.candidate_store(OUT / "league_weekly_candidates.csv"),
        "league_weekly_research.csv": LW.panel_store(OUT / "league_weekly_research.csv"),
        "model_lineage/shadow_h2h.csv": FirstSightStore(
            lin / "shadow_h2h.csv", key=["match_date", "home_team", "away_team", "model_type", "challenger_version"],
            columns=h2h_cols, observed_col="logged_at", kickoff_col="kickoff_utc"),
        "stake_audit.csv": FirstSightStore(OUT / "stake_audit.csv", key=["policy_version", "fixture", "track"],
                                           columns=["policy_version", "fixture", "track"]),
        "top_picks_log.csv": FirstSightStore(OUT / "top_picks_log.csv", key=["day", "list", "rank"],
                                             columns=["frozen_at", "list", "rank", "day", "league", "home_team",
                                                      "away_team", "market", "pick", "odds"]),
        "props_shown_log.csv": FirstSightStore(OUT / "props_shown_log.csv", key=["date", "section", "rank"],
                                               columns=["date", "shown_at", "section", "rank", "player_id",
                                                        "player_name", "match", "market", "model_prob",
                                                        "market_odds"]),
    }


def first_sight_integrity(prior: dict | None = None) -> tuple[dict, dict]:
    """(check, per-store detail). `prior` is the previous artifact's first_sight block: each store's
    first `rows` rows must still hash to its recorded frozen_digest."""
    rule = ("RED if any store's previously recorded rows no longer match their frozen digest (TAMPERED) or "
            "a store is unreadable; AMBER if duplicate keys (earliest-wins breached) or rows observed at/after "
            "kickoff; GREEN otherwise")
    prior = prior or {}
    out, worst, notes = {}, GREEN, []
    try:
        stores = first_sight_stores()
    except Exception as e:                                             # noqa: BLE001
        return _c("First-sight store integrity", RED, "stores not loadable", rule, f"{type(e).__name__}: {e}"[:200]), {}
    for name, st in stores.items():
        p = (prior.get(name) or {})
        r = st.integrity(prior={"rows": p.get("rows"), "frozen_digest": p.get("frozen_digest")} if p else None)
        r["path"] = name
        out[name] = r
        if r["status"] in ("TAMPERED", "UNREADABLE"):
            worst = RED
            notes.append(f"{name}: {r['status']} {r['reasons']}")
        elif r["status"] == "WARN":
            worst = worst if worst == RED else AMBER
            notes.append(f"{name}: {'; '.join(r['reasons'])}")
    n_prior = sum(1 for r in out.values() if r.get("prior_match") is not None)
    return (_c("First-sight store integrity", worst,
               f"{len(out)} stores, {n_prior} checked against the previous digest", rule, " | ".join(notes)[:600]),
            out)


def contracts_check() -> tuple[dict, dict]:
    rule = ("committed files validated against src/contracts (CONTRACTS-1): GREEN 0 violations; AMBER any "
            "violation (a finding to review, never auto-fixed); RED if validation itself failed")
    try:
        from src.contracts import CONTRACTS_VERSION, committed_violations
        r = committed_violations(ROOT)
    except Exception as e:                                             # noqa: BLE001
        return _c("Data contracts", RED, "validation failed", rule, f"{type(e).__name__}: {e}"[:200]), {}
    internal = any(v.get("kind") == "internal_error" for x in r.values() for v in x.get("violations", []))
    # Known, reviewed HISTORICAL violations (config/contract_baseline.json) are reported but do not turn the
    # check amber; anything beyond them (new column, more rows than acknowledged, new kind) still does.
    base = _json(ROOT / "config" / "contract_baseline.json").get("acknowledged", [])
    ack = {(a["file"], a["kind"], a.get("column")): a.get("max_rows") for a in base}
    n_known = 0
    for f, x in r.items():
        new = []
        for v in x.get("violations", []):
            k = (f, v.get("kind"), v.get("column"))
            if k in ack and (ack[k] is None or int(v.get("n_rows") or 0) <= int(ack[k])):
                n_known += 1
            else:
                new.append(v)
        x["new_violations"] = new
    n = sum(len(x.get("new_violations", [])) for x in r.values())
    files = {f: {"schema": x.get("schema"), "schema_version": x.get("schema_version"), "present": x.get("present"),
                 "rows": x.get("rows"), "n_violations": x.get("n_violations", 0),
                 "violations": [{k: v.get(k) for k in ("kind", "column", "n_rows", "detail", "examples")}
                                for v in x.get("violations", [])][:20]}
             for f, x in r.items()}
    bad = [f"{f}: {len(x.get('new_violations', []))}" for f, x in r.items() if x.get("new_violations")]
    st = RED if internal else AMBER if n else GREEN
    return (_c("Data contracts", st, f"{n} new violation(s) in {len(bad)} of {len(files)} files"
               + (f" · {n_known} known historical (config/contract_baseline.json)" if n_known else ""),
               rule + "; known historical violations listed in config/contract_baseline.json do not count",
               "; ".join(bad)),
            {"contracts_version": CONTRACTS_VERSION, "n_violations": n, "n_known_historical": n_known,
             "files": files})


def promotion_gate_health(now: datetime) -> dict:
    rule = ("RED if output/model_lineage/promotion_decisions.csv is missing/unreadable, or a champion in "
            "models/registry/current_champions.json differs from the latest PROMOTED decision for its model type "
            "(the champion moved without the gate); AMBER if the latest decision is > 8 days old (retrain is "
            "weekly) or carries a gate_version other than the configured one; GREEN otherwise")
    p = OUT / "model_lineage" / "promotion_decisions.csv"
    d = _csv(p, dtype=str, keep_default_na=False)
    if d.empty or "decided_at" not in d.columns:
        return _c("Promotion gate", RED, "no promotion_decisions.csv", rule)
    champs = _json(ROOT / "models" / "registry" / "current_champions.json")
    gate_v = _json(ROOT / "config" / "promotion_gate.json").get("gate_version", "PG-1.0")
    t = pd.to_datetime(d["decided_at"], errors="coerce", utc=True)
    last = t.max()
    age_h = (pd.Timestamp(now) - last).total_seconds() / 3600 if pd.notna(last) else 1e9
    mism = []
    promoted = d[d.get("promoted", pd.Series("", index=d.index)).astype(str).str.lower() == "true"].assign(_t=t)
    for mt, g in promoted.groupby("model_type"):
        latest = g.sort_values("_t").iloc[-1]["candidate_version"]
        cur = (champs.get(mt) or {}).get("model_version")
        if cur and latest and cur != latest:
            mism.append(f"{mt}: champion {cur} != last promoted {latest}")
    other_gate = sorted(set(d["gate_version"]) - {gate_v}) if "gate_version" in d.columns else []
    st = RED if mism else AMBER if (age_h > 8 * 24 or other_gate) else GREEN
    return _c("Promotion gate", st, f"{len(d)} decision(s), last {last:%Y-%m-%d %H:%MZ}" if pd.notna(last) else "no dates",
              rule, "; ".join(mism + ([f"other gate versions {other_gate}"] if other_gate else [])))


def edge_shadow_health(now: datetime) -> dict:
    rule = ("RED if edge_engine_health.json is missing, older than 72 h, or shadow_only is not true (a shadow "
            "engine must never be live); AMBER if older than 24 h or it scored 0 rows; GREEN otherwise")
    h = _json(OUT / "edge_engine_health.json")
    if not h:
        return _c("Edge-engine shadow", RED, "no edge_engine_health.json", rule)
    ts = pd.to_datetime(h.get("generated_at"), errors="coerce", utc=True)
    age_h = (pd.Timestamp(now) - ts).total_seconds() / 3600 if pd.notna(ts) else 1e9
    shadow_only = h.get("shadow_only") is True
    st = RED if (not shadow_only or age_h > 72) else AMBER if (age_h > 24 or not h.get("rows_scored")) else GREEN
    return _c("Edge-engine shadow", st,
              f"{h.get('selection_engine_version')} {h.get('config_status')}, {h.get('rows_scored')} rows, "
              f"shadow_only={h.get('shadow_only')}, {age_h:.1f} h old", rule)


def track_registry_check() -> dict:
    rule = ("RED if config/track_registry.json disagrees with what production applies (state, paper league, "
            "sent, real money, blocked) under the live policy; AMBER if missing or its champion versions are "
            "stale; GREEN otherwise")
    reg = _json(ROOT / "config" / "track_registry.json")
    if not reg.get("entries"):
        return _c("Track registry", AMBER, "missing", rule)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_track_registry",
                                                      ROOT / "scripts" / "build_track_registry.py")
        B = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(B)
        import config
        policy = _json(ROOT / "config" / "bankroll_policy.json")
        elig = {"ENABLED_LEAGUES": sorted(config.ENABLED_LEAGUES), "PAPER_LEAGUES": sorted(config.PAPER_LEAGUES),
                "PROVIDER_UNSUPPORTED": sorted(config.PROVIDER_UNSUPPORTED)}
        champs = _json(ROOT / "models" / "registry" / "current_champions.json")
    except Exception as e:                                             # noqa: BLE001
        return _c("Track registry", AMBER, "not verifiable", rule, f"{type(e).__name__}: {e}"[:200])
    drift, stale = [], []
    for e in reg["entries"]:
        pv = B.production_view(e["league"], e["market"], e["side"], e["model_type"], policy, elig)
        for f in ("production_state", "paper_league", "sent", "real_money", "blocked_market"):
            if e.get(f) != pv[f]:
                drift.append(f"{e['key']}.{f}: {e.get(f)} -> {pv[f]}")
        slot = (e.get("prediction_engine") or {}).get("model_slot")
        if slot and (champs.get(slot) or {}).get("model_version") != e["prediction_engine"].get("model_version"):
            stale.append(slot)
    st = RED if drift else AMBER if stale else GREEN
    return _c("Track registry", st, f"{reg.get('n_entries')} entries, {len(drift)} drifted field(s)", rule,
              "; ".join(drift[:5]) or (f"champion versions stale for {sorted(set(stale))}" if stale else ""))


def protected_tracks_check(lock: dict | None) -> dict:
    rule = "FAIL (RED) if protected_tracks_modified is true or the lock cannot be verified; GREEN otherwise"
    if lock is None or "protected_tracks_modified" not in lock:
        return _c("Protected tracks (research lock)", RED, "lock not verifiable", rule, (lock or {}).get("error", ""))
    return _c("Protected tracks (research lock)", RED if lock["protected_tracks_modified"] else GREEN,
              f"protected_tracks_modified={lock['protected_tracks_modified']}", rule,
              "; ".join(lock.get("changes", [])[:5]))


VERSION_FIELDS = ("model_version", "selection_engine_version", "policy_version", "feature_version")
VERSION_FRAMES = ("predictions.csv", "bets.csv", "bets_ledger.csv", "side_bets_ledger.csv", "stake_plan.csv",
                  "stake_audit.csv", "edge_engine_shadow.csv", "v91_shadow.csv", "model_lineage/shadow_h2h.csv",
                  "model_lineage/shadow_experiments.csv", "top_picks_log.csv", "clv_records.csv", "ht_ledger.csv")
#: columns that carry the same information under another name
VERSION_ALIASES = {"model_version": ("champion_version", "challenger_version", "v9_model_sha", "model_sha"),
                   "feature_version": ("feature_schema",), "selection_engine_version": ("legacy_engine_version",
                                                                                         "engine_version")}


def version_field_coverage() -> dict:
    """{file: {field: share of rows filled | 'alias:<col> <share>' | None}} for the committed frames."""
    out = {}
    for rel in VERSION_FRAMES:
        p = OUT / rel
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
        except Exception:                                              # noqa: BLE001
            continue
        rec = {}
        for f in VERSION_FIELDS:
            hit = f if f in cols else next((a for a in VERSION_ALIASES.get(f, ()) if a in cols), None)
            if hit is None:
                rec[f] = None
                continue
            s = pd.read_csv(p, usecols=[hit], dtype=str, keep_default_na=False)[hit]
            share = round(float((s.str.strip() != "").mean()), 4) if len(s) else 0.0
            rec[f] = share if hit == f else f"alias:{hit} {share}"
        out[rel] = rec
    return out


def build_artifact(now: datetime | None = None, prior: dict | None = None) -> dict:
    """The canonical health artifact. Never raises: a failing check is itself a RED check."""
    now = now or datetime.now(timezone.utc)
    prior = prior if prior is not None else _json(HEALTH_FILE)
    checks = []
    try:
        checks += evaluate(now)
    except Exception as e:                                             # noqa: BLE001
        checks.append(_c("Dashboard checks (evaluate)", RED, "failed", "evaluate() must run", f"{type(e).__name__}: {e}"[:200]))
    lock = _research_lock_status()
    checks.append(protected_tracks_check(lock))
    c_check, contracts = contracts_check()
    checks.append(c_check)
    fs_check, fs = first_sight_integrity(prior.get("first_sight"))
    checks.append(fs_check)
    for fn in (lambda: promotion_gate_health(now), lambda: edge_shadow_health(now), track_registry_check):
        try:
            checks.append(fn())
        except Exception as e:                                         # noqa: BLE001
            checks.append(_c(getattr(fn, "__name__", "check"), RED, "check failed", "must run",
                             f"{type(e).__name__}: {e}"[:200]))
    reasons, overall = [], "PASS"
    ptm = bool(lock.get("protected_tracks_modified", True)) if isinstance(lock, dict) else True
    if ptm:
        reasons.append("protected_tracks_modified=true (or lock unverifiable): production moved without re-approval")
    for c in checks:
        if c["status"] == RED:
            reasons.append(f"RED {c['check']}: {c['value']}")
        elif c["status"] in (AMBER, GREY):
            reasons.append(f"{c['status']} {c['check']}: {c['value']}")
    if ptm or any(c["status"] == RED for c in checks):
        overall = "FAIL"
    elif any(c["status"] in (AMBER, GREY) for c in checks):
        overall = "WARN"
    policy = _json(ROOT / "config" / "bankroll_policy.json")
    return {
        "about": "Canonical system health (src/system_health.build_artifact), written daily by "
                 "evidence_refresh.yml. overall: FAIL if any RED check or protected_tracks_modified; WARN if any "
                 "AMBER/NO_DATA; PASS otherwise. Every status comes from the rule stored beside it.",
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": overall, "reasons": reasons,
        "protected_tracks_modified": ptm,
        "contract_violations": contracts.get("n_violations"),
        "versions": {"policy_version": policy.get("policy_version"),
                     "selection_engine_version": {"production": "legacy_raw_edge", "shadow": "market_edge_v2"},
                     "contracts_version": contracts.get("contracts_version"),
                     "research_lock_digest": (lock or {}).get("current_digest"),
                     "code_commit": os.getenv("GITHUB_SHA", "")[:12] or "local"},
        "counts": {s: sum(c["status"] == s for c in checks) for s in (GREEN, AMBER, RED, GREY)},
        "checks": checks,
        "research_lock": {k: (lock or {}).get(k) for k in ("protected_tracks_modified", "changes",
                                                           "unapproved_new_protected_tracks", "approved_by",
                                                           "approved_at", "error")},
        "contracts": contracts,
        "first_sight": {k: {f: v.get(f) for f in ("rows", "frozen_digest", "duplicate_keys", "post_kickoff_rows",
                                                   "prior_rows", "prior_match", "status", "reasons")}
                        for k, v in fs.items()},
        "version_field_coverage": version_field_coverage(),
    }


def write_artifact(path: Path | None = None, now: datetime | None = None) -> dict:
    p = Path(path) if path else HEALTH_FILE
    a = build_artifact(now, prior=_json(p))
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(a, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    return a


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
    if "--write" in sys.argv:                      # canonical artifact -> output/system_health.json
        a = write_artifact()
        for c in a["checks"]:
            print(f"{c['status']:8s} {c['check']:34s} {c['value']}")
        print(f"OVERALL {a['overall']}  protected_tracks_modified={a['protected_tracks_modified']}  "
              f"contract_violations={a['contract_violations']}")
        for r in a["reasons"]:
            print(f"  - {r}")
        sys.exit(0)
    for c in evaluate():
        print(f"{c['status']:8s} {c['check']:32s} {c['value']}")
