"""Pro adapter — canonical evidence, validation, research. Read-only, over a sibling checkout."""
from __future__ import annotations

import pandas as pd

from dashboard_data.core import (Freshness, freshness_from_timestamp, newest_date,
                                 read_csv, read_json, reachable, remote_listing,
                                 repo_path, resolve_file)


def available() -> bool:
    """Reachable at all — a local checkout OR the public repo over HTTP.

    The dashboard deploys from the v9 repo alone, so ../wowzaV9-Pro exists on a developer
    machine and nowhere else. Reading it over HTTP mirrors what Pro already does to read v9.
    """
    return reachable("pro")


def _f(rel: str):
    """Resolve one file: local checkout if present, else fetched from the public repo."""
    return resolve_file("pro", rel)


def absent_reason() -> str:
    return ("Pro could not be reached — no local checkout at ../wowzaV9-Pro or ../v10, and the "
            "public repo did not respond. The dashboard shows what it can reach rather than "
            "failing.")


def system_contract() -> dict:
    """The governance contract, if Pro is present. Its SHAs say what it was verified against."""
    return read_json(_f("registry/WOWZA_SYSTEM_CONTRACT.json"))


def _season_names() -> list[str]:
    """Season store names, newest first. The season is in the DIRECTORY NAME, not a mtime.

    Works locally and remotely: raw HTTP cannot list a directory, so the remote path goes
    through the GitHub contents API.
    """
    return sorted((n for n in remote_listing("pro", "data") if n.startswith("season_")),
                  reverse=True)


def canonical_stores() -> dict:
    """The canonical season store — Pro's evidence layer.

    Each store is a DATE-PARTITIONED DIRECTORY (`fixtures/dt=2026-09-27/...`), not a single
    parquet. That is deliberate: it is how the estate stays under GitHub's 100 MB per-file wall
    without splitting a table by hand. It also means the newest partition NAME is the data's own
    date, which is a content-derived age -- exactly what is wanted, and what a file mtime is not.
    """
    seasons = _season_names()
    if not seasons:
        return {"available": False,
                "why": absent_reason() if not reachable("pro")
                       else "Pro is reachable but has no data/season_* store"}
    latest = seasons[0]
    stores: dict[str, dict] = {}
    # Listing every store's partitions costs one API call per store when remote, so the store
    # list is walked once and each store enumerated once -- never per partition.
    for store in sorted(remote_listing("pro", f"data/{latest}")):
        parts = [n[3:] for n in remote_listing("pro", f"data/{latest}/{store}")
                 if n.startswith("dt=")]
        if not parts:
            stores[store] = {"partitioned": False, "files": None}
            continue
        parts.sort()
        stores[store] = {
            "partitioned": True, "partitions": len(parts),
            "first_date": parts[0], "last_date": parts[-1],
            "freshness": freshness_from_timestamp(parts[-1] + "T23:59:59Z", "daily",
                                                  f"{store}/dt=*")}
    return {"available": True, "season": latest,
            "seasons_on_disk": seasons,
            "stores": stores}


def health_artifacts() -> dict:
    """Pro's own health files — the canary among them."""
    if not reachable("pro"):
        return {"available": False, "why": absent_reason()}
    res = {"available": True, "files": {}}
    for f in ("collect_health.json", "predict_health.json", "scheduler_health.json",
              "ml_learning_health.json", "combo_import_health.json"):
        j = read_json(_f(f"output/{f}"))
        if j:
            ts = j.get("generated_at") or j.get("asof") or j.get("timestamp")
            res["files"][f] = {"status": j.get("status") or j.get("state"),
                               "freshness": freshness_from_timestamp(ts, "daily", f)}
    return res


def shadow_learning() -> dict:
    """The walk-forward evidence: does retraining beat a frozen model?"""
    if not reachable("pro"):
        return {"available": False, "why": absent_reason()}
    d = read_csv(_f("output/shadow_learning/player_walkforward_performance.csv"))
    if d.empty:
        return {"available": False, "why": "no player walk-forward results in Pro"}
    g = d.groupby("variant")[[c for c in ("log_loss", "auc", "pr_auc") if c in d.columns]].mean()
    return {"available": True, "months": int(d["month"].nunique()) if "month" in d else None,
            "variants": g.round(5).to_dict("index"),
            "best_by_logloss": g["log_loss"].idxmin() if "log_loss" in g else None}


def research_validity() -> list[dict]:
    """Which conclusions are CURRENT and which are INVALIDATED — so dead numbers stop circulating."""
    c = system_contract()
    return c.get("research_validity_registry", []) or []


# ── League scout (Pro src/scout) ─────────────────────────────────────────────────────────────
def scout_status() -> dict:
    """Every watched league with its stage. Small JSON written by the scout report; the
    dashboard never reads the scout's parquet partitions.

    Before the first scout run the status file does not exist yet, so the watch list itself is
    shown with every league NOT_STARTED — the page is never blank and never pretends to data.
    """
    if not reachable("pro"):
        return {"available": False, "why": absent_reason()}
    j = read_json(_f("output/scout/league_status.json"))
    if j.get("leagues"):
        j["available"] = True
        j["freshness"] = freshness_from_timestamp(j.get("generated_at"), "fast", "scout report")
        return j
    reg = read_json(_f("registry/scout_leagues.json"))
    rows = [{"league_id": x["id"], "country": x["country"], "league": x["name"],
             "priority": x["priority"], "status": "NOT_STARTED", "history_seasons": 0, "results": 0,
             "fixtures_priced": 0, "upcoming_priced": 0, "model_frozen": 0,
             "settled_with_close": 0, "last_price_seen": None, "markets": {}}
            for x in reg.get("leagues", [])]
    if not rows:
        return {"available": False, "why": "the scout watch list is not in Pro yet"}
    return {"available": True, "leagues": rows, "n_leagues": len(rows),
            "status_counts": {"NOT_STARTED": len(rows)}, "freshness": None,
            "status_meaning": {"NOT_STARTED": "the scout has not completed a run yet"}}


def scout_last_run() -> dict:
    return read_json(_f("output/scout/last_run.json"))


# ── Tips Pro sends (1X2, Bet Builder) ─────────────────────────────────────────────────────────
def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _check_block(b, where: str, money: bool) -> str | None:
    """One scoreboard block ({"settled": 0} or a settled block). Returns the first contract breach.

    The P&L is checked against its OWN definition in Pro src/pipelines/tip_scoreboard.py: flat 1u,
    roi = mean(units), hit_rate = won / settled. A block that cannot reproduce itself is not shown."""
    if not isinstance(b, dict):
        return f"{where} is not an object"
    if "settled" not in b:
        return f"{where}.settled is missing"
    n = b["settled"]
    if not _is_num(n) or n < 0 or int(n) != n:
        return f"{where}.settled is not a non-negative integer"
    if n == 0:
        return None
    for k in ("won", "hit_rate") + (("units", "roi") if money else ()):
        if not _is_num(b.get(k)):
            return f"{where}.{k} missing or not a number"
    if not 0 <= b["won"] <= n:
        return f"{where}.won {b['won']} outside 0..settled {n}"
    if abs(b["hit_rate"] - b["won"] / n) > 0.0006:
        return f"{where} cannot reproduce: hit_rate {b['hit_rate']} != won/settled {b['won'] / n:.4f}"
    if money:
        if b["units"] < -n - 0.006:
            return f"{where} cannot reproduce: units {b['units']} below -settled at flat 1u"
        if abs(b["units"] - b["roi"] * n) > 0.00005 * n + 0.006:
            return f"{where} cannot reproduce: units {b['units']} != roi x settled {b['roi'] * n:.2f}"
    return None


def tip_scoreboard_contract_error(j: dict) -> str | None:
    """Why a Pro tip_scoreboard.json may NOT be rendered, or None when it honours the contract."""
    x = j.get("one_x_two", {"sent": 0})
    if not isinstance(x, dict):
        return "one_x_two is not an object"
    for name in ("headline", "paper_leagues"):
        if name in x:
            e = _check_block(x[name], f"one_x_two.{name}", money=True)
            if e:
                return e
    if not isinstance(x.get("by_pick", {}), dict):
        return "one_x_two.by_pick is not an object"
    for k, b in (x.get("by_pick") or {}).items():
        e = _check_block(b, f"one_x_two.by_pick.{k}", money=True)
        if e:
            return e
    bb = j.get("bet_builder", {"sent": 0})
    if not isinstance(bb, dict):
        return "bet_builder is not an object"
    if "headline" in bb:
        # Bet Builder has no price, so no units/roi: a calibration record only (Pro's own docstring).
        e = _check_block(bb["headline"], "bet_builder.headline", money=False)
        if e:
            return e
    return None


def tip_scoreboard() -> dict:
    """Pro's scoreboard of the tips it SENT. Cross-repo contract (Phase 2 Part 10): a payload whose shape
    changed, or whose P&L cannot reproduce itself, comes back as {"schema_error": why} with NO figures, so
    the page shows it as an error instead of P&L - and never as "no scoreboard yet"."""
    j = read_json(_f("output/tip_scoreboard.json"))
    if not j:
        return j
    err = tip_scoreboard_contract_error(j)
    fresh = freshness_from_timestamp(j.get("generated_at"), "daily", "tip scoreboard")
    if err:
        return {"schema_error": err, "generated_at": j.get("generated_at"), "freshness": fresh}
    j["freshness"] = fresh
    return j


# ── Weekly league promotion (Pro src/scout/promotion) ─────────────────────────────────────────
def promotion_review() -> dict:
    """Pro's weekly promotion review + the canonical league registry. Display only."""
    out = read_json(_f("output/scout/weekly_promotion_review.json"))
    if out:
        out["registry"] = read_json(_f("registry/wowza_league_registry.json"))
        out["freshness"] = freshness_from_timestamp(out.get("generated_at"), "weekly", "promotion review")
    return out


# ── Monthly bankroll review (Pro src/bankroll) ────────────────────────────────────────────────
def bankroll() -> dict:
    """Pro's latest recommendation and its evidence tables. Display only — never edits a policy."""
    out = {"recommendation": read_json(_f("output/bankroll/current_policy_recommendation.json"))}
    for name in ("policy_candidates", "stress_tests", "walk_forward", "track_decisions",
                 "convergence_leagues", "production_policy_path"):
        out[name] = read_csv(_f(f"output/bankroll/{name}.csv"))
    rec = out["recommendation"]
    if rec:
        out["freshness"] = freshness_from_timestamp(rec.get("generated_at", "").replace(" UTC", "Z").replace(" ", "T"),
                                                    "monthly", "bankroll review")
    return out


# ── Research index (Pro src/monitoring/research_index) ────────────────────────────────────────
def research_index() -> dict:
    """Every Pro study's LATEST verdict in one file, newest first.

    Pro rebuilds it after each study workflow, so a new finding reaches the dashboard without a
    dashboard change — adding a study means one entry in Pro's STUDIES list, nothing here.
    """
    j = read_json(_f("output/research_index.json"))
    if j:
        j["freshness"] = freshness_from_timestamp(
            str(j.get("generated_at", "")).replace(" UTC", "Z").replace(" ", "T"), "daily", "research index")
    return j


def model_lineage() -> dict:
    """Champion high-water mark evaluation (Pro src/analytics/model_lineage, weekly).

    Small CSVs only. v9's own lineage records (pointer, decisions) are read by the page from the
    local checkout — this is the part Pro computes: live results, health, the week table.
    """
    out = {"answers": read_json(_f("output/model_lineage/answers.json"))}
    for name in ("current_champions_view", "champion_health", "retrain_week_table",
                 "challenger_shadow_results", "model_churn", "champion_live_performance"):
        try:
            out[name] = read_csv(_f(f"output/model_lineage/{name}.csv"))
        except Exception:                                             # noqa: BLE001
            out[name] = pd.DataFrame()
    return out


def price_trader() -> dict:
    """Price Trader research lab (Pro src/price_trader, daily). Small files only."""
    out = {"verdict": read_json(_f("output/price_trader/v2/verdict_v2.json")),
           "health": read_json(_f("output/price_trader/price_trader_health.json")),
           "integrity": read_json(_f("output/research_integrity.json"))}
    for name in ("horizon_rule_results", "controls", "placebo_by_league"):
        try:
            out[name] = read_csv(_f(f"output/price_trader/v2/{name}.csv"))
        except Exception:                                             # noqa: BLE001
            out[name] = pd.DataFrame()
    return out


def valuable() -> dict:
    """VALUABLE deep dive (Pro src/studies/valuable, 2026-10-20) + its prospective shadow. Small files only."""
    base = "output/valuable_deep_dive/"
    out = {"results": read_json(_f(base + "valuable_results.json")),
           "shadow": read_json(_f(base + "valuable_shadow_status.json")),
           "shadow_live": read_json(_f("output/valuable_shadow/valuable_shadow_state.json"))}
    for name in ("valuable_summary", "valuable_league_market", "valuable_fair_edge_bands", "valuable_edge_bands",
                 "valuable_first_sight_tiers"):
        try:
            out[name] = read_csv(_f(base + f"{name}.csv"))
        except Exception:                                             # noqa: BLE001 - one missing file never hides the rest
            out[name] = pd.DataFrame()
    return out


def specials() -> dict:
    """Specials research tracker (Pro src/specials, owner 2026-11-09). Research only - no tips.

    Every frame/dict is EMPTY when Pro has not written it yet (the tracker is OFF until the owner sets
    PRO_SPECIALS_ENABLED); the page renders that as an empty state, never as sample data."""
    base = "output/specials/"
    out = {"eval": read_json(_f(base + "specials_eval.json")),
           "study": read_json(_f(base + "study_summary.json")),
           "cutoffs": read_json(_f(base + "cutoffs.json"))}
    for name in ("specials_log", "specials_settled", "study_calibration_grid", "study_calibration_bands"):
        try:
            out[name] = read_csv(_f(base + f"{name}.csv"), low_memory=False)
        except Exception:                                             # noqa: BLE001 - one missing file never hides the rest
            out[name] = pd.DataFrame()
    return out


def candidate_forward() -> dict:
    """Frozen candidate forward tests, daily progress (Pro src/studies/candidate_forward.py)."""
    return read_json(_f("output/edge_reassessment/candidate_forward_progress.json")) or {}


def edge2_forward() -> dict:
    """H-EDGE2 market_edge_v2 forward family, daily progress (Pro src/studies/edge2_forward.py).
    Statuses REGISTERED / ACCRUING / ELIGIBLE_FOR_REVIEW / BLOCKED_CONFIG_CHANGED; never CONFIRMED."""
    return read_json(_f("output/edge_reassessment/edge2_forward_progress.json")) or {}


def bankroll_track_evidence() -> dict:
    """The formal review's per-track evidence (Pro src/bankroll/evidence.track_evidence, keyed with v9's
    track_key, so O/U 2.5 is per side). Read by the Bankroll page's 'since last formal review' panel."""
    return read_json(_f("output/bankroll/track_evidence.json")) or {}


def tier_rules() -> pd.DataFrame:
    """Per-track evidence: does the edge rank outcomes? (Pro bankroll review, tier_rules.csv)."""
    try:
        return read_csv(_f("output/bankroll/tier_rules.csv"))
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame()


def verdict(rel: str) -> dict:
    """One study's machine-readable verdict file, e.g. 'output/player_v3/player_v3_verdict.json'."""
    return read_json(_f(rel))


# ── Upgrade studies (Pro src/studies) ─────────────────────────────────────────────────────────
def studies() -> dict:
    out = {}
    for name in ("argentina_btts", "ou_studies", "evidence", "ou_challenger"):
        out[name] = read_json(_f(f"output/studies/{name}.json"))
    return out
