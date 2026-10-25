"""Read-only views for the Market Intelligence page.

Sources (all small, all written by CI):
  * output/edge_engine_health.json   market_edge_v2 shadow run: two-sided price coverage + states per
                                     market x track (src/edge_engine_shadow.py, predict.yml)
  * output/clv_collection_health.json certified close coverage: a close observed <= 30 / <= 10 min
                                     before kickoff, by league and market (src/clv_collection_health.py)
  * output/direction_conflicts.csv   tips suppressed because they pointed the opposite goal direction
                                     to a tip that already stood (src/direction_guard.py)
  * output/edge_engine_shadow.csv    first-sight rows: LEGACY RAW EDGE next to the market-first decision.
                                     Column-projected; never more than the columns listed below.

Nothing here de-vigs, prices, scores or decides. Every number is what those writers recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
EDGE_HEALTH = OUT / "edge_engine_health.json"
CLV_HEALTH = OUT / "clv_collection_health.json"
CONFLICTS = OUT / "direction_conflicts.csv"
SHADOW = OUT / "edge_engine_shadow.csv"

SHADOW_COLS = ["prediction_ts", "match_date", "league", "home_team", "away_team", "track", "market", "side",
               "legacy_tier", "legacy_side", "legacy_raw_edge", "legacy_odds", "p_model_raw", "p_market_fair",
               "p_decision", "p_lower_bound", "ev_lower_bound", "decision_state", "decision_reason", "n_books",
               "clv_evidence_state"]
CONFLICT_COLS = ["generated_at", "match_date", "league", "home_team", "away_team", "suppressed_market",
                 "suppressed_side", "suppressed_tier", "stood_market", "stood_side", "reason"]
LEGACY_TIPPED = ("SNIPER", "MARKSMAN", "VALUABLE")


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def _csv(p: Path, cols: list[str]) -> pd.DataFrame:
    if not Path(p).exists():
        return pd.DataFrame(columns=cols)
    try:
        return pd.read_csv(p, usecols=lambda c: c in cols, dtype=str).reindex(columns=cols)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame(columns=cols)


def edge_health(path: Path | None = None) -> tuple[dict, pd.DataFrame]:
    h = _json(path or EDGE_HEALTH)
    rows = []
    for key, v in (h.get("coverage") or {}).items():
        market, _, track = key.partition("|")
        rows.append({"market": market, "track": track, "fixtures": v.get("fixtures"),
                     "two_sided": v.get("two_sided"), "two_sided_share": v.get("two_sided_share"),
                     "multi_book": v.get("multi_book"), "median_books": v.get("median_books"),
                     "legacy_tipped": v.get("legacy_tipped"),
                     "market_first_states": ", ".join(f"{k} {n}" for k, n in (v.get("states") or {}).items())})
    meta = {k: h.get(k) for k in ("generated_at", "selection_engine_version", "config_status", "devig_method",
                                  "shadow_only", "rows_scored", "current_run_book_quotes",
                                  "prospective_evidence_loaded")}
    return meta, pd.DataFrame(rows)


def _league_rows(d: dict, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"name": k, **{c: v.get(c) for c in cols}} for k, v in (d or {}).items()])


def clv_health(path: Path | None = None) -> dict:
    c = _json(path or CLV_HEALTH)
    cert_cols = ["eligible", "with_t30_close", "with_t10_close", "stale_close", "no_close",
                 "certified_t30_pct", "certified_t10_pct"]
    coll_cols = ["fixtures", "fixture_markets", "t60_pct", "t30_pct", "t10_pct", "median_last_obs_minutes"]
    since = c.get("shadow_since_measure_from") or {}
    allt = c.get("shadow_all_time") or {}
    coll = c.get("collection") or {}
    return {"generated_at": c.get("generated_at"), "close_rule": c.get("close_rule"),
            "measure_from": c.get("measure_from"), "check": c.get("check") or {},
            "since": {k: since.get(k) for k in cert_cols}, "all_time": {k: allt.get(k) for k in cert_cols},
            "since_by_league": _league_rows(since.get("by_league"), cert_cols),
            "since_by_market": _league_rows(since.get("by_market"), cert_cols),
            "collection": {k: coll.get(k) for k in ("window_days", "from", "fixtures", "fixture_markets",
                                                     "t60_pct", "t30_pct", "t10_pct")},
            "collection_by_track": _league_rows(coll.get("by_track"), coll_cols),
            "collection_by_market": _league_rows(coll.get("by_market"), coll_cols),
            "collection_by_league": _league_rows(coll.get("by_league"), coll_cols)}


def conflicts(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, summary by suppressed market x reason)."""
    d = _csv(path or CONFLICTS, CONFLICT_COLS)
    if d.empty:
        return d, pd.DataFrame(columns=["suppressed_market", "reason", "suppressed"])
    s = d.groupby(["suppressed_market", "reason"], dropna=False).size().rename("suppressed").reset_index()
    return d.sort_values("generated_at", ascending=False, kind="mergesort"), s.sort_values("suppressed",
                                                                                              ascending=False)


def legacy_vs_market(path: Path | None = None, today: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(upcoming first-sight rows from `today`, all-time summary per market x track).

    LEGACY = production's raw edge (p_model - 1/odds) and tier. MARKET-FIRST = market_edge_v2's fair
    price, decision probability, lower bound, EV lower bound and state. Shadow only."""
    d = _csv(path or SHADOW, SHADOW_COLS)
    if d.empty:
        return d, pd.DataFrame()
    for c in ("legacy_raw_edge", "legacy_odds", "p_model_raw", "p_market_fair", "p_decision", "p_lower_bound",
              "ev_lower_bound"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["legacy_tipped"] = d["legacy_tier"].isin(LEGACY_TIPPED)
    summ = (d.groupby(["market", "track"], dropna=False)
             .agg(rows=("market", "size"), legacy_tipped=("legacy_tipped", "sum"),
                  market_first_bet=("decision_state", lambda s: int((s == "BET").sum())),
                  market_first_paper=("decision_state", lambda s: int((s == "PAPER").sum())),
                  market_first_no_bet=("decision_state", lambda s: int((s == "NO_BET").sum())),
                  median_ev_lb=("ev_lower_bound", "median"))
             .reset_index())
    up = d[d["match_date"].astype(str) >= str(today)] if today else d
    up = up.sort_values(["match_date", "league", "home_team", "market"], kind="mergesort")
    return up.drop(columns=["legacy_tipped"]), summ
