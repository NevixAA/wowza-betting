"""Data for pages/18_League_Research.py ("League Repair Lab"). A VIEWER: small committed files only.

Reads output/league_weekly_research.csv (column-projected; ~400 rows a week), three small JSONs written by
src/research/league_weekly.py and config/track_registry.json. Every aggregate comes from
src/research/league_weekly_eval.py - nothing is re-derived here - so the page and the panel's own
summary cannot disagree (tests/test_league_weekly.py checks it).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from src.research import league_weekly_eval as EV

ROOT = Path(__file__).resolve().parents[1]
#: the columns the page and league_weekly_eval.prepare need - nothing else is read
VIEW_COLS = ("league_week_id", "league", "model_type", "fixture_id", "home_team", "away_team", "kickoff_utc",
             "market", "side", "league_market_rank", "legacy_rank", "prediction_engine_version",
             "selection_engine_version", "p_model_raw", "p_market_fair", "p_decision", "ev_lower_bound", "odds",
             "decision_state", "price_quality_state", "legacy_tier", "research_regime", "production_state",
             "production_selected", "actually_bet", "result", "flat_pnl", "clv_pct", "clv_prob",
             "production_selected_final", "settle_status")
REGIME_ORDER = {"REPAIR": 0, "LEARNING": 1, "PROTECTED": 2}
MARKETS = ("OU25", "BTTS", "OVER15", "OVER35")


def _out() -> Path:
    return Path(config.OUTPUT_DIR)


def load_panel(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else _out() / "league_weekly_research.csv"
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=list(VIEW_COLS))
    return pd.read_csv(p, dtype=str, keep_default_na=False, usecols=lambda c: c in VIEW_COLS)


def load_json(name: str) -> dict:
    try:
        return json.loads((_out() / name).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001 - a viewer never dies
        return {}


def registry_entries(path: Path | None = None) -> list[dict]:
    try:
        reg = json.loads(Path(path or ROOT / "config" / "track_registry.json").read_text(encoding="utf-8"))
        return list(reg.get("entries", []))
    except Exception:                                                  # noqa: BLE001
        return []


def league_regime(league: str, entries: list[dict]) -> str:
    """REPAIR if any market of the league is REPAIR, else PROTECTED if any is PROTECTED, else LEARNING."""
    regs = {e.get("research_regime") for e in entries if e.get("league") == league}
    return "REPAIR" if "REPAIR" in regs else "PROTECTED" if "PROTECTED" in regs else "LEARNING"


def league_order(entries: list[dict]) -> list[tuple[str, str]]:
    """[(league, regime)] sorted REPAIR -> LEARNING -> PROTECTED, then by name."""
    leagues = sorted({e["league"] for e in entries if e.get("league")})
    return sorted(((lg, league_regime(lg, entries)) for lg in leagues), key=lambda x: (REGIME_ORDER[x[1]], x[0]))


def track_rows(league: str, entries: list[dict]) -> pd.DataFrame:
    """Per market x side: research regime, production state, real money, sent (the registry, verbatim)."""
    rows = [{"market": e["market"], "side": e["side"], "model_type": e["model_type"],
             "research_regime": e.get("research_regime"), "production_state": e.get("production_state"),
             "real_money": e.get("real_money"), "sent": e.get("sent"), "owner_locked": e.get("owner_locked")}
            for e in entries if e.get("league") == league]
    return pd.DataFrame(rows, columns=["market", "side", "model_type", "research_regime", "production_state",
                                       "real_money", "sent", "owner_locked"])


def overview(panel: pd.DataFrame, entries: list[dict], health: dict) -> pd.DataFrame:
    """One row per league: regime, observations, gameweeks, settled, CLV coverage, current-week state."""
    d = EV.prepare(panel)
    rows = []
    for lg, regime in league_order(entries):
        g = d[d["league"] == lg] if len(d) else d
        s = g[g["settled"]] if len(g) else g
        p = s[s["priced"]] if len(s) else s
        cur = ((health.get("leagues") or {}).get(lg) or {}).get("current") or {}
        rows.append({"league": lg, "regime": regime, "observations": int(len(g)),
                     "gameweeks": int(g["league_week_id"].nunique()) if len(g) else 0,
                     "settled": int(len(s)),
                     "clv_coverage": round(float(p["clv"].notna().mean()), 3) if len(p) else None,
                     "current_week": cur.get("league_week_id", ""), "current_state": cur.get("state", "")})
    return pd.DataFrame(rows, columns=["league", "regime", "observations", "gameweeks", "settled", "clv_coverage",
                                       "current_week", "current_state"])


def league_header(panel: pd.DataFrame, league: str) -> dict:
    g = panel[panel["league"] == league] if len(panel) else panel
    return {"observations": int(len(g)), "gameweeks": int(g["league_week_id"].nunique()) if len(g) else 0,
            "prediction_engines": sorted(set(g["prediction_engine_version"]) - {""}) if len(g) else [],
            "selection_engines": sorted(set(g["selection_engine_version"]) - {""}) if len(g) else []}


def metric_table(panel: pd.DataFrame, league: str) -> pd.DataFrame:
    """league_weekly_eval.league_table, this league's rows only (identical numbers)."""
    t = EV.league_table(panel)
    return t[t["league"] == league].reset_index(drop=True)


def monotonicity(panel: pd.DataFrame, league: str) -> pd.DataFrame:
    t = EV.rank_monotonicity(panel)
    return t[t["league"] == league].reset_index(drop=True)


def history_grid(panel: pd.DataFrame, league: str, market: str) -> pd.DataFrame:
    return EV.weekly_grid(panel, league, market)
