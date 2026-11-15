"""TOP-3 per league per gameweek - the dashboard's read of src/top3_league.py's outputs (VIEWER ONLY).

Reads output/top3_league_report.json (written by the predict / update_results runs) and
output/top3_league_ledger.csv. Never computes a pick: what Telegram sent is what the page shows.
Until the owner switches TOP-3 on (TOP3_LEAGUE_ENABLED at go-live) both files are absent and the page
says so.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard_data import core

V9_DIR = Path(__file__).resolve().parents[1]
REPORT = V9_DIR / "output" / "top3_league_report.json"
LEDGER = V9_DIR / "output" / "top3_league_ledger.csv"
MARKET_LABEL = {"OU25": "O/U 2.5", "BTTS": "BTTS", "OVER15": "Over 1.5", "OVER35": "Over 3.5"}


def report() -> dict:
    return core.read_json(REPORT) or {}


def league_view(rep: dict, league: str) -> dict:
    return (rep.get("leagues") or {}).get(league) or {}


def latest_picks_frame(view: dict) -> pd.DataFrame:
    rows = view.get("latest_picks") or []
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    side = d["side"].astype(str).map(lambda s: "Yes" if s == "YES" else s.title())
    return pd.DataFrame({
        "#": d["top3_rank"], "Match": d["home_team"].astype(str) + " v " + d["away_team"].astype(str),
        "Kickoff (UTC)": d["kickoff_utc"].astype(str).str[:16].str.replace("T", " "),
        "Pick": d["market"].map(MARKET_LABEL).fillna(d["market"]) + " " + side,
        "Odds": pd.to_numeric(d["odds"], errors="coerce").round(2),
        # owner 2026-11-11 "Show calibrated edge": the calibrated edge only (NaN = not calibrated), never the raw
        "Calibrated edge pp": ((pd.to_numeric(d["edge_calibrated"], errors="coerce") * 100).round(1)
                               if "edge_calibrated" in d.columns else pd.Series(float("nan"), index=d.index)),
        # owner 2026-11-11: no tier - the slot (TOP / TIER2) and the cell's status
        "Slot": d["slot"] if "slot" in d.columns else "", "Cell": d["cell_status"] if "cell_status" in d.columns else "",
        "Result": d["result"].fillna("")})


def league_rule_line(view: dict) -> str:
    """The league's OWN line (owner 2026-11-11 "Own list": league-wide cells, outside the 3), or ""."""
    r = view.get("league_rule") or {}
    if not r.get("picks"):
        return ""
    return (f"League rule (every match, outside the 3): {r['picks']} tip(s), {r['settled']} settled, "
            f"flat P/L {float(r['flat_pnl_units']):+.2f}u")


def ledger() -> pd.DataFrame:
    return core.read_csv(LEDGER)
