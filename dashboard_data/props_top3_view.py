"""Props TOP-3 per league (owner 2026-11-22, "Like TOP-3: 3 per league, stake by proof") - dashboard data.

THE file other pages read for the props picks: output/props_top3_ledger.csv (src/props_top3.py; one row per pick,
role T1_BEST_VALUE / T2_SAFEST / T3_RISKY_VALUE, stake 0 with stake_label "No stake: props not proven yet", grading
columns result / won / flat_pnl / settle_status). Use load_picks() / latest_by_league() / role_table() /
market_table() / league_table() rather than reading the CSV by hand, so every page shows the same rows.

prediction_counts() reads output/player_ledger.csv: every prop prediction, settled (WIN / LOSS / VOID) vs waiting,
so an empty settled record is never ambiguous (0 picks yet vs picks waiting for their matches).
Pure functions apart from the loaders; nothing here writes a file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROLE_TEXT = {"T1_BEST_VALUE": "T1 best value", "T2_SAFEST": "T2 safest", "T3_RISKY_VALUE": "T3 risky value"}
ROLE_ORDER = tuple(ROLE_TEXT)
MARKET_TEXT = {"goals": "Anytime scorer", "assists": "Assist", "sot": "1+ shot on target",
               "sot2": "2+ shots on target", "cards": "Booked"}
PICKS_FILE = "props_top3_ledger.csv"
WEEKS_FILE = "props_top3_weeks.csv"
STAKE_LABEL = "No stake: props not proven yet"


def _out_dir(out_dir=None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    import config
    return Path(config.OUTPUT_DIR)


def picks_path(out_dir=None) -> Path:
    """Path of the props TOP-3 picks file (for other pages, e.g. the admin TOP-3-per-league section)."""
    return _out_dir(out_dir) / PICKS_FILE


def load_picks(out_dir=None) -> pd.DataFrame:
    """Every props TOP-3 pick, typed: odds / model_prob / value / flat_pnl numeric, kickoff UTC, role text."""
    p = picks_path(out_dir)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    for c in ("odds", "model_prob", "value", "flat_pnl", "stake"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["kickoff"] = pd.to_datetime(d.get("kickoff_utc"), errors="coerce", utc=True)
    d["Role"] = d["role"].map(ROLE_TEXT).fillna(d["role"])
    d["Market"] = d["market"].map(MARKET_TEXT).fillna(d["market"])
    d["R"] = d["result"].astype(str).str.strip().str.upper()
    return d


def load_weeks(out_dir=None) -> pd.DataFrame:
    p = _out_dir(out_dir) / WEEKS_FILE
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def latest_by_league(picks: pd.DataFrame) -> pd.DataFrame:
    """Each league's latest decided week: one row per pick, in role order (display columns)."""
    if picks is None or picks.empty:
        return pd.DataFrame()
    last = picks.groupby("league")["league_week_id"].transform("max")
    d = picks[picks["league_week_id"] == last].copy()
    d["_o"] = d["role"].map({r: i for i, r in enumerate(ROLE_ORDER)})
    d = d.sort_values(["league", "_o"])
    return pd.DataFrame({
        "League": d["league"], "Week": d["league_week_id"].str.split("|").str[-1], "Role": d["Role"],
        "Player": d["player_name"], "Team": d["team"], "Match": d["match"], "Kickoff (UTC)": d["kickoff"],
        "Market": d["Market"], "Odds": d["odds"], "Model": d["model_prob"], "Value": d["value"],
        "Value basis": d["value_basis"], "Stake": d["stake_label"].replace("", STAKE_LABEL),
        "Result": d["R"].replace("", "waiting")}).reset_index(drop=True)


def _summary(g: pd.DataFrame) -> dict:
    s = g[g["R"].isin(("WIN", "LOSS"))]
    w = int((s["R"] == "WIN").sum())
    pnl = s["flat_pnl"].dropna()
    return {"Picks": int(len(g)), "Settled": int(len(s)), "Wins": w,
            "Win %": w / len(s) * 100 if len(s) else float("nan"),
            "Profit (u)": float(pnl.sum()) if len(pnl) else float("nan"),
            "ROI %": float(pnl.mean() * 100) if len(pnl) else float("nan"),
            "Void": int((g["R"] == "VOID").sum()),
            "Waiting": int((~g["R"].isin(("WIN", "LOSS", "VOID"))).sum())}


def _table(picks: pd.DataFrame, col: str, name: str, order=None) -> pd.DataFrame:
    if picks is None or picks.empty:
        return pd.DataFrame()
    keys = list(order) if order else sorted(picks[col].dropna().unique())
    rows = [{name: k, **_summary(picks[picks[col] == k])} for k in keys if (picks[col] == k).any()]
    rows.append({name: "All props TOP-3", **_summary(picks)})
    return pd.DataFrame(rows)


def role_table(picks: pd.DataFrame) -> pd.DataFrame:
    """Settled record by role (T1 / T2 / T3), flat 1u at the pick's odds (record only - no stake)."""
    return _table(picks, "Role", "Role", [ROLE_TEXT[r] for r in ROLE_ORDER])


def market_table(picks: pd.DataFrame) -> pd.DataFrame:
    return _table(picks, "Market", "Market")


def league_table(picks: pd.DataFrame) -> pd.DataFrame:
    return _table(picks, "league", "League")


def prediction_counts(led: pd.DataFrame | None, since=None) -> dict:
    """Every prop prediction in output/player_ledger.csv (match date on/after `since` when given): total, settled
    (WIN / LOSS), void, waiting (no result yet)."""
    if led is None or led.empty or "result" not in led.columns:
        return {"predictions": 0, "settled": 0, "void": 0, "waiting": 0}
    d = led
    if since is not None and "match_date" in d.columns:
        d = d[pd.to_datetime(d["match_date"], errors="coerce") >= pd.Timestamp(since)]
    r = d["result"].astype(str).str.strip().str.upper()
    return {"predictions": int(len(d)), "settled": int(r.isin(("WIN", "LOSS")).sum()),
            "void": int((r == "VOID").sum()), "waiting": int((~r.isin(("WIN", "LOSS", "VOID"))).sum())}
