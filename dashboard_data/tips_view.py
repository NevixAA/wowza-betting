"""Read-only views for the Tips page (visual redesign 2026-10-25).

The board rows are telegram_bot.tips_table.tips_rows — the SAME rows the Telegram image draws — and are
never re-selected here. This module only:

  * attaches what production recorded about each tip (output/stake_plan.csv: deployment_state, final
    stake, and the shadow market-first columns predict writes beside it: decision_state, p_market_fair,
    p_decision, ev_lower_bound, engine_version, clv_evidence_state), and the shadow engine's point EV
    and CLV sample from output/edge_engine_shadow.csv (latest evaluation per fixture x market x side);
  * turns a board into the 'All selections' table with NUMERIC columns (formatted by column_config at
    render time, so sorting works);
  * keeps the page's text helpers (pick wording, stake note) so the test can use the same ones.

Nothing here selects, sizes, settles or changes a state. Both files are small and column-projected.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAKE_PLAN = ROOT / "output" / "stake_plan.csv"
SHADOW = ROOT / "output" / "edge_engine_shadow.csv"
TZ_NAME = "Asia/Jerusalem"

TIER_ORDER = ("SNIPER", "MARKSMAN", "VALUABLE")
TIER_ICON = {"SNIPER": "🎯", "MARKSMAN": "🔫", "VALUABLE": "💎"}
STAKE = {"SNIPER": "staked · full", "MARKSMAN": "staked · ¾", "VALUABLE": "paper"}
# SIGNAL TIER != EVIDENCE RANK (P1-6, 2026-10-22): on O/U 2.5 the edge size does not rank outcomes, so the
# bankroll policy stakes every O/U tier the same and the page must not imply SNIPER is the stronger bet.
STAKE_OU = "equal stake · tier ≠ confidence"

PLAN_COLS = ["date", "fixture", "market", "side", "deployment_state", "final_fraction", "decision_state",
             "engine_version", "p_market_fair", "p_decision", "p_lower_bound", "ev_lower_bound",
             "clv_evidence_state", "track", "generated_at"]
SHADOW_COLS = ["prediction_ts", "match_date", "home_team", "away_team", "market", "side", "ev_point",
               "ev_lower_bound", "clv_track_n", "clv_evidence_state", "n_books", "selection_engine_version"]
# stake-plan market code -> edge_engine_shadow market code
SHADOW_MARKET = {"OU25": "OU25", "OVER15": "OU15", "OVER35": "OU35", "BTTS": "BTTS"}
NOT_IN_PLAN = "NOT IN PLAN"


def market_group(m: str) -> str:
    """'O/U 2.5 UNDER' -> 'O/U 2.5'."""
    return " ".join(m.split()[:-1]) if m else m


def stake_label(tier: str, market: str) -> str:
    return STAKE_OU if market_group(str(market)) == "O/U 2.5" else STAKE.get(tier, "")


def tip_text(market: str) -> str:
    """'O/U 2.5 UNDER' -> 'Under 2.5', 'BTTS YES' -> 'BTTS — Yes' (the board's wording)."""
    parts = str(market).split()
    side = parts[-1] if parts else ""
    mk_txt = " ".join(parts[:-1]).replace("O/U ", "") if parts else ""
    side_word = {"OVER": "Over", "UNDER": "Under", "YES": "Yes"}.get(side, side.title())
    return f"BTTS — {side_word}" if mk_txt == "BTTS" else f"{side_word} {mk_txt}"


def plan_key(market: str) -> tuple[str, str]:
    """Board market label -> (stake-plan market code, side)."""
    parts = str(market).split()
    side = parts[-1].upper() if parts else ""
    grp = " ".join(parts[:-1])
    code = {"O/U 2.5": "OU25", "O/U 1.5": "OVER15", "O/U 3.5": "OVER35", "BTTS": "BTTS"}.get(grp, grp.upper())
    return code, side


def load_plan(path: Path | None = None) -> pd.DataFrame:
    p = Path(path or STAKE_PLAN)
    if not p.exists():
        return pd.DataFrame(columns=PLAN_COLS)
    try:
        return pd.read_csv(p, usecols=lambda c: c in PLAN_COLS).reindex(columns=PLAN_COLS)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame(columns=PLAN_COLS)


def load_shadow(path: Path | None = None) -> pd.DataFrame:
    """Latest shadow evaluation per (match_date, home, away, market, side)."""
    p = Path(path or SHADOW)
    if not p.exists():
        return pd.DataFrame(columns=SHADOW_COLS)
    try:
        d = pd.read_csv(p, usecols=lambda c: c in SHADOW_COLS).reindex(columns=SHADOW_COLS)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame(columns=SHADOW_COLS)
    d["match_date"] = d["match_date"].astype(str).str[:10]
    return (d.sort_values("prediction_ts", kind="mergesort")
            .drop_duplicates(["match_date", "home_team", "away_team", "market", "side"], keep="last"))


def attach(board: pd.DataFrame, plan: pd.DataFrame, shadow: pd.DataFrame | None = None) -> pd.DataFrame:
    """The board with production's and the shadow engine's recorded view of each tip. Row count and
    order are unchanged (left joins on a unique key)."""
    b = board.copy()
    if b.empty:
        return b
    keys = b["market"].map(plan_key)
    b["_pm"] = [k[0] for k in keys]
    b["_ps"] = [k[1] for k in keys]
    b["_fx"] = b["date"].astype(str).str[:10] + "|" + b["home_team"].astype(str) + "|" + b["away_team"].astype(str)
    p = plan.copy() if plan is not None else pd.DataFrame(columns=PLAN_COLS)
    p["side"] = p["side"].astype(str).str.upper()
    p = p.drop_duplicates(["fixture", "market", "side"], keep="last")
    cols = ["deployment_state", "final_fraction", "decision_state", "engine_version", "p_market_fair",
            "p_decision", "p_lower_bound", "ev_lower_bound", "clv_evidence_state", "track"]
    b = b.merge(p[["fixture", "market", "side"] + cols].rename(columns={"fixture": "_fx", "market": "_pm",
                                                                         "side": "_ps"}),
                on=["_fx", "_pm", "_ps"], how="left")
    b["in_plan"] = b["deployment_state"].notna()
    b["final_fraction"] = pd.to_numeric(b["final_fraction"], errors="coerce")
    b["production_state"] = b["deployment_state"].fillna(NOT_IN_PLAN)
    if shadow is not None and len(shadow):
        s = shadow.rename(columns={"match_date": "_d", "home_team": "_h", "away_team": "_a", "market": "_sm",
                                   "side": "_ps", "ev_lower_bound": "shadow_ev_lb",
                                   "clv_evidence_state": "shadow_clv_state"})
        b["_d"] = b["date"].astype(str).str[:10]
        b["_h"], b["_a"] = b["home_team"], b["away_team"]
        b["_sm"] = b["_pm"].map(SHADOW_MARKET)
        b = b.merge(s[["_d", "_h", "_a", "_sm", "_ps", "ev_point", "shadow_ev_lb", "clv_track_n", "n_books"]],
                    on=["_d", "_h", "_a", "_sm", "_ps"], how="left")
        b = b.drop(columns=["_d", "_h", "_a", "_sm"])
    else:
        b["ev_point"] = b["clv_track_n"] = b["n_books"] = float("nan")
    for c in ("p_market_fair", "p_decision", "p_lower_bound", "ev_lower_bound", "ev_point", "clv_track_n"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    return b.drop(columns=["_fx"])


def localise(board: pd.DataFrame, tz_name: str = TZ_NAME, now=None) -> pd.DataFrame:
    """Kickoff in local time, the local day a tip is grouped under, and whether it has started —
    the board's original rules (a 00:30 kickoff sits under the day it is played here)."""
    from zoneinfo import ZoneInfo
    try:
        tz = ZoneInfo(tz_name)
    except Exception:                                                 # noqa: BLE001 — no tzdata: UTC
        tz = ZoneInfo("UTC")
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    b = board.copy()
    ko = pd.to_datetime(b["kickoff_utc"], errors="coerce", utc=True)
    b["_ko"] = ko
    b["_local"] = ko.dt.tz_convert(tz)
    b["_day"] = b["_local"].dt.strftime("%Y-%m-%d").fillna(b["date"].astype(str))
    b["started"] = ko.notna() & (ko <= now)
    b["_t"] = b["tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    return b


def order(b: pd.DataFrame, how: str) -> pd.DataFrame:
    """'Kickoff time' or 'Best first' within each local day (the board's two orders)."""
    parts = []
    for _, g in b.groupby("_day", sort=True):
        parts.append(g.sort_values(["_ko", "_t", "edge"], ascending=[True, True, False], na_position="last")
                     if how == "Kickoff time" else g.sort_values(["_t", "edge"], ascending=[True, False]))
    return pd.concat(parts) if parts else b


def selections_table(b: pd.DataFrame, research: bool = True) -> pd.DataFrame:
    """'All selections': one row per tip, numbers kept numeric (formatted by column_config)."""
    from dashboard_ui import status_meta
    def _state(s):
        icon, word, _ = status_meta(s)
        return f"{icon} {word}"
    out = pd.DataFrame({
        "Day": pd.to_datetime(b["_day"], format="%Y-%m-%d", errors="coerce"),
        "Kickoff": b["_local"].dt.tz_localize(None),
        "Started": b["started"].astype(bool),
        "Match": b["home_team"].astype(str) + " v " + b["away_team"].astype(str),
        "League": b["league"].astype(str),
        "Rating": pd.to_numeric(b.get("rating"), errors="coerce"),
        "Pick": b["market"].map(tip_text),
        "Odds": pd.to_numeric(b["odds"], errors="coerce"),
        "Edge %": pd.to_numeric(b["edge"], errors="coerce"),
        "Model %": pd.to_numeric(b.get("model_prob"), errors="coerce") * 100,
        "Drift": (b["drift"].where(b["drift"].notna() & b["drift"].astype(str).ne("nan"))
                  if "drift" in b.columns else None),
        "Production": b["production_state"].map(_state),
        "Stake %": b["final_fraction"] * 100,
        "Tier (legacy)": b["tier"].astype(str),
        "Stake note": [stake_label(t, m) for t, m in zip(b["tier"], b["market"])],
        "Model": b["model"].map(lambda m: "Standard" if str(m) == "standard" else "New-format"),
    })
    if research:
        out["Shadow"] = b["decision_state"].map(lambda s: _state(s) if isinstance(s, str) and s else "○ NOT EVALUATED")
        out["Fair prob %"] = b["p_market_fair"] * 100
        out["Decision prob %"] = b["p_decision"] * 100
        out["EV %"] = b["ev_point"] * 100
        out["EV lower bound %"] = b["ev_lower_bound"] * 100
        out["Engine"] = b["engine_version"]
    return out.reset_index(drop=True)


def shadow_summary(b: pd.DataFrame) -> dict:
    ds = b["decision_state"].where(b["decision_state"].notna(), None) if len(b) else pd.Series(dtype=object)
    ev = pd.to_numeric(b.get("ev_lower_bound"), errors="coerce") if len(b) else pd.Series(dtype=float)
    return {"evaluated": int(ds.notna().sum()), "not_evaluated": int(ds.isna().sum()),
            "by_state": ds.dropna().value_counts().to_dict(),
            "ev_lb_median": float(ev.median()) if ev.notna().any() else None,
            "ev_lb_positive": int((ev > 0).sum()),
            "engine": (b["engine_version"].dropna().iloc[0] if len(b) and b["engine_version"].notna().any() else None)}
