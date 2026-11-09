"""Presentation for pages/11_👤_Player_Props.py (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page still loads output/player_tips.csv and output/player_ledger.csv exactly as
before (same de-duplication, same priced / unpriced split, same filters, same per-day limits). The old
page drew each day as a hand-built HTML table with its own stylesheet and hardcoded tier colours; this
module returns the SAME rows as numeric frames, rendered with st.dataframe + column_config, plus the
top-of-page cards. Props are PAPER-ONLY, permanently (invariant 2): nothing here is a bet.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import dashboard_ui as ui

TZ_NAME = "Asia/Jerusalem"
MARKET_LABEL = {
    "goals": "⚽ Anytime scorer", "goals2": "⚽ 2+ goals", "goals3": "⚽ Hat-trick",
    "assists": "🅰️ Assist",
    "sot": "🎯 1+ shot on target", "sot2": "🎯 2+ shots on target",
    "sot3": "🎯 3+ shots on target", "sot4": "🎯 4+ shots on target",
    "cards": "🟨 Booked",
}


def _tz():
    try:
        return ZoneInfo(TZ_NAME)
    except Exception:                                       # noqa: BLE001
        return ZoneInfo("UTC")


def board(d: pd.DataFrame, priced_view: bool, limit: int) -> tuple[pd.DataFrame, list[dict]]:
    """(rows shown, per-day headers) — the rows the old HTML board drew, in the same order.

    Same sort (edge desc for priced, model probability desc for unpriced, within each local day), same
    per-day `limit`, same day label. Numbers stay numbers: model / implied / EV in percent points, edge in
    percentage points, odds as odds.
    """
    tz = _tz()
    d = d.copy()
    ko = pd.to_datetime(d.get("kickoff_utc"), errors="coerce", utc=True)
    d["_local"] = ko.dt.tz_convert(tz)
    d["_day"] = d["_local"].dt.strftime("%Y-%m-%d").fillna(d["date"].dt.strftime("%Y-%m-%d"))
    if priced_view:
        d["_edge"] = (d["model_prob"] - 1 / d["market_odds"]) * 100
        d = d.sort_values(["_day", "_edge"], ascending=[True, False])
    else:
        d = d.sort_values(["_day", "model_prob"], ascending=[True, False])
    shown = d.groupby("_day", group_keys=False).head(limit)
    today = pd.Timestamp.now(tz=tz).date()
    days = []
    for day, g in shown.groupby("_day", sort=True):
        dd = pd.Timestamp(day)
        rel = {0: "Today", 1: "Tomorrow"}.get((dd.date() - today).days, "")
        total = int((d["_day"] == day).sum())
        days.append({"day": day, "title": f"{rel + ' · ' if rel else ''}{dd:%a %d %b}", "total": total,
                     "shown": len(g)})
    out = pd.DataFrame({
        "_day": shown["_day"],
        "Kickoff": shown["_local"].map(lambda t: t.strftime("%H:%M") if pd.notna(t) else "–"),
        "Player": shown["player_name"].astype(str),
        "Team": shown.get("team", pd.Series("", index=shown.index)).fillna("").astype(str),
        "Pos": shown.get("position", pd.Series(None, index=shown.index)),
        "Match": shown["match"].astype(str),
        "League": shown["league"].astype(str),
        "Market": shown["market"].map(lambda m: MARKET_LABEL.get(m, str(m))),
        "Model %": shown["model_prob"] * 100,
    })
    if priced_view:
        out["Book odds"] = shown["market_odds"]
        out["Implied %"] = 100 / shown["market_odds"]
        out["Edge (pts)"] = shown["_edge"]
        out["EV %"] = shown["ev"] * 100 if "ev" in shown.columns else None
        out["Tier"] = shown.get("tier", pd.Series("", index=shown.index)).astype(str)
    else:
        out["Fair odds"] = pd.to_numeric(shown.get("fair_odds"), errors="coerce")
    return out, days


def board_config(priced_view: bool) -> dict:
    cfg = {"Model %": ui.num_col("Model", fmt="%.0f%%", help="The model's probability."),
           "Kickoff": ui.text_col("Kickoff", help=f"Local time ({TZ_NAME})"), "Pos": ui.text_col("Pos")}
    if priced_view:
        cfg.update({"Book odds": ui.num_col("Book odds"),
                    "Implied %": ui.num_col("Implied", fmt="%.0f%%", help="1 / book odds (includes the margin)."),
                    "Edge (pts)": ui.num_col("Edge", fmt="%+.1f pts",
                                             help="Model probability minus the bookmaker's implied probability."),
                    "EV %": ui.num_col("EV", fmt="%+.0f%%"), "Tier": ui.text_col("Tier")})
    else:
        cfg["Fair odds"] = ui.num_col("Fair odds", help="1 / model probability.")
    return cfg


def render_board(d: pd.DataFrame, priced_view: bool, limit: int) -> pd.DataFrame:
    """One day heading + one sortable table per local day (kickoff in Israel time)."""
    rows, days = board(d, priced_view, limit)
    cfg = board_config(priced_view)
    for day in days:
        more = f" · top {day['shown']} shown" if day["total"] > day["shown"] else ""
        ui.eyebrow(f"{day['title']} — {day['total']} prop{'s' if day['total'] != 1 else ''}{more}")
        ui.compact_table(rows[rows["_day"] == day["day"]].drop(columns="_day"), cfg)
    return rows


def top_priced(priced: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """The k priced props where the model sits furthest above the price (paper only)."""
    if priced.empty:
        return priced
    p = priced.assign(_edge=(priced["model_prob"] - 1 / priced["market_odds"]) * 100)
    return p.sort_values("_edge", ascending=False).head(k)


def prop_card(r, container=None) -> None:
    """A priced prop: the model probability is the main number; price, implied, edge and EV beside it."""
    ko = pd.to_datetime(r.get("kickoff_utc"), errors="coerce", utc=True)
    when = ko.tz_convert(_tz()).strftime("%a %d %b %H:%M") if pd.notna(ko) else str(r.get("date"))[:10]
    ui.metric_card(MARKET_LABEL.get(r["market"], str(r["market"])), f"{r['model_prob'] * 100:.0f}%",
                   title=f"{r['player_name']} · {r.get('team', '')}",
                   supporting=[("Book odds", f"{r['market_odds']:.2f}"), ("Implied", f"{100 / r['market_odds']:.0f}%"),
                               ("Edge", f"{r['_edge']:+.1f} pts"),
                               ("EV", f"{r['ev']:+.0%}" if pd.notna(r.get("ev")) else "—"),
                               ("Kickoff", when)],
                   status=ui.status_chip("PAPER", label="PAPER · NOT A BET")
                   + (" " + ui.status_chip(str(r["tier"])) if pd.notna(r.get("tier")) and str(r.get("tier")) else ""),
                   container=container)
    (container or st).caption(f"{r['match']} · {r['league']}")


RESULTS_CONFIG = {"market_odds": ui.num_col("Odds"), "pnl": ui.num_col("P/L (u)", fmt="%+.2f")}
RAW_CONFIG = {"model_prob": ui.pct_col("Model"), "fair_odds": ui.num_col("Fair odds"),
              "market_odds": ui.num_col("Book odds"), "ev": ui.num_col("EV", fmt="%+.3f"),
              "confidence": ui.num_col("Confidence", fmt="%.3f")}


# ═══ RESULTS — settled record (rebuilt 2026-11-09, owner: "the % isn't correct, the last 2 months ════════
# were crazy results and we don't show it") ═══════════════════════════════════════════════════════════════
# The old section showed ONE all-time number over SNIPER + MARKSMAN only: 111 of its 117 bets were from
# June-July, and VALUABLE (where the recent results are) was never shown. Now: a period, every tip tier
# (VALUABLE included) with void / pending counts, a month table and a market table. Every number is
# computed from output/player_ledger.csv; P/L comes only from rows that carry a real price (pnl present).
# AVOID rows are logged for research, they are NOT tips, and are never counted in the tip record.
PERIODS = ("Last 30 days", "Last 60 days", "This month", "All time")
DEFAULT_PERIOD = "Last 60 days"
NOT_A_TIP = "AVOID"
SETTLED = ("WIN", "LOSS")


def prepare_ledger(led: pd.DataFrame) -> pd.DataFrame:
    """Normalised copy: R (result), T (tier), md (match date), pnl / odds numeric."""
    d = led.copy()
    d["R"] = d["result"].astype("string").str.strip().str.upper() if "result" in d else pd.NA
    d["T"] = d["tier"].astype("string").str.strip().str.upper() if "tier" in d else pd.NA
    d["md"] = pd.to_datetime(d["match_date"], errors="coerce") if "match_date" in d else pd.NaT
    d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce") if "pnl" in d else float("nan")
    d["odds"] = pd.to_numeric(d["market_odds"], errors="coerce") if "market_odds" in d else float("nan")
    return d


def period_start(period: str, today: pd.Timestamp) -> pd.Timestamp | None:
    """First match date in the period (inclusive); None = all time."""
    today = pd.Timestamp(today).normalize()
    if period == "Last 30 days":
        return today - pd.Timedelta(days=30)
    if period == "Last 60 days":
        return today - pd.Timedelta(days=60)
    if period == "This month":
        return today.replace(day=1)
    if period == "All time":
        return None
    raise ValueError(f"unknown period {period!r}")


def in_period(d: pd.DataFrame, period: str, today: pd.Timestamp) -> pd.DataFrame:
    start = period_start(period, today)
    return d if start is None else d[d["md"] >= start]


def _summary(g: pd.DataFrame) -> dict:
    settled = g[g["R"].isin(SETTLED)]
    priced = settled[settled["pnl"].notna()]
    n, wins = len(settled), int((settled["R"] == "WIN").sum())
    return {"Settled": n, "Wins": wins,
            "Win %": wins / n * 100 if n else float("nan"),
            "Priced": len(priced),
            "ROI %": priced["pnl"].mean() * 100 if len(priced) else float("nan"),
            "Profit (u)": priced["pnl"].sum() if len(priced) else float("nan"),
            "Avg odds": priced["odds"].mean() if len(priced) else float("nan"),
            "Void": int((g["R"] == "VOID").sum()),
            "Pending": int(g["R"].isna().sum() + (~g["R"].isin(SETTLED + ("VOID",)) & g["R"].notna()).sum()),
            "Sample": "small sample" if ui.small_sample(len(priced)) else ""}


def tips(d: pd.DataFrame) -> pd.DataFrame:
    """Tip rows only: every tier except AVOID (a logged non-tip) and rows without a tier."""
    return d[d["T"].notna() & (d["T"] != NOT_A_TIP) & (d["T"] != "NAN")]


def tier_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per tip tier present (SNIPER, MARKSMAN, VALUABLE, PAPER, ...) plus 'All tips'."""
    t = tips(d)
    order = [x for x in ("SNIPER", "MARKSMAN", "VALUABLE", "PAPER") if x in set(t["T"])]
    order += sorted(set(t["T"]) - set(order))
    rows = [{"Tier": tier, **_summary(t[t["T"] == tier])} for tier in order]
    if rows:
        rows.append({"Tier": "All tips", **_summary(t)})
    return pd.DataFrame(rows)


def month_table(d: pd.DataFrame) -> pd.DataFrame:
    """Month x tier: settled n, wins, ROI and profit (priced rows), newest month first."""
    t = tips(d)
    t = t[t["md"].notna()]
    rows = []
    for (m, tier), g in t.groupby([t["md"].dt.to_period("M"), "T"]):
        s = _summary(g)
        rows.append({"Month": str(m), "Tier": tier, "Settled": s["Settled"], "Wins": s["Wins"],
                     "ROI %": s["ROI %"], "Profit (u)": s["Profit (u)"], "Pending": s["Pending"]})
    out = pd.DataFrame(rows)
    return out.sort_values(["Month", "Tier"], ascending=[False, True]) if len(out) else out


def market_table(d: pd.DataFrame) -> pd.DataFrame:
    t = tips(d)
    rows = [{"Market": MARKET_LABEL.get(m, str(m)), **_summary(g)} for m, g in t.groupby("market")]
    out = pd.DataFrame(rows)
    return out.sort_values("Settled", ascending=False) if len(out) else out


def avoid_count(d: pd.DataFrame) -> int:
    return int((d["T"] == NOT_A_TIP).sum())


def model_era(d: pd.DataFrame, clean_from) -> pd.Series:
    """'pre-repair models' / 'clean models' per row. `clean_from` = the date the clean props models went live
    (None = not live yet, so every row is from the pre-repair models)."""
    if clean_from is None:
        return pd.Series("pre-repair models", index=d.index)
    return pd.Series(["clean models" if x else "pre-repair models"
                      for x in (d["md"] >= pd.Timestamp(clean_from)).fillna(False)], index=d.index)


TIER_CONFIG = {"Win %": ui.num_col("Win %", fmt="%.1f%%"), "ROI %": ui.num_col("ROI (1u)", fmt="%+.1f%%"),
               "Profit (u)": ui.num_col("Profit", fmt="%+.2fu"), "Avg odds": ui.num_col("Avg odds"),
               "Priced": ui.num_col("Priced", fmt="%d", help="Settled tips with a real price: ROI and profit "
                                    "use only these. Unpriced tips are counted, never given a P/L."),
               "Sample": ui.text_col("Sample", help=f"Marked below {ui.SMALL_SAMPLE_FLOOR} priced bets.")}
