"""Presentation for pages/14_🧩_Bet_Builder.py (visual redesign 2026-10-25, stage 3).

PRESENTATION ONLY. The page reads Pro's committed combo output (bet_builder_candidates.csv,
bet_builder_settled.csv, combo_dependency_matrix.csv) exactly as before; nothing here generates, prices,
settles or filters a combo differently. This module holds the settled-record arithmetic the page already
did (so the test can pin it), the candidate cards, the dependence chart and the column formats.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import dashboard_ui as ui


def settled_summary(settled: pd.DataFrame) -> dict:
    """The page's settled-record numbers (identical arithmetic to the pre-redesign page)."""
    r = settled["combo_result"]
    won, lost = int((r == "WON").sum()), int((r == "LOST").sum())
    decided = won + lost
    odds = pd.to_numeric(settled.get("fair_combo_odds", settled.get("fair_odds")), errors="coerce")
    pnl = float(((odds - 1.0).where(r == "WON", -1.0).where(r != "VOID", 0.0)
                 .where(r != "UNKNOWN", 0.0)).sum())
    # THE EXPECTATION MUST COME FROM THE DECIDED ROWS ONLY (see the page for why).
    exp = pd.to_numeric(settled.loc[r.isin(["WON", "LOST"]), "joint_probability"],
                        errors="coerce").mean() if "joint_probability" in settled.columns else None
    return {"won": won, "lost": lost, "decided": decided, "pnl": pnl, "exp": exp,
            "void": int((r == "VOID").sum()), "unknown": int((r == "UNKNOWN").sum())}


def settled_card(s: dict, container=None) -> None:
    decided, won = s["decided"], s["won"]
    exp = s["exp"]
    ui.metric_card("Hit rate", f"{100 * won / decided:.1f}%" if decided else "—", title="Settled record",
                   help="Compare against the model's expected rate beside it, not against 50%.",
                   supporting=[("Settled", f"{decided:,}"), ("Won", f"{won:,}"),
                               ("Model expected", f"{100 * exp:.1f}%" if exp is not None and exp == exp else "—"),
                               ("P/L at fair odds", f"{s['pnl']:+.2f}u"), ("Void", f"{s['void']}")],
                   status=ui.status_chip("RESEARCH", label="RESEARCH · PAPER"), n=decided, container=container)


def candidate_card(r: pd.Series, odds_col: str, container=None) -> None:
    """One combo: the fair (our) price is the main number; the naive multiplied price beside it is the point."""
    jp = pd.to_numeric(r.get("joint_probability"), errors="coerce")
    ip = pd.to_numeric(r.get("independence_probability"), errors="coerce")
    nv = pd.to_numeric(r.get("independence_fair_odds"), errors="coerce")
    dr = pd.to_numeric(r.get("dependency_ratio"), errors="coerce")
    ui.metric_card("Fair odds (ours)", ui.fmt_odds(pd.to_numeric(r.get(odds_col), errors="coerce")),
                   title=f"{r.get('match', '')} · {r.get('league', '')}",
                   supporting=[("Joint probability", ui.fmt_prob(jp, 1) if jp == jp else "—"),
                               ("If legs were independent", ui.fmt_prob(ip, 1) if ip == ip else "—"),
                               ("Naive odds", ui.fmt_odds(nv)), ("Dependency ratio", "—" if dr != dr else f"{dr:.3f}"),
                               ("Legs", str(r.get("n_legs", "—")))],
                   status=ui.status_chip("RESEARCH", label="RESEARCH · PAPER · NO MARKET PRICE"), container=container)
    (container or st).caption(f"{r.get('legs', '')} · {str(r.get('match_date', ''))[:10]}")


SETTLED_CONFIG = {"joint_probability": ui.pct_col("Joint probability"), "fair_odds": ui.num_col("Fair odds"),
                  "combo_result": ui.text_col("Result")}
CANDIDATE_CONFIG = {"joint_probability": ui.pct_col("Joint probability"),
                    "independence_probability": ui.pct_col("If independent"),
                    "dependency_ratio": ui.num_col("Dependency ratio", fmt="%.3f"),
                    "fair_odds": ui.num_col("Fair odds"), "fair_combo_odds": ui.num_col("Fair odds"),
                    "independence_fair_odds": ui.num_col("Naive odds")}
DEP_CONFIG = {"n": st.column_config.NumberColumn("n", format="%d"), "p_joint": ui.num_col("P(joint)", fmt="%.4f"),
              "independent_joint": ui.num_col("If independent", fmt="%.4f"),
              "dependency_ratio": ui.num_col("Dependency ratio", fmt="%.4f"),
              "independence_error_pp": ui.num_col("Independence error (pp)", fmt="%+.2f")}


def dependence_frame(dep: pd.DataFrame) -> pd.DataFrame:
    """The same ten pairs the page tabulated (six most positive, four most negative ratios)."""
    a = dep[dep["segment"] == "ALL"].copy()
    a["independence_error_pp"] = (100 * (a["p_joint"] - a["independent_joint"])).round(2)
    return a, pd.concat([a.nlargest(6, "dependency_ratio"), a.nsmallest(4, "dependency_ratio")])


def dependence_chart(top: pd.DataFrame):
    d = top.assign(pair=top["market_a"].astype(str) + " + " + top["market_b"].astype(str))
    return ui.pnl_bars(d, "pair", "independence_error_pp", "n", h=26 * len(d) + 60,
                       title="Multiplying the legs vs the measured joint chance",
                       axis_title="independence error (percentage points)", value_title="error (pp)",
                       n_title="fixtures")
