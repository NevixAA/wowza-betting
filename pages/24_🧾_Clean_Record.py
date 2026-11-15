"""Clean Record vs Before the repairs — the comparison the owner asked for (2026-11-11).

Owner: "adjust to count from 0 after all the fixing (to see the difference later)". The Clean Record counts
only tips generated and played on or after config.CLEAN_RECORD_START (2026-11-21, owner decision): the clean
retrained team models, the clean props models and the approved per-league x market bars. "Before the repairs"
is every earlier tip (old models, global bars, the retired tiers). Same metrics, side by side, never one total.

VIEWER ONLY. Every figure is recomputed here from the committed ledgers through dashboard_data.clean_record
(THE split) and today_view.load_settled (THE staked definition, src/staked.py). Before the start the Clean
Record column is 0 / no settled tips yet.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dashboard_access import require  # noqa: E402
require("pro")

import dashboard_ui as ui  # noqa: E402
from dashboard_data import clean_record as CR, today_view  # noqa: E402
from dashboard_data import top3_league as T3  # noqa: E402

st.set_page_config(page_title="Wowza | Clean Record", page_icon="🧾", layout="wide")


@st.cache_data(ttl=120)
def _data() -> dict:
    settled = today_view.load_settled()
    sp = CR.team_split(settled)
    led = pd.read_csv(config.OUTPUT_DIR / "player_ledger.csv", low_memory=False) \
        if (config.OUTPUT_DIR / "player_ledger.csv").exists() else pd.DataFrame()
    return {"team": CR.team_summaries(settled), "sp": sp, "props": CR.props_summaries(led),
            "top3": CR.top3_summaries(T3.ledger())}


D = _data()
ui.page_header("Clean Record vs Before the repairs",
               f"Counted from 0 on {CR.start_date()} (owner 2026-11-11), the first round on the clean models. "
               "The record before it stays visible for comparison, never mixed into the Clean Record.",
               eyebrow_text="Performance · clean record",
               badge=ui.status_chip("LIVE" if CR.started() else "NO_DATA",
                                    label="CLEAN RECORD LIVE" if CR.started() else f"STARTS {CR.start_date()}"))
CR.banner(st, ui)

_cfg = CR.table_config(ui)
SCOPES = [("Staked bets (a recorded policy stake > 0)", D["team"][CR.CLEAN]["staked"], D["team"][CR.BEFORE]["staked"]),
          ("All team tips, paper + staked (flat 1u)", D["team"][CR.CLEAN]["all"], D["team"][CR.BEFORE]["all"]),
          ("Player props (paper research)", D["props"][CR.CLEAN], D["props"][CR.BEFORE]),
          ("TOP-3 per league (paper track)", D["top3"][CR.CLEAN], D["top3"][CR.BEFORE])]

# ── headline: the Clean Record only ───────────────────────────────────────────────────────────────
ui.section_header(CR.clean_label(), "Tips generated and played on or after the start. Flat 1u per tip; the "
                  "ROI interval is a matchday block bootstrap (2,000 draws, 95%).")
cols = st.columns(len(SCOPES))
for col, (name, c, _b) in zip(cols, SCOPES):
    ui.metric_card(f"Clean Record units · {name.split(' (')[0]}", ui.fmt_units(c["units"]) if c["n"] else ui.fmt_units(0),
                   title=name,
                   supporting=[("Settled", f"{c['n']}"),
                               ("Hit rate", f"{100 * c['hit']:.1f}%" if c["n"] else "—"),
                               ("ROI", ui.fmt_roi(c["roi"]) if c["n"] else "—"),
                               ("ROI 95% CI", CR.fmt_ci(c["ci"]))],
                   status=ui.status_chip("PAPER", label="PAPER") if "paper" in name.lower() and "staked" not in name.lower()
                   else "", n=c["n"] or None, container=col)
if not any(c["n"] for _n, c, _b in SCOPES):
    ui.empty_state("NO_DATA", CR.empty_text())

# ── comparison: same metrics side by side ─────────────────────────────────────────────────────────
ui.section_header("Clean Record vs Before the repairs", "One table per scope: the Clean Record row and the "
                  f"pre-repair row ({config.PERFORMANCE_CUTOFF_DATE} to {CR.start_date()}), the same metrics.")
for name, c, b in SCOPES:
    ui.eyebrow(name)
    ui.compact_table(CR.comparison(c, b), _cfg)
st.caption(CR.BEFORE_NOTE + " TOP-3 starts as a named paper track on the start date, so it has no pre-repair "
           "row. A tip neither rule can place (no generation time and a match after the start) is in neither row: "
           f"team {D['team'][CR.UNPLACED]}, props {D['props'][CR.UNPLACED]}, TOP-3 {D['top3'][CR.UNPLACED]}.")

# ── Clean Record per market and per cell ──────────────────────────────────────────────────────────
ui.section_header("Clean Record by market and by league × market cell",
                  "Every settled team tip of the Clean Record, paper and staked, flat 1u. A cell is the league × "
                  "market-side unit the bars are approved in (Bars & Cells page).")
_c = D["sp"][CR.CLEAN]
if not len(_c):
    ui.empty_state("NO_DATA", CR.empty_text())
else:
    m1, m2 = st.columns(2)
    with m1:
        ui.eyebrow("By market")
        ui.compact_table(CR.group_table(_c, "_mkt", pnl="_pnl", date="_dt", key="clean|mkt")
                         .rename(columns={"_mkt": "Market"}), _cfg)
    with m2:
        ui.eyebrow("By cell")
        ui.compact_table(CR.group_table(CR.with_cell(_c), "Cell", pnl="_pnl", date="_dt", key="clean|cell"), _cfg)

with st.expander(f"{CR.before_label()} — by market, for comparison"):
    st.caption(CR.BEFORE_NOTE)
    ui.compact_table(CR.group_table(D["sp"][CR.BEFORE], "_mkt", pnl="_pnl", date="_dt", key="before|mkt")
                     .rename(columns={"_mkt": "Market"}), _cfg)

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
