"""Bars & Cells — every league × market cell with its owner-approved bar, status and stake (owner 2026-11-11).

"the best bar for every league and market". One row per league × market-side cell of the APPROVED proposal
(TP-2026-11-10-d999073a, owner 2026-11-11 "Approve, paper + small"); there is no global bar anywhere. A tip is
a tip only when its measure clears its cell's ONE qualifying bar (the retired Big / Medium / Small gap tiers are
history only). The stake is per cell (config/bankroll_policy.json cell_staking); every other cell is PAPER.

Below the registry: the Clean Record of each cell (tips generated and played on or after
config.CLEAN_RECORD_START), 0 / no settled tips yet before the start.

VIEWER ONLY: dashboard_data.cells_view reads the approval, the proposal, its evidence file and the policy.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require  # noqa: E402
require("pro")

import dashboard_ui as ui  # noqa: E402
from dashboard_data import cells_view as CV, clean_record as CR, today_view  # noqa: E402

st.set_page_config(page_title="Wowza | Bars & Cells", page_icon="📐", layout="wide")


@st.cache_data(ttl=300)
def _cells():
    data = CV.load()
    frame = CV.cells_frame(data)
    return data, frame


data, cells = _cells()
ap = data.get("approval") or {}
ui.page_header("Bars & Cells", "Every league × market cell: its own approved bar, its status and its stake. No "
               "global bar, no tiers.", eyebrow_text="Performance · bars per cell")
if data["problems"]:
    for p in data["problems"]:
        st.error(p)
if cells.empty:
    ui.empty_state("NO_DATA", "No approved cell table can be shown (see the messages above).")
    st.stop()

st.caption(f"Approved proposal **{ap.get('proposal_id')}** (sha256 {str(ap.get('proposal_hash'))[:12]}…), approved by "
           f"{ap.get('approved_by')} at {ap.get('approved_at')}. Applied to serving on go-live day (runbook U3b).")

cnt = CV.counts(cells)
stk = CV.staked(cells)
c1, c2, c3 = st.columns(3)
ui.metric_card("Cells", f"{len(cells)}", title="League × market-side",
               supporting=[(s, f"{n}") for s, n in cnt.items()], container=c1)
ui.metric_card("Staked cells", f"{len(stk)}", title="Real money (everything else PAPER)",
               supporting=[(f"{r['League']} · {r['Market']}", f"{r['Stake']} {r['Stake %']:.2f}%")
                           for _, r in stk.iterrows()],
               status=ui.status_chip("PAPER", label="CLOUDBET PAPER_ONLY"), container=c2)
ui.metric_card("PROVEN cells", f"{cnt['PROVEN']}", title="Evidence",
               supporting=[("PROMISING", f"{cnt['PROMISING']}"), ("UNPROVEN", f"{cnt['UNPROVEN']}"),
                           ("OFF (no tip)", f"{cnt['OFF']}")],
               status=ui.status_chip("NO_DATA", tone="NEUTRAL", label="NOTHING PROVEN YET"), container=c3)
st.caption("Status per cell from the approved study: PROMISING = positive evidence, not proof; UNPROVEN = tips on "
           "paper until the Clean Record decides; OFF = no tip in that cell. Qualifying bar = the one bar a tip's "
           "measure must clear (EV or EDGE; ALL = every match in range, a league-wide cell, listed as the league's "
           "own line outside TOP-3). TIER2 boundary = only the TOP-3 TIER2 slot rule, never a tier.")

# ── the registry ──────────────────────────────────────────────────────────────────────────────────
ui.section_header("Every league × market cell", "Filter by status or league. Stake % = the policy's base fraction "
                  "of the bankroll per tip in that cell; 0 = paper.")
f1, f2 = st.columns([1, 2])
with f1:
    stat = st.multiselect("Status", list(CV.STATUS_ORDER), default=["PROVEN", "PROMISING", "UNPROVEN"],
                          key="cells_status")
with f2:
    lg = st.multiselect("League", sorted(cells["League"].unique()), key="cells_league")
view = cells[cells["Status"].isin(stat)] if stat else cells
if lg:
    view = view[view["League"].isin(lg)]
ui.compact_table(view, {"Qualifying bar": ui.num_col("Qualifying bar", fmt="%.2f"),
                        "TIER2 boundary": ui.num_col("TIER2 boundary", fmt="%.2f"),
                        "Stake %": ui.num_col("Stake %", fmt="%.2f")})
st.caption(f"{len(view)} of {len(cells)} cells shown.")

# ── Clean Record per cell ─────────────────────────────────────────────────────────────────────────
ui.section_header(f"{CR.clean_label()} per cell", "Settled team tips of the Clean Record per cell, flat 1u, paper and "
                  "staked together; the stake column above says which cells are real money.")


@st.cache_data(ttl=120)
def _clean_cells() -> pd.DataFrame:
    sp = CR.team_split(today_view.load_settled())
    return CR.group_table(CR.with_cell(sp[CR.CLEAN]), "Cell", pnl="_pnl", date="_dt", key="clean|cell")


cc = _clean_cells()
if cc.empty:
    ui.empty_state("NO_DATA", CR.empty_text())
else:
    ui.compact_table(cc, CR.table_config(ui))

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
