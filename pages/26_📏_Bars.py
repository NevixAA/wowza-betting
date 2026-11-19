"""Bars - the current per-league x market bars, statistics only (owner 2026-11-17).

Owner: "update the dashboard with the current bars". Shows, clearly labelled:
  (a) TP-2026-11-15-bade2c0b - owner-APPROVED 2026-11-15 (protocol v5);
  (b) TP-2026-11-16-500964af - the clean-model re-run, AWAITING OWNER APPROVAL on 2026-11-20;
plus the go-live stake plan (owner 2026-11-16) and the owner's stake overrides.

VIEWER ONLY. Every figure comes from reports/dashboard/bars_statistics.json (scripts/build_bars_statistics.py,
source commits recorded in the file). No bookmaker price is shown or read: statuses, bars, odds-range rules,
bet counts, ROI with its 95% CI and mean CLV % only.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require  # noqa: E402
require("pro")

st.set_page_config(page_title="Wowza | Bars", page_icon="📏", layout="wide")

import dashboard_ui as ui  # noqa: E402
from dashboard_data import bars_stats_view as B  # noqa: E402

L = B.load()
D = L["data"]
A = B.proposal(D, "approved")
C = B.proposal(D, "clean_rerun")

ui.page_header("Bars per league and market",
               "Every league × market cell has its own bar and status, never a global one. PROVEN = the 95% CI "
               "of the ROI is above 0; PROMISING = positive but not yet proven; UNPROVEN = no proof either way; "
               "OFF = the data shows no value at any bar, so no tip.",
               eyebrow_text="Performance · bars (statistics only, no prices)")

if D is None:
    ui.empty_state("ERROR", "; ".join(L["problems"]) + ". Nothing is shown in its place.")
    st.stop()

_cfg = {"Bar": ui.num_col("Bar", fmt="%.1f%%", help="The qualifying bar a tip must clear, in the cell's measure "
                          "(EDGE = model minus market probability, EV = expected value; ALL = league-wide, every "
                          "tip of the cell)."),
        "WF ROI %": ui.num_col("WF ROI %", fmt="%+.1f%%", help="Walk-forward out-of-sample ROI of the bar."),
        "WF bets": ui.num_col("WF bets", fmt="%d"),
        "Study bets": ui.num_col("Study bets", fmt="%d", help="Bets in the selection study on own bet-time prices."),
        "Study ROI %": ui.num_col("Study ROI %", fmt="%+.1f%%"),
        "ROI 95% CI": ui.text_col("ROI 95% CI", help="Matchday block bootstrap, 95%. PROVEN needs the lower end "
                                  "above 0."),
        "CLV %": ui.num_col("CLV %", fmt="%+.1f%%", help="Mean closing-line value of the study bets."),
        "Odds range": ui.text_col("Odds range", help="The cell's odds filter (a rule, not a price)."),
        "Stake rule": ui.text_col("Stake rule", help="config/bankroll_policy.json cell_staking on the approved "
                                  "branch. A cell without an entry is PAPER: shown and recorded, stake 0.")}


def _counts_line(p) -> None:
    c = B.counts(p)
    ui.stat_line([("PROVEN", str(c["PROVEN"])), ("PROMISING", str(c["PROMISING"])),
                  ("UNPROVEN", str(c["UNPROVEN"])), ("OFF", str(c["OFF"])), ("Cells", str(len(p["cells"])))])


def _source(p) -> str:
    s = p["source"]
    rep = f", report {s['report']}" if s.get("report") else ""
    return (f"{p['proposal_id']} (sha256 {p['proposal_hash'][:12]}…), {p['entries']} entries, created "
            f"{str(p.get('created_at'))[:10]}. Source: branch {s['branch']} commit {s['commit'][:9]}, "
            f"{s['proposal']}, evidence {s['evidence']}{rep}.")


# ── the go-live stake plan ────────────────────────────────────────────────────────────────────────
plan = D.get("stake_plan") or {}
ui.section_header(f"Go-live stake plan (owner decision {plan.get('decided', '')})", plan.get("rule", ""))
_plan = B.stake_plan_frame(D)
if len(_plan):
    ui.compact_table(_plan)
for o in plan.get("overrides") or []:
    st.markdown(f"- {o['text']}")
st.caption(f"Source: {plan.get('source', '')}.")

# ── (a) the approved bars ─────────────────────────────────────────────────────────────────────────
tabs = st.tabs([f"(a) {A['label']}" if A else "(a) approved", f"(b) {C['label']}" if C else "(b) clean re-run",
                "Changes (a) → (b)"])
for tab, p, chip in ((tabs[0], A, ui.status_chip("QUALIFIED", label="APPROVED BY THE OWNER")),
                     (tabs[1], C, ui.status_chip("REGISTERED", label="AWAITING OWNER APPROVAL ON NOV 20"))):
    with tab:
        if not p:
            ui.empty_state("NO_DATA", "This proposal is not in the statistics file.")
            continue
        st.markdown(chip)
        st.caption(p["state_text"])
        _counts_line(p)
        ui.eyebrow("Cells with a stake of their own - evidence")
        stk = B.staked(p)
        if len(stk):
            ui.compact_table(stk[["League", "Track", "Market", "Status", "Bar", "Measure", "Stake rule", "WF bets",
                                  "WF ROI %", "Study bets", "Study ROI %", "ROI 95% CI", "CLV %"]], _cfg)
            st.caption("A league-wide cell (mode LEAGUE_WIDE, measure ALL) has no selection study: its evidence is "
                       "the walk-forward count and ROI only, so its CI shows '—'."
                       + (" For (b) the stake column shows the stake the same rule gives if the owner approves "
                          "(b) with the go-live stake scope." if p is C else ""))
        else:
            ui.empty_state("NO_DATA", "No cell stakes on its own: only TOP-3 tips carry a stake (0.5% flat, placed "
                                      "manually at the tip's odds).")
        ui.eyebrow("Every league × market cell")
        status = st.multiselect("Status", B.STATUS_ORDER, default=["PROVEN", "PROMISING", "UNPROVEN"],
                                key=f"st_{p['key']}")
        f = B.frame(p)
        ui.compact_table(f[f["Status"].isin(status)].reset_index(drop=True), _cfg)
        if p is C:
            st.caption("The 34 Standard half-time cells (HT Over 0.5 / 1.5) have no clean-model prediction: they "
                       "were OFF in (a) and are not re-measured in (b) (reports/P9_BARS_V5_CLEAN_MODELS.md).")
        st.caption(_source(p))

with tabs[2]:
    ch = B.changed(A, C)
    st.caption(f"{len(ch)} cells differ between (a) and (b), half-time cells not re-measured included.")
    ui.compact_table(ch)

# ── tested, not adopted ───────────────────────────────────────────────────────────────────────────
for na in D.get("not_adopted") or []:
    ui.section_header(na["title"], na["text"])
    st.markdown(ui.status_chip("REGISTERED", label="TESTED · NOT ADOPTED"))
    ui.compact_table(B.not_adopted_frame(na))
    st.caption(f"Source: branch {na['branch']} commit {na['commit'][:9]}, {na['report']} (cell counts over the "
               "194 cells; the 34 Standard half-time cells stay OFF, not re-measured).")

st.caption(D.get("evidence_note", "") + " No bookmaker price is published here (OddsAPI / API-Football licences).")

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
