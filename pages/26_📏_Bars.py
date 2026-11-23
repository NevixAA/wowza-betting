"""Bars - the per-league x market bars PRODUCTION SERVES, statistics only.

Shows the proposal named in config/threshold_applied.json (last applied by src/threshold_apply.py after the owner's
approval), its approval record, every league x market cell (OFF cells included, with their reason), the study
families, the stake rule and the owner's approval chain as history. The proposal id is never hard-coded: the file is
rebuilt from the serving config, and the page names a mismatch if the file is stale.

VIEWER ONLY. Every figure comes from reports/dashboard/bars_statistics.json (scripts/build_bars_statistics.py). No
bookmaker price is shown or read: statuses, bars, odds-range rules, bet counts, ROI with its 95% CI and mean CLV % only.
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
P = B.serving(D)

ui.page_header("Bars per league and market",
               "The bars production serves now. Every league × market cell has its own bar and status, never a "
               "global one. PROVEN = the 95% CI of the ROI is above 0; PROMISING = positive but not yet proven; "
               "UNPROVEN = no proof either way; OFF = no bar, so no tip (the reason is shown per cell).",
               eyebrow_text="Performance · bars (statistics only, no prices)")

if D is None or P is None:
    ui.empty_state("ERROR", "; ".join(L["problems"] or ["the statistics file has no serving proposal"])
                   + ". Nothing is shown in its place.")
    st.stop()

for prob in B.serving_problems(D):
    ui.empty_state("ERROR", prob)

_cfg = {"Bar": ui.num_col("Bar", fmt="%.1f%%", help="The qualifying bar a tip must clear, in the cell's measure "
                          "(EDGE = model minus market probability, EV = expected value; ALL = league-wide, every "
                          "tip of the cell)."),
        "WF ROI %": ui.num_col("WF ROI %", fmt="%+.1f%%", help="Walk-forward out-of-sample ROI of the bar."),
        "WF bets": ui.num_col("WF bets", fmt="%d"),
        "Study bets": ui.num_col("Study bets", fmt="%d", help="Bets at the final bar in the validation window."),
        "Study ROI %": ui.num_col("Study ROI %", fmt="%+.1f%%"),
        "ROI 95% CI": ui.text_col("ROI 95% CI", help="Matchday block bootstrap, 95%. PROVEN needs the lower end "
                                  "above 0. '—' = too few bets for a CI."),
        "CLV %": ui.num_col("CLV %", fmt="%+.1f%%", help="Mean closing-line value of the study bets."),
        "Odds range": ui.text_col("Odds range", help="The cell's odds filter (a rule, not a price)."),
        "Counts": ui.text_col("Counts", help="'not recorded' = the serving evidence stores no walk-forward or "
                              "study counts for this cell (OFF cells: reason and detail only). The count columns "
                              "stay empty; nothing is filled."),
        "Why OFF": ui.text_col("Why OFF", help="The bars study's reason code for an OFF cell."),
        "OFF detail": ui.text_col("OFF detail", help="The evidence's own detail text for the OFF reason."),
        "Stake rule": ui.text_col("Stake rule", help="config/bankroll_policy.json cell_staking. No cell stakes on "
                                  "its own; a TOP-3 pick carries the flat TOP-3 stake.")}

# ── the serving proposal ──────────────────────────────────────────────────────────────────────────
ui.section_header(f"Serving now: {P['proposal_id']}",
                  f"Approved by {P.get('approved_by')} on {str(P.get('approved_at'))[:10]}, applied "
                  f"{str(P.get('applied_at'))[:16].replace('T', ' ')} UTC. Protocol {P.get('protocol')}.")
st.markdown(ui.status_chip("QUALIFIED", label="SERVING · APPROVED BY THE OWNER"))
if P.get("owner_words"):
    st.caption(f"Owner approval: {P['owner_words']}.")
if P.get("supersedes"):
    st.caption(f"Supersedes {P['supersedes']}.")
c = B.counts(P)
ui.stat_line([("PROVEN", str(c["PROVEN"])), ("PROMISING", str(c["PROMISING"])), ("UNPROVEN", str(c["UNPROVEN"])),
              ("OFF", str(c["OFF"])), ("Cells", str(len(P["cells"])))])

# ── stake rule ────────────────────────────────────────────────────────────────────────────────────
plan = D.get("stake_plan") or {}
ui.section_header(f"Stake rule (owner decision {plan.get('decided', '')})", plan.get("rule", ""))
_plan = B.stake_plan_frame(D)
if len(_plan):
    ui.compact_table(_plan)
stk = B.staked(P)
if len(stk):
    ui.compact_table(stk[["League", "Track", "Market", "Status", "Bar", "Measure", "Stake rule", "WF bets",
                          "WF ROI %", "Study bets", "Study ROI %", "ROI 95% CI", "CLV %"]], _cfg)
st.caption(f"Source: {plan.get('source', '')}.")

# ── every cell ────────────────────────────────────────────────────────────────────────────────────
ui.eyebrow("Every league × market cell")
status = st.multiselect("Status", B.STATUS_ORDER, default=B.STATUS_ORDER, key="st_serving")
f = B.frame(P)
ui.compact_table(f[f["Status"].isin(status)].reset_index(drop=True), _cfg)
st.caption("Counts 'not recorded' = the field does not exist in the serving evidence: "
           f"{D.get('not_recorded', '')}. A league-wide cell (mode LEAGUE_WIDE, measure ALL) is scored on its "
           "validation bets only; a CI shows '—' when there are too few bets for one.")

ui.eyebrow("Why cells are OFF")
ui.compact_table(B.off_reasons(P))
st.caption("OFF_NO_OWN_ESTIMATE: too few of the league's own bets at the bar, so the bar would be the pooled curve, "
           "not the league's own. OFF_NO_VALUE: no bar with positive value. INSUFFICIENT_SELECTION_EVIDENCE: the "
           "whole family had too few selection-window bets for a final study. OFF_THIN: too few bets at every bar, own and pooled.")

ui.eyebrow("Study families (one per track × market)")
ui.compact_table(B.families_frame(P))

s = P["source"]
st.caption(f"{P['proposal_id']} (sha256 {P['proposal_hash'][:12]}…), {P['entries']} entries, created "
           f"{str(P.get('created_at'))[:10]}. Serving config {s['serving_config']}; approval {s['approval']}; "
           f"proposal {s['proposal']}; evidence {s['evidence']}.")

# ── history ───────────────────────────────────────────────────────────────────────────────────────
ui.section_header("Approval history", "Earlier owner-approved bars, superseded. Shown for the record only; they "
                  "no longer serve.")
ui.compact_table(B.history_frame(D))

st.caption(D.get("evidence_note", "") + " No bookmaker price is published here (OddsAPI / API-Football licences).")

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
