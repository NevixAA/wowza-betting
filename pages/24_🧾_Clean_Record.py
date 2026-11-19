"""Clean Record (from 21.11) - results on the repaired system, counted from 0 (owner 2026-11-11, restated 2026-11-17).

Owner 2026-11-17: "we need to start from 0 counting again (do not delete the old history of this season's results);
we need a comparison later. But after 21.11 it's a new page."

THE RECORD. Only tips generated AND played on or after the Clean Record start count: config/trustworthy_performance.json
clean_record_start_utc (owner decision 2026-11-11: 2026-11-21, the first round on the clean retrained models and the
approved per-league bars; written by runbook step U6 at the go-live merge). dashboard_data/clean_record.py is THE
split; this page re-derives nothing. The season before it ("Season to 20.11 (pre-repair models)") stays on its own
pages exactly as recorded and appears here only in the comparison section, never in a Clean Record total.

BEFORE THE GO-LIVE MERGE the tree has no Clean Record split (dashboard_data/clean_record.py ships with it) and no tip
of the clean models exists, so the page shows an honest empty state ("starts 21.11") and nothing else.

STATISTICS ONLY: settled counts, hit rate, flat-1u P/L, ROI with its matchday block-bootstrap 95% CI. No price column
is read for display.
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
from dashboard_data import season  # noqa: E402

st.set_page_config(page_title="Wowza | Clean Record", page_icon="🧾", layout="wide")

SPLIT = season.split_in_tree()
NOT_ENOUGH = "Not enough data yet"

if not SPLIT:
    # ── main before the go-live merge: nothing of the clean system exists yet ─────────────────────────
    ui.page_header(season.clean_label(),
                   f"Results on the repaired data and the clean models count from 0, {season.starts_text()}. "
                   f"This season's earlier results stay on their own pages, labelled \"{season.season_label()}\".",
                   eyebrow_text="Performance · clean record",
                   badge=ui.status_chip("NO_DATA", label=season.starts_text().upper()))
    if season.started():
        # The date has come but this deployment does not carry the split yet: say so, never count here.
        ui.empty_state("NO_DATA", f"The Clean Record {season.starts_text()}, but this deployment does not carry the "
                                  "clean-model split yet (it arrives with the go-live merge), so no figure is shown "
                                  "here. Nothing is counted from the pre-repair pages in its place.")
    else:
        ui.empty_state("NO_DATA", f"The Clean Record {season.starts_text()}: the first round on the clean retrained "
                                  "models and the approved per-league bars, live from the go-live merge. No tip of "
                                  "the clean system has been made or settled yet, so there is nothing to count. "
                                  "Nothing from before the start will ever be counted here.")
    ui.section_header("Staked vs not staked, per league and market",
                      "Bets, hit rate, P/L and ROI with its 95% CI, and CLV % where a certified close exists; "
                      "staked and not staked apart. Fills from the first settled Clean Record tip.")
    ui.empty_state("NO_DATA", "No settled Clean Record tip yet.")
    ui.section_header(f"Comparison: {season.clean_label()} vs {season.season_label()}",
                      "The same metrics side by side, never one total.")
    ui.empty_state("NO_DATA", f"{NOT_ENOUGH}. The comparison opens once the Clean Record has enough settled tips.")
    from utils.disclaimer import disclaimer_footer  # noqa: E402
    disclaimer_footer()
    st.stop()

# ── the go-live tree: THE split ───────────────────────────────────────────────────────────────────────
from dashboard_data import clean_record as CR, today_view  # noqa: E402
from dashboard_data import top3_league as T3  # noqa: E402


@st.cache_data(ttl=120)
def _data() -> dict:
    settled = today_view.load_settled()
    sp = CR.team_split(settled)
    led = pd.read_csv(config.OUTPUT_DIR / "player_ledger.csv", low_memory=False) \
        if (config.OUTPUT_DIR / "player_ledger.csv").exists() else pd.DataFrame()
    return {"team": CR.team_summaries(settled), "sp": sp, "props": CR.props_summaries(led),
            "top3": CR.top3_summaries(T3.ledger())}


D = _data()
ui.page_header(season.clean_label(),
               f"Counted from 0 on {CR.start_date()} (owner 2026-11-11, restated 2026-11-17): the first round on the "
               f"clean models and the approved per-league bars. \"{season.season_label()}\" stays on its own pages, "
               "exactly as recorded, and is shown here only for comparison.",
               eyebrow_text="Performance · clean record",
               badge=ui.status_chip("LIVE" if CR.started() else "NO_DATA",
                                    label="CLEAN RECORD LIVE" if CR.started() else season.starts_text().upper()))

_cfg = CR.table_config(ui)
SCOPES = [("Staked bets (a recorded policy stake > 0)", D["team"][CR.CLEAN]["staked"], D["team"][CR.BEFORE]["staked"]),
          ("All team tips, staked or not (flat 1u)", D["team"][CR.CLEAN]["all"], D["team"][CR.BEFORE]["all"]),
          ("Player props (research, no stake)", D["props"][CR.CLEAN], D["props"][CR.BEFORE]),
          ("TOP-3 per league (0.5% stake, flat; P/L flat 1u)", D["top3"][CR.CLEAN], D["top3"][CR.BEFORE])]

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
                   status=ui.status_chip("PAPER", label="NO STAKE") if "no stake" in name.lower() else "", n=c["n"] or None, container=col)
if not any(c["n"] for _n, c, _b in SCOPES):
    ui.empty_state("NO_DATA", CR.empty_text())

# ── staked vs not staked, per league and market ────────────────────────────────────────────────────────
ui.section_header("Staked vs not staked, per league and market",
                  "Every settled team tip of the Clean Record, split by THE staked definition (src/staked.py: a "
                  "recorded policy stake > 0). A cell is the league × market-side unit the bars are approved in.")
_c = D["sp"][CR.CLEAN]
if not len(_c):
    ui.empty_state("NO_DATA", CR.empty_text())
else:
    _cell = CR.with_cell(_c)
    _stk = _cell["_staked"].astype(bool).to_numpy() if "_staked" in _cell.columns else [False] * len(_cell)
    s1, s2 = st.columns(2)
    with s1:
        ui.eyebrow("Staked")
        ui.compact_table(CR.group_table(_cell[_stk], "Cell", pnl="_pnl", date="_dt", key="clean|staked|cell"), _cfg)
    with s2:
        ui.eyebrow("Not staked (flat 1u)")
        ui.compact_table(CR.group_table(_cell[[not x for x in _stk]], "Cell", pnl="_pnl", date="_dt",
                                        key="clean|paper|cell"), _cfg)
st.caption("CLV %: shown only where a certified close exists (observed at most 30 minutes before kickoff). The "
           "ledger rows carry no certified close, so no CLV % is claimed here; the live capture's certified "
           "coverage is on the Market Intelligence page.")

# ── comparison: same metrics side by side, once there is enough data ─────────────────────────────────
mins = CR.minimums()
ui.section_header(f"Comparison: {CR.clean_label()} vs {CR.before_label()}",
                  f"One table per scope, the same metrics. It opens per scope once the Clean Record has at least "
                  f"{mins['settled']} settled tips over {mins['matchdays']} matchdays (config/trustworthy_performance"
                  ".json minimums); until then it says so.")
for name, c, b in SCOPES:
    ui.eyebrow(name)
    if c["status"] == "OK":
        ui.compact_table(CR.comparison(c, b), _cfg)
    else:
        ui.empty_state("NO_DATA", f"{NOT_ENOUGH}: the Clean Record has {c['n']} settled over {c['matchdays']} "
                                  f"matchdays (needs {mins['settled']} over {mins['matchdays']}).")
st.caption(CR.BEFORE_NOTE + " TOP-3 starts on the start date with its 0.5% flat stake (owner 2026-11-17), so it has no pre-repair "
           "row. A tip neither rule can place (no generation time and a match after the start) is in neither row: "
           f"team {D['team'][CR.UNPLACED]}, props {D['props'][CR.UNPLACED]}, TOP-3 {D['top3'][CR.UNPLACED]}.")

with st.expander(f"{CR.before_label()} — by market, for reference"):
    st.caption(CR.BEFORE_NOTE)
    ui.compact_table(CR.group_table(D["sp"][CR.BEFORE], "_mkt", pnl="_pnl", date="_dt", key="before|mkt")
                     .rename(columns={"_mkt": "Market"}), _cfg)

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
