"""Predictions - every model's prediction for every market of a match, and how they turned out (owner 2026-11-22).

LEARNING RECORD, NOT BETS. The prediction ledgers (owner rule 2026-11-17: "predict every match") store every fixture
x market a model can rate - v9 team (Standard / New Format), v9 props incl. the challengers, v9 corners / cards, Pro's
score grid and Pro's combos - including the ledger-only leagues that get no tips. This page shows them per match, the
outcome once settled, and hit rate / log loss / Brier per model x market x league since the Clean Record start.

VIEWER ONLY. It reads two compact files scripts/build_predictions_index.py writes from the ledgers
(reports/dashboard/predictions_index.parquet, predictions_summary.csv; refreshed by update_results.yml). No price is
in either file and none is shown. v11 predictions are private (wowza-exec) and never shown here.
"""
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require  # noqa: E402
require("pro")

st.set_page_config(page_title="Wowza | Predictions", page_icon="🔮", layout="wide")

import dashboard_ui as ui  # noqa: E402
from dashboard_data import predictions_view as V  # noqa: E402


ui.page_header("Predictions",
               "Every model's prediction for every market of a match, and how each one turned out. A learning "
               "record, not bets: nothing here is a tip or a stake, and no price is shown.",
               eyebrow_text="Performance · prediction ledger", state="LEARNING")
st.caption("Sources: the v9 and Pro prediction ledgers and their settled views. v11: private.")


@st.cache_resource(ttl=300, show_spinner=False)
def _index():
    try:
        return V.load_index(), None
    except (OSError, ValueError, ImportError) as e:
        return None, f"reports/dashboard/{V.INDEX.name} cannot be read ({type(e).__name__})"


@st.cache_data(ttl=300, show_spinner=False)
def _summary():
    try:
        return V.load_summary(), None
    except (OSError, ValueError) as e:
        return None, f"reports/dashboard/{V.SUMMARY.name} cannot be read ({type(e).__name__})"


idx, err = _index()
if idx is None:
    ui.empty_state("ERROR", f"{err}. Nothing is shown in its place.")
    st.stop()
if idx.empty:
    ui.empty_state("NO_DATA", "The prediction ledgers hold no rows yet.")
    st.stop()

ui.stat_line([("Predictions", f"{len(idx):,}"),
              ("Matches", f"{idx['fixture_ref'].nunique():,}"),
              ("Models", str(idx["model_id"].nunique())),
              ("Latest prediction", idx["predicted_at"].max().strftime("%d.%m %H:%M UTC"))])

# ═══ PICK A MATCH ═════════════════════════════════════════════════════════════════════════════════════════════════
ui.section_header("Pick a match", "Kickoff dates in UTC. Ledger-only leagues are predicted every match but never "
                                  "tipped.")
ko = idx["match_kickoff_utc"].dropna()
lo_d, hi_d = ko.min().date(), ko.max().date()
today = pd.Timestamp.now(tz="UTC").date()
d0 = min(max(today - timedelta(days=2), lo_d), hi_d)
d1 = max(min(today + timedelta(days=3), hi_d), d0)
c1, c2, c3 = st.columns([2, 3, 2])
rng = c1.date_input("Kickoff dates", value=(d0, d1), min_value=lo_d, max_value=hi_d, key="pred_dates")
if not isinstance(rng, (tuple, list)):
    rng = (rng, rng)
start_d, end_d = (rng[0], rng[-1]) if len(rng) else (d0, d1)
leagues = V.league_options(idx)
league = c2.selectbox("League", ["ALL"] + list(leagues), key="pred_league",
                      format_func=lambda s: "All leagues" if s == "ALL" else leagues[s])
snapshot = c3.radio("Snapshot", ["first_sight", "final"], key="pred_snapshot", horizontal=True,
                    format_func=lambda s: V.SNAPSHOT_LABEL.get(s, s),
                    help="First sight = the first prediction a model made for the match. Final = the last one, "
                         "at most 120 minutes before kickoff (only where a run fell in that window).")

day = idx["match_kickoff_utc"].dt.date
sel = idx[(day >= start_d) & (day <= end_d) & (idx["snapshot"] == snapshot)]
if league != "ALL":
    sel = sel[sel["league_slug"] == league]
M = V.matches(sel)
if M.empty:
    ui.empty_state("NO_DATA", "No prediction in this date range, league and snapshot.")
else:
    labels = {r.match_key: f"{r.kickoff_utc:%d.%m %H:%M} · {r.home_team} v {r.away_team} · {r.league}"
              + (f" ({V.LEDGER_ONLY_LABEL})" if r.ledger_only else "") for r in M.itertuples()}
    mk = st.selectbox(f"Match ({len(M)} in range)", list(labels), key="pred_match", format_func=labels.get)
    all_versions = st.toggle("Show all versions", value=False, key="pred_all_versions",
                             help="Off: one row per question - the model version that was serving at this snapshot "
                                  "(the latest prediction before kickoff). On: every stored version, e.g. the legacy "
                                  "props model next to the clean rebuild. The ledger keeps every version either way.")
    rows_all = V.match_rows(sel, mk)
    rows = rows_all if all_versions else V.serving_rows(rows_all)
    m = M[M["match_key"] == mk].iloc[0]
    ui.stat_line([("Kickoff", f"{m.kickoff_utc:%a %d.%m.%Y %H:%M} UTC"), ("League", m.league),
                  ("Models", str(m.models)), ("Predictions", f"{len(rows):,}"),
                  ("Settled rows", f"{int((rows['outcome_status'] == 'SETTLED').sum()):,}")])
    if m.kickoff_source == "conflict":
        st.caption("The prediction sources disagree on this kickoff and no API-Football fixture settles it: each "
                   "row keeps its own kickoff (a data-truth finding).")
    elif int(m.refs) > 1:
        st.caption(f"One match, {int(m.refs)} provider references across the models (merged into one identity; "
                   f"kickoff: {m.kickoff_source}).")
    if m.ledger_only:
        st.markdown(ui.status_chip("OBSERVE", label="LEDGER ONLY — NO TIPS") + "  "
                    + ui.tone_text("This league is predicted for learning only; no tip is ever sent from it.",
                                   "NEUTRAL"))

    cfg = {"Probability": ui.pct_col("Probability", help="The model's probability for this selection."),
           "Pick": ui.text_col("Pick", help="The model's most likely outcome for the question."),
           "Result": ui.text_col("Result", help="hit / miss of the pick once the match is settled.")}
    tabs = st.tabs([t for t, _ in V.FAMILIES])
    for tab, (title, models) in zip(tabs, V.FAMILIES):
        with tab:
            fam = rows[rows["model_id"].isin(models)]
            if fam.empty:
                ui.empty_state("NO_DATA", f"No {title.lower()} prediction for this match"
                                          + (" (Pro's grid and combos cover Pro's own leagues)."
                                             if models[0].startswith("pro_") else "."))
                continue
            if title == "Player props":
                mkts = sorted(fam["market"].unique(), key=lambda x: list(V.MARKET_LABEL).index(x)
                              if x in V.MARKET_LABEL else 99)
                pm = st.selectbox("Props market", mkts, key="pred_props_market", format_func=V.market_label)
                fam = fam[fam["market"] == pm].sort_values("probability", ascending=False)
            if title == "Pro score grid":
                grid = V.score_grid(fam)
                if not grid.empty:
                    st.caption("Correct score: probability of each exact score (rows = home goals).")
                    ui.compact_table(grid, {c: ui.pct_col(c) for c in grid.columns if c.startswith("Away")})
                fam = fam[fam["market"] != "correct_score"] if not grid.empty else fam
                cs = rows[(rows["model_id"] == "pro_bb_dc") & (rows["market"] == "correct_score") & rows["pick"]]
                if len(cs):
                    r0 = cs.iloc[0]
                    st.caption(f"Most likely score {r0['selection']} ({r0['probability']:.1%}) · "
                               f"{V.result_text(r0)}.")
            ui.compact_table(V.prediction_table(fam), cfg)

# ═══ LEARNING RECORD ══════════════════════════════════════════════════════════════════════════════════════════════
S, serr = _summary()
ui.section_header("Learning record since the Clean Record start",
                  "Learning record, not bets. Predictions made on or after the Clean Record start, scored once "
                  "settled. Hit rate = the model's most likely outcome happened. Log loss and Brier: lower is "
                  "better. Brier is binary for two-way questions and the multi-class sum for 1X2 and correct score.")
if S is None:
    ui.empty_state("ERROR", f"{serr}. Nothing is shown in its place.")
elif S.empty:
    ui.empty_state("NO_DATA", "No prediction made since the Clean Record start yet.")
else:
    start = str(S["clean_record_start"].iloc[0])
    S = S[S["snapshot"] == snapshot]
    st.caption(f"Clean Record start {start} · snapshot: {V.SNAPSHOT_LABEL.get(snapshot, snapshot)} · settled "
               f"questions {int(S['settled'].sum()):,} of {int(S['predictions'].sum()):,}.")
    models = sorted(S["model_id"].unique(), key=lambda m: list(V.MODEL_LABEL).index(m) if m in V.MODEL_LABEL
                    else 99)
    pick = st.multiselect("Models", models, default=models, key="pred_sum_models", format_func=V.model_label)
    S = S[S["model_id"].isin(pick)]
    scfg = {"Hit rate": ui.pct_col("Hit rate"), "Avg pick probability": ui.pct_col("Avg pick probability"),
            "Log loss": ui.num_col("Log loss", fmt="%.3f"), "Brier": ui.num_col("Brier", fmt="%.3f"),
            "Avg probability (YES-type)": ui.pct_col(
                "Avg probability (YES-type)", help="Single-selection questions (player props, goal bands, team 4+, "
                                                   "combos): the mean probability the model gave the selection."),
            "Happened (YES-type)": ui.pct_col("Happened (YES-type)",
                                              help="How often that selection happened. Close to the average "
                                                   "probability = well calibrated.")}
    t1, t2 = st.tabs(["By model × market", "By model × market × league"])
    with t1:
        ui.compact_table(V.summary_display(V.summary_rollup(S, ["model_id", "market"]), with_league=False), scfg)
    with t2:
        ui.compact_table(V.summary_display(S.sort_values(["model_id", "market", "league"]), with_league=True), scfg)
    st.caption("Small samples move a lot: read a rate next to its settled count. For rare events (props, "
               "combos, goal bands) the most likely outcome is usually NO, so their hit rate is high by "
               "construction: judge them on log loss, Brier and average probability against how often it "
               "happened.")

# ═══ SUCCESS BY GAMEWEEK (owner 2026-11-23) ═══════════════════════════════════════════════════════════════════════
from dashboard_data import prediction_gw_view as GV  # noqa: E402

ui.section_header("Success by gameweek",
                  "The stored learning dataset: for every gameweek, league, market, model and model version, how often "
                  "the model's own pick happened (success), with Brier and log loss on the settled predictions. "
                  "Gameweek = the TOP-3 league week (Monday–Sunday tip days). A closed gameweek never changes; a later "
                  "correction is stored beside it. Model versions are never pooled.")


@st.cache_data(ttl=300, show_spinner=False)
def _gw():
    try:
        return GV.load(), GV.load_corrections(), None
    except (OSError, ValueError) as e:
        return None, None, f"output/{GV.DATASET.name} cannot be read ({type(e).__name__})"


G, GC, gerr = _gw()
if G is None:
    ui.empty_state("NO_DATA", f"{gerr}. It is written after settlement by the results update.")
elif G.empty:
    ui.empty_state("NO_DATA", "No gameweek in the dataset yet.")
else:
    g1, g2, g3, g4, g5 = st.columns([2, 2, 2, 2, 2])
    rec = g1.radio("Record", [True, False], format_func=GV.RECORD_LABEL.get, key="gw_record",
                   help="Clean Record = predictions made on or after the Clean Record start. The two are never mixed.")
    Gs = G[(G["clean_record"] == rec) & (G["snapshot"] == snapshot)]
    lg_opts = sorted(Gs["league"].unique())
    gw_league = g2.selectbox("League", ["ALL"] + lg_opts, key="gw_league",
                             format_func=lambda s: "All leagues" if s == "ALL" else s)
    md_opts = sorted(Gs["model_id"].unique(), key=lambda m: list(V.MODEL_LABEL).index(m)
                     if m in V.MODEL_LABEL else 99)
    gw_model = g3.selectbox("Model", ["ALL"] + md_opts, key="gw_model",
                            format_func=lambda m: "All models" if m == "ALL" else V.model_label(m))
    if gw_league != "ALL":
        Gs = Gs[Gs["league"] == gw_league]
    if gw_model != "ALL":
        Gs = Gs[Gs["model_id"] == gw_model]
    mk_opts = sorted(Gs["market"].unique(), key=lambda x: list(V.MARKET_LABEL).index(x)
                     if x in V.MARKET_LABEL else 99)
    gw_market = g4.selectbox("Market", ["ALL"] + mk_opts, key="gw_market",
                             format_func=lambda m: "All markets" if m == "ALL" else V.market_label(m))
    if gw_market != "ALL":
        Gs = Gs[Gs["market"] == gw_market]
    gw_opts = sorted(Gs["gw"].unique(), reverse=True)
    gw_pick = g5.multiselect("Gameweeks", gw_opts, default=gw_opts, key="gw_weeks")
    Gs = Gs[Gs["gw"].isin(gw_pick)]
    if Gs.empty:
        ui.empty_state("NO_DATA", "No gameweek row for these filters.")
    else:
        ui.stat_line([("Gameweeks", str(Gs["gw"].nunique())),
                      ("Predictions", f"{int(Gs['predictions'].sum()):,}"),
                      ("Settled", f"{int(Gs['settled'].sum()):,}"),
                      ("Closed rows", f"{int((Gs['status'] == 'CLOSED').sum()):,} of {len(Gs):,}")])
        tr = GV.trend(Gs, V.MODEL_LABEL, V.MARKET_LABEL)
        if len(tr) >= 2:
            st.caption("Success % by gameweek (the series with the most settled predictions; leagues summed).")
            st.line_chart(tr, height=260)
        elif len(tr) == 1:
            st.caption("One gameweek so far: the trend line starts with the second.")
        gcfg = {"Success": ui.pct_col("Success", help="Hits / settled: the model's most likely outcome happened."),
                "Brier": ui.num_col("Brier", fmt="%.3f"), "Log loss": ui.num_col("Log loss", fmt="%.3f")}
        ui.compact_table(GV.table(Gs, set(GC["row_id"]), V.MODEL_LABEL, V.MARKET_LABEL), gcfg)
        st.caption("Rates are re-derived from the stored sums. For rare events (props, combos, goal bands) the most "
                   "likely outcome is usually NO, so success is high by construction: read Brier and log loss.")

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
