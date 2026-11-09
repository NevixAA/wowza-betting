"""Specials (research) - high-scoring, 0-0 / low and unusual exact-score matches (owner 2026-11-09).

Pro's specials tracker (wowzaV9-Pro src/specials) logs EVERY upcoming fixture its score model can rate,
once, before kickoff: the full exact-score grid 0-0..6-6, the goal and margin bands, and each one against
THAT league's own history. Fixtures whose probability is unusually high for their league are flagged; the
result is graded beside the frozen row after the match. RESEARCH ONLY: nothing here is a tip or a stake.

A VIEWER: every number is read from Pro's committed files (dashboard_data/pro.specials). When the tracker
has not written anything yet the page says so; it never shows sample data.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - research tier, like the League Repair Lab

import dashboard_ui as ui  # noqa: E402
from dashboard_data import pro as PRO  # noqa: E402
from dashboard_data import specials_view as V  # noqa: E402

import importlib  # noqa: E402
PRO = importlib.reload(PRO)
V = importlib.reload(V)

st.set_page_config(page_title="Wowza | Specials", page_icon="🎲", layout="wide")

ui.page_header("Specials",
               "Matches the score model sees as unusually high-scoring, 0-0 / low-scoring or likely to end in an "
               "unusual exact score, each against the league's own history. Research only - collected to learn, "
               "never a tip or a stake.",
               eyebrow_text="Research Lab · Pro specials tracker", state="RESEARCH")

if not PRO.available():
    ui.absent(PRO.absent_reason())
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def _load():
    return PRO.specials()


data = _load()
log, settled = data.get("specials_log", pd.DataFrame()), data.get("specials_settled", pd.DataFrame())
study, ev = data.get("study") or {}, data.get("eval") or {}
rule = study.get("rule") or ev.get("rule") or {}
min_events = rule.get("min_candidate_events")
now = pd.Timestamp.now(tz="UTC")

if log.empty and not study:
    ui.empty_state("NO_DATA", "The specials tracker has not written anything yet. It is switched off until the "
                              "owner sets PRO_SPECIALS_ENABLED; the first daily run logs every rateable fixture.")
    st.stop()

quantile = rule.get("candidate_quantile")
if rule and quantile is not None:
    st.caption(f"Rule {rule.get('version')}: a fixture is flagged when its probability is above the league's "
               f"{int(round(100 * quantile))}th percentile of the model's own earlier "
               f"out-of-sample predictions for that outcome. Base rates are the league's own matches before the "
               f"log date - never a global rate.")

# ── flagged now ──────────────────────────────────────────────────────────────────────────────
ui.section_header("Flagged now", "Upcoming fixtures only, as logged before kickoff.")
if log.empty:
    ui.empty_state("NO_DATA", "No fixtures logged yet.")
else:
    for title, group in (("High-scoring", V.HIGH), ("0-0 and low-scoring", V.LOW),
                         ("Unusual exact score", ("cs_unusual",))):
        f = V.flagged_frame(log, now, group)
        st.markdown(f"**{title}**")
        if f.empty:
            st.caption("Nothing flagged in the upcoming fixtures.")
        else:
            ui.table(f, column_config={"model": ui.pct_col("Model"), "league base": ui.pct_col("League base"),
                                       "ratio": ui.num_col("Ratio", fmt="%.2f×")})

# ── one match: the grid and the bands ────────────────────────────────────────────────────────
ui.section_header("Exact-score grid", "Pick a logged fixture: the model's probability per score, 0-0 to 6-6, "
                  "against the league's own frequency.")
if not log.empty:
    lg_log = log.assign(_ko=pd.to_datetime(log["kickoff_utc"], utc=True, errors="coerce")).sort_values(
        "_ko", ascending=False)
    labels = [f"{str(r['kickoff_utc'])[:10]} · {r['league']} · {V.match_label(r)}" for _, r in lg_log.iterrows()]
    pick = st.selectbox("Fixture", range(len(labels)), format_func=lambda i: labels[i])
    row = lg_log.iloc[int(pick)]
    g = V.grid_frame(row)
    if g.empty:
        ui.empty_state("NO_DATA", "This row has no grid columns.")
    else:
        import altair as alt
        t = ui.tokens()
        base = alt.Chart(g).encode(
            x=alt.X("away:O", title="away goals", axis=alt.Axis(orient="top", labelAngle=0)),
            y=alt.Y("home:O", title="home goals"))
        tips = [alt.Tooltip("score:N"), alt.Tooltip("model:Q", format=".2%"),
                alt.Tooltip("league base:Q", format=".2%"), alt.Tooltip("ratio:Q", format=".2f")]
        rect = base.mark_rect(cornerRadius=4, stroke=t["BORDER"]).encode(
            color=alt.Color("model:Q", scale=alt.Scale(range=[t["SURFACE"], t["POSITIVE"]]),
                            legend=alt.Legend(title="model", format="%")), tooltip=tips)
        text = base.mark_text(fontSize=11).encode(text=alt.Text("model:Q", format=".1%"),
                                                  color=alt.value(t["TEXT_PRIMARY"]), tooltip=tips)
        st.altair_chart((rect + text).properties(height=360), width="stretch")
        other = row.get("p_cs_other")
        st.caption(f"Scores outside the grid (a side scoring 7+): {ui.fmt_prob(other, 2)} model · "
                   f"{ui.fmt_prob(row.get('base_cs_other'), 2)} league base. Logged {row.get('logged_at')} · "
                   f"{row.get('n_league_history')} league matches of history.")
    ui.table(V.bands_frame(row), column_config={"model": ui.pct_col("Model"),
                                                "league base": ui.pct_col("League base"),
                                                "ratio": ui.num_col("Ratio", fmt="%.2f×")})

# ── graded live record ───────────────────────────────────────────────────────────────────────
ui.section_header("Graded record (live)", "Flagged fixtures that have finished, per league and special: how "
                  "many happened against the league base rate of those same fixtures.")
gr = V.graded_frame(log, settled, min_events)
if gr.empty:
    ui.empty_state("NO_DATA", "No flagged fixture has been graded yet.")
else:
    floor = (f"INSUFFICIENT until a cell has {min_events} events. " if min_events is not None
             else "NO_FLOOR: Pro has not published its rule yet. ")
    st.caption(floor + "In v9's repaired history 6+ goal matches are 3-9% of a league's games and 0-0 is 5-12%, "
               "so 0-0 evidence builds faster.")
    ui.table(gr, column_config={"hit rate": ui.pct_col("Hit rate"), "league base": ui.pct_col("League base")})

# ── the history study ────────────────────────────────────────────────────────────────────────
ui.section_header("History study", "Walk-forward over v9's repaired history: does the model spot these matches "
                  "better than each league's own base rate?")
st_t = V.study_table(study)
if st_t.empty:
    ui.empty_state("NO_DATA", "The history study has not been written yet.")
else:
    st.caption(f"{study.get('rows')} out-of-sample predictions, {study.get('leagues')} leagues, "
               f"{study.get('date_min')} to {study.get('date_max')}. Holm correction across "
               f"{study.get('family_size')} candidate tests. Brier skill above 0 = better than the league base.")
    ui.table(st_t, column_config={"candidate hit": ui.pct_col("Candidate hit"), "their base": ui.pct_col("Their base"),
                                  "Brier skill vs base": ui.num_col("Brier skill vs base", fmt="%+.4f"),
                                  "lift": ui.num_col("Lift", fmt="%.2f×"), "p (Holm)": ui.num_col("p (Holm)", fmt="%.4f")})

grid = data.get("study_calibration_grid", pd.DataFrame())
if not grid.empty:
    ui.section_header("Score-grid calibration", "Per league: the model's average probability for each exact score "
                      "against how often it actually happened (out of sample).")
    leagues = sorted(grid["league"].unique(), key=lambda x: (x != "ALL", x))
    lg = st.selectbox("League", leagues)
    ui.table(V.calibration_frame(grid, lg),
             column_config={"mean_p": ui.pct_col("Model"), "actual": ui.pct_col("Actual"),
                            "base_mean": ui.pct_col("League base"),
                            "gap": ui.num_col("Actual − model", fmt="%+.3f")})
