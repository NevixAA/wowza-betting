"""
Live Signal History
Track all past live scanner signals to evaluate if the tool is useful.
"""
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Live History", page_icon="📜", layout="wide")

BASE_DIR     = Path(__file__).resolve().parents[1]
HISTORY_FILE = BASE_DIR / "output" / "live_signals_history.csv"

st.title("📜 Live Signal History")
st.caption("All past in-play signals — track which signal types fire most and how games ended.")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=120)
def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date", ascending=False).reset_index(drop=True)

df = load_history()

if df.empty:
    st.info("⏳ No live signal history yet. Signals are logged automatically when the live scanner fires during match hours.")
    st.markdown("""
    **What will appear here:**
    - Every live signal that fired (UNDER_HOLD, SLEEPING_GAME, COMEBACK etc.)
    - Match, score, elapsed time, fair price at the time
    - As you manually add results you'll be able to see which signal types are profitable
    """)
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total     = len(df)
matches   = df["match"].nunique() if "match" in df.columns else 0
date_from = df["date"].min().strftime("%b %d") if not df.empty else "—"
date_to   = df["date"].max().strftime("%b %d, %Y") if not df.empty else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Signals",   total)
c2.metric("Unique Matches",  matches)
c3.metric("First Signal",    date_from)
c4.metric("Latest Signal",   date_to)

st.markdown("---")

# ── Signal type breakdown ──────────────────────────────────────────────────────
st.subheader("📊 Signal Type Breakdown")

if "signal_type" in df.columns:
    sig_counts = df["signal_type"].value_counts().reset_index()
    sig_counts.columns = ["Signal Type", "Count"]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(sig_counts, use_container_width=True, hide_index=True)
    with col2:
        fig = px.bar(sig_counts, x="Signal Type", y="Count",
                     color="Count", color_continuous_scale=["#1a1a2e", "#e94560"],
                     title="Signal Frequency")
        fig.update_layout(template="plotly_dark", coloraxis_showscale=False,
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Filter ────────────────────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    sig_types = sorted(df["signal_type"].unique()) if "signal_type" in df.columns else []
    sig_filter = st.multiselect("Signal type", sig_types, default=sig_types)
with col_f2:
    leagues = sorted(df["league"].unique()) if "league" in df.columns else []
    league_filter = st.multiselect("League", leagues, default=leagues)

filtered = df.copy()
if sig_filter and "signal_type" in filtered.columns:
    filtered = filtered[filtered["signal_type"].isin(sig_filter)]
if league_filter and "league" in filtered.columns:
    filtered = filtered[filtered["league"].isin(league_filter)]

st.subheader(f"📋 {len(filtered)} Signal(s)")

# ── Table ─────────────────────────────────────────────────────────────────────
display_cols = ["date", "league", "match", "score", "elapsed_mins",
                "signal_type", "bet", "fair_under_odds", "fair_over_odds",
                "live_p_under", "live_p_over", "pre_p_over"]
show_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[show_cols].rename(columns={
        "elapsed_mins":   "min",
        "signal_type":    "signal",
        "fair_under_odds":"fair_U",
        "fair_over_odds": "fair_O",
        "live_p_under":   "P(U)",
        "live_p_over":    "P(O)",
        "pre_p_over":     "pre P(O)",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
st.caption("History grows as the live scanner runs during match hours · "
           "Add a 'result' column manually when you know match outcomes to track edge")
