"""
Live Signal History
Track all past live scanner signals to evaluate if the tool is useful.
"""
from pathlib import Path
from datetime import datetime

import sys

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Live History", page_icon="📜", layout="wide")

BASE_DIR     = Path(__file__).resolve().parents[1]
HISTORY_FILE = BASE_DIR / "output" / "live_signals_history.csv"
sys.path.insert(0, str(BASE_DIR))

st.title("📜 Live Signal History")
st.caption("All past in-play signals — track which signal types fire most and whether they'd have won.")

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


def _norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


@st.cache_data(ttl=1800, show_spinner="Loading final scores…")
def load_results_lookup():
    """Map (home, away, date) -> (final total goals, HT total goals) from match data.
    Used to settle each past live signal against the actual full-time / half-time result."""
    try:
        from src.data_loader import load_all_matches
        m = load_all_matches()
    except Exception:
        return {}
    if m is None or m.empty:
        return {}
    m = m.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    if "total_goals" not in m.columns and {"home_goals", "away_goals"}.issubset(m.columns):
        m["total_goals"] = m["home_goals"] + m["away_goals"]
    look = {}
    for r in m.itertuples(index=False):
        h, a = getattr(r, "home_team", ""), getattr(r, "away_team", "")
        d = getattr(r, "date", None)
        if pd.isna(d):
            continue
        key = (_norm(h), _norm(a), pd.Timestamp(d).strftime("%Y-%m-%d"))
        ht = getattr(r, "ht_total_goals", None)
        look[key] = (getattr(r, "total_goals", None),
                     ht if ht is not None and not pd.isna(ht) else None)
    return look


def _settle(bet: str, total, ht_total):
    """Return WIN / LOSS / None(=pending) for a live bet given the final result."""
    b = str(bet).upper()
    if "HT" in b:
        if ht_total is None or pd.isna(ht_total):
            return None
        if "OVER 0.5"  in b: return "WIN" if ht_total >= 1 else "LOSS"
        if "UNDER 0.5" in b: return "WIN" if ht_total == 0 else "LOSS"
        if "OVER 1.5"  in b: return "WIN" if ht_total >= 2 else "LOSS"
        if "UNDER 1.5" in b: return "WIN" if ht_total <= 1 else "LOSS"
        return None
    if total is None or pd.isna(total):
        return None
    if "OVER 2.5"  in b: return "WIN" if total >= 3 else "LOSS"
    if "UNDER 2.5" in b: return "WIN" if total <= 2 else "LOSS"
    return None


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

# ── Settle each signal against the final result ───────────────────────────────
lookup = load_results_lookup()


def _lookup_result(row):
    m = str(row.get("match", ""))
    if " vs " not in m or pd.isna(row["date"]):
        return (None, None)
    h, a = m.split(" vs ", 1)
    for off in (0, -1, 1):   # tolerate ±1 day (kickoff timezone drift)
        key = (_norm(h), _norm(a),
               (pd.Timestamp(row["date"]) + pd.Timedelta(days=off)).strftime("%Y-%m-%d"))
        if key in lookup:
            return lookup[key]
    return (None, None)


_res = df.apply(_lookup_result, axis=1, result_type="expand")
df["final_total"] = _res[0]
_ht = _res[1]
df["result"] = [_settle(b, t, h) for b, t, h in zip(df["bet"], df["final_total"], _ht)]

# ── Results / hit-rate ─────────────────────────────────────────────────────────
settled = df[df["result"].isin(["WIN", "LOSS"])]
pending = int((~df["result"].isin(["WIN", "LOSS"])).sum())
st.subheader("✅ Results — did the live signals win?")
if settled.empty:
    st.info(f"No signals settled yet — {pending} awaiting final scores "
            "(World Cup / matches not in the results feed stay pending).")
else:
    n = len(settled)
    wins = int((settled["result"] == "WIN").sum())
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Settled", n)
    r2.metric("Record", f"{wins}W – {n - wins}L")
    r3.metric("Hit rate", f"{wins / n * 100:.0f}%")
    r4.metric("Pending", pending)
    # per-signal-type hit rate
    by_type = (settled.assign(win=(settled["result"] == "WIN").astype(int))
               .groupby("signal_type")
               .agg(bets=("win", "size"), wins=("win", "sum"))
               .reset_index())
    by_type["hit_rate_%"] = (by_type["wins"] / by_type["bets"] * 100).round(0)
    st.dataframe(by_type.sort_values("bets", ascending=False),
                 use_container_width=True, hide_index=True)
    st.caption("⚠️ Small sample — treat as directional until many more signals accumulate. "
               "Under/Over 2.5 settled on full-time goals; HT bets on half-time goals.")

st.markdown("---")

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
                "signal_type", "bet", "result", "final_total",
                "fair_under_odds", "fair_over_odds",
                "live_p_under", "live_p_over", "pre_p_over"]
show_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[show_cols].rename(columns={
        "elapsed_mins":   "min",
        "signal_type":    "signal",
        "final_total":    "FT goals",
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
