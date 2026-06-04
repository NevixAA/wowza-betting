"""
Half-Time O/U Predictions Dashboard
Model predictions for HT OVER/UNDER 0.5 and 1.5 — standard-format leagues only.
"""
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="HT Predictions", page_icon="⏱", layout="wide")

BASE_DIR   = Path(__file__).resolve().parents[1]
PREDS_FILE = BASE_DIR / "output" / "predictions.csv"

components.html("<script>setTimeout(()=>window.location.reload(),120000)</script>", height=0)

st.title("⏱ Half-Time O/U Predictions")
st.caption("Model predictions for HT OVER/UNDER 0.5 and 1.5 · Standard-format leagues only · Updated every 4h")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

with st.expander("ℹ️ How this works"):
    st.markdown("""
    **Two HT models trained on 48,000+ matches** from 23 standard-format leagues (those with half-time score data).

    Key features used:
    - **Team HT tendency** — % of recent games where a team's matches had a goal / 2 goals in the first half
    - **HT rolling goals** — average HT goals scored/conceded in last 5 games
    - **HT attack/defense strength** — how the team's HT performance compares to league average

    **How to use:**
    - P(HT OVER 0.5) > 75% → look for HT OVER 0.5 at your bookmaker
    - P(HT UNDER 0.5) > 70% (i.e., P(over) < 30%) → look for HT UNDER 0.5
    - Compare our fair price against your bookmaker's HT market price

    ⚠️ *These models are new and still building track record. Use as supplementary signal, not primary.*
    """)

@st.cache_data(ttl=120)
def load_ht():
    if not PREDS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDS_FILE)
    if "p_ht_over05" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    df = df[df["date"] >= today]
    df = df[df["p_ht_over05"].notna()].copy()
    df["p_ht_over05"] = pd.to_numeric(df["p_ht_over05"], errors="coerce")
    df["p_ht_over15"] = pd.to_numeric(df.get("p_ht_over15"), errors="coerce")
    return df.sort_values("p_ht_over05", ascending=False).reset_index(drop=True)

df = load_ht()

if df.empty:
    st.info("⏳ No HT predictions available. Only standard-format leagues (League One, Bundesliga 2, La Liga 2, etc.) have HT data.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────
strong_over05 = df[df["p_ht_over05"] >= 0.75]
strong_under05 = df[df["p_ht_over05"] <= 0.30]
strong_over15  = df[df["p_ht_over15"] >= 0.60] if "p_ht_over15" in df.columns else pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Fixtures",      len(df))
c2.metric("⚡ Strong OVER 0.5",  len(strong_over05), help="P(HT OVER 0.5) ≥ 75%")
c3.metric("🧊 Strong UNDER 0.5", len(strong_under05), help="P(HT OVER 0.5) ≤ 30%")
c4.metric("🔥 Strong OVER 1.5",  len(strong_over15), help="P(HT OVER 1.5) ≥ 60%")

st.markdown("---")

# ── Filter ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])
with col1:
    show = st.selectbox("Show", ["All signals", "Strong OVER 0.5 only", "Strong UNDER 0.5 only", "Strong OVER 1.5 only", "All fixtures"])
with col2:
    leagues = sorted(df["league"].unique())
    league_filter = st.multiselect("League", leagues, default=leagues)

df = df[df["league"].isin(league_filter)]

if show == "Strong OVER 0.5 only":
    df = df[df["p_ht_over05"] >= 0.75]
elif show == "Strong UNDER 0.5 only":
    df = df[df["p_ht_over05"] <= 0.30]
elif show == "Strong OVER 1.5 only":
    df = df[df["p_ht_over15"] >= 0.60]
elif show == "All signals":
    df = df[(df["p_ht_over05"] >= 0.75) | (df["p_ht_over05"] <= 0.30) |
            (df["p_ht_over15"] >= 0.60) if "p_ht_over15" in df.columns
            else (df["p_ht_over05"] >= 0.75) | (df["p_ht_over05"] <= 0.30)]

if df.empty:
    st.info("No fixtures match your filters.")
    st.stop()

# ── Cards ─────────────────────────────────────────────────────────────────────
st.subheader(f"📋 {len(df)} Fixture(s)")

for _, row in df.iterrows():
    p05  = float(row["p_ht_over05"])
    p15  = float(row["p_ht_over15"]) if pd.notna(row.get("p_ht_over15")) else None
    fair_over05  = round(1 / max(p05, 0.01), 2)
    fair_under05 = round(1 / max(1 - p05, 0.01), 2)
    fair_over15  = round(1 / max(p15, 0.01), 2) if p15 else None
    fair_under15 = round(1 / max(1 - p15, 0.01), 2) if p15 else None

    # Signal detection
    signals = []
    if p05 >= 0.75:   signals.append(("⚡", "#ffdd00", f"HT OVER 0.5  ({p05*100:.0f}%) — fair {fair_over05}"))
    if p05 <= 0.30:   signals.append(("🧊", "#44aaff", f"HT UNDER 0.5 ({(1-p05)*100:.0f}%) — fair {fair_under05}"))
    if p15 and p15 >= 0.60: signals.append(("🔥", "#ff6600", f"HT OVER 1.5  ({p15*100:.0f}%) — fair {fair_over15}"))
    if p15 and p15 <= 0.25: signals.append(("🔒", "#00cc88", f"HT UNDER 1.5 ({(1-p15)*100:.0f}%) — fair {fair_under15}"))

    border = signals[0][1] if signals else "#444"
    signal_badges = " ".join(f'<span style="background:{c}22;color:{c};padding:2px 8px;border-radius:10px;font-size:0.82em">{e} {t}</span>' for e, c, t in signals) if signals else '<span style="color:#666">No strong signal</span>'

    st.markdown(f"""
    <div style="border-left:4px solid {border};padding:12px 16px;margin:6px 0;
                background:#111827;border-radius:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <b style="color:white;font-size:1.05em">{row['home_team']} vs {row['away_team']}</b>
            <span style="color:#aaa;font-size:0.85em">📅 {str(row['date'])[:10]}</span>
        </div>
        <div style="color:#90caf9;font-size:0.85em;margin:3px 0">🏆 {row.get('league','')}</div>
        <div style="margin:8px 0">{signal_badges}</div>
        <div style="display:flex;gap:24px;margin-top:6px">
            <div>
                <span style="color:#aaa;font-size:0.8em">P(HT OVER 0.5)</span><br>
                <b style="color:{'#ffdd00' if p05>=0.75 else '#44aaff' if p05<=0.30 else 'white'}">{p05*100:.0f}%</b>
            </div>
            <div>
                <span style="color:#aaa;font-size:0.8em">Fair OVER 0.5</span><br>
                <b style="color:white">{fair_over05}</b>
            </div>
            <div>
                <span style="color:#aaa;font-size:0.8em">Fair UNDER 0.5</span><br>
                <b style="color:white">{fair_under05}</b>
            </div>
            {'<div><span style="color:#aaa;font-size:0.8em">P(HT OVER 1.5)</span><br><b style="color:white">' + f'{p15*100:.0f}%' + '</b></div>' if p15 else ''}
            {'<div><span style="color:#aaa;font-size:0.8em">Fair OVER 1.5</span><br><b style="color:white">' + str(fair_over15) + '</b></div>' if p15 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
with st.expander("📊 Raw data"):
    cols = ["date","league","home_team","away_team","p_ht_over05","p_ht_over15"]
    st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True)

st.caption("AUC 0.567 (HT OVER 0.5) · AUC 0.569 (HT OVER 1.5) · 48k training rows · 23 leagues")
