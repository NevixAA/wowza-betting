"""
Sharp Money Tracker Dashboard
Odds drift signals across all enabled leagues — where is the money going?
"""
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sharp Money", page_icon="💰", layout="wide")

BASE_DIR   = Path(__file__).resolve().parents[1]
TIPS_FILE  = BASE_DIR / "output" / "sharp_tips.csv"
HIST_FILE  = BASE_DIR / "output" / "sharp_history.json"

SIGNAL_META = {
    "STEAM_STRONG": ("🔥", "#ff2222", "Steam + strong — fast money AND >10% total drift"),
    "STEAM_SHARP":  ("⚡", "#ff8800", "Steam + sharp — fast move in last window"),
    "STRONG":       ("🔴", "#ff4444", "Strong sharp money — >10% drift"),
    "SHARP":        ("🟡", "#ffaa00", "Notable move — 5–10% drift"),
    "FADING":       ("⬆️", "#888888", "Odds lengthening — money moving away"),
}

st.title("💰 Sharp Money Tracker")
st.caption("Odds drift signals across all enabled leagues. Updated every 2 hours.")

components_html = "<script>setTimeout(()=>window.location.reload(),120000)</script>"
import streamlit.components.v1 as components
components.html(components_html, height=0)

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=120)
def load_tips():
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(TIPS_FILE)

df = load_tips()

if df.empty:
    st.info("⏳ No sharp money signals yet. The tracker runs every 2 hours — check back soon.")
    st.markdown("""
    **How it works:**
    - Every 2 hours we snapshot odds for all upcoming games across 20 leagues
    - When odds move significantly from opening, it signals where money is going
    - **STRONG (>10%)** = high-confidence sharp money
    - **SHARP (5–10%)** = notable movement worth watching
    - **FADING** = odds drifting out (square/public money, avoid)
    """)
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
strong = df[df["signal"] == "STRONG"]
sharp  = df[df["signal"] == "SHARP"]
fading = df[df["signal"] == "FADING"]

steam = df[df["signal"].isin(["STEAM_STRONG", "STEAM_SHARP"])]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Signals",   len(df))
c2.metric("🔥 STEAM",        len(steam))
c3.metric("🔴 STRONG",       len(strong))
c4.metric("🟡 SHARP",        len(sharp))
c5.metric("Last Update",     df["updated_at"].iloc[0] if not df.empty else "—")

st.markdown("---")

# ── Filter ────────────────────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    sig_filter = st.multiselect("Signal", ["STEAM_STRONG", "STEAM_SHARP", "STRONG", "SHARP", "FADING"],
                                default=["STEAM_STRONG", "STEAM_SHARP", "STRONG", "SHARP"])
with col_f2:
    league_filter = st.multiselect("League", sorted(df["league"].unique()),
                                   default=list(df["league"].unique()))

filtered = df[df["signal"].isin(sig_filter) & df["league"].isin(league_filter)]

if filtered.empty:
    st.info("No signals match your filters.")
    st.stop()

# ── Signal cards ──────────────────────────────────────────────────────────────
st.subheader(f"📊 {len(filtered)} Signal(s)")

for _, row in filtered.iterrows():
    emoji, color, desc = SIGNAL_META.get(row["signal"], ("📌", "#888", row["signal"]))
    direction  = "▼ Sharp money IN" if row["drift_pct"] < 0 else "▲ Money moving OUT"
    dir_color  = "#00cc88" if row["drift_pct"] < 0 else "#ff6666"

    st.markdown(f"""
    <div style="border-left:4px solid {color}; padding:14px 18px; margin:8px 0;
                background:#111827; border-radius:6px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:1.1em; font-weight:bold; color:white">
                {emoji} {row['match']}
            </span>
            <span style="background:{color}22; color:{color}; padding:3px 10px;
                         border-radius:12px; font-size:0.85em; font-weight:bold">
                {row['signal']}
            </span>
        </div>
        <div style="color:#aaa; font-size:0.88em; margin:4px 0">
            🏆 {row['league']} &nbsp;|&nbsp; 📅 {row['date']} &nbsp;|&nbsp; 📌 {row['market']}
        </div>
        <div style="display:flex; gap:24px; margin:10px 0; flex-wrap:wrap">
            <div>
                <span style="color:#aaa; font-size:0.82em">Opening</span><br>
                <b style="color:#888; font-size:1.05em">{row['opening_odds']}</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.82em">Current</span><br>
                <b style="color:white; font-size:1.05em">{row['current_odds']}</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.82em">Drift</span><br>
                <b style="color:{color}; font-size:1.1em">{row['drift_pct']:+.1f}%</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.82em">Consensus</span><br>
                <b style="color:{'#00cc88' if row.get('consensus_pct',0)>=70 else '#ffaa00' if row.get('consensus_pct',0)>=50 else '#888'}">{row.get('consensus_pct', '—')}%</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.82em">Books</span><br>
                <b style="color:#888">{row.get('n_books', '—')}</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.82em">Snapshots</span><br>
                <b style="color:#888">{row['snapshots']}</b>
            </div>
        </div>
        <div style="color:{dir_color}; font-size:0.85em">{direction}{'&nbsp;&nbsp;🔥 <b>STEAM</b> — fast move detected' if row.get('steam') else ''}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Raw table ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Raw data"):
    st.dataframe(filtered, use_container_width=True)

st.caption("Updates every 2 hours · 20 leagues tracked · STRONG = >10% odds move from opening")
st.info("⚠️ **Note:** Opening odds = first snapshot when pipeline saw this fixture (not true market opening). "
        "A move shown here may have already happened before our system first observed it. "
        "For genuine sharp money detection, true opening odds from a historical odds API are required.")
