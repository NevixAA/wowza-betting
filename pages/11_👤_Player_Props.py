"""
Player Props Dashboard
Shows SNIPER/MARKSMAN/VALUABLE player prop signals.
"""
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Player Props | Wowza", page_icon="👤", layout="wide")
components.html("<script>setTimeout(()=>window.location.reload(),120000)</script>", height=0)

BASE_DIR   = Path(__file__).resolve().parents[1]
TIPS_FILE  = BASE_DIR / "output" / "player_tips.csv"

st.title("👤 Player Props")
st.caption("SNIPER/MARKSMAN/VALUABLE player signals — SOT, Goals, Cards, Assists")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

TIER_META = {
    "SNIPER":   ("🎯", "#e94560", "EV > 40% · Odds ≥ 5.0 · 2+ lazy factors"),
    "MARKSMAN": ("🔫", "#00aaff", "EV > 25% · Odds ≥ 4.0 · 1+ lazy factor"),
    "VALUABLE": ("💎", "#f5a623", "EV > 15% · Odds ≥ 3.0"),
    "WATCH":    ("👁",  "#666",   "Below threshold — monitor only"),
}

MARKET_EMOJI = {
    "goals":   "⚽", "sot": "🎯", "cards": "🟨",
    "assists": "🅰️", "corners": "🔄", "fouls": "👊",
}

@st.cache_data(ttl=120)
def load_tips():
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TIPS_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    return df[df["date"] >= today].copy()

df = load_tips()

if df.empty:
    st.info("⏳ No player prop signals yet. The player props model runs 30 min after each predict.")
    st.markdown("""
    **What this page shows when active:**
    - SOT, Goals, Cards, Assists signals per player per match
    - Only for SNIPER/MARKSMAN matches (our team model already flagged the game)
    - De-vigged EV with lazy market factor detection
    - Signals for: Championship, League One, Bundesliga 2, Ireland, Finland

    **Requires:** APIFOOTBALL_KEY set in GitHub Secrets for full rolling stats.
    Until then, uses season-average stats from FBref.
    """)
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
snipers   = df[df["tier"] == "SNIPER"]
marksmen  = df[df["tier"] == "MARKSMAN"]
valuables = df[df["tier"] == "VALUABLE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Signals", len(df[df["tier"] != "WATCH"]))
c2.metric("🎯 SNIPER",    len(snipers))
c3.metric("🔫 MARKSMAN",  len(marksmen))
c4.metric("💎 VALUABLE",  len(valuables))

st.markdown("---")

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    tier_filter = st.multiselect("Tier", ["SNIPER","MARKSMAN","VALUABLE"],
                                 default=["SNIPER","MARKSMAN"])
with col2:
    mkt_filter = st.multiselect("Market", sorted(df["market"].unique()),
                                default=list(df["market"].unique()))
with col3:
    min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.50, 0.05)

filtered = df[
    df["tier"].isin(tier_filter) &
    df["market"].isin(mkt_filter) &
    (df["confidence"] >= min_confidence)
].copy()

if filtered.empty:
    st.info("No signals match filters.")
    st.stop()

st.subheader(f"📋 {len(filtered)} Signal(s)")

# ── Signal cards ──────────────────────────────────────────────────────────────
for _, row in filtered.sort_values(["tier", "model_prob"], ascending=[True, False]).iterrows():
    emoji, color, desc = TIER_META.get(row["tier"], ("📌","#888",""))
    mkt_emoji = MARKET_EMOJI.get(row["market"], "📊")
    fair_odds = row.get("fair_odds", "—")
    mkt_odds  = row.get("market_odds")
    ev_val    = row.get("ev")
    conf      = float(row.get("confidence", 0))
    lazy      = str(row.get("lazy_factors", ""))
    n_games   = int(row.get("n_games", 0))

    # EV display
    ev_str  = f"EV: {ev_val:+.1%}" if ev_val is not None and not pd.isna(ev_val) else "EV: add odds"
    odds_str = f"@ {mkt_odds:.2f}" if mkt_odds and not pd.isna(mkt_odds) else f"fair {fair_odds}"
    lazy_badges = " ".join(
        f'<span style="background:#333;color:#aaa;padding:1px 6px;border-radius:8px;font-size:0.75em">{f}</span>'
        for f in lazy.split("|") if f
    )

    st.markdown(f"""
    <div style="border-left:4px solid {color};padding:12px 16px;margin:6px 0;
                background:#111827;border-radius:6px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:{color};font-weight:bold">{emoji} {row['tier']}</span>
            <span style="color:#aaa;font-size:0.85em">📅 {str(row['date'])[:10]}</span>
        </div>
        <b style="color:white;font-size:1.05em">{row['player_name']}</b>
        <span style="color:#aaa"> · {row['position']} · {row['team']}</span><br/>
        <span style="color:#90caf9">{row['league']} — {row['match']}</span><br/>
        <div style="margin:6px 0">
            <span style="color:white">{mkt_emoji} <b>{row['market'].upper()}</b></span>
            &nbsp;{odds_str}
            &nbsp;|&nbsp;<span style="color:#00cc88"><b>{ev_str}</b></span>
            &nbsp;|&nbsp;Model P: <b style="color:white">{row['model_prob']:.0%}</b>
        </div>
        <div>{lazy_badges}</div>
        <div style="color:#555;font-size:0.8em;margin-top:4px">
            Confidence: {conf:.0%} · Data: {n_games} games · {row.get('data_source','')}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
with st.expander("📊 Raw data"):
    show_cols = ["date","league","match","player_name","position","team",
                 "market","model_prob","fair_odds","market_odds","ev",
                 "tier","confidence","lazy_factors","n_games"]
    st.dataframe(filtered[[c for c in show_cols if c in filtered.columns]],
                 use_container_width=True, hide_index=True)

st.caption("Player props · Phase 1 · FBref season stats · API-Football rolling stats (requires APIFOOTBALL_KEY)")
