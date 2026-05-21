"""
Wowza Betting Dashboard — v9
Main entry point. Run with: streamlit run app.py
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

st.set_page_config(
    page_title="Wowza | Betting Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh every 5 minutes ──────────────────────────────────────────────
st_autorefresh(interval=5 * 60 * 1000, key="main_refresh")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .sniper-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #e94560;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .value-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #f5a623;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .metric-positive { color: #00c896; font-weight: bold; }
  .metric-negative { color: #e94560; font-weight: bold; }
  .tier-sniper { color: #e94560; font-weight: bold; }
  .tier-value  { color: #f5a623; font-weight: bold; }
  .drift-confirmed  { color: #00c896; }
  .drift-conflicted { color: #e94560; }
  .drift-neutral    { color: #aaa; }
  .drift-new        { color: #888; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_bets():
    f = config.OUTPUT_DIR / "bets.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_ledger():
    f = config.OUTPUT_DIR / "bets_ledger.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, dtype=str)
    if "source" not in df.columns:
        df["source"] = "live"
    else:
        df["source"] = df["source"].fillna("live")
    df["pnl"] = pd.to_numeric(df.get("pnl", pd.Series()), errors="coerce")
    df["edge_pct"] = pd.to_numeric(df.get("edge_pct", pd.Series()), errors="coerce")
    return df


# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title, col_time = st.columns([1, 6, 3])
with col_logo:
    st.markdown("## 🎯")
with col_title:
    st.markdown("# Wowza Betting Dashboard")
with col_time:
    import datetime
    st.markdown(f"<br><small>Last refresh: {datetime.datetime.now().strftime('%d/%m %H:%M')}</small>",
                unsafe_allow_html=True)

st.divider()

# ── Quick stats ───────────────────────────────────────────────────────────────
bets    = load_bets()
ledger  = load_ledger()

live = ledger[ledger["source"] == "live"] if not ledger.empty else pd.DataFrame()
scored = live[live["pnl"].notna()] if not live.empty else pd.DataFrame()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    n_sniper = len(bets[bets["signal_tier"] == "SNIPER"]) if not bets.empty else 0
    st.metric("🎯 Live SNIPERs", n_sniper)

with c2:
    n_value = len(bets[(bets["signal_tier"] == "VALUE") & (bets["bet"].isin(["OVER","UNDER"]))]) if not bets.empty else 0
    st.metric("💡 Live VALUE", n_value)

with c3:
    if not scored.empty:
        roi = scored["pnl"].sum() / len(scored) * 100
        st.metric("📈 Live ROI", f"{roi:+.1f}%")
    else:
        st.metric("📈 Live ROI", "—")

with c4:
    if not scored.empty:
        wr = (scored["pnl"] > 0).mean() * 100
        st.metric("✅ Win Rate", f"{wr:.1f}%")
    else:
        st.metric("✅ Win Rate", "—")

with c5:
    if not scored.empty:
        st.metric("📊 Resolved Bets", len(scored))
    else:
        st.metric("📊 Resolved Bets", "0")

st.divider()
st.markdown("### Navigate using the sidebar →")
st.markdown("""
| Page | What you'll find |
|---|---|
| 📊 Dashboard | Today's SNIPER & VALUE tips with drift signals |
| 📈 Performance | ROI charts, SNIPER vs VALUE breakdown, by league |
| 📋 Ledger | Full bet history — filter, search, export |
| ℹ️ Model Info | How the two models work, feature importances |
""")
