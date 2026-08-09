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

# Performance COUNTS start at the cutoff (pre-fix new-format tips were garbage). Rows before
# the cutoff stay in the ledger for CLV — they're just excluded from these aggregates.
_cut = config.PERFORMANCE_CUTOFF_DATE
if not live.empty and "generated_at" in live.columns:
    counted = live[live["generated_at"].astype(str).str[:10] >= _cut]
else:
    counted = live
scored = counted[counted["pnl"].notna()] if not counted.empty else pd.DataFrame()

# Tiers are SNIPER / MARKSMAN / VALUABLE (not "VALUE").
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    n_sniper = len(bets[bets["signal_tier"] == "SNIPER"]) if not bets.empty else 0
    st.metric("🎯 Live SNIPERs", n_sniper)

with c2:
    n_value = len(bets[(bets["signal_tier"] == "VALUABLE") & (bets["bet"].isin(["OVER", "UNDER"]))]) if not bets.empty else 0
    st.metric("💎 Live VALUABLE", n_value)

with c3:
    # P/L in UNITS on flat 1u paper stakes — NOT "ROI" (no real capital/staking).
    if not scored.empty:
        pl = scored["pnl"].sum()
        st.metric("📈 Live P/L (u)", f"{pl:+.2f}u")
    else:
        st.metric("📈 Live P/L (u)", "—")

with c4:
    if not scored.empty:
        wr = (scored["pnl"] > 0).mean() * 100
        st.metric("✅ Win Rate", f"{wr:.1f}%")
    else:
        st.metric("✅ Win Rate", "—")

with c5:
    st.metric("📊 Resolved Bets", len(scored) if not scored.empty else 0)

st.caption(f"📅 Performance counts tips generated on/after {_cut} (pre-fix new-format tips excluded; kept in CLV data).")

st.divider()
st.markdown("### Navigate using the sidebar →")
st.markdown("""
| Page | What you'll find |
|---|---|
| 📊 Dashboard | Today's tips by **model** (Standard / New-Format) and **tier** with drift signals |
| ⚡ Live | In-play, half-time and live-signal history (all live views) |
| 👤 Player Props · ⚽ Fantasy | Player-prop tips and FPL layer |
| 💼 Portfolio | ROI / PnL by model, tier and league (post-cutoff) |
| ℹ️ Model Info | How the Standard and New-Format models work |
""")
