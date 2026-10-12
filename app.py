"""
Wowza Betting Dashboard — v9
Main entry point. Run with: streamlit run app.py
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd


def _read_csv(*args, **kwargs):
    """pd.read_csv with paper leagues removed — they are collected, never counted.

    See config.PAPER_LEAGUES. Every KPI on this page reads through here so a paper
    league cannot reach a total by way of a path that forgot the rule.
    """
    import config as _cfg
    return _cfg.drop_paper_leagues(pd.read_csv(*args, **kwargs))
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
    df = _read_csv(f)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_ledger():
    f = config.OUTPUT_DIR / "bets_ledger.csv"
    if not f.exists():
        return pd.DataFrame()
    df = _read_csv(f, dtype=str)
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
# SAME BASIS AS THE DASHBOARD PAGE (fixed 2026-10-12). This header used to count EVERY tier on
# the O/U 2.5 ledger alone, so it showed -23.22u while the Dashboard showed +36.22u for the same
# day: VALUABLE is paper (never staked) and carried most of that loss, and the side markets
# (where most of the profit is) were not counted at all. Two pages, two answers. Now both read
# staked tiers (SNIPER + MARKSMAN) across every staked market, settled, from
# PERFORMANCE_CUTOFF_DATE by match date; paper VALUABLE is shown separately and labelled.
_STAKED = ("SNIPER", "MARKSMAN")
_SOURCES = [("bets_ledger.csv", None, "O/U 2.5"), ("side_bets_ledger.csv", "market", None),
            ("ht_ledger.csv", None, "HT O/U")]


@st.cache_data(ttl=60)
def load_settled() -> pd.DataFrame:
    parts = []
    cut = pd.Timestamp(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))
    for f, mkc, const in _SOURCES:
        p = config.OUTPUT_DIR / f
        if not p.exists():
            continue
        try:
            d = _read_csv(p, low_memory=False)
            if "source" in d.columns:
                d = d[d["source"].fillna("live").astype(str) == "live"]
            if "signal_tier" not in d.columns:
                continue
            d = d.assign(_dt=pd.to_datetime(d.get("match_date"), errors="coerce"),
                         _pnl=pd.to_numeric(d.get("pnl"), errors="coerce"),
                         _mkt=d[mkc].astype(str) if mkc and mkc in d.columns else const)
            parts.append(d[d["_pnl"].notna() & (d["_dt"] >= cut)][["signal_tier", "_mkt", "_pnl"]])
        except Exception:                                   # noqa: BLE001 — a viewer never dies on one file
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["signal_tier", "_mkt", "_pnl"])


bets = load_bets()
settled = load_settled()
staked = settled[settled["signal_tier"].isin(_STAKED)]
paper = settled[settled["signal_tier"] == "VALUABLE"]
_cut = config.PERFORMANCE_CUTOFF_DATE

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    _up = bets[bets["signal_tier"].isin(_STAKED) & bets["bet"].isin(["OVER", "UNDER"])] \
        if not bets.empty else bets
    st.metric("🎯 Upcoming staked tips", len(_up),
              help="SNIPER + MARKSMAN on O/U 2.5 in the current slate. All markets are on the "
                   "Dashboard page.")

with c2:
    if len(staked):
        st.metric("📈 P/L — staked (u)", f"{staked['_pnl'].sum():+.2f}u",
                  help="Flat 1u, SNIPER + MARKSMAN, every staked market (O/U 2.5, side markets, "
                       "HT), settled only. Same number as the Dashboard page.")
    else:
        st.metric("📈 P/L — staked (u)", "—")

with c3:
    if len(staked):
        st.metric("✅ Win rate — staked", f"{(staked['_pnl'] > 0).mean() * 100:.1f}%")
    else:
        st.metric("✅ Win rate — staked", "—")

with c4:
    st.metric("📊 Settled staked bets", len(staked))

with c5:
    st.metric("💎 Paper VALUABLE P/L", f"{paper['_pnl'].sum():+.2f}u" if len(paper) else "—",
              delta=f"{len(paper)} tips, not bets", delta_color="off",
              help="VALUABLE is recorded and tracked but never staked, so it is not in the P/L.")

st.caption(f"📅 Settled since {_cut} (by match date), flat 1u stakes. Per-market breakdown on the "
           f"Dashboard page — one market carries most of the total.")

st.divider()
st.markdown("### Navigate using the sidebar →")
st.markdown("""
| Page | What you'll find |
|---|---|
| 📊 Dashboard | Today's tips by **model** (Standard / New-Format) and **tier** with drift signals |
| ⚡ Live | In-play, half-time and live-signal history (all live views) |
| 👤 Player Props · ⚽ Fantasy | Player-prop tips and FPL layer |
| 💼 Portfolio | P/L by model, tier and league (post-cutoff) · bankroll sim |
| ℹ️ Model Info | How the Standard and New-Format models work |
| 🧩 Bet Builder | Same-match combos generated in Pro |
| 🔬 Pro | Evidence layer: does retraining help, **tips sent** (1X2 · Bet Builder scoreboard), **studies** (Argentina BTTS, O/U challenger, per-league evidence) |
| 🛰️ V11 Market | Market-first shadow research |
| 🔭 League Scout | ~200 leagues Wowza watches but does not bet — what is collected and how far each has got |
""")


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
