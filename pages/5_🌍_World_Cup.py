"""
World Cup 2026 — Drift Tracker Dashboard
Pure odds movement. No ML. Follow the sharp money.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="World Cup 2026", page_icon="🌍", layout="wide")

BASE_DIR        = Path(__file__).resolve().parents[1]
TIPS_FILE       = BASE_DIR / "output" / "worldcup_tips.csv"
HISTORY_FILE    = BASE_DIR / "output" / "worldcup_history.json"
MODEL_TIPS_FILE = BASE_DIR / "output" / "worldcup_model_tips.csv"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌍 World Cup 2026")
st.caption("Sharp money drift + ML model value signals · O/U 1.5 / 2.5 / 3.5 · 1X2")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_tips():
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TIPS_FILE)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_history():
    if not HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

tips    = load_tips()
history = load_history()

# ── Not yet available ─────────────────────────────────────────────────────────
if tips.empty:
    st.info("⏳ World Cup 2026 starts **June 11, 2026**. Odds tracking will begin once OddsAPI publishes fixtures (usually 2-4 weeks before kickoff).")
    st.markdown("""
    ### What this page will show:
    - **Sharp money signals** — when odds move significantly, sharp bettors are positioning
    - **O/U 2.5 drift** — is money flowing to OVER or UNDER for each match?
    - **1X2 drift** — which team is sharp money backing?
    - **Opening vs current odds** — see the full movement history per match
    - **STRONG / SHARP / FADING** signals ranked by drift magnitude
    """)
    st.stop()

@st.cache_data(ttl=120)
def load_model_tips():
    if not MODEL_TIPS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_TIPS_FILE)

model_tips = load_model_tips()

# ── KPIs ──────────────────────────────────────────────────────────────────────
strong = tips[tips["signal"] == "STRONG"]
sharp  = tips[tips["signal"] == "SHARP"]
fading = tips[tips["signal"] == "FADING"]
matches = tips["match"].nunique()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Matches Tracked", matches)
c2.metric("🔴 STRONG",       len(strong), help=">10% odds move")
c3.metric("🟡 SHARP",        len(sharp),  help="5-10% odds move")
c4.metric("⬆️ FADING",       len(fading))
c5.metric("🤖 Model Tips",   len(model_tips) if not model_tips.empty else 0)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_drift, tab_model = st.tabs(["📡 Sharp Money Drift", "🤖 ML Model Value"])

with tab_model:
    st.subheader("🤖 ML Model Fair Prices vs Market")
    st.caption("Our FT + HT models applied to WC fixtures · Compare fair price vs bookmaker odds to spot value")
    if model_tips.empty:
        st.info("⏳ Model tips will appear after the next WC tracker run.")
    else:
        for _, row in model_tips.iterrows():
            p_over = float(row.get("p_ft_over25", 0.5))
            fair_o = row.get("fair_ft_over", "—")
            fair_u = row.get("fair_ft_under", "—")
            mkt_o  = row.get("market_over25", "—")
            mkt_u  = row.get("market_under25", "—")
            val_o  = float(row.get("ft_over_value", 0))
            val_u  = float(row.get("ft_under_value", 0))
            best_val = max(val_o, val_u)
            border = "#00cc88" if best_val > 5 else "#ffaa00" if best_val > 0 else "#444"

            ht05 = row.get("p_ht_over05")
            ht15 = row.get("p_ht_over15")

            st.markdown(f"""
            <div style="border-left:4px solid {border};padding:12px 16px;margin:6px 0;background:#111827;border-radius:6px">
                <div style="display:flex;justify-content:space-between">
                    <b style="color:white;font-size:1.05em">{row['match']}</b>
                    <span style="color:#aaa;font-size:0.85em">📅 {row['date']}</span>
                </div>
                <div style="display:flex;gap:20px;margin:8px 0;flex-wrap:wrap">
                    <div><span style="color:#aaa;font-size:0.8em">P(FT OVER 2.5)</span><br><b style="color:white">{p_over*100:.0f}%</b></div>
                    <div><span style="color:#aaa;font-size:0.8em">Fair OVER 2.5</span><br><b style="color:{'#00cc88' if val_o>5 else 'white'}">{fair_o}</b></div>
                    <div><span style="color:#aaa;font-size:0.8em">Market OVER 2.5</span><br><b style="color:white">{mkt_o}</b></div>
                    <div><span style="color:#aaa;font-size:0.8em">Fair UNDER 2.5</span><br><b style="color:{'#00cc88' if val_u>5 else 'white'}">{fair_u}</b></div>
                    <div><span style="color:#aaa;font-size:0.8em">Market UNDER 2.5</span><br><b style="color:white">{mkt_u}</b></div>
                    {f'<div><span style="color:#aaa;font-size:0.8em">P(HT OVER 0.5)</span><br><b style="color:white">{float(ht05)*100:.0f}%</b></div>' if pd.notna(ht05) else ''}
                    {f'<div><span style="color:#aaa;font-size:0.8em">P(HT OVER 1.5)</span><br><b style="color:white">{float(ht15)*100:.0f}%</b></div>' if pd.notna(ht15) else ''}
                </div>
                {f'<div style="color:#00cc88;font-size:0.85em">✅ Value: OVER 2.5 +{val_o:.1f}% vs fair</div>' if val_o > 5 else ''}
                {f'<div style="color:#00cc88;font-size:0.85em">✅ Value: UNDER 2.5 +{val_u:.1f}% vs fair</div>' if val_u > 5 else ''}
            </div>
            """, unsafe_allow_html=True)

with tab_drift:
    st.subheader("📡 Sharp Money Drift Signals")

    # ── Signal filter ──────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
with col1:
    sig_filter = st.multiselect("Signal", ["STRONG", "SHARP", "FADING"],
                                default=["STRONG", "SHARP"])
with col2:
    mkt_filter = st.multiselect("Market", tips["market"].str.split(" ").str[0].unique().tolist(),
                                default=tips["market"].str.split(" ").str[0].unique().tolist())

filtered = tips[
    tips["signal"].isin(sig_filter) &
    tips["market"].str.startswith(tuple(mkt_filter))
].copy() if sig_filter and mkt_filter else tips.copy()

# ── Signal cards ──────────────────────────────────────────────────────────────
st.subheader(f"📡 Active Signals ({len(filtered)})")

if filtered.empty:
    st.info("No signals match the current filter.")
else:
    for _, row in filtered.iterrows():
        drift_color = "#ff4444" if row["drift_pct"] < 0 else "#44ff88"
        border_color = "#ff4444" if row["signal"] == "STRONG" else "#ffaa00" if row["signal"] == "SHARP" else "#888"
        direction = "▼ Shortening (sharp money IN)" if row["drift_pct"] < 0 else "▲ Lengthening (money fading)"

        st.markdown(f"""
        <div style="border-left:4px solid {border_color}; padding:12px 16px; margin:8px 0;
                    background:#1a1a2e; border-radius:4px;">
            <b style="font-size:1.1em">{row['match']}</b>
            &nbsp;&nbsp;<span style="color:#aaa; font-size:0.9em">📅 {str(row['date'])[:10]}</span>
            <br/>
            <span style="color:#ccc">🎯 {row['market']}</span>
            &nbsp;&nbsp;
            <span style="background:{border_color}22; color:{border_color}; padding:2px 8px;
                         border-radius:10px; font-size:0.85em">{row['signal']}</span>
            <br/>
            <span style="color:#aaa">Opening: <b style="color:white">{row['opening_odds']}</b>
            → Current: <b style="color:white">{row['current_odds']}</b>
            &nbsp; <b style="color:{drift_color}">{row['drift_pct']:+.1f}%</b>
            &nbsp; {direction}</span>
            <br/>
            <span style="color:#666; font-size:0.8em">Based on {row['snapshots']} snapshots · Updated {row['updated_at']}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Drift chart ───────────────────────────────────────────────────────────────
st.subheader("📊 Drift Distribution")
if len(filtered) > 0:
    fig = px.bar(
        filtered.sort_values("drift_pct"),
        x="drift_pct", y="match",
        color="signal",
        color_discrete_map={"STRONG": "#ff4444", "SHARP": "#ffaa00", "FADING": "#44aaff"},
        orientation="h",
        labels={"drift_pct": "Drift %", "match": ""},
        title="Odds Movement by Match (negative = sharp money in)",
        hover_data=["market", "opening_odds", "current_odds"],
    )
    fig.update_layout(
        template="plotly_dark", height=max(300, len(filtered) * 30),
        showlegend=True, yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Match detail (odds history) ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Match Odds History")

match_options = sorted(tips["match"].unique().tolist())
if match_options:
    selected_match = st.selectbox("Select match", match_options)

    match_keys = [k for k, v in history.items()
                  if f"{v['home']} vs {v['away']}" == selected_match]

    for key in match_keys:
        rec = history[key]
        market_label = "O/U 2.5" if rec["market"] == "totals" else "1X2"
        st.markdown(f"**{market_label}**")

        snaps = rec.get("snapshots", [])
        if len(snaps) < 2:
            st.caption("Not enough snapshots yet for history chart.")
            continue

        rows = []
        for s in snaps:
            for k, v in s["odds"].items():
                label = k.replace("odds_", "").upper()
                rows.append({"time": s["at"], "side": label, "odds": v})

        if rows:
            df_h = pd.DataFrame(rows)
            df_h["time"] = pd.to_datetime(df_h["time"])

            # Zoom Y-axis tight around actual values so movement is visible
            y_min = df_h["odds"].min()
            y_max = df_h["odds"].max()
            padding = max((y_max - y_min) * 0.5, 0.05)

            fig2 = px.line(
                df_h, x="time", y="odds", color="side",
                title=f"{selected_match} — {market_label} odds movement",
                markers=True,
                color_discrete_sequence=["#4fc3f7", "#ff7043", "#66bb6a", "#ab47bc"],
            )
            fig2.update_traces(line=dict(width=2.5), marker=dict(size=6))
            fig2.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
                height=350,
                yaxis=dict(
                    range=[y_min - padding, y_max + padding],
                    title="Odds",
                    gridcolor="#2a2a3e",
                ),
                xaxis=dict(title="", gridcolor="#2a2a3e"),
                legend=dict(title="", bgcolor="#1a1a2e", bordercolor="#333"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Data: OddsAPI · No ML — drift only")
