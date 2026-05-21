"""Performance analytics — ROI, win rate, SNIPER vs VALUE, by league."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="Performance | Wowza", page_icon="📈", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="perf_refresh")


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
    df["pnl"]        = pd.to_numeric(df.get("pnl", pd.Series()), errors="coerce")
    df["edge_pct"]   = pd.to_numeric(df.get("edge_pct", pd.Series()), errors="coerce")
    df["odds"]       = pd.to_numeric(df.get("odds", pd.Series()), errors="coerce")
    df["match_date"] = pd.to_datetime(df.get("match_date", pd.Series()), errors="coerce")
    return df


st.markdown("## 📈 Performance Analytics")

df = load_ledger()
if df.empty:
    st.warning("No ledger data yet.")
    st.stop()

# ── Source selector ────────────────────────────────────────────────────────────
source = st.radio("Data source", ["live", "backtest", "both"], horizontal=True)
if source == "both":
    data = df
elif source == "live":
    data = df[df["source"] == "live"]
else:
    data = df[df["source"] == "backtest"]

scored = data[data["pnl"].notna()].copy()

if scored.empty:
    st.info("No resolved bets yet for this source.")
    st.stop()

# ── KPI row ────────────────────────────────────────────────────────────────────
total_bets  = len(scored)
wins        = (scored["pnl"] > 0).sum()
total_pnl   = scored["pnl"].sum()
roi         = total_pnl / total_bets * 100 if total_bets else 0
win_rate    = wins / total_bets * 100 if total_bets else 0
max_dd      = (scored["pnl"].cumsum() - scored["pnl"].cumsum().cummax()).min()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Bets",   total_bets)
c2.metric("Win Rate",     f"{win_rate:.1f}%")
c3.metric("ROI",          f"{roi:+.1f}%",  delta=f"{total_pnl:+.1f}u")
c4.metric("Total PnL",    f"{total_pnl:+.1f}u")
c5.metric("Max Drawdown", f"{max_dd:.1f}u")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tier Breakdown", "📉 Equity Curve", "🏆 By League", "📅 By Month"])

# ── Tab 1: SNIPER vs VALUE ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### SNIPER vs VALUE Performance")
    tier_rows = []
    for tier in ["SNIPER", "VALUE"]:
        t = scored[scored["signal_tier"] == tier]
        if t.empty:
            continue
        n    = len(t)
        w    = (t["pnl"] > 0).sum()
        pnl  = t["pnl"].sum()
        tier_rows.append({
            "Tier":       tier,
            "Bets":       n,
            "Wins":       int(w),
            "Win Rate":   f"{w/n:.1%}",
            "Total PnL":  f"{pnl:+.2f}u",
            "ROI":        f"{pnl/n*100:+.1f}%",
            "Avg Odds":   f"{t['odds'].mean():.2f}",
        })
    if tier_rows:
        tier_df = pd.DataFrame(tier_rows)
        st.dataframe(tier_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig = go.Figure()
        for row in tier_rows:
            color = "#e94560" if row["Tier"] == "SNIPER" else "#f5a623"
            fig.add_bar(
                name=row["Tier"],
                x=[row["Tier"]],
                y=[float(row["ROI"].replace("%","").replace("+",""))],
                marker_color=color,
                text=[row["ROI"]],
                textposition="outside",
            )
        fig.update_layout(
            title="ROI % by Tier",
            yaxis_title="ROI %",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            showlegend=False,
            height=350,
        )
        fig.add_hline(y=0, line_color="#555")
        st.plotly_chart(fig, use_container_width=True)

    # Model type breakdown
    st.markdown("### Standard vs New-Format Model")
    model_rows = []
    for mt in ["standard", "new_format"]:
        t = scored[scored.get("model_type", pd.Series()) == mt] if "model_type" in scored else pd.DataFrame()
        if t.empty:
            continue
        n   = len(t)
        w   = (t["pnl"] > 0).sum()
        pnl = t["pnl"].sum()
        model_rows.append({
            "Model":     mt,
            "Bets":      n,
            "Win Rate":  f"{w/n:.1%}",
            "ROI":       f"{pnl/n*100:+.1f}%",
            "Total PnL": f"{pnl:+.2f}u",
        })
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)


# ── Tab 2: Equity curve ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Cumulative PnL Over Time")
    scored_sorted = scored.sort_values("match_date").copy()
    scored_sorted["cumulative_pnl"] = scored_sorted["pnl"].cumsum()
    scored_sorted["drawdown"] = (
        scored_sorted["cumulative_pnl"] -
        scored_sorted["cumulative_pnl"].cummax()
    )

    fig = go.Figure()

    # By tier
    for tier, color in [("SNIPER", "#e94560"), ("VALUE", "#f5a623")]:
        t = scored_sorted[scored_sorted["signal_tier"] == tier].copy()
        if t.empty:
            continue
        t["cum"] = t["pnl"].cumsum()
        fig.add_scatter(x=t["match_date"], y=t["cum"], name=tier,
                        line=dict(color=color, width=2), mode="lines")

    fig.update_layout(
        title="Cumulative PnL by Tier",
        xaxis_title="Date",
        yaxis_title="PnL (units)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=400,
        legend=dict(bgcolor="#1a1a2e"),
    )
    fig.add_hline(y=0, line_color="#555", line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    fig2 = go.Figure()
    fig2.add_scatter(
        x=scored_sorted["match_date"], y=scored_sorted["drawdown"],
        fill="tozeroy", line=dict(color="#e94560"), name="Drawdown"
    )
    fig2.update_layout(
        title="Drawdown",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=250,
        yaxis_title="PnL (units)", xaxis_title="Date",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: By League ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ROI by League")
    league_rows = []
    for lg, grp in scored.groupby("league"):
        n   = len(grp)
        w   = (grp["pnl"] > 0).sum()
        pnl = grp["pnl"].sum()
        if n < 5:
            continue
        league_rows.append({
            "League":    lg,
            "Bets":      n,
            "Win Rate":  f"{w/n:.1%}",
            "ROI":       round(pnl / n * 100, 1),
            "Total PnL": round(pnl, 2),
        })
    if league_rows:
        lg_df = pd.DataFrame(league_rows).sort_values("ROI", ascending=False)
        st.dataframe(lg_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            lg_df, x="League", y="ROI",
            color="ROI",
            color_continuous_scale=["#e94560", "#555", "#00c896"],
            color_continuous_midpoint=0,
            title="ROI % by League",
            text="ROI",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=420,
            coloraxis_showscale=False, xaxis_tickangle=-35,
        )
        fig.add_hline(y=0, line_color="#555")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: By Month ────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Monthly Breakdown")
    scored2 = scored.copy()
    scored2["month"] = scored2["match_date"].dt.to_period("M").astype(str)
    month_rows = []
    for mo, grp in scored2.groupby("month"):
        n   = len(grp)
        w   = (grp["pnl"] > 0).sum()
        pnl = grp["pnl"].sum()
        month_rows.append({
            "Month":     mo,
            "Bets":      n,
            "Win Rate":  f"{w/n:.1%}",
            "ROI":       round(pnl / n * 100, 1),
            "PnL":       round(pnl, 2),
        })
    if month_rows:
        mo_df = pd.DataFrame(month_rows)
        st.dataframe(mo_df, use_container_width=True, hide_index=True)

        fig = px.bar(mo_df, x="Month", y="PnL",
                     color="PnL",
                     color_continuous_scale=["#e94560", "#555", "#00c896"],
                     color_continuous_midpoint=0,
                     title="Monthly PnL (units)")
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=350,
            coloraxis_showscale=False,
        )
        fig.add_hline(y=0, line_color="#555")
        st.plotly_chart(fig, use_container_width=True)
