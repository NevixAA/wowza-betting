"""Model documentation — explains standard vs new-format, feature importances, backtest stats."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="Model Info | Wowza", page_icon="ℹ️", layout="wide")


@st.cache_data(ttl=300)
def load_fi(name):
    f = config.MODELS_DIR / f"feature_importances_{name}.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f)


@st.cache_data(ttl=300)
def load_backtest_summary():
    rows = []
    for name, label in [("standard", "Standard"), ("newformat", "New-Format")]:
        f = config.OUTPUT_DIR / f"backtest_by_league_{name}.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["model"] = label
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


st.markdown("## ℹ️ Model Information")

tab1, tab2, tab3 = st.tabs(["🔍 Two Model Formats", "📊 Feature Importances", "🏆 Backtest Results"])

# ── Tab 1: Explanation ─────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Standard Format Model
        ---
        **Leagues:**
        League One, League Two, Bundesliga 2, La Liga 2,
        Ligue 2, Championship, Serie B

        **Data source:** football-data.co.uk (mmz4281 format)

        **Features available:**
        - ✅ Goals (scored / conceded last 5)
        - ✅ Shots on target
        - ✅ Corners
        - ✅ Historical O/U 2.5 odds
        - ✅ Implied probability from odds
        - ✅ Attack / defense strength
        - ✅ Rest days

        **Why it's better:**
        More input features = better calibration.
        Historical O/U odds tell the model what the market
        thought *at the time*, which is a strong signal.

        **Backtest results (walk-forward):**
        - Overall ROI: **+14.2%**
        - SNIPER ROI: **+31.8%** ✅
        - VALUE ROI:  **-0.5%** ⚠️ (don't bet VALUE alone)
        - Sharpe: **2.37**
        """)

    with col2:
        st.markdown("""
        ### 🌍 New-Format Model
        ---
        **Leagues:**
        Brazil Serie A, Japan J-League, Ireland Premier,
        Austrian Bundesliga, Denmark Superliga,
        Sweden Allsvenskan, Norway Eliteserien,
        Finland Veikkausliiga, Argentina Primera,
        Mexico Liga MX, China Super League, USA MLS

        **Data source:** football-data.co.uk (/new/ format)

        **Features available:**
        - ✅ Goals (scored / conceded last 5)
        - ✅ Attack / defense strength
        - ✅ Rest days
        - ❌ No shots / corners data
        - ❌ No historical O/U odds in CSV

        **How edge is calculated:**
        Model outputs probability → compared to **live odds**
        from OddsAPI at prediction time.
        `edge = p_model - (1 / live_odds)`

        **Backtest:**
        Not possible — no historical O/U odds in CSV data.
        Performance tracked via live results only.
        """)

    st.divider()
    st.markdown("""
    ### 🔄 How Predictions Work

    ```
    1. OddsAPI → fetch upcoming fixtures + live O/U odds
    2. feature_engineering → build stats (form, strength, rest)
    3. Model → predict p(over2.5) for each fixture
    4. Edge = p_model - p_implied  (p_implied = 1/odds - vig)
    5. Tier:
         edge ≥ 10%  →  SNIPER  (bet full stake)
         edge  4-10% →  VALUE   (bet half stake / monitor)
         edge < 4%   →  AVOID
    6. Both-losing guard: if BOTH over AND under edge are
       negative, suppress the bet regardless of tier
    7. Drift: compare current odds to first recorded odds
       Confirmed  → line moved OUR way  (stronger signal)
       Conflicted → line moved AGAINST  (weaker, be cautious)
    ```
    """)


# ── Tab 2: Feature Importances ─────────────────────────────────────────────────
with tab2:
    col_std, col_nf = st.columns(2)

    with col_std:
        st.markdown("#### Standard Model")
        fi_std = load_fi("standard")
        if not fi_std.empty:
            fig = px.bar(
                fi_std.head(12).sort_values("importance"),
                x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#16213e", "#e94560"],
                title="Feature Importance (Standard)",
                text="importance_%",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", coloraxis_showscale=False,
                height=450, yaxis_title="", xaxis_title="Importance",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run retrain.py to generate feature importances.")

    with col_nf:
        st.markdown("#### New-Format Model")
        fi_nf = load_fi("newformat")
        if not fi_nf.empty:
            fig = px.bar(
                fi_nf.head(12).sort_values("importance"),
                x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#16213e", "#f5a623"],
                title="Feature Importance (New-Format)",
                text="importance_%",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", coloraxis_showscale=False,
                height=450, yaxis_title="", xaxis_title="Importance",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run retrain.py to generate feature importances.")


# ── Tab 3: Backtest Results ────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Standard Model — Backtest by League")
    lg_std = config.OUTPUT_DIR / "backtest_by_league_standard.csv"
    if lg_std.exists():
        df = pd.read_csv(lg_std)
        df = df.sort_values("roi_%", ascending=False) if "roi_%" in df else df
        st.dataframe(df, use_container_width=True, hide_index=True)

        if "roi_%" in df.columns and "league" in df.columns:
            fig = px.bar(df, x="league", y="roi_%",
                         color="roi_%",
                         color_continuous_scale=["#e94560", "#555", "#00c896"],
                         color_continuous_midpoint=0,
                         title="Backtest ROI % by League (Standard Model)",
                         text="roi_%")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", coloraxis_showscale=False,
                height=400, xaxis_tickangle=-30,
            )
            fig.add_hline(y=0, line_color="#555")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run retrain.py to generate backtest results.")

    st.markdown("#### New-Format Model")
    st.info("Backtest not available — historical O/U odds not in source CSV data. "
            "Performance tracked via live bets in the Ledger.")
