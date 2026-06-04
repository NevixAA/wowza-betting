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

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Two Model Formats", "🎯 Per-League Thresholds", "📊 Feature Importances", "🏆 Backtest Results"])

# ── Tab 1: Explanation ─────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Standard Format Model
        ---
        **Leagues:** League One, League Two, Bundesliga 2, La Liga 2, Ligue 2, Championship, Serie B

        **Data source:** football-data.co.uk (mmz4281 format)

        **Features:**
        - ✅ Goals (scored / conceded last 5)
        - ✅ HT goals (first-half rolling stats)
        - ✅ Shots on target, Corners
        - ✅ Historical O/U 2.5 odds
        - ✅ Attack / defense strength (FT + HT)
        - ✅ Team HT tendency rates
        - ✅ Rest days, Referee stats

        **Training:** COVID seasons excluded (2019/20, 2020/21 removed — empty-stadium anomaly).
        Time-decay: recent seasons weighted 2-4× higher than older data.

        **Backtest (walk-forward, post-COVID):**
        - SNIPER ROI: **+13-18% per league** ✅
        - VALUE ROI:  **+2.9%** ⚠️
        - Sharpe: **2.43**
        """)

    with col2:
        st.markdown("""
        ### 🌍 New-Format Model
        ---
        **Leagues:** Brazil, Japan, Ireland, Austria, Denmark,
        Sweden, Norway, Finland, Argentina, Mexico, China, USA MLS

        **Data source:** football-data.co.uk (/new/ format)

        **Features:**
        - ✅ Goals (scored / conceded last 5)
        - ✅ Attack / defense strength
        - ✅ Rest days
        - ❌ No shots / corners
        - ❌ No historical O/U odds in CSV

        **Backtest:** Not available — no historical odds.
        Performance tracked via live results only.

        **Note:** These leagues use global SNIPER threshold (15%)
        since no league-specific backtest is available.
        """)

    st.divider()
    st.markdown("""
    ### 🔄 How Predictions Work (v9 — post-COVID recalibrated)

    ```
    1. OddsAPI → fetch upcoming fixtures + live O/U odds
    2. feature_engineering → build stats (form, HT tendency, strength)
    3. Model → predict p(over2.5) for each fixture
    4. Edge = p_model - p_implied  (p_implied = 1/odds)
    5. Tier (per-league thresholds):
         League Two:     edge ≥ 14%  → SNIPER
         Bundesliga 2:   edge ≥ 20%  → SNIPER
         La Liga 2:      edge ≥ 20%  → SNIPER
         League One:     edge ≥ 25%  → SNIPER
         Ligue 2:        edge ≥ 25%  → SNIPER
         OVER bets:      edge ≥ 18%  → SNIPER (higher bar — less reliable)
         UNDER bets:     edge ≥ 13%  → SNIPER (more reliable historically)
         edge 4-threshold → VALUE (half stake)
         edge < 4%        → AVOID
    6. Both-losing guard: suppress if both OVER and UNDER negative
    7. Drift adjustment:
         Confirmed  → market moved our way  (stronger signal)
         Conflicted → market moved against  (downgrade tier)
    ```
    """)


# ── Tab 2: Feature Importances ─────────────────────────────────────────────────
# ── Tab 2: Per-League Thresholds ──────────────────────────────────────────────
with tab2:
    st.markdown("### 🎯 Per-League SNIPER Thresholds")
    st.caption("Each league has its own edge threshold calibrated from post-COVID backtest data.")

    thresh_data = []
    for lg, thresh in sorted(config.LEAGUE_SNIPER_THRESHOLDS.items()):
        in_enabled = lg in config.ENABLED_LEAGUES
        thresh_data.append({
            "League":           lg,
            "SNIPER Threshold": f"{thresh:.0%}",
            "Live Predictions": "✅ Active" if in_enabled else "⚠️ Training only",
            "Why this threshold": (
                "Most data, reliable at lower bar" if thresh <= 0.14
                else "Moderate bar — good edge density" if thresh <= 0.18
                else "High bar — only highest confidence"
            )
        })
    st.dataframe(pd.DataFrame(thresh_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 OVER vs UNDER Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("OVER 2.5 threshold", f"{config.SNIPER_THRESHOLD_OVER:.0%}",
                  help="OVER bets need higher edge — less reliable post-COVID")
    with col2:
        st.metric("UNDER 2.5 threshold", f"{config.SNIPER_THRESHOLD_UNDER:.0%}",
                  help="UNDER bets slightly more reliable — lower bar")

    st.markdown("---")
    st.markdown("### 🚫 COVID Season Exclusion")
    st.info(f"Training data excludes: **{', '.join(sorted(config.COVID_SEASONS))}**  \n"
            f"Reason: Empty-stadium effect created anomalous OVER 2.5 patterns (55% win rate) "
            f"that don't exist in normal football. Excluding these 2 seasons removed ~15,700 rows "
            f"but dramatically improved post-COVID prediction accuracy.")

    st.markdown("---")
    st.markdown("### ⚖️ Time-Decay Weights")
    decay_df = pd.DataFrame([
        {"Season": s, "Weight": w, "Note": "Excluded (COVID)" if w == 0 else
         "Highest weight" if w >= 4 else "High weight" if w >= 3 else
         "Medium" if w >= 2 else "Normal"}
        for s, w in sorted(config.TRAINING_DECAY_WEIGHTS.items(), reverse=True)
    ])
    st.dataframe(decay_df, use_container_width=True, hide_index=True)

# ── Tab 3: Feature Importances ─────────────────────────────────────────────────
with tab3:
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


# ── Tab 4: Backtest Results ────────────────────────────────────────────────────
with tab4:
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
