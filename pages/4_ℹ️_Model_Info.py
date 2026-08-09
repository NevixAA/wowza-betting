"""Complete system documentation — all tools, methods, and configs in one place."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="System Info | Wowza", page_icon="ℹ️", layout="wide")

st.markdown("## ℹ️ System Documentation")
st.caption("All tools, methods, thresholds and configs — one place.")

@st.cache_data(ttl=300)
def load_fi(name):
    f = config.MODELS_DIR / f"feature_importances_{name}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data(ttl=300)
def load_backtest():
    f = config.OUTPUT_DIR / "backtest_by_league_standard.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

tab_pred, tab_ht, tab_sharp, tab_live, tab_thr, tab_fi, tab_bt = st.tabs([
    "🎯 Prediction Model",
    "⏱ HT Model",
    "💰 Sharp Money",
    "⚡ Live Scanner",
    "🎚 Thresholds",
    "📊 Feature Importances",
    "🏆 Backtest",
])

# ── Tab 1: Prediction Model ────────────────────────────────────────────────────
with tab_pred:
    st.markdown("### 🎯 Pre-Match Prediction Model (O/U 2.5 FT)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Standard Format
        **Leagues:** League One, League Two, Bundesliga 2,
        La Liga 2, Ligue 2, Championship, Serie B

        **Features:**
        - Goals, shots on target, corners (last 5 games)
        - HT scoring rates (first-half tendency)
        - Attack / defense strength vs league average
        - Historical O/U 2.5 odds (market implied probability)
        - Rest days, referee foul average

        **Training:** COVID excluded (2019/20, 2020/21).
        Recent seasons weighted 2-4× (time-decay).

        **Performance:** live P/L is the real test (2026/27 season = the QA).
        Backtest figures run hot vs live — treat as directional, not a promise.
        """)
    with col2:
        st.markdown("""
        #### 🌍 New-Format
        **Leagues:** Brazil, Japan, Ireland, Austria, Denmark,
        Sweden, Norway, Finland, Argentina, Mexico, China, USA MLS

        **Features:** same rich feature set as Standard (goals, shots,
        corners, attack/defense strength, season splits, H2H) from the
        API-Football backfill — **minus** xG, inside-box and half-time
        features, which are populated in training but absent for upcoming
        fixtures (dropped 2026-08-09 to fix a calibration bug).

        **⚠️ Aug 2026 fix:** the model was under-predicting goals
        (P(over)≈0.36 vs ~0.51 real) → one-sided UNDER tips. Retrained
        without the mismatched features → P(over)≈0.50, balanced. Live
        results counted from the cutoff onward.
        """)
    st.divider()
    st.markdown("""
    #### 🔄 How it works
    ```
    1. OddsAPI  → fetch upcoming fixtures + current O/U odds
    2. Features → form, strength, HT rates, market implied prob
    3. Model    → predict p(over 2.5)
    4. Edge     = p_model − p_implied   (p_implied = 1/odds)
    5. Tier     → SNIPER / MARKSMAN / VALUABLE  (per-league thresholds)
    6. Guard    → both-losing filter suppresses conflicting bets
    7. Drift    → market movement (sharp money) confirms or conflicts the signal
    ```
    """)

# ── Tab 2: HT Model ────────────────────────────────────────────────────────────
with tab_ht:
    st.markdown("### ⏱ Half-Time O/U Prediction Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### What it predicts
        - **P(HT OVER 0.5)** — at least 1 goal in first half
        - **P(HT OVER 1.5)** — at least 2 goals in first half

        #### Training data
        - Standard-format leagues only (have HTHG/HTAG in CSVs)
        - ~48,000 matches across 23 leagues
        - Same COVID exclusion and time-decay as FT model

        #### Key features
        - Team HT tendency rate (% of recent games with HT goal)
        - HT scored / conceded last 5 games
        - HT attack / defense strength vs league half-time average
        - Combined HT goals average
        """)
    with col2:
        st.markdown("""
        #### Performance
        | Model | AUC | Notes |
        |---|---|---|
        | HT OVER 0.5 | 0.567 | Experimental |
        | HT OVER 1.5 | 0.569 | Experimental |

        ⚠️ These models are newer and still building track record.
        Use as supplementary signal — compare fair price against
        your bookmaker's HT market to find value.

        #### How to use
        Strong signal = P ≥ 75% (OVER) or P ≤ 30% (UNDER)
        Compare our **fair price** to bookmaker's offered price.
        If bookmaker pays more → value bet.
        """)

# ── Tab 3: Sharp Money ─────────────────────────────────────────────────────────
with tab_sharp:
    st.markdown("### 💰 Sharp Money Tracker")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### What it does
        Tracks odds drift across all 20 enabled leagues every 2 hours.
        "Sharp money" = odds shortening = someone positioning before the market reacts.

        #### Signal types
        | Signal | Meaning |
        |---|---|
        | 🔥 STEAM_STRONG | Fast move (>3% in 1 window) + >10% total drift |
        | ⚡ STEAM_SHARP | Fast move + 5-10% total drift |
        | 🔴 STRONG | >10% drift from opening |
        | 🟡 SHARP | 5-10% drift |
        | ⬆️ FADING | Odds lengthening (public money, avoid) |

        #### How drift is calculated
        ```
        drift % = (current_odds − opening_odds) / opening_odds
        negative = odds shortened = sharp money IN
        positive = odds lengthened = money moving away
        ```
        """)
    with col2:
        st.markdown("""
        #### Volume weighting
        - Odds are the **median across all bookmakers**, not just one
        - **Consensus %** = % of snapshots where all books moved together
        - 100% consensus + steam = strongest signal

        #### Steam detection
        Compares last snapshot vs previous snapshot.
        Move > 3% in a single 2h window = STEAM flag.

        #### Schedule
        - Runs every **2 hours** (08:00–22:00 UTC)
        - Stores opening odds on first snapshot
        - Tracks up to 50 snapshots per match/market

        #### Telegram alerts
        Sends for STEAM_STRONG, STEAM_SHARP, STRONG, SHARP signals only.
        FADING signals shown on dashboard but not alerted.
        """)

# ── Tab 4: Live Scanner ────────────────────────────────────────────────────────
with tab_live:
    st.markdown("### ⚡ Live Scanner (In-Play)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Full-time signals
        | Signal | When | Bet |
        |---|---|---|
        | 🔒 UNDER_HOLD | 45+ min, 0-1 goals, model said UNDER | UNDER 2.5 |
        | 😴 SLEEPING_GAME | 70+ min, 0-0, both teams low attack | UNDER 2.5 |
        | 📉 UNDER_RECOVERY | 2 goals at 65- min | UNDER 2.5 |
        | 💪 STRONG_STUCK | Strong team not scoring at 55-80 min | OVER 2.5 |
        | 🔥 COMEBACK | Losing team with strong attack at 60-82 min | OVER 2.5 |

        #### Half-time signals (first half only)
        | Signal | When | Bet |
        |---|---|---|
        | ⏱ HT_UNDER_0.5 | 0-0 at 35+ min | HT UNDER 0.5 |
        | 🧊 HT_UNDER_1.5 | ≤1 goal at 25+ min, P>60% | HT UNDER 1.5 |
        | ⚡ HT_OVER_0.5 | 0-0, strong teams, under 38 min | HT OVER 0.5 |
        """)
    with col2:
        st.markdown("""
        #### How fair prices are calculated
        Uses **Poisson statistics** — no live odds API needed:
        ```
        1. Pre-match model → P(over 2.5) → lambda (expected goals)
        2. Current score + elapsed time → lambda_remaining
        3. Poisson CDF → live P(under) / P(over)
        4. Fair price = 1 / probability
        ```
        Compare our fair price to your bookmaker's live screen.
        If bookmaker offers **more** than our fair price → value.

        #### Schedule
        - Every **~10 minutes** during match hours (live_scanner cron)
        - Smart cache: only calls leagues with games today
        - Live-adjusted λ from in-play shots-on-target (v2)
        - Results graded post-match; see the **⚡ Live Center → History** tab

        _All live views (in-play, half-time, history) now live on the
        single **⚡ Live Center** page._
        """)

# ── Tab 5: Thresholds ──────────────────────────────────────────────────────────
with tab_thr:
    st.markdown("### 🎚 Signal Thresholds & Calibration")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Global SNIPER", f"{config.SNIPER_THRESHOLD:.0%}")
    with col2:
        st.metric("OVER threshold", f"{config.SNIPER_THRESHOLD_OVER:.0%}", help="OVER bets less reliable post-COVID")
    with col3:
        st.metric("UNDER threshold", f"{config.SNIPER_THRESHOLD_UNDER:.0%}", help="UNDER bets more reliable")

    st.markdown("#### Per-League SNIPER Thresholds")
    st.caption("Calibrated from post-COVID walk-forward backtest. Higher bar = only highest confidence bets.")
    thresh_data = []
    for lg, thresh in sorted(config.LEAGUE_SNIPER_THRESHOLDS.items()):
        thresh_data.append({
            "League": lg,
            "SNIPER Threshold": f"{thresh:.0%}",
            "Active": "✅" if lg in config.ENABLED_LEAGUES else "Training only",
            "Rationale": "Most data → lower bar" if thresh <= 0.14
                         else "Good density" if thresh <= 0.18
                         else "High confidence only",
        })
    st.dataframe(pd.DataFrame(thresh_data), use_container_width=True, hide_index=True)

    st.markdown("#### Training Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**COVID seasons excluded:** {', '.join(sorted(config.COVID_SEASONS))}  \n"
                f"Empty stadiums caused anomalous OVER patterns (55% win rate) not seen in normal football.")
    with c2:
        decay_df = pd.DataFrame([
            {"Season": s, "Weight": w,
             "Status": "❌ Excluded" if w == 0 else f"×{w:.0f}"}
            for s, w in sorted(config.TRAINING_DECAY_WEIGHTS.items(), reverse=True)
            if w > 0 or s in config.COVID_SEASONS
        ])
        st.dataframe(decay_df, use_container_width=True, hide_index=True)

# ── Tab 6: Feature Importances ─────────────────────────────────────────────────
with tab_fi:
    st.markdown("### 📊 Feature Importances")
    col_std, col_nf = st.columns(2)
    for col, name, title, color in [
        (col_std, "standard", "Standard Model", ["#16213e", "#e94560"]),
        (col_nf,  "newformat", "New-Format Model", ["#16213e", "#f5a623"]),
    ]:
        with col:
            st.markdown(f"#### {title}")
            fi = load_fi(name)
            if not fi.empty:
                fig = px.bar(fi.head(12).sort_values("importance"),
                             x="importance", y="feature", orientation="h",
                             color="importance", color_continuous_scale=color,
                             text="importance_%")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                  font_color="white", coloraxis_showscale=False,
                                  height=420, yaxis_title="", xaxis_title="Importance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run retrain to generate feature importances.")

# ── Tab 7: Backtest ────────────────────────────────────────────────────────────
with tab_bt:
    st.markdown("### 🏆 Backtest Results")
    df = load_backtest()
    if not df.empty:
        df_sorted = df.sort_values("roi_%", ascending=False)
        st.dataframe(df_sorted, use_container_width=True, hide_index=True)
        fig = px.bar(df_sorted, x="league", y="roi_%",
                     color="roi_%",
                     color_continuous_scale=["#e94560", "#555", "#00c896"],
                     color_continuous_midpoint=0,
                     title="Walk-Forward Backtest ROI % by League (post-COVID, standard model)",
                     text="roi_%")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font_color="white", coloraxis_showscale=False,
                          height=400, xaxis_tickangle=-20)
        fig.add_hline(y=0, line_color="#555")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run retrain to generate backtest results.")

    st.markdown("#### New-Format Model")
    st.info("New-format now has full historical stats + odds via the API-Football backfill "
            "(20k+ matches). After the 2026-08-09 calibration fix, live P/L from the cutoff "
            "onward is the real test — see the 💼 Portfolio page.")

st.divider()
st.caption("v9 · GitHub Actions CI · live P/L in units (paper) · Streamlit Cloud")
