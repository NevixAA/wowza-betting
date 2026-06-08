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

tabs = st.tabs([
    "🎯 Prediction Model",
    "⏱ HT Model",
    "💰 Sharp Money",
    "🌍 World Cup",
    "⚡ Live Scanner",
    "🎚 Thresholds",
    "📊 Feature Importances",
    "🏆 Backtest",
])

# ── Tab 1: Prediction Model ────────────────────────────────────────────────────
with tabs[0]:
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

        **Backtest ROI:** +13–18% per league (walk-forward, post-COVID) ✅
        """)
    with col2:
        st.markdown("""
        #### 🌍 New-Format
        **Leagues:** Brazil, Japan, Ireland, Austria, Denmark,
        Sweden, Norway, Finland, Argentina, Mexico, China, USA MLS

        **Features:**
        - Goals (scored / conceded last 5)
        - Attack / defense strength
        - Rest days
        - ❌ No shots / corners / historical odds

        **Backtest:** Not possible — no historical O/U odds in CSV.
        Performance tracked via live results only.
        """)
    st.divider()
    st.markdown("""
    #### 🔄 How it works
    ```
    1. OddsAPI  → fetch upcoming fixtures + current O/U odds
    2. Features → form, strength, HT rates, market implied prob
    3. Model    → predict p(over 2.5)
    4. Edge     = p_model − p_implied   (p_implied = 1/odds)
    5. Tier     → SNIPER / VALUE / AVOID  (per-league thresholds)
    6. Guard    → both-losing filter suppresses conflicting bets
    7. Drift    → market movement confirms or conflicts the signal
    ```
    """)

# ── Tab 2: HT Model ────────────────────────────────────────────────────────────
with tabs[1]:
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
with tabs[2]:
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

# ── Tab 4: World Cup ───────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🌍 World Cup 2026 Tracker")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Drift tracking (same as Sharp Money)
        - Tracks **O/U 1.5, O/U 2.5, O/U 3.5** and **1X2** drift
        - Every **1 hour** (more frequent than regular leagues)
        - STRONG / SHARP / FADING signals
        - Telegram alerts for STRONG only

        #### ML Model value signals
        Our FT and HT models applied to every WC fixture:
        - **P(FT OVER 2.5)** — from standard model
        - **P(HT OVER 0.5/1.5)** — from HT model
        - **Fair price** calculated from probabilities
        - Compare to bookmaker's offered price → find value
        """)
    with col2:
        st.markdown("""
        #### Limitations
        ⚠️ Our model was trained on **club league data**, not
        international football. National team dynamics differ
        (player availability, motivation, tournament pressure).

        Use WC ML values as **directional guidance**, not
        high-confidence betting signals. The drift tracker
        (sharp money) is more reliable for WC since it
        reflects real market positioning.

        #### Dashboard tabs
        - 📡 **Sharp Money Drift** — odds movement signals
        - 🤖 **ML Model Value** — fair price vs market comparison
        """)

# ── Tab 5: Live Scanner ────────────────────────────────────────────────────────
with tabs[4]:
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
        - Every **30 minutes** during match hours (11:00–23:30 UTC)
        - Smart cache: only calls leagues with games today
        - Active league (live games): re-checks every 2 min
        - Idle league (no games): skips for 30 min

        #### World Cup
        WC games are also scanned during match hours.
        """)

# ── Tab 6: Thresholds ──────────────────────────────────────────────────────────
with tabs[5]:
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

# ── Tab 7: Feature Importances ─────────────────────────────────────────────────
with tabs[6]:
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

# ── Tab 8: Backtest ────────────────────────────────────────────────────────────
with tabs[7]:
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
    st.info("Backtest not available — no historical O/U odds in CSV data. "
            "Performance tracked via live results in the Ledger page.")

st.divider()
st.caption("v9.1 · GitHub Actions CI · OddsAPI 20K plan · Streamlit Cloud")
