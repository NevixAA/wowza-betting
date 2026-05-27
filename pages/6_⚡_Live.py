"""
Live Value Scanner Dashboard
Real-time in-play signals based on Poisson recalculation + pre-match model.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Live Scanner", page_icon="⚡", layout="wide")

BASE_DIR    = Path(__file__).resolve().parents[1]
TIPS_FILE   = BASE_DIR / "output" / "live_tips.csv"
GAMES_FILE  = BASE_DIR / "output" / "live_games.csv"

SIGNAL_META = {
    "UNDER_HOLD":     ("🔒", "#00cc88", "UNDER 2.5 — model prediction holding, time running out"),
    "SLEEPING_GAME":  ("😴", "#44aaff", "UNDER 2.5 — both teams low scoring, game going nowhere"),
    "UNDER_RECOVERY": ("📉", "#ffaa00", "UNDER 2.5 — 2 goals scored, 1 more needed for over, time left"),
    "STRONG_STUCK":   ("💪", "#ff6600", "OVER 2.5 — strong attack team can't score, will push harder"),
    "COMEBACK":       ("🔥", "#ff4444", "OVER 2.5 — losing team with strong attack pushing for goals"),
}

st.title("⚡ Live Value Scanner")
st.caption("In-play signals from Poisson recalculation. Check your bookmaker's live screen to compare odds.")

# ── Auto-refresh every 2 minutes ──────────────────────────────────────────────
import streamlit.components.v1 as components
components.html("<script>setTimeout(()=>window.location.reload(),120000)</script>", height=0)

# ── Refresh ───────────────────────────────────────────────────────────────────
col_ref, col_run = st.columns([1, 4])
with col_ref:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
with col_run:
    if st.button("▶ Run Live Scan Now"):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "src" / "live_scanner.py")],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=60
        )
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def load_tips():
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(TIPS_FILE)

@st.cache_data(ttl=120)
def load_games():
    if not GAMES_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(GAMES_FILE)

df    = load_tips()
games = load_games()

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ How this works"):
    st.markdown("""
    **No live odds API needed.** We calculate the *fair price* ourselves using Poisson statistics:

    1. Pre-match model gives P(over 2.5) → we derive expected total goals (λ)
    2. Current score + time elapsed → remaining expected goals (λ_remaining)
    3. Poisson CDF gives live P(under) / P(over)
    4. Fair price = 1 / probability

    **You check your bookmaker's live screen.** If their live odds are significantly higher than our fair price → **value bet**.

    | Signal | What it means |
    |---|---|
    | 🔒 UNDER_HOLD | Model said UNDER, game still low-scoring, time running out |
    | 😴 SLEEPING_GAME | Both teams struggle to score, 0-0 after 70+ min |
    | 📉 UNDER_RECOVERY | Score is 2 goals, only 1 more ends UNDER — check Poisson says it's still worth it |
    | 💪 STRONG_STUCK | High-attack team not scoring at 55+ min, expect them to push → OVER value |
    | 🔥 COMEBACK | Team losing with strong attack → pushing for goals → OVER value |
    """)

# ── Empty state ───────────────────────────────────────────────────────────────
if df.empty or "signal_type" not in df.columns:
    st.info("⏳ No live value signals right now. Scanner runs every 10 minutes during match hours.")
    st.markdown("""
    **When signals appear here:**
    - Match is in progress in one of our tracked leagues
    - Poisson recalculation shows the live fair price is attractive
    - Pre-match model prediction aligns with the live situation
    """)
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
under_tips = df[df["bet"].str.contains("UNDER", na=False)]
over_tips  = df[df["bet"].str.contains("OVER",  na=False)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Live Signals",    len(df))
c2.metric("UNDER Value",     len(under_tips))
c3.metric("OVER Value",      len(over_tips))
c4.metric("Last Scan",       df["updated_at"].iloc[0] if not df.empty else "—")

st.markdown("---")

# ── Signal cards ──────────────────────────────────────────────────────────────
st.subheader("🎯 Active Live Signals")

for _, row in df.iterrows():
    emoji, color, desc = SIGNAL_META.get(
        row["signal_type"], ("📌", "#888", row["signal_type"])
    )
    bet      = row["bet"]
    is_under = "UNDER" in bet
    fair_odds = row["fair_under_odds"] if is_under else row["fair_over_odds"]
    live_p    = row["live_p_under"]    if is_under else row["live_p_over"]

    # Minutes bar
    elapsed = int(row["elapsed_mins"])
    bar_pct = min(elapsed / 90 * 100, 100)

    st.markdown(f"""
    <div style="border-left:4px solid {color}; padding:14px 18px; margin:10px 0;
                background:#111827; border-radius:6px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:1.15em; font-weight:bold; color:white">
                {emoji} {row['match']}
            </span>
            <span style="background:{color}22; color:{color}; padding:3px 10px;
                         border-radius:12px; font-size:0.85em; font-weight:bold">
                {row['signal_type']}
            </span>
        </div>
        <div style="color:#aaa; font-size:0.9em; margin:4px 0">
            🏆 {row['league']} &nbsp;|&nbsp;
            ⏱ {elapsed}' &nbsp;|&nbsp;
            ⚽ Score: <b style="color:white">{row['score']}</b>
        </div>
        <div style="background:#1f2937; border-radius:4px; height:6px; margin:8px 0">
            <div style="background:{color}; width:{bar_pct:.0f}%; height:6px; border-radius:4px"></div>
        </div>
        <div style="display:flex; gap:24px; margin:8px 0">
            <div>
                <span style="color:#aaa; font-size:0.85em">Bet</span><br>
                <b style="color:white; font-size:1.1em">{bet}</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.85em">Fair Price</span><br>
                <b style="color:{color}; font-size:1.1em">{fair_odds}</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.85em">Live P({('UNDER' if is_under else 'OVER')})</span><br>
                <b style="color:white; font-size:1.1em">{live_p*100:.0f}%</b>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.85em">Pre-match P(over)</span><br>
                <b style="color:#888; font-size:1.0em">{row['pre_p_over']*100:.0f}%</b>
            </div>
        </div>
        <div style="color:#9ca3af; font-size:0.85em; margin-top:6px; border-top:1px solid #1f2937; padding-top:8px">
            📋 {row['reason']}
        </div>
        <div style="color:#4b5563; font-size:0.78em; margin-top:4px">
            ⚠️ Check your bookmaker's live screen — bet only if their odds > {fair_odds}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Raw table ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Raw data"):
    st.dataframe(df[[
        "league", "match", "score", "elapsed_mins", "signal_type",
        "bet", "fair_under_odds", "fair_over_odds",
        "live_p_under", "live_p_over", "pre_p_over", "lam_remaining"
    ]], use_container_width=True)

# ── All monitored live games ──────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"👁 All Monitored Games ({len(games)})")

if games.empty:
    st.info("No live games right now in our tracked leagues.")
else:
    signal_matches = set(df["match"].tolist()) if not df.empty else set()

    for _, g in games.iterrows():
        has_signal  = g["match"] in signal_matches
        has_pred    = str(g.get("has_prediction", "")).lower() == "true"
        border      = "#ff4444" if has_signal else ("#00cc88" if has_pred else "#444")
        badge       = "🚨 SIGNAL" if has_signal else ("📊 tracked" if has_pred else "👁 watching")
        badge_color = "#ff4444" if has_signal else ("#00cc88" if has_pred else "#666")

        p_under = g.get("live_p_under")
        fair    = g.get("fair_under_odds")
        p_str   = f"P(under): <b>{p_under}%</b> | Fair UNDER: <b>{fair}</b>" if pd.notna(p_under) and p_under else "No pre-match prediction"

        st.markdown(f"""
        <div style="border-left:3px solid {border}; padding:10px 14px; margin:6px 0;
                    background:#111827; border-radius:4px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="color:white; font-weight:bold">{g['match']}</span>
                <span style="color:#888; font-size:0.85em"> · {g['league']}</span><br/>
                <span style="color:#ccc">⏱ {g['elapsed_mins']}' &nbsp;|&nbsp; ⚽ <b>{g['score']}</b>
                &nbsp;|&nbsp; {p_str}</span>
            </div>
            <span style="background:{badge_color}22; color:{badge_color}; padding:3px 10px;
                         border-radius:10px; font-size:0.8em; white-space:nowrap">{badge}</span>
        </div>
        """, unsafe_allow_html=True)

st.caption(f"Signals refresh every 30 seconds · Only leagues where we find edge · Powered by Poisson statistics")
