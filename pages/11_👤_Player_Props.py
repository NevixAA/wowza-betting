"""Player Props — model predictions for SNIPER/VALUE matches."""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="Player Props | Wowza", page_icon="👤", layout="wide")

st.markdown("## 👤 Player Props")
st.caption(
    "Formula + ML model predictions for Anytime Goalscorer, SOT, Assists, and Yellow Cards. "
    "Only runs on SNIPER/VALUE matches — credit-efficient."
)

MARKET_LABELS = {
    "goals":   ("⚽ Anytime Goalscorer", "#e94560"),
    "assists": ("🎯 Assist",             "#f5a623"),
    "sot":     ("🔫 Shot on Target",     "#4fc3f7"),
    "cards":   ("🟨 Yellow Card",        "#ffd54f"),
}

MARKET_NOTE = {
    "goals":   "P(scores ≥ 1)",
    "assists": "P(assists ≥ 1)",
    "sot":     "P(SOT ≥ 1)",
    "cards":   "P(yellow card)",
}


@st.cache_data(ttl=120)
def load_tips() -> pd.DataFrame:
    f = config.OUTPUT_DIR / "player_tips.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    return df[df["date"] >= today].copy()


tips = load_tips()

# ── Status ────────────────────────────────────────────────────────────────────
models_exist = all(
    (config.MODELS_DIR / f"model_player_{m}.pkl").exists()
    for m in ["goals", "assists", "sot", "cards"]
)

if not models_exist:
    st.warning(
        "⚠️ Player models not trained yet. Run:\n\n"
        "```\npython -m player_model.pipeline --mode collect\n"
        "python -m player_model.pipeline --mode train\n```"
    )

if tips.empty:
    st.info(
        "No player prop tips for today yet. "
        "Run: `python -m player_model.pipeline --mode predict`"
    )
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    market_filter = st.multiselect(
        "Market", list(MARKET_LABELS.keys()),
        default=list(MARKET_LABELS.keys()),
        format_func=lambda m: MARKET_LABELS[m][0],
    )
with col2:
    match_filter = st.multiselect("Match", tips["match"].unique().tolist(), default=[])
with col3:
    tier_filter = st.multiselect(
        "Match Tier", ["SNIPER", "VALUE"], default=["SNIPER", "VALUE"]
    )

filtered = tips[tips["market"].isin(market_filter)]
if match_filter:
    filtered = filtered[filtered["match"].isin(match_filter)]
if tier_filter:
    filtered = filtered[filtered["match_tier"].isin(tier_filter)]

st.caption(f"{len(filtered)} player prop rows · Last updated: {datetime.now().strftime('%H:%M')}")
st.divider()

# ── Per-market sections ───────────────────────────────────────────────────────
for market in market_filter:
    label, color = MARKET_LABELS[market]
    note = MARKET_NOTE[market]
    df_m = filtered[filtered["market"] == market].copy()

    if df_m.empty:
        continue

    df_m = df_m.sort_values("model_prob", ascending=False)

    st.markdown(
        f"### <span style='color:{color}'>{label}</span> "
        f"<small style='color:#888'>({note})</small>",
        unsafe_allow_html=True,
    )

    # Display table
    display_cols = {
        "date":        "Date",
        "league":      "League",
        "match":       "Match",
        "match_tier":  "Match Tier",
        "player_name": "Player",
        "team":        "Team",
        "position":    "Pos",
        "model_prob":  "Model P",
        "fair_odds":   "Fair Odds",
        "bk_odds":     "BK Odds",
        "edge":        "Edge",
        "tier":        "Tier",
    }
    show = df_m[[c for c in display_cols if c in df_m.columns]].rename(columns=display_cols)
    show["Model P"] = show["Model P"].apply(lambda x: f"{x*100:.0f}%")
    show["Fair Odds"] = show["Fair Odds"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    show["BK Odds"] = show["BK Odds"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    show["Edge"] = show["Edge"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

    st.dataframe(show, use_container_width=True, hide_index=True)

# ── Add bookmaker odds manually ────────────────────────────────────────────────
st.divider()
with st.expander("💰 Add Bookmaker Odds to Calculate Edge"):
    st.caption(
        "Enter bookmaker odds for any player prop to calculate edge and tier. "
        "Changes are session-only."
    )
    col_p, col_m, col_o = st.columns([3, 2, 2])
    with col_p:
        sel_player = st.selectbox("Player", tips["player_name"].unique().tolist())
    with col_m:
        sel_market = st.selectbox(
            "Market", list(MARKET_LABELS.keys()),
            format_func=lambda m: MARKET_LABELS[m][0],
        )
    with col_o:
        bk_odds_input = st.number_input("Bookmaker Odds", min_value=1.01, value=2.50, step=0.05)

    if st.button("Calculate Edge"):
        row = tips[(tips["player_name"] == sel_player) & (tips["market"] == sel_market)]
        if row.empty:
            st.warning("Player / market combination not found in today's tips.")
        else:
            p_model  = float(row.iloc[0]["model_prob"])
            p_market = 1 / bk_odds_input
            edge     = p_model - p_market
            ev       = (p_model * bk_odds_input) - 1
            tier     = "SNIPER" if edge >= 0.08 else ("VALUE" if edge >= 0.04 else "AVOID")
            fair     = round(1 / max(p_model, 0.01), 2)

            st.markdown(f"""
            | | |
            |---|---|
            | Player | **{sel_player}** |
            | Market | {MARKET_LABELS[sel_market][0]} |
            | Model P | **{p_model*100:.0f}%** |
            | Fair Odds | **{fair}** |
            | BK Odds | **{bk_odds_input}** |
            | Implied P | {p_market*100:.0f}% |
            | Edge | **{edge*100:.1f}%** |
            | EV | **{ev:+.3f}** |
            | Tier | **{tier}** |
            """)
