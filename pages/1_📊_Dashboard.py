"""Dashboard — upcoming tips with drift signals."""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="Dashboard | Wowza", page_icon="📊", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="dash_refresh")

PYTHON = str(Path(sys.executable))
V9_DIR = str(Path(__file__).resolve().parents[1])


@st.cache_data(ttl=30)
def load_bets():
    f = config.OUTPUT_DIR / "bets.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Ensure optional columns exist
    for col in ["model_type", "drift_signal", "signal_tier", "best_edge", "best_side"]:
        if col not in df.columns:
            df[col] = ""
    df["model_type"]  = df["model_type"].fillna("").astype(str)
    df["drift_signal"] = df["drift_signal"].fillna("New").astype(str)
    df["signal_tier"]  = df["signal_tier"].fillna("").astype(str)
    return df


def drift_badge(signal: str) -> str:
    colors = {"Confirmed": "#00c896", "Conflicted": "#e94560", "Neutral": "#aaa", "New": "#666"}
    icons  = {"Confirmed": "✅", "Conflicted": "⚠️", "Neutral": "➡️", "New": "🆕"}
    c = colors.get(signal, "#666")
    i = icons.get(signal, "")
    return f'<span style="color:{c};font-weight:bold">{i} {signal}</span>'


def odds_badge(side: str, odds: float) -> str:
    color = "#4fc3f7" if side == "OVER" else "#ce93d8"
    return f'<span style="color:{color};font-weight:bold">{side} @ {odds:.2f}</span>'


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Today's Tips")

col_run, col_upd, col_time = st.columns([2, 2, 6])

with col_run:
    if st.button("🔄 Run Predict Now", use_container_width=True):
        with st.spinner("Fetching fixtures & computing predictions..."):
            result = subprocess.run(
                [PYTHON, "pipeline.py", "--mode", "predict"],
                cwd=V9_DIR, capture_output=True, text=True, timeout=120
            )
        if result.returncode == 0:
            st.success("Predictions updated!")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr[-500:]}")

with col_upd:
    if st.button("📥 Update Results Now", use_container_width=True):
        with st.spinner("Fetching match results..."):
            result = subprocess.run(
                [PYTHON, "update_results.py"],
                cwd=V9_DIR, capture_output=True, text=True, timeout=120
            )
        if result.returncode == 0:
            st.success("Results updated!")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr[-500:]}")

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_bets()

if df.empty:
    st.warning("No predictions yet. Click 'Run Predict Now' to fetch tips.")
    st.stop()

bets_file = config.OUTPUT_DIR / "bets.csv"
mod_time  = datetime.fromtimestamp(bets_file.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
st.caption(f"Last updated: {mod_time}  •  {len(df)} fixtures fetched")

# ── Filters ────────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    tier_filter = st.multiselect("Tier", ["SNIPER", "VALUE"], default=["SNIPER", "VALUE"])
with col_f2:
    available_models = sorted(df["model_type"].unique().tolist()) or ["standard", "new_format"]
    model_filter = st.multiselect("Model", available_models, default=available_models)
with col_f3:
    drift_filter = st.multiselect(
        "Drift Signal", ["Confirmed", "Neutral", "New", "Conflicted"],
        default=["Confirmed", "Neutral", "New", "Conflicted"]
    )

# Active tips only
tips = df[
    df["bet"].isin(["OVER", "UNDER"]) &
    df["signal_tier"].isin(tier_filter) &
    (df["model_type"].isin(model_filter) if model_filter else True) &
    df["drift_signal"].isin(drift_filter)
].copy()

st.divider()

# ── SNIPER section ─────────────────────────────────────────────────────────────
snipers = tips[tips["signal_tier"] == "SNIPER"].sort_values("best_edge", ascending=False)
values  = tips[tips["signal_tier"] == "VALUE"].sort_values("best_edge", ascending=False)

if not snipers.empty:
    st.markdown(f"### 🎯 SNIPER Tips &nbsp; <small style='color:#e94560'>({len(snipers)} bets)</small>",
                unsafe_allow_html=True)
    for _, row in snipers.iterrows():
        side  = row.get("best_side") or row.get("bet", "")
        odds  = row["odds_under25"] if side == "UNDER" else row["odds_over25"]
        edge  = float(row.get("best_edge", 0)) * 100
        model = "🏴󠁧󠁢󠁥󠁮󠁧󠁿" if row.get("model_type") == "standard" else "🌍"
        drift = str(row.get("drift_signal", "New"))

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:2px solid #e94560;border-radius:10px;
                    padding:14px 18px;margin-bottom:8px">
          <span style="color:#e94560;font-size:1.1em;font-weight:bold">🎯 SNIPER</span>
          &nbsp;&nbsp;{model}
          <span style="float:right;color:#aaa">{str(row['date'])[:10]}</span><br>
          <span style="color:white;font-size:1.05em">
            <b>{row['home_team']}</b> vs <b>{row['away_team']}</b>
          </span><br>
          <span style="color:#90caf9">{row.get('league','')}</span>
          &nbsp;&nbsp;
          {odds_badge(side, odds)}
          &nbsp;&nbsp;
          <span style="color:#00c896;font-weight:bold">Edge: {edge:.1f}%</span>
          &nbsp;&nbsp;
          {drift_badge(drift)}
        </div>
        """, unsafe_allow_html=True)
else:
    if "SNIPER" in tier_filter:
        st.info("No SNIPER tips matching current filters.")

# ── VALUE section ──────────────────────────────────────────────────────────────
if not values.empty:
    st.markdown(f"### 💡 VALUE Tips &nbsp; <small style='color:#f5a623'>({len(values)} bets — half stake)</small>",
                unsafe_allow_html=True)

    rows_display = []
    for _, row in values.iterrows():
        side  = row.get("best_side") or row.get("bet", "")
        odds  = row["odds_under25"] if side == "UNDER" else row["odds_over25"]
        edge  = float(row.get("best_edge", 0)) * 100
        drift = str(row.get("drift_signal", "New"))
        model = "Standard 🏴󠁧󠁢󠁥󠁮󠁧󠁿" if row.get("model_type") == "standard" else "New-Format 🌍"
        rows_display.append({
            "Date":     str(row["date"])[:10],
            "League":   row.get("league", ""),
            "Match":    f"{row['home_team']} vs {row['away_team']}",
            "Side":     side,
            "Odds":     f"{odds:.2f}",
            "Edge":     f"{edge:.1f}%",
            "Drift":    drift,
            "Model":    model,
        })

    st.dataframe(
        pd.DataFrame(rows_display),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Odds":  st.column_config.NumberColumn(format="%.2f"),
        }
    )

if tips.empty:
    st.warning("No tips match your filters.")

# ── Suppressed bets note ────────────────────────────────────────────────────────
avoided = df[df["bet"] == "AVOID"]
if not avoided.empty:
    with st.expander(f"🚫 {len(avoided)} fixtures suppressed (AVOID / both-losing guard)"):
        st.dataframe(
            avoided[["date", "league", "home_team", "away_team", "signal_tier", "model_type"]],
            use_container_width=True, hide_index=True
        )
