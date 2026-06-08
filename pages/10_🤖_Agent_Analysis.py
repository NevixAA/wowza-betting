"""Agent Analysis — run the Elite Football Analytics agent on SNIPER/VALUE picks."""
import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from agent.agent_runner import run_agent, build_prompt

st.set_page_config(page_title="Agent Analysis | Wowza", page_icon="🤖", layout="wide")

st.markdown("## 🤖 Agent Analysis — Sniper Value Signals")
st.caption("Runs the Elite Football Analytics agent on SNIPER/VALUE picks to find cross-market edges, arbitrage, and strongest signals.")

# ── API key status ─────────────────────────────────────────────────────────────
has_key = bool(os.getenv("ANTHROPIC_API_KEY", ""))
if has_key:
    st.success("✅ Anthropic API key detected — analysis runs automatically.")
else:
    st.warning(
        "⚠️ No ANTHROPIC_API_KEY found. Running in **manual mode** — "
        "copy the generated prompt and paste it into [Claude.ai](https://claude.ai) (free)."
    )

st.divider()

# ── Load picks ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_tips():
    f = config.OUTPUT_DIR / "bets.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    df = df[df["date"] >= today]
    return df[df["signal_tier"].isin(["SNIPER", "VALUE"]) & df["bet"].isin(["OVER", "UNDER"])]


tips = load_tips()

if tips.empty:
    st.info("No upcoming SNIPER/VALUE picks found. Run the predict pipeline first.")
    st.stop()

# ── Pick selector ──────────────────────────────────────────────────────────────
tips["label"] = (
    tips["signal_tier"] + " | "
    + tips["date"].dt.strftime("%d/%m")
    + "  "
    + tips["home_team"] + " vs " + tips["away_team"]
    + "  [" + tips.get("league", tips.get("home_team", "")).astype(str) + "]"
)

col_sel, col_btn = st.columns([4, 1])
with col_sel:
    selected_label = st.selectbox(
        "Select a pick to analyse:",
        tips["label"].tolist(),
        index=0,
    )
with col_btn:
    run_clicked = st.button("▶ Run Agent", use_container_width=True, type="primary")

selected_row = tips[tips["label"] == selected_label].iloc[0]

# Show pick summary card
side  = selected_row.get("best_side") or selected_row.get("bet", "")
odds  = selected_row["odds_under25"] if side == "UNDER" else selected_row["odds_over25"]
edge  = float(selected_row.get("best_edge", 0)) * 100
tier  = selected_row.get("signal_tier", "")
drift = selected_row.get("drift_signal", "New")
tier_color = "#e94560" if tier == "SNIPER" else "#f5a623"

st.markdown(f"""
<div style="background:#16213e;border:1.5px solid {tier_color};border-radius:8px;padding:12px 16px;margin:8px 0">
  <span style="color:{tier_color};font-weight:bold">{'🎯' if tier=='SNIPER' else '💡'} {tier}</span>
  &nbsp;&nbsp;
  <b style="color:white">{selected_row['home_team']} vs {selected_row['away_team']}</b>
  &nbsp;&nbsp;
  <span style="color:#90caf9">{selected_row.get('league','')}</span>
  &nbsp;|&nbsp;
  <span style="color:#00c896;font-weight:bold">{side} 2.5 @ {odds:.2f}</span>
  &nbsp;|&nbsp;
  Edge: <b style="color:#00c896">{edge:.1f}%</b>
  &nbsp;|&nbsp;
  Drift: <span style="color:#aaa">{drift}</span>
</div>
""", unsafe_allow_html=True)

# ── Run / show result ──────────────────────────────────────────────────────────
result_key = f"agent_result_{selected_label}"

if run_clicked:
    with st.spinner("Running agent analysis..."):
        result = run_agent(selected_row)
    st.session_state[result_key] = result

if result_key in st.session_state:
    result = st.session_state[result_key]

    if result["mode"] == "auto":
        st.markdown("### 📊 Agent Analysis")
        st.markdown(result["response"])
    else:
        st.markdown("### 📋 Manual Mode — Copy this prompt into Claude.ai")
        st.caption("Go to https://claude.ai → start a new chat → paste the prompt below.")
        st.code(result["response"], language="markdown")
        st.button(
            "📋 Copy to clipboard",
            on_click=lambda: st.write(
                f'<script>navigator.clipboard.writeText({repr(result["response"])})</script>',
                unsafe_allow_html=True,
            ),
        )

# ── Bulk: run all snipers ──────────────────────────────────────────────────────
st.divider()
snipers = tips[tips["signal_tier"] == "SNIPER"]

if not snipers.empty and has_key:
    st.markdown("### 🎯 Bulk — Analyse All SNIPER Picks")
    st.caption(f"{len(snipers)} SNIPER picks found")

    if st.button("▶ Run Agent on All SNIPER Picks", type="secondary"):
        all_results = []
        progress = st.progress(0)
        for i, (_, row) in enumerate(snipers.iterrows()):
            with st.spinner(f"Analysing {row['home_team']} vs {row['away_team']}..."):
                res = run_agent(row)
                all_results.append(res)
            progress.progress((i + 1) / len(snipers))

        for res in all_results:
            with st.expander(f"🎯 {res['match']}", expanded=True):
                st.markdown(res["response"])
