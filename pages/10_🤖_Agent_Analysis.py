"""Agent Analysis — coming soon."""
import streamlit as st

st.set_page_config(page_title="Agent Analysis | Wowza", page_icon="🤖", layout="wide")

st.markdown("## 🤖 Match Analysis Agent")
st.info("🚧 **Coming soon** — being rebuilt with team-specific expected goals (λ_H / λ_A from actual form data) for meaningful correct score, BTTS, and 1X2 probabilities.")

st.markdown("""
**What it will do when ready:**
- λ_H from `home_scored_last5` × defense factor — match-specific, not generic split
- λ_A from `away_scored_last5` × defense factor
- BTTS, correct score, 1X2 probabilities — all team-specific
- Arbitrage check on provided market odds
- EV calculation across all markets
- AI context notes (Gemini) for narrative risk assessment
""")
