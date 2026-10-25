"""Account — who you are, which tier you have, log in / log out, and what each tier includes.

Visual redesign 2026-10-25 (stage 3): shared page header and section header; primary / secondary buttons.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import current_tier, login_configured, user_email  # noqa: E402
import dashboard_ui as ui  # noqa: E402

tier = current_tier()
email = user_email()
label = {"free": "Free", "pro": "Wowza Pro", "admin": "Admin"}[tier]
ui.page_header("Account", "Who you are, which plan you have, and what each plan includes.",
               eyebrow_text="System · account",
               freshness=ui.status_chip("NO_DATA", label=f"PLAN · {label.upper()}", tone="INFO"))

with st.container(border=True):
    if email:
        st.markdown(f"Signed in as **{email}** · plan: **{label}**")
        if tier == "free":
            st.info("This email has no Pro licence yet. Contact us to activate Pro for it.")
        if st.button("Log out", type="secondary"):
            st.logout()
    else:
        st.markdown(f"You are using **{label}** without signing in.")
        if login_configured():
            if st.button("Log in with Google", type="primary"):
                st.login()
        else:
            st.caption("Login is not configured on this deployment yet.")

ui.section_header("Plans")
st.markdown("""
| | **Free** | **Wowza Pro** |
|---|---|---|
| Daily tips | 2 a day, chosen by evidence | **every tip, every market** |
| Success rates | last 7 days | **7 · 14 · 30 · 60 · 90 days** |
| Player props, Fantasy, Bet Builder, Live | — | ✓ |
| Bankroll sizing & portfolio | — | ✓ |
""")
st.caption("Tips are information, not advice. Past results do not guarantee future results.")
