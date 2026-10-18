"""
Wowza — entry point and ROUTER. Run with: streamlit run app.py

Since 2026-10-18 the app has three tiers (dashboard_access.py): FREE (anyone - 2 tips a day and the
last 7 days of success rates), PRO (licensed users - all tips, portfolio, bankroll, success rates over
7/14/30/60/90 days) and ADMIN (the full admin center). st.navigation shows each viewer only the pages
their tier allows; Streamlit's automatic pages/ sidebar is replaced by it. The old home screen is now
pages/00_🎯_Home.py.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_access import current_tier, pages_for  # noqa: E402

st.set_page_config(page_title="Wowza", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

tier = current_tier()
nav = st.navigation(pages_for(tier))
with st.sidebar:
    st.caption({"free": "Wowza Free", "pro": "Wowza Pro", "admin": "Admin center"}[tier])
nav.run()
