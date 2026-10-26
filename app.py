"""
Wowza — entry point and ROUTER. Run with: streamlit run app.py

Since 2026-10-18 the app has three tiers (dashboard_access.py): FREE (anyone - 2 tips a day and the
last 7 days of success rates), PRO (licensed users - all tips, portfolio, bankroll, success rates over
7/14/30/60/90 days) and ADMIN (the full admin center). st.navigation shows each viewer only the pages
their tier allows; Streamlit's automatic pages/ sidebar is replaced by it. The old home screen is now
pages/00_🎯_Home.py.

Since 2026-10-25 the sidebar is grouped by user intent (dashboard_access.NAV: Today, Performance,
Tools, Research Lab, System). Tiers still decide which pages appear; sections only decide where.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_access import current_tier, pages_for  # noqa: E402
from dashboard_data import remote_mirror  # noqa: E402

st.set_page_config(page_title="Wowza", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# SLIM deployment (2026-10-26): the app is deployed from a generated code-only repo and the data the
# pages read is mirrored from v9's public raw URLs into the same local paths (deploy/README_DASHBOARD_
# DEPLOY.md). A no-op on a full checkout. The first call in a process waits for one sync; after that
# a refresh runs at most every 10 minutes in a background thread and a render never waits for it.
if remote_mirror.slim_mode():
    if not remote_mirror.status().get("synced_at_utc"):
        with st.spinner("Fetching the latest Wowza data…"):
            _mirror = remote_mirror.ensure_fresh()
    else:
        _mirror = remote_mirror.ensure_fresh()

tier = current_tier()
nav = st.navigation(pages_for(tier))
with st.sidebar:
    st.caption({"free": "Wowza Free", "pro": "Wowza Pro", "admin": "Admin center"}[tier])
    if remote_mirror.slim_mode() and tier == "admin":
        st.caption(f"Data mirror: synced {_mirror.get('synced_at_utc') or 'never'} UTC"
                   + (f" · v9 `{str(_mirror.get('sha'))[:7]}`" if _mirror.get("sha") else ""))
nav.run()
