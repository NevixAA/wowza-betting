"""Access tiers for the dashboard: FREE / PRO / ADMIN (owner brief 2026-10-18).

    from dashboard_access import current_tier, require, WINDOWS
    tier = current_tier()            # "free" | "pro" | "admin"

WHO GETS WHAT
  free   anyone, no login            2 tips a day (top 2 by evidence) + success rates, last 7 days
  pro    logged in, email in PRO     all tips + portfolio + bankroll + success rates 7/14/30/60/90 d
  admin  logged in, email in ADMINS  everything (the admin center as it was)

LOGIN. Streamlit's built-in OIDC login (st.login, Google). Configured ONLY in the deployment's
secrets - never in code or in the repo:

    [auth]
    redirect_uri = "https://<app>.streamlit.app/oauth2callback"
    cookie_secret = "<random string>"
    client_id = "<Google OAuth client id>"
    client_secret = "<Google OAuth client secret>"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

    [access]
    admins = ["owner@example.com"]
    pro    = ["customer@example.com"]

DORMANT without [auth] secrets: everyone sees everything, exactly as before the tiers existed.
Once [auth] exists, FAIL CLOSED: not logged in, an error, or an email on neither list -> "free".
Locally, WOWZA_TIER_OVERRIDE=admin|pro|free sets the tier for testing (an environment variable a
visitor cannot set; on Streamlit Cloud only the owner can).
"""
from __future__ import annotations

import os

import streamlit as st

TIERS = ("free", "pro", "admin")
RANK = {t: i for i, t in enumerate(TIERS)}

# Success-rate windows each tier may choose from (days). The first entry is the default.
WINDOWS = {"free": [7], "pro": [7, 14, 30, 60, 90], "admin": [7, 14, 30, 60, 90, 0]}   # 0 = all time

FREE_TIPS_PER_DAY = 2


def free_tip_pick(plan, policy: dict, today, n: int = FREE_TIPS_PER_DAY):
    """The FREE tier's tips: (the n tips, their date, how many staked tips that date has).

    Only tips the approved bankroll policy would stake (stake_plan final_fraction > 0 - never a paper
    league, a PAPER / BLOCKED track or a price outside the track's caps), on the first date from
    `today` that has any, ranked by the track's evidence_score, then stake size. Pure function."""
    import pandas as pd
    if plan is None or plan.empty or "final_fraction" not in plan.columns:
        return (plan.iloc[0:0] if plan is not None else pd.DataFrame()), None, 0
    d = plan[pd.to_numeric(plan["final_fraction"], errors="coerce") > 0].copy()
    d["date"] = d["date"].astype(str).str[:10]
    d = d[d["date"] >= pd.Timestamp(today).strftime("%Y-%m-%d")]
    if d.empty:
        return d, None, 0
    day = d["date"].min()
    d = d[d["date"] == day]
    ev = {k: float(v.get("evidence_score", 0)) for k, v in (policy or {}).get("tracks", {}).items()}
    d["_ev"] = d["track"].map(ev).fillna(0.0)
    d = d.sort_values(["_ev", "final_fraction", "odds"], ascending=[False, False, True], kind="mergesort")
    return d.head(n), day, int(len(d))


def _emails(key: str) -> set[str]:
    try:
        return {str(e).strip().lower() for e in (st.secrets.get("access", {}) or {}).get(key, []) if str(e).strip()}
    except Exception:                                                # noqa: BLE001 - no secrets file
        return set()


def login_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:                                                # noqa: BLE001
        return False


def user_email() -> str | None:
    try:
        if getattr(st.user, "is_logged_in", False):
            return str(st.user.email or "").strip().lower() or None
    except Exception:                                                # noqa: BLE001
        pass
    return None


def current_tier() -> str:
    ov = os.getenv("WOWZA_TIER_OVERRIDE", "").strip().lower()
    if ov in TIERS:
        return ov
    # DORMANT UNTIL LOGIN EXISTS (owner 2026-10-20: ship the interface now). With no [auth] secrets
    # the app behaves exactly as it always has - everything visible - so merging cannot lock anyone,
    # the owner included, into Free. The moment [auth] is added to the deployment's secrets, tiers
    # switch on and the fail-closed rules below apply.
    if not login_configured():
        return "admin"
    email = user_email()
    if not email:
        return "free"
    if email in _emails("admins"):
        return "admin"
    if email in _emails("pro"):
        return "pro"
    return "free"


def allows(tier: str, needed: str) -> bool:
    return RANK.get(tier, 0) >= RANK[needed]


def require(needed: str) -> str:
    """Stop the page unless the viewer's tier is at least `needed`. Defence in depth: navigation
    already hides pages a tier cannot see, but a page must not rely on that alone."""
    tier = current_tier()
    if not allows(tier, needed):
        st.warning(f"This page is part of Wowza **{needed.title()}**. See the Account page to log in or upgrade.")
        st.stop()
    return tier


# ── navigation ───────────────────────────────────────────────────────────────────────────────
# (file, title, icon, minimum tier). The ONE place that decides who sees which page.
PAGES = [
    ("pages/free_tips.py",              "Today's free tips", "🎁", "free"),
    ("pages/00_🎯_Home.py",              "Home",              "🎯", "pro"),
    ("pages/1_📊_Dashboard.py",          "Tips",              "📊", "pro"),
    ("pages/17_🏟️_Leagues.py",           "Leagues",           "🏟️", "pro"),
    ("pages/11_👤_Player_Props.py",      "Player Props",      "👤", "pro"),
    ("pages/13_⚽_Fantasy.py",           "Fantasy",           "⚽", "pro"),
    ("pages/14_🧩_Bet_Builder.py",       "Bet Builder",       "🧩", "pro"),
    ("pages/6_⚡_Live.py",               "Live",              "⚡", "pro"),
    ("pages/5_🎯_Success_Rates.py",      "Success Rates",     "✅", "free"),
    ("pages/12_💼_Portfolio.py",         "Portfolio",         "💼", "pro"),
    ("pages/16_💰_Bankroll.py",          "Bankroll",          "💰", "pro"),
    ("pages/0_🏠_Overview.py",           "Overview",          "🏠", "admin"),
    ("pages/9_🎓_Model_Performance.py",  "Model Performance", "🎓", "admin"),
    ("pages/4_ℹ️_Model_Info.py",         "Model Info",        "ℹ️", "admin"),
    ("pages/7_🔬_Pro.py",                "Pro research",      "🔬", "admin"),
    ("pages/8_🛰️_V11_Market.py",         "V11 Market",        "🛰️", "admin"),
    ("pages/10_🤖_Agent_Analysis.py",    "Agent Analysis",    "🤖", "admin"),
    ("pages/15_🔭_League_Scout.py",      "League Scout",      "🔭", "admin"),
    ("pages/18_🔧_League_Research.py",   "League Repair Lab", "🔧", "admin"),
    ("pages/account.py",                "Account",           "👤", "free"),
]

SECTIONS = {"free": "Free", "pro": "Wowza Pro", "admin": "Admin center"}


def pages_for(tier: str) -> dict[str, list]:
    """st.navigation input: sections of st.Page for everything this tier may open."""
    out: dict[str, list] = {}
    default = "pages/00_🎯_Home.py" if allows(tier, "pro") else "pages/free_tips.py"
    for path, title, icon, need in PAGES:
        if not allows(tier, need):
            continue
        section = "Account" if path.endswith("account.py") else SECTIONS[need]
        out.setdefault(section, []).append(st.Page(path, title=title, icon=icon, default=(path == default)))
    return out
