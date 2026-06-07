"""
Access control helpers.
PUBLIC_MODE = True  →  read-only (cloud deployment, anyone can view)
PUBLIC_MODE = False →  admin mode (local or authenticated)
"""
import streamlit as st


def is_public() -> bool:
    """
    Returns True when running in public/read-only mode.
    Set via Streamlit Cloud secrets: PUBLIC_MODE = "true"
    Local app has no secret → admin mode.
    """
    try:
        return str(st.secrets.get("PUBLIC_MODE", "false")).lower() == "true"
    except Exception:
        return False


def admin_only(label: str = "") -> bool:
    """
    Show a lock notice instead of admin controls in public mode.
    Returns True when admin controls should be shown.
    """
    if is_public():
        return False
    return True
