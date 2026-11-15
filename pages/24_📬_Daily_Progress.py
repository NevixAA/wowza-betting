"""Daily Progress — the owner's daily progress report, on the dashboard (owner 2026-11-15).

The report replaces the morning e-mail (Gmail refused SMTP logins from GitHub runners) and runs until
2026-11-21. It is built every morning by the PRIVATE repo NevixAA/wowza-exec
(.github/workflows/daily_progress_dashboard.yml -> scripts/daily_progress_report.py, from real repo
data only) and committed to that repo's `progress-report` branch as report.json. This page is a
VIEWER of that one file: it shows each line exactly as the report wrote it and re-derives nothing.

WHY IT IS FETCHED, NOT MIRRORED. v9 and the slim wowza-dashboard repo are public, and every data file
in deploy/dashboard_manifest.json is downloaded from v9's public raw URLs. The report is owner-only,
so it never enters v9 or the manifest. The page reads it server-side through the GitHub API with a
READ-ONLY fine-grained token (wowza-exec only, Contents: read) kept in the deployment's Streamlit
secrets:

    [exec_report]
    token = "<fine-grained PAT>"

(locally: env WOWZA_EXEC_REPORT_TOKEN). The token is never shown, logged or put in an error message.

WHO SEES IT. ADMIN tier only (dashboard_access.PAGES), and FAIL CLOSED beyond that: when login is not
configured the tiers are dormant and everyone counts as admin, so this page then refuses to load the
report unless WOWZA_TIER_OVERRIDE=admin (an environment variable only the owner can set).
"""
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_access import require as _require_tier, login_configured  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES) - owner only, never a public tier

st.set_page_config(page_title="Wowza | Daily Progress", page_icon="📬", layout="wide")

REPO = "NevixAA/wowza-exec"
BRANCH = "progress-report"
SCHEMA = "wowza-daily-progress/1"
END_DATE = dt.date(2026, 11, 21)          # owner-set last report day (Israel date)
STALE_HOURS = 26                          # one daily run plus slack; older = yesterday's report
TZ = ZoneInfo("Asia/Jerusalem")
API = "https://api.github.com"

st.title("📬 Daily Progress")
st.caption("The owner's daily progress report until 2026-11-21, built each morning (06:20 and 07:20 Israel) "
           "from committed repo data and the GitHub Actions API. Shown exactly as written; nothing is "
           "re-derived. 'unknown' = could not be read. OddsAPI: credits and counts only, never a price.")

# ── fail closed when the tiers are dormant (no login configured -> everyone would be 'admin') ──────
if not login_configured() and os.getenv("WOWZA_TIER_OVERRIDE", "").strip().lower() != "admin":
    st.warning("Login is not configured on this deployment, so access tiers are dormant. The daily "
               "progress report is owner-only and is not loaded without login.")
    st.stop()


def _token() -> str:
    try:
        tok = (st.secrets.get("exec_report", {}) or {}).get("token")
    except Exception:                                                # noqa: BLE001 - no secrets file
        tok = None
    return str(tok or os.getenv("WOWZA_EXEC_REPORT_TOKEN") or "").strip()


def _get(path: str) -> bytes:
    """GET a file of the progress-report branch (raw). Errors never carry the token."""
    tok = _token()
    if not tok:
        raise RuntimeError("no read token configured (Streamlit secret [exec_report] token)")
    url = (f"{API}/repos/{REPO}/contents/{urllib.parse.quote(path)}"
           f"?ref={urllib.parse.quote(BRANCH, safe='')}")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.raw", "Authorization": f"Bearer {tok}",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "wowza-dashboard-daily-progress"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        hint = {401: "the token is invalid or expired", 403: "the token may not read wowza-exec contents",
                404: f"not found on {BRANCH} (or the token cannot see wowza-exec)"}.get(exc.code, "")
        raise RuntimeError(f"GitHub answered HTTP {exc.code} for {path}" + (f": {hint}" if hint else "")) from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error ({exc.reason})") from None


@st.cache_data(ttl=600, show_spinner=False)
def _fetch(path: str) -> dict:
    """Cached for 10 min. A failure raises, and Streamlit never caches an exception, so a fixed
    token or a fresh report shows on the next rerun."""
    data = json.loads(_get(path).decode("utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        got = data.get("schema") if isinstance(data, dict) else type(data).__name__
        raise RuntimeError(f"{path} has schema {got!r}, expected {SCHEMA!r}")
    return data


def _load(path: str) -> tuple[dict | None, str | None]:
    try:
        return _fetch(path), None
    except Exception as exc:                                         # noqa: BLE001
        return None, str(exc)


@st.cache_data(ttl=600, show_spinner=False)
def _history_names() -> list[str]:
    items = json.loads(_get("history").decode("utf-8"))
    return sorted((i["name"][:-5] for i in items if isinstance(i, dict) and str(i.get("name", "")).endswith(".json")),
                  reverse=True)


def _history() -> list[str]:
    try:
        return _history_names()
    except Exception:                                                # noqa: BLE001 - the latest still shows
        return []


_MD_SPECIAL = "\\`*_{}[]<>#|~$"


def _md(text: str) -> str:
    """Report text as literal markdown (repo text such as file names must not format or link)."""
    return "".join("\\" + c if c in _MD_SPECIAL else c for c in str(text))


def _line(l: dict) -> str:
    mark = "🔴 " if l.get("alert") else ""
    url = l.get("url")
    link = f" [link]({url})" if isinstance(url, str) and url.startswith("https://github.com/") else ""
    return ("    " if l.get("level") else "") + f"- {mark}{_md(l.get('text', ''))}{link}"


# ── which report ──────────────────────────────────────────────────────────────────────────────
days = _history()
choice = st.selectbox("Report", ["Latest"] + days, index=0,
                      help="Latest = report.json; earlier days = the last report of that Israel date.")
path = "report.json" if choice == "Latest" else f"history/{choice}.json"
rep, err = _load(path)
if rep is None:
    st.error(f"**The daily progress report could not be loaded** — {_md(err)}. Nothing is shown in its place.")
    st.caption(f"Source: {REPO} branch {BRANCH}, {path} (private).")
    st.stop()

# ── header: subject, age, counts ──────────────────────────────────────────────────────────────
gen_raw = str(rep.get("generated_at_utc") or "")
try:
    gen = dt.datetime.strptime(gen_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
except ValueError:
    gen = None
now = dt.datetime.now(dt.timezone.utc)
st.subheader(_md(rep.get("subject", "")))
if gen is None:
    st.error(f"Generated at: unknown (generated_at_utc = {_md(gen_raw)!s}) — age cannot be judged.")
else:
    age_h = (now - gen).total_seconds() / 3600
    st.caption(f"Generated {gen.strftime('%Y-%m-%d %H:%M')} UTC = {gen.astimezone(TZ).strftime('%Y-%m-%d %H:%M')} "
               f"Israel ({age_h:.1f} h ago). Report date (Israel): {_md(rep.get('report_date_israel', 'unknown'))}.")
    if choice == "Latest" and age_h > STALE_HOURS:
        if now.astimezone(TZ).date() > END_DATE:
            st.info(f"The daily report ended on {END_DATE.isoformat()} (owner-set). This is the last one.")
        else:
            st.warning(f"**STALE** — this report is {age_h:.0f} h old (more than {STALE_HOURS} h): this "
                       "morning's run did not publish. Check the wowza-exec workflow 'Daily progress report "
                       "(owner dashboard)'.")

sections = rep.get("sections") or []
n_alert = sum(1 for s in sections for l in (s.get("lines") or []) if l.get("alert"))
n_fail = sum(1 for s in sections if str(s.get("title", "")).startswith("5.")
             for l in (s.get("lines") or []) if l.get("alert") and l.get("url"))
c1, c2, c3 = st.columns(3)
c1.metric("Items to check", n_alert)
c2.metric("Failed runs (24 h)", n_fail)
c3.metric("Go-live", str(rep.get("go_live_date", "unknown")))

# ── the five sections, as written ─────────────────────────────────────────────────────────────
for s in sections:
    lines = s.get("lines") or []
    alerts = sum(1 for l in lines if l.get("alert"))
    title = _md(s.get("title", ""))
    with st.expander(f"{title}" + (f"  —  🔴 {alerts} to check" if alerts else ""), expanded=True):
        st.markdown("\n".join(_line(l) for l in lines) if lines else "_no lines in this section_")

st.caption(_md(rep.get("note", "")) + f"  Source: {REPO} branch {BRANCH}, {path} (private). "
           "🔴 = needs a look. Links open GitHub (owner login needed for wowza-exec).")
