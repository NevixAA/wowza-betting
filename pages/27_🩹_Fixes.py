"""Fixes - what was repaired in the data, newest first (owner 2026-11-17: "update the dashboard with ... all the
fixes we did").

VIEWER ONLY. Every line comes from reports/dashboard/repairs_log.json, built from the repo history: each repair says
when, what was wrong, what changed, the real numbers (each copied from the commit, PR or report it cites; "not
recorded" where no committed source holds one) and where it lives (main, data-truth-100 which goes live on
2026-11-21, a branch, or in progress). No price, P/L figure or credential is in the file.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require  # noqa: E402
require("pro")

st.set_page_config(page_title="Wowza | Fixes", page_icon="🩹", layout="wide")

import dashboard_ui as ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "reports" / "dashboard" / "repairs_log.json"
REPO_URL = "https://github.com/NevixAA/wowza-betting"

ui.page_header("What was fixed",
               "The data-truth repairs behind the Clean Record, newest first: what was wrong, what changed and the "
               "real numbers, each with the commit or pull request that holds it.",
               eyebrow_text="Performance · repairs log")


def _load():
    try:
        d = json.loads(LOG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"{LOG.relative_to(ROOT).as_posix()} cannot be read ({type(e).__name__})"
    if d.get("schema") != "wowza.repairs_log/1" or not isinstance(d.get("items"), list):
        return None, f"unexpected schema {d.get('schema')!r}"
    return d, None


def _ref(r: dict) -> str:
    k, i, b = r.get("kind"), str(r.get("id", "")), r.get("branch")
    repo = r.get("repo") or "NevixAA/wowza-betting"
    url = f"https://github.com/{repo}"
    tag = "" if repo == "NevixAA/wowza-betting" else f"{repo.split('/')[-1]} "
    if k == "pr" and i.lstrip("#").isdigit():
        return f"[{tag}PR {i}]({url}/pull/{i.lstrip('#')})"
    if k == "commit":
        return f"[{tag}{i}]({url}/commit/{i})" + (f" ({b})" if b else "")
    if k == "report":
        return f"`{i}`" + (f" ({b})" if b else "")
    return i


def _where_chip(where: str) -> str:
    w = where.lower()
    if "in progress" in w:
        return ui.status_chip("ACCRUING", label="IN PROGRESS")
    if "not merged" in w:
        return ui.status_chip("REGISTERED", label="ON A BRANCH")
    if w.startswith("main"):
        return ui.status_chip("LIVE", label="LIVE ON MAIN")
    return ui.status_chip("QUALIFIED", label="GOES LIVE 21.11")


D, err = _load()
if D is None:
    ui.empty_state("ERROR", f"{err}. Nothing is shown in its place.")
    st.stop()

items = sorted(D["items"], key=lambda x: str(x.get("date", "")), reverse=True)
where = pd.Series([_where_chip(i.get("where", "")) for i in items])
ui.stat_line([("Repairs listed", str(len(items))),
              ("Live on main", str(sum(str(i.get("where", "")).lower().startswith("main") for i in items))),
              ("Go live 21.11", str(sum("data-truth-100" in str(i.get("where", "")) and
                                        not str(i.get("where", "")).lower().startswith("main") for i in items))),
              ("Branch / in progress", str(sum(("not merged" in str(i.get("where", "")).lower()) or
                                               ("in progress" in str(i.get("where", "")).lower()) for i in items)))])

ui.section_header("Repairs, newest first")
for it, chip in zip(items, where):
    with st.expander(f"{it.get('date', '?')} · {it.get('title', '')}", expanded=False):
        st.markdown(chip + "  " + ui.tone_text(str(it.get("where", "")), "NEUTRAL"))
        st.markdown(f"**What was wrong.** {it.get('what_was_wrong', '')}")
        st.markdown(f"**What changed.** {it.get('what_changed', '')}")
        nums = it.get("numbers") or []
        if nums:
            st.markdown("**The numbers.**\n" + "\n".join(f"- {n}" for n in nums))
        if it.get("rules"):
            ui.compact_table(pd.DataFrame([{"Rule": r.get("rule"), "Name": r.get("name"), "Checks": r.get("checks"),
                                            "Blocks a push": {True: "yes", False: "no"}.get(r.get("gated"),
                                                                                             str(r.get("gated")))}
                                           for r in it["rules"]]))
            if it.get("rules_note"):
                st.caption(it["rules_note"])
        refs = it.get("refs") or []
        if refs:
            st.caption("Sources: " + " · ".join(_ref(r) for r in refs))

src = D.get("source_refs") or {}
st.caption(f"{D.get('about', '')} Refs: " + ", ".join(f"{k} {str(v)[:9]}" for k, v in src.items()) + ".")

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
