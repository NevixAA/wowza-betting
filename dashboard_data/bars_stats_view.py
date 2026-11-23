"""Per-league bars for the Bars page - STATISTICS ONLY: the bars production SERVES.

SOURCE (read, never written): reports/dashboard/bars_statistics.json, built by scripts/build_bars_statistics.py from
config/threshold_applied.json (the applied proposal), its approval in config/threshold_approvals.json and the
proposal's evidence file. It holds no bookmaker price: statuses, bars, the odds-range rule of a cell, bet counts, ROI,
its 95% CI and mean CLV % only. `serving_problems` compares the file with config/threshold_applied.json at view time,
so a stale file is named on the page instead of shown as current.

Nothing here has a default: a missing or unreadable file is a named problem and an empty table; a count the evidence
does not record is shown as "not recorded", never filled.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "reports" / "dashboard" / "bars_statistics.json"
APPLIED = ROOT / "config" / "threshold_applied.json"
SCHEMA = "wowza.bars_statistics/2"
STATUS_ORDER = ["PROVEN", "PROMISING", "UNPROVEN", "OFF"]
NOT_RECORDED = "not recorded"
COLUMNS = ["League", "Track", "Market", "Status", "Why OFF", "Bar", "Measure", "Odds range", "Mode", "Stake rule",
           "Counts", "WF bets", "WF ROI %", "Study bets", "Study ROI %", "ROI 95% CI", "CLV %", "OFF detail"]


def load() -> dict:
    """{'data': the file or None, 'problems': [...]}."""
    try:
        d = json.loads(STATS.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"data": None, "problems": [f"{STATS.relative_to(ROOT).as_posix()} cannot be read ({type(e).__name__})"]}
    if d.get("schema") != SCHEMA:
        return {"data": None, "problems": [f"unexpected schema {d.get('schema')!r}"]}
    return {"data": d, "problems": []}


def proposal(d: dict | None, key: str) -> dict | None:
    for p in (d or {}).get("proposals") or []:
        if p.get("key") == key:
            return p
    return None


def serving(d: dict | None) -> dict | None:
    """The proposal the file labels SERVING."""
    return proposal(d, "serving")


def applied_id() -> str | None:
    """proposal_id production serves (config/threshold_applied.json last_applied), or None if unreadable."""
    try:
        return (json.loads(APPLIED.read_text(encoding="utf-8")).get("last_applied") or {}).get("proposal_id")
    except (OSError, ValueError):
        return None


def serving_problems(d: dict | None) -> list[str]:
    """Named problems when the file's serving proposal is not what production serves."""
    p = serving(d)
    live = applied_id()
    if live is None:
        return ["config/threshold_applied.json cannot be read: the serving proposal cannot be confirmed"]
    if not p:
        return ["the statistics file has no serving proposal"]
    if p.get("proposal_id") != live:
        return [f"the statistics file shows {p.get('proposal_id')} but production serves {live}: rebuild with "
                "scripts/build_bars_statistics.py"]
    return []


def _pct(x):
    return None if x is None else round(100 * float(x), 1)


def odds_range(lo, hi) -> str:
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"≤ {float(hi):.2f}"
    if hi is None:
        return f"≥ {float(lo):.2f}"
    return f"{float(lo):.2f} – {float(hi):.2f}"


def fmt_ci(lo, hi) -> str:
    if lo is None or hi is None:
        return "—"
    return f"{100 * lo:+.1f}% … {100 * hi:+.1f}%"


def frame(p: dict | None, *, with_fraction: bool = False) -> pd.DataFrame:
    """One display row per league x market cell of a proposal, PROVEN first, OFF last (OFF shown with its reason)."""
    if not p or not p.get("cells"):
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for c in p["cells"]:
        off = c["status"] == "OFF"
        rec = bool(c.get("counts_recorded"))
        rows.append({
            "League": c["league"], "Track": c["track_label"], "Market": c["market"], "Status": c["status"],
            "Why OFF": c.get("off_reason") if off else None,
            "Bar": None if off or c["bar"] is None else round(100 * c["bar"], 1),
            "Measure": None if off else c.get("measure"),
            "Odds range": None if off else odds_range(c.get("odds_min"), c.get("odds_max")),
            "Mode": None if off else c.get("mode"),
            "Stake rule": c["stake_rule"],
            "Counts": "recorded" if rec else NOT_RECORDED,
            "WF bets": c.get("wf_bets"), "WF ROI %": _pct(c.get("wf_roi")),
            "Study bets": c.get("study_bets"), "Study ROI %": _pct(c.get("study_roi")),
            "ROI 95% CI": fmt_ci(c.get("study_ci_lo"), c.get("study_ci_hi")) if rec else NOT_RECORDED,
            "CLV %": _pct(c.get("study_clv")),
            "OFF detail": c.get("off_detail") if off else None,
            "_fraction": c.get("stake_fraction"),
        })
    f = pd.DataFrame(rows, columns=COLUMNS + ["_fraction"])
    f["_o"] = f["Status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    f = f.sort_values(["_o", "League", "Track", "Market"], na_position="last").drop(columns="_o") \
        .reset_index(drop=True)
    return f if with_fraction else f.drop(columns="_fraction")


def staked(p: dict | None) -> pd.DataFrame:
    """The cells the stake rule stakes (a recorded fraction > 0)."""
    f = frame(p, with_fraction=True)
    if not len(f):
        return frame(p)
    fr = pd.to_numeric(f["_fraction"], errors="coerce")
    return f[fr.gt(0).to_numpy()].drop(columns="_fraction").reset_index(drop=True)


def counts(p: dict | None) -> dict:
    return {s: int(((p or {}).get("counts") or {}).get(s, 0)) for s in STATUS_ORDER}


def off_reasons(p: dict | None) -> pd.DataFrame:
    """OFF cells per reason code."""
    cols = ["Why OFF", "Cells"]
    offs = [c.get("off_reason") or "—" for c in (p or {}).get("cells") or [] if c["status"] == "OFF"]
    if not offs:
        return pd.DataFrame(columns=cols)
    s = pd.Series(offs).value_counts()
    return pd.DataFrame({"Why OFF": s.index, "Cells": s.to_numpy()}, columns=cols)


def families_frame(p: dict | None) -> pd.DataFrame:
    """Per track x market family: rows in the study, configurations tested, cells on / OFF."""
    cols = ["Track", "Market", "Study rows", "Configurations tested", "Chosen configuration", "Cells on",
            "Cells OFF"]
    rows = [{"Track": f["track_label"], "Market": f["market"], "Study rows": f.get("rows"),
             "Configurations tested": f.get("configs_tested"), "Chosen configuration": f.get("chosen"),
             "Cells on": f.get("cells_on"), "Cells OFF": f.get("cells_off")}
            for f in (p or {}).get("families") or []]
    return pd.DataFrame(rows, columns=cols)


def history_frame(d: dict | None) -> pd.DataFrame:
    """The owner's approval chain (newest first), from config/threshold_approvals.json."""
    cols = ["Proposal", "State", "Approved", "Approved entries", "Supersedes"]
    rows = [{"Proposal": h["proposal_id"], "State": h["state"], "Approved": str(h.get("approved_at"))[:10],
             "Approved entries": h.get("entries"), "Supersedes": h.get("supersedes") or "—"}
            for h in (d or {}).get("history") or []]
    return pd.DataFrame(rows, columns=cols)


def stake_plan_frame(d: dict | None) -> pd.DataFrame:
    """The stake plan's real-stake cells, with their status in the serving proposal."""
    plan = (d or {}).get("stake_plan") or {}
    p = serving(d)

    def _st(c):
        for x in (p or {}).get("cells") or []:
            if (x["track"], x["mside"], x["league"]) == (c["track"], c["mside"], c["league"]):
                return x
        return None
    rows = []
    for c in plan.get("cells") or []:
        x = _st(c) or {}
        rows.append({"League": c["league"], "Track": x.get("track_label", c["track"]),
                     "Market": x.get("market", c["mside"]), "Status": x.get("status", "—"),
                     "Stake": x.get("stake_rule", "—")})
    return pd.DataFrame(rows, columns=["League", "Track", "Market", "Status", "Stake"])
