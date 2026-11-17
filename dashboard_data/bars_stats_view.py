"""Per-league bars for the Bars page - STATISTICS ONLY (owner 2026-11-17: "update the dashboard with the current bars").

SOURCE (read, never written): reports/dashboard/bars_statistics.json, built by scripts/build_bars_statistics.py from
the two proposals on their branches (commit recorded per proposal). It holds no bookmaker price: statuses, bars,
the odds-range rule of a cell, bet counts, ROI, its 95% CI and mean CLV % only.

Nothing here has a default: a missing or unreadable file is a named problem and an empty table.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "reports" / "dashboard" / "bars_statistics.json"
SCHEMA = "wowza.bars_statistics/1"
STATUS_ORDER = ["PROVEN", "PROMISING", "UNPROVEN", "OFF"]
COLUMNS = ["League", "Track", "Market", "Status", "Bar", "Measure", "Odds range", "Mode", "Stake rule",
           "WF bets", "WF ROI %", "Study bets", "Study ROI %", "ROI 95% CI", "CLV %", "Why OFF"]


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
    """One display row per league x market cell of a proposal, PROVEN first, OFF last."""
    if not p or not p.get("cells"):
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for c in p["cells"]:
        off = c["status"] == "OFF"
        rows.append({
            "League": c["league"], "Track": c["track_label"], "Market": c["market"], "Status": c["status"],
            "Bar": None if off or c["bar"] is None else round(100 * c["bar"], 1),
            "Measure": None if off else c.get("measure"),
            "Odds range": None if off else odds_range(c.get("odds_min"), c.get("odds_max")),
            "Mode": None if off else c.get("mode"),
            "Stake rule": c["stake_rule"],
            "WF bets": c.get("wf_bets"), "WF ROI %": _pct(c.get("wf_roi")),
            "Study bets": c.get("study_bets"), "Study ROI %": _pct(c.get("study_roi")),
            "ROI 95% CI": fmt_ci(c.get("study_ci_lo"), c.get("study_ci_hi")),
            "CLV %": _pct(c.get("study_clv")),
            "Why OFF": c.get("off_reason") if off else None,
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


def changed(a: dict | None, b: dict | None) -> pd.DataFrame:
    """Cells whose status differs between two proposals (a cell present in one only is listed too)."""
    cols = ["League", "Track", "Market", "Approved (a)", "Clean re-run (b)"]
    if not a or not b:
        return pd.DataFrame(columns=cols)
    ka = {(c["track_label"], c["market"], c["league"]): c["status"] for c in a["cells"]}
    kb = {(c["track_label"], c["market"], c["league"]): c["status"] for c in b["cells"]}
    rows = []
    for k in sorted(set(ka) | set(kb), key=lambda k: (k[2], k[0], k[1])):
        sa, sb = ka.get(k), kb.get(k)
        if sa != sb:
            rows.append({"League": k[2], "Track": k[0], "Market": k[1], "Approved (a)": sa or "—",
                         "Clean re-run (b)": sb or "not re-measured"})
    return pd.DataFrame(rows, columns=cols)


def not_adopted_frame(na: dict | None) -> pd.DataFrame:
    """Cell counts per variant of a tested-and-not-adopted study (copied from its report)."""
    cols = ["Variant"] + STATUS_ORDER
    rows = [{"Variant": r["variant"], **{s: r.get(s) for s in STATUS_ORDER}} for r in (na or {}).get("counts") or []]
    return pd.DataFrame(rows, columns=cols)


def stake_plan_frame(d: dict | None) -> pd.DataFrame:
    """The go-live stake plan's real-stake cells, with their status in both proposals."""
    plan = (d or {}).get("stake_plan") or {}
    a, b = proposal(d, "approved"), proposal(d, "clean_rerun")

    def _st(p, c):
        for x in (p or {}).get("cells") or []:
            if (x["track"], x["mside"], x["league"]) == (c["track"], c["mside"], c["league"]):
                return x
        return None
    rows = []
    for c in plan.get("cells") or []:
        xa, xb = _st(a, c), _st(b, c)
        x = xb or xa or {}
        rows.append({"League": c["league"], "Track": x.get("track_label", c["track"]),
                     "Market": x.get("market", c["mside"]),
                     "Status (a)": (xa or {}).get("status", "—"), "Status (b)": (xb or {}).get("status", "—"),
                     "Stake": (xa or {}).get("stake_rule", "—")})
    return pd.DataFrame(rows, columns=["League", "Track", "Market", "Status (a)", "Status (b)", "Stake"])
