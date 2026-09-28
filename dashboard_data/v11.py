"""V11 adapter — market lab. SHADOW / RESEARCH ONLY, and the adapter says so in the payload."""
from __future__ import annotations

import pandas as pd

from dashboard_data.core import (freshness_from_timestamp, newest_date, read_csv,
                                 read_json, repo_path)

#: Never let a v11 number reach the page without this attached.
STATE = "SHADOW / RESEARCH ONLY"


def available() -> bool:
    return repo_path("v11") is not None


def _out():
    p = repo_path("v11")
    return (p / "output") if p else None


def absent_reason() -> str:
    return ("V11 is not checked out beside v9. Expected at ../wowza-v11. "
            "Market-lab panels are hidden rather than shown empty.")


def shadow_log() -> pd.DataFrame:
    o = _out()
    return read_csv(o / "v11_shadow_log.csv") if o else pd.DataFrame()


def scoreboard() -> pd.DataFrame:
    o = _out()
    return read_csv(o / "v11_scoreboard.csv") if o else pd.DataFrame()


def residual() -> dict:
    """The central v11 question: does Wowza add information once the market price is known?

    Reported as MARKET vs MARKET+WOWZA on Brier and log loss. Standalone AUC is deliberately
    not the headline here -- it is not what v11 is testing.
    """
    o = _out()
    if o is None:
        return {"available": False, "why": absent_reason()}
    d = read_csv(o / "v11_residual.csv")
    if d.empty:
        return {"available": False, "why": "no residual results yet in v11/output"}
    keep = [c for c in d.columns if any(k in c.lower() for k in
                                        ("brier", "logloss", "log_loss", "n", "target", "model"))]
    return {"available": True, "state": STATE, "rows": int(len(d)),
            "table": d[keep].to_dict("records")[:40] if keep else d.head(40).to_dict("records")}


def movement() -> dict:
    """Price movement, plus the coverage that says whether the instrument measured anything."""
    o = _out()
    if o is None:
        return {"available": False, "why": absent_reason()}
    det = read_csv(o / "v11_market_movement_detail.csv")
    health = read_json(o / "v11_research_health.json")
    if det.empty and not health:
        return {"available": False, "why": "no movement detail or research health in v11/output"}
    return {
        "available": True, "state": STATE,
        "rows": int(len(det)),
        "fixtures": int(det["fixture_id"].nunique()) if "fixture_id" in det.columns else None,
        "freshness": freshness_from_timestamp(
            newest_date(det, "close_ts", "entry_ts", "kickoff_ts"), "fast",
            "v11_market_movement_detail.csv"),
        "research_health": health,
    }


def momentum() -> dict:
    """Post-fix momentum only.

    The pre-fix runs are INVALIDATED -- a merge_asof index reset attached every movement value
    to the wrong row. Those numbers were quoted for weeks. The fix landed in v11 ba21c4c on
    2026-09-10 and the artifacts were regenerated on 09-22, so what is on disk now is post-fix;
    the adapter carries that provenance so a reader never has to guess which era a number is from.
    """
    o = _out()
    if o is None:
        return {"available": False, "why": absent_reason()}
    d = read_csv(o / "v11_momentum_control.csv")
    if d.empty:
        return {"available": False, "why": "no momentum control results"}
    sig = d[d.get("excludes_zero") == True] if "excludes_zero" in d.columns else d  # noqa: E712
    return {"available": True, "state": STATE,
            "rows": int(len(d)),
            "terms_excluding_zero": int(len(sig)),
            "calc_version": (d["calc_version"].iloc[0] if "calc_version" in d.columns else None),
            "sample_status": (d["sample_status"].iloc[0] if "sample_status" in d.columns else None),
            "provenance": "post-fix (merge_asof index bug repaired 2026-09-10, regenerated 09-22)",
            "table": d.head(30).to_dict("records")}


def health() -> dict:
    o = _out()
    if o is None:
        return {"available": False, "why": absent_reason()}
    sl = shadow_log()
    return {"available": True, "state": STATE,
            "shadow_rows": int(len(sl)),
            "freshness": freshness_from_timestamp(
                newest_date(sl, "snapshot_ts", "entry_ts", "date", "kickoff_utc", "kickoff_ts"), "fast",
                "v11_shadow_log.csv"),
            "research_health": read_json(o / "v11_research_health.json")}
