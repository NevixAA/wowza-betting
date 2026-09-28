"""Pro adapter — canonical evidence, validation, research. Read-only, over a sibling checkout."""
from __future__ import annotations

import pandas as pd

from dashboard_data.core import (Freshness, freshness_from_timestamp, newest_date,
                                 read_csv, read_json, repo_path)


def available() -> bool:
    return repo_path("pro") is not None


def _out():
    p = repo_path("pro")
    return (p / "output") if p else None


def _reg():
    p = repo_path("pro")
    return (p / "registry") if p else None


def absent_reason() -> str:
    return ("Pro is not checked out beside v9. Expected at ../wowzaV9-Pro or ../v10. "
            "The dashboard shows what it can reach rather than failing.")


def system_contract() -> dict:
    """The governance contract, if Pro is present. Its SHAs say what it was verified against."""
    r = _reg()
    return read_json(r / "WOWZA_SYSTEM_CONTRACT.json") if r else {}


def _season_dirs():
    """Season stores, newest season first. The season is in the DIRECTORY NAME, not a mtime."""
    p = repo_path("pro")
    if p is None:
        return []
    d = p / "data"
    if not d.exists():
        return []
    return sorted((x for x in d.iterdir() if x.is_dir() and x.name.startswith("season_")),
                  key=lambda x: x.name, reverse=True)


def canonical_stores() -> dict:
    """The canonical season store — Pro's evidence layer.

    Each store is a DATE-PARTITIONED DIRECTORY (`fixtures/dt=2026-09-27/...`), not a single
    parquet. That is deliberate: it is how the estate stays under GitHub's 100 MB per-file wall
    without splitting a table by hand. It also means the newest partition NAME is the data's own
    date, which is a content-derived age -- exactly what is wanted, and what a file mtime is not.
    """
    seasons = _season_dirs()
    if not seasons:
        return {"available": False,
                "why": absent_reason() if repo_path("pro") is None
                       else "Pro is present but has no data/season_* store"}
    latest = seasons[0]
    stores: dict[str, dict] = {}
    for store in sorted(x for x in latest.iterdir() if x.is_dir()):
        parts = [d.name[3:] for d in store.iterdir() if d.is_dir() and d.name.startswith("dt=")]
        if not parts:
            files = list(store.glob("*.parquet"))
            stores[store.name] = {"partitioned": False, "files": len(files)}
            continue
        parts.sort()
        stores[store.name] = {
            "partitioned": True, "partitions": len(parts),
            "first_date": parts[0], "last_date": parts[-1],
            "freshness": freshness_from_timestamp(parts[-1] + "T23:59:59Z", "daily",
                                                  f"{store.name}/dt=*")}
    return {"available": True, "season": latest.name,
            "seasons_on_disk": [s.name for s in seasons],
            "stores": stores}


def health_artifacts() -> dict:
    """Pro's own health files — the canary among them."""
    o = _out()
    if o is None:
        return {"available": False, "why": absent_reason()}
    res = {"available": True, "files": {}}
    for f in ("collect_health.json", "predict_health.json", "scheduler_health.json",
              "ml_learning_health.json", "combo_import_health.json"):
        j = read_json(o / f)
        if j:
            ts = j.get("generated_at") or j.get("asof") or j.get("timestamp")
            res["files"][f] = {"status": j.get("status") or j.get("state"),
                               "freshness": freshness_from_timestamp(ts, "daily", f)}
    return res


def shadow_learning() -> dict:
    """The walk-forward evidence: does retraining beat a frozen model?"""
    o = _out()
    if o is None:
        return {"available": False, "why": absent_reason()}
    f = o / "shadow_learning" / "player_walkforward_performance.csv"
    d = read_csv(f)
    if d.empty:
        return {"available": False, "why": "no player walk-forward results in Pro"}
    g = d.groupby("variant")[[c for c in ("log_loss", "auc", "pr_auc") if c in d.columns]].mean()
    return {"available": True, "months": int(d["month"].nunique()) if "month" in d else None,
            "variants": g.round(5).to_dict("index"),
            "best_by_logloss": g["log_loss"].idxmin() if "log_loss" in g else None}


def research_validity() -> list[dict]:
    """Which conclusions are CURRENT and which are INVALIDATED — so dead numbers stop circulating."""
    c = system_contract()
    return c.get("research_validity_registry", []) or []
