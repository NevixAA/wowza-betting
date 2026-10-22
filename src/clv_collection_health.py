"""CLV collection health — is the near-kickoff close actually being captured? (P1 2026-10-22)

    from src.clv_collection_health import build, check, load_expectation

Writes output/clv_collection_health.json (from update_results' shadow pass) and backs the guard
scripts/check_clv_collection_health.py, which FAILS the workflow when certified coverage drops
below config/clv_collection_expectation.json.

Two views, because they answer different questions:

* `shadow` — the CLV that matters: settled v91_shadow rows, how many carry a CERTIFIED close
  (observed strictly before kickoff and within CLOSE_MAX_MIN = 30 minutes), within 10 minutes,
  stale, or none; by league and market; median minutes from the selected close to kickoff.
* `collection` — the capture itself, independent of what the model tipped: every fixture-market in
  the kickoff-timed archives (change-log + last-seen heartbeat) that kicked off in the last
  `window_days`, and the share with an observation inside T-30m / T-10m. This reacts in a day;
  the shadow view needs weeks of settled rows.

`generated_at` is a recorded timestamp. Nothing here reads a file mtime (CLAUDE.md: git checkout
resets mtime on every CI run).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.shadow_settlement import CLOSE_MAX_MIN, CLOSE_OK, STALE_CLOSE

HEALTH_FILE = config.OUTPUT_DIR / "clv_collection_health.json"
EXPECTATION_FILE = Path(config.BASE_DIR) / "config" / "clv_collection_expectation.json"
CLV_MARKETS = ("over25", "under25", "over15", "under15", "over35", "under35",
               "btts_yes", "btts_no")
T10 = 10.0


def _now(now) -> pd.Timestamp:
    t = pd.Timestamp(now or datetime.now(timezone.utc))
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def _track(name: str) -> str:
    n = name.lower()
    if "newformat" in n or "nf_" in n:
        return "new_format"
    if "standard" in n or "std_" in n:
        return "standard"
    return "other"


# ── shadow (certified CLV on the rows the model logged) ──────────────────────────────────────
def _shadow_counts(g: pd.DataFrame) -> dict:
    st = g["close_status"].astype(str)
    mins = pd.to_numeric(g["close_minutes_to_kickoff"], errors="coerce")
    ok = st.eq(CLOSE_OK) & mins.le(CLOSE_MAX_MIN)
    n = int(len(g))
    return {
        "eligible": n,
        "with_t30_close": int(ok.sum()),
        "with_t10_close": int((ok & mins.le(T10)).sum()),
        "stale_close": int(st.eq(STALE_CLOSE).sum()),
        "no_close": int((~st.isin([CLOSE_OK, STALE_CLOSE])).sum()),
        "certified_t30_pct": _pct(int(ok.sum()), n),
        "certified_t10_pct": _pct(int((ok & mins.le(T10)).sum()), n),
        "median_close_minutes_certified": (round(float(mins[ok].median()), 1)
                                           if ok.any() else None),
        "median_last_pre_kickoff_minutes": (round(float(mins.median()), 1)
                                            if mins.notna().any() else None),
    }


def shadow_section(df: pd.DataFrame, since: pd.Timestamp | None = None) -> dict:
    if df is None or df.empty or "result" not in df.columns:
        return {"eligible": 0}
    d = df.copy()
    for c in ("close_status", "close_minutes_to_kickoff", "market", "league"):
        if c not in d.columns:
            d[c] = ""
    d = d[d["result"].fillna("").astype(str).str.upper().isin(["WIN", "LOSS"])]
    if since is not None:
        ko = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True)
        d = d[ko >= since]
    out = _shadow_counts(d)
    if len(d):
        out["by_league"] = {str(k): _shadow_counts(g) for k, g in d.groupby("league")}
        out["by_market"] = {str(k): _shadow_counts(g) for k, g in
                            d.assign(market=d["market"].replace("", "ou25")).groupby("market")}
    return out


# ── collection (every fixture the captures saw) ──────────────────────────────────────────────
def collection_section(archives: list[tuple[str, pd.DataFrame | None]], now: pd.Timestamp,
                       window_days: float = 7.0, since: pd.Timestamp | None = None) -> dict:
    lo = now - pd.Timedelta(days=window_days)
    if since is not None and since > lo:
        lo = since                    # never judge the new capture on kickoffs from before it
    frames = []
    for name, d in archives:
        if d is None or d.empty or not {"kickoff_utc", "snapshot_ts", "match", "market",
                                        "league"} <= set(d.columns):
            continue
        d = d[d["market"].isin(CLV_MARKETS)]
        ko = pd.to_datetime(d["kickoff_utc"].astype(str), errors="coerce", utc=True)
        d = d[ko.notna() & (ko >= lo) & (ko <= now)].copy()
        if d.empty:
            continue
        d["_ko"] = ko[d.index]
        d["_ts"] = pd.to_datetime(d["snapshot_ts"].astype(str), errors="coerce", utc=True)
        d = d[d["_ts"].notna() & (d["_ts"] < d["_ko"])]          # strictly pre-kickoff only
        if d.empty:
            continue
        last = d.groupby(["league", "match", "market"], as_index=False).agg(
            _ko=("_ko", "first"), _ts=("_ts", "max"))
        last["minutes"] = (last["_ko"] - last["_ts"]).dt.total_seconds() / 60.0
        last["track"] = _track(name)
        last["archive"] = name
        frames.append(last)
    if not frames:
        return {"window_days": window_days, "from": str(lo), "fixture_markets": 0}
    a = pd.concat(frames, ignore_index=True)

    def counts(g):
        n = int(len(g))
        return {"fixture_markets": n,
                "fixtures": int(g[["league", "match"]].drop_duplicates().shape[0]),
                "t30_pct": _pct(int(g["minutes"].le(CLOSE_MAX_MIN).sum()), n),
                "t10_pct": _pct(int(g["minutes"].le(T10).sum()), n),
                "t60_pct": _pct(int(g["minutes"].le(60).sum()), n),
                "median_last_obs_minutes": round(float(g["minutes"].median()), 1)}

    out = {"window_days": window_days, "from": str(lo), **counts(a)}
    out["by_track"] = {k: counts(g) for k, g in a.groupby("track")}
    out["by_market"] = {k: counts(g) for k, g in a.groupby("market")}
    out["by_league"] = {k: counts(g) for k, g in a.groupby("league")}
    return out


# ── artifact + guard ─────────────────────────────────────────────────────────────────────────
def load_expectation(path: Path | None = None) -> dict:
    p = Path(path or EXPECTATION_FILE)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def build(shadow_df: pd.DataFrame, archives: list, now=None,
          expectation: dict | None = None) -> dict:
    now_ts = _now(now)
    exp = expectation if expectation is not None else load_expectation()
    since = pd.Timestamp(exp["measure_from"]) if exp.get("measure_from") else None
    if since is not None and since.tzinfo is None:
        since = since.tz_localize("UTC")
    h = {
        "generated_at": now_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "close_rule": f"strictly pre-kickoff, observed within {int(CLOSE_MAX_MIN)} minutes",
        "shadow_all_time": shadow_section(shadow_df),
        "shadow_since_measure_from": (shadow_section(shadow_df, since)
                                      if since is not None else None),
        "measure_from": exp.get("measure_from"),
        "collection": collection_section(archives, now_ts,
                                         float(exp.get("collection_window_days", 7)), since),
    }
    h["check"] = check(h, exp)
    return h


def check(h: dict, exp: dict) -> dict:
    """PASS / FAIL / INSUFFICIENT_SAMPLE against the expectation. Counts below `min_sample` never
    fail: a quiet week must not look like a broken capture."""
    res = {"status": "PASS", "reasons": []}
    if not exp:
        return {"status": "NO_EXPECTATION", "reasons": ["expectation file missing"]}
    sh = h.get("shadow_since_measure_from") or {}
    n_sh = int(sh.get("eligible") or 0)
    if n_sh < int(exp.get("min_shadow_sample", 40)):
        res["reasons"].append(f"shadow sample {n_sh} < {exp.get('min_shadow_sample', 40)}")
    elif (sh.get("certified_t30_pct") or 0.0) < float(exp["min_certified_t30_pct"]):
        res["status"] = "FAIL"
        res["reasons"].append(f"certified T-30m close {sh.get('certified_t30_pct')}% < "
                              f"{exp['min_certified_t30_pct']}% on {n_sh} settled shadow rows")
    co = h.get("collection") or {}
    n_co = int(co.get("fixture_markets") or 0)
    if n_co < int(exp.get("min_collection_sample", 200)):
        res["reasons"].append(f"collection sample {n_co} < {exp.get('min_collection_sample', 200)}")
    elif (co.get("t30_pct") or 0.0) < float(exp["min_collection_t30_pct"]):
        res["status"] = "FAIL"
        res["reasons"].append(f"collection T-30m {co.get('t30_pct')}% < "
                              f"{exp['min_collection_t30_pct']}% on {n_co} fixture-markets")
    if res["status"] == "PASS" and res["reasons"]:
        res["status"] = "INSUFFICIENT_SAMPLE"
    return res


def write(h: dict, path: Path | None = None) -> Path:
    p = Path(path or HEALTH_FILE)
    p.write_text(json.dumps(h, indent=2, default=str) + "\n", encoding="utf-8")
    return p
