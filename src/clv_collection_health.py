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
from collections import Counter
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
#: Every floor check() reads from config/clv_collection_expectation.json.
_EXPECTATION_KEYS = ("min_shadow_sample", "min_certified_t30_pct", "min_collection_sample",
                     "min_collection_t30_pct", "collection_window_days")


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
        # A row with no market is counted under "" (unknown), not relabelled ou25.
        out["by_market"] = {str(k): _shadow_counts(g) for k, g in
                            d.assign(market=d["market"].fillna("").astype(str)).groupby("market")}
        # P1 item 7 (2026-11-22): an all-time share mixes the pre-capture-fix weeks (W40-W42, ~8%
        # certified) with the weeks after it; the kickoff-week timeline keeps the two apart.
        wk = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True).dt.strftime("%G-W%V")
        out["by_kickoff_week"] = {str(k): _shadow_counts(g) for k, g in
                                  d.assign(_wk=wk.fillna("")).groupby("_wk")}
        out["close_feed_identity"] = _feed_identity_counts(d)
    return out


def _feed_identity_counts(d: pd.DataFrame) -> dict:
    """Counts of certified closes by feed identity (shadow_settlement.close_feed_identity): from the
    entry's own OddsAPI feed vs a Bet365 capture. Counts only; the close rule is unchanged."""
    from src.shadow_settlement import close_feed_identity
    mins = pd.to_numeric(d["close_minutes_to_kickoff"], errors="coerce")
    ok = d["close_status"].astype(str).eq(CLOSE_OK) & mins.le(CLOSE_MAX_MIN)
    src = d["close_source"] if "close_source" in d.columns else pd.Series("", index=d.index)
    mt = d["model_type"] if "model_type" in d.columns else pd.Series("", index=d.index)
    lab = [close_feed_identity(m, s, CLOSE_OK if o else "")
           for m, s, o in zip(mt.fillna(""), src.fillna(""), ok)]
    return dict(sorted(Counter(lab).items()))


# ── collection (every fixture the captures saw) ──────────────────────────────────────────────
def _incident_mask(last: pd.DataFrame, incidents: list[dict] | None) -> pd.Series:
    """True for fixture-markets whose kickoff falls in an owner-acknowledged capture incident
    (config/clv_collection_expectation.json `acknowledged_incidents`): a window whose closes were
    provably never captured (e.g. GitHub skipped the scheduled capture firings) and can never be
    recovered because odds are pre-match only. Excluded from the PASS/FAIL figure only; the
    unfiltered figure is always reported beside it (`including_incidents`)."""
    m = pd.Series(False, index=last.index)
    for inc in incidents or []:
        lo = pd.Timestamp(inc["kickoff_from"])
        hi = pd.Timestamp(inc["kickoff_to"])
        lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo
        hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi
        hit = (last["_ko"] >= lo) & (last["_ko"] < hi)
        if inc.get("track"):
            hit &= last["track"].eq(inc["track"])
        m |= hit
    return m


def collection_section(archives: list[tuple[str, pd.DataFrame | None]], now: pd.Timestamp,
                       window_days: float = 7.0, since: pd.Timestamp | None = None,
                       incidents: list[dict] | None = None) -> dict:
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
        # NO DECIMAL PRICE CAN BE <= 1.0 (PR #72): a suspended book's 1.0 quote is no observation of
        # the close, so it cannot count as coverage either (the certifier already refuses it).
        if "odds" in d.columns:
            d = d[pd.to_numeric(d["odds"], errors="coerce").gt(1.0)]
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

    full = counts(a)
    mask = _incident_mask(a, incidents)
    excluded = []
    for inc in incidents or []:
        one = _incident_mask(a, [inc])
        if one.any():
            excluded.append({"id": inc.get("id"), "reason": inc.get("reason"), **counts(a[one])})
    if mask.any():
        a = a[~mask]
    if a.empty:
        return {"window_days": window_days, "from": str(lo), "fixture_markets": 0,
                "including_incidents": full, "excluded_incidents": excluded}
    out = {"window_days": window_days, "from": str(lo), **counts(a),
           "including_incidents": full, "excluded_incidents": excluded}
    out["by_track"] = {k: counts(g) for k, g in a.groupby("track")}
    out["by_market"] = {k: counts(g) for k, g in a.groupby("market")}
    out["by_league"] = {k: counts(g) for k, g in a.groupby("league")}
    out["ladder"] = ladder_counts(a)
    out["ladder_by_league_market"] = {
        str(lg): {str(mk): ladder_counts(g2) for mk, g2 in g.groupby("market")}
        for lg, g in a.groupby("league")}
    return out


#: P1 item 7 (2026-11-22): the kickoff-relative ladder the owner asked for. A fixture-market is
#: covered at T-x when its LAST strictly pre-kickoff observation (change-log + heartbeat, odds > 1.0)
#: lies within x minutes of kickoff. Counts only - never a price.
LADDER = (("T-6h", 360.0), ("T-3h", 180.0), ("T-1h", 60.0), ("T-30m", 30.0), ("T-10m", 10.0))


def ladder_counts(g: pd.DataFrame) -> dict:
    m = pd.to_numeric(g["minutes"], errors="coerce")
    return {"fixture_markets": int(len(g)), **{k: int(m.le(v).sum()) for k, v in LADDER}}


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
        # No window configured -> the collection view is not measured (it defaulted to 7 days).
        # Go-live merge: main's acknowledged capture incidents (2026-11-14 15:00 UTC std) are kept.
        "collection": (collection_section(archives, now_ts, float(exp["collection_window_days"]), since,
                                          exp.get("acknowledged_incidents") or [])
                       if exp.get("collection_window_days") is not None else None),
    }
    h["check"] = check(h, exp)
    return h


def check(h: dict, exp: dict) -> dict:
    """PASS / FAIL / INSUFFICIENT_SAMPLE against the expectation. Counts below `min_sample` never
    fail: a quiet week must not look like a broken capture."""
    res = {"status": "PASS", "reasons": []}
    if not exp:
        return {"status": "NO_EXPECTATION", "reasons": ["expectation file missing"]}
    # The floors come from the expectation file only; a missing key is reported, never replaced by
    # a number typed here (these defaulted to 40 / 200 in code).
    missing = [k for k in _EXPECTATION_KEYS if k not in exp]
    if missing:
        return {"status": "NO_EXPECTATION", "reasons": [f"expectation file lacks {missing}"]}
    sh = h.get("shadow_since_measure_from")
    co = h.get("collection")
    # An unmeasured section is a reason (never a 0% FAIL), and the OTHER section is still judged: go-live merge
    # 2026-11-17 - returning early here hid a measured collection FAIL whenever the shadow section was not
    # measured (main's capture-incident guard, tests/test_clv_capture).
    n_sh = None if sh is None else int(sh["eligible"])
    if sh is None:
        res["reasons"].append("shadow section not measured")
    elif n_sh < int(exp["min_shadow_sample"]):
        res["reasons"].append(f"shadow sample {n_sh} < {exp['min_shadow_sample']}")
    elif sh.get("certified_t30_pct") is None:
        res["reasons"].append("certified T-30m close not measured")
    elif sh["certified_t30_pct"] < float(exp["min_certified_t30_pct"]):
        res["status"] = "FAIL"
        res["reasons"].append(f"certified T-30m close {sh.get('certified_t30_pct')}% < "
                              f"{exp['min_certified_t30_pct']}% on {n_sh} settled shadow rows")
    n_co = None if co is None else int(co["fixture_markets"])
    if co is None:
        res["reasons"].append("collection section not measured")
    elif n_co < int(exp["min_collection_sample"]):
        res["reasons"].append(f"collection sample {n_co} < {exp['min_collection_sample']}")
    elif co.get("t30_pct") is None:
        res["reasons"].append("collection T-30m not measured")
    elif co["t30_pct"] < float(exp["min_collection_t30_pct"]):
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
