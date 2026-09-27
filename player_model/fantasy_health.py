"""Machine-readable Fantasy health: what the product knows, and how old it is.

    python -m player_model.fantasy_health          # writes output/fantasy_health.json

WHY THIS EXISTS. Fantasy has already served a two-month-old FPL snapshot as if it were live,
offering an injured Doku as a captaincy pick, and nothing anywhere said so. A polished product
has to know when its own data is incomplete, and has to be able to say so on the page rather
than in a log nobody reads.

AGE NEVER COMES FROM FILE MTIME. `git checkout` resets mtime on every CI run, so anything
mtime-based reads as ~0 hours old and never refreshes -- a trap this estate has fallen into
three separate times, including in this very module's neighbour `fpl_api._cached_fetch` before
it was given a sidecar. Ages here come from `output/fpl_cache_meta.json`, the recorded
fetch-time sidecar, and an age that cannot be established is reported as UNKNOWN and treated as
infinitely old rather than as fresh.

STATUS IS A JUDGEMENT, SO IT IS SPELLED OUT. CURRENT / AGING / STALE / UNAVAILABLE per feed,
with the thresholds in the payload, so a reader can disagree with the thresholds without having
to guess what they were.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from player_model import config

OUT = config.OUTPUT_DIR
HEALTH = OUT / "fantasy_health.json"

# Hours after which a feed stops being CURRENT / starts being STALE.
THRESHOLDS = {
    "fpl_bootstrap": {"aging": 12, "stale": 36},
    "fpl_fixtures": {"aging": 12, "stale": 36},
    "projection": {"aging": 30, "stale": 72},
    "official_squads": {"aging": 72, "stale": 240},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _feed_age_hours(name: str) -> float | None:
    """Age from the RECORDED fetch time, never from mtime. None when unknowable."""
    meta_p = OUT / "fpl_cache_meta.json"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    ts = meta.get(name)
    if not ts:
        return None
    try:
        return (_now().timestamp() - float(ts)) / 3600.0
    except Exception:
        return None


def _state(age_h: float | None, key: str) -> str:
    """UNKNOWN age is treated as infinitely old, never as fresh."""
    if age_h is None:
        return "UNAVAILABLE"
    t = THRESHOLDS[key]
    if age_h <= t["aging"]:
        return "CURRENT"
    return "AGING" if age_h <= t["stale"] else "STALE"


def build() -> dict:
    h: dict = {
        "generated_at": _now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thresholds_hours": THRESHOLDS,
        "feeds": {},
        "coverage": {},
        "projection_log": {},
        "notes": [],
    }

    # ── gameweek context ─────────────────────────────────────────────────────
    try:
        from player_model.fpl_api import gameweek_context
        ctx = gameweek_context()
        h["current_gameweek"] = ctx.get("gw")
        h["deadline_utc"] = ctx.get("deadline_utc")
        h["live_gameweek"] = ctx.get("current_gw")
        h["last_finished_gameweek"] = ctx.get("last_finished_gw")
        h["season_complete"] = ctx.get("season_complete")
        if ctx.get("deadline_utc"):
            try:
                dl = pd.Timestamp(ctx["deadline_utc"]).tz_convert("UTC")
                h["hours_to_deadline"] = round((dl - pd.Timestamp(_now())).total_seconds() / 3600, 1)
            except Exception:
                h["hours_to_deadline"] = None
    except Exception as e:                                           # noqa: BLE001
        h["notes"].append(f"gameweek context unavailable: {type(e).__name__}")

    # ── feed ages ────────────────────────────────────────────────────────────
    for key, fname in (("fpl_bootstrap", "fpl_bootstrap.json"),
                       ("fpl_fixtures", "fpl_fixtures.json")):
        age = _feed_age_hours(fname)
        h["feeds"][key] = {
            "file": fname,
            "exists": (OUT / fname).exists(),
            "age_hours": round(age, 2) if age is not None else None,
            "state": _state(age, key),
        }
    for key, fname in (("official_squads", "pl_squads_official.csv"),):
        p = OUT / fname
        h["feeds"][key] = {"file": fname, "exists": p.exists(),
                           "age_hours": None, "state": "UNKNOWN_AGE" if p.exists() else "UNAVAILABLE"}

    # ── projections ──────────────────────────────────────────────────────────
    tips_p = OUT / "fantasy_tips.csv"
    if tips_p.exists():
        try:
            t = pd.read_csv(tips_p)
            avail = t["availability"].value_counts().to_dict() if "availability" in t.columns else {}
            unavailable = sum(v for k, v in avail.items()
                              if k in ("injured", "unavailable", "suspended"))
            h["coverage"] = {
                "projection_players": int(len(t)),
                "clubs": int(t["team"].nunique()) if "team" in t.columns else None,
                "available": int(avail.get("available", 0)),
                "doubtful": int(avail.get("doubtful", 0)),
                "unavailable": int(unavailable),
                "availability_breakdown": {k: int(v) for k, v in avail.items()},
                "p_start_zero": int((t["p_start"] == 0).sum()) if "p_start" in t.columns else None,
                "fpl_match_rate": (round(float(t["fpl_matched"].mean()), 4)
                                   if "fpl_matched" in t.columns else None),
                "fixture_coverage": (round(float(t["fixtures_available"].mean()), 4)
                                     if "fixtures_available" in t.columns else None),
            }
        except Exception as e:                                       # noqa: BLE001
            h["notes"].append(f"fantasy_tips unreadable: {type(e).__name__}")
    else:
        h["notes"].append("fantasy_tips.csv absent")

    # ── the projection log, and whether it can be settled at all ─────────────
    log_p = OUT / "fantasy_projection_log.csv"
    if log_p.exists():
        try:
            lg = pd.read_csv(log_p)
            tagged = int(lg["gw"].notna().sum()) if "gw" in lg.columns else 0
            days = (pd.to_datetime(lg["snapshot_date"], errors="coerce").dt.date.nunique()
                    if "snapshot_date" in lg.columns else None)
            h["projection_log"] = {
                "rows": int(len(lg)),
                "distinct_days": int(days) if days is not None else None,
                "rows_with_gameweek": tagged,
                "settleable": tagged > 0,
                "gameweeks_logged": sorted(int(g) for g in lg["gw"].dropna().unique())
                                    if "gw" in lg.columns else [],
            }
            if tagged == 0 and len(lg):
                h["notes"].append(
                    "projection log has rows but NONE carry a gameweek — forecasts cannot be "
                    "settled against actual results until new rows are written with gw stamped")
        except Exception as e:                                       # noqa: BLE001
            h["notes"].append(f"projection log unreadable: {type(e).__name__}")
    else:
        h["notes"].append("fantasy_projection_log.csv absent")

    # ── overall ──────────────────────────────────────────────────────────────
    states = [v.get("state") for v in h["feeds"].values()]
    if "STALE" in states or "UNAVAILABLE" in states:
        h["status"] = "DEGRADED"
    elif "AGING" in states or "UNKNOWN_AGE" in states:
        h["status"] = "AGING"
    else:
        h["status"] = "OK"
    h["settlement_ready"] = bool(h.get("projection_log", {}).get("settleable"))
    return h


def main() -> int:
    h = build()
    OUT.mkdir(parents=True, exist_ok=True)
    HEALTH.write_text(json.dumps(h, indent=2), encoding="utf-8")
    print(f"[fantasy_health] status={h['status']}  gw={h.get('current_gameweek')}  "
          f"settlement_ready={h['settlement_ready']}")
    for k, v in h["feeds"].items():
        print(f"   {k:<16} {v['state']:<12} age="
              f"{v['age_hours'] if v['age_hours'] is not None else '?'}")
    for n in h["notes"]:
        print(f"   NOTE: {n}")
    print(f"[fantasy_health] wrote {HEALTH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
