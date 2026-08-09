"""
API-Football usage monitor (measure-before-trim).
=================================================
Polls API-Football's /status endpoint — which is FREE (does NOT count against the daily
request limit) — and appends the cumulative daily usage to output/api_usage_log.csv. Run on
a 30-min schedule so we get a continuous usage CURVE across the day; correlate the spikes
against the known workflow cron times to see which jobs eat the quota, then right-size the
cadence for the 7,500/day Pro cap (post-2026-08-10).

ISOLATION: read-only measurement. Writes ONLY output/api_usage_log.csv. Touches no model.
"""
import os, sys
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output" / "api_usage_log.csv"
COLS = ["snapshot_ts", "date", "requests_used", "limit_day", "plan", "label"]


def poll(label: str = "poll") -> dict:
    key = os.getenv("APIFOOTBALL_KEY", "")
    if not key:
        print("[af_usage] APIFOOTBALL_KEY not set - skipping")
        return {}
    try:
        r = requests.get("https://v3.football.api-sports.io/status",
                         headers={"x-apisports-key": key}, timeout=20)
        data = r.json().get("response", {})
    except Exception as e:
        print(f"[af_usage] status fetch failed: {e}")
        return {}
    # /status returns response={} normally, but an ERROR/rate-limited call returns
    # response=[] (a list) -> .get() would crash. Treat any non-dict as empty.
    if not isinstance(data, dict):
        print("[af_usage] status response not a dict (likely rate-limited/error) - skipping")
        return {}
    reqs = data.get("requests", {}) or {}
    sub = data.get("subscription", {}) or {}
    now = datetime.now(timezone.utc)
    row = {
        "snapshot_ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": now.strftime("%Y-%m-%d"),
        "requests_used": reqs.get("current", ""),
        "limit_day": reqs.get("limit_day", ""),
        "plan": sub.get("plan", ""),
        "label": label,
    }
    new = pd.DataFrame([row], columns=COLS)
    if OUT.exists():
        try:
            new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
        except Exception:
            pass
    new.to_csv(OUT, index=False)
    print(f"[af_usage] {row['requests_used']}/{row['limit_day']} used "
          f"(plan={row['plan']}) @ {row['snapshot_ts']}")
    return row


if __name__ == "__main__":
    poll(sys.argv[1] if len(sys.argv) > 1 else "poll")
