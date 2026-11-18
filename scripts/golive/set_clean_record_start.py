"""Set the Clean Record start to the go-live merge instant (owner 2026-11-18: "count them").

    python scripts/golive/set_clean_record_start.py --at 2026-11-19T06:30:00Z
    python scripts/golive/set_clean_record_start.py --at now          # the current UTC second

Writes the ONE value in its two places, consistently, and nothing else:
  * config/trustworthy_performance.json  clean_record_start_utc (+ its _owner_confirmed note)
  * dashboard_data/season.py             OWNER_DECISION (the date part; tests/test_dashboard_records.py pins
                                         the two together)
config.PROPS_CLEAN_MODELS_LIVE_DATE is NOT moved (owner 2026-11-18: props stay on 2026-11-21).
Only the two lines change (the files keep their formatting). Refuses a value that is not a UTC instant.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "config" / "trustworthy_performance.json"
SEASON = ROOT / "dashboard_data" / "season.py"


def _instant(v: str) -> str:
    if v == "now":
        t = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    else:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", v):
            raise SystemExit(f"--at must be YYYY-MM-DDTHH:MM:SSZ (UTC) or 'now', got {v!r}")
        t = dt.datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sub_line(text: str, pattern: str, new: str, where: str) -> str:
    out, n = re.subn(pattern, lambda m: new, text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"{where}: expected exactly one line matching {pattern!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--at", required=True)
    a = ap.parse_args(argv)
    at = _instant(a.at)
    note = (f"OWNER DECISION 2026-11-18 (\\\"count them\\\"): the Clean Record starts at the go-live merge instant "
            f"{at}, when the clean refit models start serving (supersedes O4 \\\"21.11 00:00 UTC\\\" of 2026-11-17). "
            f"Old-season history is kept unchanged as the pre-repair record.")
    raw = CFG.read_text(encoding="utf-8")
    raw = re.sub(r'^(\s*)"clean_record_start_utc": "[^"]*",$', lambda m: f'{m.group(1)}"clean_record_start_utc": "{at}",',
                 raw, count=1, flags=re.M)
    raw = re.sub(r'^(\s*)"_owner_confirmed": ".*",$', lambda m: f'{m.group(1)}"_owner_confirmed": "{note}",',
                 raw, count=1, flags=re.M)
    j = json.loads(raw)
    if j.get("clean_record_start_utc") != at or "2026-11-18" not in j.get("_owner_confirmed", ""):
        raise SystemExit(f"{CFG.name}: the two lines were not rewritten")
    s = SEASON.read_text(encoding="utf-8")
    s = _sub_line(s, r'^OWNER_DECISION = "[^"]*"$', f'OWNER_DECISION = "{at[:10]}"', SEASON.name)
    CFG.write_bytes(raw.encode("utf-8"))
    SEASON.write_bytes(s.encode("utf-8"))
    print(json.dumps({"clean_record_start_utc": at, "OWNER_DECISION": at[:10]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
