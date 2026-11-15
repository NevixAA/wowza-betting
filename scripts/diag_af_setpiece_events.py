"""Diagnostic (read-only, ~25 API-Football calls): do API-Football fixture events carry any set-piece
information? Prints every distinct (type, detail, comments) on goal events for recent finished
fixtures in a few of our leagues, so the owner can see whether corners / free kicks are recorded.
Writes nothing except reports/DIAG_AF_SETPIECE_EVENTS.md.

    APIFOOTBALL_KEY=... python scripts/diag_af_setpiece_events.py
"""
from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

import requests

BASE = "https://v3.football.api-sports.io"
LEAGUES = {"Premier League": 39, "USA MLS": 253, "Sweden Allsvenskan": 113, "Japan J-League": 98, "Championship": 40}
SEASONS = {39: 2025, 253: 2026, 113: 2026, 98: 2026, 40: 2025}
PER_LEAGUE = 4


def get(path: str, params: dict) -> dict:
    r = requests.get(BASE + path, params=params, headers={"x-apisports-key": os.environ["APIFOOTBALL_KEY"]}, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> int:
    combos = collections.Counter()
    examples, lines = {}, []
    for name, lid in LEAGUES.items():
        fx = get("/fixtures", {"league": lid, "season": SEASONS[lid], "status": "FT", "last": PER_LEAGUE})["response"]
        for f in fx:
            fid = f["fixture"]["id"]
            ev = get("/fixtures/events", {"fixture": fid})["response"]
            for e in ev:
                key = (e.get("type"), e.get("detail"), e.get("comments"))
                combos[key] += 1
                examples.setdefault(key, f"{name} fixture {fid}, {e.get('time', {}).get('elapsed')}'")
        lines.append(f"- {name}: {len(fx)} fixtures checked")
    out = ["# Do API-Football events record set pieces?", "", *lines, "",
           "| type | detail | comments | count | example |", "|---|---|---|---|---|"]
    for (t, d, c), n in sorted(combos.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        out.append(f"| {t} | {d} | {c} | {n} | {examples[(t, d, c)]} |")
    text = "\n".join(out) + "\n"
    Path("reports").mkdir(exist_ok=True)
    Path("reports/DIAG_AF_SETPIECE_EVENTS.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
