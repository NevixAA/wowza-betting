"""Fail LOUDLY when certified CLV close coverage falls below the configured expectation.

    python scripts/check_clv_collection_health.py [--health PATH] [--expectation PATH]

Reads output/clv_collection_health.json (written by update_results' shadow pass) and re-applies
config/clv_collection_expectation.json, so editing the expectation takes effect without waiting
for a settlement. Exit 1 on FAIL. INSUFFICIENT_SAMPLE passes: a quiet week is not a broken
capture. A health file older than `max_health_age_hours` (by its recorded generated_at, never by
mtime) also fails — a guard reading a frozen file would pass forever.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import clv_collection_health as cch  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health", default=str(cch.HEALTH_FILE))
    ap.add_argument("--expectation", default=str(cch.EXPECTATION_FILE))
    ap.add_argument("--now", default=None, help="ISO timestamp (tests)")
    a = ap.parse_args(argv)

    exp = cch.load_expectation(Path(a.expectation))
    try:
        h = json.loads(Path(a.health).read_text(encoding="utf-8"))
    except Exception as e:                                            # noqa: BLE001
        print(f"::error::CLV collection health unreadable ({e})")
        return 1
    now = datetime.fromisoformat(a.now) if a.now else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    try:
        gen = datetime.fromisoformat(str(h["generated_at"]).replace("Z", "+00:00"))
        age_h = (now - gen).total_seconds() / 3600.0
    except Exception:                                                 # noqa: BLE001
        age_h = float("inf")                  # unknown age is infinitely old
    max_age = float(exp.get("max_health_age_hours", 30))
    if age_h > max_age:
        print(f"::error::CLV collection health is {age_h:.1f}h old (> {max_age}h) - "
              f"the settlement pass is not refreshing it")
        return 1

    res = cch.check(h, exp)
    sh = h.get("shadow_since_measure_from") or {}
    co = h.get("collection") or {}
    print(f"CLV collection health ({h.get('generated_at')}): {res['status']}")
    print(f"  shadow since {exp.get('measure_from')}: {sh.get('with_t30_close')}/"
          f"{sh.get('eligible')} certified T-30m ({sh.get('certified_t30_pct')}%), "
          f"T-10m {sh.get('certified_t10_pct')}%  [floor {exp.get('min_certified_t30_pct')}%]")
    print(f"  collection {co.get('fixture_markets')} fixture-markets: T-30m {co.get('t30_pct')}%, "
          f"T-10m {co.get('t10_pct')}%  [floor {exp.get('min_collection_t30_pct')}%]")
    for r in res["reasons"]:
        print(f"  - {r}")
    if res["status"] == "FAIL":
        print("::error::Certified CLV close coverage is below the configured expectation "
              "(config/clv_collection_expectation.json). Check the std/nf odds capture runs.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
