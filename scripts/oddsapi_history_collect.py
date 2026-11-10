"""
OddsAPI HISTORICAL collector CLI (src/oddsapi_history.py). The data goes ONLY to the private store.

  --dry-run --plan-out FILE     count the exact calls and MAXIMUM credits per tier (no network)
  --store DIR                   run: buy the plan in priority order into DIR (a checkout of the PRIVATE
                                wowza-exec branch `odds-history`), resuming from its ledger
  --store DIR --probe-s         the correct_score probe (one P7 probe event per league)

Run settings come from the environment and have no fallback:
  ODDS_API_KEY                  the provider key (a secret; never printed)
  ODDS_HIST_RUN_CREDIT_CAP      the most credits this run may spend
  ODDS_HIST_FLOOR               stop before x-requests-remaining would fall below it (>= 260,000)
Exit codes: 0 every planned tier done, 10 time budget, 20 run cap, 30 floor, 40 error / unknown cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import oddsapi_history as OH  # noqa: E402


def _need_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise SystemExit(f"{name} is not set (no fallback: the run refuses to guess it)")
    return v


def _inside_public_repo(p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-out", type=Path)
    ap.add_argument("--store", type=Path)
    ap.add_argument("--probe-s", action="store_true")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--tiers", help="comma list, a subset of the priority order (default: every tier)")
    a = ap.parse_args(argv)

    days = OH.fixture_days(OH.load_fixtures())
    tiers = OH.TIERS if not a.tiers else tuple(t for t in OH.TIERS if t.name in set(a.tiers.split(",")))

    if a.dry_run:
        if a.plan_out is None:
            raise SystemExit("--dry-run needs --plan-out")
        plan = OH.build_plan(days, tiers, s_ok=OH.s_probe_leagues(a.store))
        s = OH.summarise_plan(plan, days)
        s["planned_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        s["plan_digest"] = OH.plan_digest(plan)
        s["s_probe"] = "applied" if OH.s_probe_leagues(a.store) is not None else "not run: S counted for every team league"
        s["props_markets_from_probe"] = {k: list(v) for k, v in OH.probe_props_markets().items()}
        a.plan_out.parent.mkdir(parents=True, exist_ok=True)
        a.plan_out.write_text(json.dumps(s, indent=1, sort_keys=False) + "\n", encoding="utf-8")
        for t, v in s["tiers"].items():
            OH.say(f"{t}: calls {v['calls']:,}, max credits {v['max_credits']:,}, "
                   f"cumulative {v['cumulative_max_credits']:,}, fits {v['fits_in_usable']}")
        OH.say(f"total max credits {s['total_max_credits']:,} vs usable {s['usable_credits']:,}")
        return 0

    if a.store is None:
        raise SystemExit("a run needs --store (the PRIVATE odds-history checkout)")
    if _inside_public_repo(a.store):
        raise SystemExit("--store is inside the public wowza-betting tree: refused (redistribution prohibited)")
    floor = int(_need_env("ODDS_HIST_FLOOR"))
    cap = int(_need_env("ODDS_HIST_RUN_CREDIT_CAP"))
    if a.max_minutes is None:
        raise SystemExit("--max-minutes is required")
    meter = OH.Meter(_need_env("ODDS_API_KEY"), floor=floor, run_cap=cap, max_seconds=a.max_minutes * 60)
    if a.probe_s:
        res = OH.probe_s(meter, a.store)
        return OH.EXIT_ERROR if res.get("stopped") else OH.EXIT_DONE
    plan = OH.build_plan(days, tiers, s_ok=OH.s_probe_leagues(a.store) or set())
    run_id = os.getenv("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime("local%Y%m%d%H%M%S")
    run_id = f"{run_id}-{os.getenv('GITHUB_RUN_ATTEMPT', '1')}-{datetime.now(timezone.utc):%H%M%S}"
    store = OH.Store(a.store, run_id)
    OH.say(f"plan digest {OH.plan_digest(plan)}; units already stored {len(store.done):,}")
    code, _ = OH.Collector(store, meter, days, plan, [t.name for t in tiers]).run()
    return code


if __name__ == "__main__":
    sys.exit(main())
