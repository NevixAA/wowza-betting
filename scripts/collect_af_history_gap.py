"""
collect_af_history_gap.py - fill the af_history.parquet gap (owner decision C, 2026-11-02) under
the Data Truth collection budget.

The registry-derived season plan of scripts/backfill_af_history.py now includes the 2025 season of
the calendar-year new-format leagues, which the old typed season list never collected. This runs
that SAME backfill (same parsing, same skip list, same output) league by league, with every HTTP
call routed through the hard budget of collect_common.AFClient (45,000 calls per run-day for this
job; stop when the account has < 25,000 calls left today).

    python scripts/collect_af_history_gap.py [--league NAME]

A budget stop inside a league ends the run; that league's per-fixture statistics are already in
scripts/af_history_cache/ (restored by the workflow's actions/cache), so the next run re-reads them
at no cost and saves the league. Leagues finished before the stop are saved by the backfill itself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import AFClient, BudgetStop  # noqa: E402


class GuardedRequests:
    """Drop-in for the `requests` module inside backfill_af_history: same .get signature, every call
    checked against and counted in the AFClient budget."""

    def __init__(self, client: AFClient):
        self.client = client

    def get(self, url, headers=None, params=None, timeout=None):
        self.client.check()
        r = self.client.session.get(url, headers=headers, params=params, timeout=timeout)
        self.client._count()
        rem = r.headers.get("x-ratelimit-requests-remaining")
        try:
            if rem is not None:
                self.client.remaining = int(rem)
        except (TypeError, ValueError):
            pass
        return r


def run(league: str | None = None, client: AFClient | None = None) -> dict:
    import backfill_af_history as bah
    client = client or AFClient()
    bah.requests = GuardedRequests(client)
    leagues = [league] if league else [n for n, _, _ in bah.league_plan()]
    done = []
    try:
        for name in leagues:
            client.check()
            bah.run(target_league=name)
            done.append(name)
    except BudgetStop as e:
        print(f"[af_history_gap] budget stop: {e} (resumable; {len(done)}/{len(leagues)} league(s) saved)")
    finally:
        client.flush()
    return {"leagues": len(leagues), "done": len(done)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="af_history gap fill under the collection budget")
    ap.add_argument("--league", default=None)
    a = ap.parse_args(argv)
    run(league=a.league)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
