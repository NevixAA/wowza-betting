"""
probe_fixture_record.py - API-Football's own record of ONE fixture, to settle a provider score conflict
(owner 2026-11-15: Ireland Premier Division 2026-11-07 UCD v Drogheda, football-data 2-1 vs
API-Football 1-2). Two calls through collect_common.AFClient (budget guard):
  /fixtures?id=<id>               the fixture record now: status, score.halftime / fulltime, teams
  /fixtures/events?fixture=<id>   every goal event: minute, team, scorer, type (own goal / penalty)
Writes output/collect/fixture_probe_<id>.json; prints counts, the status and the goal events only.

    python scripts/probe_fixture_record.py 1648307 [more ids ...]      # 2 calls per id, at most 20

2026-11-15 (go-live rehearsal): also the 7 history rows whose football-data score disagrees with
API-Football (output/corrections/history_score_refused.csv, ft_status_scores_differ) - the goal
events and the fixture's elapsed minutes say which score was played (or that the match was stopped).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import AFClient, BudgetStop  # noqa: E402

MAX_CALLS = 2
MAX_TOTAL = 20


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ids = [int(a) for a in argv]
    if 2 * len(ids) > MAX_TOTAL:
        print(f"[probe] {2 * len(ids)} calls > cap {MAX_TOTAL}: refusing")
        return 1
    rc = 0
    for fid in ids:
        rc |= one(fid)
    return rc


def one(fid: int) -> int:
    budget = Path(os.getenv("RUNNER_TEMP", str(ROOT / "output" / "collect"))) / "fixture_probe_budget.json"
    client = AFClient(budget_file=budget)
    doc = {"artifact": "fixture_probe", "fixture_id": fid,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "calls": []}
    for name, ep, params in (("fixture", "fixtures", {"id": fid}),
                             ("events", "fixtures/events", {"fixture": fid}))[:MAX_CALLS]:
        try:
            body = client.get(ep, params)
        except BudgetStop as e:
            print(f"[probe] budget stop: {e}")
            break
        doc["calls"].append({"name": name, "endpoint": ep, "params": params, "ok": body is not None,
                             "response": (body or {}).get("response")})
    fx = next((c["response"] for c in doc["calls"] if c["name"] == "fixture" and c["ok"]), None) or []
    if fx:
        f = fx[0]
        print(f"[probe] fixture {fid}: status {f['fixture']['status']['short']}; "
              f"{f['teams']['home']['name']} v {f['teams']['away']['name']}; "
              f"HT {f['score']['halftime']}; FT {f['score']['fulltime']}")
    ev = next((c["response"] for c in doc["calls"] if c["name"] == "events" and c["ok"]), None) or []
    goals = [e for e in ev if e.get("type") == "Goal"]
    print(f"[probe] events {len(ev)}, goal events {len(goals)}")
    for g in goals:
        print(f"   {g['time']['elapsed']}' {g['team']['name']} ({g.get('detail')})")
    out = ROOT / "output" / "collect" / f"fixture_probe_{fid}.json"
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[probe] calls spent {len(doc['calls'])}; wrote {out.relative_to(ROOT)}")
    return 0 if len(doc["calls"]) == MAX_CALLS and all(c["ok"] for c in doc["calls"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
