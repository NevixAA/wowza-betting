"""
probe_supplement_rounds.py - the provider's competition ROUND label of every fixture in
output/corrections/history_provider_supplement.csv (owner 2026-11-11: cup / knockout rows must not
enter the supplement; the stored fixture lists hold no round label).

/fixtures?ids=<up to 20 ids> - ceil(n / 20) calls (608 fixtures -> 31), through collect_common.AFClient
(budget guard), MAX_CALLS hard cap. Writes output/collect/supplement_fixture_rounds.json: per fixture id
the provider league id / name / type-free label, season, round, status. Prints counts only.

    python scripts/probe_supplement_rounds.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import AFClient, BudgetStop  # noqa: E402

SUPPLEMENT = ROOT / "output" / "corrections" / "history_provider_supplement.csv"
OUT = ROOT / "output" / "collect" / "supplement_fixture_rounds.json"
BATCH = 20
MAX_CALLS = 40


def main() -> int:
    ids = sorted(int(x) for x in pd.read_csv(SUPPLEMENT)["provider_fixture_id"].dropna().unique())
    batches = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    if len(batches) > MAX_CALLS:
        print(f"[rounds] {len(batches)} calls needed > cap {MAX_CALLS}: refusing")
        return 1
    budget = Path(os.getenv("RUNNER_TEMP", str(ROOT / "output" / "collect"))) / "rounds_probe_budget.json"
    client = AFClient(budget_file=budget)
    got, spent, failed = {}, 0, 0
    for b in batches:
        try:
            body = client.get("fixtures", {"ids": "-".join(map(str, b))})
        except BudgetStop as e:
            print(f"[rounds] budget stop: {e}")
            break
        spent += 1
        if body is None:
            failed += 1
            continue
        for it in body.get("response", []):
            f, lg = it.get("fixture") or {}, it.get("league") or {}
            got[str(f.get("id"))] = {"league_id": lg.get("id"), "league_name": lg.get("name"),
                                     "country": lg.get("country"), "season": lg.get("season"),
                                     "round": lg.get("round"), "status": (f.get("status") or {}).get("short"),
                                     "date": (f.get("date") or "")[:10]}
    doc = {"artifact": "supplement_fixture_rounds", "generated_at": datetime.now(timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"), "calls_spent": spent, "calls_failed": failed,
           "fixtures_asked": len(ids), "fixtures_answered": len(got), "fixtures": got}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"[rounds] calls spent {spent} (failed {failed}); fixtures answered {len(got)}/{len(ids)}")
    return 0 if len(got) == len(ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
