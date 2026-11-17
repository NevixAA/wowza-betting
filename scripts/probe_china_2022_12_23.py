"""
probe_china_2022_12_23.py - settle China Super League 2022-12-23 with API-Football's own record (owner
2026-11-10). Our history holds 'Zhejiang Professional 3-0 Tianjin Jinmen Tiger' and 'Henan Songshan
Longmen 3-0 Shenzhen' on that day; the stored provider fixture lists (FT / AET / PEN only) hold
neither, which blocks the proven aliases Tianjin Jinmen Tiger -> Tianjin Teda and Henan Songshan
Longmen -> Henan Jianye (af-alias-proof/1.0 'contradicted_by_a_date').

THREE calls at most, through collect_common.AFClient (budget guard), every status kept (AWD / WO /
CANC / PST included):
  1. /fixtures?league=169&season=2022&date=2022-12-23     every fixture of the league that day
  2. /fixtures?team=837&season=2022                        Tianjin Teda: every 2022 fixture
  3. /fixtures?team=840&season=2022                        Henan Jianye: every 2022 fixture
Writes output/collect/china_2022_12_23_probe.json (fixture id, date, kick-off, clubs and ids, status,
score.fulltime / halftime per fixture of the three answers). Prints counts, fixture ids and statuses only.

    python scripts/probe_china_2022_12_23.py
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

OUT = ROOT / "output" / "collect" / "china_2022_12_23_probe.json"
CALLS = [("league_day", {"league": 169, "season": 2022, "date": "2022-12-23"}),
         ("team_837_tianjin_teda", {"team": 837, "season": 2022}),
         ("team_840_henan_jianye", {"team": 840, "season": 2022})]
MAX_CALLS = len(CALLS)


def _fx(item: dict) -> dict:
    f, t, g, s = item.get("fixture") or {}, item.get("teams") or {}, item.get("goals") or {}, item.get("score") or {}
    st = f.get("status") or {}
    return {"fixture_id": f.get("id"), "date": (f.get("date") or "")[:10], "kickoff": f.get("date"),
            "league_id": (item.get("league") or {}).get("id"), "round": (item.get("league") or {}).get("round"),
            "home_id": (t.get("home") or {}).get("id"), "home": (t.get("home") or {}).get("name"),
            "away_id": (t.get("away") or {}).get("id"), "away": (t.get("away") or {}).get("name"),
            "status": st.get("short"), "status_long": st.get("long"),
            "goals": [g.get("home"), g.get("away")],
            "fulltime": [(s.get("fulltime") or {}).get("home"), (s.get("fulltime") or {}).get("away")],
            "halftime": [(s.get("halftime") or {}).get("home"), (s.get("halftime") or {}).get("away")]}


def main() -> int:
    budget = Path(os.getenv("RUNNER_TEMP", str(ROOT / "output" / "collect"))) / "china_probe_budget.json"
    client = AFClient(budget_file=budget)
    doc = {"artifact": "china_2022_12_23_probe", "generated_at": datetime.now(timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"), "calls": []}
    spent = 0
    for name, params in CALLS:
        if spent >= MAX_CALLS:
            break
        try:
            body = client.get("fixtures", params)
        except BudgetStop as e:
            print(f"[probe] budget stop: {e}")
            break
        spent += 1
        items = [_fx(i) for i in (body or {}).get("response", [])] if body else None
        doc["calls"].append({"name": name, "params": params, "ok": body is not None,
                             "results": None if items is None else len(items), "fixtures": items})
        print(f"[probe] {name}: {'FAILED' if items is None else len(items)} fixture(s)")
        for f in items or []:
            if name == "league_day" or f["date"] == "2022-12-23" or f["status"] not in ("FT", "AET", "PEN"):
                print(f"   fixture {f['fixture_id']} {f['date']} status {f['status']}")
    doc["calls_spent"] = spent
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[probe] calls spent: {spent}; wrote {OUT.relative_to(ROOT)}")
    return 0 if spent and all(c["ok"] for c in doc["calls"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
