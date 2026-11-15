"""
Fixture lists for player-history rows whose fixture is outside the collected lists (P7 item 5).
===============================================================================================
The player re-fetch (scripts/collect_player_refetch.py) refused 6,354 rows in 247 fixtures as
`fixture_not_in_lists`: the rows carry an API-Football fixture id, but the fixture belongs to a
competition whose list was never collected (national-team matches 2025-06..2026-07 - qualifiers,
friendlies, the 2026 World Cup - and a few club extras such as a Serie B playout). Without the list,
the date and the two clubs cannot be proven, so the rows' zeros stay unverified.

THE PROOF comes from the provider itself: `/fixtures?ids=a-b-c...` (API-Football accepts up to 20 ids
per call, so about 13 calls) returns each fixture's league, season, date, teams, score and status.
Those finished fixtures are stored in output/collect/refetch_extra_fixtures.parquet (same columns as
the competition lists) and read ONLY by the player re-fetch plan, which then applies its usual proof
(same date, same clubs by player_model.club_rule.same_fixture_sides). They are deliberately NOT added
to collect_af_lists.fixture_list(): that list feeds the team-id collectors (injuries, coaches,
transfers), and national teams must not start costing calls there.

    python scripts/collect_refetch_extra_fixtures.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_af_lists as AL  # noqa: E402
from collect_common import COLLECT_DIR, AFClient, BudgetStop, read_parquet  # noqa: E402

OUT = COLLECT_DIR / "refetch_extra_fixtures.parquet"
COLS = ["fixture_id", "league_id", "league_name", "competition_type", "season", "date", "home_id",
        "home_name", "away_id", "away_name", "ft_home", "ft_away", "status", "source", "fetched_at"]
#: API-Football's documented maximum for /fixtures?ids=
IDS_PER_CALL = 20


def missing_fixture_ids() -> list[int]:
    """Fixture ids carried by re-fetch source rows that no collected list (nor this file) holds."""
    import collect_player_refetch as R
    rows = R.source_rows()
    have = set(AL.fixture_list()["fixture_id"].astype(int))
    extra = read_parquet(OUT)
    if len(extra):
        have |= set(pd.to_numeric(extra["fixture_id"], errors="coerce").dropna().astype(int))
    ids = pd.to_numeric(rows["fixture_id"], errors="coerce").dropna().astype(int)
    return sorted(set(ids) - have, reverse=True)


def parse(body: dict, fetched_at: str) -> list[dict]:
    out = []
    for f in (body or {}).get("response", []) or []:
        fx, lg, tm, gl = f.get("fixture", {}), f.get("league", {}), f.get("teams", {}), f.get("goals", {})
        if fx.get("id") is None or lg.get("id") is None:
            continue
        out.append({"fixture_id": int(fx["id"]), "league_id": int(lg["id"]), "league_name": lg.get("name"),
                    "competition_type": None, "season": str(lg.get("season")), "date": fx.get("date"),
                    "home_id": (tm.get("home") or {}).get("id"), "home_name": (tm.get("home") or {}).get("name"),
                    "away_id": (tm.get("away") or {}).get("id"), "away_name": (tm.get("away") or {}).get("name"),
                    "ft_home": gl.get("home"), "ft_away": gl.get("away"),
                    "status": (fx.get("status") or {}).get("short"),
                    "source": "api-football /fixtures?ids", "fetched_at": fetched_at})
    return out


def run(dry_run: bool = False, client: AFClient | None = None) -> dict:
    ids = missing_fixture_ids()
    chunks = [ids[i:i + IDS_PER_CALL] for i in range(0, len(ids), IDS_PER_CALL)]
    st = {"missing_fixtures": len(ids), "planned_calls": len(chunks), "calls": 0, "stored": 0,
          "not_finished": 0}
    print(f"[refetchlists] {json.dumps(st)}")
    if dry_run or not chunks:
        return st
    client = client or AFClient()
    rows = []
    try:
        for ch in chunks:
            body = client.get("fixtures", {"ids": "-".join(str(i) for i in ch)})
            if body is None:
                continue                                  # failed: asked again next run
            st["calls"] += 1
            rows.extend(parse(body, AL.now_iso()))
    except BudgetStop as e:
        print(f"[refetchlists] budget stop: {e}")
    finally:
        if rows:
            new = pd.DataFrame(rows, columns=COLS)
            st["not_finished"] = int((~new["status"].isin(AL.FINISHED)).sum())
            old = read_parquet(OUT)
            allf = pd.concat([old, new], ignore_index=True) if len(old) else new
            allf = allf.drop_duplicates("fixture_id", keep="last")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            allf.to_parquet(OUT, index=False)
            st["stored"] = int(len(new))
        if client is not None:
            client.flush()
    print(f"[refetchlists] {json.dumps(st)}")
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    run(dry_run=ap.parse_args(argv).dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
