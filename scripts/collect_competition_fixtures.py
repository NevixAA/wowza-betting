"""
collect_competition_fixtures.py - real results for the measured club strength (owner decision A).

    python scripts/collect_competition_fixtures.py            # collect what is missing
    python scripts/collect_competition_fixtures.py --dry-run  # print the plan, no fixture calls

WHAT. API-Football /fixtures?league&season (ONE call per competition-season) for
  * every registry club league and registry club cup (UEFA Champions / Europa / Conference
    League) that has an api_football_id, and
  * the continental cups of src/strength/competitions.py (Libertadores, Sudamericana, AFC and
    CONCACAF Champions League, Club World Cup), each verified first: API-Football /leagues?id must
    report type 'Cup' in country 'World', else the id is skipped and logged.
SEASONS. The provider's own season list (/leagues?id: year, start, end; one call per competition)
restricted to seasons overlapping [first match we hold, today]. No season is typed or guessed.

OUTPUT output/strength/competition_fixtures.parquet: one row per FINISHED fixture (FT/AET/PEN)
with API-Football fixture/team ids and the 90-minute score (score.fulltime; extra time and
penalties are not part of a 90-minute result). Journal: output/collect/competition_seasons.parquet
records each fetched competition-season; a season that has ENDED is never fetched again, the
current one is re-fetched every run (one call).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import COLLECT_DIR, AFClient, BudgetStop, append_parquet, history_frame, read_parquet  # noqa: E402

OUT = ROOT / "output" / "strength" / "competition_fixtures.parquet"
JOURNAL = COLLECT_DIR / "competition_seasons.parquet"
FINISHED = {"FT", "AET", "PEN"}
FIX_COLS = ["fixture_id", "league_id", "league", "league_name", "competition_type", "season", "date",
            "home_id", "home_name", "away_id", "away_name", "ft_home", "ft_away", "status"]
JOURNAL_COLS = ["league_id", "season", "season_start", "season_end", "n_fixtures", "n_finished",
                "fetched_at"]


def competitions() -> list[dict]:
    """[{league_id, league (registry name or None), competition_type, verify}]"""
    from src import league_registry as LR
    from src.strength.competitions import CROSS_LEAGUE_CUPS
    out, seen = [], set()
    for e in LR.leagues():
        ct, af = e.get("competition_type"), e.get("api_football_id")
        if ct in ("club_league", "club_cup") and af is not None and int(af) not in seen:
            seen.add(int(af))
            out.append({"league_id": int(af), "league": e["name"], "competition_type": ct, "verify": False})
    for af in CROSS_LEAGUE_CUPS:
        if af not in seen:                 # a registry cup always takes precedence
            out.append({"league_id": int(af), "league": None, "competition_type": "club_cup", "verify": True})
    return out


def data_start() -> pd.Timestamp | None:
    """The first match day of the results we hold (history banks and player_history)."""
    firsts = []
    for f in (ROOT / "output" / "fd_history.parquet", ROOT / "output" / "af_history.parquet"):
        if f.exists():
            d = pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"], errors="coerce")
            if d.notna().any():
                firsts.append(d.min())
    h = history_frame(["date"])
    if not h.empty:
        d = pd.to_datetime(h["date"], errors="coerce")
        if d.notna().any():
            firsts.append(d.min())
    return min(firsts).normalize() if firsts else None


def provider_seasons(client: AFClient, comp: dict) -> list[dict]:
    """[{season, start, end}] from /leagues?id. Empty when the id fails verification."""
    body = client.get("leagues", {"id": comp["league_id"]})
    resp = (body or {}).get("response") or []
    if len(resp) != 1:
        print(f"  [skip] league id {comp['league_id']}: /leagues returned {len(resp)} entries")
        return []
    lg, country = resp[0].get("league") or {}, resp[0].get("country") or {}
    if comp["verify"] and not (lg.get("type") == "Cup" and country.get("name") == "World"):
        print(f"  [skip] league id {comp['league_id']} ({lg.get('name')}): provider type "
              f"{lg.get('type')!r} / country {country.get('name')!r} is not a world club cup")
        return []
    comp["league_name"] = lg.get("name")
    out = []
    for s in resp[0].get("seasons") or []:
        st, en = pd.to_datetime(s.get("start"), errors="coerce"), pd.to_datetime(s.get("end"), errors="coerce")
        if s.get("year") is None or st != st or en != en:
            continue
        out.append({"season": str(s["year"]), "start": st, "end": en})
    return out


def parse(body: dict, comp: dict, season: str) -> list[dict]:
    rows = []
    for f in (body or {}).get("response") or []:
        fx, tm, sc = f.get("fixture") or {}, f.get("teams") or {}, (f.get("score") or {}).get("fulltime") or {}
        status = (fx.get("status") or {}).get("short")
        if status not in FINISHED:
            continue
        h, a = tm.get("home") or {}, tm.get("away") or {}
        rows.append({"fixture_id": fx.get("id"), "league_id": comp["league_id"], "league": comp["league"],
                     "league_name": comp.get("league_name"), "competition_type": comp["competition_type"],
                     "season": season, "date": str(fx.get("date") or "")[:10],
                     "home_id": h.get("id"), "home_name": h.get("name"),
                     "away_id": a.get("id"), "away_name": a.get("name"),
                     "ft_home": sc.get("home"), "ft_away": sc.get("away"), "status": status})
    return rows


def plan_seasons(seasons: list[dict], journal: pd.DataFrame, league_id: int, start, today) -> list[dict]:
    """Seasons overlapping [start, today], minus ENDED seasons already journaled."""
    done = set()
    if not journal.empty:
        j = journal[pd.to_numeric(journal["league_id"], errors="coerce") == league_id]
        ended = pd.to_datetime(j["season_end"], errors="coerce") < pd.to_datetime(j["fetched_at"], errors="coerce").dt.tz_localize(None).dt.normalize()
        done = set(j.loc[ended, "season"].astype(str))
    return [s for s in seasons if s["end"] >= start and s["start"] <= today and s["season"] not in done]


def run(dry_run: bool = False, client: AFClient | None = None) -> dict:
    client = client or AFClient()
    start = data_start()
    if start is None:
        print("[fixtures] no history on disk: no season span to collect")
        return {"fetched": 0}
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    journal = read_parquet(JOURNAL)
    stats = {"fetched": 0, "rows": 0, "skipped_ids": 0}
    try:
        for comp in competitions():
            seasons = provider_seasons(client, comp)
            if not seasons:
                stats["skipped_ids"] += 1
                continue
            todo = plan_seasons(seasons, journal, comp["league_id"], start, today)
            print(f"  {comp['league'] or comp.get('league_name')} (id {comp['league_id']}): "
                  f"{len(todo)} season(s) to fetch")
            if dry_run:
                continue
            for s in todo:
                body = client.get("fixtures", {"league": comp["league_id"], "season": s["season"]})
                if body is None:
                    continue                              # failed: nothing recorded, retried next run
                rows = parse(body, comp, s["season"])
                if rows:
                    append_parquet(OUT, rows, ["fixture_id"], FIX_COLS)
                n_all = len(body.get("response") or [])
                append_parquet(JOURNAL, [{"league_id": comp["league_id"], "season": s["season"],
                                          "season_start": s["start"].date().isoformat(),
                                          "season_end": s["end"].date().isoformat(),
                                          "n_fixtures": n_all, "n_finished": len(rows),
                                          "fetched_at": datetime.now(timezone.utc).isoformat()}],
                               ["league_id", "season"], JOURNAL_COLS)
                stats["fetched"] += 1
                stats["rows"] += len(rows)
    except BudgetStop as e:
        print(f"[fixtures] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        client.flush()
    print(f"[fixtures] {stats}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect cross-league competition fixtures")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run(dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
