"""
collect_af_history_gap.py - fill the af_history.parquet gap (owner decision C, 2026-11-02) under
the Data Truth collection budget.

The registry-derived season plan of scripts/backfill_af_history.py now includes the 2025 season of
the calendar-year new-format leagues, which the old typed season list never collected. This runs
that SAME backfill (same parsing, same skip list, same output) league by league, with every HTTP
call routed through the hard budget of collect_common.AFClient (45,000 calls per run-day for this
job; stop when the account has < 25,000 calls left today).

    python scripts/collect_af_history_gap.py [--league NAME]
    python scripts/collect_af_history_gap.py --insidebox [--dry-run]

A budget stop inside a league ends the run; that league's per-fixture statistics are already in
scripts/af_history_cache/ (restored by the workflow's actions/cache), so the next run re-reads them
at no cost and saves the league. Leagues finished before the stop are saved by the backfill itself.

--insidebox (Data Truth round 6): API-Football 'Shots insidebox' for EVERY league in af_history.
  backfill_af_history parses 'Shots insidebox' (HIB/AIB) since 2026-09, but only for new-format
  leagues and only for fixtures it had not banked yet: its skip list skips a banked fixture even when
  its HIB is empty, and Standard rows were banked by data_loader's write-back without it. So 77% of
  af_history rows carry no inside-box count. This mode:
    1. targets every af_history row without HIB or AIB (any league with an api_football_id in
       config/leagues.json);
    2. finds its API-Football fixture id WITHOUT a call: the row's own fixture_id, else the fixture
       lists already collected (output/strength/competition_fixtures.parquet from step 1 and
       output/collect/af_ht_scores.parquet) - same league id, same date (adjacent date only when the
       same date has none), clubs resolved with src.team_names.resolve_result; more than one
       candidate, or one fixture claimed by rows of different dates -> refused, counted;
    3. asks /leagues?id (1 call per league, journaled) which seasons have fixture statistics and
       skips the rest (provider has none: no call spent);
    4. reads /fixtures/statistics per fixture - from scripts/af_history_cache/stats_<id>.json when
       already cached (NO call), else one budget-guarded call whose body is cached there too.
       Home/away are taken by the provider's team ids, never by response order. A count the provider
       does not give stays missing (never 0). Journal output/collect/af_fixture_stats.parquet: a
       fixture stored there (also an empty answer) is never fetched again.
  OUTPUT beside the original (af_history.parquet is NEVER rewritten):
    output/collect/af_history_stats_fill.parquet  af_history's (league, date, home_team, away_team)
        + HIB, AIB (and HXG, AXG from the same response, no extra call), fixture_id, match_rule,
        source, matched_at. src/data_loader._enrich_af_stats_fill reads it (nan_only) for training
        and serving alike.
    output/collect/af_stats_fill_audit.json       per league: lacking, matched, refused, no provider
        statistics, filled; calls made / planned.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import (COLLECT_DIR, FLUSH_EVERY, AFClient, BudgetStop,  # noqa: E402
                            append_parquet, read_parquet)

AF_HISTORY = ROOT / "output" / "af_history.parquet"
FIXTURE_SOURCES = (ROOT / "output" / "strength" / "competition_fixtures.parquet",
                   COLLECT_DIR / "af_ht_scores.parquet")
STATS_STORE = COLLECT_DIR / "af_fixture_stats.parquet"
COVERAGE = COLLECT_DIR / "af_stats_coverage.parquet"
FILL = COLLECT_DIR / "af_history_stats_fill.parquet"
AUDIT = COLLECT_DIR / "af_stats_fill_audit.json"
STATS_CACHE = ROOT / "scripts" / "af_history_cache"
SOURCE = "api-football /fixtures/statistics"
STATS_COLS = ["fixture_id", "league_id", "season", "home_id", "away_id", "HIB", "AIB", "HXG", "AXG",
              "n_teams", "read_from", "fetched_at"]
COVERAGE_COLS = ["league_id", "season", "statistics_fixtures", "fetched_at"]
FILL_COLS = ["league", "date", "home_team", "away_team", "HIB", "AIB", "HXG", "AXG", "fixture_id",
             "af_date", "match_rule", "source", "matched_at"]


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


# ── --insidebox ─────────────────────────────────────────────────────────────────────────────────

def _af_ids() -> dict[str, int]:
    """{registry name: api_football_id} for EVERY registry league with an id (Standard included)."""
    from src import league_registry as LR
    return {e["name"]: int(e["api_football_id"]) for e in LR.leagues()
            if e.get("api_football_id") is not None}


def _int_or_none(v):
    try:
        return None if v is None or v != v else int(v)
    except (TypeError, ValueError):
        return None


def lacking_insidebox(af: pd.DataFrame | None = None) -> pd.DataFrame:
    """af_history rows without HIB or AIB, in a registry league with an API-Football id, with
    `canon` (registry name) and `league_id`."""
    from src import league_registry as LR
    if af is None:
        af = pd.read_parquet(AF_HISTORY)
    ids = _af_ids()
    h = af.copy()
    for c in ("HIB", "AIB", "fixture_id", "home_team_id", "away_team_id"):
        if c not in h.columns:
            h[c] = None
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].isin(ids)]
    h = h[h["HIB"].isna() | h["AIB"].isna()].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h = h[h["date"].notna() & h["home_team"].notna() & h["away_team"].notna()]
    h["league_id"] = h["canon"].map(ids).astype(int)
    return h.sort_values("date", ascending=False, kind="mergesort")


def fixture_index(sources=None) -> pd.DataFrame:
    """The collected API-Football fixture lists: one row per fixture id."""
    cols = ["fixture_id", "league_id", "season", "date", "home_id", "home_name", "away_id", "away_name"]
    parts = []
    for p in (FIXTURE_SOURCES if sources is None else sources):
        x = read_parquet(p)
        if x.empty or not set(cols) <= set(x.columns):
            continue
        parts.append(x[cols])
    if not parts:
        return pd.DataFrame(columns=cols)
    fx = pd.concat(parts, ignore_index=True).dropna(subset=["fixture_id"])
    fx["fixture_id"] = fx["fixture_id"].astype(int)
    fx["league_id"] = pd.to_numeric(fx["league_id"], errors="coerce")
    fx["season"] = fx["season"].astype(str)
    fx["_d"] = pd.to_datetime(fx["date"], errors="coerce").dt.normalize()
    return fx.drop_duplicates("fixture_id", keep="last").reset_index(drop=True)


def match_fixtures(lacking: pd.DataFrame, fx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Each lacking af_history row -> (fixture_id, season, home_id, away_id, match_rule) or a counted
    refusal. Rule: the row's own fixture_id; else same league id + date, clubs resolved home->home
    and away->away (team_names.resolve_result), adjacent date only when the same date has none."""
    from src.season_resolver import resolve_api_season
    from src.team_names import resolve_result
    stats: dict = defaultdict(lambda: defaultdict(int))
    by_day = {k: g for k, g in fx.groupby(["league_id", "_d"])} if len(fx) else {}
    by_id = fx.set_index("fixture_id") if len(fx) else pd.DataFrame()
    memo: dict = {}

    def _res(name, pool, key):
        k = key + (str(name),)
        if k not in memo:
            memo[k] = resolve_result(str(name), [str(c) for c in pool])
        return memo[k]

    picks = []
    for r in lacking.itertuples(index=False):
        st = stats[r.canon]
        st["lacking"] += 1
        fid = _int_or_none(r.fixture_id)
        if fid is not None:
            row = by_id.loc[fid] if len(by_id) and fid in by_id.index else None
            picks.append({"league": r.league,
                          "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                          "canon": r.canon, "league_id": r.league_id, "fixture_id": fid,
                          # the provider's own season label, else src.season_resolver on the date
                          "season": (str(row["season"]) if row is not None
                                     else resolve_api_season(r.canon, r.date.date())),
                          "home_id": _int_or_none(r.home_team_id), "away_id": _int_or_none(r.away_team_id),
                          "af_date": str(pd.Timestamp(r.date).date()), "match_rule": "af_fixture_id"})
            continue
        found, rule = pd.DataFrame(), None
        for days, name in (((0,), "same_day"), ((-1, 1), "adjacent_day")):
            ks = [(r.league_id, r.date + pd.Timedelta(days=k)) for k in days]
            parts = [by_day[k] for k in ks if k in by_day]
            if not parts:
                continue
            pool = pd.concat(parts)
            h = _res(r.home_team, pool["home_name"].unique(), (r.league_id, r.date, name, "h"))
            a = _res(r.away_team, pool["away_name"].unique(), (r.league_id, r.date, name, "a"))
            if h is None or a is None:
                continue
            found = pool[(pool["home_name"] == h) & (pool["away_name"] == a)]
            if len(found):
                rule = name
                break
        if len(found) == 0:
            st["unmatched"] += 1
            continue
        if found["fixture_id"].nunique() > 1:
            st["refused_ambiguous"] += 1
            continue
        f = found.iloc[0]
        picks.append({"league": r.league, "date": r.date, "home_team": r.home_team,
                      "away_team": r.away_team, "canon": r.canon, "league_id": r.league_id,
                      "fixture_id": int(f["fixture_id"]), "season": str(f["season"]),
                      "home_id": _int_or_none(f["home_id"]), "away_id": _int_or_none(f["away_id"]),
                      "af_date": str(f["date"])[:10], "match_rule": rule})
    m = pd.DataFrame(picks, columns=["league", "date", "home_team", "away_team", "canon", "league_id",
                                     "fixture_id", "season", "home_id", "away_id", "af_date",
                                     "match_rule"])
    if not m.empty:
        # one fixture claimed by rows of DIFFERENT dates: two matches for one fixture -> refused
        twice = m.groupby("fixture_id")["date"].transform("nunique") > 1
        for lg, n in m.loc[twice, "canon"].value_counts().items():
            stats[lg]["refused_fixture_claimed_twice"] += int(n)
        m = m[~twice]
        for lg, n in m["canon"].value_counts().items():
            stats[lg]["matched"] += int(n)
    return m.reset_index(drop=True), {k: dict(v) for k, v in stats.items()}


def parse_coverage(body: dict, league_id: int) -> list[dict]:
    """/leagues?id -> one row per season: does API-Football hold fixture statistics for it."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for item in (body or {}).get("response") or []:
        for s in item.get("seasons") or []:
            cov = ((s.get("coverage") or {}).get("fixtures") or {}).get("statistics_fixtures")
            if s.get("year") is None:
                continue
            rows.append({"league_id": int(league_id), "season": str(s["year"]),
                         "statistics_fixtures": bool(cov) if cov is not None else None,
                         "fetched_at": now})
    return rows


def parse_stats(resp, home_id, away_id) -> dict:
    """HIB/AIB ('Shots insidebox') and HXG/AXG ('expected_goals') by the provider's TEAM IDS. No
    response, or teams that are not the fixture's two clubs -> all missing (never 0)."""
    import backfill_af_history as bah
    out = {"HIB": None, "AIB": None, "HXG": None, "AXG": None, "n_teams": len(resp or [])}
    by_team = {_int_or_none((t.get("team") or {}).get("id")): t.get("statistics") or []
               for t in (resp or [])}
    if home_id is None or away_id is None or home_id not in by_team or away_id not in by_team:
        return out
    h, a = by_team[home_id], by_team[away_id]
    out.update(HIB=bah._stat(h, "Shots insidebox"), AIB=bah._stat(a, "Shots insidebox"),
               HXG=bah._statf(h, "expected_goals"), AXG=bah._statf(a, "expected_goals"))
    return out


def _cached_stats(fid: int, cache_dir: Path = STATS_CACHE):
    f = Path(cache_dir) / f"stats_{fid}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_fill(m: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if m.empty or store.empty:
        fill = pd.DataFrame(columns=FILL_COLS)
    else:
        s = store.drop_duplicates("fixture_id", keep="last").set_index("fixture_id")
        j = m[m["fixture_id"].isin(s.index)].copy()
        for c in ("HIB", "AIB", "HXG", "AXG"):
            j[c] = j["fixture_id"].map(s[c])
        j = j[j[["HIB", "AIB", "HXG", "AXG"]].notna().any(axis=1)]
        j["date"] = pd.to_datetime(j["date"]).dt.strftime("%Y-%m-%d")
        j["source"], j["matched_at"] = SOURCE, now
        fill = j[FILL_COLS]
    old = read_parquet(FILL)
    if not old.empty:
        fill = pd.concat([old[FILL_COLS], fill], ignore_index=True)
    fill = fill.drop_duplicates(subset=["league", "date", "home_team", "away_team"], keep="last")
    FILL.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILL.with_suffix(".tmp.parquet")
    fill.reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(FILL)
    return fill


def run_insidebox(dry_run: bool = False, client: AFClient | None = None,
                  af: pd.DataFrame | None = None, cache_dir: Path = STATS_CACHE) -> dict:
    lacking = lacking_insidebox(af)
    m, per_league = match_fixtures(lacking, fixture_index())
    cov = read_parquet(COVERAGE)
    store = read_parquet(STATS_STORE)
    have = set(pd.to_numeric(store["fixture_id"], errors="coerce").dropna().astype(int)) \
        if not store.empty else set()
    have_cov = (set(pd.to_numeric(cov["league_id"], errors="coerce").dropna().astype(int))
                if "league_id" in cov.columns else set())
    need_cov = sorted(set(m["league_id"].astype(int)) - have_cov) if len(m) else []
    todo = m.drop_duplicates("fixture_id")
    todo = todo[~todo["fixture_id"].isin(have)]
    cached = [f for f in todo["fixture_id"] if _cached_stats(f, cache_dir) is not None]
    stats = {"lacking_rows": int(len(lacking)), "matched_rows": int(len(m)),
             "fixtures_to_read": int(len(todo)), "fixtures_in_local_cache": len(cached),
             "planned_coverage_calls": len(need_cov),
             "planned_stats_calls_upper_bound": int(len(todo) - len(cached)), "calls": 0}
    print(f"[insidebox] {json.dumps(stats)}")
    if dry_run:
        stats["per_league"] = per_league
        return stats
    client = client or AFClient()
    batch: list[dict] = []
    now = lambda: datetime.now(timezone.utc).isoformat()                      # noqa: E731
    try:
        for lid in need_cov:
            body = client.get("leagues", {"id": int(lid)})
            if body is None:
                continue
            stats["calls"] += 1
            rows = parse_coverage(body, lid)
            if rows:
                append_parquet(COVERAGE, rows, ["league_id", "season"], COVERAGE_COLS)
        cov = read_parquet(COVERAGE)
        ok = {(int(r.league_id), str(r.season)): r.statistics_fixtures for r in cov.itertuples()} \
            if not cov.empty else {}
        for r in todo.itertuples(index=False):
            key = (int(r.league_id), str(r.season))
            has = ok.get(key)
            if not (has is not None and has == has and bool(has)):
                per_league.setdefault(r.canon, {})
                tag = "no_provider_statistics" if key in ok else "coverage_unknown"
                per_league[r.canon][tag] = per_league[r.canon].get(tag, 0) + 1
                continue
            body = _cached_stats(r.fixture_id, cache_dir)
            read_from = "cache"
            if body is None:
                body = client.get("fixtures/statistics", {"fixture": int(r.fixture_id)})
                if body is None:
                    continue                                  # failed: nothing recorded, retried
                stats["calls"] += 1
                read_from = "api"
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                (Path(cache_dir) / f"stats_{int(r.fixture_id)}.json").write_text(
                    json.dumps(body, ensure_ascii=False), encoding="utf-8")
            hid, aid = _int_or_none(r.home_id), _int_or_none(r.away_id)
            batch.append({"fixture_id": int(r.fixture_id), "league_id": int(r.league_id),
                          "season": str(r.season), "home_id": hid, "away_id": aid,
                          **parse_stats(body.get("response"), hid, aid),
                          "read_from": read_from, "fetched_at": now()})
            if len(batch) >= FLUSH_EVERY:
                append_parquet(STATS_STORE, batch, ["fixture_id"], STATS_COLS)
                batch = []
    except BudgetStop as e:
        print(f"[insidebox] budget stop: {e} (resumable)")
        stats["budget_stop"] = str(e)
    finally:
        if batch:
            append_parquet(STATS_STORE, batch, ["fixture_id"], STATS_COLS)
        client.flush()
    fill = _write_fill(m, read_parquet(STATS_STORE))
    for lg, n in fill["league"].value_counts().items():
        per_league.setdefault(lg, {})["filled_total"] = int(n)
    stats["filled_rows_total"] = int(len(fill))
    stats["per_league"] = per_league
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps({"written_at": datetime.now(timezone.utc).isoformat(),
                                 "note": "Shots insidebox / expected_goals for af_history rows that "
                                         "lack them, from API-Football; af_history.parquet itself "
                                         "is not modified.", **stats},
                                indent=1, sort_keys=True, default=str), encoding="utf-8")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="af_history gap fill under the collection budget")
    ap.add_argument("--league", default=None)
    ap.add_argument("--insidebox", action="store_true",
                    help="collect 'Shots insidebox' for af_history rows lacking it (all leagues)")
    ap.add_argument("--dry-run", action="store_true", help="--insidebox: plan and count calls only")
    a = ap.parse_args(argv)
    if a.insidebox:
        run_insidebox(dry_run=a.dry_run)
        return 0
    run(league=a.league)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
