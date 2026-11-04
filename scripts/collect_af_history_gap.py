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
    python scripts/collect_af_history_gap.py --fullstats        # 0 API calls, see run_fullstats

A budget stop inside a league ends the run; that league's per-fixture statistics are already in
scripts/af_history_cache/ (restored by the workflow's actions/cache), so the next run re-reads them
at no cost and saves the league. Leagues finished before the stop are saved by the backfill itself.

--insidebox (Data Truth round 6): API-Football 'Shots insidebox' for EVERY league in af_history.
  backfill_af_history parses 'Shots insidebox' (HIB/AIB) since 2026-09, but only for new-format
  leagues and only for fixtures it had not banked yet: its skip list skips a banked fixture even when
  its HIB is empty, and Standard rows were banked by data_loader's write-back without it. So 77% of
  af_history rows carry no inside-box count. This mode:
    1. targets every af_history row without HIB or AIB (any league with an api_football_id in
       config/leagues.json) and, since fix round 9 (owner 2026-11-04), every af_history match whose
       SERVED shots, corners or fouls are missing: a row the loader wrote back (no fixture_id, no FT
       score - src/af_history_provenance) serves none of its own values, so it is served only where
       the --fullstats sidecar holds them (stats_targets; run_insidebox says which fixtures are read);
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


#: the served statistics whose absence makes an af_history match a collection target (owner
#: 2026-11-04, fix round 9): shots, corners, fouls - fullstats sidecar column per side
SERVED_STATS = {"shots": ("HS", "AS", "home_shots", "away_shots"),
                "corners": ("HC", "AC", "home_corners", "away_corners"),
                "fouls": ("HF", "AF", "home_fouls", "away_fouls")}


def stats_targets(af: pd.DataFrame | None = None, fullstats: pd.DataFrame | None = None,
                  statsfill: pd.DataFrame | None = None) -> pd.DataFrame:
    """--insidebox targets since fix round 9 (owner 2026-11-04): every af_history match whose SERVED
    inside-box shots are missing (before: rows with HIB/AIB null) PLUS every match whose SERVED
    shots, corners or fouls are missing, both under the loader's provenance rule
    (src/af_history_provenance): a written-back row serves none of its own values, so it is served
    only where a sidecar holds the value for that match (same key the loader joins on: af_history's
    league, date, home_team, away_team - the --fullstats sidecar, and for inside-box also the
    --insidebox fill); a provider row serves its own HS/HC/HF/HIB, else the sidecar's. Same columns
    as lacking_insidebox plus `reason` ('insidebox', 'served_stats' or 'insidebox+served_stats')."""
    from src import league_registry as LR
    from src.af_history_provenance import written_back_mask
    if af is None:
        af = pd.read_parquet(AF_HISTORY)
    if fullstats is None:
        fullstats = read_parquet(FULL_FILL)
    ids = _af_ids()
    h = af.copy()
    for c in ("HIB", "AIB", "fixture_id", "home_team_id", "away_team_id") + tuple(
            c for v in SERVED_STATS.values() for c in v[:2]):
        if c not in h.columns:
            h[c] = None
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].isin(ids)].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h = h[h["date"].notna() & h["home_team"].notna() & h["away_team"].notna()]
    wb = written_back_mask(h)
    key = ["league", "_d", "home_team", "away_team"]
    side = pd.DataFrame(columns=key)
    if fullstats is not None and not fullstats.empty and "league" in fullstats.columns:
        side = fullstats.copy()
        side["_d"] = pd.to_datetime(side["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for c in ("league", "home_team", "away_team"):
            side[c] = side[c].astype(str)
        side = side.drop_duplicates(key, keep="last")
    left = h[["league", "home_team", "away_team"]].astype(str).assign(
        _d=h["date"].dt.strftime("%Y-%m-%d"))[key]
    hk = left.merge(side, on=key, how="left")
    hk.index = h.index
    missing = pd.Series(False, index=h.index)
    for own_h, own_a, sc_h, sc_a in SERVED_STATS.values():
        served_h = pd.to_numeric(h[own_h], errors="coerce").where(~wb)
        served_a = pd.to_numeric(h[own_a], errors="coerce").where(~wb)
        if sc_h in hk.columns:
            served_h = served_h.fillna(pd.to_numeric(hk[sc_h], errors="coerce"))
            served_a = served_a.fillna(pd.to_numeric(hk[sc_a], errors="coerce"))
        missing |= served_h.isna() | served_a.isna()
    # inside-box under the same provenance rule: a written-back row's HIB/AIB is not served either
    # (the standard leagues' inside-box and xG came ONLY from such rows, which the pre-round-9 target
    # 'HIB is null' never selected), so it is a target unless a sidecar holds the value
    if statsfill is None:
        statsfill = read_parquet(FILL)
    ib_h = pd.to_numeric(h["HIB"], errors="coerce").where(~wb)
    ib_a = pd.to_numeric(h["AIB"], errors="coerce").where(~wb)
    if "home_insidebox" in hk.columns:
        ib_h = ib_h.fillna(pd.to_numeric(hk["home_insidebox"], errors="coerce"))
        ib_a = ib_a.fillna(pd.to_numeric(hk["away_insidebox"], errors="coerce"))
    if statsfill is not None and not statsfill.empty and {"HIB", "AIB", "league"} <= set(statsfill.columns):
        sf = statsfill.copy()
        sf["_d"] = pd.to_datetime(sf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for c in ("league", "home_team", "away_team"):
            sf[c] = sf[c].astype(str)
        sk = left.merge(sf.drop_duplicates(key, keep="last")[key + ["HIB", "AIB"]], on=key, how="left")
        sk.index = h.index
        ib_h = ib_h.fillna(pd.to_numeric(sk["HIB"], errors="coerce"))
        ib_a = ib_a.fillna(pd.to_numeric(sk["AIB"], errors="coerce"))
    ib = ib_h.isna() | ib_a.isna()
    h = h[ib | missing].copy()
    h["reason"] = ["insidebox+served_stats" if a and b else "insidebox" if a else "served_stats"
                   for a, b in zip(ib[h.index], missing[h.index])]
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
    response, or teams that are not the fixture's two clubs -> all missing (never 0). Inside-box is
    a count read by backfill_af_history._stat under the owner's team null rule (2026-11-04: null in
    an answered block = 0); xG is not a count, its null stays missing."""
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
                  af: pd.DataFrame | None = None, cache_dir: Path = STATS_CACHE,
                  fullstats: pd.DataFrame | None = None, fixtures: pd.DataFrame | None = None,
                  statsfill: pd.DataFrame | None = None) -> dict:
    """Targets: stats_targets (fix round 9, owner 2026-11-04) - rows lacking inside-box AND matches
    whose served shots / corners / fouls are missing. Which fixtures are read:
      * an inside-box target: not yet in the journal (as before) nor read into the --fullstats
        sidecar (whose answer already holds the inside-box value, or the provider's null);
      * a served-stats target: its answer has not been read into the --fullstats sidecar yet (that
        sidecar holds every value of an answer it read, so a fixture in it needs nothing more), and
        the journal does not record an EMPTY provider answer for it (n_teams 0: nothing to read).
        A journaled fixture whose cached answer was lost is read again - its corners and fouls were
        never stored anywhere else.
    A fixture whose answer is in scripts/af_history_cache costs no call; seasons the provider flags
    without statistics are skipped (no call). The --fullstats step that follows in the collection job
    parses every newly cached answer into the sidecar."""
    lacking = stats_targets(af, fullstats, statsfill)
    m, per_league = match_fixtures(lacking, fixture_index() if fixtures is None else fixtures)
    cov = read_parquet(COVERAGE)
    store = read_parquet(STATS_STORE)
    full = read_parquet(FULL_FILL) if fullstats is None else fullstats
    have = set(pd.to_numeric(store["fixture_id"], errors="coerce").dropna().astype(int)) \
        if not store.empty else set()
    empty = (set(pd.to_numeric(store.loc[pd.to_numeric(store["n_teams"], errors="coerce") == 0,
                                         "fixture_id"], errors="coerce").dropna().astype(int))
             if not store.empty and "n_teams" in store.columns else set())
    in_full = (set(pd.to_numeric(full["fixture_id"], errors="coerce").dropna().astype(int))
               if full is not None and not full.empty and "fixture_id" in full.columns else set())
    have_cov = (set(pd.to_numeric(cov["league_id"], errors="coerce").dropna().astype(int))
                if "league_id" in cov.columns else set())
    need_cov = sorted(set(m["league_id"].astype(int)) - have_cov) if len(m) else []
    if len(m):
        why = lacking.assign(date=lacking["date"].dt.strftime("%Y-%m-%d"))[
            ["league", "date", "home_team", "away_team", "reason"]].drop_duplicates(
            ["league", "date", "home_team", "away_team"])
        m = m.assign(_ds=pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")).merge(
            why.rename(columns={"date": "_ds"}), on=["league", "_ds", "home_team", "away_team"],
            how="left").drop(columns="_ds")
        m["reason"] = m["reason"].fillna("insidebox")
    else:
        m["reason"] = pd.Series(dtype=object)
    by_fix = m.groupby("fixture_id")["reason"].agg(lambda s: "+".join(sorted(set("+".join(s).split("+")))))
    todo = m.drop_duplicates("fixture_id").copy()
    todo["reason"] = todo["fixture_id"].map(by_fix).astype(str)
    wants_ib = (todo["reason"].str.contains("insidebox") & ~todo["fixture_id"].isin(have)
                & ~todo["fixture_id"].isin(in_full))
    wants_st = (todo["reason"].str.contains("served_stats") & ~todo["fixture_id"].isin(in_full)
                & ~todo["fixture_id"].isin(empty))
    todo = todo[wants_ib | wants_st]
    cached = [f for f in todo["fixture_id"] if _cached_stats(f, cache_dir) is not None]
    ok_cov = ({(int(r.league_id), str(r.season)): r.statistics_fixtures for r in cov.itertuples()}
              if not cov.empty and "statistics_fixtures" in cov.columns else {})
    cached_set = set(cached)
    flag = [(ok_cov.get((int(r.league_id), str(r.season))), int(r.fixture_id) in cached_set)
            for r in todo.itertuples(index=False)]
    no_prov = sum(1 for f, _ in flag if f is not None and f == f and not bool(f))
    unknown = sum(1 for f, _ in flag if f is None or f != f)
    n_cached = len(cached)
    covered_calls = sum(1 for f, c in flag if not c and not (f is not None and f == f and not bool(f)))
    stats = {"lacking_rows": int(len(lacking)),
             "target_rows_by_reason": {k: int(v) for k, v in lacking["reason"].value_counts().items()},
             "matched_rows": int(len(m)),
             "fixtures_to_read": int(len(todo)),
             "fixtures_to_read_by_reason": {k: int(v) for k, v in todo["reason"].value_counts().items()},
             "fixtures_in_local_cache": n_cached,
             "fixtures_skipped_no_provider_statistics": int(no_prov),
             "fixtures_coverage_unknown": int(unknown),
             "planned_coverage_calls": len(need_cov),
             "planned_stats_calls_upper_bound": int(len(todo) - n_cached),
             "planned_stats_calls_excluding_no_statistics_seasons": int(covered_calls), "calls": 0}
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


# ── --fullstats ─────────────────────────────────────────────────────────────────────────────────
# Owner 2026-11-04: every statistic API-Football returns per fixture, for every af_history match whose
# /fixtures/statistics answer is ALREADY cached (scripts/af_history_cache/stats_<id>.json, written by
# backfill_af_history and --insidebox). ZERO API calls: a fixture without a cached answer is counted
# ('not_cached') and left missing. Home/away by the provider's TEAM IDS, never response order.
# TEAM NULL RULE (owner 2026-11-04, fix round 9; src/af_team_stats is the one reading): in an
# ANSWERED team block a null COUNT ('Red Cards', 'Offsides', 'Corner Kicks', 'Fouls', shots,
# 'Goalkeeper Saves', 'Blocked Shots', 'Free Kicks') is 0; no answer is missing; percentages,
# expected_goals and goals_prevented stay missing when null; 'Total passes' / 'Passes accurate' stay
# missing too, even in an answered block (owner decision 2026-11-04, fix round 10: a null pass count
# means "not tracked"; see src/af_team_stats). Until fix round 9 every null was missing.
# A re-run re-parses every cached answer, so the committed sidecar takes the rule on the next run.

#: provider statistic type -> column stem (home_<stem> / away_<stem>). Possession and Passes % are
#: percentages ('55%' -> 55.0); expected_goals and goals_prevented are decimals. 'Free Kicks' is
#: stored since fix round 9 (2026-11-04): the fullstats audit reported it as the one unknown type.
FULL_STATS = {"Shots on Goal": "sot", "Shots off Goal": "shots_off", "Total Shots": "shots",
              "Blocked Shots": "blocked", "Shots insidebox": "insidebox",
              "Shots outsidebox": "outsidebox", "Fouls": "fouls", "Corner Kicks": "corners",
              "Offsides": "offsides", "Ball Possession": "possession", "Yellow Cards": "yellow",
              "Red Cards": "red", "Goalkeeper Saves": "saves", "Total passes": "passes",
              "Passes accurate": "passes_accurate", "Passes %": "pass_pct",
              "expected_goals": "xg", "goals_prevented": "goals_prevented",
              "Free Kicks": "free_kicks"}
FULL_FILL = COLLECT_DIR / "af_history_fullstats_fill.parquet"
FULL_AUDIT = COLLECT_DIR / "af_fullstats_audit.json"
FULL_COLS = (["league", "date", "home_team", "away_team", "fixture_id", "home_id", "away_id"]
             + [f"{side}_{stem}" for stem in FULL_STATS.values() for side in ("home", "away")]
             + ["match_rule", "source", "matched_at"])


def parse_full_stats(resp, home_id, away_id) -> dict | None:
    """{home_<stem>/away_<stem>: value} by the provider's team ids, under the owner's team null rule
    (src/af_team_stats.team_values: an answered block's null count is 0, an unanswered block or an
    unlisted type is missing, non-counts never 0); None when the answer does not hold BOTH of the
    fixture's clubs (nothing is placed by position). Also returns the provider types this map does
    not know under '_unknown' (logged, never stored)."""
    from src.af_team_stats import team_values
    by_team = {_int_or_none((t.get("team") or {}).get("id")): t.get("statistics") or []
               for t in (resp or [])}
    if home_id is None or away_id is None or home_id not in by_team or away_id not in by_team:
        return None
    out: dict = {"_unknown": set()}
    for side, tid in (("home", home_id), ("away", away_id)):
        out["_unknown"] |= {s.get("type") for s in by_team[tid]
                            if s.get("type") is not None and s.get("type") not in FULL_STATS}
        vals = team_values(by_team[tid])
        for typ, stem in FULL_STATS.items():
            out[f"{side}_{stem}"] = vals.get(typ)
    return out


def run_fullstats(af: pd.DataFrame | None = None, cache_dir: Path = STATS_CACHE,
                  fixtures: pd.DataFrame | None = None) -> dict:
    """Every af_history row in a registry league with an API-Football id -> its fixture id (same
    rules as --insidebox: own fixture_id, else the collected fixture lists, clubs proven by
    team_names.resolve_result, ambiguity refused) -> the cached statistics answer. No API call."""
    from src import league_registry as LR
    if af is None:
        af = pd.read_parquet(AF_HISTORY)
    ids = _af_ids()
    h = af.copy()
    for c in ("fixture_id", "home_team_id", "away_team_id"):
        if c not in h.columns:
            h[c] = None
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].isin(ids)].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h = h[h["date"].notna() & h["home_team"].notna() & h["away_team"].notna()]
    h["league_id"] = h["canon"].map(ids).astype(int)
    m, per_league = match_fixtures(h, fixture_index() if fixtures is None else fixtures)
    # team ids the --insidebox journal recorded for a fixture fill a match without them
    store = read_parquet(STATS_STORE)
    sid = (store.dropna(subset=["fixture_id"]).drop_duplicates("fixture_id", keep="last")
           .set_index("fixture_id")) if not store.empty else pd.DataFrame()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, unknown = [], set()
    counts: dict = defaultdict(lambda: defaultdict(int))
    for r in m.itertuples(index=False):
        hid, aid = _int_or_none(r.home_id), _int_or_none(r.away_id)
        if (hid is None or aid is None) and len(sid) and int(r.fixture_id) in sid.index:
            hid = hid if hid is not None else _int_or_none(sid.loc[int(r.fixture_id), "home_id"])
            aid = aid if aid is not None else _int_or_none(sid.loc[int(r.fixture_id), "away_id"])
        body = _cached_stats(int(r.fixture_id), cache_dir)
        if body is None:
            counts[r.canon]["not_cached"] += 1
            continue
        resp = body.get("response") if isinstance(body, dict) else body
        parsed = parse_full_stats(resp, hid, aid)
        if parsed is None:
            counts[r.canon]["teams_not_in_answer"] += 1
            continue
        unknown |= parsed.pop("_unknown")
        if all(v is None for v in parsed.values()):
            counts[r.canon]["empty_answer"] += 1
            continue
        counts[r.canon]["filled"] += 1
        rows.append({"league": r.league, "date": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                     "home_team": r.home_team, "away_team": r.away_team,
                     "fixture_id": int(r.fixture_id), "home_id": hid, "away_id": aid, **parsed,
                     "match_rule": r.match_rule, "source": SOURCE, "matched_at": now})
    fill = pd.DataFrame(rows, columns=FULL_COLS)
    old = read_parquet(FULL_FILL)
    if not old.empty:
        # reindex: a sidecar written before a column existed (home/away_free_kicks, fix round 9)
        # keeps that column missing for its rows until this run re-parses them
        fill = pd.concat([old.reindex(columns=FULL_COLS), fill], ignore_index=True)
    fill = fill.drop_duplicates(subset=["league", "date", "home_team", "away_team"], keep="last")
    num = [c for c in FULL_COLS if c.startswith(("home_", "away_")) and c not in ("home_team", "away_team")]
    fill[num] = fill[num].apply(pd.to_numeric, errors="coerce")    # None -> NaN (missing), never 0
    FULL_FILL.parent.mkdir(parents=True, exist_ok=True)
    tmp = FULL_FILL.with_suffix(".tmp.parquet")
    fill.reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(FULL_FILL)
    for lg, st in per_league.items():
        counts[lg].update({k: v for k, v in st.items()})
    stats = {"history_rows": int(len(h)), "matched_rows": int(len(m)),
             "filled_rows_this_run": len(rows), "filled_rows_total": int(len(fill)), "calls": 0,
             "unknown_provider_types": sorted(unknown),
             "non_null_share": {c: round(float(fill[c].notna().mean()), 4)
                                for c in FULL_COLS if c.startswith(("home_", "away_"))
                                and c not in ("home_team", "away_team", "home_id", "away_id")}
             if len(fill) else {},
             "per_league": {k: dict(v) for k, v in sorted(counts.items())}}
    print(f"[fullstats] rows {stats['history_rows']:,}; matched {stats['matched_rows']:,}; "
          f"filled {stats['filled_rows_total']:,}; unknown types {stats['unknown_provider_types']}; "
          f"0 API calls")
    FULL_AUDIT.write_text(json.dumps({"written_at": datetime.now(timezone.utc).isoformat(),
                                      "note": "Every /fixtures/statistics value for af_history "
                                              "matches whose answer is already cached; 0 API calls; "
                                              "a value the provider does not give is missing, never "
                                              "0; af_history.parquet is not modified.", **stats},
                                     indent=1, sort_keys=True, default=str), encoding="utf-8")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="af_history gap fill under the collection budget")
    ap.add_argument("--league", default=None)
    ap.add_argument("--insidebox", action="store_true",
                    help="collect /fixtures/statistics for af_history rows lacking inside-box or "
                         "served shots / corners / fouls (all leagues)")
    ap.add_argument("--fullstats", action="store_true",
                    help="every cached /fixtures/statistics value for af_history (0 API calls)")
    ap.add_argument("--dry-run", action="store_true", help="--insidebox: plan and count calls only")
    a = ap.parse_args(argv)
    if a.fullstats:
        run_fullstats()
        return 0
    if a.insidebox:
        run_insidebox(dry_run=a.dry_run)
        return 0
    run(league=a.league)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
