"""
backfill_af_history.py
======================
One-time historical backfill: fetch match statistics from API-Football
for all new-format leagues and save as output/af_history.parquet.

This promotes new-format leagues from "goals only" to full-feature
training (shots, corners, fouls, cards) — same richness as standard leagues.

Cost: ~16,000–20,000 API-Football credits (one-time).
Cache: scripts/af_history_cache/ — never re-fetches completed fixtures.
Output: output/af_history.parquet

Usage:
    python scripts/backfill_af_history.py            # full run
    python scripts/backfill_af_history.py --dry-run  # estimate call count only
    python scripts/backfill_af_history.py --league "<registry league name>"  # single league
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE_DIR = Path(__file__).parent / "af_history_cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT = ROOT / "output" / "af_history.parquet"
OUTPUT.parent.mkdir(exist_ok=True)

_BASE    = "https://v3.football.api-sports.io"
_KEY     = os.getenv("APIFOOTBALL_KEY", "")
_HEADERS = {"x-apisports-key": _KEY}

# ── League config ─────────────────────────────────────────────────────────────
# Every NEW-FORMAT league in config/leagues.json (incl. the API-Football-only training leagues),
# with its API-Football id from the registry. Seasons are not typed here: they are the provider
# seasons (src/season_resolver) spanned by the league's OWN history - from its first match in
# fd_history / af_history to today - or, for an API-Football-only league with no history yet,
# its registered api_football_extra_seasons. A league with neither is skipped and logged.

def _history_start() -> dict:
    """{canonical league: earliest match date} over the committed history parquets."""
    from src.league_registry import canonical_name
    first: dict = {}
    for f in (ROOT / "output" / "fd_history.parquet", OUTPUT):
        if not f.exists():
            continue
        try:
            h = pd.read_parquet(f, columns=["league", "date"])
        except Exception as e:                                          # noqa: BLE001
            print(f"  [warn] {f.name} unreadable ({e}) — not used for season spans")
            continue
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        for lg, d in h.dropna(subset=["date"]).groupby("league")["date"].min().items():
            c = canonical_name(lg) or str(lg)
            first[c] = min(d, first.get(c, d))
    return first


def league_plan() -> list[tuple[str, int, list[str]]]:
    """[(league, af_league_id, [seasons])] for every registry new-format league."""
    from src import league_registry as LR
    from src.season_resolver import resolve_api_season, is_known
    starts = _history_start()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    plan = []
    for e in LR.leagues():
        if e.get("model_type") != "new_format":
            continue
        name, af_id = e["name"], e.get("api_football_id")
        if af_id is None:
            print(f"  [skip] {name}: no api_football_id in config/leagues.json")
            continue
        if name in starts:
            # one probe per month across the span: every season the history covers
            probes = pd.date_range(starts[name].normalize(), today, freq="MS").append(pd.DatetimeIndex([today]))
            seasons = sorted({s for s in (resolve_api_season(name, d.date()) for d in probes) if is_known(s)})
        else:
            seasons = [str(s) for s in (e.get("api_football_extra_seasons") or [])]
        if not seasons:
            print(f"  [skip] {name}: no history and no registered seasons — nothing to backfill")
            continue
        plan.append((name, int(af_id), seasons))
    return plan


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, cache_key: str, dry_run: bool = False,
         max_age_h: float | None = None) -> dict | None:
    """Cached GET. `max_age_h=None` caches FOREVER; a number expires the entry.

    WHY AN EXPIRY EXISTS AT ALL. This cache had none, and the workflow restores it on every run
    (`restore-keys: af-history-`). So the FIXTURES LIST for each league-season was fetched once,
    around 2026-08-18, and replayed unchanged ever since — it contained only the matches finished
    on that date, every one of which was already in the parquet. Result: zero new rows, no diff,
    no commit, and a green job every morning for 18 days while af_history.parquet stayed frozen
    at 2026-08-18. The classic shape here — the workflow was never failing, it was succeeding at
    doing nothing.

    The distinction that matters: a FINISHED fixture's statistics never change, so those stay
    cached permanently and cost nothing. A season's fixture LIST changes every time a match is
    played, so it must expire.
    """
    cache_f = CACHE_DIR / f"{cache_key}.json"
    if cache_f.exists():
        fresh = True
        if max_age_h is not None:
            age_h = (time.time() - cache_f.stat().st_mtime) / 3600.0
            fresh = age_h < max_age_h
        if fresh:
            return json.loads(cache_f.read_text(encoding="utf-8"))

    if dry_run:
        return None

    if not _KEY:
        print("[backfill] APIFOOTBALL_KEY not set — set env var and retry")
        sys.exit(1)

    try:
        r = requests.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params, timeout=20)
        remaining = r.headers.get("x-ratelimit-requests-remaining", "?")
        if r.status_code == 429:
            print(f"  [rate limit] sleeping 60s...")
            time.sleep(60)
            r = requests.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params, timeout=20)
        if r.status_code != 200:
            print(f"  [error] {endpoint} {params}: HTTP {r.status_code}")
            return None
        data = r.json()
        cache_f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.35)   # ~170 calls/min max — safe for pro tier
        return data
    except Exception as e:
        print(f"  [error] {endpoint}: {e}")
        return None


def _stat(stats_list: list, stat_type: str) -> int | None:
    for s in stats_list:
        if s.get("type") == stat_type:
            val = s.get("value")
            if val is None or val == "None":
                return None
            try:
                return int(str(val).replace("%", "").strip())
            except (ValueError, TypeError):
                return None
    return None


def _statf(stats_list: list, stat_type: str) -> float | None:
    """Float version of _stat, for statistics that are NOT counts.

    `_stat` casts with int(), so int("0.33") raises and returns None. Every statistic it was
    written for is a count, so that was correct — but expected_goals is 0.33, and reading it
    through _stat would have silently produced None for EVERY fixture while looking wired.
    That is the same failure as the columns this is meant to fix: present, plausible, empty.

    Deliberately a second function rather than a change to _stat: a dozen call sites depend on
    getting an int back, and widening the shared helper to satisfy one caller would quietly
    change the type of everything else.
    """
    for s in stats_list:
        if s.get("type") == stat_type:
            val = s.get("value")
            if val is None or val == "None":
                return None
            try:
                return float(str(val).replace("%", "").strip())
            except (ValueError, TypeError):
                return None
    return None


# ── Fetch season fixtures ─────────────────────────────────────────────────────

def fetch_season(league_name: str, league_id: int, season: str, dry_run: bool) -> list[dict]:
    ck = f"fixtures_{league_id}_{season}"
    # SIX HOURS, not forever. This is the list of matches in a season and it grows every matchday;
    # caching it permanently is what froze af_history.parquet for 18 days. Six hours still spares
    # the repeated calls inside one run and across the same morning, while guaranteeing a daily
    # job sees yesterday's results. One call per league-season per run — 16 leagues is 16 calls.
    data = _get("fixtures", {"league": league_id, "season": season}, ck, dry_run, max_age_h=6.0)

    if data is None:
        return []

    fixtures = data.get("response", [])
    completed = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "FT"]
    print(f"  {league_name} {season}: {len(completed)}/{len(fixtures)} completed fixtures")
    return completed


def fetch_stats(fixture_id: int, dry_run: bool) -> dict | None:
    ck = f"stats_{fixture_id}"
    data = _get("fixtures/statistics", {"fixture": fixture_id}, ck, dry_run)
    if data is None:
        return None
    return data.get("response")


# ── Parse one fixture into a row ──────────────────────────────────────────────

def parse_fixture(league_name: str, season: str, f: dict, stats_resp) -> dict | None:
    fix  = f.get("fixture", {})
    teams = f.get("teams", {})
    goals = f.get("goals", {})

    date_str = fix.get("date", "")[:10]
    home = teams.get("home", {}).get("name", "")
    away = teams.get("away", {}).get("name", "")
    fthg = goals.get("home")
    ftag = goals.get("away")

    if not home or not away or fthg is None or ftag is None:
        return None

    row = {
        "league":     league_name,
        "season":     season,
        "date":       date_str,
        "home_team":  home,
        "away_team":  away,
        # Provider ids from the same response: the fixture and both clubs are identified by id,
        # not only by their spelling (the skip list keys on fixture_id).
        "fixture_id":   fix.get("id"),
        "home_team_id": teams.get("home", {}).get("id"),
        "away_team_id": teams.get("away", {}).get("id"),
        "FTHG":       int(fthg),
        "FTAG":       int(ftag),
        "FTR":        "H" if fthg > ftag else ("A" if ftag > fthg else "D"),
        # Shot columns — None if API-Football doesn't have stats for this fixture
        "HS":  None, "AS":  None,
        "HST": None, "AST": None,
        "HC":  None, "AC":  None,
        "HF":  None, "AF":  None,
        "HY":  None, "AY":  None,
        "HR":  None, "AR":  None,
        # Declared here too, not only in the stats branch: a fixture with no statistics response
        # must still produce these keys, or the rows would have ragged columns and the parquet
        # schema would depend on which fixture happened to be first.
        "HXG": None, "AXG": None,
        "HIB": None, "AIB": None,
    }

    if stats_resp and len(stats_resp) >= 2:
        h_stats = stats_resp[0].get("statistics", [])
        a_stats = stats_resp[1].get("statistics", [])

        row["HS"]  = _stat(h_stats, "Total Shots")
        row["AS"]  = _stat(a_stats, "Total Shots")
        row["HST"] = _stat(h_stats, "Shots on Goal")
        row["AST"] = _stat(a_stats, "Shots on Goal")
        row["HC"]  = _stat(h_stats, "Corner Kicks")
        row["AC"]  = _stat(a_stats, "Corner Kicks")
        row["HF"]  = _stat(h_stats, "Fouls")
        row["AF"]  = _stat(a_stats, "Fouls")
        row["HY"]  = _stat(h_stats, "Yellow Cards")
        row["AY"]  = _stat(a_stats, "Yellow Cards")
        row["HR"]  = _stat(h_stats, "Red Cards")
        row["AR"]  = _stat(a_stats, "Red Cards")
        # xG AND INSIDE-BOX SHOTS. Both arrive in the SAME response already being fetched and
        # paid for, and were simply never read out of it. The model has declared
        # home_xg_last5 / away_xg_last5 / home_insidebox_last5 / away_insidebox_last5 in
        # FEATURE_COLS all along; verified in the live store, all four were 100% NULL — 0 of
        # 27,610 rows — because af_history.parquet carried no such columns, so data_loader's
        # merge had nothing to fill them with and the model median-imputed a constant.
        #
        # Names are exactly as the API returns them ('expected_goals' lowercase with an
        # underscore, 'Shots insidebox' with no space in "insidebox") — read off a live
        # response rather than guessed, because _stat matches the type string exactly.
        #
        # Costs nothing: no extra call, no extra credit. Rows already in the parquet keep NaN,
        # so coverage builds forward from the next af_history_extend run.
        row["HXG"] = _statf(h_stats, "expected_goals")   # float, not a count
        row["AXG"] = _statf(a_stats, "expected_goals")
        row["HIB"] = _stat(h_stats, "Shots insidebox")
        row["AIB"] = _stat(a_stats, "Shots insidebox")

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def run(target_league: str | None = None, dry_run: bool = False) -> None:
    all_rows: list[dict] = []
    total_api_calls = 0
    total_fixtures  = 0

    from src.league_registry import canonical_name
    _target = None if target_league is None else canonical_name(target_league)
    if target_league is not None and _target is None:
        print(f"  [error] --league {target_league!r} is not a registered league name or alias")
        return
    leagues_to_run = [(n, i, s) for n, i, s in league_plan() if _target is None or n == _target]

    # ── Keep the CURRENT season in scope ──────────────────────────────────────
    # The season lists above were written for the one-time pre-season backfill and stop at
    # 2024/2025. Nothing added the live season, so current-season fixtures never entered
    # af_history.parquet — which is exactly why data_loader's step 2 kept trying to fetch
    # them from the API on EVERY predict run (~3 min per league, forever).
    try:
        sys.path.insert(0, str(ROOT))
        import config as _cfg
        # League-aware season (R2 finding 6). An unresolvable league adds nothing (never a guessed year).
        from src.season_resolver import resolve_api_season, is_known
        _current = {n: (lambda s_: s_ if is_known(s_) else "")(resolve_api_season(n)) for n, _, _ in leagues_to_run}
    except Exception:
        _current = {}
    leagues_to_run = [
        (n, i, s + ([_current[n]] if _current.get(n) and _current[n] not in s else []))
        for n, i, s in leagues_to_run
    ]

    # ── Use the COMMITTED parquet as memory ───────────────────────────────────
    # scripts/af_history_cache/ is untracked, so in CI it starts empty and an uncached run
    # would re-fetch all ~20,000 historical fixtures. af_history.parquet IS committed, so
    # treat it as the record of what we already have and skip those fixtures entirely.
    # That is what makes this safe to run on a schedule: the first run collects the current
    # season, every later run collects only what is genuinely new.
    # Skip set keyed on API-Football's fixture_id where the row carries one (rows written since
    # 2026-11-01); older rows only have names, and those names are API-Football's own spelling
    # from the same endpoint, so the legacy key compares like with like.
    existing_keys: set = set()
    existing_ids: set = set()
    if OUTPUT.exists():
        try:
            _ex = pd.read_parquet(OUTPUT)
            _ex["date"] = pd.to_datetime(_ex["date"], errors="coerce")
            if "fixture_id" in _ex.columns:
                existing_ids = {int(x) for x in pd.to_numeric(_ex["fixture_id"], errors="coerce").dropna()}
                _ex = _ex[_ex["fixture_id"].isna()]
            existing_keys = {
                (str(r.league), str(r.date)[:10], str(r.home_team), str(r.away_team))
                for r in _ex.itertuples()
            }
            print(f"  af_history.parquet already holds {len(existing_ids) + len(existing_keys):,} fixtures — "
                  f"these will be skipped, not re-fetched")
        except Exception as e:
            print(f"  [warn] could not read existing af_history ({e}) — no skip list")
    skipped = 0

    for league_name, league_id, seasons in leagues_to_run:
        print(f"\n{'='*60}")
        print(f"  {league_name}  (AF id={league_id})")
        print(f"{'='*60}")

        for season in seasons:
            fixtures = fetch_season(league_name, league_id, season, dry_run)
            total_api_calls += 1  # fixtures list call

            for i, f in enumerate(fixtures):
                # Already in the committed parquet? Then we have this fixture's stats and
                # must not spend a call on it again. This is the whole point of the skip
                # list — completed results never change.
                fix_id = f["fixture"]["id"]
                _tm = f.get("teams", {}) or {}
                _key = (league_name,
                        str((f.get("fixture", {}) or {}).get("date", ""))[:10],
                        str((_tm.get("home") or {}).get("name", "")),
                        str((_tm.get("away") or {}).get("name", "")))
                if fix_id in existing_ids or _key in existing_keys:
                    skipped += 1
                    continue

                stats_resp = fetch_stats(fix_id, dry_run)
                total_api_calls += 1

                row = parse_fixture(league_name, season, f, stats_resp)
                if row:
                    all_rows.append(row)

                if (i + 1) % 50 == 0:
                    print(f"    ... {i+1}/{len(fixtures)} done  (total rows so far: {len(all_rows)})")

            total_fixtures += len(fixtures)

    print(f"\n{'='*60}")
    print(f"  Total API calls: {total_api_calls}")
    print(f"  Total fixtures:  {total_fixtures}")
    print(f"  Skipped (already in af_history.parquet): {skipped}")
    print(f"  Total rows:      {len(all_rows)}")

    if not all_rows and skipped:
        print("  Nothing new to collect — af_history.parquet is already current.")
        return

    if dry_run:
        print("  [dry-run] no data saved")
        return

    if not all_rows:
        print("  No rows collected — check API key and cache")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["league", "date"]).reset_index(drop=True)

    # Merge with existing af_history if it exists (incremental runs)
    if OUTPUT.exists():
        existing = pd.read_parquet(OUTPUT)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["league", "date", "home_team", "away_team"], keep="last")
        df = df.sort_values(["league", "date"]).reset_index(drop=True)

    df.to_parquet(OUTPUT, index=False)
    pct_with_shots = df["HS"].notna().mean() * 100
    print(f"\n  Saved {len(df)} rows → {OUTPUT.name}")
    print(f"  Shot data coverage: {pct_with_shots:.1f}%")
    print(f"\n  Next step: run pipeline.py --mode train to retrain team model")
    print(f"  Note: leagues stay as NEW_FORMAT — API-Football and FD data are kept separate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill API-Football match stats for new-format leagues")
    parser.add_argument("--dry-run", action="store_true", help="Estimate call count without hitting API")
    parser.add_argument("--league", type=str, default=None, help="Run for a single league only")
    args = parser.parse_args()

    run(target_league=args.league, dry_run=args.dry_run)
