"""
collection_complete.py - is the Data Truth real-data collection COMPLETE? (post-collection step 1,
2026-11-07). Read-only: NO API call, nothing written (except --json FILE, never under output/).

    python scripts/collection_complete.py                # table + verdict; exit 0 only when complete
    python scripts/collection_complete.py --json FILE    # also the full report as JSON

WHAT. For every collector of .github/workflows/data_truth_collect.yml (fixtures, htscores incl.
--add-ft / --add-status, cupstats, afgap, insidebox, fullstats, sidelined, profiles, injuries,
coaches, transfers, lineups, playerrefetch) it runs that collector's OWN planning function on the
committed files and reports:
  * calls_left      the API-Football calls the collector would plan now, split into
      backlog       calls that should already have been made (budget stop, failed request) - the
                    collection is not complete while any is left;
      steady_state  calls that exist only because new matches were played / a season is open / a
                    scheduled transfer-window refresh is due: they recur on every run for ever;
  * unreachable     what the collector will never get, by reason (the provider has no statistics /
                    lineups for the season, invalid keys, refused identity, fixtures not in the
                    fixture lists, empty provider answers, ...): counted, never a call.
NEW-MATCH RULE. A planned item is steady state when its match date is on or after the day before
the collector's last journaled fetch (the fixture lists are refreshed at the start of every run, so a
match of that day or later can postdate the last run); anything older is backlog.
TWO COLLECTORS ARE PLANNED OFFLINE here, because their own --dry-run calls the API:
  * fixtures  (collect_competition_fixtures.run(dry_run=True) asks /leagues per competition): read
    from its journal - every registry competition journaled, the OPEN seasons re-read each run;
  * afgap     (backfill_af_history has no call-free dry run): the league-season fixture lists it
    re-reads each run, and every FT fixture of those seasons in step 1's lists that af_history lacks.
AUDITS. The audit JSONs in output/collect are checked for internal consistency (the counts add up,
and agree with the files they describe). An inconsistent audit makes the verdict NOT complete.

EXIT 0 only when every collector has 0 backlog calls and every audit check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import COLLECT_DIR, DAILY_CAP, read_parquet  # noqa: E402
import collect_common as CC  # noqa: E402

#: the steady-state window: a match dated from the day before the collector's last fetch is "new"
NEW_MATCH_SLACK = pd.Timedelta(days=1)
STEPS = ("fixtures", "htscores", "cupstats", "afgap", "insidebox", "fullstats", "sidelined", "profiles",
         "injuries", "coaches", "transfers", "lineups", "playerrefetch")


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────

def last_fetch(journal: pd.DataFrame, col: str = "fetched_at") -> pd.Timestamp | None:
    """The newest fetch time of a journal (UTC, naive), None when the journal is empty."""
    if journal is None or journal.empty or col not in journal.columns:
        return None
    t = pd.to_datetime(journal[col], errors="coerce", utc=True).dropna()
    return None if t.empty else t.max().tz_localize(None)


def split_new(dates, last: pd.Timestamp | None) -> tuple[int, int]:
    """(steady_state, backlog) for planned items with these match dates (the NEW-MATCH RULE)."""
    d = pd.to_datetime(pd.Series(list(dates), dtype=object), errors="coerce", utc=True).dt.tz_localize(None)
    if last is None:
        return 0, int(len(d))
    new = d >= (pd.Timestamp(last).normalize() - NEW_MATCH_SLACK)
    return int(new.sum()), int((~new).sum())


def result(name: str, backlog: int, steady: int, recurring: bool, unreachable: dict | None = None,
           steady_note: str = "", detail: dict | None = None, planner: str = "") -> dict:
    return {"collector": name, "planner": planner, "calls_left": int(backlog) + int(steady),
            "backlog": int(backlog), "steady_state": int(steady), "recurring": bool(recurring),
            "steady_state_note": steady_note,
            "unreachable": {k: int(v) for k, v in (unreachable or {}).items() if v},
            "detail": detail or {}}


def _tally(d: dict, k: str) -> int:
    """One per-league REASON counter of a collector audit. The collectors count reasons in a
    defaultdict and write a reason only when it occurred, so an absent reason is a tally of none
    (a counter, not a value standing in for missing data). Totals are read with _total instead."""
    v = d.get(k)
    return int(v) if v is not None else 0


def _total(d: dict, k: str) -> int | None:
    """A TOTAL of an audit: None when the audit does not state it (a check on it then fails)."""
    v = (d or {}).get(k)
    return int(v) if v is not None else None


def _plus(*xs):
    """Sum of totals; None when any is unknown."""
    return None if any(x is None for x in xs) else sum(xs)


def _sum_per_league(per_league: dict, keys) -> Counter:
    c: Counter = Counter()
    for d in (per_league or {}).values():
        for k in keys:
            c[k] += _tally(d, k)
    return c


# ── the collectors ──────────────────────────────────────────────────────────────────────────────

def check_fixtures() -> dict:
    import collect_competition_fixtures as F
    comps = F.competitions()
    j = read_parquet(F.JOURNAL)
    have = set(pd.to_numeric(j["league_id"], errors="coerce").dropna().astype(int)) if len(j) else set()
    never = [c["league_id"] for c in comps if c["league_id"] not in have]
    open_seasons = 0
    if len(j):
        end = pd.to_datetime(j["season_end"], errors="coerce")
        at = pd.to_datetime(j["fetched_at"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        open_seasons = int((end >= at).sum())
    # every run: one /leagues per competition + one /fixtures per open season
    steady = len(comps) - len(never) + open_seasons
    return result("fixtures", backlog=len(never), steady=steady, recurring=True,
                  steady_note=f"{len(comps) - len(never)} /leagues (season list) + {open_seasons} open "
                              f"season list(s) re-read every run",
                  detail={"competitions": len(comps), "journaled_league_seasons": int(len(j)),
                          "competitions_never_journaled": never,
                          "note": "the provider season list is not stored; a competition never journaled "
                                  "is counted as backlog (it may also be an id that fails verification)"},
                  planner="offline: competitions() + competition_seasons journal")


def check_htscores() -> dict:
    import collect_ht_scores as H
    lacking = H.lacking_rows()
    journal = read_parquet(H.JOURNAL)
    todo = H.plan(lacking, journal)
    jk = {(int(r.league_id), str(r.season)) for r in journal.itertuples(index=False)} if len(journal) else set()
    re_read = sum((t["league_id"], t["season"]) in jk for t in todo)
    backlog = len(todo) - re_read
    # --add-ft: the no-call completion IN MEMORY (add_ft would write it), then the plan
    scores, n_comp = H.complete_ft_from_competition_fixtures(read_parquet(H.SCORES))
    ft_todo = H.ft_plan(scores, read_parquet(H.FT_JOURNAL))
    # --add-status --status-all-leagues
    comp = pd.read_parquet(H.COMP_FIXTURES, columns=["league_id", "league", "season", "competition_type", "date"]) \
        if H.COMP_FIXTURES.exists() else None
    sj = read_parquet(H.STATUS_JOURNAL)
    st_todo = H.status_plan(journal, read_parquet(H.FT_JOURNAL), sj,
                            comp.drop(columns="date") if comp is not None else None, all_leagues=True)
    st_new = 0
    if st_todo and comp is not None:
        first = comp.groupby(["league_id", "season"])["date"].min()
        s_new, _ = split_new([first.get((t["league_id"], t["season"])) for t in st_todo], last_fetch(sj))
        st_new = s_new
    audit = _json(COLLECT_DIR / "ht_fill_audit.json")
    unr = _sum_per_league(audit.get("per_league"), ("unmatched", "refused_ambiguous",
                                                     "refused_fixture_claimed_twice", "refused_ft_mismatch"))
    return result("htscores", backlog=backlog + len(ft_todo) + len(st_todo) - st_new,
                  steady=re_read + st_new, recurring=True,
                  steady_note="a journaled league-season with a match lacking HT dated after its fetch; "
                              "a new league-season's one-time status re-read",
                  unreachable={f"history_rows_{k}": v for k, v in unr.items()},
                  detail={"history_rows_lacking_ht": int(len(lacking)),
                          "ht_lists_to_fetch": len(todo), "ht_lists_re_read_new_matches": re_read,
                          "add_ft_completed_without_call_in_memory": int(n_comp),
                          "add_ft_lists_to_re_read": len(ft_todo),
                          "add_status_lists_to_re_read": len(st_todo),
                          "fill_rows_total_per_audit": audit.get("filled_rows_total")},
                  planner="collect_ht_scores.plan / ft_plan / status_plan(all_leagues=True)")


def check_cupstats() -> dict:
    import collect_cup_player_stats as C
    fx = read_parquet(C.FIXTURES)
    start, leagues = C._start_and_leagues()
    j = read_parquet(C.JOURNAL)
    todo = C.plan(fx, C.cup_ids(), start, j, leagues)
    meta = read_parquet(C.META)
    m_todo, pending = C.meta_plan(j, meta, fx)
    lf = last_fetch(j)
    s1, b1 = split_new(todo["date"] if len(todo) else [], lf)
    s2, b2 = split_new(m_todo["date"] if len(m_todo) else [], last_fetch(meta))
    n_new = len(todo) + len(m_todo)
    ids_calls = -(-n_new // C.IDS_PER_CALL) if n_new else 0
    empty = int((pd.to_numeric(j["n_rows"], errors="coerce") == 0).sum()) if len(j) else 0
    empty_corners = int((meta["corners_status"] == "empty").sum()) if len(meta) else 0
    # new fixtures also need their /fixtures?ids= and, at most, /fixtures/statistics
    backlog = b1 + b2 + len(pending) + (ids_calls if (b1 + b2) else 0)
    steady = s1 + s2 + (ids_calls if not (b1 + b2) else 0)
    return result("cupstats", backlog=backlog, steady=steady, recurring=True,
                  steady_note="newly finished UEFA cup fixtures (players + referee/corners)",
                  unreachable={"fixtures_provider_answered_no_player_rows": empty,
                               "fixtures_provider_statistics_empty_no_corners": empty_corners},
                  detail={"fixtures_to_fetch": int(len(todo)), "meta_to_fetch": int(len(m_todo)),
                          "corners_pending": int(len(pending)), "journaled_fixtures": int(len(j))},
                  planner="collect_cup_player_stats.plan + meta_plan (= its --dry-run)")


def check_afgap() -> dict:
    import backfill_af_history as B
    from src.season_resolver import is_known, resolve_api_season
    plan = B.league_plan()
    cur = {n: resolve_api_season(n) for n, _, _ in plan}
    plan = [(n, i, s + ([cur[n]] if is_known(cur[n]) and cur[n] not in s else [])) for n, i, s in plan]
    n_lists = sum(len(s) for _, _, s in plan)
    fx = read_parquet(ROOT / "output" / "strength" / "competition_fixtures.parquet")
    af = pd.read_parquet(B.OUTPUT) if B.OUTPUT.exists() else pd.DataFrame(columns=["league", "date"])
    ids = (set(pd.to_numeric(af["fixture_id"], errors="coerce").dropna().astype(int))
           if "fixture_id" in af.columns else set())
    ex = af[af["fixture_id"].isna()] if "fixture_id" in af.columns else af
    keys = {(str(r.league), str(r.date)[:10], str(r.home_team), str(r.away_team)) for r in ex.itertuples()}
    missing_dates, not_in_lists = [], []
    have = {(int(a), str(b)) for a, b in zip(fx["league_id"], fx["season"])} if len(fx) else set()
    for name, lid, seasons in plan:
        for s in seasons:
            if (lid, str(s)) not in have:
                not_in_lists.append(f"{name} {s}")
                continue
            f = fx[(fx["league_id"] == lid) & (fx["season"].astype(str) == str(s)) & (fx["status"] == "FT")]
            for r in f.itertuples(index=False):
                if int(r.fixture_id) in ids or (name, str(r.date)[:10], str(r.home_name), str(r.away_name)) in keys:
                    continue
                missing_dates.append(r.date)
    lf = pd.to_datetime(af["date"], errors="coerce").max() if len(af) else None
    s, b = split_new(missing_dates, lf)
    return result("afgap", backlog=b, steady=n_lists + s, recurring=True,
                  steady_note=f"{n_lists} league-season fixture list(s) re-read every run (6-hour cache) "
                              f"+ statistics of newly finished fixtures",
                  detail={"leagues": len(plan), "league_season_lists": n_lists,
                          "ft_fixtures_in_lists_not_in_af_history": len(missing_dates),
                          "planned_seasons_not_in_step1_lists": not_in_lists,
                          "note": "statistics already in the CI cache (scripts/af_history_cache) cost no "
                                  "call; the cache is not on this machine"},
                  planner="offline: backfill_af_history.league_plan + step 1 fixture lists vs af_history")


def check_insidebox() -> dict:
    import collect_af_history_gap as G
    st = G.run_insidebox(dry_run=True)                     # the collector's own plan (prints its JSON)
    # the same selection, per fixture (cross-checked against the collector's own counts below)
    lacking = G.stats_targets()
    m, per_league = G.match_fixtures(lacking, G.fixture_index())
    store = read_parquet(G.STATS_STORE)
    full = read_parquet(G.FULL_FILL)
    cov = read_parquet(G.COVERAGE)
    have = set(pd.to_numeric(store["fixture_id"], errors="coerce").dropna().astype(int)) if len(store) else set()
    nt = (store.drop_duplicates("fixture_id", keep="last").assign(
        fixture_id=lambda x: pd.to_numeric(x["fixture_id"], errors="coerce")).set_index("fixture_id")["n_teams"]
        if len(store) else pd.Series(dtype=float))
    empty = set(nt[pd.to_numeric(nt, errors="coerce") == 0].index.astype(int))
    in_full = set(pd.to_numeric(full["fixture_id"], errors="coerce").dropna().astype(int)) if len(full) else set()
    ok = {(int(r.league_id), str(r.season)): r.statistics_fixtures for r in cov.itertuples()} if len(cov) else {}
    why = lacking.assign(_ds=lacking["date"].dt.strftime("%Y-%m-%d"))[
        ["league", "_ds", "home_team", "away_team", "reason"]].drop_duplicates(["league", "_ds", "home_team", "away_team"])
    if len(m):
        m = m.assign(_ds=pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")).merge(
            why, on=["league", "_ds", "home_team", "away_team"], how="left")
        m["reason"] = m["reason"].fillna("insidebox")
    else:
        m["reason"] = pd.Series(dtype=object)
    by_fix = m.groupby("fixture_id")["reason"].agg(lambda s: "+".join(sorted(set("+".join(s).split("+")))))
    todo = m.drop_duplicates("fixture_id").copy()
    todo["reason"] = todo["fixture_id"].map(by_fix).astype(str)
    wants = ((todo["reason"].str.contains("insidebox") & ~todo["fixture_id"].isin(have) & ~todo["fixture_id"].isin(in_full))
             | (todo["reason"].str.contains("served_stats") & ~todo["fixture_id"].isin(in_full)
                & ~todo["fixture_id"].isin(empty)))
    todo = todo[wants]
    flag = [ok.get((int(a), str(b))) for a, b in zip(todo["league_id"], todo["season"])]
    no_stats = pd.Series([f is not None and f == f and not bool(f) for f in flag], index=todo.index)
    unknown = pd.Series([f is None or f != f for f in flag], index=todo.index)
    answered = todo["fixture_id"].isin(have)
    callable_ = todo[~no_stats & ~answered]
    s, b = split_new(callable_["date"], last_fetch(store))
    checks = {"fixtures_to_read": (int(len(todo)), st["fixtures_to_read"]),
              "no_provider_statistics": (int(no_stats.sum()), st["fixtures_skipped_no_provider_statistics"])}
    unr_rows = _sum_per_league(per_league, ("unmatched", "refused_ambiguous", "refused_fixture_claimed_twice"))
    n_empty_targets = int(m.drop_duplicates("fixture_id")["fixture_id"].isin(empty).sum()) if len(m) else 0
    return result("insidebox", backlog=b + st["planned_coverage_calls"], steady=s, recurring=True,
                  steady_note="newly finished matches whose statistics are not read yet",
                  unreachable={"fixtures_season_without_provider_statistics": int(no_stats.sum()),
                               "fixtures_provider_answer_empty": n_empty_targets,
                               "fixtures_answered_but_not_placed_by_fullstats": int((answered & ~no_stats).sum()),
                               **{f"history_rows_{k}": v for k, v in unr_rows.items()}},
                  detail={"collector_plan": {k: st[k] for k in (
                              "lacking_rows", "matched_rows", "fixtures_to_read", "fixtures_in_local_cache",
                              "fixtures_skipped_no_provider_statistics", "planned_coverage_calls",
                              "planned_stats_calls_excluding_no_statistics_seasons")},
                          "coverage_unknown_fixtures": int(unknown.sum()),
                          "cross_check_with_collector": {k: {"here": a, "collector": c, "equal": a == c}
                                                          for k, (a, c) in checks.items()},
                          "answered_not_placed_note": "the answer is journaled (output/collect/"
                              "af_fixture_stats.parquet) and cached in CI, so it costs no call; --fullstats "
                              "refuses to place it (it matches on the WHOLE af_history, where another row of "
                              "a different date claims the same fixture)"},
                  planner="collect_af_history_gap.run_insidebox(dry_run=True)")


def check_fullstats() -> dict:
    a = _json(COLLECT_DIR / "af_fullstats_audit.json")
    c = _sum_per_league(a.get("per_league"), ("not_cached", "teams_not_in_answer", "empty_answer", "unmatched",
                                              "refused_ambiguous", "refused_fixture_claimed_twice"))
    return result("fullstats", backlog=0, steady=0, recurring=True,
                  steady_note="0 API calls by design: re-parses every cached answer each run",
                  unreachable={f"history_rows_{k}": v for k, v in c.items()},
                  detail={k: a.get(k) for k in ("history_rows", "matched_rows", "filled_rows_total", "written_at")},
                  planner="af_fullstats_audit.json (the step makes no call; running it would rewrite the sidecar)")


def check_sidelined() -> dict:
    import collect_sidelined as S
    h = CC.history_frame(["player_id", "date"])
    have = read_parquet(S.PLAYERS)
    todo = S.todo_players(h, have)
    hp = h.assign(player_id=pd.to_numeric(h["player_id"], errors="coerce"))
    first = hp.groupby("player_id")["date"].min()
    s, b = split_new([first.get(p) for p in todo], last_fetch(have))
    return result("sidelined", backlog=b, steady=s, recurring=True,
                  steady_note="players new to player_history",
                  unreachable={"history_rows_player_id_not_a_provider_id": S.invalid_rows(h)},
                  detail={"players_to_ask": len(todo), "players_journaled": int(len(have))},
                  planner="collect_sidelined.todo_players (= its --dry-run)")


def check_profiles() -> dict:
    import collect_player_profiles as P
    p = P.plan_keys()
    kf = p["keys"]
    have = read_parquet(P.OUT)
    dates = []
    if p["todo"] and len(kf):
        ok = kf[kf["problem"].isna()].assign(_k=lambda x: list(zip(x["player_id"].astype(int), x["season_key"].astype(str))))
        first = ok.groupby("_k")["date"].min()
        dates = [first.get(k) for k in p["todo"]]
    s, b = split_new(dates, last_fetch(have))
    return result("profiles", backlog=b, steady=s, recurring=True,
                  steady_note="(player, season) pairs new to player_history",
                  unreachable={f"pairs_{k}": v for k, v in p["skipped"].items()},
                  detail={"pairs_to_ask": len(p["todo"]), "history_rows_behind_skipped": p["skipped_rows"],
                          "profiles_stored": int(len(have))},
                  planner="collect_player_profiles.plan_keys (= its --dry-run)")


def check_injuries() -> dict:
    import collect_af_lists as AL
    import collect_injuries as I
    p = I.plan()
    sj = read_parquet(I.SEASON_JOURNAL)
    stored = {(int(r.league_id), str(r.season)) for r in sj.itertuples(index=False)} if len(sj) else set()
    rf = AL.registry_fixtures()
    first = rf.groupby(["league_id", "season"])["date"].min() if len(rf) else pd.Series(dtype=object)
    regrown = [r for r in p["seasons"] if (int(r.league_id), str(r.season)) in stored]
    never = [r for r in p["seasons"] if (int(r.league_id), str(r.season)) not in stored]
    s_new, b_never = split_new([first.get((int(r.league_id), str(r.season))) for r in never], last_fetch(sj))
    cov_new = 0
    if p["coverage"]:
        cov = read_parquet(AL.COVERAGE)
        lseas = p["league_seasons"]
        have = AL.done_keys(AL.COVERAGE, ["league_id", "season"])
        miss = [(int(a), str(b)) for a, b in zip(lseas["league_id"], lseas["season"]) if (str(a), str(b)) not in have]
        s_c, _ = split_new([first.get(k) for k in miss], last_fetch(cov))
        cov_new = len(p["coverage"]) if s_c == len(miss) else 0
    empty = int((sj["status"] == "empty").sum()) if len(sj) else 0
    return result("injuries", backlog=b_never + len(p["fallback"]) + len(p["coverage"]) - cov_new,
                  steady=len(regrown) + s_new + cov_new, recurring=True,
                  steady_note="open league-seasons with newly finished fixtures (1 call each) and new seasons",
                  unreachable={"league_seasons_provider_answer_empty": empty},
                  detail={"league_seasons": int(len(p["league_seasons"])), "journaled": int(len(sj)),
                          "status": sj["status"].value_counts().to_dict() if len(sj) else {},
                          "fixture_fallback_calls": int(len(p["fallback"]))},
                  planner="collect_injuries.plan (= its --dry-run)")


def _teams_check(name: str, module, n_key: str) -> dict:
    import collect_af_lists as AL
    todo = module.todo_teams()
    j = read_parquet(module.JOURNAL)
    fx = AL.fixture_list()
    t = pd.concat([fx[["home_id", "date"]].rename(columns={"home_id": "team_id"}),
                   fx[["away_id", "date"]].rename(columns={"away_id": "team_id"})], ignore_index=True)
    first = t.groupby("team_id")["date"].min()
    init = [tid for tid, w in todo if w == AL.INITIAL]
    refresh = [tid for tid, w in todo if w != AL.INITIAL]
    s, b = split_new([first.get(tid) for tid in init], last_fetch(j))
    empty = int((pd.to_numeric(j[n_key], errors="coerce") == 0).sum()) if len(j) else 0
    return result(name, backlog=b, steady=s + len(refresh), recurring=True,
                  steady_note="teams new to the fixture lists + the scheduled refresh after each "
                              "transfer-window close (2 Sep / 3 Feb)",
                  unreachable={"journal_rows_provider_answer_empty": empty},
                  detail={"teams_in_fixture_lists": len(AL.team_ids(fx)), "initial_due": len(init),
                          "refresh_due": len(refresh), "journal_rows": int(len(j))},
                  planner=f"collect_{name}.todo_teams (= its --dry-run)")


def check_coaches() -> dict:
    import collect_coaches as M
    return _teams_check("coaches", M, "n_coaches")


def check_transfers() -> dict:
    import collect_transfers as M
    return _teams_check("transfers", M, "n_players")


def check_lineups() -> dict:
    import collect_af_lists as AL
    import collect_lineups as L
    p = L.plan()
    cov = AL.coverage_map("lineups")
    todo = p["todo"]
    flag = pd.Series([cov.get((int(a), str(b))) for a, b in zip(todo["league_id"], todo["season"])],
                     index=todo.index, dtype=object)
    no_cov = flag.map(lambda v: v is False)
    unknown = flag.map(lambda v: v is None)
    j = read_parquet(L.JOURNAL)
    s, b = split_new(todo.loc[~no_cov, "date"], last_fetch(j))
    empty = int((pd.to_numeric(j["n_teams"], errors="coerce") == 0).sum()) if len(j) else 0
    return result("lineups", backlog=b + len(p["coverage"]), steady=s, recurring=True,
                  steady_note="newly finished registry fixtures",
                  unreachable={"fixtures_season_without_provider_lineups": int(no_cov.sum()),
                               "fixtures_provider_answer_empty": empty},
                  detail={"registry_finished_fixtures": p["registry_fixtures"], "journaled": int(len(j)),
                          "to_fetch": int((~no_cov).sum()), "coverage_unknown_fixtures": int(unknown.sum()),
                          "to_fetch_oldest": str(todo.loc[~no_cov, "date"].min()) if (~no_cov).any() else None},
                  planner="collect_lineups.plan (= its --dry-run)")


def check_playerrefetch() -> dict:
    import collect_player_refetch as R
    p = R.plan()
    j = read_parquet(R.JOURNAL)
    a = _json(R.AUDIT)
    refused_in_answers = int(pd.to_numeric(j["n_refused"], errors="coerce").sum()) if len(j) else 0
    unr = {f"fixtures_{k}": v for k, v in p["refused_fixtures"].items()}
    unr.update({f"rows_{k}": v for k, v in p["refused_rows"].items()})
    unr["rows_refused_in_answers"] = refused_in_answers
    unr["fixtures_provider_answer_empty"] = int((j["status"] == "empty").sum()) if len(j) else 0
    return result("playerrefetch", backlog=p["planned_calls"], steady=0, recurring=False,
                  steady_note="none: rows written since fix round 9 are verified at parse (provider_rule)",
                  unreachable=unr,
                  detail={k: p[k] for k in ("rows", "rows_verified_at_parse", "rows_to_verify", "fixtures",
                                            "proven_fixtures", "journaled_fixtures", "planned_calls")}
                  | {"rows_stored": a.get("stored_rows"),
                     "rows_refused_last_run_by_reason": (a.get("run") or {}).get("rows_refused")},
                  planner="collect_player_refetch.plan (= its --dry-run)")


CHECKS = {"fixtures": check_fixtures, "htscores": check_htscores, "cupstats": check_cupstats,
          "afgap": check_afgap, "insidebox": check_insidebox, "fullstats": check_fullstats,
          "sidelined": check_sidelined, "profiles": check_profiles, "injuries": check_injuries,
          "coaches": check_coaches, "transfers": check_transfers, "lineups": check_lineups,
          "playerrefetch": check_playerrefetch}


# ── audit consistency ───────────────────────────────────────────────────────────────────────────

def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _chk(out: list, file: str, name: str, ok: bool, detail="") -> None:
    out.append({"file": file, "check": name, "ok": bool(ok), "detail": detail})


def _rows(path: Path) -> int | None:
    if not Path(path).exists():
        return None
    import pyarrow.parquet as pq
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def audit_checks(d: Path = COLLECT_DIR) -> list[dict]:
    """Internal consistency of the audit JSONs in output/collect (and of each against its file)."""
    d = Path(d)
    out: list[dict] = []
    refused = ("unmatched", "refused_ambiguous", "refused_fixture_claimed_twice")

    b = _json(d / "af_budget.json")
    if b:
        over = {k: v for k, v in b.items() if int(v) > CC._CAP_MAX}
        _chk(out, "af_budget.json", f"no day above the hard override ceiling {CC._CAP_MAX:,}", not over, over)
        _chk(out, "af_budget.json", "every day a non-negative count",
             all(isinstance(v, int) and v >= 0 for v in b.values()))
        override = {k: v for k, v in b.items() if int(v) > DAILY_CAP}
        if override:
            _chk(out, "af_budget.json", f"days above the default cap {DAILY_CAP:,} (a COLLECT_AF_DAILY_CAP "
                                        f"override day, allowed up to {CC._CAP_MAX:,})", True, override)

    a = _json(d / "ht_fill_audit.json")
    if a:
        pl = a.get("per_league") or {}
        bad = {lg: v for lg, v in pl.items()
               if _tally(v, "lacking") != sum(_tally(v, k) for k in
                                              ("filled", "refused_ft_mismatch", "no_fixture_stored") + refused)}
        _chk(out, "ht_fill_audit.json", "per league: lacking = filled + unmatched + refused (ambiguous, "
                                        "claimed twice, FT mismatch)", not bad, bad)
        n_mm = sum(_tally(v, "refused_ft_mismatch") for v in pl.values())
        csv = d / "ht_fill_ft_mismatch.csv"
        n_csv = len(pd.read_csv(csv)) if csv.exists() else None
        _chk(out, "ht_fill_audit.json", "refused_ft_mismatch = per-league sum = rows of ht_fill_ft_mismatch.csv",
             _total(a, "refused_ft_mismatch") == n_mm == n_csv, {"audit": a.get("refused_ft_mismatch"),
                                                                 "per_league": n_mm, "csv": n_csv})
        n_fill = _rows(d / "fd_history_ht_fill.parquet")
        _chk(out, "ht_fill_audit.json", "filled_rows_total = rows of fd_history_ht_fill.parquet",
             n_fill is not None and _total(a, "filled_rows_total") == n_fill,
             {"audit": a.get("filled_rows_total"), "file": n_fill})
        n_filled = sum(_tally(v, "filled") for v in pl.values())
        tot = _total(a, "filled_rows_total")
        _chk(out, "ht_fill_audit.json", "rows filled this run <= filled_rows_total",
             tot is not None and n_filled <= tot, {"filled": n_filled})

    a = _json(d / "af_stats_fill_audit.json")
    if a:
        pl = a.get("per_league") or {}
        _chk(out, "af_stats_fill_audit.json", "target_rows_by_reason sums to lacking_rows",
             sum((a.get("target_rows_by_reason") or {}).values()) == _total(a, "lacking_rows"))
        _chk(out, "af_stats_fill_audit.json", "fixtures_to_read_by_reason sums to fixtures_to_read",
             sum((a.get("fixtures_to_read_by_reason") or {}).values()) == _total(a, "fixtures_to_read"))
        c = _sum_per_league(pl, ("lacking", "matched", "filled_total") + refused)
        _chk(out, "af_stats_fill_audit.json", "per-league lacking / matched sum to the totals",
             c["lacking"] == _total(a, "lacking_rows") and c["matched"] == _total(a, "matched_rows"),
             {"lacking": c["lacking"], "matched": c["matched"]})
        bad = {lg: v for lg, v in pl.items() if "lacking" in v and _tally(v, "lacking") !=
               _tally(v, "matched") + sum(_tally(v, k) for k in refused)}
        _chk(out, "af_stats_fill_audit.json", "per league: lacking = matched + unmatched + refused", not bad, bad)
        to_read = _total(a, "fixtures_to_read")
        cached = _total(a, "fixtures_in_local_cache")
        _chk(out, "af_stats_fill_audit.json", "planned_stats_calls_upper_bound = fixtures_to_read - in local cache",
             None not in (to_read, cached) and _total(a, "planned_stats_calls_upper_bound") == to_read - cached)
        lhs = _plus(_total(a, "fixtures_skipped_no_provider_statistics"),
                    _total(a, "planned_stats_calls_excluding_no_statistics_seasons"))
        _chk(out, "af_stats_fill_audit.json", "no-statistics fixtures + planned calls <= fixtures_to_read",
             None not in (lhs, to_read) and lhs <= to_read)
        n = _rows(d / "af_history_stats_fill.parquet")
        _chk(out, "af_stats_fill_audit.json", "filled_rows_total = file rows = per-league filled_total sum",
             a.get("filled_rows_total") == n == c["filled_total"],
             {"audit": a.get("filled_rows_total"), "file": n, "per_league": c["filled_total"]})

    a = _json(d / "af_fullstats_audit.json")
    if a:
        pl = a.get("per_league") or {}
        c = _sum_per_league(pl, ("lacking", "matched", "filled", "not_cached", "teams_not_in_answer",
                                 "empty_answer") + refused)
        _chk(out, "af_fullstats_audit.json", "per-league rows / matched / filled sum to the totals",
             c["lacking"] == _total(a, "history_rows") and c["matched"] == _total(a, "matched_rows")
             and c["filled"] == _total(a, "filled_rows_this_run"),
             {k: c[k] for k in ("lacking", "matched", "filled")})
        bad = {lg: v for lg, v in pl.items()
               if _tally(v, "lacking") != _tally(v, "matched") + sum(_tally(v, k) for k in refused)
               or _tally(v, "matched") != sum(_tally(v, k) for k in
                                              ("filled", "not_cached", "teams_not_in_answer", "empty_answer"))}
        _chk(out, "af_fullstats_audit.json", "per league: rows = matched + refused; matched = filled + not cached "
                                             "+ teams not in answer + empty answer", not bad, bad)
        n = _rows(d / "af_history_fullstats_fill.parquet")
        _chk(out, "af_fullstats_audit.json", "filled_rows_total = rows of af_history_fullstats_fill.parquet",
             a.get("filled_rows_total") == n, {"audit": a.get("filled_rows_total"), "file": n})
        _chk(out, "af_fullstats_audit.json", "calls = 0", a.get("calls") == 0)

    a = _json(d / "player_history_refetch_audit.json")
    if a:
        p = a.get("plan") or {}
        _chk(out, "player_history_refetch_audit.json", "plan: rows = sum of rows_by_source = verified + to verify",
             _total(p, "rows") is not None and _total(p, "rows") == sum((p.get("rows_by_source") or {}).values())
             == _plus(_total(p, "rows_verified_at_parse"), _total(p, "rows_to_verify")))
        proven = _total(p, "proven_fixtures")
        _chk(out, "player_history_refetch_audit.json", "plan: fixtures = proven + refused; planned = proven - journaled",
             proven is not None and _total(p, "fixtures") == proven + sum((p.get("refused_fixtures") or {}).values())
             and _total(p, "journaled_fixtures") is not None
             and _total(p, "planned_calls") == proven - _total(p, "journaled_fixtures"))
        rf = read_parquet(d / "player_history_refetch.parquet")
        j = read_parquet(d / "player_refetch_fixtures.parquet")
        by_src = rf["source"].value_counts().to_dict() if len(rf) else {}
        _chk(out, "player_history_refetch_audit.json", "stored_rows (and by source) = the sidecar's rows",
             a.get("stored_rows") == len(rf) and a.get("stored_rows_by_source") == by_src,
             {"audit": a.get("stored_rows"), "file": len(rf)})
        jn = int(pd.to_numeric(j["n_rows"], errors="coerce").sum()) if len(j) else 0
        jr = int(pd.to_numeric(j["n_refused"], errors="coerce").sum()) if len(j) else 0
        _chk(out, "player_history_refetch_audit.json", "journal: fixtures, status, n_rows = stored rows, "
                                                       "n_refused = rows_refused_in_answers",
             a.get("journal_fixtures") == len(j) and jn == len(rf) and a.get("rows_refused_in_answers") == jr
             and a.get("journal_by_status") == (j["status"].value_counts().to_dict() if len(j) else {}),
             {"journal_n_rows": jn, "journal_n_refused": jr})
        run = a.get("run") or {}
        _chk(out, "player_history_refetch_audit.json", "last run: refused by reason <= refused in the journal; "
                                                       "rows stored <= sidecar rows",
             sum((run.get("rows_refused") or {}).values()) <= jr
             and _total(run, "rows_stored") is not None and _total(run, "rows_stored") <= len(rf))
        played = int((pd.to_numeric(rf["minutes_refetch"], errors="coerce") > 0).sum()) if len(rf) else 0
        bad = {}
        for c, f in (a.get("fields") or {}).items():
            rows = _total(f, "rows")
            parts = ("sent_value", "sent_zero", "null_read_as_zero", "missing")
            if all(k in f for k in parts) and sum(int(f[k]) for k in parts) != rows:
                bad[c] = "sent_value + sent_zero + null_read_as_zero + missing != rows"
            if rows != played:
                bad[c] = f"rows {rows} != played rows in the sidecar {played}"
            oz = _plus(*(_total(f, k) for k in ("orig_zero_now_zero", "orig_zero_now_missing", "orig_zero_now_value")))
            if oz is None or rows is None or oz > rows:
                bad[c] = "orig_zero_* missing or exceed rows"
        _chk(out, "player_history_refetch_audit.json", "per field: the provider-behaviour counts add up to the "
                                                       "played rows", not bad, bad)

    # lineups: the journal's team counts against the stored partitions
    j = read_parquet(d / "lineups_fixtures.parquet")
    part = d / "lineups"
    if len(j) and part.exists():
        n_teams = sum(_rows(p) for p in part.glob("teams_*.parquet"))     # glob: every file exists
        _chk(out, "lineups_fixtures.parquet", "sum of journal n_teams = rows of lineups/teams_*.parquet",
             int(pd.to_numeric(j["n_teams"], errors="coerce").sum()) == n_teams,
             {"journal": int(pd.to_numeric(j["n_teams"], errors="coerce").sum()), "partitions": n_teams})
    return out


# ── verdict ─────────────────────────────────────────────────────────────────────────────────────

def verdict(results: list[dict], audits: list[dict]) -> dict:
    backlog = {r["collector"]: r["backlog"] for r in results if r["backlog"]}
    errors = [r["collector"] for r in results if r.get("error")]
    failed = [f"{a['file']}: {a['check']}" for a in audits if not a["ok"]]
    complete = not backlog and not failed and not errors
    return {"complete": complete, "backlog_calls": sum(backlog.values()), "backlog_by_collector": backlog,
            "steady_state_calls_per_run_now": sum(r["steady_state"] for r in results),
            "audit_checks": len(audits), "audit_failures": failed, "collector_errors": errors}


def run(steps=STEPS) -> dict:
    results = []
    for s in steps:
        try:
            results.append(CHECKS[s]())
        except Exception as e:                                             # noqa: BLE001
            results.append(result(s, 0, 0, False) | {"error": f"{type(e).__name__}: {e}"})
    audits = audit_checks()
    return {"generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_calls_made": 0, "collectors": results, "audits": audits, "verdict": verdict(results, audits)}


def render(rep: dict) -> str:
    lines = ["", f"{'collector':<14}{'backlog':>9}{'steady':>9}  recurring  unreachable"]
    for r in rep["collectors"]:
        unr = ", ".join(f"{k} {v:,}" for k, v in r["unreachable"].items()) or "-"
        lines.append(f"{r['collector']:<14}{r['backlog']:>9,}{r['steady_state']:>9,}  "
                     f"{'yes' if r['recurring'] else 'no ':<9}  {unr}" + (f"  ERROR {r['error']}" if r.get("error") else ""))
    lines.append("")
    for a in rep["audits"]:
        lines.append(f"[{'ok ' if a['ok'] else 'BAD'}] {a['file']}: {a['check']}"
                     + ("" if a["ok"] and not a["detail"] else f"  {json.dumps(a['detail'], default=str)}"))
    v = rep["verdict"]
    lines.append("")
    lines.append(f"VERDICT: {'COLLECTION COMPLETE' if v['complete'] else 'NOT COMPLETE'} - backlog "
                 f"{v['backlog_calls']:,} call(s) {v['backlog_by_collector']}; steady state now "
                 f"{v['steady_state_calls_per_run_now']:,} call(s) per run; audit failures "
                 f"{len(v['audit_failures'])}; 0 API calls made")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="is the Data Truth collection complete? (read-only, no API call)")
    ap.add_argument("--json", help="write the full report here (never under output/)")
    ap.add_argument("--steps", default=",".join(STEPS), help="comma list of collectors to check")
    a = ap.parse_args(argv)
    if a.json and (ROOT / "output").resolve() in Path(a.json).resolve().parents:
        raise SystemExit("refusing to write under output/ (this check is read-only)")
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in CHECKS]
    if unknown:
        raise SystemExit(f"unknown collector(s) {unknown}; known: {list(CHECKS)}")
    rep = run(steps)
    print(render(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    return 0 if rep["verdict"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
