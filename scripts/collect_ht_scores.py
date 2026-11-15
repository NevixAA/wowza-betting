"""
collect_ht_scores.py - half-time scores for the new-format leagues (Data Truth round 2, HT data gap).

    python scripts/collect_ht_scores.py            # fetch what is missing, then match
    python scripts/collect_ht_scores.py --dry-run  # print the plan (league-seasons, calls); no call
    python scripts/collect_ht_scores.py --match-only  # re-match the stored fixtures; no call

THE GAP. output/fd_history.parquet (football-data.co.uk) carries ht_home_goals / ht_away_goals for
only 5-27% of the new-format leagues' matches (measured 2026-11-02 on 42,084 rows of the 15
registry `model_type: new_format` leagues) and for 0 of the 2,188 played since 2026-07-01.

WHAT. API-Football /fixtures?league&season (ONE call per league-season) returns every fixture with
`score.halftime`. For every new-format league-season that has a history match WITHOUT a half-time
score (league from config/leagues.json, season from src.season_resolver on that match's date):
  1. fetch the season once -> output/collect/af_ht_scores.parquet: one row per FINISHED fixture
     (FT/AET/PEN) with its API-Football fixture id, club ids/names and score.halftime. A null
     half-time score is stored as missing, never 0.
  2. match each history row lacking a half-time score to ONE fixture: same league, the history's
     clubs resolved onto the fixture's clubs with src.team_names (resolve, then the API-Football
     result-spelling aliases) home->home and away->away, on the same date; only when the same date
     has no candidate, on the adjacent dates (kick-off in UTC vs the local date). More than one
     candidate, or one fixture claimed by history rows of DIFFERENT dates -> refused (counted,
     nothing written). History rows of the same date that resolve onto the same fixture are one
     match under two club spellings (the history repeats such matches) and each gets the score.

OUTPUT (beside the original - fd_history.parquet is NEVER rewritten):
  output/collect/fd_history_ht_fill.parquet  league, date, home_team, away_team (the history's own
      spellings), HTHG, HTAG (the provider's half-time score), fixture_id, af_date, match_rule,
      source, matched_at - the same key + HTHG/HTAG layout as af_ht_history.parquet.
  output/collect/ht_fill_audit.json          per league: rows lacking HT, filled, refused (why),
      plus the calls made. Journal output/collect/ht_seasons.parquet: a fetched league-season is not
      fetched again unless the history has a match without HT dated after that fetch.

BUDGET. Every call goes through collect_common.AFClient (45,000 calls per run-day for the whole
collection job, stop below 25,000 remaining). A budget stop is a clean, resumable exit.

FULL-TIME SCORE BESIDE THE HALF-TIME SCORE (owner decision 2026-11-04, "yes"). The Phase 2 HT repair
could check a fill row against the provider's full-time score for only 1,489 of 33,259 rows, because
af_ht_scores stored score.halftime and dropped score.fulltime from the very same response. Now:
  * parse() keeps score.fulltime (ft_home / ft_away; missing when the provider gives none, never 0);
  * stored fixtures that predate this are completed first from output/strength/
    competition_fixtures.parquet - the SAME provider field (score.fulltime of /fixtures?league&season)
    for the SAME fixture id, collected by step 1, no call - and, with --add-ft, by re-reading the
    league-season fixture list for whatever is still lacking (one call per league-season, about 120
    at most; resumable: journal output/collect/ht_ft_seasons.parquet, budget-guarded by AFClient);
  * every fill row carries provider_ft_home / provider_ft_away and ft_check. A row whose provider FT
    differs from the history's FT is REFUSED (no HT written) and listed in
    output/collect/ht_fill_ft_mismatch.csv with the provider status - a different full-time score
    means the pairing is not proven, or the history holds an extra-time score (an AET/PEN final).

HISTORY CORRECTIONS (owner decisions 2026-11-04). History rows listed in output/corrections/
history_removed_duplicates.csv (the same match filed twice one day apart) and
history_excluded_cup_rows.csv (proven domestic-cup fixtures filed under the league) are not matched:
they are not league matches of the history the loader serves (src/history_corrections.py). Without
this the surviving copy of a duplicate kept being refused as "claimed by two dates".

90-MINUTE SCORE (owner rule "settlement is on the 90-minute score", fix8/history2 2026-11-04). The
FT check above refused 4 fill rows of knockout / play-off matches (Romania 2023-05-26 and 06-01,
Liga MX 2023-05-29, MLS Cup 2022-11-05) because the history holds the after-extra-time score. The
history score of a proven AET/PEN fixture is corrected BESIDE the history
(output/corrections/history_score_90min.csv, built by scripts/build_history_corrections.py) and
lacking_rows() reads the corrected score, so those rows now pass the FT check against the provider's
score.fulltime. parse() now also keeps score.extratime / score.penalty (et_* / pen_*; missing when
the provider gives none) - the proof that a history score is 90' + extra time - and keeps AWARDED
fixtures (AWD / WO) with their status: they are never matched for a half-time score (claims() takes
FT/AET/PEN only) but scripts/build_history_corrections.py flags a history row proven to be one, since
an awarded result is not a played score.

STATUS RE-READ (fix9/defaults, owner 2026-11-04). The 120 league-season lists stored on 2026-11-02 were
parsed before extra time / penalties / awarded fixtures were kept, so no awarded (AWD / WO) or
abandoned (ABD) match could be detected in that history. --add-status re-reads each stored
league-season list ONCE (journal output/collect/ht_status_seasons.parquet; resumable, budget-guarded by
AFClient; about 120 calls - `--add-status --dry-run` prints the count) and stores score.extratime,
score.penalty and the statuses AWD / WO / ABD. parse() now keeps ABD too; an abandoned fixture is,
like an awarded one, never a source of a half-time or full-time score (claims() takes FT/AET/PEN only).
--status-all-leagues extends the re-read to every other registry league-season of step 1's fixture
lists (output/strength/competition_fixtures.parquet), so the whole history can be checked.
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

from collect_common import COLLECT_DIR, AFClient, BudgetStop, append_parquet, read_parquet  # noqa: E402

HISTORY = ROOT / "output" / "fd_history.parquet"
SCORES = COLLECT_DIR / "af_ht_scores.parquet"
FILL = COLLECT_DIR / "fd_history_ht_fill.parquet"
AUDIT = COLLECT_DIR / "ht_fill_audit.json"
JOURNAL = COLLECT_DIR / "ht_seasons.parquet"
FT_JOURNAL = COLLECT_DIR / "ht_ft_seasons.parquet"
FT_MISMATCH = COLLECT_DIR / "ht_fill_ft_mismatch.csv"
#: step 1's league fixture lists (scripts/collect_competition_fixtures.py): score.fulltime by fixture id
COMP_FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
FINISHED = {"FT", "AET", "PEN"}
#: the provider's "technical loss" / "walk over": a result decided off the pitch, not a played score
AWARDED = {"AWD", "WO"}
#: abandoned and not resumed: whatever score stood when play stopped is not a played result
ABANDONED = {"ABD"}
#: kept by parse() beside the finished fixtures, never matched for a score
NOT_PLAYED = AWARDED | ABANDONED
STATUS_JOURNAL = COLLECT_DIR / "ht_status_seasons.parquet"
STATUS_JOURNAL_COLS = ["league_id", "season", "league", "scope", "n_fixtures", "n_kept", "n_awarded",
                       "n_abandoned", "n_extratime", "fetched_at"]
SOURCE = "api-football /fixtures score.halftime"
SCORE_COLS = ["fixture_id", "league", "league_id", "season", "date", "kickoff", "home_id", "home_name",
              "away_id", "away_name", "ht_home", "ht_away", "status", "ft_home", "ft_away", "ft_source",
              "et_home", "et_away", "pen_home", "pen_away"]
FILL_COLS = ["league", "date", "home_team", "away_team", "HTHG", "HTAG", "fixture_id", "af_date",
             "match_rule", "source", "matched_at", "provider_ft_home", "provider_ft_away", "ft_check"]
JOURNAL_COLS = ["league_id", "season", "league", "n_fixtures", "n_finished", "n_with_ht", "fetched_at"]
FT_JOURNAL_COLS = ["league_id", "season", "league", "n_finished", "n_with_ft", "fetched_at"]
FT_SRC_FETCH = "api-football /fixtures score.fulltime"
FT_SRC_COMP = "api-football /fixtures score.fulltime (competition_fixtures, same fixture id)"


def new_format_leagues() -> dict[str, int]:
    """{registry name: api_football_id} for every `model_type: new_format` league with an id."""
    from src import league_registry as LR
    return {e["name"]: int(e["api_football_id"]) for e in LR.leagues()
            if e.get("model_type") == "new_format" and e.get("api_football_id") is not None}


def lacking_rows(history: pd.DataFrame | None = None, corrections: bool = True) -> pd.DataFrame:
    """History rows of the new-format leagues without a half-time score, with canonical league,
    API-Football league id and API season (src.season_resolver). Unknown season -> dropped, logged.
    `corrections`: drop the rows the owner's history corrections remove (src.history_corrections);
    False only for scripts/build_history_corrections.py, which DERIVES those lists from the raw rows."""
    from src import league_registry as LR
    from src.season_resolver import is_known, resolve_api_season
    if history is None:
        history = pd.read_parquet(HISTORY, columns=["league", "date", "home_team", "away_team",
                                                    "home_goals", "away_goals",
                                                    "ht_home_goals", "ht_away_goals"])
    if corrections:
        from src.history_corrections import drop_corrected_rows
        history = drop_corrected_rows(history, tag="ht")
    nf = new_format_leagues()
    h = history.copy()
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].isin(nf)]
    h = h[h["ht_home_goals"].isna() | h["ht_away_goals"].isna()].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h[h["date"].notna() & h["home_team"].notna() & h["away_team"].notna()]
    h["league_id"] = h["canon"].map(nf)
    h["season"] = [resolve_api_season(c, d) for c, d in zip(h["canon"], h["date"])]
    # astype(bool): on an EMPTY frame map() gives an object Series, and h[<empty object Series>]
    # selects COLUMNS - every column vanished and plan() died on KeyError 'canon' (Phase 2 HT repair)
    bad = ~h["season"].map(is_known).astype(bool)
    if bad.any():
        print(f"[ht] {int(bad.sum())} history row(s) skipped: API season unknown for "
              f"{sorted(h.loc[bad, 'canon'].unique())}")
    return h[~bad]


def plan(lacking: pd.DataFrame, journal: pd.DataFrame) -> list[dict]:
    """League-seasons to fetch: every (league_id, season) with a history row lacking HT, minus the
    journaled ones whose fetch is newer than that league-season's latest lacking match."""
    last_fetch = {}
    if not journal.empty:
        for r in journal.itertuples(index=False):
            t = pd.to_datetime(r.fetched_at, errors="coerce", utc=True)
            if t == t:
                last_fetch[(int(r.league_id), str(r.season))] = t.tz_localize(None).normalize()
    out = []
    g = lacking.groupby(["canon", "league_id", "season"])["date"].agg(["max", "size"]).reset_index()
    for r in g.itertuples(index=False):
        key = (int(r.league_id), str(r.season))
        if key in last_fetch and r.max < last_fetch[key]:
            continue
        out.append({"league": r.canon, "league_id": key[0], "season": key[1], "n_lacking": int(r.size)})
    return sorted(out, key=lambda x: (x["league"], x["season"]))


def _int_or_none(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _pair_or_none(d: dict) -> tuple:
    """(home, away) of a provider score block; (None, None) unless BOTH sides are given."""
    h, a = _int_or_none((d or {}).get("home")), _int_or_none((d or {}).get("away"))
    return (h, a) if h is not None and a is not None else (None, None)


def parse(body: dict, league: str, league_id: int, season: str) -> list[dict]:
    """Finished fixtures with the provider's half-time score (None when the provider gives none), and
    awarded (AWD / WO) and abandoned (ABD) ones with their status. Extra time / penalties as the
    provider gives them."""
    rows = []
    for f in (body or {}).get("response") or []:
        fx, tm = f.get("fixture") or {}, f.get("teams") or {}
        score = f.get("score") or {}
        ht = score.get("halftime") or {}
        status = (fx.get("status") or {}).get("short")
        if status not in FINISHED | NOT_PLAYED:
            continue
        h, a = tm.get("home") or {}, tm.get("away") or {}
        ft = score.get("fulltime") or {}
        ft_h, ft_a = _int_or_none(ft.get("home")), _int_or_none(ft.get("away"))
        et_h, et_a = _pair_or_none(score.get("extratime"))
        pen_h, pen_a = _pair_or_none(score.get("penalty"))
        ko = str(fx.get("date") or "")
        rows.append({"fixture_id": _int_or_none(fx.get("id")), "league": league, "league_id": league_id,
                     "season": season, "date": ko[:10], "kickoff": ko,
                     "home_id": _int_or_none(h.get("id")), "home_name": h.get("name"),
                     "away_id": _int_or_none(a.get("id")), "away_name": a.get("name"),
                     "ht_home": _int_or_none(ht.get("home")), "ht_away": _int_or_none(ht.get("away")),
                     "status": status, "ft_home": ft_h, "ft_away": ft_a,
                     # a half-given FT is no FT: both sides or neither (never one side + a missing 0)
                     "ft_source": FT_SRC_FETCH if ft_h is not None and ft_a is not None else None,
                     "et_home": et_h, "et_away": et_a, "pen_home": pen_h, "pen_away": pen_a})
        if rows[-1]["ft_source"] is None:
            rows[-1]["ft_home"] = rows[-1]["ft_away"] = None
    return [r for r in rows if r["fixture_id"] is not None]


def _ft_pair(a, b):
    """(home, away) as ints, or None when either side is missing."""
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            return None
        return int(a), int(b)
    except (TypeError, ValueError):
        return None


def complete_ft_from_competition_fixtures(scores: pd.DataFrame,
                                          comp: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """Stored score rows without ft_home/ft_away take score.fulltime from step 1's league fixture
    lists BY FIXTURE ID and the same league id (no call). A value stored already is never replaced.
    Returns (scores, rows completed)."""
    scores = scores.copy()
    for c in ("ft_home", "ft_away", "ft_source"):
        if c not in scores.columns:
            scores[c] = None
    if scores.empty:
        return scores, 0
    if comp is None:
        if not COMP_FIXTURES.exists():
            return scores, 0
        comp = pd.read_parquet(COMP_FIXTURES, columns=["fixture_id", "league_id", "ft_home", "ft_away"])
    comp = comp.dropna(subset=["ft_home", "ft_away"]).drop_duplicates("fixture_id")
    by_id = {int(r.fixture_id): (int(r.league_id), int(r.ft_home), int(r.ft_away))
             for r in comp.itertuples(index=False)}
    lack = scores["ft_home"].isna() | scores["ft_away"].isna()
    n = 0
    for i in scores.index[lack]:
        hit = by_id.get(int(scores.at[i, "fixture_id"]))
        if hit is None or hit[0] != int(scores.at[i, "league_id"]):
            continue                                     # another league's id: never borrowed
        scores.at[i, "ft_home"], scores.at[i, "ft_away"] = hit[1], hit[2]
        scores.at[i, "ft_source"] = FT_SRC_COMP
        n += 1
    return scores, n


def ft_plan(scores: pd.DataFrame, ft_journal: pd.DataFrame) -> list[dict]:
    """League-seasons whose stored FINISHED fixtures still lack a provider FT after the no-call
    completion, minus those already re-read (journal): one call each."""
    if scores.empty:
        return []
    # an awarded fixture (AWD / WO) often has no score.fulltime at all: it is no reason to re-read
    fin = scores["status"].isin(FINISHED) if "status" in scores.columns else pd.Series(True, index=scores.index)
    lack = scores[fin & (scores["ft_home"].isna() | scores["ft_away"].isna())]
    done = set()
    if not ft_journal.empty:
        done = {(int(r.league_id), str(r.season)) for r in ft_journal.itertuples(index=False)}
    out = []
    for (lid, season, lg), g in lack.groupby(["league_id", "season", "league"]):
        if (int(lid), str(season)) not in done:
            out.append({"league": lg, "league_id": int(lid), "season": str(season), "n_lacking_ft": len(g)})
    return sorted(out, key=lambda x: (x["league"], x["season"]))


def _resolve(name, candidates, canon=None, date=None):
    """resolve_result, else (P7 item 1) the league-scoped alias PROVEN from same-fixture evidence
    (config/af_history_proven_aliases.json) when `canon` and the row `date` are given."""
    from src.team_names import resolve_result
    if not len(candidates):
        return None
    if canon is None or date is None:
        return resolve_result(str(name), [str(c) for c in candidates])
    from prove_af_history_aliases import resolve_with_proven
    return resolve_with_proven(str(name), [str(c) for c in candidates], canon, str(date)[:10],
                               resolve_result)


def claims(lacking: pd.DataFrame, scores: pd.DataFrame, stats: dict | None = None,
           require_ht: bool = True, statuses: set | None = None) -> pd.DataFrame:
    """Every history row's ONE candidate fixture (same league, clubs proven home->home and
    away->away, same day else adjacent day), BEFORE the two-dates refusal and the FT check. Also the
    input of scripts/build_history_corrections.py, which proves duplicate history rows from these
    claims. Columns: FILL_COLS (FT check not yet applied) + _canon, _hg, _ag (history FT), kickoff,
    status, provider_home, provider_away, et_home, et_away, pen_home, pen_away.
    `require_ht` False (the history corrections builder): every fixture of `statuses` (default
    FINISHED) is a candidate, HTHG / HTAG missing where the provider gives none."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if stats is None:
        stats = defaultdict(lambda: defaultdict(int))
    extra = ["et_home", "et_away", "pen_home", "pen_away"]
    cols = FILL_COLS + ["_canon", "_hg", "_ag", "kickoff", "status", "provider_home", "provider_away"] + extra
    sc = scores.dropna(subset=["ht_home", "ht_away"]).copy() if require_ht else scores.copy()
    for c in ["ft_home", "ft_away", "kickoff", "status", "ht_home", "ht_away"] + extra:
        if c not in sc.columns:
            sc[c] = None
    if "status" in scores.columns:
        # an awarded fixture (AWD / WO) is no played match: never a source of a half-time score
        sc = sc[sc["status"].isin(statuses or FINISHED)]
    sc["_d"] = pd.to_datetime(sc["date"], errors="coerce").dt.normalize()
    by_day = {k: g for k, g in sc.groupby(["league", "_d"])}
    # The history repeats some matches row for row (e.g. one season under two labels); a repeat is
    # ONE match (one fill row, keyed like the sidecar), not two matches claiming one fixture.
    key = ["canon", "date", "home_team", "away_team"]
    rep = lacking.duplicated(subset=key)
    for lg, n in lacking.loc[rep, "canon"].value_counts().items():
        stats[lg]["history_repeat_rows"] += int(n)
    lacking = lacking[~rep]
    picks = []
    for r in lacking.itertuples(index=False):
        st = stats[r.canon]
        st["lacking"] += 1
        d = pd.Timestamp(r.date).normalize()
        found, rule = [], None
        for days, name in (((0,), "same_day"), ((-1, 1), "adjacent_day")):
            parts = [by_day[(r.canon, d + pd.Timedelta(days=k))] for k in days
                     if (r.canon, d + pd.Timedelta(days=k)) in by_day]
            if not parts:
                continue
            pool = pd.concat(parts)
            h = _resolve(r.home_team, pool["home_name"].unique(), r.canon, d.date())
            a = _resolve(r.away_team, pool["away_name"].unique(), r.canon, d.date())
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
        pft = _ft_pair(f["ft_home"], f["ft_away"])
        pht = _ft_pair(f["ht_home"], f["ht_away"])
        picks.append({"league": r.league, "date": r.date, "home_team": r.home_team,
                      "away_team": r.away_team, "HTHG": pht[0] if pht else None,
                      "HTAG": pht[1] if pht else None,
                      "fixture_id": int(f["fixture_id"]), "af_date": f["date"], "match_rule": rule,
                      "source": SOURCE, "matched_at": now,
                      "provider_ft_home": pft[0] if pft else None,
                      "provider_ft_away": pft[1] if pft else None, "ft_check": None,
                      "_canon": r.canon, "_hg": getattr(r, "home_goals", None),
                      "_ag": getattr(r, "away_goals", None), "kickoff": f["kickoff"],
                      "status": f["status"], "provider_home": f["home_name"],
                      "provider_away": f["away_name"],
                      **{c: _int_or_none(f[c]) if pd.notna(f[c]) else None for c in extra}})
    return pd.DataFrame(picks, columns=cols)


def match(lacking: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(fill rows, per-league counts). See the module docstring for the rule. The FT mismatches
    refused this run are in the per-league counts and in attrs['ft_mismatch'] of the fill frame."""
    stats: dict = defaultdict(lambda: defaultdict(int))
    if scores.empty:
        for lg, n in lacking["canon"].value_counts().items():
            stats[lg]["lacking"] = int(n)
            stats[lg]["no_fixture_stored"] = int(n)
        out = pd.DataFrame(columns=FILL_COLS)
        out.attrs["ft_mismatch"] = pd.DataFrame()
        return out, {k: dict(v) for k, v in stats.items()}
    fill = claims(lacking, scores, stats)
    mism = pd.DataFrame()
    if not fill.empty:
        # One fixture claimed by several history rows. Rows of the SAME date whose clubs both
        # resolved onto that fixture's two clubs are the same match under two spellings (the
        # history carries e.g. 'Austin' / 'Austin FC' rows for one MLS game): each gets the score,
        # counted. Claims from DIFFERENT dates are two matches for one fixture: all refused.
        n_dates = fill.groupby("fixture_id")["date"].transform("nunique")
        many = fill["fixture_id"].duplicated(keep=False)
        twice = many & (n_dates > 1)
        for lg, n in fill.loc[twice, "_canon"].value_counts().items():
            stats[lg]["refused_fixture_claimed_twice"] += int(n)
        for lg, n in fill.loc[many & ~twice, "_canon"].value_counts().items():
            stats[lg]["history_rows_same_match_two_spellings"] += int(n)
        fill = fill[~twice].copy()
        # FULL-TIME CHECK (owner 2026-11-04): the provider's score.fulltime must equal the history's
        # FT where both are known; otherwise the pairing is not proven and the row is refused.
        hft = [_ft_pair(a, b) for a, b in zip(fill["_hg"], fill["_ag"])]
        pft = [_ft_pair(a, b) for a, b in zip(fill["provider_ft_home"], fill["provider_ft_away"])]
        fill["ft_check"] = ["provider_ft_missing" if p is None else "history_ft_missing" if h is None
                            else "equal" if p == h else "mismatch" for h, p in zip(hft, pft)]
        bad = fill["ft_check"] == "mismatch"
        for lg, n in fill.loc[bad, "_canon"].value_counts().items():
            stats[lg]["refused_ft_mismatch"] += int(n)
        mism = fill.loc[bad, ["league", "date", "home_team", "away_team", "_hg", "_ag", "fixture_id",
                              "af_date", "kickoff", "status", "provider_home", "provider_away",
                              "provider_ft_home", "provider_ft_away", "HTHG", "HTAG"]] \
            .rename(columns={"_hg": "history_ft_home", "_ag": "history_ft_away",
                             "HTHG": "provider_ht_home", "HTAG": "provider_ht_away"}).reset_index(drop=True)
        fill = fill[~bad]
        for lg, n in fill["_canon"].value_counts().items():
            stats[lg]["filled"] += int(n)
    out = fill[FILL_COLS].reset_index(drop=True)
    out.attrs["ft_mismatch"] = mism
    return out, {k: dict(v) for k, v in stats.items()}


def add_ft(client: AFClient | None = None, fetch: bool = False) -> dict:
    """Complete score.fulltime on the stored fixtures: first from step 1's fixture lists (no call),
    then - only with `fetch` - by re-reading each league-season still lacking it (one call each,
    journaled in FT_JOURNAL, so a budget stop resumes where it stopped). Returns counts."""
    out = {"ft_completed_from_competition_fixtures": 0, "ft_calls": 0, "ft_planned_calls": 0}
    scores = read_parquet(SCORES)
    if scores.empty:
        return out
    scores, n = complete_ft_from_competition_fixtures(scores)
    out["ft_completed_from_competition_fixtures"] = n
    if n:
        tmp = SCORES.with_suffix(".tmp.parquet")
        scores[[c for c in SCORE_COLS if c in scores.columns]].to_parquet(tmp, index=False)
        tmp.replace(SCORES)
    todo = ft_plan(scores, read_parquet(FT_JOURNAL))
    out["ft_planned_calls"] = len(todo)
    print(f"[ht-ft] {n:,} stored fixture(s) completed with score.fulltime from competition_fixtures "
          f"(0 calls); {len(todo)} league-season(s) still lacking FT = {len(todo)} call(s)"
          + ("" if fetch else " - not fetched (run with --add-ft)"))
    if not fetch or not todo:
        return out
    client = client or AFClient()
    try:
        for t in todo:
            body = client.get("fixtures", {"league": t["league_id"], "season": t["season"]})
            if body is None:
                continue                                    # failed: nothing recorded, retried next run
            out["ft_calls"] += 1
            rows = parse(body, t["league"], t["league_id"], t["season"])
            if rows:
                append_parquet(SCORES, rows, ["fixture_id"], SCORE_COLS)
            append_parquet(FT_JOURNAL, [{"league_id": t["league_id"], "season": t["season"],
                                         "league": t["league"],
                                         "n_finished": sum(r["status"] in FINISHED for r in rows),
                                         "n_with_ft": sum(r["ft_home"] is not None for r in rows),
                                         "fetched_at": datetime.now(timezone.utc).isoformat()}],
                           ["league_id", "season"], FT_JOURNAL_COLS)
    except BudgetStop as e:
        print(f"[ht-ft] budget stop: {e} (resumable)")
        out["ft_budget_stop"] = str(e)
    finally:
        client.flush()
    return out


def status_plan(journal: pd.DataFrame, ft_journal: pd.DataFrame, status_journal: pd.DataFrame,
                comp: pd.DataFrame | None = None, all_leagues: bool = False) -> list[dict]:
    """League-season lists to re-read ONCE for extra time / penalties / AWD / WO / ABD: every stored
    HT list (journal + FT journal; scope 'ht_lists') and, with all_leagues, every other registry
    league-season of step 1's fixture lists (scope 'registry_leagues'), minus those already re-read
    (status journal). One call each."""
    done = set()
    if not status_journal.empty:
        done = {(int(r.league_id), str(r.season)) for r in status_journal.itertuples(index=False)}
    out, seen = [], set()
    for j in (journal, ft_journal):
        if j is None or j.empty:
            continue
        for r in j.itertuples(index=False):
            k = (int(r.league_id), str(r.season))
            if k in done or k in seen:
                continue
            seen.add(k)
            out.append({"league": r.league, "league_id": k[0], "season": k[1], "scope": "ht_lists"})
    if all_leagues and comp is not None and not comp.empty:
        c = comp[comp["league"].notna()]
        if "competition_type" in c.columns:
            c = c[c["competition_type"] == "club_league"]
        for (lid, season, lg), _ in c.groupby(["league_id", "season", "league"]):
            k = (int(lid), str(season))
            if k in done or k in seen:
                continue
            seen.add(k)
            out.append({"league": lg, "league_id": k[0], "season": k[1], "scope": "registry_leagues"})
    return sorted(out, key=lambda x: (x["scope"] != "ht_lists", x["league"], x["season"]))


def add_status(client: AFClient | None = None, fetch: bool = False, all_leagues: bool = False) -> dict:
    """Re-read the stored league-season lists once (status_plan) and store every finished, awarded
    and abandoned fixture with score.extratime / score.penalty and its status in af_ht_scores (the
    NEW parse wins over the stored row of the same fixture id: same provider fields, plus the ones
    the old parse dropped). Journaled per league-season, so a budget stop resumes. Returns counts."""
    comp = None
    if all_leagues and COMP_FIXTURES.exists():
        comp = pd.read_parquet(COMP_FIXTURES, columns=["league_id", "league", "season", "competition_type"])
    todo = status_plan(read_parquet(JOURNAL), read_parquet(FT_JOURNAL), read_parquet(STATUS_JOURNAL),
                       comp, all_leagues)
    from collections import Counter
    by_scope = dict(Counter(t["scope"] for t in todo))
    out = {"status_planned_calls": len(todo), "status_planned_by_scope": by_scope, "status_calls": 0,
           "awarded_stored": 0, "abandoned_stored": 0}
    print(f"[ht-status] {len(todo)} league-season list(s) to re-read once for extra time / penalties / "
          f"AWD / WO / ABD = {len(todo)} API-Football call(s) {by_scope}"
          + ("" if fetch else " - not fetched (dry run)"))
    if not fetch or not todo:
        return out
    client = client or AFClient()
    try:
        for t in todo:
            body = client.get("fixtures", {"league": t["league_id"], "season": t["season"]})
            if body is None:
                continue                                    # failed: nothing recorded, retried next run
            out["status_calls"] += 1
            rows = parse(body, t["league"], t["league_id"], t["season"])
            if rows:
                append_parquet(SCORES, rows, ["fixture_id"], SCORE_COLS)
            n_awd = sum(r["status"] in AWARDED for r in rows)
            n_abd = sum(r["status"] in ABANDONED for r in rows)
            out["awarded_stored"] += n_awd
            out["abandoned_stored"] += n_abd
            append_parquet(STATUS_JOURNAL, [{"league_id": t["league_id"], "season": t["season"],
                                             "league": t["league"], "scope": t["scope"],
                                             "n_fixtures": len(body.get("response") or []),
                                             "n_kept": len(rows), "n_awarded": n_awd, "n_abandoned": n_abd,
                                             "n_extratime": sum(r["et_home"] is not None for r in rows),
                                             "fetched_at": datetime.now(timezone.utc).isoformat()}],
                           ["league_id", "season"], STATUS_JOURNAL_COLS)
    except BudgetStop as e:
        print(f"[ht-status] budget stop: {e} (resumable)")
        out["status_budget_stop"] = str(e)
    finally:
        client.flush()
    print(f"[ht-status] {json.dumps(out)}")
    return out


def run(dry_run: bool = False, match_only: bool = False, client: AFClient | None = None,
        history: pd.DataFrame | None = None, fetch_ft: bool = False) -> dict:
    """`fetch_ft` (--add-ft): also re-read the league-seasons whose stored fixtures still lack the
    provider FT after the no-call completion (budget-guarded, resumable). Ignored with match_only."""
    lacking = lacking_rows(history)
    todo = plan(lacking, read_parquet(JOURNAL))
    print(f"[ht] {len(lacking):,} new-format history row(s) without a half-time score; "
          f"{len(todo)} league-season(s) to fetch = {len(todo)} API-Football call(s)")
    for t in todo:
        print(f"  {t['league']} (id {t['league_id']}) season {t['season']}: {t['n_lacking']} row(s) lacking HT")
    stats = {"planned_calls": len(todo), "calls": 0}
    if dry_run:
        return stats
    if not match_only and todo:
        client = client or AFClient()
        try:
            for t in todo:
                body = client.get("fixtures", {"league": t["league_id"], "season": t["season"]})
                if body is None:
                    continue                                # failed: nothing recorded, retried next run
                stats["calls"] += 1
                rows = parse(body, t["league"], t["league_id"], t["season"])
                if rows:
                    append_parquet(SCORES, rows, ["fixture_id"], SCORE_COLS)
                append_parquet(JOURNAL, [{"league_id": t["league_id"], "season": t["season"],
                                          "league": t["league"],
                                          "n_fixtures": len(body.get("response") or []),
                                          "n_finished": sum(r["status"] in FINISHED for r in rows),
                                          "n_with_ht": sum(r["ht_home"] is not None for r in rows),
                                          "fetched_at": datetime.now(timezone.utc).isoformat()}],
                               ["league_id", "season"], JOURNAL_COLS)
        except BudgetStop as e:
            print(f"[ht] budget stop: {e} (resumable)")
            stats["budget_stop"] = str(e)
        finally:
            client.flush()
    stats.update(add_ft(client=client, fetch=fetch_ft and not match_only and "budget_stop" not in stats))
    fill, per_league = match(lacking, read_parquet(SCORES))
    mism = fill.attrs.pop("ft_mismatch", pd.DataFrame())        # attrs must not reach parquet
    stats["refused_ft_mismatch"] = int(len(mism))
    FT_MISMATCH.parent.mkdir(parents=True, exist_ok=True)
    (mism if len(mism) else pd.DataFrame(columns=[
        "league", "date", "home_team", "away_team", "history_ft_home", "history_ft_away", "fixture_id",
        "af_date", "kickoff", "status", "provider_home", "provider_away", "provider_ft_home",
        "provider_ft_away", "provider_ht_home", "provider_ht_away"])).to_csv(FT_MISMATCH, index=False)
    old = read_parquet(FILL)
    key = ["league", "date", "home_team", "away_team"]
    if not old.empty:
        for c in FILL_COLS:                    # a sidecar written before the FT columns existed
            if c not in old.columns:
                old[c] = None
        # a stored row of a history row the owner's corrections remove is no longer a league match
        from src.history_corrections import corrected_mask
        gone = corrected_mask(old)
        stats["stored_rows_dropped_history_corrections"] = int(gone.sum())
        old = old[~gone]
        # A history row RE-EVALUATED this run carries this run's answer - including "refused" or
        # "unmatched". It used to keep its stored score unless re-matched now, so a row a later run
        # refuses (a second history row of another date claiming the same fixture) kept a stale
        # score, and an empty match result left the whole stored file untouched (Phase 2 HT repair,
        # 2026-11-03). Rows outside this run's scope (no longer lacking HT, season unknown) stay.
        old = old[FILL_COLS].copy()
        seen = set(zip(lacking["league"].astype(str), pd.to_datetime(lacking["date"]),
                       lacking["home_team"].astype(str), lacking["away_team"].astype(str)))
        re_eval = pd.Series([k in seen for k in zip(old["league"].astype(str), pd.to_datetime(old["date"]),
                                                     old["home_team"].astype(str), old["away_team"].astype(str))],
                            index=old.index, dtype=bool)
        new_keys = set(zip(fill["league"].astype(str), pd.to_datetime(fill["date"]),
                           fill["home_team"].astype(str), fill["away_team"].astype(str)))
        dropped = [k for k, r in zip(zip(old["league"].astype(str), pd.to_datetime(old["date"]),
                                         old["home_team"].astype(str), old["away_team"].astype(str)), re_eval)
                   if r and k not in new_keys]
        stats["stored_rows_dropped_now_unanswered"] = len(dropped)
        old = old[~re_eval]
        fill = pd.concat([old, fill], ignore_index=True) if not fill.empty else old
    fill = fill.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    fill.attrs = {}
    FILL.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILL.with_suffix(".tmp.parquet")
    fill.to_parquet(tmp, index=False)
    tmp.replace(FILL)
    stats["filled_rows_total"] = int(len(fill))
    stats["per_league"] = per_league
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps({"written_at": datetime.now(timezone.utc).isoformat(),
                                 "history": str(HISTORY.relative_to(ROOT)),
                                 "note": "Half-time scores for history rows that lack them, from API-"
                                         "Football; fd_history.parquet itself is not modified.",
                                 **stats}, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(f"[ht] {json.dumps({k: v for k, v in stats.items() if k != 'per_league'})}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect half-time scores for the new-format leagues")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--match-only", action="store_true")
    ap.add_argument("--add-ft", action="store_true",
                    help="re-read league-seasons whose stored fixtures lack score.fulltime "
                         "(about 120 calls at most, budget-guarded, resumable)")
    ap.add_argument("--add-status", action="store_true",
                    help="ONLY re-read each stored league-season list once for extra time, penalties and "
                         "AWD / WO / ABD (about 120 calls, budget-guarded, resumable; with --dry-run: count)")
    ap.add_argument("--status-all-leagues", action="store_true",
                    help="with --add-status: also every other registry league-season of competition_fixtures")
    a = ap.parse_args(argv)
    if a.add_status:
        add_status(fetch=not a.dry_run, all_leagues=a.status_all_leagues)
        return 0
    run(dry_run=a.dry_run, match_only=a.match_only, fetch_ft=a.add_ft)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
