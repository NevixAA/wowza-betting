"""
build_history_corrections.py - the owner's history corrections of 2026-11-04, PROVEN from provider data.

    python scripts/build_history_corrections.py            # derive both lists from stored data; no call
    python scripts/build_history_corrections.py --fetch    # + fetch the national cup fixture lists the
                                                           #   unplaced rows need (budget-guarded)
    python scripts/build_history_corrections.py --dry-run  # print the plan (calls); write nothing

Writes BESIDE the history (output/fd_history.parquet is never rewritten); src/history_corrections.py
applies the lists in the loader and in the HT collector:

1. DUPLICATES -> output/corrections/history_removed_duplicates.csv (owner: "duplicates remove").
   The Phase 2 HT repair refused 1,198 history rows because ONE provider fixture was claimed by
   history rows of TWO dates (MLS 706, Brazil 200, Argentina 180, Mexico 112): the same match filed
   twice, one day apart. WHY one day apart: API-Football dates a fixture by its UTC kick-off; the
   Americas play evening matches (19:00-22:30 local = 22:00-03:30 UTC), so a late kick-off is one
   calendar day later in UTC than at the ground. One history copy carries the local date, the other
   the UTC date. Rule (scripts/collect_ht_scores.claims, the HT matcher itself): every row's ONE
   candidate fixture - same league, both clubs proven by src.team_names.resolve_result home->home and
   away->away, same day else adjacent day. For a fixture claimed from two dates the copy dated on the
   PROVIDER's fixture date stays and the other copy is removed, only when:
     * both copies claim the SAME provider fixture id (so both clubs resolve onto its two clubs),
     * the full-time scores of all copies agree wherever both hold one,
     * the dates are exactly one day apart and one copy is on the provider date.
   Anything else is a CONFLICT: counted, listed (history_duplicate_conflicts.csv), left alone.
   Each removed row keeps its full original values plus the kept row's key and the evidence
   (fixture id, provider date, UTC kick-off, local kick-off date by the registry timezone).

2. CUP ROWS -> output/corrections/history_excluded_cup_rows.csv (owner: "cups are not in Wowza tips;
   keep cups only as player data"). Candidates: new-format history rows with NO provider fixture of
   their league within +-1 day, inside the span the provider's league fixture lists cover (the Phase 2
   finding: 407 Argentina rows 2023-08..2024-05, Romania 2, Japan 1). Proof: the league's country's
   national cups (API-Football /leagues?country&type=cup, provider type 'Cup' and the registry
   country), their fixture lists (/fixtures?league&season, one call per cup-season that covers a
   candidate date), and exactly ONE cup fixture on the row's date (else the adjacent day) with both
   clubs proven home->home / away->away and the same full-time score where both hold one. A row that
   cannot be proven stays in the history and is listed with its reason
   (history_cup_unproven_rows.csv). Cup fixture lists: output/collect/national_cups.parquet,
   national_cup_fixtures.parquet, national_cup_seasons.parquet (journal: a stored cup-season that has
   ended is never fetched again).

fix8/history2 follow-ups (2026-11-04; the owner rules already decided: the provider's date decides
duplicates, settlement is on the 90-minute score, corrections go beside the originals). All three
use only stored provider data - no call - and one pool of provider fixtures: the HT store
(af_ht_scores, FT completed from competition_fixtures) plus step 1's league fixture lists
(output/strength/competition_fixtures.parquet), one row per fixture id.

3. LOADER-MERGE DUPLICATES (appended to history_removed_duplicates.csv, found_by
   'loader_merge_pair'). Rule 1 only saw rows lacking a half-time score. Now EVERY pair the loader's
   own de-duplication merges (src.data_loader.identity_before_one_match + src.integrity.
   duplicate_matches.duplicate_pairs, on fd_history + the backtest archive, after the other lists)
   whose two rows are both history rows on different days goes through the same proof: both rows
   claim the SAME provider fixture id (clubs resolved home->home / away->away), the same FT, one copy
   on the provider date and the other one day off -> the off-date copy is removed. Anything else is a
   conflict (listed, left to the loader). A same-day pair has no date to decide and is left to the
   loader; a pair with an archive row is out of scope (a history correction never removes or keeps a
   non-history row) - both counted. The audit counts how many survivors CHANGE versus the loader's
   tie-break ("most populated, then earliest").
4. 90-MINUTE SCORE -> output/corrections/history_score_90min.csv. Every history row (all leagues)
   proven to be a provider fixture finished AET or PEN (claim as above, the fixture not claimed from
   two dates) whose FT is not the provider's score.fulltime but >= it on both sides is served at the
   90-minute score (original kept in the list). A history FT below the 90-minute score on a side is
   not an extra-time score: refused (UNIDENTIFIABLE), listed in history_score_refused.csv. et_check
   records whether the history FT equals 90' + score.extratime where the provider's extra time is
   stored (it was not stored before fix8/history2: 'extratime_not_stored'); it is evidence, not a
   gate - the fixture is proven by its clubs and date, and the 90-minute score is the provider's own.
   The same list carries every FT-status disagreement (provider FT != history FT on a played
   fixture): never corrected, reason 'ft_status_scores_differ' (UNIDENTIFIABLE - two sources, no
   proof which is right).
5. AWARDED -> output/corrections/history_awarded_results.csv: a history row proven to be a provider
   fixture with status AWD / WO / ABD is excluded (not a played score), with its reason. The stores
   held no AWD / WO fixture before fix8/history2 (both collectors kept FT/AET/PEN only); the 120 old
   league-season lists are re-read once for them by collect_ht_scores.py --add-status (fix9/defaults).

fix9/defaults owner decisions (2026-11-04), no call:
6. NOT PLAYED (OWNER_NOT_PLAYED) -> the awarded list: Queretaro v Atlas 2022-03-05 was abandoned after
   crowd violence and awarded; every history copy proven to be provider fixture 816403 is excluded with
   that reason (the provider lists it FT 0-1, so rule 5 cannot see it).
7. SAME-DAY COPY WITH A WRONG SCORE -> history_removed_duplicates.csv (found_by
   'same_day_copy_wrong_score_provider_agrees'): "the provider plus an independent agreeing copy is
   proven". Two or more same-day history copies of one provider FT fixture, one agreeing with
   score.fulltime and one not: the disagreeing copy is removed as a duplicate (MLS 2023-11-25 Orlando
   City v Columbus Crew: provider 0-2, copy 'Orlando City SC' 0-2, copy 'Orlando City' 0-0). A lone
   disagreeing copy stays UNIDENTIFIABLE in history_score_refused.csv.

Audit: output/corrections/history_corrections_audit.json. BUDGET: every call goes through
collect_common.AFClient (job cap, account floor); MAX_CALLS per run on top. A budget stop is a clean,
resumable exit: what is stored is used, the rest stays unproven (and in).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_ht_scores as H  # noqa: E402
from collect_common import COLLECT_DIR, AFClient, BudgetStop, append_parquet, read_parquet  # noqa: E402

HISTORY = ROOT / "output" / "fd_history.parquet"
OUT_DIR = ROOT / "output" / "corrections"
DUP_OUT = OUT_DIR / "history_removed_duplicates.csv"
DUP_CONFLICTS = OUT_DIR / "history_duplicate_conflicts.csv"
CUP_OUT = OUT_DIR / "history_excluded_cup_rows.csv"
CUP_UNPROVEN = OUT_DIR / "history_cup_unproven_rows.csv"
SCORE_OUT = OUT_DIR / "history_score_90min.csv"
SCORE_REFUSED = OUT_DIR / "history_score_refused.csv"
AWARDED_OUT = OUT_DIR / "history_awarded_results.csv"
AWARDED_COLS = ["league", "date", "home_team", "away_team", "home_goals", "away_goals", "fixture_id",
                "provider_status", "reason"]
AUDIT = OUT_DIR / "history_corrections_audit.json"
KEY = ["league", "date", "home_team", "away_team"]
EXTRA_TIME = {"AET", "PEN"}
SCORE_OUT_COLS = KEY + ["history_ft_home", "history_ft_away", "ft90_home", "ft90_away", "fixture_id",
                        "provider_status", "provider_date", "provider_home", "provider_away", "et_home",
                        "et_away", "pen_home", "pen_away", "match_rule", "et_check", "found_by"]
SCORE_REFUSED_COLS = KEY + ["history_ft_home", "history_ft_away", "provider_ft_home", "provider_ft_away",
                            "fixture_id", "provider_status", "provider_date", "provider_home",
                            "provider_away", "reason", "verdict"]
COMP_FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
CUPS = COLLECT_DIR / "national_cups.parquet"
CUP_FIXTURES = COLLECT_DIR / "national_cup_fixtures.parquet"
CUP_JOURNAL = COLLECT_DIR / "national_cup_seasons.parquet"
MAX_CALLS = 40
FINISHED = {"FT", "AET", "PEN"}
#: list_kind "all_types" = from the full /leagues?country list (2026-11-05); blank = the older cup-only list
CUPS_COLS = ["country", "league_id", "name", "type", "season", "start", "end", "fetched_at", "list_kind"]
CUP_FX_COLS = ["fixture_id", "league_id", "cup_name", "country", "season", "date", "kickoff",
               "home_id", "home_name", "away_id", "away_name", "ft_home", "ft_away", "status"]
CUP_JOURNAL_COLS = ["league_id", "season", "country", "season_end", "n_fixtures", "n_finished", "fetched_at"]


# ---- shared ------------------------------------------------------------------------------------

def _registry() -> dict:
    """{canonical new-format league: registry entry}."""
    from src import league_registry as LR
    return {e["name"]: e for e in LR.leagues()
            if e.get("model_type") == "new_format" and e.get("api_football_id") is not None}


def _ft(a, b):
    return H._ft_pair(a, b)


def _local_date(kickoff: str, tz: str | None):
    """The kick-off's calendar date at the ground (registry timezone), or None."""
    t = pd.to_datetime(kickoff, errors="coerce", utc=True)
    if t is pd.NaT or t != t or not tz:
        return None
    try:
        return t.tz_convert(tz).strftime("%Y-%m-%d")
    except Exception:                                                  # noqa: BLE001
        return None


def load_history() -> pd.DataFrame:
    h = pd.read_parquet(HISTORY)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    return h


def provider_scores() -> pd.DataFrame:
    """af_ht_scores with score.fulltime completed from step 1's fixture lists (in memory)."""
    sc = read_parquet(H.SCORES)
    if sc.empty:
        return sc
    comp = read_parquet(COMP_FIXTURES)
    sc, _ = H.complete_ft_from_competition_fixtures(sc, comp if not comp.empty else None)
    return sc


# ---- 1. duplicates -----------------------------------------------------------------------------

def duplicate_removals(history: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(removed rows with evidence, conflicts, counts). See the module docstring."""
    reg = _registry()
    lack = H.lacking_rows(history[[c for c in KEY + ["home_goals", "away_goals", "ht_home_goals",
                                                     "ht_away_goals"] if c in history.columns]],
                          corrections=False)
    picks = H.claims(lack, scores)
    stats = {"claims": int(len(picks))}
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame(), stats
    picks = picks.rename(columns={"_canon": "canon", "_hg": "hist_hg", "_ag": "hist_ag"})
    picks["day"] = pd.to_datetime(picks["date"]).dt.strftime("%Y-%m-%d")
    nd = picks.groupby("fixture_id")["day"].transform("nunique")
    two = picks[nd > 1].copy()
    stats["rows_claiming_a_fixture_from_two_dates"] = int(len(two))
    stats["by_league"] = {k: int(v) for k, v in two["canon"].value_counts().items()}
    stats["fixtures"] = int(two["fixture_id"].nunique())
    removed, conflicts = [], []
    for fid, g in two.groupby("fixture_id"):
        pdate = str(g["af_date"].iloc[0])[:10]
        tz = (reg.get(g["canon"].iloc[0]) or {}).get("timezone")
        local = _local_date(str(g["kickoff"].iloc[0]), tz)
        ev = {"fixture_id": int(fid), "provider_date": pdate, "provider_kickoff_utc": g["kickoff"].iloc[0],
              "provider_local_date": local, "provider_timezone": tz,
              "provider_home": g["provider_home"].iloc[0], "provider_away": g["provider_away"].iloc[0],
              "provider_ft_home": g["provider_ft_home"].iloc[0], "provider_ft_away": g["provider_ft_away"].iloc[0],
              "provider_status": g["status"].iloc[0]}
        fts = {_ft(a, b) for a, b in zip(g["hist_hg"], g["hist_ag"])} - {None}
        days = sorted(set(g["day"]))
        gap = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days
        kept = g[g["day"] == pdate]
        why = None
        if len(fts) > 1:
            why = "score_differs_between_copies"
        elif gap != 1:
            why = f"dates_{gap}_days_apart"
        elif kept.empty:
            why = "no_copy_on_provider_date"
        if why:
            for r in g.itertuples(index=False):
                conflicts.append({"league": r.league, "date": r.day, "home_team": r.home_team,
                                  "away_team": r.away_team, "home_goals": r.hist_hg, "away_goals": r.hist_ag,
                                  "conflict": why, **ev})
            continue
        k0 = kept.sort_values(["home_team", "away_team"]).iloc[0]
        for r in g[g["day"] != pdate].itertuples(index=False):
            removed.append({"league": r.league, "date": r.day, "home_team": r.home_team,
                            "away_team": r.away_team, "kept_date": pdate, "kept_home_team": k0["home_team"],
                            "kept_away_team": k0["away_team"], "kept_rows": int(len(kept)),
                            "removed_copy_is_local_date": (local == r.day) if local else None, **ev})
    rem = pd.DataFrame(removed)
    con = pd.DataFrame(conflicts)
    if len(con):
        con["found_by"] = "ht_claim_two_dates"
    if not rem.empty:
        rem["found_by"] = "ht_claim_two_dates"
        # the FULL original rows (every column fd_history holds; a key the history repeats -> each copy)
        orig = history.copy()
        orig["date"] = orig["date"].dt.strftime("%Y-%m-%d")
        rem = rem.merge(orig, on=KEY, how="left", validate="one_to_many")
        lead = KEY + [c for c in history.columns if c not in KEY]
        rem = rem[lead + [c for c in rem.columns if c not in lead]]
    stats.update({"removed_rows": int(len(rem)), "removed_keys": int(len(removed)),
                  "conflict_rows": int(len(con)),
                  "conflict_fixtures": int(con["fixture_id"].nunique()) if len(con) else 0,
                  "removed_by_league": {k: int(v) for k, v in rem["league"].value_counts().items()} if len(rem) else {},
                  "conflicts_by_reason": {k: int(v) for k, v in con["conflict"].value_counts().items()} if len(con) else {}})
    if len(rem):
        loc = rem.drop_duplicates(KEY)["removed_copy_is_local_date"]
        stats["timezone"] = {"removed_copy_dated_by_local_kickoff_date": int((loc == True).sum()),  # noqa: E712
                             "removed_copy_not_local_date": int((loc == False).sum()),            # noqa: E712
                             "local_date_unknown": int(loc.isna().sum()),
                             "kept_copy_is_provider_utc_date": int(len(rem.drop_duplicates(KEY)))}
    return rem, con, stats


# ---- provider pool and whole-history claims (fix8/history2) -------------------------------------

def provider_pool(scores: pd.DataFrame | None = None, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every stored provider fixture of a registry league, one row per fixture id, in the HT store's
    layout (H.SCORE_COLS): the HT store first (it carries kick-off, half-time, extra time), then step
    1's league fixture lists for the fixtures the HT store lacks. Same provider fields, no call."""
    sc = provider_scores() if scores is None else scores
    comp = read_parquet(COMP_FIXTURES) if comp is None else comp
    parts = []
    if not sc.empty:
        parts.append(sc.reindex(columns=H.SCORE_COLS))
    if not comp.empty:
        c = comp[comp["league"].notna()].copy()
        c["kickoff"] = None
        c["ft_source"] = H.FT_SRC_COMP
        parts.append(c.reindex(columns=H.SCORE_COLS))
    if not parts:
        return pd.DataFrame(columns=H.SCORE_COLS)
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.dropna(subset=["fixture_id"]).drop_duplicates("fixture_id", keep="first")
    return pool.reset_index(drop=True)


def _with_canon(history: pd.DataFrame) -> pd.DataFrame:
    from src import league_registry as LR
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    return h[h["canon"].notna() & h["date"].notna() & h["home_team"].notna() & h["away_team"].notna()]


def near(history: pd.DataFrame, pool: pd.DataFrame, statuses: set) -> pd.DataFrame:
    """History rows (with canon) within +-1 day of a pool fixture of their league whose status is in
    `statuses` - the only rows that can claim one; keeps the whole-history claim cheap."""
    p = pool[pool["status"].isin(statuses)]
    if p.empty or history.empty:
        return history.iloc[0:0]
    d = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    days = set()
    for lg, dd in zip(p["league"], d):
        for k in (-1, 0, 1):
            days.add((lg, dd + pd.Timedelta(days=k)))
    m = [(lg, dd) in days for lg, dd in zip(history["canon"], history["date"].dt.normalize())]
    return history[m]


def history_claims(rows: pd.DataFrame, pool: pd.DataFrame, statuses: set | None = None) -> pd.DataFrame:
    """H.claims over history rows (any league), every fixture of `statuses` a candidate, no half-time
    score required. History FT as hist_hg / hist_ag (itertuples renames '_' fields); adds `day`
    (history date) and `pdate` (provider date)."""
    cols = H.FILL_COLS + ["canon", "hist_hg", "hist_ag", "kickoff", "status", "provider_home",
                          "provider_away", "et_home", "et_away", "pen_home", "pen_away", "day", "pdate"]
    if rows.empty or pool.empty:
        return pd.DataFrame(columns=cols)
    c = H.claims(rows, pool, require_ht=False, statuses=statuses or (H.FINISHED | H.NOT_PLAYED))
    c = c.rename(columns={"_canon": "canon", "_hg": "hist_hg", "_ag": "hist_ag"})
    c["day"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d")
    c["pdate"] = c["af_date"].astype(str).str[:10]
    return c[cols]


# ---- 4. 90-minute score / 5. awarded results ----------------------------------------------------

def score_corrections(history: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                           pd.DataFrame, dict]:
    """(90-minute corrections, refused rows, awarded rows, counts) over the WHOLE history (`history`
    after the removal lists). See the module docstring, rules 4 and 5."""
    h = _with_canon(history)
    statuses = H.FINISHED | H.NOT_PLAYED          # AWD / WO and (fix9/defaults) abandoned ABD
    c = history_claims(h, pool, statuses)
    st = {"history_rows_claimed": int(len(c))}
    if c.empty:
        return pd.DataFrame(columns=SCORE_OUT_COLS), pd.DataFrame(columns=SCORE_REFUSED_COLS), \
            pd.DataFrame(), st
    two = c.groupby("fixture_id")["day"].transform("nunique") > 1
    st["claims_refused_fixture_claimed_from_two_dates"] = int(two.sum())
    c = c[~two]
    proven, refused, awarded = [], [], []
    for r in c.itertuples(index=False):
        hft, pft = _ft(r.hist_hg, r.hist_ag), _ft(r.provider_ft_home, r.provider_ft_away)
        base = {"league": r.league, "date": r.day, "home_team": r.home_team, "away_team": r.away_team,
                "fixture_id": int(r.fixture_id), "provider_status": r.status, "provider_date": r.pdate,
                "provider_home": r.provider_home, "provider_away": r.provider_away}
        if r.status in H.NOT_PLAYED:
            awarded.append({**base, "home_goals": r.hist_hg, "away_goals": r.hist_ag,
                            "provider_ft_home": pft[0] if pft else None,
                            "provider_ft_away": pft[1] if pft else None, "match_rule": r.match_rule})
            continue
        if hft is None or pft is None or hft == pft:
            continue
        ref = {**base, "history_ft_home": hft[0], "history_ft_away": hft[1],
               "provider_ft_home": pft[0], "provider_ft_away": pft[1], "verdict": "UNIDENTIFIABLE"}
        if r.status not in EXTRA_TIME:
            refused.append({**ref, "reason": "ft_status_scores_differ"})
            continue
        if hft[0] < pft[0] or hft[1] < pft[1]:
            refused.append({**ref, "reason": "history_below_90min_score"})
            continue
        et = _ft(r.et_home, r.et_away)
        et_check = ("extratime_not_stored" if et is None else "equals_90_plus_extratime"
                    if hft == (pft[0] + et[0], pft[1] + et[1]) else "differs_from_90_plus_extratime")
        pen = _ft(r.pen_home, r.pen_away)
        proven.append({**base, "history_ft_home": hft[0], "history_ft_away": hft[1],
                       "ft90_home": pft[0], "ft90_away": pft[1],
                       "et_home": et[0] if et else None, "et_away": et[1] if et else None,
                       "pen_home": pen[0] if pen else None, "pen_away": pen[1] if pen else None,
                       "match_rule": r.match_rule, "et_check": et_check,
                       "found_by": "provider_status_" + str(r.status)})
    prov = pd.DataFrame(proven, columns=SCORE_OUT_COLS)
    refd = pd.DataFrame(refused, columns=SCORE_REFUSED_COLS)
    awd = pd.DataFrame(awarded)
    xt = c[c["status"].isin(EXTRA_TIME)]
    st.update({
        "claims_of_aet_pen_fixtures": int(len(xt)),
        "aet_pen_history_already_90min": int(sum(_ft(a, b) == _ft(x, y) and _ft(a, b) is not None
                                                 for a, b, x, y in zip(xt["hist_hg"], xt["hist_ag"],
                                                                       xt["provider_ft_home"], xt["provider_ft_away"]))),
        "corrected_to_90min": int(len(prov)),
        "corrected_by_league": {k: int(v) for k, v in prov["league"].value_counts().items()} if len(prov) else {},
        "refused_by_reason": {k: int(v) for k, v in refd["reason"].value_counts().items()} if len(refd) else {},
        "awarded_rows": int(len(awd)),
        "pool_fixtures_by_status": {k: int(v) for k, v in pool["status"].value_counts().items()}})
    return prov, refd, awd, st


# ---- 6. owner decisions of fix9/defaults (2026-11-04) -------------------------------------------

#: Provider fixtures the OWNER ruled are not played matches although the provider lists them FT.
#: Every history row proven to be the fixture (same claim rule as above, any spelling) is excluded
#: like an awarded result, with this reason and the provider fixture id. Keyed by the PROVIDER's id;
#: the history rows and their recorded scores are found, never typed.
OWNER_NOT_PLAYED = {
    816403: ("Queretaro v Atlas, Liga MX 2022-03-05: abandoned after crowd violence and awarded - "
             "not a played match (owner 2026-11-04; the provider lists it FT, the history holds two "
             "copies: the awarded 0-3 and the 0-1 standing when play stopped)"),
}
#: why the provider + an agreeing history copy proves a same-day duplicate with a wrong score
AGREE_FOUND_BY = "same_day_copy_wrong_score_provider_agrees"


def owner_not_played(history: pd.DataFrame, pool: pd.DataFrame,
                     decisions: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """(history rows proven to be an OWNER_NOT_PLAYED fixture, counts), in the awarded list's layout
    plus `reason`. A decided fixture no history row claims is counted (listed nowhere)."""
    decisions = OWNER_NOT_PLAYED if decisions is None else decisions
    cols = KEY + ["home_goals", "away_goals", "fixture_id", "provider_status", "reason"]
    st = {"decided_fixtures": len(decisions), "rows": 0, "fixtures_not_claimed": []}
    if not decisions or pool.empty:
        st["fixtures_not_claimed"] = sorted(decisions)
        return pd.DataFrame(columns=cols), st
    sub = pool[pool["fixture_id"].isin(list(decisions))]
    statuses = set(sub["status"].dropna().unique()) or H.FINISHED
    h = near(_with_canon(history), sub, statuses)
    c = history_claims(h, sub, statuses)
    rows = [{"league": r.league, "date": r.day, "home_team": r.home_team, "away_team": r.away_team,
             "home_goals": r.hist_hg, "away_goals": r.hist_ag, "fixture_id": int(r.fixture_id),
             "provider_status": r.status, "reason": decisions[int(r.fixture_id)]}
            for r in c.itertuples(index=False) if int(r.fixture_id) in decisions]
    out = pd.DataFrame(rows, columns=cols)
    st["rows"] = int(len(out))
    st["fixtures_not_claimed"] = sorted(set(decisions) - set(out["fixture_id"].astype(int)))
    return out, st


def agreeing_copy_removals(history: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Owner rule 2026-11-04: "the provider plus an independent agreeing copy is proven". A provider
    fixture finished FT that two or more history rows of the SAME day claim (one match under two club
    spellings, clubs proven home->home / away->away) where at least one copy's FT EQUALS the provider's
    score.fulltime and another copy's FT differs: each differing copy is a duplicate with a wrong
    score -> removed (history_removed_duplicates.csv, found_by AGREE_FOUND_BY), the agreeing copy kept.
    AET / PEN fixtures are left to the 90-minute rule; one copy alone is never 'proven' either way
    (it stays in history_score_refused.csv, UNIDENTIFIABLE)."""
    st = {"same_day_multi_copy_fixtures_with_disagreement": 0, "removed_rows": 0, "removed_by_league": {}}
    h = near(_with_canon(history), pool, {"FT"})
    c = history_claims(h, pool, {"FT"})
    if c.empty:
        return pd.DataFrame(), st
    c = c[c.groupby("fixture_id")["day"].transform("nunique") == 1]
    removed = []
    for fid, g in c.groupby("fixture_id"):
        if len(g) < 2:
            continue
        pft = _ft(g["provider_ft_home"].iloc[0], g["provider_ft_away"].iloc[0])
        if pft is None:
            continue
        hft = [_ft(a, b) for a, b in zip(g["hist_hg"], g["hist_ag"])]
        agree = [i for i, x in enumerate(hft) if x == pft]
        differ = [i for i, x in enumerate(hft) if x is not None and x != pft]
        if not agree or not differ:
            continue
        st["same_day_multi_copy_fixtures_with_disagreement"] += 1
        k0 = g.iloc[sorted(agree, key=lambda i: (str(g["home_team"].iloc[i]), str(g["away_team"].iloc[i])))[0]]
        for i in differ:
            r = g.iloc[i]
            removed.append({"league": r["league"], "date": r["day"], "home_team": r["home_team"],
                            "away_team": r["away_team"], "home_goals": r["hist_hg"], "away_goals": r["hist_ag"],
                            "kept_date": k0["day"], "kept_home_team": k0["home_team"],
                            "kept_away_team": k0["away_team"], "kept_rows": len(agree),
                            "fixture_id": int(fid), "provider_date": r["pdate"],
                            "provider_home": r["provider_home"], "provider_away": r["provider_away"],
                            "provider_ft_home": pft[0], "provider_ft_away": pft[1], "provider_status": r["status"],
                            "found_by": AGREE_FOUND_BY})
    rem = pd.DataFrame(removed)
    if rem.empty:
        return rem, st
    # the FULL original row of each removed copy (key AND its recorded score: a repeated key keeps
    # only the copy with the wrong score)
    orig = history.copy()
    orig["date"] = pd.to_datetime(orig["date"]).dt.strftime("%Y-%m-%d")
    orig["_ftk"] = [_ft(a, b) for a, b in zip(orig["home_goals"], orig["away_goals"])]
    rem["_ftk"] = [_ft(a, b) for a, b in zip(rem["home_goals"], rem["away_goals"])]
    rem = rem.drop(columns=["home_goals", "away_goals"]).merge(orig, on=KEY + ["_ftk"], how="left")
    rem = rem.drop(columns=["_ftk"])
    lead = KEY + [c for c in history.columns if c not in KEY]
    rem = rem[lead + [c for c in rem.columns if c not in lead]]
    st["removed_rows"] = int(len(rem))
    st["removed_by_league"] = {k: int(v) for k, v in rem["league"].value_counts().items()}
    return rem, st


# ---- 3. every pair the loader merges -------------------------------------------------------------

_HK = ["_hk_league", "_hk_date", "_hk_home", "_hk_away"]


def loader_pairs(history: pd.DataFrame, corr: pd.DataFrame, scores: pd.DataFrame,
                 archive: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the duplicate pairs the loader's one-match step merges today, the frame they index), on
    fd_history (+ the backtest archive) after the lists `corr` / `scores`. Each row carries its
    history key (_hk_*) and source; the frame also carries `_filled` (the loader's 'most populated'
    count) for the survivor comparison."""
    from src import data_loader as DL
    from src.integrity.duplicate_matches import duplicate_pairs
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["_hk_league"], h["_hk_date"] = h["league"].astype(str), h["date"].dt.strftime("%Y-%m-%d")
    h["_hk_home"], h["_hk_away"] = h["home_team"].astype(str), h["away_team"].astype(str)
    parts = [h.assign(row_source="fd_cache")]
    if archive is not None and len(archive):
        parts.append(DL._drop_misfiled_league_rows(archive, "history-corrections:archive")
                     .assign(row_source="backtest_archive"))
    m = pd.concat(parts, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)
    x = DL.identity_before_one_match(m, corrections=corr, scores=scores)
    # the loader's survivor rule counts populated cells WITHOUT these helper columns
    x["_filled"] = x.drop(columns=_HK).notna().sum(axis=1)
    return duplicate_pairs(x.drop(columns=["_filled"])), x


def pair_removals(history: pd.DataFrame, pool: pd.DataFrame, corr: pd.DataFrame, scores: pd.DataFrame,
                  archive: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(removed rows with evidence, conflicts, counts) for rule 3 of the module docstring."""
    from src.history_corrections import HISTORY_SOURCES
    from src.integrity.duplicate_matches import SOURCE_PRIORITY
    pairs, x = loader_pairs(history, corr, scores, archive)
    st = {"loader_pairs": int(len(pairs)),
          "same_day_pairs_left_to_loader": int((pairs["gap_days"] == 0).sum()) if len(pairs) else 0}
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), st
    src = x["row_source"]
    both = pairs["i"].map(src).isin(HISTORY_SOURCES) & pairs["j"].map(src).isin(HISTORY_SOURCES)
    st["pairs_with_a_non_history_row_out_of_scope"] = int((~both & (pairs["gap_days"] > 0)).sum())
    pairs = pairs[both & (pairs["gap_days"] > 0)].reset_index(drop=True)
    st["history_pairs_one_to_three_days_apart"] = int(len(pairs))
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), st
    ix = sorted(set(pairs["i"]) | set(pairs["j"]))
    rows = x.loc[ix, _HK].copy()
    rows.columns = KEY
    orig = history.copy()
    orig["date"] = pd.to_datetime(orig["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ft = orig.drop_duplicates(KEY).set_index(KEY)[["home_goals", "away_goals"]]
    k = list(zip(rows["league"], rows["date"], rows["home_team"], rows["away_team"]))
    rows["home_goals"] = [ft.loc[t, "home_goals"] if t in ft.index else None for t in k]
    rows["away_goals"] = [ft.loc[t, "away_goals"] if t in ft.index else None for t in k]
    cl = history_claims(_with_canon(rows.drop_duplicates(KEY)), pool, H.FINISHED)
    by_key = {(r.league, r.day, r.home_team, r.away_team): r for r in cl.itertuples(index=False)}
    reg = _registry()
    prio = x["row_source"].map(SOURCE_PRIORITY).fillna(len(SOURCE_PRIORITY))
    removed, conflicts = [], []
    for p in pairs.itertuples(index=False):
        ki = tuple(x.loc[p.i, _HK])
        kj = tuple(x.loc[p.j, _HK])
        ci, cj = by_key.get(ki), by_key.get(kj)
        ev = {"loader_gap_days": int(p.gap_days)}
        why = None
        if ci is None or cj is None:
            why = "a_copy_claims_no_provider_fixture"
        elif ci.fixture_id != cj.fixture_id:
            why = "copies_claim_different_fixtures"
        else:
            # round 7's proof: the copies' FT agree (the history-vs-provider FT is rule 4's business)
            hi, hj = _ft(ci.hist_hg, ci.hist_ag), _ft(cj.hist_hg, cj.hist_ag)
            pdate = ci.pdate
            if hi is None or hi != hj:
                why = "score_differs_between_copies"
            elif abs((pd.Timestamp(ki[1]) - pd.Timestamp(kj[1])).days) != 1:
                why = f"dates_{abs((pd.Timestamp(ki[1]) - pd.Timestamp(kj[1])).days)}_days_apart"
            elif pdate not in (ki[1], kj[1]):
                why = "no_copy_on_provider_date"
            ev.update({"fixture_id": int(ci.fixture_id), "provider_date": pdate,
                       "provider_kickoff_utc": ci.kickoff, "provider_home": ci.provider_home,
                       "provider_away": ci.provider_away, "provider_ft_home": ci.provider_ft_home,
                       "provider_ft_away": ci.provider_ft_away, "provider_status": ci.status})
        if why:
            for kk, rr in ((ki, ci), (kj, cj)):
                conflicts.append({"league": kk[0], "date": kk[1], "home_team": kk[2], "away_team": kk[3],
                                  "home_goals": rr.hist_hg if rr is not None else None,
                                  "away_goals": rr.hist_ag if rr is not None else None,
                                  "conflict": why, "found_by": "loader_merge_pair", **ev})
            continue
        keep_i = ki[1] == ev["provider_date"]
        kept, gone, g_ix, k_ix = (ki, kj, p.j, p.i) if keep_i else (kj, ki, p.i, p.j)
        # the loader's own survivor (src.integrity.duplicate_matches.dedupe_matches order)
        order = sorted((p.i, p.j), key=lambda r: (prio[r], -x.at[r, "_filled"], x.at[r, "date"], str(r)))
        tz = (reg.get(LR_canon(kept[0])) or {}).get("timezone")
        removed.append({"league": gone[0], "date": gone[1], "home_team": gone[2], "away_team": gone[3],
                        "kept_date": kept[1], "kept_home_team": kept[2], "kept_away_team": kept[3],
                        "kept_rows": 1, "provider_timezone": tz,
                        "provider_local_date": _local_date(str(ev["provider_kickoff_utc"]), tz),
                        "loader_survivor_was": "kept_copy" if order[0] == k_ix else "removed_copy",
                        "found_by": "loader_merge_pair", **ev})
    rem, con = pd.DataFrame(removed), pd.DataFrame(conflicts)
    st["pairs_proven"] = int(len(rem))
    if len(rem):
        rem = rem.drop_duplicates(KEY)
        kept = set(zip(rem["league"], rem["kept_date"], rem["kept_home_team"], rem["kept_away_team"]))
        clash = [k in kept for k in zip(rem["league"], rem["date"], rem["home_team"], rem["away_team"])]
        if any(clash):              # a row kept by one pair and removed by another: not proven
            bad = rem[clash]
            st["removed_and_kept_clash"] = int(len(bad))
            con = pd.concat([con, bad[KEY + ["fixture_id"]].assign(
                conflict="row_kept_by_one_pair_removed_by_another", found_by="loader_merge_pair")],
                ignore_index=True)
            rem = rem[[not c for c in clash]]
        o = history.copy()
        o["date"] = pd.to_datetime(o["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        rem = rem.merge(o, on=KEY, how="left", validate="one_to_many")
        lead = KEY + [c for c in history.columns if c not in KEY]
        rem = rem[lead + [c for c in rem.columns if c not in lead]]
    st.update({"removed_rows": int(len(rem)),
               "removed_by_league": {k: int(v) for k, v in rem["league"].value_counts().items()} if len(rem) else {},
               "survivor_changes_vs_loader_tiebreak": 0,
               "conflict_rows": int(len(con)),
               "conflicts_by_reason": {k: int(v) for k, v in con["conflict"].value_counts().items()} if len(con) else {}})
    if len(rem):
        st["survivor_changes_vs_loader_tiebreak"] = int((rem.drop_duplicates(KEY)["loader_survivor_was"]
                                                         == "removed_copy").sum())
    return rem, con, st


def LR_canon(name):
    from src import league_registry as LR
    return LR.canonical_name(name) if isinstance(name, str) else None


def load_archive() -> pd.DataFrame:
    """The backtest archive exactly as the loader reads it (src.data_loader._load_backtest_archive)."""
    from src import data_loader as DL
    return DL._load_backtest_archive()


# ---- 2. cup rows -------------------------------------------------------------------------------

def unplaced_rows(history: pd.DataFrame, league_fx: pd.DataFrame) -> pd.DataFrame:
    """New-format history rows with NO provider fixture of their league within +-1 day, inside the
    first..last date the provider's league lists cover for that league. `league_fx`: league (canonical),
    date of every stored provider league fixture."""
    from src import league_registry as LR
    reg = _registry()
    h = history.copy()
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].isin(reg) & h["date"].notna()].copy()
    fx = league_fx.copy()
    fx["_d"] = pd.to_datetime(fx["date"], errors="coerce").dt.normalize()
    days = {lg: set(g["_d"].dropna()) for lg, g in fx.groupby("league")}
    span = {lg: (min(s), max(s)) for lg, s in days.items() if s}
    d = h["date"].dt.normalize()
    one = pd.Timedelta(days=1)
    out = []
    for i, lg, dd in zip(h.index, h["canon"], d):
        if lg not in span or not (span[lg][0] <= dd <= span[lg][1]):
            continue
        s = days[lg]
        if dd in s or dd - one in s or dd + one in s:
            continue
        out.append(i)
    return h.loc[out]


def league_fixture_days() -> pd.DataFrame:
    """Every stored provider fixture of a new-format league: step 1's lists + the HT store."""
    reg = _registry()
    ids = {int(e["api_football_id"]): n for n, e in reg.items()}
    parts = []
    comp = read_parquet(COMP_FIXTURES)
    if not comp.empty:
        c = comp[comp["league_id"].isin(ids)]
        parts.append(pd.DataFrame({"league": c["league_id"].map(ids), "date": c["date"]}))
    sc = read_parquet(H.SCORES)
    if not sc.empty:
        s = sc[sc["league_id"].isin(ids)]
        parts.append(pd.DataFrame({"league": s["league_id"].map(ids), "date": s["date"]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["league", "date"])


def prove_cup_rows(cand: pd.DataFrame, cup_fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(proven cup rows with evidence, unproven rows with reason). `cand` carries canon; `cup_fx` the
    stored national cup fixtures (with `country`)."""
    reg = _registry()
    if cand.empty:
        return pd.DataFrame(), pd.DataFrame()
    fx = cup_fx.copy() if not cup_fx.empty else pd.DataFrame(columns=CUP_FX_COLS)
    fx["_d"] = pd.to_datetime(fx["date"], errors="coerce").dt.normalize()
    by = {k: g for k, g in fx.groupby(["country", "_d"])}
    proven, unproven = [], []
    for i, r in cand.iterrows():
        country = (reg.get(r["canon"]) or {}).get("country")
        d = pd.Timestamp(r["date"]).normalize()
        base = {c: r[c] for c in cand.columns if c != "canon"}
        base["date"] = d.strftime("%Y-%m-%d")
        found, rule, why = None, None, "no_cup_fixture_stored_within_pm1_day"
        for ks, nm in (((0,), "same_day"), ((-1, 1), "adjacent_day")):
            parts = [by[(country, d + pd.Timedelta(days=k))] for k in ks if (country, d + pd.Timedelta(days=k)) in by]
            if not parts:
                continue
            pool = pd.concat(parts)
            hh = H._resolve(r["home_team"], pool["home_name"].dropna().unique())
            aa = H._resolve(r["away_team"], pool["away_name"].dropna().unique())
            if hh is None or aa is None:
                why = "club_not_resolved_among_cup_fixtures"
                continue
            f = pool[(pool["home_name"] == hh) & (pool["away_name"] == aa)]
            if len(f):
                found, rule = f, nm
                break
            why = "clubs_resolved_no_cup_fixture_pairing"
        if found is None:
            unproven.append({**base, "reason": why})
            continue
        if found["fixture_id"].nunique() > 1:
            unproven.append({**base, "reason": "ambiguous_more_than_one_cup_fixture"})
            continue
        f = found.iloc[0]
        hft, pft = _ft(r.get("home_goals"), r.get("away_goals")), _ft(f["ft_home"], f["ft_away"])
        if hft is not None and pft is not None and hft != pft:
            unproven.append({**base, "reason": "full_time_score_differs", "fixture_id": int(f["fixture_id"]),
                             "provider_ft_home": pft[0], "provider_ft_away": pft[1]})
            continue
        proven.append({**base, "cup_league_id": int(f["league_id"]), "cup_name": f["cup_name"],
                       "cup_season": str(f["season"]), "fixture_id": int(f["fixture_id"]),
                       "provider_date": str(f["date"])[:10], "provider_kickoff_utc": f["kickoff"],
                       "provider_home": f["home_name"], "provider_away": f["away_name"],
                       "provider_ft_home": pft[0] if pft else None,
                       "provider_ft_away": pft[1] if pft else None,
                       "provider_status": f["status"], "match_rule": rule,
                       "ft_check": "equal" if (hft and pft) else "one_side_missing"})
    return pd.DataFrame(proven), pd.DataFrame(unproven)


def _registry_league_ids() -> set:
    """api_football_id of EVERY registry league (any model type): never a 'cup' candidate."""
    from src import league_registry as LR
    return {int(e["api_football_id"]) for e in LR.leagues() if e.get("api_football_id") is not None}


def parse_cups(body: dict, country: str, registry_ids: set | None = None) -> list[dict]:
    """Domestic competitions of `country` OTHER than the registry's league(s), from /leagues?country:
    provider type 'Cup', or provider type 'League' that is not a registry league. Run-day 4
    (2026-11-05) proved the type filter too narrow: API-Football files 'Copa de la Liga Profesional'
    (Argentina, the 2023-08..2024-05 window of the 407 rows) as type 'League', so the cup-only list
    never held it and 0 of 407 rows could be proven. The proof itself is unchanged and strict (one
    fixture of that competition on the date, both clubs proven, same score), so listing the country's
    other competitions can only place a row on a match that really belongs to another competition.
    The country must be the registry country (anything else is not proof)."""
    reg = _registry_league_ids() if registry_ids is None else set(registry_ids)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for e in (body or {}).get("response") or []:
        lg, ct = e.get("league") or {}, e.get("country") or {}
        if ct.get("name") != country or lg.get("id") is None:
            continue
        if not (lg.get("type") == "Cup" or (lg.get("type") == "League" and int(lg["id"]) not in reg)):
            continue
        for s in e.get("seasons") or []:
            if s.get("year") is None or not s.get("start") or not s.get("end"):
                continue
            rows.append({"country": country, "league_id": int(lg["id"]), "name": lg.get("name"),
                         "type": lg.get("type"), "season": str(s["year"]), "start": str(s["start"]),
                         "end": str(s["end"]), "fetched_at": now, "list_kind": "all_types"})
    return rows


def parse_cup_fixtures(body: dict, cup: dict) -> list[dict]:
    rows = []
    for f in (body or {}).get("response") or []:
        fx, tm = f.get("fixture") or {}, f.get("teams") or {}
        st = (fx.get("status") or {}).get("short")
        if st not in FINISHED or fx.get("id") is None:
            continue
        ft = (f.get("score") or {}).get("fulltime") or {}
        pair = _ft(ft.get("home"), ft.get("away"))
        h, a = tm.get("home") or {}, tm.get("away") or {}
        ko = str(fx.get("date") or "")
        rows.append({"fixture_id": int(fx["id"]), "league_id": int(cup["league_id"]), "cup_name": cup["name"],
                     "country": cup["country"], "season": str(cup["season"]), "date": ko[:10], "kickoff": ko,
                     "home_id": h.get("id"), "home_name": h.get("name"), "away_id": a.get("id"),
                     "away_name": a.get("name"), "ft_home": pair[0] if pair else None,
                     "ft_away": pair[1] if pair else None, "status": st})
    return rows


def cup_plan(cand: pd.DataFrame, cups: pd.DataFrame, journal: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """(countries whose cup list must be fetched, cup-seasons to fetch). A country is fetched when no
    cup list is stored, or the stored one predates a candidate row. A cup-season is fetched when its
    [start-1d, end+1d] covers a candidate row of its country and it is not journaled after its end."""
    reg = _registry()
    if cand.empty:
        return [], []
    cand = cand.assign(country=cand["canon"].map(lambda c: (reg.get(c) or {}).get("country")))
    countries = []
    for c, g in cand.groupby("country"):
        got = cups[cups["country"] == c] if not cups.empty else cups
        # a list stored before 2026-11-05 is cup-only (no list_kind 'all_types'): fetch the full list once
        full = got["list_kind"].eq("all_types").any() if "list_kind" in got.columns else False
        if got.empty or not full or \
                pd.to_datetime(got["fetched_at"], utc=True).max().tz_localize(None) < g["date"].max():
            countries.append(c)
    done = set()
    if not journal.empty:
        ended = pd.to_datetime(journal["season_end"], errors="coerce") < \
            pd.to_datetime(journal["fetched_at"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        done = {(int(r.league_id), str(r.season)) for r in journal[ended].itertuples(index=False)}
    todo = []
    if not cups.empty:
        one = pd.Timedelta(days=1)
        for r in cups.drop_duplicates(["league_id", "season"]).itertuples(index=False):
            if (int(r.league_id), str(r.season)) in done:
                continue
            st, en = pd.to_datetime(r.start, errors="coerce"), pd.to_datetime(r.end, errors="coerce")
            if st != st or en != en:
                continue
            dates = cand.loc[cand["country"] == r.country, "date"]
            if ((dates >= st - one) & (dates <= en + one)).any():
                todo.append({"country": r.country, "league_id": int(r.league_id), "name": r.name,
                             "season": str(r.season), "end": en})
    return countries, todo


def fetch_cups(cand: pd.DataFrame, client: AFClient | None = None, dry_run: bool = False) -> dict:
    """Fetch the national cup lists and the cup-season fixture lists the candidates need."""
    st = {"calls": 0, "planned_country_calls": 0, "planned_fixture_calls": 0}
    countries, _ = cup_plan(cand, read_parquet(CUPS), read_parquet(CUP_JOURNAL))
    st["planned_country_calls"] = len(countries)
    if dry_run:
        _, todo = cup_plan(cand, read_parquet(CUPS), read_parquet(CUP_JOURNAL))
        st["planned_fixture_calls"] = len(todo)
        print(f"[cups] plan: {len(countries)} cup list call(s) {countries}; then fixture lists of the "
              f"cup-seasons covering the rows ({len(todo)} known now)")
        return st
    client = client or AFClient()
    try:
        for c in countries:
            if st["calls"] >= MAX_CALLS:
                raise BudgetStop(f"MAX_CALLS {MAX_CALLS} for this step")
            body = client.get("leagues", {"country": c})        # all types; parse_cups filters
            if body is None:
                continue
            st["calls"] += 1
            rows = parse_cups(body, c)
            if rows:
                append_parquet(CUPS, rows, ["league_id", "season"], CUPS_COLS)
            print(f"[cups] {c}: {len({r['league_id'] for r in rows})} domestic competition(s) other than "
                  f"the registry league(s) from the provider")
        _, todo = cup_plan(cand, read_parquet(CUPS), read_parquet(CUP_JOURNAL))
        st["planned_fixture_calls"] = len(todo)
        for t in todo:
            if st["calls"] >= MAX_CALLS:
                raise BudgetStop(f"MAX_CALLS {MAX_CALLS} for this step")
            body = client.get("fixtures", {"league": t["league_id"], "season": t["season"]})
            if body is None:
                continue
            st["calls"] += 1
            rows = parse_cup_fixtures(body, t)
            if rows:
                append_parquet(CUP_FIXTURES, rows, ["fixture_id"], CUP_FX_COLS)
            append_parquet(CUP_JOURNAL, [{"league_id": t["league_id"], "season": t["season"],
                                          "country": t["country"], "season_end": t["end"].date().isoformat(),
                                          "n_fixtures": len(body.get("response") or []), "n_finished": len(rows),
                                          "fetched_at": datetime.now(timezone.utc).isoformat()}],
                           ["league_id", "season"], CUP_JOURNAL_COLS)
            print(f"[cups] {t['name']} (id {t['league_id']}) {t['season']}: {len(rows)} finished fixture(s)")
    except BudgetStop as e:
        print(f"[cups] budget stop: {e} (resumable)")
        st["budget_stop"] = str(e)
    finally:
        client.flush()
    return st


# ---- run ---------------------------------------------------------------------------------------

def _write(df: pd.DataFrame, path: Path, cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (df if len(df) else pd.DataFrame(columns=cols)).to_csv(path, index=False)


def _as_corrections(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """A list frame in src.history_corrections.read_corrections' layout (in memory)."""
    cols = KEY + ["home_goals", "away_goals", "correction", "kept_date", "kept_home_team", "kept_away_team"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["correction"] = kind
    return d.reindex(columns=cols)


def _scores_frame(sc: pd.DataFrame) -> pd.DataFrame:
    return sc[KEY + ["history_ft_home", "history_ft_away", "ft90_home", "ft90_away"]] if len(sc) else \
        pd.DataFrame(columns=KEY + ["history_ft_home", "history_ft_away", "ft90_home", "ft90_away"])


def follow_ups(history: pd.DataFrame, scores: pd.DataFrame, rem: pd.DataFrame, cups: pd.DataFrame,
               archive: pd.DataFrame | None = None) -> dict:
    """Rules 3-5 (fix8/history2). Two passes, because the loader applies the 90-minute list before
    its one-match step and a proven duplicate removal decides which copy is corrected:
      1. scores and awarded rows on the history after the round-7 lists;
      2. the loader's pairs under those lists -> loader-merge duplicates;
      3. scores and awarded rows again on the history after ALL removal lists (the final lists)."""
    from src.history_corrections import apply_history_corrections
    pool = provider_pool(scores)
    base = pd.concat([_as_corrections(rem, "duplicate_removed"), _as_corrections(cups, "cup_excluded")],
                     ignore_index=True)
    none = _scores_frame(pd.DataFrame())
    h1 = apply_history_corrections(history, corr=base, scores=none)
    sc1, _, awd1, _ = score_corrections(h1, pool)
    corr1 = pd.concat([base, _as_corrections(awd1, "awarded_excluded")], ignore_index=True)
    prem, pcon, pst = pair_removals(history, pool, corr1, _scores_frame(sc1), archive)
    corr2 = pd.concat([corr1, _as_corrections(prem, "duplicate_removed")], ignore_index=True)
    h2 = apply_history_corrections(history, corr=corr2, scores=none)
    # fix9/defaults (owner 2026-11-04): the fixtures the owner ruled not played, then the same-day
    # copies proven wrong by the provider plus an agreeing copy - before the final score lists, so a
    # removed copy is no longer listed as an UNIDENTIFIABLE score disagreement.
    onp, onp_st = owner_not_played(h2, pool)
    corr3 = pd.concat([corr2, _as_corrections(onp, "awarded_excluded")], ignore_index=True)
    h3 = apply_history_corrections(history, corr=corr3, scores=none)
    agree, agree_st = agreeing_copy_removals(h3, pool)
    corr4 = pd.concat([corr3, _as_corrections(agree, "duplicate_removed")], ignore_index=True)
    h4 = apply_history_corrections(history, corr=corr4, scores=none)
    sc, refused, awd, sst = score_corrections(h4, pool)
    return {"pool": pool, "pair_removed": prem, "pair_conflicts": pcon, "pair_stats": pst,
            "score": sc, "score_refused": refused, "awarded": awd, "score_stats": sst,
            "owner_not_played": onp, "owner_not_played_stats": onp_st,
            "agree_removed": agree, "agree_stats": agree_st}


def run(fetch: bool = False, dry_run: bool = False, client: AFClient | None = None) -> dict:
    history = load_history()
    audit: dict = {"history": HISTORY.relative_to(ROOT).as_posix(), "history_rows": int(len(history))}
    scores = provider_scores()
    rem, con, dst = duplicate_removals(history, scores)
    audit["duplicates"] = dst
    cand = unplaced_rows(history, league_fixture_days())
    audit["cups"] = {"unplaced_rows": int(len(cand)),
                     "unplaced_by_league": {k: int(v) for k, v in cand["canon"].value_counts().items()},
                     "unplaced_lacking_ht": int((cand["ht_home_goals"].isna() | cand["ht_away_goals"].isna()).sum())
                     if len(cand) else 0}
    if fetch or dry_run:
        audit["cups"]["fetch"] = fetch_cups(cand, client=client, dry_run=dry_run)
    if dry_run:
        print(json.dumps(audit, indent=1, default=str))
        return audit
    proven, unproven = prove_cup_rows(cand, read_parquet(CUP_FIXTURES))
    audit["cups"].update({
        "proven_cup_rows": int(len(proven)), "unproven_rows": int(len(unproven)),
        "proven_by_league_cup": ({f"{a} / {b}": int(n) for (a, b), n in
                                  proven.groupby(["league", "cup_name"]).size().items()} if len(proven) else {}),
        "unproven_by_reason": ({k: int(v) for k, v in unproven["reason"].value_counts().items()}
                               if len(unproven) else {})})
    fu = follow_ups(history, scores, rem, proven, load_archive())
    audit["loader_merge_duplicates"] = fu["pair_stats"]
    audit["score_90min"] = fu["score_stats"]
    audit["owner_not_played"] = fu["owner_not_played_stats"]
    audit["same_day_copy_provider_agrees"] = fu["agree_stats"]
    rem = pd.concat([rem, fu["pair_removed"]], ignore_index=True) if len(fu["pair_removed"]) else rem
    rem = pd.concat([rem, fu["agree_removed"]], ignore_index=True) if len(fu["agree_removed"]) else rem
    con = pd.concat([con, fu["pair_conflicts"]], ignore_index=True) if len(fu["pair_conflicts"]) else con
    hist_cols = list(history.columns)
    _write(rem, DUP_OUT, hist_cols + ["kept_date", "kept_home_team", "kept_away_team", "fixture_id", "found_by"])
    _write(con, DUP_CONFLICTS, KEY + ["home_goals", "away_goals", "conflict", "fixture_id", "found_by"])
    _write(proven, CUP_OUT, hist_cols + ["cup_league_id", "cup_name", "fixture_id"])
    _write(unproven, CUP_UNPROVEN, hist_cols + ["reason"])
    _write(fu["score"], SCORE_OUT, SCORE_OUT_COLS)
    _write(fu["score_refused"], SCORE_REFUSED, SCORE_REFUSED_COLS)
    awd = fu["awarded"].copy()
    if len(awd):
        awd["reason"] = "provider status " + awd["provider_status"].astype(str) + ": result not played on the pitch"
    awd = pd.concat([d for d in (awd, fu["owner_not_played"]) if len(d)], ignore_index=True) \
        if len(awd) or len(fu["owner_not_played"]) else pd.DataFrame()
    _write(awd, AWARDED_OUT, AWARDED_COLS)
    audit["written_at"] = datetime.now(timezone.utc).isoformat()
    AUDIT.write_text(json.dumps(audit, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(f"[history-corrections] {json.dumps({k: v for k, v in audit.items()}, default=str)[:2000]}")
    return audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the owner's history corrections (cup rows, duplicates)")
    ap.add_argument("--fetch", action="store_true", help="fetch national cup fixture lists (budget-guarded)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run(fetch=a.fetch, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
