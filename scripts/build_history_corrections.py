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
AUDIT = OUT_DIR / "history_corrections_audit.json"
COMP_FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
CUPS = COLLECT_DIR / "national_cups.parquet"
CUP_FIXTURES = COLLECT_DIR / "national_cup_fixtures.parquet"
CUP_JOURNAL = COLLECT_DIR / "national_cup_seasons.parquet"
MAX_CALLS = 40
KEY = ["league", "date", "home_team", "away_team"]
FINISHED = {"FT", "AET", "PEN"}
CUPS_COLS = ["country", "league_id", "name", "type", "season", "start", "end", "fetched_at"]
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
    if not rem.empty:
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


def parse_cups(body: dict, country: str) -> list[dict]:
    """National cups of `country` from /leagues?country&type=cup: provider type must be 'Cup' and its
    country the registry country (anything else is not proof of a domestic cup)."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for e in (body or {}).get("response") or []:
        lg, ct = e.get("league") or {}, e.get("country") or {}
        if lg.get("type") != "Cup" or ct.get("name") != country or lg.get("id") is None:
            continue
        for s in e.get("seasons") or []:
            if s.get("year") is None or not s.get("start") or not s.get("end"):
                continue
            rows.append({"country": country, "league_id": int(lg["id"]), "name": lg.get("name"),
                         "type": lg.get("type"), "season": str(s["year"]), "start": str(s["start"]),
                         "end": str(s["end"]), "fetched_at": now})
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
        if got.empty or pd.to_datetime(got["fetched_at"], utc=True).max().tz_localize(None) < g["date"].max():
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
            body = client.get("leagues", {"country": c, "type": "cup"})
            if body is None:
                continue
            st["calls"] += 1
            rows = parse_cups(body, c)
            if rows:
                append_parquet(CUPS, rows, ["league_id", "season"], CUPS_COLS)
            print(f"[cups] {c}: {len({r['league_id'] for r in rows})} national cup(s) from the provider")
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


def run(fetch: bool = False, dry_run: bool = False, client: AFClient | None = None) -> dict:
    history = load_history()
    audit: dict = {"history": HISTORY.relative_to(ROOT).as_posix(), "history_rows": int(len(history))}
    rem, con, dst = duplicate_removals(history, provider_scores())
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
    hist_cols = list(history.columns)
    _write(rem, DUP_OUT, hist_cols + ["kept_date", "kept_home_team", "kept_away_team", "fixture_id"])
    _write(con, DUP_CONFLICTS, KEY + ["home_goals", "away_goals", "conflict", "fixture_id"])
    _write(proven, CUP_OUT, hist_cols + ["cup_league_id", "cup_name", "fixture_id"])
    _write(unproven, CUP_UNPROVEN, hist_cols + ["reason"])
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
