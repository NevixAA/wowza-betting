"""
build_af_history_corrections.py - the owner's duplicate rule (2026-11-04) for output/af_history.parquet,
PROVEN from stored provider data. 0 API calls.

    python scripts/build_af_history_corrections.py              # write the two lists
    python scripts/build_af_history_corrections.py --dry-run    # print the counts; write nothing

Every af_history row of a registry league with an API-Football id is matched to its provider fixture
exactly as scripts/collect_af_history_gap.py matches it (the row's own fixture id, else the collected
fixture lists - output/strength/competition_fixtures.parquet and output/collect/af_ht_scores.parquet -
same league id, same day else the adjacent day, clubs resolved home->home / away->away by
src.team_names.resolve_result, more than one candidate refused), but the fixtures claimed by rows of
TWO dates are kept instead of refused. Each such fixture is a candidate pair; it is PROVEN when
  * every copy claims the SAME provider fixture id (by construction) and both clubs of every copy are
    that fixture's clubs: by the provider's team ids where the row stores them, else by name as the
    matcher resolved them (a row matched by its own fixture id without team ids is resolved by name
    against the provider's clubs of that league and day);
  * the copies are exactly one day apart and at least one copy is on the provider's fixture date
    (the UTC date of the kick-off, af_ht_scores.kickoff, else the list's date);
  * the full-time score agrees between the copies and with the provider's score.fulltime wherever
    both hold one.
Then every copy NOT on the provider date is removed (src/af_history_corrections applies it); the kept
copy is the one on the provider date (a provider row before a written-back one where the provider date
holds two spellings). Anything else is a conflict: listed, left alone.

PASS 2 (fix15/dates, owner 2026-11-07, the same proof): one provider fixture held on the PROVIDER's
date by two or more af_history rows under different club spellings ('Club Leon' / 'Leon'; mostly
rows the old loader bank wrote back under the football-data spelling beside the provider's). Proven
when every copy claims the same provider fixture id (by construction), both clubs of every copy are
the fixture's clubs (as above) and the full-time score agrees between the copies and with the
provider wherever both hold one. One copy stays (the same order as pass 1: provider row, fixture id,
most statistics, spelling; for a fixture of pass 1 it is pass 1's kept copy) and every other copy is
removed (found_by af_same_day_spelling), its provider values filling the kept copy's empty cells
(src.af_history_corrections.plan_fill: a written-back copy gives nothing; two provider copies holding
different values make the cell missing - the provider's own fixture statistics answer, the
--fullstats / --insidebox sidecars, fills it when held). A fixture whose pass-1 pair is unproven is
left alone; an unproven group is a conflict (listed, left alone).

OUTPUT (af_history.parquet is never rewritten):
  output/corrections/af_history_removed_duplicates.csv   every removed copy with ALL its original
      af_history values, the kept copy's key, the evidence and the effect on the kept copy
      (src.af_history_corrections.plan_fill: cells filled / made missing / unproven cells cleared).
  output/corrections/af_history_duplicate_conflicts.csv   every copy of an unproven pair + reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_af_history_gap as G  # noqa: E402
from collect_common import read_parquet  # noqa: E402
from src import af_history_corrections as AC  # noqa: E402

COMP_FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
HT_SCORES = ROOT / "output" / "collect" / "af_ht_scores.parquet"
KEY = AC.KEY
EVIDENCE = ["provider_fixture_id", "provider_date", "provider_kickoff_utc", "provider_local_date",
            "provider_timezone", "provider_home", "provider_away", "provider_home_id", "provider_away_id",
            "provider_ft_home", "provider_ft_away", "provider_status"]
REMOVED_EXTRA = (AC.KEPT + ["kept_rows_on_provider_date", "removed_is_written_back", "kept_is_written_back",
                            "removed_match_rule", "kept_match_rule", "removed_club_proof", "kept_club_proof",
                            "removed_copy_is_local_date"] + EVIDENCE
                 + ["cells_filled", "cells_made_missing", "unproven_cells_cleared", "found_by"])
CONFLICT_COLS = KEY + ["FTHG", "FTAG", "fixture_id", "written_back", "match_rule", "conflict",
                       "found_by"] + EVIDENCE
#: found_by of pass 1 (fix14/afdupes) and pass 2 (fix15/dates)
FOUND_TWO_DATES = "af_claim_two_dates"
FOUND_SAME_DAY = "af_same_day_spelling"


def _timezones() -> dict:
    from src import league_registry as LR
    return {e["name"]: e.get("timezone") for e in LR.leagues()}


def _local_date(kickoff, tz):
    t = pd.to_datetime(kickoff, errors="coerce", utc=True)
    if t is pd.NaT or t != t or not tz:
        return None
    try:
        return t.tz_convert(tz).strftime("%Y-%m-%d")
    except Exception:                                                  # noqa: BLE001
        return None


def provider_records(comp: pd.DataFrame | None = None, ht: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per provider fixture id: the HT store first (it carries the UTC kick-off), then the
    league fixture lists. provider_date = the kick-off's UTC date, else the list date."""
    comp = read_parquet(COMP_FIXTURES) if comp is None else comp
    ht = read_parquet(HT_SCORES) if ht is None else ht
    cols = ["fixture_id", "league_id", "date", "kickoff", "home_id", "home_name", "away_id", "away_name",
            "ft_home", "ft_away", "status"]
    parts = [x.reindex(columns=cols) for x in (ht, comp) if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame(columns=cols + ["provider_date"])
    p = pd.concat(parts, ignore_index=True).dropna(subset=["fixture_id"])
    p["fixture_id"] = p["fixture_id"].astype(int)
    p = p.drop_duplicates("fixture_id", keep="first").set_index("fixture_id", drop=False)
    k = pd.to_datetime(p["kickoff"], errors="coerce", utc=True)
    p["provider_date"] = k.dt.strftime("%Y-%m-%d").where(k.notna(),
                                                         pd.to_datetime(p["date"], errors="coerce")
                                                         .dt.strftime("%Y-%m-%d"))
    return p


def _int(v):
    try:
        return None if v is None or v != v else int(v)
    except (TypeError, ValueError):
        return None


def _ft(a, b):
    a, b = _int(a), _int(b)
    return None if a is None or b is None else (a, b)


def club_proof(row: dict, prov, day_pool: dict) -> str | None:
    """'team_ids' / 'names' when the row's two clubs are the provider fixture's, else None."""
    from src.team_names import resolve_result
    hid, aid = _int(row.get("home_team_id")), _int(row.get("away_team_id"))
    if hid is not None and aid is not None:
        return "team_ids" if (hid, aid) == (_int(prov["home_id"]), _int(prov["away_id"])) else None
    if row.get("match_rule") in ("same_day", "adjacent_day"):
        return "names"                       # resolved home->home / away->away by the matcher itself
    pool_h, pool_a = day_pool.get(int(prov["fixture_id"]), ([prov["home_name"]], [prov["away_name"]]))
    h = resolve_result(str(row["home_team"]), [str(x) for x in pool_h])
    a = resolve_result(str(row["away_team"]), [str(x) for x in pool_a])
    return "names" if (h == prov["home_name"] and a == prov["away_name"]) else None


def _kept_order(r: dict) -> tuple:
    """Which copy on the provider date stays: a provider row before a written-back one, a row with a
    fixture id first, then the most statistic cells, then the spelling (stable)."""
    return (r["_wb"], _int(r.get("fixture_id")) is None,
            -sum(0 if AC._empty(r.get(c)) else 1 for c in AC.STAT_COLS),
            str(r["home_team"]), str(r["away_team"]))


def build(af: pd.DataFrame | None = None, fixtures: pd.DataFrame | None = None,
          prov: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(removed rows, conflict rows, counts)."""
    from src import league_registry as LR
    from src.af_history_provenance import written_back_mask
    af = pd.read_parquet(G.AF_HISTORY) if af is None else af
    ids = G._af_ids()
    h = af.copy()
    for c in ("fixture_id", "home_team_id", "away_team_id"):
        if c not in h.columns:
            h[c] = None
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].isin(ids)].copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h = h[h["date"].notna() & h["home_team"].notna() & h["away_team"].notna()]
    h["league_id"] = h["canon"].map(ids).astype(int)
    fx = G.fixture_index() if fixtures is None else fixtures
    m, _ = G.match_fixtures(h, fx, refuse_claimed_twice=False)
    prov = provider_records() if prov is None else prov
    tz = _timezones()
    m = m.copy()
    m["day"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
    two = m[m["claimed_twice"]].copy()
    orig = af.copy()
    orig["_day"] = pd.to_datetime(orig["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    orig_by_key = {k: r for k, r in zip(zip(orig["league"].astype(str), orig["_day"],
                                             orig["home_team"].astype(str), orig["away_team"].astype(str)),
                                         orig.drop(columns="_day").to_dict("records"))}
    wb_by_key = dict(zip(zip(orig["league"].astype(str), orig["_day"], orig["home_team"].astype(str),
                             orig["away_team"].astype(str)), written_back_mask(af).tolist()))
    # the provider's clubs of the league on the fixture's day +-1 (resolve pool for id-matched rows)
    fxd = fx.groupby(["league_id", "_d"]) if len(fx) else None
    removed, conflicts = [], []
    st: dict = defaultdict(Counter)

    def _rows(g: pd.DataFrame) -> list[dict]:
        out = []
        for r in g.to_dict("records"):
            k = (str(r["league"]), r["day"], str(r["home_team"]), str(r["away_team"]))
            out.append({**orig_by_key[k], "match_rule": r["match_rule"], "day": r["day"], "_key": k,
                        "_wb": bool(wb_by_key[k])})
        return out

    def _evidence(fid: int, lg: str):
        """(provider record or None, evidence dict, resolve pool)."""
        p = prov.loc[int(fid)] if int(fid) in prov.index else None
        ev = {"provider_fixture_id": int(fid)}
        pool = {}
        if p is None:
            return p, ev, pool
        pdate = str(p["provider_date"])
        ev.update({"provider_date": pdate, "provider_kickoff_utc": p["kickoff"],
                   "provider_local_date": _local_date(p["kickoff"], tz.get(lg)),
                   "provider_timezone": tz.get(lg), "provider_home": p["home_name"],
                   "provider_away": p["away_name"], "provider_home_id": _int(p["home_id"]),
                   "provider_away_id": _int(p["away_id"]), "provider_ft_home": _int(p["ft_home"]),
                   "provider_ft_away": _int(p["ft_away"]), "provider_status": p["status"]})
        if fxd is not None:
            d0 = pd.Timestamp(pdate)
            parts = [fxd.get_group((p["league_id"], d0 + pd.Timedelta(days=k)))
                     for k in (-1, 0, 1) if (p["league_id"], d0 + pd.Timedelta(days=k)) in fxd.groups]
            if parts:
                pp = pd.concat(parts)
                pool[int(fid)] = (pp["home_name"].unique().tolist(), pp["away_name"].unique().tolist())
        return p, ev, pool

    def _score_why(rows: list[dict], p) -> str | None:
        fts = {_ft(r.get("FTHG"), r.get("FTAG")) for r in rows} - {None}
        pft = _ft(p["ft_home"], p["ft_away"])
        if any(r["_proof"] is None for r in rows):
            return "clubs_not_proven"
        if len(fts) > 1:
            return "score_differs_between_copies"
        if fts and pft is not None and fts != {pft}:
            return "score_differs_from_provider"
        return None

    def _conflict(rows, why, ev, found_by):
        for r in rows:
            conflicts.append({**{c: r.get(c) for c in KEY}, "date": r["day"], "FTHG": r.get("FTHG"),
                              "FTAG": r.get("FTAG"), "fixture_id": r.get("fixture_id"),
                              "written_back": r["_wb"], "match_rule": r["match_rule"],
                              "conflict": why, "found_by": found_by, **ev})

    def _remove(r, k0, kept_now, n_kept, ev, lg, found_by, prefix):
        new, filled, missing, cleared = AC.plan_fill(kept_now, r)
        kept_now.update(new)
        st[lg][f"{prefix}rows_removed"] += 1
        st[lg][f"{prefix}removed_written_back"] += int(r["_wb"])
        st[lg][f"{prefix}kept_written_back"] += int(k0["_wb"])
        st[lg][f"{prefix}cells_filled"] += len(filled)
        st[lg][f"{prefix}cells_made_missing"] += len(missing)
        st[lg][f"{prefix}unproven_cells_cleared"] += len(cleared)
        out = {c: v for c, v in r.items() if not str(c).startswith("_") and c not in ("match_rule", "day")}
        out["date"] = r["day"]
        out.update({"kept_date": k0["day"], "kept_home_team": k0["home_team"],
                    "kept_away_team": k0["away_team"], "kept_rows_on_provider_date": n_kept,
                    "removed_is_written_back": r["_wb"], "kept_is_written_back": k0["_wb"],
                    "removed_match_rule": r["match_rule"], "kept_match_rule": k0["match_rule"],
                    "removed_club_proof": r["_proof"], "kept_club_proof": k0["_proof"],
                    "removed_copy_is_local_date": (ev["provider_local_date"] == r["day"])
                    if ev.get("provider_local_date") else None, **ev,
                    "cells_filled": ";".join(filled), "cells_made_missing": ";".join(missing),
                    "unproven_cells_cleared": ";".join(cleared), "found_by": found_by})
        removed.append(out)

    # ---- pass 1 (fix14/afdupes): one fixture claimed from TWO dates -------------------------------
    kept_state: dict = {}            # proven fixture -> (kept copy, its values after the fills)
    unproven_two: set = set()
    for fid, g in two.groupby("fixture_id"):
        lg = str(g["canon"].iloc[0])
        st[lg]["pairs"] += 1
        st[lg]["rows_in_pairs"] += len(g)
        rows = _rows(g)
        p, ev, pool = _evidence(fid, lg)
        why = None
        if p is None:
            why = "provider_fixture_not_in_lists"
        else:
            pdate = str(p["provider_date"])
            for r in rows:
                r["_proof"] = club_proof(r, p, pool)
            days = sorted({r["day"] for r in rows})
            gap = (pd.Timestamp(days[-1]) - pd.Timestamp(days[0])).days
            why = _score_why(rows, p)
            if why is None and (gap != 1 or len(days) != 2):
                why = f"dates_{gap}_days_apart"
            elif why is None and pdate not in days:
                why = "no_copy_on_provider_date"
        if why:
            st[lg][f"conflict_{why}"] += 1
            unproven_two.add(int(fid))
            _conflict(rows, why, ev, FOUND_TWO_DATES)
            continue
        kept = sorted([r for r in rows if r["day"] == pdate], key=_kept_order)
        k0 = kept[0]
        kept_now = {c: v for c, v in k0.items() if not str(c).startswith("_")}
        st[lg]["proven"] += 1
        for r in rows:
            if r["day"] != pdate:
                _remove(r, k0, kept_now, len(kept), ev, lg, FOUND_TWO_DATES, "")
        kept_state[int(fid)] = (k0, kept_now)

    # ---- pass 2 (fix15/dates): one fixture held on the provider date under two or more spellings --
    m["pdate"] = [str(prov.at[int(f), "provider_date"]) if int(f) in prov.index else None
                  for f in m["fixture_id"]]
    on = m[m["day"] == m["pdate"]]
    on = on[on.groupby("fixture_id")["day"].transform("size") > 1]
    for fid, g in on.groupby("fixture_id"):
        lg = str(g["canon"].iloc[0])
        st[lg]["same_day_groups"] += 1
        st[lg]["same_day_rows_in_groups"] += len(g)
        if int(fid) in unproven_two:
            st[lg]["same_day_left_two_date_pair_unproven"] += 1
            continue
        rows = _rows(g)
        p, ev, pool = _evidence(fid, lg)
        for r in rows:
            r["_proof"] = club_proof(r, p, pool)
        why = _score_why(rows, p)
        if why:
            st[lg][f"same_day_conflict_{why}"] += 1
            _conflict(rows, why, ev, FOUND_SAME_DAY)
            continue
        if int(fid) in kept_state:
            k0, kept_now = kept_state[int(fid)]
        else:
            k0 = sorted(rows, key=_kept_order)[0]
            kept_now = {c: v for c, v in k0.items() if not str(c).startswith("_")}
        st[lg]["same_day_proven"] += 1
        for r in rows:
            if r["_key"] == k0["_key"]:
                continue
            _remove(r, k0, kept_now, len(rows), ev, lg, FOUND_SAME_DAY, "same_day_")
    hist_cols = list(af.columns)
    rem = pd.DataFrame(removed, columns=hist_cols + REMOVED_EXTRA)
    con = pd.DataFrame(conflicts, columns=CONFLICT_COLS)
    stats = {"history_rows": int(len(h)), "matched_rows": int(len(m)),
             "rows_claiming_a_fixture_from_two_dates": int(len(two)),
             "rows_in_same_day_groups": int(len(on)),
             "per_league": {k: dict(v) for k, v in sorted(st.items())},
             "total": dict(sum((Counter(v) for v in st.values()), Counter())), "calls": 0}
    return rem, con, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="af_history duplicate rule (owner 2026-11-04), 0 API calls")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rem, con, stats = build()
    print(json.dumps(stats, indent=1, default=str))
    if a.dry_run:
        return 0
    AC.CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rem.to_csv(AC.AF_DUP_REMOVED, index=False)
    con.to_csv(AC.AF_DUP_CONFLICTS, index=False)
    print(f"[af-history-corrections] {len(rem)} removed copy(ies) -> {AC.AF_DUP_REMOVED.name}; "
          f"{len(con)} conflict row(s) -> {AC.AF_DUP_CONFLICTS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
