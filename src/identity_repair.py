"""
Club identity repair for the v9_repaired recipe (Standard forensic repair, 2026-10-28).
======================================================================================

THE DEFECT (found by the Pro data audit, R2): the training frame stores thousands of Standard
fixtures TWICE — once per source spelling (Hamburg / Hamburger SV, Hull / Hull City,
Sheffield Weds / Sheffield Wednesday, Sp Gijon / Sporting Gijon, ...). data_loader's
`_canonicalise_clubs` only merges spellings that normalise to the same key, and these do not.
Each copy is a separate "team" to the feature builder, so the duplicate rows double-count
matches AND each club's rolling history is split in two (invariant 11 / brief test 14).

THE REPAIR is evidence-based, never fuzzy:
  * an explicit alias map {league: {alias: canonical}} may be supplied (R2 publishes one at
    Pro output/standard_repair/identity_aliases.json, version identity/1.0), and/or
  * `derive_same_day_aliases` finds pairs from the data itself: two rows in the same league on
    the same DATE with the same SCORE whose home names differ AND away names differ are the
    same match twice; a name pair becomes an alias only when that happens on at least
    `min_dates` distinct dates and the pairing is one-to-one within the league.
Then rows are deduplicated on (date, league, home, away), keeping the most complete row, and
fixtures whose copies disagree on the score are QUARANTINED (dropped and reported), not guessed.

Legacy production never calls this module.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_VERSION = "v9r-identity-1.0"


def load_alias_map(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(lg): {str(k): str(v) for k, v in (m or {}).items()}
            for lg, m in (d.get("map", d) or {}).items()}


def derive_same_day_aliases(df: pd.DataFrame, min_dates: int = 3) -> dict:
    """{league: {alias: canonical}} from same-day same-score duplicate evidence."""
    d = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    d["_d"] = pd.to_datetime(d["date"]).dt.normalize()
    ev: dict = defaultdict(Counter)       # league -> Counter[(a, b)] (a < b) over distinct dates
    last_seen: dict = {}
    for (lg, _dt, _hg, _ag), g in d.groupby(["league", "_d", "home_goals", "away_goals"], sort=False):
        if len(g) < 2:
            continue
        rows = list(zip(g["home_team"].astype(str), g["away_team"].astype(str)))
        seen = set()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (h1, a1), (h2, a2) = rows[i], rows[j]
                if h1 != h2 and a1 != a2:
                    for x, y in ((h1, h2), (a1, a2)):
                        k = tuple(sorted((x, y)))
                        if k not in seen:
                            seen.add(k)
                            ev[lg][k] += 1
    for (lg, t), dt in pd.concat([
            d[["league", "home_team", "_d"]].rename(columns={"home_team": "t"}),
            d[["league", "away_team", "_d"]].rename(columns={"away_team": "t"})]
    ).groupby(["league", "t"])["_d"].max().items():
        last_seen[(lg, t)] = dt
    played_pairs: dict = defaultdict(set)
    for lg, h, a in zip(d["league"], d["home_team"].astype(str), d["away_team"].astype(str)):
        played_pairs[lg].add(tuple(sorted((h, a))))
    out: dict = {}
    for lg, cnt in ev.items():
        # Evidence alone is not enough: two DIFFERENT matches with the same score on the same
        # day recur over seasons (measured: Ajax/Feyenoord, Racing/River would pass a pure count).
        # The names must also be the same club by src/team_names' league-scoped resolver.
        # And two spellings of ONE club never play each other (Yokohama F. Marinos v Yokohama FC
        # passes the resolver but is a derby between two clubs).
        met = played_pairs.get(lg, set())
        strong = [(k, n) for k, n in cnt.items()
                  if n >= min_dates and k not in met and _same_club(*k)]
        partners: dict = defaultdict(list)
        for (x, y), n in strong:
            partners[x].append((n, y))
            partners[y].append((n, x))
        m = {}
        for (x, y), n in strong:
            # one-to-one: each name's strongest partner must be the other
            if max(partners[x])[1] != y or max(partners[y])[1] != x:
                continue
            canon, alias = sorted((x, y), key=lambda t: (last_seen.get((lg, t), pd.Timestamp(0)), t),
                                  reverse=True)
            m[alias] = canon
        if m:
            out[lg] = m
    return out


def _same_club(x: str, y: str) -> bool:
    from src.team_names import resolve
    return resolve(x, [y]) == y or resolve(y, [x]) == x


def merge_maps(*maps: dict) -> dict:
    out: dict = defaultdict(dict)
    for mp in maps:
        for lg, m in (mp or {}).items():
            out[lg].update(m)
    return {lg: _resolve_chains(m) for lg, m in out.items()}


def _resolve_chains(m: dict) -> dict:
    res = {}
    for a in m:
        seen, c = {a}, m[a]
        while c in m and c not in seen:
            seen.add(c)
            c = m[c]
        if c != a:
            res[a] = c
    return res


def canonicalise_and_dedupe(df: pd.DataFrame, alias_map: dict | None = None,
                            derive: bool = True, min_dates: int = 3):
    """Apply aliases (given and/or derived), dedupe, quarantine score conflicts.
    Returns (frame, report)."""
    explicit = alias_map or {}
    derived = derive_same_day_aliases(df, min_dates) if derive else {}
    # An explicit (audited) map wins: a derived pair touching any name the explicit map already
    # governs is dropped, so the two can never point a club in opposite directions (a cycle).
    for lg, m in list(derived.items()):
        gov = set(explicit.get(lg, {})) | set(explicit.get(lg, {}).values())
        derived[lg] = {a: c for a, c in m.items() if a not in gov and c not in gov}
    amap = merge_maps(derived, explicit)
    d = df.copy()
    n_renamed = 0
    for c in ("home_team", "away_team"):
        old = d[c].astype(str)
        new = [amap.get(lg, {}).get(t, t) for lg, t in zip(d["league"].astype(str), old)]
        n_renamed += int((old.to_numpy() != np.asarray(new, dtype=object)).sum())
        d[c] = new
    d["_d"] = pd.to_datetime(d["date"]).dt.normalize()
    key = ["_d", "league", "home_team", "away_team"]
    played = d["home_goals"].notna() & d["away_goals"].notna()
    sc = d[played].groupby(key)[["home_goals", "away_goals"]].nunique().max(axis=1)
    bad = set(sc[sc > 1].index)
    kk = list(zip(d["_d"], d["league"], d["home_team"], d["away_team"]))
    conflict = np.array([k in bad for k in kk])
    quarantine = d[conflict].drop(columns=["_d"]).assign(reason="score_conflict_across_spellings")
    d = d[~conflict]
    before = len(d)
    d = (d.assign(_c=d.notna().sum(axis=1))
           .sort_values(["_d", "_c"], kind="mergesort")
           .drop_duplicates(key, keep="last").drop(columns=["_c", "_d"]))
    d = d.sort_values("date", kind="mergesort").reset_index(drop=True)
    report = {"identity_version": IDENTITY_VERSION,
              "aliases": {lg: dict(sorted(m.items())) for lg, m in sorted(amap.items())},
              "n_aliases": int(sum(len(m) for m in amap.values())),
              "team_cells_renamed": n_renamed, "duplicate_rows_removed": int(before - len(d)),
              "quarantined_fixtures": int(len(quarantine)),
              "quarantine": quarantine}
    return d, report


def split_history_check(df: pd.DataFrame) -> pd.DataFrame:
    """Remaining same-day same-score collisions after repair (should be ~0 per league)."""
    d = df[df["home_goals"].notna()].copy()
    d["_d"] = pd.to_datetime(d["date"]).dt.normalize()
    g = d.groupby(["league", "_d", "home_goals", "away_goals"]).size()
    return g[g > 1].reset_index(name="rows")
