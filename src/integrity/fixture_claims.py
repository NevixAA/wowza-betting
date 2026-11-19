"""One provider fixture, one history row - the duplicate check by the PROVIDER's fixture id (2026-11-10).

WHY A SECOND CHECK. src.integrity.duplicate_matches pairs two rows when both clubs are the same
canonical club (or, same day and same score, ONE side is). A match filed twice under BOTH clubs'
old and new names at once - China Super League 2025-02-23 'Tianjin Teda v Meizhou Kejia' next to
'Tianjin Jinmen Tiger v Meizhou Hakka' (af_history, found by the OddsAPI history matcher on
2026-11-10) - has no side in common by name, so the name rule cannot see it. The provider's own
fixture list can: both rows are API-Football fixture 1340905.

THE CLAIM (the matcher of scripts/collect_ht_scores.claims and scripts/prove_af_history_aliases,
nothing looser). A row claims a provider fixture of its league when
  * it stores that fixture id (rule 'fixture_id'), else
  * SAME DAY, both clubs resolved home->home and away->away (src.team_names.resolve_result, else the
    league-scoped PROVEN alias of config/af_history_proven_aliases.json on that date) onto exactly
    one finished fixture (rule 'both_clubs'), else
  * SAME DAY, one club resolved, it plays exactly one finished fixture of the league that day on that
    side, and the row's full-time score is that fixture's score.fulltime (rule
    'forced_opponent+score': a club does not play twice in a day), else
  * ADJACENT DAY (+-1, the UTC/local date split), both clubs resolved onto exactly one fixture AND
    the row's score is the fixture's score (rule 'both_clubs_adjacent_day').
Anything else claims nothing. A provider fixture claimed by TWO OR MORE rows is one match filed
twice: the frame double-counts it.

Provider pool: the stored fixture lists (output/collect/af_ht_scores.parquet, then
output/strength/competition_fixtures.parquet; one row per fixture id, statuses FT / AET / PEN), the
pool scripts/build_history_corrections.provider_pool builds. No API call.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HT_SCORES = ROOT / "output" / "collect" / "af_ht_scores.parquet"
COMP_FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
PROVEN_ALIASES = ROOT / "config" / "af_history_proven_aliases.json"
FINISHED = ("FT", "AET", "PEN")
POOL_COLS = ["fixture_id", "league", "season", "date", "home_id", "home_name", "away_id", "away_name",
             "ft_home", "ft_away", "status"]


def provider_pool(ht: pd.DataFrame | None = None, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Finished provider fixtures, one row per fixture id: the HT store first, the league fixture
    lists for the rest; FT from the lists where the HT store has none (same fixture id).
    Empty when neither file is readable."""
    def _read(p):
        try:
            return pd.read_parquet(p)
        except Exception:                                              # noqa: BLE001
            return pd.DataFrame()
    ht = _read(HT_SCORES) if ht is None else ht
    comp = _read(COMP_FIXTURES) if comp is None else comp
    parts = [d.reindex(columns=POOL_COLS) for d in (ht, comp) if d is not None and not d.empty]
    if not parts:
        return pd.DataFrame(columns=POOL_COLS)
    pool = pd.concat(parts, ignore_index=True)
    pool["fixture_id"] = pd.to_numeric(pool["fixture_id"], errors="coerce")
    pool = pool.dropna(subset=["fixture_id", "league", "home_name", "away_name"])
    for c in ("ft_home", "ft_away"):
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    # FT of a fixture: the first copy that holds one (HT store first, then the list)
    ft = (pool.dropna(subset=["ft_home", "ft_away"]).drop_duplicates("fixture_id")
          .set_index("fixture_id")[["ft_home", "ft_away"]])
    pool = pool.drop_duplicates("fixture_id", keep="first").reset_index(drop=True)
    pool["ft_home"] = pool["fixture_id"].map(ft["ft_home"])
    pool["ft_away"] = pool["fixture_id"].map(ft["ft_away"])
    pool = pool[pool["status"].isin(FINISHED)].copy()
    pool["fixture_id"] = pool["fixture_id"].astype("int64")
    pool["_d"] = pd.to_datetime(pool["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return pool.dropna(subset=["_d"]).reset_index(drop=True)


def load_proven_aliases(path: Path = PROVEN_ALIASES) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("map", {})
    except Exception:                                                  # noqa: BLE001
        return {}


def _lookup(aliases: dict, league: str, name: str, day: str):
    """scripts/prove_af_history_aliases.lookup (date-segmented league-scoped alias)."""
    v = (aliases.get(league) or {}).get(name)
    if v is None or isinstance(v, str):
        return v
    for seg in v:
        if (seg["valid_from"] is None or day >= seg["valid_from"]) and \
                (seg["valid_to"] is None or day < seg["valid_to"]):
            return seg["provider"]
    return None


class _Day:
    __slots__ = ("names", "by_pair", "by_home", "by_away")

    def __init__(self, g: pd.DataFrame):
        recs = list(zip(g["fixture_id"], g["home_name"].astype(str), g["away_name"].astype(str),
                        g["ft_home"], g["ft_away"]))
        self.names = sorted({r[1] for r in recs} | {r[2] for r in recs})
        self.by_pair, self.by_home, self.by_away = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in recs:
            self.by_pair[(r[1], r[2])].append(r)
            self.by_home[r[1]].append(r)
            self.by_away[r[2]].append(r)


def _same(rec, hg, ag) -> bool:
    return pd.notna(hg) and pd.notna(ag) and rec[3] == float(hg) and rec[4] == float(ag)


def claims(rows: pd.DataFrame, pool: pd.DataFrame, aliases: dict | None = None,
           league_col: str = "league", home_goals: str = "home_goals", away_goals: str = "away_goals",
           fixture_col: str | None = None) -> pd.DataFrame:
    """Each row's ONE provider fixture (module rule). Columns: idx (rows' index label), league,
    date, fixture_id, rule, day_off. `rows[league_col]` must hold registry (canonical) names - the
    pool's `league`. Rows that claim nothing are absent."""
    from src.team_names import resolve_result
    cols = ["idx", "league", "date", "fixture_id", "rule", "day_off"]
    if rows is None or rows.empty or pool is None or pool.empty:
        return pd.DataFrame(columns=cols)
    aliases = load_proven_aliases() if aliases is None else aliases
    days = {k: _Day(g) for k, g in pool.groupby(["league", "_d"], sort=False)}
    league_names = {lg: sorted(set(g["home_name"].astype(str)) | set(g["away_name"].astype(str)))
                    for lg, g in pool.groupby("league", sort=False)}
    memo_lg: dict = {}
    memo_day: dict = {}

    def res(lg, d, name, day):
        if name in day.names:
            return name
        # a unique hit among ALL the league's provider names that plays this day is the unique hit
        # among the day's names too (a subset): resolve once per league, per day only when needed
        k = (lg, name)
        if k not in memo_lg:
            memo_lg[k] = resolve_result(name, league_names.get(lg, []))
        hit = memo_lg[k]
        if hit is not None and hit in day.names:
            return hit
        k2 = (lg, d, name)
        if k2 not in memo_day:
            hit = resolve_result(name, day.names)
            if hit is None:
                p = _lookup(aliases, lg, name, d)
                hit = p if p in day.names else None
            memo_day[k2] = hit
        return memo_day[k2]

    dt = pd.to_datetime(rows["date"], errors="coerce")
    fids = (pd.to_numeric(rows[fixture_col], errors="coerce") if fixture_col and fixture_col in rows.columns
            else pd.Series(np.nan, index=rows.index))
    out = []
    one = pd.Timedelta(days=1)
    for idx, lg, t, h, a, hg, ag, fid in zip(rows.index, rows[league_col], dt, rows["home_team"].astype(str),
                                            rows["away_team"].astype(str), rows[home_goals],
                                            rows[away_goals], fids):
        if not isinstance(lg, str) or pd.isna(t):
            continue
        d0 = t.strftime("%Y-%m-%d")
        if pd.notna(fid):
            out.append((idx, lg, d0, int(fid), "fixture_id", 0))
            continue
        got = None
        day = days.get((lg, d0))
        if day is not None:
            rh, ra = res(lg, d0, h, day), res(lg, d0, a, day)
            if rh and ra:
                f = day.by_pair.get((rh, ra), [])
                if len(f) == 1:
                    got = (f[0][0], "both_clubs", 0)
            elif rh or ra:
                f = day.by_home.get(rh, []) if rh else day.by_away.get(ra, [])
                if len(f) == 1 and _same(f[0], hg, ag):
                    got = (f[0][0], "forced_opponent+score", 0)
        if got is None:
            for off in (-1, 1):
                d = (t + off * one).strftime("%Y-%m-%d")
                dd = days.get((lg, d))
                if dd is None:
                    continue
                rh, ra = res(lg, d, h, dd), res(lg, d, a, dd)
                if rh and ra:
                    f = dd.by_pair.get((rh, ra), [])
                    if len(f) == 1 and _same(f[0], hg, ag):
                        got = (f[0][0], "both_clubs_adjacent_day", off)
                        break
        if got is not None:
            out.append((idx, lg, d0) + got)
    return pd.DataFrame(out, columns=cols)


def duplicate_claims(c: pd.DataFrame) -> pd.DataFrame:
    """The claims of every provider fixture claimed by two or more rows (sorted by league, fixture,
    date)."""
    if c is None or c.empty:
        return pd.DataFrame(columns=list(c.columns) if c is not None else [])
    n = c.groupby("fixture_id")["idx"].transform("nunique")
    return c[n > 1].sort_values(["league", "fixture_id", "date"], kind="mergesort")
