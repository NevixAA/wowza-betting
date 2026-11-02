"""
As-of-date reads of the measured strength tables (built by src/strength/build.py).

Every read is POINT-IN-TIME: a club's rating at date D is its rating after its last match on a day
strictly before D; a league's strength at D is its latest snapshot on or before D (a snapshot is
itself computed from matches strictly before its day) and no older than LEAGUE_WINDOW_DAYS. Below
the minimum samples, or with no table on disk, the value is NaN. Names are mapped to club keys with
src.team_names.resolve inside the league's identity scope (results.ClubIndex), ambiguity refused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.strength import elo, results

DIR = Path(__file__).resolve().parents[2] / "output" / "strength"


def _ns(s: pd.Series) -> pd.Series:
    """Day-normalised, tz-naive datetime64[ns] (merge_asof needs one resolution on both sides)."""
    d = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    return d.astype("datetime64[ns]")


class Strength:
    def __init__(self, ratings: pd.DataFrame, leagues: pd.DataFrame, names: pd.DataFrame):
        # one key dtype on both sides of merge_asof (parquet may hand back pandas' string dtype)
        ratings = ratings.assign(date=_ns(ratings["date"]), club_key=ratings["club_key"].astype(object))
        leagues = leagues.assign(date=_ns(leagues["date"]), league=leagues["league"].astype(object))
        self.ratings = ratings.sort_values("date").reset_index(drop=True)
        self.leagues = leagues.sort_values("date").reset_index(drop=True)
        self.index = results.ClubIndex()
        for s, n, k in zip(names["scope"], names["name"], names["club_key"]):
            self.index.add(s, n, k)
        self._key_cache: dict = {}

    @property
    def empty(self) -> bool:
        return self.ratings.empty

    def club_key(self, name, league) -> str | None:
        """Club key of `name` playing in `league`: resolved in the league's country scope; for a
        cup / unknown-scope competition, only when exactly one scope resolves the name."""
        k = (name, league)
        if k not in self._key_cache:
            scope = results.scope_of(league)
            if scope is not None:
                self._key_cache[k] = self.index.resolve(name, scope)
            elif results.competition_type(league) == "club_cup":
                self._key_cache[k] = self.index.resolve_any_scope(name)
            else:
                self._key_cache[k] = None
        return self._key_cache[k]

    def club_elo_asof(self, keys, dates) -> pd.DataFrame:
        """DataFrame(index aligned with the inputs) of club_elo, club_elo_n: the club's rating
        after its last match on a day before `date`, NaN below MIN_CLUB_MATCHES matches."""
        q = pd.DataFrame({"club_key": pd.Series(list(keys), dtype=object),
                          "date": _ns(pd.Series(list(dates)))})
        q["_i"] = np.arange(len(q))
        out = pd.DataFrame({"club_elo": np.nan, "club_elo_n": np.nan}, index=q.index)
        ok = q["club_key"].notna() & q["date"].notna()
        if ok.any() and not self.ratings.empty:
            r = self.ratings[["club_key", "date", "rating", "n_matches"]]
            hit = pd.merge_asof(q[ok].sort_values("date"), r, on="date", by="club_key",
                                allow_exact_matches=False).set_index("_i")
            n = hit["n_matches"]
            out.loc[hit.index, "club_elo"] = hit["rating"].where(n >= elo.MIN_CLUB_MATCHES).to_numpy()
            out.loc[hit.index, "club_elo_n"] = n.to_numpy()
        return out

    def league_asof(self, leagues, dates) -> pd.DataFrame:
        """league_elo (strength), league_elo_rank (1 = strongest linked league) as of `date`."""
        canon = [results._canon(x) for x in leagues]
        q = pd.DataFrame({"league": pd.Series(canon, dtype=object),
                          "date": _ns(pd.Series(list(dates)))})
        q["_i"] = np.arange(len(q))
        out = pd.DataFrame({"league_elo": np.nan, "league_elo_rank": np.nan}, index=q.index)
        ok = q["league"].notna() & q["date"].notna()
        if ok.any() and not self.leagues.empty:
            lg = self.leagues[["league", "date", "strength", "rank"]].rename(columns={"date": "snap"})
            lg["date"] = lg["snap"]
            hit = pd.merge_asof(q[ok].sort_values("date"), lg.sort_values("date"), on="date",
                                by="league", tolerance=pd.Timedelta(days=elo.LEAGUE_WINDOW_DAYS)
                                ).set_index("_i")
            out.loc[hit.index, "league_elo"] = hit["strength"].to_numpy()
            out.loc[hit.index, "league_elo_rank"] = hit["rank"].to_numpy()
        return out


_LOADED: dict = {}


def load(directory: Path | None = None) -> Strength:
    """The strength tables on disk (cached per directory and file mtimes). Missing tables -> an
    empty Strength whose every lookup is NaN (logged once)."""
    d = Path(directory or DIR)
    files = [d / "club_ratings.parquet", d / "league_strength.parquet", d / "club_names.parquet"]
    stamp = tuple(f.stat().st_mtime if f.exists() else None for f in files)
    key = str(d)
    if key in _LOADED and _LOADED[key][0] == stamp:
        return _LOADED[key][1]
    if not all(f.exists() for f in files):
        print(f"[strength] no measured strength tables in {d} (run python -m src.strength.build): "
              f"club / league strength features are NaN")
        s = Strength(pd.DataFrame(columns=["club_key", "date", "rating", "n_matches"]),
                     pd.DataFrame(columns=["league", "date", "strength", "rank"]),
                     pd.DataFrame(columns=["scope", "name", "club_key"]))
    else:
        s = Strength(pd.read_parquet(files[0]), pd.read_parquet(files[1]), pd.read_parquet(files[2]))
    _LOADED[key] = (stamp, s)
    return s
