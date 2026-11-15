"""
team_identity.py - API-Football TEAM ID identity for player rows and the opponent tables.

WHY (owner question 2026-11-02, UEFA club cups in props). The opponent features of a fixture were
looked up under '<team name>|<identity scope>', and a club cup is its own scope ('club_cup'). In a
Champions League / Europa League / Conference League tie the opponent was therefore looked up in
the cup's own history - which player_history did not hold at all - so every opponent feature of
the 2,193 cup props tips since August was NaN on the branch (typed constants on main), even for
Real Madrid or Arsenal with dozens of league matches on file.

ONE IDENTITY ACROSS COMPETITIONS. A club is 'af:<API-Football team id>' in every competition:
  * a player row's own / opponent id comes from the row (collected cup rows carry team_id and
    opponent_id), else from the API-Football fixture it belongs to
    (output/strength/competition_fixtures.parquet, written by scripts/collect_competition_fixtures:
    fixture_id -> home_id / away_id), the row's side decided by club_rule.side_of (exact name, then
    src.team_names identity; neither side proven -> no id, never a guess);
  * a NAME (an upcoming fixture's opponent) resolves to an id among the clubs that played THAT
    competition in the collected fixtures (exact provider name, then src.team_names.resolve,
    ambiguity refused), else through the measured-strength name index (src.strength.lookup,
    the league's country scope; a cup: exactly one scope must resolve it).
The same key is used by feature_engineering (training AND serving) and league_quality.

These are provider identifiers read from collected files; no value is typed here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FIX_FILE = "competition_fixtures.parquet"
_FIX_COLS = ["fixture_id", "league_id", "league", "league_name", "date", "home_id", "home_name",
             "away_id", "away_name", "ft_home", "ft_away", "status"]
_CACHE: dict = {}


def _dir() -> Path:
    # Read at call time, so a test that points src.strength.lookup.DIR elsewhere sees no tables.
    from src.strength import lookup
    return Path(lookup.DIR)


def key(team_id) -> str | None:
    """'af:<id>' for a real API-Football team id (> 0), else None."""
    try:
        v = int(team_id)
    except (TypeError, ValueError):
        return None
    return f"af:{v}" if v > 0 else None


def _canon(league):
    if not isinstance(league, str) or not league:
        return None
    return config.league_registry().canonical_name(league)


def fixture_results() -> pd.DataFrame:
    """The collected API-Football finished fixtures (ids, provider names, 90-minute score), one row
    per fixture_id, with `competition` = the registry canonical name (or the provider's name for a
    continental cup that is not a registry league). Empty frame when the file is absent."""
    p = _dir() / FIX_FILE
    stamp = (str(p), p.stat().st_mtime if p.exists() else None)
    hit = _CACHE.get("fx")
    if hit is not None and hit[0] == stamp:
        return hit[1]
    if not p.exists():
        fx = pd.DataFrame(columns=_FIX_COLS + ["competition"])
    else:
        try:
            fx = pd.read_parquet(p)
        except Exception as e:                                              # noqa: BLE001
            print(f"[team_identity] {p.name} unreadable ({e}): no fixture ids")
            fx = pd.DataFrame(columns=_FIX_COLS)
        for c in _FIX_COLS:
            if c not in fx.columns:
                fx[c] = None
        fx = fx[_FIX_COLS].copy()
        for c in ("fixture_id", "home_id", "away_id", "ft_home", "ft_away", "league_id"):
            fx[c] = pd.to_numeric(fx[c], errors="coerce")
        fx = fx.dropna(subset=["fixture_id"]).drop_duplicates("fixture_id", keep="last")
        comp = {lg: (_canon(lg) or lg) for lg in pd.unique(fx["league"].dropna())}
        fx["competition"] = fx["league"].map(comp).where(fx["league"].notna(), fx["league_name"])
        fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
        fx = fx.reset_index(drop=True)
    _CACHE.clear()
    _CACHE["fx"] = (stamp, fx)
    return fx


def _id_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    v = pd.to_numeric(df[col], errors="coerce")
    return v.where(v > 0)


def row_team_ids(df: pd.DataFrame, fx: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """(own team id, opponent team id) per player row, float with NaN where unknown.

    The row's own team_id / opponent_id first; else the fixture's ids with the side decided by
    club_rule.side_of(team, fixture home name, fixture away name). A row whose fixture we do not
    hold, or whose club name proves neither side, has no id (counted by the callers' logs)."""
    tid, oid = _id_col(df, "team_id"), _id_col(df, "opponent_id")
    need = (tid.isna() | oid.isna())
    if not need.any() or "fixture_id" not in df.columns or "team" not in df.columns:
        return tid, oid
    fx = fixture_results() if fx is None else fx
    if fx.empty:
        return tid, oid
    from .club_rule import side_of
    sub = pd.DataFrame({"fixture_id": pd.to_numeric(df.loc[need, "fixture_id"], errors="coerce"),
                        "team": df.loc[need, "team"].astype(str)})
    pairs = sub.drop_duplicates().merge(
        fx[["fixture_id", "home_id", "home_name", "away_id", "away_name"]], on="fixture_id", how="inner")
    if pairs.empty:
        return tid, oid
    side = [side_of(t, h, a) for t, h, a in zip(pairs["team"], pairs["home_name"], pairs["away_name"])]
    side = pd.Series(side, index=pairs.index, dtype=object)
    pairs["_tid"] = np.where(side == True, pairs["home_id"],                         # noqa: E712
                             np.where(side == False, pairs["away_id"], np.nan))      # noqa: E712
    pairs["_oid"] = np.where(side == True, pairs["away_id"],                         # noqa: E712
                             np.where(side == False, pairs["home_id"], np.nan))      # noqa: E712
    got = sub.reset_index().merge(pairs[["fixture_id", "team", "_tid", "_oid"]],
                                  on=["fixture_id", "team"], how="left").set_index("index")
    tid = tid.copy()
    oid = oid.copy()
    tid.loc[got.index] = tid.loc[got.index].where(tid.loc[got.index].notna(), got["_tid"].astype(float))
    oid.loc[got.index] = oid.loc[got.index].where(oid.loc[got.index].notna(), got["_oid"].astype(float))
    return tid.where(tid > 0), oid.where(oid > 0)


def _pool(competition) -> dict:
    """{provider club name: {team ids}} of the clubs that played `competition` in the collected
    fixtures (None: every collected competition)."""
    fx = fixture_results()
    ck = ("pool", competition, id(fx))
    if ck in _CACHE:
        return _CACHE[ck]
    out: dict = {}
    if not fx.empty:
        sub = fx if competition is None else fx[fx["competition"] == competition]
        for n, i in zip(pd.concat([sub["home_name"], sub["away_name"]]),
                        pd.concat([sub["home_id"], sub["away_id"]])):
            if isinstance(n, str) and n and i == i:
                out.setdefault(n, set()).add(int(i))
    _CACHE[ck] = out
    return out


def club_key(name, league, strength=None) -> str | None:
    """'af:<id>' of the club called `name` in a fixture of `league` (see module docstring), None
    when nothing resolves it unambiguously. `strength`: an injected src.strength.lookup.Strength
    (default: the tables on disk)."""
    if not isinstance(name, str) or not name.strip():
        return None
    ck = ("key", name, league, id(strength))
    if ck in _CACHE:
        return _CACHE[ck]
    comp = _canon(league) or league
    pool = _pool(comp)
    hit = None
    if name in pool and len(pool[name]) == 1:
        hit = key(next(iter(pool[name])))
    if hit is None and pool:
        nm = config._v9_src("team_names").resolve(name, list(pool))
        if nm and len(pool[nm]) == 1:
            hit = key(next(iter(pool[nm])))
    if hit is None:
        # The exact provider name in ANY collected competition, when it names exactly one club
        # (a cup tie's club that has not played this cup before; a promoted club).
        allp = _pool(None)
        if name in allp and len(allp[name]) == 1:
            hit = key(next(iter(allp[name])))
    if hit is None:
        try:
            from src.strength import lookup
            S = strength if strength is not None else lookup.load()
            hit = S.club_key(name, league)
        except Exception as e:                                              # noqa: BLE001
            print(f"[team_identity] strength name index unavailable ({e})")
            hit = None
    _CACHE[ck] = hit
    return hit
