"""
league_quality.py — measured club / league strength for the player features
==========================================================================
OWNER DECISION A (2026-11-02): "measure our own strength". The strength comes from src/strength: a
time-ordered Elo fitted ONLY on real results we hold (football-data / API-Football history plus the
collected league and cross-league cup fixtures), K and home advantage FITTED on held-out log loss.
A club with too few rated matches, and a league whose country is not linked to the reference
component by real cross-league matches, is NaN. No hand tiers, no priors.

MEASURED COLUMNS (as of the row's match date; only matches on EARLIER days are seen):
  team_club_elo                 the player's club rating                        (log-odds scale)
  opp_club_elo                  the opponent club's rating
  league_elo                    league strength = mean rating of its rated clubs (linked leagues)
  league_elo_rank               1 = strongest linked league on that day
  player_career_avg_league_elo  mean league_elo over the player's earlier rows  (>= MIN_CAREER_ROWS)
  player_career_avg_opp_elo     mean opp_club_elo over the player's earlier rows (>= MIN_CAREER_ROWS)
and, in feature_engineering._add_composites,
  context_elo_odds              exp(player_career_avg_opp_elo - opp_club_elo): the Elo odds ratio
                                between the opposition his record was earned against and this one
                                (> 1: this opponent is weaker than his usual opposition)
  elo_mismatch_goals / _sot     goals_pg / sot_pg x context_elo_odds

RENAMED, NOT REDEFINED. The legacy columns league_tier, league_quality, player_career_avg_quality,
context_quality_discount, quality_mismatch_goals and quality_mismatch_sot meant HAND tiers (1-4,
0.3-1.0); the serving payloads were trained on them. Their meaning cannot be measured, so they stay
NaN (LEGACY_COLUMNS) and the measured replacements above carry new names. The props model can adopt
the new names at its next retrain (config.PLAYER_FEATURE_COLS is not owned here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEGACY_COLUMNS = ("league_tier", "league_quality", "player_career_avg_quality",
                  "context_quality_discount", "quality_mismatch_goals", "quality_mismatch_sot")
QUALITY_COLUMNS = LEGACY_COLUMNS[:3]          # kept for callers of enrich_league_quality
MEASURED_COLUMNS = ("team_club_elo", "opp_club_elo", "league_elo", "league_elo_rank",
                    "player_career_avg_league_elo", "player_career_avg_opp_elo")

MIN_CAREER_ROWS = 5           # earlier rows with a measured value behind a career average


def league_tier(league) -> float:
    """The legacy HAND tier of `league`: never measured -> NaN for every league. The measured
    strength is league_strength(league, date)."""
    return float("nan")


def league_strength(league, date) -> float:
    """Measured strength (mean club Elo) of `league` as of `date`, or NaN."""
    from src.strength import lookup
    return float(lookup.load().league_asof([league], [date])["league_elo"].iloc[0])


def enrich_league_quality(history: list[dict]) -> list[dict]:
    """Legacy entry point (pipeline calls it before build_features): writes the LEGACY quality
    columns as NaN. The measured columns are attached by build_features (attach_strength), with
    ONE definition for training and serving."""
    if not history:
        return history
    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    for c in QUALITY_COLUMNS:
        df[c] = np.nan
    return df.to_dict("records")


def _keys(S, names: pd.Series, leagues: pd.Series) -> pd.Series:
    """Club key of each (name, league) through the strength name index (team_names.resolve in the
    league's identity scope, ambiguity refused); None when unresolved."""
    pairs = pd.Series(list(zip(names, leagues)), index=names.index)
    cache = {p: S.club_key(p[0], p[1]) for p in pd.unique(pairs)}
    return pairs.map(cache)


def attach_strength(df: pd.DataFrame, strength=None) -> pd.DataFrame:
    """Add MEASURED_COLUMNS to a player-match frame (player_id, date, league, team, opponent).
    Career averages use only the player's EARLIER rows (shift 1). Returns a new frame."""
    from src.strength import lookup
    S = strength if strength is not None else lookup.load()
    out = df.copy()
    for c in MEASURED_COLUMNS:
        out[c] = np.nan
    out["_scope"] = None
    if out.empty or S.empty or not {"date", "league", "team"} <= set(out.columns):
        return out
    opp = out["opponent"] if "opponent" in out.columns else \
        out["away_team"].where(out["is_home"].astype(bool), out["home_team"])
    dates = pd.to_datetime(out["date"], errors="coerce")
    own_k = _keys(S, out["team"], out["league"])
    opp_k = _keys(S, opp, out["league"])
    out["team_club_elo"] = S.club_elo_asof(own_k, dates)["club_elo"].to_numpy()
    out["opp_club_elo"] = S.club_elo_asof(opp_k, dates)["club_elo"].to_numpy()
    lg = S.league_asof(out["league"], dates)
    out["league_elo"] = lg["league_elo"].to_numpy()
    out["league_elo_rank"] = lg["league_elo_rank"].to_numpy()
    out["_scope"] = _scopes(out["league"])
    if "player_id" in out.columns:
        order = np.lexsort((np.arange(len(out)), dates.to_numpy(), out["player_id"].to_numpy()))
        o = out.iloc[order]
        # league_elo is on the linked cross-league scale: averaged over the whole career. Club
        # ratings share a scale only inside one identity scope (a country's divisions; a cup), so
        # the opponent average uses the player's earlier rows in the SAME scope as the row.
        for src, feat, keys in (("league_elo", "player_career_avg_league_elo", [o["player_id"]]),
                                ("opp_club_elo", "player_career_avg_opp_elo",
                                 [o["player_id"], o["_scope"].fillna("")])):
            v = o[src]
            prev_sum = v.fillna(0.0).groupby(keys).cumsum() - v.fillna(0.0)
            prev_n = v.notna().astype(int).groupby(keys).cumsum() - v.notna().astype(int)
            avg = (prev_sum / prev_n.where(prev_n > 0)).where(prev_n >= MIN_CAREER_ROWS)
            out.loc[o.index, feat] = avg.to_numpy()
    return out


def _scopes(leagues: pd.Series) -> pd.Series:
    """Identity scope per row: the club league's country, the cup itself, else None."""
    from src.strength import results
    cache = {}
    for lg in pd.unique(leagues):
        s = results.scope_of(lg)
        if s is None and results.competition_type(lg) == "club_cup":
            s = f"club_cup:{results._canon(lg) or lg}"
        cache[lg] = s
    return leagues.map(cache)


def career_now(rows: pd.DataFrame, league=None) -> dict:
    """Serving: the career averages over a player's stored rows (all before the fixture); the
    opponent average over the rows in the fixture league's identity scope (training definition)."""
    res = {}
    scope = _scopes(pd.Series([league]))[0] if league is not None else None
    for src, feat in (("league_elo", "player_career_avg_league_elo"),
                      ("opp_club_elo", "player_career_avg_opp_elo")):
        sub = rows
        if src == "opp_club_elo":
            sub = rows[rows["_scope"] == scope] if (scope is not None and "_scope" in rows.columns) \
                else rows.iloc[0:0]
        v = pd.to_numeric(sub[src], errors="coerce") if src in sub.columns else pd.Series(dtype=float)
        res[feat] = float(v.mean()) if v.notna().sum() >= MIN_CAREER_ROWS else float("nan")
    return res


def fixture_strength(team, opponent, league, date, strength=None) -> dict:
    """Serving: team_club_elo / opp_club_elo / league_elo / league_elo_rank for one fixture."""
    from src.strength import lookup
    S = strength if strength is not None else lookup.load()
    d = pd.to_datetime(date, errors="coerce")
    res = {c: float("nan") for c in ("team_club_elo", "opp_club_elo", "league_elo", "league_elo_rank")}
    if S.empty or d != d:
        return res
    ks = [S.club_key(team, league), S.club_key(opponent, league)]
    e = S.club_elo_asof(ks, [d, d])["club_elo"]
    res["team_club_elo"], res["opp_club_elo"] = float(e.iloc[0]), float(e.iloc[1])
    lg = S.league_asof([league], [d])
    res["league_elo"], res["league_elo_rank"] = float(lg["league_elo"].iloc[0]), float(lg["league_elo_rank"].iloc[0])
    return res
