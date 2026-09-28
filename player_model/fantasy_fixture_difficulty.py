"""Two fixture difficulties, because an opponent is not equally hard in both directions.

    from player_model.fantasy_fixture_difficulty import team_strengths, rate_fixtures

WHY ONE NUMBER IS NOT ENOUGH. Official FDR is a single 1-5 rating per fixture, so a team is
simply "hard" or "easy". Real opponents are not symmetric. A side that concedes freely but
scores freely is a GOOD fixture for your forwards and a BAD one for your defenders, and a single
number cannot say that -- it averages the two opposite answers into a middling one that is wrong
for both.

So there are two:

    ATTACK difficulty   how hard this opponent is to SCORE against  (their defence)
    DEFENCE difficulty  how likely this opponent is to SCORE        (their attack)

Use attack difficulty for forwards and attacking midfielders, defence difficulty for defenders
and keepers whose points come from clean sheets.

BUILT FROM WOWZA'S OWN MATCH DATA, WHICH TURNED OUT TO BE THE ONLY OPTION. There is no Premier
League team data anywhere in the estate -- fd_history and af_history hold 0 PL rows, because
Wowza bets second divisions and the only-our-leagues rule kept the top flight out. The PL does
exist in player_history, so team strength is aggregated up from player rows: 1,560 fixtures of
goals scored and conceded, per team, split home and away.

IT DOES NOT REPLACE OFFICIAL FDR. Both are exposed. FDR is what every other FPL tool shows and
what most managers reason in; this is an additional view, and where the two disagree that
disagreement is the interesting part rather than something to hide.

RECENCY WITHOUT THROWING AWAY HISTORY. Strength is an exponentially weighted mean over each
team's matches, so this season dominates while earlier seasons still anchor a side that has
played six games. A promoted team with almost no history is flagged rather than assigned a
confident rating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_model import config

PARQUET = config.BASE_DIR / "player_history.parquet"
FANTASY_LEAGUE = "Premier League"
#: Matches after which a team's rating is treated as reliable.
MIN_MATCHES = 8
#: Half-life in matches for the recency weighting.
HALF_LIFE = 12.0


def _team_match_frame(parquet_path=None) -> pd.DataFrame:
    """One row per (fixture, team): goals for, goals against, home flag."""
    cols = ["league", "fixture_id", "date", "team", "opponent", "is_home",
            "goals", "goals_conceded"]
    df = pd.read_parquet(parquet_path or PARQUET, columns=cols)
    pl = df[df["league"] == FANTASY_LEAGUE].copy()
    if pl.empty:
        return pl
    pl["date"] = pd.to_datetime(pl["date"], errors="coerce")
    # Player rows repeat the team-level score, so take one row per team per fixture. Goals FOR
    # must be SUMMED across the team's players; goals against is already a team figure.
    g = (pl.groupby(["fixture_id", "team"], as_index=False)
           .agg(date=("date", "first"), opponent=("opponent", "first"),
                is_home=("is_home", "first"),
                gf=("goals", "sum"), ga=("goals_conceded", "max")))
    return g.dropna(subset=["date", "team", "opponent"]).sort_values("date")


def team_strengths(parquet_path=None) -> pd.DataFrame:
    """Per-team attack and defence rates, overall and split home/away."""
    g = _team_match_frame(parquet_path)
    if g.empty:
        return g
    rows = []
    for team, t in g.groupby("team"):
        t = t.sort_values("date")
        n = len(t)
        # Exponential recency weights: newest match weighted 1, halving every HALF_LIFE matches.
        age = np.arange(n - 1, -1, -1)
        w = 0.5 ** (age / HALF_LIFE)
        def wm(col, mask=None):
            s = t[col].to_numpy(float); ww = w
            if mask is not None:
                s, ww = s[mask], w[mask]
            return float(np.average(s, weights=ww)) if len(s) and ww.sum() > 0 else np.nan
        home = t["is_home"].astype(bool).to_numpy()
        rows.append({"team": team, "matches": n,
                     "gf_pg": wm("gf"), "ga_pg": wm("ga"),
                     "gf_home": wm("gf", home), "ga_home": wm("ga", home),
                     "gf_away": wm("gf", ~home), "ga_away": wm("ga", ~home),
                     "reliable": n >= MIN_MATCHES})
    s = pd.DataFrame(rows)
    # Index against the league so 1.0 means "exactly average".
    lg_gf, lg_ga = s["gf_pg"].mean(), s["ga_pg"].mean()
    s["attack_index"] = (s["gf_pg"] / lg_gf).round(3)      # >1 = scores more than average
    s["defence_index"] = (s["ga_pg"] / lg_ga).round(3)     # >1 = concedes more than average
    return s.sort_values("attack_index", ascending=False).reset_index(drop=True)


def _scale_from(reference: pd.Series) -> np.ndarray:
    """The four league-wide cut points that turn an index into a 1-5 rating.

    DIFFICULTY IS AN ABSOLUTE SCALE, NOT A RANK WITHIN THE INPUT. The first version ranked
    whatever rows it was handed, so asking about two fixtures against the same opponent rated
    both 5 -- they were the top of a two-row list. Luton came out at attack difficulty 5, the
    hardest team in the league to score against, when they were the easiest. The cut points now
    come from the 20 teams once, and every fixture is measured against them.
    """
    return np.quantile(reference.dropna().to_numpy(float), [0.2, 0.4, 0.6, 0.8])


def _apply_scale(x: pd.Series, cuts: np.ndarray) -> pd.Series:
    return pd.Series(np.digitize(x.to_numpy(float), cuts) + 1, index=x.index).clip(1, 5)


def rate_fixtures(opponents: pd.DataFrame, strengths: pd.DataFrame | None = None) -> pd.DataFrame:
    """Given rows with `opponent` (and optional `is_home`), attach both difficulties.

    attack_difficulty  1 = easiest to score against, 5 = hardest
    defence_difficulty 1 = least likely to score against you, 5 = most
    """
    s = strengths if strengths is not None else team_strengths()
    if s is None or s.empty or opponents.empty:
        return opponents.assign(attack_difficulty=np.nan, defence_difficulty=np.nan,
                                difficulty_reliable=False)
    idx = s.set_index("team")
    o = opponents.copy()
    # Harder to score against = concedes FEWER goals, so invert the defence index.
    o["_opp_concede"] = o["opponent"].map(idx["defence_index"])
    o["_opp_score"] = o["opponent"].map(idx["attack_index"])
    o["difficulty_reliable"] = o["opponent"].map(idx["reliable"]).fillna(False)
    rel = s[s["reliable"]] if s["reliable"].any() else s
    # Cut points from the LEAGUE, applied to whatever fixtures were asked about.
    a_cuts = _scale_from(-rel["defence_index"])
    d_cuts = _scale_from(rel["attack_index"])
    o["attack_difficulty"] = _apply_scale(-o["_opp_concede"].fillna(rel["defence_index"].mean()), a_cuts)
    o["defence_difficulty"] = _apply_scale(o["_opp_score"].fillna(rel["attack_index"].mean()), d_cuts)
    return o.drop(columns=["_opp_concede", "_opp_score"])


def disagreements(strengths: pd.DataFrame | None = None, top: int = 8) -> pd.DataFrame:
    """Teams where the two difficulties diverge most — the whole reason for splitting them.

    If attack and defence difficulty always agreed, one number would be sufficient and this
    module would be wasted effort. This is the check that says whether it is.
    """
    s = strengths if strengths is not None else team_strengths()
    if s is None or s.empty:
        return pd.DataFrame()
    d = s[s["reliable"]].copy()
    d["attack_difficulty"] = _apply_scale(-d["defence_index"], _scale_from(-d["defence_index"]))
    d["defence_difficulty"] = _apply_scale(d["attack_index"], _scale_from(d["attack_index"]))
    d["split"] = (d["attack_difficulty"] - d["defence_difficulty"]).abs()
    return (d.sort_values("split", ascending=False)
            .head(top)[["team", "matches", "attack_index", "defence_index",
                        "attack_difficulty", "defence_difficulty", "split"]]
            .reset_index(drop=True))
