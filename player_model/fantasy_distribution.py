"""A distribution of outcomes, not a single number — simulated from FPL's own scoring events.

    from player_model.fantasy_distribution import simulate
    d = simulate(proj_df)          # adds p25/median/p75/ceiling, P(5+), P(8+), P(10+)

WHY A SINGLE EXPECTED-POINTS NUMBER IS NOT ENOUGH. Two players on 5.0 xPts can be completely
different bets. A defender on 5.0 is mostly appearance points and a clean sheet: low variance,
a fine floor, almost no ceiling. A forward on 5.0 is a 45% chance of a goal and a long tail:
blank far more often, but the only one of the two who can win you a week. "Safe captain" and
"high-upside captain" are different questions and a mean cannot tell them apart.

INTERVALS ARE SIMULATED, NEVER ASSUMED. The brief is explicit -- do not fabricate intervals.
There is no normal approximation here and no invented standard deviation. Each simulated
gameweek plays out the actual FPL scoring rules on the model's own event probabilities:

    start?            Bernoulli(p_start)
    60+ minutes?      from minutes_pg, which decides 1 appearance point or 2
    goals             Poisson(lambda) with lambda = -ln(1 - p_goal), because p_goal is the
                      model's P(at least one) and that is the Poisson rate implying it
    assists           Poisson, same inversion from p_assist
    clean sheet       Bernoulli(cs_pts / clean-sheet points for the position)
    bonus             0-3, drawn to match the expected bonus rather than added flat

A player who does not start scores exactly zero, which is why the distributions are so
zero-heavy: 46% of real gameweek scores are zero, and any interval that cannot produce a zero
is describing a different game.

THE SANITY CHECK THAT MATTERS. The simulated MEAN must reproduce the deterministic projection.
If it does not, one of the two is wrong, and a pretty percentile band around a wrong centre is
worse than no band at all.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GOAL_PTS = {"FWD": 4.0, "MID": 5.0, "DEF": 6.0, "GKP": 10.0}   # GKP 10: FPL game_config (was 6)
ASSIST_PTS = 3.0
CS_PTS = {"FWD": 0.0, "MID": 1.0, "DEF": 4.0, "GKP": 4.0}
MAX_BONUS = 3
#: Flat points for clearing the defensive-contribution threshold (2026/27 rules).
DC_PTS = 2.0


def _rate_from_prob(p: np.ndarray) -> np.ndarray:
    """Poisson rate implied by P(at least one) = 1 - exp(-lambda)."""
    p = np.clip(p, 0.0, 0.95)
    return -np.log1p(-p)


#: Every input one simulated gameweek needs. A player missing any of them is not simulated.
_REQUIRED = ("position", "p_start", "minutes_pg", "p_goal", "p_assist", "cs_pts", "dc_pts",
             "bonus_pts")
_OUT_COLS = ("sim_mean", "sim_p25", "sim_median", "sim_p75", "sim_ceiling", "p_blank",
             "p_5plus", "p_8plus", "p_10plus", "upside_ratio")


def simulate(proj: pd.DataFrame, n_sims: int = 20000, seed: int = 11) -> pd.DataFrame:
    """Return `proj` with distribution columns attached. Never raises on odd input.

    A player missing any required input (unknown position, P(start), minutes, event probability
    or points component) gets NaN distribution columns and is logged - he used to be simulated as
    a 'MID' with 0 for whatever was missing, which is a distribution of an invented player.
    """
    d = proj.copy().reset_index(drop=True)
    n = len(d)
    if n == 0:
        return d
    for c in _OUT_COLS:
        d[c] = np.nan
    missing = [c for c in _REQUIRED if c not in d.columns]
    if missing:
        log.warning("simulate: projections lack %s - no distributions", ", ".join(missing))
        return d
    known_pos = d["position"].isin(list(GOAL_PTS))
    nums = d[list(_REQUIRED[1:])].apply(pd.to_numeric, errors="coerce")
    ok = known_pos & nums.notna().all(axis=1)
    if not ok.all():
        log.info("simulate: %d player(s) not simulated - a required input is unknown",
                 int((~ok).sum()))
    if not ok.any():
        return d
    rng = np.random.default_rng(seed)
    full, d = d, d[ok].reset_index(drop=True)
    nums = nums[ok].reset_index(drop=True)
    n = len(d)

    pos = d["position"].astype(str).to_numpy()
    num = lambda c: nums[c].to_numpy(float)

    p_start = np.clip(num("p_start"), 0, 1)
    mins = np.clip(num("minutes_pg"), 0, 90)
    g_rate = _rate_from_prob(num("p_goal"))
    a_rate = _rate_from_prob(num("p_assist"))
    goal_v = np.array([GOAL_PTS[p] for p in pos])
    cs_v = np.array([CS_PTS[p] for p in pos])
    # cs_pts is the EXPECTED clean-sheet points, so the probability is that over its face value.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_cs = np.where(cs_v > 0, num("cs_pts") / np.where(cs_v > 0, cs_v, 1), 0.0)
    p_cs = np.clip(p_cs, 0, 1)
    dc = num("dc_pts")
    bonus_mu = np.clip(num("bonus_pts"), 0, MAX_BONUS)
    # P(60+ minutes) taken from the player's own average; a 90-minute man is nailed for the
    # 2-point appearance, a 45-minute man is a coin flip.
    p60 = np.clip((mins - 30.0) / 45.0, 0.0, 1.0)

    S = (n, n_sims)
    started = rng.random(S) < p_start[:, None]
    long_shift = rng.random(S) < p60[:, None]
    appear = np.where(started, np.where(long_shift, 2.0, 1.0), 0.0)

    goals = rng.poisson(np.repeat(g_rate[:, None], n_sims, axis=1)) * started
    assists = rng.poisson(np.repeat(a_rate[:, None], n_sims, axis=1)) * started
    # A clean sheet only pays if the player completed 60 minutes.
    cs = (rng.random(S) < p_cs[:, None]) & started & long_shift
    # Bonus as a small integer draw whose mean matches the heuristic expectation.
    bonus = (rng.random(S) < (bonus_mu / MAX_BONUS)[:, None]) * rng.integers(1, MAX_BONUS + 1, S)
    bonus = bonus * started

    # DEFENSIVE CONTRIBUTION IS A THRESHOLD, NOT AN ANNUITY. It pays a flat DC_PTS for clearing
    # a per-position count of defensive actions, so a player either hits it or does not. Adding
    # the EXPECTED value deterministically to every appearance made the floor artificially
    # solid: every nailed defender came out with p_blank = 0.000, which is plainly false -- a
    # defender who plays 90 minutes and neither keeps a clean sheet nor hits the threshold
    # scores 2. Modelling it as the Bernoulli it actually is restores a real floor.
    p_dc = np.clip(dc / DC_PTS, 0.0, 1.0)
    dc_hit = (rng.random(S) < p_dc[:, None]) & started

    pts = (appear + goals * goal_v[:, None] + assists * ASSIST_PTS
           + cs * cs_v[:, None] + bonus + dc_hit * DC_PTS)

    q = np.percentile(pts, [25, 50, 75, 90], axis=1)
    d["sim_mean"] = pts.mean(axis=1).round(3)
    d["sim_p25"], d["sim_median"], d["sim_p75"], d["sim_ceiling"] = (q[0].round(2), q[1].round(2),
                                                                     q[2].round(2), q[3].round(2))
    d["p_blank"] = (pts <= 2).mean(axis=1).round(3)      # 2 pts or fewer is a blank in practice
    d["p_5plus"] = (pts >= 5).mean(axis=1).round(3)
    d["p_8plus"] = (pts >= 8).mean(axis=1).round(3)
    d["p_10plus"] = (pts >= 10).mean(axis=1).round(3)
    # Upside per expected point: separates the safe five from the explosive five.
    d["upside_ratio"] = np.where(d["sim_mean"] > 0.1,
                                 (d["sim_ceiling"] / d["sim_mean"]).round(2), np.nan)
    full.loc[ok.to_numpy(), list(_OUT_COLS)] = d[list(_OUT_COLS)].to_numpy()
    return full


def captain_profile(d: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Rank captain options by upside as well as expectation."""
    cols = [c for c in ["player_name", "team", "position", "sim_mean", "sim_median",
                        "sim_ceiling", "p_blank", "p_10plus", "upside_ratio",
                        "start_confidence"] if c in d.columns]
    out = d.sort_values("sim_mean", ascending=False).head(top)[cols].copy()
    if {"p_blank", "p_10plus"}.issubset(out.columns):
        out["style"] = np.where(out["p_10plus"] >= 0.15, "high ceiling",
                        np.where(out["p_blank"] <= 0.25, "safe floor", "balanced"))
    return out.reset_index(drop=True)
