"""
Time-ordered Elo on the LOG-ODDS scale, fitted on the results we hold.

MODEL. Before a match, P(home result) = sigmoid(r_home - r_away + H); the observed result S is
1 / 0.5 / 0 (90-minute win / draw / loss). After it, r_home += K (S - P) and r_away -= K (S - P).
Ratings move only AFTER a match is played, and lookups (lookup.py) read ratings from strictly
earlier days, so a feature at date D never sees a result from D or later.

FITTED, NOT TYPED. K and H are chosen by minimising the log loss of the ONLINE predictions on the
last HOLDOUT_FRACTION of matches (each prediction is made with the ratings before that match), over
the search ranges K_RANGE / H_RANGE: a coarse log/linear grid then a refinement around the best
point. Every evaluated (K, H, loss) is written to output/strength/elo_fit.json so the choice is
auditable. Predictions involving a club with fewer than MIN_CLUB_MATCHES prior matches are not
scored (a club's first matches measure its entry point, not the model).

SCALE ORIGIN. Only rating DIFFERENCES are identified. The first clubs of a competition start at 0,
the origin of the scale (not a strength prior: shifting every rating by a constant changes no
prediction). A club that enters later starts at the CURRENT MEAN rating of the clubs already in
that competition - a measured quantity - and its own rating is reported as NaN until it has
MIN_CLUB_MATCHES matches. League strength (lookup.py) additionally requires the league's country to
be linked to the reference component by MIN_LINK_MATCHES real cross-league matches, because two
leagues that never meet have no common scale.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Minimum samples (Data Truth: below these the value is NaN, never a prior).
MIN_CLUB_MATCHES = 10      # a club's prior rated matches before its rating is reported
MIN_LEAGUE_CLUBS = 8       # rated clubs a league needs before its mean is its strength
MIN_LINK_MATCHES = 10      # cross-league matches between two countries before they share a scale
LEAGUE_WINDOW_DAYS = 365   # a club belongs to a league's strength if it played there this recently

# Fit protocol (search ranges and hold-out share, not values of the parameters themselves).
HOLDOUT_FRACTION = 0.3
K_RANGE = (0.005, 1.0)     # log-odds per unit surprise
H_RANGE = (0.0, 1.0)       # log-odds home advantage
COARSE_GRID = 9


def _arrays(m: pd.DataFrame):
    clubs = pd.Index(pd.unique(np.concatenate([m["home_key"].to_numpy(), m["away_key"].to_numpy()])))
    hi = clubs.get_indexer(m["home_key"])
    ai = clubs.get_indexer(m["away_key"])
    ci, comps = pd.factorize(m["competition"])
    hg = m["home_goals"].to_numpy(float)
    ag = m["away_goals"].to_numpy(float)
    s = np.where(hg > ag, 1.0, np.where(hg < ag, 0.0, 0.5))
    return clubs, hi.tolist(), ai.tolist(), ci.tolist(), s.tolist(), len(comps)


def run(m: pd.DataFrame, k: float, h: float, scored_from: int | None = None,
        record: bool = False, _arr=None) -> dict:
    """One chronological pass. Returns {'loss', 'n_scored'} and, with record=True, per-match
    pre/post ratings and prior match counts (aligned with m's rows) plus the club index."""
    clubs, hi, ai, ci, s, ncomp = _arr or _arrays(m)
    nclub = len(clubs)
    r = [None] * nclub
    n = [0] * nclub
    pools = [[] for _ in range(nclub)]
    member = [set() for _ in range(ncomp)]
    psum = [0.0] * ncomp
    pcnt = [0] * ncomp
    start = len(s) if scored_from is None else scored_from
    loss = 0.0
    nsc = 0
    N = len(s)
    if record:
        pre_h = np.empty(N); pre_a = np.empty(N); post_h = np.empty(N); post_a = np.empty(N)
        n_h = np.empty(N, dtype=np.int64); n_a = np.empty(N, dtype=np.int64)
    exp, log = math.exp, math.log
    for i in range(N):
        a, b, c = hi[i], ai[i], ci[i]
        for x in (a, b):
            if r[x] is None:
                r[x] = psum[c] / pcnt[c] if pcnt[c] else 0.0
            if x not in member[c]:
                member[c].add(x)
                pools[x].append(c)
                psum[c] += r[x]
                pcnt[c] += 1
        ra, rb = r[a], r[b]
        e = 1.0 / (1.0 + exp(-(ra - rb + h)))
        si = s[i]
        if i >= start and n[a] >= MIN_CLUB_MATCHES and n[b] >= MIN_CLUB_MATCHES:
            e2 = min(max(e, 1e-12), 1 - 1e-12)
            loss -= si * log(e2) + (1.0 - si) * log(1.0 - e2)
            nsc += 1
        d = k * (si - e)
        if record:
            pre_h[i], pre_a[i], n_h[i], n_a[i] = ra, rb, n[a], n[b]
        r[a] = ra + d
        r[b] = rb - d
        for p in pools[a]:
            psum[p] += d
        for p in pools[b]:
            psum[p] -= d
        n[a] += 1
        n[b] += 1
        if record:
            post_h[i], post_a[i] = r[a], r[b]
    out = {"loss": (loss / nsc) if nsc else float("nan"), "n_scored": nsc}
    if record:
        out.update(pre_h=pre_h, pre_a=pre_a, post_h=post_h, post_a=post_a, n_h=n_h, n_a=n_a)
    return out


def fit(m: pd.DataFrame, verbose: bool = True) -> dict:
    """Fit K and H on held-out online log loss. Returns {'k', 'h', 'loss', 'n_scored', 'grid':[...]}.
    Raises ValueError when there are no scoreable held-out matches (nothing to fit on)."""
    if m.empty:
        raise ValueError("no matches to fit the rating on")
    arr = _arrays(m)
    start = int(len(m) * (1.0 - HOLDOUT_FRACTION))
    evals: dict = {}

    def loss(k, h):
        key = (round(float(k), 6), round(float(h), 6))
        if key not in evals:
            evals[key] = run(m, key[0], key[1], scored_from=start, _arr=arr)
        return evals[key]["loss"]

    ks = np.geomspace(*K_RANGE, COARSE_GRID)
    hs = np.linspace(*H_RANGE, COARSE_GRID)
    for k in ks:
        for h in hs:
            loss(k, h)
    if not any(v["n_scored"] for v in evals.values()):
        raise ValueError("no held-out match has two clubs with enough prior matches to score")
    # Refine: halve the step around the incumbent until the steps are < 2% of the range.
    best = min(evals, key=lambda kh: evals[kh]["loss"])
    lk_step = (np.log(K_RANGE[1]) - np.log(K_RANGE[0])) / (COARSE_GRID - 1)
    h_step = (H_RANGE[1] - H_RANGE[0]) / (COARSE_GRID - 1)
    while h_step > 0.02 * (H_RANGE[1] - H_RANGE[0]):
        lk_step /= 2.0
        h_step /= 2.0
        k0, h0 = best
        for dk in (-1, 0, 1):
            for dh in (-1, 0, 1):
                k = float(np.clip(np.exp(np.log(k0) + dk * lk_step), *K_RANGE))
                h = float(np.clip(h0 + dh * h_step, *H_RANGE))
                loss(k, h)
        best = min(evals, key=lambda kh: evals[kh]["loss"])
    res = evals[best]
    out = {"k": best[0], "h": best[1], "loss": res["loss"], "n_scored": res["n_scored"],
           "n_matches": int(len(m)), "holdout_from_date": str(m["date"].iloc[start])[:10],
           "holdout_fraction": HOLDOUT_FRACTION, "k_range": list(K_RANGE), "h_range": list(H_RANGE),
           "min_club_matches": MIN_CLUB_MATCHES,
           "grid": [{"k": k, "h": h, "loss": v["loss"], "n_scored": v["n_scored"]}
                    for (k, h), v in sorted(evals.items())]}
    if verbose:
        print(f"[strength] fitted K={out['k']:.4f} H={out['h']:.4f} held-out log loss "
              f"{out['loss']:.5f} on {out['n_scored']:,} matches ({len(evals)} evaluations)")
    return out


def club_rating_table(m: pd.DataFrame, k: float, h: float) -> pd.DataFrame:
    """Long table: one row per (club, match) with the rating AFTER the match and the number of
    matches it has played including this one. lookup.py reads it strictly before a date."""
    res = run(m, k, h, record=True)
    home = pd.DataFrame({"club_key": m["home_key"].to_numpy(), "date": m["date"].to_numpy(),
                         "competition": m["competition"].to_numpy(), "rating": res["post_h"],
                         "n_matches": res["n_h"] + 1, "scope": m["home_scope"].to_numpy()})
    away = pd.DataFrame({"club_key": m["away_key"].to_numpy(), "date": m["date"].to_numpy(),
                         "competition": m["competition"].to_numpy(), "rating": res["post_a"],
                         "n_matches": res["n_a"] + 1, "scope": m["away_scope"].to_numpy()})
    t = pd.concat([home, away], ignore_index=True)
    t["_o"] = np.r_[np.arange(len(m)), np.arange(len(m))]
    return t.sort_values(["date", "_o"]).drop(columns="_o").reset_index(drop=True)
