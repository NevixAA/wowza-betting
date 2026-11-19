"""Per-league negative-binomial count models for corners and cards -> P(count > line).
====================================================================================

Per league x family three count models, each a log-link GLM fitted by Poisson IRLS with an NB2
dispersion estimated by moments on the training residuals (alpha = 0 -> Poisson):

    home team count   log mu = b0 + b1 log1p(home_for) + b2 log1p(away_against)
    away team count   log mu = c0 + c1 log1p(away_for) + c2 log1p(home_against)
    match total       log mu = d0 + d1 log1p(home_for + home_against) + d2 log1p(away_for + away_against)

The league's own intercepts carry its level (every league its own model, never pooled). A line L
(x.5) settles OVER when count > L, so P(OVER) = 1 - CDF(floor(L)). Parameters are stored as JSON
(models/side_models/side_models.json) - no pickle, so the served numbers are readable and diffable.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy import stats

PARTS = ("home", "away", "total")


def design(f, part: str) -> np.ndarray:
    """Design matrix of one part from a features frame (columns of src.side_models.features)."""
    hf, ha = np.asarray(f["home_for"], float), np.asarray(f["home_against"], float)
    af, aa = np.asarray(f["away_for"], float), np.asarray(f["away_against"], float)
    if part == "home":
        x1, x2 = hf, aa
    elif part == "away":
        x1, x2 = af, ha
    elif part == "total":
        x1, x2 = hf + ha, af + aa
    else:
        raise ValueError(part)
    return np.column_stack([np.ones(len(hf)), np.log1p(x1), np.log1p(x2)])


def fit_poisson(X: np.ndarray, y: np.ndarray, iters: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Poisson GLM (log link) by IRLS. Deterministic; starts from the mean."""
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(float(np.mean(y)), 1e-9))
    for _ in range(iters):
        eta = X @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        W = mu
        XtW = X.T * W
        new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """NB2 dispersion by moments: Var = mu + alpha mu^2. Never negative (0 = Poisson)."""
    a = float(np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2))
    return max(0.0, a)


def fit_part(f, y, part: str) -> dict:
    X = design(f, part)
    y = np.asarray(y, float)
    beta = fit_poisson(X, y)
    mu = np.exp(X @ beta)
    return {"coef": [float(b) for b in beta], "alpha": nb_alpha(y, mu), "n": int(len(y)),
            "mean_y": float(np.mean(y))}


def mu_of(params: dict, f, part: str) -> np.ndarray:
    return np.exp(design(f, part) @ np.asarray(params["coef"], float))


def p_over(mu: np.ndarray, alpha: float, line: float) -> np.ndarray:
    """P(count > line) for x.5 lines under NB2(mu, alpha) (Poisson when alpha == 0)."""
    k = math.floor(float(line))
    mu = np.asarray(mu, float)
    if alpha <= 0:
        return 1.0 - stats.poisson.cdf(k, mu)
    n = 1.0 / alpha
    p = n / (n + mu)
    return 1.0 - stats.nbinom.cdf(k, n, p)


def params_sha(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
