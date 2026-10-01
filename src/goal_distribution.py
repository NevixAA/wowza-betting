"""§7 — a coherent score distribution. One model, every market derived from it. RESEARCH.

    from src.goal_distribution import DixonColes
    m = DixonColes().fit(train_df)
    probs = m.predict(test_df)      # p_over15, p_over25, p_over35, p_btts, p_home/draw/away

WHY A DISTRIBUTION RATHER THAN FOUR CLASSIFIERS. v9 trains Over 1.5, Over 2.5, Over 3.5 and BTTS
as unrelated binary models. Nothing makes them agree, and they routinely cannot all be true at
once: a fixture can come back with P(Over 1.5) BELOW P(Over 2.5), which is impossible — every
match with three goals also has two. Pro's `consistency_violations.csv` exists because this
happens. Four models also mean four sets of parameters fitted to the same ~26k fixtures, when
one score distribution explains all of them.

A score distribution cannot be incoherent. P(Over 2.5) is the tail of the same matrix that gives
P(BTTS) and P(Home), so the ordering holds by construction, and 1X2 comes free.

THE MODEL. Independent Poisson with a Dixon-Coles low-score correction:

    lambda_home = league_mu_home x attack[home] x defence[away]
    lambda_away = league_mu_away x attack[away] x defence[home]
    P(i,j)      = Poisson(i; lambda_home) x Poisson(j; lambda_away) x tau(i, j, rho)

`tau` is the Dixon-Coles adjustment on 0-0, 1-0, 0-1 and 1-1, where independent Poisson is
known to misfit — low-scoring games are more correlated than independence implies, which matters
directly for BTTS and Under markets.

STRICTLY AS-OF. Strengths come from matches strictly BEFORE the fixture being scored. Ratings
are built on the training block only and never updated with holdout results, so nothing a model
sees could have been unknown at kickoff.

NOT OPTIMISED FOR ROI. Per the brief this is judged on forecasting quality alone — log loss,
Brier, calibration, and RPS for 1X2 — never on simulated profit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MAX_GOALS = 10          # the matrix is truncated here; P(>10 goals in a half-decent league) ~ 0
PRIOR_MATCHES = 25.0    # empirical-Bayes shrinkage of a team's strength toward the league mean


def _tau(i: np.ndarray, j: np.ndarray, lh: float, la: float, rho: float) -> np.ndarray:
    """Dixon-Coles correction on the four lowest scorelines.

    Independent Poisson understates draws and 0-0 in particular. rho < 0 pushes probability
    into 0-0 and 1-1 and out of 1-0 and 0-1, which is the empirically observed shape.
    """
    t = np.ones_like(i, dtype=float)
    t = np.where((i == 0) & (j == 0), 1.0 - lh * la * rho, t)
    t = np.where((i == 0) & (j == 1), 1.0 + lh * rho, t)
    t = np.where((i == 1) & (j == 0), 1.0 + la * rho, t)
    t = np.where((i == 1) & (j == 1), 1.0 - rho, t)
    return np.clip(t, 1e-6, None)


@dataclass
class DixonColes:
    """Team strengths + a low-score correction, fitted per league."""
    rho: float = -0.05
    max_goals: int = MAX_GOALS
    prior: float = PRIOR_MATCHES
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    league_mu: dict = field(default_factory=dict)
    fitted_rows: int = 0

    def fit(self, df: pd.DataFrame) -> "DixonColes":
        """Estimate strengths from completed matches. Uses only the frame it is given."""
        d = df.dropna(subset=["home_goals", "away_goals", "home_team", "away_team", "league"])
        self.fitted_rows = len(d)
        for lg, g in d.groupby("league"):
            hg = pd.to_numeric(g["home_goals"], errors="coerce")
            ag = pd.to_numeric(g["away_goals"], errors="coerce")
            mu_h, mu_a = float(hg.mean()), float(ag.mean())
            if not np.isfinite(mu_h) or mu_h <= 0 or not np.isfinite(mu_a) or mu_a <= 0:
                continue
            self.league_mu[lg] = (mu_h, mu_a)
            # Attack = goals scored relative to the league's venue mean; defence = conceded.
            # Both shrunk toward 1.0, so a team with four matches does not get a 1.8 multiplier.
            for team in pd.unique(pd.concat([g["home_team"], g["away_team"]]).dropna()):
                h = g[g["home_team"] == team]
                a = g[g["away_team"] == team]
                n = len(h) + len(a)
                if n == 0:
                    continue
                scored = (pd.to_numeric(h["home_goals"], errors="coerce").sum()
                          + pd.to_numeric(a["away_goals"], errors="coerce").sum())
                conceded = (pd.to_numeric(h["away_goals"], errors="coerce").sum()
                            + pd.to_numeric(a["home_goals"], errors="coerce").sum())
                exp_s = len(h) * mu_h + len(a) * mu_a
                exp_c = len(h) * mu_a + len(a) * mu_h
                self.attack[(lg, team)] = float((scored + self.prior) / (exp_s + self.prior))
                self.defence[(lg, team)] = float((conceded + self.prior) / (exp_c + self.prior))
        return self

    def _lambdas(self, lg, home, away) -> tuple[float, float]:
        mu = self.league_mu.get(lg)
        if mu is None:
            mu = (float(np.mean([m[0] for m in self.league_mu.values()])) if self.league_mu else 1.4,
                  float(np.mean([m[1] for m in self.league_mu.values()])) if self.league_mu else 1.1)
        ah = self.attack.get((lg, home), 1.0)
        dh = self.defence.get((lg, home), 1.0)
        aa = self.attack.get((lg, away), 1.0)
        da = self.defence.get((lg, away), 1.0)
        return (float(np.clip(mu[0] * ah * da, 0.05, 6.0)),
                float(np.clip(mu[1] * aa * dh, 0.05, 6.0)))

    def matrix(self, lh: float, la: float) -> np.ndarray:
        """Joint score probability matrix, Dixon-Coles corrected and renormalised."""
        from scipy.stats import poisson
        k = np.arange(self.max_goals + 1)
        ph, pa = poisson.pmf(k, lh), poisson.pmf(k, la)
        m = np.outer(ph, pa)
        i, j = np.meshgrid(k, k, indexing="ij")
        m = m * _tau(i, j, lh, la, self.rho)
        return m / m.sum()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Every market, derived from one matrix per fixture — coherent by construction."""
        out = []
        for r in df.itertuples(index=False):
            lh, la = self._lambdas(getattr(r, "league", None),
                                   getattr(r, "home_team", None), getattr(r, "away_team", None))
            m = self.matrix(lh, la)
            k = np.arange(self.max_goals + 1)
            i, j = np.meshgrid(k, k, indexing="ij")
            tot = i + j
            out.append({
                "lambda_home": lh, "lambda_away": la, "lambda_total": lh + la,
                "p_over15": float(m[tot > 1.5].sum()),
                "p_over25": float(m[tot > 2.5].sum()),
                "p_over35": float(m[tot > 3.5].sum()),
                "p_btts": float(m[1:, 1:].sum()),
                "p_home": float(np.tril(m, -1).sum()),
                "p_draw": float(np.trace(m)),
                "p_away": float(np.triu(m, 1).sum()),
            })
        return pd.DataFrame(out, index=df.index)


def rps(p_home: np.ndarray, p_draw: np.ndarray, p_away: np.ndarray,
        outcome: np.ndarray) -> float:
    """Ranked Probability Score for 1X2 — the proper scoring rule for an ORDERED outcome.

    Log loss treats home/draw/away as unrelated labels. RPS knows a draw sits between the two
    wins, so predicting a home win when it was a draw is penalised less than predicting a home
    win when it was an away win. Lower is better. `outcome` is 0=home, 1=draw, 2=away.
    """
    P = np.cumsum(np.column_stack([p_home, p_draw, p_away]), axis=1)
    O = np.zeros((len(outcome), 3))
    O[np.arange(len(outcome)), outcome.astype(int)] = 1.0
    O = np.cumsum(O, axis=1)
    return float(((P - O) ** 2).sum(axis=1).mean() / 2.0)


def coherence_violations(p: pd.DataFrame) -> dict:
    """Count orderings that are logically impossible. A distribution must score zero here.

    Separate binary classifiers have no mechanism preventing these; this is the structural
    argument for the distribution, independent of whether it also scores better.
    """
    return {
        "over15_below_over25": int((p["p_over15"] < p["p_over25"]).sum()),
        "over25_below_over35": int((p["p_over25"] < p["p_over35"]).sum()),
        "btts_above_over15": int((p["p_btts"] > p["p_over15"]).sum()),
        "1x2_not_summing_to_1": int((np.abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) > 1e-6).sum()),
        "n": int(len(p)),
    }
