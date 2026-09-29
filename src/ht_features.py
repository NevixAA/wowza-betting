"""Half-time-specific features. Built for the HT models, used by nothing else.

WHY A SEPARATE MODULE. The HT models were trained on the STANDARD feature set — a set designed
for full-time Over/Under 2.5. It carries corners, fouls, shot ratios and the full-time O/U
implied probabilities, and exactly four genuinely half-time columns, each a 5-match rolling
binary rate that can take only eleven distinct values. A 5-match rolling average of a coin-ish
event is mostly noise: measured alone it reaches AUC 0.522, barely above the 0.500 of a constant.

WHAT THIS BUILDS INSTEAD. Half-time goals are a LOW-COUNT process (~1.0 goals per match by the
break), so the right shape is a rate model, not a rolling average:

    lambda_home = league_ht_home_mean  x  home_attack  x  away_defence
    lambda_away = league_ht_away_mean  x  away_attack  x  home_defence
    P(at least one HT goal) = 1 - exp(-(lambda_home + lambda_away))

This is the standard multiplicative (Dixon-Coles style) construction, applied to the first half
only. It pools information the rolling rate throws away: every match a team has played, weighted
by recency, shrunk toward the league mean so a team with six matches does not get a confident
rate.

SHRINKAGE IS THE POINT, NOT A DETAIL. With ~0.5 HT goals per team per match, a team needs many
matches before its own rate says anything. Empirical-Bayes shrinkage toward the league mean is
what stops a 3-match hot streak from becoming a 0.9 attack multiplier. `PRIOR_MATCHES` is the
number of league-average matches blended in; at 30 a team needs roughly a full season before its
own history dominates.

EVERYTHING IS AS-OF. Each fixture sees only matches played STRICTLY BEFORE it. The accumulators
walk the frame in date order and are updated AFTER the row is emitted, so a fixture can never
contribute to its own features. This is the same discipline the walk-forward backtests use, and
it is enforced here rather than assumed, because a rate model is unusually easy to leak through:
a single groupby-transform over the whole frame would look right and be wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: League-average matches blended into every team rate. Higher = more shrinkage.
PRIOR_MATCHES = 30.0

#: Exponential recency half-life, in matches. Older form still counts, just less.
HALF_LIFE = 40.0

#: Multiplicative strengths are clipped here. A team is not four times more dangerous than
#: average in the first half; a value that extreme is a small-sample artifact.
STRENGTH_LO, STRENGTH_HI = 0.45, 2.20

HT_FEATURE_COLS = [
    "ht_lambda_home", "ht_lambda_away", "ht_lambda_total",
    "ht_p_over05_poisson", "ht_p_over15_poisson",
    "ht_home_attack", "ht_home_defence", "ht_away_attack", "ht_away_defence",
    "ht_league_base", "ht_league_over05_base",
    "ht_home_n", "ht_away_n",
    "ht_home_over05_ewm", "ht_away_over05_ewm",
    "ht_home_scored_ewm", "ht_home_conceded_ewm",
    "ht_away_scored_ewm", "ht_away_conceded_ewm",
    "ht_h2h_over05", "ht_h2h_n",
    "ht_ft_share_home", "ht_ft_share_away",
]


class _Acc:
    """Recency-weighted accumulator: effective match count and weighted mean."""

    __slots__ = ("w", "sw", "decay")

    def __init__(self, half_life: float = HALF_LIFE):
        self.w = 0.0          # sum of weights
        self.sw = 0.0         # sum of weight * value
        self.decay = 0.5 ** (1.0 / max(half_life, 1.0))

    def add(self, v: float) -> None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return
        self.w = self.w * self.decay + 1.0
        self.sw = self.sw * self.decay + float(v)

    @property
    def n(self) -> float:
        return self.w

    def mean(self, prior: float, prior_w: float = PRIOR_MATCHES) -> float:
        """Shrunk toward `prior`. With no history this returns the prior exactly."""
        return (self.sw + prior * prior_w) / (self.w + prior_w)


def build_ht_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach HT_FEATURE_COLS. Requires date, league, home_team, away_team,
    ht_home_goals, ht_away_goals; uses home_goals/away_goals when present.

    Rows whose half-time score is missing still RECEIVE features (they may be scored later);
    they simply do not UPDATE the accumulators, because there is nothing to learn from them.
    """
    need = {"date", "league", "home_team", "away_team", "ht_home_goals", "ht_away_goals"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"build_ht_features needs {sorted(missing)}")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    # Stable sort: ties on date must keep a deterministic order, or two runs disagree.
    d = d.sort_values(["date", "league", "home_team", "away_team"],
                      kind="mergesort").reset_index(drop=True)

    hh = pd.to_numeric(d["ht_home_goals"], errors="coerce").to_numpy(float)
    ha = pd.to_numeric(d["ht_away_goals"], errors="coerce").to_numpy(float)
    fh = pd.to_numeric(d.get("home_goals"), errors="coerce").to_numpy(float) \
        if "home_goals" in d.columns else np.full(len(d), np.nan)
    fa = pd.to_numeric(d.get("away_goals"), errors="coerce").to_numpy(float) \
        if "away_goals" in d.columns else np.full(len(d), np.nan)
    tot = hh + ha

    lg = d["league"].astype(str).to_numpy()
    home = d["home_team"].astype(str).to_numpy()
    away = d["away_team"].astype(str).to_numpy()

    # league accumulators
    lg_h, lg_a, lg_o05 = {}, {}, {}
    # team accumulators: scored/conceded at HT, split by venue
    t_hs, t_hc, t_as, t_ac = {}, {}, {}, {}
    t_o05_h, t_o05_a = {}, {}
    t_share_h, t_share_a = {}, {}          # HT goals as a fraction of the team's FT goals
    h2h: dict[tuple, _Acc] = {}

    def acc(store: dict, key) -> _Acc:
        a = store.get(key)
        if a is None:
            a = store[key] = _Acc()
        return a

    out = {c: np.full(len(d), np.nan) for c in HT_FEATURE_COLS}
    # Global fallbacks so the very first fixtures are not NaN-blank.
    G_H, G_A, G_O05 = 0.55, 0.45, 0.70

    for i in range(len(d)):
        L, H, A = lg[i], home[i], away[i]
        la_h, la_a, la_o = acc(lg_h, L), acc(lg_a, L), acc(lg_o05, L)
        base_h = la_h.mean(G_H, 20.0)
        base_a = la_a.mean(G_A, 20.0)
        base_o = la_o.mean(G_O05, 20.0)

        hs, hc = acc(t_hs, H), acc(t_hc, H)          # this team, at HOME
        aws, awc = acc(t_as, A), acc(t_ac, A)        # that team, AWAY

        # Multiplicative strengths, each shrunk to its own league baseline.
        h_att = np.clip(hs.mean(base_h) / max(base_h, 1e-6), STRENGTH_LO, STRENGTH_HI)
        h_def = np.clip(hc.mean(base_a) / max(base_a, 1e-6), STRENGTH_LO, STRENGTH_HI)
        a_att = np.clip(aws.mean(base_a) / max(base_a, 1e-6), STRENGTH_LO, STRENGTH_HI)
        a_def = np.clip(awc.mean(base_h) / max(base_h, 1e-6), STRENGTH_LO, STRENGTH_HI)

        lam_h = base_h * h_att * a_def
        lam_a = base_a * a_att * h_def
        lam = lam_h + lam_a

        out["ht_lambda_home"][i] = lam_h
        out["ht_lambda_away"][i] = lam_a
        out["ht_lambda_total"][i] = lam
        out["ht_p_over05_poisson"][i] = 1.0 - np.exp(-lam)
        out["ht_p_over15_poisson"][i] = 1.0 - np.exp(-lam) * (1.0 + lam)
        out["ht_home_attack"][i] = h_att
        out["ht_home_defence"][i] = h_def
        out["ht_away_attack"][i] = a_att
        out["ht_away_defence"][i] = a_def
        out["ht_league_base"][i] = base_h + base_a
        out["ht_league_over05_base"][i] = base_o
        out["ht_home_n"][i] = hs.n
        out["ht_away_n"][i] = aws.n
        out["ht_home_scored_ewm"][i] = hs.mean(base_h)
        out["ht_home_conceded_ewm"][i] = hc.mean(base_a)
        out["ht_away_scored_ewm"][i] = aws.mean(base_a)
        out["ht_away_conceded_ewm"][i] = awc.mean(base_h)
        out["ht_home_over05_ewm"][i] = acc(t_o05_h, H).mean(base_o)
        out["ht_away_over05_ewm"][i] = acc(t_o05_a, A).mean(base_o)
        out["ht_ft_share_home"][i] = acc(t_share_h, H).mean(0.42, 10.0)
        out["ht_ft_share_away"][i] = acc(t_share_a, A).mean(0.42, 10.0)

        k = (L, *sorted((H, A)))
        hx = acc(h2h, k)
        out["ht_h2h_over05"][i] = hx.mean(base_o, 5.0)
        out["ht_h2h_n"][i] = hx.n

        # ── UPDATE ONLY AFTER EMITTING. This ordering is the no-leakage guarantee. ──
        if not (np.isnan(hh[i]) or np.isnan(ha[i])):
            la_h.add(hh[i]); la_a.add(ha[i]); la_o.add(1.0 if tot[i] >= 1 else 0.0)
            hs.add(hh[i]); hc.add(ha[i])
            aws.add(ha[i]); awc.add(hh[i])
            acc(t_o05_h, H).add(1.0 if tot[i] >= 1 else 0.0)
            acc(t_o05_a, A).add(1.0 if tot[i] >= 1 else 0.0)
            hx.add(1.0 if tot[i] >= 1 else 0.0)
            if not np.isnan(fh[i]) and fh[i] > 0:
                acc(t_share_h, H).add(hh[i] / fh[i])
            if not np.isnan(fa[i]) and fa[i] > 0:
                acc(t_share_a, A).add(ha[i] / fa[i])

    for c, v in out.items():
        d[c] = v
    return d


def attach_ht_features(historical: pd.DataFrame,
                       upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for UPCOMING fixtures, built from `historical`.

    THIS EXISTS BECAUSE TRAINING AND PREDICT USE DIFFERENT BUILDERS. Training goes through
    `feature_engineering.build_features`; prediction goes through `build_upcoming_features`. A
    feature present in one and absent in the other is the single most dangerous shape in this
    codebase: at predict time `model._prep` imputes an all-NaN column to 0.0, the scaler maps
    0.0 to roughly z=-37, and the logistic base model collapses. That is documented in
    pipeline.py's `_NF_DROP` block, where exactly this crushed new-format P(over) from ~0.51
    to ~0.36 before anyone noticed.

    So the two paths must produce the SAME columns by construction, which is what this does:
    it appends the upcoming rows to the historical frame, runs the one accumulator, and returns
    only the upcoming rows. Upcoming fixtures have no half-time score, so they receive features
    and update nothing — every one of them sees the full history and none sees another.

    With `upcoming=None` it simply returns the historical frame with features attached, which
    is what the training path wants.
    """
    if upcoming is None or upcoming.empty:
        return build_ht_features(historical)

    up = upcoming.copy()
    for c in ("ht_home_goals", "ht_away_goals"):
        if c not in up.columns:
            up[c] = np.nan
        else:
            # A fixture that has not kicked off cannot have a half-time score. If one is
            # present it is stale data, and letting it update the accumulator would leak.
            up[c] = np.nan
    up["_is_upcoming"] = True

    hist = historical.copy()
    hist["_is_upcoming"] = False
    # Align the two frames in ONE reindex each. Assigning missing columns in a loop triggers
    # pandas' fragmentation warning and is genuinely slow here — this runs inside the 15-minute
    # predict loop, where wall-clock is the binding constraint, not quota.
    cols = sorted(set(hist.columns) | set(up.columns))
    both = build_ht_features(
        pd.concat([hist.reindex(columns=cols), up.reindex(columns=cols)], ignore_index=True))
    out = both[both["_is_upcoming"] == True].drop(columns=["_is_upcoming"])  # noqa: E712
    return out.reset_index(drop=True)
