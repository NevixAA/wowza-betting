"""As-of LEAGUE-LEVEL features (research, 2026-11-01). Challenger A of the league-level / market-anchor study.

WHY. The team models' inputs are mostly team-relative ("attack strength vs league average"), and the
logit-calibration study showed the base models average leagues away: on the repaired Standard O/U
holdout the across-league spread of mean P(over) is 0.014 against 0.055 actual. These features give
the base models the league's own scoring level explicitly.

STRICTLY AS-OF. A row dated d sees only league matches with date < d. Matches on the same day never
see each other (aggregation is per (league, date), then cumulated and shifted one DATE), so a
Saturday fixture cannot learn from another Saturday result. Unplayed rows (no full-time score)
receive features and contribute nothing.

Features (NaN until a league has MIN_PRIOR prior matches - the model's imputer handles them):
    lg_goals_pg_{N}   mean total goals over the league's last N prior matches
    lg_o25_rate_{N}   share of those matches over 2.5      (also o15 / o35 / btts)
    lg_goals_pg_season  season-to-date mean total goals (season column; prior dates only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (100, 300)
MIN_PRIOR = 30
TARGETS = {"o25": lambda g: g > 2.5, "o15": lambda g: g > 1.5, "o35": lambda g: g > 3.5}


def feature_names() -> list[str]:
    out = []
    for n in WINDOWS:
        out += [f"lg_goals_pg_{n}", f"lg_o25_rate_{n}", f"lg_o15_rate_{n}", f"lg_o35_rate_{n}", f"lg_btts_rate_{n}"]
    return out + ["lg_goals_pg_season"]


def _window(cum: np.ndarray, cum_n: np.ndarray, prev_idx: np.ndarray, n: int) -> np.ndarray:
    """Mean over the last n matches strictly before each date (by cumulative counts per date)."""
    out = np.full(len(prev_idx), np.nan)
    for k, p in enumerate(prev_idx):
        if p < 0:
            continue
        tot_n = cum_n[p]
        if tot_n < MIN_PRIOR:
            continue
        start_n = max(tot_n - n, 0)
        # first date index whose cumulative count exceeds start_n -> window starts inside it;
        # whole-date granularity: take the dates after the last one with cum_n <= start_n
        j = np.searchsorted(cum_n, start_n, side="right") - 1
        base, base_n = (cum[j], cum_n[j]) if j >= 0 else (0.0, 0.0)
        cnt = tot_n - base_n
        out[k] = (cum[p] - base) / cnt if cnt > 0 else np.nan
    return out


def add_league_level(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with feature_names() added. Needs date, league, home_goals, away_goals (+ season)."""
    d = df.copy()
    d["_date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    hg = pd.to_numeric(d.get("home_goals"), errors="coerce")
    ag = pd.to_numeric(d.get("away_goals"), errors="coerce")
    played = hg.notna() & ag.notna()
    tg = (hg + ag).where(played)
    d["_n"] = played.astype(float)
    d["_g"] = tg.fillna(0.0)
    for k, f in TARGETS.items():
        d[f"_{k}"] = f(tg).where(played).fillna(0).astype(float)
    d["_btts"] = ((hg > 0) & (ag > 0)).where(played).fillna(0).astype(float)
    sums = ["_n", "_g", "_o25", "_o15", "_o35", "_btts"]
    for c in feature_names():
        d[c] = np.nan
    for lg, idx in d.groupby("league").groups.items():
        sub = d.loc[idx]
        daily = sub.groupby("_date")[sums].sum().sort_index()
        dates = daily.index.to_numpy()
        cum = daily.cumsum()
        cum_n = cum["_n"].to_numpy()
        # for each row: index of the last date strictly before its date
        pos = np.searchsorted(dates, sub["_date"].to_numpy(), side="left") - 1
        for n in WINDOWS:
            d.loc[idx, f"lg_goals_pg_{n}"] = _window(cum["_g"].to_numpy(), cum_n, pos, n)
            for k in ("o25", "o15", "o35", "btts"):
                d.loc[idx, f"lg_{k}_rate_{n}"] = _window(cum[f"_{k}"].to_numpy(), cum_n, pos, n)
        if "season" in sub.columns:
            for s, sidx in sub.groupby("season").groups.items():
                ss = sub.loc[sidx]
                sd = ss.groupby("_date")[["_n", "_g"]].sum().sort_index().cumsum()
                sp = np.searchsorted(sd.index.to_numpy(), ss["_date"].to_numpy(), side="left") - 1
                gn, gg = sd["_n"].to_numpy(), sd["_g"].to_numpy()
                val = np.array([gg[p] / gn[p] if p >= 0 and gn[p] >= 10 else np.nan for p in sp])
                d.loc[sidx, "lg_goals_pg_season"] = val
    return d.drop(columns=["_date"] + sums)
