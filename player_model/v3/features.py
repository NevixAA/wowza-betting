"""V3 feature families used by Fantasy (port of Pro src/player_v3/features.py, same formulas).

  BASE_V9      v9's PLAYER_FEATURE_COLS as stored in player_history
  FORM_SHRUNK  per-90 rates (last 5, last 10, career) shrunk toward a position x league prior:
               n/(n+k) rate + k/(n+k) prior, n in 90-minute blocks, k = 5
  TEAM_ENV     the team's and the opponent's recent SOT / shots for and against (last 5, last 10)

Only the columns those families need are computed (Pro's FORM_MH also had 3-game, season and EWMA
rates; they feed nothing Fantasy uses). The prior is fitted on rows before `prior_cutoff` only.
"""
from __future__ import annotations

import pandas as pd

from player_model.v3 import data as D

STATS = ["shots_on_target", "shots_total", "goals", "assists", "key_passes", "fouls_drawn",
         "fouls_committed", "yellow_cards", "duels_won", "dribbles_success", "dribbles_attempted"]
K_SHRINK = 5.0


def base_cols() -> list[str]:
    from player_model import config as pcfg
    return list(pcfg.PLAYER_FEATURE_COLS)


def form(h: pd.DataFrame) -> pd.DataFrame:
    cols = STATS + ["minutes"]
    roll = D.asof_rolling(h, "player_id", cols, windows=(5, 10), how="sum")
    cum = D.asof_rolling(h, "player_id", cols, windows=(), expanding=True, how="sum")
    out = {}
    for s in STATS:
        for w in (5, 10):
            out[f"{s}_p90_l{w}"] = D.per90(roll[f"{s}_l{w}"], roll[f"minutes_l{w}"])
        out[f"{s}_p90_career"] = D.per90(cum[f"{s}_cum"], cum["minutes_cum"])
    out["minutes_career"] = cum["minutes_cum"]
    out["minutes_l5_sum"] = roll["minutes_l5"]
    return pd.DataFrame(out, index=h.index)


def shrunk(h: pd.DataFrame, f: pd.DataFrame, prior_cutoff: pd.Timestamp, k: float = K_SHRINK) -> pd.DataFrame:
    tr = h["date"] < prior_cutoff
    g = h[tr].groupby(["position", "league"])
    prior = {s: (g[s].sum() / g["minutes"].sum() * 90.0) for s in STATS}
    key = pd.MultiIndex.from_frame(h[["position", "league"]])
    out = {}
    for s in STATS:
        p = pd.Series(prior[s].reindex(key).to_numpy(), index=h.index)
        p = p.fillna(float((h.loc[tr, s].sum() / h.loc[tr, "minutes"].sum()) * 90.0))
        for w in ("l5", "l10", "career"):
            mins = f["minutes_career"] if w == "career" else f["minutes_l5_sum"] * (2 if w == "l10" else 1)
            n = (mins / 90.0).fillna(0)
            out[f"{s}_shr_{w}"] = (n / (n + k)) * f[f"{s}_p90_{w}"].fillna(p) + (k / (n + k)) * p
    return pd.DataFrame(out, index=h.index)


def team_env(h: pd.DataFrame, tt: pd.DataFrame) -> pd.DataFrame:
    tr = D.asof_rolling(tt, "team", ["team_sot", "team_shots", "opp_sot", "opp_shots"], windows=(5, 10))
    env = tt[["fixture_id", "team"]].join(tr).rename(columns=lambda c: c if c in ("fixture_id", "team") else f"env_{c}")
    opp_env = env.rename(columns={"team": "opponent", **{c: c.replace("env_", "oppenv_") for c in env.columns if c.startswith("env_")}})
    e = h[["fixture_id", "team", "opponent"]].merge(env, on=["fixture_id", "team"], how="left") \
        .merge(opp_env, on=["fixture_id", "opponent"], how="left")
    e.index = h.index
    return e.drop(columns=["fixture_id", "team", "opponent"])


def build(h: pd.DataFrame, prior_cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Appearance rows with BASE_V9 + FORM_SHRUNK + TEAM_ENV columns, and the family -> columns map."""
    fam = {"BASE_V9": [c for c in base_cols() if c in h.columns]}
    s = shrunk(h, form(h), prior_cutoff)
    e = team_env(h, D.team_totals(h))
    fam["FORM_SHRUNK"], fam["TEAM_ENV"] = list(s.columns), list(e.columns)
    return pd.concat([h, s, e], axis=1), fam
