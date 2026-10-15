"""Expected-minutes model (port of Pro src/player_v3/minutes.py, same features and HGB settings).

THE POPULATION PROBLEM. player_history holds only APPEARANCES - a benched or absent player has no
row - so a model trained on it can never learn P(start). The panel rebuilds every club fixture inside
each player x club tenure; a fixture with no appearance is 0 minutes, started 0.

LIVE EXTENSION (new in the port). In the study a tenure ended at the player's last appearance - known
only in hindsight. Live, a player who missed his club's last three matches must look absent three
times, not "played last match". So for each player's CURRENT club (his latest appearance, within
`recent_days`) the panel is extended to the club's latest fixture, and one synthetic NEXT row
(fixture_id < 0, dated `asof`) is added for the projection. Training uses tenure rows only, exactly
as the study did; extended and synthetic rows are only ever predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_model.v3 import data as D

THRESH = (60, 75, 90)
FEATS = ["minutes_l1", "minutes_l3", "minutes_l5", "minutes_l10", "started_l1", "started_l3",
         "started_l5", "started_l10", "appeared_l3", "appeared_l5", "appeared_l10", "m60_l5",
         "m60_l10", "minutes_cum", "started_cum", "appeared_cum", "app_minutes_l10",
         "app_minutes_cum", "app_started_l10", "app_started_cum", "season_started_cum",
         "days_since_app", "rest_days", "congestion14", "n_prior", "age", "home",
         "pos_D", "pos_M", "pos_F", "pos_G"]
HGB = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=100,
           early_stopping=False, random_state=0)


def build_panel(h: pd.DataFrame, asof: pd.Timestamp | None = None, recent_days: int = 120) -> pd.DataFrame:
    tt = D.team_totals(h)[["fixture_id", "team", "date", "league", "opponent"]]
    ten = h.groupby(["player_id", "team"])["date"].agg(first="min", last="max").reset_index()
    ten["end"] = ten["last"]
    if asof is not None:
        latest = h.sort_values("date").groupby("player_id").tail(1)[["player_id", "team", "date"]]
        latest = latest[latest["date"] >= asof - pd.Timedelta(days=recent_days)]
        club_last = tt.groupby("team")["date"].max()
        cur = ten.set_index(["player_id", "team"]).index.isin(
            latest.set_index(["player_id", "team"]).index)
        ten.loc[cur, "end"] = ten.loc[cur, "team"].map(club_last).fillna(ten.loc[cur, "last"])
        ten["current"] = cur
    else:
        ten["current"] = False
    rows = []
    for team, g in ten.groupby("team", sort=False):
        fx = tt[tt["team"] == team].sort_values("date")
        if fx.empty:
            continue
        dates = fx["date"].to_numpy()
        for r in g.itertuples(index=False):
            lo, hi = np.searchsorted(dates, r.first, "left"), np.searchsorted(dates, r.end, "right")
            if hi - lo < 2:
                continue
            part = fx.iloc[lo:hi]
            rows.append(pd.DataFrame({"player_id": r.player_id, "team": team,
                                      "fixture_id": part["fixture_id"].to_numpy(),
                                      "date": part["date"].to_numpy(),
                                      "league": part["league"].to_numpy(),
                                      "opponent": part["opponent"].to_numpy(),
                                      "in_tenure": part["date"].to_numpy() <= np.datetime64(r.last)}))
            if r.current:
                rows.append(pd.DataFrame({"player_id": [r.player_id], "team": [team],
                                          "fixture_id": [-1], "date": [asof], "league": [part["league"].iloc[-1]],
                                          "opponent": [np.nan], "in_tenure": [False]}))
    p = pd.concat(rows, ignore_index=True)
    app = h[["player_id", "fixture_id", "minutes", "started", "position", "age", "is_home"]]
    p = p.merge(app, on=["player_id", "fixture_id"], how="left")
    p["appeared"] = p["minutes"].notna().astype(int)
    p["minutes"] = p["minutes"].fillna(0).clip(0, 120)
    p["started"] = p["started"].fillna(0).astype(int)
    p = p.sort_values(["player_id", "date", "fixture_id"], kind="mergesort").reset_index(drop=True)
    for c in ("position", "age"):
        p[c] = p.groupby("player_id")[c].ffill()
    home = h.groupby(["fixture_id", "team"])["is_home"].first().rename("home").reset_index()
    p = p.merge(home, on=["fixture_id", "team"], how="left")
    # synthetic next rows share fixture_id -1: give each a unique negative id so as-of ordering is stable
    syn = p["fixture_id"] == -1
    p.loc[syn, "fixture_id"] = -(np.arange(int(syn.sum())) + 1)
    return p.drop(columns=["is_home"]).sort_values(["date", "fixture_id", "player_id"]).reset_index(drop=True)


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    key = ["player_id", "team"]
    p["app_minutes"] = p["minutes"].where(p["appeared"] == 1)
    p["app_started"] = p["started"].where(p["appeared"] == 1)
    for k in THRESH:
        p[f"m{k}"] = (p["minutes"] >= k).astype(int)
    roll = D.asof_rolling(p, key, ["minutes", "started", "appeared", "m60"], windows=(1, 3, 5, 10))
    cum = D.asof_rolling(p, key, ["minutes", "started", "appeared"], windows=(), expanding=True)
    appv = D.asof_rolling(p, key, ["app_minutes", "app_started"], windows=(10,), expanding=True)
    p = pd.concat([p, roll, cum, appv], axis=1)
    p["season"] = np.where(p["date"].dt.month >= 7, p["date"].dt.year, p["date"].dt.year - 1)
    sea = D.asof_rolling(p, key + ["season"], ["started"], windows=(), expanding=True, prefix="season_")
    p = pd.concat([p, sea], axis=1)
    order = p.sort_values(["date", "fixture_id"], kind="mergesort").index
    q = p.loc[order]
    last_app = q["date"].where(q["appeared"] == 1).groupby([q["player_id"], q["team"]]).shift(1)
    last_app = last_app.groupby([q["player_id"], q["team"]]).ffill()
    p.loc[order, "days_since_app"] = (q["date"] - last_app).dt.days
    prev_fx = q.groupby([q["player_id"], q["team"]])["date"].shift(1)
    p.loc[order, "rest_days"] = (q["date"] - prev_fx).dt.days
    p["n_prior"] = p.groupby(key).cumcount()
    tf = (p[["team", "fixture_id", "date"]].drop_duplicates(["team", "fixture_id"])
            .sort_values(["team", "date"]).set_index("date"))
    tf["congestion14"] = (tf.groupby("team")["fixture_id"].transform(
        lambda s: s.rolling("14D", closed="left").count()))
    p = p.merge(tf.reset_index()[["team", "fixture_id", "congestion14"]], on=["team", "fixture_id"], how="left")
    return p[p["n_prior"] >= 1].reset_index(drop=True)


def design(p: pd.DataFrame) -> pd.DataFrame:
    X = p.copy()
    for c in "DMFG":
        X[f"pos_{c}"] = (X["position"] == c).astype(float)
    X["home"] = pd.to_numeric(X["home"], errors="coerce")
    X["days_since_app"] = X["days_since_app"].clip(upper=365)
    return X[FEATS].astype(float)


def fit_predict(p: pd.DataFrame) -> pd.DataFrame:
    """Fit on tenure rows (as the study), predict every row incl. extended / synthetic ones."""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGC
    from sklearn.ensemble import HistGradientBoostingRegressor as HGR
    X = design(p)
    tr = p["in_tenure"].to_numpy(bool)
    out = pd.DataFrame({"player_id": p["player_id"], "fixture_id": p["fixture_id"], "date": p["date"],
                        "team": p["team"]})
    for t, y in {"start": p["started"], **{f"m{k}": p[f"m{k}"] for k in THRESH}}.items():
        out[f"xm_{t}"] = HGC(**HGB).fit(X[tr], y.to_numpy()[tr]).predict_proba(X)[:, 1]
    out["xm_minutes"] = np.clip(HGR(**HGB).fit(X[tr], p["minutes"].to_numpy()[tr]).predict(X), 0, 95)
    return out
