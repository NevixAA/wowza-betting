"""V3 data layer: player history, team totals and AS-OF rolling features (port of Pro src/player_v3/data.py).

THE ONE RULE. A feature for match m may only use matches strictly before m. Every rolling helper
shifts by one row inside its group before aggregating; groups are sorted by (date, fixture_id).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = Path(__file__).resolve().parents[2] / "player_history.parquet"

RAW = ["fixture_id", "player_id", "player_name", "date", "league", "season", "team", "opponent",
       "is_home", "position", "minutes", "started", "goals", "assists", "shots_total",
       "shots_on_target", "yellow_cards", "red_cards", "key_passes", "fouls_committed",
       "fouls_drawn", "duels_total", "duels_won", "tackles_total", "interceptions",
       "dribbles_attempted", "dribbles_success", "dribbles_past", "saves", "goals_conceded",
       "passes_total", "offsides", "height_cm", "age", "rating"]


def load_history(cols: list[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    import glob

    import pyarrow.parquet as pq
    p = Path(path or HISTORY)
    want = cols or RAW
    first = sorted(glob.glob(str(p / "*.parquet")))[0] if p.is_dir() else str(p)
    have = set(pq.read_schema(first).names)
    d = pd.read_parquet(p, columns=[c for c in want if c in have])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "player_id", "fixture_id"])
    d["player_id"] = d["player_id"].astype("int64")
    d["fixture_id"] = d["fixture_id"].astype("int64")
    if "position" in d.columns:
        d["position"] = d["position"].astype(str).str.upper().str[:1].replace({"": "U", "N": "U"})
    return d.sort_values(["date", "fixture_id", "player_id"]).reset_index(drop=True)


TEAM_SUM = {"shots_total": "team_shots", "shots_on_target": "team_sot", "goals": "team_goals",
            "assists": "team_assists", "key_passes": "team_kp", "fouls_committed": "team_fouls",
            "fouls_drawn": "team_fouls_drawn", "yellow_cards": "team_cards", "minutes": "team_minutes",
            "duels_total": "team_duels", "dribbles_attempted": "team_dribbles"}


def team_totals(h: pd.DataFrame) -> pd.DataFrame:
    """Per (fixture, team): realised totals of that match plus the opponent's (= conceded)."""
    have = {k: v for k, v in TEAM_SUM.items() if k in h.columns}
    t = (h.groupby(["fixture_id", "team"], sort=False)
           .agg(date=("date", "first"), league=("league", "first"), opponent=("opponent", "first"),
                **{v: (k, "sum") for k, v in have.items()})
           .reset_index())
    opp = t[["fixture_id", "team", *have.values()]].rename(
        columns={"team": "opponent", **{v: v.replace("team_", "opp_") for v in have.values()}})
    return t.merge(opp, on=["fixture_id", "opponent"], how="left")


def asof_rolling(frame: pd.DataFrame, key: str | list[str], cols: list[str],
                 windows=(3, 5, 10), ewm_spans=(), expanding: bool = False,
                 how: str = "mean", prefix: str = "") -> pd.DataFrame:
    """Shifted rolling aggregates of `cols` within `key`, aligned to `frame`'s index."""
    keys = [key] if isinstance(key, str) else list(key)
    order = frame.sort_values(["date", "fixture_id"], kind="mergesort").index
    f = frame.loc[order, keys + cols]
    g = f.groupby(keys, sort=False)
    out = {}
    for c in cols:
        prev = g[c].shift(1)
        pg = prev.groupby([f[k] for k in keys], sort=False)
        for w in windows:
            r = pg.rolling(w, min_periods=1)
            out[f"{prefix}{c}_l{w}"] = (r.mean() if how == "mean" else r.sum()).reset_index(
                level=list(range(len(keys))), drop=True)
        for s in ewm_spans:
            out[f"{prefix}{c}_ewm{s}"] = pg.transform(lambda x: x.ewm(span=s, min_periods=1).mean())
        if expanding:
            out[f"{prefix}{c}_cum"] = (pg.cumsum() if how == "sum" else
                                       pg.transform(lambda x: x.expanding().mean()))
    res = pd.DataFrame(out, index=order)
    return res.reindex(frame.index)


def per90(num: pd.Series, minutes: pd.Series) -> pd.Series:
    return num / minutes.replace(0, np.nan) * 90.0
