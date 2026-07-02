"""FANTASY tips family — FPL-style expected-points projections from the prop models.

Premier League only (v1). Fantasy is a PREDICTION CONTEST (no bookmaker / vig / edge),
so the prop model's accuracy (AUC 0.69-0.86) converts directly to value — unlike prop
BETTING, which is efficiently priced (see memory project_prop_edge_backtest).

This family is fully separate from the SNIPER/MARKSMAN betting family: no odds, no edge,
no tiers — just the model's raw probabilities ranked into fantasy points.

v1 uses each player's LATEST form row (parquet). When PL resumes (~Aug) the upcoming-
fixture / opponent-adjusted path can be layered on via build_upcoming_features.
Enhancement TODO: clean-sheet points for DEF/GK from the team O/U model (v1 = attack only).
"""
from __future__ import annotations
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

from player_model import config
from player_model.model import load_model, predict_proba

FANTASY_LEAGUE = "Premier League"
GOAL_PTS = {"F": 4.0, "M": 5.0, "D": 6.0, "G": 6.0}   # FPL goal points by position
_PARQUET = Path(__file__).resolve().parents[1] / "player_history.parquet"
_OUT = Path(__file__).resolve().parents[1] / "output" / "fantasy_tips.csv"


def _ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def build_fantasy_projections(parquet_path: Path | None = None, min_minutes: float = 45.0) -> pd.DataFrame:
    """Return per-player FPL-style expected-points projections for the fantasy league.

    fantasy_pts = P(goal)*GOAL_PTS[pos] + P(assist)*3 + P(sot2)*1 (bonus proxy)
                  + appearance (2 if minutes_pg>=60 else 1)
    """
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = df[df["league"] == FANTASY_LEAGUE].copy()
    if pl.empty:
        return pd.DataFrame()
    # latest form row per player
    pl = pl.sort_values("date").groupby("player_id", as_index=False).tail(1)
    mins = pd.to_numeric(pl.get("minutes_pg", 0), errors="coerce").fillna(0)
    pl = pl[mins >= min_minutes].copy()
    if pl.empty:
        return pd.DataFrame()

    for market, col in [("goals", "p_goal"), ("assists", "p_assist"), ("sot2", "p_sot2")]:
        payload = load_model(market)
        pl[col] = predict_proba(pl, payload).values if payload is not None else 0.0

    m = pd.to_numeric(pl["minutes_pg"], errors="coerce").fillna(60)
    pl["appearance"] = np.where(m >= 60, 2.0, 1.0)
    pl["goal_pts"] = pl["position"].map(GOAL_PTS).fillna(5.0)
    pl["fantasy_pts"] = (
        pl["p_goal"] * pl["goal_pts"]
        + pl["p_assist"] * 3.0
        + pl["p_sot2"] * 1.0
        + pl["appearance"]
    ).round(3)

    keep = [c for c in ["player_name", "team", "position", "minutes_pg",
                        "p_goal", "p_assist", "p_sot2", "fantasy_pts"] if c in pl.columns]
    out = pl[keep].sort_values("fantasy_pts", ascending=False).reset_index(drop=True)
    out["overall_rank"] = np.arange(1, len(out) + 1)
    out["pos_rank"] = out.groupby("position")["fantasy_pts"].rank(ascending=False, method="first").astype(int)
    out["captain_pick"] = out["overall_rank"] <= 3
    for c in ["p_goal", "p_assist", "p_sot2"]:
        out[c] = out[c].round(3)
    return out


def generate_fantasy_tips(write: bool = True) -> pd.DataFrame:
    """Build projections and (optionally) write output/fantasy_tips.csv. Returns the df."""
    proj = build_fantasy_projections()
    if write and not proj.empty:
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        proj.to_csv(_OUT, index=False)
    return proj


# Squad-view params surfaced on the dashboard (whatever exists in the parquet).
_SQUAD_PARAMS = [
    "minutes_pg", "goals_pg", "assists_pg", "sot_pg", "shots_pg", "cards_pg",
    "age", "height_cm", "rating_pg", "player_career_avg_quality",
    "saves_pg", "gk_save_rate",
]
_SQUADS_OUT = Path(__file__).resolve().parents[1] / "output" / "pl_squads.csv"
_OFFICIAL_SQUADS = Path(__file__).resolve().parents[1] / "output" / "pl_squads_official.csv"


def build_squads(parquet_path: Path | None = None) -> pd.DataFrame:
    """Per-club Premier League squad view: each player's role + every key parquet param
    (latest row per player). If output/pl_squads_official.csv exists (from the API daily
    refresh — transfer-window accurate), it's the authoritative roster: players are filtered
    to the current squad and current club/position come from it. Otherwise falls back to
    each player's most-recent team in the parquet."""
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = df[df["league"] == FANTASY_LEAGUE].copy()
    if pl.empty:
        return pd.DataFrame()
    pl = pl.sort_values("date").groupby("player_id", as_index=False).tail(1)
    param_cols = [c for c in _SQUAD_PARAMS if c in pl.columns]
    params = pl[["player_id"] + param_cols].copy()

    if _OFFICIAL_SQUADS.exists():
        # Official current roster (all 20 clubs, transfer-accurate) LEFT-joined to parquet
        # params — so new signings show up too (params blank until they have PL history).
        try:
            off = pd.read_csv(_OFFICIAL_SQUADS)[["player_id", "player_name", "team", "position"]]
            sq = off.merge(params, on="player_id", how="left")
        except Exception:
            sq = pl[["player_id", "player_name", "team", "position"] + param_cols].copy()
    else:
        sq = pl[["player_id", "player_name", "team", "position"] + param_cols].copy()

    for c in param_cols:
        sq[c] = pd.to_numeric(sq[c], errors="coerce").round(2)
    sq = sq.sort_values(["team", "position", "goals_pg"], ascending=[True, True, False])
    _SQUADS_OUT.parent.mkdir(parents=True, exist_ok=True)
    sq.to_csv(_SQUADS_OUT, index=False)
    return sq


def refresh_official_squads(league_id: int | None = None, seasons=("2026", "2025")) -> int:
    """Fetch official current squads via API-Football -> output/pl_squads_official.csv
    (transfer-window accurate; run daily in CI). Picks the first season that has teams.
    Returns player count, or 0 if unavailable (no key / off-season / endpoint)."""
    from player_model.api_football import get_league_teams, get_pl_squads
    lid = league_id or config.PROP_LEAGUES.get(FANTASY_LEAGUE, 39)
    season = next((s for s in seasons if len(get_league_teams(lid, s)) >= 15), None)
    if not season:
        return 0
    rows = get_pl_squads(lid, season)
    if not rows:
        return 0
    df = pd.DataFrame(rows)[["player_id", "player_name", "team", "position", "number", "age"]]
    _OFFICIAL_SQUADS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OFFICIAL_SQUADS, index=False)
    return len(df)


if __name__ == "__main__":
    p = generate_fantasy_tips(write=True)
    if p.empty:
        print("no fantasy projections (no PL data / models missing)")
    else:
        print(f"fantasy projections: {len(p)} players -> {_OUT.name}")
        print("TOP 10:")
        for r in p.head(10).itertuples():
            print(f"  {r.overall_rank:>2}. {_ascii(r.player_name):<24} {r.position} -> {r.fantasy_pts}")
    sq = build_squads()
    print(f"squads: {len(sq)} players across {sq['team'].nunique() if not sq.empty else 0} clubs -> {_SQUADS_OUT.name}")
