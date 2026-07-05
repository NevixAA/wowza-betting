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


# Squad-view stats are recomputed over a USER-SELECTABLE window (last-N games) from the
# RAW per-game columns, so the dashboard can show 5/10/20/38/multi-season form dynamically.
# (parquet age/height_cm are ~84% placeholder defaults -> excluded.)
_RAW_MAP = {  # display "_pg" stat -> raw per-game source column
    "goals_pg": "goals", "assists_pg": "assists", "sot_pg": "shots_on_target",
    "shots_pg": "shots_total", "cards_pg": "yellow_cards", "minutes_pg": "minutes",
    "rating_pg": "rating", "saves_pg": "saves",
}
_SQUADS_OUT = Path(__file__).resolve().parents[1] / "output" / "pl_squads.csv"
_OFFICIAL_SQUADS = Path(__file__).resolve().parents[1] / "output" / "pl_squads_official.csv"


def build_squads(n: int = 5, parquet_path: Path | None = None, write: bool = False) -> pd.DataFrame:
    """Per-club PL squad view with each player's form over the LAST `n` games (recomputed
    from raw per-game columns, so n is fully dynamic: 5/10/20/38/multi-season). Roster comes
    from the official squad feed when present (transfer-accurate, all 20 clubs; new signings
    show with blank stats until they have PL history), else from parquet most-recent team.
    Adds `games_used` (actual games available <= n). write=True snapshots to pl_squads.csv."""
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = df[df["league"] == FANTASY_LEAGUE].copy()
    if pl.empty:
        return pd.DataFrame()
    pl = pl.sort_values("date")
    raw_cols = [r for r in _RAW_MAP.values() if r in pl.columns]

    recent = pl.groupby("player_id").tail(n)                       # last n games per player
    stats = recent.groupby("player_id")[raw_cols].mean().rename(
        columns={v: k for k, v in _RAW_MAP.items() if v in raw_cols})
    stats["games_used"] = recent.groupby("player_id").size()
    latest = pl.groupby("player_id").tail(1).set_index("player_id")
    if "player_career_avg_quality" in latest.columns:
        stats["player_career_avg_quality"] = latest["player_career_avg_quality"]
    stats = stats.reset_index()

    if _OFFICIAL_SQUADS.exists():
        try:
            base = pd.read_csv(_OFFICIAL_SQUADS)[["player_id", "player_name", "team", "position"]]
        except Exception:
            base = latest.reset_index()[["player_id", "player_name", "team", "position"]]
    else:
        base = latest.reset_index()[["player_id", "player_name", "team", "position"]]
    sq = base.merge(stats, on="player_id", how="left")

    for c in [c for c in sq.columns if c not in ("player_id", "player_name", "team", "position")]:
        sq[c] = pd.to_numeric(sq[c], errors="coerce").round(2)
    sort_key = "goals_pg" if "goals_pg" in sq.columns else "player_name"
    sq = sq.sort_values(["team", "position", sort_key], ascending=[True, True, False])
    if write:
        _SQUADS_OUT.parent.mkdir(parents=True, exist_ok=True)
        sq.to_csv(_SQUADS_OUT, index=False)
    return sq


def player_form(n: int = 5, parquet_path: Path | None = None) -> pd.DataFrame:
    """Per-player last-N-game form KPIs + a `form` list (last-N goals, for a sparkline).
    Recomputed from raw per-game columns so n is fully dynamic. Keyed by player_id."""
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = df[df["league"] == FANTASY_LEAGUE].sort_values("date")
    if pl.empty:
        return pd.DataFrame(columns=["player_id"])
    recent = pl.groupby("player_id").tail(n)
    agg = recent.groupby("player_id").agg(
        f_goals_pg=("goals", "mean"),
        f_sot_pg=("shots_on_target", "mean") if "shots_on_target" in pl.columns else ("goals", "mean"),
        f_min_pg=("minutes", "mean") if "minutes" in pl.columns else ("goals", "mean"),
        f_games=("goals", "size"),
    ).round(2)
    agg["form"] = recent.groupby("player_id")["goals"].apply(lambda s: [int(x) for x in s.tolist()])
    return agg.reset_index()


def upcoming_by_team(next_n: int = 5) -> dict:
    """{team_name: [next opponents]} for upcoming PL fixtures. Empty off-season (no key /
    no fixtures) — the caller renders a graceful placeholder until the season starts."""
    try:
        from player_model.api_football import get_upcoming_fixtures
        lid = config.PROP_LEAGUES.get(FANTASY_LEAGUE, 39)
        season = config.PROP_SEASONS.get(FANTASY_LEAGUE, "2025")
        fx = get_upcoming_fixtures(lid, season, next_n=next_n * 12)  # a few GWs to cover all clubs
        out: dict[str, list] = {}
        for f in fx:
            h = f.get("teams", {}).get("home", {}).get("name")
            a = f.get("teams", {}).get("away", {}).get("name")
            if h and a:
                out.setdefault(h, []).append(f"{a} (H)")
                out.setdefault(a, []).append(f"{h} (A)")
        return {t: v[:next_n] for t, v in out.items()}
    except Exception:
        return {}


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
    sq = build_squads(write=True)
    print(f"squads: {len(sq)} players across {sq['team'].nunique() if not sq.empty else 0} clubs -> {_SQUADS_OUT.name}")
