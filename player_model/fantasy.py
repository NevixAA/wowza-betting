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
GOAL_PTS = {"F": 4.0, "M": 5.0, "D": 6.0, "G": 6.0}   # legacy parquet-position goal pts
# FPL goal points keyed by FPL position codes (GKP/DEF/MID/FWD)
FPL_GOAL_PTS = {"FWD": 4.0, "MID": 5.0, "DEF": 6.0, "GKP": 6.0}
# Clean-sheet points by position (DEF/GK 4, MID 1, FWD 0)
FPL_CS_PTS   = {"DEF": 4.0, "GKP": 4.0, "MID": 1.0, "FWD": 0.0}
# Defensive-contribution proxy thresholds (actions/game). APPROX: our source gives
# tackles+interceptions+blocks+duels_won but NOT clearances or ball-recoveries, so these
# sit below FPL's real 10 (DEF CBIT) / 12 (MID+FWD tackles+recoveries) cut-offs.
_DC_THRESH = {"DEF": 5.0, "GKP": 999.0, "MID": 4.5, "FWD": 4.0}
_POS_ALIAS = {"F": "FWD", "M": "MID", "D": "DEF", "G": "GKP",
              "FWD": "FWD", "MID": "MID", "DEF": "DEF", "GKP": "GKP", "GK": "GKP"}
_PARQUET = Path(__file__).resolve().parents[1] / "player_history.parquet"
_OUT = Path(__file__).resolve().parents[1] / "output" / "fantasy_tips.csv"


def _ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def _recent_def_actions(pl_all: pd.DataFrame, n: int = 10) -> dict:
    """Recent mean defensive actions/game per player (tackles+interceptions+blocks+duels_won).
    Approximates FPL CBIT — source lacks clearances/ball-recoveries (flagged at call site)."""
    cols = [c for c in ("tackles_total", "interceptions", "blocks", "duels_won") if c in pl_all.columns]
    if not cols:
        return {}
    recent = pl_all.sort_values("date").groupby("player_id").tail(n)
    da = recent.groupby("player_id")[cols].mean().sum(axis=1)
    return {k: float(v) for k, v in da.items()}


def _cs_prob(fdr_list: list) -> float:
    """P(clean sheet) from the team's next fixtures' official FDR, via a Poisson on opponent
    xG (harder fixture -> higher opp xG -> lower CS prob). NaN when no fixtures (off-season)."""
    import math
    if not fdr_list:
        return float("nan")
    ps = [math.exp(-(0.5 + 0.42 * float(f.get("fdr") or 3))) for f in fdr_list]
    return float(np.mean(ps)) if ps else float("nan")


def build_fantasy_projections(parquet_path: Path | None = None, min_minutes: float = 45.0,
                              use_fpl: bool = True, next_n: int = 5) -> pd.DataFrame:
    """Per-player FPL expected-points projections.

    fantasy_pts = appearance + attack (P(goal)*pos_pts + P(assist)*3 + P(sot2)*1 bonus proxy)
                  + defensive-contribution (approx) + clean-sheet (DEF/GK/MID, from FDR).

    When use_fpl, joins to the LIVE official FPL squad (transfer-accurate team/position/price)
    and attaches availability/injury flags. Falls back to parquet team/position if the FPL API
    is unavailable (off-network), so it never hard-fails.
    """
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl_all = df[df["league"] == FANTASY_LEAGUE].copy()
    if pl_all.empty:
        return pd.DataFrame()
    pl = pl_all.sort_values("date").groupby("player_id", as_index=False).tail(1)
    mins = pd.to_numeric(pl.get("minutes_pg", 0), errors="coerce").fillna(0)
    pl = pl[mins >= min_minutes].copy()
    if pl.empty:
        return pd.DataFrame()

    for market, col in [("goals", "p_goal"), ("assists", "p_assist"), ("sot2", "p_sot2")]:
        payload = load_model(market)
        pl[col] = predict_proba(pl, payload).values if payload is not None else 0.0

    m = pd.to_numeric(pl["minutes_pg"], errors="coerce").fillna(60)
    pl["appearance"] = np.where(m >= 60, 2.0, 1.0)

    da_map = _recent_def_actions(pl_all)
    pl["def_actions_pg"] = pl["player_id"].map(da_map).fillna(0.0) if da_map else 0.0

    # ── LIVE FPL layer: transfer-accurate team/position/price + availability + FDR ──
    fpl, fdr_map = pd.DataFrame(), {}
    if use_fpl:
        try:
            from player_model import fpl_api
            fpl = fpl_api.players_df()
            fdr_map = fpl_api.upcoming_fdr(next_n=next_n)
            norm = fpl_api._norm
        except Exception:
            fpl, fdr_map, norm = pd.DataFrame(), {}, (lambda s: str(s).lower())
    else:
        norm = lambda s: str(s).lower()

    # defaults from parquet (fallback when FPL unmatched / unavailable)
    pl["live_team"]         = pl.get("team", "")
    pl["price"]             = np.nan
    pl["availability"]      = "unknown"
    pl["injured"]           = False
    pl["doubtful"]          = False
    pl["chance_of_playing"] = np.nan
    pl["fpl_matched"]       = False
    pl["_fpl_pos"]          = ""

    if not fpl.empty:
        by_full = {k: r for k, r in fpl.set_index("match_key").iterrows()}
        by_web  = {k: r for k, r in fpl.set_index("web_key").iterrows()}
        for i, r in pl.iterrows():
            k = norm(r.get("player_name", ""))
            rec = by_full.get(k)
            if rec is None:
                rec = by_web.get(k)
            if rec is not None:
                pl.at[i, "live_team"]         = rec["team"]
                pl.at[i, "price"]             = rec["price"]
                pl.at[i, "availability"]      = rec["availability"]
                pl.at[i, "injured"]           = bool(rec["injured"])
                pl.at[i, "doubtful"]          = bool(rec["doubtful"])
                pl.at[i, "chance_of_playing"] = rec["chance_of_playing"]
                pl.at[i, "fpl_matched"]       = True
                pl.at[i, "_fpl_pos"]          = rec["position"]
        # A player NOT in the current FPL set has left the PL (e.g. Salah -> Besiktas) or is
        # not fantasy-relevant -> DROP from FPL tips (never show them at a stale old club).
        pl = pl[pl["fpl_matched"]].copy()
        if pl.empty:
            return pd.DataFrame()

    # scoring position: FPL position if matched, else parquet position (aliased to FPL codes)
    pos = pl["_fpl_pos"].where(pl["_fpl_pos"].astype(bool), pl.get("position", "MID").astype(str))
    pl["position"] = pos.map(lambda p: _POS_ALIAS.get(str(p), "MID"))
    pl["team"] = pl["live_team"]   # `team` is now the LIVE club (fixes stale Salah etc.)

    pl["goal_pts"]   = pl["position"].map(FPL_GOAL_PTS).fillna(5.0)
    pl["attack_pts"] = (pl["p_goal"] * pl["goal_pts"] + pl["p_assist"] * 3.0 + pl["p_sot2"] * 1.0).round(3)

    thr = pl["position"].map(_DC_THRESH).fillna(6.0)
    pl["dc_pts"] = (2.0 * (pl["def_actions_pg"] / thr).clip(0, 1)).round(3)

    if fdr_map:
        pl["cs_pts"] = pl.apply(
            lambda r: 0.0 if (_p := _cs_prob(fdr_map.get(r["team"], []))) != _p
            else round(_p * FPL_CS_PTS.get(r["position"], 0.0), 3), axis=1)
    else:
        pl["cs_pts"] = 0.0

    pl["fantasy_pts"] = (pl["appearance"] + pl["attack_pts"] + pl["dc_pts"] + pl["cs_pts"]).round(3)
    pl["value"] = (pl["fantasy_pts"] / pl["price"]).round(3)

    keep = [c for c in ["player_name", "team", "position", "price", "minutes_pg",
                        "p_goal", "p_assist", "p_sot2", "attack_pts", "dc_pts", "cs_pts",
                        "def_actions_pg", "fantasy_pts", "value", "availability", "injured",
                        "doubtful", "chance_of_playing", "fpl_matched"] if c in pl.columns]
    out = pl[keep].sort_values("fantasy_pts", ascending=False).reset_index(drop=True)
    out["overall_rank"] = np.arange(1, len(out) + 1)
    out["pos_rank"] = out.groupby("position")["fantasy_pts"].rank(ascending=False, method="first").astype(int)
    # captain: top 3 among AVAILABLE players (never captain an injured one)
    healthy = ~out["injured"].astype(bool)
    out["captain_pick"] = False
    top3 = out[healthy].head(3).index
    out.loc[top3, "captain_pick"] = True
    for c in ["p_goal", "p_assist", "p_sot2"]:
        out[c] = out[c].round(3)
    return out


def _team_defense_map(pl: pd.DataFrame) -> dict:
    """{team_name: goals-conceded-per-game} approx from the opponent-concede feature.
    Higher = leakier defence = easier fixture for attackers."""
    if pl.empty or "opp_goals_conceded_pg" not in pl.columns:
        return {}
    is_home = pd.to_numeric(pl.get("is_home", 0), errors="coerce").fillna(0)
    opp = np.where(is_home == 1, pl.get("away_team", ""), pl.get("home_team", ""))
    tmp = pl.assign(_opp=opp)
    d = tmp.groupby("_opp")["opp_goals_conceded_pg"].mean()
    return {str(k): float(v) for k, v in d.items() if pd.notna(v) and str(k).strip()}


def build_fantasy_projections_fixtures(next_n: int = 5, parquet_path: Path | None = None) -> pd.DataFrame:
    """Fantasy projections with an opponent-adjusted (fixture-difficulty) layer.

    Base points come from build_fantasy_projections() — the model's player-vs-player matchup
    features are already baked into p_goal/p_assist/p_sot2. This layers a TEAM-vs-TEAM
    adjustment: each player's points are scaled by the difficulty of their team's next
    `next_n` fixtures (opponent goals-conceded rate × home/away), so an easy run of games
    lifts projected points and a hard run lowers them.

    Adds columns: next_fixtures, avg_fdr (mean multiplier; >1 = easy run), fixture_adj_pts,
    fixtures_available (False off-season → fixture_adj_pts == base fantasy_pts).
    """
    base = build_fantasy_projections(parquet_path)
    if base.empty:
        return base
    base = base.copy()
    base["next_fixtures"] = ""
    base["avg_fdr"] = np.nan
    base["fixture_adj_pts"] = base["fantasy_pts"]

    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = df[df["league"] == FANTASY_LEAGUE].copy()
    dmap = _team_defense_map(pl)
    fixtures = upcoming_by_team(next_n)     # {team: ["Opp (H)", ...]} — empty off-season

    if not fixtures or not dmap:
        base["fixtures_available"] = False
        return base
    base["fixtures_available"] = True

    lg_avg = float(np.mean(list(dmap.values()))) or 1.0
    _norm = lambda s: "".join(c for c in str(s).lower() if c.isalnum())
    dmap_n = {_norm(k): v for k, v in dmap.items()}
    fix_n  = {_norm(k): v for k, v in fixtures.items()}

    adj, fdr, fxs = [], [], []
    for _, r in base.iterrows():
        opps = fix_n.get(_norm(r.get("team", "")), [])
        if not opps:
            adj.append(r["fantasy_pts"]); fdr.append(np.nan); fxs.append(""); continue
        mults = []
        for o in opps:
            is_h = o.strip().endswith("(H)")
            conc = dmap_n.get(_norm(o.rsplit("(", 1)[0]), lg_avg)
            m = (conc / lg_avg) * (1.10 if is_h else 0.92)   # leaky opp + home = easier
            mults.append(min(max(m, 0.6), 1.5))
        mult = float(np.mean(mults))
        # scale ONLY the opponent-dependent expectation, NOT the fixed appearance points
        appr = 2.0 if (pd.to_numeric(r.get("minutes_pg"), errors="coerce") or 60) >= 60 else 1.0
        adj.append(round(appr + mult * (r["fantasy_pts"] - appr), 3))
        fdr.append(round(mult, 2))
        fxs.append(" · ".join(opps))
    base["fixture_adj_pts"] = adj
    base["avg_fdr"] = fdr
    base["next_fixtures"] = fxs
    return base.sort_values("fixture_adj_pts", ascending=False).reset_index(drop=True)


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
