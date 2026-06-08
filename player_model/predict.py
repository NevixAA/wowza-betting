"""
Player prop predictions for today's SNIPER/VALUE picks.
Only activates for matches already flagged by the main ML model — credit-efficient.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from . import config
from .data_fetcher import fetch_fixture_players, parse_players, fetch_finished_fixtures
from .feature_engineering import build_upcoming_features
from .model import load_model, predict_proba


def _fair_odds(p: float) -> float:
    return round(1 / max(p, 0.01), 2)


def _tier(edge: float) -> str:
    if edge >= config.SNIPER_EDGE:
        return "SNIPER"
    if edge >= config.VALUE_EDGE:
        return "VALUE"
    return "AVOID"


def run_player_predictions(
    bets_csv: Path,
    history_df: pd.DataFrame,
    season: str = config.API_SEASON,
) -> pd.DataFrame:
    """
    For each SNIPER/VALUE match in bets_csv, fetch confirmed lineup via
    API-Football and generate player prop tips.

    Returns DataFrame with player prop tips.
    """
    if not bets_csv.exists():
        return pd.DataFrame()

    bets = pd.read_csv(bets_csv)
    bets = bets[bets["signal_tier"].isin(["SNIPER", "VALUE"])].copy()
    bets["date"] = pd.to_datetime(bets["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    bets = bets[bets["date"] >= today]

    if bets.empty:
        return pd.DataFrame()

    # Load all 4 models
    payloads = {m: load_model(m) for m in config.MARKETS}
    missing  = [m for m, p in payloads.items() if p is None]
    if missing:
        print(f"[player_model] Models not trained yet: {missing}. Run pipeline --mode train first.")
        return pd.DataFrame()

    all_tips = []

    for _, match_row in bets.iterrows():
        home = match_row["home_team"]
        away = match_row["away_team"]
        league = match_row.get("league", "")
        date_str = str(match_row["date"])[:10]

        # Find fixture_id via league fixture list (cached)
        league_id = _find_league_id(league)
        if league_id is None:
            continue

        fixture_id = _find_fixture_id(league_id, season, home, away)
        if fixture_id is None:
            print(f"  [player] No fixture found: {home} vs {away}")
            continue

        # Fetch lineup / player data (1 API call, cached)
        try:
            raw_players = fetch_fixture_players(fixture_id)
        except Exception as e:
            print(f"  [player] API error for fixture {fixture_id}: {e}")
            continue

        # Build upcoming feature rows
        upcoming = _parse_lineup(raw_players, home, away)
        if not upcoming:
            continue

        feat_df = build_upcoming_features(upcoming, history_df)
        if feat_df.empty:
            continue

        # Predict all 4 markets
        for market, payload in payloads.items():
            probs = predict_proba(feat_df, payload)
            feat_df[f"p_{market}"] = probs.values

        # Build tip rows
        for _, player_row in feat_df.iterrows():
            for market in config.MARKETS:
                p_model = float(player_row[f"p_{market}"])
                fair    = _fair_odds(p_model)

                tip = {
                    "date":        date_str,
                    "league":      league,
                    "match":       f"{home} vs {away}",
                    "match_tier":  match_row.get("signal_tier", ""),
                    "player_name": player_row["player_name"],
                    "team":        player_row["team"],
                    "position":    player_row.get("position", ""),
                    "market":      market,
                    "model_prob":  round(p_model, 3),
                    "fair_odds":   fair,
                    "bk_odds":     None,   # to be filled when bookmaker odds available
                    "edge":        None,
                    "tier":        None,
                }
                all_tips.append(tip)

    tips_df = pd.DataFrame(all_tips) if all_tips else pd.DataFrame()
    if not tips_df.empty:
        output = config.OUTPUT_DIR / "player_tips.csv"
        tips_df.to_csv(output, index=False)
        print(f"[player_model] {len(tips_df)} player prop rows → {output}")

    return tips_df


def _find_league_id(league_name: str) -> int | None:
    for name, lid in config.TRAINING_LEAGUES.items():
        if name.lower() in league_name.lower() or league_name.lower() in name.lower():
            return lid
    return None


def _find_fixture_id(
    league_id: int, season: str, home: str, away: str
) -> int | None:
    """Find fixture_id from cached fixture list using fuzzy team name match."""
    try:
        fixtures = fetch_finished_fixtures(league_id, season)
    except Exception:
        return None

    home_l = home.lower()
    away_l = away.lower()
    for fix in fixtures:
        h = fix["teams"]["home"]["name"].lower()
        a = fix["teams"]["away"]["name"].lower()
        if (home_l in h or h in home_l) and (away_l in a or a in away_l):
            return fix["fixture"]["id"]
    return None


def _parse_lineup(raw_players: list[dict], home: str, away: str) -> list[dict]:
    """Extract player dicts from fixture/players API response."""
    players = []
    for team_block in raw_players:
        team_name = team_block["team"]["name"]
        is_home   = int(home.lower() in team_name.lower() or team_name.lower() in home.lower())
        opponent  = away if is_home else home

        for p in team_block["players"]:
            info  = p["player"]
            stats = p.get("statistics", [{}])[0]
            games = stats.get("games", {})
            mins  = games.get("minutes") or 0
            if mins < 20:
                continue

            players.append({
                "player_id":   info["id"],
                "player_name": info["name"],
                "team":        team_name,
                "opponent":    opponent,
                "is_home":     is_home,
                "position":    (games.get("position") or "").upper(),
                "minutes":     mins,
            })
    return players
