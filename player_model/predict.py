"""
Player prop predictions for today's SNIPER/VALUE picks.
Uses Sofascore season stats — no extra API calls for predictions.
Only activates for matches already flagged by the main model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .model import load_model, predict_proba


def _fair_odds(p: float) -> float:
    return round(1 / max(p, 0.01), 2)


def run_player_predictions(
    bets_csv: Path,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each SNIPER/VALUE match in bets_csv, find players from both teams
    in history_df and generate player prop predictions.
    No extra API calls — uses season stats already collected.
    """
    if not bets_csv.exists() or history_df.empty:
        return pd.DataFrame()

    bets = pd.read_csv(bets_csv)
    bets = bets[bets["signal_tier"].isin(["SNIPER", "VALUE"])].copy()
    bets["date"] = pd.to_datetime(bets["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    bets  = bets[bets["date"] >= today]

    if bets.empty:
        return pd.DataFrame()

    # Load 4 models
    payloads = {m: load_model(m) for m in config.MARKETS}
    missing  = [m for m, p in payloads.items() if p is None]
    if missing:
        print(f"[player_model] Models not trained yet: {missing}. Run --mode train first.")
        return pd.DataFrame()

    all_tips = []

    for _, match_row in bets.iterrows():
        home      = match_row["home_team"]
        away      = match_row["away_team"]
        league    = match_row.get("league", "")
        date_str  = str(match_row["date"])[:10]
        match_tier= match_row.get("signal_tier", "")

        # Find players from both teams in history
        team_players = history_df[
            history_df["team"].str.lower().isin([home.lower(), away.lower()])
        ].copy()

        if team_players.empty:
            # Try fuzzy: team name contains home/away
            mask = (
                history_df["team"].str.lower().str.contains(home.lower()[:6], na=False) |
                history_df["team"].str.lower().str.contains(away.lower()[:6], na=False)
            )
            team_players = history_df[mask].copy()

        if team_players.empty:
            continue

        # Add match context
        team_players["is_home"] = team_players["team"].str.lower().apply(
            lambda t: 1.0 if home.lower() in t or t in home.lower() else 0.0
        )

        # Predict all 4 markets
        for market, payload in payloads.items():
            probs = predict_proba(team_players, payload)
            team_players[f"p_{market}"] = probs.values

        # Build tip rows
        for _, player_row in team_players.iterrows():
            for market in config.MARKETS:
                p_model = float(player_row.get(f"p_{market}", 0))
                if p_model < 0.40:   # skip very unlikely props
                    continue

                fair = _fair_odds(p_model)
                tier = (
                    "SNIPER" if p_model >= (0.5 + config.SNIPER_EDGE) else
                    "VALUE"  if p_model >= (0.5 + config.VALUE_EDGE)  else
                    "WATCH"
                )

                all_tips.append({
                    "date":        date_str,
                    "league":      league,
                    "match":       f"{home} vs {away}",
                    "match_tier":  match_tier,
                    "player_name": player_row["player_name"],
                    "team":        player_row["team"],
                    "position":    player_row.get("position", ""),
                    "market":      market,
                    "model_prob":  round(p_model, 3),
                    "fair_odds":   fair,
                    "bk_odds":     None,
                    "edge":        None,
                    "tier":        tier,
                })

    if not all_tips:
        return pd.DataFrame()

    tips_df = pd.DataFrame(all_tips)
    # Keep only strongest signals per player per market
    tips_df = tips_df.sort_values("model_prob", ascending=False)
    tips_df = tips_df.drop_duplicates(subset=["match", "player_name", "market"])

    output = config.OUTPUT_DIR / "player_tips.csv"
    tips_df.to_csv(output, index=False)
    print(f"[player_model] {len(tips_df)} player prop rows → {output}")
    return tips_df
