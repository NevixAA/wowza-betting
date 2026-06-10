"""
Player Prop Predictions v2
===========================
Generates player prop signals for today's SNIPER/MARKSMAN matches.
Uses de-vigged edge, relative edge floors, and 5-component confidence scoring.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd

from . import config

_APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY", "")
from .feature_engineering import build_upcoming_features, compute_ges
from .model import load_model, predict_proba


# ── Edge calculations (v2 — proper de-vig) ────────────────────────────────────

def _devig_fair_prob(market_odds: float) -> float:
    """
    Compute fair (de-vigged) implied probability.
    Uses assumed overround based on odds level.
    """
    if not market_odds or market_odds <= 1.0:
        return 0.5
    raw = 1.0 / market_odds
    # Find assumed overround
    overround = config.OVERROUND_BY_ODDS[99.0]  # default high
    for min_odds, ovr in sorted(config.OVERROUND_BY_ODDS.items()):
        if market_odds <= min_odds:
            overround = ovr
            break
    return raw / overround


def _ev(p_model: float, market_odds: float) -> float:
    return round(p_model * market_odds - 1, 4)


def _relative_edge(p_model: float, fair_prob: float) -> float:
    if fair_prob <= 0:
        return 0.0
    return round((p_model - fair_prob) / fair_prob, 4)


def _edge_passes_floor(rel_edge: float, market_odds: float) -> bool:
    """Check relative edge meets the floor for this odds tier."""
    for min_odds in sorted(config.RELATIVE_EDGE_FLOORS.keys(), reverse=True):
        if market_odds >= min_odds:
            return rel_edge >= config.RELATIVE_EDGE_FLOORS[min_odds]
    return rel_edge >= config.RELATIVE_EDGE_FLOORS[min(config.RELATIVE_EDGE_FLOORS.keys())]


def _kelly_stake(ev: float, market_odds: float) -> float:
    if market_odds <= 1.0:
        return 0.0
    full_kelly = ev / (market_odds - 1)
    return max(0.0, round(full_kelly * config.KELLY_FRACTION, 4))


# ── Confidence scoring ────────────────────────────────────────────────────────

def _confidence_score(row: dict, lazy_factor_count: int = 0) -> float:
    """5-component confidence score (0.0-1.0)."""
    n_games = row.get("n_games", 0)

    # Data volume (0->0, 5->0.40, 10->0.70, 20->1.0)
    vol_score = min(1.0, n_games / 20.0) * 0.70 + (min(n_games, 5) / 5.0) * 0.30

    # Recency (exponential decay — most recent game weighted highest)
    recency = 0.8  # default when no per-match data

    # Model agreement (always 1.0 for single model, will improve with XGBoost)
    model_agree = 1.0

    # Lazy factor contribution
    lazy_score = min(lazy_factor_count / 2.0, 1.0)

    # Minutes certainty
    p_start = row.get("p_start", 0.8)  # P(player starts)
    min_cert = float(p_start)

    score = (
        config.CONFIDENCE_FLOORS.get("VALUABLE", 0.50) * 0  # zero-point base
        + 0.30 * vol_score
        + 0.25 * recency
        + 0.20 * model_agree
        + 0.15 * lazy_score
        + 0.10 * min_cert
    )
    return round(score, 3)


# ── Tier classification ───────────────────────────────────────────────────────

def _classify_tier(
    ev: float, market_odds: float, rel_edge: float,
    confidence: float, lazy_count: int, ges: float = 0.5,
    market: str = ""
) -> str:
    """Classify signal into SNIPER/MARKSMAN/VALUABLE/WATCH."""
    is_goals_sot = market in ("goals", "sot")

    # GES gate for goals/SOT
    if is_goals_sot and ges < config.GES_SUPPRESS:
        return "WATCH"

    if (ev >= config.SNIPER_EV and market_odds >= 5.0
            and confidence >= config.CONFIDENCE_FLOORS["SNIPER"]
            and lazy_count >= 2
            and _edge_passes_floor(rel_edge, market_odds)
            and (not is_goals_sot or ges >= config.GES_SNIPER_MIN)):
        return "SNIPER"

    if (ev >= config.MARKSMAN_EV and market_odds >= 4.0
            and confidence >= config.CONFIDENCE_FLOORS["MARKSMAN"]
            and lazy_count >= 1
            and _edge_passes_floor(rel_edge, market_odds)
            and (not is_goals_sot or ges >= config.GES_MARKSMAN_MIN)):
        return "MARKSMAN"

    if (ev >= config.VALUABLE_EV
            and confidence >= config.CONFIDENCE_FLOORS["VALUABLE"]
            and _edge_passes_floor(rel_edge, market_odds)):
        return "VALUABLE"

    return "WATCH"


# ── Lazy market factor detection ──────────────────────────────────────────────

def _detect_lazy_factors(row: dict, feat_row: dict, market: str) -> list[str]:
    """Identify which lazy market factors are active for this player/match."""
    factors = []

    if market == "sot":
        sp_score = feat_row.get("set_piece_threat_score", 0.0)
        # Threshold: aerial 45% × corners at league avg (6) × 0.30 sp rate = 0.135
        if sp_score > 0.135 and (feat_row.get("pos_defender", 0) or feat_row.get("pos_midfielder", 0)):
            factors.append("SET_PIECE")

    if market == "cards":
        ref_strict = feat_row.get("referee_strictness", 0)
        if ref_strict > 0.5:
            factors.append("REFEREE")

    if market == "sot" and feat_row.get("opp_sot_conceded_pg", 4.5) > 5.5:
        factors.append("KEEPER_WEAK")

    if feat_row.get("minutes_pg", 75) < 65 and row.get("minutes_est", 75) > 75:
        factors.append("MINUTES")

    return factors


# ── Main prediction function ──────────────────────────────────────────────────

def run_player_predictions(
    bets_csv: Path,
    history_df: pd.DataFrame,
    referee_profiles: dict | None = None,
    match_contexts: dict | None = None,
) -> pd.DataFrame:
    """
    Generate player prop signals for ALL upcoming fixtures in PROP_LEAGUES.
    Decoupled from team model — runs for Premier League, Bundesliga etc.
    even when we don't have a team O/U signal for those leagues.

    bets_csv:         output/bets.csv (used to add context for our standard leagues)
    history_df:       player season stats from FBref training data
    referee_profiles: {match_key: {strictness_score, yellows_per_game}}
    match_contexts:   {match_key: {team_corners_per90, opp_sot_conceded_pg, ...}}
    """
    if history_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()

    # Primary: ALL upcoming fixtures in PROP_LEAGUES (decoupled from team model)
    prop_matches = []
    if _APIFOOTBALL_KEY:
        try:
            from player_model.api_football import get_upcoming_fixtures
            for league, lg_id in config.PROP_LEAGUES.items():
                season = config.PROP_SEASONS.get(league, "2025")
                fixtures = get_upcoming_fixtures(lg_id, season, next_n=5)
                for fix in fixtures:
                    teams = fix.get("teams", {})
                    dt    = fix.get("fixture", {}).get("date", "")[:10]
                    prop_matches.append({
                        "home_team":    teams.get("home", {}).get("name", ""),
                        "away_team":    teams.get("away", {}).get("name", ""),
                        "league":       league,
                        "date":         pd.Timestamp(dt) if dt else today,
                        "signal_tier":  "PROP_LEAGUE",
                        "fixture_id":   fix.get("fixture", {}).get("id"),
                    })
        except Exception as e:
            print(f"[predict] API-Football fixtures fetch failed: {e}")

    # Fallback / supplement: our team model matches (bets.csv)
    if bets_csv and bets_csv.exists():
        bets_df = pd.read_csv(bets_csv)
        bets_df["date"] = pd.to_datetime(bets_df["date"], errors="coerce")
        bets_df = bets_df[bets_df["date"] >= today]
        for _, r in bets_df.iterrows():
            prop_matches.append({
                "home_team":   r["home_team"],
                "away_team":   r["away_team"],
                "league":      r.get("league", ""),
                "date":        r["date"],
                "signal_tier": r.get("signal_tier", ""),
                "fixture_id":  None,
            })

    if not prop_matches:
        return pd.DataFrame()

    # Deduplicate
    seen = set()
    bets_list = []
    for m in prop_matches:
        key = f"{m['home_team']}|{m['away_team']}|{str(m['date'])[:10]}"
        if key not in seen:
            seen.add(key)
            bets_list.append(m)

    import pandas as _pd
    bets = _pd.DataFrame(bets_list)

    payloads = {m: load_model(m) for m in config.MARKETS}
    missing  = [m for m, p in payloads.items() if p is None]
    if missing:
        print(f"[player_model] Models not trained: {missing}. Run --mode train first.")
        return pd.DataFrame()

    all_tips = []

    for _, match_row in bets.iterrows():
        home      = match_row["home_team"]
        away      = match_row["away_team"]
        league    = match_row.get("league", "")
        date_str  = str(match_row["date"])[:10]
        match_key = f"{home}|{away}|{date_str}"

        ref = (referee_profiles or {}).get(match_key, {})
        ctx = (match_contexts or {}).get(match_key, {})

        # Find players from both teams
        team_players = history_df[
            history_df["team"].str.lower().isin([home.lower(), away.lower()])
        ].copy()
        if team_players.empty:
            mask = (
                history_df["team"].str.lower().str.contains(home.lower()[:5], na=False) |
                history_df["team"].str.lower().str.contains(away.lower()[:5], na=False)
            )
            team_players = history_df[mask].copy()
        if team_players.empty:
            continue

        # Add match context
        team_players["is_home"] = team_players["team"].str.lower().apply(
            lambda t: 1.0 if home.lower()[:5] in t or t[:5] in home.lower() else 0.0
        )
        team_players["opponent"] = team_players["team"].apply(
            lambda t: away if home.lower()[:5] in t.lower() else home
        )

        upcoming_list = team_players.to_dict("records")
        feat_df = build_upcoming_features(upcoming_list, history_df, ref, ctx)

        if feat_df.empty:
            continue

        # Predict all markets
        for market, payload in payloads.items():
            probs = predict_proba(feat_df, payload)
            feat_df[f"p_{market}"] = probs.values

        for i, feat_row in feat_df.iterrows():
            for market in config.MARKETS:
                p_model = float(feat_row.get(f"p_{market}", 0))
                if p_model < 0.15:
                    continue

                n_games = int(feat_row.get("n_games", 0))
                if n_games < config.MIN_GAMES_SIGNAL:
                    continue

                # Fair odds (we don't have market odds yet — placeholder)
                fair_o = round(1 / max(p_model, 0.01), 2)

                # GES for goals/SOT
                ges = compute_ges(
                    feat_row.to_dict(),
                    opp_weakness=feat_row.get("opp_goals_conceded_pg", 1.0),
                ) if market in ("goals", "sot") else 0.5

                lazy_factors = _detect_lazy_factors(
                    feat_row.to_dict(), feat_row.to_dict(), market
                )
                confidence = _confidence_score(feat_row.to_dict(), len(lazy_factors))

                all_tips.append({
                    "date":          date_str,
                    "league":        league,
                    "match":         f"{home} vs {away}",
                    "match_tier":    match_row.get("signal_tier", ""),
                    "player_name":   feat_row.get("player_name", ""),
                    "team":          feat_row.get("team", ""),
                    "position":      feat_row.get("position", ""),
                    "market":        market,
                    "model_prob":    round(p_model, 3),
                    "fair_odds":     fair_o,
                    "market_odds":   None,   # to be filled when odds available
                    "fair_implied":  None,
                    "edge_abs":      None,
                    "edge_rel":      None,
                    "ev":            None,
                    "ges":           ges if market in ("goals", "sot") else None,
                    "confidence":    confidence,
                    "lazy_factors":  "|".join(lazy_factors),
                    "n_games":       n_games,
                    "data_source":   feat_row.get("data_source", ""),
                    "tier":          "WATCH",   # updated when market odds available
                    "kelly_stake":   None,
                })

    if not all_tips:
        return pd.DataFrame()

    tips_df = pd.DataFrame(all_tips)
    tips_df = tips_df.sort_values("model_prob", ascending=False)
    tips_df = tips_df.drop_duplicates(subset=["match", "player_name", "market"])

    output = config.OUTPUT_DIR / "player_tips.csv"
    tips_df.to_csv(output, index=False)
    print(f"[player_model] {len(tips_df)} player prop rows -> {output}")
    return tips_df


def enrich_with_odds(tips_df: pd.DataFrame, odds_data: dict) -> pd.DataFrame:
    """
    Enrich player tips with market odds, compute EV, edge, tier.
    odds_data: {"{player_name}|{market}": market_odds}
    Called separately when odds are available.
    """
    if tips_df.empty:
        return tips_df

    for idx, row in tips_df.iterrows():
        key        = f"{row['player_name']}|{row['market']}"
        mkt_odds   = odds_data.get(key)
        if not mkt_odds:
            continue

        p_model    = float(row["model_prob"])
        fair_prob  = _devig_fair_prob(mkt_odds)
        ev_val     = _ev(p_model, mkt_odds)
        rel_edge   = _relative_edge(p_model, fair_prob)
        confidence = float(row["confidence"])
        lazy_count = len(str(row.get("lazy_factors", "")).split("|")) if row.get("lazy_factors") else 0
        ges        = float(row["ges"]) if row.get("ges") is not None else 0.5
        market     = row["market"]

        tier = _classify_tier(ev_val, mkt_odds, rel_edge, confidence, lazy_count, ges, market)

        tips_df.at[idx, "market_odds"]  = mkt_odds
        tips_df.at[idx, "fair_implied"] = round(fair_prob, 4)
        tips_df.at[idx, "edge_abs"]     = round(p_model - fair_prob, 4)
        tips_df.at[idx, "edge_rel"]     = rel_edge
        tips_df.at[idx, "ev"]           = ev_val
        tips_df.at[idx, "tier"]         = tier
        tips_df.at[idx, "kelly_stake"]  = _kelly_stake(ev_val, mkt_odds) if tier != "WATCH" else 0.0

    return tips_df
