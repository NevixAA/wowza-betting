"""
Player Prop Predictions v2
===========================
Generates player prop signals for today's SNIPER/MARKSMAN matches.
Uses de-vigged edge, relative edge floors, and 5-component confidence scoring.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

from . import config

_APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY", "")
from .api_football import _norm_name as _norm_player_name
from .feature_engineering import build_upcoming_features, compute_ges
from .model import load_model, predict_proba

import numpy as np


def _compute_market_caps(history_path: Path) -> dict[str, float]:
    """
    Compute per-market probability caps = top-1% career rate among players
    with >= 10 matches in player_history.parquet.
    No model should output a probability above the best real player's career rate.
    """
    _COL_THRESHOLD = {
        "goals":   ("goals",           1),
        "goals2":  ("goals",           2),
        "assists": ("assists",         1),
        "sot":     ("shots_on_target", 1),
        "sot2":    ("shots_on_target", 2),
        "sot3":    ("shots_on_target", 3),
        "cards":   ("yellow_cards",    1),
    }
    caps: dict[str, float] = {}
    if not history_path.exists():
        return {mkt: 0.95 for mkt in _COL_THRESHOLD}
    hist = pd.read_parquet(history_path)
    for mkt, (col, thr) in _COL_THRESHOLD.items():
        if col not in hist.columns:
            caps[mkt] = 0.95
            continue
        rates = (hist.groupby("player_name")
                     .filter(lambda g: len(g) >= 10)
                     .groupby("player_name")[col]
                     .apply(lambda x: (x >= thr).mean()))
        caps[mkt] = round(float(rates.quantile(0.99)), 4)
    return caps


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


def _team_strength_mult(team_win_prob: float, market: str) -> float:
    """
    Bidirectional team-strength adjustment on model probability.

    Favourite  (win_prob > 60%): boost  — up to ×1.30 at 90%+
    Balanced   (40% – 60%)     : no change
    Underdog   (win_prob < 40%): penalty — down to ×0.40 at 5%-

    Cards exempt: card rate doesn't scale cleanly with team dominance.
    Assists: 70% of the adjustment.
    """
    if market == "cards":
        return 1.0

    if team_win_prob >= 0.60:
        raw = min(team_win_prob - 0.60, 0.30)   # 0.0 → 0.30
        adj = raw
        if market == "assists":
            adj *= 0.70
        return round(1.0 + adj, 3)

    if team_win_prob < 0.40:
        # Extended penalty for heavy underdogs (Panama vs England = ~5% win prob)
        # Linear from 0.40 (no change) → 0.05 (×0.40 max penalty)
        raw = min(0.40 - team_win_prob, 0.35)   # 0.0 → 0.35
        adj = raw * (0.60 / 0.35)               # scale so 0.35 raw → 0.60 penalty
        adj = min(adj, 0.60)                    # cap at ×0.40 floor
        if market == "assists":
            adj *= 0.70
        return round(max(1.0 - adj, 0.40), 3)

    return 1.0  # 40–60%: balanced game, no adjustment


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
    market: str = "", position: str = ""
) -> str:
    """Classify signal into SNIPER/MARKSMAN/VALUABLE/AVOID (WATCH removed).

    Applies a real-odds-backtest role×market gate: clear money-losing combos
    (defenders→goals, multi-SOT/goals2) are capped at VALUABLE (data-only) rather
    than producing a real SNIPER/MARKSMAN tip.
    """
    is_goals_sot = market in ("goals", "sot")

    # GES gate for goals/SOT — below the floor = no signal
    if is_goals_sot and ges < config.GES_SUPPRESS:
        return "AVOID"

    # Use relative edge gates instead of hard market_odds floors.
    # rel_edge = (model_prob - fair_prob) / fair_prob — odds-level-agnostic.
    _sniper_re   = getattr(config, "REL_EDGE_SNIPER",   0.20)
    _marksman_re = getattr(config, "REL_EDGE_MARKSMAN", 0.12)

    if (ev >= config.SNIPER_EV and rel_edge >= _sniper_re
            and confidence >= config.CONFIDENCE_FLOORS["SNIPER"]
            and lazy_count >= 2
            and _edge_passes_floor(rel_edge, market_odds)
            and (not is_goals_sot or ges >= config.GES_SNIPER_MIN)):
        tier = "SNIPER"
    elif (ev >= config.MARKSMAN_EV and rel_edge >= _marksman_re
            and confidence >= config.CONFIDENCE_FLOORS["MARKSMAN"]
            and lazy_count >= 1
            and _edge_passes_floor(rel_edge, market_odds)
            and (not is_goals_sot or ges >= config.GES_MARKSMAN_MIN)):
        tier = "MARKSMAN"
    elif (ev >= config.VALUABLE_EV
            and confidence >= config.CONFIDENCE_FLOORS["VALUABLE"]
            and _edge_passes_floor(rel_edge, market_odds)):
        tier = "VALUABLE"
    else:
        return "AVOID"

    # ── Role × market gate (real-odds backtest Jun 2026) ──────────────────────
    # Cap clear money-losers at VALUABLE (data-only): multi-SOT/goals2 markets,
    # and defenders for anytime-goalscorer.
    pos = (position or "").strip().upper()[:1]
    if tier in ("SNIPER", "MARKSMAN") and (
        market in getattr(config, "VALUABLE_ONLY_MARKETS", set())
        or (pos, market) in getattr(config, "VALUABLE_ONLY_ROLE_MARKET", set())
    ):
        tier = "VALUABLE"
    return tier


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

    # Compute caps once per run from the actual history distribution
    _history_path = config.BASE_DIR / "player_history.parquet"
    _MARKET_CAPS = _compute_market_caps(_history_path)
    print(f"[player_model] market caps (top-1% real player): { {k: v for k,v in _MARKET_CAPS.items()} }")

    today = pd.Timestamp.now().normalize()

    # Primary: ALL upcoming fixtures in PROP_LEAGUES (decoupled from team model)
    prop_matches = []
    if _APIFOOTBALL_KEY:
        try:
            from player_model.api_football import get_upcoming_fixtures
            for league, lg_id in config.PROP_LEAGUES.items():
                season = config.PROP_SEASONS.get(league, "2025")
                n_fix = 20 if league == "World Cup" else 5
                fixtures = get_upcoming_fixtures(lg_id, season, next_n=n_fix)
                for fix in fixtures:
                    teams    = fix.get("teams", {})
                    dt_full  = fix.get("fixture", {}).get("date", "")
                    dt       = dt_full[:10]
                    prop_matches.append({
                        "home_team":    teams.get("home", {}).get("name", ""),
                        "away_team":    teams.get("away", {}).get("name", ""),
                        "league":       league,
                        "date":         pd.Timestamp(dt) if dt else today,
                        "kickoff_utc":  dt_full,
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

    # ── Pre-fetch injury lists once per league (4 h cache) ────────────────────
    injured_cache: dict[str, set] = {}
    if _APIFOOTBALL_KEY:
        try:
            from player_model.api_football import get_injured_players as _gip
            from datetime import date as _date
            _today_str = _date.today().isoformat()
            for _lg, _lg_id in config.PROP_LEAGUES.items():
                _szn = config.PROP_SEASONS.get(_lg, "2025")
                injured_cache[_lg] = _gip(_lg_id, _szn, _today_str)
            _n_inj = sum(len(v) for v in injured_cache.values())
            if _n_inj:
                print(f"[predict] Injury filter: {_n_inj} injured/suspended players loaded")
        except Exception as _e:
            print(f"[predict] Injury fetch failed: {_e}")

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

        # Team win probabilities for opponent-weakness multiplier
        home_win_prob = float(ctx.get("home_win_prob", 0.5))
        away_win_prob = float(ctx.get("away_win_prob", 0.5))

        # ── Lineup check: build confirmed starter set for this fixture ─────────
        lineup_starters: set[str] = set()
        lineup_available = False
        fid = match_row.get("fixture_id")
        if _APIFOOTBALL_KEY and fid and not pd.isna(fid):
            try:
                from player_model.api_football import get_fixture_lineup as _gfl
                _lineup = _gfl(int(fid))
                if _lineup:
                    for _team_players in _lineup.values():
                        for _p in _team_players:
                            if _p.get("started"):
                                lineup_starters.add(_norm_player_name(_p.get("player_name", "")))
                    # Only treat lineup as confirmed when both teams have starters posted
                    # (empty startXI = lineup not yet submitted → don't drop any player)
                    if len(lineup_starters) >= 18:
                        lineup_available = True
            except Exception as _e:
                print(f"[predict] Lineup fetch failed (fixture {fid}): {_e}")

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
            player_norm = _norm_player_name(str(feat_row.get("player_name", "")))

            # Skip players not in confirmed starting XI when lineup is confirmed
            if lineup_available and player_norm not in lineup_starters:
                continue

            # Starter proxy: without a confirmed lineup, only tip players who average real
            # starter minutes — avoids tipping fringe/rotation players who may not start.
            if not lineup_available:
                _mpg = feat_row.get("minutes_pg", None)
                if _mpg is not None and not pd.isna(_mpg) and float(_mpg) < config.MIN_STARTER_MINUTES:
                    continue

            # Skip injured / suspended players
            if player_norm in injured_cache.get(league, set()):
                continue

            _is_gk = str(feat_row.get("position", "")).strip().upper().startswith("G")
            _is_wc = "world cup" in str(league).lower()

            for market in config.MARKETS:
                if market == "cards":
                    # The cards model is well-calibrated for CLUB leagues (mean ~0.13 vs the
                    # 12.8% base rate, AUC 0.69). World Cup features get imputed to league
                    # averages, inflating WC card predictions to ~0.40 for everyone. But ~0.40
                    # is roughly CORRECT for genuinely card-prone players, so for WC we apply a
                    # higher bar: gate on the player's REAL booking history (their true rate ≈
                    # the prediction) and drop the rest. Club cards pass through (model is fine).
                    if _is_wc:
                        _real_card_rate = max(float(feat_row.get("cards_pg", 0) or 0),
                                              float(feat_row.get("season_cards_pg", 0) or 0))
                        if _real_card_rate < config.WC_CARD_MIN_RATE:
                            continue
                    # GKs allowed for cards — a bunker/time-wasting GK with real booking history
                    # is a legitimate card tip; the history gate already filters the rest.
                else:
                    # Goalkeepers don't score / shoot / assist — skip them for those markets.
                    if _is_gk:
                        continue
                p_model = float(feat_row.get(f"p_{market}", 0))
                # Team-strength multiplier: boost for heavy favourites, penalty for underdogs
                _is_home = float(feat_row.get("is_home", 0.5)) > 0.5
                _win_prob = home_win_prob if _is_home else away_win_prob
                p_model = p_model * _team_strength_mult(_win_prob, market)

                # Hard quality-mismatch cap: player career tier << opponent tier
                # e.g. Panama striker (career_quality=0.30) vs England CBs (opp_quality=1.0)
                # context_quality_discount = 0.30 → apply extra cap so model can't output
                # inflated probabilities for weak-league players vs elite opposition
                _career_q  = float(feat_row.get("player_career_avg_quality", 0.65))
                _opp_q     = float(feat_row.get("opp_def_player_quality", 1.0))
                _ctx_disc  = _career_q / max(_opp_q, 0.1)
                if _ctx_disc < 0.60 and market in ("goals", "goals2", "sot", "sot2", "sot3"):
                    # Discount scales: ctx_disc=0.50 → ×0.50, ctx_disc=0.30 → ×0.30
                    p_model = p_model * _ctx_disc

                # Cap at realistic maximum then absolute ceiling of 0.95
                p_model = min(p_model, _MARKET_CAPS.get(market, 0.90), 0.95)

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
                # Confirmed starter adds a lazy factor (confidence boost)
                if lineup_available and player_norm in lineup_starters and "STARTER" not in lazy_factors:
                    lazy_factors = ["STARTER"] + lazy_factors
                confidence = _confidence_score(feat_row.to_dict(), len(lazy_factors))
                # World Cup non-card props use imputed national-team features (live ROI ~ -33%)
                # → flag low-confidence and discount so they rank below club-league signals.
                if _is_wc and market != "cards":
                    confidence *= config.WC_PROP_CONF_PENALTY
                    lazy_factors = lazy_factors + ["WC_LOW_CONF"]

                all_tips.append({
                    "date":          date_str,
                    "kickoff_utc":   match_row.get("kickoff_utc", ""),
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
                    "tier":          "AVOID",   # updated when market odds available
                    "kelly_stake":   None,
                })

    if not all_tips:
        return pd.DataFrame()

    tips_df = pd.DataFrame(all_tips)
    tips_df = tips_df.sort_values("model_prob", ascending=False)
    # Dedup on normalized name so accent/hyphen variants (e.g. "Vinícius Júnior" vs
    # "Vinicius Junior") don't produce duplicate Telegram signals for the same player.
    tips_df["_name_norm"] = tips_df["player_name"].apply(_norm_player_name)
    tips_df = tips_df.drop_duplicates(subset=["match", "_name_norm", "market"])
    tips_df = tips_df.drop(columns=["_name_norm"])

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
        lazy_count = len([f for f in str(row.get("lazy_factors", "")).split("|")
                          if f and f != "WC_LOW_CONF"])   # WC_LOW_CONF is a PENALTY, not a positive factor
        ges        = float(row["ges"]) if row.get("ges") is not None else 0.5
        market     = row["market"]

        tier = _classify_tier(ev_val, mkt_odds, rel_edge, confidence, lazy_count, ges, market, row.get("position", ""))

        tips_df.at[idx, "market_odds"]  = mkt_odds
        tips_df.at[idx, "fair_implied"] = round(fair_prob, 4)
        tips_df.at[idx, "edge_abs"]     = round(p_model - fair_prob, 4)
        tips_df.at[idx, "edge_rel"]     = rel_edge
        tips_df.at[idx, "ev"]           = ev_val
        tips_df.at[idx, "tier"]         = tier
        tips_df.at[idx, "kelly_stake"]  = _kelly_stake(ev_val, mkt_odds) if tier != "AVOID" else 0.0

    return tips_df


# Markets with no live Odds API coverage — tiered via calibration base rates.
_NO_ODDS_MARKETS = {"assists"}


def enrich_no_odds_markets(tips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign tier for markets that have no Odds API coverage (assists).
    Uses calibration base rates as synthetic bookmaker odds so these tips
    can earn VALUABLE/MARKSMAN/SNIPER tier on model confidence alone.

    Synthetic price = 1 / (base_rate × 1.10)  — assumes 10% overround.
    """
    if tips_df.empty:
        return tips_df

    cal_path = config.OUTPUT_DIR / "player_props_calibration.json"
    try:
        with open(cal_path) as _f:
            calibration = json.load(_f)
    except Exception:
        return tips_df

    for idx, row in tips_df.iterrows():
        market = row["market"]
        if market not in _NO_ODDS_MARKETS:
            continue
        if row.get("market_odds"):   # already enriched by real Odds API
            continue

        cal       = calibration.get(market, {})
        base_rate = cal.get("base_rate")
        if not base_rate:
            continue

        # Synthetic market odds: base rate × 1.10 overround
        implied_prob = base_rate * 1.10
        mkt_odds     = round(1.0 / implied_prob, 2)

        p_model    = float(row["model_prob"])
        # Fair prob = base_rate (we know the true calibrated probability directly)
        fair_prob  = base_rate
        ev_val     = _ev(p_model, mkt_odds)
        rel_edge   = _relative_edge(p_model, fair_prob)
        confidence = float(row["confidence"])
        lazy_count = len([f for f in str(row.get("lazy_factors", "")).split("|")
                          if f and f != "WC_LOW_CONF"])   # WC_LOW_CONF is a PENALTY, not a positive factor
        ges        = float(row["ges"]) if row.get("ges") is not None else 0.5

        tier = _classify_tier(ev_val, mkt_odds, rel_edge, confidence, lazy_count, ges, market, row.get("position", ""))

        src = str(row.get("data_source", "") or "")
        tips_df.at[idx, "market_odds"]  = mkt_odds
        tips_df.at[idx, "fair_implied"] = round(fair_prob, 4)
        tips_df.at[idx, "edge_abs"]     = round(p_model - fair_prob, 4)
        tips_df.at[idx, "edge_rel"]     = rel_edge
        tips_df.at[idx, "ev"]           = ev_val
        tips_df.at[idx, "tier"]         = tier
        tips_df.at[idx, "data_source"]  = (src + "|calibration_implied").lstrip("|")
        tips_df.at[idx, "kelly_stake"]  = _kelly_stake(ev_val, mkt_odds) if tier != "AVOID" else 0.0

    return tips_df
