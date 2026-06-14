"""
Player Model Pipeline v2
========================
Usage:
  python -m player_model.pipeline --mode collect   # fetch FBref training data
  python -m player_model.pipeline --mode train     # train 4 models
  python -m player_model.pipeline --mode predict   # generate today's player tips
  python -m player_model.pipeline --mode all       # collect + train + predict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Force UTF-8 stdout so team names with non-ASCII chars (ü, é, etc.) don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_v9 = Path(__file__).resolve().parents[1]
load_dotenv(_v9 / ".env")
sys.path.insert(0, str(_v9))

from player_model import config
from player_model.data_fetcher import (
    collect_history, collect_match_history, collect_national_team_history,
    FBREF_LEAGUES, EUROPEAN_CUPS, APIFOOTBALL_LEAGUES,
)
from player_model.feature_engineering import build_features
from player_model.model import train, save_model
from player_model.predict import run_player_predictions, enrich_with_odds
from player_model.odds_fetcher import fetch_prop_odds, match_odds_to_tips
from player_model.ledger import append_player_signals

HISTORY_CACHE = config.BASE_DIR / "player_history.parquet"


# ── Phase 2: Collect ──────────────────────────────────────────────────────────

def mode_collect(extended: bool = False, last_n: int = 100) -> None:
    leagues = {**APIFOOTBALL_LEAGUES, **EUROPEAN_CUPS} if extended else APIFOOTBALL_LEAGUES
    label = "core + European cups" if extended else "core leagues only"
    print(f"[collect] Fetching per-match player stats ({label}, last {last_n} fixtures/league)...")
    rows = collect_match_history(leagues=leagues, last_n=last_n)
    if not rows:
        print("[collect] No data — FBref may be rate-limiting. Try again in a few minutes.")
        return

    df = build_features(rows)
    if df.empty:
        print("[collect] Feature engineering returned empty DataFrame.")
        return

    df.to_parquet(HISTORY_CACHE, index=False)
    print(f"[collect] Saved {len(df)} player rows, {df['player_id'].nunique()} players -> {HISTORY_CACHE.name}")

    # Summary
    for market in ["target_goals", "target_sot", "target_cards", "target_assists"]:
        if market in df.columns:
            rate = df[market].mean()
            print(f"  {market}: {rate:.1%} positive rate")


# ── WC26 national team collect ────────────────────────────────────────────────

def mode_collect_wc(last_n: int = 10) -> None:
    print(f"[collect-wc] Fetching WC2026 national team player stats (last {last_n} fixtures/team)...")
    rows = collect_national_team_history(last_n=last_n)
    if not rows:
        print("[collect-wc] No data collected.")
        return

    new_df = build_features(rows)
    if new_df.empty:
        print("[collect-wc] Feature engineering returned empty DataFrame.")
        return

    # Merge with existing history (club leagues) — deduplicate on fixture+player
    if HISTORY_CACHE.exists():
        existing = pd.read_parquet(HISTORY_CACHE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["fixture_id", "player_id"])
        combined = combined.reset_index(drop=True)
    else:
        combined = new_df

    combined.to_parquet(HISTORY_CACHE, index=False)
    wc_players = new_df["player_id"].nunique()
    print(f"[collect-wc] +{len(new_df)} WC rows ({wc_players} players) merged -> {len(combined)} total rows")

    for market in ["target_goals", "target_sot", "target_cards", "target_assists"]:
        if market in new_df.columns:
            rate = new_df[market].mean()
            print(f"  {market}: {rate:.1%} positive rate (WC data)")


# ── Phase 3: Train ────────────────────────────────────────────────────────────

def mode_train() -> None:
    if not HISTORY_CACHE.exists():
        print("[train] No history. Run --mode collect first.")
        return

    df = pd.read_parquet(HISTORY_CACHE)
    print(f"[train] Training on {len(df)} rows, {df['player_id'].nunique()} players.")

    for market in config.MARKETS:
        print(f"\n  Training market: {market}")
        try:
            results = train(df, market)
            save_model(results, market)
        except ValueError as e:
            print(f"  [SKIP] {market}: {e}")
        except Exception as e:
            print(f"  [ERROR] {market}: {e}")

    print("\n[train] Done.")


# ── Predict ───────────────────────────────────────────────────────────────────

def mode_predict() -> None:
    if not HISTORY_CACHE.exists():
        print("[predict] No history. Run --mode collect + train first.")
        return

    history = pd.read_parquet(HISTORY_CACHE)

    # Try to enrich with API-Football recent form if key available
    import os
    api_key = os.getenv("APIFOOTBALL_KEY", "")
    if api_key:
        print("[predict] API-Football key found — will fetch recent match stats for rolling features.")
        _enrich_with_recent_form(history)
    else:
        print("[predict] No APIFOOTBALL_KEY — using season stats only (less precise rolling features).")

    tips = run_player_predictions(
        bets_csv=config.OUTPUT_DIR / "bets.csv",
        history_df=history,
    )

    if tips.empty:
        print("[predict] No player tips generated.")
        return

    # Enrich with live market odds from The Odds API
    from player_model.odds_fetcher import _load_odds_key
    odds_api_key = _load_odds_key()
    if odds_api_key:
        print("[predict] Fetching player prop odds from The Odds API...")
        try:
            odds_raw = fetch_prop_odds(tips)
            if odds_raw:
                odds_mapped = match_odds_to_tips(tips, odds_raw)
                tips = enrich_with_odds(tips, odds_mapped)
                # Re-save enriched CSV
                tips.to_csv(config.OUTPUT_DIR / "player_tips.csv", index=False)
                print(f"[predict] Odds enriched for {len(odds_mapped)} player/market pairs")
            else:
                print("[predict] No player prop odds returned from Odds API")
        except Exception as e:
            print(f"[predict] Odds enrichment failed: {e}")
    else:
        print("[predict] No ODDS_API_KEY — skipping odds enrichment (tier will stay WATCH)")

    sniper   = tips[tips["tier"] == "SNIPER"]
    marksman = tips[tips["tier"] == "MARKSMAN"]
    valuable = tips[tips["tier"] == "VALUABLE"]
    print(f"[predict] SNIPER:{len(sniper)}  MARKSMAN:{len(marksman)}  VALUABLE:{len(valuable)}")

    n_new = append_player_signals(tips)
    if n_new:
        print(f"[predict] player_ledger: +{n_new} new signal(s) recorded")


def _enrich_with_recent_form(history: pd.DataFrame) -> None:
    """Use API-Football to update rolling stats for players in upcoming matches."""
    try:
        from player_model.api_football import get_recent_fixtures, get_fixture_player_stats
        from player_model.feature_engineering import build_rolling_features

        bets_csv = config.OUTPUT_DIR / "bets.csv"
        if not bets_csv.exists():
            return

        bets = pd.read_csv(bets_csv)
        leagues_needed = bets["league"].unique().tolist()

        for league, lg_id in config.PROP_LEAGUES.items():
            if not any(league.lower() in l.lower() for l in leagues_needed):
                continue
            season = config.PROP_SEASONS.get(league, "2025")
            fixtures = get_recent_fixtures(lg_id, season, last_n=5)
            print(f"  [{league}] {len(fixtures)} recent fixtures fetched")

            for fix in fixtures[:3]:  # limit to 3 most recent to save API calls
                fix_id = fix.get("fixture", {}).get("id")
                if not fix_id:
                    continue
                player_stats = get_fixture_player_stats(fix_id)
                # Update rolling stats in history for matched players
                for stat in player_stats:
                    pid = stat.get("player_id")
                    if pid and len(history[history["player_id"] == pid]) > 0:
                        idx = history[history["player_id"] == pid].index[-1]
                        # Update with most recent match data
                        mins = int(stat.get("minutes_played") or 0)
                        if mins > 0:
                            history.at[idx, "goals_pg"]  = stat.get("goals", 0)
                            history.at[idx, "sot_pg"]    = stat.get("shots_on_target", 0)
                            history.at[idx, "cards_pg"]  = stat.get("yellow_card", 0)

    except Exception as e:
        print(f"  [enrich] Error enriching with API-Football: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Player Model Pipeline")
    parser.add_argument("--mode", choices=["collect", "collect-wc", "train", "predict", "all"], required=True)
    parser.add_argument("--extended", action="store_true",
                        help="Also collect Champions League / Europa League / Conference League")
    parser.add_argument("--last-n", type=int, default=99,
                        help="Number of recent fixtures to collect per league (max 99, default: 99)")
    args = parser.parse_args()

    if args.mode == "collect" or args.mode == "all":
        mode_collect(extended=args.extended, last_n=args.last_n)
    if args.mode == "collect-wc":
        mode_collect_wc(last_n=args.last_n)
    if args.mode == "train" or args.mode == "all":
        mode_train()
    if args.mode == "predict" or args.mode == "all":
        mode_predict()


if __name__ == "__main__":
    main()
