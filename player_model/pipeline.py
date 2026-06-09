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

_v9 = Path(__file__).resolve().parents[1]
load_dotenv(_v9 / ".env")
sys.path.insert(0, str(_v9))

from player_model import config
from player_model.data_fetcher import collect_history, FBREF_LEAGUES
from player_model.feature_engineering import build_features
from player_model.model import train, save_model
from player_model.predict import run_player_predictions

HISTORY_CACHE = config.BASE_DIR / "player_history.parquet"


# ── Phase 2: Collect ──────────────────────────────────────────────────────────

def mode_collect() -> None:
    print("[collect] Fetching player season stats from FBref (free)...")
    rows = collect_history(leagues=FBREF_LEAGUES)
    if not rows:
        print("[collect] No data — FBref may be rate-limiting. Try again in a few minutes.")
        return

    df = build_features(rows)
    if df.empty:
        print("[collect] Feature engineering returned empty DataFrame.")
        return

    df.to_parquet(HISTORY_CACHE, index=False)
    print(f"[collect] Saved {len(df)} player rows, {df['player_id'].nunique()} players → {HISTORY_CACHE.name}")

    # Summary
    for market in ["target_goals", "target_sot", "target_cards", "target_assists"]:
        if market in df.columns:
            rate = df[market].mean()
            print(f"  {market}: {rate:.1%} positive rate")


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

    sniper   = tips[tips["tier"] == "SNIPER"]
    marksman = tips[tips["tier"] == "MARKSMAN"]
    valuable = tips[tips["tier"] == "VALUABLE"]
    print(f"[predict] SNIPER:{len(sniper)}  MARKSMAN:{len(marksman)}  VALUABLE:{len(valuable)}")


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
                        mins = stat.get("minutes_played", 0)
                        if mins > 0:
                            history.at[idx, "goals_pg"]  = stat.get("goals", 0)
                            history.at[idx, "sot_pg"]    = stat.get("shots_on_target", 0)
                            history.at[idx, "cards_pg"]  = stat.get("yellow_card", 0)

    except Exception as e:
        print(f"  [enrich] Error enriching with API-Football: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Player Model Pipeline")
    parser.add_argument("--mode", choices=["collect", "train", "predict", "all"], required=True)
    args = parser.parse_args()

    if args.mode == "collect" or args.mode == "all":
        mode_collect()
    if args.mode == "train" or args.mode == "all":
        mode_train()
    if args.mode == "predict" or args.mode == "all":
        mode_predict()


if __name__ == "__main__":
    main()
