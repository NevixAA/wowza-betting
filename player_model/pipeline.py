"""
Player Model Pipeline
=====================
Usage:
  python -m player_model.pipeline --mode collect   # one-time: fetch training data
  python -m player_model.pipeline --mode train     # train 4 models
  python -m player_model.pipeline --mode predict   # generate today's player tips

Credit budget for collect: capped at 800 API calls by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_v9 = Path(__file__).resolve().parents[1]
load_dotenv(_v9 / ".env")
sys.path.insert(0, str(_v9))

from player_model import config
from player_model.data_fetcher import collect_history
from player_model.feature_engineering import build_features
from player_model.model import train, save_model
from player_model.predict import run_player_predictions

HISTORY_CACHE = config.BASE_DIR / "player_history.parquet"


def mode_collect(max_fixtures: int = 800) -> None:
    print(f"[collect] Fetching history (cap={max_fixtures} API calls)...")
    rows = collect_history(max_fixtures=max_fixtures)
    if not rows:
        print("[collect] No data collected.")
        return

    df = pd.DataFrame(rows)
    df = build_features(df)
    df.to_parquet(HISTORY_CACHE, index=False)
    print(f"[collect] Saved {len(df)} player rows → {HISTORY_CACHE}")


def mode_train() -> None:
    if not HISTORY_CACHE.exists():
        print("[train] No history found. Run --mode collect first.")
        return

    df = pd.read_parquet(HISTORY_CACHE)
    print(f"[train] Training on {len(df)} rows, {df['player_id'].nunique()} players.")

    for market in config.MARKETS:
        print(f"\n  Training market: {market}")
        try:
            results = train(df, market)
            save_model(results, market)
        except Exception as e:
            print(f"  [WARN] {market} failed: {e}")

    print("\n[train] Done.")


def mode_predict() -> None:
    if not HISTORY_CACHE.exists():
        print("[predict] No history. Run --mode collect + train first.")
        return

    history = pd.read_parquet(HISTORY_CACHE)
    tips = run_player_predictions(
        bets_csv=config.OUTPUT_DIR / "bets.csv",
        history_df=history,
    )

    if tips.empty:
        print("[predict] No player tips generated.")
    else:
        print(f"[predict] {len(tips)} player prop rows written.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Player Model Pipeline")
    parser.add_argument(
        "--mode", choices=["collect", "train", "predict"], required=True
    )
    parser.add_argument(
        "--max-fixtures", type=int, default=800,
        help="Max API calls for collect mode (default 800)",
    )
    args = parser.parse_args()

    if args.mode == "collect":
        mode_collect(args.max_fixtures)
    elif args.mode == "train":
        mode_train()
    elif args.mode == "predict":
        mode_predict()


if __name__ == "__main__":
    main()
