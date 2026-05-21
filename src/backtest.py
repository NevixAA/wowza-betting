"""
Walk-Forward Backtesting Engine
=================================
STRICT no-leakage protocol:
  1. Sort all data chronologically.
  2. Train on first BACKTEST_MIN_TRAIN matches.
  3. Predict next BACKTEST_WALK_SIZE matches (model never sees these).
  4. Slide window forward by BACKTEST_WALK_SIZE, retrain on all data so far.
  5. Repeat until end of data.

Metrics reported:
  total_bets, win_rate, roi_%
  total_profit_units, total_staked_units
  sharpe_ratio (annualized, based on daily P&L)
  max_drawdown_units
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.model import train as train_model, predict_proba, FEATURE_COLS
from src.betting import evaluate_value

log = logging.getLogger(__name__)


def _compute_pnl(row: dict) -> float:
    if row.get("bet") == "AVOID":
        return 0.0
    stake = row.get("bet_stake", 1.0) or 0.0
    if stake == 0:
        return 0.0
    target_val = row.get("over25", np.nan)
    if pd.isna(target_val):
        return 0.0
    is_over_bet = row["bet"] == "OVER"
    won = (is_over_bet and target_val == 1.0) or (not is_over_bet and target_val == 0.0)
    odds = row["odds_over25"] if is_over_bet else row["odds_under25"]
    return stake * (odds - 1) if won else -stake


def run_backtest(
    df: pd.DataFrame,
    target: str = "over25",
    walk_size: int = None,
    min_train: int = None,
    enabled_leagues: set = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run walk-forward backtest.

    Parameters
    ----------
    df            : feature-engineered DataFrame from feature_engineering.build_features()
    target        : binary target column name
    walk_size     : how many matches to predict per window
    min_train     : minimum training rows before first prediction
    enabled_leagues : restrict backtest to these leagues (None = all)

    Returns
    -------
    results_df : every predicted match with bet decision + P&L
    summary    : dict with aggregate metrics
    """
    walk_size = walk_size or config.BACKTEST_WALK_SIZE
    min_train = min_train or config.BACKTEST_MIN_TRAIN

    df = df.dropna(subset=[target, "odds_over25", "odds_under25"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    if enabled_leagues:
        df = df[df["league"].isin(enabled_leagues)]

    if len(df) < min_train + walk_size:
        raise ValueError(
            f"Not enough data for backtest: {len(df)} rows "
            f"(need >= {min_train + walk_size})"
        )

    all_preds = []
    window_count = 0

    for start in range(min_train, len(df), walk_size):
        train_df = df.iloc[:start].copy()
        test_df  = df.iloc[start : start + walk_size].copy()

        if len(test_df) == 0:
            break

        try:
            model_results = train_model(train_df, target=target, train_ratio=0.85)
        except Exception as e:
            log.warning(f"Window {window_count}: training failed — {e}")
            continue

        payload = {
            "models":       {k: v["model"] for k, v in model_results.items()},
            "feature_cols": model_results[next(iter(model_results))]["feature_cols"],
        }

        try:
            p = predict_proba(test_df, payload=payload)
            test_df["p_over25"] = p.values
        except Exception as e:
            log.warning(f"Window {window_count}: prediction failed — {e}")
            continue

        test_df = evaluate_value(test_df)
        all_preds.append(test_df)
        window_count += 1

    if not all_preds:
        raise RuntimeError("Backtest generated no predictions")

    results_df = pd.concat(all_preds, ignore_index=True)

    # Compute P&L
    results_df["pnl"] = [_compute_pnl(r) for r in results_df.to_dict("records")]
    results_df["cumulative_pnl"] = results_df["pnl"].cumsum()

    # Drawdown
    peak = results_df["cumulative_pnl"].cummax()
    results_df["drawdown"] = results_df["cumulative_pnl"] - peak

    # Summary metrics
    bet_rows     = results_df[results_df["bet"] != "AVOID"]
    total_bets   = len(bet_rows)
    wins         = int((bet_rows["pnl"] > 0).sum())
    total_profit = float(results_df["pnl"].sum())
    total_staked = float(bet_rows["bet_stake"].sum())
    roi          = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    max_dd       = float(results_df["drawdown"].min())

    # Sharpe (annualized from daily P&L)
    daily = results_df.groupby("date")["pnl"].sum()
    sharpe = float((daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0)

    # By-league breakdown
    league_summary = (
        bet_rows.groupby("league")
        .apply(lambda g: pd.Series({
            "bets":   len(g),
            "wins":   int((g["pnl"] > 0).sum()),
            "profit": round(float(g["pnl"].sum()), 3),
            "roi_%":  round(float(g["pnl"].sum() / g["bet_stake"].sum() * 100)
                            if g["bet_stake"].sum() > 0 else 0, 2),
        }))
        .reset_index()
    )

    summary = {
        "total_matches_backtested": len(results_df),
        "total_bets":               total_bets,
        "wins":                     wins,
        "win_rate":                 round(wins / total_bets, 4) if total_bets else 0.0,
        "total_profit_units":       round(total_profit, 3),
        "total_staked_units":       round(total_staked, 3),
        "roi_%":                    round(roi, 2),
        "max_drawdown_units":       round(max_dd, 3),
        "sharpe_ratio":             round(sharpe, 3),
        "windows_trained":          window_count,
    }

    log.info(
        f"Backtest complete: {total_bets} bets | "
        f"WR={summary['win_rate']:.1%} | ROI={roi:.1f}% | "
        f"Sharpe={sharpe:.2f} | MaxDD={max_dd:.2f}u"
    )

    return results_df, summary, league_summary
