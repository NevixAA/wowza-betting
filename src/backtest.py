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
    _checkpoint_path = config.OUTPUT_DIR / "backtest_checkpoint_over25.csv"
    _CHECKPOINT_EVERY = 50

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

        if window_count % _CHECKPOINT_EVERY == 0:
            try:
                pd.concat(all_preds, ignore_index=True).to_csv(_checkpoint_path, index=False)
                log.info(f"[checkpoint] {window_count} windows → {_checkpoint_path.name}")
            except Exception:
                pass

    if not all_preds:
        raise RuntimeError("Backtest generated no predictions")

    results_df = pd.concat(all_preds, ignore_index=True)

    # Compute P&L
    results_df["pnl"] = [_compute_pnl(r) for r in results_df.to_dict("records")]
    results_df["cumulative_pnl"] = results_df["pnl"].cumsum()

    # Drawdown
    peak = results_df["cumulative_pnl"].cummax()
    results_df["drawdown"] = results_df["cumulative_pnl"] - peak

    # ── Tier breakdown ────────────────────────────────────────────────────────
    # signal_tier column: SNIPER / MARKSMAN / VALUABLE / AVOID
    # Headline ROI uses SNIPER + MARKSMAN only (VALUABLE is info-only, not a tip).
    # VALUABLE is reported separately so it never pollutes the performance metric.

    def _tier_stats(rows: pd.DataFrame) -> dict:
        if rows.empty:
            return {"bets": 0, "wins": 0, "win_rate": 0.0, "profit": 0.0, "staked": 0.0, "roi_%": 0.0}
        wins_  = int((rows["pnl"] > 0).sum())
        profit = float(rows["pnl"].sum())
        staked = float(rows["bet_stake"].sum())
        return {
            "bets":     len(rows),
            "wins":     wins_,
            "win_rate": round(wins_ / len(rows), 4),
            "profit":   round(profit, 3),
            "staked":   round(staked, 3),
            "roi_%":    round(profit / staked * 100, 2) if staked > 0 else 0.0,
        }

    sniper_rows   = results_df[results_df["signal_tier"] == "SNIPER"]
    marksman_rows = results_df[results_df["signal_tier"] == "MARKSMAN"]
    valuable_rows = results_df[results_df["signal_tier"] == "VALUABLE"]

    sniper_stats   = _tier_stats(sniper_rows)
    marksman_stats = _tier_stats(marksman_rows)
    valuable_stats = _tier_stats(valuable_rows)

    # Headline = SNIPER + MARKSMAN (placed tips only)
    placed_rows  = results_df[results_df["signal_tier"].isin(["SNIPER", "MARKSMAN"])]
    total_bets   = len(placed_rows)
    wins         = int((placed_rows["pnl"] > 0).sum())
    total_profit = float(placed_rows["pnl"].sum())
    total_staked = float(placed_rows["bet_stake"].sum())
    roi          = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    max_dd       = float(results_df["drawdown"].min())

    # Sharpe (annualized from daily P&L — placed tips only)
    daily  = placed_rows.groupby("date")["pnl"].sum() if not placed_rows.empty else pd.Series(dtype=float)
    sharpe = float((daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0)

    # By-league breakdown (placed tips)
    league_summary = (
        placed_rows.groupby("league")
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
        # Placed tips (SNIPER + MARKSMAN) — headline metrics
        "total_bets":               total_bets,
        "wins":                     wins,
        "win_rate":                 round(wins / total_bets, 4) if total_bets else 0.0,
        "total_profit_units":       round(total_profit, 3),
        "total_staked_units":       round(total_staked, 3),
        "roi_%":                    round(roi, 2),
        "max_drawdown_units":       round(max_dd, 3),
        "sharpe_ratio":             round(sharpe, 3),
        "windows_trained":          window_count,
        # Per-tier breakdown
        "sniper_bets":              sniper_stats["bets"],
        "sniper_roi_%":             sniper_stats["roi_%"],
        "sniper_win_rate":          sniper_stats["win_rate"],
        "marksman_bets":            marksman_stats["bets"],
        "marksman_roi_%":           marksman_stats["roi_%"],
        "marksman_win_rate":        marksman_stats["win_rate"],
        "valuable_bets":            valuable_stats["bets"],
        "valuable_roi_%":           valuable_stats["roi_%"],
    }

    log.info(
        f"Backtest complete: {total_bets} placed tips (S:{sniper_stats['bets']} M:{marksman_stats['bets']}) | "
        f"WR={summary['win_rate']:.1%} | ROI={roi:.1f}% | "
        f"Sharpe={sharpe:.2f} | MaxDD={max_dd:.2f}u"
    )

    return results_df, summary, league_summary


# ── Side-market walk-forward backtest ─────────────────────────────────────────

_OVERROUND = 1.08  # assumed bookmaker margin for side markets

# Market-average fallback odds when bookmaker odds aren't in the CSV.
# Used only when odds_col is all-NaN (e.g. Football-Data new-format leagues).
_DEFAULT_ODDS = {"btts": 1.85, "over15": 1.40, "over35": 2.60}

def _side_tier(edge: float) -> str:
    if edge >= 0.10: return "SNIPER"
    if edge >= 0.08: return "MARKSMAN"
    if edge >= 0.04: return "VALUABLE"
    return "AVOID"


def run_side_market_backtest(
    df: pd.DataFrame,
    target: str,
    walk_size: int = None,
    min_train: int = None,
    enabled_leagues: set = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Walk-forward backtest for a single side market (btts / over15 / over35).

    Uses the same strict no-leakage protocol as run_backtest().
    Tier thresholds: SNIPER ≥10%, MARKSMAN ≥8%, VALUABLE ≥4% edge.
    Headline ROI uses SNIPER + MARKSMAN only.

    Parameters
    ----------
    target : column name for the binary outcome AND odds (e.g. "btts" → odds_btts)
    """
    walk_size = walk_size or config.BACKTEST_WALK_SIZE
    min_train = min_train or config.BACKTEST_MIN_TRAIN
    odds_col  = f"odds_{target}"

    # Only require the target outcome — odds are filled with market defaults if absent
    df = df.dropna(subset=[target]).copy()
    if odds_col not in df.columns:
        df[odds_col] = np.nan
    missing_odds = df[odds_col].isna().sum()
    if missing_odds > 0:
        fallback = _DEFAULT_ODDS.get(target, 2.00)
        df[odds_col] = df[odds_col].fillna(fallback)
        log.info(
            f"[{target}] {missing_odds}/{len(df)} rows missing {odds_col} — "
            f"filled with market default {fallback}"
        )
    df = df.sort_values("date").reset_index(drop=True)

    if enabled_leagues:
        df = df[df["league"].isin(enabled_leagues)]

    if len(df) < min_train + walk_size:
        raise ValueError(
            f"[{target}] Not enough data for backtest: {len(df)} rows "
            f"(need >= {min_train + walk_size})"
        )

    all_preds   = []
    window_count = 0
    _checkpoint_path = config.OUTPUT_DIR / f"backtest_checkpoint_{target}.csv"
    _CHECKPOINT_EVERY = 50

    for start in range(min_train, len(df), walk_size):
        train_df = df.iloc[:start].copy()
        test_df  = df.iloc[start : start + walk_size].copy()

        if len(test_df) == 0:
            break

        try:
            model_results = train_model(train_df, target=target, train_ratio=0.85)
        except Exception as e:
            log.warning(f"[{target}] Window {window_count}: training failed — {e}")
            continue

        payload = {
            "models":       {k: v["model"] for k, v in model_results.items()},
            "feature_cols": model_results[next(iter(model_results))]["feature_cols"],
        }

        try:
            p = predict_proba(test_df, payload=payload)
            test_df[f"p_{target}"] = p.values
        except Exception as e:
            log.warning(f"[{target}] Window {window_count}: prediction failed — {e}")
            continue

        # Compute edge + tier
        test_df["fair_prob"]  = (1.0 / test_df[odds_col]) / _OVERROUND
        test_df["edge"]       = test_df[f"p_{target}"] - test_df["fair_prob"]
        test_df["signal_tier"] = test_df["edge"].apply(_side_tier)
        test_df["bet"]        = test_df["signal_tier"].apply(
            lambda t: "BET" if t != "AVOID" else "AVOID"
        )

        # P&L — 1 unit stake for SNIPER/MARKSMAN, 0.5 for VALUABLE
        def _stake(tier):
            return 1.0 if tier == "SNIPER" else (0.75 if tier == "MARKSMAN" else 0.5)

        test_df["bet_stake"] = test_df["signal_tier"].apply(_stake)
        test_df.loc[test_df["bet"] == "AVOID", "bet_stake"] = 0.0

        def _pnl(row):
            if row["bet"] == "AVOID":
                return 0.0
            stake = row["bet_stake"]
            won   = row[target] == 1
            return stake * (row[odds_col] - 1) if won else -stake

        test_df["pnl"] = [_pnl(r) for r in test_df.to_dict("records")]

        all_preds.append(test_df)
        window_count += 1

        if window_count % _CHECKPOINT_EVERY == 0:
            try:
                pd.concat(all_preds, ignore_index=True).to_csv(_checkpoint_path, index=False)
                log.info(f"[checkpoint/{target}] {window_count} windows → {_checkpoint_path.name}")
            except Exception:
                pass

    if not all_preds:
        raise RuntimeError(f"[{target}] Backtest generated no predictions")

    results_df = pd.concat(all_preds, ignore_index=True)
    results_df["cumulative_pnl"] = results_df["pnl"].cumsum()
    peak = results_df["cumulative_pnl"].cummax()
    results_df["drawdown"] = results_df["cumulative_pnl"] - peak

    def _tier_stats(rows):
        if rows.empty:
            return {"bets": 0, "wins": 0, "win_rate": 0.0, "profit": 0.0, "staked": 0.0, "roi_%": 0.0}
        wins_  = int((rows["pnl"] > 0).sum())
        profit = float(rows["pnl"].sum())
        staked = float(rows["bet_stake"].sum())
        return {
            "bets": len(rows), "wins": wins_,
            "win_rate": round(wins_ / len(rows), 4),
            "profit": round(profit, 3), "staked": round(staked, 3),
            "roi_%": round(profit / staked * 100, 2) if staked > 0 else 0.0,
        }

    sniper_stats   = _tier_stats(results_df[results_df["signal_tier"] == "SNIPER"])
    marksman_stats = _tier_stats(results_df[results_df["signal_tier"] == "MARKSMAN"])
    valuable_stats = _tier_stats(results_df[results_df["signal_tier"] == "VALUABLE"])

    placed = results_df[results_df["signal_tier"].isin(["SNIPER", "MARKSMAN"])]
    total_bets   = len(placed)
    total_profit = float(placed["pnl"].sum())
    total_staked = float(placed["bet_stake"].sum())
    roi          = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    max_dd       = float(results_df["drawdown"].min())

    league_summary = (
        placed.groupby("league")
        .apply(lambda g: pd.Series({
            "bets":   len(g),
            "wins":   int((g["pnl"] > 0).sum()),
            "profit": round(float(g["pnl"].sum()), 3),
            "roi_%":  round(float(g["pnl"].sum() / g["bet_stake"].sum() * 100)
                            if g["bet_stake"].sum() > 0 else 0, 2),
        }))
        .reset_index()
    ) if not placed.empty else pd.DataFrame()

    summary = {
        "market":                   target,
        "total_matches_backtested": len(results_df),
        "total_bets":               total_bets,
        "wins":                     int((placed["pnl"] > 0).sum()) if not placed.empty else 0,
        "win_rate":                 round(int((placed["pnl"] > 0).sum()) / total_bets, 4) if total_bets else 0.0,
        "total_profit_units":       round(total_profit, 3),
        "total_staked_units":       round(total_staked, 3),
        "roi_%":                    round(roi, 2),
        "max_drawdown_units":       round(max_dd, 3),
        "windows_trained":          window_count,
        "sniper_bets":              sniper_stats["bets"],
        "sniper_roi_%":             sniper_stats["roi_%"],
        "marksman_bets":            marksman_stats["bets"],
        "marksman_roi_%":           marksman_stats["roi_%"],
        "valuable_bets":            valuable_stats["bets"],
        "valuable_roi_%":           valuable_stats["roi_%"],
    }

    log.info(
        f"[{target}] Backtest: {total_bets} tips (S:{sniper_stats['bets']} M:{marksman_stats['bets']}) | "
        f"ROI={roi:.1f}%"
    )

    return results_df, summary, league_summary


def optimize_side_market_thresholds(
    results_df: pd.DataFrame,
    target: str,
    min_bets: int = 20,
    edge_min: float = 0.04,
    edge_max: float = 0.20,
    edge_step: float = 0.01,
) -> dict:
    """
    Per-league edge threshold optimizer for a side market.

    For each league in results_df, grid-searches edge thresholds to find
    the cutoff that maximizes ROI (minimum `min_bets` bets required).

    Returns
    -------
    dict: {
        league: {
            "sniper_th":  float,   # optimal threshold (use as SNIPER cutoff)
            "marksman_th": float,  # sniper_th - 0.02 (minimum 0.04)
            "roi":        float,   # ROI at optimal threshold
            "bets":       int,
            "drop":       bool,    # True if no threshold yields ROI > 0
        }
    }
    """
    import json

    thresholds = np.arange(edge_min, edge_max + edge_step / 2, edge_step)
    results = {}

    for league, grp in results_df.groupby("league"):
        best_roi   = -999.0
        best_th    = edge_min
        best_bets  = 0

        for th in thresholds:
            bets = grp[grp["edge"] >= th]
            if len(bets) < min_bets:
                continue
            staked = float(bets["bet_stake"].sum())
            if staked <= 0:
                continue
            roi = float(bets["pnl"].sum() / staked * 100)
            if roi > best_roi:
                best_roi  = roi
                best_th   = float(round(th, 4))
                best_bets = len(bets)

        drop = (best_roi <= 0 or best_bets < min_bets)
        results[league] = {
            "sniper_th":   best_th,
            "marksman_th": max(round(best_th - 0.02, 4), edge_min),
            "roi":         round(best_roi, 2) if not drop else None,
            "bets":        best_bets,
            "drop":        drop,
        }

    kept    = sum(1 for v in results.values() if not v["drop"])
    dropped = sum(1 for v in results.values() if v["drop"])
    log.info(
        f"[{target}] Threshold opt: {kept} leagues kept, {dropped} dropped "
        f"(no threshold yielded ROI > 0 with {min_bets}+ bets)"
    )
    return results
