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
    # A FILLED PRICE IS NOT A PRICE. The fill stays — the model still scores these rows and
    # they still contribute to TRAINING, which needs no odds — but they are flagged and can
    # never be counted as bets, because an "edge" against a constant is not an edge.
    #
    # WHY THIS MATTERED. With a constant price, fair_prob = (1/1.40)/1.08 = 0.6614 is also
    # constant, so `edge = p_model - 0.6614` and the tier is a bare model-probability threshold
    # with no market in it. The reported ROI was the payout of an invented number. Measured on
    # the committed outputs: over15 was 12,186/12,187 rows (100.0%) at the 1.40 default, and
    # 2,121 of its 2,122 placed SNIPER+MARKSMAN tips sat on that constant. Those numbers went on
    # to certify over15 as an approved market for Bundesliga 2 (+13.5%), Championship (+7.86%),
    # League Two (+9.97%) and Serie B (+4.21%) in output/league_roi_config.json, which
    # telegram_bot/notifier.py reads to decide what to send.
    #
    # The contrast is the argument: btts was certified on real prices (0 of its 925 placed tips
    # used the default) and is running at +27.3% ROI live over 167 settled bets, while over15 was
    # certified entirely on the constant and is running at -1.0% over 63. Honest certification
    # and live performance agree; fabricated certification and live performance do not.
    df["price_synthetic"] = df[odds_col].isna()
    missing_odds = int(df["price_synthetic"].sum())
    if missing_odds > 0:
        fallback = _DEFAULT_ODDS.get(target, 2.00)
        df[odds_col] = df[odds_col].fillna(fallback)
        pct = 100.0 * missing_odds / max(len(df), 1)
        log.warning(
            f"[{target}] {missing_odds}/{len(df)} rows ({pct:.1f}%) have NO real {odds_col} — "
            f"filled with {fallback} for scoring only and EXCLUDED from bets/ROI"
        )
        if missing_odds == len(df):
            log.error(
                f"[{target}] NO PRICED DATA AT ALL — 100% of rows are synthetic. This market "
                f"cannot be backtested and must not be certified. Any ROI reported for it "
                f"previously was the payout of a constant."
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
        # Synthetic-priced rows can never be bets. NO_PRICE is deliberately a distinct label
        # from AVOID: AVOID means the edge was measured and rejected, NO_PRICE means there was
        # nothing to measure against. Collapsing the two is how 2,121 unpriced rows became
        # "placed tips" with a reported ROI.
        _synth = test_df["price_synthetic"].fillna(False).astype(bool)
        test_df.loc[_synth, "signal_tier"] = "NO_PRICE"
        test_df.loc[_synth, "edge"] = np.nan
        test_df["bet"]        = test_df["signal_tier"].apply(
            lambda t: "BET" if t not in ("AVOID", "NO_PRICE") else "AVOID"
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
    min_oos_bets: int = 20,
    edge_min: float = 0.04,
    edge_max: float = 0.20,
    edge_step: float = 0.01,
) -> dict:
    """
    Per-league edge threshold optimizer for a side market — now with an out-of-sample gate.

    WHAT WAS WRONG. This function used to be a pure in-sample grid search: for each league, pick
    the threshold with the best ROI over ALL the data, and mark `drop` only when no threshold
    cleared zero. Nothing was ever held back, so the number it produced was the best of ~17
    candidates chosen on the same rows it was scored on. That is retrospective tuning, which
    invariant 6 forbids, and `optimize_standard_thresholds` had a walk-forward pass for exactly
    this reason while this one did not.

    It is not a theoretical concern. Pro's threshold study (2026-09-22) fitted per-league bars on
    an earlier slice and applied them to a later one: of the four cells with enough data to
    check, THREE got worse out of sample — Norway -29.7pp, China -40.9pp, Finland -61.6pp. An
    in-sample-optimal threshold is not merely unproven, it actively cost money on fresh fixtures.

    THE FIX, mirroring the standard optimizer exactly so the two behave alike:
      * `sniper_th` is still the all-data best — it remains the most stable point estimate, and
        it is what gets used once a league is approved.
      * A WALK-FORWARD pass now tunes on prior seasons and bets the next one blind, producing
        `roi_oos` / `bets_oos`.
      * `approved` is True only when that out-of-sample ROI is positive on >= min_oos_bets.

    APPROVAL NEVER SILENCES A LEAGUE. Per the 2026/27 live-test policy, an unapproved league
    still emits tips at the GLOBAL bar so it keeps generating a CLV/ROI record — the whole point
    is to learn about it, and a league you stopped tipping is a league you can never learn about.
    Approval decides whether its OWN learned threshold is trusted over the global one, nothing
    more.

    `drop` is kept for backward compatibility with the merged best_params_side_markets.json that
    pipeline._generate_side_bets already reads, and it keeps its old in-sample meaning. Read
    `approved` for the question "has this threshold been shown to work on data it was not fitted
    to".

    Returns
    -------
    dict: {league: {sniper_th, marksman_th, roi, bets, drop, roi_oos, bets_oos, approved}}
    """
    thresholds = np.arange(edge_min, edge_max + edge_step / 2, edge_step)
    results = {}

    def _best(sub: pd.DataFrame, floor: int):
        """(threshold, bets, roi) maximising ROI on `sub`, or None if never enough bets."""
        best = None
        for th in thresholds:
            sel = sub[sub["edge"] >= th]
            if len(sel) < floor:
                continue
            staked = float(sel["bet_stake"].sum())
            if staked <= 0:
                continue
            roi = float(sel["pnl"].sum() / staked * 100)
            if best is None or roi > best[2]:
                best = (float(round(th, 4)), len(sel), roi)
        return best

    has_season = "season" in results_df.columns
    seasons = sorted(results_df["season"].astype(str).unique()) if has_season else []

    for league, grp in results_df.groupby("league"):
        ins = _best(grp, min_bets)
        best_th = ins[0] if ins else edge_min
        best_bets = ins[1] if ins else 0
        best_roi = ins[2] if ins else -999.0

        # Walk-forward: tune on the seasons before, bet the next one blind. Identical shape to
        # optimize_standard_thresholds so a reader comparing the two sees one method, not two.
        oos_rows = []
        for i in range(1, len(seasons)):
            train = grp[grp["season"].astype(str).isin(seasons[:i])]
            test = grp[grp["season"].astype(str) == seasons[i]]
            bt = _best(train, max(10, min_bets // 2))
            if bt is None or test.empty:
                continue
            sel = test[test["edge"] >= bt[0]]
            if len(sel):
                oos_rows.append(sel)
        oos = pd.concat(oos_rows) if oos_rows else grp.iloc[0:0]
        oos_staked = float(oos["bet_stake"].sum()) if len(oos) else 0.0
        oos_roi = float(oos["pnl"].sum() / oos_staked * 100) if oos_staked > 0 else None
        oos_bets = len(oos)

        drop = (best_roi <= 0 or best_bets < min_bets)
        approved = bool(oos_roi is not None and oos_roi > 0 and oos_bets >= min_oos_bets)
        results[league] = {
            "sniper_th":   best_th,
            "marksman_th": max(round(best_th - 0.02, 4), edge_min),
            "roi":         round(best_roi, 2) if not drop else None,
            "bets":        best_bets,
            "drop":        drop,
            "roi_oos":     round(oos_roi, 2) if oos_roi is not None else None,
            "bets_oos":    oos_bets,
            "approved":    approved,
        }

    kept     = sum(1 for v in results.values() if not v["drop"])
    approved = sum(1 for v in results.values() if v["approved"])
    log.info(
        f"[{target}] Threshold opt: {len(results)} leagues, {kept} profitable in-sample, "
        f"{approved} APPROVED out-of-sample (OOS ROI > 0 on {min_oos_bets}+ blind bets). "
        f"Unapproved leagues still tip at the global bar."
    )
    return results


def optimize_standard_thresholds(
    results_df: pd.DataFrame,
    min_bets: int = 30,
    min_oos_bets: int = 20,
    edge_min: float = 0.04,
    edge_max: float = 0.25,
    edge_step: float = 0.01,
) -> dict:
    """
    Per-league SNIPER edge-threshold optimizer for the STANDARD O/U 2.5 model.

    The standard model historically used HAND-SET per-league thresholds
    (config.LEAGUE_SNIPER_THRESHOLDS, calibrated once by hand). This is the
    auto-tuning equivalent of optimize_side_market_thresholds — which only ever
    existed for the newer side markets (BTTS / Over1.5 / Over3.5).

    Method (overfit-resistant, so an in-sample mirage is never marked real-money):
      * Flat-stake PnL is computed directly from best_side + outcome + odds, so ANY
        candidate threshold can be scored — independent of the tier the backtest
        actually staked (this correctly includes leagues that a stale guard zeroed).
      * COVID seasons are excluded (config.COVID_SEASONS).
      * DEPLOYED `sniper_th` = grid value maximising ROI on ALL non-COVID data
        (>= min_bets) — most data → most stable point estimate.
      * A WALK-FORWARD out-of-sample pass (tune on prior seasons, bet the next one
        blind) gives `roi_oos`; a league is only `approved` when that OOS ROI is
        positive on >= min_oos_bets. Approval gates REAL MONEY only — signals for
        every league still fire (paper/CLV), so this never silences a tip.

    Returns
    -------
    dict: {league: {sniper_th, marksman_th, roi_insample, bets_insample,
                    roi_oos, bets_oos, approved}}
    """
    df = results_df.copy()
    ow = df["over25"] == 1
    pnl_over  = np.where(ow,  df["odds_over25"]  - 1.0, -1.0)
    pnl_under = np.where(~ow, df["odds_under25"] - 1.0, -1.0)
    df["_bpnl"] = np.where(df["best_side"] == "OVER",  pnl_over,
                  np.where(df["best_side"] == "UNDER", pnl_under, np.nan))

    covid = set(getattr(config, "COVID_SEASONS", {"2019/20", "2020/21"}))
    d = df[
        ~df["season"].astype(str).isin(covid)
        & df["best_side"].isin(["OVER", "UNDER"])
        & df["best_edge"].notna()
        & df["_bpnl"].notna()
    ].copy()

    grid    = np.round(np.arange(edge_min, edge_max + edge_step / 2, edge_step), 4)
    seasons = sorted(d["season"].astype(str).unique())

    def _best_threshold(sub: pd.DataFrame, floor_bets: int):
        best = None
        for th in grid:
            sel = sub[sub["best_edge"] >= th]
            if len(sel) < floor_bets:
                continue
            roi = float(sel["_bpnl"].mean() * 100)
            if best is None or roi > best[2]:
                best = (float(th), len(sel), roi)
        return best  # (threshold, bets, roi) or None

    results: dict = {}
    for lg, g in d.groupby("league"):
        ins = _best_threshold(g, min_bets)

        # Walk-forward OOS: tune on prior seasons, apply to each following season.
        oos_rows = []
        for i in range(1, len(seasons)):
            train = g[g["season"].astype(str).isin(seasons[:i])]
            test  = g[g["season"].astype(str) == seasons[i]]
            bt = _best_threshold(train, max(15, min_bets // 2))
            if bt is None or test.empty:
                continue
            sel = test[test["best_edge"] >= bt[0]]
            if len(sel):
                oos_rows.append(sel)
        oos      = pd.concat(oos_rows) if oos_rows else g.iloc[0:0]
        oos_roi  = float(oos["_bpnl"].mean() * 100) if len(oos) else None
        oos_bets = len(oos)

        if ins is None:
            results[lg] = {
                "sniper_th": None, "marksman_th": None,
                "roi_insample": None, "bets_insample": 0,
                "roi_oos": round(oos_roi, 2) if oos_roi is not None else None,
                "bets_oos": oos_bets, "approved": False,
            }
            continue

        th, bets, roi = ins
        approved = (oos_roi is not None and oos_roi > 0 and oos_bets >= min_oos_bets)
        results[lg] = {
            "sniper_th":     round(th, 4),
            "marksman_th":   round(max(th - 0.02, edge_min), 4),
            "roi_insample":  round(roi, 2),
            "bets_insample": bets,
            "roi_oos":       round(oos_roi, 2) if oos_roi is not None else None,
            "bets_oos":      oos_bets,
            "approved":      bool(approved),
        }

    kept = sum(1 for v in results.values() if v["approved"])
    log.info(
        f"[standard] Threshold opt: {len(results)} leagues, {kept} approved "
        f"(OOS ROI > 0 on {min_oos_bets}+ out-of-sample bets)"
    )
    return results
