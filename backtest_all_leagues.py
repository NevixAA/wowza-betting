"""
Full-league walk-forward backtest — v9
=======================================
Runs the v9 backtest across ALL 11 leagues (no ENABLED_LEAGUES filter).
Outputs:
  - Per-league summary
  - Per-league × per-season breakdown
  - Saved to v9/output/backtest_all_leagues.csv
             v9/output/backtest_season_breakdown.csv
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from src.data_loader import load_all_matches
from src.feature_engineering import build_features
from src.backtest import run_backtest

ALL_LEAGUES = config.STANDARD_FORMAT_LEAGUES | config.NEW_FORMAT_LEAGUES


def _season_breakdown(results_df: pd.DataFrame) -> pd.DataFrame:
    bets = results_df[results_df["bet"].isin(["OVER", "UNDER"])].copy()
    if bets.empty:
        return pd.DataFrame()

    rows = []
    for (league, season), grp in bets.groupby(["league", "season"]):
        wins = (grp["pnl"] > 0).sum()
        n    = len(grp)
        pnl  = grp["pnl"].sum()
        rows.append({
            "league":   league,
            "season":   season,
            "bets":     n,
            "wins":     wins,
            "win_rate": round(wins / n, 3) if n else 0,
            "pnl":      round(pnl, 2),
            "roi_%":    round(pnl / n * 100, 2) if n else 0,
        })

    df = pd.DataFrame(rows).sort_values(["league", "season"]).reset_index(drop=True)
    return df


def main():
    log.info("=" * 70)
    log.info("FULL-LEAGUE BACKTEST  (all leagues in Excel, all seasons)")
    log.info("=" * 70)

    log.info("Loading historical data...")
    raw = load_all_matches()
    log.info(f"  {len(raw):,} matches | {raw['league'].nunique()} leagues | "
             f"{raw['date'].min().date()} → {raw['date'].max().date()}")

    log.info("Engineering features...")
    feat = build_features(raw)
    valid = feat.dropna(subset=["over25", "home_scored_last5"])
    log.info(f"  {len(valid):,} rows with full features")

    # Show what we have per league
    log.info("\nData coverage per league:")
    for league, grp in valid.groupby("league"):
        seasons = sorted(grp["season"].unique())
        log.info(f"  {league:25s}: {len(grp):5,} matches | {len(seasons)} seasons | "
                 f"{grp['date'].min().date()} → {grp['date'].max().date()}")

    def _run_backtest_group(feat_subset, leagues, label, prefix):
        log.info(f"\nRunning walk-forward backtest — {label} ...")
        results_df, summary, league_summary = run_backtest(feat_subset, enabled_leagues=leagues)

        out_file = config.OUTPUT_DIR / f"backtest_all_leagues_{prefix}.csv"
        results_df.to_csv(out_file, index=False)

        bets = results_df[results_df["bet"].isin(["OVER", "UNDER"])].copy()
        if bets.empty:
            return results_df, pd.DataFrame()
        lg = bets.groupby("league").apply(lambda g: pd.Series({
            "bets":     len(g),
            "wins":     (g["pnl"] > 0).sum(),
            "win_rate": round((g["pnl"] > 0).mean(), 3),
            "pnl":      round(g["pnl"].sum(), 2),
            "roi_%":    round(g["pnl"].sum() / len(g) * 100, 2),
            "drawdown": round(g["pnl"].cumsum().sub(g["pnl"].cumsum().cummax()).min(), 2),
        })).reset_index().sort_values("roi_%", ascending=False)

        lg_file = config.OUTPUT_DIR / f"backtest_all_leagues_by_league_{prefix}.csv"
        lg.to_csv(lg_file, index=False)

        print(f"\n{'=' * 70}")
        print(f"  OVERALL SUMMARY — {label}")
        print("=" * 70)
        for k, v in summary.items():
            print(f"  {k:35s}: {v}")
        print(f"\n  BY LEAGUE [{label}] (ranked by ROI):")
        print(lg.to_string(index=False))

        return results_df, lg

    # ── Standard model backtest ────────────────────────────────────────────────
    std_leagues = config.STANDARD_FORMAT_LEAGUES
    std_feat    = valid[valid["league"].isin(std_leagues)]
    std_results, std_lg = _run_backtest_group(std_feat, std_leagues, "STANDARD", "standard")

    # ── New-format model backtest ──────────────────────────────────────────────
    nf_leagues = config.NEW_FORMAT_LEAGUES
    nf_feat    = valid[valid["league"].isin(nf_leagues)]
    nf_results, nf_lg = _run_backtest_group(nf_feat, nf_leagues, "NEW-FORMAT", "newformat")

    results_df = pd.concat([std_results, nf_results], ignore_index=True)
    out_file = config.OUTPUT_DIR / "backtest_all_leagues.csv"
    results_df.to_csv(out_file, index=False)
    log.info(f"\nCombined results → {out_file}")

    lg = pd.concat([std_lg, nf_lg], ignore_index=True).sort_values("roi_%", ascending=False)

    lg_file = config.OUTPUT_DIR / "backtest_all_leagues_by_league.csv"
    lg.to_csv(lg_file, index=False)

    # Season breakdown
    season_df = _season_breakdown(results_df)
    s_file = config.OUTPUT_DIR / "backtest_season_breakdown.csv"
    season_df.to_csv(s_file, index=False)
    log.info(f"Season breakdown → {s_file}")

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")

    print("\n" + "=" * 70)
    print("  BY LEAGUE (ranked by ROI)")
    print("=" * 70)
    print(lg.to_string(index=False))

    print("\n" + "=" * 70)
    print("  BY LEAGUE × SEASON")
    print("=" * 70)
    if not season_df.empty:
        # Pivot for readability
        pivot = season_df.pivot_table(
            index="league", columns="season",
            values="roi_%", aggfunc="first"
        ).round(1)
        print("\n  ROI % per season:")
        print(pivot.fillna("—").to_string())

        print("\n  Win rate per season:")
        wr_pivot = season_df.pivot_table(
            index="league", columns="season",
            values="win_rate", aggfunc="first"
        ).round(3)
        print(wr_pivot.fillna("—").to_string())

        print("\n  Bets placed per season:")
        b_pivot = season_df.pivot_table(
            index="league", columns="season",
            values="bets", aggfunc="first"
        ).astype("Int64")
        print(b_pivot.fillna(0).to_string())

    print("\n" + "=" * 70)
    print("  FULL SEASON DETAIL")
    print("=" * 70)
    print(season_df.to_string(index=False))

    return results_df, season_df


if __name__ == "__main__":
    main()
