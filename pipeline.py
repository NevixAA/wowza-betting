"""
v9 Football Prediction Pipeline — Hybrid v7 + v8
==================================================
Entry point for all pipeline modes.

Usage
-----
    python pipeline.py --mode train
    python pipeline.py --mode predict
    python pipeline.py --mode backtest
    python pipeline.py --mode all

Modes
-----
    train    : load data, engineer features, train + calibrate models, save
    predict  : load trained models, fetch upcoming fixtures, output bets
    backtest : run strict walk-forward backtest, output results + summary
    all      : train → predict → backtest

v9 additions over v8
---------------------
    - Odds drift tracking  : every predict run snapshots odds → drift signal
                             adjusts SNIPER/VALUE tiers at output time
    - Both-losing guard    : leagues flagged by v7 optimizer are set to AVOID
    - Postponement detect  : fixtures missing since last run are flagged
    - 3-tier output        : SNIPER (>=0.10) / VALUE (0.04-0.10) / AVOID (<0.04)
"""
from __future__ import annotations

import argparse
import logging
import sys
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

import config
from src.data_loader import load_all_matches
from src.feature_engineering import build_features
from src.model import train as train_model, save_models, load_models, get_feature_importances
from src.betting import generate_bets
from src.backtest import run_backtest
from src.ledger import append_tips, print_ledger


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def _train_one(valid: "pd.DataFrame", label: str, model_file) -> dict:
    """Train one model (standard or new-format) and save it."""
    log.info(f"  [{label}] {len(valid):,} rows — training ensemble ...")
    results = train_model(valid)
    save_models(results, model_file=model_file)

    payload = load_models(model_file=model_file)
    fi = get_feature_importances(payload)
    if not fi.empty:
        fi_file = config.MODELS_DIR / f"feature_importances_{label}.csv"
        fi.to_csv(fi_file, index=False)
        print(f"\nTop 10 features [{label}]:")
        print(fi.head(10).to_string(index=False))
    return results


def mode_train() -> tuple:
    log.info("=" * 60)
    log.info("MODE: TRAIN  (standard model + new-format model — separate)")
    log.info("=" * 60)

    log.info("Loading historical data...")
    raw = load_all_matches()
    log.info(f"  {len(raw):,} matches | {raw['league'].nunique()} leagues | "
             f"{raw['date'].min().date()} → {raw['date'].max().date()}")

    log.info("Engineering features...")
    feat  = build_features(raw)
    valid = feat.dropna(subset=["over25", "home_scored_last5"])
    log.info(f"  {len(valid):,} rows with full features")

    # ── Standard model — trained only on standard-format league data ────────
    log.info("\nTraining STANDARD model ...")
    std_valid = valid[valid["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    log.info(f"  Standard leagues: {sorted(std_valid['league'].unique())}")
    _train_one(std_valid, "standard", config.MODEL_FILE_STANDARD)

    # ── New-format model — trained only on new-format league data ───────────
    log.info("\nTraining NEW-FORMAT model ...")
    nf_valid = valid[valid["league"].isin(config.NEW_FORMAT_LEAGUES)]
    log.info(f"  New-format leagues: {sorted(nf_valid['league'].unique())}")
    if len(nf_valid) >= config.BACKTEST_MIN_TRAIN:
        _train_one(nf_valid, "newformat", config.MODEL_FILE_NEWFORMAT)
    else:
        log.warning(f"  Not enough new-format data ({len(nf_valid)} rows) — skipping new-format model")

    return feat, None


# ── PREDICT ───────────────────────────────────────────────────────────────────

def mode_predict(historical: "pd.DataFrame" = None) -> "pd.DataFrame":
    log.info("=" * 60)
    log.info("MODE: PREDICT  (v9 — drift + both-losing + postponement)")
    log.info("=" * 60)

    import pandas as pd
    from src.predict import predict_upcoming

    if historical is None:
        historical = load_all_matches()

    payload_std = load_models(model_file=config.MODEL_FILE_STANDARD)
    payload_nf  = load_models(model_file=config.MODEL_FILE_NEWFORMAT) \
                  if config.MODEL_FILE_NEWFORMAT.exists() else None

    preds, postponed = predict_upcoming(
        historical, payload_std,
        payload_newformat=payload_nf,
        days_ahead=7,
    )

    if preds is None or preds.empty:
        log.warning("No predictions generated.")
        return preds

    # ── Save full predictions ────────────────────────────────────────────────
    pred_file = config.OUTPUT_DIR / "predictions.csv"
    preds.to_csv(pred_file, index=False)
    log.info(f"Predictions saved → {pred_file}  ({len(preds)} fixtures)")

    # ── Save postponed list ──────────────────────────────────────────────────
    if postponed:
        pst_file = config.OUTPUT_DIR / "postponed.csv"
        pd.DataFrame(postponed).to_csv(pst_file, index=False)
        log.info(f"Postponed fixtures → {pst_file}  ({len(postponed)} matches)")

    # ── Generate bets (SNIPER + VALUE) ───────────────────────────────────────
    bets = generate_bets(preds)

    if not bets.empty:
        bets_file = config.OUTPUT_DIR / "bets.csv"
        bets.to_csv(bets_file, index=False)
        log.info(f"Tips saved → {bets_file}  ({len(bets)} tips)")

        # Append to persistent ledger (never overwrites, deduped per fixture)
        append_tips(bets)

        display = [
            "date", "league", "home_team", "away_team",
            "best_side", "odds_over25", "odds_under25",
            "best_edge", "bet", "signal_tier",
            "drift_signal", "over_drift", "under_drift",
        ]
        display = [c for c in display if c in bets.columns]

        for tier, label, note in [
            ("SNIPER", "SNIPER  (edge >= 0.10)", "bet full stake"),
            ("VALUE",  "VALUE   (edge 0.04-0.10)", "bet half stake / monitor"),
        ]:
            rows = bets[bets["signal_tier"] == tier]
            print("\n" + "=" * 90)
            print(f"  {label}  — {note}  [{len(rows)} tips]")
            print("=" * 90)
            if rows.empty:
                print("  (none this week)")
            else:
                print(rows[display].to_string(index=False))

        # Both-losing summary
        bl_count = int(preds.get("both_losing", pd.Series(False)).sum()) if "both_losing" in preds.columns else 0

        print(f"\n  SNIPER      : {(bets['signal_tier'] == 'SNIPER').sum()}")
        print(f"  VALUE       : {(bets['signal_tier'] == 'VALUE').sum()}")
        print(f"  AVOID       : {len(preds) - len(bets)} fixtures below threshold")
        if bl_count:
            print(f"  BOTH-LOSING : {bl_count} fixtures suppressed (v7 guard active)")
        if postponed:
            print(f"  POSTPONED   : {len(postponed)} fixtures disappeared since last run")

        # Drift summary
        if "drift_signal" in bets.columns:
            print(f"\n  Drift breakdown (tips only):")
            print(bets["drift_signal"].value_counts().to_string())
    else:
        print("\n  No tips this week (all fixtures below 0.04 edge).")
        if postponed:
            print(f"  {len(postponed)} fixture(s) postponed since last run.")

    return preds


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def _run_one_backtest(feat: "pd.DataFrame", leagues: set, label: str, out_prefix: str) -> tuple:
    results_df, summary, league_summary = run_backtest(feat, enabled_leagues=leagues)

    bt_file = config.OUTPUT_DIR / f"backtest_results_{out_prefix}.csv"
    results_df.to_csv(bt_file, index=False)
    lg_file = config.OUTPUT_DIR / f"backtest_by_league_{out_prefix}.csv"
    league_summary.to_csv(lg_file, index=False)

    print("\n" + "=" * 60)
    print(f"  BACKTEST SUMMARY — {label}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")
    print(f"\n  BY LEAGUE [{label}]:")
    print(league_summary.to_string(index=False))

    return results_df, summary


def mode_backtest(feat: "pd.DataFrame" = None) -> tuple:
    log.info("=" * 60)
    log.info("MODE: BACKTEST  (walk-forward, separate standard / new-format)")
    log.info("=" * 60)

    if feat is None:
        raw  = load_all_matches()
        feat = build_features(raw)

    std_leagues = config.STANDARD_FORMAT_LEAGUES & config.ENABLED_LEAGUES
    nf_leagues  = config.NEW_FORMAT_LEAGUES  & config.ENABLED_LEAGUES

    log.info(f"Standard leagues: {sorted(std_leagues)}")
    std_results, std_summary = _run_one_backtest(feat, std_leagues, "STANDARD", "standard")

    log.info(f"New-format leagues: {sorted(nf_leagues)}")
    nf_results, nf_summary = _run_one_backtest(feat, nf_leagues, "NEW-FORMAT", "newformat")

    # Also save a combined file for the full ledger view
    import pandas as pd
    combined = pd.concat([std_results, nf_results], ignore_index=True)
    combined.to_csv(config.OUTPUT_DIR / "backtest_results.csv", index=False)

    return std_results, std_summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v9 Football Prediction Pipeline (hybrid v7+v8)")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "backtest", "all"],
        default="predict",
        help="Pipeline mode (default: predict)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        mode_train()
    elif args.mode == "predict":
        mode_predict()
    elif args.mode == "backtest":
        mode_backtest()
    elif args.mode == "all":
        feat, _ = mode_train()
        mode_predict(load_all_matches())
        mode_backtest(feat)



if __name__ == "__main__":
    main()
