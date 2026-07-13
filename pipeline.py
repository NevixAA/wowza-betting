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

import pandas as pd

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
from src.backtest import run_backtest, run_side_market_backtest, optimize_side_market_thresholds
from src.ledger import append_tips, print_ledger


# ── Side-market bet generation ────────────────────────────────────────────────

def _load_side_market_thresholds() -> dict:
    """Load per-league thresholds from best_params_side_markets.json if it exists."""
    import json
    th_file = config.MODELS_DIR / "best_params_side_markets.json"
    if th_file.exists():
        with open(th_file) as f:
            return json.load(f)
    return {}


def _generate_side_bets(preds: "pd.DataFrame", side_markets: dict) -> "pd.DataFrame":
    """
    Generate SNIPER/MARKSMAN/VALUABLE tips for BTTS / Over 1.5 / Over 3.5.
    Uses per-league optimized thresholds from best_params_side_markets.json when
    available, falling back to fixed global thresholds.
    Leagues marked 'drop=True' in the optimizer output are suppressed entirely.
    """
    import pandas as pd
    import numpy as np
    frames = []
    OVERROUND    = 1.08
    league_params = _load_side_market_thresholds()  # {target: {league: {sniper_th, marksman_th, drop}}}

    for target in side_markets:
        prob_col = f"p_{target}"
        odds_col = f"odds_{target}"
        if prob_col not in preds.columns or odds_col not in preds.columns:
            continue
        df = preds[[
            "date", "league", "home_team", "away_team", prob_col, odds_col
        ]].dropna(subset=[prob_col, odds_col]).copy()
        if df.empty:
            continue

        market_params = league_params.get(target, {})

        df["model_prob"]  = df[prob_col]
        df["market_odds"] = df[odds_col]
        df["fair_prob"]   = (1.0 / df["market_odds"]) / OVERROUND
        df["edge"]        = df["model_prob"] - df["fair_prob"]
        df["ev"]          = df["model_prob"] * df["market_odds"] - 1.0
        df["market"]      = target

        def _tier(row):
            lg = row["league"]
            lp = market_params.get(lg, {})
            # Drop leagues the optimizer flagged as unprofitable
            if lp.get("drop", False):
                return "AVOID"
            sniper_th   = lp.get("sniper_th",   0.10)
            marksman_th = lp.get("marksman_th",  0.08)
            edge = row["edge"]
            if edge >= sniper_th:   return "SNIPER"
            if edge >= marksman_th: return "MARKSMAN"
            if edge >= 0.04:        return "VALUABLE"
            return "AVOID"

        df["signal_tier"] = df.apply(_tier, axis=1)
        df = df[df["signal_tier"] != "AVOID"]
        df = df.sort_values("edge", ascending=False)
        frames.append(df[["date", "league", "home_team", "away_team",
                           "market", "market_odds", "model_prob",
                           "fair_prob", "edge", "ev", "signal_tier"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("edge", ascending=False)


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def _train_one(valid: "pd.DataFrame", label: str, model_file,
               target: str = "over25", weights=None) -> dict:
    """Train one model and save it. target specifies which column to predict."""
    log.info(f"  [{label}] {len(valid):,} rows — training ensemble (target={target})...")
    results = train_model(valid, target=target, sample_weight=weights)
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

    # ── Exclude COVID seasons ────────────────────────────────────────────────
    if config.EXCLUDE_COVID_SEASONS:
        before = len(raw)
        raw = raw[~raw["season"].isin(config.COVID_SEASONS)]
        log.info(f"  Excluded COVID seasons {config.COVID_SEASONS}: "
                 f"{before - len(raw):,} rows removed → {len(raw):,} remaining")

    log.info("Engineering features...")
    feat  = build_features(raw)
    valid = feat.dropna(subset=["over25", "home_scored_last5"])
    log.info(f"  {len(valid):,} rows with full features")

    # ── Build time-decay sample weights ─────────────────────────────────────
    def _get_weights(df):
        weights = df["season"].map(
            lambda s: config.TRAINING_DECAY_WEIGHTS.get(s, config.DEFAULT_DECAY_WEIGHT)
        ).fillna(config.DEFAULT_DECAY_WEIGHT)
        return weights.values

    # ── Standard model — trained only on standard-format league data ────────
    log.info("\nTraining STANDARD model ...")
    std_valid = valid[valid["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    std_weights = _get_weights(std_valid)
    log.info(f"  Standard leagues: {sorted(std_valid['league'].unique())}")
    log.info(f"  Seasons in training: {sorted(std_valid['season'].unique())}")
    _train_one(std_valid, "standard", config.MODEL_FILE_STANDARD, weights=std_weights)

    # ── New-format model — trained only on new-format league data ───────────
    log.info("\nTraining NEW-FORMAT model ...")
    nf_valid = valid[valid["league"].isin(config.NEW_FORMAT_LEAGUES)]
    nf_weights = _get_weights(nf_valid)
    log.info(f"  New-format leagues: {sorted(nf_valid['league'].unique())}")
    if len(nf_valid) >= config.BACKTEST_MIN_TRAIN:
        _train_one(nf_valid, "newformat", config.MODEL_FILE_NEWFORMAT, weights=nf_weights)
    else:
        log.warning(f"  Not enough new-format data ({len(nf_valid)} rows) — skipping new-format model")

    # ── HT models — all leagues with HT data (not just std_valid) ───────────
    log.info("\nTraining HT models (all leagues with HTHG/HTAG data)...")
    ht_valid = valid.dropna(subset=["ht_over05", "home_ht_over05_rate"])
    log.info(f"  HT leagues: {sorted(ht_valid['league'].unique())}")
    if len(ht_valid) >= config.BACKTEST_MIN_TRAIN:
        log.info(f"  HT data: {len(ht_valid):,} rows with HT scores")
        _train_one(ht_valid, "ht_over05", config.HT_MODEL_FILE_05, target="ht_over05")
        ht_valid15 = valid.dropna(subset=["ht_over15", "home_ht_over15_rate"])
        _train_one(ht_valid15, "ht_over15", config.HT_MODEL_FILE_15, target="ht_over15")
    else:
        log.warning(f"  Not enough HT data ({len(ht_valid)} rows) — skipping HT models")

    # ── Side-market models (BTTS / O1.5 / O3.5) — all leagues with odds data ──
    # nf_valid rows now have odds_btts/over15/over35 when af_odds_history.parquet
    # has been backfilled; rows without odds simply drop out via dropna().
    log.info("\nTraining side-market models (BTTS / Over 1.5 / Over 3.5)...")
    _sm_all = pd.concat([std_valid, nf_valid], ignore_index=True)
    for target, model_file in config.SIDE_MARKETS.items():
        sm_valid = _sm_all.dropna(subset=[target])
        if len(sm_valid) >= config.BACKTEST_MIN_TRAIN:
            log.info(f"  [{target}] {len(sm_valid):,} rows  "
                     f"base rate={sm_valid[target].mean():.1%}")
            _train_one(sm_valid, target, model_file,
                       target=target, weights=_get_weights(sm_valid))
        else:
            log.warning(f"  [{target}] Not enough data ({len(sm_valid)}) — skipping")

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

    # Side-market payloads (BTTS / O1.5 / O3.5) — optional, skip if not trained yet
    side_payloads = {}
    for target, model_file in config.SIDE_MARKETS.items():
        if model_file.exists():
            side_payloads[target] = load_models(model_file=model_file)

    preds, postponed = predict_upcoming(
        historical, payload_std,
        payload_newformat=payload_nf,
        side_payloads=side_payloads,
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

    # ── Side-market tips (BTTS / O1.5 / O3.5) ───────────────────────────────
    side_bets = _generate_side_bets(preds, config.SIDE_MARKETS)
    if not side_bets.empty:
        side_bets_file = config.OUTPUT_DIR / "side_bets.csv"
        side_bets.to_csv(side_bets_file, index=False)
        log.info(f"Side-market tips → {side_bets_file}  ({len(side_bets)} tips)")
        for mkt, label in config.SIDE_MARKET_LABELS.items():
            n = (side_bets["market"] == mkt).sum()
            log.info(f"  {label}: {n} tips")

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
            ("SNIPER",   "SNIPER    (per-league threshold)", "full stake"),
            ("MARKSMAN", "MARKSMAN  (edge 8%-threshold)",   "3/4 stake"),
            ("VALUABLE", "VALUABLE  (edge 4-8%)",           "half stake / monitor"),
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


def mode_backtest(feat: "pd.DataFrame" = None, only: str = None) -> tuple:
    """Walk-forward backtest, standard + new-format computed SEPARATELY (already isolated).
    only=None      -> both (+ side markets) — default, unchanged (monthly cron behaviour).
    only='standard'-> standard 2.5 + side markets only (new-format skipped).
    only='newformat'-> new-format only (standard + side markets skipped; standard results
                       CSV is NOT regenerated -> zero effect on the standard model path)."""
    log.info("=" * 60)
    log.info(f"MODE: BACKTEST  (walk-forward; only={only or 'all'})")
    log.info("=" * 60)

    if feat is None:
        raw  = load_all_matches()
        feat = build_features(raw)

    std_leagues = config.STANDARD_FORMAT_LEAGUES & config.ENABLED_LEAGUES
    nf_leagues  = config.NEW_FORMAT_LEAGUES  & config.ENABLED_LEAGUES

    std_results = std_summary = nf_results = nf_summary = None
    if only in (None, "standard"):
        log.info(f"Standard leagues: {sorted(std_leagues)}")
        std_results, std_summary = _run_one_backtest(feat, std_leagues, "STANDARD", "standard")
    if only in (None, "newformat"):
        log.info(f"New-format leagues: {sorted(nf_leagues)}")
        nf_results, nf_summary = _run_one_backtest(feat, nf_leagues, "NEW-FORMAT", "newformat")

    # Combined ledger file only when BOTH ran (don't clobber it on a per-model run)
    if std_results is not None and nf_results is not None:
        import pandas as pd
        combined = pd.concat([std_results, nf_results], ignore_index=True)
        combined.to_csv(config.OUTPUT_DIR / "backtest_results.csv", index=False)

    # Single-model run (only='standard' or 'newformat'): skip the side-market backtests
    # entirely — they're a separate concern, run them alone via `--mode backtest-side --market X`.
    # This is what makes per-model runs fast (no side-market walk-forwards bundled in).
    if only is not None:
        return (std_results, std_summary) if only == "standard" else (nf_results, nf_summary)

    # ── Side-market backtests (BTTS / Over 1.5 / Over 3.5) ──────────────────
    log.info("\nBacktesting side markets (BTTS / Over 1.5 / Over 3.5)...")
    import json
    all_side_thresholds = {}  # {target: {league: {sniper_th, marksman_th, roi, bets, drop}}}

    for target, label in config.SIDE_MARKET_LABELS.items():
        try:
            sm_results, sm_summary, sm_league = run_side_market_backtest(
                feat, target=target, enabled_leagues=std_leagues
            )
            bt_file = config.OUTPUT_DIR / f"backtest_results_{target}.csv"
            sm_results.to_csv(bt_file, index=False)
            lg_file = config.OUTPUT_DIR / f"backtest_by_league_{target}.csv"
            sm_league.to_csv(lg_file, index=False)

            print(f"\n{'=' * 60}")
            print(f"  BACKTEST — {label}")
            print(f"{'=' * 60}")
            for k, v in sm_summary.items():
                print(f"  {k:35s}: {v}")
            if not sm_league.empty:
                print(f"\n  BY LEAGUE [{label}]:")
                print(sm_league.to_string(index=False))

            # ── Per-league threshold optimization ────────────────────────────
            log.info(f"\nOptimizing per-league thresholds for {label}...")
            league_ths = optimize_side_market_thresholds(sm_results, target=target)
            all_side_thresholds[target] = league_ths

            print(f"\n  PER-LEAGUE THRESHOLDS [{label}]:")
            print(f"  {'League':<30} {'SNIPER th':>10} {'MARKSMAN th':>12} {'ROI%':>8} {'Bets':>6} {'Drop':>6}")
            print(f"  {'-'*72}")
            for lg, v in sorted(league_ths.items()):
                roi_str = f"{v['roi']:.1f}%" if v['roi'] is not None else "—"
                print(f"  {lg:<30} {v['sniper_th']:>10.2f} {v['marksman_th']:>12.2f} "
                      f"{roi_str:>8} {v['bets']:>6} {'DROP' if v['drop'] else '':>6}")

        except Exception as e:
            log.warning(f"  [{target}] Backtest failed: {e}")

    # Save all side-market thresholds to JSON
    if all_side_thresholds:
        th_file = config.MODELS_DIR / "best_params_side_markets.json"
        with open(th_file, "w") as f:
            json.dump(all_side_thresholds, f, indent=2)
        log.info(f"Side-market thresholds saved → {th_file}")

    return std_results, std_summary


# ── Entry point ───────────────────────────────────────────────────────────────

def mode_backtest_side(feat: "pd.DataFrame" = None, market: str = None) -> None:
    """Run ONLY the side-market backtests (BTTS / Over 1.5 / Over 3.5).
    Skips the standard 2.5 walk-forward backtest entirely.
    market: if set, run only that one market (used for parallel dispatch).
    """
    import json, subprocess, sys
    log.info("=" * 60)
    log.info("MODE: BACKTEST-SIDE  (BTTS / Over 1.5 / Over 3.5 only)")
    log.info("=" * 60)

    # If no specific market requested, spawn 3 parallel subprocesses (3× faster)
    if market is None:
        procs = []
        for t in config.SIDE_MARKET_LABELS:
            p = subprocess.Popen(
                [sys.executable, __file__, "--mode", "backtest-side", "--market", t],
                cwd=Path(__file__).parent,
            )
            procs.append((t, p))
            log.info(f"Spawned backtest-side --market {t} (PID {p.pid})")
        for t, p in procs:
            p.wait()
            log.info(f"[{t}] subprocess finished (exit {p.returncode})")
        # Merge per-market JSON files into one combined best_params_side_markets.json
        all_side_thresholds = {}
        for t in config.SIDE_MARKET_LABELS:
            f = config.MODELS_DIR / f"best_params_{t}.json"
            if f.exists():
                all_side_thresholds[t] = json.loads(f.read_text())
        if all_side_thresholds:
            th_file = config.MODELS_DIR / "best_params_side_markets.json"
            th_file.write_text(json.dumps(all_side_thresholds, indent=2))
            log.info(f"Combined thresholds saved → {th_file}")
        return

    if feat is None:
        raw  = load_all_matches()
        feat = build_features(raw)

    std_leagues = config.STANDARD_FORMAT_LEAGUES & config.ENABLED_LEAGUES
    all_side_thresholds = {}
    markets_to_run = {market: config.SIDE_MARKET_LABELS[market]} if market else config.SIDE_MARKET_LABELS

    for target, label in markets_to_run.items():
        try:
            sm_results, sm_summary, sm_league = run_side_market_backtest(
                feat, target=target, enabled_leagues=std_leagues
            )
            (config.OUTPUT_DIR / f"backtest_results_{target}.csv").parent.mkdir(exist_ok=True)
            sm_results.to_csv(config.OUTPUT_DIR / f"backtest_results_{target}.csv", index=False)
            sm_league.to_csv(config.OUTPUT_DIR / f"backtest_by_league_{target}.csv", index=False)

            print(f"\n{'=' * 60}")
            print(f"  BACKTEST — {label}")
            print(f"{'=' * 60}")
            for k, v in sm_summary.items():
                print(f"  {k:35s}: {v}")
            if not sm_league.empty:
                print(f"\n  BY LEAGUE [{label}]:")
                print(sm_league.to_string(index=False))

            log.info(f"\nOptimizing per-league thresholds for {label}...")
            league_ths = optimize_side_market_thresholds(sm_results, target=target)
            all_side_thresholds[target] = league_ths

            print(f"\n  PER-LEAGUE THRESHOLDS [{label}]:")
            print(f"  {'League':<30} {'SNIPER th':>10} {'MARKSMAN th':>12} {'ROI%':>8} {'Bets':>6} {'Drop':>6}")
            print(f"  {'-'*72}")
            for lg, v in sorted(league_ths.items()):
                roi_str = f"{v['roi']:.1f}%" if v['roi'] is not None else "—"
                print(f"  {lg:<30} {v['sniper_th']:>10.2f} {v['marksman_th']:>12.2f} "
                      f"{roi_str:>8} {v['bets']:>6} {'DROP' if v['drop'] else '':>6}")

        except Exception as e:
            log.warning(f"  [{target}] Backtest failed: {e}")

    if all_side_thresholds:
        # Save per-market file (used by parallel dispatcher to merge)
        if market:
            m_file = config.MODELS_DIR / f"best_params_{market}.json"
            m_file.write_text(json.dumps(all_side_thresholds.get(market, {}), indent=2))
        else:
            th_file = config.MODELS_DIR / "best_params_side_markets.json"
            with open(th_file, "w") as f:
                json.dump(all_side_thresholds, f, indent=2)
            log.info(f"Side-market thresholds saved → {th_file}")


def main():
    parser = argparse.ArgumentParser(description="v9 Football Prediction Pipeline (hybrid v7+v8)")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "backtest", "backtest-side", "all"],
        default="predict",
        help="Pipeline mode (default: predict)",
    )
    parser.add_argument(
        "--market",
        choices=["btts", "over15", "over35"],
        default=None,
        help="Run backtest-side for a single market only (used internally for parallel dispatch)",
    )
    parser.add_argument(
        "--only",
        choices=["standard", "newformat"],
        default=None,
        help="Backtest ONE model family only (isolation): 'newformat' skips standard entirely "
             "(standard results not regenerated); 'standard' skips new-format. Default: both.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        mode_train()
    elif args.mode == "predict":
        mode_predict()
    elif args.mode == "backtest":
        mode_backtest(only=args.only)
    elif args.mode == "backtest-side":
        mode_backtest_side(market=args.market)
    elif args.mode == "all":
        feat, _ = mode_train()
        mode_predict(load_all_matches())
        mode_backtest(feat)



if __name__ == "__main__":
    main()
