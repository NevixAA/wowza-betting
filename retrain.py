"""
retrain.py — End-of-season data refresh + model retraining

Downloads the latest CSVs from football-data.co.uk, retrains BOTH models
(standard and new-format separately), runs two backtests, and prints a
before/after comparison.

Usage
-----
    python retrain.py                  # full refresh (download + train + backtest)
    python retrain.py --no-download    # skip download, just retrain + backtest
    python retrain.py --download-only  # download CSVs only, no training
    python retrain.py --season 2627    # override target season (default: auto)

Season code format: YYMM → "2627" = 2026/27 season.
Auto-detection: month < 7 → current ending season; month >= 7 → new season.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

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

METRICS_FILE = config.OUTPUT_DIR / "backtest_metrics_history.json"
DATA_DIR     = config.DATA_DIR / "data" / "football_data"

# ── League download maps ──────────────────────────────────────────────────────

# Standard mmz4281 format: URL = /mmz4281/{season}/{code}.csv
# Contains full stats: shots, corners, fouls, O/U odds
_STANDARD_LEAGUES = {
    "League One":              "E2",
    "League Two":              "E3",
    "Bundesliga 2":            "D2",
    "Ligue 2":                 "F2",
    "La Liga 2":               "SP2",
    "Championship":            "E1",
    "Serie B":                 "I2",
    "Greek Super League":      "G1",
    "National League":         "EC",
    "Portuguese Primeira Liga": "P1",
    "Scottish Championship":   "SC1",
    "Scottish League One":     "SC2",
    "Scottish League Two":     "SC3",
}

# "New" format: URL = /new/{code}.csv  (multi-season, goals + 1X2 only, no O/U odds)
# season_type: "winter" = YYYY/YYYY format; "year" = integer year
_NEW_FORMAT_LEAGUES = {
    "Denmark Superliga":          {"code": "DNK", "filter": "Superliga",       "season_type": "winter"},
    "Austrian Bundesliga":        {"code": "AUT", "filter": "Bundesliga",       "season_type": "winter"},
    "Romanian Superliga":         {"code": "ROM", "filter": "Superliga",        "season_type": "winter"},
    "Mexico Liga MX":             {"code": "MEX", "filter": "Liga MX",          "season_type": "winter"},
    "Sweden Allsvenskan":         {"code": "SWE", "filter": "Allsvenskan",      "season_type": "year"},
    "Norway Eliteserien":         {"code": "NOR", "filter": "Eliteserien",      "season_type": "year"},
    "Finland Veikkausliiga":      {"code": "FIN", "filter": "Veikkausliiga",    "season_type": "year"},
    "Ireland Premier Division":   {"code": "IRL", "filter": "Premier Division", "season_type": "year"},
    "Argentina Primera Division": {"code": "ARG", "filter": "Liga Profesional", "season_type": "year"},
    "Brazil Serie A":             {"code": "BRA", "filter": "Serie A",          "season_type": "year"},
    "Japan J-League":             {"code": "JPN", "filter": "J1 League",        "season_type": "year"},
    "China Super League":         {"code": "CHN", "filter": "Super League",     "season_type": "year"},
    "USA MLS":                    {"code": "USA", "filter": "MLS",              "season_type": "year"},
}

# Winter → internal season code mapping
_WINTER_MAP = {
    "2019/2020": "2020", "2020/2021": "2021", "2021/2022": "2122",
    "2022/2023": "2223", "2023/2024": "2324", "2024/2025": "2425",
    "2025/2026": "2526", "2026/2027": "2627",
}
# Calendar-year → internal season code mapping (prefix 'y' to avoid clash)
_YEAR_MAP = {
    2020: "y2020", 2021: "y2021", 2022: "y2022", 2023: "y2023",
    2024: "y2024", 2025: "y2025", 2026: "y2026",
}


def _auto_season() -> str:
    today = date.today()
    if today.month >= 7:
        y1, y2 = today.year % 100, (today.year + 1) % 100
    else:
        y1, y2 = (today.year - 1) % 100, today.year % 100
    return f"{y1:02d}{y2:02d}"


# ── Downloaders ───────────────────────────────────────────────────────────────

def download_standard(league: str, code: str, season: str) -> bool:
    url  = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    dest = DATA_DIR / code / f"{code}_{season}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            log.warning(f"  {league} ({code}): HTTP {r.status_code} — skipped")
            return False

        df = pd.read_csv(StringIO(r.text), on_bad_lines="skip", low_memory=False)
        if "HomeTeam" not in df.columns:
            log.warning(f"  {league}: no HomeTeam column — skipped")
            return False
        df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()]
        if len(df) < 5:
            log.warning(f"  {league}: too few rows ({len(df)}) — skipped")
            return False

        df.to_csv(dest, index=False)
        log.info(f"  {league:35s} → {dest.name}  ({len(df)} rows)")
        return True

    except Exception as e:
        log.warning(f"  {league}: {e}")
        return False


def download_new_format(league: str, cfg: dict, season: str) -> bool:
    """Download and split a new-format file into per-season CSVs."""
    code       = cfg["code"]
    flt        = cfg.get("filter")
    season_type = cfg["season_type"]

    url      = f"https://www.football-data.co.uk/new/{code}.csv"
    dest_dir = DATA_DIR / code
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            log.warning(f"  {league} ({code}): HTTP {r.status_code} — skipped")
            return False

        raw = pd.read_csv(StringIO(r.text), encoding="utf-8-sig",
                          on_bad_lines="skip", low_memory=False)

        if flt and "League" in raw.columns:
            raw = raw[raw["League"].astype(str).str.contains(flt, na=False, case=False)]

        if "Season" not in raw.columns:
            log.warning(f"  {league}: no Season column — skipped")
            return False

        saved = []
        if season_type == "winter":
            for season_str, season_code in _WINTER_MAP.items():
                chunk = raw[raw["Season"].astype(str).str.strip() == season_str].copy()
                if chunk.empty:
                    continue
                _save_new_format_chunk(chunk, code, season_code, dest_dir)
                saved.append(season_code)
        else:  # calendar year
            for year_int, season_code in _YEAR_MAP.items():
                # Season column may be int or str
                chunk = raw[raw["Season"].apply(
                    lambda s: str(s).strip().rstrip(".0") == str(year_int)
                )].copy()
                if chunk.empty:
                    continue
                _save_new_format_chunk(chunk, code, season_code, dest_dir)
                saved.append(season_code)

        if saved:
            log.info(f"  {league:35s} → {code}_[{', '.join(saved)}].csv")
        return bool(saved)

    except Exception as e:
        log.warning(f"  {league}: {e}")
        return False


def _save_new_format_chunk(chunk: pd.DataFrame, code: str, season_code: str,
                           dest_dir: Path) -> None:
    """Standardise and save one season chunk."""
    out = pd.DataFrame()
    out["Date"]     = chunk.get("Date", chunk.get("date"))
    out["HomeTeam"] = chunk.get("Home", chunk.get("HomeTeam", "")).astype(str).str.strip()
    out["AwayTeam"] = chunk.get("Away", chunk.get("AwayTeam", "")).astype(str).str.strip()
    out["FTHG"]     = pd.to_numeric(chunk.get("HG", chunk.get("FTHG")), errors="coerce")
    out["FTAG"]     = pd.to_numeric(chunk.get("AG", chunk.get("FTAG")), errors="coerce")
    out = out[out["HomeTeam"].notna() & out["AwayTeam"].notna()]
    out = out[out["FTHG"].notna() & out["FTAG"].notna()]
    dest = dest_dir / f"{code}_{season_code}.csv"
    out.to_csv(dest, index=False)


def download_all(season: str) -> None:
    log.info(f"Downloading season {season} data ...")

    log.info(f"  Standard-format leagues ({len(_STANDARD_LEAGUES)}):")
    for league, code in _STANDARD_LEAGUES.items():
        download_standard(league, code, season)
        time.sleep(0.3)

    log.info(f"  New-format leagues ({len(_NEW_FORMAT_LEAGUES)}):")
    for league, cfg in _NEW_FORMAT_LEAGUES.items():
        download_new_format(league, cfg, season)
        time.sleep(0.3)

    log.info("Download complete.")


# ── Metrics history ───────────────────────────────────────────────────────────

def _load_metrics() -> dict:
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_metrics(history: dict) -> None:
    METRICS_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _baseline(history: dict, track: str, k: int = 6) -> dict | None:
    """The bar a new model must clear: the BEST of the last `k` recorded runs on this track.

    NOT simply the previous run. That distinction is cheap under weekly retraining and load-
    bearing under daily, which is the cadence from 2026-09-22 to 2026-10-05.

    THE RATCHET. `_promotion_gate` allows a drop of up to RETRAIN_MAX_ROI_DROP_PP against its
    baseline. If the baseline is "yesterday", every day may legally lose the full tolerance and
    each step passes on its own: 10.0 -> 8.1 -> 6.2 -> 4.3 clears a 2pp gate four days running
    while shedding 5.7 points. Holding the baseline at the best of a recent window makes the bar
    stop moving down with you, so cumulative drift is measured against where the model actually
    was rather than against its own decline.

    Bounded to `k` runs rather than all time on purpose: football changes, and a model should not
    be held forever against a number produced in a different scoring environment. Six is roughly
    six weeks weekly, or a week daily.
    """
    runs = [v for key, v in sorted(history.items(), reverse=True)
            if key.endswith(f"_{track}") and isinstance(v, dict)]
    recent = runs[:k]
    scored = [r for r in recent if isinstance(r.get("roi_%"), (int, float))]
    if not scored:
        return recent[0] if recent else None
    return max(scored, key=lambda r: r["roi_%"])


def _promotion_gate(label: str, prev: dict | None, curr: dict) -> tuple[bool, str]:
    """May this freshly trained model replace the incumbent? Returns (promote, reason).

    WHY THIS EXISTS. `save_models()` used to run immediately after `train_model()` and BEFORE
    `run_backtest()`, with `_print_comparison()` afterwards only printing. So every retrain went
    live unconditionally and the comparison was decoration — a model that backtested worse
    replaced a better one and nothing stopped it, or even recorded that it had happened.

    Note what the backtest is and is not. `run_backtest` walk-forward-retrains inside each fold,
    so it measures the PROCEDURE on history, not the specific pickle just fitted. That is still
    the right gate input — if the procedure has degraded on history, the artifact it produced is
    not one to ship — but it is not a test of that artifact, and this gate should not be
    described as one.

    Deliberately conservative and deliberately simple. Pro has the real thing
    (`v10/src/models/registry.py::evaluate_gate()` — chronological_4block, logloss/brier, ECE,
    clv_n); v9 is frozen and cannot import it. This is a floor, not a substitute.

    ON UNMEASURABLE CASES: promote, loudly. The instinct is to fail closed, but the cost
    asymmetry runs the other way here. Failing closed on a missing metric key freezes the model
    forever, silently — exactly the class of bug this whole pass is cleaning up. Failing open is
    no worse than the behaviour being replaced, and it says so on stdout.
    """
    if os.getenv("RETRAIN_FORCE_PROMOTE", "").strip() == "1":
        return True, "RETRAIN_FORCE_PROMOTE=1 set — gate bypassed by hand"

    curr_bets = curr.get("total_bets")
    if isinstance(curr_bets, (int, float)) and curr_bets <= 0:
        return False, "the new model placed 0 backtest bets — nothing was measured, so nothing is proven"

    if not prev:
        return True, "no previous backtest on record — nothing to regress against"

    tol = float(os.getenv("RETRAIN_MAX_ROI_DROP_PP", "2.0"))
    p_roi, c_roi = prev.get("roi_%"), curr.get("roi_%")
    if not (isinstance(p_roi, (int, float)) and isinstance(c_roi, (int, float))):
        return True, f"roi_% missing on one side (prev={p_roi!r}, curr={c_roi!r}) — cannot compare"

    drop = p_roi - c_roi
    if drop > tol:
        return False, (f"ROI fell {drop:.2f}pp ({p_roi:.2f} -> {c_roi:.2f}), beyond the "
                       f"{tol:.2f}pp tolerance")
    return True, f"ROI {p_roi:.2f} -> {c_roi:.2f} ({-drop:+.2f}pp), within the {tol:.2f}pp tolerance"


def _print_comparison(label: str, prev: dict | None, curr: dict) -> None:
    keys = [
        ("roi_%",              "ROI % (S+MM placed)"),
        ("sharpe_ratio",       "Sharpe"),
        ("max_drawdown_units", "Max DD (u)"),
        ("win_rate",           "Win Rate (S+MM)"),
        ("total_bets",         "Placed Bets (S+MM)"),
    ]
    tier_keys = [
        ("sniper_bets",    "sniper_roi_%",    "SNIPER"),
        ("marksman_bets",  "marksman_roi_%",  "MARKSMAN"),
        ("valuable_bets",  "valuable_roi_%",  "VALUABLE (info)"),
    ]
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST COMPARISON — {label}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'Previous':>10} {'Current':>10}  {'Δ':>6}")
    print("-" * 60)
    for key, lbl in keys:
        cur = curr.get(key, "—")
        prv = prev.get(key, "—") if prev else "—"
        delta = f"{cur - prv:+.3f}" if isinstance(cur, float) and isinstance(prv, float) else ""
        print(f"  {lbl:<30} {str(prv):>10} {str(cur):>10}  {delta:>6}")
    print("-" * 60)
    print(f"  {'Tier':<14} {'Bets':>6} {'ROI%':>8}   {'Bets':>6} {'ROI%':>8}")
    for bets_key, roi_key, tier_lbl in tier_keys:
        pb = prev.get(bets_key, "—") if prev else "—"
        pr = prev.get(roi_key,  "—") if prev else "—"
        cb = curr.get(bets_key, "—")
        cr = curr.get(roi_key,  "—")
        print(f"  {tier_lbl:<14} {str(pb):>6} {str(pr):>8}   {str(cb):>6} {str(cr):>8}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-of-season retrain + backtest")
    parser.add_argument("--no-download",   action="store_true",
                        help="Skip CSV download")
    parser.add_argument("--download-only", action="store_true",
                        help="Download CSVs only, no training")
    parser.add_argument("--season",        default=None,
                        help="Season code e.g. 2627 (default: auto)")
    args = parser.parse_args()

    season = args.season or _auto_season()
    log.info(f"Target season: {season}")

    # ── 1. Download ────────────────────────────────────────────────────────────
    if not args.no_download:
        download_all(season)
    else:
        log.info("Skipping download (--no-download).")

    if args.download_only:
        log.info("Download-only mode — done.")
        return

    # ── 2. Load + feature engineering ─────────────────────────────────────────
    from src.data_loader import load_all_matches
    from src.feature_engineering import build_features
    from src.model import train as train_model, save_models, load_models, get_feature_importances
    from src.backtest import run_backtest

    raw  = load_all_matches(force=True)
    log.info(f"Loaded {len(raw):,} matches | {raw['league'].nunique()} leagues | "
             f"{raw['date'].min().date()} → {raw['date'].max().date()}")

    # ── REFUSE TO RETRAIN WITHOUT HISTORICAL ODDS ────────────────────────────────
    # A retrain that cannot BACKTEST is more dangerous than no retrain at all: it would happily
    # overwrite the production models and then be unable to validate a single per-league
    # threshold, because src/backtest.py drops every row missing odds_over25/odds_under25.
    #
    # This became reachable on 2026-09-05, when football-data.co.uk returned 503 for many hours
    # and the loader fell back to a cache seeded from API-Football data — which carries results
    # and shots but NO PRICES. Training would have "succeeded" on 28,179 rows and silently
    # shipped a model nobody could measure.
    #
    # football-data is the only source of historical closing odds: API-Football's /odds is
    # pre-match only (proven at ~830 calls — 0 of 3 fixtures in every season 2019-2025). So when
    # the prices are absent the correct action is to STOP and keep yesterday's validated model.
    _price_cols = [c for c in ("odds_over25", "odds_under25") if c in raw.columns]
    _priced = int(raw[_price_cols].notna().all(axis=1).sum()) if len(_price_cols) == 2 else 0
    if _priced < config.BACKTEST_MIN_TRAIN:
        log.error(
            f"ABORTING RETRAIN: only {_priced:,} of {len(raw):,} rows carry historical odds "
            f"(need >= {config.BACKTEST_MIN_TRAIN:,}). The models on disk are LEFT UNTOUCHED.\n"
            f"  Cause is almost certainly football-data.co.uk being unreachable — it is the only "
            f"source of historical closing prices, and the loader falls back to an odds-less "
            f"cache when it is down.\n"
            f"  Retraining now would overwrite a validated model with one that cannot be "
            f"backtested. Re-run once the source returns.")
        return

    feat  = build_features(raw)
    valid = feat.dropna(subset=["over25", "home_scored_last5"])
    log.info(f"{len(valid):,} rows with full features")

    history = _load_metrics()
    run_key = datetime.utcnow().strftime("%Y-%m-%d")

    # ── 3. Train + backtest STANDARD model ────────────────────────────────────
    std_valid = valid[valid["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    log.info(f"\nTraining STANDARD model on {len(std_valid):,} rows ...")
    log.info(f"  Leagues: {sorted(std_valid['league'].unique())}")

    # AN EMPTY TRACK IS A DATA FAULT, NOT A TRAINING TASK. Observed today: the loader fell back
    # to a cache built from af_history.parquet, which is a NEW-FORMAT-only backfill, so not one
    # standard-format league was present and this line read "Training STANDARD model on 0 rows"
    # before dying inside sklearn with an unhelpful error.
    #
    # Refusing here says WHICH track has no data and leaves that model on disk untouched, rather
    # than crashing the whole run — the new-format model can still retrain when the standard one
    # cannot, and vice versa.
    # PER-TRACK, NOT GLOBAL. The earlier guard counted priced rows across the WHOLE frame and
    # passed at 1,430 — but almost all of those are new-format, priced by our own nf captures.
    # The STANDARD subset had 224, and run_backtest needs 460, so training "succeeded" and then
    # died in the backtest. A model that cannot be backtested must not be saved, and the two
    # tracks have completely different price coverage, so the question has to be asked per track.
    _std_priced = int(std_valid[["odds_over25", "odds_under25"]].notna().all(axis=1).sum()) \
        if {"odds_over25", "odds_under25"} <= set(std_valid.columns) else 0
    if len(std_valid) < config.BACKTEST_MIN_TRAIN or _std_priced < (config.BACKTEST_MIN_TRAIN + config.BACKTEST_WALK_SIZE):
        log.error(f"SKIPPING STANDARD model: {len(std_valid):,} rows, of which {_std_priced:,} "
                  f"carry odds (need >= {config.BACKTEST_MIN_TRAIN + config.BACKTEST_WALK_SIZE:,} priced to backtest). "
                  f"{config.MODEL_FILE_STANDARD.name} is LEFT UNTOUCHED.\n"
                  f"  Standard-format historical PRICES come from football-data.co.uk, which is "
                  f"unreachable; our own forward captures cover the new-format leagues far better "
                  f"than the second divisions. The new-format model below still retrains.")
        std_results = None
    else:
        # TRAIN INTO MEMORY ONLY. The save now happens after the backtest, behind the gate.
        std_results = train_model(std_valid)

    std_leagues = config.STANDARD_FORMAT_LEAGUES & config.ENABLED_LEAGUES
    if std_results is None:
        # SKIP THE STANDARD BACKTEST, DO NOT RETURN. This was `return`, and it returned from
        # retrain() entirely — so a standard track with no prices did not merely skip its own
        # model, it silently took the NEW-FORMAT model down with it, along with the side-market
        # backtests below. Sections 4+ are dead code whenever football-data is unreachable.
        #
        # That is why "retrain doesn't work" was the whole run and not one track: the new-format
        # data was complete the entire time (priced by our own captures) and never got the chance
        # to train. The two tracks are independent by invariant 1 and must fail independently.
        log.error("SKIPPING STANDARD backtest: the model was not retrained this run. "
                  "Continuing to the NEW-FORMAT track, which has its own data and its own "
                  "price coverage.")
    else:
        std_df, std_summary, std_lg = run_backtest(std_valid, enabled_leagues=std_leagues)
        std_df.to_csv(config.OUTPUT_DIR / "backtest_results_standard.csv", index=False)
        std_lg.to_csv(config.OUTPUT_DIR / "backtest_by_league_standard.csv", index=False)

        print("\n" + "=" * 60)
        print("  BACKTEST — STANDARD MODEL")
        print("=" * 60)
        for k, v in std_summary.items():
            print(f"  {k:35s}: {v}")
        print("\n  BY LEAGUE:"); print(std_lg.to_string(index=False))

        prev_std = history.get(f"{run_key}_standard") or \
                   next((v for k, v in sorted(history.items(), reverse=True)
                         if "_standard" in k), None)
        _print_comparison("STANDARD", prev_std, std_summary)

        # ── THE GATE ──────────────────────────────────────────────────────────
        promote, why = _promotion_gate("STANDARD", _baseline(history, "standard"), std_summary)
        if promote:
            save_models(std_results, model_file=config.MODEL_FILE_STANDARD)
            log.info(f"STANDARD model PROMOTED — {why}")
        else:
            log.error(f"STANDARD model NOT PROMOTED — {why}. "
                      f"{config.MODEL_FILE_STANDARD.name} is LEFT UNTOUCHED; the incumbent "
                      f"keeps serving. Set RETRAIN_FORCE_PROMOTE=1 to override by hand.")

        # Feature importances describe whatever is ACTUALLY live, which after a blocked
        # promotion is the incumbent, not what was just fitted. Reading the file rather than
        # `std_results` is what keeps those two from drifting apart.
        try:
            fi = get_feature_importances(load_models(model_file=config.MODEL_FILE_STANDARD))
            if not fi.empty:
                fi.to_csv(config.MODELS_DIR / "feature_importances_standard.csv", index=False)
                print("\nTop 10 features [STANDARD — the LIVE model]:")
                print(fi.head(10).to_string(index=False))
        except FileNotFoundError:
            log.warning(f"No {config.MODEL_FILE_STANDARD.name} on disk — skipping "
                        f"feature importances.")

        history[f"{run_key}_standard"] = {**std_summary, "season": season,
                                          "promoted": promote, "gate_reason": why}

    # ── 4. Train + backtest NEW-FORMAT model ──────────────────────────────────
    nf_valid = valid[valid["league"].isin(config.NEW_FORMAT_LEAGUES)]
    log.info(f"\nTraining NEW-FORMAT model on {len(nf_valid):,} rows ...")
    log.info(f"  Leagues: {sorted(nf_valid['league'].unique())}")

    if len(nf_valid) >= config.BACKTEST_MIN_TRAIN:
        # TRAIN INTO MEMORY ONLY — save is gated below, same as the standard track. The two
        # gates are SEPARATE and neither can block or promote the other (invariant 1).
        nf_results = train_model(nf_valid)

        nf_leagues = config.NEW_FORMAT_LEAGUES & config.ENABLED_LEAGUES
        nf_df, nf_summary, nf_lg = run_backtest(nf_valid, enabled_leagues=nf_leagues)
        nf_df.to_csv(config.OUTPUT_DIR / "backtest_results_newformat.csv", index=False)
        nf_lg.to_csv(config.OUTPUT_DIR / "backtest_by_league_newformat.csv", index=False)

        print("\n" + "=" * 60)
        print("  BACKTEST — NEW-FORMAT MODEL")
        print("=" * 60)
        for k, v in nf_summary.items():
            print(f"  {k:35s}: {v}")
        print("\n  BY LEAGUE:"); print(nf_lg.to_string(index=False))

        prev_nf = history.get(f"{run_key}_newformat") or \
                  next((v for k, v in sorted(history.items(), reverse=True)
                        if "_newformat" in k), None)
        _print_comparison("NEW-FORMAT", prev_nf, nf_summary)

        # ── THE GATE (new-format's own, independent of standard's) ────────────
        nf_promote, nf_why = _promotion_gate("NEW-FORMAT", _baseline(history, "newformat"),
                                             nf_summary)
        if nf_promote:
            save_models(nf_results, model_file=config.MODEL_FILE_NEWFORMAT)
            log.info(f"NEW-FORMAT model PROMOTED — {nf_why}")
        else:
            log.error(f"NEW-FORMAT model NOT PROMOTED — {nf_why}. "
                      f"{config.MODEL_FILE_NEWFORMAT.name} is LEFT UNTOUCHED; the incumbent "
                      f"keeps serving. Set RETRAIN_FORCE_PROMOTE=1 to override by hand.")

        try:
            fi_nf = get_feature_importances(load_models(model_file=config.MODEL_FILE_NEWFORMAT))
            if not fi_nf.empty:
                fi_nf.to_csv(config.MODELS_DIR / "feature_importances_newformat.csv",
                             index=False)
                print("\nTop 10 features [NEW-FORMAT — the LIVE model]:")
                print(fi_nf.head(10).to_string(index=False))
        except FileNotFoundError:
            log.warning(f"No {config.MODEL_FILE_NEWFORMAT.name} on disk — skipping "
                        f"feature importances.")

        history[f"{run_key}_newformat"] = {**nf_summary, "season": season,
                                           "promoted": nf_promote, "gate_reason": nf_why}
    else:
        log.warning(f"Not enough new-format data ({len(nf_valid)} rows < "
                    f"{config.BACKTEST_MIN_TRAIN}) — new-format model not trained")

    # ── 5. Save metrics ────────────────────────────────────────────────────────
    _save_metrics(history)
    log.info(f"Metrics saved → {METRICS_FILE}")

    print(f"\n  Retrain complete.")
    print(f"  Standard model → {config.MODEL_FILE_STANDARD}")
    print(f"  New-format model → {config.MODEL_FILE_NEWFORMAT}")
    print(f"  Run 'python pipeline.py --mode predict' for new tips.\n")


if __name__ == "__main__":
    main()
