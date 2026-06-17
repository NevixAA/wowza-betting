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

    feat  = build_features(raw)
    valid = feat.dropna(subset=["over25", "home_scored_last5"])
    log.info(f"{len(valid):,} rows with full features")

    history = _load_metrics()
    run_key = datetime.utcnow().strftime("%Y-%m-%d")

    # ── 3. Train + backtest STANDARD model ────────────────────────────────────
    std_valid = valid[valid["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    log.info(f"\nTraining STANDARD model on {len(std_valid):,} rows ...")
    log.info(f"  Leagues: {sorted(std_valid['league'].unique())}")

    std_results = train_model(std_valid)
    save_models(std_results, model_file=config.MODEL_FILE_STANDARD)

    payload_std = load_models(model_file=config.MODEL_FILE_STANDARD)
    fi = get_feature_importances(payload_std)
    if not fi.empty:
        fi.to_csv(config.MODELS_DIR / "feature_importances_standard.csv", index=False)
        print("\nTop 10 features [STANDARD]:")
        print(fi.head(10).to_string(index=False))

    std_leagues = config.STANDARD_FORMAT_LEAGUES & config.ENABLED_LEAGUES
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
    history[f"{run_key}_standard"] = {**std_summary, "season": season}

    # ── 4. Train + backtest NEW-FORMAT model ──────────────────────────────────
    nf_valid = valid[valid["league"].isin(config.NEW_FORMAT_LEAGUES)]
    log.info(f"\nTraining NEW-FORMAT model on {len(nf_valid):,} rows ...")
    log.info(f"  Leagues: {sorted(nf_valid['league'].unique())}")

    if len(nf_valid) >= config.BACKTEST_MIN_TRAIN:
        nf_results = train_model(nf_valid)
        save_models(nf_results, model_file=config.MODEL_FILE_NEWFORMAT)

        payload_nf = load_models(model_file=config.MODEL_FILE_NEWFORMAT)
        fi_nf = get_feature_importances(payload_nf)
        if not fi_nf.empty:
            fi_nf.to_csv(config.MODELS_DIR / "feature_importances_newformat.csv", index=False)
            print("\nTop 10 features [NEW-FORMAT]:")
            print(fi_nf.head(10).to_string(index=False))

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
        history[f"{run_key}_newformat"] = {**nf_summary, "season": season}
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
