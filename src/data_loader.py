"""
Historical match data loader.

Reads every raw match sheet from the summary XLSX workbook and returns
a clean, consolidated DataFrame sorted by date.

Sheet naming convention: {League}_{YYMM}  e.g. "Championship_2526"

Output columns
──────────────
date, league, season, home_team, away_team
home_goals, away_goals, total_goals, ftr
over25, btts
home_corners, away_corners, home_fouls, away_fouls
home_shots, away_shots, home_sot, away_sot
home_sot_ratio, away_sot_ratio
odds_over25, odds_under25
referee
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import logging

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

_SEASON_MAP = {
    # European winter seasons (YYMM code → readable label)
    "2526": "2025/26", "2425": "2024/25", "2324": "2023/24",
    "2223": "2022/23", "2122": "2021/22", "2021": "2020/21",
    "2020": "2019/20",
    # Calendar-year seasons for summer leagues (Brazil, Norway, Finland, etc.)
    # Files saved as CODE_y2025.csv to avoid conflict with winter "2021"→"2020/21"
    "y2026": "2026", "y2025": "2025", "y2024": "2024",
    "y2023": "2023", "y2022": "2022", "y2021": "2021", "y2020": "2020",
}

# Preferred odds columns in order (first found with data wins)
_OVER_COLS  = ["AvgC>2.5", "MaxC>2.5", "B365C>2.5", "PC>2.5",
               "Avg>2.5",  "Max>2.5",  "B365>2.5",  "P>2.5"]
_UNDER_COLS = ["AvgC<2.5", "MaxC<2.5", "B365C<2.5", "PC<2.5",
               "Avg<2.5",  "Max<2.5",  "B365<2.5",  "P<2.5"]

# Module-level cache so we only parse the XLSX once per process
_CACHE: Optional[pd.DataFrame] = None


def _pick_odds(raw: pd.DataFrame, cols: list[str]) -> pd.Series:
    for col in cols:
        if col in raw.columns:
            s = pd.to_numeric(raw[col], errors="coerce")
            if s.notna().sum() > len(raw) * 0.1:   # at least 10% coverage
                return s
    return pd.Series(np.nan, index=raw.index)


def load_all_matches(xlsx_path: Optional[Path] = None, force: bool = False) -> pd.DataFrame:
    """Load every match sheet from the summary workbook into one clean DataFrame."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    path = xlsx_path or config.SUMMARY_XLSX
    xl   = pd.ExcelFile(path)

    frames = []
    for sheet in xl.sheet_names:
        m = re.match(r"^(.+)_(\d{4})$", sheet)
        if not m:
            continue
        league, season_code = m.group(1), m.group(2)
        if season_code not in _SEASON_MAP:
            continue

        try:
            raw = xl.parse(sheet)
        except Exception:
            continue

        df = pd.DataFrame()
        df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
        df = df[df["date"].notna()].copy()

        df["league"]    = league
        df["season"]    = _SEASON_MAP[season_code]
        df["home_team"] = raw.get("HomeTeam", np.nan)
        df["away_team"] = raw.get("AwayTeam", np.nan)

        for out_col, src_col in [
            ("home_goals",   "FTHG"),
            ("away_goals",   "FTAG"),
            ("home_corners", "HC"),
            ("away_corners", "AC"),
            ("home_fouls",   "HF"),
            ("away_fouls",   "AF"),
            ("home_shots",   "HS"),
            ("away_shots",   "AS"),
            ("home_sot",     "HST"),
            ("away_sot",     "AST"),
        ]:
            df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")

        df["ftr"]      = raw.get("FTR", np.nan)
        df["referee"]  = raw.get("Referee", np.nan)
        df["odds_over25"]  = _pick_odds(raw, _OVER_COLS)
        df["odds_under25"] = _pick_odds(raw, _UNDER_COLS)

        frames.append(df)

    # ── CSV overrides: prefer downloaded CSVs over Excel sheets ─────────────────
    csv_dir = path.parent / "data" / "football_data"
    if csv_dir.exists():
        for league_name, fd_code in {
            # Standard mmz4281 format — downloaded by retrain.py
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
            "Scottish Premiership":    "SC0",
            "Belgian First Division A": "B1",
            "Dutch Eredivisie":        "N1",
            "Turkish Super Lig":       "T1",
            # New format — downloaded by retrain.py
            "Denmark Superliga":            "DNK",
            "Austrian Bundesliga":          "AUT",
            "Romanian Superliga":           "ROM",
            "Sweden Allsvenskan":           "SWE",
            "Norway Eliteserien":           "NOR",
            "Finland Veikkausliiga":        "FIN",
            "Ireland Premier Division":     "IRL",
            "Argentina Primera Division":   "ARG",
            "Brazil Serie A":               "BRA",
            "Japan J-League":               "JPN",
            "Mexico Liga MX":               "MEX",
            "China Super League":           "CHN",
            "USA MLS":                      "USA",
            "Swiss Super League":           "CHE",
            "Poland Ekstraklasa":           "PL1",
        }.items():
            code_dir = csv_dir / fd_code
            if not code_dir.exists():
                continue
            for csv_path in sorted(code_dir.glob(f"{fd_code}_*.csv")):
                season_code = csv_path.stem.split("_")[-1]
                if season_code not in _SEASON_MAP:
                    continue
                try:
                    raw = pd.read_csv(csv_path, low_memory=False)
                    df = pd.DataFrame()
                    df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
                    df = df[df["date"].notna()].copy()
                    df["league"]    = league_name
                    df["season"]    = _SEASON_MAP[season_code]
                    df["home_team"] = raw.get("HomeTeam", np.nan)
                    df["away_team"] = raw.get("AwayTeam", np.nan)
                    for out_col, src_col in [
                        ("home_goals","FTHG"),("away_goals","FTAG"),
                        ("home_corners","HC"),("away_corners","AC"),
                        ("home_fouls","HF"),("away_fouls","AF"),
                        ("home_shots","HS"),("away_shots","AS"),
                        ("home_sot","HST"),("away_sot","AST"),
                    ]:
                        df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
                    df["ftr"]           = raw.get("FTR", np.nan)
                    df["referee"]       = raw.get("Referee", np.nan)
                    df["odds_over25"]   = _pick_odds(raw, _OVER_COLS)
                    df["odds_under25"]  = _pick_odds(raw, _UNDER_COLS)
                    df = df[df["home_team"].notna() & df["away_team"].notna()]
                    # Replace the matching Excel frame with this clean CSV data
                    frames = [f for f in frames if not (
                        (f["league"] == league_name).all() and
                        (f["season"] == _SEASON_MAP[season_code]).all()
                    )]
                    frames.append(df)
                    log.debug(f"CSV override: {league_name} {_SEASON_MAP[season_code]} ({len(df)} rows)")
                except Exception as e:
                    log.warning(f"Failed to load CSV override {csv_path}: {e}")

    if not frames:
        raise RuntimeError(f"No match sheets found in {path}")

    out = pd.concat(frames, ignore_index=True)
    out = out[out["home_team"].notna() & out["away_team"].notna()]
    out = out.sort_values("date").reset_index(drop=True)

    # Derived columns
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["over25"]      = (out["total_goals"] > 2.5).astype(float)
    out["btts"]        = (
        (out["home_goals"] > 0) & (out["away_goals"] > 0)
    ).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["home_sot_ratio"] = np.where(
            out["home_shots"] > 0,
            out["home_sot"] / out["home_shots"],
            np.nan,
        )
        out["away_sot_ratio"] = np.where(
            out["away_shots"] > 0,
            out["away_sot"] / out["away_shots"],
            np.nan,
        )

    # ── Sanity check: flag German teams appearing in Danish league slots ─────────
    _GERMAN_TEAMS = {
        "Bayern Munich", "Dortmund", "Leverkusen", "RB Leipzig", "Ein Frankfurt",
        "Wolfsburg", "Freiburg", "Hoffenheim", "Stuttgart", "Union Berlin",
        "Augsburg", "Mainz", "Hertha", "St Pauli", "FC Koln", "Heidenheim",
        "Werder Bremen", "M'gladbach",
    }
    for dk_league in ("Denmark Superliga", "Denmark 1st Div"):
        dk = out[out["league"] == dk_league]
        if dk.empty:
            continue
        all_teams = set(dk["home_team"].dropna()) | set(dk["away_team"].dropna())
        contaminated = all_teams & _GERMAN_TEAMS
        if contaminated:
            import warnings
            warnings.warn(
                f"DATA CONTAMINATION: {dk_league} contains German teams: {contaminated}. "
                f"Run: python download_league_data.py",
                stacklevel=2,
            )

    _CACHE = out
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_all_matches()
    print(f"Loaded {len(df):,} matches across {df['league'].nunique()} leagues")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Over 2.5 rate: {df['over25'].mean():.1%}")
    print(df[["date","league","home_team","away_team","home_goals","away_goals","over25"]].tail(8).to_string())
