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

# Side-market odds (not available in all leagues/seasons — NaN where absent)
_BTTS_COLS  = ["AvgBTSH", "MaxBTSH", "BbAvBTSH", "BBTSH"]
_OVER15_COLS = ["Avg>1.5", "Max>1.5", "BbAv>1.5", "B365>1.5"]
_OVER35_COLS = ["Avg>3.5", "Max>3.5", "BbAv>3.5", "B365>3.5"]

# Real historical O/U + BTTS odds for new-format leagues, backfilled from The Odds API.
# Replaces the bug where new-format odds_over25/under25 were mapped to 1X2 home/away odds.
from pathlib import Path as _Path
_NF_ODDS_FILE = _Path(__file__).resolve().parents[1] / "output" / "newformat_odds_history.csv"
_nf_odds_cache = None
def _nf_real_odds() -> dict:
    """Lookup {(YYYY-MM-DD, league, 'Home vs Away'): {market: odds}} from the backfill."""
    global _nf_odds_cache
    if _nf_odds_cache is None:
        _nf_odds_cache = {}
        if _NF_ODDS_FILE.exists():
            try:
                _o = pd.read_csv(_NF_ODDS_FILE)
                # ── DROP FIRST-HALF BTTS MISLABELLED AS FULL-MATCH (found 2026-08-23) ──────
                #
                # capture_nf_odds_forward.py carried the identical substring-matching defect as
                # the standard capture: `"Both Teams Score" in name` also matches bet 34, "Both
                # Teams Score - First Half" (Yes ~5.50), and the parse loop assigned on every
                # match so the last one won. Both parsers are fixed, so no NEW contamination
                # arrives.
                #
                # THE NEW-FORMAT DAMAGE IS FAR WORSE THAN THE STANDARD TRACK'S and was missed on
                # the first pass because only the standard file was measured:
                #     standard_sidemarket_odds_history.csv    271 / 6,358 btts_yes  =  4.3%
                #     newformat_odds_history.csv              931 / 3,778 btts_yes  = 24.6%
                #
                # A quarter of new-format BTTS training prices were first-half quotes. Left in,
                # they teach the model that BTTS pays ~6.0 in leagues where it pays ~1.9 — which
                # is not noise, it is a systematic bias toward a fictional longshot edge.
                #
                # Threshold and evidence: see _enrich_with_standard_sidemarket_odds. The two
                # populations do not overlap, so this cannot discard a genuine price.
                # DROPPED FROM THIS READ ONLY — the raw capture file is left untouched so the
                # contamination stays auditable.
                _bad = ((_o["market"].astype(str) == "btts_yes")
                        & (pd.to_numeric(_o["odds"], errors="coerce") > 3.20))
                if _bad.any():
                    log.warning(f"[nf_odds] dropped {int(_bad.sum())} btts_yes row(s) with odds "
                                f"> 3.20 — first-half BTTS mislabelled as full-match "
                                f"(2026-08-23 parser fix). Raw file unchanged.")
                    _o = _o[~_bad]
                for _r in _o.itertuples(index=False):
                    _nf_odds_cache.setdefault(
                        (str(_r.snapshot_date), str(_r.league), str(_r.match)), {}
                    )[str(_r.market)] = float(_r.odds)
            except Exception:
                pass
    return _nf_odds_cache

# Module-level cache so we only parse the XLSX once per process
_CACHE: Optional[pd.DataFrame] = None


def _pick_odds(raw: pd.DataFrame, cols: list[str]) -> pd.Series:
    for col in cols:
        if col in raw.columns:
            s = pd.to_numeric(raw[col], errors="coerce")
            if s.notna().sum() > len(raw) * 0.1:   # at least 10% coverage
                return s
    return pd.Series(np.nan, index=raw.index)


def _ci_download_all() -> list[pd.DataFrame]:
    """
    CI / GitHub Actions fallback: download current + recent seasons directly
    from football-data.co.uk when local Excel/CSV files are not available.
    Returns list of DataFrames (one per league/season).
    """
    import requests
    from io import StringIO
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    STD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"
    SEASONS = ["2526", "2425", "2324", "2223"]  # last 4 seasons

    std_leagues = {
        "League One": "E2", "League Two": "E3", "Championship": "E1",
        "Bundesliga 2": "D2", "La Liga 2": "SP2", "Ligue 2": "F2",
        "Serie B": "I2", "Greek Super League": "G1",
        "Belgian First Division A": "B1", "Dutch Eredivisie": "N1",
        "Turkish Super Lig": "T1", "Scottish Premiership": "SC0",
        "Scottish Championship": "SC1", "Portuguese Primeira Liga": "P1",
        "National League": "EC",
    }
    new_leagues = {
        "Denmark Superliga": "DNK", "Austrian Bundesliga": "AUT",
        "Sweden Allsvenskan": "SWE", "Norway Eliteserien": "NOR",
        "Finland Veikkausliiga": "FIN", "Ireland Premier Division": "IRL",
        "Argentina Primera Division": "ARG", "Brazil Serie A": "BRA",
        "Japan J-League": "JPN", "Mexico Liga MX": "MEX",
        "China Super League": "CHN", "USA MLS": "USA",
    }

    frames = []

    # Standard format (multiple seasons)
    for league, code in std_leagues.items():
        for season in SEASONS:
            season_label = f"20{season[:2]}/{season[2:]}"
            url = STD_URL.format(season=season, code=code)
            try:
                r = requests.get(url, timeout=15, headers=HEADERS)
                if r.status_code != 200:
                    continue
                raw = pd.read_csv(StringIO(r.text), encoding="utf-8-sig",
                                  on_bad_lines="skip", low_memory=False)
                df = pd.DataFrame()
                df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
                df = df[df["date"].notna()].copy()
                if df.empty:
                    continue
                df["league"] = league
                df["season"] = season_label
                df["home_team"] = raw.get("HomeTeam", np.nan)
                df["away_team"] = raw.get("AwayTeam", np.nan)
                for out_col, src_col in [
                    ("home_goals","FTHG"),("away_goals","FTAG"),
                    ("ht_home_goals","HTHG"),("ht_away_goals","HTAG"),
                    ("home_corners","HC"),("away_corners","AC"),
                    ("home_fouls","HF"),("away_fouls","AF"),
                    ("home_shots","HS"),("away_shots","AS"),
                    ("home_sot","HST"),("away_sot","AST"),
                ]:
                    df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
                df["odds_over25"]  = _pick_odds(raw, _OVER_COLS)
                df["odds_under25"] = _pick_odds(raw, _UNDER_COLS)
                df["odds_btts"]    = _pick_odds(raw, _BTTS_COLS)
                df["odds_over15"]  = _pick_odds(raw, _OVER15_COLS)
                df["odds_over35"]  = _pick_odds(raw, _OVER35_COLS)
                df["ftr"] = raw.get("FTR", np.nan)
                df = df[df["home_team"].notna() & df["away_team"].notna()]
                frames.append(df)
            except Exception as e:
                # Silently skipping here means a league-season quietly vanishes from TRAINING
                # data and the model is refit on less history with nothing to show for it.
                log.warning(f"[data] {league} {season_label} skipped ({type(e).__name__}: {e})")
                continue

    # New format (single file, all seasons)
    for league, code in new_leagues.items():
        url = NEW_URL.format(code=code)
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            if r.status_code != 200:
                continue
            raw = pd.read_csv(StringIO(r.text), encoding="utf-8-sig",
                              on_bad_lines="skip", low_memory=False)
            if "Season" in raw.columns:
                raw = raw[raw["Season"].astype(str).str.contains(
                    "2022|2023|2024|2025|2026")].copy()
            df = pd.DataFrame()
            df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
            df = df[df["date"].notna()].copy()
            if df.empty:
                continue
            df["league"] = league
            df["season"] = raw.get("Season", "2025/26").values[:len(df)] if "Season" in raw.columns else "2025/26"
            df["home_team"] = raw.get("Home", np.nan)
            df["away_team"] = raw.get("Away", np.nan)
            for out_col, src_col in [
                ("home_goals","HG"),("away_goals","AG"),
            ]:
                df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
            # Real O/U + BTTS odds from the historical backfill (NOT 1X2 — that was the bug).
            _nfo   = _nf_real_odds()
            _days  = df["date"].dt.strftime("%Y-%m-%d")
            _mkeys = df["home_team"].astype(str) + " vs " + df["away_team"].astype(str)
            _rws   = [_nfo.get((d, league, m), {}) for d, m in zip(_days, _mkeys)]
            df["odds_over25"]  = [r.get("over25",   np.nan) for r in _rws]
            df["odds_under25"] = [r.get("under25",  np.nan) for r in _rws]
            df["odds_over15"]  = [r.get("over15",   np.nan) for r in _rws]
            df["odds_over35"]  = [r.get("over35",   np.nan) for r in _rws]
            df["odds_btts"]    = [r.get("btts_yes", np.nan) for r in _rws]
            df = df[df["home_team"].notna() & df["away_team"].notna()]
            frames.append(df)
        except Exception as e:
            # Same trap as the standard loop above: a new-format league dropping out of
            # training used to be completely invisible.
            log.warning(f"[data] {league} (new-format) skipped ({type(e).__name__}: {e})")
            continue

    log.info(f"CI download: {len(frames)} league/season files loaded")
    return frames


def _load_api_football_only_leagues() -> list[pd.DataFrame]:
    """
    Load historical fixture results for leagues not on football-data.co.uk.
    Uses API-Football `/fixtures` endpoint, permanently cached.
    Returns DataFrames with the same columns as the FD loader output.
    """
    import os as _os
    api_key = _os.getenv("APIFOOTBALL_KEY", "") or config.API_KEY
    if not api_key:
        return []

    if not hasattr(config, "API_FOOTBALL_ONLY_LEAGUES"):
        return []

    try:
        from src.api_football_ou import fetch_league_fixture_results
    except ImportError:
        try:
            from api_football_ou import fetch_league_fixture_results
        except ImportError:
            return []

    frames = []
    for league in config.API_FOOTBALL_ONLY_LEAGUES:
        league_id = config.API_FOOTBALL_IDS.get(league)
        if not league_id:
            continue
        seasons = config.API_FOOTBALL_EXTRA_SEASONS.get(league, [])
        for season_year in seasons:
            try:
                df = fetch_league_fixture_results(league_id, season_year, league)
                if df.empty:
                    continue
                df["league"] = league
                df["season"] = season_year
                df["ftr"] = np.where(
                    df["home_goals"] > df["away_goals"], "H",
                    np.where(df["home_goals"] < df["away_goals"], "A", "D"),
                )
                for col in ["home_shots", "away_shots", "home_sot", "away_sot",
                            "home_corners", "away_corners", "home_fouls", "away_fouls",
                            "odds_over25", "odds_under25",
                            "odds_btts", "odds_over15", "odds_over35"]:
                    if col not in df.columns:
                        df[col] = np.nan
                frames.append(df)
            except Exception as e:
                log.warning(f"[api_football] {league} {season_year}: {e}")

    if frames:
        log.info(f"[api_football] Loaded {sum(len(f) for f in frames)} rows "
                 f"from {len(frames)} API-Football-only league/seasons")
        # Merge historical BTTS/O1.5/O3.5 odds backfill into new-format rows
        combined = pd.concat(frames, ignore_index=True)
        combined = _enrich_with_af_odds(combined)
        return [combined]
    return frames


def _enrich_with_af_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill odds_btts / odds_over15 / odds_over35 for new-format leagues using
    the API-Football historical odds backfill (output/af_odds_history.parquet).
    Only fills NaN cells — never overwrites existing odds.
    """
    af_odds_path = Path(__file__).resolve().parents[1] / "output" / "af_odds_history.parquet"
    if not af_odds_path.exists():
        return df

    try:
        from src.api_football_ou import _norm_name
    except ImportError:
        _norm_name = lambda x: str(x).lower().strip()

    try:
        odds_hist = pd.read_parquet(af_odds_path)
        odds_hist["date"] = pd.to_datetime(odds_hist["date"], errors="coerce")
        odds_hist["_h"] = odds_hist["home_team"].apply(_norm_name)
        odds_hist["_a"] = odds_hist["away_team"].apply(_norm_name)
        odds_hist["_d"] = odds_hist["date"].dt.strftime("%Y-%m-%d")

        df["_h"] = df["home_team"].apply(_norm_name)
        df["_a"] = df["away_team"].apply(_norm_name)
        df["_d"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        lookup = odds_hist.set_index(["_h", "_a", "_d"])[
            ["odds_btts", "odds_over15", "odds_over35"]
        ]

        idx = pd.MultiIndex.from_arrays([df["_h"], df["_a"], df["_d"]])
        matched = lookup.reindex(idx).reset_index(drop=True)

        for col in ["odds_btts", "odds_over15", "odds_over35"]:
            mask = df[col].isna() & matched[col].notna()
            df.loc[mask, col] = matched.loc[mask, col].values
            if mask.any():
                log.info(f"  [af_odds] {col}: filled {mask.sum()} rows from af_odds_history")

        df = df.drop(columns=["_h", "_a", "_d"])
    except Exception as e:
        log.warning(f"[af_odds] Could not merge odds history: {e}")

    return df


def _enrich_with_standard_sidemarket_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Fill odds_btts / odds_over15 / odds_over35 for STANDARD 2nd-division leagues from the
    real backfill output/standard_sidemarket_odds_history.csv. football-data.co.uk carries no
    BTTS/O1.5/O3.5 columns for the 2nd-divs, so those were synthetic (a flat ~1.85) — this plugs
    in REAL prices so the BTTS / side-market backtests reflect what a book would actually pay.
    NaN-only fill; NEVER touches odds_over25 — the standard O/U money-market stays 100%
    football-data, so the O/U backtest is provably unchanged. [v10 2026-07-13]"""
    path = Path(__file__).resolve().parents[1] / "output" / "standard_sidemarket_odds_history.csv"
    if not path.exists():
        return df
    try:
        from src.api_football_ou import _norm_name
    except ImportError:
        _norm_name = lambda x: str(x).lower().strip()
    try:
        _MKT = {"btts_yes": "odds_btts", "over15": "odds_over15", "over35": "odds_over35"}
        oh = pd.read_csv(path)
        oh = oh[oh["market"].isin(_MKT)].copy()
        if oh.empty:
            return df

        # ── DROP FIRST-HALF BTTS PRICES MISLABELLED AS FULL-MATCH (found 2026-08-23) ──────
        #
        # The capture parser matched the bet NAME by substring, so "Both Teams Score - First
        # Half" (bet 34, Yes ~5.50) overwrote "Both Teams Score" (bet 8, Yes ~1.91) whenever
        # Bet365 offered both. The parser is fixed, so no NEW contamination arrives — but 271 of
        # 6,358 historical btts_yes rows (4.3%, 2026-08-06 to 08-23) are first-half prices, and
        # `aggfunc="last"` below would feed them straight into odds_btts for training and the
        # BTTS backtest. A 6.0 price where the truth is 1.9 does not look like an outlier to any
        # downstream check; it looks like a longshot with a huge edge.
        #
        # THE TWO POPULATIONS DO NOT OVERLAP, so this filter cannot catch a real price:
        #     clean full-match btts_yes   1.34 - 2.97
        #     contaminated (first half)   3.40 - 7.00
        #     observations in 0.31-0.33 implied (odds 3.03-3.22):  ZERO
        # 3.20 sits inside that empty gap. A full-match BTTS-YES at 3.20 implies a 31% chance
        # both teams score across 90 minutes, which does not occur in professional football.
        #
        # Rows are DROPPED FROM THIS READ ONLY. Nothing is deleted from the CSV — the raw
        # capture stays intact so the contamination remains auditable.
        _BTTS_YES_MAX = 3.20
        _bad = (oh["market"] == "btts_yes") & (pd.to_numeric(oh["odds"], errors="coerce")
                                               > _BTTS_YES_MAX)
        if _bad.any():
            log.warning(f"  [std_sidemarket] dropped {int(_bad.sum())} btts_yes row(s) with "
                        f"odds > {_BTTS_YES_MAX} — first-half BTTS mislabelled as full-match "
                        f"(see 2026-08-23 parser fix). Raw file unchanged.")
            oh = oh[~_bad].copy()
        if oh.empty:
            return df
        parts = oh["match"].astype(str).str.split(" vs ", n=1, expand=True)
        oh["_h"] = parts[0].apply(_norm_name)
        oh["_a"] = (parts[1] if parts.shape[1] > 1 else "").apply(_norm_name)
        oh["_d"] = oh["match_date"].astype(str).str[:10]
        oh["_col"] = oh["market"].map(_MKT)
        piv = oh.pivot_table(index=["_h", "_a", "_d"], columns="_col", values="odds", aggfunc="last")

        df["_h"] = df["home_team"].apply(_norm_name)
        df["_a"] = df["away_team"].apply(_norm_name)
        df["_d"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        idx = pd.MultiIndex.from_arrays([df["_h"], df["_a"], df["_d"]])
        matched = piv.reindex(idx).reset_index(drop=True)
        for col in ("odds_btts", "odds_over15", "odds_over35"):   # NOTE: over25 deliberately excluded
            if col in matched.columns and col in df.columns:
                mask = df[col].isna() & matched[col].notna()
                if mask.any():
                    df.loc[mask, col] = matched.loc[mask, col].values
                    log.info(f"  [std_sidemarket] {col}: filled {int(mask.sum())} rows from real backfill")
        df = df.drop(columns=["_h", "_a", "_d"])
    except Exception as e:
        log.warning(f"[std_sidemarket] merge failed: {e}")
    return df


def _enrich_with_api_shots(df: pd.DataFrame) -> pd.DataFrame:
    """
    For new-format leagues that have no shot data from football-data.co.uk,
    fill home_shots / away_shots / home_sot / away_sot from API-Football.
    Skips silently when API_KEY is not set or API returns nothing.
    """
    import os as _os
    api_key = _os.getenv("APIFOOTBALL_KEY", "") or config.API_KEY

    try:
        from src.api_football_ou import _norm_name
    except ImportError:
        _norm_name = lambda x: str(x).lower().strip()

    # Pre-initialize xG/insidebox so merge suffixes work correctly for new columns
    for _col in ["home_xg", "away_xg", "home_insidebox", "away_insidebox"]:
        if _col not in df.columns:
            df[_col] = np.nan

    # ── Step 1: merge from af_history.parquet (historical backfill, one-time run) ──
    af_history_path = Path(__file__).resolve().parents[1] / "output" / "af_history.parquet"
    if af_history_path.exists():
        try:
            hist = pd.read_parquet(af_history_path)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist["_home_norm"] = hist["home_team"].apply(_norm_name)
            hist["_away_norm"] = hist["away_team"].apply(_norm_name)
            hist["_date_str"]  = hist["date"].dt.strftime("%Y-%m-%d")

            # Map af_history column names → internal names
            col_map = {
                "HS": "home_shots", "AS": "away_shots",
                "HST": "home_sot",  "AST": "away_sot",
                "HC": "home_corners", "AC": "away_corners",
                "HF": "home_fouls",   "AF": "away_fouls",
                "HY": "home_yellows", "AY": "away_yellows",
            }
            hist = hist.rename(columns=col_map)
            hist_cols = [c for c in col_map.values() if c in hist.columns]

            df["_home_norm"] = df["home_team"].apply(_norm_name)
            df["_away_norm"] = df["away_team"].apply(_norm_name)
            df["_date_str"]  = df["date"].dt.strftime("%Y-%m-%d")

            merged = df.merge(
                hist[["_home_norm", "_away_norm", "_date_str", "league"] + hist_cols]
                    .rename(columns={c: f"{c}_h" for c in hist_cols}),
                on=["_home_norm", "_away_norm", "_date_str"],
                how="left",
                suffixes=("", "_h"),
            )

            for col in hist_cols:
                hcol = f"{col}_h"
                if hcol not in merged.columns:
                    continue
                if col not in df.columns:
                    df[col] = np.nan
                orig = df[col].values
                hist_vals = merged[hcol].values
                df[col] = [float(h) if pd.isna(o) and pd.notna(h) else o
                           for o, h in zip(orig, hist_vals)]

            df.drop(columns=["_home_norm", "_away_norm", "_date_str"], errors="ignore", inplace=True)
            n_with_shots = int(df["home_shots"].notna().sum()) if "home_shots" in df.columns else 0
            log.info(f"[af_history] merged backfill: {n_with_shots} rows now have shot data")
        except Exception as e:
            log.warning(f"[af_history] failed to merge af_history.parquet: {e}")

    # ── Step 1b: merge HT scores from af_ht_history.parquet ──────────────────
    af_ht_path = Path(__file__).resolve().parents[1] / "output" / "af_ht_history.parquet"
    if af_ht_path.exists():
        try:
            ht_hist = pd.read_parquet(af_ht_path)
            ht_hist["date"] = pd.to_datetime(ht_hist["date"], errors="coerce")
            ht_hist["_home_norm"] = ht_hist["home_team"].apply(_norm_name)
            ht_hist["_away_norm"] = ht_hist["away_team"].apply(_norm_name)
            ht_hist["_date_str"]  = ht_hist["date"].dt.strftime("%Y-%m-%d")

            if "_home_norm" not in df.columns:
                df["_home_norm"] = df["home_team"].apply(_norm_name)
                df["_away_norm"] = df["away_team"].apply(_norm_name)
                df["_date_str"]  = df["date"].dt.strftime("%Y-%m-%d")

            ht_merged = df.merge(
                ht_hist[["_home_norm", "_away_norm", "_date_str", "HTHG", "HTAG"]]
                    .rename(columns={"HTHG": "HTHG_h", "HTAG": "HTAG_h"}),
                on=["_home_norm", "_away_norm", "_date_str"],
                how="left",
            )
            for src, dst in [("HTHG_h", "ht_home_goals"), ("HTAG_h", "ht_away_goals")]:
                if src not in ht_merged.columns:
                    continue
                if dst not in df.columns:
                    df[dst] = np.nan
                orig = df[dst].values
                fill = ht_merged[src].values
                df[dst] = [float(h) if pd.isna(o) and pd.notna(h) else o
                           for o, h in zip(orig, fill)]

            df.drop(columns=["_home_norm", "_away_norm", "_date_str"], errors="ignore", inplace=True)
            n_ht = int(df["ht_home_goals"].notna().sum()) if "ht_home_goals" in df.columns else 0
            log.info(f"[af_ht_history] merged HT scores: {n_ht} rows now have halftime data")
        except Exception as e:
            log.warning(f"[af_ht_history] failed to merge af_ht_history.parquet: {e}")

    if not api_key:
        return df

    try:
        from src.api_football_ou import fetch_shots_for_league, _norm_name
    except ImportError:
        return df

    for league in config.NEW_FORMAT_LEAGUES:
        league_id = config.API_FOOTBALL_IDS.get(league)
        if not league_id:
            continue

        mask = df["league"] == league
        if not mask.any():
            continue

        # Skip only if xG is already populated (shots from CSV is fine, but xG always comes from API)
        if "home_xg" in df.columns and df.loc[mask, "home_xg"].notna().mean() > 0.5:
            continue

        season = config.API_FOOTBALL_SEASONS.get(league, "2025")
        # Also try the previous calendar year — many leagues span two seasons
        # (e.g. Brazil 2026 is in-progress but 2025 is fully completed in FD)
        prev_season = str(int(season) - 1)
        all_shots = []
        for s in [season, prev_season]:
            try:
                s_df = fetch_shots_for_league(league_id, s, league)
                if not s_df.empty:
                    s_df["_date_str"] = s_df["date"].dt.strftime("%Y-%m-%d")
                    all_shots.append(s_df)
            except Exception as e:
                log.warning(f"[api_football_ou] {league} season {s} shots fetch failed: {e}")
        if not all_shots:
            continue
        shots_df = (
            pd.concat(all_shots, ignore_index=True)
            .drop_duplicates(subset=["_home_norm", "_away_norm", "_date_str"])
            if len(all_shots) > 1 else all_shots[0]
        )

        if shots_df.empty:
            continue

        # Build normalised keys for the main df rows belonging to this league
        league_rows = df[mask].copy()
        league_rows["_home_norm"] = league_rows["home_team"].apply(_norm_name)
        league_rows["_away_norm"] = league_rows["away_team"].apply(_norm_name)
        league_rows["_date_str"]  = league_rows["date"].dt.strftime("%Y-%m-%d")

        api_cols = ["home_shots", "away_shots", "home_sot", "away_sot",
                    "home_corners", "away_corners", "home_fouls", "away_fouls",
                    "home_xg", "away_xg", "home_insidebox", "away_insidebox"]
        avail_cols = [c for c in api_cols if c in shots_df.columns]
        merged = league_rows.merge(
            shots_df[["_home_norm", "_away_norm", "_date_str"] + avail_cols],
            on=["_home_norm", "_away_norm", "_date_str"],
            how="left",
            suffixes=("", "_api"),
        )

        idx = df.index[mask]
        for col in avail_cols:
            api_col = f"{col}_api"
            if api_col not in merged.columns:
                continue
            orig = df.loc[idx, col].values
            api  = merged[api_col].values
            filled = [float(a) if pd.isna(o) and pd.notna(a) else o
                      for o, a in zip(orig, api)]
            df.loc[idx, col] = filled

        n_filled = int(df.loc[mask, "home_shots"].notna().sum())
        log.info(f"[api_football_ou] {league}: {n_filled}/{int(mask.sum())} rows now have shot data")

    # Recompute SOT ratios after enrichment
    with np.errstate(divide="ignore", invalid="ignore"):
        df["home_sot_ratio"] = np.where(
            df["home_shots"] > 0, df["home_sot"] / df["home_shots"], np.nan
        )
        df["away_sot_ratio"] = np.where(
            df["away_shots"] > 0, df["away_sot"] / df["away_shots"], np.nan
        )

    return df


def load_all_matches(xlsx_path: Optional[Path] = None, force: bool = False) -> pd.DataFrame:
    """Load every match sheet from the summary workbook into one clean DataFrame."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    path = xlsx_path or config.SUMMARY_XLSX

    # ── CI / GitHub Actions fallback ─────────────────────────────────────────
    if not path.exists():
        log.info("Local Excel not found — downloading data from football-data.co.uk (CI mode)...")
        ci_frames = _ci_download_all()
        if not ci_frames:
            raise RuntimeError("CI download failed — no data loaded")
        ci_frames.extend(_load_api_football_only_leagues())
        out = pd.concat(ci_frames, ignore_index=True)
        out = out[out["home_team"].notna() & out["away_team"].notna()]
        out = out.sort_values("date").reset_index(drop=True)
        out["total_goals"] = out["home_goals"] + out["away_goals"]
        out["over25"] = (out["total_goals"] > 2.5).astype(float)
        out["over15"] = (out["total_goals"] > 1.5).astype(float)
        out["over35"] = (out["total_goals"] > 3.5).astype(float)
        out["btts"] = ((out["home_goals"] > 0) & (out["away_goals"] > 0)).astype(float)
        if "ht_home_goals" in out.columns and out["ht_home_goals"].notna().any():
            out["ht_total_goals"] = out["ht_home_goals"] + out["ht_away_goals"]
            out["ht_over05"] = (out["ht_total_goals"] >= 1).astype(float)
            out["ht_over15"] = (out["ht_total_goals"] >= 2).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["home_sot_ratio"] = np.where(out.get("home_shots", pd.Series(dtype=float)) > 0,
                out.get("home_sot", pd.Series(dtype=float)) / out.get("home_shots", pd.Series(dtype=float)), np.nan)
            out["away_sot_ratio"] = np.where(out.get("away_shots", pd.Series(dtype=float)) > 0,
                out.get("away_sot", pd.Series(dtype=float)) / out.get("away_shots", pd.Series(dtype=float)), np.nan)
        out = _enrich_with_api_shots(out)
        out = _enrich_with_standard_sidemarket_odds(out)   # real BTTS/O1.5/O3.5 for 2nd-divs
        _CACHE = out
        return out

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
            ("home_goals",      "FTHG"),
            ("away_goals",      "FTAG"),
            ("ht_home_goals",   "HTHG"),
            ("ht_away_goals",   "HTAG"),
            ("home_corners",    "HC"),
            ("away_corners",    "AC"),
            ("home_fouls",      "HF"),
            ("away_fouls",      "AF"),
            ("home_shots",      "HS"),
            ("away_shots",      "AS"),
            ("home_sot",        "HST"),
            ("away_sot",        "AST"),
        ]:
            df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")

        df["ftr"]      = raw.get("FTR", np.nan)
        df["referee"]  = raw.get("Referee", np.nan)
        df["odds_over25"]  = _pick_odds(raw, _OVER_COLS)
        df["odds_under25"] = _pick_odds(raw, _UNDER_COLS)
        df["odds_btts"]    = _pick_odds(raw, _BTTS_COLS)
        df["odds_over15"]  = _pick_odds(raw, _OVER15_COLS)
        df["odds_over35"]  = _pick_odds(raw, _OVER35_COLS)

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
                        ("ht_home_goals","HTHG"),("ht_away_goals","HTAG"),
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
                    df["odds_btts"]     = _pick_odds(raw, _BTTS_COLS)
                    df["odds_over15"]   = _pick_odds(raw, _OVER15_COLS)
                    df["odds_over35"]   = _pick_odds(raw, _OVER35_COLS)
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

    # ── Direct HT download: big-5 leagues not in local files ─────────────────
    # These leagues aren't used for prediction but add ~1,700 rows of HT data
    _HT_EXTRA = {
        "Premier League":    "E0",
        "Bundesliga 1":      "D1",
        "La Liga":           "SP1",
        "Serie A":           "I1",
        "Ligue 1":           "F1",
    }
    _STD_URL = "https://www.football-data.co.uk/mmz4281/2526/{code}.csv"
    for lg_name, code in _HT_EXTRA.items():
        # Skip if already loaded via local CSV
        if any(lg_name in str(f.get("league", "")) for f in [{"league": f["league"].iloc[0]} for f in frames if not f.empty]):
            continue
        try:
            import requests as _req
            from io import StringIO as _SIO
            r = _req.get(_STD_URL.format(code=code), timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            raw = pd.read_csv(_SIO(r.text), encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
            if "FTHG" not in raw.columns:
                continue
            df = pd.DataFrame()
            df["date"]      = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
            df = df[df["date"].notna()].copy()
            df["league"]    = lg_name
            df["season"]    = "2025/26"
            df["home_team"] = raw.get("HomeTeam", np.nan)
            df["away_team"] = raw.get("AwayTeam", np.nan)
            for out_col, src_col in [
                ("home_goals","FTHG"),("away_goals","FTAG"),
                ("ht_home_goals","HTHG"),("ht_away_goals","HTAG"),
                ("home_shots","HS"),("away_shots","AS"),
                ("home_sot","HST"),("away_sot","AST"),
                ("home_corners","HC"),("away_corners","AC"),
                ("home_fouls","HF"),("away_fouls","AF"),
            ]:
                df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
            df["ftr"] = raw.get("FTR", np.nan)
            df["odds_over25"]  = _pick_odds(raw, _OVER_COLS)
            df["odds_under25"] = _pick_odds(raw, _UNDER_COLS)
            df["odds_btts"]    = _pick_odds(raw, _BTTS_COLS)
            df["odds_over15"]  = _pick_odds(raw, _OVER15_COLS)
            df["odds_over35"]  = _pick_odds(raw, _OVER35_COLS)
            df = df[df["home_team"].notna() & df["away_team"].notna()]
            frames.append(df)
            log.debug(f"HT extra: {lg_name} {len(df)} rows")
        except Exception as e:
            log.debug(f"HT extra {lg_name}: {e}")

    if not frames:
        raise RuntimeError(f"No match sheets found in {path}")

    frames.extend(_load_api_football_only_leagues())
    out = pd.concat(frames, ignore_index=True)
    out = out[out["home_team"].notna() & out["away_team"].notna()]
    out = out.sort_values("date").reset_index(drop=True)

    # Derived columns
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["over25"]      = (out["total_goals"] > 2.5).astype(float)
    out["over15"]      = (out["total_goals"] > 1.5).astype(float)
    out["over35"]      = (out["total_goals"] > 3.5).astype(float)
    out["btts"]        = (
        (out["home_goals"] > 0) & (out["away_goals"] > 0)
    ).astype(float)

    # HT targets (only populated when HTHG/HTAG available)
    if "ht_home_goals" in out.columns and out["ht_home_goals"].notna().any():
        out["ht_total_goals"] = out["ht_home_goals"] + out["ht_away_goals"]
        out["ht_over05"]      = (out["ht_total_goals"] >= 1).astype(float)
        out["ht_over15"]      = (out["ht_total_goals"] >= 2).astype(float)

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

    # ── API-Football shot enrichment for new-format leagues ──────────────────────
    out = _enrich_with_api_shots(out)
    # ── Real BTTS/O1.5/O3.5 odds for standard 2nd-divs (was synthetic 1.85) ───────
    out = _enrich_with_standard_sidemarket_odds(out)

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
                f"Rows auto-removed.",
                stacklevel=2,
            )
            out = out[~(
                (out["league"] == dk_league) &
                (out["home_team"].isin(contaminated) | out["away_team"].isin(contaminated))
            )]

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
