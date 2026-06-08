"""
Player Stats Data Fetcher — FBref (free, no API key needed).
Extends the existing fbref_scraper.py pattern already in the project.

Scrapes per-player season stats from FBref standard stat tables:
  - shooting: goals, shots, shots_on_target
  - passing: assists, key_passes
  - misc: yellow_cards, red_cards, fouls

Cached for 7 days. Polite 4s delay between requests.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from . import config

CACHE_FILE    = config.CACHE_FILE
CACHE_DAYS    = 7
REQUEST_DELAY = 4.0   # polite scraping

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://fbref.com/",
}

# FBref competition IDs → (comp_id, url_slug)
FBREF_LEAGUES: dict[str, tuple] = {
    "Championship":   (10,  "Championship"),
    "League One":     (15,  "League-One"),
    "League Two":     (16,  "League-Two"),
    "Bundesliga 2":   (33,  "2-Bundesliga"),
    "Ligue 2":        (60,  "Ligue-2"),
    "La Liga 2":      (17,  "Segunda-Division"),
    "Serie B":        (18,  "Serie-B"),
    "Premier League": (9,   "Premier-League"),
    "La Liga":        (12,  "La-Liga"),
    "Bundesliga":     (20,  "Bundesliga"),
    "Ligue 1":        (13,  "Ligue-1"),
}

# FBref table IDs for each stat type
STAT_TABLES = {
    "standard": "stats_standard",
    "shooting":  "stats_shooting",
    "misc":      "stats_misc",
}


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(data: dict) -> None:
    CACHE_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _stale(entry: dict) -> bool:
    ts = entry.get("fetched_at", "")
    if not ts:
        return True
    try:
        return datetime.now() - datetime.fromisoformat(ts) > timedelta(days=CACHE_DAYS)
    except Exception:
        return True


# ── FBref scraper ─────────────────────────────────────────────────────────────

def _fetch_fbref_table(comp_id: int, slug: str, table_id: str) -> Optional[pd.DataFrame]:
    """Fetch one FBref stats table for a competition."""
    url = f"https://fbref.com/en/comps/{comp_id}/{table_id}/{slug}-Stats"
    time.sleep(REQUEST_DELAY)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"    [FBref] {r.status_code} — {url}")
            return None
        tables = pd.read_html(r.text, attrs={"id": table_id})
        if not tables:
            return None
        df = tables[0].copy()
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                f"{b}" if str(a).startswith("Unnamed") else f"{a}_{b}"
                if b and str(b) != "nan" else str(a)
                for a, b in df.columns
            ]
        return df
    except Exception as e:
        print(f"    [FBref] Error: {e}")
        return None


def _parse_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Extract player, team, position, appearances, minutes, goals, assists."""
    cols = {c.lower(): c for c in df.columns}

    def _col(*candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    name_col = _col("player")
    team_col = _col("squad")
    pos_col  = _col("pos")
    mp_col   = _col("mp", "matches")
    min_col  = _col("min", "minutes")
    gls_col  = _col("gls", "goals")
    ast_col  = _col("ast", "assists")

    if not name_col:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name or name.lower() in ("player", ""):
            continue
        rows.append({
            "player_name": name,
            "team":        str(row.get(team_col, "")).strip() if team_col else "",
            "position":    str(row.get(pos_col, "")).strip()  if pos_col  else "",
            "appearances": _safe_int(row.get(mp_col))         if mp_col   else 0,
            "minutes":     _safe_int(row.get(min_col))        if min_col  else 0,
            "goals":       _safe_int(row.get(gls_col))        if gls_col  else 0,
            "assists":     _safe_int(row.get(ast_col))        if ast_col  else 0,
        })
    return pd.DataFrame(rows)


def _parse_shooting(df: pd.DataFrame) -> pd.DataFrame:
    """Extract shots total and shots on target."""
    cols = {c.lower(): c for c in df.columns}

    def _col(*candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    name_col = _col("player")
    sh_col   = _col("sh", "shots")
    sot_col  = _col("sot", "shots on target")

    if not name_col:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name or name.lower() in ("player", ""):
            continue
        rows.append({
            "player_name":     name,
            "shots_total":     _safe_int(row.get(sh_col))  if sh_col  else 0,
            "shots_on_target": _safe_int(row.get(sot_col)) if sot_col else 0,
        })
    return pd.DataFrame(rows)


def _parse_misc(df: pd.DataFrame) -> pd.DataFrame:
    """Extract yellow cards."""
    cols = {c.lower(): c for c in df.columns}

    def _col(*candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    name_col = _col("player")
    yc_col   = _col("cyel", "yellow cards", "yel", "crdy")

    if not name_col:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name or name.lower() in ("player", ""):
            continue
        rows.append({
            "player_name":  name,
            "yellow_cards": _safe_int(row.get(yc_col)) if yc_col else 0,
        })
    return pd.DataFrame(rows)


def _safe_int(val) -> int:
    try:
        return int(float(str(val).replace(",", "")))
    except Exception:
        return 0


# ── Public: fetch all players for a league ────────────────────────────────────

def fetch_league_player_stats(
    league: str,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return flat list of player stat dicts for one league.
    Scraped from FBref. Cached for CACHE_DAYS.
    """
    info = FBREF_LEAGUES.get(league)
    if not info or info[0] is None:
        return []

    comp_id, slug = info

    cache = _load_cache()
    key   = f"fbref_players|{league}"
    entry = cache.get(key, {})

    if not force_refresh and entry and not _stale(entry):
        return entry.get("players", [])

    print(f"  [FBref] Fetching player stats: {league}...")

    # Standard stats (appearances, goals, assists)
    std_df = _fetch_fbref_table(comp_id, slug, "stats_standard")
    if std_df is None or std_df.empty:
        print(f"  [FBref] No standard stats for {league}")
        return []
    base = _parse_standard(std_df)
    if base.empty:
        return []

    # Shooting stats
    sht_df = _fetch_fbref_table(comp_id, slug, "stats_shooting")
    if sht_df is not None and not sht_df.empty:
        shots = _parse_shooting(sht_df)
        if not shots.empty:
            base = base.merge(shots, on="player_name", how="left")

    # Misc stats (cards)
    misc_df = _fetch_fbref_table(comp_id, slug, "stats_misc")
    if misc_df is not None and not misc_df.empty:
        misc = _parse_misc(misc_df)
        if not misc.empty:
            base = base.merge(misc, on="player_name", how="left")

    # Fill missing cols
    for col in ["shots_total", "shots_on_target", "yellow_cards"]:
        if col not in base.columns:
            base[col] = 0
        base[col] = base[col].fillna(0).astype(int)

    base["league"]    = league
    base["player_id"] = base["player_name"].apply(lambda n: abs(hash(n)) % 10_000_000)
    base["rating"]    = 0.0

    players = base[base["appearances"] >= config.MIN_APPEARANCES].to_dict("records")

    cache[key] = {
        "fetched_at": datetime.now().isoformat(),
        "players":    players,
    }
    _save_cache(cache)
    print(f"  [FBref] Cached {len(players)} players for {league}")
    return players


# ── Bulk collection ───────────────────────────────────────────────────────────

def collect_history(
    max_fixtures: int = 800,
    leagues: dict | None = None,
    seasons: list[str] | None = None,
) -> list[dict]:
    """Collect player stats from FBref for all supported leagues."""
    if leagues is None:
        leagues = FBREF_LEAGUES

    all_rows: list[dict] = []
    for league in leagues:
        rows = fetch_league_player_stats(league, force_refresh=False)
        all_rows.extend(rows)
        print(f"  [{league}] {len(rows)} players")

    print(f"[DONE] {len(all_rows)} total player rows.")
    return all_rows
