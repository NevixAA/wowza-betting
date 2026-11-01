"""
Signal ledgers for player props and sharp money.
Appends actionable signals to CSV files for historical performance tracking.

Player ledger: records SNIPER/MARKSMAN/VALUABLE prop tips from each predict run.
Sharp ledger:  records STEAM_STRONG/STEAM_SHARP/STRONG signals from each sharp run.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

PLAYER_LEDGER_FILE = config.OUTPUT_DIR / "player_ledger.csv"
SHARP_LEDGER_FILE  = config.OUTPUT_DIR / "sharp_ledger.csv"

PLAYER_LEDGER_COLS = [
    "signal_date", "match_date", "league", "home_team", "away_team",
    # player_id / fixture_id carried from the tip so this ledger can be joined EXACTLY to
    # player_history.parquet (which has both) instead of on normalised player names. Rows
    # written before 2026-08-15 have them blank; readers must fall back to the name.
    "player_id", "fixture_id",
    "player_name", "team", "position", "market", "tier",
    "model_prob", "market_odds", "ev", "edge_rel", "n_games",
    "is_played", "result", "pnl", "notes", "resolved_date",
]

SHARP_LEDGER_COLS = [
    "signal_date", "match_date", "league", "home_team", "away_team",
    "market_label", "side", "signal", "opening_odds", "current_odds",
    "drift_pct", "consensus_pct", "n_books",
    "result", "pnl", "resolved_date",
]

# Track EVERY pick the model produces (2026/27: props are a calibrated-but-unproven TEST
# model — keep the whole prediction record for calibration/accuracy, not just the top tiers).
PLAYER_TRACKED_TIERS  = {"SNIPER", "MARKSMAN", "VALUABLE", "WATCH", "PAPER", "AVOID"}
SHARP_TRACKED_SIGNALS = {"STEAM_STRONG", "STEAM_SHARP", "STRONG"}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _split_match(match_str: str) -> tuple[str, str]:
    """'Arsenal vs Liverpool' → ('Arsenal', 'Liverpool')"""
    parts = str(match_str).split(" vs ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return str(match_str).strip(), ""


def _parse_side(market_label: str) -> str:
    """Extract bet side from sharp market label string."""
    label = market_label.upper()
    if "O/U" in label and "OVER" in label:
        return "OVER"
    if "O/U" in label and "UNDER" in label:
        return "UNDER"
    if "1X2 HOME" in label:
        return "HOME"
    if "1X2 AWAY" in label:
        return "AWAY"
    if "DRAW" in label:
        return "DRAW"
    return ""


def _has_id(x) -> bool:
    try:
        return int(float(x)) > 0
    except (TypeError, ValueError):
        return False


def _ledger_key(r) -> tuple:
    """Identity of a ledger row: (player_id, fixture_id, market), or (player_id, match_date, market)
    when the fixture has no API-Football id. Was (player_name, market, match_date) - two players
    sharing a name on one day collapsed into one row."""
    pid = int(float(r.get("player_id"))) if _has_id(r.get("player_id")) else None
    fid = int(float(r.get("fixture_id"))) if _has_id(r.get("fixture_id")) else None
    if fid is not None:
        return ("fixture", pid, fid, str(r.get("market", "")))
    return ("date", pid, str(r.get("match_date", ""))[:10], str(r.get("market", "")))


def append_player_signals(
    tips_df: pd.DataFrame,
    ledger_file: Path = PLAYER_LEDGER_FILE,
) -> int:
    """
    Append SNIPER/MARKSMAN/VALUABLE signals from tips_df to player_ledger.csv.
    Deduplicates on IDS: (player_id, fixture_id, market), or (player_id, match_date, market) for a
    fixture without an API-Football id. A tip without a match date or without a player_id is
    refused (logged) - it cannot be settled honestly.
    Returns count of newly appended rows.
    """
    if tips_df.empty:
        return 0

    signals = tips_df[tips_df["tier"].isin(PLAYER_TRACKED_TIERS)].copy()
    if signals.empty:
        return 0

    today = _today()
    rows = []
    refused = 0
    for _, r in signals.iterrows():
        home, away = _split_match(r.get("match", ""))
        _md = str(r.get("date") or "")[:10]
        if not _md or _md in ("nan", "NaT", "None") or not _has_id(r.get("player_id")):
            refused += 1          # no match date / no player id (was dated "today")
            continue
        rows.append({
            "signal_date":   today,
            "match_date":    _md,
            "league":        r.get("league", ""),
            "home_team":     home,
            "away_team":     away,
            "player_id":     r.get("player_id", ""),
            "fixture_id":    r.get("fixture_id", ""),
            # API-Football sends some names with doubled spaces ("Paris Josua  Brunner")
            "player_name":   " ".join(str(r.get("player_name", "") or "").split()),
            "team":          r.get("team", ""),
            "position":      r.get("position", ""),
            "market":        r.get("market", ""),
            "tier":          r.get("tier", ""),
            "model_prob":    r.get("model_prob", ""),
            "market_odds":   r.get("market_odds", ""),
            "ev":            r.get("ev", ""),
            "edge_rel":      r.get("edge_rel", ""),
            "n_games":       r.get("n_games", ""),
            "is_played":     "",
            "result":        "",
            "pnl":           "",
            "notes":         "",
            "resolved_date": "",
        })

    if refused:
        log.info(f"player_ledger: {refused} signal(s) refused (no match date or no player_id)")
    if not rows:
        return 0
    new_df = pd.DataFrame(rows, columns=PLAYER_LEDGER_COLS)

    if ledger_file.exists():
        existing = pd.read_csv(ledger_file, dtype=str)
        existing_keys = {_ledger_key(r) for r in existing.to_dict("records")}
        new_df = new_df[~new_df.apply(lambda r: _ledger_key(r) in existing_keys, axis=1)]
        if new_df.empty:
            log.info("player_ledger: all signals already recorded")
            return 0
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(ledger_file, index=False)
    log.info(f"player_ledger: +{len(new_df)} signal(s) → {ledger_file.name}")
    return len(new_df)


def append_sharp_signals(
    tips_df: pd.DataFrame,
    ledger_file: Path = SHARP_LEDGER_FILE,
) -> int:
    """
    Append STEAM_STRONG/STEAM_SHARP/STRONG signals to sharp_ledger.csv.
    Deduplicates on (home_team, away_team, match_date, side) — one entry per match/side.
    Returns count of newly appended rows.
    """
    if tips_df.empty:
        return 0

    signals = tips_df[tips_df["signal"].isin(SHARP_TRACKED_SIGNALS)].copy()
    if signals.empty:
        return 0

    today = _today()
    rows = []
    for _, r in signals.iterrows():
        home, away = _split_match(r.get("match", ""))
        label = str(r.get("market", ""))
        side  = _parse_side(label)
        rows.append({
            "signal_date":   today,
            "match_date":    str(r.get("date", today))[:10],
            "league":        r.get("league", ""),
            "home_team":     home,
            "away_team":     away,
            "market_label":  label,
            "side":          side,
            "signal":        r.get("signal", ""),
            "opening_odds":  r.get("opening_odds", ""),
            "current_odds":  r.get("current_odds", ""),
            "drift_pct":     r.get("drift_pct", ""),
            "consensus_pct": r.get("consensus_pct", ""),
            "n_books":       r.get("n_books", ""),
            "result":        "",
            "pnl":           "",
            "resolved_date": "",
        })

    new_df = pd.DataFrame(rows, columns=SHARP_LEDGER_COLS)

    if ledger_file.exists():
        existing = pd.read_csv(ledger_file, dtype=str)
        existing_keys = set(
            zip(
                existing["home_team"], existing["away_team"],
                existing["match_date"], existing["side"],
            )
        )
        new_df = new_df[
            ~new_df.apply(
                lambda r: (r["home_team"], r["away_team"], r["match_date"], r["side"])
                          in existing_keys,
                axis=1,
            )
        ]
        if new_df.empty:
            log.info("sharp_ledger: all signals already recorded")
            return 0
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(ledger_file, index=False)
    log.info(f"sharp_ledger: +{len(new_df)} signal(s) → {ledger_file.name}")
    return len(new_df)
