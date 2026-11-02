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
    """'Arsenal vs Liverpool' → ('Arsenal', 'Liverpool'); no match string -> (None, None)."""
    if match_str is None or (isinstance(match_str, float) and match_str != match_str):
        return None, None
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


def _id(x) -> int | None:
    return int(float(x)) if _has_id(x) else None


def _ledger_key(r) -> tuple:
    """Identity of a ledger row: (player_id, fixture_id, market) - issue #1 step 1. Was
    (player_name, market, match_date): two players sharing a name on one day collapsed into one
    row. Legacy rows written before every tip carried a fixture id key on (player_id, match_date,
    market) so a re-seen tip is still recognised (see _legacy_key)."""
    return ("fixture", _id(r.get("player_id")), _id(r.get("fixture_id")), str(r.get("market")))


def _legacy_key(r) -> tuple:
    """(player_id, match_date, market): how a NEW row is matched against a legacy row that has no
    fixture_id. Ids only - never the player's name."""
    return ("date", _id(r.get("player_id")), str(r.get("match_date"))[:10], str(r.get("market")))


def append_player_signals(
    tips_df: pd.DataFrame,
    ledger_file: Path = PLAYER_LEDGER_FILE,
) -> int:
    """
    Append SNIPER/MARKSMAN/VALUABLE signals from tips_df to player_ledger.csv.
    Deduplicates on IDS: (player_id, fixture_id, market) - issue #1 step 1. A tip without a match
    date, without a player_id or without an API-Football fixture_id is REFUSED (logged with the
    reason) - it cannot be settled honestly. Legacy ledger rows without a fixture_id still block
    a duplicate of the same (player_id, match_date, market).
    Returns count of newly appended rows.
    """
    if tips_df.empty:
        return 0

    signals = tips_df[tips_df["tier"].isin(PLAYER_TRACKED_TIERS)].copy()
    if signals.empty:
        return 0

    today = _today()
    rows = []
    refused = {"no match date": 0, "no player_id": 0, "no fixture_id": 0}
    for _, r in signals.iterrows():
        home, away = _split_match(r.get("match"))
        _md = str(r.get("date"))[:10] if r.get("date") is not None else ""
        if not _md or _md in ("nan", "NaT", "None"):
            refused["no match date"] += 1          # (was dated "today")
            continue
        if not _has_id(r.get("player_id")):
            refused["no player_id"] += 1
            continue
        if not _has_id(r.get("fixture_id")):
            refused["no fixture_id"] += 1
            continue
        name = r.get("player_name")
        rows.append({
            "signal_date":   today,
            "match_date":    _md,
            "league":        r.get("league"),
            "home_team":     home,
            "away_team":     away,
            "player_id":     _id(r.get("player_id")),
            "fixture_id":    _id(r.get("fixture_id")),
            # API-Football sends some names with doubled spaces ("Paris Josua  Brunner")
            "player_name":   " ".join(str(name).split()) if name is not None else None,
            "team":          r.get("team"),
            "position":      r.get("position"),
            "market":        r.get("market"),
            "tier":          r.get("tier"),
            "model_prob":    r.get("model_prob"),
            "market_odds":   r.get("market_odds"),
            "ev":            r.get("ev"),
            "edge_rel":      r.get("edge_rel"),
            "n_games":       r.get("n_games"),
            "is_played":     "",
            "result":        "",
            "pnl":           "",
            "notes":         "",
            "resolved_date": "",
        })

    if any(refused.values()):
        log.info("player_ledger: signal(s) refused - "
                 + ", ".join(f"{k}: {v}" for k, v in refused.items() if v))
    if not rows:
        return 0
    new_df = pd.DataFrame(rows, columns=PLAYER_LEDGER_COLS)
    # One row per (player_id, fixture_id, market) within this batch as well.
    new_df = new_df[~new_df.apply(_ledger_key, axis=1).duplicated()]

    if ledger_file.exists():
        existing = pd.read_csv(ledger_file, dtype=str)
        recs = existing.to_dict("records")
        existing_keys = {_ledger_key(r) for r in recs if _has_id(r.get("fixture_id"))}
        legacy_keys = {_legacy_key(r) for r in recs if not _has_id(r.get("fixture_id"))}
        new_df = new_df[~new_df.apply(lambda r: _ledger_key(r) in existing_keys
                                      or _legacy_key(r) in legacy_keys, axis=1)]
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
    undated = 0
    for _, r in signals.iterrows():
        # A signal without a match date is refused (it was dated "today", a date it never had).
        _md = str(r.get("date"))[:10] if r.get("date") is not None else ""
        if not _md or _md in ("nan", "NaT", "None"):
            undated += 1
            continue
        home, away = _split_match(r.get("match", ""))
        label = str(r.get("market", ""))
        side  = _parse_side(label)
        rows.append({
            "signal_date":   today,
            "match_date":    _md,
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

    if undated:
        log.info(f"sharp_ledger: {undated} signal(s) refused (no match date)")
    if not rows:
        return 0
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
