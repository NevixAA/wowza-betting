"""
Live Scanner Loop
=================
Runs continuously. 30s when games are live, 5min when idle.
Start this once — it runs all day and self-throttles.

Usage:
    python live_loop.py
    (or via C:\\WowzaBot\\live_loop.bat — add to startup)
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.live_scanner import run as scan, _fetch_live_scores
from telegram_bot.notifier import notify_live_signals

MATCH_HOUR_START = 11   # don't scan before 11:00
MATCH_HOUR_END   = 24   # don't scan after midnight
INTERVAL_LIVE    = 120  # seconds between scans when games are active (2 min)
INTERVAL_IDLE    = 600  # seconds between scans when no live games (10 min)
INTERVAL_SLEEP   = 600  # seconds when outside match hours (10 min)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _in_match_hours() -> bool:
    h = datetime.now().hour
    return MATCH_HOUR_START <= h < MATCH_HOUR_END


def main():
    log.info("=" * 50)
    log.info("  Wowza Live Scanner Loop started")
    log.info(f"  Active: {MATCH_HOUR_START}:00 - {MATCH_HOUR_END}:00")
    log.info(f"  Live interval: {INTERVAL_LIVE}s | Idle: {INTERVAL_IDLE}s")
    log.info("=" * 50)

    scan_count = 0
    signal_count = 0

    while True:
        try:
            if not _in_match_hours():
                next_hour = MATCH_HOUR_START if datetime.now().hour >= MATCH_HOUR_END else MATCH_HOUR_START
                log.info(f"Outside match hours. Sleeping {INTERVAL_SLEEP//60}min...")
                time.sleep(INTERVAL_SLEEP)
                continue

            # Quick check: any live games right now?
            live_games = _fetch_live_scores()

            if not live_games:
                log.debug("No live games. Idle scan in 5min.")
                time.sleep(INTERVAL_IDLE)
                continue

            # Games are live — run full scan
            scan_count += 1
            log.info(f"[Scan #{scan_count}] {len(live_games)} live game(s) — scanning...")
            tips = scan()

            if tips:
                sent = notify_live_signals()
                signal_count += sent
                log.info(f"  {len(tips)} signal(s) | {sent} alert(s) sent | Total alerts: {signal_count}")

            time.sleep(INTERVAL_LIVE)

        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
