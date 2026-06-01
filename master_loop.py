"""
Wowza Master Loop
=================
Single process that handles ALL scheduling internally.
Replaces Task Scheduler entirely — more reliable, no admin needed.

Schedule:
  Predict      — every 4h starting 06:00
  UpdateResults— every 2h starting 07:00
  Telegram     — 5 min after each predict
  GitPush      — 10 min after each predict
  WorldCup     — every 2h starting 08:00
  LiveScanner  — every 30s when games active, 5min idle

Start: runs at Windows login via Startup folder shortcut.
Stop:  close the window or Task Manager → python.exe
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PY  = Path(r"C:\WowzaVenv\Scripts\python.exe")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Schedule config ───────────────────────────────────────────────────────────
PREDICT_HOURS       = {6, 10, 14, 18, 22}
UPDATE_HOURS        = {7, 9, 11, 13, 15, 17, 19, 21, 23}
WORLDCUP_HOURS      = {8, 10, 12, 14, 16, 18, 20, 22}
SHARP_HOURS         = {8, 10, 12, 14, 16, 18, 20, 22}
LIVE_HOUR_START     = 11
LIVE_HOUR_END       = 24
LIVE_INTERVAL_LIVE  = 120   # seconds when games active (2 min)
LIVE_INTERVAL_IDLE  = 600   # seconds when no games (10 min)


# ── Task runner ───────────────────────────────────────────────────────────────

def _run(script: str, args: list[str] = []) -> bool:
    cmd = [str(VENV_PY), str(BASE_DIR / script)] + args
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, timeout=300,
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode == 0:
            log.info(f"  OK: {script}")
        else:
            log.warning(f"  FAIL: {script} — {result.stderr[-200:]}")
        return result.returncode == 0
    except Exception as e:
        log.error(f"  ERROR: {script} — {e}")
        return False


# ── Live scanner (inline — avoids subprocess overhead every 30s) ──────────────

def _live_tick() -> int:
    """Run one live scan cycle. Returns number of live games found."""
    try:
        from src.live_scanner import _fetch_live_scores, run as live_run
        from telegram_bot.notifier import notify_live_signals
        games = _fetch_live_scores()
        if games:
            live_run()
            notify_live_signals()
        return len(games)
    except Exception as e:
        log.error(f"Live scan error: {e}")
        return 0


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("  Wowza Master Loop — all tasks in one process")
    log.info(f"  Predict: {sorted(PREDICT_HOURS)}")
    log.info(f"  Results: {sorted(UPDATE_HOURS)}")
    log.info(f"  Live:    {LIVE_HOUR_START}:00-{LIVE_HOUR_END}:00 (30s/5min)")
    log.info("=" * 55)

    last_predict  = datetime.min
    last_results  = datetime.min
    last_worldcup = datetime.min
    last_sharp    = datetime.min
    last_telegram = datetime.min
    last_gitpush  = datetime.min

    while True:
        now  = datetime.now()
        hour = now.hour

        try:
            # ── Predict every 4h ────────────────────────────────────────────
            if hour in PREDICT_HOURS and (now - last_predict).total_seconds() > 3500:
                log.info(f"[{now.strftime('%H:%M')}] Running PREDICT...")
                _run("pipeline.py", ["--mode", "predict"])
                last_predict  = now
                last_telegram = datetime.min   # reset so telegram runs after predict
                last_gitpush  = datetime.min

            # ── Telegram 5 min after predict ────────────────────────────────
            if (last_predict > datetime.min
                    and (now - last_predict).total_seconds() > 300
                    and (now - last_telegram).total_seconds() > 3500):
                log.info(f"[{now.strftime('%H:%M')}] Running TELEGRAM...")
                _run("telegram_bot/notifier.py")
                last_telegram = now

            # ── Git push 10 min after predict ───────────────────────────────
            if (last_predict > datetime.min
                    and (now - last_predict).total_seconds() > 600
                    and (now - last_gitpush).total_seconds() > 3500):
                log.info(f"[{now.strftime('%H:%M')}] Running GIT PUSH...")
                _run("scheduler/git_push_outputs.py")
                last_gitpush = now

            # ── Update results every 2h ──────────────────────────────────────
            if hour in UPDATE_HOURS and (now - last_results).total_seconds() > 3500:
                log.info(f"[{now.strftime('%H:%M')}] Running UPDATE RESULTS...")
                _run("update_results.py")
                last_results = now

            # ── World Cup every 2h ───────────────────────────────────────────
            if hour in WORLDCUP_HOURS and (now - last_worldcup).total_seconds() > 3500:
                log.info(f"[{now.strftime('%H:%M')}] Running WORLD CUP...")
                _run("worldcup/tracker.py")
                last_worldcup = now

            # ── Sharp money tracker every 2h ─────────────────────────────────
            if hour in SHARP_HOURS and (now - last_sharp).total_seconds() > 3500:
                log.info(f"[{now.strftime('%H:%M')}] Running SHARP TRACKER...")
                _run("src/sharp_tracker.py")
                last_sharp = now

            # ── Live scanner ─────────────────────────────────────────────────
            if LIVE_HOUR_START <= hour < LIVE_HOUR_END:
                n_live = _live_tick()
                interval = LIVE_INTERVAL_LIVE if n_live > 0 else LIVE_INTERVAL_IDLE
                time.sleep(interval)
            else:
                time.sleep(60)  # outside match hours, check every minute

        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as e:
            log.error(f"Loop error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
