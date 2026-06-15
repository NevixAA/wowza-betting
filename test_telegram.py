"""
Telegram notifier integration test.

Usage:
  python test_telegram.py                # dry-run: preview all messages, no send
  python test_telegram.py --send         # actually send to Telegram
  python test_telegram.py --force        # clear dedup cache first, then dry-run
  python test_telegram.py --force --send # clear dedup + send everything
  python test_telegram.py --fn wc        # test only one function (wc/sharp/props/digest/sniper/weekly)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent

# ── Patch _send before importing notifier ────────────────────────────────────

_sent_messages: list[str] = []
_dry_run = True

def _mock_send(token: str, chat_id: str, text: str) -> bool:
    _sent_messages.append(text)
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)
    return True

import telegram_bot.notifier as notifier

_real_send = notifier._send


def _patched_send(token: str, chat_id: str, text: str) -> bool:
    _sent_messages.append(text)
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)
    if not _dry_run:
        return _real_send(token, chat_id, text)
    return True


notifier._send = _patched_send


# ── Test runners ──────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_sniper():
    _section("notify_new_snipers() — player SNIPER/MARKSMAN/VALUABLE")
    n = notifier.notify_new_snipers()
    print(f"  → {n} message(s) sent")


def test_props():
    _section("notify_player_props() — WC player props with odds + EV")
    n = notifier.notify_player_props()
    print(f"  → {n} message(s) sent")


def test_wc():
    _section("notify_wc_strong() — WC drift signals (STRONG/SHARP/FADING)")
    n = notifier.notify_wc_strong()
    print(f"  → {n} message(s) sent")


def test_sharp():
    _section("notify_sharp_strong() — domestic sharp money")
    n = notifier.notify_sharp_strong()
    print(f"  → {n} message(s) sent")


def test_live():
    _section("notify_live_signals() — live in-play O/U tips")
    n = notifier.notify_live_signals()
    print(f"  → {n} message(s) sent")


def test_ht():
    _section("notify_ht_tips() — HT model tips")
    n = notifier.notify_ht_tips()
    print(f"  → {n} message(s) sent")


def test_digest():
    _section("notify_daily_digest() — model/sharp/WC morning digest")
    before = len(_sent_messages)
    ok = notifier.notify_daily_digest()
    n = len(_sent_messages) - before
    print(f"  → {'sent ✓' if ok or n > 0 else 'nothing to send / error'}")


def test_weekly():
    _section("notify_weekly_summary() — model/sharp/WC weekly summary")
    before = len(_sent_messages)
    ok = notifier.notify_weekly_summary()
    n = len(_sent_messages) - before
    print(f"  → {'sent ✓' if ok or n > 0 else 'nothing to send / error'}")


def test_props_digest():
    _section("notify_props_daily_digest() — player props daily briefing")
    before = len(_sent_messages)
    ok = notifier.notify_props_daily_digest()
    n = len(_sent_messages) - before
    print(f"  → {'sent ✓' if ok or n > 0 else 'nothing to send / error'}")


def test_props_weekly():
    _section("notify_props_weekly_summary() — player props weekly summary")
    before = len(_sent_messages)
    ok = notifier.notify_props_weekly_summary()
    n = len(_sent_messages) - before
    print(f"  → {'sent ✓' if ok or n > 0 else 'nothing to send / error'}")


def test_agent():
    _section("notify_agent_analysis() — agent research output")
    n = notifier.notify_agent_analysis()
    print(f"  → {n} message(s) sent")


ALL_TESTS = {
    "sniper":        test_sniper,
    "props":         test_props,
    "props_digest":  test_props_digest,
    "props_weekly":  test_props_weekly,
    "wc":            test_wc,
    "sharp":         test_sharp,
    "live":          test_live,
    "ht":            test_ht,
    "digest":        test_digest,
    "weekly":        test_weekly,
    "agent":         test_agent,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _dry_run, _sent_messages

    parser = argparse.ArgumentParser(description="Telegram notifier test harness")
    parser.add_argument("--send",  action="store_true", help="Actually send to Telegram (default: dry-run)")
    parser.add_argument("--force", action="store_true", help="Clear dedup cache before running")
    parser.add_argument("--fn",    choices=list(ALL_TESTS.keys()), help="Run only one function")
    args = parser.parse_args()

    _dry_run = not args.send
    mode = "DRY-RUN (no Telegram send)" if _dry_run else "LIVE SEND to Telegram"
    print(f"\n{'#' * 60}")
    print(f"  Telegram notifier test — {mode}")
    print(f"{'#' * 60}")

    if args.force:
        nf = BASE_DIR / "telegram_bot" / "notified.json"
        if nf.exists():
            backup = nf.with_suffix(".json.bak")
            backup.write_text(nf.read_text(encoding="utf-8"), encoding="utf-8")
            nf.write_text(json.dumps({"keys": []}, indent=2), encoding="utf-8")
            print(f"\n[force] Cleared notified.json (backup → {backup.name})")
        else:
            print("\n[force] notified.json didn't exist — nothing to clear")

    tests = {args.fn: ALL_TESTS[args.fn]} if args.fn else ALL_TESTS

    total_sent = 0
    for name, fn in tests.items():
        before = len(_sent_messages)
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
        total_sent += len(_sent_messages) - before

    print(f"\n{'#' * 60}")
    print(f"  Done — {total_sent} message(s) {'sent to Telegram' if not _dry_run else 'previewed (dry-run)'}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
