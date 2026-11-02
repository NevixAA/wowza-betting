"""
Telegram notifier preview harness (a CLI script, NOT a pytest module).

Usage:
  python test_telegram.py                # dry-run: preview all messages, nothing leaves the machine
  python test_telegram.py --send         # actually send to Telegram
  python test_telegram.py --force        # clear the (sandboxed) dedup cache first, then dry-run
  python test_telegram.py --force --send # clear the REAL dedup cache + send everything
  python test_telegram.py --fn wc        # run only one function (wc/sharp/props/digest/sniper/weekly)

DRY RUN REALLY IS DRY (estate scan 2026-11-02, finding g). It used to patch only notifier._send,
so every path that posts by itself (tips_table.send_photo, daily_pack albums, _send_one) sent for
real, with the real token and chat, from a "dry run". Now, from the moment this module is imported:
  * every HTTP request to api.telegram.org is intercepted at requests.Session.request and answered
    with a fake 200 - the transport, not one helper, is mocked;
  * the notifier is given placeholder credentials, never the real token / chat id;
  * the dedup files (notified.json & co.) are copied to a temp sandbox, so a preview does not mark
    real tips as already sent.
Live sending needs --send on the command line and is refused whenever pytest is loaded.
The runner functions are named run_*, so pytest never collects them as tests.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
TELEGRAM_HOST = "api.telegram.org"
DRY_TOKEN, DRY_CHAT = "DRY_RUN_TOKEN", "DRY_RUN_CHAT"

_sent_messages: list[str] = []
#: (METHOD, url with the token redacted) of every Telegram call the dry run intercepted.
_blocked_calls: list[tuple[str, str]] = []
_live = False


# ── transport guard (installed at import) ────────────────────────────────────

_REAL_REQUEST = requests.sessions.Session.request


def _redact(url: str) -> str:
    return re.sub(r"/bot[^/]+/", "/bot***/", str(url))


class _FakeTelegramResponse:
    status_code = 200
    ok = True
    text = '{"ok": true, "result": {}}'

    def json(self):
        return {"ok": True, "result": {}}

    def raise_for_status(self):
        return None


def _guarded_request(self, method, url, *args, **kwargs):
    if TELEGRAM_HOST in str(url) and not _live:
        _blocked_calls.append((str(method).upper(), _redact(url)))
        return _FakeTelegramResponse()
    return _REAL_REQUEST(self, method, url, *args, **kwargs)


requests.sessions.Session.request = _guarded_request


def enable_live_send() -> None:
    """The ONLY way to let messages out. Refused inside any pytest process."""
    global _live
    if "pytest" in sys.modules or "_pytest" in sys.modules:
        raise RuntimeError("live Telegram sending is never allowed from a test run")
    _live = True
    notifier._load_config = _REAL_LOAD_CONFIG


# ── notifier, with placeholder credentials and a sandboxed dedup state ──────

import telegram_bot.notifier as notifier  # noqa: E402

_REAL_LOAD_CONFIG = notifier._load_config
notifier._load_config = lambda: {"token": DRY_TOKEN, "chat_id": DRY_CHAT}

_SANDBOX = Path(tempfile.mkdtemp(prefix="wowza_tg_dryrun_"))
for _f in Path(notifier.BASE_DIR).glob("*.json"):
    if _f.name != "bot_config.json":
        shutil.copy2(_f, _SANDBOX / _f.name)
_REAL_DEDUP = {k: getattr(notifier, k) for k in (
    "BASE_DIR", "NOTIFIED_FILE", "SHARP_NOTIFIED_FILE", "WC_NOTIFIED_FILE",
    "PLAYER_NOTIFIED_FILE", "PLAYER_KICKOFF_FILE", "_SHARP_MOVE_FILE") if hasattr(notifier, k)}
for _k, _v in _REAL_DEDUP.items():
    setattr(notifier, _k, _SANDBOX if _k == "BASE_DIR" else _SANDBOX / Path(_v).name)
if (Path(_REAL_DEDUP.get("_SHARP_MOVE_FILE", "")).exists()
        and "_SHARP_MOVE_FILE" in _REAL_DEDUP):
    shutil.copy2(_REAL_DEDUP["_SHARP_MOVE_FILE"], notifier._SHARP_MOVE_FILE)

_real_send = notifier._send


def _patched_send(token: str, chat_id: str, text: str) -> bool:
    _sent_messages.append(text)
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)
    return _real_send(token, chat_id, text)      # dry run: the transport guard answers


notifier._send = _patched_send


# ── runners ──────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_sniper():
    _section("notify_new_snipers() — player SNIPER/MARKSMAN/VALUABLE")
    print(f"  → {notifier.notify_new_snipers()} message(s)")


def run_props():
    _section("notify_player_props() — WC player props with odds + EV")
    print(f"  → {notifier.notify_player_props()} message(s)")


def run_wc():
    _section("notify_wc_strong() — WC drift signals (STRONG/SHARP/FADING)")
    print(f"  → {notifier.notify_wc_strong()} message(s)")


def run_sharp():
    _section("notify_sharp_strong() — domestic sharp money")
    print(f"  → {notifier.notify_sharp_strong()} message(s)")


def run_live():
    _section("notify_live_signals() — live in-play O/U tips")
    print(f"  → {notifier.notify_live_signals()} message(s)")


def run_ht():
    _section("notify_ht_tips() — HT model tips")
    print(f"  → {notifier.notify_ht_tips()} message(s)")


def _run_bool(title, fn):
    _section(title)
    before = len(_sent_messages)
    ok = fn()
    n = len(_sent_messages) - before
    print(f"  → {'ok ✓' if ok or n > 0 else 'nothing to send / error'}")


def run_digest():
    _run_bool("notify_daily_digest() — model/sharp/WC morning digest", notifier.notify_daily_digest)


def run_weekly():
    _run_bool("notify_weekly_summary() — model/sharp/WC weekly summary", notifier.notify_weekly_summary)


def run_props_digest():
    _run_bool("notify_props_daily_digest() — player props daily briefing",
              notifier.notify_props_daily_digest)


def run_props_weekly():
    _run_bool("notify_props_weekly_summary() — player props weekly summary",
              notifier.notify_props_weekly_summary)


def run_agent():
    _section("notify_agent_analysis() — agent research output")
    print(f"  → {notifier.notify_agent_analysis()} message(s)")


RUNNERS = {
    "sniper":        run_sniper,
    "props":         run_props,
    "props_digest":  run_props_digest,
    "props_weekly":  run_props_weekly,
    "wc":            run_wc,
    "sharp":         run_sharp,
    "live":          run_live,
    "ht":            run_ht,
    "digest":        run_digest,
    "weekly":        run_weekly,
    "agent":         run_agent,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram notifier preview harness")
    parser.add_argument("--send",  action="store_true", help="Actually send to Telegram (default: dry-run)")
    parser.add_argument("--force", action="store_true", help="Clear dedup cache before running")
    parser.add_argument("--fn",    choices=list(RUNNERS.keys()), help="Run only one function")
    args = parser.parse_args()

    if args.send:
        enable_live_send()
        for _k, _v in _REAL_DEDUP.items():          # live: the real dedup state
            setattr(notifier, _k, _v)
    mode = "LIVE SEND to Telegram" if _live else "DRY-RUN (nothing leaves this machine)"
    print(f"\n{'#' * 60}")
    print(f"  Telegram notifier harness — {mode}")
    print(f"{'#' * 60}")

    if args.force:
        nf = Path(notifier.NOTIFIED_FILE)
        if nf.exists():
            backup = nf.with_suffix(".json.bak")
            backup.write_text(nf.read_text(encoding="utf-8"), encoding="utf-8")
            nf.write_text(json.dumps({"keys": []}, indent=2), encoding="utf-8")
            print(f"\n[force] Cleared {nf} (backup → {backup.name})")
        else:
            print(f"\n[force] {nf} didn't exist — nothing to clear")

    runners = {args.fn: RUNNERS[args.fn]} if args.fn else RUNNERS
    total = 0
    for name, fn in runners.items():
        before = len(_sent_messages)
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
        total += len(_sent_messages) - before

    print(f"\n{'#' * 60}")
    print(f"  Done — {total} message(s) {'sent to Telegram' if _live else 'previewed (dry-run)'}"
          + ("" if _live else f"; {len(_blocked_calls)} Telegram call(s) intercepted"))
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
