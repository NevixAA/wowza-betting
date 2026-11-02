"""Estate scan 2026-11-02 finding (g): test_telegram.py's "dry run" really sent.

It patched only notifier._send; tips_table.send_photo, daily_pack and anything calling
requests.post directly went out for real, with the real token and chat. The harness is exercised
in a SUBPROCESS (its import patches requests/notifier process-wide), with a tripwire on the HTTP
adapter underneath: any byte that would leave the machine is recorded and fails the test."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "test_telegram.py"

_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import requests, requests.adapters
leaked = []
def tripwire(self, request, **kw):
    leaked.append(str(request.url))
    raise RuntimeError("network")
requests.adapters.HTTPAdapter.send = tripwire

import test_telegram as h
from telegram_bot import notifier, tips_table
out = {}
r = requests.post("https://api.telegram.org/botREAL_TOKEN_123/sendMessage",
                  data={"chat_id": "REAL_CHAT", "text": "x"}, timeout=5)
out["fake_status"] = r.status_code
out["send_ok"] = bool(notifier._send("t", "c", "hello"))
try:
    out["photo_ok"] = bool(tips_table.send_photo("t", "c", b"png", "cap"))
except Exception as e:
    out["photo_ok"] = "raised " + type(e).__name__
out["creds"] = notifier._load_config()
out["sandboxed"] = str(notifier.NOTIFIED_FILE).startswith(str(h._SANDBOX))
out["blocked"] = h._blocked_calls
out["previewed"] = h._sent_messages
import pytest  # noqa: F401  - a pytest process must never be able to go live
try:
    h.enable_live_send()
    out["live_refused"] = False
except RuntimeError:
    out["live_refused"] = True
out["leaked"] = leaked
print("JSON" + json.dumps(out))
"""


def _probe():
    env = {**os.environ, "TELEGRAM_TOKEN": "REAL_TOKEN_123", "TELEGRAM_CHAT_ID": "REAL_CHAT",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, "-c", _PROBE, str(ROOT)], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=300)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("JSON")), None)
    assert line, f"probe failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}"
    return json.loads(line[4:])


def test_dry_run_never_reaches_telegram():
    out = _probe()
    assert out["leaked"] == [], "a dry run put a request on the wire"
    assert out["fake_status"] == 200 and out["send_ok"] is True and out["photo_ok"] is True
    assert out["previewed"] == ["hello"]
    urls = [u for _, u in out["blocked"]]
    assert any("sendPhoto" in u for u in urls) and any("sendMessage" in u for u in urls)
    assert not any("REAL_TOKEN_123" in u for u in urls), "token must be redacted in the log"


def test_dry_run_uses_placeholder_credentials_and_sandboxed_dedup():
    out = _probe()
    assert out["creds"] == {"token": "DRY_RUN_TOKEN", "chat_id": "DRY_RUN_CHAT"}
    assert out["sandboxed"] is True
    assert out["live_refused"] is True


def test_harness_defines_no_pytest_tests():
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert not [n for n in names if n.startswith(("test", "Test"))], names


def test_conftest_blocks_telegram_for_every_test():
    import pytest
    import requests
    with pytest.raises(AssertionError, match="api.telegram.org"):
        requests.post("https://api.telegram.org/botX/sendMessage", data={"chat_id": "1"}, timeout=1)
    assert "TELEGRAM_TOKEN" not in os.environ and "TELEGRAM_CHAT_ID" not in os.environ
