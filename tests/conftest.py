"""Shared test isolation (added 2026-10-25 with src/contracts).

Writers now call src.contracts.shadow.observe after they write, and observe records into
output/contract_violations.json. Point it at a per-test temp file so no test ever writes into the
real output/ directory.

Since 2026-11-05 (Phase 2 KEY=ci) the run is also OFFLINE (every non-loopback socket is refused) and
the session FAILS if it changed any file git can see. See the block at the end.
"""
import hashlib
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_real_telegram(monkeypatch):
    """No test can reach Telegram (estate scan 2026-11-02, finding g): real credentials are removed
    from the environment, and any request to api.telegram.org that gets as far as the HTTP
    transport fails the test. Tests that exercise sending mock _send / requests.post above this."""
    import requests

    for k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    real_send = requests.sessions.Session.send

    def guarded(self, request, **kw):
        if "api.telegram.org" in str(getattr(request, "url", "")):
            raise AssertionError("a test tried to reach api.telegram.org - mock the transport")
        return real_send(self, request, **kw)
    monkeypatch.setattr(requests.sessions.Session, "send", guarded)
    yield


@pytest.fixture(autouse=True)
def _isolate_contract_violations(tmp_path, monkeypatch):
    monkeypatch.setenv("WOWZA_CONTRACT_VIOLATIONS", str(tmp_path / "contract_violations.json"))
    yield


# ── offline + clean tree (Phase 2 KEY=ci, 2026-11-05) ───────────────────────────────────────────
# A full run used to rewrite output/fpl_bootstrap.json, fpl_cache_meta.json and fpl_fixtures.json:
# a test reached the real FPL API through player_model.fpl_api's cache refresh. Tests run in CI with
# no API key and no network, so every non-loopback connection is refused here, and the session
# fails if the run changed any file git can see.
_LOOPBACK = ("127.", "::1", "localhost")


def _is_loopback(addr) -> bool:
    host = addr[0] if isinstance(addr, tuple) and addr else addr
    return isinstance(host, str) and (host.startswith(_LOOPBACK) or host == "")


_REAL_CONNECT = {}


def _refuse(addr):
    raise OSError(f"tests run offline: a test tried to connect to {addr!r} - "
                  "use a fixture or mock the client")


def _offline_connect(self, addr):
    if self.family == getattr(socket, "AF_UNIX", None) or _is_loopback(addr):
        return _REAL_CONNECT["connect"](self, addr)
    _refuse(addr)


def _offline_connect_ex(self, addr):
    if self.family == getattr(socket, "AF_UNIX", None) or _is_loopback(addr):
        return _REAL_CONNECT["connect_ex"](self, addr)
    _refuse(addr)


def pytest_configure(config):
    # Session-wide, not a function fixture: module-scoped fixtures (AppTest page renders) are set up
    # before any function-scoped autouse fixture and would run unguarded.
    if not _REAL_CONNECT:
        _REAL_CONNECT.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex)
        socket.socket.connect = _offline_connect
        socket.socket.connect_ex = _offline_connect_ex


def pytest_unconfigure(config):
    if _REAL_CONNECT:
        socket.socket.connect = _REAL_CONNECT.pop("connect")
        socket.socket.connect_ex = _REAL_CONNECT.pop("connect_ex")


def _tree_state():
    """`git status --porcelain` of the repo plus each tracked-and-modified file's content hash, or
    None when git is unavailable. Untracked files count too (a test must not create them)."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    state = {}
    for line in r.stdout.splitlines():
        path = line[3:].strip().strip('"')
        p = ROOT / path
        try:
            state[line] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
        except OSError:
            state[line] = "?"
    return state


def pytest_sessionstart(session):
    session.config._wowza_tree = _tree_state()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_wowza_tree", None)
    if before is None:
        return
    after = _tree_state()
    if after is None or after == before:
        return
    changed = sorted(set(after.items()) ^ set(before.items()))
    lines = sorted({k for k, _ in changed})
    msg = ("TEST HYGIENE: the run changed files in the repo tree (write to tmp_path, or monkeypatch "
           "config.OUTPUT_DIR):\n  " + "\n  ".join(lines))
    tr = session.config.pluginmanager.getplugin("terminalreporter")
    if tr is not None:
        tr.write_line(msg, red=True)
    else:
        print(msg)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
