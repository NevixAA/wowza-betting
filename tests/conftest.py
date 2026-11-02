"""Shared test isolation (added 2026-10-25 with src/contracts).

Writers now call src.contracts.shadow.observe after they write, and observe records into
output/contract_violations.json. Point it at a per-test temp file so no test ever writes into the
real output/ directory.
"""
import pytest


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
