"""Shared test isolation (added 2026-10-25 with src/contracts).

Writers now call src.contracts.shadow.observe after they write, and observe records into
output/contract_violations.json. Point it at a per-test temp file so no test ever writes into the
real output/ directory.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_contract_violations(tmp_path, monkeypatch):
    monkeypatch.setenv("WOWZA_CONTRACT_VIOLATIONS", str(tmp_path / "contract_violations.json"))
    yield
