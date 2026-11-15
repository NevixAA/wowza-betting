"""scripts/cell_staking_from_proposal.py --set-fraction / --fraction-note / --record-approval (go-live rehearsal
2026-11-15, defect 5b). Runs against a COPY of config/bankroll_policy.json in tmp_path; the committed policy is
never written."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cell_staking_from_proposal as G  # noqa: E402

ARG_CELL = "new_format|BTTS|Argentina Primera Division"
NOTE = "owner test words: Argentina 0.5% from go-live"


def _proposal_path() -> Path:
    latest = json.loads((ROOT / "output" / "threshold_proposals" / "latest.json").read_text(encoding="utf-8"))
    return ROOT / latest["path"]


@pytest.fixture
def pol(tmp_path, monkeypatch):
    committed = ROOT / "config" / "bankroll_policy.json"
    before = hashlib.sha256(committed.read_bytes()).hexdigest()
    p = tmp_path / "bankroll_policy.json"
    p.write_bytes(committed.read_bytes())
    monkeypatch.setattr(G, "POLICY", p)
    yield p
    assert hashlib.sha256(committed.read_bytes()).hexdigest() == before


def _cells(p):
    return json.loads(p.read_text(encoding="utf-8"))["cell_staking"]


def test_defaults_unchanged(pol):
    before = _cells(pol)
    assert G.main(["--proposal", str(_proposal_path())]) == 0
    after = _cells(pol)
    assert after["cells"] == before["cells"] and after["approved"] == before["approved"]
    assert after["cells"][ARG_CELL]["base_fraction"] == 0.005          # owner 2026-11-15 "Reduce to 0.5%"


def test_set_fraction_records_owner_words(pol):
    state = _cells(pol)["cells"][ARG_CELL]["state"]
    assert G.main(["--proposal", str(_proposal_path()), "--set-fraction", f"{ARG_CELL}=0.004",
                   "--fraction-note", NOTE]) == 0
    c = _cells(pol)["cells"][ARG_CELL]
    assert c["base_fraction"] == 0.004 and c["base_fraction_source"] == NOTE
    assert c["state"] == state                                             # only the fraction changes


def test_set_fraction_dry_run_writes_nothing(pol):
    raw = pol.read_bytes()
    assert G.main(["--proposal", str(_proposal_path()), "--dry-run", "--set-fraction", f"{ARG_CELL}=0.005",
                   "--fraction-note", NOTE, "--record-approval"]) == 0
    assert pol.read_bytes() == raw


@pytest.mark.parametrize("args", [
    ["--set-fraction", f"{ARG_CELL}=0.005"],                                   # no owner words
    ["--set-fraction", "new_format|BTTS|Japan J-League=0.005", "--fraction-note", NOTE],   # never creates a stake
    ["--set-fraction", f"{ARG_CELL}=0", "--fraction-note", NOTE],
    ["--set-fraction", f"{ARG_CELL}=abc", "--fraction-note", NOTE],
])
def test_set_fraction_refusals_write_nothing(pol, args):
    raw = pol.read_bytes()
    assert G.main(["--proposal", str(_proposal_path()), *args]) == 2
    assert pol.read_bytes() == raw


def test_record_approval_writes_the_proposal_id_and_hash(pol):
    prop = json.loads(_proposal_path().read_text(encoding="utf-8"))
    assert G.main(["--proposal", str(_proposal_path()), "--record-approval"]) == 0
    assert _cells(pol)["approved"] == {"proposal_id": prop["proposal_id"], "proposal_hash": prop["proposal_hash"]}


def test_record_approval_refuses_an_edited_proposal(pol, tmp_path):
    prop = json.loads(_proposal_path().read_text(encoding="utf-8"))
    prop["proposal_hash"] = "0" * 64
    bad = tmp_path / "edited.json"
    bad.write_text(json.dumps(prop), encoding="utf-8")
    raw = pol.read_bytes()
    assert G.main(["--proposal", str(bad), "--record-approval"]) == 2
    assert pol.read_bytes() == raw
