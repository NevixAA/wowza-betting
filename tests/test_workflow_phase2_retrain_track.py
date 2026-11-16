"""phase2_retrain.yml runs only the dispatched track(s).

The new_format job needs about 10 GB peak in the comparison step and dies out of memory on
GitHub's 16 GB runner (run 42895611758, 2026-11-10), so New-Format is run locally. A dispatch
must therefore run standard alone by default and new_format only when the owner chooses it.
The owner gate (vars.PHASE2_RETRAIN_ENABLED) must stay on the job.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "phase2_retrain.yml"

# The only matrix expression shape this test knows how to evaluate: a JSON lookup table
# indexed by the input. Anything else fails loudly instead of being half-evaluated.
_LOOKUP = re.compile(r"^\$\{\{\s*fromJSON\('(?P<table>.*)'\)\[inputs\.track\]\s*\}\}$")


@pytest.fixture(scope="module")
def doc() -> dict:
    return yaml.safe_load(WF.read_text(encoding="utf-8"))


def _on(doc: dict) -> dict:
    # pyyaml (YAML 1.1) reads the bare key `on` as boolean True.
    return doc.get("on", doc.get(True))


def _track_input(doc: dict) -> dict:
    return _on(doc)["workflow_dispatch"]["inputs"]["track"]


def _matrix_for(doc: dict, track: str) -> list[str]:
    expr = doc["jobs"]["retrain"]["strategy"]["matrix"]["track"]
    m = _LOOKUP.match(str(expr).strip())
    assert m, f"unexpected matrix expression: {expr!r}"
    table = json.loads(m.group("table"))
    assert track in table, f"choice {track!r} has no matrix entry (would be null on GitHub)"
    return table[track]


def test_dispatch_only_no_other_triggers(doc):
    assert set(_on(doc)) == {"workflow_dispatch"}


def test_track_input_exists_with_standard_default(doc):
    inp = _track_input(doc)
    assert inp["type"] == "choice"
    assert inp["options"] == ["standard", "new_format", "both"]
    assert inp["default"] == "standard"


def test_every_choice_maps_to_exactly_its_tracks(doc):
    assert _matrix_for(doc, "standard") == ["standard"]
    assert _matrix_for(doc, "new_format") == ["new_format"]
    assert _matrix_for(doc, "both") == ["standard", "new_format"]


def test_new_format_runs_only_when_chosen(doc):
    for choice in _track_input(doc)["options"]:
        runs_nf = "new_format" in _matrix_for(doc, choice)
        assert runs_nf == (choice in ("new_format", "both")), choice
    assert "new_format" not in _matrix_for(doc, _track_input(doc)["default"])


def test_matrix_tracks_are_harness_tracks(doc):
    for choice in _track_input(doc)["options"]:
        for t in _matrix_for(doc, choice):
            assert t in ("standard", "new_format")


def test_owner_gate_unchanged(doc):
    job = doc["jobs"]["retrain"]
    assert job["if"] == "vars.PHASE2_RETRAIN_ENABLED == 'true'"
    assert job["permissions"] == {"contents": "read"}
    fit = next(s for s in job["steps"] if s.get("name", "").startswith("Fit and score"))
    assert fit["env"]["PHASE2_RETRAIN_ENABLED"] == "${{ vars.PHASE2_RETRAIN_ENABLED }}"
    assert '--tracks "${{ matrix.track }}"' in fit["run"]
