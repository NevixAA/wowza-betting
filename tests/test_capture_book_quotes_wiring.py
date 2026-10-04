"""Both capture scripts must retain per-book quotes the SAME way.

This estate's recurring failure is two copies of one idea drifting apart: the two feature
builders, `model_type_for_league` duplicated into v11, `COLLECT_SEASONS` frozen while
`PROP_SEASONS` rolled. These assert the std and new-format captures stay identical where it
matters, and that the wiring cannot silently drop data.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "standard": ROOT / "scripts" / "capture_std_sidemarket_odds_forward.py",
    "new_format": ROOT / "scripts" / "capture_nf_odds_forward.py",
}
WORKFLOWS = {
    "standard": ROOT / ".github" / "workflows" / "std_odds_capture.yml",
    "new_format": ROOT / ".github" / "workflows" / "nf_odds_capture.yml",
}


@pytest.mark.parametrize("track", sorted(SCRIPTS))
def test_both_scripts_gate_on_the_same_env_var(track):
    assert 'CAPTURE_BOOK_QUOTES = os.getenv("CAPTURE_BOOK_QUOTES"' in SCRIPTS[track].read_text(
        encoding="utf-8")


@pytest.mark.parametrize("track", sorted(SCRIPTS))
def test_the_bookmaker_filter_is_dropped_only_when_capture_is_on(track):
    """Default request must stay byte-for-byte what it always was."""
    s = SCRIPTS[track].read_text(encoding="utf-8")
    assert "if not want_raw:\n                params[\"bookmaker\"] = BET365" in s


@pytest.mark.parametrize("track", sorted(SCRIPTS))
def test_the_consensus_parser_still_keeps_bet365_only(track):
    """With the filter dropped the payload carries every book. If the parser did not filter
    internally, the consensus archive would silently change meaning."""
    s = SCRIPTS[track].read_text(encoding="utf-8")
    head = s[s.index("def _parse_all_odds"):]
    assert 'if bk.get("id") != BET365:' in head[:600]


@pytest.mark.parametrize("track", sorted(SCRIPTS))
def test_each_script_tags_its_own_model_type(track):
    """Invariant 1: the two tracks never mix. A mis-tagged quote would pool them."""
    assert f'model_type="{track}"' in SCRIPTS[track].read_text(encoding="utf-8")


@pytest.mark.parametrize("track", sorted(SCRIPTS))
def test_quotes_are_written_before_the_no_rows_early_return(track):
    """book_rows can be non-empty while rows is EMPTY -- the consensus parser keeps Bet365 only,
    so a fixture priced by eight other books yields no consensus rows and eight books of quotes.
    Returning first would discard exactly the fixtures per-book capture exists for."""
    s = SCRIPTS[track].read_text(encoding="utf-8")
    assert s.index("if CAPTURE_BOOK_QUOTES and book_rows:") < s.index("    if not rows:")


@pytest.mark.parametrize("track", sorted(WORKFLOWS))
def test_the_workflow_enables_capture(track):
    wf = yaml.safe_load(WORKFLOWS[track].read_text(encoding="utf-8"))
    envs = [st.get("env", {}) for job in wf["jobs"].values() for st in job["steps"]]
    assert any(str(e.get("CAPTURE_BOOK_QUOTES", "")) == "1" for e in envs)


@pytest.mark.parametrize("track", sorted(WORKFLOWS))
def test_the_workflow_stages_the_quotes_it_captures(track):
    """Capturing without committing is the 2026-08-15 failure: every step green, nothing saved."""
    assert "git add -f output/book_quotes" in WORKFLOWS[track].read_text(encoding="utf-8")
