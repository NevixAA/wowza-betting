"""A preregistration that can be edited after seeing data is not a preregistration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REG = Path(__file__).resolve().parents[1] / "registry" / "preregistered_hypotheses.json"
D = json.loads(REG.read_text(encoding="utf-8"))
H = {h["id"]: h for h in D["hypotheses"]}

#: Values as registered BEFORE any confirmation data existed. If a later edit changes one of
#: these, the hypothesis is void and must be superseded instead -- this test is the tripwire.
FROZEN = {
    "H-BTTS-01": {"n_min": 150, "confirmation_starts": "2026-10-02",
                  "primary_metric": "ROI per bet, flat 1u, settled at entry price"},
    "H-BTTS-02": {"n_min": 150, "confirmation_starts": "2026-10-02",
                  "primary_metric": "ROI per bet, flat 1u"},
}


@pytest.mark.parametrize("hid,fields", FROZEN.items())
def test_an_open_hypothesis_is_never_edited(hid, fields):
    for k, v in fields.items():
        assert H[hid][k] == v, (
            f"{hid}.{k} changed after its confirmation window opened — that VOIDS the "
            f"hypothesis. Supersede it with a new entry instead.")


@pytest.mark.parametrize("hid", sorted(H))
def test_every_hypothesis_states_how_it_can_FAIL(hid):
    r = H[hid].get("rejection_criterion", "")
    assert r and len(r) > 20, f"{hid} has no falsifiable rejection criterion"


@pytest.mark.parametrize("hid", sorted(H))
def test_every_hypothesis_fixes_its_sample_size_and_start(hid):
    assert isinstance(H[hid].get("n_min"), int) and H[hid]["n_min"] >= 100
    assert H[hid].get("confirmation_starts")


def test_the_argentina_family_is_frozen():
    f = D["frozen_families"]["BTTS_ARGENTINA"]
    assert set(f["covers"]) <= set(H)
    assert "no threshold change" in f["rule"]


def test_the_btts_family_covers_roi_clv_and_calibration():
    """ROI alone cannot distinguish a real edge from luck. The family needs all three."""
    metrics = " ".join(H[h]["primary_metric"].lower() for h in
                       D["frozen_families"]["BTTS_ARGENTINA"]["covers"])
    assert "roi" in metrics and "clv" in metrics and "brier" in metrics


def test_argentina_and_ex_argentina_are_separate_hypotheses():
    """If one entry covered both, a win in Argentina would carry ex-Argentina with it."""
    assert "EXCEPT Argentina" in H["H-BTTS-02"]["definition"]
    assert "Argentina Primera Division" in H["H-BTTS-01"]["definition"]


def test_missing_market_data_is_excluded_not_zeroed():
    """Counting an unpriced selection as CLV=0 would drag every mean toward zero and make a
    thin sample look safe."""
    assert "excluded from n, not counted as zero" in H["H-BTTS-03"]["rejection_criterion"]


def test_two_months_in_one_window_are_not_independent():
    assert "NOT independent" in D["independence_rule"]


def test_a_multiple_comparison_policy_exists():
    assert "Benjamini-Hochberg" in D["multiple_comparisons_policy"]
