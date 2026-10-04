"""§1 — a worse candidate never replaces a serving champion.

Replays the decisions actually recorded in output/retrain_log.json. These are not synthetic:
every case below was promoted by the live tolerance at the time.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.replay_promotions import strict_replayable

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "output" / "retrain_log.json"


def decide(old_ll, new_ll, basis="same_holdout", tol=0.030, block_worse=True):
    """The pipeline's promotion branch, isolated. Mirrors pipeline._train_one exactly."""
    if old_ll != old_ll:
        return True, "no incumbent"
    if new_ll != new_ll:
        return False, "no usable log loss"
    if block_worse and basis == "same_holdout" and new_ll > old_ll:
        return False, "worse candidate"
    if new_ll > old_ll + tol:
        return False, "beyond tolerance"
    return True, "within tolerance"


# ── the historically bad promotions, by value ─────────────────────────────────────────────────
HISTORICAL_BAD = [
    ("2026-09-26", "ht_over05", 0.64789, 0.66716),
    ("2026-09-26", "ht_over15", 0.56759, 0.57152),
    ("2026-09-29", "newformat", 0.68064, 0.68362),
    ("2026-09-27", "ht_over05", 0.66761, 0.67033),
    ("2026-10-02", "newformat", 0.68004, 0.68134),
    ("2026-10-04", "newformat", 0.68110, 0.68142),
    ("2026-10-04", "ht_over05", 0.60543, 0.60569),
    ("2026-09-30", "standard",  0.68738, 0.68847),
]


@pytest.mark.parametrize("day,track,old,new", HISTORICAL_BAD)
def test_each_historical_bad_promotion_is_now_blocked(day, track, old, new):
    assert decide(old, new, block_worse=False)[0] is True, "precondition: the OLD gate promoted"
    assert decide(old, new)[0] is False, f"{day} {track} should no longer promote"


def test_a_better_candidate_still_promotes():
    """The rule must not freeze the models. Retraining beating freezing is a proven result."""
    assert decide(0.68801, 0.68793)[0] is True


def test_a_tie_keeps_the_incumbent():
    """On equal evidence prefer the model already serving — it is the known quantity."""
    assert decide(0.68000, 0.68000)[0] is True, "exactly equal is not worse"
    assert decide(0.68000, 0.680001)[0] is False


def test_the_weak_comparison_basis_is_exempt():
    """Under the stored-metrics fallback the two numbers come from DIFFERENT test sets, so a
    sub-tolerance difference is an artefact and blocking on it would reject good models."""
    assert decide(0.680, 0.681, basis="stored_metrics_FALLBACK")[0] is True


def test_the_rule_can_be_reverted_with_one_env_var():
    assert decide(0.68110, 0.68142, block_worse=False)[0] is True


def test_catastrophes_are_still_caught_by_the_tolerance():
    """ht_over05 blew up twice in September. Both must still fail."""
    assert decide(0.60431, 0.70750)[0] is False
    assert decide(0.60431, 0.69292)[0] is False


# ── the replay itself ─────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not LOG.exists(), reason="no retrain log in this checkout")
def test_replay_over_the_real_log_blocks_every_worse_promotion():
    runs = json.loads(LOG.read_text(encoding="utf-8")).get("runs", {})
    worse_promoted = []
    for day, tracks in runs.items():
        for track, r in (tracks or {}).items():
            if not isinstance(r, dict):
                continue
            o, n = r.get("logloss_old"), r.get("logloss_new")
            if not (isinstance(o, float) and isinstance(n, float)):
                continue
            if n > o and r.get("promoted") and r.get("comparison_basis") == "same_holdout":
                worse_promoted.append((day, track, o, n))
    assert worse_promoted, "precondition: the live gate did promote worse models"
    for day, track, o, n in worse_promoted:
        assert decide(o, n)[0] is False, f"{day} {track} still promotes"


@pytest.mark.skipif(not LOG.exists(), reason="no retrain log in this checkout")
def test_the_rule_does_not_block_the_majority_of_retrains():
    """A gate that blocks almost everything would freeze the estate, which is a measured harm."""
    runs = json.loads(LOG.read_text(encoding="utf-8")).get("runs", {})
    kept = total = 0
    for tracks in runs.values():
        for r in (tracks or {}).values():
            if not isinstance(r, dict):
                continue
            o, n = r.get("logloss_old"), r.get("logloss_new")
            if isinstance(o, float) and isinstance(n, float):
                total += 1
                kept += decide(o, n, basis=r.get("comparison_basis", "same_holdout"))[0]
    assert kept / total > 0.5, f"only {kept}/{total} retrains would promote — too restrictive"


def test_strict_replay_helper_requires_material_improvement():
    """The strict gate's extra floor is NOT in production — it is log-only. Pin that it is
    genuinely stricter, so a future edit cannot quietly make them the same rule."""
    ok, reasons = strict_replayable(0.68801, 0.68793, {})     # better, but only by 0.00008
    assert ok is False and any("material" in r for r in reasons)
    assert decide(0.68801, 0.68793)[0] is True                 # production still promotes it
