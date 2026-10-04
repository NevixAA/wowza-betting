"""The v9.1 gate runs ALONGSIDE the live one and must never be able to affect it.

The whole value of a log-only wiring is that it is provably inert. These tests check the two
ways it could stop being inert: by raising, and by being read as a decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline


class _Boom:
    """A payload whose every use raises — stands in for any breakage inside the shadow gate."""
    def __getitem__(self, k):  raise RuntimeError("synthetic failure")
    def get(self, *a, **k):    raise RuntimeError("synthetic failure")


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"over25": rng.integers(0, 2, n).astype(float),
                         "league": rng.choice(["A", "B"], n)})


def test_a_failing_shadow_gate_never_raises():
    """A measurement must not be able to break a retrain. It returns, it does not propagate."""
    out = pipeline._shadow_gate("T", _Boom(), {}, _frame(), "over25", True, "live reason")
    assert out["available"] is False


def test_it_is_skipped_rather_than_guessed_on_a_tiny_holdout():
    out = pipeline._shadow_gate("T", object(), {}, _frame(10), "over25", True, "r")
    assert out["available"] is False


def test_no_holdout_at_all_is_handled():
    assert pipeline._shadow_gate("T", object(), {}, None, "over25", True, "r")["available"] is False


def test_the_live_decision_is_passed_in_not_derived():
    """The signature takes the live verdict as an ARGUMENT. It is recorded, never recomputed —
    which is what makes it structurally impossible for this function to change one."""
    import inspect
    sig = inspect.signature(pipeline._shadow_gate)
    assert "promoted_by_live_gate" in sig.parameters


def test_shadow_gate_is_called_after_the_promotion_decision_is_already_made():
    """Order matters: if it ran BEFORE `promote` was set, a future edit could feed it back in."""
    src = inspect_source()
    i_decide = src.index("if promote:\n        save_models(")
    i_shadow = src.index("shadow = _shadow_gate(")
    assert i_decide < i_shadow, "the shadow gate must run after the model is already saved"


def inspect_source() -> str:
    import inspect
    return inspect.getsource(pipeline)


def test_shadow_result_is_recorded_but_not_consulted():
    """`v91_gate` appears in the retrain record and nowhere in any branch."""
    src = inspect_source()
    assert '"v91_gate": shadow' in src
    for bad in ("if shadow", "shadow[", "shadow.get", "and shadow", "or shadow"):
        assert bad not in src, f"the shadow verdict is being read as a decision: {bad}"
