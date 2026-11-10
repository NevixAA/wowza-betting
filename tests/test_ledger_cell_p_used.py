"""The ledgers record the probability the bars_opt cell served (cell_p_used, 2026-11-10), read by wowza-exec to
re-check the SAME cell measure at the Cloudbet price. Missing stays blank, never 0."""
from __future__ import annotations

import math

from src import ledger as L


def test_both_ledgers_declare_cell_p_used_at_the_end():
    assert L.LEDGER_COLS[-1] == "cell_p_used"
    assert L.SIDE_LEDGER_COLS[-1] == "cell_p_used"


def test_missing_cell_probability_stays_missing_never_zero():
    assert math.isnan(L._num(None)) and math.isnan(L._num("")) and L._num("0.61") == 0.61
