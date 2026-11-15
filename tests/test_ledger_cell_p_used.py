"""The ledgers record the probability the bars_opt cell served (cell_p_used, 2026-11-10), read by wowza-exec to
re-check the SAME cell measure at the Cloudbet price. Missing stays blank, never 0."""
from __future__ import annotations

import math

from src import ledger as L


def test_both_ledgers_declare_the_appended_cell_columns_in_order():
    # cell_p_used (2026-11-10), then cell_status / qualifying_bar / edge_calibrated_pct (owner 2026-11-11)
    tail = ["cell_p_used", "cell_status", "qualifying_bar", "edge_calibrated_pct"]
    assert L.LEDGER_COLS[-4:] == tail and L.SIDE_LEDGER_COLS[-4:] == tail


def test_missing_cell_probability_stays_missing_never_zero():
    assert math.isnan(L._num(None)) and math.isnan(L._num("")) and L._num("0.61") == 0.61
