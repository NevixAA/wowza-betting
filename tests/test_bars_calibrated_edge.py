"""Pre-registered calibrated-edge bars study (src/bars_opt/calibrated_edge.py): skeleton pieces and the gate."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.bars_opt import calibrated_edge as C
from src.bars_opt import study as S

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_space_is_the_pre_registered_30():
    cf = C.configs()
    assert len(cf) == 30 and len(set(cf)) == 30
    model = [c for c in cf if c[0] != S.LEAGUE_WIDE_SOURCE]
    assert len(model) == 24 and {c[1] for c in model} == {C.CAL_MEASURE}
    assert {c[0] for c in model} == set(S.SOURCES)
    assert {c[1] for c in cf if c[0] == S.LEAGUE_WIDE_SOURCE} == {S.LEAGUE_WIDE_MEASURE}


def test_own_lambda_needs_own_rows_and_is_clipped():
    rng = np.random.default_rng(0)
    n = 200
    fair = rng.uniform(.3, .7, n)
    p = fair + rng.uniform(0, .2, n)
    won = rng.uniform(size=n) < fair + 0.3 * (p - fair)                 # true slope 0.3
    rows = pd.DataFrame({"fair": fair, "p_used": p, "won": won})
    lam = C.own_lambda(rows)
    assert 0 <= lam <= 1 and abs(lam - 0.3) < 0.35
    assert np.isnan(C.own_lambda(rows.head(C.MIN_OWN_ROWS - 1)))       # too few own rows: none, never filled
    assert C.own_lambda(rows.assign(won=True)) == 1.0                    # clipped


def test_league_lambdas_shrink_only_inside_the_league():
    rows = pd.DataFrame({"track": "t", "league": ["A"] * 120 + ["B"] * 60, "mside": ["m1"] * 60 + ["m2"] * 60 + ["m1"] * 60,
                         "fair": 0.5, "p_used": 0.6, "won": [True] * 120 + [False] * 60})
    lam = C.league_lambdas(rows)
    assert lam[("t", "B", "m1")] == 0.0 and lam[("t", "A", "m1")] == 1.0   # league B never pulled toward A


def test_cal_edge_is_lambda_times_served_gap_and_missing_stays_missing():
    e = C.cal_edge([0.6, 0.6], [0.5, 0.5], [0.5, np.nan])
    assert np.isclose(e[0], 0.05) and np.isnan(e[1])


def test_gate_refuses_today_and_run_never_runs():
    why = C.gate("2026-12")
    assert any("not enabled" in w for w in why)
    assert any("before the pre-registered first month" in w for w in C.gate("2026-11"))
    with pytest.raises(SystemExit):
        C.run("2026-12")


def test_protocol_is_pinned_and_states_the_evaluation_rows():
    flag = json.loads((ROOT / C.FLAG_FILE).read_text(encoding="utf-8"))
    assert flag["run_enabled"] is False
    assert C.sha256_lf(ROOT / C.PROTOCOL_FILE) == flag["protocol_sha256"]
    proto = json.loads((ROOT / C.PROTOCOL_FILE).read_text(encoding="utf-8"))
    assert proto["code"]["sha256_lf"] == C.sha256_lf(ROOT / "src/bars_opt/calibrated_edge.py")
    assert "2026-12-01" in proto["evaluation"]["rows_that_count"] and "no upper edge" in proto["machinery"]["no_cap"]
    assert proto["run"]["date"].startswith("2026-12-01")


def test_a_changed_protocol_is_refused(tmp_path):
    for rel in (C.FLAG_FILE, C.PROTOCOL_FILE):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    with open(tmp_path / C.PROTOCOL_FILE, "a", encoding="utf-8") as f:
        f.write(" ")
    assert any("not the pinned version" in w for w in C.gate("2026-12", tmp_path))
