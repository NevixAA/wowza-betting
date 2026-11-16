"""TOP-3 v7 tiers (src/research/top3_tiers.py, config/top3_tiers_protocol_v7.json): synthetic rows only."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.research import top3_tiers as T

ROOT = Path(__file__).resolve().parents[1]


def _pool():
    return pd.DataFrame({
        "track": ["s"] * 5, "league": ["L"] * 5,
        "mside": ["over25_over", "over15_over", "over35_over", "btts_yes", "over25_under"],
        "row_id": ["a", "b", "c", "d", "b"], "date": pd.to_datetime(["2026-01-05"] * 5),
        "odds": [1.9, 1.3, 3.1, 2.0, 2.6], "p_cal": [.55, .80, .35, .52, .45],
        "cal_ev": [.05, .04, .08, .04, .17], "pnl": [.9, .3, -1, 1, 1.6]})


def test_three_named_tiers_distinct_and_one_direction():
    t = T.pick_tiers(_pool(), {"s|L": {"odds_min": 2.5, "margin": 0.0}})
    assert list(t["tier"]) == list(T.TIERS)
    assert t.iloc[0]["row_id"] == "b" and t.iloc[0]["mside"] == "over25_under"     # highest cal_ev
    assert t.iloc[1]["row_id"] == "a"            # b's Over 1.5 is the opposite direction of b's Under 2.5
    assert t.iloc[2]["row_id"] == "c"            # odds >= 2.5
    assert len(set(zip(t["row_id"], t["market"]))) == 3


def test_no_threshold_no_t3_never_padded():
    t = T.pick_tiers(_pool(), {})
    assert "T3_RISKY_VALUE" not in set(t["tier"]) and len(t) == 2


def test_fit_t3_needs_min_dev_picks_no_fallback():
    fit = T.fit_t3(_pool())
    assert fit["s|L"]["threshold"] is None


def test_stake_by_proof():
    assert T.stake_label("PROMISING") == "STAKE" and T.stake_label("PROVEN") == "STAKE"
    assert T.stake_label("UNPROVEN") == "PAPER" and T.stake_label(None) == "PAPER"


def test_shortfall_causes():
    fx = pd.DataFrame({"track": ["s", "s"], "league": ["L", "M"], "row_id": ["a", "z"],
                       "date": pd.to_datetime(["2026-01-05", "2026-01-05"]), "has_form": [True, True]})
    slots = pd.DataFrame({"track": ["s"], "league": ["L"], "mside": ["over25_over"], "row_id": ["a"],
                          "date": pd.to_datetime(["2026-01-05"]), "model": [True], "price": [False],
                          "lam": [True], "value": [False]})
    sf = T.shortfall(fx, slots, pd.DataFrame(columns=["track", "league", "gw", "tier"]), {("s", "L"): 1},
                     ["2026-W02", "2026-W03"], {})
    c = {(r.league, r.gw): r.cause for r in sf.itertuples()}
    assert c[("L", "2026-W02")] == "NO_PRICE" and c[("L", "2026-W03")] == "NO_FIXTURES"
    assert c[("M", "2026-W02")] == "ALL_CELLS_OFF"


def test_protocol_pins_code():
    pr = json.loads((ROOT / "config" / "top3_tiers_protocol_v7.json").read_text(encoding="utf-8"))
    for k, f in (("sha256_lf", pr["code"]["file"]), ("script_sha256_lf", pr["code"]["script"])):
        assert hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == pr["code"][k]
