"""The owner's FRESH HOLDOUT protocol (src/bars_opt/fresh_holdout.py, src/bars_opt/hist_study.run,
config/bars_fresh_holdout_protocol.json). Synthetic frames only; no store, no study on real data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import league_bars as LB  # noqa: E402
from src.bars_opt import fresh_holdout as FH  # noqa: E402
from src.bars_opt import hist_study as H  # noqa: E402
from src.bars_opt import study as S  # noqa: E402


def _family(spec):
    """spec: [(start, n_days, rows_per_day, fresh)] -> eligible-family frame with a fresh column."""
    rows = []
    for start, n_days, per, fresh in spec:
        for d in pd.date_range(start, periods=n_days, freq="7D"):
            for k in range(per):
                lg = ("A", "B")[k % 2]
                rows.append({"track": "t", "mside": "over15_over", "league": lg, "row_id": f"{d.date()}|{lg}|{k}",
                             "date": d, "p_side": 0.6, "fair": 0.55, "odds": 1.8, "odds_c": 1.8, "won": True,
                             "pnl": 0.8, "fresh": fresh})
    return pd.DataFrame(rows)


# ── the protocol file ────────────────────────────────────────────────────────────────────────────

def test_protocol_is_pre_registered_with_the_owner_rules():
    proto, sha = FH.protocol()
    assert len(sha) == 64
    r = proto["proven_rule"]
    assert r["min_holdout_bets"] == LB.MIN_HOLDOUT_BETS == 150
    assert r["P_RESOLUTION_HITS"] == S.P_RESOLUTION_HITS == 10
    assert "why_earlier_searches_do_not_enter" in r and "holm_family" in r
    assert FH.MIN_RUN_ROWS == 375
    assert "monthly" in proto["procedure"]["no_peek"]


# ── which rows are fresh ─────────────────────────────────────────────────────────────────────────

def test_only_history_priced_rows_no_earlier_run_saw_are_fresh():
    p = pd.DataFrame({"row_id": ["r1", "r2", "r3", "r4"], "market": ["over15"] * 4,
                      "price_source": ["oddsapi_hist", "oddsapi_hist", "own", None]})
    fresh = FH.mark_fresh(p, seen={("r2", "over15")})
    assert fresh.tolist() == [True, False, False, False]


def test_earlier_runs_cover_every_committed_universe_priced_row():
    from src.bars_opt import search_ledger as L
    try:
        seen, runs = FH.earlier_runs()
    except L.LedgerError as e:
        pytest.skip(str(e))
    u = pd.read_parquet(ROOT / FH.UNIVERSE_PATHS["new_format"], columns=["row_id", "market", "yes", "no"])
    k = u[u["yes"].notna() & u["no"].notna()]
    assert set(zip(k["row_id"], k["market"])) <= seen
    assert {r["kind"] for r in runs} >= {"universe"}


# ── the window ───────────────────────────────────────────────────────────────────────────────────

def test_most_recent_long_fresh_run_is_split_and_later_stragglers_are_skipped():
    f = _family([("2023-01-07", 20, 10, False),      # touched (own-priced earlier)
                 ("2023-06-03", 60, 10, True),       # 600 fresh rows: the run
                 ("2024-08-03", 10, 10, False),      # touched again
                 ("2025-01-04", 3, 10, True)])       # short later straggler: cannot host a test
    w = FH.holdout_window(f)
    assert w["status"] == "OK" and w["later_short_runs_skipped"] == 1
    assert w["run_start"] == pd.Timestamp("2023-06-03") and w["run_rows"] == 600
    sel, ho = FH.split_family(f, w)
    assert ho["fresh"].all() and ho["date"].min() >= w["holdout_start"] and ho["date"].max() <= w["run_end"]
    assert sel["date"].max() < w["holdout_start"]
    assert not set(sel["row_id"]) & set(ho["row_id"])
    assert (f["date"] > w["run_end"]).sum() > 0 and not ((sel["date"] > w["run_end"]).any() or (ho["date"] > w["run_end"]).any())
    share = ho["date"].nunique() / f[(f["date"] >= w["run_start"]) & (f["date"] <= w["run_end"])]["date"].nunique()
    assert abs(share - LB.IN_PERIOD_CHECK_SHARE) < 0.1


def test_one_touched_row_breaks_a_run():
    f = _family([("2023-06-03", 100, 10, True)])
    f.loc[f["date"] == pd.Timestamp("2023-06-03") + pd.Timedelta(days=7 * 30), "fresh"] = [True] * 9 + [False]
    w = FH.holdout_window(f)
    assert w["status"] == "OK" and w["run_rows"] == 690          # the later 69 dates only
    assert w["run_start"] == pd.Timestamp("2023-06-03") + pd.Timedelta(days=7 * 31)


def test_family_without_a_long_fresh_run_gets_no_historical_test():
    f = _family([("2023-01-07", 50, 10, False), ("2024-01-06", 30, 10, True)])
    w = FH.holdout_window(f)
    assert w["status"] == FH.NO_FRESH
    sel, ho = FH.split_family(f, w)
    assert sel.empty and ho.empty


def test_coverage_counts_cells_with_150_rows():
    f = _family([("2023-01-07", 20, 10, False), ("2023-06-03", 100, 10, True)])
    fam, lg = FH.coverage([{"track": "t", "mside": "over15_over", "frame": f, "window": FH.holdout_window(f)}])
    assert fam[0]["status"] == "OK" and fam[0]["cells_ge_150"] == sum(r["ge_150"] for r in lg)
    assert fam[0]["holdout_rows"] == sum(r["holdout_rows"] for r in lg)


# ── the single test: no peek, once, Holm counted before ─────────────────────────────────────────

def _bars():
    return {"mode": S.MODE_MODEL, "valuable_th": 0.02, "marksman_th": 0.04, "sniper_th": 0.06,
            "shrink_weight_at_valuable": 0.5, "wf_bets": 50, "wf_roi": 0.05, "wf_roi_shrunk": 0.04,
            "tier2": False, "tier2_bets": 0, "tier2_roi": None}


@pytest.fixture
def rigged(tmp_path, monkeypatch):
    from src import threshold_proposals as TP
    f = _family([("2023-01-07", 20, 10, False), ("2023-06-03", 100, 10, True)])
    fams = [{"track": "standard", "mside": "over15_over", "frame": f, "window": FH.holdout_window(f)},
            {"track": "standard", "mside": "over25_over", "frame": f.assign(fresh=False),
             "window": {"status": FH.NO_FRESH, "reason": "touched"}}]
    _, ho = FH.split_family(f, fams[0]["window"])
    priced = tmp_path / "priced"
    priced.mkdir()
    for t in ("standard", "new_format"):
        pd.DataFrame({"x": [1]}).to_parquet(priced / f"universe_{t}.parquet")
    log = {"select": [], "bet": []}

    def select_core(sel, seed=0, allow_blend=False):
        log["select"].append(sel.copy())
        return {"reality_check": {}, "inner_table": [], "chosen": "RAW|EDGE|max-|min-", "off": None,
                "bars": {"A": _bars(), "B": _bars() | {"off": "NO_POSITIVE_BAR"}}, "league_switch": []}

    def bet_with(sel, e, choice):
        log["bet"].append((sel.copy(), e.copy()))
        b = e.assign(tier2_ok=False)
        return b, {lg: choice["bars"][lg] for lg in choice["bars"]}

    monkeypatch.setattr(H, "gate", lambda store: (True, []))
    monkeypatch.setattr(FH, "protocol_committed", lambda: (True, ""))
    monkeypatch.setattr(FH, "earlier_runs", lambda: (set(), [{"kind": "universe"}]))
    monkeypatch.setattr(FH, "families", lambda priced, seen: fams)
    monkeypatch.setattr(S, "select_core", select_core)
    monkeypatch.setattr(S, "bet_with", bet_with)
    monkeypatch.setattr(S, "run_monthly", lambda *a, **k: (_ for _ in ()).throw(AssertionError("monthly peeks")))
    monkeypatch.setattr(S, "run_select", lambda *a, **k: (_ for _ in ()).throw(AssertionError("old SELECT_END path")))
    monkeypatch.setattr(TP, "PROPOSAL_DIR", tmp_path / "proposals")
    return {"tmp": tmp_path, "priced": priced, "ho": ho, "log": log}


def test_selection_never_sees_a_holdout_row_and_each_cell_is_tested_once(rigged):
    res = H.run(rigged["tmp"] / "store", rigged["priced"], rigged["tmp"] / "study", workers=1)
    log, ho = rigged["log"], rigged["ho"]
    assert len(log["select"]) == 1                                   # the NO_FRESH family is never selected
    sel = log["select"][0]
    assert not set(sel["row_id"]) & set(ho["row_id"]) and sel["date"].max() < ho["date"].min()
    assert len(log["bet"]) == 1                                      # one test per family
    _, e = log["bet"][0]
    assert set(e["row_id"]) <= set(ho["row_id"]) and set(e["league"]) == {"A"}   # OFF league B never scored
    assert res["holm_family"] == 1 and res["families_without_fresh_holdout"] == 1
    prop = json.loads(Path(res["proposal"]).read_text(encoding="utf-8"))
    assert prop["protocol_sha256"] == FH.protocol()[1] and prop["serving_change"] is False
    assert res["proposal_id"].startswith("TP-FH-")
    assert prop["cells"][0]["family_size"] == 1
    assert {x["mside"] for x in prop["no_fresh_holdout"]} == {"over25_over"}
    frozen = json.loads((rigged["tmp"] / "study" / "fresh_choices.json").read_text(encoding="utf-8"))
    assert frozen["holm_family"] == 1 and frozen["protocol_sha256"] == prop["protocol_sha256"]


def test_a_second_test_on_the_same_output_is_refused(rigged):
    H.run(rigged["tmp"] / "store", rigged["priced"], rigged["tmp"] / "study", workers=1)
    with pytest.raises(SystemExit, match="ONCE"):
        H.run(rigged["tmp"] / "store", rigged["priced"], rigged["tmp"] / "study", workers=1)


def test_an_uncommitted_protocol_blocks_the_run(rigged, monkeypatch):
    monkeypatch.setattr(FH, "protocol_committed", lambda: (False, "has uncommitted changes"))
    with pytest.raises(SystemExit, match="not pre-registered"):
        H.run(rigged["tmp"] / "store", rigged["priced"], rigged["tmp"] / "study", workers=1)


def test_selection_worker_refuses_a_holdout_row():
    f = _family([("2023-06-03", 10, 2, True)])
    with pytest.raises(AssertionError, match="holdout row"):
        H._select((f, {f["row_id"].iloc[0]}, "t", "m"))
