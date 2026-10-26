"""Market Intelligence: LEGACY vs MARKET-FIRST migration comparison (market-first brief, 2026-10-26).

Pins that the page's comparison section shows exactly what output/edge_engine_comparison.json and
output/edge_engine_backtest.csv say, per track x market, with PAPER and DEPLOYABLE never pooled; that the
prospective counts are the shadow log's; that an H-EDGE2 status of CONFIRMED is never displayed as one; and
that the page renders headlessly (AppTest, absolute path, no network).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_data import market_intel as M

ROOT = Path(__file__).resolve().parents[1]
PAGE = next((ROOT / "pages").glob("21_*.py"))
CMP, BT, SHADOW = ROOT / "output" / "edge_engine_comparison.json", ROOT / "output" / "edge_engine_backtest.csv", \
    ROOT / "output" / "edge_engine_shadow.csv"
needs_artifacts = pytest.mark.skipif(not (CMP.exists() and BT.exists()), reason="comparison artifacts not present")

EDGE2 = {"generated_at": "2026-10-25T13:18:22Z", "forward_start": "2026-10-26T00:00:00Z",
         "registration_intact": True, "engine_hashes_match_registration": True, "confirmed_assignable_here": False,
         "production_changed": "NO", "status_counts": {"REGISTERED": 1, "ACCRUING": 1},
         "hypotheses": [
             {"id": "H-EDGE2-NF-OU25-01", "track": "new_format", "market": "OU25", "min_n_settled_two_sided": 400,
              "min_gameweeks": 8, "PRE_REGISTRATION_NOT_COUNTED": 99, "status": "REGISTERED", "progress_pct": 0.0,
              "forward": {"logged": 0, "two_sided": 0, "settled_two_sided": 0, "gameweeks": 0}},
             {"id": "H-EDGE2-NF-O15-01", "track": "new_format", "market": "OU15", "min_n_settled_two_sided": 150,
              "min_gameweeks": 8, "PRE_REGISTRATION_NOT_COUNTED": 99, "status": "ACCRUING", "progress_pct": 4.0,
              "forward": {"logged": 12, "two_sided": 7, "settled_two_sided": 6, "gameweeks": 1}}]}


@pytest.fixture(scope="module")
def mig():
    return M.migration_comparison()


@pytest.fixture(scope="module")
def cmp_json():
    return json.loads(CMP.read_text(encoding="utf-8"))


def _num(v):
    return np.nan if v is None else float(v)


@needs_artifacts
def test_new_engine_rows_equal_the_json(mig, cmp_json):
    e = mig["engines"]
    n_checked = 0
    for key, cell in cmp_json["cells"].items():
        track, fam = key.split("|")
        lab = M.cell_label(track, fam)
        for k, name in M.NEW_ENGINES:
            want = cell["engines_two_sided_universe"].get(k) or {"n": 0}
            got = e[(e["cell"] == lab) & (e["engine"] == name)]
            assert len(got) == 1, (lab, name)
            got = got.iloc[0]
            assert got["basis"] == "PAPER"                                 # market-first is shadow only
            assert got["n"] == int(want.get("n") or 0)
            ci = want.get("roi_ci95") or [None, None]
            for col, v in (("roi_pct", want.get("roi_pct")), ("roi_ci_lo", ci[0]), ("roi_ci_hi", ci[1]),
                           ("pnl_u", want.get("pnl_u")), ("max_drawdown_u", want.get("max_drawdown_u")),
                           ("clv_coverage_pct", want.get("clv_coverage_pct")), ("clv_mean_pct", want.get("clv_mean_pct")),
                           ("hit_pct", want.get("hit_pct"))):
                np.testing.assert_equal(_num(got[col]), _num(v), err_msg=f"{lab} {name} {col}")
            n_checked += 1
    assert n_checked == 3 * len(cmp_json["cells"])


@needs_artifacts
def test_legacy_split_is_never_pooled_and_reconciles_with_the_json(mig, cmp_json):
    e = mig["engines"]
    leg = e[e["engine"] == "LEGACY raw edge"]
    assert set(e["basis"]) <= {"DEPLOYABLE", "PAPER"}
    assert not e.duplicated(["cell", "engine", "basis"]).any()
    assert not e["cell"].str.contains("ALL|POOLED", case=False).any()
    for key, cell in cmp_json["cells"].items():
        lab = M.cell_label(*key.split("|"))
        part = leg[leg["cell"] == lab]
        assert set(part["basis"]) == {"DEPLOYABLE", "PAPER"}, lab
        want = cell["engines_two_sided_universe"]["LEGACY"]
        assert int(part["n"].sum()) == int(want["n"]), lab
        assert float(part["pnl_u"].fillna(0).sum()) == pytest.approx(float(want["pnl_u"]), abs=0.02), lab
    assert bool(mig["reconcile"]["reconciled"].all())


@needs_artifacts
def test_legacy_split_counts_from_the_raw_csv(mig):
    """Independent of edge_replay: n and P/L per (cell, basis) straight from the backtest rows."""
    b = pd.read_csv(BT, dtype=str, keep_default_na=False)
    policy = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
    b = b[(b["legacy_bet"] == "True") & (b["two_sided"] == "True") & (b["y_a"] != "")]
    b = b.assign(odds=pd.to_numeric(b["legacy_odds"], errors="coerce"), y=b["y_a"].astype(float))
    b = b[b["odds"] > 1]
    won = np.where(b["legacy_side"].isin(["OVER", "YES"]), b["y"], 1 - b["y"])
    b = b.assign(pnl=np.where(won == 1, b["odds"] - 1, -1.0),
                 basis=["DEPLOYABLE" if M.deployable(lg, t, f, tier, o, policy)[0] else "PAPER"
                        for lg, t, f, tier, o in zip(b["league"], b["track"], b["family"], b["legacy_tier"], b["odds"])])
    leg = mig["engines"][mig["engines"]["engine"] == "LEGACY raw edge"]
    for (t, f, basis), g in b.groupby(["track", "family", "basis"]):
        row = leg[(leg["cell"] == M.cell_label(t, f)) & (leg["basis"] == basis)].iloc[0]
        assert row["n"] == len(g)
        assert row["pnl_u"] == pytest.approx(round(g["pnl"].sum(), 2), abs=0.011)
    # standard tracks are PAPER in the policy: nothing on them can be DEPLOYABLE
    assert (leg[leg["track"] == "standard"].query("basis == 'DEPLOYABLE'")["n"] == 0).all()


@needs_artifacts
def test_probability_gain_and_agreement_equal_the_json(mig, cmp_json):
    for key, cell in cmp_json["cells"].items():
        track, fam = key.split("|")
        lab = M.cell_label(track, fam)
        for k, name in M.PROB_SOURCES:
            got = mig["probability"].query("cell == @lab and source == @name").iloc[0]
            want = cell["probability_quality_side_a"][k]
            for col in ("n", "brier", "logloss", "calibration_slope", "ece"):
                np.testing.assert_equal(_num(got[col]), _num(want.get(col)), err_msg=f"{lab} {k} {col}")
        g = mig["brier_gain"].query("cell == @lab").iloc[0]
        gw = cell["model_minus_market_brier_gain"]
        assert (g["brier_gain"], g["ci_lo"], g["ci_hi"]) == (gw["mean"], gw["ci95"][0], gw["ci95"][1])
        for k in ("NEW_EVLB", "NEW_QUALIFIED"):
            a = mig["agreement"].query("cell == @lab and new_engine == @k").iloc[0]
            w = cell["agreement_two_sided"][k]
            for col in ("legacy_only", "new_only", "both_same_side", "disagree_side", "neither"):
                assert a[col] == w[col], (lab, k, col)
            tot = sum(w.values())
            assert a["two_sided_rows"] == tot == cell["two_sided_at_entry"]
            assert a["agreement_rate"] == pytest.approx((w["both_same_side"] + w["neither"]) / tot, abs=1e-4)


def test_deployable_rule_follows_the_policy():
    pol = {"tracks": {"new_format|OU25": {"state": "ACTIVE_LOW", "base_fraction": 0.005, "max_odds": 2.5},
                      "new_format|BTTS": {"state": "ACTIVE", "base_fraction": 0.0125, "max_odds": 2.6},
                      "new_format|OVER15": {"state": "ACTIVE_LOW", "base_fraction": 0.005, "min_odds": 1.45,
                                            "max_odds": 1.9},
                      "standard|OU25": {"state": "PAPER", "base_fraction": 0.0}},
           "paper_leagues": ["USA MLS"], "portfolio": {"hard_max_odds": 3.0},
           "tier_multipliers": {"SNIPER": 1.1, "MARKSMAN": 0.9},
           "track_tier_multipliers": {"new_format|OU25": {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0}}}
    D = M.deployable
    assert D("Brazil Serie A", "new_format", "OU25", "VALUABLE", 2.0, pol)[0]          # OU25 stakes every tier
    assert not D("Brazil Serie A", "new_format", "BTTS", "VALUABLE", 2.0, pol)[0]      # tier not staked
    assert D("Brazil Serie A", "new_format", "BTTS", "SNIPER", 2.0, pol)[0]
    assert D("Championship", "standard", "OU25", "SNIPER", 2.0, pol) == (False, "track PAPER")
    assert D("USA MLS", "new_format", "OU25", "SNIPER", 2.0, pol) == (False, "paper league")
    assert D("Brazil Serie A", "new_format", "OU25", "SNIPER", 2.6, pol) == (False, "odds above track cap")
    assert D("Brazil Serie A", "new_format", "OU15", "SNIPER", 1.40, pol) == (False, "odds below track floor")
    assert D("Brazil Serie A", "new_format", "OU25", "SNIPER", 3.2, pol) == (False, "odds blocked")


@pytest.mark.skipif(not SHADOW.exists(), reason="shadow log not present")
def test_shadow_sample_equals_the_shadow_log():
    s = pd.read_csv(SHADOW, dtype=str, keep_default_na=False)
    meta, tab = M.shadow_sample(forward_start="2026-10-26T00:00:00Z")
    assert meta["rows"] == len(s) == int(tab["rows"].sum())
    ts = pd.to_datetime(s["prediction_ts"], utc=True, format="mixed")
    for (mt, mk), g in s.groupby(["model_type", "market"]):
        r = tab[tab["cell"] == M.cell_label(mt, mk)].iloc[0]
        assert r["rows"] == len(g)
        assert r["two_sided"] == int(g["market_source"].isin(M.TWO_SIDED_SOURCES).sum())
        assert r["since_forward_start"] == int((ts[g.index] >= pd.Timestamp("2026-10-26T00:00:00Z")).sum())
        for state, n in g["decision_state"].value_counts().items():
            assert r[f"state_{state}"] == n


def test_edge2_status_never_shows_confirmed():
    meta, tab = M.edge2_status(EDGE2)
    assert list(tab["status"]) == ["REGISTERED", "ACCRUING"] and meta["invalid_status"] == []
    assert list(tab["market"]) == ["OU25", "OVER15"] and tab["settled_two_sided"].tolist() == [0, 6]
    bad = json.loads(json.dumps(EDGE2))
    bad["hypotheses"][1]["status"] = "CONFIRMED"
    meta, tab = M.edge2_status(bad)
    assert "CONFIRMED" not in set(tab["status"]) and tab["status"].iloc[1] == "INVALID (CONFIRMED)"
    assert meta["invalid_status"] == ["H-EDGE2-NF-O15-01"]
    assert M.edge2_status({})[1].empty


def _pro_dir(tmp_path: Path, progress: dict) -> Path:
    d = tmp_path / "pro"
    (d / "output" / "edge_reassessment").mkdir(parents=True)
    (d / "output" / "edge_reassessment" / "edge2_forward_progress.json").write_text(json.dumps(progress),
                                                                                    encoding="utf-8")
    return d


def _render(monkeypatch, pro_candidates: list):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from dashboard_data import core, pro
    # Pages that importlib.reload(pro) while another test has core.reachable patched leave pro holding the
    # patched function after that test's undo: rebind pro to core's real helpers for this render.
    for name in ("reachable", "resolve_file", "read_json"):
        monkeypatch.setattr(pro, name, getattr(core, name))
    monkeypatch.setattr(core, "RAW_BASE", {})                       # never the network
    monkeypatch.setitem(core.REPOS, "pro", {**core.REPOS["pro"], "candidates": pro_candidates})
    monkeypatch.setitem(core.REPOS, "v11", {**core.REPOS["v11"], "candidates": []})
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    st.cache_data.clear()
    at = AppTest.from_file(str(PAGE.resolve()), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at) -> str:
    return "\n".join([x.value for x in at.markdown] + [x.value for x in at.caption] + [x.value for x in at.subheader])


@needs_artifacts
def test_page_renders_the_comparison_with_pro(monkeypatch, tmp_path, mig):
    pytest.importorskip("streamlit.testing.v1")
    at = _render(monkeypatch, [_pro_dir(tmp_path, EDGE2)])
    txt = _text(at)
    assert "◷ RETROSPECTIVE]" in txt and "▸ PROSPECTIVE]" in txt                    # the two banners
    assert txt.count("RETROSPECTIVE · REPLAY, NOT FORWARD EVIDENCE") == mig["engines"]["cell"].nunique()
    assert "Migration comparison — legacy raw edge vs market-first" in txt
    frames = [d.value for d in at.dataframe]
    # first comparison tab = first cell; its engine table is the module's rows for that cell
    first = mig["engines"]["cell"].iloc[0]
    want = mig["engines"][mig["engines"]["cell"] == first].drop(columns=["cell", "track", "market", "cell_status"])
    hit = [f for f in frames if list(f.columns) == list(want.columns)]
    assert hit, "engine table not rendered"
    pd.testing.assert_frame_equal(hit[0].reset_index(drop=True), want.reset_index(drop=True), check_dtype=False)
    assert len(hit) == mig["engines"]["cell"].nunique()             # one engine table per cell, never pooled
    e2 = [f for f in frames if "pre_registration_not_counted" in f.columns]
    assert e2 and list(e2[0]["status"]) == ["REGISTERED", "ACCRUING"]
    from dashboard_data import core
    assert core.repo_path("pro") == tmp_path / "pro"
    import streamlit as st
    st.cache_data.clear()


@needs_artifacts
def test_page_renders_offline_and_refuses_confirmed(monkeypatch, tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    at = _render(monkeypatch, [])
    assert "FEED NOT CONNECTED" in _text(at)
    bad = json.loads(json.dumps(EDGE2))
    bad["hypotheses"][0]["status"] = "CONFIRMED"
    at = _render(monkeypatch, [_pro_dir(tmp_path, bad)])
    assert any("CONFIRMED is never assignable" in e.value for e in at.error)
    e2 = [d.value for d in at.dataframe if "pre_registration_not_counted" in d.value.columns][0]
    assert "CONFIRMED" not in set(e2["status"])
    import streamlit as st
    st.cache_data.clear()
