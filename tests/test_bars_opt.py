"""Bars optimization study (src/bars_opt, reports/P4_BARS_OPTIMIZATION.md) on synthetic data only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.bars_opt import study as S


def _frame(n_days=420, leagues=("A", "B", "C"), regime=None, seed=0, start="2025-06-01"):
    """Synthetic family frame. `regime` = a league where EVERY match wins more often than the fair price."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for d in days:
        for lg in leagues:
            fair = float(rng.uniform(0.35, 0.65))
            p = float(np.clip(fair + rng.normal(0, 0.05), 0.05, 0.95))
            true = min(0.95, fair + (0.12 if lg == regime else 0.0))
            won = bool(rng.random() < true)
            odds = round(1 / fair * 0.97, 3)
            rows.append({"track": "t", "mside": "btts_yes", "league": lg, "row_id": f"{d.date()}|{lg}|h|a",
                         "date": d, "p_side": p, "fair": fair, "odds": odds, "odds_c": odds, "won": won,
                         "pnl": odds - 1.0 if won else -1.0})
    return pd.DataFrame(rows)


def test_configuration_count_is_the_preregistered_54():
    c = S.configs()
    assert len(c) == 54 == 4 * 2 * 6 + 6
    assert len({S.cfg_name(x) for x in c}) == 54
    assert all(S.parse_cfg(S.cfg_name(x)) == x for x in c)


def test_select_never_reads_the_final_window():
    f = _frame(n_days=520, start="2025-06-01")
    assert f["date"].max() >= S.FINAL_START
    a = S.select_core(S.select_rows(f, S.SELECT_END))
    g = f.copy()
    late = g["date"] >= S.FINAL_START
    g.loc[late, "won"] = ~g.loc[late, "won"]                      # scramble every FINAL outcome
    g.loc[late, "pnl"] = np.where(g.loc[late, "won"], g.loc[late, "odds"] - 1, -1.0)
    b = S.select_core(S.select_rows(g, S.SELECT_END))
    assert json.dumps(a, default=str) == json.dumps(b, default=str)


def test_bars_are_ordered_per_league_and_never_a_pooled_stand_in():
    f = _frame()
    thin = f[f["league"] == "C"].head(10)                        # league C: 10 rows only
    f = pd.concat([f[f["league"] != "C"], thin], ignore_index=True)
    bars, _ = S.fit_cfg(f, ("RAW", "EDGE", None, None))
    for lg, b in bars.items():
        if "off" in b:
            continue
        assert b["valuable_th"] <= b["marksman_th"] <= b["sniper_th"]
        assert b["bar_basis"] == "OWN_BETS"
    assert "off" in bars["C"]                                     # no global / pooled bar served to C


def test_league_wide_regime_is_found_and_labelled():
    f = _frame(n_days=360, regime="B", seed=3)
    core = S.select_core(f)
    assert core["chosen"] is not None
    b = core["bars"]["B"]
    assert "off" not in b and b["mode"] == S.MODE_LW              # the regime league is served LEAGUE_WIDE
    for lg in ("A", "C"):                                         # no regime, no model skill: never LEAGUE_WIDE
        assert "off" in core["bars"][lg] or core["bars"][lg]["mode"] == S.MODE_MODEL


def test_league_wide_source_bets_every_eligible_match():
    f = _frame(n_days=200, regime="A", seed=5)
    bars, scorer = S.fit_cfg(f, ("NONE", "ALL", None, None))
    assert "off" not in bars["A"]
    out = S.apply_bars(scorer(f[f["league"] == "A"]), {"A": bars["A"]}, S.MODE_LW)
    assert len(out) == int((f["league"] == "A").sum())
    assert (out["tier"] == S.MODE_LW).all()
    assert np.allclose(out["ev_est"], bars["A"]["wf_roi_shrunk"])


def test_tier2_requires_its_own_positive_band_evidence():
    f = _frame(seed=7)
    bars, _ = S.fit_cfg(f, ("RAW", "EDGE", None, None))
    for b in bars.values():
        if "off" in b:
            continue
        if b["tier2"]:
            assert b["tier2_bets"] >= S.MIN_OWN_BETS and b["tier2_roi"] > 0
        if b["marksman_th"] == b["valuable_th"]:
            assert b["tier2"] is False and b["tier2_bets"] == 0


def _bets(rows):
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"])
    for k, v in {"track": "t", "pnl": 0.5, "odds": 2.0, "odds_c": 2.0, "tier2_ok": False}.items():
        if k not in d:
            d[k] = v
    return d


def test_top3_caps_at_three_never_pads_and_respects_direction():
    rows = [{"league": "A", "date": "2026-09-07", "row_id": f"f{i}", "mside": "btts_yes", "ev_est": 0.1 + i / 100,
             "band": "TOP_M"} for i in range(5)]
    rows += [{"league": "B", "date": "2026-09-07", "row_id": "g1", "mside": "btts_yes", "ev_est": 0.2, "band": "TOP_M"}]
    rows += [{"league": "A", "date": "2026-09-08", "row_id": "f4", "mside": "over25_under", "ev_est": 0.9, "band": "TOP_M"}]
    t = S.top3(_bets(rows), "TOP_ONLY_V")
    assert t.groupby(["league", "gw"]).size().max() == 3
    assert (t["league"] == "B").sum() == 1                        # never padded to 3
    a = t[t["league"] == "A"]
    assert not ({"f4"} <= set(a["row_id"]) and a[a["row_id"] == "f4"]["mside"].nunique() > 1)


def test_top3_tier2_fills_only_after_every_top_bet():
    rows = [{"league": "A", "date": "2026-09-07", "row_id": "t1", "mside": "btts_yes", "ev_est": 0.01, "band": "TOP_M"},
            {"league": "A", "date": "2026-09-07", "row_id": "t2", "mside": "btts_yes", "ev_est": 0.50, "band": "BAND_VM",
             "tier2_ok": True},
            {"league": "A", "date": "2026-09-07", "row_id": "t3", "mside": "btts_yes", "ev_est": 0.60, "band": "BAND_VM",
             "tier2_ok": False}]
    t = S.top3(_bets(rows), "TOP_M_PLUS_TIER2")
    assert list(t["row_id"]) == ["t1", "t2"]                      # TOP first even at lower EV; t3 not TIER2-eligible
    assert list(t["slot"]) == ["TOP", "TIER2"]


def test_final_refuses_a_choice_that_differs_from_the_refit():
    f = _frame(n_days=520, start="2025-06-01", seed=11, regime="A")
    sel = S.select_rows(f, S.SELECT_END)
    core = json.loads(json.dumps(S.select_core(sel), default=str))
    if not core["chosen"]:
        pytest.skip("synthetic family produced no choice")
    choice = {"track": "t", "mside": "btts_yes", "select_leagues": sorted(sel["league"].unique()),
              "baseline_bars": {}} | core
    S.final_family((f, choice))                                   # the committed choice reproduces
    lg = next(k for k, v in choice["bars"].items() if "off" not in v)
    choice["bars"][lg]["valuable_th"] = 0.29
    with pytest.raises(RuntimeError):
        S.final_family((f, choice))


def test_monthly_proposal_writes_only_its_own_folder(tmp_path, monkeypatch):
    f = _frame(n_days=300, seed=13)
    monkeypatch.setattr(S, "load_families", lambda _u: [(f, "t", "btts_yes")])
    res = S.monthly_proposal(tmp_path / "u", "2026-04", tmp_path / "out")
    files = list((tmp_path / "out").iterdir())
    assert [p.name for p in files] == ["bars_opt_2026-04.json"]
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert res["entries"] == len(data)
    for e in data:
        assert "off" in e or e["valuable_th"] <= e["marksman_th"] <= e["sniper_th"]


def test_module_has_no_global_bar_and_no_typed_default():
    import inspect
    src = inspect.getsource(S)
    assert "SIDE_GLOBAL_BARS" not in src and "GLOBAL" not in src.replace("global or pooled", "")
    assert "fillna(" not in src
