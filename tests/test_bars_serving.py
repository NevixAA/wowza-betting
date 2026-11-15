"""Serving of the optimized per-league bars (src/bars_opt/serve.py, src/bars_opt/cells.py; owner 2026-11-09).

    python -m pytest tests/test_bars_serving.py -q
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.bars_opt import cells as C
from src.bars_opt import serve as BS
from src.bars_opt import study as S

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "output" / "bars_opt"


def _synth(n=600, seed=0, leagues=("L1", "L2", "L3")):
    rng = np.random.default_rng(seed)
    fair = rng.uniform(0.3, 0.7, n)
    p = np.clip(fair + rng.normal(0, 0.05, n), 0.05, 0.95)
    won = rng.uniform(size=n) < fair
    odds = 1.0 / fair * 0.95
    return pd.DataFrame({"league": rng.choice(list(leagues), n), "p_side": p, "fair": fair, "odds": odds,
                         "won": won, "pnl": np.where(won, odds - 1, -1.0),
                         "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(n) // 5, unit="D")})


def _cell(**kw):
    c = {"mside": "btts_yes", "track": "new_format", "league": "L1", "month": "2026-11", "config": "RAW|EDGE|max-|min-",
         "config_basis": "SELECTED", "source": "RAW", "measure": "EDGE", "odds_max": None, "odds_min": None,
         "mode": "MODEL", "valuable_th": 0.02, "marksman_th": 0.04, "sniper_th": 0.06, "bar_basis": "OWN_BETS",
         "shrink_weight_at_valuable": 0.8, "tier2": False, "wf_roi_shrunk": 0.05, "source_params": {},
         "verdict": "UNPROVEN"}
    c.update(kw)
    return c


# ── one implementation: the serialized source equals the fitted source ──────────────────────────
@pytest.mark.parametrize("src", ["BLEND", "LEARNED", "SHRINK", "RAW"])
def test_source_params_rebuild_the_fitted_function(src):
    t = _synth()
    e = _synth(seed=1)
    params = S.source_params(src, t)
    json.dumps(params)                                                   # plain JSON, storable in a cell
    a = S.source_fn(src, params)(e)
    b = S.fit_source(src, t)(e)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
    if src in ("BLEND", "LEARNED"):                                      # equals sklearn's own predict_proba
        from sklearn.linear_model import LogisticRegression
        if src == "BLEND":
            X = np.c_[S._logit(t["p_side"]), S._logit(t["fair"])]
            m = LogisticRegression(penalty=None, max_iter=2000).fit(X, t["won"].astype(int))
            ref = m.predict_proba(np.c_[S._logit(e["p_side"]), S._logit(e["fair"])])[:, 1]
        else:
            lg = sorted(t["league"].unique())
            m = LogisticRegression(C=S.LEARNED_C, max_iter=3000).fit(S._learned_design(t, lg), t["won"].astype(int))
            ref = m.predict_proba(S._learned_design(e, lg))[:, 1]
        np.testing.assert_allclose(a, ref, atol=1e-9)


# ── parity with the study on REAL committed rows ─────────────────────────────────────────────────
def _family(track, mside):
    u = pd.read_parquet(UNIVERSE / f"universe_{track}.parquet")
    return S.family_frame(u, mside)


@pytest.mark.parametrize("track,mside,cfg", [("new_format", "btts_yes", ("LEARNED", "EV", 2.5, None)),
                                             ("new_format", "over25_under", ("SHRINK", "EDGE", None, 1.4)),
                                             ("standard", "over25_over", ("RAW", "EDGE", None, None)),
                                             ("new_format", "over25_over", ("NONE", "ALL", 2.5, None))])
def test_served_tier_equals_the_study_tier_on_real_rows(track, mside, cfg):
    if not (UNIVERSE / f"universe_{track}.parquet").exists():
        pytest.skip("committed universe missing")
    f = _family(track, mside)
    sel, e = f[f["date"] < S.SELECT_END], f[f["date"] >= S.SELECT_END]
    bars, scorer = S.fit_cfg(sel, cfg, allow_blend=True)
    mode = S.MODE_LW if cfg[0] == S.LEAGUE_WIDE_SOURCE else S.MODE_MODEL
    study = S.apply_bars(scorer(e), bars, mode)
    params = S.source_params(cfg[0], sel)
    served_n, checked = 0, 0
    for lg, b in bars.items():
        if "off" in b:
            continue
        sp = params if cfg[0] != "SHRINK" else {"weights": {lg: params["weights"][lg]}}
        cell = _cell(mside=mside, track=track, league=lg, config=S.cfg_name(cfg), source=cfg[0], measure=cfg[1],
                     odds_max=cfg[2], odds_min=cfg[3], mode=mode, valuable_th=b["valuable_th"],
                     marksman_th=b["marksman_th"], sniper_th=b["sniper_th"], tier2=b.get("tier2"),
                     wf_roi_shrunk=b.get("wf_roi_shrunk"), source_params=sp)
        assert BS.validate_cell(cell) == ""
        rows = e[e["league"] == lg]
        res = BS.score(rows, cell)
        # owner 2026-11-11: the served tip is QUALIFIED exactly on the rows the study kept (measure >= the
        # cell's one approved bar, valuable_th); the retired tier split is never served
        st = study[study["league"] == lg].set_index("row_id")["tier"]
        served = res["tier"].astype(str)
        got = dict(zip(rows.loc[served != "", "row_id"], served[served != ""]))
        assert set(got) == set(st.index), lg
        assert set(got.values()) <= {"QUALIFIED"}
        served_n += len(got)
        checked += len(rows)
    assert checked > 0


# ── per-cell rules ────────────────────────────────────────────────────────────────────────────────
def test_odds_range_is_the_cells_own():
    rows = pd.DataFrame({"league": "L1", "p_side": [0.60, 0.60, 0.60], "fair": [0.50, 0.50, 0.50],
                         "odds": [1.30, 1.90, 3.00]})
    res = BS.score(rows, _cell(odds_min=1.4, odds_max=2.5))
    assert list(res["reason"]) == [BS.R_RANGE, "", BS.R_RANGE]
    assert list(res["tier"]) == ["", "QUALIFIED", ""]


def test_league_wide_cell_tips_every_match_in_range_as_qualified():
    rows = pd.DataFrame({"league": "L1", "p_side": [0.10, 0.90], "fair": [0.50, 0.50], "odds": [1.9, 1.9]})
    res = BS.score(rows, _cell(source="NONE", measure="ALL", mode="LEAGUE_WIDE", config="NONE|ALL|max-|min-"))
    assert list(res["tier"]) == ["QUALIFIED", "QUALIFIED"] and set(res["bar_mode"]) == {"LEAGUE_WIDE"}


def test_no_fair_price_no_tier():
    rows = pd.DataFrame({"league": "L1", "p_side": [0.6], "fair": [np.nan], "odds": [1.9]})
    res = BS.score(rows, _cell())
    assert res["tier"].iloc[0] == "" and res["reason"].iloc[0] == BS.R_NO_FAIR


@pytest.mark.parametrize("bad", [{"valuable_th": 0.05, "marksman_th": 0.04}, {"source": "MAGIC"},
                                 {"source": "BLEND", "source_params": {}}, {"mode": "LEAGUE_WIDE"},
                                 {"sniper_th": float("nan")}])
def test_invalid_cells_are_refused(bad):
    assert BS.validate_cell(_cell(**bad)) != ""


def test_unapproved_cell_is_no_league_bar():
    lp = {"bars_opt": {"yes": _cell()}}
    cell, why = BS.served_cell(lp, "btts", "L1", "yes", applied=frozenset())
    assert cell is None and why.startswith("NO_LEAGUE_BAR") and "not applied" in why
    cell, why = BS.served_cell(lp, "btts", "L1", "yes", applied=frozenset({"btts|L1"}))
    assert cell is not None and why == ""
    cell, _ = BS.served_cell(lp, "btts", "L2", "yes", applied=frozenset({"btts|L2"}))
    assert cell is None                                                  # a cell serves only its own league


def test_learned_blend_weight_is_per_league_and_reported():
    t = _synth(n=900)
    t["edge"] = t["p_side"] - t["fair"]
    t["p_used"] = t["p_side"]
    thin = t[(t["league"] != "L3") | (t.index % 40 == 0)]                # L3 keeps only a handful of rows
    strict = S.league_bars(thin, False)
    blend = S.league_bars(thin, False, allow_blend=True)
    off = [lg for lg, b in strict.items() if b.get("off") == S.OFF_POOL]
    for lg in off:
        assert "off" not in blend[lg]
        assert blend[lg]["bar_basis"] == S.BASIS_BLEND
        assert 0.0 <= blend[lg]["shrink_weight_at_valuable"] <= 1.0


# ── O/U 2.5: separate Over and Under cells in evaluate_value ─────────────────────────────────────
def _ou_rows(league):
    base = {"league": league, "date": "2026-11-21", "home_team": "A", "away_team": "B",
            "home_scored_last5": 7, "away_scored_last5": 6, "home_conceded_last5": 5, "away_conceded_last5": 6}
    return pd.DataFrame([dict(base, p_over25=0.62, odds_over25=1.95, odds_under25=1.95),
                         dict(base, home_team="C", p_over25=0.38, odds_over25=1.95, odds_under25=1.95)])


def _install_ou(monkeypatch, entry, league="Serie B"):
    import src.betting as B
    import src.side_thresholds as ST
    monkeypatch.setattr(ST, "applied_entries", lambda path=None: frozenset({f"ou25|{league}"}))
    monkeypatch.setattr(B, "_STANDARD_TH_CACHE", {league: entry})


def test_ou25_over_and_under_are_served_by_their_own_cells(monkeypatch):
    from src.betting import evaluate_value
    lg = "Serie B"
    over = _cell(mside="over25_over", league=lg, valuable_th=0.05, marksman_th=0.08, sniper_th=0.11)
    under = _cell(mside="over25_under", league=lg, valuable_th=0.20, marksman_th=0.25, sniper_th=0.30)
    _install_ou(monkeypatch, {"bars_opt": {"over": over, "under": under}})
    df = evaluate_value(_ou_rows(lg))
    a, b = df.iloc[0], df.iloc[1]
    assert a["bet"] == "OVER" and a["signal_tier"] == "QUALIFIED"        # edge ~0.12 >= over bar (one bar per cell)
    assert b["bet"] == "AVOID" and b["signal_tier"] == "AVOID"            # under edge ~0.12 < under valuable 0.20
    assert a["bar_config_over"] == "RAW|EDGE|max-|min-" and a["threshold_source"] == "LEAGUE"
    assert not bool(a["min_odds_blocked"]) and not bool(a["max_odds_blocked"])


def test_ou25_league_with_only_an_over_cell_never_tips_under(monkeypatch):
    from src.betting import evaluate_value
    lg = "Serie B"
    _install_ou(monkeypatch, {"bars_opt": {"over": _cell(mside="over25_over", league=lg)}})
    df = evaluate_value(_ou_rows(lg))
    assert df.iloc[1]["bet"] == "AVOID" and df.iloc[0]["bet"] == "OVER"


def test_ou25_cell_odds_range_replaces_the_global_min_odds(monkeypatch):
    from src.betting import evaluate_value
    lg = "Serie B"
    _install_ou(monkeypatch, {"bars_opt": {"over": _cell(mside="over25_over", league=lg, odds_max=1.80)}})
    monkeypatch.setattr(config, "MIN_OVER_ODDS", 1.01)
    df = evaluate_value(_ou_rows(lg))
    assert df.iloc[0]["bet"] == "AVOID"                                  # 1.95 > the cell's own max 1.80


# ── side markets and TOP-3 read the same cells ────────────────────────────────────────────────────
def test_side_bets_use_the_cell(monkeypatch):
    import pipeline as P
    import src.side_thresholds as ST
    from src.integrity import side_edge_devig as SE
    lg = "Argentina Primera Division"
    cell = _cell(league=lg, source="NONE", measure="ALL", mode="LEAGUE_WIDE", config="NONE|ALL|max2.5|min-",
                 odds_max=2.5)
    side = {"btts": {lg: {"bars_opt": {"yes": cell}}}}
    monkeypatch.setattr(ST, "applied_entries", lambda path=None: frozenset({f"btts|{lg}"}))
    monkeypatch.setattr(P, "_load_side_market_thresholds", lambda: side)
    monkeypatch.setattr(SE, "load_quotes", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(SE, "production_fair_probs", lambda df, target, q, asof: pd.Series(0.5, index=df.index))
    rows = []
    for i, odds in enumerate((1.90, 2.80)):
        r = {"date": "2026-11-21", "league": lg, "home_team": f"H{i}", "away_team": f"A{i}", "model_type": "new_format",
             "home_scored_last5": 7, "away_scored_last5": 6, "home_conceded_last5": 5, "away_conceded_last5": 6}
        for m in ("btts", "over15", "over35"):
            r[f"p_{m}"], r[f"odds_{m}"] = (0.40, odds) if m == "btts" else (None, None)
        rows.append(r)
    out = P._generate_side_bets(pd.DataFrame(rows), config.SIDE_MARKETS)
    assert list(out["home_team"]) == ["H0"]                              # 2.80 is outside the cell's range
    assert out["signal_tier"].iloc[0] == "QUALIFIED" and out["bar_mode"].iloc[0] == "LEAGUE_WIDE"


def test_top3_qualifies_and_ranks_with_the_cell():
    from src import top3_league as T
    lg = "Brazil Serie A"
    cell = _cell(league=lg, source="RAW", measure="EV", config="RAW|EV|max-|min-", valuable_th=0.0,
                 marksman_th=0.05, sniper_th=0.10)
    ctx = {"standard": {}, "side": {"btts": {lg: {"bars_opt": {"yes": cell}}}}, "applied": {f"btts|{lg}"}}
    d = pd.DataFrame([{"league": lg, "market": "BTTS", "side": "YES", "p_model_raw": pm, "p_market_fair": 0.5,
                       "odds": 1.9, "price_quality_state": "TWO_SIDED_MULTI_BOOK"} for pm in (0.50, 0.60, 0.56)])
    q = T.qualify(d, ctx)
    assert list(q["tier"]) == ["", "QUALIFIED", "QUALIFIED"]            # below the bar: no tip, a reason
    assert list(q["reason"]) == ["BELOW_LEAGUE_BAR", "", ""]
    # owner 2026-11-11: the cell's p_used and fair travel with the tip for the T3 ranking (no EV rank)
    assert q["p_used"].iloc[1] > q["p_used"].iloc[2] and set(q["slot"].iloc[1:]) == {"TOP"}
    assert q["mside"].iloc[1] == "btts_yes"


# ── no global value anywhere in the serving adapter ───────────────────────────────────────────────
def test_adapter_reads_no_global_value():
    for mod in (BS, C):
        src = inspect.getsource(mod)
        code = src.split('"""', 2)[2]
        for banned in ("SIDE_GLOBAL_BARS", "MIN_OVER_ODDS", "MIN_UNDER_ODDS", "NF_MAX_OU_ODDS",
                       "VALUABLE_THRESHOLD", "MARKSMAN_THRESHOLD", "SNIPER_THRESHOLD", "1.08"):
            assert banned not in code, (mod.__name__, banned)


def test_production_candidates_shape():
    fam = {"cells": [_cell(mside="over25_over", league="Serie B", prod_market="ou25", side="over"),
                     _cell(mside="over25_under", league="Serie B", prod_market="ou25", side="under",
                           verdict="PROMISING"),
                     _cell(league="Brazil Serie A", prod_market="btts", side="yes")]}
    out = C.production_candidates([fam])
    assert set(out["ou25"]["Serie B"]["bars_opt"]) == {"over", "under"}
    assert out["btts"]["Brazil Serie A"]["bars_opt"]["yes"]["league"] == "Brazil Serie A"
    assert out["ou25"]["Serie B"]["approved"] is False                   # PROVEN only when every cell is
    from src import threshold_proposals as TP
    assert "bars_opt" in TP.SERVING_FIELDS and "valuable_th" in TP.SERVING_FIELDS


def test_blend_cell_without_own_bets_at_its_bar_is_off_never_a_pooled_stand_in():
    fam = {"track": "new_format", "mside": "btts_yes", "off": [],
           "cells": [_cell(league="L1", bar_basis=S.BASIS_BLEND, wf_bets=0, shrink_weight_at_valuable=0.0),
                     _cell(league="L2", bar_basis=S.BASIS_BLEND, wf_bets=5, shrink_weight_at_valuable=0.3),
                     _cell(league="L3", bar_basis="OWN_BETS", wf_bets=0)]}
    out = C.refilter([fam])[0]
    assert [c["league"] for c in out["cells"]] == ["L2", "L3"]
    assert out["off"] == [{"league": "L1", "reason": C.OFF_NO_OWN, "detail": out["off"][0]["detail"]}]
    assert C.OWN_VARIANCE_MIN == 2                                       # the same rule choose_bars uses
    assert "n\"] >= 2" in inspect.getsource(__import__("src.league_bars", fromlist=["x"]).choose_bars)
