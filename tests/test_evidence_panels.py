"""Evidence panels (owner brief 2026-10-26, part D) — DISPLAY ONLY.

Pins: each panel's numbers equal its source artifact (or an independent recomputation of the same rule),
the "NO POLICY CHANGE" statement is on the Bankroll page, the Bankroll page cannot write a policy, the
mirrored constants agree with the production modules they copy, and the three pages render headlessly
(AppTest, ABSOLUTE paths, Pro / V11 unreachable unless a fake Pro checkout is given).
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_data import evidence_panels as EP

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "pages"
BANKROLL = next(PAGES.glob("16_*.py"))
MODEL_LAB = next(PAGES.glob("20_*.py"))
MARKET_INTEL = next(PAGES.glob("21_*.py"))


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────────
def _offline(monkeypatch, pro_dir: Path | None = None):
    """Pro / V11 unreachable over HTTP; Pro optionally a fake local checkout."""
    from dashboard_data import core
    monkeypatch.setattr(core, "RAW_BASE", {})
    monkeypatch.setitem(core.REPOS, "pro", {**core.REPOS["pro"], "candidates": [pro_dir] if pro_dir else []})
    monkeypatch.setitem(core.REPOS, "v11", {**core.REPOS["v11"], "candidates": []})


def _render(page: Path, monkeypatch, timeout: int = 180):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    st.cache_data.clear()
    at = AppTest.from_file(str(page.resolve()), default_timeout=timeout).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at) -> str:
    return "\n".join([x.value for x in at.markdown] + [x.value for x in at.caption] + [x.value for x in at.subheader])


def _ledgers():
    """Synthetic main + side ledgers in the live column layout."""
    main = pd.DataFrame([
        # date, league, side, tier, odds, result, pnl, clv_pct, model_type, source
        ("2026-10-10", "Argentina Primera Division", "OVER", "SNIPER", 2.0, "WIN", 1.0, 2.0, "new_format", "live"),
        ("2026-10-11", "Argentina Primera Division", "UNDER", "VALUABLE", 1.9, "LOSS", -1.0, -1.0, "new_format", "live"),
        ("2026-10-27", "Brazil Serie A", "OVER", "MARKSMAN", 2.2, "LOSS", -1.0, 0.5, "new_format", "live"),
        ("2026-10-28", "Brazil Serie A", "OVER", "SNIPER", 2.1, "WIN", 1.1, np.nan, "new_format", "live"),
        ("2026-10-28", "USA MLS", "OVER", "SNIPER", 2.1, "WIN", 1.1, 1.0, "new_format", "live"),       # paper league
        ("2026-10-28", "Brazil Serie A", "UNDER", "SNIPER", 2.1, "WIN", 1.1, 1.0, "new_format", "backfill"),
        ("2026-10-29", "Brazil Serie A", "UNDER", "SNIPER", 2.0, None, None, None, "new_format", "live"),  # unsettled
    ], columns=["match_date", "league", "side", "signal_tier", "odds", "result", "pnl", "clv_pct", "model_type",
                "source"])
    main["home_team"], main["away_team"] = "H", "A"
    side = pd.DataFrame([
        ("2026-10-20", "Liga MX", "btts", "SNIPER", 1.9, "WIN", 0.9, 1.5, "new_format", "live", ""),
        ("2026-10-27", "Liga MX", "btts", "VALUABLE", 1.9, "LOSS", -1.0, 0.0, "new_format", "live", ""),
        ("2026-10-27", "Liga MX", "over15", "SNIPER", 1.55, "WIN", 0.55, 2.0, "new_format", "live", ""),
        ("2026-10-27", "Liga MX", "over15", "MARKSMAN", 1.40, "WIN", 0.40, 1.0, "new_format", "live", ""),  # fill price
        ("2026-10-28", "Liga MX", "over15", "VALUABLE", 1.6, "LOSS", -1.0, np.nan, "new_format", "live", ""),
        ("2026-10-28", "Liga MX", "over15", "SNIPER", np.nan, "WIN", 0.5, 1.0, "new_format", "live", ""),   # no price
        ("2026-10-28", "Liga MX", "over15", "SNIPER", 1.5, "LOSS", -1.0, 1.0, "standard", "live", ""),       # other track
    ], columns=["match_date", "league", "market", "signal_tier", "odds", "result", "pnl", "clv_pct", "model_type",
                "source", "notes"])
    side["home_team"], side["away_team"] = "H", "A"
    return main, side


POLICY = {"tracks": {"new_format|OU25_OVER": {"state": "ACTIVE_LOW"}, "new_format|OU25_UNDER": {"state": "ACTIVE_LOW"},
                     "new_format|BTTS": {"state": "ACTIVE"}, "standard|OU25_OVER": {"state": "PAPER"}},
          "tier_multipliers": {"SNIPER": 1.1, "MARKSMAN": 0.9},
          "track_tier_multipliers": {"new_format|OU25_OVER": {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0},
                                     "new_format|OU25_UNDER": {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0}}}
TRACK_EVIDENCE = {"as_of": "2026-10-26", "evidence_cutoff": "2026-10-25", "generated_at": "2026-10-26 12:00 UTC",
                  "tracks": {"new_format|OU25_OVER": {"n": 147, "matchdays": 46, "roi": 0.0399, "clv_n": 144,
                                                      "clv_median": 0.0047},
                             "new_format|BTTS": {"n": 133, "matchdays": 45, "roi": 0.2254, "clv_n": 116,
                                                 "clv_median": 0.0}}}
RECOMMENDATION = {"generated_at": "2026-10-26 12:00 UTC", "policy_version": "balanced+review-2026-10",
                  "review": {"as_of": "2026-10-26", "evidence_cutoff": "2026-10-25", "POLICY_CHANGE_RECOMMENDED": "NO",
                             "RECOMMENDED_POLICY": "CURRENT_WITH_RISK_REDUCTIONS_ONLY"}}


# ── 1. Bankroll: since last formal review ───────────────────────────────────────────────────────────
def test_mirrored_rules_equal_production():
    """evidence_panels copies small rules instead of importing writer modules — they must not drift."""
    import config
    from src import backtest, bankroll_policy, market_edge
    for mt in ("new_format", "standard", "unknown"):
        for mk in ("OU25", "BTTS", "OVER15", "OVER35", "ou25"):
            for sd in ("OVER", "UNDER", "YES", None, float("nan"), "over", ""):
                assert EP.track_key(mt, mk, sd) == bankroll_policy.track_key(mt, mk, sd), (mt, mk, sd)
    assert EP.SIDE_SPLIT_MARKETS == bankroll_policy.SIDE_SPLIT_MARKETS
    assert EP.PERFORMANCE_CUTOFF == config.PERFORMANCE_CUTOFF_DATE
    assert EP._paper_leagues() == set(config.PAPER_LEAGUES)
    assert EP.O15_FILL_PRICE == backtest._DEFAULT_ODDS["over15"] == market_edge.SYNTHETIC_DEFAULT_ODDS["OU15"]


def test_since_review_numbers_equal_independent_computation():
    main, side = _ledgers()
    bets = EP.load_bets(main, side)
    # paper league, non-live and unsettled rows never count
    assert "USA MLS" not in set(bets["league"]) and len(bets) == 4 + 7
    assert (bets["market"] == "OU25").sum() == 4
    meta, rows = EP.since_review(TRACK_EVIDENCE, RECOMMENDATION, policy=POLICY, bets=bets, today="2026-10-30")
    r = rows.set_index("track")
    # review-date figures are Pro's own, verbatim
    assert r.loc["new_format|OU25_OVER", "review_n"] == 147 and r.loc["new_format|OU25_OVER", "review_roi"] == 0.0399
    assert r.loc["new_format|OU25_OVER", "review_clv_coverage"] == round(144 / 147, 3)
    assert r.loc["new_format|BTTS", "review_roi"] == 0.2254
    # since-review: dated after the 2026-10-25 cutoff; OU25 OVER = 27/10 MARKSMAN loss + 28/10 SNIPER win
    o = r.loc["new_format|OU25_OVER"]
    assert o["since_n"] == 2 and o["since_matchdays"] == 2
    assert o["since_roi"] == round((-1.0 + 1.1) / 2, 4) and o["since_units"] == 0.1
    assert o["since_clv_median"] == 0.005 and o["since_clv_coverage"] == 0.5
    # BTTS keeps the global ladder: VALUABLE (multiplier 0) is not evidence -> the 27/10 loss is excluded
    assert r.loc["new_format|BTTS", "since_n"] == 0
    # rolling windows end at `today` (30/10): 14d = 16/10.. ; 30d = 30/09..
    assert o["roll14_n"] == 2 and o["roll30_n"] == 3 and o["roll30_roi"] == round((1.0 - 1.0 + 1.1) / 3, 4)
    assert set(rows["direction"]) == {"INSUFFICIENT"}                  # n < 30 everywhere
    assert meta["evidence_cutoff"] == "2026-10-25" and meta["policy_change_recommended"] == "NO"


def test_review_tier_rule_is_applied_to_both_periods_and_never_mutates_the_policy():
    main, side = _ledgers()
    rules = pd.DataFrame([{"track": "new_format|BTTS", "rule": "EQUAL (edge does not rank outcomes)"},
                          {"track": "new_format|OU25_OVER", "rule": "LADDER (edge ranks outcomes)"}])
    before = json.dumps(POLICY, sort_keys=True)
    _, rows = EP.since_review(TRACK_EVIDENCE, RECOMMENDATION, policy=POLICY, bets=EP.load_bets(main, side),
                              today="2026-10-30", tier_rules=rules)
    assert json.dumps(POLICY, sort_keys=True) == before
    r = rows.set_index("track")
    assert r.loc["new_format|BTTS", "since_n"] == 1                    # EQUAL -> the VALUABLE row counts
    assert r.loc["new_format|OU25_OVER", "since_n"] == 2               # LADDER -> SNIPER 1.1, MARKSMAN 0.9 > 0


@pytest.mark.parametrize("units,expected", [
    ([0.5] * 40, "IMPROVING"), ([-1.0] * 40, "DETERIORATING"), ([1.0, -1.0] * 20, "STABLE"), ([1.0] * 10, "INSUFFICIENT")])
def test_direction_rule(units, expected):
    days = pd.to_datetime([f"2026-10-{1 + i % 8:02d}" for i in range(len(units))])
    since = pd.DataFrame({"unit": units, "date": days})
    d = EP.direction(since, 0.0)
    assert d["state"] == expected, d
    if expected != "INSUFFICIENT":
        assert d["lo"] <= float(np.mean(units)) <= d["hi"]
    assert EP.direction(since, None)["state"] == "INSUFFICIENT"


def test_bankroll_page_cannot_write_policy_static():
    """The page and every dashboard module it draws with: no policy writer imported, no write call."""
    files = [BANKROLL, ROOT / "dashboard_data" / "evidence_panels.py", ROOT / "dashboard_data" / "evidence_view.py",
             ROOT / "dashboard_data" / "bankroll_view.py"]
    forbidden_mods = ("src.bankroll_policy", "src.stake_plan", "src.bankroll", "scripts", "src.promotion_gate",
                      "src.allocation", "src.ledger")
    write_attrs = {"write_text", "write_bytes", "to_csv", "to_parquet", "to_json", "to_excel", "dump", "unlink",
                   "mkdir", "touch", "rmdir"}                    # (DataFrame.rename is not a write)
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if isinstance(node, ast.ImportFrom) and node.module == "src":
                    names += [f"src.{a.name}" for a in node.names]
                for n in names:
                    assert not any(n == m or n.startswith(m + ".") for m in forbidden_mods), (f.name, n)
            if isinstance(node, ast.Call):
                fn = node.func
                assert not (isinstance(fn, ast.Attribute) and fn.attr in write_attrs), (f.name, fn.attr)
                if isinstance(fn, ast.Name) and fn.id == "open":
                    modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)] + \
                            [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                    assert not any(set(str(m)) & set("wax+") for m in modes), (f.name, modes)


def test_bankroll_page_renders_offline_states_no_policy_change_and_writes_nothing(monkeypatch):
    _offline(monkeypatch)
    policy = ROOT / "config" / "bankroll_policy.json"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    writes = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+") and str(Path(str(file)).resolve()).startswith(str((ROOT / "config").resolve())):
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    at = _render(BANKROLL, monkeypatch)
    text = _text(at)
    assert EP.NO_POLICY_CHANGE in text
    assert "FEED NOT CONNECTED" in text                                 # Pro unreachable -> no review to compare
    assert not writes and hashlib.sha256(policy.read_bytes()).hexdigest() == digest


def test_bankroll_panel_shows_the_review_and_ledger_numbers(monkeypatch, tmp_path):
    """A fake Pro checkout carrying a review: the cards show Pro's review ROI and the ledger's since figures."""
    br = tmp_path / "output" / "bankroll"
    br.mkdir(parents=True)
    (br / "track_evidence.json").write_text(json.dumps(TRACK_EVIDENCE), encoding="utf-8")
    (br / "current_policy_recommendation.json").write_text(json.dumps(RECOMMENDATION), encoding="utf-8")
    _offline(monkeypatch, tmp_path)
    at = _render(BANKROLL, monkeypatch)
    text = _text(at)
    assert EP.NO_POLICY_CHANGE in text and "Since last formal bankroll review" in text
    meta, rows = EP.since_review(TRACK_EVIDENCE, RECOMMENDATION)
    cards = [(m.label, m.value) for m in at.metric if m.label == "Since-review ROI"]
    real = rows[rows["state"].isin(["ACTIVE", "ACTIVE_LOW", "ACTIVE_STRONG", "PROBATION", "DEGRADED"])]
    assert len(cards) == len(real)
    import dashboard_ui as ui
    want = sorted("—" if pd.isna(x) else ui.fmt_roi(float(x)) for x in real["since_roi"])
    assert sorted(v for _, v in cards) == want
    assert "+4.0% · n=147" in text                                     # Pro's review ROI for NF OU25 OVER, verbatim


# ── 2. Model Lab: H-MODEL-01 ─────────────────────────────────────────────────────────────────────────
def test_hmodel_rows_equal_the_eval_artifact():
    ev = json.loads(EP.SHADOW_EVAL.read_text(encoding="utf-8")) if EP.SHADOW_EVAL.exists() else {}
    if not ev.get("results"):
        pytest.skip("no shadow_experiments_eval.json in this checkout")
    meta, rows = EP.hmodel_rows()
    assert len(rows) == len(ev["results"]) and meta["min_n"] == 1000
    for _, r in rows.iterrows():
        src = ev["results"][f"{r['experiment']}|{r['track']}"]
        assert r["settled_paired_n"] == src["settled_paired"]
        assert r["delta_log_loss"] == src.get("delta_log_loss") and r["delta_brier"] == src.get("delta_brier")
        assert r["ll_champion"] == src.get("ll_champion_raw") and r["brier_challenger"] == src.get("brier_challenger")
        assert r["required_n"] == 1000 and r["progress"] == min(1.0, src["settled_paired"] / 1000)
        assert r["status"] == "SHADOW — not promoted"
        assert r["req_n_met"] is (src["settled_paired"] >= 1000)


def test_evaluate_brier_ci_added_and_log_loss_ci_unchanged():
    """The extra Brier CI shares each resample with the log-loss CI, so the log-loss CI is bit-identical to
    the pre-2026-10-26 algorithm (copied verbatim below)."""
    from src import shadow_experiments as SE
    rng = np.random.default_rng(1)
    n = 120
    led = pd.DataFrame({"experiment_id": "E", "model_type": "standard",
                        "match_date": [f"2026-10-{1 + i % 12:02d}" for i in range(n)],
                        "y": rng.integers(0, 2, n), "p_champion_raw": rng.uniform(0.3, 0.7, n),
                        "p_challenger": rng.uniform(0.3, 0.7, n)})
    led["p_served"] = led["p_champion_raw"]
    res = SE.evaluate(led.astype(str), n_boot=300, seed=7)["E|standard"]
    y, a, b = led["y"].to_numpy(float), led["p_champion_raw"].to_numpy(float), led["p_challenger"].to_numpy(float)
    dl = SE._ll(y, b) - SE._ll(y, a)
    dates = led["match_date"].to_numpy()
    ud = np.unique(dates)
    old_rng = np.random.default_rng(7)
    by = {k: dl[dates == k] for k in ud}
    bs = [np.concatenate([by[k] for k in old_rng.choice(ud, len(ud))]).mean() for _ in range(300)]
    assert res["delta_log_loss_ci95"] == pytest.approx([np.percentile(bs, 2.5), np.percentile(bs, 97.5)], abs=1e-12)
    lo, hi = res["delta_brier_ci95"]
    assert lo <= res["delta_brier"] <= hi and res["n_matchdays"] == 12


def test_model_lab_renders_with_the_hmodel_panel(monkeypatch):
    _offline(monkeypatch)
    at = _render(MODEL_LAB, monkeypatch)
    meta, rows = EP.hmodel_rows()
    if rows.empty:
        return
    text = _text(at)
    assert "H-MODEL-01 — champion vs HGB challenger, per track" in text and "SHADOW — NOT PROMOTED" in text
    vals = [m.value for m in at.metric if m.label == "Δ log loss (HGB − champion)"]
    assert vals == [f"{x:+.4f}" for x in rows["delta_log_loss"]]


# ── 3. Market Intelligence: H-EDGE2 ──────────────────────────────────────────────────────────────────
def _shadow_rows():
    def r(i, pts, ko, track="new_format", market="OU25", state="NO_BET", src="MEDIAN_TWO_SIDED_BOOKS",
          side="OVER", odds=2.0, league="L1", ver="market_edge_v2"):
        return {"selection_engine_version": ver, "prediction_ts": pts, "fixture_id": f"f{i}", "kickoff_utc": ko,
                "league": league, "model_type": track, "market": market, "decision_state": state,
                "market_source": src, "side": side, "odds": odds}
    return pd.DataFrame([
        r(1, "2026-10-25T09:00:00Z", "2026-10-27T18:00:00Z"),                         # pre-registration
        r(2, "2026-10-26T09:00:00Z", "2026-10-27T18:00:00Z", state="QUALIFIED", odds=2.3, league="L2"),
        r(2, "2026-10-26T12:00:00Z", "2026-10-27T18:00:00Z", state="NO_BET"),         # later sight: ignored
        r(3, "2026-10-26T09:00:00Z", "2026-10-28T18:00:00Z", src="ONE_SIDED"),
        r(4, "2026-10-29T09:00:00Z", "2026-10-28T18:00:00Z"),                         # after kickoff
        r(5, "2026-10-26T09:00:00Z", "2026-10-28T18:00:00Z", ver="legacy"),
        r(6, "2026-10-26T09:00:00Z", "2026-10-28T18:00:00Z", market="BTTS", side="YES")])


def test_edge2_report_counts_status_words_and_interim_policy():
    prog = {"forward_start": "2026-10-26T00:00:00Z", "engine_version": "market_edge_v2", "production_changed": "NO",
            "hypotheses": [
                {"id": "A", "track": "new_format", "market": "OU25", "status": "ACCRUING", "min_n_settled_two_sided": 400,
                 "min_gameweeks": 8, "forward": {"settled_two_sided": 3, "gameweeks": 1, "logged": 3},
                 "metrics": {"ll_diff_mean": -0.1}, "secondary": {"selection": {"roi": 0.5}},
                 "descriptive": {"matchdays": 2}, "progress_pct": 0.8},
                {"id": "B", "track": "new_format", "market": "BTTS", "status": "ELIGIBLE_FOR_REVIEW",
                 "min_n_settled_two_sided": 400, "min_gameweeks": 8, "forward": {"settled_two_sided": 401},
                 "metrics": {"ll_diff_mean": -0.01, "ll_diff_ci95": [-0.02, 0.0]},
                 "secondary": {"selection": {"n": 5, "roi": 0.1}}, "progress_pct": 100.0},
                {"id": "C", "track": "standard", "market": "OU25", "status": "CONFIRMED", "forward": {}}]}
    meta, items = EP.edge2_report(prog, _shadow_rows())
    a, b, c = items
    assert a["status"] == "ACCRUING" and a["metrics"] == {} and a["secondary"] == {}       # interim: withheld
    assert a["matchdays"] == 2 and a["settled"] == 3
    assert b["metrics"]["ll_diff_mean"] == -0.01 and b["secondary"]["selection"]["roi"] == 0.1
    assert c["status"] == "INVALID (CONFIRMED)" and c["metrics"] == {}
    sh = a["shadow"]                                     # rows 2 (first sight) and 3 count; 1/4/5/6 do not
    assert sh["logged"] == 2 and sh["two_sided"] == 1 and sh["selections"] == 1 and sh["selection_rate"] == 0.5
    assert sh["sides_selected"] == {"OVER": 1} and sh["odds_bands_selected"]["(2.2, 2.5]"] == 1
    assert sh["leagues"] == 2 and sh["top_league_share"] == 0.5
    assert b["shadow"]["logged"] == 1 and meta["production_changed"] == "NO"


def test_edge2_shadow_counts_equal_the_v9_shadow_artifact():
    s = EP._read(EP.SHADOW)
    if s.empty:
        pytest.skip("no edge_engine_shadow.csv")
    fs = "2026-10-26T00:00:00Z"
    sh = EP.shadow_forward(s, "new_format", "OU25", fs)
    x = s[s["selection_engine_version"].eq("market_edge_v2") & s["model_type"].eq("new_format") & s["market"].eq("OU25")].copy()
    x["p"] = pd.to_datetime(x["prediction_ts"], utc=True)
    x["k"] = pd.to_datetime(x["kickoff_utc"], utc=True)
    x = x[x["p"] < x["k"]].sort_values("p", kind="mergesort").drop_duplicates(["fixture_id", "market"], keep="first")
    x = x[x["p"] >= pd.Timestamp(fs)]
    assert sh["logged"] == len(x)
    assert sh["two_sided"] == int(x["market_source"].isin(EP.TWO_SIDED).sum())
    assert sh["selections"] == int(x["decision_state"].isin(["QUALIFIED", "STRONG"]).sum())


# ── 4. NF Over 1.5 real price only ───────────────────────────────────────────────────────────────────
def test_over15_real_price_excludes_filled_prices_and_matches_the_ledger():
    _, side = _ledgers()
    meta, rows = EP.over15_real_price(side, today="2026-10-30")
    ex = meta["exclusions"]
    assert ex == {"settled_rows": 4, "no_price": 1, "fill_constant_1.40": 1, "settled_at_fair_odds": 0, "kept": 2}
    allr = rows[(rows["window"] == "season") & (rows["gap_band"] == "ALL")].iloc[0]
    assert allr["n"] == 2 and allr["roi"] == round((0.55 - 1.0) / 2, 4) and allr["units"] == -0.45
    assert allr["ledger_clv_coverage"] == 0.5 and allr["ledger_clv_mean_pct"] == 2.0
    assert allr["certified_clv_coverage"] == 0.0 and allr["certified_clv_mean_pct"] is None
    assert bool(allr["small_sample"])
    sn = rows[(rows["window"] == "season") & (rows["gap_band"] == "SNIPER")].iloc[0]
    assert sn["n"] == 1 and sn["roi"] == 0.55
    assert set(rows["window"]) == {"30d", "60d", "season"}


def test_over15_on_the_committed_ledger():
    s = EP._read(EP.SIDE_BETS)
    if s.empty:
        pytest.skip("no side ledger")
    meta, rows = EP.over15_real_price()
    x = s[s["market"].eq("over15") & s["model_type"].eq("new_format") & s["source"].fillna("live").eq("live")
          & s["result"].isin(["WIN", "LOSS"])]
    x = x[pd.to_datetime(x["match_date"].astype(str).str[:10]) >= pd.Timestamp("2026-08-10")]
    x = x[pd.to_numeric(x["odds"], errors="coerce").gt(1.0) & ~pd.to_numeric(x["odds"], errors="coerce").round(4).eq(1.40)]
    allr = rows[(rows["window"] == "season") & (rows["gap_band"] == "ALL")].iloc[0]
    assert allr["n"] == len(x)
    assert allr["roi"] == pytest.approx(round(float(pd.to_numeric(x["pnl"]).mean()), 4))
    by = rows[(rows["window"] == "season") & (rows["gap_band"] != "ALL")]
    assert by["n"].sum() == len(x)


def test_market_intelligence_renders_with_both_panels(monkeypatch):
    _offline(monkeypatch)
    at = _render(MARKET_INTEL, monkeypatch, timeout=240)
    text = _text(at)
    assert "H-EDGE2 forward experiment report" in text
    assert "New-format Over 1.5 — REAL PRICE ONLY" in text
    chips = [m.value for m in at.markdown if "-badge[" in m.value]
    assert not any("CONFIRMED" in c for c in chips), chips          # no status chip ever says CONFIRMED
    meta, rows = EP.over15_real_price(today=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    if not rows.empty and rows["n"].max():
        import dashboard_ui as ui
        want = rows[(rows["gap_band"] == "ALL")].set_index("window").loc[["30d", "60d", "season"], "roi"]
        got = [m.value for m in at.metric if m.label == "ROI (flat 1u)"]
        assert got == ["—" if pd.isna(v) else ui.fmt_roi(float(v)) for v in want]


def test_market_intelligence_edge2_cards_equal_pro_progress(monkeypatch, tmp_path):
    prog = {"generated_at": "2026-10-26T12:00:00Z", "forward_start": "2026-10-26T00:00:00Z",
            "engine_version": "market_edge_v2", "registration_intact": True, "production_changed": "NO",
            "hypotheses": [{"id": h, "track": t, "market": m, "status": s, "min_n_settled_two_sided": 400,
                            "min_gameweeks": 8, "progress_pct": p,
                            "forward": {"logged": 6, "two_sided": 6, "settled_two_sided": n, "gameweeks": 0,
                                        "selections_QUALIFIED_STRONG": 0, "states": {"NO_BET": 6}},
                            "descriptive": {"matchdays": 0}, "metrics": None, "secondary": None}
                           for h, t, m, s, p, n in (("H-EDGE2-NF-OU25-01", "new_format", "OU25", "ACCRUING", 1.5, 6),
                                                    ("H-EDGE2-STD-OU25-01", "standard", "OU25", "REGISTERED", 0.0, 0))]}
    d = tmp_path / "output" / "edge_reassessment"
    d.mkdir(parents=True)
    (d / "edge2_forward_progress.json").write_text(json.dumps(prog), encoding="utf-8")
    _offline(monkeypatch, tmp_path)
    at = _render(MARKET_INTEL, monkeypatch, timeout=240)
    got = [m.value for m in at.metric if m.label == "Progress"]
    assert got == ["1.5%", "0.0%"]
    chips = " ".join(m.value for m in at.markdown if "-badge[" in m.value)
    assert "ACCRUING" in chips and "REGISTERED" in chips
    assert "withheld until ELIGIBLE_FOR_REVIEW" in _text(at)
