"""Non-regression for the 2026-10-25 visual redesign, stage 2b: League Repair Lab, Leagues, Overview, Live.

The redesign moved and restyled values; it must not change any of them. Each test either renders the page
headlessly (streamlit.testing AppTest, absolute path) and compares what it shows with the canonical function
the page draws from, or checks a view helper against the expression the old page used. A mismatch means
the presentation changed a number.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest

from src.research import league_weekly_eval as EV

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT / "pages" / "18_🔧_League_Research.py"
OVERVIEW = ROOT / "pages" / "0_🏠_Overview.py"
LIVE = ROOT / "pages" / "6_⚡_Live.py"


def _render(page: Path, monkeypatch, timeout=180):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    st.cache_data.clear()
    at = AppTest.from_file(str(page.resolve()), default_timeout=timeout).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metrics(at) -> list[tuple[str, str]]:
    return [(m.label, m.value) for m in at.metric]


def _panel(weeks=10, league="League One"):
    """Settled League One OU25 panel (as tests/test_league_weekly._settled_panel), one rank missing in the
    newest week so the heatmap has to draw an explicit NO ROW cell."""
    rng = random.Random(7)
    rows = []
    for w in range(weeks):
        for rk in range(1, 6):
            if w == weeks - 1 and rk == 5:
                continue
            res = rng.choice(["WIN", "LOSS"]) if w < weeks - 1 else ""
            rows.append({"league": league, "market": "OU25", "model_type": "standard",
                         "side": rng.choice(["OVER", "UNDER"]), "league_week_id": f"{league}|2026-W{30 + w:02d}",
                         "league_market_rank": str(rk), "kickoff_utc": f"2026-07-{1 + w:02d}T15:00:00Z",
                         "home_team": f"H{w}{rk}", "away_team": f"A{w}{rk}", "fixture_id": f"f{w}{rk}",
                         "ev_lower_bound": str(0.02 - 0.01 * rk), "odds": "1.9",
                         "decision_state": "QUALIFIED" if rk == 1 else "NO_BET",
                         "price_quality_state": "TWO_SIDED_MULTI_BOOK", "result": res,
                         "flat_pnl": "" if not res else str(0.9 if res == "WIN" else -1.0),
                         "clv_pct": str(rng.uniform(-3, 3)) if rk < 4 and res else "", "clv_prob": "",
                         "p_model_raw": "0.55", "p_market_fair": "0.52", "p_decision": "0.53",
                         "production_selected": "True" if rk == 2 else "False", "production_selected_final": "",
                         "settle_status": "SETTLED" if res else "", "prediction_engine_version": "standard_x",
                         "selection_engine_version": "market_edge_v2", "legacy_rank": str(rk), "legacy_tier": "AVOID",
                         "research_regime": "REPAIR", "production_state": "PAPER", "actually_bet": "False"})
    return pd.DataFrame(rows)


# ── League Repair Lab ─────────────────────────────────────────────────────────────────────────
def test_heatmap_is_the_weekly_grid_cell_for_cell():
    from dashboard_data import league_research_view as V
    d = _panel()
    hm = V.heatmap_frame(d, "League One", "OU25")
    grid = EV.weekly_grid(d, "League One", "OU25")
    assert len(hm) == 5 * len(grid)                                   # every gameweek x #1..#5 is a cell
    assert list(dict.fromkeys(hm["league_week_id"])) == list(grid["league_week_id"])      # newest first
    for _, c in hm.iterrows():
        want = grid.loc[grid["league_week_id"] == c["league_week_id"], c["rank"]].iloc[0]
        assert c["cell"] == want                                       # tooltip text == history table text
        if c["outcome"] == "NO ROW":
            assert want == ""
        else:
            assert c["outcome"] in ("WIN", "LOSS", "PENDING")
            assert (c["outcome"] if c["outcome"] != "PENDING" else "pending") in want
            assert c["outcome"] in c["label"]                          # colour is never the only cue
            assert ("PROD" in c["label"]) == ("| PROD" in want)
    assert {"WIN", "LOSS", "PENDING", "NO ROW"} <= set(hm["outcome"])
    spec = V.heatmap_chart(hm).to_dict()
    assert len(spec["layer"]) == 2 and spec["layer"][1]["mark"]["type"] == "text"


def test_heatmap_empty_panel_is_an_empty_state_not_an_empty_chart():
    from dashboard_data import league_research_view as V
    assert V.heatmap_frame(pd.DataFrame(), "League One", "OU25").empty


def test_strip_and_engine_frames_are_league_table_rows():
    from dashboard_data import league_research_view as V
    d = _panel()
    t = EV.league_table(d)
    t = t[(t["league"] == "League One") & (t["market"] == "OU25")]
    strip = V.rank_strip_rows(t)
    assert set(strip["subset"]) == {"RANK1", "TOP2", "TOP3", "TOP5"}
    pd.testing.assert_frame_equal(strip, t[t["subset"].isin(["RANK1", "TOP2", "TOP3", "TOP5"])].reset_index(drop=True))
    ef = V.engine_frame(t)
    for r in ef.to_dict("records"):
        sub = {v: k for k, v in V.ENGINE_SUBSETS}[r["selection"]]
        src = t[(t["side"] == r["side"]) & (t["subset"] == sub)].iloc[0]
        for c in ("n_rows", "n_settled", "n_priced_settled", "roi", "clv_mean", "win_rate", "status"):
            assert (pd.isna(r[c]) and pd.isna(src[c])) or r[c] == src[c], (r["selection"], c)
    pf = V.prob_frame(t)
    top5 = t[t["subset"] == "TOP5"]
    for r in pf.to_dict("records"):
        key = {v: k for k, v in V.PROB_SOURCES}[r["source"]]
        assert r["brier"] == top5[top5["side"] == r["side"]].iloc[0][f"brier_{key}"]


def test_repair_lab_renders_canonical_values(tmp_path, monkeypatch):
    """Render on a settled synthetic panel: the TOP 1/2/3/5 cards and the strip show league_table's numbers."""
    import config
    import dashboard_ui as ui
    d = _panel()
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    d.to_csv(tmp_path / "league_weekly_research.csv", index=False)
    health = {"generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "tracked_leagues": 20,
              "leagues_with_current_week_data": 1, "missing_leagues": [], "expected_candidate_rows": 20,
              "actual_candidate_rows": 19, "panel_rows": len(d), "settled_rows": 45, "pending_rows": 4,
              "clv_coverage": 0.6, "protected_tracks_modified": False, "leagues": {}}
    (tmp_path / "league_market_health.json").write_text(json.dumps(health), encoding="utf-8")
    (tmp_path / "weekly_league_report.json").write_text(json.dumps({"leagues": {}}), encoding="utf-8")
    at = _render(REPAIR, monkeypatch)
    at.selectbox(key="lr_league").set_value("League One").run()
    assert not at.exception, [e.value for e in at.exception]
    m = _metrics(at)
    assert ("Tracked leagues", "20") in m and ("CLV coverage", "60%") in m
    t = EV.league_table(d)
    t = t[(t["league"] == "League One") & (t["market"] == "OU25")]
    rois = [v for lab, v in m if lab == "ROI (flat 1u)"]
    want = [ui.fmt_roi(r["roi"]) for r in V_strip(t)]
    assert rois[:len(want)] == want                                    # OU25 tab renders first
    # the history TABLE under the heatmap is still league_weekly_eval.weekly_grid, exactly
    grids = [df.value for df in at.dataframe if list(df.value.columns)[:2] == ["league_week_id", "#1"]]
    pd.testing.assert_frame_equal(grids[0].reset_index(drop=True), EV.weekly_grid(d, "League One", "OU25"),
                                  check_dtype=False)
    # PROTECTED is governance (blue), never an error/warning element
    assert not at.error and not at.warning


def V_strip(t):
    from dashboard_data import league_research_view as V
    rows = V.rank_strip_rows(t)
    out = []
    for (_mt, _side), g in rows.groupby(["model_type", "side"], sort=True):
        by = {r["subset"]: r for r in g.to_dict("records")}
        out += [by[s] for s, _ in V.STRIP if s in by]
    return out


def test_repair_lab_real_data_matches_health_json(monkeypatch):
    from dashboard_data import league_research as LR
    from dashboard_data import league_research_view as V
    h = LR.load_json("league_market_health.json")
    if not h:
        pytest.skip("no league_market_health.json in this checkout")
    at = _render(REPAIR, monkeypatch)
    m = dict((lab, v) for lab, v in _metrics(at))
    assert m["Tracked leagues"] == str(h.get("tracked_leagues", "—"))
    assert m["CLV coverage"] == V.pct0(h.get("clv_coverage"))
    rc = V.regime_counts(LR.league_order(LR.registry_entries()))
    assert m["Leagues in REPAIR"] == str(rc["REPAIR"])
    pd.testing.assert_frame_equal(at.dataframe[0].value.reset_index(drop=True),
                                  LR.overview(LR.load_panel(), LR.registry_entries(), h), check_dtype=False)


# ── Overview ──────────────────────────────────────────────────────────────────────────────────
def test_overall_status_rules():
    from dashboard_data import overview_view as V
    g, a, r = {"status": "GREEN"}, {"status": "AMBER"}, {"status": "RED"}
    assert V.overall([g, g]) == "GREEN" and V.overall([g, a]) == "AMBER" and V.overall([a, r, g]) == "RED"
    assert V.overall([]) == "NO_DATA"
    assert V.count_line([g, a, a, r]) == "1 red · 2 amber · 1 green"


def test_overview_renders_same_values_offline(monkeypatch):
    """Pro / V11 unreachable (no network in tests): the page still renders, says FEED NOT CONNECTED, and
    the production numbers equal the adapter's."""
    import dashboard_ui as ui
    from dashboard_data import core, v9
    monkeypatch.setattr(core, "RAW_BASE", {})
    for k in ("pro", "v11"):
        monkeypatch.setitem(core.REPOS, k, {**core.REPOS[k], "candidates": []})
    at = _render(OVERVIEW, monkeypatch)
    m = dict(_metrics(at))
    perf = v9.performance()
    if perf.get("available"):
        tracks = {k: v for k, v in perf["by_track"].items() if k != "unknown"}
        assert m["Staked P/L"] == ui.units(sum(t["staked_only"]["pnl"] for t in tracks.values()))
        for name, t in tracks.items():
            assert m[name.replace("_", "-")] == ui.units(t["staked_only"]["pnl"])
    from src.system_health import evaluate
    from dashboard_data import overview_view as V
    md = " ".join(x.value for x in at.markdown)
    assert V.count_line(evaluate()) in md
    assert "FEED NOT CONNECTED" in md


# ── Live ──────────────────────────────────────────────────────────────────────────────────────
def _old_card(row):
    """The pre-redesign HTML card's expressions, copied verbatim from pages/6_⚡_Live.py at 50af39e4."""
    bet = row["bet"]; is_under = "UNDER" in bet
    fair_odds = row["fair_under_odds"] if is_under else row["fair_over_odds"]
    live_p = row["live_p_under"] if is_under else row["live_p_over"]
    elapsed = int(row["elapsed_mins"]); bar_pct = min(elapsed / 90 * 100, 100)
    return {"match": row["match"], "league": row["league"], "minute": elapsed, "score": row["score"], "bet": bet,
            "fair": f"{fair_odds}", "live": f"{live_p*100:.0f}%", "pre": f"{row['pre_p_over']*100:.0f}%",
            "bar": f"{bar_pct:.0f}", "side": "UNDER" if is_under else "OVER"}


@pytest.mark.parametrize("row", [
    {"league": "Championship", "match": "Hull City vs Leeds", "score": "0-0", "elapsed_mins": 71, "pre_p_over": 0.48,
     "live_p_under": 0.8123, "live_p_over": 0.1877, "fair_under_odds": 1.23, "fair_over_odds": 5.33,
     "signal_type": "UNDER_HOLD", "bet": "UNDER 2.5"},
    {"league": "Bundesliga 2", "match": "Hertha vs Schalke", "score": "0-1", "elapsed_mins": 95, "pre_p_over": 0.66,
     "live_p_under": 0.3512, "live_p_over": 0.6488, "fair_under_odds": 2.85, "fair_over_odds": 1.54,
     "signal_type": "COMEBACK", "bet": "OVER 2.5"}])
def test_live_card_shows_the_old_card_values(row):
    from dashboard_data import live_view as LV
    old, v = _old_card(pd.Series(row)), LV.signal_values(pd.Series(row))
    stats = dict(v["stats"])
    assert (v["match"], v["league"], v["minute"], v["score"]) == (old["match"], old["league"], old["minute"], old["score"])
    assert stats["Bet"] == old["bet"] and stats["Fair price"] == old["fair"]
    assert stats[f"Live P({old['side']})"] == old["live"] and stats["Pre-match P(over)"] == old["pre"]
    assert old["fair"] in v["note"]
    assert round(min(v["minute"] / LV.FULL_TIME, 1.0) * 100) == int(old["bar"])
    assert v["icon"] and LV.signal_chip(row["signal_type"]).startswith(":")    # icon + word chip, no hex


def test_live_page_renders_with_shared_cards(monkeypatch):
    at = _render(LIVE, monkeypatch)
    assert [t.label for t in at.tabs] == ["⚡ Live Now", "⏱ Half-Time", "📜 History"]
    hist = ROOT / "output" / "live_signals_history.csv"
    if hist.exists():
        n = len(pd.read_csv(hist))
        if n:
            assert ("Total Signals", str(n)) in _metrics(at)
    for m in at.markdown:
        assert "background:" not in m.value and "color:white" not in m.value.replace(" ", "")
