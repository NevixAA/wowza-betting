"""Data truth, delivery layer (owner rules 2026-11-01): Telegram, dashboard pages and scripts show a
missing value as missing, take leagues from the registry and match clubs through the resolver.

One test per behaviour change on the dt/delivery branch.
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from telegram_bot import notifier as N  # noqa: E402
from src import league_registry as LR  # noqa: E402

DAY = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")


def _page_funcs(glob: str, *names: str, extra: dict | None = None) -> dict:
    """Pure helper functions lifted out of a Streamlit page (the page itself needs a server)."""
    src = next((ROOT / "pages").glob(glob)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns: dict = {"pd": pd, "np": np, **(extra or {})}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []                           # drop @st.cache_data
            exec(compile(ast.Module([node], []), glob, "exec"), ns)
    return ns


# ── Telegram ────────────────────────────────────────────────────────────────────────────────────
def test_kickoff_uses_registry_zone_and_labels_utc_when_none():
    ko = f"{DAY} 14:00:00"
    local = N._fmt_kickoff({"league": "Championship", "date": DAY, "kickoff_utc": ko})
    assert local.startswith(DAY) and not local.endswith(" UTC")          # Europe/London: GMT/BST
    # A multi-country cup has no zone in the registry: shown in UTC and SAID to be UTC.
    assert LR.timezone("Champions League") is None
    assert N._fmt_kickoff({"league": "Champions League", "date": DAY, "kickoff_utc": ko}) == f"{DAY} 14:00 UTC"
    assert N._fmt_kickoff({"league": "Nowhere League", "date": DAY, "kickoff_utc": ko}).endswith(" UTC")


def test_missing_numbers_and_drift_display_as_missing():
    assert N._fmt_num(None, ".2f") == "–" and N._fmt_num(float("nan"), "+.0f") == "–"
    assert N._fmt_num("1.95", ".2f") == "1.95"
    assert N._drift_text(None) == N._drift_text(float("nan")) == N._drift_text("") == "no drift data"
    assert N._drift_text("New") == "New"                                  # a real signal stays


def _bets(tmp_path, **kw):
    r = {"date": DAY, "kickoff_utc": f"{DAY} 15:00:00", "league": "Championship", "home_team": "Hull City",
         "away_team": "Leeds United", "best_side": "OVER", "bet": "OVER", "odds_over25": 1.95,
         "odds_under25": 1.9, "best_edge": 0.10, "signal_tier": "SNIPER", "drift_signal": "Neutral",
         "model_type": "standard"}
    r.update(kw)
    pd.DataFrame([r]).to_csv(tmp_path / "bets.csv", index=False)


@pytest.fixture
def tg(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_load_notified", lambda *a, **k: set())
    monkeypatch.setattr(N, "_save_notified", lambda *a, **k: None)
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    monkeypatch.setattr(N, "_send", lambda token, chat, text: sent.append(text) or True)
    return sent


def test_tip_without_recorded_edge_is_not_sent(tmp_path, tg, capsys):
    _bets(tmp_path, best_edge=np.nan)
    assert N.notify_new_snipers() == 0 and tg == []
    assert "SKIP unsendable tip" in capsys.readouterr().out


def test_tip_without_a_model_is_not_labelled_new_format(tmp_path, tg):
    _bets(tmp_path, league="Nowhere League", model_type=np.nan)        # unregistered -> 'unknown'
    assert N.notify_new_snipers() == 0 and tg == []


def test_tip_without_drift_says_so(tmp_path, tg):
    _bets(tmp_path, drift_signal=np.nan)
    assert N.notify_new_snipers() == 1
    assert "Drift: no drift data" in tg[0] and "Drift: New" not in tg[0]


def test_fixture_lookup_uses_the_resolver_within_league_and_day():
    idx = {N._fixture_bucket("Championship", DAY): {("Hull", "Leeds"): 1.9}}
    assert N._fixture_lookup(idx, "Championship", DAY, "Hull City", "Leeds United") == 1.9
    assert N._fixture_lookup(idx, "Championship", "2001-01-01", "Hull City", "Leeds United") is None
    assert N._fixture_lookup(idx, "League One", DAY, "Hull City", "Leeds United") is None


def test_prop_league_filter_is_the_registry_props_flag():
    s = pd.Series(["Premier League", "Championship", "Nowhere League", None])
    want = [LR.canonical_name(x) in set(LR.prop_leagues()) for x in s]
    assert list(N._is_prop_league(s)) == want and want[-2:] == [False, False]


def test_fantasy_guard_takes_the_fpl_league_from_the_registry(monkeypatch):
    seen = {}

    def _fx(league_id, season, next_n):
        seen.update(id=league_id, season=season)
        return []
    import player_model.api_football as af
    monkeypatch.setattr(af, "get_upcoming_fixtures", _fx)
    assert N.send_fantasy_tips() == 0
    assert seen["id"] == LR.prop_leagues()[LR.fpl_league()]
    src = (ROOT / "telegram_bot" / "notifier.py").read_text(encoding="utf-8")
    assert 'PROP_LEAGUES["Premier League"]' not in src and "__read_csv(" not in src


def test_cashout_without_fixture_is_not_sent(tmp_path, tg, monkeypatch, capsys):
    today = datetime.now().strftime("%Y-%m-%d")                  # the sender's own 'today'
    key = f"PLAYER|{today}|A Player|goals"
    monkeypatch.setenv("PROPS_INSTANT_ALERTS", "1")
    monkeypatch.setattr(N, "PLAYER_NOTIFIED_FILE", tmp_path / "pn.json")
    monkeypatch.setattr(N, "BASE_DIR", tmp_path)
    ko = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    (tmp_path / "kc.json").write_text(json.dumps({key: ko}), encoding="utf-8")
    monkeypatch.setattr(N, "PLAYER_KICKOFF_FILE", tmp_path / "kc.json")
    monkeypatch.setattr(N, "_load_notified", lambda *a, **k: {key} if a and "pn" in str(a[0]) else set())
    # the benched player's own tip row is gone from today's file, so the fixture is unknown
    pd.DataFrame([{"date": today, "player_name": "Other", "market": "goals", "tier": "SNIPER",
                   "league": "Premier League", "match": "X vs Y"}]).to_csv(tmp_path / "player_tips.csv",
                                                                            index=False)
    assert N.notify_lineup_cashout() == 0
    assert tg == [] and "SKIP cashout" in capsys.readouterr().out


# ── daily pack ──────────────────────────────────────────────────────────────────────────────────
def test_unlabelled_source_rows_never_enter_the_record(capsys):
    from telegram_bot import daily_pack as dp
    d = pd.DataFrame({"source": ["live", np.nan, "backtest"], "x": [1, 2, 3]})
    assert list(dp._live_only(d, "t.csv")["x"]) == [1]
    assert "without a recorded source" in capsys.readouterr().out


def test_bet_builder_missing_probability_is_a_dash(monkeypatch):
    from telegram_bot import daily_pack as dp
    sel = pd.DataFrame([{"match_date": "2026-10-18", "league": "L", "match": "A vs B", "legs": "x (5%)",
                         "joint_probability": np.nan, "fair_odds": np.nan}])
    monkeypatch.setattr(dp, "_pro_bet_builder", lambda: sel)
    seen = []
    monkeypatch.setattr(dp, "render_table", lambda title, cols, rows, **k: seen.extend(rows) or [b"png"])
    dp.bet_builder_image("2026-10-18")
    assert "all legs –" in seen[0]["band"] and "0.0%" not in seen[0]["band"]


def test_tips_table_untagged_model_is_derived_or_dash():
    from telegram_bot import tips_table as tt
    assert tt._model_of({"model_type": np.nan, "league": "Championship"}) == "standard"
    # no registered model: None (config.model_type_for_league no longer returns 'unknown')
    assert tt._model_of({"model_type": "", "league": "Nowhere League"}) is None


# ── dashboard pages ─────────────────────────────────────────────────────────────────────────────
def test_agent_analysis_gives_no_verdict_without_inputs():
    ns = _page_funcs("10_*_Agent_Analysis.py", "_num", "analyse_row", "calc_markets", "score_matrix",
                     "dixon_coles_tau", "poisson", "poisson_cdf", "verdict", "ev", "kelly",
                     extra={"math": __import__("math")})
    row = pd.Series({"league": "Championship", "best_side": "OVER", "odds_over25": 1.9, "odds_under25": 1.9,
                     "best_edge": 0.1, "p_over25": 0.6, "league_avg_goals": 2.6,
                     "home_attack_str": np.nan, "away_attack_str": 1.1, "home_defense_str": 1.0,
                     "away_defense_str": 0.9, "signal_tier": "SNIPER"})
    shares = {"Championship": {"home_share": 0.56, "fh_share": 0.45}}
    out = ns["analyse_row"](row, shares)
    assert out["missing"] == ["home_attack_str"] and "NO VERDICT" in out["verdict"]
    row["home_attack_str"] = 1.2
    out = ns["analyse_row"](row, shares)
    assert out["missing"] == [] and out["lhg"] == pytest.approx(2.6 * 0.56)
    # the league's own split, not a global 55/45
    out2 = ns["analyse_row"](row, {"Championship": {"home_share": 0.50, "fh_share": None}})
    assert out2["lhg"] == pytest.approx(1.3) and np.isnan(out2["mkts"]["p_ht_over05"])
    assert "home-goal share" in " ".join(ns["analyse_row"](row, {})["missing"])


def test_success_rates_unpriced_win_has_no_return():
    src = next((ROOT / "pages").glob("5_*_Success_Rates.py")).read_text(encoding="utf-8")
    assert "fillna(2.0)" not in src and '"unknown")' not in src


def test_portfolio_kelly_skips_rows_without_edge():
    ns = _page_funcs("12_*_Portfolio.py", "simulate")
    b = pd.DataFrame({"match_date": pd.to_datetime(["2026-10-01", "2026-10-02"]), "odds": [2.0, 2.0],
                      "result": ["WIN", "WIN"], "tier": ["SNIPER", "SNIPER"], "edge": [0.1, np.nan]})
    eq, stats, games = ns["simulate"](b, 1000, {"SNIPER": 0.125}, 5.0, "Fractional Kelly")
    assert stats["skipped_no_edge"] == 1 and len(games) == 1 and not eq["bank"].isna().any()


def test_pages_show_missing_counts_as_dash():
    for glob, banned in (("7_*_Pro.py", ["get('N_required', 150)", "or 0):+.1%", "−5.8%"]),
                         ("16_*_Bankroll.py", ["get('required', 6)"]),
                         ("20_*_Model_Lab.py", ["or 1000", "or 0.001", "or 180"]),
                         ("13_*_Fantasy.py", ['fillna("available")', "get('residual_mode', 'shadow')"]),
                         ("6_*_Live.py", ["for off in (0, -1, 1)"])):
        src = next((ROOT / "pages").glob(glob)).read_text(encoding="utf-8")
        for b in banned:
            assert b not in src, (glob, b)


def test_pro_band_line_is_computed_from_the_band_files():
    ns = _page_funcs("7_*_Pro.py", "_dash", "_band_line")
    eb = pd.DataFrame({"edge_type": "raw_edge", "kind": "aggregate", "band": ["3-4", "3-4"], "market": "over25",
                       "model_type": ["new_format", "standard"], "N": [10, 10], "PnL": [1.0, -3.0],
                       "CLV_pct": [1.0, 3.0], "CLV_n": [10, 10]})
    line = ns["_band_line"](eb, pd.DataFrame())
    assert "3–4%: **-10.0%** (n 20, CLV +2.00)" in line and "4–6%: —" in line and "BTTS fair edge 4–8%: —" in line


def test_free_tip_ranking_puts_unscored_tracks_last():
    import dashboard_access as A
    plan = pd.DataFrame({"date": [DAY] * 3, "track": ["a", "b", "c"], "final_fraction": [0.01] * 3,
                         "odds": [2.0] * 3})
    pol = {"tracks": {"a": {"evidence_score": -1.0}, "b": {}, "c": {"evidence_score": 2.0}}}
    picks, _, _ = A.free_tip_pick(plan, pol, DAY, n=3)
    assert list(picks["track"]) == ["c", "a", "b"]                 # b has no score: last, not 0


# ── scripts ─────────────────────────────────────────────────────────────────────────────────────
def test_af_backfill_leagues_come_from_the_registry():
    import backfill_af_history as B
    plan = {n: (i, s) for n, i, s in B.league_plan()}
    want = {e["name"] for e in LR.leagues() if e.get("model_type") == "new_format"}
    assert set(plan) == want
    for n, (i, s) in plan.items():
        assert i == LR.entry(n)["api_football_id"] and s == sorted(set(s)) and s
    assert "LEAGUES = [" not in (ROOT / "scripts" / "backfill_af_history.py").read_text(encoding="utf-8")
    assert "LEAGUES = [" not in (ROOT / "scripts" / "backfill_af_odds.py").read_text(encoding="utf-8")


def test_xg_backfill_seasons_follow_each_league_rule():
    import datetime as dt
    sys.path.insert(0, str(ROOT))
    import backfill_xg as X
    d = dt.date(2026, 11, 1)
    assert X.get_seasons_for_league("Championship", d) == ["2023", "2024", "2025", "2026"]
    assert X.get_seasons_for_league("Nowhere League", d) == []


def test_fantasy_backtest_drops_unknown_positions_and_missing_stats(capsys):
    import fantasy_backtest as F
    base = {c: 1 for c in F.STAT_COLS}
    pl = pd.DataFrame([{**base, "position": "M"}, {**base, "position": "??"},
                       {**base, "position": "D", "key_passes": np.nan}])
    out = F.complete_rows(pl)
    assert list(out["pos"]) == ["MID"] and "dropped 2" in capsys.readouterr().out


def test_regime_score_is_none_when_an_input_is_missing():
    import build_league_regimes as B
    sm = {"evidence_n": 40, "independent_matchdays": 10, "recent_roi": {"roi": -0.1},
          "max_drawdown_units": None, "calibration_error": 0.02}
    clv = {"n_certified": 0, "mean_clv_pct": None}
    rs = B.repair_score("REPAIR", sm, clv, {"status": "NO_DATA"})
    assert rs["repair_priority_terms"]["drawdown"] is None and rs["repair_priority_score"] is None
    sm["max_drawdown_units"] = 2.0
    assert B.repair_score("REPAIR", sm, clv, {"status": "NO_DATA"})["repair_priority_score"] is not None
    # no evidence behind a term -> it weighs 0 (computed), even with the input missing
    sm.update(evidence_n=0, recent_roi={"roi": None}, max_drawdown_units=None)
    assert B.repair_score("LEARNING", sm, clv, {"status": "NO_DATA"})["repair_priority_score"] == 0


def test_track_registry_keeps_absent_policy_values_absent():
    import build_track_registry as T
    import inspect
    src = inspect.getsource(T)
    assert '"hard_max_odds", 3.0' not in src and '"base_fraction", 0.0' not in src


def test_quota_unknown_stops_the_odds_backfill(monkeypatch, tmp_path):
    import backfill_af_odds as O

    class R:
        status_code, headers = 200, {}
        def json(self):
            return {"response": []}
    monkeypatch.setattr(O, "_KEY", "k")
    monkeypatch.setattr(O, "ODDS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(O.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(O, "_quota_hit", False)
    assert O._fetch_one({"fixture_id": 1}) is None and O._quota_hit is True


def test_movement_entry_band_needs_real_timestamps():
    import movement_segments as M
    s = pd.DataFrame({"odds": [2.0], "signal_date": ["2026-10-01"], "match_date": ["2026-10-02"]})
    out = M._bands(s)
    assert pd.isna(out["entry_band"].iloc[0])                     # dates are not timestamps


def test_edge_replay_has_no_residual_weight_default():
    src = (ROOT / "scripts" / "edge_engine_replay.py").read_text(encoding="utf-8")
    assert ".get(key, 0.10)" not in src


def test_snapshot_prop_leagues_from_registry():
    src = (ROOT / "scripts" / "snapshot_prop_odds.py").read_text(encoding="utf-8")
    assert "config.PROP_LEAGUES" not in src and "LR.prop_leagues()" in src and '"fixture_id"' in src


def test_cloudbet_probe_has_no_league_word_list():
    src = (ROOT / "scripts" / "cloudbet_probe.py").read_text(encoding="utf-8")
    assert '"league one"' not in src and 'provider_name(lg, "cloudbet")' in src
