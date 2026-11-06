"""fix9/defaults (owner 2026-11-04): the real defaults a wide scan of data-truth-100 found.

Each test pins one rule: no invented value reaches data, tips, stakes or displays; behaviour is
otherwise unchanged (same threshold VALUES, same skips).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import config


# ── 1. the side-market GLOBAL bar is ONE declared policy, labelled everywhere ───────────────────
def test_global_bar_values_unchanged_and_declared_once():
    from src import side_thresholds as ST
    assert config.SIDE_GLOBAL_BARS == {"sniper": 0.10, "marksman": 0.08, "valuable": 0.04}
    assert ST.global_bars() == (0.10, 0.08, 0.04)


@pytest.mark.parametrize("lp, expect", [
    ({"sniper_th": 0.06, "marksman_th": 0.04, "approved": True}, (0.06, 0.04, 0.04, "LEAGUE")),
    ({"sniper_th": 0.06, "marksman_th": 0.04, "approved": False}, (0.10, 0.08, 0.04, "GLOBAL")),
    ({}, (0.10, 0.08, 0.04, "GLOBAL")),                                   # league not in the file
    (None, (0.10, 0.08, 0.04, "GLOBAL")),
    ({"sniper_th": 0.07, "marksman_th": 0.05, "drop": False}, (0.07, 0.05, 0.04, "LEAGUE")),  # pre-`approved` file
    ({"sniper_th": 0.07, "marksman_th": 0.05, "drop": True}, (0.10, 0.08, 0.04, "GLOBAL")),
    ({"sniper_th": 0.07, "approved": True}, (0.07, 0.08, 0.04, "GLOBAL")),  # one own bar: not a league tier
])
def test_side_bars_rule(lp, expect):
    from src.side_thresholds import side_bars
    got = side_bars(lp)
    assert got[:3] == pytest.approx(expect[:3]) and got[3] == expect[3]


def test_three_callers_use_the_same_choice():
    from src.integrity import side_edge_devig as SE
    from src.research import edge_replay as ER
    from src.side_thresholds import side_bars
    params = {"btts": {"Championship": {"sniper_th": 0.09, "marksman_th": 0.07, "approved": True},
                       "Ligue 2": {"sniper_th": 0.07, "marksman_th": 0.05, "approved": False}}}
    for lg in ("Championship", "Ligue 2", "Argentina Primera Division"):
        b = side_bars(params["btts"].get(lg, {}))
        assert SE.thresholds(params, "btts", lg) == pytest.approx(b[:3])
        assert SE.thresholds_with_source(params, "btts", lg)[3] == b[3]
    assert SE.DEFAULT_BARS == (0.10, 0.08, 0.04)
    # edge_replay: p 0.60 at odds 2.00 -> edge 0.60 - 0.5/1.08 = 0.137 -> SNIPER on every bar here
    t, e = ER.legacy_side_tier(0.60, 2.00, "Ligue 2", "btts", params)
    assert t == "SNIPER" and e == pytest.approx(0.60 - 0.5 / ER.SIDE_OVERROUND)
    # 0.09 edge: MARKSMAN on the global bar (0.10 / 0.08), SNIPER on Championship's own 0.09
    p = 0.095 + (1 / 2.0) / ER.SIDE_OVERROUND
    assert ER.legacy_side_tier(p, 2.0, "Ligue 2", "btts", params)[0] == "MARKSMAN"
    assert ER.legacy_side_tier(p, 2.0, "Championship", "btts", params)[0] == "SNIPER"


def test_ledger_writes_threshold_source(tmp_path, monkeypatch):
    from src import ledger as L
    monkeypatch.setattr(L, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    df = pd.DataFrame([
        {"date": "2026-11-05", "league": "Championship", "home_team": "A", "away_team": "B", "market": "btts",
         "side": "YES", "signal_tier": "SNIPER", "market_odds": 2.0, "edge": 0.11, "ev": 0.2,
         "model_prob": 0.6, "model_type": "standard", "threshold_source": "LEAGUE"},
        {"date": "2026-11-05", "league": "Argentina Primera Division", "home_team": "C", "away_team": "D",
         "market": "btts", "side": "YES", "signal_tier": "MARKSMAN", "market_odds": 2.0, "edge": 0.09,
         "ev": 0.1, "model_prob": 0.6, "model_type": "new_format", "threshold_source": "GLOBAL"}])
    L.append_side_market_tips(df)
    out = pd.read_csv(tmp_path / "side_bets_ledger.csv")
    assert out.set_index("home_team")["threshold_source"].to_dict() == {"A": "LEAGUE", "C": "GLOBAL"}
    assert "threshold_source" in L.SIDE_LEDGER_COLS


def test_telegram_and_dashboard_label_the_global_bar():
    from telegram_bot import notifier as N
    from telegram_bot import tips_table as TT
    from dashboard_data import tips_view as TV
    assert "GLOBAL" in N.threshold_source_line("GLOBAL")
    assert N.threshold_source_line("LEAGUE") == "" and N.threshold_source_line(None) == ""
    assert N.threshold_source_line(float("nan")) == ""
    assert TT.gap_text("SNIPER", "GLOBAL").endswith(TT.GLOBAL_MARK)
    assert not TT.gap_text("SNIPER", "LEAGUE").endswith(TT.GLOBAL_MARK)
    assert TT._threshold_source(float("nan")) is None and TT._threshold_source("global") == "GLOBAL"
    assert TV.BAR_LABEL["GLOBAL"] == "GLOBAL"            # the 'Bar' column reads "Bar: GLOBAL" (dashtruth)


def test_png_draws_the_global_footnote():
    pytest.importorskip("PIL")
    from telegram_bot import tips_table as TT
    rows = pd.DataFrame([{"time": "12:00", "league": "Argentina Primera Division", "rating": 5,
                          "home_team": "A", "away_team": "B", "model": "new_format", "tier": "SNIPER",
                          "market": "BTTS YES", "odds": 2.0, "edge": 11.0, "threshold_source": "GLOBAL"}])
    assert TT.render_pngs(rows, "t")                                  # renders with the marked Gap


# ── 2. the FPL minutes baseline reads real history, missing stays missing ──────────────────────
def _panel():
    rows = []
    # player 1: GW1 start 90', GW2 start 45', GW3 sub 20', GW4 (pending) ; single fixtures
    for gw, st, mins in ((1, 1, 90), (2, 1, 45), (3, 0, 20)):
        rows.append({"element": 1, "gw": gw, "status": "FINAL", "n_fix": 1, "actual_starts": st,
                     "actual_minutes": mins})
    rows.append({"element": 1, "gw": 4, "status": "PENDING", "n_fix": 1})
    # player 2: GW1 double gameweek (excluded), GW2 not played
    rows.append({"element": 2, "gw": 1, "status": "FINAL", "n_fix": 2, "actual_starts": 2, "actual_minutes": 180})
    rows.append({"element": 2, "gw": 2, "status": "FINAL", "n_fix": 1, "actual_starts": 0, "actual_minutes": 0})
    rows.append({"element": 2, "gw": 3, "status": "PENDING", "n_fix": 1})
    return pd.DataFrame(rows)


def test_own_minutes_history_uses_only_earlier_single_fixture_gameweeks():
    from player_model.fantasy_panel import add_own_minutes_history
    d = add_own_minutes_history(_panel()).set_index(["element", "gw"])
    assert math.isnan(d.loc[(1, 1), "hist_play_share"])               # nothing before GW1
    assert d.loc[(1, 2), "hist_p60_given_start"] == 1.0                # GW1: start reaching 90'
    assert d.loc[(1, 3), "hist_p60_given_start"] == 0.5                # 90' and 45'
    assert d.loc[(1, 4), "hist_play_share"] == 1.0 and d.loc[(1, 4), "hist_gws"] == 3
    assert d.loc[(1, 4), "hist_starts"] == 2 and d.loc[(1, 4), "hist_p60_given_start"] == 0.5
    assert d.loc[(2, 2), "hist_gws"] == 0                              # the double gameweek never counts
    assert d.loc[(2, 3), "hist_play_share"] == 0.0                     # a real zero (did not play)
    assert math.isnan(d.loc[(2, 3), "hist_p60_given_start"])           # no start -> unknown, not 0.92


def test_heuristic_minutes_has_no_typed_numbers():
    from player_model.fantasy_residual import heuristic_minutes
    x = pd.DataFrame({"chance_next_f": [1.0, np.nan, 1.0, 1.0], "min_per_team_game": [80, 80, 80, 80],
                      "start_rate": [0.8, 0.8, np.nan, 0.8], "hist_play_share": [0.9, 0.9, 0.9, np.nan],
                      "hist_p60_given_start": [0.75, 0.75, 0.75, 0.75], "n_fix": [1, 1, 1, 1]})
    h = heuristic_minutes(x)
    assert h.loc[0, "m60"] == pytest.approx(1.0 * 0.8 * 0.75)
    assert h.loc[0, "m0"] == pytest.approx(1 - 0.9)
    assert h.loc[1:].isna().all().all()                                # any missing input -> missing
    blank = heuristic_minutes(x.assign(n_fix=0))
    assert (blank["m0"] == 0.998).all()                                # a known blank stays a real zero
    no_hist = heuristic_minutes(x.drop(columns=["hist_play_share", "hist_p60_given_start"]).iloc[:1])
    assert no_hist.isna().all().all()


# ── 3. small defaults ───────────────────────────────────────────────────────────────────────────
def test_kelly_and_stake_on_a_missing_price_are_missing():
    from src import betting as B
    assert math.isnan(B._kelly(0.6, float("nan")))
    assert math.isnan(B._kelly(float("nan"), 2.0))
    assert B._kelly(0.6, 1.0) == 0.0                                   # a real price with no edge
    assert B._kelly(0.6, 2.0) == pytest.approx(0.2)
    assert math.isnan(B._stake(float("nan"), 0.6, float("nan"), 0.04))
    assert B._stake(0.01, 0.6, 2.0, 0.04) == 0.0


def test_settled_stats_count_settled_rows_only():
    from dashboard_data.v9 import _settled_stats
    t = pd.DataFrame({"result": ["WIN", "LOSS", "WIN"], "pnl": [1.0, -1.0, np.nan]})
    s = _settled_stats(t)
    assert s["n"] == s["n_settled"] == 2 and s["n_without_pnl"] == 1
    assert s["roi"] == 0.0 and s["hit"] == 0.5
    none = _settled_stats(pd.DataFrame({"result": ["WIN"], "pnl": [np.nan]}))
    assert none["n"] == 0 and none["pnl"] is None and none["roi"] is None


def test_prop_prices_never_default_to_zero():
    from player_model import odds_fetcher as OF
    bms = [{"title": "B1", "markets": [{"key": "player_shots_on_target", "outcomes": [
        {"name": "Over", "description": "Joe Bloggs", "point": 0.5, "price": 1.9},
        {"name": "Over", "description": "No Price", "point": 0.5},
        {"name": "Over", "description": "No Point", "price": 3.0}]}]}]
    best = OF._best_price(bms, "player_shots_on_target", 0.5)
    assert set(best) == {OF._norm("Joe Bloggs")} and best[OF._norm("Joe Bloggs")] == 1.9
    assert OF._price({"price": None}) is None and OF._price({"price": "x"}) is None
    assert OF._price({}) is None and OF._price({"price": "2.5"}) == 2.5


def test_fixture_and_player_ids_never_default_to_zero(tmp_path, monkeypatch):
    from player_model import data_fetcher as DF
    assert DF._fixture_id_of({}) is None
    assert DF._fixture_id_of({"fixture": {"id": 0}}) is None
    assert DF._fixture_id_of({"fixture": {"id": 7}}) == 7 and DF._fixture_id_of({"id": 9, "_raw": 1}) == 9
    assert DF._is_cached(None) is False


def test_policy_base_fraction_reader():
    from src.bankroll_policy import positive_fraction
    assert positive_fraction({"base_fraction": 0.01}) is True
    assert positive_fraction({"base_fraction": 0}) is False
    assert positive_fraction({}) is False and positive_fraction({"base_fraction": None}) is False


def test_engine_clv_sample_unknown_is_not_zero():
    from src import market_edge as ME
    import inspect
    src = inspect.getsource(ME._evaluate)
    assert 'clv.get("n", 0)' not in src and 'clv_track_n=clv.get("n")' in src


# ── 4. rule 2 scans all production code for the new classes ────────────────────────────────────
def test_rule2_patterns_catch_the_new_classes():
    from src.integrity import data_truth as DT
    code = "\n".join([
        'price = float(o.get("price", 0))',
        'th = lp.get("sniper_th", 0.10)',
        'n = int(x or 0)',
        'k = [kelly(p, o) for p, o in zip(p, odds.fillna(0))]',
        'ok = d.get("name", "x")',                                     # a string default: not numeric
        'v = x or 0.5',                                                # not a zero
    ])
    kinds = [h["kind"] for h in DT.scan_file("x.py", code)]
    assert kinds.count("numeric_get_default") == 2
    assert kinds.count("or_zero") == 1 and kinds.count("fillna_zero_price") == 1
    doc = '"""It was `int(x or 0)` and .get("price", 0)."""\nx = 1\n'
    assert DT.scan_file("y.py", doc) == []                              # docstring prose is not code


def test_rule2_scans_all_production_code_and_not_tests():
    from src.integrity import data_truth as DT
    files = set(DT.SCAN_FILES)
    assert {"pipeline.py", "src/betting.py", "player_model/fantasy_residual.py",
            "dashboard_data/v9.py", "src/research/edge_replay.py", "src/integrity/side_edge_devig.py",
            "player_model/odds_fetcher.py"} <= files
    assert not any(f.startswith(("tests/", "scripts/audit/")) for f in files)
    assert len(files) > 250


def test_allowlist_entries_all_have_reasons_and_baseline_stays_zero():
    from src.integrity import data_truth as DT
    allow = json.loads(DT.ALLOWLIST.read_text(encoding="utf-8"))["entries"]
    assert allow and all(str(e.get("reason") or "").strip() for e in allow)
    assert DT._baseline().get("entries") == []
    assert int(json.loads(DT.RATCHET.read_text(encoding="utf-8"))["max_known_defaults"]) == 0
    # none of the fixed sites is allow-listed
    codes = " ".join(e["code"] for e in allow)
    for fixed in ('lp.get("sniper_th"', 'o.get("price", 0)', "ov_odds.fillna(0)", 'get("player_id", 0)',
                  '.get("id", 0)', 'pnl"), errors="coerce").fillna(0)', 'p_over25", 0.5'):
        assert fixed not in codes


def test_rule2_is_green_on_this_tree():
    from src.integrity import data_truth as DT
    c = DT.defaults()
    assert c["status"] == DT.GREEN, c


def test_rule2_red_on_a_new_default_and_on_an_unreasoned_entry(monkeypatch, tmp_path):
    from src.integrity import data_truth as DT
    monkeypatch.setattr(DT, "scan_defaults", lambda files=None: [
        {"file": "src/x.py", "kind": "numeric_get_default", "code": 'p = r.get("price", 0)'}])
    assert DT.defaults()["status"] == DT.RED
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"entries": [{"file": "src/x.py", "kind": "numeric_get_default",
                                              "code": 'p = r.get("price", 0)', "reason": ""}]}))
    monkeypatch.setattr(DT, "ALLOWLIST", allow)
    assert DT.defaults()["status"] == DT.RED                            # no reason -> refused
    allow.write_text(json.dumps({"entries": [{"file": "src/x.py", "kind": "numeric_get_default",
                                              "code": 'p = r.get("price", 0)', "reason": "counter"}]}))
    assert DT.defaults()["status"] == DT.GREEN


# ── 6. history follow-ups ───────────────────────────────────────────────────────────────────────
def test_committed_lists_hold_the_two_owner_decisions():
    from src import history_corrections as HC
    awd = pd.read_csv(HC.AWARDED_EXCLUDED)
    q = awd[awd["fixture_id"] == 816403]
    assert len(q) == 2 and set(q["home_team"]) == {"Queretaro", "Club Queretaro"}
    assert q["reason"].str.contains("abandoned").all()
    dup = pd.read_csv(HC.DUP_REMOVED, low_memory=False)
    o = dup[(dup["league"] == "USA MLS") & (dup["date"] == "2023-11-25") & (dup["away_team"] == "Columbus Crew")]
    assert len(o) == 1 and o.iloc[0]["home_team"] == "Orlando City"
    assert (o.iloc[0]["home_goals"], o.iloc[0]["away_goals"]) == (0, 0)
    assert o.iloc[0]["kept_home_team"] == "Orlando City SC" and o.iloc[0]["found_by"] == \
        "same_day_copy_wrong_score_provider_agrees"
    ref = pd.read_csv(HC.CORRECTIONS_DIR / "history_score_refused.csv")
    assert not ref["fixture_id"].isin([816403, 1142091]).any()


#: rows in output/fd_history.parquet when this test was written (2026-11-04). The file only grows
#: (Update Results appends), so this is a floor, never an exact count.
FD_HISTORY_ROW_FLOOR = 65940


def test_loader_corrections_remove_them_from_the_real_history():
    import hashlib
    from src import history_corrections as HC
    path = HC.ROOT / "output" / "fd_history.parquet"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    h = pd.read_parquet(path)
    before = h.copy(deep=True)
    out = HC.apply_history_corrections(h)
    d = pd.to_datetime(out["date"])
    q = out[(d == "2022-03-05") & out["away_team"].eq("Atlas") & out["home_team"].str.contains("Queretaro")]
    assert q.empty
    o = out[(d == "2023-11-25") & out["away_team"].eq("Columbus Crew") & out["home_team"].str.startswith("Orlando")]
    assert len(o) == 1 and (o.iloc[0]["home_goals"], o.iloc[0]["away_goals"]) == (0.0, 2.0)
    # the original is never rewritten: neither the frame handed in nor the file on disk
    pd.testing.assert_frame_equal(h, before)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    # bots append to the file: a floor and invariants, not a hard count of a growing file
    assert len(h) >= FD_HISTORY_ROW_FLOOR
    keys = HC._keys(h)
    assert len(keys) == len(set(keys))                                 # no duplicate (league, date, home, away)
    # the removals are exactly the listed rows whose key AND recorded FT match (counted independently)
    corr = HC.read_corrections()
    listed = {}
    for r, k in zip(corr.itertuples(index=False), HC._keys(corr)):
        listed.setdefault(k, set()).add(HC._ft(r.home_goals, r.away_goals))
    eligible = h["row_source"].isin(HC.HISTORY_SOURCES).tolist() if "row_source" in h.columns else [True] * len(h)
    expect = sum(1 for k, ok, a, b in zip(keys, eligible, h["home_goals"], h["away_goals"])
                 if ok and k in listed and HC._ft(a, b) in listed[k])
    rep = out.attrs["history_corrections"]
    assert rep["rows_removed"] == expect > 0
    assert len(out) == len(h) - expect
    ko = HC._keys(out)
    assert len(ko) == len(set(ko))


def _bh():
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_history_corrections")


def test_agreeing_copy_rule_on_synthetic_rows(monkeypatch):
    B = _bh()
    hist = pd.DataFrame({"league": ["USA MLS"] * 3, "date": pd.to_datetime(["2023-11-25"] * 3),
                         "home_team": ["Orlando City SC", "Orlando City", "Austin"],
                         "away_team": ["Columbus Crew", "Columbus Crew", "Dallas"],
                         "home_goals": [0.0, 0.0, 1.0], "away_goals": [2.0, 0.0, 1.0]})
    claims = pd.DataFrame({
        "league": ["USA MLS"] * 3, "day": ["2023-11-25"] * 3, "pdate": ["2023-11-25"] * 3,
        "home_team": ["Orlando City SC", "Orlando City", "Austin"],
        "away_team": ["Columbus Crew", "Columbus Crew", "Dallas"],
        "hist_hg": [0.0, 0.0, 1.0], "hist_ag": [2.0, 0.0, 1.0], "fixture_id": [1, 1, 2],
        "provider_ft_home": [0, 0, 1], "provider_ft_away": [2, 2, 1], "status": ["FT"] * 3,
        "provider_home": ["Orlando City SC"] * 2 + ["Austin"], "provider_away": ["Columbus Crew"] * 2 + ["Dallas"]})
    monkeypatch.setattr(B, "near", lambda h, pool, st: h)
    monkeypatch.setattr(B, "_with_canon", lambda h: h)
    monkeypatch.setattr(B, "history_claims", lambda h, pool, st: claims)
    rem, st = B.agreeing_copy_removals(hist, pd.DataFrame({"fixture_id": [1, 2]}))
    assert len(rem) == 1 and rem.iloc[0]["home_team"] == "Orlando City"
    assert rem.iloc[0]["kept_home_team"] == "Orlando City SC" and st["removed_rows"] == 1
    # a lone disagreeing copy is NOT proven either way
    monkeypatch.setattr(B, "history_claims", lambda h, pool, st: claims.iloc[[1]])
    rem2, _ = B.agreeing_copy_removals(hist, pd.DataFrame({"fixture_id": [1]}))
    assert rem2.empty


def test_owner_not_played_lists_every_claiming_copy(monkeypatch):
    B = _bh()
    claims = pd.DataFrame({"league": ["Mexico Liga MX"] * 2, "day": ["2022-03-05"] * 2,
                           "home_team": ["Queretaro", "Club Queretaro"], "away_team": ["Atlas"] * 2,
                           "hist_hg": [0.0, 0.0], "hist_ag": [3.0, 1.0], "fixture_id": [816403, 816403],
                           "status": ["FT", "FT"]})
    monkeypatch.setattr(B, "near", lambda h, pool, st: h)
    monkeypatch.setattr(B, "_with_canon", lambda h: h)
    monkeypatch.setattr(B, "history_claims", lambda h, pool, st: claims)
    out, st = B.owner_not_played(pd.DataFrame(), pd.DataFrame({"fixture_id": [816403], "status": ["FT"]}))
    assert len(out) == 2 and st["fixtures_not_claimed"] == []
    assert out["reason"].str.contains("abandoned").all()


def test_status_reread_mode_plans_each_old_list_once():
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    H = importlib.import_module("collect_ht_scores")
    body = {"response": [
        {"fixture": {"id": 1, "date": "2022-03-05T00:00:00+00:00", "status": {"short": "ABD"}},
         "teams": {"home": {"id": 1, "name": "A"}, "away": {"id": 2, "name": "B"}},
         "score": {"halftime": {"home": 0, "away": 0}, "fulltime": {"home": None, "away": None}}},
        {"fixture": {"id": 2, "date": "2022-03-05T00:00:00+00:00", "status": {"short": "AET"}},
         "teams": {"home": {"id": 3, "name": "C"}, "away": {"id": 4, "name": "D"}},
         "score": {"halftime": {"home": 0, "away": 0}, "fulltime": {"home": 1, "away": 1},
                   "extratime": {"home": 1, "away": 0}, "penalty": {"home": None, "away": None}}},
        {"fixture": {"id": 3, "date": "2022-03-05T00:00:00+00:00", "status": {"short": "PST"}},
         "teams": {"home": {"id": 5, "name": "E"}, "away": {"id": 6, "name": "F"}}, "score": {}}]}
    rows = {r["fixture_id"]: r for r in H.parse(body, "X", 1, "2021")}
    assert set(rows) == {1, 2} and rows[1]["status"] == "ABD" and rows[1]["ft_home"] is None
    assert (rows[2]["et_home"], rows[2]["et_away"]) == (1, 0) and rows[2]["pen_home"] is None
    j = pd.DataFrame({"league_id": [1, 1, 2], "season": ["2021", "2022", "2021"], "league": ["X", "X", "Y"]})
    done = pd.DataFrame({"league_id": [1], "season": ["2021"]})
    plan = H.status_plan(j, pd.DataFrame(), done)
    assert [(p["league_id"], p["season"]) for p in plan] == [(1, "2022"), (2, "2021")]
    comp = pd.DataFrame({"league_id": [9, 1], "league": ["Z", "X"], "season": [2020, "2022"],
                         "competition_type": ["club_league", "club_league"]})
    plan2 = H.status_plan(j, pd.DataFrame(), done, comp, all_leagues=True)
    assert len(plan2) == 3 and plan2[-1]["scope"] == "registry_leagues"


def test_status_reread_dry_run_counts_the_committed_old_lists(capsys):
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    H = importlib.import_module("collect_ht_scores")
    out = H.add_status(fetch=False)
    # Recount the plan from the committed journals: every stored HT list (journal + FT journal)
    # minus the lists the status journal already re-read. The collection keeps growing all three
    # (the status journal also holds registry-league lists), so this is a set difference.
    def keys(p):
        if not p.exists():
            return set()
        d = pd.read_parquet(p, columns=["league_id", "season"])
        return {(int(a), str(b)) for a, b in zip(d["league_id"], d["season"])}
    todo = (keys(H.JOURNAL) | keys(H.FT_JOURNAL)) - keys(H.STATUS_JOURNAL)
    assert out["status_planned_calls"] == len(todo)
    assert out["status_calls"] == 0
