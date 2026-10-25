"""Weekly per-league research panel (src/research/league_weekly.py + league_weekly_eval.py), 2026-10-25.

Synthetic edge-shadow logs in a tmp output dir; nothing here reads or writes the committed output/.
"""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import market_edge as ME
from src.research import league_weekly as LW
from src.research import league_weekly_eval as EV

ROOT = Path(__file__).resolve().parents[1]
SAT = "2026-10-31"                     # ISO 2026-W44 (Mon 26 Oct .. Sun 1 Nov)
FIRST_SEEN = "2026-10-27T09:00:00Z"
LOCK_KO = "2026-10-27T19:45:00Z"       # Tuesday-night fixture = the week's first kickoff
LEAGUES = ("League One", "Brazil Serie A")


def _registry(tmp_path):
    entries = []
    for lg, mt, st in (("League One", "standard", "PAPER"), ("Brazil Serie A", "new_format", "ACTIVE_LOW")):
        for m, sides in (("OU25", ("OVER", "UNDER")), ("BTTS", ("YES",)), ("OVER15", ("OVER",)), ("OVER35", ("OVER",))):
            for s in sides:
                entries.append({"key": f"{lg}|{m}|{s}|{mt}", "league": lg, "market": m, "side": s, "model_type": mt,
                                "research_regime": "REPAIR" if lg == "League One" and m == "OU25" else "LEARNING",
                                "production_state": "BLOCKED" if m == "OVER35" else st})
    p = tmp_path / "track_registry.json"
    p.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return p


def _row(league, home, away, fam, ko=f"{SAT}T15:00:00Z", seen=FIRST_SEEN, ev_lb=-0.05, side=None, src="MEDIAN_TWO_SIDED_BOOKS",
         p_model=0.55, legacy_edge=0.02, legacy_tier="AVOID", fid=None, state="NO_BET", odds=1.95):
    a, b = ME.SIDES[fam]
    side = side if side is not None else a
    priced = src in LW.TWO_SIDED
    mt = "standard" if league == "League One" else "new_format"
    nan = ""
    return {
        "selection_engine_version": "market_edge_v2", "prediction_ts": seen,
        "fixture_id": fid or f"{ko[:10]}|{home}|{away}", "match_date": ko[:10], "kickoff_utc": ko,
        "league": league, "home_team": home, "away_team": away, "model_type": mt, "market": fam,
        "legacy_engine_version": "legacy_raw_edge", "legacy_raw_edge": legacy_edge, "legacy_tier": legacy_tier,
        "legacy_side": a, "legacy_threshold": 0.04, "legacy_odds": odds,
        "new_state": state, "new_side": side if priced else "", "new_reason": "x", "track": mt,
        "decision_state": state, "decision_reason": "x", "side": side if priced else "",
        "p_model_raw": p_model, "p_market_fair": 0.5 if priced else nan, "p_market_fair_power": nan,
        "p_market_fair_proportional": nan, "devig_method": "power", "p_decision": 0.51 if priced else nan,
        "p_lower_bound": 0.49 if priced else nan, "market_residual_raw": 0.2 if priced else nan,
        "market_residual_adjusted": nan, "residual_weight": 0.1, "odds": odds if priced else nan,
        "odds_source": "ROW", "best_book_odds": odds, "market_overround": 0.06 if priced else nan,
        "ev_point": (ev_lb + 0.03) if priced else nan, "ev_lower_bound": ev_lb if priced else nan,
        "effective_sample_size": 0, "uncertainty_penalty": 0.02 if priced else nan, "penalty_breakdown": "",
        "clv_track_n": 0, "clv_track_mean": nan, "clv_track_median": nan, "clv_track_positive_pct": nan,
        "clv_evidence_state": "NONE", "market_source": src if priced else "NONE",
        "books_used": "a;b" if priced else "", "n_books": 2 if priced else 0, "n_one_sided_books": 0,
        "n_invalid_books": 0, "price_timestamp": seen if priced else "", "minutes_to_kickoff": 100.0,
        "price_age_minutes": 1.0,
    }


def _shadow(n_fixtures=7, leagues=LEAGUES, seed=0):
    rows = []
    for lg in leagues:
        for i in range(n_fixtures):
            ko = LOCK_KO if i == 0 else f"{SAT}T15:00:00Z"
            for fam in ("OU25", "BTTS", "OU15", "OU35"):
                rows.append(_row(lg, f"{lg[:3]}H{i}", f"{lg[:3]}A{i}", fam, ko=ko, ev_lb=round(-0.01 * i - 0.001, 4)))
    return pd.DataFrame(rows)


@pytest.fixture
def env(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    return {"out": out, "reg": _registry(tmp_path),
            "kick": pd.DataFrame(columns=["league", "home_team", "away_team", "match_date", "kickoff_utc"]),
            "lock": {"protected_tracks_modified": False}}


def _run(env, shadow, now):
    shadow.to_csv(env["out"] / "edge_engine_shadow.csv", index=False)
    r = LW.run(now=now, out_dir=env["out"], registry_path=env["reg"], kickoffs=env["kick"], lock=env["lock"])
    assert "error" not in r, r
    return r


def _panel(env):
    return pd.read_csv(env["out"] / "league_weekly_research.csv", dtype=str, keep_default_na=False)


AFTER_LOCK = pd.Timestamp("2026-10-27T20:00:00Z")


def test_rows_per_league_and_max_five(env):
    _run(env, _shadow(7), AFTER_LOCK)
    p = _panel(env)
    assert set(p["league"]) == set(LEAGUES)
    g = p.groupby(["league_week_id", "market"]).size()
    assert (g == 5).all() and len(g) == len(LEAGUES) * 4
    assert set(p["league_market_rank"].astype(int)) == {1, 2, 3, 4, 5}
    assert (p["selection_mode"] == "WEEKLY_LEAGUE_RESEARCH").all()


def test_fewer_fixtures_never_manufactured(env):
    _run(env, _shadow(3), AFTER_LOCK)
    assert (_panel(env).groupby(["league_week_id", "market"]).size() == 3).all()


def test_ranking_is_rs1_and_deterministic(env, tmp_path):
    sh = _shadow(7)
    _run(env, sh, AFTER_LOCK)
    a = _panel(env)
    one = a[(a["league"] == "League One") & (a["market"] == "OU25")].sort_values("league_market_rank")
    ev = one["ev_lower_bound"].astype(float).tolist()
    assert ev == sorted(ev, reverse=True)
    env2 = dict(env, out=tmp_path / "o2")
    env2["out"].mkdir()
    _run(env2, sh.sample(frac=1.0, random_state=3).reset_index(drop=True), AFTER_LOCK)
    b = _panel(env2)
    cols = [c for c in LW.PANEL_FROZEN if c != "created_at"]
    key = ["league_week_id", "market", "league_market_rank"]
    pd.testing.assert_frame_equal(a.sort_values(key)[cols].reset_index(drop=True),
                                  b.sort_values(key)[cols].reset_index(drop=True))


def test_first_sight_byte_identical_reruns(env):
    sh = _shadow(7)
    _run(env, sh, AFTER_LOCK)
    files = [env["out"] / n for n in ("league_weekly_research.csv", "league_weekly_candidates.csv")]
    before = [f.read_bytes() for f in files]
    _run(env, sh, AFTER_LOCK + pd.Timedelta(hours=1))
    assert [f.read_bytes() for f in files] == before
    # a later, different opinion of the same fixtures changes nothing already frozen
    later = sh.copy()
    later["prediction_ts"] = "2026-10-28T09:00:00Z"
    later["ev_lower_bound"] = 0.5
    later["kickoff_utc"] = later["kickoff_utc"].where(later["kickoff_utc"] != LOCK_KO, "2026-10-28T19:45:00Z")
    _run(env, later, AFTER_LOCK + pd.Timedelta(days=1))
    assert [f.read_bytes() for f in files] == before


def test_no_post_kickoff_observation_and_late_arrivals_are_not_members(env):
    sh = _shadow(3, leagues=("League One",))
    late = _row("League One", "LateH", "LateA", "OU25", seen="2026-10-28T09:00:00Z", ev_lb=0.30)   # after lock
    post = _row("League One", "PostH", "PostA", "OU25", ko="2026-10-27T08:00:00Z", ev_lb=0.40)    # seen after kickoff
    _run(env, pd.concat([sh, pd.DataFrame([late, post])], ignore_index=True), pd.Timestamp("2026-10-28T10:00:00Z"))
    p = _panel(env)
    assert not p["home_team"].isin(["LateH", "PostH"]).any()
    c = pd.read_csv(env["out"] / "league_weekly_candidates.csv", dtype=str, keep_default_na=False)
    assert "LateH" in set(c["home_team"]) and "PostH" not in set(c["home_team"])
    assert (pd.to_datetime(p["first_seen_at"], utc=True) < pd.to_datetime(p["week_lock_ts"], utc=True)).all()
    assert (pd.to_datetime(p["first_seen_at"], utc=True) < pd.to_datetime(p["kickoff_utc"], utc=True)).all()
    h = json.loads((env["out"] / "league_market_health.json").read_text())
    wk = [v for v in h["leagues"]["League One"].values() if v["league_week_id"].endswith("W44")]
    assert wk and wk[0]["late_after_lock"] == 1                  # the late OU25 candidate row


def test_before_lock_is_provisional_only(env):
    _run(env, _shadow(7), pd.Timestamp("2026-10-27T12:00:00Z"))
    assert not (env["out"] / "league_weekly_research.csv").exists() or _panel(env).empty
    rep = json.loads((env["out"] / "weekly_league_report.json").read_text())
    rows = rep["leagues"]["League One"]["top5"]["OU25"]
    assert len(rows) == 5 and {r["status"] for r in rows} == {"PROVISIONAL"}


def test_missing_price_fails_closed(env):
    sh = _shadow(3, leagues=("League One",))
    sh = pd.concat([sh, pd.DataFrame([_row("League One", "NoPH", "NoPA", "OU25", src="NONE", ev_lb=0.9)])])
    _run(env, sh, AFTER_LOCK)
    p = _panel(env)
    x = p[p["home_team"] == "NoPH"].iloc[0]
    assert x["price_quality_state"] == "INSUFFICIENT" and x["league_market_rank"] == "4"
    assert x["decision_state"] in ("NO_BET", "OBSERVE") and x["ev_lower_bound"] == ""


def test_legacy_and_new_fields_are_distinct_and_copied(env):
    sh = _shadow(5, leagues=("League One",))
    sh.loc[sh["market"] == "OU25", "legacy_raw_edge"] = [0.01, 0.05, 0.03, 0.02, 0.04]
    _run(env, sh, AFTER_LOCK)
    p = _panel(env)
    o = p[p["market"] == "OU25"].sort_values("league_market_rank")
    assert o["legacy_edge"].astype(float).tolist() == [0.01, 0.05, 0.03, 0.02, 0.04]
    assert o["legacy_rank"].astype(int).tolist() == [5, 1, 3, 4, 2]
    assert o["league_market_rank"].astype(int).tolist() == [1, 2, 3, 4, 5]
    assert {"legacy_edge", "legacy_tier", "legacy_rank", "ev_lower_bound", "decision_state"} <= set(p.columns)


def test_gameweek_ids_deterministic():
    assert LW.league_week_id("USA MLS", "2026-11-02T01:30:00Z") == "USA MLS|2026-W44"     # Sunday night stays
    assert LW.league_week_id("League One", "2026-11-02T19:45:00Z") == "League One|2026-W45"
    assert LW.league_week_id("X", "2026-10-31T15:00:00Z") == LW.league_week_id("X", "2026-10-31T15:00:00Z")


def test_no_duplicate_fixture_in_ranking(env):
    sh = _shadow(3, leagues=("League One",))
    dup = sh[sh["home_team"] == "LeaH1"].copy()
    dup["fixture_id"] = "oddsapi-event-xyz"
    dup["prediction_ts"] = "2026-10-27T10:00:00Z"
    _run(env, pd.concat([sh, dup], ignore_index=True), AFTER_LOCK)
    p = _panel(env)
    assert not p.duplicated(["league_week_id", "market", "match_date", "home_team", "away_team"]).any()
    assert not p.duplicated(["league_week_id", "market", "fixture_id"]).any()


def _archives(home, away, ko, odds_side, odds_opp, mins=10):
    t = (pd.Timestamp(ko) - pd.Timedelta(minutes=mins)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [{"snapshot_ts": t, "match": f"{home} vs {away}", "market": "over25", "odds": odds_side,
             "match_date": ko[:10], "league": "League One"},
            {"snapshot_ts": t, "match": f"{home} vs {away}", "market": "under25", "odds": odds_opp,
             "match_date": ko[:10], "league": "League One"}]
    return [("standard_odds_history.csv", pd.DataFrame(rows))]


def test_settlement_appends_without_touching_frozen(env):
    sh = _shadow(3, leagues=("League One",))
    sh.loc[sh["market"] == "OU25", "side"] = "OVER"
    _run(env, sh, AFTER_LOCK)
    before = _panel(env)
    v91 = pd.DataFrame([{"fixture_id": f"{LOCK_KO[:10]}|LeaH0|LeaA0", "match_date": LOCK_KO[:10], "home_team": "LeaH0",
                         "away_team": "LeaA0", "home_goals": "2", "away_goals": "1", "result_source": "football-data"}])
    v91.to_csv(env["out"] / "v91_shadow.csv", index=False)
    r = LW.settle(now=pd.Timestamp("2026-10-28T03:00:00Z"), out_dir=env["out"],
                  archives=_archives("LeaH0", "LeaA0", LOCK_KO, 1.90, 2.00), kickoffs=env["kick"], lock=env["lock"],
                  registry_path=env["reg"])
    assert r["settled"] == 4                                   # one fixture x four markets
    after = _panel(env)
    pd.testing.assert_frame_equal(before[LW.PANEL_FROZEN], after[LW.PANEL_FROZEN])
    x = after[(after["home_team"] == "LeaH0") & (after["market"] == "OU25")].iloc[0]
    assert x["result"] == "WIN" and float(x["flat_pnl"]) == pytest.approx(0.95)
    assert float(x["closing_odds"]) == 1.9 and float(x["clv_pct"]) == pytest.approx((1.95 / 1.9 - 1) * 100, abs=1e-3)
    fair = ME.power_devig(1.90, 2.00)
    assert float(x["closing_market_fair_prob"]) == pytest.approx(fair, abs=1e-6)
    assert float(x["clv_prob"]) == pytest.approx(fair - 0.5, abs=1e-6)
    assert LW.panel_store(env["out"] / "league_weekly_research.csv").verify()["ok"] is True
    # graded once: a second pass changes nothing
    b2 = (env["out"] / "league_weekly_research.csv").read_bytes()
    LW.settle(now=pd.Timestamp("2026-10-29T03:00:00Z"), out_dir=env["out"], archives=[], kickoffs=env["kick"],
              lock=env["lock"], registry_path=env["reg"])
    assert (env["out"] / "league_weekly_research.csv").read_bytes() == b2


def test_stale_close_is_not_certified_and_insufficient_has_no_pnl(env):
    sh = _shadow(1, leagues=("League One",)).astype(object)
    sh.loc[sh["market"] == "OU25", ["market_source", "ev_lower_bound", "side"]] = ["NONE", "", ""]
    _run(env, sh, AFTER_LOCK)
    pd.DataFrame([{"fixture_id": f"{LOCK_KO[:10]}|LeaH0|LeaA0", "match_date": LOCK_KO[:10], "home_team": "LeaH0",
                   "away_team": "LeaA0", "home_goals": "0", "away_goals": "0", "result_source": "x"}]
                 ).to_csv(env["out"] / "v91_shadow.csv", index=False)
    LW.settle(now=pd.Timestamp("2026-10-28T03:00:00Z"), out_dir=env["out"],
              archives=_archives("LeaH0", "LeaA0", LOCK_KO, 1.90, 2.00, mins=90), kickoffs=env["kick"],
              lock=env["lock"], registry_path=env["reg"])
    p = _panel(env)
    x = p[p["market"] == "OU25"].iloc[0]
    assert x["price_quality_state"] == "INSUFFICIENT" and x["result"] in ("WIN", "LOSS") and x["flat_pnl"] == ""
    assert x["closing_odds"] == "" and x["close_status"] == "STALE_CLOSE_GT_30MIN"


def test_reconstructed_side_matches_engine_exactly():
    P = ME.load_params()
    now, ko = pd.Timestamp("2026-10-27T10:00:00Z"), pd.Timestamp("2026-10-27T19:45:00Z")
    qs = [ME.BookQuote("a", 2.30, 1.62, now), ME.BookQuote("b", 2.25, 1.65, now), ME.BookQuote("c", 2.4, 1.6, now)]
    c = ME.Candidate(family="BTTS", track="standard", p_model=0.32, kickoff=ko, now=now, quotes=qs,
                     row_odds_a=2.35, row_odds_b=1.63, league="League One", league_history_n=500)
    out = ME.evaluate(c, P)
    sides = {s["side"]: s for s in out.pop("_sides")}
    assert out["side"] == "NO"                                          # engine chose the other side
    row = {**{k: ("" if v is None else v) for k, v in out.items()}, "market": "BTTS", "legacy_side": "YES",
           "legacy_odds": 2.35, "market_residual_raw": out["market_residual_raw"]}
    v = LW.side_view(row, "YES", P, (2.35, 1.63))
    y = sides["YES"]
    assert v["side_source"] == "RECONSTRUCTED" and v["decision_state"] == "NO_BET"
    for k_view, k_eng in (("p_market_fair", "p_market_fair"), ("p_decision", "p_decision"),
                          ("p_lower_bound", "p_lower_bound"), ("ev_point", "ev_point"),
                          ("ev_lower_bound", "ev_lower_bound"), ("uncertainty", "uncertainty_penalty"),
                          ("market_residual", "market_residual_raw"), ("odds", "odds")):
        assert v[k_view] == pytest.approx(y[k_eng], abs=2e-6), k_view


def test_research_rows_never_touch_production(env, tmp_path):
    """Static: the module writes only its own outputs. Dynamic: production inputs are byte-identical after
    run + settle."""
    src = (ROOT / "src" / "research" / "league_weekly.py").read_text(encoding="utf-8")
    for forbidden in ("bankroll_policy", "league_market_regimes", "notified.json", "gate_resolver",
                      "LEAGUE_MARKSMAN", "THRESHOLD", "telegram"):
        assert forbidden not in src, forbidden
    writes = re.findall(r"(\w+)\s*\.\s*(?:to_csv|write_text|write_bytes)\(", src)
    assert set(writes) <= {"tmp", "new", "out"}, writes          # tmp-then-replace JSON/CSV and the store
    sh = _shadow(3)
    sh.to_csv(env["out"] / "edge_engine_shadow.csv", index=False)
    prod = {"stake_plan.csv": "date,fixture,market,side,stake_amount,deployment_state\n",
            "bets_ledger.csv": "source,match_date,home_team,away_team,side\n",
            "side_bets_ledger.csv": "source,match_date,home_team,away_team,market,side\n",
            "stake_audit.csv": "fixture,market,side,stake_amount,deployment_state\n",
            "predictions.csv": "league,date,home_team,away_team\n"}
    for n, t in prod.items():
        (env["out"] / n).write_text(t, encoding="utf-8")
    cfg = [ROOT / "config" / n for n in ("bankroll_policy.json", "league_market_regimes.json", "track_registry.json")]
    h0 = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [env["out"] / n for n in prod] + cfg}
    LW.run(now=AFTER_LOCK, out_dir=env["out"], registry_path=env["reg"], kickoffs=env["kick"], lock=env["lock"])
    LW.settle(now=AFTER_LOCK + pd.Timedelta(days=2), out_dir=env["out"], archives=[], kickoffs=env["kick"],
              lock=env["lock"], registry_path=env["reg"])
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in h0} == h0
    p = _panel(env)
    assert (p["selection_mode"] == "WEEKLY_LEAGUE_RESEARCH").all()
    assert set(p["actually_bet"]) <= {"False"}                 # nothing staked in this plan -> never a bet


def test_health_distinguishes_no_fixtures_from_missing_rows(env):
    kick = pd.DataFrame([{"league": "Brazil Serie A", "home_team": "X", "away_team": "Y", "match_date": SAT,
                          "kickoff_utc": f"{SAT}T15:00:00Z"}])
    env = dict(env, kick=kick)
    _run(env, _shadow(3, leagues=("League One",)), AFTER_LOCK)
    h = json.loads((env["out"] / "league_market_health.json").read_text())
    st = {lg: v["current"]["state"] for lg, v in h["leagues"].items()}
    assert st["Brazil Serie A"] == "MISSING_ROWS" and st["League One"] == "FROZEN"
    assert h["missing_leagues"] == ["Brazil Serie A"]
    from src import system_health as S
    reg = {"tracked_leagues": list(LEAGUES), "regime_counts": {}}
    ok = {"protected_tracks_modified": False, "changes": [], "unapproved_new_protected_tracks": []}
    c = S.weekly_research_panel(reg, ok, h, now=AFTER_LOCK.to_pydatetime())
    assert c["status"] == "AMBER" and "Brazil Serie A" in c["detail"]
    h2 = dict(h, missing_leagues=[], missing_leagues_previous_week=[])
    assert S.weekly_research_panel(reg, ok, h2, now=AFTER_LOCK.to_pydatetime())["status"] == "GREEN"
    assert S.weekly_research_panel(reg, dict(ok, protected_tracks_modified=True), h2)["status"] == "RED"


def _settled_panel():
    rng = random.Random(7)
    rows = []
    for w in range(10):
        for rk in range(1, 6):
            res = rng.choice(["WIN", "LOSS"])
            odds = 1.9
            rows.append({"league": "League One", "market": "OU25", "model_type": "standard",
                         "side": rng.choice(["OVER", "UNDER"]), "league_week_id": f"League One|2026-W{30 + w:02d}",
                         "league_market_rank": str(rk), "kickoff_utc": f"2026-07-{1 + w:02d}T15:00:00Z",
                         "home_team": f"H{w}{rk}", "away_team": f"A{w}{rk}", "fixture_id": f"f{w}{rk}",
                         "ev_lower_bound": str(0.02 - 0.01 * rk), "odds": str(odds),
                         "decision_state": "QUALIFIED" if rk == 1 else "NO_BET",
                         "price_quality_state": "TWO_SIDED_MULTI_BOOK", "result": res,
                         "flat_pnl": str(odds - 1 if res == "WIN" else -1.0),
                         "clv_pct": str(rng.uniform(-3, 3)) if rk < 4 else "", "clv_prob": "",
                         "p_model_raw": "0.55", "p_market_fair": "0.52", "p_decision": "0.53",
                         "production_selected": "True" if rk == 2 else "False", "production_selected_final": "",
                         "settle_status": "SETTLED", "prediction_engine_version": "standard_x",
                         "selection_engine_version": "market_edge_v2", "legacy_rank": str(rk), "legacy_tier": "AVOID",
                         "research_regime": "REPAIR", "production_state": "PAPER", "actually_bet": "False"})
    return pd.DataFrame(rows)


def test_eval_metrics_and_not_enough_data():
    d = _settled_panel()
    t = EV.league_table(d)
    top5 = t[(t["subset"] == "TOP5")]
    assert top5["n_settled"].sum() == 50 and set(top5["side"]) == {"OVER", "UNDER"}     # sides separate
    assert set(t["status"]) == {"NOT_ENOUGH_DATA"}                                      # < 30 per side-subset
    r1 = t[t["subset"] == "RANK1"]
    assert r1["n_settled"].sum() == 10
    one = EV.prepare(d)
    m = EV.metrics(EV.subset(one, "TOP5"), "k")
    pnl = pd.to_numeric(d["flat_pnl"])
    assert m["roi"] == pytest.approx(round(pnl.sum() / 50, 4)) and m["gameweeks"] == 10
    assert m["roi_ci_lo"] <= m["roi"] <= m["roi_ci_hi"]
    assert EV.metrics(EV.subset(one, "TOP5"), "k") == m                                  # seeded, deterministic
    mono = EV.rank_monotonicity(d)
    assert set(mono["rank"]) == {1, 2, 3, 4, 5}
    empty = EV.league_table(pd.DataFrame())
    assert empty.empty
    assert EV.evaluate(pd.DataFrame())["rows"] == 0


def test_page_reproduces_eval_exactly(tmp_path, monkeypatch):
    import config
    from dashboard_data import league_research as LR
    d = _settled_panel()
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    d.to_csv(tmp_path / "league_weekly_research.csv", index=False)
    panel = LR.load_panel()
    want = EV.league_table(d)
    got = LR.metric_table(panel, "League One")
    pd.testing.assert_frame_equal(got, want[want["league"] == "League One"].reset_index(drop=True))
    pd.testing.assert_frame_equal(LR.history_grid(panel, "League One", "OU25"), EV.weekly_grid(d, "League One", "OU25"))
    page = next((ROOT / "pages").glob("18_*_League_Research.py")).read_text(encoding="utf-8")
    assert "LR.metric_table(" in page and "LR.history_grid(" in page
    assert ".groupby(" not in page and ".agg(" not in page and "read_parquet" not in page
    from dashboard_access import PAGES
    assert any(p.startswith("pages/18_") and need == "admin" for p, *_, need in PAGES)
    order = [rg for _, rg in LR.league_order([{"league": "A", "research_regime": "PROTECTED"},
                                              {"league": "B", "research_regime": "LEARNING"},
                                              {"league": "C", "research_regime": "REPAIR"}])]
    assert order == ["REPAIR", "LEARNING", "PROTECTED"]


def test_contract_for_panel_rows(env):
    from src.contracts import SCHEMAS, validate
    _run(env, _shadow(5), AFTER_LOCK)
    v = validate(_panel(env), SCHEMAS["LEAGUE_WEEKLY_RESEARCH_ROW"])
    assert v == [], v
