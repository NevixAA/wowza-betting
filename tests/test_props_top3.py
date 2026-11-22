"""Props TOP-3 per league per gameweek (owner 2026-11-22, "Like TOP-3: 3 per league, stake by proof").

T1 best value / T2 safest / T3 risky value among priced props with positive value; the team TOP-3's league-week and
freeze conventions; the shared coherence gate; stake 0 with "No stake: props not proven yet"; no retired tier label
(SNIPER / MARKSMAN / VALUABLE) and no "paper" anywhere in the props path."""
from __future__ import annotations

import inspect
import json
import re

import pandas as pd
import pytest

from src import props_top3 as P
from src import tip_coherence as TC

NOW = pd.Timestamp("2026-11-23T10:00:00Z")
KO = "2026-11-24T19:45:00+00:00"          # Tuesday evening -> lock; freeze = lock - 120 min
ENV = {"TOP3_LEAGUE_ENABLED": "true", "TOP3_FREEZE_LEAD_MIN": "120"}
RETIRED = re.compile(r"\b(SNIPER|MARKSMAN|VALUABLE)\b", re.IGNORECASE)


def _row(pid, name, market, p, odds, fair=None, basis="assumed_overround", ko=KO, pos="F", match="Home FC vs Away",
         fixture=1, league="Championship"):
    return {"date": ko[:10], "kickoff_utc": ko, "league": league, "match": match, "fixture_id": fixture,
            "player_id": pid, "player_name": name, "team": "Home FC", "position": pos, "market": market,
            "model_prob": p, "market_odds": odds, "fair_implied": fair, "fair_basis": basis, "tier": "AVOID"}


def _board():
    return pd.DataFrame([
        _row(1, "Value Guy", "goals", 0.40, 3.50),                         # EV +0.40 vs quote
        _row(2, "Safe Guy", "sot", 0.80, 1.40),                            # EV +0.12, highest prob
        _row(3, "Long Guy", "goals", 0.12, 11.0),                          # EV +0.32, longest odds
        _row(4, "Fair Guy", "sot", 0.50, 1.80, fair=0.45, basis="two_sided_devig"),   # +0.111 vs fair
        _row(5, "Neg Guy", "sot", 0.30, 2.50),                             # EV -0.25: never a pick
        _row(6, "No Price", "sot", 0.90, None),                            # unpriced: not a candidate
        _row(7, "Rare Guy", "sot3", 0.30, 9.0),                            # rare market: research only
        _row(8, "Keeper", "goals", 0.30, 9.0, pos="G"),                    # GK attacking: blocked
    ])


ALL = lambda s: pd.Series(True, index=s.index)  # noqa: E731


def _cands(now=NOW):
    return P.build_candidates(_board(), now, ALL)


def test_candidates_are_priced_upcoming_non_rare_and_state_their_value_basis():
    c = _cands().set_index("player_id")
    assert set(c.index) == {"1", "2", "3", "4", "5"}
    assert c.loc["4", "value_basis"] == P.BASIS_FAIR and abs(float(c.loc["4", "value"]) - (0.5 / 0.45 - 1)) < 1e-6
    assert c.loc["1", "value_basis"] == P.BASIS_QUOTE and abs(float(c.loc["1", "value"]) - 0.40) < 1e-6
    assert c["league_week_id"].eq("Championship|2026-W48").all()
    assert P.build_candidates(_board(), pd.Timestamp("2026-11-25T00:00:00Z"), ALL).empty   # after kickoff


def test_roles_t1_best_value_t2_safest_t3_longest_odds_and_never_a_non_positive_one():
    c = _cands()
    freeze = pd.Timestamp(KO) - pd.Timedelta(minutes=120)
    picks, excl = P.select_week(c, freeze, freeze)
    by = {x["role"]: x["player_name"] for x in picks}
    assert by == {"T1_BEST_VALUE": "Value Guy", "T2_SAFEST": "Safe Guy", "T3_RISKY_VALUE": "Long Guy"}
    assert excl[P.R_NO_VALUE] == 1 and excl[P.R_RANK] == 1        # Neg Guy out; Fair Guy ranked below


def test_never_padded_and_late_or_started_candidates_skipped():
    c = _cands()
    c = c[c["player_id"].isin(["1", "5"])]
    freeze = pd.Timestamp(KO) - pd.Timedelta(minutes=120)
    picks, excl = P.select_week(c, freeze, freeze)
    assert [x["player_name"] for x in picks] == ["Value Guy"] and excl == {P.R_NO_VALUE: 1}
    late = c.assign(first_seen_at="2026-11-24T19:00:00Z")
    assert P.select_week(late, freeze, freeze)[0] == []
    assert P.select_week(c, freeze, pd.Timestamp(KO))[1].get(P.R_STARTED) == 1


def test_freeze_due_waits_for_the_freeze_respects_the_start_and_carries_no_stake():
    c = _cands()
    st = pd.Timestamp("2026-11-22T20:00:00Z")
    early = pd.Timestamp(KO) - pd.Timedelta(minutes=121)
    assert P.freeze_due(c, set(), early, 120, st)[0].empty
    at = pd.Timestamp(KO) - pd.Timedelta(minutes=119)
    picks, weeks = P.freeze_due(c, set(), at, 120, st)
    assert len(picks) == 3 and len(weeks) == 1 and int(weeks.iloc[0]["n_picks"]) == 3
    assert picks["stake"].astype(float).eq(0.0).all()
    assert picks["stake_label"].eq("No stake: props not proven yet").all()
    assert list(picks["role"]) == list(P.ROLES) and list(picks["props_rank"]) == [1, 2, 3]
    assert P.freeze_due(c, {"Championship|2026-W48"}, at, 120, st)[0].empty        # decided once
    assert P.freeze_due(c, set(), at, 120, pd.Timestamp("2026-11-25T00:00:00Z"))[0].empty   # before the start
    assert P.freeze_due(c, set(), at, 120, None)[0].empty                           # no start -> nothing


def test_prop_events_and_the_hard_list():
    goal = TC.prop_event("goals")
    assert goal == TC.Event("FT", "PLAYER", "OVER") and TC.prop_event("sot") == goal
    assert TC.prop_event("cards") is None and TC.prop_event("nonsense") is None
    for other, mm in ((TC.Event("FT", "TOTAL", "UNDER", 0.5), True), (TC.Event("FT", "BTTS", "NO"), True),
                      (TC.Event("FT", "TOTAL", "UNDER", 2.5), True), (TC.Event("HT", "TOTAL", "UNDER", 0.5), True),
                      (TC.Event("FT", "TOTAL", "OVER", 2.5), False), (TC.Event("FT", "BTTS", "YES"), False)):
        assert TC.mismatch(goal, other).mismatch is mm, other
    assert TC.mismatch(goal, TC.prop_event("assists")).mismatch is False


def test_gate_blocks_a_scorer_against_a_sent_under_and_the_next_prop_takes_the_slot():
    c = _cands()
    fx = TC.fixture_id("2026-11-24", "Home", "Away")
    gate = TC.CoherenceGate([TC.Candidate(fx, TC.Event("FT", "TOTAL", "UNDER", 2.5), "sent TOP-3")], {})
    board = pd.DataFrame([{"league": "Championship", "date": "2026-11-24", "home_team": "Home", "away_team": "Away"}])
    at = pd.Timestamp(KO) - pd.Timedelta(minutes=119)
    picks, weeks = P.freeze_due(c, set(), at, 120, pd.Timestamp("2026-11-22T20:00:00Z"), gate, board)
    # every goal-direction prop of that match mismatches the sent Under: no pick, each one counted
    assert picks.empty and json.loads(weeks.iloc[0]["excluded"])[P.R_MISMATCH] == 4
    assert set(picks.columns) >= {"board_home_team", "board_away_team"}
    # a booking prop has no goal relation and stands
    cards = P.build_candidates(pd.DataFrame([_row(9, "Card Guy", "cards", 0.5, 2.6)]), NOW, ALL)
    gate2 = TC.CoherenceGate([TC.Candidate(fx, TC.Event("FT", "TOTAL", "UNDER", 2.5), "sent TOP-3")], {})
    picks2, _ = P.freeze_due(cards, set(), at, 120, pd.Timestamp("2026-11-22T20:00:00Z"), gate2, board)
    assert list(picks2["player_name"]) == ["Card Guy"]


def test_board_names_resolve_spelling_differences_and_refuse_ambiguity():
    b = pd.DataFrame([{"league": "Argentina Primera Division", "date": "2026-11-23", "home_team": "Instituto de Córdoba",
                       "away_team": "Atlético Huracán"},
                      {"league": "Argentina Primera Division", "date": "2026-11-23", "home_team": "Gimnasia L.P.",
                       "away_team": "Newells Old Boys"}])
    assert P.board_names(b, "Argentina Primera Division", "2026-11-23", "Instituto Cordoba", "Huracan") == \
        ("Instituto de Córdoba", "Atlético Huracán")
    assert P.board_names(b, "Argentina Primera Division", "2026-11-23", "Boca Juniors", "River Plate") is None
    assert P.board_names(b, "Serie A", "2026-11-23", "Instituto Cordoba", "Huracan") is None


def test_run_freeze_settle_notify_end_to_end(tmp_path):
    od = tmp_path
    _board().to_csv(od / "player_tips.csv", index=False)
    assert P.run(NOW, od, env={}) == {"enabled": False} and not (od / P.CANDIDATES_NAME).exists()
    gate = TC.CoherenceGate([], {})
    r1 = P.run(NOW, od, env=ENV, gate=gate, prop_league=ALL)
    assert r1["candidates_appended"] == 5 and r1["picks_appended"] == 0
    at = pd.Timestamp(KO) - pd.Timedelta(minutes=100)
    r2 = P.run(at, od, env=ENV, gate=gate, prop_league=ALL)
    assert r2["picks_appended"] == 3 and r2["weeks_decided"] == 1
    assert P.run(at + pd.Timedelta(minutes=5), od, env=ENV, gate=gate, prop_league=ALL)["picks_appended"] == 0
    sent = []
    res = P.notify(od, ENV, sender=lambda t: sent.append(t) or True, now=at, gate=gate)
    assert res["sent"] == 1 and len(sent) == 1
    msg = sent[0]
    assert "PROPS TOP-3 Championship" in msg and "T1 best value" in msg and "T2 safest" in msg \
        and "T3 risky value" in msg and "No stake: props not proven yet" in msg
    assert not RETIRED.search(msg) and "paper" not in msg.lower()
    assert P.notify(od, ENV, sender=lambda t: sent.append(t) or True, now=at, gate=gate)["sent"] == 0   # once
    # settle from the player ledger (the same player results the prediction ledger uses)
    pd.DataFrame([{"fixture_id": "1", "player_id": "1", "market": "goals", "result": "WIN"},
                  {"fixture_id": "1", "player_id": "2", "market": "sot", "result": "LOSS"},
                  {"fixture_id": "1", "player_id": "3", "market": "goals", "result": ""}]
                 ).to_csv(od / "player_ledger.csv", index=False)
    assert P.settle(pd.Timestamp("2026-11-24T20:00:00Z"), od, ENV)["settled"] == 0       # too early
    assert P.settle(pd.Timestamp("2026-11-25T08:00:00Z"), od, ENV)["settled"] == 2
    led = pd.read_csv(od / P.PICKS_NAME, dtype=str).set_index("player_id")
    assert led.loc["1", "result"] == "WIN" and float(led.loc["1", "flat_pnl"]) == pytest.approx(2.5)
    assert led.loc["2", "flat_pnl"] == "-1.0" and pd.isna(led.loc["3", "result"])
    rep = json.loads((od / P.REPORT_NAME).read_text(encoding="utf-8"))
    assert rep["leagues"]["Championship"]["settled"] == 2
    # the gate now sees the sent props picks (a later Under on that match would be the mismatch)
    sent_props = TC.sent_from_props_top3(od)
    assert len(sent_props) == 3 and all(c.event.kind == "PLAYER" for c in sent_props)


def test_never_sent_after_kickoff(tmp_path):
    od = tmp_path
    _board().to_csv(od / "player_tips.csv", index=False)
    gate = TC.CoherenceGate([], {})
    P.run(NOW, od, env=ENV, gate=gate, prop_league=ALL)
    P.run(pd.Timestamp(KO) - pd.Timedelta(minutes=100), od, env=ENV, gate=gate, prop_league=ALL)
    sent = []
    P.notify(od, ENV, sender=lambda t: sent.append(t) or True, now=pd.Timestamp(KO) + pd.Timedelta(minutes=1),
             gate=gate)
    assert sent == []


def test_no_retired_tier_no_paper_and_no_stake_in_the_props_path():
    from player_model import pipeline as PL
    from player_model import predict as PP
    from telegram_bot import notifier as N
    src = inspect.getsource(N.notify_player_props)
    body = src.split('"""')[-1]                                   # code after the docstring
    assert not RETIRED.search(body) and "paper" not in body.lower()
    assert "props_top3" in body
    assert not hasattr(PP, "tag_paper_feed") and not hasattr(PP, "_kelly_stake")
    assert PP._classify_tier(0.60, 2.5, 0.40, 1.0, 3, 0.9, "sot2", "F") == PP.RESEARCH_TIER == "RESEARCH"
    psrc = inspect.getsource(PL)
    assert '"PAPER"' not in psrc and "tag_paper_feed" not in psrc
    assert 'drop(columns=["kelly_stake"]' in psrc
    assert not RETIRED.search(inspect.getsource(P).split('"""', 2)[-1])   # module code below its docstring
    for retired_fn in (N.notify_props_daily_digest,):
        assert retired_fn() is False


def test_data_truth_rule_7_covers_the_props_top3_ledger():
    from src.integrity import data_truth as DT
    assert ("props_top3_ledger.csv", "match_date", "role", None) in DT.TIP_RECORDS


def test_dashboard_view_reads_the_picks_file(tmp_path):
    from dashboard_data import props_top3_view as V
    od = tmp_path
    _board().to_csv(od / "player_tips.csv", index=False)
    gate = TC.CoherenceGate([], {})
    P.run(NOW, od, env=ENV, gate=gate, prop_league=ALL)
    P.run(pd.Timestamp(KO) - pd.Timedelta(minutes=100), od, env=ENV, gate=gate, prop_league=ALL)
    assert V.picks_path(od).name == P.PICKS_NAME
    picks = V.load_picks(od)
    lat = V.latest_by_league(picks)
    assert list(lat["Role"]) == ["T1 best value", "T2 safest", "T3 risky value"]
    assert lat["Stake"].eq("No stake: props not proven yet").all() and lat["Result"].eq("waiting").all()
    rt = V.role_table(picks).set_index("Role")
    assert rt.loc["All props TOP-3", "Picks"] == 3 and rt.loc["All props TOP-3", "Waiting"] == 3
    assert not V.market_table(picks).empty and not V.league_table(picks).empty
    led = pd.DataFrame({"match_date": ["2026-11-20", "2026-11-22", "2026-11-22", "2026-11-23"],
                        "result": ["WIN", "LOSS", "VOID", ""]})
    assert V.prediction_counts(led, "2026-11-21") == {"predictions": 3, "settled": 1, "void": 1, "waiting": 1}
    assert V.prediction_counts(pd.DataFrame()) == {"predictions": 0, "settled": 0, "void": 0, "waiting": 0}


def test_wording_scan_fails_on_paper_in_the_props_page_sources(tmp_path):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ui_wording_scan", root / "scripts" / "ui_wording_scan.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    word = re.compile(r"\bpaper\b", re.IGNORECASE)
    # the real props page + views carry no user-facing "paper" literal
    assert W.scan_telegram_sources(word, root, W.STATIC_PAGE_SOURCES) == []
    assert any(g.startswith("pages/11_") for g in W.STATIC_PAGE_SOURCES)
    assert {"dashboard_data/props_view.py", "dashboard_data/props_top3_view.py"} <= set(W.STATIC_PAGE_SOURCES)
    assert "src/props_top3.py" in W.TELEGRAM_SOURCES
    # a planted literal is caught
    (tmp_path / "dashboard_data").mkdir()
    (tmp_path / "dashboard_data" / "props_view.py").write_text('X = st.caption("Paper only")\n', encoding="utf-8")
    hits = W.scan_telegram_sources(word, tmp_path, W.STATIC_PAGE_SOURCES)
    assert len(hits) == 1 and hits[0]["kind"].startswith("source literal")
