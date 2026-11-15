"""UEFA club cups in props (owner question 2026-11-02, fix5/cups).

(a) a player's form / minutes / rest days must include his cup appearances (collected cup rows);
(b) opponent features in ANY fixture come from the opponent's own recent matches in ALL
    competitions, keyed by the API-Football team id - one definition for training and serving;
(c) a cup fixture's league strength is measured (src/strength club ratings), never a hand tier;
(d) the collector: one /fixtures/players call per UEFA cup fixture, budget-guarded, journaled.
Everything offline (synthetic frames, fake HTTP session).

    python -m pytest -p no:cacheprovider tests/test_cup_props_features.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_common as CC  # noqa: E402
from player_model import config as C  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model import league_quality as LQ  # noqa: E402
from player_model.setpiece import features as SPF  # noqa: E402
from src.strength import elo, lookup  # noqa: E402

IDS = {"Arsenal": 42, "Chelsea": 49, "Everton": 45, "Fulham": 36,
       "Club Brugge KV": 569, "Gent": 631, "Anderlecht": 554, "Genk": 742}
D0 = pd.Timestamp("2025-08-02")


def _player_rows(fid, date, league, home, away, goals_home=1, goals_away=0, ids=True):
    rows = []
    for team, ih in ((home, 1), (away, 0)):
        for j, pos in enumerate(("G", "D", "M", "F")):
            r = {"fixture_id": fid, "player_id": IDS[team] * 100 + j, "player_name": f"{team}-{pos}",
                 "date": date, "league": league, "home_team": home, "away_team": away, "team": team,
                 "is_home": ih, "referee": "", "season": "2025", "position": pos, "minutes": 90,
                 "started": 1, "goals": (goals_home if ih else goals_away) if pos == "F" else 0,
                 "assists": 0, "shots_total": 2, "shots_on_target": 2 if pos == "F" else 0,
                 "yellow_cards": 0, "key_passes": 1, "rating": 7.0, "fouls_committed": 1,
                 "fouls_drawn": 1, "duels_total": 4, "duels_won": 2, "team_corners": 5,
                 "red_cards": 0, "passes_total": 30, "offsides": 0, "tackles_total": 1,
                 "interceptions": 1, "dribbles_attempted": 1, "dribbles_success": 1,
                 "dribbles_past": 0, "goals_conceded": (goals_away if ih else goals_home) if pos == "G" else 0,
                 "saves": 2 if pos == "G" else 0, "penalty_scored": 0, "penalty_missed": 0,
                 "penalty_won": 0, "penalty_saved": 0}
            if ids:
                r["team_id"], r["opponent_id"] = (IDS[home], IDS[away]) if ih else (IDS[away], IDS[home])
            rows.append(r)
    return rows


def _fx(fid, date, league, home, away, gh, ga, lid, status="FT"):
    return {"fixture_id": fid, "league_id": lid, "league": league, "league_name": league,
            "competition_type": "club_league", "season": "2025", "date": str(date.date()),
            "home_id": IDS[home], "home_name": home, "away_id": IDS[away], "away_name": away,
            "ft_home": gh, "ft_away": ga, "status": status}


def _world():
    """8 Premier League rounds (player rows), 8 Belgian rounds (fixture scores only), then a
    Champions League round: Arsenal v Club Brugge KV and Chelsea v Gent (player rows of the PL
    clubs only, as before the cup collection)."""
    rows, fx = [], []
    fid = 9_100_000
    for r in range(8):
        d = D0 + pd.Timedelta(days=7 * r)
        pairs = [("Arsenal", "Chelsea"), ("Everton", "Fulham")] if r % 2 == 0 else \
                [("Chelsea", "Everton"), ("Fulham", "Arsenal")]
        for h, a in pairs:
            fid += 1
            # Chelsea concede 2 at home in round r; a player-row goal sum would miss the own goal:
            # the 90-minute score says 2, the scorer rows say 1.
            gh, ga = (1, 0) if h != "Fulham" else (0, 2)
            rows += _player_rows(fid, d, "Premier League", h, a, gh, ga)
            fx.append(_fx(fid, d, "Premier League", h, a, gh + (1 if a == "Chelsea" else 0), ga, 39))
        for h, a, gh, ga in (("Club Brugge KV", "Gent", 0, 3), ("Anderlecht", "Genk", 1, 1)):
            fid += 1
            fx.append(_fx(fid, d + pd.Timedelta(days=1), "Belgian First Division A", h, a, gh, ga, 144))
    cup_day = D0 + pd.Timedelta(days=7 * 8 + 2)
    rows += [r for r in _player_rows(9_200_001, cup_day, "Champions League", "Arsenal", "Club Brugge KV", 2, 0,
                                     ids=False) if r["team"] == "Arsenal"]
    fx.append(_fx(9_200_001, cup_day, "Champions League", "Arsenal", "Club Brugge KV", 2, 0, 2))
    return rows, pd.DataFrame(fx), cup_day


@pytest.fixture()
def world(tmp_path, monkeypatch):
    rows, fx, cup_day = _world()
    sdir = tmp_path / "strength"
    sdir.mkdir()
    fx.to_parquet(sdir / "competition_fixtures.parquet", index=False)
    monkeypatch.setattr(lookup, "DIR", sdir)              # no strength tables: club Elo NaN
    monkeypatch.setattr(FE, "COLLECT_DIR", tmp_path)
    monkeypatch.setattr(SPF, "DIR", tmp_path)

    def _clear():
        try:
            from player_model import team_identity as TI
            TI._CACHE.clear()
        except ImportError:                     # the pre-fix code has no team_identity
            pass
        lookup._LOADED.clear()
        FE._COLLECTED.clear()
        FE._LINEUP_CACHE.clear()
    _clear()
    yield rows, fx, cup_day
    _clear()


def _chelsea_conceded(fx, before):
    """Chelsea's 90-minute goals conceded in its last ROLLING_N fixtures before `before`."""
    f = fx[(fx["home_name"] == "Chelsea") | (fx["away_name"] == "Chelsea")].copy()
    f["d"] = pd.to_datetime(f["date"])
    f = f[f["d"] < before].sort_values("d").tail(C.ROLLING_N)
    return float(np.where(f["home_name"] == "Chelsea", f["ft_away"], f["ft_home"]).mean())


# ── (b) opponent features across all competitions ──────────────────────────────────────────────

def test_serving_cup_opponent_reads_its_league_matches(world):
    rows, fx, cup_day = world
    hist = pd.DataFrame(rows)                           # the raw history serving reads
    up = [{"player_id": IDS["Arsenal"] * 100 + 3, "fixture_id": 990001, "team": "Arsenal",
           "home_team": "Arsenal", "away_team": "Chelsea", "is_home": 1, "league": "Champions League",
           "date": str((cup_day + pd.Timedelta(days=30)).date())}]
    out = FE.build_upcoming_features(up, hist)
    got = out["opp_goals_conceded_pg"].iloc[0]
    # Before the fix: the cup is its own scope with no Chelsea rows -> NaN.
    assert got == pytest.approx(_chelsea_conceded(fx, cup_day + pd.Timedelta(days=30)))
    assert out["opp_sot_conceded_pg"].notna().iloc[0] and out["opp_gk_save_rate"].notna().iloc[0]


def test_training_cup_row_opponent_from_all_its_competitions(world):
    rows, fx, cup_day = world
    b = FE.build_features(rows)
    cup = b[(b["league"] == "Champions League") & (b["position"] == "F")]
    assert len(cup) == 1
    # Club Brugge KV: no player rows anywhere, but its Belgian results are collected: conceded 3
    # in each of its last ROLLING_N matches -> 3.0; shots / keeper quantities need player rows: NaN.
    assert cup["opp_goals_conceded_pg"].iloc[0] == pytest.approx(3.0)
    assert math.isnan(cup["opp_sot_conceded_pg"].iloc[0])
    assert cup["team_id"].iloc[0] == IDS["Arsenal"] and cup["opponent_id"].iloc[0] == IDS["Club Brugge KV"]


def test_goals_conceded_is_the_90_minute_score_not_the_scorer_rows(world):
    rows, fx, _ = world
    t = FE.team_match_table(pd.DataFrame(rows))
    ch = t[t["_tkey"] == f"af:{IDS['Chelsea']}"].merge(fx[["fixture_id", "home_name", "ft_home", "ft_away"]],
                                                        on="fixture_id")
    want = np.where(ch["home_name"] == "Chelsea", ch["ft_away"], ch["ft_home"])
    assert (ch["goals_conceded"].to_numpy() == want).all()
    assert (ch["goals_conceded"] == 2).any()               # the own-goal fixture: 2, scorer rows say 1


def test_one_key_for_training_and_serving(world):
    rows, fx, cup_day = world
    hist = FE.build_features(rows)
    t = FE.team_match_table(hist)
    from player_model import team_identity as TI
    # the same opponent name in a league fixture and in a cup fixture is the same club key
    assert TI.club_key("Chelsea", "Premier League") == TI.club_key("Chelsea", "Champions League") \
        == f"af:{IDS['Chelsea']}"
    assert TI.club_key("Club Brugge KV", "Champions League") == f"af:{IDS['Club Brugge KV']}"
    assert TI.club_key("Nobody FC", "Champions League") is None
    assert set(t["_tkey"]) >= {f"af:{i}" for i in IDS.values()}


def test_unmeasured_fixtures_are_skipped_not_counted():
    # A club's cup tie we hold only the score for must not erase its shots-based state: the state
    # is the mean of its last n MEASURED fixtures (training and serving alike).
    t = pd.DataFrame({"_tkey": ["af:1"] * 5 + ["af:2"] * 2, "fixture_id": range(7),
                      "sot_conceded": [1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0]})
    got = FE._roll_measured(t, "sot_conceded", 2, 2)
    assert got.iloc[:3].isna().all()                         # fewer than 2 measured before
    assert got.iloc[3] == pytest.approx(1.5) and got.iloc[4] == pytest.approx(1.5)
    assert got.iloc[5:].isna().all()                         # per team
    # serving computes the same rolling: build_upcoming_features runs build_features itself
    # (tests/test_props_train_serve_parity_2026_11_09.py proves the serving rows equal training's)


# ── (c) league strength of a cup fixture: measured ─────────────────────────────────────────────

def test_cup_strength_is_the_measured_mean_of_its_clubs(monkeypatch):
    monkeypatch.setattr(elo, "MIN_LEAGUE_CLUBS", 3)
    monkeypatch.setattr(elo, "MIN_CLUB_MATCHES", 2)
    day0 = pd.Timestamp("2025-09-01")
    rat = []
    for k, (key, base) in enumerate((("af:1", 0.3), ("af:2", 0.1), ("af:3", -0.1), ("af:4", 0.5))):
        for i in range(3):
            rat.append({"club_key": key, "date": day0 + pd.Timedelta(days=i), "rating": base,
                        "n_matches": i + 1, "competition": "Premier League"})
        if key != "af:4":                                    # af:4 never played the cup
            rat.append({"club_key": key, "date": day0 + pd.Timedelta(days=10), "rating": base,
                        "n_matches": 4, "competition": "Champions League"})
    S = lookup.Strength(pd.DataFrame(rat), pd.DataFrame(columns=["league", "date", "strength", "rank"]),
                        pd.DataFrame(columns=["scope", "name", "club_key"]))
    got = LQ.cup_strength(S, ["Champions League", "Premier League", "Champions League"],
                          [day0 + pd.Timedelta(days=20), day0 + pd.Timedelta(days=20),
                           day0 + pd.Timedelta(days=10)])
    assert got.iloc[0] == pytest.approx((0.3 + 0.1 - 0.1) / 3)     # members only, ratings before the day
    assert math.isnan(got.iloc[1])                                  # not a cup
    assert math.isnan(got.iloc[2])                                  # no cup match BEFORE that day
    monkeypatch.setattr(elo, "MIN_LEAGUE_CLUBS", 4)
    assert math.isnan(LQ.cup_strength(S, ["Champions League"], [day0 + pd.Timedelta(days=20)]).iloc[0])
    # wired into attach_strength (training) and fixture_strength (serving)
    monkeypatch.setattr(elo, "MIN_LEAGUE_CLUBS", 3)
    df = pd.DataFrame({"player_id": [1], "date": [day0 + pd.Timedelta(days=20)], "league": ["Champions League"],
                       "team": ["A"], "opponent": ["B"], "team_id": [1], "opponent_id": [4]})
    out = LQ.attach_strength(df, strength=S)
    assert out["league_elo"].iloc[0] == pytest.approx(0.1)
    assert out["opp_club_elo"].iloc[0] == pytest.approx(0.5)        # by API-Football id, no name lookup
    fs = LQ.fixture_strength("A", "B", "Champions League", day0 + pd.Timedelta(days=20), strength=S)
    assert fs["league_elo"] == pytest.approx(0.1) and math.isnan(fs["league_elo_rank"])


# ── (a) cup appearances in form / rest days ─────────────────────────────────────────────────────

def _cup_rows_for(rows, cup_day):
    """The collected cup rows (what scripts/collect_cup_player_stats.py writes) of the cup tie."""
    out = []
    for r in _player_rows(9_200_001, cup_day, "Champions League", "Arsenal", "Club Brugge KV", 0, 0):
        r = dict(r, date=str(cup_day.date()), fixture_status="FT")
        out.append(r)
    return pd.DataFrame(out)


def test_serving_rest_days_and_form_include_cup_appearances(world):
    from player_model.cup_history import with_cup_rows
    rows, fx, cup_day = world
    league_only = [r for r in rows if r["league"] != "Champions League"]
    hist = FE.build_features(league_only)
    cups = _cup_rows_for(rows, cup_day)
    served = with_cup_rows(hist, cups)
    assert len(served) == len(hist) + len(cups)
    pid = IDS["Arsenal"] * 100 + 3
    fday = cup_day + pd.Timedelta(days=3)
    up = [{"player_id": pid, "fixture_id": 990002, "team": "Arsenal", "home_team": "Arsenal",
           "away_team": "Chelsea", "is_home": 1, "league": "Premier League", "date": str(fday.date())}]
    before = FE.build_upcoming_features(up, hist).iloc[0]
    after = FE.build_upcoming_features(up, served).iloc[0]
    assert after["rest_days"] == 3 and before["rest_days"] > 3
    assert after["goals_pg"] != before["goals_pg"]                  # the 2-goal cup game counts
    assert with_cup_rows(served, cups) is served                    # nothing twice


def test_merge_cups_appends_only_new_90_minute_rows(world, monkeypatch):
    from player_model import pipeline as P
    rows, fx, cup_day = world
    league_only = [r for r in rows if r["league"] != "Champions League"]
    hist = FE.build_features(league_only)
    cups = _cup_rows_for(rows, cup_day)
    aet = cups.copy()
    aet["fixture_id"], aet["fixture_status"] = 9_200_002, "AET"
    aet["date"] = str((cup_day + pd.Timedelta(days=1)).date())
    both = pd.concat([cups, aet], ignore_index=True)
    combined, st = P.merge_cup_rows(hist, both)
    assert st["extra_time_context_only"] == len(aet)
    added = combined[combined["league"] == "Champions League"]
    assert set(added["fixture_id"]) == {9_200_001} and st["appended"] == len(added) > 0
    # every stored row is unchanged
    old = combined[combined["league"] != "Champions League"].sort_values(["fixture_id", "player_id"])
    ref = hist.sort_values(["fixture_id", "player_id"])
    pd.testing.assert_frame_equal(old[ref.columns].reset_index(drop=True), ref.reset_index(drop=True),
                                  check_dtype=False)
    # the appended row's features were built with the league context: rest days from the league
    f = added[added["position"] == "F"].iloc[0]
    assert f["rest_days"] == 9 and f["n_prev_games"] >= 1 and f["opp_goals_conceded_pg"] == pytest.approx(3.0)


# ── (d) the collector ─────────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code, self.headers = body, status, {"x-ratelimit-requests-remaining": "70000"}

    def json(self):
        return self._b


class _Session:
    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params))
        if url.endswith("/status"):
            return _Resp({"response": {}})
        if "fixture" not in params:               # /fixtures?ids= (referee + corners sidecar)
            return _Resp({"response": [], "errors": []})
        return _Resp(self.bodies.get(params["fixture"], {"response": [], "errors": []}))


def _af_block(team, tid, players):
    return {"team": {"id": tid, "name": team},
            "players": [{"player": {"id": pid, "name": f"P{pid}"},
                         "statistics": [{"games": {"minutes": mins, "position": "F", "substitute": False,
                                                   "rating": "7.1"},
                                         "goals": {"total": g, "assists": None, "conceded": 0, "saves": None},
                                         "shots": {"total": 2, "on": 1}, "cards": {"yellow": 0, "red": 0},
                                         "passes": {"total": 20, "key": 1}, "duels": {}, "fouls": {},
                                         "tackles": {}, "dribbles": {}, "penalty": {}}]}
                        for pid, mins, g in players]}


def test_collector_plan_parse_journal_and_budget(tmp_path, monkeypatch):
    import collect_cup_player_stats as K
    fx = pd.DataFrame([
        {**_fx(1, pd.Timestamp("2025-10-01"), "Champions League", "Arsenal", "Club Brugge KV", 2, 0, 2)},
        {**_fx(2, pd.Timestamp("2025-10-02"), "Europa League", "Gent", "Genk", 1, 1, 3), "status": "AET"},
        {**_fx(3, pd.Timestamp("2022-01-01"), "Champions League", "Gent", "Genk", 0, 0, 2)},       # before history
        {**_fx(4, pd.Timestamp("2025-10-03"), "Premier League", "Arsenal", "Chelsea", 0, 0, 39)},   # not a cup
        {**_fx(5, pd.Timestamp("2025-10-04"), "Conference League", "Gent", "Anderlecht", 0, 0, 848)},
        {**_fx(6, pd.Timestamp("2025-10-05"), "Champions League", "Genk", "Gent", 0, 0, 2), "status": "PST"},
    ])
    cups = {2: "Champions League", 3: "Europa League", 848: "Conference League"}
    p = K.plan(fx, cups, pd.Timestamp("2022-08-12"), pd.DataFrame(), ("Premier League",))
    # cup fixtures only, finished only, inside the window; a fixture with a club of a history
    # league first, then newest first
    assert p["fixture_id"].tolist() == [1, 5, 2]
    journal = pd.DataFrame([{"fixture_id": 5}])
    assert K.plan(fx, cups, pd.Timestamp("2022-08-12"), journal, ())["fixture_id"].tolist() == [2, 1]

    body = {"response": [_af_block("Arsenal", 42, [(7, 90, 1), (8, 5, 0)]),
                         _af_block("Club Brugge KV", 569, [(9, 90, 0)]),
                         _af_block("Somebody", 1, [(10, 90, 0)])], "errors": []}
    rows = K.parse(body, p.iloc[0])
    by = {r["player_id"]: r for r in rows}
    assert set(by) == {7, 9}                                          # < 10 minutes and the stray block: out
    assert by[7]["league"] == "Champions League" and by[7]["is_home"] == 1
    assert (by[7]["team_id"], by[7]["opponent_id"]) == (42, 569) and by[9]["team_id"] == 569
    assert by[7]["fixture_status"] == "FT" and by[7]["team_corners"] is None
    # assists: null for everyone in an answered fixture -> 0 for a player who played (owner's refined
    # null rule 2026-11-04, fix round 9); the rating is not a count and is kept as sent
    assert by[7]["goals"] == 1 and by[7]["assists"] == 0 and by[7]["count_source"] == "provider_rule"
    assert K.parse({"response": [], "errors": []}, p.iloc[0]) == []           # not answered: no row, no 0

    # run: one call per fixture, rows + journal written, an empty answer journaled too
    out, jrn = tmp_path / "rows.parquet", tmp_path / "journal.parquet"
    fx.to_parquet(tmp_path / "fx.parquet", index=False)
    monkeypatch.setattr(K, "FIXTURES", tmp_path / "fx.parquet")
    monkeypatch.setattr(K, "OUT", out)
    monkeypatch.setattr(K, "JOURNAL", jrn)
    monkeypatch.setattr(K, "META", tmp_path / "meta.parquet")
    monkeypatch.setattr(K, "cup_ids", lambda: cups)
    monkeypatch.setattr(K, "_start_and_leagues", lambda: (pd.Timestamp("2022-08-12"), ("Premier League",)))
    s = _Session({1: body})
    c = CC.AFClient(key="k", session=s, budget_file=tmp_path / "b.json", pause_s=0)
    st = K.run(client=c)
    assert st["fetched"] == 3 and st["rows"] == 2
    assert [p_["fixture"] for u, p_ in s.calls if "fixtures/players" in u] == [1, 5, 2]
    assert len(pd.read_parquet(out)) == 2 and sorted(pd.read_parquet(jrn)["fixture_id"]) == [1, 2, 5]
    # resumable: a second run asks nothing
    s2 = _Session({})
    assert K.run(client=CC.AFClient(key="k", session=s2, budget_file=tmp_path / "b.json", pause_s=0))["todo"] == 0
    assert not [u for u, _ in s2.calls if "fixtures/players" in u]
    # budget: the cap stops before the call, cleanly, nothing recorded for the unfetched fixture
    jrn.unlink()
    (tmp_path / "b.json").write_text(json.dumps({CC.utc_day(): 10}))
    st = K.run(client=CC.AFClient(key="k", session=_Session({1: body}), budget_file=tmp_path / "b.json",
                                  daily_cap=11, pause_s=0))
    assert st["fetched"] == 1 and "budget_stop" in st and len(pd.read_parquet(jrn)) == 1


def test_workflow_runs_the_cup_step_with_the_budgeted_collector():
    text = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    assert "python scripts/collect_cup_player_stats.py" in text and "contains(env.STEPS, 'cupstats')" in text
    assert text.index("1. Cross-league competition fixtures") < text.index("8. UEFA cup player stats")
    for line in text.splitlines():
        if "default:" in line and "fixtures" in line or line.strip().startswith("STEPS:"):
            assert "cupstats" in line
