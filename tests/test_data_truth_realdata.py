"""Data Truth real-data round (dt2/realdata, owner decisions A-D of 2026-11-02): measured club /
league strength, collected profiles / penalties / injuries, Understat set pieces, corner matchup,
and the budgeted collectors. Everything offline (fake HTTP sessions, synthetic frames).

    python -m pytest tests/test_data_truth_realdata.py -q
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
from player_model.setpiece import features as SPF, mapping as SPM, understat as SPU  # noqa: E402
from src.strength import build as SB, elo, lookup, results  # noqa: E402


# ── fake HTTP ────────────────────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, body, remaining=None, status=200):
        self._b, self.status_code = body, status
        self.headers = {} if remaining is None else {"x-ratelimit-requests-remaining": str(remaining)}
        self.text = json.dumps(body)

    def json(self):
        return self._b


class _Session:
    def __init__(self, remaining_seq, status_body=None):
        self.rem = list(remaining_seq)
        self.calls = []
        self.status_body = status_body

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params))
        if url.endswith("/status"):
            return _Resp(self.status_body or {"response": {}})
        return _Resp({"response": [], "errors": []}, self.rem.pop(0) if self.rem else None)


def test_budget_floor_stops_before_the_call(tmp_path):
    s = _Session([30_000, 24_999, 50_000])
    c = CC.AFClient(key="k", session=s, budget_file=tmp_path / "b.json", pause_s=0)
    assert c.get("players", {"id": 1}) is not None
    assert c.get("players", {"id": 2}) is not None          # header now says 24,999 remaining
    with pytest.raises(CC.BudgetStop):
        c.get("players", {"id": 3})
    assert len([u for u, _ in s.calls if not u.endswith("/status")]) == 2


def test_budget_daily_cap_and_journal(tmp_path):
    bf = tmp_path / "b.json"
    bf.write_text(json.dumps({CC.utc_day(): 9}))
    c = CC.AFClient(key="k", session=_Session([70_000] * 5), budget_file=bf, daily_cap=10, pause_s=0)
    c.get("sidelined", {"player": 1})
    with pytest.raises(CC.BudgetStop):
        c.get("sidelined", {"player": 2})
    c.flush()
    assert json.loads(bf.read_text())[CC.utc_day()] == 10


def test_budget_status_endpoint_is_read_first(tmp_path):
    s = _Session([], status_body={"response": {"requests": {"current": 60_000, "limit_day": 75_000}}})
    c = CC.AFClient(key="k", session=s, budget_file=tmp_path / "b.json", pause_s=0)
    with pytest.raises(CC.BudgetStop):                      # 15,000 left < 25,000 floor
        c.get("players", {"id": 1})
    assert [u for u, _ in s.calls] == [f"{CC.BASE_URL}/status"]


def test_budget_stop_is_not_swallowed_by_a_broad_except():
    assert not issubclass(CC.BudgetStop, Exception)
    with pytest.raises(CC.BudgetStop):
        try:
            raise CC.BudgetStop("x")
        except Exception:                                    # what backfill_af_history._get does
            pass


def test_guarded_requests_counts_and_stops(tmp_path):
    import collect_af_history_gap as G
    c = CC.AFClient(key="k", session=_Session([26_000, 24_000]), budget_file=tmp_path / "b.json", pause_s=0)
    g = G.GuardedRequests(c)
    g.get(f"{CC.BASE_URL}/fixtures", params={"league": 1})
    g.get(f"{CC.BASE_URL}/fixtures", params={"league": 2})
    assert c.calls_today() == 2 and c.remaining == 24_000
    with pytest.raises(CC.BudgetStop):
        g.get(f"{CC.BASE_URL}/fixtures", params={"league": 3})


# ── collectors' parsers ─────────────────────────────────────────────────────────────────────────

def test_profile_parse_real_values_only():
    import collect_player_profiles as P
    body = {"response": [{"player": {"height": "187 cm", "weight": None, "birth": {"date": "1999-04-02"}},
                          "statistics": [{"penalty": {"scored": 2, "missed": 1}},
                                         {"penalty": {"scored": 1, "missed": None}},   # not both -> skipped
                                         {"penalty": {"scored": 0, "missed": 0}}]}]}
    r = P.parse(body, 7, "2024")
    assert r["status"] == "ok" and r["height_cm"] == 187.0 and math.isnan(r["weight_kg"])
    assert r["birth_date"] == "1999-04-02"
    assert (r["penalty_scored"], r["penalty_missed"], r["pen_entries"]) == (2.0, 1.0, 2)
    e = P.parse({"response": []}, 7, "2023")
    assert e["status"] == "empty" and math.isnan(e["height_cm"]) and math.isnan(e["penalty_scored"])
    none = P.parse({"response": [{"player": {}, "statistics": [{"penalty": {"scored": None, "missed": None}}]}]}, 8, "2024")
    assert none["pen_entries"] == 0 and math.isnan(none["penalty_scored"])


def test_sidelined_parse_keeps_an_empty_answer():
    import collect_sidelined as S
    p, spells = S.parse({"response": []}, 5)
    assert p["n_spells"] == 0 and spells == []
    p, spells = S.parse({"response": [{"type": "Knee", "start": "2024-01-02", "end": None},
                                      {"type": "x", "start": None, "end": "2024-02-01"}]}, 5)
    assert p["n_spells"] == 1 and spells[0]["end"] is None


def test_competition_fixture_parse_and_season_plan():
    import collect_competition_fixtures as F
    comp = {"league_id": 13, "league": None, "competition_type": "club_cup", "league_name": "CONMEBOL Libertadores"}
    body = {"response": [
        {"fixture": {"id": 1, "date": "2024-04-03T00:30:00+00:00", "status": {"short": "AET"}},
         "teams": {"home": {"id": 10, "name": "A"}, "away": {"id": 11, "name": "B"}},
         "score": {"fulltime": {"home": 1, "away": 1}}},
        {"fixture": {"id": 2, "date": "2024-04-04", "status": {"short": "NS"}},
         "teams": {"home": {"id": 10}, "away": {"id": 12}}, "score": {"fulltime": {}}}]}
    rows = F.parse(body, comp, "2024")
    assert len(rows) == 1 and (rows[0]["ft_home"], rows[0]["ft_away"]) == (1, 1)  # 90-minute score
    seasons = [{"season": "2023", "start": pd.Timestamp("2023-02-01"), "end": pd.Timestamp("2023-11-30")},
               {"season": "2024", "start": pd.Timestamp("2024-02-01"), "end": pd.Timestamp("2024-11-30")},
               {"season": "2018", "start": pd.Timestamp("2018-02-01"), "end": pd.Timestamp("2018-11-30")}]
    j = pd.DataFrame([{"league_id": 13, "season": "2023", "season_end": "2023-11-30",
                       "fetched_at": "2024-01-05T00:00:00+00:00"},
                      {"league_id": 13, "season": "2024", "season_end": "2024-11-30",
                       "fetched_at": "2024-06-05T00:00:00+00:00"}])
    plan = F.plan_seasons(seasons, j, 13, pd.Timestamp("2019-07-01"), pd.Timestamp("2024-07-01"))
    assert [s["season"] for s in plan] == ["2024"]         # ended+journaled skipped, pre-span skipped


def test_cup_verification_refuses_a_non_world_cup(tmp_path):
    import collect_competition_fixtures as F

    class S(_Session):
        def get(self, url, headers=None, params=None, timeout=None):
            if url.endswith("/status"):
                return _Resp({"response": {}})
            return _Resp({"response": [{"league": {"type": "League", "name": "X"},
                                        "country": {"name": "Brazil"}, "seasons": []}], "errors": []}, 60_000)
    c = CC.AFClient(key="k", session=S([]), budget_file=tmp_path / "b.json", pause_s=0)
    assert F.provider_seasons(c, {"league_id": 13, "verify": True}) == []


# ── measured strength ───────────────────────────────────────────────────────────────────────────

def _matches(n_rounds=30, link=0):
    """Two countries' leagues (4 clubs each, round robin) plus `link` cross-country cup matches in
    which country A's clubs always win."""
    rows = []
    d = pd.Timestamp("2023-01-07")
    a = [f"af:{i}" for i in range(1, 5)]
    b = [f"af:{i}" for i in range(11, 15)]
    for r in range(n_rounds):
        for clubs, comp, sc in ((a, "Premier League", "club_league:England"),
                                (b, "Serie A", "club_league:Italy")):
            pairs = [(clubs[0], clubs[1]), (clubs[2], clubs[3])] if r % 3 == 0 else \
                    [(clubs[0], clubs[2]), (clubs[1], clubs[3])] if r % 3 == 1 else \
                    [(clubs[0], clubs[3]), (clubs[1], clubs[2])]
            for h, w in pairs:
                rows.append({"date": d, "competition": comp, "season": "2023", "source": "collected",
                             "home_key": h, "away_key": w, "home_goals": 2 if h in (a[0], b[0]) else 1,
                             "away_goals": 1, "home_scope": sc, "away_scope": sc})
        d += pd.Timedelta(days=7)
    for k in range(link):                                  # mid-week, during the league season
        rows.append({"date": pd.Timestamp("2023-01-10") + pd.Timedelta(days=7 * k),
                     "competition": "Champions League", "season": "2023",
                     "source": "collected", "home_key": a[k % 4], "away_key": b[k % 4], "home_goals": 2,
                     "away_goals": 0, "home_scope": "club_league:England", "away_scope": "club_league:Italy"})
    m = pd.DataFrame(rows)
    m["date"] = pd.to_datetime(m["date"]).astype("datetime64[ns]")
    return m.sort_values("date", kind="stable").reset_index(drop=True)


def test_elo_is_fitted_not_typed_and_in_range():
    f = elo.fit(_matches(), verbose=False)
    assert elo.K_RANGE[0] <= f["k"] <= elo.K_RANGE[1] and elo.H_RANGE[0] <= f["h"] <= elo.H_RANGE[1]
    assert f["n_scored"] > 0 and len(f["grid"]) > elo.COARSE_GRID ** 2 - 1


def test_club_rating_reads_only_earlier_days_and_needs_a_minimum_sample():
    m = _matches()
    t = elo.club_rating_table(m, 0.2, 0.1)
    S = lookup.Strength(t, pd.DataFrame(columns=["league", "date", "strength", "rank"]),
                        pd.DataFrame(columns=["scope", "name", "club_key"]))
    day = m["date"].iloc[-1]
    got = S.club_elo_asof(["af:1", "af:1"], [day, day + pd.Timedelta(days=1)])
    before = t[(t["club_key"] == "af:1") & (t["date"] < day)].iloc[-1]["rating"]
    after = t[(t["club_key"] == "af:1") & (t["date"] <= day)].iloc[-1]["rating"]
    assert got["club_elo"].iloc[0] == pytest.approx(before)        # the match ON day D is not seen
    assert got["club_elo"].iloc[1] == pytest.approx(after)
    early = S.club_elo_asof(["af:1"], [m["date"].iloc[0] + pd.Timedelta(days=14)])
    assert math.isnan(early["club_elo"].iloc[0])                   # < MIN_CLUB_MATCHES -> NaN
    assert t[t["club_key"] == "af:1"]["rating"].iloc[-1] > t[t["club_key"] == "af:4"]["rating"].iloc[-1]


def test_league_strength_needs_a_real_cross_league_link(monkeypatch):
    monkeypatch.setattr(elo, "MIN_LEAGUE_CLUBS", 4)
    m = _matches(link=0)
    ls = SB.league_strength_table(m, elo.club_rating_table(m, 0.2, 0.1))
    assert ls["strength"].isna().all() and not ls["linked"].any()
    m = _matches(link=elo.MIN_LINK_MATCHES + 4)
    ls = SB.league_strength_table(m, elo.club_rating_table(m, 0.2, 0.1))
    last = ls.dropna(subset=["strength"]).groupby("league").last()
    assert set(last.index) == {"Premier League", "Serie A"}
    assert last.loc["Premier League", "strength"] > last.loc["Serie A", "strength"]
    assert last.loc["Premier League", "rank"] == 1


def test_continental_cup_rows_get_a_cup_key(tmp_path):
    c = pd.DataFrame([{"fixture_id": 1, "league_id": 13, "league": None, "season": "2024", "date": "2024-04-03",
                       "home_id": 10, "home_name": "A", "away_id": 11, "away_name": "B",
                       "ft_home": 1, "ft_away": 0, "status": "FT"}])
    p = tmp_path / "cf.parquet"
    c.to_parquet(p, index=False)
    m, _, rep = results.load_results(collected=p, af=tmp_path / "none.parquet", fd=tmp_path / "none.parquet",
                                     verbose=False)
    assert m["competition"].tolist() == ["af_cup:13"] and results.competition_type("af_cup:13") == "club_cup"


def test_attach_strength_career_average_uses_earlier_rows_only():
    m = _matches()
    t = elo.club_rating_table(m, 0.2, 0.1)
    names = pd.DataFrame({"scope": "club_league:England", "name": ["Arsenal", "Chelsea", "Everton", "Fulham"],
                          "club_key": ["af:1", "af:2", "af:3", "af:4"]})
    S = lookup.Strength(t, pd.DataFrame(columns=["league", "date", "strength", "rank"]), names)
    dates = sorted(m.loc[m["competition"] == "Premier League", "date"].unique())[12:22]
    df = pd.DataFrame({"player_id": 1, "date": dates, "league": "Premier League", "team": "Arsenal",
                       "opponent": ["Chelsea", "Everton", "Fulham"] * 3 + ["Chelsea"]})
    out = LQ.attach_strength(df, strength=S)
    assert out["opp_club_elo"].notna().all()
    k = LQ.MIN_CAREER_ROWS
    assert out["player_career_avg_opp_elo"].iloc[:k].isna().all()
    assert out["player_career_avg_opp_elo"].iloc[k] == pytest.approx(out["opp_club_elo"].iloc[:k].mean())
    assert out["league_elo"].isna().all()                          # no linked league in this table
    assert math.isnan(LQ.league_tier("Premier League"))


# ── Understat set pieces ────────────────────────────────────────────────────────────────────────

def _hexed(obj) -> str:
    return "".join(f"\\x{b:02x}" for b in json.dumps(obj).encode("utf-8"))


def test_understat_parsers_json_and_html():
    dates = [{"id": "11", "isResult": True, "h": {"id": "1", "title": "Arsenal"}, "a": {"id": "2", "title": "Chelsea"},
              "goals": {"h": "2", "a": "1"}, "datetime": "2024-08-17 14:00:00"}]
    assert SPU.parse_league(json.dumps({"dates": dates}), "EPL", "2024")[0]["match_id"] == 11
    html = f"<script>var datesData = JSON.parse('{_hexed(dates)}');</script>"
    assert SPU.parse_league(html, "EPL", "2024")[0]["h_title"] == "Arsenal"
    shots = {"h": [{"id": "5", "player": "Gabriel Magalhães", "player_id": "9", "situation": "FromCorner",
                    "result": "Goal", "xG": "0.3", "player_assisted": "Bukayo Saka"}], "a": []}
    rost = {"h": {"1": {"player_id": "9", "player": "Gabriel Magalhães", "time": "90", "team_id": "1"}}, "a": {}}
    html = (f"var shotsData = JSON.parse('{_hexed(shots)}'); var rostersData = JSON.parse('{_hexed(rost)}');")
    s, r = SPU.parse_match(html, 11)
    assert s[0]["player"] == "Gabriel Magalhães" and s[0]["situation"] == "FromCorner" and r[0]["time"] == 90.0
    assert SPU.parse_match("<html></html>", 11) is None


def test_player_name_compatibility():
    assert SPM.compatible("M. Salah", "Mohamed Salah")
    assert SPM.compatible("Lars-Jørgen Salvesen", "Lars-Jorgen Salvesen")
    assert not SPM.compatible("M. Salah", "Mohamed Elneny")
    assert not SPM.compatible("A. B.", "Anything Bob")                  # no full token


def _us_world():
    hist = []
    for fid, day, h, a in ((1, "2024-08-17", "Arsenal", "Chelsea"), (2, "2024-08-24", "Chelsea", "Arsenal")):
        for team, ih in ((h, 1), (a, 0)):
            for pid, nm in (({"Arsenal": 100, "Chelsea": 200}[team], "B. Saka" if team == "Arsenal" else "C. Palmer"),
                            ({"Arsenal": 101, "Chelsea": 201}[team], "Gabriel" if team == "Arsenal" else "L. Colwill")):
                hist.append({"fixture_id": fid, "date": pd.Timestamp(day), "league": "Premier League",
                             "home_team": h, "away_team": a, "team": team, "is_home": ih,
                             "player_id": pid, "player_name": nm})
    hist = pd.DataFrame(hist)
    matches = pd.DataFrame([
        {"match_id": 11, "datetime": "2024-08-16 19:00:00", "h_title": "Arsenal", "a_title": "Chelsea", "is_result": True},
        {"match_id": 12, "datetime": "2024-08-24 14:00:00", "h_title": "Chelsea", "a_title": "Arsenal", "is_result": True},
        {"match_id": 13, "datetime": "2024-09-30 14:00:00", "h_title": "Chelsea", "a_title": "Arsenal", "is_result": True}])
    rosters = pd.DataFrame([
        {"match_id": mid, "h_a": side, "us_player_id": uid, "player": nm, "time": 90.0}
        for mid, ars in ((11, "h"), (12, "a"))
        for side, uid, nm in ((ars, 1, "Bukayo Saka"), (ars, 2, "Gabriel Magalhães"),
                              ("a" if ars == "h" else "h", 3, "Cole Palmer"),
                              ("a" if ars == "h" else "h", 4, "Levi Colwill"))])
    return hist, matches, rosters


def test_understat_mapping_by_date_clubs_and_names():
    hist, matches, rosters = _us_world()
    mm, rep = SPM.map_matches(matches, hist)
    assert dict(zip(mm["match_id"], mm["fixture_id"])) == {11: 1, 12: 2}   # 1-day timezone shift OK
    assert rep["no_window"] == 1
    pm, prep = SPM.map_players(rosters, mm, hist)
    got = dict(zip(pm["us_player_id"], pm["player_id"]))
    # 'Gabriel' (one full token) is compatible with 'Gabriel Magalhães'; all four map, 2 votes each
    assert got == {1: 100, 2: 101, 3: 200, 4: 201} and (pm["n_votes"] == 2).all()


def test_understat_ambiguity_is_refused():
    hist, matches, rosters = _us_world()
    hist.loc[hist["player_id"] == 201, "player_name"] = "C. Palmer"       # two Chelsea 'C. Palmer'
    mm, _ = SPM.map_matches(matches, hist)
    pm, rep = SPM.map_players(rosters, mm, hist)
    assert 3 not in set(pm["us_player_id"]) and rep["ambiguous"] >= 2


def test_setpiece_features_min_sample_and_no_same_day_leak(monkeypatch):
    monkeypatch.setattr(SPF, "MIN_TEAM_SP_EVENTS", 2)
    monkeypatch.setattr(SPF, "MIN_US_MINUTES", 90)
    hist, matches, rosters = _us_world()
    mm, _ = SPM.map_matches(matches, hist)
    pm, _ = SPM.map_players(rosters, mm, hist)
    shots = pd.DataFrame([
        {"shot_id": 1, "match_id": 11, "h_a": "h", "us_player_id": 2, "situation": "FromCorner", "result": "Goal",
         "xG": 0.4, "player_assisted": "Bukayo Saka"},
        {"shot_id": 2, "match_id": 11, "h_a": "h", "us_player_id": 2, "situation": "SetPiece", "result": "MissedShot",
         "xG": 0.1, "player_assisted": "Bukayo Saka"},
        {"shot_id": 3, "match_id": 11, "h_a": "h", "us_player_id": 1, "situation": "OpenPlay", "result": "Goal",
         "xG": 0.5, "player_assisted": None}])
    app = SPF.appearances(shots, rosters, mm, pm, hist[["fixture_id", "date"]])
    q = SPF.asof(app, [101, 101, 100], ["2024-08-17", "2024-08-18", "2024-08-18"])
    assert q.iloc[0].isna().all()                                   # its own match day: not seen
    assert q.loc[1, "us_sp_receiver_share"] == 1.0 and q.loc[1, "us_sp_xg_p90"] == pytest.approx(0.5)
    assert q.loc[1, "us_sp_sot_p90"] == pytest.approx(1.0)
    assert q.loc[2, "us_sp_taker_share"] == 1.0 and q.loc[2, "us_sp_receiver_share"] == 0.0
    monkeypatch.setattr(SPF, "MIN_TEAM_SP_EVENTS", 3)
    assert SPF.asof(app, [100], ["2024-08-18"])["us_sp_taker_share"].isna().all()   # below sample


# ── feature wiring ──────────────────────────────────────────────────────────────────────────────

def _rows(n=8):
    rows = []
    d0 = pd.Timestamp("2025-08-01")
    for r in range(n):
        for team, ih, opp in (("Arsenal", 1, "Chelsea"), ("Chelsea", 0, "Arsenal")):
            for pid, pos in (((1 if team == "Arsenal" else 2), "F"), ((3 if team == "Arsenal" else 4), "D")):
                rows.append({"fixture_id": 100 + r, "player_id": pid, "player_name": f"P{pid}",
                             "date": d0 + pd.Timedelta(days=7 * r), "league": "Premier League",
                             "home_team": "Arsenal", "away_team": "Chelsea", "team": team, "is_home": ih,
                             "season": "2025", "position": pos, "minutes": 90, "started": 1, "goals": 0,
                             "assists": 0, "shots_total": 1, "shots_on_target": 1, "yellow_cards": 0,
                             "key_passes": 0, "rating": 7.0, "fouls_committed": 0, "fouls_drawn": 0,
                             "duels_total": 2, "duels_won": 1,
                             "team_corners": 0 if r == 2 else (5 if team == "Arsenal" else 3),
                             "red_cards": 0, "passes_total": 10, "offsides": 0, "tackles_total": 0,
                             "interceptions": 0, "dribbles_attempted": 0, "dribbles_success": 0,
                             "dribbles_past": 0, "goals_conceded": 0, "saves": 0,
                             "penalty_scored": 1 if (pid == 1 and r == 5) else 0, "penalty_missed": 0,
                             "penalty_won": 0, "height_cm": 180.0, "weight_kg": 75.0, "age": 25.0})
    return rows


@pytest.fixture()
def collect_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(FE, "COLLECT_DIR", tmp_path)
    monkeypatch.setattr(SPF, "DIR", tmp_path)
    monkeypatch.setattr(lookup, "DIR", tmp_path / "strength")      # no strength tables: NaN
    FE._COLLECTED.clear()
    yield tmp_path
    FE._COLLECTED.clear()


def test_unrecorded_corners_are_missing_not_zero():
    df = FE._blank_unrecorded_corners(pd.DataFrame(_rows()))
    assert df.loc[df["fixture_id"] == 102, "team_corners"].isna().all()
    assert (df.loc[df["fixture_id"] == 103, "team_corners"] > 0).all()


def test_corner_matchup_profiles_penalties_injuries(collect_dir, monkeypatch):
    monkeypatch.setattr(C, "MIN_LEAGUE_MATCHES", 2)
    pd.DataFrame([{"player_id": 1, "season": "2024", "status": "ok", "height_cm": 191.0, "weight_kg": 84.0,
                   "birth_date": "2000-08-01", "penalty_scored": 3.0, "penalty_missed": 1.0, "pen_entries": 2},
                  {"player_id": 2, "season": "2025", "status": "ok", "height_cm": np.nan, "weight_kg": np.nan,
                   "birth_date": None, "penalty_scored": np.nan, "penalty_missed": np.nan, "pen_entries": 0}]
                 ).to_parquet(collect_dir / "player_profiles.parquet", index=False)
    pd.DataFrame([{"player_id": 3, "n_spells": 1, "fetched_at": "x"}, {"player_id": 4, "n_spells": 0, "fetched_at": "x"}]
                 ).to_parquet(collect_dir / "sidelined_players.parquet", index=False)
    pd.DataFrame([{"player_id": 3, "type": "Knee", "start": "2025-08-10", "end": "2025-08-20"}]
                 ).to_parquet(collect_dir / "sidelined_spells.parquet", index=False)
    b = FE.build_features(_rows())
    p1 = b[b["player_id"] == 1].sort_values("date").reset_index(drop=True)
    assert (p1["height_cm"] == 191.0).all() and p1["age"].iloc[0] == pytest.approx(
        (p1["date"].iloc[0] - pd.Timestamp("2000-08-01")).days / 365.25)
    # prior season record 3/4 attempts (>= MIN_PEN_ATTEMPTS) from the collected profile; the
    # in-season penalty of round 5 joins from round 6 on
    assert p1["penalty_conversion_rate"].iloc[0] == pytest.approx(3 / 4)
    assert p1["penalty_conversion_rate"].iloc[-1] == pytest.approx(4 / 5)
    p2 = b[b["player_id"] == 2]
    assert p2["height_cm"].isna().all() and p2["penalty_conversion_rate"].isna().all()
    p3 = b[b["player_id"] == 3].set_index("date")
    assert p3.loc["2025-08-22", "days_since_last_injury"] == 2.0
    assert p3.loc["2025-08-22", "return_from_injury_flag"] == 1.0
    p4 = b[b["player_id"] == 4]
    assert (p4["chronic_injury_risk"] == 0).all() and p4["days_since_last_injury"].isna().all()
    # corners: the 0-0 fixture is blank; the matchup is measured once the opponent has a sample
    late = b[(b["player_id"] == 1) & (b["date"] >= "2025-09-12")]
    assert late["corner_matchup_index"].notna().all()
    for c in (*LQ.LEGACY_COLUMNS, *FE.LEGACY_SETPIECE_COLUMNS, *SPF.SETPIECE_COLUMNS, *LQ.MEASURED_COLUMNS):
        assert b[c].isna().all(), c
    for c in ("context_elo_odds", "elo_mismatch_goals", "elo_mismatch_sot", "corner_matchup_index",
              "opp_corners_conceded_pg", *LQ.MEASURED_COLUMNS, *SPF.SETPIECE_COLUMNS):
        assert c in b.columns


def test_serving_reads_the_collected_data(collect_dir):
    pd.DataFrame([{"player_id": 1, "season": "2024", "status": "ok", "height_cm": 191.0, "weight_kg": 84.0,
                   "birth_date": "2000-08-01", "penalty_scored": 3.0, "penalty_missed": 1.0, "pen_entries": 2}]
                 ).to_parquet(collect_dir / "player_profiles.parquet", index=False)
    pd.DataFrame([{"player_id": 1, "n_spells": 0, "fetched_at": "x"}]).to_parquet(
        collect_dir / "sidelined_players.parquet", index=False)
    pd.DataFrame(columns=["player_id", "type", "start", "end"]).to_parquet(
        collect_dir / "sidelined_spells.parquet", index=False)
    FE._SERVE_CACHE.clear()
    hist = FE.build_features(_rows())
    up = [{"player_id": 1, "team": "Arsenal", "opponent": "Chelsea", "is_home": 1, "position": "F",
           "fixture_league": "Premier League", "fixture_date": "2025-10-01"}]
    out = FE.build_upcoming_features(up, hist, {}, {}, FE._build_history_index(hist))
    r = out.iloc[0]
    assert r["height_cm"] == 191.0 and r["age"] == pytest.approx(
        (pd.Timestamp("2025-10-01") - pd.Timestamp("2000-08-01")).days / 365.25)
    assert r["penalty_conversion_rate"] == pytest.approx(4 / 5)
    assert r["chronic_injury_risk"] == 0 and math.isnan(r["days_since_last_injury"])
    for c in (*LQ.LEGACY_COLUMNS, *SPF.SETPIECE_COLUMNS):
        assert math.isnan(r[c]), c
    assert not [c for c in C.PLAYER_FEATURE_COLS if c not in out.columns]
