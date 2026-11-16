"""fix/history-supplement (owner 2026-11-10): provider fixtures the served history lacks are added BESIDE
the banks (src/history_supplement.py, scripts/build_history_supplement.py); China Super League
2022-12-23's two cancelled fixtures are excluded as not played (scripts/build_history_corrections.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_history_corrections as BHC  # noqa: E402
import build_history_supplement as BHS  # noqa: E402
from src import history_supplement as HS  # noqa: E402
from src.integrity import fixture_claims as FC  # noqa: E402

LG = "Austrian Bundesliga"


def _sup(rows):
    return pd.DataFrame([{"date": pd.Timestamp(d), "league": LG, "season": "2025/26", "home_team": h,
                          "away_team": a, "home_goals": hg, "away_goals": ag, "ht_home_goals": None,
                          "ht_away_goals": None, "provider_fixture_id": fid}
                         for d, h, a, hg, ag, fid in rows])


def test_add_supplement_adds_only_keys_the_frame_lacks():
    df = pd.DataFrame({"date": [pd.Timestamp("2026-05-17")], "league": [LG], "home_team": ["Sturm Graz"],
                       "away_team": ["SK Rapid"], "home_goals": [2.0], "away_goals": [0.0],
                       "row_source": ["fd_cache"]})
    sup = _sup([("2026-05-17", "Sturm Graz", "SK Rapid", 2.0, 0.0, 1531259),
                ("2026-05-19", "Ried", "Wolfsberger AC", 2.0, 1.0, 1545407)])
    out = HS.add_supplement(df, sup)
    assert len(out) == 2
    new = out.iloc[1]
    assert (new["home_team"], new["row_source"]) == ("Ried", HS.ROW_SOURCE)
    assert "provider_fixture_id" not in out.columns
    assert out.attrs["history_supplement"] == {"rows": 1, "per_league": {LG: 1}}


def test_supplement_can_be_switched_off(monkeypatch, tmp_path):
    p = tmp_path / "s.csv"
    _sup([("2026-05-19", "Ried", "Wolfsberger AC", 2.0, 1.0, 1545407)]).to_csv(p, index=False)
    assert len(HS.read_supplement(p)) == 1
    monkeypatch.setenv("WOWZA_HISTORY_SUPPLEMENT", "0")
    assert HS.read_supplement(p).empty


def test_a_row_without_a_score_is_never_read(tmp_path):
    p = tmp_path / "s.csv"
    _sup([("2026-05-19", "Ried", "Wolfsberger AC", None, None, 1545407)]).to_csv(p, index=False)
    assert HS.read_supplement(p).empty


def test_supplement_rows_rank_like_api_football_in_the_one_match_rule():
    from src.integrity.duplicate_matches import SOURCE_PRIORITY
    assert SOURCE_PRIORITY[HS.ROW_SOURCE] == SOURCE_PRIORITY["api_football"]


# ── the builder ─────────────────────────────────────────────────────────────────────────────────
def _pool(rows):
    recs = [dict(zip(["fixture_id", "league", "season", "date", "home_id", "home_name", "away_id", "away_name",
                      "ft_home", "ft_away", "status"], r)) for r in rows]
    return FC.provider_pool(ht=pd.DataFrame(recs), comp=pd.DataFrame())


def _served(rows):
    return pd.DataFrame([{"date": pd.Timestamp(d), "league": LG, "season": s, "home_team": h, "away_team": a,
                          "home_goals": hg, "away_goals": ag} for d, s, h, a, hg, ag in rows])


POOL = _pool([
    (1, LG, "2025", "2025-08-01", 10, "Red Bull Salzburg", 20, "Rapid Vienna", 1, 0, "FT"),
    (2, LG, "2025", "2025-08-08", 20, "Rapid Vienna", 30, "Ried", 2, 2, "FT"),
    (3, LG, "2025", "2026-05-19", 30, "Ried", 40, "Wolfsberger AC", 2, 1, "FT"),       # play-off: missing
    (4, LG, "2025", "2025-07-20", 10, "Red Bull Salzburg", 30, "Ried", 1, 1, "FT"),    # before the window
    (5, LG, "2025", "2026-05-22", 30, "Ried", 20, "Rapid Vienna", 3, 0, "FT"),         # awarded (no HT)
    (6, LG, "2025", "2026-05-25", 20, "Rapid Vienna", 50, "Brand New Club", 1, 1, "PEN"),
    (7, LG, "2026", "2026-08-01", 10, "Red Bull Salzburg", 20, "Rapid Vienna", 0, 0, "FT"),  # current season
    (8, LG, "2026", "2026-08-08", 40, "Wolfsberger AC", 10, "Red Bull Salzburg", 0, 1, "FT"),
])
HT = pd.DataFrame({"fixture_id": [3, 5, 6], "ht_home": [0.0, None, 1.0], "ht_away": [0.0, None, 0.0]})
SERVED = _served([("2025-08-01", "2025/26", "Salzburg", "SK Rapid", 1.0, 0.0),
                  ("2025-08-08", "2025/26", "SK Rapid", "Ried", 2.0, 2.0),
                  ("2026-08-08", "2026/27", "Wolfsberger AC", "Salzburg", 0.0, 1.0)])


@pytest.fixture(autouse=True)
def _spellings(monkeypatch):
    # 'Salzburg' / 'SK Rapid' are the served spellings of provider teams 10 / 20 (as the real history)
    from src import team_names
    real = team_names.resolve_result
    table = {"Salzburg": "Red Bull Salzburg", "SK Rapid": "Rapid Vienna"}
    monkeypatch.setattr(team_names, "resolve_result",
                        lambda n, c: table[n] if n in table and table[n] in c else real(n, c))


def test_builder_adds_the_missing_finished_season_fixture_in_served_spellings():
    add, ref, per = BHS.build(SERVED, POOL, HT, corrected=set())
    got = {int(f): r for f, r in zip(add["provider_fixture_id"], add.to_dict("records"))}
    assert set(got) == {3, 6}
    r = got[3]
    assert (r["date"], r["home_team"], r["away_team"], r["home_goals"], r["away_goals"]) == \
        ("2026-05-19", "Ried", "Wolfsberger AC", 2.0, 1.0)
    assert (r["ht_home_goals"], r["ht_away_goals"], r["season"]) == (0.0, 0.0, "2025/26")
    assert r["gap"] == "after_last_served"
    # a club the served history never had keeps the provider's name; the other side is served spelling
    assert (got[6]["home_team"], got[6]["away_team"]) == ("SK Rapid", "Brand New Club")
    assert got[6]["away_name_rule"] == "provider_name_new_club"
    reasons = dict(zip(ref["provider_fixture_id"], ref["reason"]))
    assert reasons[4].startswith("window_edge")
    assert reasons[5].startswith("provider record holds no half-time score")
    assert 7 not in set(add["provider_fixture_id"]) | set(ref["provider_fixture_id"])   # current season


def test_builder_never_adds_a_fixture_an_owner_list_names():
    add, ref, _ = BHS.build(SERVED, POOL, HT, corrected={3})
    assert 3 not in set(add["provider_fixture_id"])
    assert dict(zip(ref["provider_fixture_id"], ref["reason"]))[3] == "named by an owner correction list"


def test_builder_refuses_next_to_an_unclaimed_served_row():
    served = pd.concat([SERVED, _served([("2026-05-20", "2025/26", "Unknown A", "Unknown B", 2.0, 1.0)])],
                       ignore_index=True)
    add, ref, _ = BHS.build(served, POOL, HT, corrected=set())
    assert 3 not in set(add["provider_fixture_id"])
    assert dict(zip(ref["provider_fixture_id"], ref["reason"]))[3].startswith("an unclaimed served row")


# ── China 2022-12-23: cancelled, not played ─────────────────────────────────────────────────────
def test_cancelled_fixtures_are_placed_by_the_complete_day_list():
    days = pd.DataFrame([
        {"fixture_id": 1, "league": "China Super League", "date": "2022-12-23", "home_name": "Wuhan Zall",
         "away_name": "Dalian Aerbin", "ft_home": 3, "ft_away": 0, "status": "FT"},
        {"fixture_id": 927036, "league": "China Super League", "date": "2022-12-23",
         "home_name": "Hangzhou Greentown", "away_name": "Tianjin Teda", "ft_home": None, "ft_away": None,
         "status": "CANC"},
        {"fixture_id": 926930, "league": "China Super League", "date": "2022-12-23", "home_name": "Henan Jianye",
         "away_name": "Shenzhen Ruby FC", "ft_home": None, "ft_away": None, "status": "CANC"}])
    hist = pd.DataFrame({"league": ["China Super League"] * 3, "date": pd.to_datetime(["2022-12-23"] * 3),
                         "home_team": ["Wuhan Zall", "Zhejiang Professional", "Henan Songshan Longmen"],
                         "away_team": ["Dalian Aerbin", "Tianjin Jinmen Tiger", "Shenzhen"],
                         "home_goals": [3.0, 3.0, 3.0], "away_goals": [0.0, 0.0, 0.0]})
    out, st = BHC.owner_not_played_by_day(hist, days, decisions={927036: "a", 926930: "b"})
    assert dict(zip(out["home_team"], out["fixture_id"])) == {"Zhejiang Professional": 927036,
                                                              "Henan Songshan Longmen": 926930}
    # Zhejiang resolves (alias store); Henan v Shenzhen is placed by its proven alias or, without one,
    # by elimination - either way one row per cancelled fixture
    assert sum(st["by_rule"].values()) == 2 and set(st["by_rule"]) <= {"one_club_resolved", "day_elimination"}
    assert set(out["provider_status"]) == {"CANC"}


def test_a_played_fixture_is_never_a_not_played_decision():
    days = pd.DataFrame([{"fixture_id": 1, "league": "China Super League", "date": "2022-12-23",
                          "home_name": "Wuhan Zall", "away_name": "Dalian Aerbin", "ft_home": 3, "ft_away": 0,
                          "status": "FT"}])
    hist = pd.DataFrame({"league": ["China Super League"], "date": pd.to_datetime(["2022-12-23"]),
                         "home_team": ["Wuhan Zall"], "away_team": ["Dalian Aerbin"],
                         "home_goals": [3.0], "away_goals": [0.0]})
    out, _ = BHC.owner_not_played_by_day(hist, days, decisions={1: "x"})
    assert out.empty


# ── the committed lists ─────────────────────────────────────────────────────────────────────────
def test_committed_supplement_keeps_league_play_offs_and_refuses_cups():
    # owner 2026-11-11 "keep league play-offs": league rounds and the league's own play-off ties stay
    # (the Austrian 2025/26 Conference League play-off included); cups / cup-style title knockouts and
    # placement games are refused, each (league, label) listed with its decision
    import json
    rounds = {int(k): v["round"] for k, v in json.loads(BHS.ROUNDS.read_text(encoding="utf-8"))["fixtures"].items()}
    s = pd.read_csv(HS.SUPPLEMENT)
    assert len(s) and all(BHS.round_reason(rounds.get(int(f)), lg) is None
                          for f, lg in zip(s["provider_fixture_id"], s["league"]))
    assert s["provider_fixture_id"].is_unique
    assert s[["home_goals", "away_goals", "ht_home_goals", "ht_away_goals"]].notna().all().all()
    assert {1545407, 1545664, 1545665} <= set(s["provider_fixture_id"])
    assert not s["league"].isin(["Argentina Primera Division", "USA MLS", "Mexico Liga MX"]).any() or         all(rounds[int(f)].startswith(("Regular Season",)) for f in
            s.loc[s["league"].isin(["Argentina Primera Division", "USA MLS", "Mexico Liga MX"]), "provider_fixture_id"])
    ref = pd.read_csv(BHS.REFUSED)
    arg = ref[(ref["league"] == "Argentina Primera Division") & ref["reason"].str.startswith("cup / cup-style")]
    assert len(arg) == 6
    rr = pd.read_csv(BHS.ROUND_RULES)
    assert set(rr["decision"]) <= {"keep", "exclude"} and rr["reason"].str.len().gt(0).all()
    assert int(rr["fixtures"].sum()) == int(len(s)) + int(ref["reason"].str.startswith(("cup /", "placement")).sum())


def test_round_decisions_keep_league_play_offs_refuse_cups_and_placement():
    keep = [("Austrian Bundesliga", "Final"), ("Austrian Bundesliga", "Semi-finals"),
            ("Championship", "Promotion Play-offs - Final"), ("Dutch Eredivisie", "Relegation Round"),
            ("Scottish Premiership", "Relegation/Promotion"), ("Belgian First Division A", "Championship Round - 3"),
            ("Japan J-League", "Regular Season - 4"), ("Finland Veikkausliiga", "Europa League Play-offs - 1st Round")]
    for lg, lab in keep:
        assert BHS.round_reason(lab, lg) is None, (lg, lab)
    out = [("Argentina Primera Division", "1st Phase - Quarter-finals"), ("USA MLS", "MLS Cup - Final"),
           ("USA MLS", "Play-In Round"), ("Mexico Liga MX", "Apertura - Finals"), ("China Super League", "Semi-finals"),
           ("Japan J-League", "5th place"), ("China Super League", "9th Place Final")]
    for lg, lab in out:
        assert BHS.round_reason(lab, lg) is not None, (lg, lab)
    assert BHS.round_reason("Regular Season - 2", "USA MLS") is None
    assert BHS.round_reason(None, "Championship").startswith("round label unknown")


def test_committed_china_cancelled_rows_are_excluded_as_not_played():
    aw = pd.read_csv(ROOT / "output" / "corrections" / "history_awarded_results.csv")
    c = aw[aw["fixture_id"].isin([927036, 926930])]
    assert set(zip(c["home_team"], c["away_team"], c["provider_status"])) == {
        ("Zhejiang Professional", "Tianjin Jinmen Tiger", "CANC"), ("Henan Songshan Longmen", "Shenzhen", "CANC")}


def test_edinburgh_city_is_one_club_in_scottish_league_one():
    # af-alias-proof/1.0 proved FC Edinburgh -> Edinburgh City (35 dates, league-scoped); the registry
    # pair makes the served history use one spelling, so the play-offs can be placed
    import json
    from src import league_registry as LR
    assert ("Edinburgh City", "FC Edinburgh") in LR.same_club_spellings("Scottish League One")
    assert LR.same_club_spellings("Scottish League Two") == []
    proven = json.loads((ROOT / "config" / "af_history_proven_aliases.json").read_text(encoding="utf-8"))
    assert proven["map"]["Scottish League One"]["FC Edinburgh"] == "Edinburgh City"
    s = pd.read_csv(HS.SUPPLEMENT)
    e = s[(s["league"] == "Scottish League One")
          & (s["home_team"].str.contains("Edinburgh") | s["away_team"].str.contains("Edinburgh"))]
    assert len(e) >= 1 and not (s["home_team"] == "FC Edinburgh").any() and not (s["away_team"] == "FC Edinburgh").any()
