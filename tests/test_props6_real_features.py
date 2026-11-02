"""fix6/props6 (owner rules 2026-11-02: no fake data, no proxies). Each test fails on
origin/data-truth-100 769db5be.

  1. set-piece proxies purged (train AND serve), typed clip bounds / constants gone from measured
     features, typed-weight formulas withheld from the training feature list;
  2. prev_season_* and shot_accuracy_l10 (real values, explicit minimum samples, one definition);
  3. rest_days on the COMPLETE match history (league + collected cup rows), train and serve;
  4. the cup collector stores team corners (/fixtures/statistics) and the referee (/fixtures);
  5. merge-cups writes only the stored history's columns and carries the collected corners/referee.

Everything offline (synthetic frames, fake HTTP session).

    python -m pytest -p no:cacheprovider tests/test_props6_real_features.py -q
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from player_model import config as C  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model.setpiece import features as SPF  # noqa: E402
from src.strength import lookup  # noqa: E402

IDS = {"Arsenal": 42, "Chelsea": 49, "Everton": 45, "Fulham": 36}
PURGED = ("sp_taker_score", "sp_receiver_score", "set_piece_threat_score", "set_piece_corner_matchup",
          "defender_sot_role_index", "defender_sp_edge", "defender_sot_edge",
          "sp_threat_vs_weak_sp_defense", "sp_taker_assist_edge", "opp_sp_goals_conceded_pg",
          "career_sp_assist_rate", "sp_goals_pg", "aerial_height_sp_composite",
          "sp_goal_probability_composite", "sp_assist_pg", "career_sp_goals_rate", "sp_goals_share")


def _player_rows(fid, date, home, away, season, shots=2, sot=1, minutes=90, league="Premier League"):
    rows = []
    for team, ih in ((home, 1), (away, 0)):
        for j, pos in enumerate(("G", "D", "M", "F")):
            rows.append({
                "fixture_id": fid, "player_id": IDS[team] * 100 + j, "player_name": f"{team}-{pos}",
                "date": date, "league": league, "home_team": home, "away_team": away, "team": team,
                "is_home": ih, "referee": "Ref A", "season": season, "position": pos,
                "minutes": minutes, "started": 1, "goals": 1 if (pos == "F" and ih) else 0,
                "assists": 0, "shots_total": shots, "shots_on_target": sot if pos == "F" else 0,
                "yellow_cards": 0, "key_passes": 1, "rating": 7.0, "fouls_committed": 1,
                "fouls_drawn": 1, "duels_total": 4, "duels_won": 2, "team_corners": 5,
                "red_cards": 0, "passes_total": 30, "offsides": 0, "tackles_total": 1,
                "interceptions": 1, "dribbles_attempted": 1, "dribbles_success": 1,
                "dribbles_past": 0, "goals_conceded": 0, "saves": 2 if pos == "G" else 0,
                "penalty_scored": 0, "penalty_missed": 0, "penalty_won": 0, "penalty_saved": 0,
                "sp_goal": 1 if (pos == "D" and ih) else 0, "headed_goal": 0, "fk_goal": 0,
                "sp_assist": 0, "fk_assist": 0,
                "team_id": IDS[team], "opponent_id": IDS[away] if ih else IDS[home]})
    return rows


def _season_rows(season: str, start: str, rounds: int, fid0: int, **kw):
    rows = []
    for r in range(rounds):
        d = pd.Timestamp(start) + pd.Timedelta(days=7 * r)
        pairs = [("Arsenal", "Chelsea"), ("Everton", "Fulham")] if r % 2 == 0 else \
                [("Chelsea", "Everton"), ("Fulham", "Arsenal")]
        for k, (h, a) in enumerate(pairs):
            rows += _player_rows(fid0 + 10 * r + k, d, h, a, season, **kw)
    return rows


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """No strength tables, no collected data, no Understat: only the rows a test passes."""
    sdir = tmp_path / "strength"
    sdir.mkdir()
    monkeypatch.setattr(lookup, "DIR", sdir)
    monkeypatch.setattr(FE, "COLLECT_DIR", tmp_path)
    monkeypatch.setattr(SPF, "DIR", tmp_path)
    from player_model import cup_history as CH
    monkeypatch.setattr(CH, "CUP_ROWS_FILE", tmp_path / "cup_player_rows.parquet")
    monkeypatch.setattr(CH, "CUP_META_FILE", tmp_path / "cup_fixture_meta.parquet", raising=False)

    def _clear():
        from player_model import team_identity as TI
        TI._CACHE.clear()
        lookup._LOADED.clear()
        FE._COLLECTED.clear()
        FE._SERVE_CACHE.clear()
    _clear()
    yield tmp_path
    _clear()


# â”€â”€ 1. set-piece proxies purged â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_setpiece_proxies_are_not_built_for_training_or_serving(iso):
    rows = _season_rows("2024", "2024-08-10", 8, 7_000_000)
    b = FE.build_features(rows)
    assert not [c for c in PURGED if c in b.columns]
    up = [{"player_id": IDS["Arsenal"] * 100 + 1, "team": "Arsenal", "opponent": "Chelsea", "is_home": 1,
           "position": "D", "fixture_league": "Premier League", "fixture_date": "2024-10-20"}]
    out = FE.build_upcoming_features(up, b, {}, {}, FE._build_history_index(b))
    assert len(out) == 1 and not [c for c in PURGED if c in out.columns]
    # the Understat set-piece features are the only set-piece columns (NaN: no Understat data)
    assert all(c in out.columns and math.isnan(out[c].iloc[0]) for c in SPF.SETPIECE_COLUMNS)


def test_setpiece_proxies_never_in_the_training_feature_list_or_contract():
    from player_model import props_contract as PC
    assert not set(PURGED) & set(C.PLAYER_FEATURE_COLS)
    assert set(PURGED) <= set(C.SETPIECE_PROXIES_REMOVED)
    assert not set(PURGED) & set(PC.load_contract()["features"])


def test_no_set_piece_lazy_factor():
    from player_model import predict as P
    feat = {"set_piece_threat_score": 9.0, "pos_defender": 1, "pos_midfielder": 0,
            "opp_sot_conceded_pg": 1.0, "minutes_pg": 90}
    assert "SET_PIECE" not in P._detect_lazy_factors({}, feat, "sot")
    assert "set_piece_threat_score" not in inspect.getsource(P)


def _one_row(**vals) -> pd.DataFrame:
    df = pd.DataFrame([vals])
    FE._ensure(df, FE._COMPOSITE_INPUTS)
    return FE._add_composites(df)


def test_measured_ratios_are_not_clipped_to_typed_bounds():
    f = _one_row(opp_goals_conceded_pg=4.0, lg_goals_per_team=1.0, goals_pg=2.0, sot_pg=1.5,
                 opp_def_player_rating_pg=8.0, lg_def_rating=1.0, opp_top_def_rating=8.0,
                 lg_def_rating_top=1.0, opp_def_aerial_win_rate=0.6, lg_def_aerial=0.5,
                 offsides_pg=3.0, pos_forward=1, pos_midfielder=0, pos_defender=0,
                 fouls_per90=1.0, dribbles_pg=0.0, dribble_success_rate=1.0, opp_mid_cards_pg=1.0,
                 referee_yellows_pg=0.1, lg_yellows_per_match=4.0).iloc[0]
    assert f["opp_strength_index"] == pytest.approx(4.0)            # was clipped to 3.0
    assert f["quality_adj_goals_pg"] == pytest.approx(8.0)          # was clipped to 5.0
    assert f["opp_def_player_quality"] == pytest.approx(64 * 1.2)   # was clipped to 4.0
    assert f["offside_aggressiveness"] == pytest.approx(3.0)        # was clipped to 2.0
    assert f["shooting_efficiency_index"] == pytest.approx(2.0 / 1.5)   # was clipped to 1.0
    assert f["card_clash_index"] == pytest.approx(1.0 * 1.0 * 0.1 / 4.0)  # ratio floor 0.1 gone


def test_foul_committer_ratio_is_the_exact_share(iso):
    rows = _season_rows("2024", "2024-08-10", 4, 7_100_000)
    b = FE.build_features(rows)
    # every row: 1 committed, 1 drawn -> exactly 0.5 (was 1 / 2.01)
    assert (b["foul_committer_ratio"].dropna() == 0.5).all() and b["foul_committer_ratio"].notna().any()


def _two_seasons(prev_rounds=8):
    return (_season_rows("2023", "2023-08-12", prev_rounds, 7_200_000, shots=2, sot=1)
            + _season_rows("2024", "2024-08-10", 6, 7_300_000, shots=4, sot=4))


ARS_F = IDS["Arsenal"] * 100 + 3


def test_prev_season_player_and_team_features_training(iso):
    b = FE.build_features(_two_seasons()).sort_values(["player_id", "date"])
    p = b[b["player_id"] == ARS_F].set_index("date")
    first24 = p.loc["2024-08-10"]
    # the player's real 2023 values: 8 matches, a goal in each of the 4 home games, 16 shots / 8 SOT
    assert first24["prev_season_appearances"] == 8
    assert first24["prev_season_goals_pg"] == pytest.approx(0.5)
    assert first24["prev_season_shots_pg"] == pytest.approx(2.0)
    assert first24["prev_season_shot_accuracy"] == pytest.approx(0.5)
    # own names: this season's value is still missing on the season's first match (never filled)
    assert math.isnan(first24["season_goals_pg"])
    # the teams' real 2023 values (Arsenal scored at home only and conceded away only; Chelsea too)
    assert first24["prev_season_team_goals_pg"] == pytest.approx(0.5)
    assert first24["prev_season_team_conceded_pg"] == pytest.approx(0.5)
    assert first24["prev_season_opp_goals_pg"] == pytest.approx(0.5)
    # the first season has no previous season held: NaN, not 0
    assert p.loc[p.index < "2024-01-01", "prev_season_goals_pg"].isna().all()
    assert p.loc[p.index < "2024-01-01", "prev_season_appearances"].isna().all()


def test_prev_season_minimum_sample(iso):
    b = FE.build_features(_two_seasons(prev_rounds=4))
    r = b[(b["player_id"] == ARS_F) & (b["date"] == "2024-08-10")].iloc[0]
    assert r["prev_season_appearances"] == 4                       # a real count
    assert math.isnan(r["prev_season_goals_pg"])                   # < MIN_PREV_SEASON_MATCHES (5)
    assert math.isnan(r["prev_season_shot_accuracy"])              # 8 shots < MIN_SHOTS_FOR_ACCURACY


def test_shot_accuracy_l10_is_a_separate_longer_window(iso):
    b = FE.build_features(_two_seasons()).sort_values(["player_id", "date"])
    p = b[b["player_id"] == ARS_F].reset_index(drop=True)
    # p row k = the player's (k+1)th match (the first is dropped: no prior match). 2023: 2 shots /
    # 1 SOT a match; 2024: 4 / 4.
    hist = [(2, 1)] * 8 + [(4, 4)] * 6
    for k in range(len(p)):
        prev = hist[max(0, k + 1 - C.SHOT_ACC_WINDOW):k + 1]
        sh, so = sum(x for x, _ in prev), sum(y for _, y in prev)
        want = so / sh if sh >= C.MIN_SHOTS_FOR_ACCURACY else float("nan")
        assert p.loc[k, "shot_accuracy_l10"] == pytest.approx(want, nan_ok=True), k
    assert p["shot_accuracy_l10"].iloc[-1] != pytest.approx(p["shot_accuracy_rate"].iloc[-1])


def _write_fixtures(rows, sdir):
    """competition_fixtures.parquet of the rows' fixtures (the club-id source serving resolves by)."""
    f = pd.DataFrame(rows).drop_duplicates("fixture_id")
    fx = pd.DataFrame({"fixture_id": f["fixture_id"], "league_id": 39, "league": f["league"],
                       "league_name": f["league"], "competition_type": "club_league", "season": f["season"],
                       "date": pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d"),
                       "home_id": f["home_team"].map(IDS), "home_name": f["home_team"],
                       "away_id": f["away_team"].map(IDS), "away_name": f["away_team"],
                       "ft_home": 1, "ft_away": 0, "status": "FT"})
    fx.to_parquet(sdir / "competition_fixtures.parquet", index=False)


def test_prev_season_and_l10_serving_use_the_training_definition(iso):
    rows = _two_seasons()
    _write_fixtures(rows, iso / "strength")
    b = pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d["date"]))   # every match held
    up = [{"player_id": ARS_F, "team": "Arsenal", "opponent": "Chelsea", "is_home": 1, "position": "F",
           "fixture_league": "Premier League", "fixture_date": "2024-10-20"}]
    r = FE.build_upcoming_features(up, b, {}, {}, FE._build_history_index(b)).iloc[0]
    assert r["prev_season_goals_pg"] == pytest.approx(0.5) and r["prev_season_appearances"] == 8
    assert r["prev_season_shot_accuracy"] == pytest.approx(0.5)
    assert r["prev_season_team_goals_pg"] == pytest.approx(0.5)
    assert r["prev_season_opp_conceded_pg"] == pytest.approx(0.5)
    hist = [(2, 1)] * 8 + [(4, 4)] * 6
    last = hist[-C.SHOT_ACC_WINDOW:]
    assert r["shot_accuracy_l10"] == pytest.approx(sum(y for _, y in last) / sum(x for x, _ in last))
    assert set(FE.PREV_SEASON_COLUMNS) | {"shot_accuracy_l10"} <= set(C.PLAYER_FEATURE_COLS)


# â”€â”€ 3. rest days on the complete match history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _cup_file(tmp, day, fid=7_900_001, minutes=90, status="FT"):
    """A collected cup row file (what scripts/collect_cup_player_stats.py writes) with one cup tie
    of Arsenal on `day`."""
    rows = [dict(r, league="Champions League", fixture_status=status, date=str(pd.Timestamp(day).date()),
                 minutes=minutes)
            for r in _player_rows(fid, pd.Timestamp(day), "Arsenal", "Chelsea", "2024")]
    pd.DataFrame(rows).to_parquet(tmp / "cup_player_rows.parquet", index=False)


def test_training_rest_days_count_collected_cup_matches_and_short_cameos(iso):
    rows = _season_rows("2024", "2024-08-10", 4, 7_400_000)        # Arsenal: 08-10, 08-17, 08-24, 08-31
    # a 5-minute league cameo on 08-27 (not a training row: < 10 minutes) and a cup tie on 08-21
    rows += [dict(r, minutes=5) for r in _player_rows(7_400_900, pd.Timestamp("2024-08-27"),
                                                       "Arsenal", "Everton", "2024")]
    _cup_file(iso, "2024-08-21", status="AET")                      # extra time: still a match played
    b = FE.build_features(rows)
    p = b[b["player_id"] == ARS_F].set_index("date")["rest_days"]
    assert p.loc["2024-08-24"] == 3        # since the cup tie (was 7: league rows only)
    assert p.loc["2024-08-31"] == 4        # since the cameo (was 7)
    assert p.loc["2024-08-17"] == 7


def test_rest_days_are_not_capped(iso):
    rows = (_season_rows("2024", "2024-03-02", 2, 7_500_000)          # Arsenal 03-02, 03-09
            + _season_rows("2024", "2024-05-18", 2, 7_510_000))       # Arsenal 05-18 (70 days later)
    b = FE.build_features(rows)
    p = b[b["player_id"] == ARS_F].set_index("date")["rest_days"]
    assert p.loc["2024-05-18"] == 70       # was clipped to 30


def test_serving_rest_days_count_collected_cup_matches(iso):
    rows = _season_rows("2024", "2024-08-10", 4, 7_600_000)
    hist = pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d["date"]))
    _cup_file(iso, "2024-09-03")
    up = [{"player_id": ARS_F, "team": "Arsenal", "opponent": "Chelsea", "is_home": 1, "position": "F",
           "fixture_league": "Premier League", "fixture_date": "2024-09-07"}]
    r = FE.build_upcoming_features(up, hist, {}, {}, FE._build_history_index(hist)).iloc[0]
    assert r["rest_days"] == 4             # since the collected cup tie (was 7: league history only)
    FE._SERVE_CACHE.clear()
    up[0]["fixture_date"] = "2024-11-30"
    r = FE.build_upcoming_features(up, hist, {}, {}, FE._build_history_index(hist)).iloc[0]
    assert r["rest_days"] == 88            # uncapped (was 30)


# â”€â”€ 4. cup collector: corners + referee, budget-guarded, counted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _Resp:
    def __init__(self, body):
        self._b, self.status_code, self.headers = body, 200, {"x-ratelimit-requests-remaining": "70000"}

    def json(self):
        return self._b


class _MetaSession:
    """Fake API-Football: /fixtures?ids= answers with referee (+ statistics for even ids);
    /fixtures/statistics answers with corners."""
    def __init__(self):
        self.calls = []

    @staticmethod
    def _stats(fid):
        return [{"team": {"id": 1000 + fid}, "statistics": [{"type": "Corner Kicks", "value": 6}]},
                {"team": {"id": 2000 + fid}, "statistics": [{"type": "Corner Kicks", "value": None}]}]

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1] if "statistics" not in url else "fixtures/statistics", params))
        if url.endswith("/status"):
            return _Resp({"response": {}})
        if url.endswith("/fixtures/statistics"):
            return _Resp({"response": self._stats(int(params["fixture"])), "errors": []})
        if url.endswith("/fixtures"):
            out = []
            for f in str(params["ids"]).split("-"):
                f = int(f)
                r = {"fixture": {"id": f, "referee": f"Ref {f}" if f != 3 else None}}
                if f % 2 == 0:
                    r["statistics"] = self._stats(f)
                out.append(r)
            return _Resp({"response": out, "errors": []})
        return _Resp({"response": [], "errors": []})


def _meta_world(tmp, monkeypatch, n=21):
    import collect_cup_player_stats as K
    monkeypatch.setattr(K, "META", tmp / "meta.parquet")
    monkeypatch.setattr(K, "JOURNAL", tmp / "journal.parquet")
    jr = [{"fixture_id": f, "league": "Champions League", "season": "2025", "date": f"2025-10-{1 + f % 28:02d}",
           "n_rows": 30, "fetched_at": "x"} for f in range(1, n + 1)]
    jr.append({"fixture_id": 99, "league": "Champions League", "season": "2025", "date": "2025-10-01",
               "n_rows": 0, "fetched_at": "x"})                                  # no rows: no meta call
    pd.DataFrame(jr).to_parquet(tmp / "journal.parquet", index=False)
    fx = pd.DataFrame({"fixture_id": list(range(1, n + 1)) + [99],
                       "home_id": [1000 + f for f in range(1, n + 1)] + [1],
                       "away_id": [2000 + f for f in range(1, n + 1)] + [2]})
    return K, fx


def test_cup_meta_corners_and_referee_counted_and_resumable(tmp_path, monkeypatch):
    import collect_common as CC
    K, fx = _meta_world(tmp_path, monkeypatch)
    s = _MetaSession()
    c = CC.AFClient(key="k", session=s, budget_file=tmp_path / "b.json", pause_s=0)
    st = K.run_meta(c, fx)
    ids_calls = [p for u, p in s.calls if u == "fixtures"]
    stat_calls = [p["fixture"] for u, p in s.calls if u == "fixtures/statistics"]
    assert len(ids_calls) == 2 and all(len(p["ids"].split("-")) <= 20 for p in ids_calls)   # 21 -> 20 + 1
    assert sorted(stat_calls) == [f for f in range(1, 22) if f % 2]     # only where statistics were not embedded
    assert c.calls_run == 2 + 11 and st["calls_fixtures"] == 2 and st["calls_statistics"] == 11
    m = pd.read_parquet(tmp_path / "meta.parquet").set_index("fixture_id")
    assert len(m) == 21 and 99 not in m.index
    assert (m["corners_status"] == "ok").all()
    assert (m["home_corners"] == 6).all() and m["away_corners"].isna().all()          # null -> NaN, never 0
    assert m.loc[5, "referee"] == "Ref 5" and pd.isna(m.loc[3, "referee"])
    # resumable: a second run asks nothing
    s2 = _MetaSession()
    K.run_meta(CC.AFClient(key="k", session=s2, budget_file=tmp_path / "b.json", pause_s=0), fx)
    assert not [u for u, _ in s2.calls if u != "status"]


def test_cup_meta_budget_stop_is_clean(tmp_path, monkeypatch):
    import json
    import collect_common as CC
    K, fx = _meta_world(tmp_path, monkeypatch)
    (tmp_path / "b.json").write_text(json.dumps({CC.utc_day(): 100}))
    c = CC.AFClient(key="k", session=_MetaSession(), budget_file=tmp_path / "b.json", daily_cap=101, pause_s=0)
    st = K.run_meta(c, fx)
    assert "budget_stop" in st and c.calls_run == 1
    m = pd.read_parquet(tmp_path / "meta.parquet")
    assert len(m) == 20 and (m["corners_status"] == "pending").sum() == 10            # resumable state


def test_cup_rows_carry_the_sidecar_corners_and_referee(tmp_path):
    from player_model import cup_history as CH
    rows = pd.DataFrame([
        {"fixture_id": 5, "player_id": 1, "team_id": 1005, "team_corners": None, "referee": None},
        {"fixture_id": 5, "player_id": 2, "team_id": 2005, "team_corners": None, "referee": None},
        {"fixture_id": 6, "player_id": 3, "team_id": 1006, "team_corners": 4.0, "referee": "Own Ref"},
        {"fixture_id": 7, "player_id": 4, "team_id": 1007, "team_corners": None, "referee": None}])
    meta = pd.DataFrame([
        {"fixture_id": 5, "referee": "Ref 5", "home_id": 1005, "away_id": 2005, "home_corners": 6.0,
         "away_corners": 3.0, "corners_status": "ok"},
        {"fixture_id": 6, "referee": "Ref 6", "home_id": 1006, "away_id": 2006, "home_corners": 9.0,
         "away_corners": 1.0, "corners_status": "ok"}])
    rows.to_parquet(tmp_path / "rows.parquet", index=False)
    meta.to_parquet(tmp_path / "meta.parquet", index=False)
    out = CH.cup_rows(tmp_path / "rows.parquet", tmp_path / "meta.parquet").set_index("player_id")
    assert out.loc[1, "team_corners"] == 6.0 and out.loc[2, "team_corners"] == 3.0
    assert out.loc[1, "referee"] == "Ref 5"
    assert out.loc[3, "team_corners"] == 4.0 and out.loc[3, "referee"] == "Own Ref"   # the row's own value wins
    assert math.isnan(out.loc[4, "team_corners"]) and pd.isna(out.loc[4, "referee"])     # no sidecar row
    assert pd.read_parquet(tmp_path / "rows.parquet")["team_corners"].isna().sum() == 3  # file not rewritten


# ── 5. merge-cups ready to run ──────────────────────────────────────────────────────────────────

def test_merge_cups_writes_only_stored_columns_and_the_sidecar_values(iso, monkeypatch):
    from player_model import pipeline as P
    rows = _season_rows("2024", "2024-08-10", 4, 7_700_000)
    stored = FE.build_features(rows)
    # the stored history predates this build's new features
    stored = stored.drop(columns=[c for c in (*FE.PREV_SEASON_COLUMNS, "shot_accuracy_l10") if c in stored.columns])
    hdir = iso / "player_history.parquet"
    P._save_history(stored, hdir)
    monkeypatch.setattr(P, "HISTORY_CACHE", hdir)
    _cup_file(iso, "2024-08-21")
    for r in ("team_corners", "referee"):                                   # what the collector writes
        f = pd.read_parquet(iso / "cup_player_rows.parquet")
        f[r] = None
        f.to_parquet(iso / "cup_player_rows.parquet", index=False)
    pd.DataFrame([{"fixture_id": 7_900_001, "date": "2024-08-21", "referee": "Cup Ref",
                   "home_id": IDS["Arsenal"], "away_id": IDS["Chelsea"], "home_corners": 7.0,
                   "away_corners": 2.0, "corners_status": "ok", "fetched_at": "x"}]
                 ).to_parquet(iso / "cup_fixture_meta.parquet", index=False)
    FE._COLLECTED.clear()
    P.mode_merge_cups()
    out = pd.read_parquet(hdir)
    assert set(out.columns) == set(stored.columns)                  # no column on the cup rows alone
    cup = out[out["league"] == "Champions League"]
    assert len(cup) > 0 and len(out) == len(stored) + len(cup)
    assert set(cup["referee"]) == {"Cup Ref"}
    assert set(pd.to_numeric(cup.loc[cup["team"] == "Arsenal", "team_corners"])) == {7.0}
    assert set(pd.to_numeric(cup.loc[cup["team"] == "Chelsea", "team_corners"])) == {2.0}
    # every stored row is untouched (a committed row is never rewritten)
    old = out[out["league"] != "Champions League"].sort_values(["fixture_id", "player_id"]).reset_index(drop=True)
    ref = stored.sort_values(["fixture_id", "player_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(old["rest_days"], ref["rest_days"], check_dtype=False)


def test_typed_weight_formulas_are_withheld_from_training():
    typed = {"penalty_threat_score", "card_risk_index", "midfield_engine_score",
             "progressive_carrier_score", "disciplinary_pressure_index", "foul_magnet_score",
             "defensive_vulnerability_index", "creative_playmaker_score", "opp_mid_discipline"}
    assert not typed & set(C.PLAYER_FEATURE_COLS)
    assert typed <= set(C.TYPED_FORMULA_FEATURES_WITHHELD)
