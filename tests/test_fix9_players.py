"""Fix round 9 'players' (owner decisions 2026-11-04): provenance of re-collected counts through the
history merge, the season of national-team rows, the profile collector's key check, and what prop
settlement does with a null count. Updated in fix round 10 'settle' (owner 2026-11-04): the season of
a national-team row is the PROVIDER'S league.season (no resolver), and settlement reads a null count
with the shared null rule (tests/test_fix10_settle.py has the full settlement cases).

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_fix9_players.py -q

No network: every provider answer is a fake; every output goes to tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_player_profiles as PROF  # noqa: E402
from player_model import data_fetcher as DF  # noqa: E402
from player_model import history_integrity as HI  # noqa: E402
from player_model import player_stat_nulls as PSN  # noqa: E402


# ── 2. a re-collected row parsed with the rule replaces its counts as a set ────────────────────────

def _row(fid, pid, date, **vals):
    base = {c: 0 for c in PSN.HISTORY_COUNT_COLS}
    base.update(rating=7.0, minutes=90)
    base.update(vals)
    return {"fixture_id": fid, "player_id": pid, "date": pd.Timestamp(date), **base}


def test_merge_keeps_a_rule_rows_missing_count_missing():
    existing = pd.DataFrame([_row(1, 10, "2026-10-30"), _row(2, 10, "2026-09-01"), _row(3, 10, "2026-10-30")])
    fresh = pd.DataFrame([
        {**_row(1, 10, "2026-10-30", goals=1, saves=None, rating=None), PSN.SOURCE_COL: PSN.PARSED_WITH_RULE},
        {**_row(2, 10, "2026-09-01", goals=4, saves=None), PSN.SOURCE_COL: PSN.PARSED_WITH_RULE},   # frozen
        {**_row(3, 10, "2026-10-30", goals=2, saves=None), PSN.SOURCE_COL: None},                   # old-style
    ])
    out, st = HI.merge_history(existing, fresh, today="2026-11-04")
    by = out.set_index(["fixture_id", "player_id"])
    # live + parsed with the rule: the fresh missing value stays missing (was refilled with the stored 0)
    assert by.loc[(1, 10), "goals"] == 1 and pd.isna(by.loc[(1, 10), "saves"]) and pd.isna(by.loc[(1, 10), "rating"])
    assert by.loc[(1, 10), PSN.SOURCE_COL] == PSN.PARSED_WITH_RULE
    # frozen: stored row kept exactly, so it stays unlabelled (-> zeros_unverified, re-fetch reaches it)
    assert by.loc[(2, 10), "goals"] == 0 and pd.isna(by.loc[(2, 10), PSN.SOURCE_COL])
    # a fresh row without the label keeps combine_first semantics
    assert by.loc[(3, 10), "goals"] == 2 and by.loc[(3, 10), "saves"] == 0
    assert st["refreshed_counts_as_set"] == 1


def test_cup_merge_carries_the_provenance_column():
    src = (ROOT / "player_model" / "pipeline.py").read_text(encoding="utf-8")
    body = src[src.index("def merge_cup_rows"):src.index("def mode_merge_cups")]
    assert "c != SOURCE_COL" in body and "or c == SOURCE_COL" in body


# ── 3. national-team rows: the PROVIDER's season (fix round 10), never "" and never resolved ──────

def test_national_row_season_is_the_providers_or_missing():
    assert DF.provider_fixture_season({"league": {"name": "Friendlies", "season": 2026}}) == "2026"
    assert DF.provider_fixture_season({"league": {"name": "Friendlies", "season": "2025"}}) == "2025"
    for bad in ({"league": {"name": "Friendlies"}}, {"league": {"season": None}}, {"league": {"season": ""}},
                {"league": {"season": "26"}}, {"league": {"season": True}}, {}, None):
        assert DF.provider_fixture_season(bad) is None
    assert not hasattr(DF, "national_row_season")                  # the resolver path is gone


def test_national_team_collect_writes_the_provider_season(monkeypatch, tmp_path):
    from src import season_resolver as SR
    monkeypatch.setattr(DF, "NATIONAL_SEASONS_FILE", tmp_path / "national_fixture_seasons.csv")
    fixture ={"fixture": {"id": 77, "date": "2026-03-31T18:00:00+00:00", "status": {"short": "FT"}},
               "teams": {"home": {"id": 1, "name": "A"}, "away": {"id": 2, "name": "B"}},
               "league": {"name": "Friendlies"}}

    def _api(ep, params):
        if ep == "teams":
            return {"response": [{"team": {"id": 1, "name": "A"}}]}
        if ep == "fixtures":
            return {"response": [fixture]}
        raise AssertionError(ep)
    monkeypatch.setattr(DF, "_api_get", _api)
    monkeypatch.setattr(DF, "_fixture_cache_path", lambda fid: tmp_path / f"{fid}.json")
    monkeypatch.setattr(DF, "_fetch_fixture_statistics", lambda fid: {})
    monkeypatch.setattr(DF, "_fetch_fixture_player_stats", lambda fid: [
        {"team": {"id": 1, "name": "A"}, "players": [{"player": {"id": 5, "name": "P5"},
                                                       "statistics": [{"games": {"minutes": 90}}]}]}])
    n0 = len(SR.degradations())
    rows = DF.collect_national_team_history(last_n=1)
    assert len(rows) == 1 and rows[0]["season"] is None                      # no provider season: missing
    assert len(SR.degradations()) == n0 + 1                                   # recorded, never a year
    assert not (tmp_path / "national_fixture_seasons.csv").exists()
    assert rows[0][PSN.SOURCE_COL] == PSN.PARSED_WITH_RULE and rows[0]["goals"] == 0
    fixture["league"]["season"] = 2026                                        # the provider sends it
    assert DF.collect_national_team_history(last_n=1)[0]["season"] == "2026"
    rec = pd.read_csv(tmp_path / "national_fixture_seasons.csv", dtype=str)
    assert rec[["fixture_id", "league", "date", "season"]].values.tolist() == [["77", "Friendlies", "2026-03-31", "2026"]]
    DF.collect_national_team_history(last_n=1)                                # a re-run never duplicates
    assert len(pd.read_csv(tmp_path / "national_fixture_seasons.csv")) == 1


# ── 3. the profile collector never sends an invalid key ───────────────────────────────────────────

def _hist():
    return pd.DataFrame([
        {"fixture_id": 1, "player_id": 10, "season": "2025", "league": "Premier League", "date": "2025-09-01"},
        {"fixture_id": 2, "player_id": 11, "season": "", "league": "Premier League", "date": "2026-03-01"},
        {"fixture_id": 3, "player_id": 12, "season": "", "league": "Friendlies", "date": "2026-03-31"},
        {"fixture_id": 3, "player_id": 13, "season": None, "league": "Friendlies", "date": "2026-03-31"},
        {"fixture_id": 4, "player_id": 0, "season": "2025", "league": "Premier League", "date": "2025-09-01"},
        {"fixture_id": 5, "player_id": 0, "season": "", "league": "Friendlies", "date": "2026-03-31"},
        {"fixture_id": 6, "player_id": None, "season": "2025", "league": "Premier League", "date": "2025-09-01"},
    ])


class _Client:
    def __init__(self):
        self.calls = []

    def get(self, ep, params):
        self.calls.append((ep, dict(params)))
        return {"response": []}

    def flush(self):
        pass


def test_profile_keys_are_checked_before_any_call(monkeypatch, tmp_path):
    # fix round 10: a blank season is the PROVIDER's recorded fixture season or nothing (no resolver)
    seasons = {2: "2025"}                       # collect-wc recorded fixture 2's league.season
    monkeypatch.setattr(PROF, "OUT", tmp_path / "player_profiles.parquet")
    monkeypatch.setattr(PROF, "provider_seasons", lambda path=None: dict(seasons))
    p = PROF.plan_keys(_hist(), pd.DataFrame())
    assert sorted(p["todo"]) == sorted([(10, "2025"), (11, "2025")])           # blank season proven
    assert p["provider_fixture_rows"] == 1
    assert p["skipped_rows"] == {"player_id_invalid": 3, "season_unresolved": 2}
    assert p["skipped"]["season_unresolved"] == 2
    rows = PROF.invalid_key_rows(p["keys"])
    assert len(rows) == 5 and set(rows["problem"]) == {"player_id_invalid", "season_unresolved"}
    # without the provider record fixture 2's pair is unresolved too: never a resolved / typed year
    p0 = PROF.plan_keys(_hist(), pd.DataFrame(), seasons={})
    assert p0["todo"] == [(10, "2025")] and p0["skipped_rows"]["season_unresolved"] == 3
    pl = "2025"
    assert PROF.key_problem(0, "2025") == "player_id_invalid" and PROF.key_problem(None, "2025") == "player_id_invalid"
    assert PROF.key_problem(5, "") == "season_unresolved" and PROF.key_problem(5, None) == "season_unresolved"
    assert PROF.key_problem(5, "2025") is None and PROF.key_problem("5.0", "2025.0") is None

    # run(): only the valid keys reach the client; the invalid ones are listed beside the history
    monkeypatch.setattr(PROF, "history_frame", lambda cols: _hist())
    c = _Client()
    out_csv = tmp_path / "invalid.csv"
    st = PROF.run(client=c, invalid_out=out_csv)
    sent = {(prm["id"], prm["season"]) for _, prm in c.calls}
    assert sent == {(10, "2025"), (11, pl)} and st["fetched"] == 2
    assert all(int(i) > 0 and str(s).strip() for i, s in sent)
    assert len(pd.read_csv(out_csv)) == 5
    # stored now -> nothing is asked again; the invalid ones are never asked
    c2 = _Client()
    assert PROF.run(client=c2)["todo"] == 0 and c2.calls == []


# ── 4. prop settlement and a null count: the shared null rule (fix round 10) ──────────────────────

def test_settlement_with_a_null_shots_count(monkeypatch):
    """Fix round 10 (owner 2026-11-04): a player who PLAYED with shots.on null, in a fixture the
    provider answered, has 0 shots on target -> LOSS, whatever the team 'Shots on Goal' total says
    (until fix round 10: LOSS only when that total reconciled, otherwise pending for ever). The
    sheet itself settles it; the team-total proof path is not needed."""
    import update_results as U
    sheet = [{"team": {"id": 1}, "players": [
        {"player": {"id": 7, "name": "Seven"}, "statistics": [{"games": {"minutes": 90}, "shots": {"on": None},
                                                               "goals": {"total": None}, "cards": {}}]},
        {"player": {"id": 8, "name": "Eight"}, "statistics": [{"games": {"minutes": 90}, "shots": {"on": 2},
                                                               "goals": {"total": 1}, "cards": {}}]}]},
        {"team": {"id": 2}, "players": [{"player": {"id": 9}, "statistics": [{"games": {"minutes": 90}}]}]}]
    team_sot = {"v": 2}

    def _af(ep, params, hours):
        if ep == "/fixtures/players":
            return sheet
        if ep == "/fixtures/statistics":
            return [{"team": {"id": 1}, "statistics": [{"type": "Shots on Goal", "value": team_sot["v"]}]}]
        return []
    monkeypatch.setattr(U, "_af_response", _af)
    players = U._fixture_players(55)
    assert players["answered"] is True
    rec = players["by_id"][7]
    assert rec["shots_on_target"] == 0 and "shots_on_target" in rec["rule_zero"]   # the rule's 0
    assert U._resolve_player_market("sot", rec) == "LOSS"                       # settled on the sheet
    assert U._rule_decided("sot", rec, "LOSS") == ["shots_on_target"]
    team_sot["v"] = 3                                                           # total not accounted for
    assert U._resolve_player_market("sot", U._fixture_players(55)["by_id"][7]) == "LOSS"   # still LOSS
    # a sent value is kept as sent and is not a rule decision
    assert players["by_id"][8]["shots_on_target"] == 2 and players["by_id"][8]["rule_zero"] == [
        "assists", "red_card", "yellow_card"]
