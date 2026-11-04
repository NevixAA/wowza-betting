"""Pre-kickoff capture (owner approval 2026-11-04; switched off until 2026-11-08).

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_prematch_capture.py -q

No network: the API-Football session is a counting fake that also moves a fake clock, so the test can
tell the fetch time from the pass start time. Every output goes to tmp_path.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import capture_prematch as CP  # noqa: E402
import collect_common as CC  # noqa: E402
from collect_common import AFClient  # noqa: E402

REG, OTHER = 40, 999                       # Championship's api_football_id; a league outside the registry
LEAGUES = {REG: "Championship"}
KO = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)
FID = 1001


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


class _Resp:
    def __init__(self, body, remaining="70000", status=200):
        self.status_code, self._b = status, body
        self.headers = {"x-ratelimit-requests-remaining": remaining}

    def json(self):
        return self._b


def _fixture_item(fid, league, ko, status="NS", home=(10, "Leeds"), away=(20, "Derby")):
    return {"fixture": {"id": fid, "date": ko.isoformat(), "timestamp": int(ko.timestamp()),
                        "status": {"short": status}},
            "league": {"id": league, "season": 2026},
            "teams": {"home": {"id": home[0], "name": home[1]}, "away": {"id": away[0], "name": away[1]}}}


def _list_body():
    return {"get": "fixtures", "parameters": {"date": "x"}, "errors": [], "results": 4, "response": [
        _fixture_item(FID, REG, KO),
        _fixture_item(1002, OTHER, KO),                              # not a registry league
        _fixture_item(1003, REG, KO, status="PST"),                  # postponed: never planned
        _fixture_item(1004, REG, KO, status="TBD"),                  # kickoff time not defined
    ]}


def _odds_body():
    return {"errors": [], "paging": {"current": 1, "total": 1}, "response": [{
        "fixture": {"id": FID, "timestamp": int(KO.timestamp())}, "update": "2026-11-14T13:40:00+00:00",
        "bookmakers": [{"id": 8, "name": "Bet365", "bets": [
            {"id": 1, "name": "Match Winner", "values": [{"value": "Home", "odd": "2.10"}]},
            {"id": 92, "name": "Anytime Goal Scorer", "values": [
                {"value": "Joel Piroe", "odd": "2.75"},
                {"value": "P. Bamford", "odd": "3.10"},
                {"value": "Nobody Known", "odd": "9.00"}]},
            {"id": 215, "name": "Away Player Shots On Target Total", "values": [
                {"value": "Jerry Yates - 1", "odd": "1.80"}]},
        ]}]}]}


def _lineups_body():
    def p(pid, name):
        return {"player": {"id": pid, "name": name, "number": 9, "pos": "F", "grid": "4:1"}}
    return {"errors": [], "response": [
        {"team": {"id": 10, "name": "Leeds"}, "formation": "4-2-3-1", "coach": {"id": 7, "name": "D. Farke"},
         "startXI": [p(501, "Joel Piroe"), p(502, "Patrick Bamford")], "substitutes": [p(None, "Unknown Kid")]},
        {"team": {"id": 20, "name": "Derby"}, "formation": "3-5-2", "coach": {"id": None, "name": None},
         "startXI": [p(601, "Jerry Yates")], "substitutes": []}]}


def _injuries_body():
    return {"errors": [], "response": [{
        "player": {"id": 777, "name": "Ethan Ampadu", "type": "Missing Fixture", "reason": "Knee Injury"},
        "team": {"id": 10, "name": "Leeds"}, "fixture": {"id": FID, "timestamp": int(KO.timestamp())}}]}


class _Session:
    """Counts calls by endpoint; each call moves the clock by `step` (the fetch takes time)."""

    def __init__(self, clock: Clock, step=timedelta(seconds=40), fail=(), remaining="70000"):
        self.clock, self.step, self.fail, self.remaining = clock, step, set(fail), remaining
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, headers=None, params=None, timeout=None):
        ep = url.rsplit("/v3.football.api-sports.io/", 1)[-1]
        if ep == "status":
            return _Resp({"response": {"requests": {"current": 0, "limit_day": 75000}}})
        self.calls.append((ep, dict(params or {})))
        self.clock.t = self.clock.t + self.step
        if ep in self.fail:
            return _Resp({}, status=500)
        body = {"fixtures": _list_body, "odds": _odds_body, "fixtures/lineups": _lineups_body,
                "injuries": _injuries_body}[ep]()
        return _Resp(body, remaining=self.remaining)


def _client(tmp, sess, cap=CP.DAILY_CAP, floor=CP.FLOOR):
    return AFClient(key="k", session=sess, budget_file=tmp / "prematch" / "af_budget.json",
                    daily_cap=cap, floor=floor, pause_s=0)


def _pass(tmp, clock, sess=None, client=None, tag="20261114T1000Z-1"):
    sess = sess or _Session(clock)
    client = client or _client(tmp, sess)
    st = CP.run_pass(client=client, root=tmp / "prematch", now_fn=clock, leagues=LEAGUES, tag=tag,
                     is_enabled=True)
    return st, sess


def _timeline(tmp, start_min=125, end_min=-5, every=5, tag="20261114T1000Z-1"):
    """Passes every `every` minutes from KO-start_min to KO-end_min; one client = one workflow run."""
    clock = Clock(KO - timedelta(minutes=start_min))
    sess = _Session(clock)
    client = _client(tmp, sess)
    t = KO - timedelta(minutes=start_min)
    while t <= KO - timedelta(minutes=end_min):
        clock.t = t
        CP.run_pass(client=client, root=tmp / "prematch", now_fn=clock, leagues=LEAGUES, tag=tag,
                    is_enabled=True)
        t += timedelta(minutes=every)
    return sess


def _by_ep(sess):
    out = {}
    for ep, p in sess.calls:
        out[ep] = out.get(ep, 0) + 1
    return out


# -- 1. slot selection at each lead time -----------------------------------------------------------

@pytest.mark.parametrize("mbk, expect", [
    (121, []), (120, ["injuries_120"]), (95, ["injuries_120"]), (85, []), (75, ["odds_75"]),
    (56, ["odds_75"]), (50, []), (45, ["lineups_45"]), (25, ["lineups_45"]), (20, []),
    (15, ["odds_15"]), (2, ["odds_15"]), (1, []), (-10, []),
])
def test_slot_selection_at_each_lead_time(mbk, expect):
    fx = {"fixture_id": FID, "kickoff": KO, "status_short": "NS"}
    plan, _ = CP.due_slots([fx], set(), KO - timedelta(minutes=mbk))
    assert [s.name for _, s in plan] == expect


def test_a_window_already_closed_is_counted_missed_never_fetched():
    fx = {"fixture_id": FID, "kickoff": KO, "status_short": "NS"}
    plan, why = CP.due_slots([fx], set(), KO - timedelta(minutes=40))
    assert [s.name for _, s in plan] == ["lineups_45"]
    assert why == {"missed_injuries_120": 1, "missed_odds_75": 1}


def test_only_not_started_fixtures_with_a_known_kickoff_are_planned():
    now = KO - timedelta(minutes=100)
    fxs = [{"fixture_id": 1, "kickoff": KO, "status_short": "PST"},
           {"fixture_id": 2, "kickoff": None, "status_short": "NS"},
           {"fixture_id": 3, "kickoff": KO, "status_short": "TBD"}]
    plan, why = CP.due_slots(fxs, set(), now)
    assert plan == [] and why == {"status_PST": 1, "kickoff_unknown": 1, "status_TBD": 1}


# -- 2. a full timeline: every slot exactly once, never twice ----------------------------------------

def test_each_slot_is_fetched_once_and_stored_with_its_timing(tmp_path):
    sess = _timeline(tmp_path)
    eps = _by_ep(sess)
    assert eps["injuries"] == 1 and eps["odds"] == 2 and eps["fixtures/lineups"] == 1
    assert all(p["fixture"] == FID for ep, p in sess.calls if ep != "fixtures")   # 1002-1004 never asked
    assert eps["fixtures"] == 3                    # 130 min of passes, a list re-read every 60 min
    root = tmp_path / "prematch"
    j = CP.read_table("slots", root)
    assert sorted(j["slot"]) == ["injuries_120", "lineups_45", "odds_15", "odds_75"]
    for r in j.itertuples():
        s = CP.SLOT_BY_NAME[r.slot]
        assert s.latest <= r.minutes_before_kickoff <= s.target
        assert r.kickoff_utc == KO.isoformat(timespec="seconds") and r.league == "Championship"
    odds = CP.read_table("prop_odds", root)
    assert set(odds["slot"]) == {"odds_75", "odds_15"}
    assert set(odds["bet_name"]) == {"Anytime Goal Scorer", "Away Player Shots On Target Total"}
    assert set(odds["bookmaker_id"]) == {8}
    yates = odds[odds["value_raw"] == "Jerry Yates - 1"].iloc[0]
    assert yates["player_name"] == "Jerry Yates" and yates["line"] == 0.5 and yates["bet_prefix"] == "away"
    bets = CP.read_table("odds_bets", root)
    assert bets.set_index("bet_name")["is_player_prop"].to_dict() == {
        "Match Winner": False, "Anytime Goal Scorer": True, "Away Player Shots On Target Total": True}
    lp = CP.read_table("lineup_players", root)
    assert set(lp["player_id"].dropna()) == {501, 502, 601} and lp["player_id"].isna().sum() == 1
    lt = CP.read_table("lineup_teams", root)
    assert lt.set_index("team_id")["coach_id"].isna().to_dict() == {10: False, 20: True}   # None stays None
    inj = CP.read_table("injuries", root)
    assert list(inj["player_id"]) == [777] and inj["minutes_before_kickoff"].iloc[0] > 90
    raw = CP.read_table("raw", root)
    assert set(raw["raw_scope"]) == {"full", "registry_items"}
    assert '"Match Winner"' in raw[raw["slot"] == "odds_75"]["raw_json"].iloc[0]   # whole answer kept


def test_a_second_run_never_fetches_a_journaled_slot_again(tmp_path):
    _timeline(tmp_path, start_min=125, end_min=30)                 # run 1: injuries, odds_75, lineups
    sess = _timeline(tmp_path, start_min=125, end_min=-5, tag="20261114T1300Z-2")  # run 2 replays it
    eps = _by_ep(sess)
    assert eps.get("injuries", 0) == 0 and eps.get("fixtures/lineups", 0) == 0
    assert eps["odds"] == 1                                         # only odds_15 was still open
    j = CP.read_table("slots", tmp_path / "prematch")
    assert j.groupby(["fixture_id", "endpoint", "slot"]).size().max() == 1


def test_a_failed_request_records_nothing_and_is_asked_again(tmp_path):
    clock = Clock(KO - timedelta(minutes=118))
    sess = _Session(clock, fail={"injuries"})
    st, _ = _pass(tmp_path, clock, sess=sess)
    assert st["failed"] == 1 and CP.read_table("slots", tmp_path / "prematch").empty
    sess.fail.clear()
    clock.t = KO - timedelta(minutes=110)
    _pass(tmp_path, clock, sess=sess, client=_client(tmp_path, sess))
    assert list(CP.read_table("slots", tmp_path / "prematch")["slot"]) == ["injuries_120"]


def test_an_empty_answer_is_journaled_and_never_asked_again(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_injuries_body", lambda: {"errors": [], "response": []})
    clock = Clock(KO - timedelta(minutes=118))
    _pass(tmp_path, clock)
    clock.t = KO - timedelta(minutes=100)
    _, sess = _pass(tmp_path, clock)
    assert "injuries" not in _by_ep(sess)
    j = CP.read_table("slots", tmp_path / "prematch")
    assert list(j["status"]) == ["empty"] and list(j["n_items"]) == [0]


# -- 3. captured_at is the real fetch time ------------------------------------------------------------

def test_captured_at_is_the_time_the_answer_arrived_not_the_pass_start(tmp_path):
    start = KO - timedelta(minutes=118)
    clock = Clock(start)
    sess = _Session(clock, step=timedelta(minutes=3))                # each call takes 3 minutes
    _pass(tmp_path, clock, sess=sess)
    j = CP.read_table("slots", tmp_path / "prematch").iloc[0]
    # the list call ended at start+3, the injuries call was requested then and answered at start+6
    assert j["requested_at_utc"] == (start + timedelta(minutes=3)).isoformat(timespec="seconds")
    assert j["captured_at_utc"] == (start + timedelta(minutes=6)).isoformat(timespec="seconds")
    assert j["minutes_before_kickoff"] == 112.0
    assert j["kickoff_utc_response"] == KO.isoformat(timespec="seconds")
    lst = CP.read_table("lists", tmp_path / "prematch").iloc[0]
    assert lst["captured_at_utc"] == (start + timedelta(minutes=3)).isoformat(timespec="seconds")
    assert lst["n_fixtures_all"] == 4 and lst["n_registry"] == 3


# -- 4. player ids: proven identity only -------------------------------------------------------------

ROSTER = [{"player_id": 501, "team_id": 10, "player_name": "Joel Piroe"},
          {"player_id": 502, "team_id": 10, "player_name": "Patrick Bamford"},
          {"player_id": 503, "team_id": 10, "player_name": "Dan James"},
          {"player_id": 603, "team_id": 20, "player_name": "Dan James"},       # same name, other club
          {"player_id": 604, "team_id": 20, "player_name": "Kane Wilson"},
          {"player_id": 605, "team_id": 20, "player_name": "Kyle Wilson"},
          {"player_id": None, "team_id": 20, "player_name": "Unknown Kid"}]


def test_player_id_mapping_refuses_every_unproven_name():
    names = ["Joel Piroe", "JOËL  PIROE", "P. Bamford", "Dan James", "K. Wilson", "Nobody Known",
             "Unknown Kid", "Joel"]
    m = CP.map_player_names(names, ROSTER)
    assert m["Joel Piroe"]["player_id"] == 501 and m["Joel Piroe"]["match_method"] == "exact"
    assert m["JOËL  PIROE"]["player_id"] == 501                       # accent / case / space only
    assert m["P. Bamford"]["player_id"] == 502 and m["P. Bamford"]["match_method"] == "initial_surname"
    assert m["P. Bamford"]["team_id"] == 10
    for n, why in (("Dan James", "ambiguous"), ("K. Wilson", "ambiguous"), ("Nobody Known", "unmatched"),
                   ("Unknown Kid", "roster_entry_without_id"), ("Joel", "unmatched")):
        assert m[n]["player_id"] is None and m[n]["status"] == why, n


def test_initial_surname_is_refused_when_two_quoted_names_share_it():
    roster = [{"player_id": 1, "team_id": 10, "player_name": "Jack Smith"}]
    m = CP.map_player_names(["J. Smith", "John Smith"], roster)
    assert m["J. Smith"]["player_id"] is None and m["John Smith"]["player_id"] is None


def test_odds_names_get_ids_from_the_fixtures_own_lineup_after_the_last_window(tmp_path):
    _timeline(tmp_path, start_min=125, end_min=-5)
    ids = CP.read_table("prop_player_ids", tmp_path / "prematch").set_index("player_name")
    assert ids.loc["Joel Piroe", "player_id"] == 501 and ids.loc["Joel Piroe", "team_id"] == 10
    assert ids.loc["P. Bamford", "player_id"] == 502
    assert ids.loc["P. Bamford", "match_method"] == "initial_surname"
    assert ids.loc["Jerry Yates", "player_id"] == 601
    assert pd.isna(ids.loc["Nobody Known", "player_id"]) and ids.loc["Nobody Known", "status"] == "unmatched"
    assert set(ids["roster_slot"]) == {"lineups_45"}


def test_no_lineup_means_no_player_id(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_lineups_body", lambda: {"errors": [], "response": []})
    _timeline(tmp_path, start_min=125, end_min=-5)
    ids = CP.read_table("prop_player_ids", tmp_path / "prematch")
    assert len(ids) == 4 and ids["player_id"].isna().all() and set(ids["status"]) == {"no_roster"}


# -- 5. budget ---------------------------------------------------------------------------------------

def test_budget_journal_and_floor_are_this_jobs_own():
    assert CP.BUDGET_FILE != CC.BUDGET_FILE and CP.BUDGET_FILE.parent.name == "prematch"
    assert CP.FLOOR == CC.FLOOR - CP.DAILY_CAP and CP.FLOOR < CC.FLOOR
    assert 0 < CP.DAILY_CAP <= 1_000


def test_the_daily_cap_stops_the_pass_cleanly_and_keeps_what_was_fetched(tmp_path):
    clock = Clock(KO - timedelta(minutes=118))
    sess = _Session(clock)
    st, _ = _pass(tmp_path, clock, sess=sess, client=_client(tmp_path, sess, cap=1))
    assert "budget_stop" in st and len(sess.calls) == 1               # the list only; no injuries call
    assert CP.read_table("lists", tmp_path / "prematch")["n_registry"].tolist() == [3]
    assert CP.read_table("slots", tmp_path / "prematch").empty
    budget = (tmp_path / "prematch" / "af_budget.json").read_text()
    assert ": 1" in budget


def test_the_account_floor_stops_before_the_call(tmp_path):
    clock = Clock(KO - timedelta(minutes=118))
    sess = _Session(clock, remaining=str(CP.FLOOR - 1))
    st, _ = _pass(tmp_path, clock, sess=sess)
    assert "account floor" in st["budget_stop"] and len(sess.calls) == 1   # the list answer said < floor


def test_the_big_jobs_budget_journal_is_never_touched(tmp_path, monkeypatch):
    big = tmp_path / "collect" / "af_budget.json"
    big.parent.mkdir(parents=True)
    big.write_text('{"2026-11-14": 12345}')
    monkeypatch.setattr(CC, "BUDGET_FILE", big)
    _timeline(tmp_path, start_min=125, end_min=100)
    assert big.read_text() == '{"2026-11-14": 12345}'


# -- 6. the switch is off by default -------------------------------------------------------------------

@pytest.mark.parametrize("val", [None, "", "false", "True", "1", "yes"])
def test_the_script_makes_no_call_unless_the_switch_is_exactly_true(tmp_path, monkeypatch, val):
    if val is None:
        monkeypatch.delenv(CP.ENABLE_VAR, raising=False)
    else:
        monkeypatch.setenv(CP.ENABLE_VAR, val)
    clock = Clock(KO - timedelta(minutes=118))
    sess = _Session(clock)
    st = CP.run_pass(client=_client(tmp_path, sess), root=tmp_path / "prematch", now_fn=clock,
                     leagues=LEAGUES, tag="20261114T1000Z-1")
    assert st.get("switched_off") is True and sess.calls == [] and not (tmp_path / "prematch").exists()


def test_the_workflow_is_gated_on_the_repository_variable():
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "prematch_capture.yml").read_text(encoding="utf-8"))
    job = doc["jobs"]["capture"]
    assert "vars.PREMATCH_CAPTURE_ENABLED == 'true'" in job["if"]
    steps = {s.get("name"): s for s in job["steps"]}
    loop = steps["Capture loop (a pass every 5 minutes for ~175 minutes)"]
    assert loop["if"].startswith("vars.PREMATCH_CAPTURE_ENABLED == 'true'")
    assert loop["env"]["PREMATCH_CAPTURE_ENABLED"] == "${{ vars.PREMATCH_CAPTURE_ENABLED }}"
    for name, s in steps.items():                                   # nothing else gets the key
        if "APIFOOTBALL_KEY" in (s.get("env") or {}):
            assert name == loop.get("name")
    assert doc["concurrency"] == {"group": "prematch-capture", "cancel-in-progress": False}
    on = doc[True]                                                    # PyYAML reads `on:` as True
    assert "workflow_dispatch" in on and on["workflow_dispatch"]["inputs"]["mode"]["default"] == "dry-run"
    crons = [c["cron"] for c in on["schedule"]]
    assert crons == ["7 */3 * * *"]                                   # <= 8 firings/day (delivered)
    txt = (ROOT / ".github" / "workflows" / "prematch_capture.yml").read_text(encoding="utf-8")
    assert "git pull -q --rebase --autostash -X theirs" in txt and "git add -f output/prematch" in txt
    assert "collect_trigger" not in txt and "data-truth-100" not in txt


def test_the_big_collection_does_not_share_the_concurrency_group_or_files():
    big = yaml.safe_load((ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8"))
    assert big["concurrency"]["group"] != "prematch-capture"
    assert "output/prematch" not in (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")


# -- 7. storage ------------------------------------------------------------------------------------

def test_chunks_are_partitioned_by_month_and_roll_before_they_grow(tmp_path, monkeypatch):
    _timeline(tmp_path, start_min=125, end_min=100)
    files = sorted(p.relative_to(tmp_path / "prematch").as_posix()
                   for p in (tmp_path / "prematch").rglob("*.parquet"))
    assert "slots/2026-11/20261114T1000Z-1.parquet" in files
    assert all(f.split("/")[1] == "2026-11" for f in files)
    monkeypatch.setattr(CP, "ROLL_BYTES", 1)
    CP._append_chunk(tmp_path / "prematch", "lists", "2026-11", "20261114T1000Z-1",
                     [{"list_date": "2026-11-15", "captured_at_utc": "2026-11-14T13:00:00+00:00"}])
    assert (tmp_path / "prematch" / "lists" / "2026-11" / "20261114T1000Z-1_p2.parquet").exists()
    assert len(CP.read_table("lists", tmp_path / "prematch")) == 2   # 1 list read + the rolled row


def test_an_all_missing_column_keeps_its_type_across_chunks(tmp_path):
    root = tmp_path / "prematch"
    base = {"fixture_id": 1, "slot": "odds_15", "captured_at_utc": "2026-11-14T14:50:00+00:00"}
    CP._append_chunk(root, "prop_odds", "2026-11", "20261114T1000Z-1", [{**base, "row": 0, "odd": None}])
    CP._append_chunk(root, "prop_odds", "2026-11", "20261114T1300Z-2", [{**base, "row": 1, "odd": 2.5}])
    d = pd.read_parquet(root / "prop_odds" / "2026-11")               # a reader can take the directory
    assert len(d) == 2 and d["odd"].isna().sum() == 1


# -- 8. dry run ------------------------------------------------------------------------------------

def test_dry_run_counts_real_fixtures_and_makes_no_call(monkeypatch):
    monkeypatch.setattr(CP.AL, "registry_af_ids", lambda: {REG})
    fx = pd.DataFrame([{"fixture_id": i, "league_id": REG if i < 5 else OTHER, "season": "2026",
                        "date": "2026-11-01" if i < 3 else "2026-11-02", "home_id": 1, "away_id": 2,
                        "status": "FT"} for i in range(7)])
    est = CP.estimate(days=2, end="2026-11-03", fx=fx)
    assert est["per_day"] == {"2026-11-01": 3, "2026-11-02": 2}
    lc = CP.list_calls_per_day()
    assert est["calls_per_day_max"] == 3 * 4 + lc and lc == est["list_calls_per_day"]
    assert 24 <= lc <= 30
