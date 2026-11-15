"""Special-market quotes (src/specials_quotes.py): kept from the /odds answers the captures already make.

Owner 2026-11-10: collect correct-score odds with the live collection, zero extra calls, real quotes only.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import specials_quotes as SQ

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=5)


def _payload(fid=1001, ko=KO, update="2026-11-10T11:55:00+00:00"):
    return {"response": [{
        "fixture": {"id": fid, "timestamp": int(ko.timestamp()), "date": ko.isoformat()},
        "update": update,
        "bookmakers": [
            {"id": 8, "name": "Bet365", "bets": [
                {"id": 5, "name": "Goals Over/Under", "values": [
                    {"value": "Over 0.5", "odd": "1.06"}, {"value": "Under 0.5", "odd": "10.00"},
                    {"value": "Over 1.5", "odd": "1.30"}, {"value": "Over 2.5", "odd": "1.90"},
                    {"value": "Over 4.5", "odd": "6.50"}, {"value": "Under 4.5", "odd": "1.10"},
                    {"value": "Over 5.5", "odd": "13.00"}, {"value": "Under 5.5", "odd": "1.02"},
                    {"value": "Over 6.5", "odd": "26.00"}]},
                {"id": 6, "name": "Goals Over/Under First Half", "values": [
                    {"value": "Over 0.5", "odd": "1.40"}, {"value": "Under 0.5", "odd": "2.80"}]},
                {"id": 26, "name": "Goals Over/Under - Second Half", "values": [
                    {"value": "Over 4.5", "odd": "21.00"}]},
                {"id": 10, "name": "Exact Score", "values": [
                    {"value": "1:0", "odd": "7.00"}, {"value": "0:0", "odd": "11.00"},
                    {"value": "4:3", "odd": "151.00"}, {"value": "2:1", "odd": "N/A"}]},
                {"id": 31, "name": "Correct Score - First Half", "values": [{"value": "0:0", "odd": "3.10"}]},
                {"id": 62, "name": "Exact Score - First Half", "values": [{"value": "1:0", "odd": "3.60"}]},
                {"id": 16, "name": "Total - Home", "values": [
                    {"value": "Over 3.5", "odd": "17.00"}, {"value": "Under 3.5", "odd": "1.01"},
                    {"value": "Over 1.5", "odd": "2.40"}]},
                {"id": 17, "name": "Total - Away", "values": [{"value": "Over 3.5", "odd": "34.00"}]},
                {"id": 105, "name": "Total - Home First Half", "values": [{"value": "Over 3.5", "odd": "99.00"}]},
                {"id": 8, "name": "Both Teams Score", "values": [
                    {"value": "Yes", "odd": "1.80"}, {"value": "No", "odd": "1.95"}]},
                {"id": 1, "name": "Match Winner", "values": [
                    {"value": "Home", "odd": "2.10"}, {"value": "Draw", "odd": "3.30"},
                    {"value": "Away", "odd": "3.60"}]}]},
            {"id": 11, "name": "1xBet", "bets": [
                {"id": 10, "name": "Exact Score", "values": [
                    {"value": "1:0", "odd": "6.80"}, {"value": "0:0", "odd": "1.00"}]},
                {"id": 5, "name": "Goals Over/Under", "values": [
                    {"value": "Over 2.5", "odd": "1.95"}, {"value": "Under 2.5", "odd": "1.92"}]}]},
        ]}]}


def _parse(**kw):
    a = dict(capture="std", af_fixture_id=1001, league="Championship", kickoff_utc="",
             home="Leeds", away="Hull", captured_at=NOW)
    a.update(kw)
    p = a.pop("payload", _payload())
    return SQ.parse_specials(p, **a)


def test_keeps_exactly_the_requested_full_time_quotes():
    rows = _parse()
    got = {(r["bookmaker"], r["bet"], r["selection"], r["odds"]) for r in rows}
    assert got == {
        ("Bet365", "Goals Over/Under", "Over 0.5", 1.06), ("Bet365", "Goals Over/Under", "Under 0.5", 10.0),
        ("Bet365", "Goals Over/Under", "Over 4.5", 6.5), ("Bet365", "Goals Over/Under", "Under 4.5", 1.1),
        ("Bet365", "Goals Over/Under", "Over 5.5", 13.0), ("Bet365", "Goals Over/Under", "Under 5.5", 1.02),
        ("Bet365", "Goals Over/Under", "Over 6.5", 26.0),
        ("Bet365", "Exact Score", "1:0", 7.0), ("Bet365", "Exact Score", "0:0", 11.0),
        ("Bet365", "Exact Score", "4:3", 151.0),
        ("Bet365", "Total - Home", "Over 3.5", 17.0), ("Bet365", "Total - Home", "Under 3.5", 1.01),
        ("Bet365", "Total - Away", "Over 3.5", 34.0),
        ("1xBet", "Exact Score", "1:0", 6.8),
    }


def test_first_and_second_half_bets_never_enter():
    rows = _parse()
    assert not any("Half" in r["bet"] for r in rows)
    # the first-half 1:0 (3.60) and second-half Over 4.5 (21.00) never overwrite the full-time ones
    assert [r["odds"] for r in rows if r["bet"] == "Exact Score" and r["selection"] == "1:0"
            and r["bookmaker"] == "Bet365"] == [7.0]
    assert [r["odds"] for r in rows if r["selection"] == "Over 4.5"] == [6.5]


def test_no_invented_quotes():
    rows = _parse()
    # 2:1 "N/A" and 1xBet 0:0 at 1.00 are not prices; Over 1.5 / 2.5 live in book_quotes
    assert not any(r["selection"] == "2:1" for r in rows)
    assert not any(r["bookmaker"] == "1xBet" and r["selection"] == "0:0" for r in rows)
    assert not any(r["selection"] in ("Over 1.5", "Over 2.5", "Under 2.5") for r in rows)


def test_row_identity_fields():
    r = _parse()[0]
    assert r["captured_at"] == "2026-11-10T12:00:00Z"
    assert r["provider_update"] == "2026-11-10T11:55:00Z"
    assert r["af_fixture_id"] == 1001 and r["kickoff_utc"] == "2026-11-10T17:00:00Z"
    assert r["fixture_key"] == "Championship|2026-11-10|Leeds|Hull"
    assert r["capture"] == "std" and r["ladder_band"] == "T-6h" and r["minutes_to_kickoff"] == 300.0
    assert set(SQ.COLUMNS) - {"obs_reason"} <= set(r)


def test_pre_kickoff_only():
    assert _parse(captured_at=KO) == []
    assert _parse(captured_at=KO + timedelta(minutes=1)) == []
    # unknown kickoff: not stored
    p = _payload()
    p["response"][0]["fixture"] = {"id": 1001}
    assert _parse(payload=p) == []
    # the answer's own kickoff wins over the capture's
    assert _parse(kickoff_utc=(NOW - timedelta(hours=1)).isoformat()) != []


def test_answer_for_another_fixture_is_ignored():
    assert _parse(af_fixture_id=999) == []


def test_unknown_capture_rejected():
    with pytest.raises(ValueError):
        _parse(capture="other")


def test_append_open_change_heartbeat_and_append_only(tmp_path):
    first = _parse()
    assert SQ.append_specials(first, base=tmp_path) == len(first)
    f = tmp_path / "2026-11" / "2026-11-10_std.csv"
    assert f.exists()
    before = f.read_text(encoding="utf-8")
    # same prices again, still FAR: nothing new
    assert SQ.append_specials(_parse(captured_at=NOW + timedelta(minutes=30)), base=tmp_path) == 0
    # one price moves
    p = _payload()
    p["response"][0]["bookmakers"][0]["bets"][3]["values"][0]["odd"] = "7.50"
    assert SQ.append_specials(_parse(payload=p, captured_at=NOW + timedelta(hours=1)), base=tmp_path) == 1
    # at T-30m every unchanged quote gets ONE heartbeat
    t30 = KO - timedelta(minutes=25)
    n = SQ.append_specials(_parse(payload=p, captured_at=t30), base=tmp_path)
    assert n == len(first)
    assert SQ.append_specials(_parse(payload=p, captured_at=t30 + timedelta(minutes=5)), base=tmp_path) == 0
    d = pd.read_csv(f)
    assert f.read_text(encoding="utf-8").startswith(before)          # append-only
    assert d["obs_reason"].value_counts().to_dict() == {"open": len(first), "change": 1,
                                                         "heartbeat": len(first)}
    assert set(d.columns) == set(SQ.COLUMNS)


def test_one_file_per_producer(tmp_path):
    SQ.append_specials(_parse(capture="nf"), base=tmp_path)
    SQ.append_specials(_parse(capture="prematch_75"), base=tmp_path)
    SQ.append_specials(_parse(capture="prematch_15"), base=tmp_path)
    names = sorted(p.name for p in tmp_path.rglob("*.csv"))
    assert names == ["2026-11-10_nf.csv", "2026-11-10_prematch.csv"]
    d = pd.read_csv(tmp_path / "2026-11" / "2026-11-10_prematch.csv")
    # each prematch slot is its own capture: both kept in full
    assert d.groupby("capture").size().to_dict() == {"prematch_15": 14, "prematch_75": 14}


# -- mock HTTP: the capture scripts write the sidecar with ZERO extra calls and unchanged outputs ------

def _load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_t", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    status_code = 200
    headers = {"x-ratelimit-requests-remaining": "60000"}

    def __init__(self, body):
        self._b = body

    def json(self):
        return json.loads(json.dumps(self._b))


def _run_capture(monkeypatch, tmp_path, script, league, with_specials=True):
    import src.book_quotes as BQ
    import src.season_resolver as SR
    mod = _load_script(script)
    now = datetime.now(timezone.utc)
    ko = (now + timedelta(hours=3)).replace(microsecond=0)
    fixtures = [{"fixture": {"id": 1001, "date": ko.isoformat()},
                 "teams": {"home": {"name": "Leeds"}, "away": {"name": "Hull"}}},
                {"fixture": {"id": 1002, "date": ko.isoformat()},
                 "teams": {"home": {"name": "Stoke"}, "away": {"name": "Derby"}}}]
    import config
    lid = config.API_FOOTBALL_IDS[league]
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append((url, dict(params)))
        return _Resp(_payload(fid=params["fixture"], ko=ko))

    monkeypatch.setattr(mod, "_KEY", "test")
    monkeypatch.setattr(mod, "CAPTURE_BOOK_QUOTES", True)
    monkeypatch.setattr(mod, "OUT", tmp_path / "history.csv")
    monkeypatch.setattr(mod, "get_upcoming_fixtures",
                        lambda l, s, next_n=0: fixtures if l == lid else [])
    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod.odds_heartbeat, "record", lambda *a, **k: None)
    monkeypatch.setattr(SR, "resolve_api_season", lambda *a, **k: "2026")
    monkeypatch.setattr(BQ, "QUOTES_DIR", tmp_path / "book_quotes")
    monkeypatch.setattr(BQ, "LEGACY_FILE", tmp_path / "book_quotes.csv")
    monkeypatch.setattr(SQ, "SPECIALS_DIR", tmp_path / "specials_quotes")
    if not with_specials:
        def boom(*a, **k):
            raise RuntimeError("specials disabled for the comparison run")
        monkeypatch.setattr(SQ, "parse_specials", boom)
    mod.run()
    return calls


@pytest.mark.parametrize("script,league,capture", [
    ("capture_std_sidemarket_odds_forward", "Championship", "std"),
    ("capture_nf_odds_forward", "USA MLS", "nf"),
])
def test_capture_writes_sidecar_with_zero_extra_calls(monkeypatch, tmp_path, script, league, capture):
    on, off = tmp_path / "on", tmp_path / "off"
    calls_on = _run_capture(monkeypatch, on, script, league, with_specials=True)
    calls_off = _run_capture(monkeypatch, off, script, league, with_specials=False)
    # zero extra calls: exactly one /odds request per fixture, identical with or without specials
    assert calls_on == calls_off
    assert [c[1]["fixture"] for c in calls_on] == [1001, 1002]
    # existing outputs unchanged (snapshot times differ by run, so compare everything else)
    h_on = pd.read_csv(on / "history.csv").drop(columns=["snapshot_ts", "snapshot_date"])
    h_off = pd.read_csv(off / "history.csv").drop(columns=["snapshot_ts", "snapshot_date"])
    pd.testing.assert_frame_equal(h_on, h_off)
    drop = ["snapshot_ts", "minutes_to_kickoff", "ladder_band"]
    b_on = pd.concat(pd.read_csv(f) for f in sorted((on / "book_quotes").glob("*.csv"))).drop(columns=drop)
    b_off = pd.concat(pd.read_csv(f) for f in sorted((off / "book_quotes").glob("*.csv"))).drop(columns=drop)
    pd.testing.assert_frame_equal(b_on.reset_index(drop=True), b_off.reset_index(drop=True))
    assert not (off / "specials_quotes").exists()
    files = list((on / "specials_quotes").rglob(f"*_{capture}.csv"))
    assert len(files) == 1
    d = pd.read_csv(files[0])
    assert set(d["capture"]) == {capture} and set(d["league"]) == {league}
    assert sorted(d["af_fixture_id"].unique()) == [1001, 1002]
    assert (d[d["bet"] == "Exact Score"].groupby("af_fixture_id")["selection"].nunique() == 3).all()
    assert len(d) == 2 * 14


def test_workflows_stage_the_sidecar():
    for wf in ("std_odds_capture.yml", "nf_odds_capture.yml"):
        assert "git add -f output/specials_quotes" in (ROOT / ".github" / "workflows" / wf).read_text(
            encoding="utf-8")
