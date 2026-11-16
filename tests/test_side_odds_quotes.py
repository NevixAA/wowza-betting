"""Corners/cards quotes (src/side_odds_quotes.py), owner 2026-11-16 (audit gap G1).

Kept from the /odds answers the captures already make (0 extra calls), real quotes only, pre-kickoff only,
written ONLY to the private store (marker + outside the code checkout), never to a public path or log.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import side_odds_quotes as SO

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=5)


def _entry(fid=1001, ko=KO, lid=40):
    return {
        "league": {"id": lid, "name": "Championship", "country": "England", "season": 2026},
        "fixture": {"id": fid, "timestamp": int(ko.timestamp()), "date": ko.isoformat()},
        "update": "2026-11-16T11:55:00+00:00",
        "bookmakers": [
            {"id": 8, "name": "Bet365", "bets": [
                {"id": 45, "name": "Corners Over Under", "values": [
                    {"value": "Over 9.5", "odd": "1.90"}, {"value": "Under 9.5", "odd": "1.90"},
                    {"value": "Over 10.25", "odd": "2.05"}, {"value": "Exactly 10", "odd": "9.00"}]},
                {"id": 57, "name": "Home Corners Over/Under", "values": [
                    {"value": "Over 4.5", "odd": "1.80"}, {"value": "Under 4.5", "odd": "N/A"}]},
                {"id": 58, "name": "Away Corners Over/Under", "values": [{"value": "Over 3.0", "odd": "1.70"}]},
                {"id": 80, "name": "Cards Over/Under", "values": [
                    {"value": "Over 3.5", "odd": "1.85"}, {"value": "Under 3.5", "odd": "1.00"}]},
                {"id": 150, "name": "Home Team Yellow Cards", "values": [{"value": "Over 1.5", "odd": "1.60"}]},
                {"id": 335, "name": "Red Cards Over/Under", "values": [{"value": "Over 0.5", "odd": "4.50"}]},
                # not kept: first half, handicap, 1x2, a renamed id, goals
                {"id": 77, "name": "Total Corners (1st Half)", "values": [{"value": "Over 4.5", "odd": "2.00"}]},
                {"id": 56, "name": "Corners Asian Handicap", "values": [{"value": "Home -1", "odd": "1.90"}]},
                {"id": 55, "name": "Corners 1x2", "values": [{"value": "Home", "odd": "1.50"}]},
                {"id": 45, "name": "Corners Over/Under (renamed)", "values": [{"value": "Over 8.5", "odd": "1.50"}]},
                {"id": 5, "name": "Goals Over/Under", "values": [{"value": "Over 2.5", "odd": "1.90"}]}]},
            {"id": 11, "name": "1xBet", "bets": [
                {"id": 45, "name": "Corners Over Under", "values": [{"value": "Over 9.5", "odd": "1.95"}]},
                {"id": 83, "name": "Away Team Total Cards", "values": [{"value": "Under 2.5", "odd": "1.75"}]}]},
            {"name": "no id", "bets": [
                {"id": 45, "name": "Corners Over Under", "values": [{"value": "Over 9.5", "odd": "1.99"}]}]},
        ]}


KEPT = {("Bet365", 45, "Over 9.5", 1.9), ("Bet365", 45, "Under 9.5", 1.9), ("Bet365", 45, "Over 10.25", 2.05),
        ("Bet365", 57, "Over 4.5", 1.8), ("Bet365", 58, "Over 3.0", 1.7), ("Bet365", 80, "Over 3.5", 1.85),
        ("Bet365", 150, "Over 1.5", 1.6), ("Bet365", 335, "Over 0.5", 4.5),
        ("1xBet", 45, "Over 9.5", 1.95), ("1xBet", 83, "Under 2.5", 1.75)}


def _parse(resp=None, **kw):
    a = dict(capture="std", captured_at=NOW, names={1001: ("Leeds", "Hull")},
             leagues={40: ("Championship", None)}, expect_fixture_id=1001)
    a.update(kw)
    return SO.parse(resp if resp is not None else [_entry()], **a)


def test_keeps_exactly_full_time_corners_and_cards_over_under():
    rows = _parse()
    assert {(r["bookmaker"], r["bet_id"], r["selection"], r["odds"]) for r in rows} == KEPT


def test_row_fields():
    r = next(x for x in _parse() if x["bet_id"] == 57)
    assert r["snapshot_ts"] == "2026-11-16T12:00:00Z" and r["kickoff_utc"] == "2026-11-16T17:00:00Z"
    assert (r["af_fixture_id"], r["league_id"], r["league"], r["country"]) == (1001, 40, "Championship", "England")
    assert (r["home_team"], r["away_team"]) == ("Leeds", "Hull")
    assert (r["market"], r["team"], r["line"], r["side"]) == ("corners_ou_home", "home", 4.5, "over")
    assert r["minutes_to_kickoff"] == 300.0 and r["ladder_band"] == "T-6h"
    assert set(r) == set(SO.COLUMNS) - {"obs_reason"}


def test_markets_per_bet_id():
    assert {r["bet_id"]: r["market"] for r in _parse()} == {
        45: "corners_ou_match", 57: "corners_ou_home", 58: "corners_ou_away", 80: "cards_ou_match",
        83: "cards_ou_away", 150: "yellow_cards_ou_home", 335: "red_cards_ou_match"}


def test_no_invented_values():
    rows = _parse(names={})                          # names unknown -> missing, never guessed
    assert all(r["home_team"] is None and r["away_team"] is None for r in rows)
    assert all(r["odds"] > 1.0 for r in rows)        # 'N/A' and 1.00 are not prices


def test_pre_kickoff_only_and_unknown_kickoff_dropped():
    assert _parse(captured_at=KO) == []
    assert _parse(captured_at=KO + timedelta(minutes=1)) == []
    e = _entry()
    e["fixture"] = {"id": 1001}
    assert _parse(resp=[e]) == []                    # no kickoff anywhere
    assert len(_parse(resp=[e], kickoffs={1001: KO.isoformat()})) == len(KEPT)


def test_answer_for_another_fixture_is_ignored():
    assert _parse(resp=[_entry(fid=999)]) == []


def test_unknown_capture_rejected():
    with pytest.raises(ValueError):
        _parse(capture="prematch_75")


def test_unkept_side_bets_are_reported_by_name_only():
    SO.UNKEPT.clear()
    rows = _parse()
    line = SO.summary(rows, len(rows))
    assert "Total Corners (1st Half)" in line and "Corners Over/Under (renamed)" in line
    assert "1.9" not in line and "4.5" not in line   # no price (and no line value) in the log line
    assert "0 extra API calls" in line


# -- the private store ---------------------------------------------------------------------------

def _store(tmp_path):
    d = tmp_path / "store"
    d.mkdir()
    (d / SO.MARKER).write_text("private", encoding="utf-8")
    return d


def test_store_off_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv(SO.ENV, raising=False)
    assert SO.store_dir() is None
    assert SO.append(_parse()) == 0


def test_store_refuses_without_marker(tmp_path):
    with pytest.raises(SO.StoreRefused):
        SO.append(_parse(), base=tmp_path)
    assert not any(tmp_path.rglob("*.csv"))


def test_store_refuses_inside_the_code_checkout():
    inside = ROOT / "output"
    marker = inside / SO.MARKER
    created = not marker.exists()
    marker.write_text("x", encoding="utf-8")
    try:
        with pytest.raises(SO.StoreRefused):
            SO.store_dir(inside)
    finally:
        if created:
            marker.unlink()


def test_append_open_change_heartbeat_append_only(tmp_path):
    SO._STATE.clear()
    base = _store(tmp_path)
    assert SO.append(_parse(), base=base) == len(KEPT)
    assert SO.append(_parse(captured_at=NOW + timedelta(minutes=10)), base=base) == 0     # nothing moved
    e = _entry()
    e["bookmakers"][0]["bets"][0]["values"][0]["odd"] = "1.80"                          # one price moves
    assert SO.append(_parse(resp=[e], captured_at=NOW + timedelta(minutes=20)), base=base) == 1
    near = NOW + timedelta(hours=4, minutes=35)                                          # T-25m rung
    assert SO.append(_parse(resp=[e], captured_at=near), base=base) == len(KEPT)          # heartbeat
    files = list((base / "data").rglob("*.csv"))
    assert [f.name for f in files] == ["2026-11-16_std.csv"]
    d = pd.read_csv(files[0])
    assert list(d.columns) == SO.COLUMNS
    assert d["obs_reason"].value_counts().to_dict() == {"open": len(KEPT), "heartbeat": len(KEPT), "change": 1}
    # a new process (empty state) resumes from the file, not from scratch
    SO._STATE.clear()
    assert SO.append(_parse(resp=[e], captured_at=near + timedelta(minutes=1)), base=base) == 0


def test_one_file_per_producer(tmp_path):
    SO._STATE.clear()
    base = _store(tmp_path)
    SO.append(_parse(capture="std") + _parse(capture="nf"), base=base)
    SO.append(_parse(capture="scout_sweep", names={}, leagues={}, expect_fixture_id=None), base=base)
    names = sorted(p.name for p in (base / "data").rglob("*.csv"))
    assert names == ["2026-11-16_nf.csv", "2026-11-16_std.csv", "2026-11-16_sweep.csv"]


# -- the capture scripts: 0 extra calls, existing outputs byte-for-byte unchanged --------------------

def _load_script(name):
    spec = importlib.util.spec_from_file_location(name + "_so", ROOT / "scripts" / f"{name}.py")
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


def _run_capture(monkeypatch, tmp_path, script, league, ko, store):
    import config
    import src.book_quotes as BQ
    import src.season_resolver as SR
    import src.specials_quotes as SQ
    mod = _load_script(script)
    fixtures = [{"fixture": {"id": f, "date": ko.isoformat()},
                 "teams": {"home": {"name": h}, "away": {"name": a}}}
                for f, h, a in ((1001, "Leeds", "Hull"), (1002, "Stoke", "Derby"))]
    lid = config.API_FOOTBALL_IDS[league]
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append((url, dict(params)))
        e = _entry(fid=params["fixture"], ko=ko, lid=lid)
        return _Resp({"response": [e]})

    monkeypatch.setattr(mod, "_KEY", "test")
    monkeypatch.setattr(mod, "CAPTURE_BOOK_QUOTES", True)
    monkeypatch.setattr(mod, "OUT", tmp_path / "history.csv")
    monkeypatch.setattr(mod, "get_upcoming_fixtures", lambda l, s, next_n=0: fixtures if l == lid else [])
    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod.odds_heartbeat, "record", lambda *a, **k: None)
    monkeypatch.setattr(SR, "resolve_api_season", lambda *a, **k: "2026")
    monkeypatch.setattr(BQ, "QUOTES_DIR", tmp_path / "book_quotes")
    monkeypatch.setattr(BQ, "LEGACY_FILE", tmp_path / "book_quotes.csv")
    monkeypatch.setattr(SQ, "SPECIALS_DIR", tmp_path / "specials_quotes")
    if store is None:
        monkeypatch.delenv(SO.ENV, raising=False)
    else:
        monkeypatch.setenv(SO.ENV, str(store))
    SO._STATE.clear()
    mod.run()
    return calls


@pytest.mark.parametrize("script,league,capture", [
    ("capture_std_sidemarket_odds_forward", "Championship", "std"),
    ("capture_nf_odds_forward", "USA MLS", "nf"),
])
def test_capture_keeps_side_odds_with_zero_extra_calls(monkeypatch, tmp_path, capsys, script, league, capture):
    on, off = tmp_path / "on", tmp_path / "off"
    on.mkdir(); off.mkdir()
    store = _store(tmp_path)
    ko = (datetime.now(timezone.utc) + timedelta(hours=3)).replace(microsecond=0)
    calls_on = _run_capture(monkeypatch, on, script, league, ko, store)
    out = capsys.readouterr().out
    calls_off = _run_capture(monkeypatch, off, script, league, ko, None)
    assert calls_on == calls_off                       # 0 extra calls
    assert [c[1]["fixture"] for c in calls_on] == [1001, 1002]
    h_on = pd.read_csv(on / "history.csv").drop(columns=["snapshot_ts", "snapshot_date"])
    h_off = pd.read_csv(off / "history.csv").drop(columns=["snapshot_ts", "snapshot_date"])
    pd.testing.assert_frame_equal(h_on, h_off)         # the public archive is unchanged
    for sub in ("specials_quotes", "book_quotes"):     # nothing corners/cards in a public output
        for f in (on / sub).rglob("*.csv"):
            assert "Corners" not in f.read_text(encoding="utf-8")
    files = list((store / "data").rglob(f"*_{capture}.csv"))
    assert len(files) == 1
    d = pd.read_csv(files[0])
    assert set(d["capture"]) == {capture} and set(d["league"]) == {league}
    assert sorted(d["af_fixture_id"].unique()) == [1001, 1002]
    assert len(d) == 2 * len(KEPT)
    assert "[side_odds] +20 row(s)" in out and "odds calls this pass: 2" in out
    assert "1.85" not in out and "1.95" not in out     # no price in the log


def test_workflows_gate_and_push_the_private_store():
    for wf, prod in (("std_odds_capture.yml", "std"), ("nf_odds_capture.yml", "nf")):
        t = (ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8")
        assert t.count("vars.SIDE_ODDS_ENABLED == 'true'") == 2
        assert "bash scripts/side_odds_store.sh open" in t
        assert f"bash scripts/side_odds_store.sh push {prod}" in t
        assert "secrets.SIDE_ODDS_STORE_KEY" in t
        assert "side_odds" not in t.split("Commit and push")[1].split("Push corners/cards")[0]


def test_shared_file_markers():
    t = (ROOT / "src" / "side_odds_quotes.py").read_text(encoding="utf-8")
    assert "from src" not in t and "import config" not in t   # self-contained: shared verbatim with Pro


#: sha256 (LF line endings) of the shared file; Pro's tests pin the same value for src/scout/side_odds.py
SHARED_SHA256 = "058316def2ba7bb68b37707253ad30baf3b5cfd34a613b5867dbb897baf41491"


def test_shared_file_is_the_pinned_version():
    import hashlib
    b = (ROOT / "src" / "side_odds_quotes.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(b).hexdigest() == SHARED_SHA256
