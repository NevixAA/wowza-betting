"""OddsAPI per-book near-kickoff close capture (owner 2026-11-23): floor + cap guards, bands, retries, the
private-store rule, and the entry bookmaker recorded on predict / ledger / shadow rows."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import config
from src import oddsapi_book_close as BC

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cap_bc", ROOT / "scripts" / "capture_oddsapi_book_close.py")
CAP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CAP)

NOW = datetime(2026, 11, 28, 14, 40, tzinfo=timezone.utc)
KO = datetime(2026, 11, 28, 15, 0, tzinfo=timezone.utc)          # 20 min ahead -> CLOSE_30


def _budget(**kw):
    a = dict(floor=260_000, run_cap=20, day_cap=100, month_cap=1000)
    a.update(kw)
    return BC.Budget(**a)


# ── budget: floor and caps ─────────────────────────────────────────────────────────────────────────
def test_floor_below_the_reserve_is_refused():
    with pytest.raises(ValueError):
        _budget(floor=259_999)
    assert BC.FLOOR_MIN == 260_000


def test_no_paid_call_while_remaining_is_unknown():
    b = _budget()
    assert not b.can_spend(1, NOW) and "unknown" in b.stopped


@pytest.mark.parametrize("remaining,ok", [(260_000, False), (260_001, True), (5_000_000, True)])
def test_floor_guard_keeps_remaining_at_or_above_260k(remaining, ok):
    b = _budget()
    b.remaining = remaining
    assert b.can_spend(1, NOW) is ok
    if not ok:
        assert b.stopped.startswith("floor")


def test_run_day_and_month_caps():
    b = _budget(run_cap=2, day_cap=3, month_cap=4)
    b.remaining = 600_000
    b.book(2, NOW)
    assert not b.can_spend(1, NOW) and b.stopped == "run cap 2"
    b2 = _budget(run_cap=50, day_cap=3, month_cap=4)
    b2.remaining = 600_000
    b2.usage["by_day"][NOW.strftime("%Y-%m-%d")] = 3
    assert not b2.can_spend(1, NOW) and b2.stopped == "day cap 3"
    b3 = _budget(run_cap=50, day_cap=50, month_cap=4)
    b3.remaining = 600_000
    b3.usage["by_month"][NOW.strftime("%Y-%m")] = 4
    assert not b3.can_spend(1, NOW) and b3.stopped == "month cap 4"


def test_cost_unknown_stops_every_further_paid_call():
    b = _budget()
    b.remaining = 600_000
    assert b.read_headers({"x-requests-remaining": "599999"}) is None
    b.cost_unknown = True
    assert not b.can_spend(1, NOW)


def test_usage_journal_persists_across_runs(tmp_path):
    p = tmp_path / "ledger" / "usage.json"
    b = _budget()
    b.book(7, NOW)
    b.save(p)
    b2 = BC.Budget.load(p, floor=260_000, run_cap=20, day_cap=10, month_cap=1000)
    b2.remaining = 600_000
    assert b2.can_spend(3, NOW) and not b2.can_spend(4, NOW) and b2.stopped == "day cap 10"


# ── bands and parsing ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mins,band", [(31, None), (30, "CLOSE_30"), (10.5, "CLOSE_30"), (10, "CLOSE_10"),
                                       (1.5, "CLOSE_10"), (1, None), (-3, None)])
def test_bands(mins, band):
    assert BC.band_of(mins) == band


def _pred(*kos, sport="soccer_efl_champ", league="Championship"):
    return pd.DataFrame({"league": [league] * len(kos), "_oa_sport_key": [sport] * len(kos),
                         "kickoff_utc": [k.strftime("%Y-%m-%dT%H:%M:%SZ") for k in kos],
                         "home_team": [f"H{i}" for i in range(len(kos))],
                         "away_team": [f"A{i}" for i in range(len(kos))]})


def test_one_slot_per_sport_and_kickoff_and_due_skips_done_bands():
    slots = BC.slots_from_predictions(_pred(KO, KO, KO + timedelta(hours=3)))
    assert len(slots) == 2
    d = BC.due(slots, pd.DataFrame(columns=BC.JOURNAL_COLS), NOW)
    assert [x["band"] for x in d] == ["CLOSE_30"]
    j = pd.DataFrame([{"sport_key": "soccer_efl_champ", "kickoff_utc": BC.iso(KO), "band": "CLOSE_30",
                       "status": "OK"}])
    assert BC.due(slots, j, NOW) == []


def test_quote_rows_drop_suspended_and_inplay_quotes():
    ev = [{"id": "e1", "commence_time": BC.iso(KO), "home_team": "H", "away_team": "A",
           "bookmakers": [{"key": "b1", "markets": [{"key": "totals", "outcomes": [
               {"name": "Over", "point": 2.5, "price": 1.0}, {"name": "Under", "point": 2.5, "price": 1.9}]}]}]},
          {"id": "e2", "commence_time": BC.iso(NOW - timedelta(minutes=5)), "home_team": "X", "away_team": "Y",
           "bookmakers": [{"key": "b1", "markets": [{"key": "totals", "outcomes": [
               {"name": "Over", "point": 2.5, "price": 2.0}]}]}]}]
    rows = BC.quote_rows(ev, league="L", sport_key="s", band="CLOSE_30", captured=NOW)
    assert [(r["bookmaker"], r["side"], r["market"]) for r in rows] == [("b1", "UNDER", "OU25")]


# ── the runner with a fake provider ────────────────────────────────────────────────────────────────
class _R:
    def __init__(self, status=200, payload=None, remaining="600000", last="1"):
        self.status_code = status
        self._p = payload if payload is not None else []
        self.headers = {"x-requests-remaining": remaining}
        if last is not None:
            self.headers["x-requests-last"] = last

    def json(self):
        return self._p


def _event(ko):
    return [{"id": "e1", "commence_time": BC.iso(ko), "home_team": "H", "away_team": "A",
             "bookmakers": [{"key": "bk", "markets": [{"key": "totals", "outcomes": [
                 {"name": "Over", "point": 2.5, "price": 1.91}, {"name": "Under", "point": 2.5, "price": 1.93}]}]}]}]


class _Fake:
    def __init__(self, odds=None, remaining=600_000):
        self.calls, self.remaining, self.odds = [], remaining, odds

    def __call__(self, url, params=None, timeout=None):
        assert params.get("apiKey") == "k"
        self.calls.append(url)
        if url.endswith("/sports") or url.endswith("/events"):
            return _R(remaining=str(self.remaining), last="0")
        if self.odds is not None:
            return self.odds(url, params)
        self.remaining -= 1
        return _R(payload=_event(KO), remaining=str(self.remaining))


def _args(tmp_path, pred, **kw):
    pp = tmp_path / "predictions.csv"
    pred.to_csv(pp, index=False)
    a = dict(store=str(tmp_path / "store"), predictions=str(pp), mode="capture", floor=260_000, run_cap=20,
             day_cap=100, month_cap=1000, max_calls=3, loop_minutes=0, poll_seconds=90.0)
    a.update(kw)
    return SimpleNamespace(**a)


def _clock(start):
    t = {"now": start}

    def now():
        return t["now"]

    def sleep(s):
        t["now"] = t["now"] + timedelta(seconds=s)
    return now, sleep


def test_capture_buys_both_bands_once_and_stores_privately(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    fake = _Fake()
    now, sleep = _clock(NOW)
    out = CAP.run(_args(tmp_path, _pred(KO), loop_minutes=60), get=fake, now=now, sleep=sleep)
    paid = [u for u in fake.calls if u.endswith("/odds")]
    assert len(paid) == 2 and out["credits"] == 2
    j = BC.read_journal(tmp_path / "store")
    assert sorted(j["band"]) == ["CLOSE_10", "CLOSE_30"] and set(j["status"]) == {"OK"}
    files = list((tmp_path / "store" / "data").rglob("*.csv"))
    assert len(files) == 1 and len(pd.read_csv(files[0])) == 4
    log = capsys.readouterr().out
    assert "1.91" not in log and "1.93" not in log and "\"k\"" not in log       # counts only, never a price/key
    usage = json.loads((tmp_path / "store" / "ledger" / "usage.json").read_text())
    assert sum(usage["by_day"].values()) == 2


def test_capture_respects_the_run_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    kos = [KO + timedelta(minutes=0)]
    pred = pd.concat([_pred(*kos, sport=f"s{i}", league=f"L{i}") for i in range(5)], ignore_index=True)
    fake = _Fake()
    now, sleep = _clock(NOW)
    out = CAP.run(_args(tmp_path, pred, run_cap=3), get=fake, now=now, sleep=sleep)
    assert out["credits"] == 3 and out["stopped"] == "run cap 3"
    assert len([u for u in fake.calls if u.endswith("/odds")]) == 3


def test_capture_makes_no_paid_call_at_the_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    fake = _Fake(remaining=260_000)
    now, sleep = _clock(NOW)
    out = CAP.run(_args(tmp_path, _pred(KO)), get=fake, now=now, sleep=sleep)
    assert out["credits"] == 0 and out["stopped"].startswith("floor")
    assert not [u for u in fake.calls if u.endswith("/odds")]


def test_provider_error_is_retried_once_then_given_up(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    fake = _Fake(odds=lambda url, p: _R(status=502, last="0"))
    now, sleep = _clock(NOW)
    CAP.run(_args(tmp_path, _pred(KO), loop_minutes=5, poll_seconds=60), get=fake, now=now, sleep=sleep)
    j = BC.read_journal(tmp_path / "store")
    assert list(j["status"]) == ["HTTP_502", "HTTP_502"]          # MAX_ATTEMPTS for the open band, no loop
    assert not (tmp_path / "store" / "data").exists()


def test_answer_without_cost_header_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    pred = pd.concat([_pred(KO, sport=f"s{i}", league=f"L{i}") for i in range(3)], ignore_index=True)
    fake = _Fake(odds=lambda url, p: _R(payload=_event(KO), last=None))
    now, sleep = _clock(NOW)
    out = CAP.run(_args(tmp_path, pred), get=fake, now=now, sleep=sleep)
    assert len([u for u in fake.calls if u.endswith("/odds")]) == 1 and "cost unknown" in out["stopped"]


def test_test_mode_is_stored_apart_and_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    far = [NOW + timedelta(hours=h) for h in (3, 4, 5, 6)]
    pred = pd.concat([_pred(k, sport=f"s{i}", league=f"L{i}") for i, k in enumerate(far)], ignore_index=True)
    fake = _Fake(odds=lambda url, p: _R(payload=_event(far[0])))
    now, sleep = _clock(NOW)
    out = CAP.run(_args(tmp_path, pred, mode="test", max_calls=3, run_cap=2), get=fake, now=now, sleep=sleep)
    assert out["credits"] == 2 and out["stopped"] == "run cap 2"
    assert (tmp_path / "store" / "test").exists() and not (tmp_path / "store" / "data").exists()
    assert set(BC.read_journal(tmp_path / "store")["band"]) == {"TEST"}


def test_store_inside_the_public_repo_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    a = _args(tmp_path, _pred(KO), store=str(ROOT / "output" / "book_close_store"))
    with pytest.raises(SystemExit):
        CAP.run(a, get=_Fake())
    assert not (ROOT / "output" / "book_close_store").exists()


def test_cli_refuses_a_floor_below_260k(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    with pytest.raises(SystemExit):
        CAP.main(["--store", str(tmp_path / "s"), "--mode", "dry-run", "--run-cap", "1", "--floor", "100000"])


def test_workflow_is_gated_capped_and_private():
    wf = (ROOT / ".github" / "workflows" / "oddsapi_book_close.yml").read_text(encoding="utf-8")
    assert "vars.ODDSAPI_BOOK_CLOSE == '1'" in wf
    assert "--floor 260000" in wf and "secrets.PRICE_STORE_KEY" in wf and "oddsapi-book-close" in wf
    assert "test mode is capped at 20 credits" in wf
    assert "git add -A" not in wf and "output/book_odds" not in wf
    # cron is often dropped: completed capture / predict runs also start it, behind the same switch
    assert "workflow_run:" in wf and "- Predict" in wf
    assert "(github.event_name == 'schedule' || github.event_name == 'workflow_run') && vars.ODDSAPI_BOOK_CLOSE == '1'" in wf


# ── entry bookmaker ────────────────────────────────────────────────────────────────────────────────
class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def test_predict_records_the_book_behind_each_entry_price(tmp_path, monkeypatch):
    import requests

    import src.btts_odds as B
    import src.predict as P
    import src.rolling_store as RS
    league = sorted(set(config.ODDS_API_SPORT_KEYS) & set(config.ENABLED_LEAGUES))[0]
    monkeypatch.setattr(config, "ODDS_API_SPORT_KEYS", {league: config.ODDS_API_SPORT_KEYS[league]})
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(B, "fetch", lambda rows: rows)
    monkeypatch.setattr(RS, "roll", lambda *a, **k: {})
    ko = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def book(key, outs):
        return {"key": key, "markets": [{"key": "totals", "outcomes": [
            {"name": n, "point": 2.5, "price": p} for n, p in outs]}]}

    ev = [{"commence_time": ko, "home_team": "H", "away_team": "A",
           "bookmakers": [book("overonly", [("Over", 2.0)]), book("both", [("Over", 2.1), ("Under", 1.8)])]}]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(ev))
    out = P._fetch_odds_api()
    assert out.iloc[0]["odds_over25_book"] == "overonly" and out.iloc[0]["odds_under25_book"] == "both"


def test_ledger_entry_book_follows_the_side_and_stays_blank_when_unknown():
    from src import ledger as L
    row = {"odds_over25_book": "b_over", "odds_under25_book": "b_under"}
    assert L._entry_book(row, "OVER") == "b_over" and L._entry_book(row, "UNDER") == "b_under"
    assert L._entry_book({}, "OVER") == "" and L._entry_book({"odds_over25_book": float("nan")}, "OVER") == ""
    assert L.LEDGER_COLS[-1] == "entry_bookmaker"


def test_shadow_schema_carries_the_entry_book():
    from src import shadow_compare as sc
    assert sc.COLUMNS[-1] == "entry_bookmaker" and "entry_bookmaker" not in sc.GRADING_ONLY
