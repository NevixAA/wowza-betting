"""OU 1.5/3.5 historical backfill: exact parsing, credit guards, isolation from training data."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backfill_std_ou15_history as B  # noqa: E402

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _oc(name, point, price, desc=None):
    d = {"name": name, "point": point, "price": price}
    if desc:
        d["description"] = desc
    return d


def _event(books):
    return {"id": "e1", "home_team": "Leeds United", "away_team": "Burnley",
            "commence_time": "2024-09-14T11:30:00Z", "bookmakers": books}


def _book(key, markets):
    return {"key": key, "markets": markets}


# ── parser ─────────────────────────────────────────────────────────────────────────────────────

def test_exact_market_and_point_two_sided():
    ev = _event([_book("pinnacle", [{"key": "alternate_totals", "outcomes": [
        _oc("Over", 1.5, 1.28), _oc("Under", 1.5, 3.8), _oc("Over", 3.5, 3.16), _oc("Under", 3.5, 1.38),
        _oc("Over", 2.5, 1.9), _oc("Under", 2.5, 1.95), _oc("Over", 1.25, 1.2), _oc("Under", 1.75, 2.9)]}])])
    p = B.book_pairs(ev)
    assert p[1.5] == {"pinnacle": (1.28, 3.8)}
    assert p[3.5] == {"pinnacle": (3.16, 1.38)}


def test_first_half_and_other_markets_never_contaminate():
    ev = _event([_book("coolbet", [
        {"key": "alternate_totals_h1", "outcomes": [_oc("Over", 1.5, 3.4), _oc("Under", 1.5, 1.3)]},
        {"key": "totals_h1", "outcomes": [_oc("Over", 1.5, 3.5), _oc("Under", 1.5, 1.28)]},
        {"key": "totals", "outcomes": [_oc("Over", 1.5, 1.01), _oc("Under", 1.5, 15.0)]},
        {"key": "alternate_team_totals", "outcomes": [_oc("Over", 1.5, 2.0, "Leeds United"),
                                                      _oc("Under", 1.5, 1.8, "Leeds United")]},
        {"key": "alternate_totals", "outcomes": [_oc("Over", 1.5, 2.0, "Leeds United"),
                                                 _oc("Under", 1.5, 1.8, "Leeds United")]},
    ])])
    assert B.book_pairs(ev) == {1.5: {}, 3.5: {}}


def test_one_sided_duplicate_and_label_variants_are_dropped():
    ev = _event([
        _book("onesided", [{"key": "alternate_totals", "outcomes": [_oc("Over", 1.5, 1.3)]}]),
        _book("dup", [{"key": "alternate_totals", "outcomes": [
            _oc("Over", 1.5, 1.3), _oc("Over", 1.5, 1.32), _oc("Under", 1.5, 3.5)]}]),
        _book("label", [{"key": "alternate_totals", "outcomes": [
            _oc("Over 1.5", 1.5, 1.3), _oc("under", 1.5, 3.5)]}]),
        _book("garbage", [{"key": "alternate_totals", "outcomes": [
            _oc("Over", 1.5, 1.05), _oc("Under", 1.5, 1.05)]}]),
        _book("ok", [{"key": "alternate_totals", "outcomes": [_oc("Over", "1.5", 1.3), _oc("Under", 1.5, 3.5)]}]),
    ])
    assert B.book_pairs(ev)[1.5] == {"ok": (1.3, 3.5)}


def test_rows_median_books_and_research_label():
    payload = {"timestamp": "2024-09-13T23:55:39Z", "data": _event([
        _book("pinnacle", [{"key": "alternate_totals", "outcomes": [_oc("Over", 1.5, 1.28), _oc("Under", 1.5, 3.8)]}]),
        _book("coolbet", [{"key": "alternate_totals", "outcomes": [_oc("Over", 1.5, 1.27), _oc("Under", 1.5, 3.67)]}]),
        _book("williamhill", [{"key": "alternate_totals", "outcomes": [_oc("Over", 1.5, 1.25), _oc("Under", 1.5, 3.75)]}]),
    ])}
    meta = {"id": "e1", "league": "Championship", "home": "Leeds United", "away": "Burnley",
            "commence": "2024-09-14T11:30:00Z"}
    rows = B.rows_for_event(payload, meta, "2024-09-14T00:00:00Z",
                            name_map=lambda lg, n: {"Leeds United": "Leeds"}.get(n, n))
    by = {r["market"]: r for r in rows}
    assert set(by) == {"over15", "under15"}                      # no 3.5 quoted -> no 3.5 rows
    assert by["over15"]["odds"] == 1.27 and by["under15"]["odds"] == 3.75
    assert by["over15"]["n_books"] == 3 and by["over15"]["odds_pinnacle"] == 1.28
    assert by["over15"]["price_class"] == "ODDSAPI_HISTORICAL_SNAPSHOT"
    assert by["over15"]["match"] == "Leeds vs Burnley" and by["over15"]["name_resolved"] == 1
    assert by["over15"]["match_date"] == "2024-09-14" and by["over15"]["snapshot_ts"].startswith("2024-09-13")
    assert set(by["over15"]) == set(B.COLS)


def test_snapshot_is_pre_match():
    assert B.snapshot_ts_for("2025-11-08T13:00:00Z") == "2025-11-08T00:00:00Z"
    assert B.snapshot_ts_for("2025-11-08T00:15:00Z") == "2025-11-07T22:15:00Z"


def test_output_is_isolated_from_training_file():
    assert B.OUT_CSV.parent.name == "research" and B.STATE.parent.name == "research"
    assert B.OUT_CSV.name != "standard_sidemarket_odds_history.csv"
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8")
    assert "std_ou15" not in src and "output/research" not in src


# ── guards ─────────────────────────────────────────────────────────────────────────────────────

def _cfg(**kw):
    c = {"start_threshold": 60000, "floor_min": 25000, "run_cap": 8000, "live_burn_default": 3400,
         "cycle_days": 30, "safety": 1.15, "dynamic": True}
    c.update(kw)
    return c


def _st(**kw):
    st = {"cursor": {}, "events": {}, "ledger": [], "spent_total": 0, "reset_ts": None, "complete": False}
    st.update(kw)
    return st


def test_no_run_below_start_threshold():
    go, budget, why = B.decide(59999, _st(), NOW, _cfg(dynamic=False))
    assert not go and budget == 0 and "START_THRESHOLD" in why


def test_no_run_when_complete():
    assert B.decide(5_000_000, _st(complete=True), NOW, _cfg())[0] is False


def test_fixed_floor_budget_is_capped():
    go, budget, _ = B.decide(100000, _st(), NOW, _cfg(dynamic=False))
    assert go and budget == 8000
    go, budget, _ = B.decide(30000, _st(), NOW, _cfg(dynamic=False, start_threshold=0))
    assert go and budget == 5000                                    # 30,000 - 25,000 floor


def test_dynamic_reserve_protects_a_month_of_live_usage():
    # 100k plan just reset, live burn 3,400/day, 30 days left -> reserve 117,300 > 100k: no run
    go, _, why = B.decide(100000, _st(), NOW, _cfg())
    assert not go and "no room above the floor" in why
    # a bigger plan leaves real room above the reserve
    go, budget, _ = B.decide(4_900_000, _st(), NOW, _cfg())
    assert go and budget == 8000
    # late in a cycle the reserve shrinks with the days left
    reset = (NOW - timedelta(days=27)).strftime("%Y-%m-%dT%H:%M:%SZ")
    floor, _ = B.compute_floor(_st(reset_ts=reset), NOW, _cfg())
    assert floor == 25000                                           # 3 days x 3,400 x 1.15 < min


def test_measured_burn_excludes_own_spend_and_detects_reset():
    st = _st()
    t0 = NOW - timedelta(days=4)
    B.record_balance(st, 90000, 10000, t0 - timedelta(days=1))
    B.record_balance(st, 1000, 99000, t0)                           # used dropped -> reset
    assert st["reset_ts"] == t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    st["spent_total"] = 8000
    B.record_balance(st, 1000 + 4 * 5000 + 8000, 0, NOW)            # 5,000/day live + 8,000 ours
    assert B.measured_live_burn(st, NOW) == pytest.approx(5000)
    floor, why = B.compute_floor(st, NOW, _cfg())
    assert floor == int(-(-5000 * 26 * 1.15 // 1)) and "measured 5,000" in why


class _Resp:
    def __init__(self, status, body, last, remaining, used=0):
        self.status_code, self._body, self.text = status, body, str(body)
        self.headers = {"x-requests-last": str(last), "x-requests-remaining": str(remaining),
                        "x-requests-used": str(used)}

    def json(self):
        return self._body


class _Session:
    """Fake OddsAPI: every events list returns one finished event; each odds call costs 10."""

    def __init__(self, remaining, odds_status=200):
        self.remaining, self.odds_status, self.calls = remaining, odds_status, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/events"):
            d = params["date"][:10]
            self.remaining -= 1
            return _Resp(200, {"data": [{"id": f"ev{len(self.calls)}", "home_team": "A", "away_team": "B",
                                         "commence_time": f"{d}T15:00:00Z"}]}, 1, self.remaining)
        if self.odds_status != 200:
            return _Resp(self.odds_status, {"message": "nope"}, 0, self.remaining)
        self.remaining -= 10
        return _Resp(200, {"timestamp": "x", "data": _event([_book("pinnacle", [{"key": "alternate_totals",
                     "outcomes": [_oc("Over", 1.5, 1.3), _oc("Under", 1.5, 3.5)]}])])}, 10, self.remaining)


@pytest.fixture
def tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT_DIR", tmp_path)
    monkeypatch.setattr(B, "OUT_CSV", tmp_path / "out.csv")
    monkeypatch.setattr(B, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(B, "make_name_map", lambda: None)
    return tmp_path


def test_run_stops_at_floor_and_resumes(tmp_out):
    s = _Session(remaining=25_400)
    assert B.run("k", budget=10_000, floor=25_000, session=s) == 0
    assert s.remaining >= 25_000                                    # never crossed the floor
    st = B.load_state()
    done = [m for m in st["events"].values() if m["status"] == "done"]
    assert done and st["spent_total"] == 25_400 - s.remaining and not st["complete"]
    n_rows = sum(1 for _ in (tmp_out / "out.csv").open()) - 1
    assert n_rows == 2 * len(done)                                  # over15 + under15 per event
    # resume: already-done events are never asked again
    s2 = _Session(remaining=25_100)
    B.run("k", budget=10_000, floor=25_000, session=s2)
    st2 = B.load_state()
    assert all(st2["events"][e]["status"] == "done" for e in (k for k, m in st["events"].items()
                                                                  if m["status"] == "done"))


def test_run_respects_per_run_cap(tmp_out):
    s = _Session(remaining=1_000_000)
    B.run("k", budget=55, floor=0, session=s)
    assert 1_000_000 - s.remaining <= 55


@pytest.mark.parametrize("code", [401, 422, 429])
def test_hard_stop_codes(tmp_out, code):
    s = _Session(remaining=1_000_000, odds_status=code)
    assert B.run("k", budget=500, floor=0, session=s) == 3
    odds_calls = [c for c in s.calls if c.endswith("/odds")]
    assert len(odds_calls) == 1                                     # stopped on the first refusal
