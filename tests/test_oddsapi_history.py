"""OddsAPI historical collector (src/oddsapi_history.py) and its read-side adapter: mock HTTP only,
no network, no credits. Every store is a tmp dir; nothing under output/ is written."""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import oddsapi_history as OH  # noqa: E402
from src import oddsapi_history_adapter as AD  # noqa: E402

LG, SK, D = "USA MLS", "soccer_usa_mls", "2024-06-01"
KEY = "SECRETKEY123"
DAYS = {(LG, D): [("Austin", "Dallas"), ("Portland Timbers", "Seattle Sounders")]}
EV = [{"id": "ev-aus", "commence_time": "2024-06-02T00:30:00Z", "home_team": "Austin FC", "away_team": "FC Dallas"},
      {"id": "ev-por", "commence_time": "2024-06-02T02:30:00Z", "home_team": "Portland Timbers",
       "away_team": "Seattle Sounders FC"},
      {"id": "ev-xyz", "commence_time": "2024-06-02T01:00:00Z", "home_team": "Nowhere United",
       "away_team": "Elsewhere City"}]


class Resp:
    def __init__(self, status, body, last, remaining):
        self.status_code, self._body = status, body
        self.headers = {} if last is None else {"x-requests-last": str(last), "x-requests-remaining": str(remaining)}
        if last is None and remaining is not None:
            self.headers = {"x-requests-remaining": str(remaining)}

    def json(self):
        return self._body


def _book(markets):
    return {"key": "book1", "title": "Book 1", "last_update": "2024-06-01T23:00:00Z", "markets": markets}


def _event_odds(ev):
    return {"id": ev["id"], "commence_time": ev["commence_time"], "home_team": ev["home_team"],
            "away_team": ev["away_team"], "bookmakers": [
                _book([{"key": "btts", "outcomes": [{"name": "Yes", "price": 1.8}, {"name": "No", "price": 2.0}]},
                       {"key": "alternate_totals", "outcomes": [
                           {"name": "Over", "price": 1.25, "point": 1.5}, {"name": "Under", "price": 3.9, "point": 1.5},
                           {"name": "Over", "price": 7.5, "point": 4.5}, {"name": "Under", "price": 1.08, "point": 4.5},
                           {"name": "Over", "point": 5.5},                       # no price -> NOT a row
                           {"name": "Under", "price": "n/a", "point": 5.5}]}]),  # not a price -> NOT a row
                {"key": "book2", "title": "Book 2", "last_update": "x", "markets": [
                    {"key": "btts", "outcomes": [{"name": "Yes", "price": 1.85}, {"name": "No", "price": 1.95}]}]}]}


class FakeAPI:
    """Bills like the docs: lookups 1 (0 when empty), odds 10 x markets RETURNED x regions."""

    def __init__(self, remaining=1_000_000, events=EV, report_cost=True, raise_exc=False):
        self.remaining, self.events, self.report_cost, self.raise_exc = remaining, events, report_cost, raise_exc
        self.calls = []

    def get(self, url, params=None, timeout=None):
        if self.raise_exc:
            raise ConnectionError(f"failed to reach {url}?apiKey={params.get('apiKey')}")
        self.calls.append((url, dict(params)))
        if url.endswith("/sports"):
            return Resp(200, [], 0, self.remaining)
        if url.endswith("/events"):
            data = [e for e in self.events]
            cost = 1 if data else 0
            body = {"timestamp": params["date"], "data": data}
        elif "/events/" in url:
            eid = url.split("/events/")[1].split("/")[0]
            ev = next(e for e in self.events if e["id"] == eid)
            data = _event_odds(ev)
            mk = {m["key"] for b in data["bookmakers"] for m in b["markets"]} & set(params["markets"].split(","))
            cost = 10 * len(mk) * len(params["regions"].split(","))
            body = {"timestamp": params["date"], "data": data}
        else:
            data = [{**e, "bookmakers": [_book([{"key": "h2h", "outcomes": [
                {"name": e["home_team"], "price": 2.1}, {"name": "Draw", "price": 3.3},
                {"name": e["away_team"], "price": 3.4}]}])]} for e in self.events]
            data.append({"id": "other-day", "commence_time": "2024-06-09T00:00:00Z", "home_team": "A",
                         "away_team": "B", "bookmakers": [_book([{"key": "h2h", "outcomes": [
                             {"name": "A", "price": 2.0}]}])]})
            cost = 10 * len(params["markets"].split(",")) * len(params["regions"].split(","))
            body = {"timestamp": params["date"], "data": data}
        self.remaining -= cost
        return Resp(200, body, cost if self.report_cost else None, self.remaining)


def _tiers(*names):
    return tuple(t for t in OH.TIERS if t.name in names)


def _plan(days=DAYS, names=("B1",), **kw):
    return OH.build_plan(days, _tiers(*names), props=kw.get("props", {}), s_ok=kw.get("s_ok"), nf={LG})


def _run(tmp_path, api, names=("B1",), floor=OH.MIN_FLOOR, cap=100_000, run_id="r1", minutes=60, days=DAYS):
    store = OH.Store(tmp_path / "store", run_id)
    meter = OH.Meter(KEY, floor=floor, run_cap=cap, max_seconds=minutes * 60, session=api, pause=0,
                     sleep=lambda s: None)
    code, reason = OH.Collector(store, meter, days, _plan(days, names), names).run()
    return code, reason, store, meter


def _odds(tmp_path):
    return AD.load_store(tmp_path / "store")


# ── the plan ─────────────────────────────────────────────────────────────────────────────────────
def test_plan_counts_calls_and_maximum_credits_exactly():
    plan = _plan(names=("B1", "C1", "C4"))
    s = OH.summarise_plan(plan, DAYS)["tiers"]
    assert s["B1"]["odds_calls"] == 2 and s["B1"]["lookup_calls"] == 1 and s["B1"]["max_credits"] == 2 * 20 + 1
    assert s["C1"]["lookup_calls"] == 0 and s["C1"]["max_credits"] == 40      # the lookup is bought once
    assert [u.regions for u in plan["units"]["C4"]] == [("uk",), ("uk",)]    # eu already requested by B1
    assert s["C4"]["cumulative_max_credits"] == 41 + 40 + 40


def test_a_later_tier_never_rebuys_a_requested_market():
    days = {(LG, "2021-06-01"): [("Austin", "Dallas")]}
    plan = OH.build_plan(days, _tiers("B2", "C3", "C5"), props={}, nf={LG})
    c3 = {u.label: u.markets for u in plan["units"]["C3"]}
    assert c3 == {"T-48h": ("h2h", "totals"), "close": ("h2h",)}             # B2 bought close totals
    assert sorted(u.label for u in plan["units"]["C5"]) == sorted(["T-72h", "T-24h", "T-12h", "T-6h", "T-2h", "T-1h"])


def test_props_tier_uses_only_the_probe_markets_and_s_tier_only_confirmed_leagues():
    days = {("Premier League", "2024-05-19"): [("Arsenal", "Everton")], (LG, D): [("Austin", "Dallas")]}
    plan = OH.build_plan(days, _tiers("B3", "S"), props={"Premier League": ("player_assists",)}, s_ok=set(), nf={LG})
    assert [(u.league, u.markets, u.max_cost) for u in plan["units"]["B3"]] == [
        ("Premier League", ("player_assists",), 40)]
    assert plan["units"]["S"] == []                                          # S probe confirmed nothing
    plan = OH.build_plan(days, _tiers("S"), props={}, s_ok=None, nf={LG})    # dry run: maximum
    assert [(u.league, u.max_cost) for u in plan["units"]["S"]] == [(LG, 20)]


def test_probe_props_markets_reads_what_the_probe_saw_returned(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"leagues": {"Ligue 1": {"props": {"markets": {"player_shots_on_target": {}}}},
                                         "La Liga": {"props": {"books": 0, "markets": {}}}}}))
    assert OH.probe_props_markets(f) == {"Ligue 1": ("player_shots_on_target",)}


def test_dedupe_day_merges_the_same_match_from_two_sources():
    rows = [("Austin FC", "FC Dallas", "af"), ("Portland Timbers", "Seattle Sounders", "af"),
            ("Austin", "Dallas", "fd")]
    assert OH.dedupe_day(rows) == [("Austin FC", "FC Dallas"), ("Portland Timbers", "Seattle Sounders")]


# ── the run ──────────────────────────────────────────────────────────────────────────────────────
def test_run_buys_matched_events_only_and_stores_real_prices(tmp_path):
    api = FakeAPI()
    code, _, store, meter = _run(tmp_path, api)
    assert code == OH.EXIT_DONE
    odds_calls = [c for c in api.calls if "/events/" in c[0]]
    assert sorted(c[0].split("/events/")[1].split("/")[0] for c in odds_calls) == ["ev-aus", "ev-por"]
    assert all(c[1]["markets"] == "btts,alternate_totals" and c[1]["regions"] == "eu" for c in odds_calls)
    # close = kickoff - 5 minutes
    assert {c[1]["date"] for c in odds_calls} == {"2024-06-02T00:25:00Z", "2024-06-02T02:25:00Z"}
    o = _odds(tmp_path)
    assert o["price"].notna().all() and (o["price"] > 1).all()
    assert set(o["point"].dropna()) == {1.5, 4.5}                             # every line kept, 5.5 had no price
    assert len(o) == 2 * (2 + 4 + 2)
    assert meter.spent == 1 + 20 + 20
    run = json.loads(next((tmp_path / "store" / "ledger" / "runs").glob("*.json")).read_text())
    assert run["credits_spent"] == 41 and run["exit_code"] == 0
    m = pd.read_csv(next((tmp_path / "store" / "ledger" / "match").glob("*.csv")))
    assert m["kind"].tolist() == ["unmatched_event"]                          # recorded, never bought


def test_resume_skips_everything_already_stored(tmp_path):
    _run(tmp_path, FakeAPI())
    api = FakeAPI()
    code, _, _, meter = _run(tmp_path, api, run_id="r2")
    assert code == OH.EXIT_DONE and meter.spent == 0
    assert [c[0] for c in api.calls] == [OH.API + "/sports"]                  # only the free credits read


def test_floor_stops_before_a_call_could_cross_it(tmp_path):
    api = FakeAPI(remaining=OH.MIN_FLOOR + 21)
    code, reason, _, meter = _run(tmp_path, api)
    assert code == OH.EXIT_FLOOR and "floor" in reason
    assert meter.remaining >= OH.MIN_FLOOR
    assert meter.spent == 1 + 20                                              # lookup + one event, no more


def test_floor_below_the_owner_minimum_is_refused():
    with pytest.raises(ValueError):
        OH.Meter(KEY, floor=OH.MIN_FLOOR - 1, run_cap=10, max_seconds=60, session=FakeAPI())


def test_run_cap_stops_the_run(tmp_path):
    code, reason, _, meter = _run(tmp_path, FakeAPI(), cap=25)
    assert code == OH.EXIT_RUN_CAP and meter.spent <= 25


def test_an_unreported_cost_stops_the_run(tmp_path):
    code, reason, _, _ = _run(tmp_path, FakeAPI(report_cost=False))
    assert code == OH.EXIT_ERROR and "cost" in reason


def test_time_budget_stops_the_run(tmp_path):
    code, _, _, meter = _run(tmp_path, FakeAPI(), minutes=0)
    assert code == OH.EXIT_TIME and meter.spent == 0


def test_logs_carry_no_data_no_names_no_key(tmp_path, capsys):
    _run(tmp_path, FakeAPI())
    _run(tmp_path / "x", FakeAPI(raise_exc=True))
    out = capsys.readouterr()
    text = out.out + out.err
    for banned in (KEY, "Austin", "Dallas", "Portland", "ev-aus", "1.8", "7.5", "apiKey", "https://"):
        assert banned not in text, banned
    assert "credits spent" in text


def test_featured_snapshot_is_timed_off_the_first_matched_kickoff(tmp_path):
    api = FakeAPI()
    days = {(LG, "2023-07-01"): DAYS[(LG, D)]}
    ev = [{**e, "commence_time": e["commence_time"].replace("2024-06-02", "2023-07-02")} for e in EV]
    api.events = ev
    code, _, _, _ = _run(tmp_path, api, names=("B4",), days=days)
    assert code == OH.EXIT_DONE
    snaps = sorted(c[1]["date"] for c in api.calls if c[0].endswith("/odds") and "/events/" not in c[0])
    assert snaps[0] == "2023-06-29T00:30:00Z" and snaps[-1] == "2023-07-02T00:25:00Z" and len(snaps) == 8
    o = _odds(tmp_path)
    assert "other-day" not in set(o["event_id"])                              # outside the league-day


def test_cli_refuses_a_store_inside_the_public_repo_and_missing_settings(monkeypatch):
    import oddsapi_history_collect as CLI
    monkeypatch.setattr(CLI.OH, "load_fixtures", lambda: pd.DataFrame(columns=["league", "date", "home", "away", "source"]))
    with pytest.raises(SystemExit, match="public"):
        CLI.main(["--store", str(ROOT / "output" / "x"), "--max-minutes", "1"])
    monkeypatch.delenv("ODDS_HIST_FLOOR", raising=False)
    with pytest.raises(SystemExit, match="ODDS_HIST_FLOOR"):
        CLI.main(["--store", str(ROOT.parent / "elsewhere"), "--max-minutes", "1"])


def test_probe_s_records_presence_only(tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"leagues": {LG: {"sport_key": SK, "event": {"id": "ev-aus", "kickoff": EV[0]["commence_time"]}},
                                         "Saudi Pro League": {"sport_key": "x", "event": "none"}}}))
    api = FakeAPI()
    meter = OH.Meter(KEY, floor=OH.MIN_FLOOR, run_cap=1000, max_seconds=60, session=api, pause=0, sleep=lambda s: None)
    res = OH.probe_s(meter, tmp_path / "store", probe_file=p)
    assert res["leagues"][LG]["correct_score_returned"] is False                # the fake has no correct_score
    assert res["leagues"]["Saudi Pro League"] == {"skipped": "no P7 probe event"}
    assert OH.s_probe_leagues(tmp_path / "store") == set()
    assert "price" not in json.dumps(res)


# ── the adapter ──────────────────────────────────────────────────────────────────────────────────
def _universe():
    rows = []
    for m in ("over15", "over25", "btts"):
        rows += [{"row_id": f"{D}|{LG}|Austin|Dallas", "date": pd.Timestamp(D), "league": LG, "home": "Austin",
                  "away": "Dallas", "market": m},
                 {"row_id": f"{D}|{LG}|Portland Timbers|Seattle Sounders", "date": pd.Timestamp(D), "league": LG,
                  "home": "Portland Timbers", "away": "Seattle Sounders", "market": m}]
    return pd.DataFrame(rows)


def test_adapter_attaches_real_prices_by_fixture_id(tmp_path):
    _run(tmp_path, FakeAPI())
    long, summary, unmatched = AD.attach(_universe(), _odds(tmp_path))
    assert set(long["fixture_id"]) == {f"{D}|{LG}|Austin|Dallas", f"{D}|{LG}|Portland Timbers|Seattle Sounders"}
    assert {"over15", "over45", "btts"} <= set(long["market"])
    u = AD.attach_to_universe(_universe(), summary)
    a = u[(u["home"] == "Austin")].set_index("market")
    assert a.loc["over15", "oh_close_yes_best"] == 1.25 and a.loc["over15", "oh_close_no_best"] == 3.9
    assert a.loc["btts", "oh_close_yes_n_books"] == 2 and a.loc["btts", "oh_close_yes_best"] == 1.85
    assert np.isnan(a.loc["over25", "oh_close_yes_best"])                     # never priced -> stays missing
    assert unmatched["kind"].tolist() == []


def test_adapter_never_joins_across_leagues_and_reports_unmatched(tmp_path):
    _run(tmp_path, FakeAPI())
    u = _universe().assign(league="Japan J-League")                           # same names, other league
    long, _, unmatched = AD.attach(u, _odds(tmp_path))
    assert long.empty
    assert (unmatched["kind"] == "fixture_without_event").sum() == 2
    assert (unmatched["kind"] == "event_without_fixture").sum() == 2


def test_adapter_counts_one_quote_per_book():
    odds = pd.DataFrame([
        {"unit_key": "u1", "endpoint": "event_odds", "tier": "B1", "snapshot_label": "close", "event_id": "e",
         "sport_key": SK, "league": LG, "requested_at": "t", "snapshot_ts": "2024-06-02T00:20:00Z",
         "commence_time": "2024-06-02T00:30:00Z", "home_team": "Austin FC", "away_team": "FC Dallas",
         "bookmaker": "b", "market": mk, "outcome_name": "Over", "outcome_description": None, "point": 2.5,
         "price": p} for mk, p in (("alternate_totals", 1.9), ("totals", 1.95))])
    odds.loc[1, "unit_key"] = "u2"
    _, s, _ = AD.attach(_universe().iloc[[0]], odds)
    row = s[(s["market"] == "over25") & (s["side"] == "yes")].iloc[0]
    assert row["n_books"] == 1


def test_classify_places_only_exact_outcomes():
    assert AD.classify("alternate_totals", "Over", None, 6.5, "H", "A")["market"] == "over65"
    assert AD.classify("alternate_totals", "Over", None, 2.25, "H", "A")["market"] == "total_line"
    assert AD.classify("h2h", "Someone", None, None, "H", "A") is None
    assert AD.classify("team_totals", "Over", "A", 1.5, "H", "A")["market"] == "team_total_away"
    assert AD.classify("team_totals", "Over", "Z", 1.5, "H", "A") is None
