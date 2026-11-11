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
    monkeypatch.setattr(CLI.OH, "load_fixtures",
                        lambda: (pd.DataFrame(columns=["league", "date", "home", "away", "source"]), {}))
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
    u = AD.attach_to_universe(_universe(), summary, purpose="close")
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
    assert (unmatched["kind"] == "fixture_without_event").sum() == 2       # MLS events never seen by Japan


def test_adapter_counts_one_quote_per_book():
    odds = pd.DataFrame([
        {"unit_key": "u1", "endpoint": "event_odds", "tier": "B1", "snapshot_label": "close", "event_id": "e",
         "sport_key": SK, "league": LG, "league_date": D, "requested_at": "t", "snapshot_ts": "2024-06-02T00:20:00Z",
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


# ── matching: v9 spellings, proven OddsAPI aliases, rescheduled events, +-1 day ─────────────────────
def _ev(i, h, a, ko):
    return {"id": i, "home_team": h, "away_team": a, "commence_time": ko}


def test_resolver_uses_v9_identity_spellings_and_refuses_two_clubs():
    spell = {LG: {"SK Rapid": {"SK Rapid", "Rapid Vienna"}, "Austin": {"Austin"}}}
    R = OH.Resolver(spell)
    assert R.club(LG, "Rapid Vienna", ["SK Rapid", "Austin"]) == "SK Rapid"
    assert R.club(LG, "Wiener Sportclub", ["SK Rapid", "Austin"]) is None    # unproven: never guessed
    assert OH.Resolver(spell, {LG: {"Wiener Sportclub": "SK Rapid"}}).club(LG, "Wiener Sportclub", ["SK Rapid"]) == "SK Rapid"


def _three_days(oddsapi_name="Leuven", dates=("2024-01-06", "2024-01-13", "2024-01-20")):
    days, ebd = {}, {}
    for i, d in enumerate(dates):
        days[(LG, d)] = [("Charleroi", "Oud-Heverlee Leuven"), ("Gent", "Genk")]
        ebd[(LG, d)] = [_ev(f"a{i}", "Charleroi", oddsapi_name, f"{d}T15:00:00Z"),
                        _ev(f"b{i}", "Gent", "Genk", f"{d}T18:00:00Z")]
    return days, ebd


def test_forced_opponent_on_three_days_proves_a_league_scoped_alias():
    days, ebd = _three_days()
    spell = OH.club_spellings({}, days)
    proof = OH.prove_provider_aliases(days, ebd, spell)
    assert proof["map"] == {LG: {"Leuven": "Oud-Heverlee Leuven"}}
    R = OH.Resolver(spell, proof["map"])
    m, rep = OH.match_day(LG, "2024-01-13", days[(LG, "2024-01-13")], ebd, R)
    assert len(m) == 2 and rep["unmatched_fixtures"] == []
    # the alias is league-scoped: another league never sees it
    assert OH.Resolver(spell, proof["map"]).club("Japan J-League", "Leuven", ["Oud-Heverlee Leuven"]) is None


def test_two_days_of_evidence_prove_nothing():
    days, ebd = _three_days(dates=("2024-01-06", "2024-01-13"))
    assert OH.prove_provider_aliases(days, ebd, OH.club_spellings({}, days))["map"] == {}


def test_a_name_pointing_at_two_clubs_is_refused():
    days, ebd = _three_days()
    days[(LG, "2024-01-27")] = [("Charleroi", "Westerlo"), ("Gent", "Genk")]
    ebd[(LG, "2024-01-27")] = [_ev("x", "Charleroi", "Leuven", "2024-01-27T15:00:00Z"),
                               _ev("y", "Gent", "Genk", "2024-01-27T18:00:00Z")]
    proof = OH.prove_provider_aliases(days, ebd, OH.club_spellings({}, days))
    assert proof["map"] == {} and proof["refused"]["name_points_at_two_clubs"] == 1


def test_a_rescheduled_match_takes_the_event_on_the_played_date_only():
    d = "2025-11-10"
    fx = [("Dep. Riestra", "Independiente")]
    R = OH.Resolver({LG: {"Dep. Riestra": {"Dep. Riestra", "Deportivo Riestra"}, "Independiente": {"Independiente"}}})
    ebd = {(LG, d): [_ev("stale", "Deportivo Riestra", "Independiente", "2025-11-09T21:00:00Z"),
                     _ev("real", "Deportivo Riestra", "Independiente", "2025-11-10T22:00:00Z")]}
    m, rep = OH.match_day(LG, d, fx, ebd, R)
    assert m[fx[0]]["id"] == "real" and rep["ambiguous"] == []
    ebd = {(LG, d): [_ev("one", "Deportivo Riestra", "Independiente", "2025-11-10T12:00:00Z"),
                     _ev("two", "Deportivo Riestra", "Independiente", "2025-11-10T22:00:00Z")]}
    m, rep = OH.match_day(LG, d, fx, ebd, R)
    assert m == {} and rep["ambiguous"] == fx                                 # two on the day: neither


def test_an_adjacent_day_event_needs_both_clubs_proven():
    d = "2024-03-10"
    fx = [("Austin", "Dallas"), ("Portland Timbers", "Seattle Sounders")]
    R = OH.Resolver(OH.club_spellings({}, {(LG, d): fx}))
    ebd = {(LG, d): [], (LG, "2024-03-11"): [_ev("late", "Austin FC", "FC Dallas", "2024-03-12T13:00:00Z"),
                                              _ev("half", "Austin FC", "Nowhere", "2024-03-12T15:00:00Z")]}
    m, rep = OH.match_day(LG, d, fx, ebd, R)
    assert {f: e["id"] for f, e in m.items()} == {("Austin", "Dallas"): "late"}
    assert rep["matched_adjacent_day"] == 1


def test_run_proves_aliases_from_its_listings_then_buys_and_unmatched_units_stay_retryable(tmp_path):
    dates = ["2024-06-01", "2024-06-08", "2024-06-15"]
    days = {(LG, d): [("Austin", "Dallas"), ("Portland Timbers", "Seattle")] for d in dates}

    class API(FakeAPI):
        def get(self, url, params=None, timeout=None):
            if url.endswith("/events"):
                d = params["commenceTimeFrom"][:10]
                nxt = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                self.events = [{"id": f"aus-{nxt}", "commence_time": f"{nxt}T00:30:00Z", "home_team": "Austin FC",
                                "away_team": "FC Dallas"},
                               {"id": f"por-{nxt}", "commence_time": f"{nxt}T02:30:00Z",
                                "home_team": "Portland Timbers", "away_team": "Seattle Sounders FC"}]
            elif "/events/" in url:
                eid = url.split("/events/")[1].split("/")[0]
                self.events = [{"id": eid, "commence_time": "2024-06-02T00:30:00Z", "home_team": "x", "away_team": "y"}]
            return super().get(url, params, timeout)

    api = API()
    code, _, store, meter = _run(tmp_path, api, days=days)
    assert code == OH.EXIT_DONE
    lookups = [c for c in api.calls if c[0].endswith("/events")]
    odds = [c for c in api.calls if "/events/" in c[0]]
    assert len(lookups) == 3 and len(odds) == 6        # 'Seattle Sounders FC' proven on 3 days -> all bought
    al = json.loads((tmp_path / "store" / "ledger" / "provider_aliases.json").read_text())
    assert al["map"] == {LG: {"Seattle Sounders FC": "Seattle"}}

    # with only 2 days nothing is proven: those units are NOT done and a later run retries them
    days2 = {k: v for k, v in days.items() if k[1] != "2024-06-15"}
    api2 = API()
    _run(tmp_path / "two", api2, days=days2)
    led = pd.concat([pd.read_csv(f) for f in (tmp_path / "two" / "store" / "ledger" / "units").glob("*.csv")])
    assert len([c for c in api2.calls if "/events/" in c[0]]) == 2            # Austin only
    assert set(led["status"]) <= {"ok", "empty"} and led["unit_key"].str.contains("por-").sum() == 0
    api3 = API()
    _run(tmp_path / "two", api3, days=days, run_id="r2")
    assert len([c for c in api3.calls if c[0].endswith("/events")]) == 1      # only the NEW day's listing
    assert len([c for c in api3.calls if "/events/" in c[0]]) == 4            # 3 Portland retried + new Austin


# ── provider faults: bounded retries; failed cost from the free /sports, live spend excluded ───────
F = OH.MIN_FLOOR
START = F + 1000


class Script:
    """Paid calls answer from `steps` ((status, last, remaining) or an exception); GET /sports
    answers from `sports` in order (the last value repeats)."""

    def __init__(self, steps, sports=(START,)):
        self.steps, self.sports, self.paid, self.sports_reads = list(steps), list(sports), 0, 0

    def get(self, url, params=None, timeout=None):
        if url.endswith("/sports"):
            self.sports_reads += 1
            v = self.sports.pop(0) if len(self.sports) > 1 else self.sports[0]
            return Resp(200, [], 0, v) if v is not None else Resp(200, [], None, None)
        self.paid += 1
        st = self.steps.pop(0)
        if isinstance(st, Exception):
            raise st
        status, last, rem = st
        return Resp(status, {"timestamp": "t", "data": []} if status == 200 else None, last, rem)


def _meter(steps, sports=(START,)):
    sleeps, booked = [], []
    m = OH.Meter(KEY, floor=OH.MIN_FLOOR, run_cap=10 ** 9, max_seconds=600, session=Script(steps, sports),
                 pause=0, sleep=sleeps.append, on_reconciled=lambda k, t, c: booked.append((k, t, c)))
    m.remaining = m.remaining_start = START
    return m, sleeps, booked


def test_a_502_then_success_costs_only_what_is_reported():
    #            after the 502     after the sleep
    m, sleeps, booked = _meter([(502, None, None), (200, 20, START - 20)], sports=[START, START])
    status, _, c = m.call("/x", {}, 20, "B1", key="u1")
    assert status == 200 and c == 20 and m.spent == 20 and sleeps == [5] and booked == []
    assert m.live_spend_during_backoff == 0 and m.s.sports_reads == 2


def test_a_silently_billed_502_is_booked_to_the_failed_request():
    m, _, booked = _meter([(502, None, None), (200, 20, START - 40)], sports=[START - 20, START - 20])
    assert m.call("/x", {}, 20, "B1", key="u1")[0] == 200
    assert m.spent == 40 and booked == [("u1", "B1", 20)]
    assert m.reconciled == [{"unit_key": "u1", "tier": "B1", "credits": 20}]


def test_a_live_burst_during_the_back_off_is_excluded_and_does_not_stop():
    # the 502 cost nothing; while we slept the live predictions spent 300 credits on the shared key
    m, _, booked = _meter([(502, None, None), (200, 20, START - 320)], sports=[START, START - 300])
    status, _, _ = m.call("/x", {}, 20, "B1", key="u1")
    assert status == 200 and m.spent == 20 and booked == []
    assert m.live_spend_during_backoff == 300


def test_an_unexplained_drop_beyond_one_request_stops_the_run():
    m, _, _ = _meter([(502, None, None)], sports=[START - 100])
    with pytest.raises(OH.Stop) as e:
        m.call("/x", {}, 20, "B1", key="u1")
    assert e.value.code == OH.EXIT_ERROR and "unexplained" in e.value.reason


def test_sports_without_a_remaining_header_stops_the_run():
    m, _, _ = _meter([(502, None, None)], sports=[None])
    with pytest.raises(OH.Stop) as e:
        m.call("/x", {}, 20, "B1")
    assert "remaining unknown" in e.value.reason


def test_three_retries_then_failed_and_the_unit_stays_not_done(tmp_path):
    m, sleeps, _ = _meter([(502, None, None)] * 4)
    status, body, c = m.call("/x", {}, 20, "B1", key="u1")
    assert status == "failed" and body is None and sleeps == [5, 20, 60] and m.failed_requests == 1
    assert m.s.paid == 4 and m.s.sports_reads == 4 + 3        # one after every failure, one after each sleep

    class Flaky(FakeAPI):
        def get(self, url, params=None, timeout=None):
            if "/events/ev-aus/" in url:
                self.calls.append((url, dict(params)))
                return Resp(502, None, None, None)
            return super().get(url, params, timeout)
    api = Flaky()
    code, _, store, meter = _run(tmp_path, api)
    assert code == OH.EXIT_DONE                                 # the run went on to the next unit
    led = pd.concat([pd.read_csv(f) for f in (tmp_path / "store" / "ledger" / "units").glob("*.csv")])
    assert "failed_after_retries" in set(led["status"])
    assert not any("ev-aus" in k for k in store.done)            # NOT done: retried on resume
    assert any("ev-por" in k for k in store.done)


def test_more_than_twenty_failed_requests_stop_the_run():
    m, _, _ = _meter([(503, None, None)] * 200)
    with pytest.raises(OH.Stop) as e:
        for _ in range(30):
            m.call("/x", {}, 1, "B1")
    assert m.failed_requests == OH.MAX_FAILED_PER_RUN + 1 and "failed after retries" in e.value.reason


def test_a_429_backs_off_and_retries():
    m, sleeps, _ = _meter([(429, 0, START), (200, 20, START - 20)])
    status, _, c = m.call("/x", {}, 20, "B1")
    assert status == 200 and sleeps == [5] and m.spent == 20


def test_a_timeout_is_retried_like_a_provider_fault():
    import requests
    m, sleeps, _ = _meter([requests.exceptions.Timeout("https://x?apiKey=SECRET"), (200, 20, START - 20)])
    assert m.call("/x", {}, 20, "B1")[0] == 200 and sleeps == [5]


def test_other_4xx_are_never_retried():
    m, sleeps, _ = _meter([(400, 0, START), (200, 20, START - 20)])
    assert m.call("/x", {}, 20, "B1")[0] == 400 and sleeps == [] and m.s.paid == 1


def test_a_success_without_cost_headers_still_stops():
    m, _, _ = _meter([(200, None, None)])
    with pytest.raises(OH.Stop) as e:
        m.call("/x", {}, 20, "B1")
    assert "cost not reported" in e.value.reason


# ── BT: the bet-time snapshot (owner 2026-11-11) and the shared read-path rules ────────────────────
def test_bt_is_ordered_after_b4_and_before_b3():
    order = list(OH.TIER_ORDER)
    assert order.index("B4") < order.index("BT") < order.index("B3")


def test_bt_takes_one_snapshot_per_event_at_the_track_lead_time_and_reuses_b1_lookups():
    days = {("USA MLS", D): [("Austin", "Dallas")],                 # new_format
            ("Championship", D): [("Leeds", "Hull")]}               # standard
    plan = OH.build_plan(days, _tiers("B1", "BT"), props={}, nf={"USA MLS"})
    bt = {u.league: u for u in plan["units"]["BT"]}
    assert set(bt) == {"USA MLS", "Championship"} and len(plan["units"]["BT"]) == 2
    assert bt["USA MLS"].offset_s == -OH.BT_LEAD_HOURS["new_format"] * 3600
    assert bt["Championship"].offset_s == -OH.BT_LEAD_HOURS["standard"] * 3600
    assert {u.label for u in plan["units"]["BT"]} == {"bet"}
    assert bt["USA MLS"].markets == ("btts", "alternate_totals") and bt["USA MLS"].regions == ("eu",)
    assert plan["lookups"]["BT"] == []                               # B1's listings, never re-bought


def test_bt_run_buys_at_kickoff_minus_the_lead_and_stores_role_inputs(tmp_path):
    api = FakeAPI()
    days = {(LG, D): DAYS[(LG, D)]}
    store = OH.Store(tmp_path / "store", "r1")
    meter = OH.Meter(KEY, floor=OH.MIN_FLOOR, run_cap=10_000, max_seconds=600, session=api, pause=0,
                     sleep=lambda s: None)
    plan = OH.build_plan(days, _tiers("B1", "BT"), props={}, nf={LG})
    OH.Collector(store, meter, days, plan, ["BT"]).run()
    snaps = sorted(c[1]["date"] for c in api.calls if "/events/" in c[0])
    lead = OH.BT_LEAD_HOURS["new_format"]
    assert snaps == sorted([(pd.Timestamp("2024-06-02T00:30:00Z") - pd.Timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            (pd.Timestamp("2024-06-02T02:30:00Z") - pd.Timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%SZ")])
    o = _odds(tmp_path)
    assert set(o["tier"]) == {"BT"} and set(o["snapshot_label"]) == {"bet"}


def test_snapshot_roles():
    r = OH.snapshot_role
    assert r("BT", "bet") == "BET"
    assert r("B1", "close") == r("B4", "close") == r("B2", "close") == r("C4", "close") == "CLOSE"
    assert r("C1", "T-48h") == r("C3", "T-48h") == "OPEN"
    assert r("B4", "T-48h") == "MOVEMENT"


def _long_odds(rows):
    base = {"unit_key": "u", "endpoint": "event_odds", "sport_key": SK, "league": LG, "league_date": D,
            "requested_at": "t", "event_id": "e", "home_team": "Austin FC", "away_team": "FC Dallas",
            "bookmaker": "b", "market": "btts", "outcome_description": None, "point": None,
            "commence_time": "2024-06-02T00:30:00Z"}
    out = []
    for i, r in enumerate(rows):
        out.append({**base, "unit_key": f"u{i}", "bookmaker": f"b{i}", **r})
    return pd.DataFrame(out)


def test_quotes_at_or_after_the_actual_kickoff_are_dropped():
    odds = _long_odds([
        # a RESCHEDULED event: the listing said 00:30, the match was brought forward to 00:00; the close
        # snapshots (requested off the stale 00:30) carry the new kickoff -> they are post-kickoff
        {"tier": "B1", "snapshot_label": "close", "snapshot_ts": "2024-06-02T00:25:00Z", "outcome_name": "Yes",
         "price": 1.9, "commence_time": "2024-06-02T00:00:00Z"},
        {"tier": "B1", "snapshot_label": "close", "snapshot_ts": "2024-06-02T00:30:00Z", "outcome_name": "Yes",
         "price": 1.7, "commence_time": "2024-06-02T00:00:00Z"},
        {"tier": "C1", "snapshot_label": "T-48h", "snapshot_ts": "2024-05-31T00:30:00Z", "outcome_name": "Yes",
         "price": 2.0},
        {"tier": "BT", "snapshot_label": "bet", "snapshot_ts": None, "outcome_name": "Yes", "price": 2.1}])
    long, _, _ = AD.attach(_universe().iloc[[0]], odds)
    # actual kickoff = 00:00 (the latest snapshot's statement); no snapshot time -> pre-kickoff unprovable
    assert sorted(long["price"]) == [2.0]
    assert long.attrs["dropped"]["post_kickoff"] == 2 and long.attrs["dropped"]["no_snapshot_ts"] == 1


def test_prices_at_or_below_1_01_are_dropped():
    odds = _long_odds([
        {"tier": "B1", "snapshot_label": "close", "snapshot_ts": "2024-06-02T00:25:00Z", "outcome_name": "Yes", "price": 1.01},
        {"tier": "B1", "snapshot_label": "close", "snapshot_ts": "2024-06-02T00:25:00Z", "outcome_name": "No", "price": 1.02}])
    long, _, _ = AD.attach(_universe().iloc[[0]], odds)
    assert list(long["price"]) == [1.02] and long.attrs["dropped"]["price_le_1_01"] == 1


def test_every_quote_carries_its_role_and_a_bet_price_never_comes_from_close():
    odds = _long_odds([
        {"tier": "BT", "snapshot_label": "bet", "snapshot_ts": "2024-05-26T00:00:00Z", "outcome_name": "Yes", "price": 2.2},
        {"tier": "B1", "snapshot_label": "close", "snapshot_ts": "2024-06-02T00:25:00Z", "outcome_name": "Yes", "price": 1.9},
        {"tier": "C1", "snapshot_label": "T-48h", "snapshot_ts": "2024-05-31T00:30:00Z", "outcome_name": "Yes", "price": 2.0}])
    long, summary, _ = AD.attach(_universe().iloc[[0]], odds)
    assert dict(zip(long["tier"], long["role"])) == {"BT": "BET", "B1": "CLOSE", "C1": "OPEN"}
    assert list(AD.prices_for(long, "bet")["price"]) == [2.2]
    assert list(AD.prices_for(long, "close")["price"]) == [1.9]
    with pytest.raises(AD.CloseAsBetError):
        AD.prices_for(long, "bet", role="CLOSE")
    with pytest.raises(AD.CloseAsBetError):
        AD.attach_to_universe(_universe().iloc[[0]], summary, purpose="bet", role="CLOSE")
    u = AD.attach_to_universe(_universe(), summary, purpose="bet").set_index(["home", "market"])
    assert u.loc[("Austin", "btts"), "oh_bet_yes_best"] == 2.2
    c = AD.attach_to_universe(_universe(), summary, purpose="close").set_index(["home", "market"])
    assert c.loc[("Austin", "btts"), "oh_close_yes_best"] == 1.9
    assert "oh_bet_yes_best" not in c.columns


def test_a_quote_taken_after_the_kickoff_it_carried_is_dropped_even_if_the_match_moved_later():
    # at 00:25 the provider said kickoff 00:00 (in play); a later listing postponed it to the next day
    odds = _long_odds([
        {"tier": "B1", "snapshot_label": "close", "snapshot_ts": "2024-06-02T00:25:00Z", "outcome_name": "Yes",
         "price": 1.9, "commence_time": "2024-06-02T00:00:00Z"},
        {"tier": "C1", "snapshot_label": "T-48h", "snapshot_ts": "2024-06-02T06:00:00Z", "outcome_name": "Yes",
         "price": 2.0, "commence_time": "2024-06-03T00:30:00Z"}])
    long, _, _ = AD.attach(_universe().iloc[[0]], odds)
    assert list(long["price"]) == [2.0] and long.attrs["dropped"]["post_kickoff"] == 1
