"""Prop prices are scoped to the tip's OWN fixture (2026-10-25).

Both price sources used to collapse every fixture into one {"player|market": max price} lookup,
so a namesake playing elsewhere the same day lent his price. Real cases from 2026-10-24:

* Matheuzinho (Vitoria v Atletico Paranaense) goals carried 15.00 — Bet365's price for the OTHER
  Matheuzinho in Vasco v Corinthians. His own quotes were Betano 3.85 / 3.90.
* A. Marin (Girona v Celta de Vigo II) sot 5.00 / sot2 36.00 — Adrian Marin's 1xBet prices in
  Orlando City v New York City FC (MLS). Alvaro Marin had no SOT quote in his own match.
* Andres Martin (Racing Santander v Espanyol) sot 5.50 — Alex Martin's Bet365 price in Cordoba v
  Eibar, reached through the initial+surname fallback.
"""
import math

import pandas as pd
import pytest

from player_model import odds_fetcher as of
from player_model import prop_odds_af as af
from player_model.predict import enrich_with_odds

DAY = "2026-10-24"
VITORIA = "Vitoria vs Atletico Paranaense"
VASCO = "Vasco DA Gama vs Corinthians"
GIRONA = "Girona vs Celta de Vigo II"
CORDOBA = "Cordoba vs Eibar"
ORLANDO = "Orlando City SC vs New York City FC"
RACING = "Racing Santander vs Espanyol"


def _tip(player, market, league, match, date=DAY):
    return {"date": date, "league": league, "match": match, "player_name": player,
            "market": market, "model_prob": 0.30, "confidence": 0.6, "ges": 0.5,
            "lazy_factors": "", "position": "F", "market_odds": float("nan"),
            "fair_implied": float("nan"), "edge_abs": float("nan"), "edge_rel": float("nan"),
            "ev": float("nan"), "tier": "AVOID", "kelly_stake": float("nan")}


# ── API-Football /odds payloads, one per fixture id ──────────────────────────────────────
def _af_bet(name, values):
    return {"name": name, "values": [{"value": v, "odd": str(o)} for v, o in values]}


AF_ODDS = {
    1: [{"bookmakers": [  # Vitoria v Atletico-PR: OUR Matheuzinho
        {"name": "Betano", "bets": [_af_bet("Anytime Goal Scorer", [("Matheuzinho", 3.85)])]},
        {"name": "Betano2", "bets": [_af_bet("Anytime Goal Scorer", [("Matheuzinho", 3.90)])]},
    ]}],
    2: [{"bookmakers": [  # Vasco v Corinthians: the OTHER Matheuzinho
        {"name": "Bet365", "bets": [_af_bet("Anytime Goal Scorer", [("Matheuzinho", 15.0)])]},
    ]}],
    3: [{"bookmakers": [  # Girona v Celta II: Alvaro Marin goals only, no SOT line
        {"name": "Bet365", "bets": [_af_bet("Anytime Goal Scorer", [("Alvaro Marin", 4.0)])]},
    ]}],
    4: [{"bookmakers": [  # Cordoba v Eibar: Alex Martin SOT (a different "A. Martin")
        {"name": "Bet365", "bets": [_af_bet("Player Shots On Target Total",
                                            [("Alex Martin - 1+", 5.5)])]},
    ]}],
}
AF_FIXTURE_IDS = {("Vitoria", "Atletico Paranaense"): 1, ("Vasco DA Gama", "Corinthians"): 2,
                  ("Girona", "Celta de Vigo II"): 3, ("Cordoba", "Eibar"): 4}

# ── OddsAPI: MLS (Adrian Marin) and La Liga (Andres Martin, goals only) ───────────────────
ODDSAPI_EVENTS = {
    "soccer_usa_mls": [{"id": "ev_orl", "home_team": "Orlando City SC",
                        "away_team": "New York City FC", "commence_time": f"{DAY}T23:30:00Z"}],
    "soccer_spain_la_liga": [{"id": "ev_rac", "home_team": "Racing Santander",
                              "away_team": "Espanyol", "commence_time": f"{DAY}T19:00:00Z"}],
}
ODDSAPI_ODDS = {
    "ev_orl": [{"title": "1xBet", "markets": [
        {"key": "player_shots_on_target", "outcomes": [
            {"name": "Over", "description": "Adrian Marin", "point": 0.5, "price": 5.0},
            {"name": "Under", "description": "Adrian Marin", "point": 0.5, "price": 9.0},
            {"name": "Over", "description": "Adrian Marin", "point": 1.5, "price": 36.0},
        ]}]}],
    "ev_rac": [{"title": "William Hill", "markets": [
        {"key": "player_goal_scorer_anytime", "outcomes": [
            {"name": "Yes", "description": "Andres Martin", "price": 2.9},
        ]}]}],
}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """fetch_prop_odds with every network call and file write stubbed out."""
    def fake_oddsapi_get(path, params):
        if path.endswith("/events"):
            return ODDSAPI_EVENTS.get(path.split("/")[1], [])
        ev = path.split("/")[3]
        return {"bookmakers": ODDSAPI_ODDS.get(ev, [])}

    import player_model.api_football as apf
    monkeypatch.setattr(of, "_get", fake_oddsapi_get)
    monkeypatch.setattr(of, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(of, "validate_sport_keys", lambda **k: {})
    monkeypatch.setattr(of, "_load_coverage", lambda: {})
    monkeypatch.setattr(of, "_save_coverage", lambda cov: None)
    monkeypatch.setattr(of, "_append_quotes", lambda q: None)
    monkeypatch.setattr(of, "_append_odds_history", lambda r: None)
    monkeypatch.setattr(af, "_prop_leagues", lambda: {"Brazil Serie A": 71, "La Liga 2": 141})
    monkeypatch.setattr(af, "_prop_seasons", lambda: {"Brazil Serie A": "2026", "La Liga 2": "2026"})
    monkeypatch.setattr(af, "find_upcoming_fixture_id",
                        lambda lid, season, d, h, a: AF_FIXTURE_IDS.get((h, a)))
    monkeypatch.setattr(apf, "_get",
                        lambda path, params, **k: {"response": AF_ODDS.get(params["fixture"], [])})
    af.QUOTES.clear()


def _run_pipeline(tips):
    """Exactly what player_model.pipeline does: fetch -> match -> enrich."""
    raw = of.fetch_prop_odds(tips)
    mapped = of.match_odds_to_tips(tips, raw)
    return enrich_with_odds(tips.copy(), mapped)


def _odds(out, player, market, match):
    v = out[(out.player_name == player) & (out.market == market) & (out.match == match)]
    assert len(v) == 1
    return v["market_odds"].iloc[0]


def _board():
    return pd.DataFrame([
        _tip("Matheuzinho", "goals", "Brazil Serie A", VITORIA),
        _tip("Matheuzinho", "goals", "Brazil Serie A", VASCO),
        _tip("A. Marin", "sot", "La Liga 2", GIRONA),
        _tip("A. Marin", "sot2", "La Liga 2", GIRONA),
        _tip("A. Marin", "goals", "La Liga 2", GIRONA),
        _tip("Andrés Martín", "sot", "La Liga", RACING),
        _tip("Andrés Martín", "goals", "La Liga", RACING),
        _tip("Adrián Marín", "sot", "USA MLS", ORLANDO),
        _tip("Alex Martin", "sot", "La Liga 2", CORDOBA),
    ])


# ── End to end through the real pipeline functions (fails on the pre-fix code) ───────────
def test_namesake_in_other_fixture_does_not_lend_price(wired):
    out = _run_pipeline(_board())
    # Own-fixture best across books, never the other Matheuzinho's 15.00.
    assert _odds(out, "Matheuzinho", "goals", VITORIA) == pytest.approx(3.90)
    assert _odds(out, "Matheuzinho", "goals", VASCO) == pytest.approx(15.0)


def test_initial_surname_fallback_cannot_cross_fixtures(wired):
    out = _run_pipeline(_board())
    # Adrian Marin (MLS) and Alex Martin (Cordoba v Eibar) must not price these tips.
    assert math.isnan(_odds(out, "A. Marin", "sot", GIRONA))
    assert math.isnan(_odds(out, "A. Marin", "sot2", GIRONA))
    assert math.isnan(_odds(out, "Andrés Martín", "sot", RACING))
    # ...but the fallback still works INSIDE the tip's own fixture.
    assert _odds(out, "A. Marin", "goals", GIRONA) == pytest.approx(4.0)
    assert _odds(out, "Andrés Martín", "goals", RACING) == pytest.approx(2.9)
    assert _odds(out, "Alex Martin", "sot", CORDOBA) == pytest.approx(5.5)


def test_under_is_never_the_tip_price(wired):
    out = _run_pipeline(_board())
    assert _odds(out, "Adrián Marín", "sot", ORLANDO) == pytest.approx(5.0)   # not Under 9.0


# ── Unit level ───────────────────────────────────────────────────────────────────────────
def _fp(*rows):
    fp = {}
    for d, m, p, mk, o in rows:
        of._add_fixture_price(fp, d, m, of._norm(p), mk, o)
    return fp


def test_exact_name_scoped_to_fixture():
    fp = _fp((DAY, VASCO, "Matheuzinho", "goals", 15.0))
    tips = pd.DataFrame([_tip("Matheuzinho", "goals", "Brazil Serie A", VITORIA)])
    assert of.match_odds_to_tips(tips, {"matheuzinho|goals": 15.0}, fixture_prices=fp) == {}


def test_own_fixture_best_price_across_books():
    fp = _fp((DAY, VITORIA, "Matheuzinho", "goals", 3.85), (DAY, VITORIA, "Matheuzinho", "goals", 3.90),
             (DAY, VASCO, "Matheuzinho", "goals", 15.0))
    tips = pd.DataFrame([_tip("Matheuzinho", "goals", "Brazil Serie A", VITORIA)])
    got = of.match_odds_to_tips(tips, {}, fixture_prices=fp)
    assert got == {of.tip_price_key("Matheuzinho", "goals", DAY, VITORIA): 3.90}


def test_fixture_matched_across_club_spellings_and_kickoff_date():
    # OddsAPI spells clubs differently and dates by UTC kickoff (can be the next day).
    fp = _fp(("2026-10-25", "Orlando City vs New York City", "Adrian Marin", "sot", 5.0))
    tips = pd.DataFrame([_tip("Adrián Marín", "sot", "USA MLS", ORLANDO)])
    assert list(of.match_odds_to_tips(tips, {}, fixture_prices=fp).values()) == [5.0]
    # Same clubs a week later is a different fixture.
    fp2 = _fp(("2026-10-31", ORLANDO, "Adrian Marin", "sot", 5.0))
    assert of.match_odds_to_tips(tips, {}, fixture_prices=fp2) == {}


def test_initial_surname_ambiguous_within_fixture_is_refused():
    fp = _fp((DAY, GIRONA, "Alvaro Marin", "goals", 4.0), (DAY, GIRONA, "Adrian Marin", "goals", 6.0))
    tips = pd.DataFrame([_tip("A. Marin", "goals", "La Liga 2", GIRONA)])
    assert of.match_odds_to_tips(tips, {}, fixture_prices=fp) == {}


def test_reserve_side_is_not_the_first_team_fixture():
    fp = _fp((DAY, "Girona vs Celta de Vigo", "Alvaro Marin", "goals", 4.0))
    tips = pd.DataFrame([_tip("A. Marin", "goals", "La Liga 2", GIRONA)])
    assert of.match_odds_to_tips(tips, {}, fixture_prices=fp) == {}


def test_no_fixture_map_prices_nothing(monkeypatch):
    monkeypatch.setattr(of, "_LAST_FIXTURE_PRICES", None)
    tips = pd.DataFrame([_tip("Matheuzinho", "goals", "Brazil Serie A", VITORIA)])
    assert of.match_odds_to_tips(tips, {"matheuzinho|goals": 15.0}) == {}


def test_oddsapi_no_side_is_never_the_yes_price():
    bms = [{"markets": [{"key": "player_goal_scorer_anytime", "outcomes": [
        {"name": "Yes", "description": "Erling Haaland", "price": 1.30},
        {"name": "No", "description": "Erling Haaland", "price": 3.40}]}]}]
    assert of._best_price(bms, "player_goal_scorer_anytime", None) == {"erling haaland": 1.30}


def test_clv_close_uses_own_fixture_only(monkeypatch):
    from player_model import clv_tracker
    import src.rolling_store as rs
    rec = {"bet_id": f"PLAYER|{DAY}|Matheuzinho|goals", "match": VITORIA, "odds_close": ""}
    closes = {}
    monkeypatch.setattr(clv_tracker.clv_capture, "_load", lambda: [rec])
    monkeypatch.setattr(clv_tracker.clv_capture, "capture_close",
                        lambda bid, o, u: closes.__setitem__(bid, o))
    hist = pd.DataFrame([
        {"snapshot_ts": f"{DAY}T10:00:00Z", "match_date": DAY, "match": VITORIA,
         "player": "Matheuzinho", "market": "goals", "odds": 3.90},
        {"snapshot_ts": f"{DAY}T12:00:00Z", "match_date": DAY, "match": VASCO,
         "player": "Matheuzinho", "market": "goals", "odds": 15.0},
    ])
    monkeypatch.setattr(rs, "read_all", lambda path, dates=None: hist.copy())
    assert clv_tracker.close_out() == 1
    assert closes == {rec["bet_id"]: 3.90}
