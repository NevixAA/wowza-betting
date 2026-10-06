"""The runner. Every test here is about not staking twice and not staking blind."""
from __future__ import annotations

import pandas as pd
import pytest

import src.cloudbet.bot as bot
from src.cloudbet.selection import SelectionRules
from src.cloudbet.staking import StakingRules


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "LEDGER", tmp_path / "cloudbet_bets.csv")
    return bot.LEDGER


def _cands(n=3, market="ou25", side="OVER"):
    return pd.DataFrame([{
        "fixture_id": f"fx{i}", "match_key": f"m{i}", "league": "Argentina Primera Division",
        "home_team": f"H{i}", "away_team": f"A{i}", "kickoff_utc": "2026-10-07T18:00:00Z",
        "market": market, "side": side, "edge": 0.20 - i / 100,
        "signal_tier": "SNIPER", "model_type": "new_format", "our_odds": 2.00,
        "selection_label": f"H{i} v A{i}", "bet_kind": "team",
        "settlement_status": "ALIGNED", "date": "2026-10-07",
    } for i in range(n)])


def _events(n=3):
    return {"Argentina Primera Division": [
        {"id": f"ev{i}", "home": {"name": f"H{i}"}, "away": {"name": f"A{i}"},
         "markets": {"soccer.total_goals": {"submarkets": {"total=2.5": {"selections": [
             {"outcome": "over", "price": 2.10, "status": "SELECTION_ENABLED"},
             {"outcome": "under", "price": 1.80, "status": "SELECTION_ENABLED"}]}}}}}
        for i in range(n)]}


def _run(monkeypatch, ledger, cands=None, events=None, **kw):
    monkeypatch.setattr(bot, "build_candidates", lambda **_: (cands if cands is not None
                                                              else _cands()))
    return bot.run(mode="DRY", bankroll=2000.0,
                   events_by_league=events if events is not None else _events(), **kw)


# ── pricing happens BEFORE selection ──────────────────────────────────────────────────────────
def test_edge_is_recomputed_against_cloudbets_price(monkeypatch, ledger):
    """Our 0.20 edge was against odds 2.00. Cloudbet is 2.10, a better price, so the edge GROWS
    by the difference in implied probability — and that is the number the cap sorts on."""
    out = _run(monkeypatch, ledger)
    assert len(out) == 3
    shift = 1 / 2.00 - 1 / 2.10
    assert out.iloc[0]["edge_at_cloudbet_price"] == pytest.approx(0.20 + shift, abs=1e-6)
    assert out.iloc[0]["edge_at_our_price"] == pytest.approx(0.20)


def test_a_worse_cloudbet_price_shrinks_the_edge(monkeypatch, ledger):
    ev = _events()
    for e in ev["Argentina Primera Division"]:
        e["markets"]["soccer.total_goals"]["submarkets"]["total=2.5"]["selections"][0]["price"] = 1.80
    out = _run(monkeypatch, ledger, events=ev)
    assert out.iloc[0]["edge_at_cloudbet_price"] < out.iloc[0]["edge_at_our_price"]


def test_nothing_is_bet_without_a_cloudbet_price(monkeypatch, ledger):
    """We do not bet into a price we cannot see."""
    assert len(_run(monkeypatch, ledger, events={})) == 0


def test_an_unmatched_fixture_is_not_bet(monkeypatch, ledger):
    ev = {"Argentina Primera Division": [
        {"id": "x", "home": {"name": "Boca Juniors"}, "away": {"name": "River Plate"},
         "markets": {}}]}
    assert len(_run(monkeypatch, ledger, events=ev)) == 0


# ── the daily ledger is the real cap ──────────────────────────────────────────────────────────
def test_the_cap_counts_bets_already_placed_today(monkeypatch, ledger):
    prior = pd.DataFrame([{
        "placed_at": pd.Timestamp.now(tz="UTC").isoformat(), "accepted": True,
        "reference_id": "old", "fixture_id": "other", "stake": 50.0}])
    for c in bot.LEDGER_COLUMNS:
        if c not in prior.columns:
            prior[c] = pd.NA
    prior[bot.LEDGER_COLUMNS].to_csv(ledger, index=False)
    out = _run(monkeypatch, ledger, cands=_cands(5),
               rules=SelectionRules(max_bets_per_day=3))
    assert len(out) == 2, "the cap ignored what was already placed today"


def test_a_fixture_already_bet_today_is_never_bet_again(monkeypatch, ledger):
    prior = pd.DataFrame([{
        "placed_at": pd.Timestamp.now(tz="UTC").isoformat(), "accepted": True,
        "reference_id": "r", "fixture_id": "fx0", "stake": 50.0}])
    for c in bot.LEDGER_COLUMNS:
        if c not in prior.columns:
            prior[c] = pd.NA
    prior[bot.LEDGER_COLUMNS].to_csv(ledger, index=False)
    out = _run(monkeypatch, ledger)
    assert "fx0" not in set(out["fixture_id"])


def test_a_rejected_bet_does_not_consume_a_daily_slot():
    """A rejection cost no money. Counting it would silently shrink the day's capacity."""
    led = pd.DataFrame([{"placed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                         "accepted": False, "stake": 50.0, "fixture_id": "f"}])
    for c in bot.LEDGER_COLUMNS:
        if c not in led.columns:
            led[c] = pd.NA
    assert len(bot._placed_today(led)) == 0


def test_a_dry_run_still_leaves_an_audit_trail(monkeypatch, ledger):
    _run(monkeypatch, ledger)
    d = pd.read_csv(ledger)
    assert len(d) == 3 and set(d["mode"]) == {"DRY"} and not d["accepted"].any()


def test_the_ledger_records_both_prices_and_the_settlement_status(monkeypatch, ledger):
    """Without our price beside theirs, nobody can later ask how much edge execution cost."""
    _run(monkeypatch, ledger)
    d = pd.read_csv(ledger)
    for c in ("our_odds", "cloudbet_odds", "edge_at_our_price", "edge_at_cloudbet_price",
              "settlement_status", "reference_id"):
        assert d[c].notna().all(), c


# ── staking ───────────────────────────────────────────────────────────────────────────────────
def test_the_daily_exposure_cap_stops_the_run(monkeypatch, ledger):
    """20 bets at 4.5% is 90% of bankroll in one day, and slates correlate."""
    out = _run(monkeypatch, ledger, cands=_cands(20),
               rules=SelectionRules(max_bets_per_day=20),
               staking=StakingRules(fraction=0.045, max_fraction_per_day=0.10))
    assert 0 < len(out) <= 3
    assert pd.to_numeric(out["stake"]).sum() <= 2000 * 0.10 + 1e-6
