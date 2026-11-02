"""ONE definition of "Staked" across the dashboard (estate scan 2026-11-02, P1 dashboard items).

Staked = SNIPER + MARKSMAN, source == live, settled, match date >= PERFORMANCE_CUTOFF_DATE, outside
config.PAPER_LEAGUES, never a player prop (props are paper-only). dashboard_data.today_view owns it
(is_staked / staked_settled); Home, Tips, Leagues, Overview and Success Rates read it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import units as U  # noqa: E402


def test_is_staked_applies_every_clause():
    import config
    from dashboard_data.today_view import is_staked
    cut = pd.Timestamp(config.PERFORMANCE_CUTOFF_DATE)
    paper = sorted(config.PAPER_LEAGUES)[0]
    ok = dict(tier="SNIPER", source="live", dt=cut, league="Championship", ledger="bets_ledger.csv", pnl=0.9)
    rows = [ok,
            dict(ok, tier="VALUABLE"),                       # paper tier
            dict(ok, source="backtest"),                     # not live
            dict(ok, dt=cut - pd.Timedelta(days=1)),         # before the cutoff
            dict(ok, dt=pd.NaT),                             # no match date -> not countable
            dict(ok, league=paper),                          # paper league
            dict(ok, ledger="player_ledger.csv"),            # props are paper-only
            dict(ok, pnl=float("nan")),                      # unsettled
            dict(ok, tier="MARKSMAN")]
    f = pd.DataFrame(rows)
    m = is_staked(f["tier"], source=f["source"], dt=f["dt"], league=f["league"], ledger=f["ledger"], pnl=f["pnl"])
    assert m.tolist() == [True, False, False, False, False, False, False, False, True]


def test_overview_staked_uses_the_definition(monkeypatch):
    """v9.performance (Overview 'Staked P/L') counted WIN/LOSS rows of any source and kept rows with no
    match date; Home counted live rows with a date. Same ledgers, two different 'Staked'."""
    import config
    from dashboard_data import v9
    cut = pd.Timestamp(config.PERFORMANCE_CUTOFF_DATE, tz="UTC")
    base = dict(league="Championship", model_type="standard", signal_tier="SNIPER", result="WIN",
                pnl=0.9, source="live", match_date=cut + pd.Timedelta(days=3), generated_at=cut, odds=1.9)
    led = pd.DataFrame([base,
                        dict(base, source="backtest", pnl=5.0),           # not live
                        dict(base, match_date=pd.NaT, pnl=7.0),           # no match date
                        dict(base, result="VOID", pnl=0.0),               # settled void: a staked bet
                        dict(base, signal_tier="VALUABLE", pnl=-1.0)])    # paper tier
    monkeypatch.setattr(v9, "ledger", lambda *a, **k: led.copy())
    s = v9.performance()["by_track"]["standard"]["staked_only"]
    assert (s["n"], s["pnl"]) == (2, 0.9)


def test_committed_pages_agree_on_staked():
    """Measured on the committed ledgers: Home/Tips (today_view), Leagues (leagues.totals) and
    Overview (v9.performance) give the same staked n and P/L."""
    from dashboard_data import leagues as L, today_view, v9
    st = today_view.staked_settled()
    home = (len(st), U.fmt(U.total(st["_pnl"])))
    tot = L.totals(L.settled())
    leagues = (tot["tips"], U.fmt(tot["pl"]))
    perf = v9.performance()
    tr = [t["staked_only"] for k, t in perf.get("by_track", {}).items() if k != "unknown"]
    overview = (sum(t["n"] for t in tr), U.fmt(U.total(t["pnl"] for t in tr)))
    assert home == leagues == overview, (home, leagues, overview)


def test_success_rates_staked_scope_uses_the_definition():
    src = next((ROOT / "pages").glob("5_*Success_Rates.py")).read_text(encoding="utf-8")
    assert "_is_staked(d[\"_tier\"], source=d[\"_live\"], dt=d[\"_date\"], league=d[\"_league\"]," in src
    assert "ledger=d[\"_ledger\"]" in src
    assert 'd = d[d["_tier"].isin(STAKED)]' not in src            # the old tier-only filter took props


def test_tips_page_reads_the_one_staked_reader():
    src = next((ROOT / "pages").glob("1_*Dashboard.py")).read_text(encoding="utf-8")
    assert "_TV.staked_settled()" in src and "_MKT_SOURCES" not in src


def test_portfolio_does_not_call_staked_side_markets_paper():
    from dashboard_data import portfolio_view as V
    assert V.MARKET_STATE["Side/BTTS"][0] != "PAPER"
    assert V.MARKET_STATE["Player Props"][0] == "PAPER"
