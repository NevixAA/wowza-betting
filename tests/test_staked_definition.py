"""ONE definition of "Staked" across the dashboard (owner 2026-11-04; supersedes the 2026-11-02 tier basis).

Staked = the bankroll policy RECORDED a stake > 0 for the tip on a real-money track (src/staked.py:
stake_audit.csv / stake_plan.csv, deployment_state in REAL_STATES, base_fraction > 0), plus the
viewer clauses: source == live, settled, match date >= PERFORMANCE_CUTOFF_DATE, outside
config.PAPER_LEAGUES, never a player prop. dashboard_data.today_view owns the viewer side
(is_staked / staked_settled); Home, Tips, Leagues, Overview and Success Rates read it. A tier is not
a stake: SNIPER on a PAPER track is NOT staked; VALUABLE with a recorded stake IS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import staked as S  # noqa: E402
from src import units as U  # noqa: E402


def _rec(home, away, state="ACTIVE", base=0.0125, amount=1.0, market="BTTS", side="YES",
         model="new_format", date="2026-10-20", gen="2026-10-19 08:00:00", pv="pv1", reason="OK"):
    return dict(date=date, fixture=f"{date}|{home}|{away}", league="Championship", model_type=model,
                market=market, side=side, deployment_state=state, base_fraction=base, stake_amount=amount,
                final_fraction=amount / 100, reason=reason, policy_version=pv, generated_at=gen)


def test_is_staked_applies_every_clause():
    import config
    from dashboard_data.today_view import is_staked
    cut = pd.Timestamp(config.PERFORMANCE_CUTOFF_DATE)
    paper = sorted(config.PAPER_LEAGUES)[0]
    ok = dict(status=S.STAKED, source="live", dt=cut, league="Championship", ledger="bets_ledger.csv", pnl=0.9)
    rows = [ok,
            dict(ok, status=S.NOT_STAKED),                   # recorded, zero stake (e.g. PAPER track)
            dict(ok, status=S.PRE_POLICY),                   # before the policy: no stake to prove
            dict(ok, status=S.NO_RECORD),                    # UNIDENTIFIABLE: never assumed staked
            dict(ok, source="backtest"),                     # not live
            dict(ok, dt=cut - pd.Timedelta(days=1)),         # before the cutoff
            dict(ok, dt=pd.NaT),                             # no match date -> not countable
            dict(ok, league=paper),                          # paper league
            dict(ok, ledger="player_ledger.csv"),            # props are paper-only
            dict(ok, pnl=float("nan"))]                      # unsettled
    f = pd.DataFrame(rows)
    m = is_staked(f["status"], source=f["source"], dt=f["dt"], league=f["league"], ledger=f["ledger"], pnl=f["pnl"])
    assert m.tolist() == [True] + [False] * 9


def test_overview_staked_uses_the_definition(monkeypatch):
    """v9.performance staked_only = recorded stake, not tier: a SNIPER on a PAPER track is out, a
    VALUABLE with a recorded stake is in; flat-1u pnl and P/L at the recorded stake are both carried."""
    import config
    from dashboard_data import v9
    cut = pd.Timestamp(config.PERFORMANCE_CUTOFF_DATE, tz="UTC")
    md = pd.Timestamp("2026-10-20", tz="UTC")
    assert md >= cut
    base = dict(league="Championship", model_type="new_format", signal_tier="SNIPER", result="WIN", market="btts",
                side="YES", pnl=0.9, source="live", match_date=md, generated_at=pd.Timestamp("2026-10-19", tz="UTC"),
                odds=1.9)
    led = pd.DataFrame([dict(base, home_team="A", away_team="B"),                         # staked 2u
                        dict(base, home_team="C", away_team="D", pnl=-1.0),               # PAPER track
                        dict(base, home_team="E", away_team="F", signal_tier="VALUABLE", pnl=1.1),  # staked 0.5u
                        dict(base, home_team="G", away_team="H", source="backtest", pnl=5.0)])
    rec = S.records_from(pd.DataFrame([_rec("A", "B", amount=2.0), _rec("C", "D", state="PAPER", base=0.0, amount=0.0,
                                                                      reason="NO_REAL_STAKE_PAPER"),
                                       _rec("E", "F", amount=0.5), _rec("G", "H")]))
    monkeypatch.setattr(v9, "ledger", lambda *a, **k: led.copy())
    monkeypatch.setattr(S, "load_records", lambda *a, **k: rec)
    s = v9.performance()["by_track"]["new_format"]["staked_only"]
    assert (s["n"], s["pnl"]) == (2, 2.0)
    assert s["stake_units"] == 2.5 and s["stake_pnl"] == round(2.0 * 0.9 + 0.5 * 1.1, 2)


def test_committed_pages_agree_on_staked():
    """Measured on the committed ledgers + stake records: Home/Tips (today_view), Leagues
    (leagues.totals) and Overview (v9.performance) give the same staked n and P/L."""
    from dashboard_data import leagues as L, today_view, v9
    st = today_view.staked_settled()
    home = (len(st), U.fmt(U.total(st["_pnl"])))
    tot = L.totals(L.settled())
    leagues = (tot["tips"], U.fmt(tot["pl"]))
    perf = v9.performance()
    tr = [t["staked_only"] for k, t in perf.get("by_track", {}).items() if k != "unknown"]
    overview = (sum(t["n"] for t in tr), U.fmt(U.total(t["pnl"] for t in tr)))
    assert home == leagues == overview, (home, leagues, overview)


def test_committed_staked_rows_all_carry_a_recorded_stake():
    """No staked row without a recorded stake > 0 - the invariant the old tier basis broke."""
    from dashboard_data import today_view
    st = today_view.staked_settled()
    assert (st["_stake_status"] == S.STAKED).all()
    assert (pd.to_numeric(st["_stake_units"], errors="coerce") > 0).all()


def test_success_rates_staked_scope_uses_the_definition():
    src = next((ROOT / "pages").glob("5_*Success_Rates.py")).read_text(encoding="utf-8")
    assert "_is_staked(d[\"_stake_status\"], source=d[\"_live\"], dt=d[\"_date\"], league=d[\"_league\"]," in src
    assert "ledger=d[\"_ledger\"]" in src
    assert 'd = d[d["_tier"].isin(STAKED)]' not in src            # the old tier-only filter took props


def test_tips_page_reads_the_one_staked_reader():
    src = next((ROOT / "pages").glob("1_*Dashboard.py")).read_text(encoding="utf-8")
    assert "_TV.staked_settled()" in src and "_MKT_SOURCES" not in src
    assert 'board["tier"].isin(["SNIPER", "MARKSMAN"])' not in src


def test_portfolio_does_not_call_staked_side_markets_paper():
    from dashboard_data import portfolio_view as V
    assert V.MARKET_STATE["Side/BTTS"][0] != "PAPER"
    assert V.MARKET_STATE["Player Props"][0] == "PAPER"


def test_no_page_or_view_decides_staked_by_tier():
    """Every 'staked' decision goes through src/staked.py. These are the tier filters that used to
    define it; none may come back in dashboard_data/ or pages/."""
    bad = ['["tier"].isin(STAKED)', '["signal_tier"].isin(STAKED)', 'd["staked"] = d["tier"]',
           '.isin(("SNIPER", "MARKSMAN"))', 'final_fraction"] > 0]']
    hits = []
    for p in list((ROOT / "dashboard_data").glob("*.py")) + list((ROOT / "pages").glob("*.py")):
        t = p.read_text(encoding="utf-8")
        hits += [f"{p.name}: {b}" for b in bad if b in t]
    assert not hits, hits
