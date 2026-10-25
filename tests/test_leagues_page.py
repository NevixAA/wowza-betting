"""Leagues page + Monday Leagues picture (owner 2026-10-25): one data module, two renderers.

dashboard_data/leagues.py feeds pages/17_🏟️_Leagues.py and telegram_bot/daily_pack.leagues_image, so
the app and Telegram cannot disagree. Tested on a small synthetic ledger: numbers by market and tier,
sides not pooled, paper leagues shown but never totalled, and Telegram == page on the same input.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "17_🏟️_Leagues.py"
TODAY = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
SOON = (pd.Timestamp(TODAY) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
RECENT = (pd.Timestamp(TODAY) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")


def _main(date, league, side, tier, odds, result, pnl, edge, source="live", home="H", away="A"):
    return {"source": source, "generated_at": f"{date}T06:00:00Z", "match_date": date, "league": league,
            "home_team": home, "away_team": away, "side": side, "odds": odds, "edge_pct": edge,
            "signal_tier": tier, "result": result, "pnl": pnl, "model_type": ""}


def _side(date, league, market, tier, odds, result, pnl, edge, mp, home="H", away="A"):
    return {"source": "live", "generated_at": f"{date}T06:00:00Z", "match_date": date, "league": league,
            "home_team": home, "away_team": away, "market": market, "odds": odds, "edge_pct": edge,
            "model_prob": mp, "signal_tier": tier, "result": result, "pnl": pnl, "model_type": "new_format"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    import config
    from src import stake_plan
    from dashboard_data import core
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(stake_plan, "load_league_ratings", lambda *a, **k: {"Brazil Serie A": 9, "League One": 2})
    monkeypatch.setattr(core, "RAW_BASE", {})                         # no network in tests
    monkeypatch.setitem(core.REPOS, "pro", {**core.REPOS["pro"], "candidates": []})
    try:
        from telegram_bot import tips_table
        monkeypatch.setattr(tips_table, "_RATINGS", None)
    except Exception:                                                   # noqa: BLE001
        pass
    main = [
        _main("2026-10-01", "Brazil Serie A", "OVER", "SNIPER", 2.0, "WIN", 1.0, 10.0, home="B1"),
        _main("2026-10-02", "Brazil Serie A", "UNDER", "MARKSMAN", 2.5, "LOSS", -1.0, 6.0, home="B2"),
        _main(RECENT, "Brazil Serie A", "OVER", "VALUABLE", 2.2, "WIN", 1.2, 4.0, home="B3"),
        _main("2026-10-03", "Brazil Serie A", "OVER", "SNIPER", 2.0, "WIN", 1.0, 9.0, source="backtest", home="BX"),
        _main("2026-07-01", "Brazil Serie A", "OVER", "SNIPER", 2.0, "WIN", 1.0, 9.0, home="BOLD"),   # pre-cutoff
        _main(SOON, "Brazil Serie A", "OVER", "SNIPER", 2.1, None, None, 7.0, home="BNEXT"),          # upcoming
        _main(RECENT, "League One", "OVER", "SNIPER", 1.9, "LOSS", -1.0, 5.0, home="L1"),
        _main("2026-10-05", "USA MLS", "UNDER", "SNIPER", 3.0, "WIN", 2.0, 12.0, home="M1"),          # paper league
        _main(SOON, "USA MLS", "UNDER", "SNIPER", 3.0, None, None, 12.0, home="MNEXT"),
    ]
    side = [
        _side("2026-10-01", "Brazil Serie A", "btts", "SNIPER", 2.0, "WIN", 1.0, 8.0, 0.60, home="B1"),
        _side(RECENT, "Brazil Serie A", "over15", "MARKSMAN", 1.5, "LOSS", -1.0, 9.0, 0.75, home="B4"),
    ]
    pd.DataFrame(main).to_csv(tmp_path / "bets_ledger.csv", index=False)
    pd.DataFrame(side).to_csv(tmp_path / "side_bets_ledger.csv", index=False)
    return tmp_path


def _ov():
    from dashboard_data import leagues as L
    d = L.settled()
    return d, L.overview(d, TODAY, up=L.upcoming(TODAY, 7), rat=L.ratings(), pol=L.policy(), reg=L.regimes())


def test_rows_basis(env):
    from dashboard_data import leagues as L
    d = L.settled()
    br = d[d["league"] == "Brazil Serie A"]
    assert len(br) == 5                                  # backtest, pre-cutoff and unsettled rows are out
    assert set(br["mkt"]) == {"O/U 2.5 Over", "O/U 2.5 Under", "BTTS", "Over 1.5"}
    s = L.summarise(br[br["staked"]])
    assert (s["tips"], s["won"], s["lost"]) == (4, 2, 2)
    assert s["pl"] == pytest.approx(0.0) and s["hit"] == pytest.approx(0.5) and s["roi"] == pytest.approx(0.0)
    assert s["avg_odds"] == pytest.approx((2.0 + 2.5 + 2.0 + 1.5) / 4)
    assert s["avg_edge"] == pytest.approx((10 + 6 + 8 + 9) / 4)
    # O/U 2.5 model probability = 1/odds + edge; side markets use the recorded model_prob
    assert s["avg_model_prob"] == pytest.approx((0.6 + (0.4 + 0.06) + 0.60 + 0.75) / 4)
    assert s["avg_implied"] == pytest.approx((0.5 + 0.4 + 0.5 + 1 / 1.5) / 4)


def test_by_market_and_tier(env):
    from dashboard_data import leagues as L
    d = L.settled()
    br = d[d["league"] == "Brazil Serie A"]
    bm = L.by_market(br).set_index("market")
    assert list(bm.index) == ["O/U 2.5 Over", "O/U 2.5 Under", "BTTS", "Over 1.5"]       # sides not pooled
    assert bm.loc["O/U 2.5 Over", "tips"] == 1 and bm.loc["O/U 2.5 Over", "pl"] == pytest.approx(1.0)
    assert bm.loc["O/U 2.5 Under", "tips"] == 1 and bm.loc["O/U 2.5 Under", "pl"] == pytest.approx(-1.0)
    assert bm.loc["Over 1.5", "hit"] == 0 and bm.loc["BTTS", "avg_odds"] == pytest.approx(2.0)
    paper = L.by_market(br, ("VALUABLE",)).set_index("market")
    assert list(paper.index) == ["O/U 2.5 Over"] and paper.loc["O/U 2.5 Over", "pl"] == pytest.approx(1.2)
    bt = L.by_tier(br).set_index("tier")
    assert list(bt.index) == ["SNIPER", "MARKSMAN", "VALUABLE (paper)"]
    assert bt.loc["SNIPER", "tips"] == 2 and bt.loc["SNIPER", "pl"] == pytest.approx(2.0)
    assert bt.loc["MARKSMAN", "pl"] == pytest.approx(-2.0)
    assert bt.loc["VALUABLE (paper)", "roi"] == pytest.approx(1.2)


def test_paper_league_shown_not_totalled(env):
    import config
    from dashboard_data import leagues as L
    d, ov = _ov()
    mls = ov.set_index("league").loc["USA MLS"]
    assert bool(mls["paper"]) and mls["state"] == L.PAPER_STATE and mls["tips"] == 1
    assert ov.iloc[-1]["league"] == "USA MLS" or not ov.iloc[-1]["tips"]   # paper leagues sorted last
    t = L.totals(d)
    assert (t["tips"], t["pl"]) == (5, pytest.approx(-1.0))                # Brazil 4 + League One 1, no MLS
    # identical to the record picture's basis (daily_pack.settled_tips with paper leagues dropped)
    from telegram_bot.daily_pack import settled_tips
    r = settled_tips()
    r = r[(r["date"] >= str(config.PERFORMANCE_CUTOFF_DATE)[:10]) & r["pnl"].notna() & r["tier"].isin(L.STAKED)]
    assert len(r) == t["tips"] and r["pnl"].sum() == pytest.approx(t["pl"])
    tab = L.overview_table(ov).set_index("League")
    assert "USA MLS (paper)" in tab.index


def test_overview_context_columns(env):
    from dashboard_data import leagues as L
    _, ov = _ov()
    o = ov.set_index("league")
    assert o.loc["Brazil Serie A", "rating"] == 9 and o.loc["League One", "rating"] == 2
    assert o.loc["Brazil Serie A", "upcoming"] == 1 and o.loc["USA MLS", "upcoming"] == 0   # paper never sent
    assert o.loc["Brazil Serie A", "track"] == "new-format" and o.loc["League One", "track"] == "standard"
    # last 7 days: Over 1.5 MARKSMAN loss (-1); the VALUABLE win is paper and not in pl_7d
    assert o.loc["Brazil Serie A", "pl_7d"] == pytest.approx(-1.0)
    assert o.loc["Brazil Serie A", "valuable_pl"] == pytest.approx(1.2)
    assert L.overview_table(ov).set_index("League").loc["Brazil Serie A", "Rating"] == "9/10"


def test_telegram_equals_page_table(env):
    """The Telegram Leagues picture draws exactly the page's overview table."""
    from dashboard_data import leagues as L
    from telegram_bot import daily_pack as P
    _, ov = _ov()
    tab = L.overview_table(ov[ov["tips"] > 0])
    rows = P.league_rows(tab)
    cols = [c for c, _ in P.LEAGUE_COLUMNS]
    assert len(rows) == len(tab)
    for r, t in zip(rows, tab.to_dict("records")):
        cell = dict(zip(cols, r["cells"]))
        assert cell["League"] == t["League"] and cell["Rating"] == t["Rating"]
        assert cell["Tips"] == str(t["Tips"]) and cell["W-L"] == t["W-L"]
        assert cell["P/L"] == f"{t['P/L (u)']:+.2f}u" and cell["Last 7d"] == f"{t['P/L 7d (u)']:+.2f}u"
        assert cell["ROI"] == f"{t['ROI %']:+.0f}%" and cell["Hit"] == f"{t['Hit %']:.0f}%"
        assert cell["Avg odds"] == f"{t['Avg odds']:.2f}" and cell["Avg edge"] == f"{t['Avg edge %']:.1f}%"
        assert cell["Best market"] == t["Best market"]
    pngs = P.leagues_image(TODAY)
    assert pngs and pngs[0][:8] == b"\x89PNG\r\n\x1a\n"
    wk = P.last_week_leagues_image(TODAY)
    assert wk and wk[0][:8] == b"\x89PNG\r\n\x1a\n"
    lw = L.last_week(L.settled(), TODAY).set_index("league")
    assert set(lw.index) == {"Brazil Serie A", "League One"}             # only leagues with tips last week
    assert lw.loc["Brazil Serie A", "tips"] == 1 and lw.loc["Brazil Serie A", "valuable_pl"] == pytest.approx(1.2)


def test_page_renders_same_table(env):
    """Render the page headlessly and compare its overview table with the module's (and so Telegram's)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit as st
    from dashboard_data import leagues as L
    st.cache_data.clear()
    at = testing.AppTest.from_file(str(PAGE), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    shown = at.dataframe[0].value
    _, ov = _ov()
    want = L.overview_table(ov)
    assert list(shown["League"]) == list(want["League"])
    for c in ("Tips", "P/L (u)", "ROI %", "Hit %", "Avg odds", "Avg edge %", "P/L 7d (u)", "Upcoming"):
        pd.testing.assert_series_equal(shown[c].reset_index(drop=True), want[c].reset_index(drop=True),
                                       check_names=False, check_dtype=False)
    at.selectbox(key="lg_pick").set_value("USA MLS").run()
    assert not at.exception, [e.value for e in at.exception]
    st.cache_data.clear()


def test_page_uses_shared_table_and_is_registered():
    src = PAGE.read_text(encoding="utf-8")
    assert "leagues.overview_table(ov)" in src and '_require_tier("pro")' in src
    import dashboard_access as A
    assert any(p == "pages/17_🏟️_Leagues.py" and need == "pro" for p, *_, need in A.PAGES)


def test_page_cards_and_profile_carry_module_values(env):
    """Visual redesign 2026-10-25 (stage 2b): the headline card, league cards and the profile's three cards
    show the module's numbers (same strings the old ten-metric layout showed)."""
    testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit as st
    import dashboard_ui as ui
    from dashboard_data import leagues as L
    from dashboard_data import leagues_view as V
    st.cache_data.clear()
    at = testing.AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    d, ov = _ov()
    tot = L.totals(d)
    metrics = [(m.label, m.value) for m in at.metric]
    assert ("P/L", ui.units(tot["pl"])) == metrics[0]                    # the headline card
    cards = {r["league"]: r for r in ov.to_dict("records") if r["tips"] > 0 and not r["paper"]}
    staked = [v for lab, v in metrics if lab == "P/L (staked)"]
    assert staked == [ui.units(cards[k]["pl"]) for k in cards]           # one card per league, overview order
    at.selectbox(key="lg_pick").set_value("Brazil Serie A").run()
    assert not at.exception, [e.value for e in at.exception]
    row = ov.set_index("league").loc["Brazil Serie A"]
    m = [(x.label, x.value) for x in at.metric]
    assert ("P/L", ui.units(row["pl"])) in m[1:] and ("Rating", V.rating(row["rating"])) in m
    assert ("Avg odds", V.fmt_num(row["avg_odds"])) in m
    md = " ".join(x.value for x in at.markdown)
    assert V.fmt_roi(row["roi"]) in md and V.fmt_pct(row["hit"], 1) in md
    st.cache_data.clear()
