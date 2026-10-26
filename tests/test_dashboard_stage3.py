"""Non-regression for the 2026-10-25 visual redesign, stage 3 (every remaining page).

The redesign moved values into cards, replaced hand-built HTML / plotly with the shared components and
changed pre-formatted strings into numbers formatted at render time. It must not change any value. Each
test either renders a page headlessly (streamlit.testing AppTest, ABSOLUTE path, Pro / V11 unreachable so
no network is touched) and compares what it shows with the computation the page always used, or pins a
view helper against the exact expression the pre-redesign page used (copied here verbatim as `_old_*`).
"""
from __future__ import annotations

import html as _html
import re
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "pages"
P = {k: next(PAGES.glob(f"{k}_*.py")) if not (PAGES / f"{k}.py").exists() else PAGES / f"{k}.py"
     for k in ("12", "16", "11", "14", "5", "9", "4", "7", "8", "15", "10", "19", "20", "21", "22",
               "account", "free_tips")}


def _offline(monkeypatch):
    """Pro and V11 unreachable: no local checkout, no HTTP (tests never touch the network)."""
    from dashboard_data import core
    monkeypatch.setattr(core, "RAW_BASE", {})
    for k in ("pro", "v11"):
        monkeypatch.setitem(core.REPOS, k, {**core.REPOS[k], "candidates": []})


def _render(page: Path, monkeypatch, tier: str = "admin", timeout: int = 180):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", tier)
    st.cache_data.clear()
    at = AppTest.from_file(str(page.resolve()), default_timeout=timeout).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _metrics(at) -> list[tuple[str, str]]:
    return [(m.label, m.value) for m in at.metric]


def _text(at) -> str:
    return "\n".join([x.value for x in at.markdown] + [x.value for x in at.caption]
                     + [x.value for x in at.title] + [x.value for x in at.subheader])


# ══ Portfolio ═════════════════════════════════════════════════════════════════════════════════════
def _portfolio_defs(monkeypatch):
    """The page's own loaders + normalize() + simulate(), executed from the page source (everything above
    the UI section) — so the test computes with exactly the code the page runs."""
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    src = P["12"].read_text(encoding="utf-8")
    ns = {"__file__": str(P["12"]), "__name__": "portfolio_defs"}
    exec(compile(src.split("# ── UI")[0], str(P["12"]), "exec"), ns)        # noqa: S102 - our own page source
    return ns


def test_portfolio_cards_and_tables_equal_the_simulation(monkeypatch):
    import config
    ns = _portfolio_defs(monkeypatch)
    from dashboard_data import portfolio_view as V
    parts = [d for d in (ns["normalize"](ns["load_main_ledger"](), "main"),
                         ns["normalize"](ns["load_player_ledger"](), "player"),
                         ns["normalize"](ns["load_side_ledger"](), "side")) if d is not None and not d.empty]
    if not parts:
        pytest.skip("no ledgers in this checkout")
    allbets = pd.concat(parts, ignore_index=True)
    allbets = allbets[allbets["source"] == "live"]                                # default: Live only
    allbets = allbets[allbets["generated_at"].astype(str) >= config.PERFORMANCE_CUTOFF_DATE]   # post-fix default
    tf = {"SNIPER": 0.125, "MARKSMAN": 0.0625, "VALUABLE": 0.0}
    res = {m: ns["simulate"](allbets[allbets["market"] == m], 1000, tf, 5.0, "Fractional Kelly")
           for m in ("Standard O/U", "New-Format", "Player Props", "Side/BTTS")}
    at = _render(P["12"], monkeypatch)
    finals = [v for lab, v in _metrics(at) if lab == "Final bankroll"]
    want = [V.headline(r[1])["final"] for r in res.values() if r is not None]
    assert finals == want
    text = _text(at)
    for r in res.values():
        if r is not None:
            h = V.headline(r[1])
            for k in ("roi", "bets", "win", "max_dd", "weeks", "best", "worst"):
                assert h[k] in text or any(h[k] == m.value or h[k] in str(getattr(m, "delta", ""))
                                           for m in at.metric), (k, h[k])
    # every weekly table is weekly_frame() of that market's simulation, cell for cell
    weekly = [d.value for d in at.dataframe if list(d.value.columns) == ["week", "bets", "staked", "pnl", "bank",
                                                                        "drawdown"]]
    for got, r in zip(weekly, [r for r in res.values() if r is not None]):
        pd.testing.assert_frame_equal(got.reset_index(drop=True), V.weekly_frame(r[0]).reset_index(drop=True),
                                      check_dtype=False)


def test_portfolio_frames_equal_the_old_expressions():
    from dashboard_data import portfolio_view as V
    wk = pd.to_datetime(["2026-09-07", "2026-09-14", "2026-09-21"])
    eq = pd.DataFrame({"week": wk, "bets": [3, 0, 5], "staked": [12.345, 0.0, 30.04], "pnl": [-4.44, 0.0, 7.777],
                       "bank": [995.6, 995.6, 1003.4], "drawdown": [-0.0044, -0.0044, 0.0]})
    old = eq.assign(week=eq["week"].dt.strftime("%Y-%m-%d"), staked=eq["staked"].round(1), pnl=eq["pnl"].round(1),
                    bank=eq["bank"].round(0), drawdown=(eq["drawdown"] * 100).round(1),
                    )[["week", "bets", "staked", "pnl", "bank", "drawdown"]]
    pd.testing.assert_frame_equal(V.weekly_frame(eq), old)
    # the equity curve is the old st.line_chart series: seeded one week earlier at the starting bank
    ef = V.equity_frame(eq, 1000)
    curve = eq.set_index("week")["bank"]
    curve.loc[curve.index.min() - pd.Timedelta(days=7)] = 1000
    assert ef["bank"].tolist() == curve.sort_index().tolist()
    s = {"final": 1234.5, "roi": 0.2345, "n_bets": 17, "win_pct": 0.4706, "best_week": 88.2, "worst_week": -41.9,
         "max_dd": -0.0712, "weeks": 6}
    assert V.headline(s) == {"final": "1,234U", "roi": "+23.4%", "bets": "17", "win": "47% win", "best": "+88U",
                             "worst": "-42U", "max_dd": "-7.1%", "weeks": "6 wks"}


# ══ Bankroll ══════════════════════════════════════════════════════════════════════════════════════
def _old_pct(x):
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):.2f}%"


def test_bankroll_track_rows_equal_the_old_pct_strings():
    import json
    from dashboard_data import bankroll_view as V
    prod = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
    keys = list(prod["tracks"])
    rec = {"tracks": {keys[0]: {"base_fraction": 0.0075}}}
    dec = pd.DataFrame([{"track": keys[0], "decision": "KEEP", "n": 412, "matchdays": 31, "roi": 0.0312,
                         "shrunk_roi": 0.0207, "reasons": "x" * 300}])
    new = V.track_rows(prod, rec, dec)
    for _, r in new.iterrows():
        k = r["Track"].replace(" · ", "|")
        t = prod["tracks"][k]
        assert V.pct(None if r["Base (production)"] is None else r["Base (production)"] / 100) == _old_pct(t.get("base_fraction"))
    r0 = new.iloc[0]
    assert _old_pct(r0["Recommended"] / 100) == "0.75%" and _old_pct(r0["ROI"] / 100) == "3.12%"
    assert _old_pct(r0["Shrunk ROI"] / 100) == "2.07%" and r0["n"] == 412 and len(r0["Why"]) == 160
    assert pd.isna(new.iloc[1]["Recommended"]) and pd.isna(new.iloc[1]["Decision"])
    assert _old_pct(new.iloc[1]["Recommended"]) == "—"                         # a missing value still reads '—'


def test_bankroll_plan_rows_equal_the_old_arithmetic():
    from dashboard_data import bankroll_view as V
    sp = pd.DataFrame({"date": ["2026-10-25", "2026-10-25", "2026-10-26", "2026-10-26"],
                       "fixture": list("abcd"), "league": ["L"] * 4, "track": ["t"] * 4,
                       "signal_tier": ["SNIPER"] * 4, "odds": [1.9, 2.0, 2.1, 1.8],
                       "final_fraction": [0.0125, 0.005, 0.0, 0.0075]})
    show, by_day = V.plan_rows(sp)
    staked = sp[sp["final_fraction"] > 0]
    old_by_day = (staked.groupby("date")["final_fraction"].sum() * 100).round(2).rename("% of bankroll").reset_index()
    pd.testing.assert_frame_equal(by_day[["date", "% of bankroll"]], old_by_day)
    assert by_day["tips"].tolist() == [2, 1]
    assert show["stake % of bankroll"].tolist() == [1.25, 0.5, 0.75]


def test_bankroll_page_offline_shows_the_stake_plan(monkeypatch):
    _offline(monkeypatch)
    at = _render(P["16"], monkeypatch, tier="pro")
    sp = pd.read_csv(ROOT / "output" / "stake_plan.csv")
    m = dict(_metrics(at))
    assert m["With a stake"] == f"{int((sp['final_fraction'] > 0).sum()):,}"
    assert f"{len(sp):,}" in _text(at)                                       # Tips sized, now a supporting figure
    assert "NO DATA" in _text(at)                                            # no Pro review offline: empty state


# ══ Player Props ══════════════════════════════════════════════════════════════════════════════════
def _old_board(d: pd.DataFrame, priced_view: bool, limit: int) -> str:
    """pages/11_Player_Props._board before the redesign, verbatim apart from the CSS/colour constants."""
    MARKET_LABEL = {"goals": "⚽ Anytime scorer", "sot": "🎯 1+ shot on target", "assists": "🅰️ Assist"}
    tz = ZoneInfo("Asia/Jerusalem")
    d = d.copy()
    ko = pd.to_datetime(d.get("kickoff_utc"), errors="coerce", utc=True)
    d["_local"] = ko.dt.tz_convert(tz)
    d["_day"] = d["_local"].dt.strftime("%Y-%m-%d").fillna(d["date"].dt.strftime("%Y-%m-%d"))
    if priced_view:
        d["_edge"] = (d["model_prob"] - 1 / d["market_odds"]) * 100
        d = d.sort_values(["_day", "_edge"], ascending=[True, False])
    else:
        d = d.sort_values(["_day", "model_prob"], ascending=[True, False])
    shown = d.groupby("_day", group_keys=False).head(limit)
    out = []
    for day, g in shown.groupby("_day", sort=True):
        for _, r in g.iterrows():
            kot = r["_local"].strftime("%H:%M") if pd.notna(r["_local"]) else "–"
            player = f"<b>{_html.escape(str(r['player_name']))}</b>"
            model = f"<b>{r['model_prob'] * 100:.0f}%</b>"
            if priced_view:
                e = r["_edge"]
                cells = (f"<td class='num'><b>{r['market_odds']:.2f}</b></td>"
                         f"<td class='num'>{e:+.1f} pts</td><td>{r.get('tier', '')}</td>")
            else:
                fair = r.get("fair_odds")
                cells = f"<td class='num'>{'–' if pd.isna(fair) else f'{float(fair):.2f}'}</td>"
            out.append(f"<tr><td class='ko'>{kot}</td><td>{player}</td><td>{MARKET_LABEL.get(r['market'], r['market'])}"
                       f"</td><td class='num'>{model}</td>{cells}</tr>")
    return "\n".join(out)


def _props_frame():
    rng = np.random.default_rng(3)
    now = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    rows = []
    for i in range(40):
        ko = now + pd.Timedelta(hours=int(rng.integers(0, 60)))
        priced = i % 3 != 0
        rows.append({"date": ko.tz_convert(None).normalize(), "kickoff_utc": ko.isoformat() if i % 7 else None,
                     "league": "L", "match": f"H{i} vs A{i}", "player_name": f"Player {i}", "team": "T",
                     "position": "F", "market": ["goals", "sot", "assists"][i % 3],
                     "model_prob": float(rng.uniform(0.1, 0.8)),
                     "fair_odds": float(rng.uniform(1.2, 9)), "market_odds": float(rng.uniform(1.5, 6)) if priced else np.nan,
                     "ev": float(rng.uniform(-0.3, 0.3)) if priced else np.nan, "tier": "PAPER"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("priced_view,limit", [(True, 60), (False, 25), (True, 3)])
def test_props_board_rows_equal_the_old_html_board(priced_view, limit):
    from dashboard_data import props_view as V
    d = _props_frame()
    d = d[d["market_odds"].notna()] if priced_view else d[d["market_odds"].isna()]
    old = re.findall(r"<tr><td class='ko'>(.*?)</td><td><b>(.*?)</b></td><td>(.*?)</td><td class='num'><b>(\d+)%</b>"
                     r"</td>(.*?)</tr>", _old_board(d, priced_view, limit))
    rows, days = V.board(d, priced_view, limit)
    assert len(old) == len(rows) and sum(x["shown"] for x in days) == len(rows)
    for o, (_, r) in zip(old, rows.iterrows()):
        assert o[0] == r["Kickoff"] and _html.unescape(o[1]) == r["Player"] and o[2] == r["Market"]
        assert o[3] == f"{r['Model %']:.0f}"
        if priced_view:
            odds, edge = re.findall(r"<b>([\d.]+)</b>", o[4])[0], re.findall(r"([+\-][\d.]+) pts", o[4])[0]
            assert odds == f"{r['Book odds']:.2f}" and edge == f"{r['Edge (pts)']:+.1f}"
        else:
            fair = re.findall(r"<td class='num'>(.*?)</td>", o[4])[0]
            assert fair == ("–" if pd.isna(r["Fair odds"]) else f"{r['Fair odds']:.2f}")


def test_props_page_renders_the_same_headline(monkeypatch):
    at = _render(P["11"], monkeypatch, tier="pro")
    m = dict(_metrics(at))
    if "Upcoming props" not in m:
        pytest.skip("no upcoming props in this checkout")
    n = lambda s: int(str(s).replace(",", ""))  # noqa: E731
    assert n(m["Upcoming props"]) == n(m["Priced by a bookmaker"]) + n(m["Model only (no price)"])
    assert "<table" not in _text(at) and "pp-wrap" not in _text(at)         # the hand-built HTML board is gone
    assert len(at.dataframe) >= 1                                          # boards are sortable dataframes


# ══ Bet Builder ═══════════════════════════════════════════════════════════════════════════════════
def test_builder_settled_summary_equals_the_old_arithmetic():
    from dashboard_data import builder_view as V
    settled = pd.DataFrame({"combo_result": ["WON", "LOST", "LOST", "VOID", "UNKNOWN", "WON"],
                            "fair_odds": [2.5, 3.0, 1.8, 2.2, 4.0, 1.6],
                            "joint_probability": [0.4, 0.33, 0.55, 0.45, 0.25, 0.62]})
    s = V.settled_summary(settled)
    r = settled["combo_result"]
    odds = pd.to_numeric(settled.get("fair_combo_odds", settled.get("fair_odds")), errors="coerce")
    pnl = float(((odds - 1.0).where(r == "WON", -1.0).where(r != "VOID", 0.0).where(r != "UNKNOWN", 0.0)).sum())
    exp = pd.to_numeric(settled.loc[r.isin(["WON", "LOST"]), "joint_probability"], errors="coerce").mean()
    assert (s["won"], s["lost"], s["decided"], s["void"], s["unknown"]) == (2, 2, 4, 1, 1)
    assert s["pnl"] == pnl and s["exp"] == exp


def test_builder_dependence_frame_is_the_old_ten_pairs():
    from dashboard_data import builder_view as V
    rng = np.random.default_rng(1)
    dep = pd.DataFrame({"segment": ["ALL"] * 14 + ["L1"] * 3, "market_a": list("abcdefghijklmnopq"),
                        "market_b": list("qponmlkjihgfedcba"), "n": rng.integers(100, 9000, 17),
                        "p_joint": rng.uniform(0.1, 0.5, 17), "independent_joint": rng.uniform(0.1, 0.5, 17),
                        "dependency_ratio": rng.uniform(0.6, 1.8, 17)})
    a, top = V.dependence_frame(dep)
    oa = dep[dep["segment"] == "ALL"].copy()
    oa["independence_error_pp"] = (100 * (oa["p_joint"] - oa["independent_joint"])).round(2)
    otop = pd.concat([oa.nlargest(6, "dependency_ratio"), oa.nsmallest(4, "dependency_ratio")])
    pd.testing.assert_frame_equal(top, otop)
    spec = V.dependence_chart(top).to_dict()
    assert spec["encoding"]["x"]["title"].startswith("independence error")


def test_builder_page_offline_is_feed_not_connected(monkeypatch):
    _offline(monkeypatch)
    from dashboard_data import core
    monkeypatch.setattr(core, "reachable", lambda repo: False)
    at = _render(P["14"], monkeypatch, tier="pro")
    assert "FEED NOT CONNECTED" in _text(at)


# ══ Success rates ═════════════════════════════════════════════════════════════════════════════════
def test_success_ci_frame_keeps_the_old_decisive_rule():
    from dashboard_data import success_view as V
    shown = pd.DataFrame({"model": ["a", "b", "c"], "market": ["OU25"] * 3, "tier": ["SNIPER"] * 3, "n": [40, 50, 8],
                          "roi_pct": [12.0, -3.0, 40.0], "roi_lo": [2.0, -20.0, np.nan], "roi_hi": [22.0, 14.0, np.nan]})
    ch = V.ci_frame(shown)
    assert len(ch) == 2 and ch["decisive"].tolist() == [False, True]          # sorted by roi_pct, NaN rows dropped
    assert set(ch["reading"]) == {"interval clears zero", "interval spans zero"}
    spec = V.ci_chart(ch).to_dict()
    assert any(l.get("mark", {}).get("type") == "rule" for l in spec["layer"])  # whiskers + zero rule


@pytest.mark.parametrize("tier", ["admin", "free"])
def test_success_rates_page_renders_for_each_tier(monkeypatch, tier):
    at = _render(P["5"], monkeypatch, tier=tier)
    assert any(t.value == "Success rates" for t in at.title)
    assert any(e.label == "Advanced filters" for e in at.expander)
    if at.dataframe:
        df = at.dataframe[0].value
        assert list(df.columns)[:4] == ["model", "market", "tier", "n"]
        assert (df["n"] >= (10 if tier == "admin" else 5)).all()             # the min-n default per tier


# ══ Model Info ════════════════════════════════════════════════════════════════════════════════════
def test_model_info_thresholds_are_numbers_with_the_same_value(monkeypatch):
    import config
    at = _render(P["4"], monkeypatch)
    from dashboard_data.model_info_view import THRESH_COL          # "Big gap threshold" (src/tier_labels)
    t = next(d.value for d in at.dataframe if THRESH_COL in d.value.columns)
    want = {lg: f"{v:.0%}" for lg, v in config.LEAGUE_SNIPER_THRESHOLDS.items()}
    for _, r in t.iterrows():
        assert f"{r[THRESH_COL]:.0f}%" == want[r["League"]]


# ══ Research (Pro) ════════════════════════════════════════════════════════════════════════════════
def test_research_load_all_returns_the_same_call_values(monkeypatch):
    from dashboard_data import overview_view, research_view as V
    monkeypatch.setattr(overview_view, "prefetch_pro_listings", lambda: None)
    vals = {name: {"name": name, "x": i} for i, name in enumerate(V.CALLS)}

    def _boom():
        raise RuntimeError("unreadable")
    fake = SimpleNamespace(**{n: (lambda n=n: vals[n]) for n in V.CALLS})
    fake.studies = _boom
    got = V.load_all(fake)
    assert set(got) == set(V.CALLS)
    for n in V.CALLS:
        assert got[n] == ({} if n == "studies" else vals[n])


def test_research_heat_frame_is_the_old_pivot():
    from dashboard_data import research_view as V
    lm = pd.DataFrame({"league": ["A", "A", "B", "B", "C"], "market": ["OU25", "BTTS", "OU25", "BTTS", "OU25"],
                       "N": [12, 30, 9, 40, 15], "ROI_shrunk_pct": [3.2, -1.1, 8.0, 0.4, -6.5]})
    h = lm[lm.N >= 10].pivot_table(index="league", columns="market", values="ROI_shrunk_pct")
    d = V.valuable_heat_frame(lm)
    for _, r in d.iterrows():
        assert r["ROI_shrunk_pct"] == h.loc[r["league"], r["market"]]
    assert len(d) == int(h.notna().sum().sum())


@pytest.mark.parametrize("key", ["7", "8", "15", "9"])
def test_research_pages_render_offline_as_not_connected(monkeypatch, key):
    _offline(monkeypatch)
    from dashboard_data import core
    monkeypatch.setattr(core, "reachable", lambda repo: False)
    at = _render(P[key], monkeypatch)
    assert at.title, "no page header"
    if key != "9":
        assert "FEED NOT CONNECTED" in _text(at)


# ══ Agent Analysis ════════════════════════════════════════════════════════════════════════════════
def test_agent_verdict_words_unchanged_and_tones_semantic():
    src = P["10"].read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(src[src.index("def verdict("):src.index("# ── Load data")], "verdict", "exec"), ns)  # noqa: S102
    v = ns["verdict"]
    assert v("OVER", 0.6, 0.7, 20)[0::2] == ("✅ APPROVED", "Both models agree. Strong edge confirmed.")
    assert v("OVER", 0.6, 0.7, 10)[0] == "✅ APPROVED" and v("OVER", 0.6, 0.7, 3)[0] == "⚠️ WEAK"
    assert v("OVER", 0.52, 0.49, 3)[0] == "⚠️ NEUTRAL" and v("OVER", 0.9, 0.3, 3)[0] == "❌ DISAGREE"
    assert {v("OVER", 0.6, 0.7, 20)[1], v("OVER", 0.6, 0.7, 3)[1], v("OVER", 0.9, 0.3, 3)[1]} == \
        {"POSITIVE", "WARNING", "NEGATIVE"}


def test_agent_page_renders(monkeypatch):
    at = _render(P["10"], monkeypatch)
    assert at.title


# ══ admin pages 19–22, account, free tips ═════════════════════════════════════════════════════════
@pytest.mark.parametrize("key,tier", [("19", "admin"), ("20", "admin"), ("21", "admin"), ("22", "admin"),
                                      ("account", "free"), ("free_tips", "free")])
def test_aligned_pages_render_with_one_header(monkeypatch, key, tier):
    at = _render(P[key], monkeypatch, tier=tier)
    assert len(at.title) == 1


def test_recorded_badge_agrees_with_recorded():
    from dashboard_data import freshness as F

    class _St:                                                    # recorded() writes through st.*; swallow it
        def __getattr__(self, _):
            return lambda *a, **k: None
    now = pd.Timestamp.now(tz="UTC")
    m = {"artifacts": []}
    for ts in (now.isoformat(), (now - pd.Timedelta(hours=80)).isoformat(), None):
        state, chip = F.recorded_badge("output/x.json", ts, 36, "x", m)
        assert state == F.recorded(_St(), "output/x.json", ts, 36, "x", m)
        assert chip.startswith(":")
