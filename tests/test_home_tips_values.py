"""Non-regression for the 2026-10-25 visual redesign of Home and Tips.

The redesign moved and restyled values; it must not change any of them. Each test renders the page
headlessly (streamlit.testing AppTest, absolute path), pulls the displayed values, and compares them
with (a) a FROZEN copy of the pre-redesign computation and (b) the canonical data functions the page
now calls. A mismatch means the presentation changed a number.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "pages" / "00_🎯_Home.py"

STAT = re.compile(r"<span class='wz-stat-l'>(.*?)</span><span class='wz-stat-v'>(.*?)</span>")


def _render(page: Path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    at = AppTest.from_file(str(page.resolve()), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _stats(at) -> dict:
    import html
    out = {}
    for m in at.markdown:
        for k, v in STAT.findall(m.value):
            out.setdefault(html.unescape(k), html.unescape(v))
    return out


def _metrics(at) -> dict:
    out = {}
    for m in at.metric:
        out.setdefault(m.label, m.value)
    return out


# ── frozen pre-redesign Home logic (pages/00_🎯_Home.py at 4af43f28), copied verbatim ─────────────
def _old_home_values():
    import config

    def _read_csv(*a, **k):
        return config.drop_paper_leagues(pd.read_csv(*a, **k))

    f = config.OUTPUT_DIR / "bets.csv"
    bets = _read_csv(f) if f.exists() else pd.DataFrame()
    parts = []
    cut = pd.Timestamp(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))
    for fn, mkc, const in [("bets_ledger.csv", None, "O/U 2.5"), ("side_bets_ledger.csv", "market", None),
                           ("ht_ledger.csv", None, "HT O/U")]:
        p = config.OUTPUT_DIR / fn
        if not p.exists():
            continue
        try:
            d = _read_csv(p, low_memory=False)
            if "source" in d.columns:
                d = d[d["source"].fillna("live").astype(str) == "live"]
            if "signal_tier" not in d.columns:
                continue
            d = d.assign(_dt=pd.to_datetime(d.get("match_date"), errors="coerce"),
                         _pnl=pd.to_numeric(d.get("pnl"), errors="coerce"),
                         _mkt=d[mkc].astype(str) if mkc and mkc in d.columns else const)
            parts.append(d[d["_pnl"].notna() & (d["_dt"] >= cut)][["signal_tier", "_mkt", "_pnl"]])
        except Exception:                                   # noqa: BLE001
            continue
    settled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["signal_tier", "_mkt", "_pnl"])
    staked = settled[settled["signal_tier"].isin(("SNIPER", "MARKSMAN"))]
    paper = settled[settled["signal_tier"] == "VALUABLE"]
    up = (bets[bets["signal_tier"].isin(("SNIPER", "MARKSMAN")) & bets["bet"].isin(["OVER", "UNDER"])]
          if not bets.empty else bets)
    return {
        "🎯 Upcoming staked tips": str(len(up)),
        "📈 P/L — staked (u)": f"{staked['_pnl'].sum():+.2f}u" if len(staked) else "—",
        "✅ Win rate — staked": f"{(staked['_pnl'] > 0).mean() * 100:.1f}%" if len(staked) else "—",
        "📊 Settled staked bets": str(len(staked)),
        "💎 Paper VALUABLE P/L": f"{paper['_pnl'].sum():+.2f}u" if len(paper) else "—",
        "paper_delta": f"{len(paper)} tips, not bets",
    }


# Where each pre-redesign Home value lives now (label moved, value identical).
HOME_MOVES = {"🎯 Upcoming staked tips": ("stat", "Upcoming staked tips (O/U 2.5)"),
              "📈 P/L — staked (u)": ("metric", "Staked P/L"),
              "✅ Win rate — staked": ("stat", "Win rate — staked"),
              "📊 Settled staked bets": ("stat", "Settled staked bets")}


def test_home_values_match_pre_redesign(monkeypatch):
    old = _old_home_values()
    at = _render(HOME, monkeypatch)
    metrics, stats = _metrics(at), _stats(at)
    for old_label, (kind, new_label) in HOME_MOVES.items():
        got = (metrics if kind == "metric" else stats).get(new_label)
        if old[old_label] == "—" and got is None:
            continue
        assert got == old[old_label], f"{old_label} -> {new_label}: {got!r} != {old[old_label]!r}"
    caps = " ".join(c.value for c in at.caption)
    if old["💎 Paper VALUABLE P/L"] != "—":
        assert f"**{old['💎 Paper VALUABLE P/L']}** · {old['paper_delta']}" in caps


def test_home_new_values_come_from_canonical_functions(monkeypatch):
    import json
    from dashboard_data import production_view, today_view
    at = _render(HOME, monkeypatch)
    metrics, stats = _metrics(at), _stats(at)
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    plan = production_view.stake_plan(today=today)
    assert metrics["Active tips"] == str(int((plan["final_fraction"] > 0).sum()))
    act = today_view.action(plan)
    assert stats["Paper / no stake"] == str(act["paper"])
    sh = json.loads((ROOT / "output" / "system_health.json").read_text(encoding="utf-8"))
    assert metrics["System health"] == str(sh.get("overall"))
    perf = today_view.performance(today_view.load_settled(), today=pd.Timestamp.now().normalize())
    if perf["n"]:
        assert stats["ROI"] == f"{perf['pnl'] / perf['n'] * 100:+.1f}%"
        assert stats["Drawdown from peak"] == f"{perf['drawdown']:+.2f}u"
        assert perf["drawdown"] <= 0
    # needs-attention lists exactly the non-GREEN checks, never a healthy one
    shown = {m.value.split("**")[1] for m in at.markdown if "&nbsp;**" in m.value and "-badge[" in m.value}
    not_green = {c["check"] for c in sh.get("checks", []) if c.get("status") != "GREEN"}
    green = {c["check"] for c in sh.get("checks", []) if c.get("status") == "GREEN"}
    assert not_green <= shown and not (green & shown)
    nk = today_view.next_kickoff(plan, today_view.kickoffs())
    assert stats["Next active kickoff"] == (f"{nk.tz_convert('Asia/Jerusalem'):%a %H:%M}" if pd.notna(nk) else "—")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# TIPS
# ═════════════════════════════════════════════════════════════════════════════════════════════
TIPS = ROOT / "pages" / "1_📊_Dashboard.py"


def _selections(at) -> pd.DataFrame:
    for d in at.dataframe:
        if list(d.value.columns[:4]) == ["Day", "Kickoff", "Started", "Match"]:
            return d.value
    raise AssertionError("no 'All selections' table on the Tips page")


def test_tips_board_is_tips_rows_unchanged(monkeypatch):
    """All selections = telegram_bot.tips_table.tips_rows (the Telegram image's rows), every row, with
    the same pick, odds, edge and tier; and the four summary metrics are computed from those rows."""
    from telegram_bot.tips_table import tips_rows
    from dashboard_data import tips_view as TV
    at = _render(TIPS, monkeypatch)
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    want = tips_rows(today, "2099-12-31")
    # the page's DEFAULT filters, unchanged from before the redesign: league and model multiselects
    # default to every league / model in the upcoming bets.csv slate, so a ledger tip from a league
    # with no fixture in the current slate is not on the default board.
    import config
    bets = config.drop_paper_leagues(pd.read_csv(config.OUTPUT_DIR / "bets.csv"))
    bets = bets[pd.to_datetime(bets["date"], errors="coerce") >= pd.Timestamp.now().normalize()]
    if len(want) and len(bets):
        want = want[want["league"].isin(set(bets["league"].dropna()))]
    got = _selections(at) if len(want) else pd.DataFrame()
    m = _metrics(at)
    if want.empty:
        return
    assert m["Tips shown"] == str(len(want)) and len(got) == len(want)
    assert m["Staked (SNIPER + MARKSMAN)"] == str(int(want["tier"].isin(["SNIPER", "MARKSMAN"]).sum()))
    assert m["Fixtures"] == str(want[["date", "home_team", "away_team"]].drop_duplicates().shape[0])

    def key(match, pick, odds, edge, tier):
        return (match, pick, round(float(odds), 2), None if pd.isna(edge) else round(float(edge), 2), tier)
    exp = sorted(key(f"{r.home_team} v {r.away_team}", TV.tip_text(r.market), r.odds, r.edge, r.tier)
                 for r in want.itertuples())
    obs = sorted(key(r["Match"], r["Pick"], r["Odds"], r["Edge %"], r["Tier (legacy)"]) for _, r in got.iterrows())
    assert obs == exp
    # numbers stay numbers: sorting the table must sort by value, not by text
    for c in ("Odds", "Edge %", "Stake %"):
        assert pd.api.types.is_numeric_dtype(got[c]), c


# ── frozen pre-redesign Tips settled/CLV logic (pages/1_📊_Dashboard.py at 4af43f28) ─────────────
def _old_tips_settled_metrics() -> dict:
    import config

    def _read_csv(*a, **k):
        return config.drop_paper_leagues(pd.read_csv(*a, **k))
    cut = pd.Timestamp(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))
    parts = []
    for f, mkc, const in [("bets_ledger.csv", None, "O/U 2.5"), ("side_bets_ledger.csv", "market", None),
                          ("ht_ledger.csv", None, "HT O/U")]:
        fp = config.OUTPUT_DIR / f
        if not fp.exists():
            continue
        try:
            d = _read_csv(fp, low_memory=False)
            d["_dt"] = pd.to_datetime(d.get("match_date"), errors="coerce")
            d["_pnl"] = pd.to_numeric(d.get("pnl"), errors="coerce")
            d["_mkt"] = (d[mkc].astype(str) if mkc and mkc in d.columns else const)
            keep = d.get("signal_tier").isin(("SNIPER", "MARKSMAN")) & d["_pnl"].notna() & (d["_dt"] >= cut)
            if keep.any():
                parts.append(d[keep][["_mkt", "_dt", "_pnl"]])
        except Exception:                                    # noqa: BLE001
            continue
    out = {}
    if parts:
        s = pd.concat(parts, ignore_index=True)
        tot = float(s["_pnl"].sum())
        out["P/L — all staked markets"] = (f"{tot:+.2f}u", f"{tot / len(s):+.3f}u per bet")
        out["Settled bets"] = (f"{len(s):,}", None)
        out["Days covered"] = (f"{s['_dt'].dt.date.nunique()}", None)
    cl = _read_csv(config.OUTPUT_DIR / "bets_ledger.csv", low_memory=False)
    cl["_clv"] = pd.to_numeric(cl.get("clv_pct"), errors="coerce")
    cl["_o"] = pd.to_numeric(cl.get("odds"), errors="coerce")
    cl["_c"] = pd.to_numeric(cl.get("closing_odds"), errors="coerce")
    raw = cl[cl["_clv"].notna()]
    ok = raw[raw["_clv"].abs() <= 25.0]
    if parts and len(ok):
        flat = ok["_c"] == ok["_o"]
        moved = ok[~flat]
        out["Median CLV"] = (f"{ok['_clv'].median():+.2f}%", None)
        out["Clean observations"] = (f"{len(ok):,}", f"−{len(raw) - len(ok)} rejected")
        out["Beat the close"] = (f"{100 * (moved['_clv'] > 0).mean():.1f}%", None)
        out["No recorded move"] = (f"{100 * flat.mean():.0f}%", None)
    return out


def test_tips_settled_and_clv_metrics_match_pre_redesign(monkeypatch):
    old = _old_tips_settled_metrics()
    at = _render(TIPS, monkeypatch)
    got = {}
    for m in at.metric:
        got.setdefault(m.label, (m.value, getattr(m, "delta", None) or None))
    for label, val in old.items():
        assert got.get(label) == val, f"{label}: {got.get(label)!r} != {val!r}"


def test_tips_attach_keeps_rows_and_copies_stake_plan_values():
    """Production / shadow columns are a left join: same rows, same order, values copied verbatim."""
    from telegram_bot.tips_table import tips_rows
    from dashboard_data import tips_view as TV
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    b = tips_rows(today, "2099-12-31")
    if b.empty:
        pytest.skip("no tips in this checkout")
    plan = TV.load_plan()
    a = TV.attach(b, plan, TV.load_shadow())
    assert len(a) == len(b)
    assert a[["home_team", "away_team", "market", "odds"]].astype(str).values.tolist() == \
        b[["home_team", "away_team", "market", "odds"]].astype(str).values.tolist()
    p = plan.assign(side=plan["side"].astype(str).str.upper()).drop_duplicates(["fixture", "market", "side"], keep="last")
    p = p.set_index(["fixture", "market", "side"])
    for _, r in a[a["in_plan"]].iterrows():
        code, side = TV.plan_key(r["market"])
        src = p.loc[(f"{str(r['date'])[:10]}|{r['home_team']}|{r['away_team']}", code, side)]
        assert r["deployment_state"] == src["deployment_state"]
        for c in ("final_fraction", "p_market_fair", "p_decision", "ev_lower_bound"):
            assert (pd.isna(r[c]) and pd.isna(src[c])) or abs(float(r[c]) - float(src[c])) < 1e-12, c
    assert set(a.loc[~a["in_plan"], "production_state"]) <= {TV.NOT_IN_PLAN}


def test_market_labels_round_trip():
    from dashboard_data import tips_view as TV
    assert TV.plan_key("O/U 2.5 UNDER") == ("OU25", "UNDER")
    assert TV.plan_key("BTTS YES") == ("BTTS", "YES")
    assert TV.plan_key("O/U 1.5 OVER") == ("OVER15", "OVER")
    assert TV.plan_key("O/U 3.5 OVER") == ("OVER35", "OVER")
    assert TV.tip_text("O/U 2.5 UNDER") == "Under 2.5" and TV.tip_text("BTTS YES") == "BTTS — Yes"
    assert TV.stake_label("SNIPER", "O/U 2.5 OVER") == TV.STAKE_OU
    assert TV.stake_label("VALUABLE", "BTTS YES") == "paper"
