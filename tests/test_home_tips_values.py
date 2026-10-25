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
