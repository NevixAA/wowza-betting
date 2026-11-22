"""Admin Board (owner 2026-11-22): pages/29_🧭_Admin_Board.py + dashboard_data/admin_board.py + private_prices.py.

Pins: the bars come from the APPLIED config through the serving rule, every league x market x side has a row and
a cell without its own owner-applied bar says NO BAR; TOP-3 reads both ledger shapes (stake_effect / stake +
stake_reason) and never takes a price from the ledger itself; odds come from the private store with the read-only
token (fallback: the public file; neither -> "odds: private"); the token never reaches the page or an error; the
tracker registry is well formed and its v9 workflows exist; the page renders for admin only, network mocked.
"""
import copy
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from dashboard_data import admin_board as AB
from dashboard_data import private_prices as PP

ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/29_🧭_Admin_Board.py"
TOKEN = "github_pat_TESTTOKEN_never_shown_0123456789"


# ── bars ────────────────────────────────────────────────────────────────────────────────────────
def _real_inputs():
    inp = AB.load_bars_inputs()
    assert inp["problems"] == []
    return inp


def test_bars_cover_every_league_market_side_from_the_applied_config():
    inp = _real_inputs()
    d = AB.bars_frame(inp)
    leagues = AB.universe(inp["leagues"])
    assert len(d) == len(leagues) * len(AB.MARKET_SIDES)
    assert set(d["Status"]) <= set(AB.STATUS_ORDER)
    # sides the serving code has no cell for, and the HT lines (no approved HT bar exists), are NO BAR
    assert (d.loc[d["Market"].str.startswith("HT"), "Status"] == AB.NO_BAR).all()
    assert (d.loc[(d["Market"] == "BTTS") & (d["Side"] == "NO"), "Status"] == AB.NO_BAR).all()
    # every served cell names the proposal that applied it and has a bar
    served = d[d["Status"].isin(["PROVEN", "PROMISING", "UNPROVEN"])]
    assert len(served) > 0
    assert served["Proposal"].notna().all() and served["Bar %"].notna().all()
    nb = d[d["Status"] == AB.NO_BAR]
    assert nb["Why no bar"].notna().all() and nb["Bar %"].isna().all()
    assert not {"odds", "Odds", "price"} & set(d.columns)


def _one_cell(inp):
    for league, lp in inp["standard"].items():
        cells = (lp or {}).get("bars_opt") or {}
        if "over" in cells:
            return league, lp
    pytest.skip("no ou25 over cell in the applied config")


def test_unapplied_and_dropped_cells_are_not_served():
    inp = _real_inputs()
    league, lp = _one_cell(inp)
    base = AB.bars_frame(inp)
    row = base[(base["League"] == league) & (base["Market"] == "O/U 2.5") & (base["Side"] == "OVER")].iloc[0]
    assert row["Status"] in ("PROVEN", "PROMISING", "UNPROVEN")
    not_applied = copy.deepcopy(inp)
    for h in not_applied["applied"]["history"]:
        h["entries"] = [e for e in h["entries"] if e != f"ou25|{league}"]
    r2 = AB.bars_frame(not_applied)
    r2 = r2[(r2["League"] == league) & (r2["Market"] == "O/U 2.5") & (r2["Side"] == "OVER")].iloc[0]
    assert r2["Status"] == AB.NOT_APPLIED and pd.isna(r2["Bar %"])
    dropped = copy.deepcopy(inp)
    dropped["standard"][league]["drop"] = True
    r3 = AB.bars_frame(dropped)
    r3 = r3[(r3["League"] == league) & (r3["Market"] == "O/U 2.5") & (r3["Side"] == "OVER")].iloc[0]
    assert r3["Status"] == AB.OFF


# ── TOP-3 ───────────────────────────────────────────────────────────────────────────────────────
def _ledger(new_shape: bool):
    rows = [
        {"pick_id": "a1", "league_week_id": "L1|2026-W47", "top3_rank": 1, "league": "L1", "home_team": "H", "away_team": "A",
         "kickoff_utc": "2026-11-21T20:00:00Z", "market": "OVER15", "side": "OVER", "odds": 9.99,
         "edge_calibrated": 0.015, "stake_effect": "REAL 0.5% of bankroll, flat", "result": "WIN", "flat_pnl": 0.5,
         "settle_status": "SETTLED"},
        {"pick_id": "a2", "league_week_id": "L1|2026-W47", "top3_rank": 2, "league": "L1", "home_team": "H2", "away_team": "A2",
         "kickoff_utc": "2026-11-22T20:00:00Z", "market": "BTTS", "side": "YES", "odds": 8.88,
         "edge_calibrated": None, "stake_effect": "NONE (TOP-3 never sizes a stake)", "result": None, "flat_pnl": None,
         "settle_status": None},
    ]
    if new_shape:
        rows[0].update(stake=0.005, stake_reason="STAKED")
        rows[1].update(stake=0.0, stake_reason="NO_STAKE_BELOW_BAR")
    return pd.DataFrame(rows)


@pytest.mark.parametrize("new_shape", [False, True])
def test_top3_frame_both_shapes_and_no_ledger_price(new_shape):
    d = AB.top3_frame(_ledger(new_shape), None)
    assert list(d["Staked"]) == ["Yes", "No"]
    assert d.loc[0, "Stake"] == "0.5% stake"
    assert d.loc[1, "Stake"] == ("No stake: below bar" if new_shape else "No stake")
    assert d["Odds at tip"].isna().all()                       # never read from the ledger itself
    assert list(d["Tier"]) == ["T1 best value", "T2 safest"]
    assert d.loc[0, "Result"] == "WIN" and d.loc[1, "Result"] == "pending"
    assert d.loc[0, "Calibrated value pp"] == 1.5
    d2 = AB.top3_frame(_ledger(new_shape), {"a1": 1.5})
    assert d2.loc[0, "Odds at tip"] == 1.5 and pd.isna(d2.loc[1, "Odds at tip"])


def test_top3_weeks_explain_empty_slots():
    wk = pd.DataFrame([{"league_week_id": "L1|2026-W47", "league": "L1", "n_candidates": 20, "n_qualifying": 0,
                        "n_picks": 0, "excluded": json.dumps({"NO_LEAGUE_BAR": 15, "NO_TWO_SIDED_PRICE": 5})}])
    d = AB.top3_weeks(wk, _ledger(True))
    assert d.loc[0, "Empty slots"] == 3 and d.loc[0, "Week"] == "2026-W47"
    assert d.loc[0, "Why (excluded candidates)"] == "no league bar 15 · no two sided price 5"
    assert d.loc[0, "Staked"] == 1


def test_odds_from_csv_keeps_real_prices_only():
    txt = "pick_id,odds\na,1.9\nb,\nc,0.5\n"
    assert AB.odds_from_csv(txt) == {"a": 1.9}


# ── tracker ─────────────────────────────────────────────────────────────────────────────────────
def test_tracker_registry_is_well_formed_and_v9_workflows_exist():
    reg, probs = AB.load_tracker()
    assert reg is not None and probs == []
    ids = [t["id"] for t in reg["tools"]]
    assert len(ids) == len(set(ids))
    for t in reg["tools"]:
        if t["repo"] == "v9":
            for w in t.get("workflows") or []:
                assert (ROOT / ".github" / "workflows" / w).exists(), w
        assert "paper" not in (t["name"] + t["purpose"] + t.get("schedule", "") + t.get("record_note", "")).lower()


def test_pick_and_record_text():
    doc = {"models": [{"model_id": "m", "market": "x", "rows": 2}, {"model_id": "m", "market": "y", "rows": 3},
                      {"model_id": "n", "rows": 9}], "a": {"b": 4}, "items": [1, 2, 3]}
    assert AB.pick(doc, "models[model_id=m].rows") == 5
    assert AB.pick(doc, "models[model_id=m,market=y].rows") == 3
    assert AB.pick(doc, "a.b") == 4 and AB.pick(doc, "items#len") == 3 and AB.pick(doc, "a.zz") is None
    rec = {"kind": "json", "fields": [{"label": "bets", "key": "a.b", "gate": 150}, {"label": "gone", "key": "q"}]}
    assert AB.record_text(rec, json.dumps(doc), "") == "bets 4 / 150 · gone missing"
    csvrec = {"kind": "csv", "fields": [{"label": "rows"}, {"label": "settled", "nonempty": "result"}]}
    assert AB.record_text(csvrec, "id,result\n1,W\n2,\n", "") == "rows 2 · settled 1"
    assert AB.record_text(rec, None, "HTTP 404") == "not readable (HTTP 404)"


def test_tracker_frame_states_and_links():
    reg = {"repos": {"v9": {"slug": "o/r", "label": "r"}},
           "tools": [{"id": "a", "name": "A", "repo": "v9", "purpose": "p", "workflows": ["x.yml"]},
                     {"id": "b", "name": "B", "repo": "v9", "purpose": "p", "workflows": ["y_paper.yml"]},
                     {"id": "c", "name": "C", "repo": "v9", "purpose": "p"}]}
    states = {"v9": {".github/workflows/x.yml": {"state": "active"}, ".github/workflows/y_paper.yml": {"state": "disabled_manually"}}}
    runs = {"v9": {".github/workflows/x.yml": {"created_at": "2026-11-22T10:00:00Z", "conclusion": "failure"}}}
    d = AB.tracker_frame(reg, states, runs, {"a": "rows 1"}, {})
    assert list(d["Enabled"]) == ["enabled", "disabled", "n/a (no workflow)"]
    assert d.loc[0, "Conclusion"] == "failure" and d.loc[0, "Last run (UTC)"] == "2026-11-22 10:00"
    assert d.loc[1, "Link"] == "https://github.com/o/r/actions"          # no retired word inside a URL
    assert d.loc[0, "Link"].endswith("/actions/workflows/x.yml")


# ── private store reader ────────────────────────────────────────────────────────────────────────
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_store_reader_uses_the_token_and_never_leaks_it(monkeypatch):
    seen = []

    def urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("Authorization")))
        if "missing" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        return _Resp(b"pick_id,odds\nz,2.0\n")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv(PP.SECRET, TOKEN)
    txt = PP.public_file_text("wowza-betting", "output/top3_league_ledger.csv")
    assert AB.odds_from_csv(txt) == {"z": 2.0}
    url, auth = seen[0]
    assert url.startswith("https://api.github.com/repos/NevixAA/wowza-exec/contents/wowza-betting/main/output/")
    assert url.endswith("?ref=af-prices-live") and auth == f"Bearer {TOKEN}"
    with pytest.raises(PP.StoreUnavailable) as e:
        PP.get("missing.csv")
    assert TOKEN not in str(e.value) and "not in the private store" in str(e.value)
    monkeypatch.delenv(PP.SECRET)
    with pytest.raises(PP.StoreUnavailable, match="token missing"):
        PP.get("x")


# ── page smoke (AppTest, network mocked) ────────────────────────────────────────────────────────
def _fake_net(store_has_ledger: bool):
    calls = []

    def urlopen(req, timeout=None):
        url = req.full_url
        calls.append((url, req.get_header("Authorization")))
        if "/contents/" in url:
            if store_has_ledger and "top3_league_ledger.csv" in url:
                return _Resp(b"pick_id,odds\n")
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if url.endswith("/actions/workflows?per_page=100"):
            return _Resp(json.dumps({"workflows": [{"path": ".github/workflows/predict.yml", "state": "active"}]}).encode())
        if "/runs" in url:
            return _Resp(json.dumps({"workflow_runs": [{"path": ".github/workflows/predict.yml", "created_at":
                                                        "2026-11-22T10:00:00Z", "conclusion": "success"}]}).encode())
        if url.startswith("https://raw.githubusercontent.com/"):
            return _Resp(b'{"rows": 7}')
        raise AssertionError(f"unexpected url {url}")
    return urlopen, calls


def _render(monkeypatch, tier="admin", token=TOKEN, store_has_ledger=False, ledger=None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    fake, calls = _fake_net(store_has_ledger)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", tier)
    if token:
        monkeypatch.setenv(PP.SECRET, token)
    else:
        monkeypatch.delenv(PP.SECRET, raising=False)
    if ledger is not None:
        monkeypatch.setattr(AB, "LEDGER", ledger)
    at = AppTest.from_file(str(ROOT / PAGE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at, calls


def _text(at) -> str:
    parts = []
    for kind in ("title", "subheader", "markdown", "caption", "error", "warning", "info", "tabs"):
        for el in getattr(at, kind, []):
            parts.append(str(getattr(el, "value", "")) + " " + str(getattr(el, "label", "")))
    for df in at.dataframe:
        parts.append(df.value.to_csv())
    return "\n".join(parts)


def test_page_renders_all_three_sections_for_admin(monkeypatch):
    at, calls = _render(monkeypatch)
    txt = _text(at)
    assert "Admin Board" in txt and "Applied config:" in txt
    assert "League-weeks and why slots are empty" in txt
    assert "Registry: config/research_tracker.json" in txt
    assert TOKEN not in txt
    assert "paper" not in txt.lower()
    assert len(at.dataframe) == 4


def test_page_odds_private_when_no_store_and_no_public_prices(monkeypatch, tmp_path):
    led = pd.read_csv(ROOT / "output" / "top3_league_ledger.csv")
    if led.empty:
        led = _ledger(True)
    p = tmp_path / "ledger.csv"
    led.drop(columns=[c for c in AB.PRICE_COLUMNS if c in led.columns]).to_csv(p, index=False)
    at, _ = _render(monkeypatch, token=None, ledger=p)
    txt = _text(at)
    assert "odds: private (token missing)" in txt


@pytest.mark.parametrize("tier", ["free", "pro"])
def test_page_is_admin_only(monkeypatch, tier):
    at, calls = _render(monkeypatch, tier=tier)
    assert calls == []
    assert "Applied config:" not in _text(at)
