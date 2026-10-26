"""The FIRST LOGGED TIER STANDS (owner decision 2026-10-26).

Until then src/ledger.py raised a logged tip's signal_tier in place when a later predict run produced a
higher tier ("best tier wins"). Results were then reported against a tier the owner was never sent:
first-sight tier matched the ledger on 72% of O/U 2.5 tips. Now:
  * a later, higher tier never changes the ledger row's tier (O/U 2.5 and side markets);
  * it is recorded once per (tip, tier) in the append-only research log ledger_tier_seen.csv;
  * existing rows are never rewritten;
  * Telegram never re-sends a tip because its tier rose (the dedup key has no tier), and the morning
    album / top picks read the ledger, so they show the first logged tier.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

import config
from src import ledger as L

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _bet(tier="VALUABLE", edge=0.05, side="UNDER", home="Mito", odds_u=1.89):
    return {"date": TODAY, "league": "Japan J-League", "home_team": home, "away_team": "Shimizu",
            "bet": side, "best_side": side, "odds_over25": 2.45, "odds_under25": odds_u,
            "best_edge": edge, "signal_tier": tier, "drift_signal": "", "model_type": "new_format"}


def _side(tier="VALUABLE", edge=0.05, odds=2.10, market="btts", home="Boca"):
    return {"date": TODAY, "league": "Argentina Primera Division", "home_team": home, "away_team": "River",
            "market": market, "market_odds": odds, "edge": edge, "ev": edge * odds, "model_prob": 0.55,
            "signal_tier": tier, "model_type": "new_format"}


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    monkeypatch.setattr(L, "_version_fields", lambda *a, **k: {k: "" for k in L.VERSION_FIELDS})
    return tmp_path


def _seen(out):
    p = out / L.TIER_SEEN_NAME
    return pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame()


# ── O/U 2.5 ledger ───────────────────────────────────────────────────────────────────────────────
def test_a_later_run_with_a_bigger_edge_does_not_change_the_ou_tier(out):
    L.append_tips(pd.DataFrame([_bet("VALUABLE", edge=0.05)]))
    L.append_tips(pd.DataFrame([_bet("MARKSMAN", edge=0.09)]))
    L.append_tips(pd.DataFrame([_bet("SNIPER", edge=0.16)]))
    led = pd.read_csv(out / "bets_ledger.csv")
    assert len(led) == 1 and led.iloc[0]["signal_tier"] == "VALUABLE"
    s = _seen(out)
    assert list(s["observed_tier"]) == ["MARKSMAN", "SNIPER"]
    assert set(s["ledger_tier"]) == {"VALUABLE"} and set(s["ledger"]) == {"ou25"}
    assert list(s["observed_edge_pct"].astype(float)) == [9.0, 16.0]


def test_the_research_log_records_each_higher_tier_once(out):
    L.append_tips(pd.DataFrame([_bet("VALUABLE")]))
    for _ in range(3):
        L.append_tips(pd.DataFrame([_bet("SNIPER", edge=0.16)]))
    L.append_tips(pd.DataFrame([_bet("MARKSMAN", edge=0.09)]))      # below the best seen: not new
    s = _seen(out)
    assert list(s["observed_tier"]) == ["SNIPER"]
    assert L.tier_seen_store(out / L.TIER_SEEN_NAME).verify()["ok"] is True


def test_a_same_or_lower_tier_re_see_writes_nothing(out):
    L.append_tips(pd.DataFrame([_bet("MARKSMAN", edge=0.09)]))
    before = (out / "bets_ledger.csv").read_bytes()
    L.append_tips(pd.DataFrame([_bet("MARKSMAN", edge=0.10)]))
    L.append_tips(pd.DataFrame([_bet("VALUABLE", edge=0.05)]))
    assert (out / "bets_ledger.csv").read_bytes() == before
    assert _seen(out).empty


def test_existing_rows_are_not_rewritten(out):
    """A row logged before the change (possibly already upgraded) keeps its tier; other rows untouched."""
    L.append_tips(pd.DataFrame([_bet("SNIPER", edge=0.16), _bet("VALUABLE", home="Other")]))
    led0 = pd.read_csv(out / "bets_ledger.csv", dtype=str)
    L.append_tips(pd.DataFrame([_bet("VALUABLE", edge=0.05), _bet("SNIPER", edge=0.2, home="Other")]))
    led1 = pd.read_csv(out / "bets_ledger.csv", dtype=str)
    assert list(led1["signal_tier"]) == list(led0["signal_tier"]) == ["SNIPER", "VALUABLE"]
    pd.testing.assert_frame_equal(led1, led0)            # commit B: edge/kelly/drift frozen too


# ── side-market ledger ───────────────────────────────────────────────────────────────────────────
def test_a_later_run_with_a_bigger_edge_does_not_change_the_side_tier(out):
    L.append_side_market_tips(pd.DataFrame([_side("VALUABLE", edge=0.05, odds=2.10)]))
    L.append_side_market_tips(pd.DataFrame([_side("SNIPER", edge=0.12, odds=2.25)]))
    led = pd.read_csv(out / "side_bets_ledger.csv")
    assert len(led) == 1 and led.iloc[0]["signal_tier"] == "VALUABLE"
    s = _seen(out)
    assert len(s) == 1 and s.iloc[0]["ledger"] == "side" and s.iloc[0]["selection"] == "btts"
    assert s.iloc[0]["observed_tier"] == "SNIPER" and float(s.iloc[0]["observed_odds"]) == 2.25
    assert float(s.iloc[0]["ledger_odds"]) == 2.10


# ── consumers ────────────────────────────────────────────────────────────────────────────────────
def test_telegram_never_resends_a_tip_because_its_tier_rose(out, monkeypatch):
    from telegram_bot import notifier as N
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", out)
    notified, sent = set(), []
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_load_notified", lambda *a, **k: notified)
    monkeypatch.setattr(N, "_save_notified", lambda keys, *a, **k: notified.update(keys))
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    monkeypatch.setattr(N, "_send", lambda token, chat, text: sent.append(text) or True)
    pd.DataFrame([_bet("VALUABLE")]).to_csv(out / "bets.csv", index=False)
    assert N.notify_new_snipers() == 1
    pd.DataFrame([_bet("SNIPER", edge=0.16)]).to_csv(out / "bets.csv", index=False)
    assert N.notify_new_snipers() == 0                      # same fixture + side: never re-sent
    assert not any("upgrad" in m.lower() for m in sent)


def test_album_and_top_picks_show_the_first_logged_tier(out):
    from src import top_picks
    from telegram_bot import tips_table
    L.append_tips(pd.DataFrame([_bet("VALUABLE")]))
    L.append_tips(pd.DataFrame([_bet("SNIPER", edge=0.16)]))
    L.append_side_market_tips(pd.DataFrame([_side("MARKSMAN", edge=0.09)]))
    L.append_side_market_tips(pd.DataFrame([_side("SNIPER", edge=0.12)]))
    rows = tips_table.tips_rows("2000-01-01", "2100-01-01")
    assert sorted(rows["tier"]) == ["MARKSMAN", "VALUABLE"]
    tips = top_picks.load_tips()
    assert sorted(tips["signal_tier"]) == ["MARKSMAN", "VALUABLE"]


def test_fix_ledger_tiers_apply_is_retired():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "scripts/fix_ledger_tiers.py", "--apply"], capture_output=True, text=True)
    assert r.returncode != 0 and "retired" in (r.stderr + r.stdout)


def test_predict_commits_the_research_log():
    wf = open(".github/workflows/predict.yml", encoding="utf-8").read()
    assert "output/ledger_tier_seen.csv output/ledger_tier_seen.csv.first_sight.json" in wf


# ── commit B (pending owner approval): odds / edge frozen at first log ─────────────────────────────
def test_ou_edge_kelly_drift_are_frozen_at_first_log(out):
    L.append_tips(pd.DataFrame([{**_bet("VALUABLE", edge=0.05), "drift_signal": "New"}]))
    before = (out / "bets_ledger.csv").read_bytes()
    L.append_tips(pd.DataFrame([{**_bet("SNIPER", edge=0.16, odds_u=2.05), "drift_signal": "Confirmed"}]))
    assert (out / "bets_ledger.csv").read_bytes() == before          # the ledger file is not even rewritten
    s = _seen(out).iloc[0]
    assert float(s["observed_edge_pct"]) == 16.0 and float(s["observed_odds"]) == 2.05
    assert float(s["ledger_edge_pct"]) == 5.0 and float(s["ledger_odds"]) == 1.89


def test_side_odds_edge_ev_are_frozen_at_first_log(out):
    L.append_side_market_tips(pd.DataFrame([_side("VALUABLE", edge=0.05, odds=2.10)]))
    before = (out / "side_bets_ledger.csv").read_bytes()
    L.append_side_market_tips(pd.DataFrame([_side("SNIPER", edge=0.12, odds=2.25)]))
    assert (out / "side_bets_ledger.csv").read_bytes() == before
    led = pd.read_csv(out / "side_bets_ledger.csv").iloc[0]
    assert led["odds"] == 2.10 and led["edge_pct"] == 5.0           # settled + CLV-graded at this price
    assert float(_seen(out).iloc[0]["observed_odds"]) == 2.25       # the later price is research only


def test_a_new_tip_after_a_re_see_still_appends_without_touching_old_rows(out):
    L.append_side_market_tips(pd.DataFrame([_side("VALUABLE", odds=2.10)]))
    L.append_side_market_tips(pd.DataFrame([_side("SNIPER", edge=0.12, odds=2.25),
                                            _side("MARKSMAN", edge=0.09, odds=1.95, home="Lanus")]))
    led = pd.read_csv(out / "side_bets_ledger.csv")
    assert list(led["home_team"]) == ["Boca", "Lanus"]
    assert list(led["odds"]) == [2.10, 1.95] and list(led["signal_tier"]) == ["VALUABLE", "MARKSMAN"]
