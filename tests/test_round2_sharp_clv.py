"""Round-2 data-truth findings owned by fix4/sharp_clv (2026-11-02), pinned so they cannot come back.

V1  the sharp record shown in Telegram is the RESTATED one, or UNDER RESTATEMENT - never the P/L booked
    at opening_odds.
V2  first-half prices stored as closes are out of every CLV consumer (sidecar list).
HT  half-time bets with no market price are out of every HT P/L display (sidecar list).
TD  tip day uses a 10:00 Asia/Jerusalem WALL-CLOCK cut-off (DST end 2026-10-25).
TP  a top pick is never frozen after kickoff, and the 25 that were are excluded from grading.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import config
from src import clv_corrections as CC

REPO = Path(__file__).resolve().parents[1]
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
YDAY = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


# ── TD: tip day on the DST change day ────────────────────────────────────────────────────────────
def test_tip_day_cutoff_is_wall_clock_on_dst_end():
    from src.tip_day import tip_day_of
    # 2026-10-25 07:00 UTC = 09:00 Israel (DST ended at 02:00): BEFORE the 10:00 rollover -> 10-24.
    assert tip_day_of("2026-10-25T07:00:00Z") == "2026-10-24"
    # 08:00 UTC = 10:00 Israel winter time: the new tip day starts exactly here.
    assert tip_day_of("2026-10-26T07:59:00Z") == "2026-10-25"
    assert tip_day_of("2026-10-26T08:00:00Z") == "2026-10-26"
    # summer time: 10:00 Israel = 07:00 UTC
    assert tip_day_of("2026-10-20T06:59:00Z") == "2026-10-19"
    assert tip_day_of("2026-10-20T07:00:00Z") == "2026-10-20"
    assert tip_day_of(None, "2026-10-25") == "2026-10-25"


# ── TP: freeze guard + grading exclusion ─────────────────────────────────────────────────────────
def _tips(day="2026-10-24"):
    rows = []
    for i, (ko, odds) in enumerate([("2026-10-24T00:15:00Z", 1.40), ("2026-10-24T06:00:00Z", 1.50),
                                    ("2026-10-24T18:00:00Z", 1.60), ("2026-10-24T19:00:00Z", 1.70)]):
        rows.append({"day": day, "match_day": day, "kickoff_utc": ko, "league": "Japan J-League",
                     "home_team": f"H{i}", "away_team": f"A{i}", "market": "OU25_UNDER", "odds": odds,
                     "implied": 1 / odds, "model_p": 0.6, "signal_tier": "MARKSMAN", "model_type": "new_format",
                     "result": "", "generated_at": "2026-10-23 10:00"})
    return pd.DataFrame(rows)


def test_freeze_never_lists_a_match_that_has_kicked_off():
    from src import top_picks as TP
    d = TP.todays_lists("2026-10-24", tips=_tips(), now=pd.Timestamp("2026-10-24T06:03:57Z"))
    assert set(d["home_team"]) == {"H2", "H3"}             # H0 (00:15) and H1 (06:00) had kicked off


def test_freeze_day_passes_the_freeze_time_to_the_guard():
    src = (REPO / "src" / "top_picks.py").read_text(encoding="utf-8")
    i = src.index("def freeze_day")
    assert "todays_lists(day, now=now)" in src[i:i + 1200]


def test_freezing_a_past_day_lists_nothing(tmp_path, monkeypatch):
    from src import top_picks as TP
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(TP, "LOG", tmp_path / "top_picks_log.csv")
    monkeypatch.setattr(TP, "load_tips", lambda: _tips())
    assert TP.freeze_day("2026-10-24").empty and not (tmp_path / "top_picks_log.csv").exists()


def test_frozen_after_kickoff_is_listed_and_graded_excluded(tmp_path, monkeypatch):
    from src import top_picks as TP
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    log = tmp_path / "top_picks_log.csv"
    monkeypatch.setattr(TP, "LOG", log)
    t = _tips()
    f = t.assign(frozen_at="2026-10-24T06:03:57Z", list="SAFEST", rank=range(1, 5),
                 pick="Under 2.5", p_win=None, ev=None)
    f[["frozen_at", "list", "rank", "day", "league", "home_team", "away_team", "market", "pick", "odds",
       "p_win", "ev", "signal_tier", "model_type"]].to_csv(log, index=False)
    ko = t.assign(match_date=t["day"])[["league", "match_date", "home_team", "away_team", "kickoff_utc"]]
    late = TP.frozen_after_kickoff(TP.frozen(), ko=ko)
    assert list(late["home_team"]) == ["H0", "H1"]
    (tmp_path / "corrections").mkdir()
    late.to_csv(tmp_path / "corrections" / "top_picks_frozen_after_kickoff.csv", index=False)
    settled = t.assign(result=["WIN", "WIN", "LOSS", "WIN"])
    monkeypatch.setattr(TP, "load_tips", lambda: settled)
    g = TP.graded()
    assert dict(zip(g["home_team"], g["result"])) == {"H0": "EXCLUDED", "H1": "EXCLUDED", "H2": "LOSS", "H3": "WIN"}
    assert list(g["result_stored"]) == ["WIN", "WIN", "LOSS", "WIN"]        # original kept beside
    sb = TP.scoreboard()
    assert int(sb.loc[sb["list"] == "SAFEST", "picks"].iloc[0]) == 2


def test_committed_top_picks_sidecar_lists_the_25_late_picks():
    late = pd.read_csv(REPO / "output" / "corrections" / "top_picks_frozen_after_kickoff.csv")
    assert len(late) == 25
    assert sorted(late["day"].unique()) == ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-24", "2026-10-25"]
    assert (pd.to_datetime(late["frozen_at"], utc=True) > pd.to_datetime(late["kickoff_utc"], utc=True)).all()


# ── V1: sharp record in Telegram ─────────────────────────────────────────────────────────────────
def _sharp_ledger(out: Path):
    pd.DataFrame([
        {"signal_date": YDAY, "match_date": YDAY, "league": "Ireland Premier Division", "home_team": "Bohemians", "away_team": "Dundalk",
         "market_label": "1X2 HOME (Bohemians)", "side": "HOME", "signal": "STRONG", "opening_odds": 3.0,
         "current_odds": 2.5, "result": "WIN", "pnl": 2.0},
        {"signal_date": YDAY, "match_date": YDAY, "league": "Ireland Premier Division", "home_team": "Derry City", "away_team": "Sligo Rovers",
         "market_label": "1X2 AWAY (Sligo Rovers)", "side": "AWAY", "signal": "STRONG", "opening_odds": 4.0,
         "current_odds": 3.2, "result": "LOSS", "pnl": -1.0},
    ]).to_csv(out / "sharp_ledger.csv", index=False)


@pytest.fixture
def tg(tmp_path, monkeypatch):
    from telegram_bot import notifier, tips_table
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(notifier, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(notifier, "_load_notified", lambda *a, **k: set())
    monkeypatch.setattr(notifier, "_save_notified", lambda *a, **k: None)
    monkeypatch.setattr(notifier, "_send", lambda token, chat, text: sent.append(text) or True)
    monkeypatch.setattr(tips_table, "send_photo", lambda *a, **k: True)
    _sharp_ledger(tmp_path)
    return notifier, sent, tmp_path


def _restate(out: Path):
    (out / "corrections").mkdir(exist_ok=True)
    led = pd.read_csv(out / "sharp_ledger.csv")
    led.assign(pnl_restated=[1.5, -1.0])[CC.SHARP_KEY + ["pnl_restated"]] \
        .to_csv(out / "corrections" / "sharp_ledger_restated.csv", index=False)


def test_weekly_summary_never_shows_the_booked_sharp_pnl(tg):
    notifier, sent, _ = tg
    notifier.notify_weekly_summary()
    msg = sent[0]
    assert "UNDER RESTATEMENT" in msg
    assert "+1.00u" not in msg.split("Sharp Tracker")[1]                 # booked 2.0 - 1.0


def test_weekly_summary_shows_the_restated_sharp_record(tg):
    notifier, sent, out = tg
    _restate(out)
    notifier.notify_weekly_summary()
    msg = sent[0]
    assert "UNDER RESTATEMENT" not in msg and "+0.50u" in msg            # restated 1.5 - 1.0


def test_daily_digest_sharp_lines_are_restated_or_under_restatement(tg):
    notifier, sent, out = tg
    notifier.notify_daily_digest()
    res = [m for m in sent if "DAILY RESULTS" in m][0]
    assert res.count("UNDER RESTATEMENT") == 2 and "+1.00u" not in res
    sent.clear()
    _restate(out)
    notifier.notify_daily_digest()
    res = [m for m in sent if "DAILY RESULTS" in m][0]
    assert "UNDER RESTATEMENT" not in res and res.count("+0.50u") == 2


def test_unrestated_settled_rows_are_excluded_not_booked(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    _sharp_ledger(tmp_path)
    (tmp_path / "corrections").mkdir()
    led = pd.read_csv(tmp_path / "sharp_ledger.csv")
    led.iloc[:1].assign(pnl_restated=1.5)[CC.SHARP_KEY + ["pnl_restated"]] \
        .to_csv(tmp_path / "corrections" / "sharp_ledger_restated.csv", index=False)
    r, st = CC.sharp_record(led)
    assert st == {"state": "RESTATED", "unrestated": 1, "source": st["source"]}
    assert r["pnl"].tolist()[0] == 1.5 and pd.isna(r["pnl"].tolist()[1])
    assert r["pnl_stored"].tolist() == [2.0, -1.0]


# ── V2: first-half closes out of every CLV consumer ──────────────────────────────────────────────
def _bets(out: Path):
    b = pd.DataFrame([
        {"source": "live", "generated_at": "2026-07-14 09:00", "match_date": "2026-07-16", "league": "Brazil Serie A",
         "home_team": "Botafogo", "away_team": "Santos", "side": "UNDER", "odds": 1.96, "signal_tier": "MARKSMAN",
         "model_type": "new_format", "closing_odds": 1.20, "clv_pct": 63.33, "result": "WIN", "pnl": 0.96},
        {"source": "live", "generated_at": "2026-07-14 09:00", "match_date": "2026-07-16", "league": "Brazil Serie A",
         "home_team": "Gremio", "away_team": "Bahia", "side": "OVER", "odds": 2.00, "signal_tier": "MARKSMAN",
         "model_type": "new_format", "closing_odds": 2.10, "clv_pct": -4.76, "result": "LOSS", "pnl": -1.0},
    ])
    b.to_csv(out / "bets_ledger.csv", index=False)
    (out / "corrections").mkdir(exist_ok=True)
    b.iloc[:1].to_csv(out / "corrections" / "clv_first_half_contaminated.csv", index=False)
    return b


def test_sharp_split_ignores_quarantined_clv(tmp_path, monkeypatch):
    from telegram_bot import notifier
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    b = _bets(tmp_path)
    lines = []
    notifier._sharp_split_lines(b, lines)
    assert not any("confirm" in x for x in lines)                        # the 63% "confirm" was a 1H price
    assert any("disagree: 0W/1L" in x for x in lines)


def test_evidence_panels_clv_drops_quarantined_rows(tmp_path, monkeypatch):
    from dashboard_data import evidence_panels as EP
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    b = _bets(tmp_path)
    d = EP.load_bets(bets=b, side_bets=pd.DataFrame(), from_date="2026-01-01")
    assert d["clv"].notna().sum() == 1 and round(float(d["clv"].dropna().iloc[0]), 4) == -0.0476


def test_exclusion_keeps_the_stored_value_beside(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    b = _bets(tmp_path)
    c = CC.exclude_contaminated_clv(b.assign(side=b["side"].str.lower()))   # key match is case-normalised
    assert c["clv_quarantined"].tolist() == [True, False]
    assert pd.isna(c["clv_pct"].iloc[0]) and c["clv_pct_stored"].iloc[0] == 63.33
    assert c["closing_odds_stored"].iloc[0] == 1.20 and c["pnl"].iloc[0] == 0.96   # P/L untouched


def test_dashboard_clv_panel_reads_through_the_exclusion():
    src = next((REPO / "pages").glob("1_*Dashboard.py")).read_text(encoding="utf-8")
    i = src.index("def _clv_rows")
    assert "exclude_contaminated_clv(" in src[i:i + 600]


def test_committed_clv_sidecar_matches_the_finding():
    s = pd.read_csv(REPO / "output" / "corrections" / "clv_first_half_contaminated.csv")
    assert len(s) == 170 and set(s["model_type"]) == {"new_format"} and (s["clv_pct"].abs() > 50).all()
    b = pd.read_csv(REPO / "output" / "bets_ledger.csv", usecols=["match_date", "league", "home_team", "away_team",
                                                                    "side", "clv_pct", "model_type"], low_memory=False)
    clean = CC.exclude_contaminated_clv(b)
    nf = clean[clean["model_type"] == "new_format"]
    assert int(clean["clv_quarantined"].sum()) == 170
    # The finding is pinned to the rows the sidecar lists (a committed, fixed file), not to the live
    # ledger: bots append settled bets, so the ledger-wide means move (7.94 -> 7.75 stored as of
    # 2026-11-06). Every listed row is found once in the ledger, with the stored CLV the sidecar recorded.
    assert round(float(s["clv_pct"].mean()), 2) == 48.15
    q = clean[clean["clv_quarantined"]]
    assert sorted(pd.to_numeric(q["clv_pct_stored"], errors="coerce").round(2)) == sorted(s["clv_pct"].round(2))
    assert q["clv_pct"].isna().all()
    # The ledger-wide means, recomputed from the committed file without the module under test
    raw = b[b["model_type"] == "new_format"]
    raw_v = pd.to_numeric(raw["clv_pct"], errors="coerce")
    key = ["match_date", "league", "home_team", "away_team", "side"]
    listed = set(map(tuple, s[key].astype(str).apply(lambda c: c.str.strip()).values))
    in_list = raw[key].astype(str).apply(lambda c: c.str.strip()).apply(tuple, axis=1).isin(listed)
    stored_mean = float(pd.to_numeric(nf["clv_pct_stored"], errors="coerce").mean())
    clean_mean = float(pd.to_numeric(nf["clv_pct"], errors="coerce").mean())
    assert stored_mean == pytest.approx(float(raw_v.mean()))
    assert clean_mean == pytest.approx(float(raw_v[~in_list].mean()))
    assert int(in_list.sum()) == 170
    # the finding itself: the listed closes inflate the stored mean; no unlisted new_format row
    # carries a first-half-sized |CLV| (the detection rule's threshold)
    assert stored_mean > clean_mean
    assert not (raw_v[~in_list].abs() > CC.CLV_CONTAMINATION_ABS).any()
    a = pd.read_csv(REPO / "output" / "corrections" / "clv_first_half_contaminated_archive_rows.csv")
    assert len(a) and set(a["status"]) == {"QUARANTINED"}


# ── HT: unpriced bets out of P/L ─────────────────────────────────────────────────────────────────
def test_ht_priced_mask_honours_the_unpriced_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    h = pd.DataFrame([
        {"source": "live", "generated_at": "2026-10-01 10:00", "match_date": "2026-10-02", "league": "League One",
         "home_team": "A", "away_team": "B", "market": "ht_under15", "side": "UNDER", "entry_odds": None,
         "fair_odds": 1.28, "result": "WIN", "pnl": 0.28},
        {"source": "live", "generated_at": "2026-10-01 10:00", "match_date": "2026-10-02", "league": "League One",
         "home_team": "C", "away_team": "D", "market": "ht_under15", "side": "UNDER", "entry_odds": 1.40,
         "fair_odds": 1.28, "result": "LOSS", "pnl": -1.0},
        {"source": "live", "generated_at": "2026-10-01 10:00", "match_date": "2026-10-02", "league": "League One",
         "home_team": "E", "away_team": "F", "market": "ht_under15", "side": "UNDER", "entry_odds": 1.30,
         "fair_odds": 1.28, "result": "WIN", "pnl": 0.30},
    ])
    (tmp_path / "corrections").mkdir()
    h.iloc[[2]].to_csv(tmp_path / "corrections" / "ht_ledger_unpriced.csv", index=False)
    assert CC.ht_priced_mask(h).tolist() == [False, True, False]


def test_committed_ht_sidecar_lists_the_47_unpriced_bets():
    s = pd.read_csv(REPO / "output" / "corrections" / "ht_ledger_unpriced.csv")
    h = pd.read_csv(REPO / "output" / "ht_ledger.csv")
    settled = h[h["result"].isin(["WIN", "LOSS"])]
    assert len(s) == 47 and round(float(s["pnl"].sum()), 2) == -12.16
    real = settled[CC.ht_priced_mask(settled)]
    assert len(real) == 14 and round(float(real["pnl"].sum()), 2) == 5.57
