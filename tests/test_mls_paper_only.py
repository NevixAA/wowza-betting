"""Paper leagues: collect EVERYTHING, send nothing, count nothing, bet nothing.

USA MLS became a paper league on 2026-10-07 (owner instruction). The first attempt, on
2026-10-06, removed it from ENABLED_LEAGUES — and that stopped far more than tips: sharp_tracker
and live_scanner both iterate ENABLED_LEAGUES, so MLS movement and in-play collection stopped
too, and with no tips there was no CLV or shadow-log record. The owner wants all of it kept so
the league can be upgraded later on evidence.

So MLS stays enabled for COLLECTION, and every send path, every KPI and the betting bot filter
it out. These tests pin both halves — the collection that must keep happening and the leaks that
must not.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

import config

MLS = "USA MLS"
ROOT = Path(__file__).resolve().parents[1]


# ── collection must keep running ──────────────────────────────────────────────────────────────
def test_mls_is_still_predicted_and_collected():
    """ENABLED_LEAGUES drives prediction, sharp tracking and the live scanner. Dropping MLS
    from it was what stopped its movement data on 2026-10-06."""
    assert MLS in config.ENABLED_LEAGUES


def test_mls_still_trains_and_has_an_odds_source():
    assert MLS in config.NEW_FORMAT_LEAGUES
    assert MLS in config.ODDS_API_SPORT_KEYS


@pytest.mark.parametrize("path", ["src/sharp_tracker.py", "src/live_scanner.py"])
def test_the_collectors_that_iterate_enabled_leagues_will_see_mls(path):
    """If either collector stopped reading ENABLED_LEAGUES this test would be meaningless —
    pin the dependency the paper-league design relies on."""
    assert "ENABLED_LEAGUES" in (ROOT / path).read_text(encoding="utf-8")


# ── it is a paper league ──────────────────────────────────────────────────────────────────────
def test_mls_is_a_paper_league():
    assert config.is_paper_league(MLS)
    assert config.is_paper_league(" USA MLS ")      # whitespace in a ledger must not leak it
    assert not config.is_paper_league("Argentina Primera Division")


def test_drop_paper_leagues_removes_only_paper_rows():
    d = pd.DataFrame({"league": [MLS, "La Liga 2", MLS, "Serie B"], "pnl": [1, 2, 3, 4]})
    out = config.drop_paper_leagues(d)
    assert list(out["league"]) == ["La Liga 2", "Serie B"]


def test_drop_paper_leagues_is_safe_on_frames_without_a_league_column():
    d = pd.DataFrame({"x": [1, 2]})
    assert config.drop_paper_leagues(d).equals(d)
    assert config.drop_paper_leagues(pd.DataFrame()).empty


# ── nothing is sent ───────────────────────────────────────────────────────────────────────────
def test_every_notifier_read_goes_through_the_paper_filter():
    """37 CSV reads feed Telegram sends and digest KPIs. One direct pd.read_csv that bypasses
    the filter is one path by which a paper-league tip reaches the channel."""
    src = (ROOT / "telegram_bot" / "notifier.py").read_text(encoding="utf-8")
    direct = len(re.findall(r"pd\.read_csv\(", src))
    assert direct == 1, f"{direct} direct pd.read_csv calls — only the filtered helper may call it"
    assert "drop_paper_leagues(pd.read_csv(" in src


def test_the_notifier_reader_actually_drops_mls(tmp_path):
    from telegram_bot import notifier
    f = tmp_path / "x.csv"
    pd.DataFrame({"league": [MLS, "Serie B"], "pnl": [1, 2]}).to_csv(f, index=False)
    assert list(notifier._read_csv(f)["league"]) == ["Serie B"]


def test_the_pipeline_strips_paper_leagues_from_the_send_sources_only():
    """bets.csv / side_bets.csv are what the senders read; the ledgers are the collection
    record and must still receive the rows."""
    src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert "config.drop_paper_leagues(bets)" in src
    assert "config.drop_paper_leagues(side_bets).to_csv" in src
    # the ledger appends still receive the UNFILTERED frames
    assert "append_tips(bets)" in src and "append_side_market_tips(side_bets)" in src


# ── nothing is counted ────────────────────────────────────────────────────────────────────────
def test_the_dashboard_ledger_excludes_mls_by_default():
    from dashboard_data import v9
    d = v9.ledger()
    if d.empty:
        pytest.skip("no ledger in this checkout")
    assert MLS not in set(d["league"].astype(str))
    full = v9.ledger(include_paper_leagues=True)
    assert len(full) >= len(d), "the raw record must still be reachable"


@pytest.mark.parametrize("page", ["pages/12_💼_Portfolio.py", "pages/1_📊_Dashboard.py",
                                  "pages/5_🎯_Success_Rates.py", "app.py"])
def test_every_page_that_reads_a_ledger_directly_filters_it(page):
    src = (ROOT / page).read_text(encoding="utf-8")
    assert len(re.findall(r"pd\.read_csv\(", src)) == 1, f"{page} bypasses the paper filter"
    assert "drop_paper_leagues(pd.read_csv(" in src


# ── nothing is bet ────────────────────────────────────────────────────────────────────────────
def test_the_cloudbet_bot_never_sees_a_paper_league():
    """The bot reads the ledger, which DOES contain MLS — so without its own filter a paper
    league would reach real money by the one path that bypasses Telegram."""
    src = (ROOT / "src" / "cloudbet" / "candidates.py").read_text(encoding="utf-8")
    assert "drop_paper_leagues(out)" in src


# ── the decision is documented ────────────────────────────────────────────────────────────────
def test_the_evidence_is_recorded_on_the_staked_basis():
    """An earlier comment quoted all-tier figures, which overstated the case. The record must
    carry the staked numbers, including that the CI includes zero."""
    src = (ROOT / "config.py").read_text(encoding="utf-8")
    block = src[src.index("PAPER_LEAGUES: set"):] if "PAPER_LEAGUES: set" in src else ""
    head = src[src.index("# ── Paper leagues"):src.index("PAPER_LEAGUES: set")]
    assert "72 bets over 20 matchdays" in head and "includes zero" in head
    assert "USA MLS" in block


def test_no_league_was_dropped_by_accident():
    expected = {
        "League One", "League Two", "Bundesliga 2", "La Liga 2", "Ligue 2",
        "Championship", "Serie B",
        "Austrian Bundesliga", "Sweden Allsvenskan", "Norway Eliteserien",
        "Finland Veikkausliiga", "Denmark Superliga", "Ireland Premier Division",
        "Argentina Primera Division", "Brazil Serie A", "Japan J-League",
        "Mexico Liga MX", "China Super League", "USA MLS", "Romanian Superliga",
    }
    assert config.ENABLED_LEAGUES == expected, (
        f"unexpected: +{config.ENABLED_LEAGUES - expected} -{expected - config.ENABLED_LEAGUES}")
