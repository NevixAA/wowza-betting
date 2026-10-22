"""Rolling live file + per-match-date archive (src/rolling_store.py, 2026-10-22).

The point of the change is to stop git re-storing the full odds history on every run WITHOUT
changing what any writer dedupes or any reader sees. These tests pin exactly that.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import src.rolling_store as rs

TODAY = date.today()


def _d(offset: int) -> str:
    return (TODAY + timedelta(days=offset)).strftime("%Y-%m-%d")


def _book_rows():
    """Book-snapshot shaped rows across past, recent and upcoming match dates."""
    rows = []
    for md_off in (-30, -9, -3, -2, -1, 0, 3):
        for k, odds in enumerate((1.91, 1.95, 1.91)):
            rows.append({"snapshot_ts": f"{_d(md_off - 11)}T1{k}:00:00Z", "league": "L",
                         "match_date": _d(md_off), "kickoff_utc": f"{_d(md_off)}T19:00:00Z",
                         "match": f"A{md_off} vs B", "bookmaker": "bk", "market": "OU25",
                         "side": "OVER", "odds": odds})
    rows.append({**rows[0], "match_date": "not-a-date"})          # never archived
    return pd.DataFrame(rows)


def _write(df, p):
    df.to_csv(p, index=False, lineterminator="\n")


def _canon(df):
    return (df.astype(str).sort_values(list(df.columns)).reset_index(drop=True))


def test_roll_moves_only_finished_match_dates_and_is_idempotent(tmp_path):
    live = tmp_path / "book_odds_snapshots.csv"
    orig = _book_rows()
    _write(orig, live)
    s1 = rs.roll(live, keep_days=2)
    assert s1["moved"] == 9 and s1["parts"] == 3                 # -30, -9, -3
    adir = tmp_path / "book_odds_snapshots_archive"
    assert sorted(p.stem for p in adir.glob("*.csv")) == sorted([_d(-30), _d(-9), _d(-3)])
    left = pd.read_csv(live, dtype=str)
    assert left["match_date"].min() >= _d(-2) or "not-a-date" in set(left["match_date"])
    assert "not-a-date" in set(left["match_date"])
    before = {p.name: p.read_bytes() for p in adir.glob("*.csv")}
    live_before = live.read_bytes()
    s2 = rs.roll(live, keep_days=2)
    assert s2["moved"] == 0
    assert live.read_bytes() == live_before
    assert {p.name: p.read_bytes() for p in adir.glob("*.csv")} == before


def test_read_all_equals_the_original_content(tmp_path):
    live = tmp_path / "book_odds_snapshots.csv"
    orig = _book_rows()
    _write(orig, live)
    original_text = pd.read_csv(live, dtype=str, keep_default_na=False)
    rs.roll(live, keep_days=2)
    back = rs.read_all(live, dtype=str, keep_default_na=False)
    assert len(back) == len(original_text)
    pd.testing.assert_frame_equal(_canon(back), _canon(original_text))


def test_archive_and_live_files_are_written_with_lf(tmp_path):
    live = tmp_path / "x.csv"
    _write(_book_rows(), live)
    rs.roll(live, keep_days=2)
    for p in [live, *(tmp_path / "x_archive").glob("*.csv")]:
        assert b"\r\n" not in p.read_bytes()


def test_rolling_twice_into_an_existing_part_does_not_duplicate(tmp_path):
    """Two runs that both archive the same day (or a union merge re-adding rows) self-heal."""
    live = tmp_path / "h.csv"
    orig = _book_rows()
    _write(orig, live)
    rs.roll(live, keep_days=2)
    _write(orig, live)                                            # same rows come back
    rs.roll(live, keep_days=2)
    part = pd.read_csv(tmp_path / "h_archive" / f"{_d(-30)}.csv")
    assert len(part) == 3


def test_read_all_dates_filter_and_missing_everything(tmp_path):
    live = tmp_path / "h.csv"
    assert rs.read_all(live).empty
    _write(_book_rows(), live)
    rs.roll(live, keep_days=2)
    only = rs.read_all(live, dates=[_d(-9)], dtype=str)
    assert set(only["match_date"]) - {_d(-9)} <= set(pd.read_csv(live, dtype=str)["match_date"])
    assert _d(-30) not in set(only["match_date"])


def test_roll_never_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_bytes(b"\x00\x01garbage")
    rs.roll(p)                                                    # must not raise
    rs.roll(tmp_path / "missing.csv")


# ── writers ──────────────────────────────────────────────────────────────────

def test_book_writer_dedupe_unchanged_for_an_upcoming_match_with_old_first_quotes(tmp_path,
                                                                                  monkeypatch):
    """An upcoming match first quoted >7 days ago keeps its whole curve in the live file, so the
    consecutive-distinct dedupe still sees the previous price and writes nothing for an unmoved
    book -- exactly as before the file became rolling."""
    import config
    from src import btts_odds
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    live = tmp_path / "book_odds_snapshots.csv"
    _write(_book_rows(), live)
    btts_odds.append_book_rows([])                                # no-op
    up = _d(3)
    first_seen = pd.read_csv(live).query("match_date == @up")["snapshot_ts"].min()
    assert first_seen < _d(-7)                                    # earliest quote >7 days old
    unmoved = {"snapshot_ts": f"{_d(0)}T12:00:00Z", "league": "L", "match_date": up,
               "kickoff_utc": f"{up}T19:00:00Z", "match": "A3 vs B", "bookmaker": "bk",
               "market": "OU25", "side": "OVER", "odds": 1.91}
    btts_odds.append_book_rows([unmoved])
    d = pd.read_csv(live)
    assert (d["match_date"] == up).sum() == 3                    # unmoved -> nothing written
    assert d["match_date"].min() >= _d(-2) or "not-a-date" in set(d["match_date"])
    btts_odds.append_book_rows([{**unmoved, "odds": 2.05}])       # moved -> one row
    d = pd.read_csv(live)
    assert (d["match_date"] == up).sum() == 4
    full = rs.read_all(live)
    assert len(full) == len(_book_rows()) + 1


def _prop_rec(md, odds, player="harry kane"):
    return {"match_date": md, "league": "Premier League", "match": "X vs Y", "player": player,
            "market": "goals", "odds": odds, "source": "oddsapi", "bookmaker": ""}


def test_prop_history_writer_rolls_and_keeps_first_seen_dedupe(tmp_path, monkeypatch):
    from player_model import odds_fetcher as of
    live = tmp_path / "player_prop_odds_history.csv"
    monkeypatch.setattr(of, "_ODDS_HISTORY_FILE", live)
    old = pd.DataFrame([{**_prop_rec(_d(-10), 3.0), "snapshot_date": _d(-12),
                         "snapshot_ts": f"{_d(-12)}T10:00:00Z"},
                        {**_prop_rec(_d(4), 3.5), "snapshot_date": _d(-8),
                         "snapshot_ts": f"{_d(-8)}T10:00:00Z"}])
    old = old.reindex(columns=["snapshot_date", "snapshot_ts", "match_date", "league", "match",
                               "player", "market", "odds", "source", "bookmaker"])
    _write(old, live)
    of._append_odds_history([_prop_rec(_d(4), 3.5),               # same price, first seen 12d ago
                             _prop_rec(_d(4), 3.8)])              # new price
    d = pd.read_csv(live)
    assert list(d["odds"]) == [3.5, 3.8]                          # dedupe intact, old match gone
    assert (tmp_path / "player_prop_odds_history_archive" / f"{_d(-10)}.csv").exists()
    assert len(rs.read_all(live)) == 3


def test_quotes_go_to_a_daily_file_and_dedupe_against_previous_days(tmp_path, monkeypatch):
    from player_model import odds_fetcher as of
    monkeypatch.setattr(of, "_QUOTES_DIR", tmp_path)
    q = {"match_date": _d(3), "league": "L", "match": "X vs Y", "player": "p", "market": "goals",
         "side": "yes", "point": None, "odds": 4.0, "bookmaker": "bk", "source": "oddsapi"}
    prev = pd.DataFrame([{**q, "snapshot_ts": f"{_d(-1)}T10:00:00Z"}]) \
        .reindex(columns=["snapshot_ts", *of._QUOTE_KEY, "source"])
    prev.to_csv(tmp_path / f"{_d(-1)}.csv", index=False)
    of._append_quotes([q, {**q, "odds": 4.5}])
    today = tmp_path / f"{TODAY:%Y-%m-%d}.csv"
    assert today.exists()
    d = pd.read_csv(today)
    assert list(d["odds"]) == [4.5]                               # 4.0 already seen yesterday
    of._append_quotes([{**q, "odds": 4.5}])
    assert len(pd.read_csv(today)) == 1
    assert not list(tmp_path.glob("????-??.csv"))                 # no monthly file any more


def test_quotes_dedupe_against_the_frozen_legacy_monthly_file(tmp_path, monkeypatch):
    """prop_quotes/2026-10.csv is left in place (frozen) by the migration; a price already in it
    must not be re-logged into a day-file."""
    from player_model import odds_fetcher as of
    monkeypatch.setattr(of, "_QUOTES_DIR", tmp_path)
    q = {"match_date": _d(3), "league": "L", "match": "X vs Y", "player": "p", "market": "sot",
         "side": "over", "point": 0.5, "odds": 1.8, "bookmaker": "bk", "source": "oddsapi"}
    pd.DataFrame([{**q, "snapshot_ts": f"{_d(-2)}T10:00:00Z"}]) \
        .reindex(columns=["snapshot_ts", *of._QUOTE_KEY, "source"]) \
        .to_csv(tmp_path / f"{TODAY:%Y-%m}.csv", index=False)
    of._append_quotes([q])
    assert not (tmp_path / f"{TODAY:%Y-%m-%d}.csv").exists()


# ── readers ──────────────────────────────────────────────────────────────────

def test_clv_close_out_finds_a_close_that_was_archived(tmp_path, monkeypatch):
    from player_model import clv_tracker
    from src import clv_capture
    live = tmp_path / "player_prop_odds_history.csv"
    md = _d(-6)
    h = pd.DataFrame([{"snapshot_date": md, "snapshot_ts": f"{md}T{t}:00:00Z", "match_date": md,
                       "league": "PL", "match": "X vs Y", "player": "harry kane",
                       "market": "goals", "odds": o, "source": "oddsapi", "bookmaker": ""}
                      for t, o in (("10", 3.0), ("18", 2.6))])
    _write(h, live)
    rs.roll(live)
    assert pd.read_csv(live).empty
    monkeypatch.setattr(clv_capture, "_load",
                        lambda: [{"bet_id": f"PLAYER|{md}|Harry Kane|goals", "odds_close": ""}])
    closed = []
    monkeypatch.setattr(clv_capture, "capture_close", lambda b, o, u: closed.append((b, o)))
    assert clv_tracker.close_out(odds_history_path=live) == 1
    assert closed == [(f"PLAYER|{md}|Harry Kane|goals", 2.6)]


def test_curve_index_sees_archived_history(tmp_path):
    from src.market_movement import CurveIndex
    live = tmp_path / "player_prop_odds_history.csv"
    md = _d(-6)
    h = pd.DataFrame([{"snapshot_date": md, "snapshot_ts": f"{md}T1{t}:00:00Z", "match_date": md,
                       "league": "PL", "match": "X vs Y", "player": "p", "market": "goals",
                       "odds": 2.0 + t} for t in range(3)])
    _write(h, live)
    rs.roll(live)
    idx = CurveIndex(sources=[("player_prop_odds_history.csv", "props")], out_dir=tmp_path)
    assert idx.n_rows == 3
