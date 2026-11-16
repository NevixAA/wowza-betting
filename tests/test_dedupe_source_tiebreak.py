"""Duplicate fixtures of EQUAL completeness: the primary source wins, whatever the input order.

Regression for Pro "Standard Repair" run 43917999786 (2026-11-16): Antwerp v Beerschot VA
(2024-09-29) existed in football-data and in the backtest archive with the same number of populated
fields. _canonicalise_clubs kept whichever copy sorted last, and that order came from an unstable
date sort over the whole merged frame, so unrelated new rows flipped it to the archive copy. A lone
BACKTEST_ARCHIVE row in a football-data season then broke Pro's source-regime test.

    python -m pytest tests/test_dedupe_source_tiebreak.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as DL  # noqa: E402

P = DL.SOURCE_PRIORITY
COL = DL.SOURCE_PRIORITY_COL


def _row(price, prio, corners=np.nan):
    return {"date": pd.Timestamp("2024-09-29"), "league": "Belgian First Division A", "season": "2024/25",
            "home_team": "Antwerp", "away_team": "Beerschot VA", "home_goals": 5.0, "away_goals": 0.0,
            "odds_over25": price, "home_corners": corners, COL: prio}


def _filler(n):
    # unrelated fixtures on the same date: their count must never decide the tie
    return [{"date": pd.Timestamp("2024-09-29"), "league": "Belgian First Division A", "season": "2024/25",
             "home_team": f"Club {i}", "away_team": f"Club {i + 100}", "home_goals": 1.0, "away_goals": 1.0,
             "odds_over25": 1.9, "home_corners": np.nan, COL: P["football_data_cache"]} for i in range(n)]


def _winner(rows):
    out = DL._canonicalise_clubs(pd.DataFrame(rows))
    m = out["home_team"].eq("Antwerp")
    assert int(m.sum()) == 1, "the duplicate fixture must collapse to exactly one row"
    return out.loc[m].iloc[0]


def test_tie_goes_to_football_data_in_either_order_and_any_frame_size():
    fd = _row(1.37, P["football_data_cache"])
    arch = _row(1.40, P["backtest_archive"])
    for n in (0, 1, 7, 50):
        for rows in ([fd, arch] + _filler(n), [arch, fd] + _filler(n), _filler(n) + [arch, fd]):
            w = _winner(rows)
            assert w["odds_over25"] == 1.37
            assert w[COL] == P["football_data_cache"]


def test_fresh_download_beats_cache_and_api_football_beats_archive_on_a_tie():
    assert _winner([_row(1.37, P["football_data_download"]), _row(1.50, P["football_data_cache"])])["odds_over25"] == 1.37
    assert _winner([_row(1.50, P["football_data_cache"]), _row(1.37, P["football_data_download"])])["odds_over25"] == 1.37
    assert _winner([_row(1.37, P["api_football"]), _row(1.40, P["backtest_archive"])])["odds_over25"] == 1.37
    assert _winner([_row(1.40, P["backtest_archive"]), _row(1.37, P["api_football"])])["odds_over25"] == 1.37


def test_completeness_still_decides_first():
    # the richer copy wins even from the lowest-priority source: priority only breaks ties
    w = _winner([_row(1.37, P["football_data_download"]), _row(1.40, P["backtest_archive"], corners=6.0)])
    assert w["odds_over25"] == 1.40 and w["home_corners"] == 6.0


def test_priority_column_is_not_counted_as_a_populated_field():
    # a row whose only extra non-null cell is the priority column must not out-rank a real field
    a = _row(1.37, P["football_data_download"])
    b = _row(1.40, P["backtest_archive"], corners=6.0)
    a.pop(COL)
    w = _winner([a, b])
    assert w["home_corners"] == 6.0


def test_without_priority_column_old_rule_holds():
    rows = [_row(1.37, 0), _row(1.40, 0, corners=6.0)]
    for r in rows:
        r.pop(COL)
    w = _winner(rows)
    assert w["home_corners"] == 6.0 and COL not in w.index


def test_load_all_matches_tags_sources_and_drops_the_column():
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8")
    body = src[src.index("def load_all_matches"):]
    assert 'out.sort_values("date", kind="mergesort")' in body
    i_canon = body.index("out = _canonicalise_clubs(out)")
    assert "out.drop(columns=[SOURCE_PRIORITY_COL]" in body[i_canon:i_canon + 200]
    assert 'SOURCE_PRIORITY["backtest_archive"]' in body
