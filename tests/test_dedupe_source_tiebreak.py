"""Duplicate fixtures of EQUAL completeness: the primary source wins, whatever the input order.

Port of main PR #38 (931c7671d) to data-truth-100. Regression for Pro "Standard Repair" run
43917999786 (2026-11-16): Antwerp v Beerschot VA (2024-09-29) existed in football-data and in the
backtest archive with the same number of populated fields. _canonicalise_clubs kept whichever copy
sorted last, and that order came from an unstable date sort over the whole merged frame, so
unrelated new rows could flip it to the archive copy.

On data-truth-100 every merged row already carries `row_source`, and the one-match de-duplication
(src.integrity.duplicate_matches.SOURCE_PRIORITY) already ranks sources by it. _canonicalise_clubs
now breaks a completeness tie by that SAME rule, so the two de-duplication steps agree.

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
from src.integrity.duplicate_matches import SOURCE_PRIORITY  # noqa: E402


def _row(price, src, corners=np.nan):
    r = {"date": pd.Timestamp("2024-09-29"), "league": "Belgian First Division A", "season": "2024/25",
         "home_team": "Antwerp", "away_team": "Beerschot VA", "home_goals": 5.0, "away_goals": 0.0,
         "odds_over25": price, "home_corners": corners}
    if src is not None:
        r["row_source"] = src
    return r


def _filler(n):
    # unrelated fixtures on the same date: their count must never decide the tie
    return [{"date": pd.Timestamp("2024-09-29"), "league": "Belgian First Division A", "season": "2024/25",
             "home_team": f"Club {i}", "away_team": f"Club {i + 100}", "home_goals": 1.0, "away_goals": 1.0,
             "odds_over25": 1.9, "home_corners": np.nan, "row_source": "fd_cache"} for i in range(n)]


def _winner(rows):
    out = DL._canonicalise_clubs(pd.DataFrame(rows))
    m = out["home_team"].eq("Antwerp")
    assert int(m.sum()) == 1, "the duplicate fixture must collapse to exactly one row"
    return out.loc[m].iloc[0]


def test_tie_goes_to_football_data_in_either_order_and_any_frame_size():
    fd = _row(1.37, "fd_cache")
    arch = _row(1.40, "backtest_archive")
    for n in (0, 1, 7, 50):
        for rows in ([fd, arch] + _filler(n), [arch, fd] + _filler(n), _filler(n) + [arch, fd]):
            w = _winner(rows)
            assert w["odds_over25"] == 1.37
            assert w["row_source"] == "fd_cache"


def test_source_order_matches_the_one_match_rule():
    # football_data > fd_cache > api_football > backtest_archive, in either input order
    for hi, lo in (("football_data", "fd_cache"), ("fd_cache", "api_football"),
                   ("api_football", "backtest_archive"), ("football_data", "backtest_archive")):
        assert SOURCE_PRIORITY[hi] < SOURCE_PRIORITY[lo]
        assert _winner([_row(1.37, hi), _row(1.50, lo)])["odds_over25"] == 1.37
        assert _winner([_row(1.50, lo), _row(1.37, hi)])["odds_over25"] == 1.37


def test_unknown_source_ranks_last():
    assert _winner([_row(1.37, "backtest_archive"), _row(1.50, "someone_else")])["odds_over25"] == 1.37
    assert _winner([_row(1.50, "someone_else"), _row(1.37, "backtest_archive")])["odds_over25"] == 1.37


def test_completeness_still_decides_first():
    # the richer copy wins even from the lowest-priority source: the source only breaks ties
    w = _winner([_row(1.37, "football_data"), _row(1.40, "backtest_archive", corners=6.0)])
    assert w["odds_over25"] == 1.40 and w["home_corners"] == 6.0


def test_no_helper_column_left_behind():
    out = DL._canonicalise_clubs(pd.DataFrame([_row(1.37, "fd_cache"), _row(1.40, "backtest_archive")]))
    assert list(out.columns) == list(pd.DataFrame([_row(1.37, "fd_cache")]).columns)


def test_without_row_source_old_rule_holds():
    w = _winner([_row(1.37, None), _row(1.40, None, corners=6.0)])
    assert w["home_corners"] == 6.0 and "row_source" not in w.index


def test_loader_date_sort_before_identity_is_stable():
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8")
    body = src[src.index("def load_all_matches"):]
    i_ident = body.index("out = one_club_one_match(out)")
    assert 'out = out.sort_values("date", kind="mergesort").reset_index(drop=True)' in body[:i_ident]
