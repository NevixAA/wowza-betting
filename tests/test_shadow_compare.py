"""Tests for the prospective shadow log and the multiple-comparison controls."""
from __future__ import annotations
import numpy as np, pandas as pd, pytest
import src.shadow_compare as sc


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SHADOW_FILE", tmp_path / "v91_shadow.csv")


def _rows(n=5, day="2026-10-05"):
    return [{"fixture_id": i, "logged_at": "2026-10-02T09:00:00", "match_date": day,
             "league": "L", "p_v9": 0.5, "p_v91": 0.52} for i in range(n)]


def test_append_writes_the_full_schema():
    sc.append_shadow(_rows())
    d = pd.read_csv(sc.SHADOW_FILE)
    assert list(d.columns) == sc.COLUMNS


def test_a_fixture_is_logged_once_however_often_predict_runs():
    """predict runs every 15 minutes; a re-log would turn a prospective record retrospective."""
    assert sc.append_shadow(_rows()) == 5
    assert sc.append_shadow(_rows()) == 0
    assert len(pd.read_csv(sc.SHADOW_FILE)) == 5


def _rows_from(start, n, day="2026-10-05"):
    return [{**r, "fixture_id": r["fixture_id"] + start} for r in _rows(n, day)]


def test_new_fixtures_still_append():
    sc.append_shadow(_rows(3))
    assert sc.append_shadow(_rows_from(10, 3, day="2026-10-06")) == 3
    assert len(pd.read_csv(sc.SHADOW_FILE)) == 6


def test_a_moved_kickoff_never_logs_the_fixture_twice():
    """2026-11-23 data truth 9 RED: Rosario Central v Barracas Central was first logged for 2026-11-28
    15:00; the provider moved it to 2026-11-29 00:00 and the next predict run appended a second row,
    because the key held the match date. First sight wins: the fixture id x market is logged once."""
    first = [{"fixture_id": "3c55be4d38980e11540c96a994695d51", "market": "ou25",
              "logged_at": "2026-11-23T02:06:03+00:00", "match_date": "2026-11-28 00:00:00",
              "kickoff_utc": "2026-11-28 15:00:00", "p_v9": 0.4826}]
    moved = [{**first[0], "logged_at": "2026-11-23T08:30:40+00:00", "match_date": "2026-11-29 00:00:00",
              "kickoff_utc": "2026-11-29 00:00:00", "p_v9": 0.4827}]
    assert sc.append_shadow(first) == 1
    assert sc.append_shadow(moved) == 0
    d = pd.read_csv(sc.SHADOW_FILE, dtype=str)
    assert len(d) == 1 and d.iloc[0]["match_date"] == "2026-11-28 00:00:00"
    assert d.iloc[0]["p_v9"] == "0.4826"                      # the first-sight opinion, never revised
    # another market on the same fixture is a different unit and still appends
    assert sc.append_shadow([{**moved[0], "market": "btts"}]) == 1


def test_one_batch_never_writes_a_fixture_twice():
    rows = _rows(2) + [{**_rows(1)[0], "match_date": "2026-10-06", "p_v9": 0.9}]
    assert sc.append_shadow(rows) == 2
    d = pd.read_csv(sc.SHADOW_FILE)
    assert len(d) == 2 and d.loc[d["fixture_id"] == 0, "p_v9"].iloc[0] == 0.5   # first row wins
    assert sc.append_shadow(rows) == 0


def test_rows_without_a_fixture_id_are_never_merged_across_dates():
    rows = [{"fixture_id": "", "match_date": "2026-10-05", "league": "L"},
            {"fixture_id": "", "match_date": "2026-10-06", "league": "L"}]
    assert sc.append_shadow(rows) == 2


def test_prospective_window_excludes_rows_logged_before_registration():
    sc.append_shadow([
        {"fixture_id": 1, "logged_at": "2026-09-01T00:00:00", "match_date": "2026-10-05",
         "result": "WIN"},                                   # played after, PREDICTED before
        {"fixture_id": 2, "logged_at": "2026-10-03T00:00:00", "match_date": "2026-10-05",
         "result": "LOSS"}])
    w = sc.prospective_window("2026-10-02")
    assert list(w["fixture_id"]) == [2], \
        "a fixture predicted before registration is not prospective, whenever it was played"


def test_prospective_window_only_returns_settled_rows():
    sc.append_shadow([
        {"fixture_id": 1, "logged_at": "2026-10-03T00:00:00", "match_date": "2026-10-05",
         "result": "WIN"},
        {"fixture_id": 2, "logged_at": "2026-10-03T00:00:00", "match_date": "2026-10-05"}])
    assert len(sc.prospective_window("2026-10-02")) == 1


# ── FDR ───────────────────────────────────────────────────────────────────────────────────────
def test_fdr_rejects_a_lone_lucky_cell_among_many_nulls():
    """The movement study's exact shape: 19 cells, one at p~0.05, nothing real."""
    p = [0.047] + list(np.linspace(0.2, 0.98, 18))
    out = sc.fdr(p, q=0.10)
    assert not out["survives"].any(), "a single p=0.047 among 19 tests must not survive BH"


def test_fdr_keeps_genuinely_strong_results():
    p = [1e-6, 1e-5, 1e-4] + list(np.linspace(0.3, 0.99, 17))
    assert sc.fdr(p, q=0.10)["survives"].sum() >= 3


def test_fdr_handles_empty_input():
    assert len(sc.fdr([])) == 0


def test_whites_reality_check_punishes_the_winner_of_a_search():
    """A best-of-many statistic must be judged against the distribution of best-of-many."""
    rng = np.random.default_rng(0)
    boot_best = np.array([rng.normal(0, 1, 40).max() for _ in range(2000)])
    assert sc.whites_reality_check(1.9, boot_best) > 0.05, \
        "a value that is unremarkable as a maximum of 40 draws must not read as significant"
    assert sc.whites_reality_check(5.0, boot_best) < 0.05
