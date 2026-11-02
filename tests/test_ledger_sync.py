"""Ledger race (estate scan 2026-11-02, P1): concurrent CI writers must never drop each other's rows.

The workflows used to publish ledgers with `git pull --rebase -X theirs`, which resolves a
conflicting hunk wholesale in favour of the run being replayed. These tests pin
  * the defect itself, on a real throwaway git repo (test_rebase_x_theirs_drops_a_concurrent_row);
  * the row-level three-way merge that replaces it (src.ledger.merge_ledger_rows);
  * sync-snapshot + sync-push end to end against a bare "origin" with a racing writer.
"""
from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src import ledger as L

KEY = L.SYNC_LEDGERS["output/bets_ledger.csv"]
COLS = ["source", "match_date", "home_team", "away_team", "side", "odds", "result", "pnl", "notes"]


def _df(rows):
    return pd.DataFrame(rows, columns=COLS).astype(str)


def _row(home, result="", pnl="", odds="1.90", date="2026-10-30"):
    return ["live", date, home, f"{home} B", "OVER", odds, result, pnl, ""]


def test_merge_keeps_concurrent_append_and_applies_this_runs_settlement():
    base = _df([_row("A"), _row("C")])
    ours = _df([_row("A", "WIN", "0.9"), _row("C")])                    # this run settled A
    theirs = _df([_row("A"), _row("C"), _row("D", date="2026-11-03")])  # predict appended D
    merged, st = L.merge_ledger_rows(base, ours, theirs, KEY)
    assert list(merged["home_team"]) == ["A", "C", "D"]
    assert merged.loc[merged["home_team"] == "A", "result"].item() == "WIN"
    assert merged.loc[merged["home_team"] == "A", "pnl"].item() == "0.9"
    assert st["cells_applied"] == 2 and st["added"] == 0


def test_merge_appends_this_runs_new_rows_and_keeps_theirs_settlements():
    base = _df([_row("A"), _row("C")])
    ours = _df([_row("A"), _row("C"), _row("E", date="2026-11-04")])    # predict appended E
    theirs = _df([_row("A", "LOSS", "-1.0"), _row("C", "WIN", "0.9")])  # update_results settled
    merged, st = L.merge_ledger_rows(base, ours, theirs, KEY)
    assert list(merged["home_team"]) == ["A", "C", "E"]
    assert list(merged["result"]) == ["LOSS", "WIN", ""]
    assert st["added"] == 1 and st["cells_applied"] == 0


def test_merge_first_logged_row_stands_when_both_logged_it():
    base = _df([])
    ours = _df([_row("A", odds="2.10")])
    theirs = _df([_row("A", odds="1.95")])
    merged, st = L.merge_ledger_rows(base, ours, theirs, KEY)
    assert len(merged) == 1 and merged["odds"].item() == "1.95"


def test_merge_deletion_only_when_untouched_upstream():
    base = _df([_row("A"), _row("B")])
    ours = _df([_row("A")])                                              # this run removed B
    merged, st = L.merge_ledger_rows(base, ours, _df([_row("A"), _row("B")]), KEY)
    assert list(merged["home_team"]) == ["A"] and st["deleted"] == 1
    merged, st = L.merge_ledger_rows(base, ours, _df([_row("A"), _row("B", "WIN", "0.9")]), KEY)
    assert list(merged["home_team"]) == ["A", "B"] and st["delete_refused"] == 1


def test_merge_preserves_text_exactly():
    text = "source,match_date,home_team,away_team,side,odds,result,pnl,notes\n" \
           "live,2026-10-30,A,A B,OVER,1.900,,,\n"
    t = L._read_csv_text(text)
    merged, _ = L.merge_ledger_rows(t, t, t, KEY)
    buf = io.StringIO()
    merged.to_csv(buf, index=False, lineterminator="\n")
    assert buf.getvalue() == text


def test_player_key_prefers_player_id_and_falls_back_to_name():
    k = L.SYNC_LEDGERS["output/player_ledger.csv"]
    a = {"match_date": "d", "home_team": "h", "away_team": "a", "market": "sot",
         "player_id": "123", "player_name": "X"}
    b = dict(a, player_id="")
    assert L.ledger_row_key(a, k)[-1] == "player_id=123"
    assert L.ledger_row_key(b, k)[-1] == "player_name=X"


# ── real git: the defect and the fix ──────────────────────────────────────────

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git not on PATH")


def _g(cwd, *args, check=True):
    return subprocess.run(["git", "-C", str(cwd), *args], check=check, capture_output=True, text=True)


def _csv(rows):
    buf = io.StringIO()
    _df(rows).to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


def _setup(tmp_path, rows):
    origin = tmp_path / "origin.git"
    _g(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    seed = tmp_path / "seed"
    _g(tmp_path, "clone", "-q", str(origin), str(seed))
    for c in (seed,):
        _g(c, "config", "user.name", "t"); _g(c, "config", "user.email", "t@t")
    (seed / "output").mkdir()
    (seed / "output" / "bets_ledger.csv").write_text(_csv(rows), encoding="utf-8")
    _g(seed, "add", "-A"); _g(seed, "commit", "-q", "-m", "seed"); _g(seed, "push", "-q", "origin", "HEAD:main")
    clones = []
    for n in ("run_a", "run_b"):
        c = tmp_path / n
        _g(tmp_path, "clone", "-q", str(origin), str(c))
        _g(c, "config", "user.name", "t"); _g(c, "config", "user.email", "t@t")
        clones.append(c)
    return origin, clones


def _origin_rows(tmp_path, origin):
    chk = tmp_path / "check"
    if chk.exists():
        shutil.rmtree(chk)
    _g(tmp_path, "clone", "-q", str(origin), str(chk))
    return pd.read_csv(chk / "output" / "bets_ledger.csv", dtype=str, keep_default_na=False)


@needs_git
def test_rebase_x_theirs_drops_a_concurrent_row(tmp_path):
    """THE FINDING, reproduced: run A (predict) appends D and pushes first; run B
    (update_results) settled C - the row next to D - and publishes with the old
    `pull --rebase -X theirs`. D is gone from origin and every step reported success."""
    origin, (a, b) = _setup(tmp_path, [_row("A"), _row("C")])
    (a / "output" / "bets_ledger.csv").write_text(_csv([_row("A"), _row("C"), _row("D", date="2026-11-03")]), encoding="utf-8")
    _g(a, "commit", "-qam", "predict"); _g(a, "push", "-q", "origin", "HEAD:main")
    (b / "output" / "bets_ledger.csv").write_text(_csv([_row("A"), _row("C", "WIN", "0.9")]), encoding="utf-8")
    _g(b, "commit", "-qam", "results")
    _g(b, "pull", "-q", "--rebase", "-X", "theirs", "origin", "main"); _g(b, "push", "-q", "origin", "HEAD:main")
    assert "D" not in set(_origin_rows(tmp_path, origin)["home_team"])


@needs_git
def test_sync_push_keeps_the_concurrent_row(tmp_path):
    origin, (a, b) = _setup(tmp_path, [_row("A"), _row("C")])
    (a / "output" / "bets_ledger.csv").write_text(_csv([_row("A"), _row("C"), _row("D", date="2026-11-03")]), encoding="utf-8")
    _g(a, "commit", "-qam", "predict"); _g(a, "push", "-q", "origin", "HEAD:main")
    (b / "output" / "bets_ledger.csv").write_text(_csv([_row("A"), _row("C", "WIN", "0.9")]), encoding="utf-8")
    snap = tmp_path / "snap"
    assert L.sync_snapshot(snap, repo=b) == ["output/bets_ledger.csv"]
    # the job's own commit no longer carries the ledger
    assert _g(b, "diff", "--quiet", "HEAD", "--", "output/bets_ledger.csv", check=False).returncode == 0
    assert L.sync_push(snap, "results (ledgers)", repo=b, attempts=2) == 0
    got = _origin_rows(tmp_path, origin)
    assert list(got["home_team"]) == ["A", "C", "D"]
    assert got.loc[got["home_team"] == "C", "result"].item() == "WIN"


@needs_git
def test_sync_snapshot_ignores_unchanged_ledgers(tmp_path):
    origin, (a, _b) = _setup(tmp_path, [_row("A")])
    snap = tmp_path / "snap"
    assert L.sync_snapshot(snap, repo=a) == []
    assert L.sync_push(snap, "x", repo=a) == 0
