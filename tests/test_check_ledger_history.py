"""scripts/check_ledger_history.py: finds rows that existed and later vanished, and classifies them."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_ledger_history as H  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
HDR = "source,generated_at,match_date,league,home_team,away_team,side,odds,result,pnl\n"


def _row(src, home, gen="2026-10-01 10:00", side="OVER", odds="1.9", res=""):
    return f"{src},{gen},2026-10-05,L,{home},{home} B,{side},{odds},{res},\n"


def _commit(repo, text, msg, date):
    (repo / "output" / "bets_ledger.csv").write_text(HDR + text, encoding="utf-8", newline="\n")
    env = dict(os.environ, GIT_COMMITTER_DATE=date, GIT_AUTHOR_DATE=date)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", msg], check=True, env=env)


def test_finds_the_lost_row_and_spares_the_purge_and_the_rename(tmp_path):
    repo = tmp_path / "r"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    (repo / "output").mkdir()
    base = _row("live", "A") + _row("live", "B", gen="2026-10-01 11:00") + _row("backtest", "Z")
    _commit(repo, base, "auto: predict", "2026-10-01T10:00:00")
    # the race: an update_results rebase drops B
    _commit(repo, _row("live", "A", res="WIN") + _row("backtest", "Z"), "auto: results", "2026-10-02T10:00:00")
    # a name repair re-keys A (same fingerprint, new home name)
    _commit(repo, _row("live", "A2", res="WIN") + _row("backtest", "Z"), "data: rename", "2026-10-03T10:00:00")
    # the owner's backtest purge
    _commit(repo, _row("live", "A2", res="WIN"), "data: remove backtest rows", "2026-11-01T10:00:00")
    blobs = H.Blobs(repo)
    try:
        r = H.check_ledger(repo, "HEAD", "output/bets_ledger.csv", max_commits=2, blobs=blobs)
    finally:
        blobs.close()
    assert r["commits"] == 4 and r["sampled"] == 2
    assert r["vanished"] == {"lost_in_bot_commit": 1, "rekeyed": 1, "intentional_backtest_purge": 1}
    assert r["examples"]["lost_in_bot_commit"][0]["key"][2] == "B"
    assert r["by_commit"][0]["subject"] == "auto: results"
