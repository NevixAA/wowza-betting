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

pytestmark = pytest.mark.skipif(shutil.which(H.GIT) is None and not Path(H.GIT).exists(), reason="git not found")
HDR = "source,generated_at,match_date,league,home_team,away_team,side,odds,result,pnl\n"


def _row(src, home, gen="2026-10-01 10:00", side="OVER", odds="1.9", res=""):
    return f"{src},{gen},2026-10-05,L,{home},{home} B,{side},{odds},{res},\n"


def _commit(repo, text, msg, date):
    (repo / "output" / "bets_ledger.csv").write_text(HDR + text, encoding="utf-8", newline="\n")
    env = dict(os.environ, GIT_COMMITTER_DATE=date, GIT_AUTHOR_DATE=date)
    subprocess.run([H.GIT, "-C", str(repo), "add", "-A"], check=True)
    subprocess.run([H.GIT, "-C", str(repo), "commit", "-q", "-m", msg], check=True, env=env)


def test_finds_the_lost_row_and_spares_the_purge_and_the_rename(tmp_path):
    repo = tmp_path / "r"
    subprocess.run([H.GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run([H.GIT, "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run([H.GIT, "-C", str(repo), "config", "user.email", "t@t"], check=True)
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


PL_HDR = "signal_date,match_date,home_team,away_team,market,player_name,tier,model_prob,result,pnl,player_id\n"


def _pl(name, sig="2026-10-01", prob="0.40", res=""):
    return f"{sig},2026-10-09,H,A,goals,{name},AVOID,{prob},{res},,\n"


def _pcommit(repo, text, msg, date):
    (repo / "output" / "player_ledger.csv").write_text(PL_HDR + text, encoding="utf-8", newline="\n")
    env = dict(os.environ, GIT_COMMITTER_DATE=date, GIT_AUTHOR_DATE=date)
    subprocess.run([H.GIT, "-C", str(repo), "add", "-A"], check=True)
    subprocess.run([H.GIT, "-C", str(repo), "commit", "-q", "-m", msg], check=True, env=env)


def test_restore_writes_lost_rows_beside_the_ledger_with_first_sight_values(tmp_path):
    repo = tmp_path / "r"
    subprocess.run([H.GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run([H.GIT, "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run([H.GIT, "-C", str(repo), "config", "user.email", "t@t"], check=True)
    (repo / "output").mkdir()
    _pcommit(repo, _pl("Keep") + _pl("Lost", prob="0.31") + _pl("Back", prob="0.22"),
             "auto: player props", "2026-10-01T10:00:00")
    _pcommit(repo, _pl("Keep", res="WIN"), "auto: player props", "2026-10-02T10:00:00")
    # 'Back' is logged again later with a different signal_date / model_prob
    _pcommit(repo, _pl("Keep", res="WIN") + _pl("Back", sig="2026-10-05", prob="0.27"),
             "auto: player props", "2026-10-05T10:00:00")
    led = (repo / "output" / "player_ledger.csv").read_bytes()
    out = repo / "output" / "corrections"
    res = H.restore(repo, "HEAD", "output/player_ledger.csv", 50,
                    out / "r.csv", out / "x.csv", out / "r.md")
    assert res["lost"] == 1 and res["relogged"] == 1 and res["relogged_changed"] == 1
    assert (repo / "output" / "player_ledger.csv").read_bytes() == led       # ledger untouched
    import csv as _csv
    lost = list(_csv.DictReader((out / "r.csv").open(encoding="utf-8")))
    assert lost[0]["player_name"] == "Lost" and lost[0]["model_prob"] == "0.31"
    assert lost[0]["removed_subject"] == "auto: player props" and lost[0]["first_seen_commit"]
    back = list(_csv.DictReader((out / "x.csv").open(encoding="utf-8")))
    assert back[0]["model_prob"] == "0.22" and back[0]["changed_columns"] == "signal_date|model_prob"
    assert "Removing commits" in (out / "r.md").read_text(encoding="utf-8")
