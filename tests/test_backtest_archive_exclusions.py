"""D1 (owner 2026-11-23 "Yes, do all"): archive rows no provider confirms are excluded from the loader
read, listed beside the untouched archive in output/corrections/backtest_archive_excluded_rows.csv."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_loader as DL  # noqa: E402

LIST = ROOT / DL.ARCHIVE_EXCLUDED


def _arch():
    return pd.DataFrame({"date": pd.to_datetime(["2022-10-23", "2022-10-23", "2022-11-12"]),
                         "league": ["Belgian First Division A"] * 3,
                         "home_team": ["Standard", "Gent", "Charleroi"],
                         "away_team": ["Anderlecht", "Seraing", "Mechelen"],
                         "home_goals": [5, 2, 0], "away_goals": [0, 1, 5]})


def test_listed_rows_leave_the_read_and_nothing_else():
    out = DL._drop_excluded_archive_rows(_arch())
    assert out["home_team"].tolist() == ["Gent"]


def test_the_list_records_originals_and_evidence():
    ex = pd.read_csv(LIST, dtype=str)
    assert len(ex) == 2
    assert set(ex["home_team"]) == {"Standard", "Charleroi"}
    assert (ex["fd_raw_ft_check"] == "provider_not_played").all()
    for c in ("archive_home_goals", "archive_away_goals", "archive_row_index", "reason", "decided_by"):
        assert ex[c].notna().all()


def test_listed_rows_still_exist_in_the_untouched_archive():
    arch = ROOT / "output" / "backtest_all_leagues.csv"
    if not arch.exists():
        return
    raw = pd.read_csv(arch, usecols=["date", "league", "home_team", "away_team", "home_goals", "away_goals"],
                      low_memory=False)
    ex = pd.read_csv(LIST, dtype=str)
    for _, r in ex.iterrows():
        hit = raw[(raw["league"] == r["league"]) & (raw["date"].astype(str).str[:10] == r["date"])
                  & (raw["home_team"] == r["home_team"]) & (raw["away_team"] == r["away_team"])]
        assert len(hit) == 1
        assert int(hit["home_goals"].iloc[0]) == int(r["archive_home_goals"])
        assert int(hit.index[0]) == int(r["archive_row_index"])


def test_a_listed_row_not_found_is_logged_not_widened(tmp_path, caplog):
    p = tmp_path / "x.csv"
    p.write_text("league,date,home_team,away_team\nBelgian First Division A,2022-10-24,Nobody,Else\n", encoding="utf-8")
    import logging
    with caplog.at_level(logging.WARNING):
        out = DL._drop_excluded_archive_rows(_arch(), excluded=p)
    assert len(out) == 3
    assert "not found" in caplog.text
