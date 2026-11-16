"""The verified git-restored af_history provider rows (owner 2026-11-16: "yes, make the models use them").

The loader reads them as a separate, labelled source, right after step 1's af_history merge (both exits), nan_only: a cell that holds a
value is never changed, a cell the verified file does not hold stays missing. The committed verified file
holds only rows whose FT score equals API-Football's, carries provenance + source commit per row, and
no excluded cell or row reaches it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data_loader as DL

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / DL.AF_RESTORED_ROWS
EXCLUDED = ROOT / "output" / "corrections" / "af_history_restored_provider_rows_2026-11_excluded.csv"
LG = "Brazil Serie A"


def _frame():
    return pd.DataFrame({
        "league": [LG, LG, LG], "season": ["2024"] * 3,
        "date": pd.to_datetime(["2024-04-13", "2024-04-14", "2024-04-20"]),
        "home_team": ["Palmeiras", "Flamengo", "Santos"], "away_team": ["Vitoria", "Sao Paulo", "Gremio"],
        "home_goals": [2, 1, 0], "away_goals": [0, 1, 0],
        "home_corners": [np.nan, 4.0, np.nan], "away_corners": [np.nan, 5.0, np.nan],
        "home_yellows": [np.nan, np.nan, 1.0], "away_yellows": [np.nan, np.nan, np.nan],
        "home_shots": [np.nan, 9.0, np.nan], "away_shots": [np.nan, 11.0, np.nan],
        "home_sot": [np.nan, 3.0, np.nan], "away_sot": [np.nan, 4.0, np.nan],
    })


def _verified(tmp_path):
    v = pd.DataFrame({
        "league": [LG, LG], "date": pd.to_datetime(["2024-04-13", "2024-04-14"]),
        "home_team": ["Palmeiras", "Flamengo"], "away_team": ["Vitoria", "Sao Paulo"], "season": ["2024"] * 2,
        "FTHG": [2.0, 1.0], "FTAG": [0.0, 1.0], "FTR": ["H", "D"],
        "HS": [14.0, 99.0], "AS": [7.0, 99.0], "HST": [5.0, 99.0], "AST": [2.0, 99.0],
        "HC": [6.0, 99.0], "AC": [np.nan, 99.0],            # AC excluded by verification -> stays missing
        "HY": [2.0, 3.0], "AY": [1.0, 2.0],
        "provenance": ["restored"] * 2, "restored_from_commit": ["a" * 40] * 2,
    })
    p = tmp_path / DL.AF_RESTORED_ROWS
    p.parent.mkdir(parents=True)
    v.to_parquet(p, index=False)
    return tmp_path


def test_fills_only_missing_cells_and_never_overwrites(tmp_path):
    before = _frame()
    after = DL._enrich_af_restored_rows(before.copy(), root=_verified(tmp_path))
    assert list(after.columns)[: len(before.columns)] == list(before.columns)
    for c in before.columns:
        had = before[c].notna()
        assert (after.loc[had, c] == before.loc[had, c]).all(), c         # nothing present changes
    assert after.at[0, "home_corners"] == 6.0 and after.at[0, "home_shots"] == 14.0
    assert pd.isna(after.at[0, "away_corners"])                            # excluded cell stays missing
    assert after.at[1, "home_corners"] == 4.0 and after.at[1, "home_shots"] == 9.0  # 99s never land
    assert after.at[1, "home_yellows"] == 3.0                              # a missing cell is filled
    pd.testing.assert_series_equal(after.loc[2, before.columns], before.loc[2], check_names=False)
    assert after.attrs["af_restored_fill"]["filled_cells"]["home_corners"] == 1


def test_absent_file_leaves_the_frame_unchanged(tmp_path):
    before = _frame()
    after = DL._enrich_af_restored_rows(before.copy(), root=tmp_path)
    pd.testing.assert_frame_equal(after, before)


def test_the_fill_runs_after_step_1_on_both_exits():
    import inspect
    src = inspect.getsource(DL._enrich_with_api_shots)
    i = src.index("_enrich_af_restored_rows(df)")
    assert src.index("Step 1: merge from af_history.parquet") < i < src.index("if not api_key:")


@pytest.mark.skipif(not VERIFIED.exists(), reason="no verified restored file")
def test_committed_verified_file_is_labelled_and_clean():
    v = pd.read_parquet(VERIFIED)
    assert len(v) > 0
    assert v["provenance"].notna().all() and v["restored_from_commit"].str.fullmatch(r"[0-9a-f]{40}").all()
    assert v["verified_by"].str.startswith(("API-Football FT", "football-data")).all()
    k = ["league", "date", "home_team", "away_team"]
    assert not v.duplicated(k).any()
    for c, cap in (("HC", 30), ("AC", 30), ("HY", 12), ("AY", 12), ("HF", 50), ("AF", 50)):
        x = v[c].dropna()
        assert ((x >= 0) & (x <= cap) & (x % 1 == 0)).all(), c
    ex = pd.read_csv(EXCLUDED)
    rows = ex[ex["column"] == "*row*"]
    vk = set(zip(v["league"], pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d"), v["home_team"], v["away_team"]))
    rk = set(zip(rows["league"], pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d"), rows["home_team"],
                 rows["away_team"]))
    assert not (vk & rk)                                                   # an excluded row never served
    cells = ex[ex["column"] != "*row*"]
    vi = v.assign(_d=pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")).set_index(
        ["league", "_d", "home_team", "away_team"])
    for r in cells.head(200).itertuples(index=False):
        key = (r.league, pd.Timestamp(r.date).strftime("%Y-%m-%d"), r.home_team, r.away_team)
        if key in vi.index:
            assert pd.isna(vi.loc[key, r.column]), (key, r.column)        # an excluded cell is missing
