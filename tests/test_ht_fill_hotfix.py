"""HT hotfix on main (2026-11-03, owner approved): the half-time sidecar
output/collect/fd_history_ht_fill.parquet (real API-Football score.halftime, collected on
data-truth-100) fills MISSING ht_home_goals / ht_away_goals. Proven club identity only; a stored
HT score is never overwritten; a null HT stays missing (never 0).

    python -m pytest -q tests/test_ht_fill_hotfix.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loader as DL

LG = "Brazil Serie A"
OLD, NEW = "Atletico Paranaense", "Athletico-PR"


def _r(d, h, a, hg, ag, hth=np.nan, hta=np.nan, lg=LG):
    return {"date": pd.Timestamp(d), "league": lg, "season": None, "home_team": h, "away_team": a,
            "home_goals": float(hg), "away_goals": float(ag), "ht_home_goals": hth,
            "ht_away_goals": hta}


def _write_fill(root, rows):
    p = root / "output" / "collect" / "fd_history_ht_fill.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "HTHG", "HTAG"]) \
        .to_parquet(p, index=False)


def test_fill_fills_missing_and_rebuilds_labels(tmp_path):
    df = pd.DataFrame([_r("2024-07-07", "Fortaleza", "Santos", 2, 2),
                       _r("2024-07-08", "Bahia", "Gremio", 1, 0, 1.0, 0.0)])
    _write_fill(tmp_path, [(LG, "2024-07-07", "Fortaleza", "Santos", 1, 1)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    row = out[out["home_team"] == "Fortaleza"].iloc[0]
    assert (row["ht_home_goals"], row["ht_away_goals"]) == (1.0, 1.0)
    assert row["ht_total_goals"] == 2 and row["ht_over15"] == 1.0


def test_stored_ht_is_never_overwritten(tmp_path):
    df = pd.DataFrame([_r("2024-07-08", "Bahia", "Gremio", 1, 0, 1.0, 0.0)])
    _write_fill(tmp_path, [(LG, "2024-07-08", "Bahia", "Gremio", 0, 0)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    assert out["ht_home_goals"].iloc[0] == 1.0


def test_null_ht_in_fill_stays_missing(tmp_path):
    df = pd.DataFrame([_r("2024-07-07", "Fortaleza", "Santos", 2, 2),
                       _r("2024-07-08", "Bahia", "Gremio", 1, 0, 1.0, 0.0)])
    _write_fill(tmp_path, [(LG, "2024-07-07", "Fortaleza", "Santos", None, None)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    assert out.loc[out["home_team"] == "Fortaleza", "ht_home_goals"].isna().all()
    assert out.loc[out["home_team"] == "Fortaleza", "ht_over05"].isna().all()


def test_other_league_row_is_not_placed(tmp_path):
    df = pd.DataFrame([_r("2024-07-07", "Fortaleza", "Santos", 2, 2)])
    _write_fill(tmp_path, [("Argentina Primera", "2024-07-07", "Fortaleza", "Santos", 1, 1)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    assert out["ht_home_goals"].isna().all()


def test_renamed_club_is_placed_through_the_applied_identity_map(tmp_path, monkeypatch):
    df = pd.DataFrame([_r("2024-07-07", "Fortaleza", NEW, 2, 2)])
    monkeypatch.setattr(DL, "IDENTITY_ALIASES", {LG: {OLD: NEW}})
    _write_fill(tmp_path, [(LG, "2024-07-07", "Fortaleza", OLD, 1, 0)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    assert (out["ht_home_goals"].iloc[0], out["ht_away_goals"].iloc[0]) == (1.0, 0.0)


def test_ambiguous_alias_stays_unplaced(tmp_path, monkeypatch):
    df = pd.DataFrame([_r("2024-07-07", "Fortaleza", NEW, 2, 2),
                       _r("2024-07-14", "Santos", "Other Club", 0, 0)])
    monkeypatch.setattr(DL, "IDENTITY_ALIASES", {LG: {OLD: NEW, "Atletico Paranaense ": "Other Club"}})
    _write_fill(tmp_path, [(LG, "2024-07-07", "Fortaleza", OLD, 1, 0)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    assert out["ht_home_goals"].isna().all()
