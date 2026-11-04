"""R2-C finding 12: fixture-level enrichment joins can never expand, shift or guess.

    python -m pytest tests/test_integrity_joins.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.integrity import joins as IJ  # noqa: E402


def _left():
    # NON-range index on purpose: _canonicalise_clubs leaves one, and the legacy joins broke on it.
    return pd.DataFrame({"league": ["A", "A", "B", "A"], "h": ["x", "y", "x", "z"],
                         "a": ["p", "q", "p", "r"], "d": ["2026-01-01"] * 4,
                         "shots": [np.nan, np.nan, np.nan, 7.0]}, index=[40, 7, 99, 3])


def _right():
    return pd.DataFrame({
        "league": ["A", "A", "A", "B", "A", "A"],
        "h": ["x", "x", "y", "x", "y", "z"], "a": ["p", "p", "q", "p", "q", "r"],
        "d": ["2026-01-01"] * 6,
        "shots": [10.0, 10.0, 5.0, 3.0, 6.0, 99.0],       # (A,x,p) agrees; (A,y,q) conflicts
        "xg": [1.1, np.nan, 0.5, 0.3, 0.5, 2.0]})


def test_no_expansion_and_rows_stay_on_their_fixture():
    IJ.reset()
    out, rep = IJ.safe_left_enrich(_left(), _right(), keys=["league", "h", "a", "d"],
                                   value_cols=["shots", "xg"], name="t")
    assert len(out) == 4 and list(out.index) == [40, 7, 99, 3]
    assert out.loc[40, "shots"] == 10.0 and out.loc[40, "xg"] == 1.1     # duplicates that agree
    assert out.loc[99, "shots"] == 3.0                                   # league B's own match
    assert np.isnan(out.loc[7, "shots"]) and out.loc[7, "xg"] == 0.5     # conflicting cell left NaN
    assert out.loc[3, "shots"] == 7.0                                    # NaN-only: never overwritten
    assert rep.output_rows == rep.left_rows == 4
    assert rep.ambiguous_key_groups == 1 and rep.ambiguous_cells == {"shots": 1}
    assert rep.status == "WARN" and "t" in IJ.reports()


def test_join_without_league_or_fixture_id_is_refused():
    with pytest.raises(ValueError):
        IJ.safe_left_enrich(_left(), _right(), keys=["h", "a", "d"], value_cols=["shots"], name="bad")


def test_fixture_id_key_is_accepted():
    l = pd.DataFrame({"fixture_id": [1, 2], "v": [np.nan, np.nan]})
    r = pd.DataFrame({"fixture_id": [2, 1], "v": [20.0, 10.0]})
    out, rep = IJ.safe_left_enrich(l, r, keys=["fixture_id"], value_cols=["v"], name="fid")
    assert list(out["v"]) == [10.0, 20.0] and rep.status == "PASS"


def test_legacy_positional_zip_shifts_and_the_audit_sees_it():
    """Reproduces the af_history defect: merge on a league-free key with a duplicated right key,
    then copy back positionally — every row after the duplicate gets its neighbour's value."""
    left = pd.DataFrame({"h": ["x", "y", "z"], "a": ["p", "q", "r"], "d": ["D"] * 3,
                         "shots": [np.nan] * 3})
    right = pd.DataFrame({"h": ["x", "x", "y", "z"], "a": ["p", "p", "q", "r"], "d": ["D"] * 4,
                          "shots": [1.0, 1.0, 2.0, 3.0]})
    merged = left.merge(right.rename(columns={"shots": "s_h"}), on=["h", "a", "d"], how="left")
    legacy = [h if pd.isna(o) else o for o, h in zip(left["shots"], merged["s_h"])]
    assert legacy == [1.0, 1.0, 2.0]                  # y got x's value, z got y's: the shift
    rep = IJ.audit_legacy_merge(left, right, ["h", "a", "d"], name="legacy", register=False)
    assert rep.legacy_expansion_rows == 1 and rep.legacy_misaligned_rows == 2
    assert rep.status == "FAIL" and "key omits league/fixture_id" in rep.notes


def test_league_free_lookup_audit_reports_cross_league_values():
    left = pd.DataFrame({"league": ["Denmark 1st Div"], "_h": ["x"], "_a": ["y"], "_d": ["D"]})
    right = pd.DataFrame({"league": ["Bundesliga 2"], "_h": ["x"], "_a": ["y"], "_d": ["D"]})
    rep = IJ.audit_league_free_lookup(left, right, ["_h", "_a", "_d"], name="lf", register=False)
    assert rep.ambiguous_key_groups == 1 and rep.status == "WARN"


def test_write_join_health(tmp_path):
    IJ.reset()
    IJ.safe_left_enrich(_left(), _right(), keys=["league", "h", "a", "d"], value_cols=["shots"], name="w")
    doc = IJ.write_join_health(tmp_path / "j.json", {"x": 1})
    assert doc["status"] == "WARN" and (tmp_path / "j.json").exists() and doc["x"] == 1


def _real_rows():
    """PROVIDER rows only (fix round 9, owner 2026-11-04): a row the loader once wrote back (no
    fixture_id, no FT score) is no longer served, so it is no truth for the join either."""
    from src.af_history_provenance import written_back_mask
    h = pd.read_parquet(ROOT / "output" / "af_history.parquet")
    h = h[~written_back_mask(h)]
    h = h[h["league"].isin(["Brazil Serie A", "USA MLS", "Argentina Primera Division"])]
    return h.sort_values("date").tail(400)


@pytest.mark.skipif(not (ROOT / "output" / "af_history.parquet").exists(), reason="no af_history")
def test_data_loader_safe_join_matches_af_history_truth(monkeypatch):
    """Contract on the committed bank: with WOWZA_SAFE_ENRICH_JOINS=1 every filled shot value is the
    value af_history holds for THAT league/date/fixture; the frame keeps its rows and index."""
    import config
    from src import data_loader as DL
    monkeypatch.setenv("APIFOOTBALL_KEY", "")
    monkeypatch.setattr(config, "API_KEY", "", raising=False)
    monkeypatch.setenv("WOWZA_SAFE_ENRICH_JOINS", "1")
    truth = _real_rows()
    truth = truth.drop_duplicates(["league", "date", "home_team", "away_team"])
    left = truth[["league", "date", "home_team", "away_team"]].copy()
    left["date"] = pd.to_datetime(left["date"])
    for c in ("home_shots", "away_shots", "home_sot", "away_sot"):
        left[c] = np.nan
    left.index = np.arange(len(left)) * 3 + 11                       # non-range index
    IJ.reset()
    out = DL._enrich_with_api_shots(left.copy())
    assert len(out) == len(left) and list(out.index) == list(left.index)
    rep = IJ.reports()["af_history_shots"]
    assert rep["output_rows"] == rep["left_rows"]
    m = out.merge(truth.assign(date=pd.to_datetime(truth["date"]))[
        ["league", "date", "home_team", "away_team", "HS"]], on=["league", "date", "home_team", "away_team"])
    got = m.dropna(subset=["home_shots"])
    assert len(got) > 50
    assert (got["home_shots"] == got["HS"]).all()


def test_data_loader_legacy_join_is_audited_as_fail(monkeypatch, tmp_path):
    """The legacy (league-less) join shape is still audited. Fix round 9: the committed bank's
    duplicate keys were written-back rows, which are no longer joined, so the expansion is shown on
    a bank holding one key in two leagues (provider rows)."""
    import config
    from src import data_loader as DL
    monkeypatch.setenv("APIFOOTBALL_KEY", "")
    monkeypatch.setattr(config, "API_KEY", "", raising=False)
    monkeypatch.setenv("WOWZA_SAFE_ENRICH_JOINS", "0")      # the legacy path, now rollback-only
    bank = pd.DataFrame([{"league": lg, "date": pd.Timestamp("2025-03-01"), "home_team": "Boca Juniors",
                          "away_team": "River Plate", "FTHG": 1.0, "FTAG": 1.0, "fixture_id": fid, "HS": hs}
                         for lg, fid, hs in (("Argentina Primera Division", 1, 10.0), ("USA MLS", 2, 7.0))])
    p = tmp_path / "af_history.parquet"
    bank.to_parquet(p)
    monkeypatch.setattr(DL, "AF_HISTORY_PATH", p)
    left = pd.DataFrame([{"league": lg, "date": pd.Timestamp("2025-03-01"), "home_team": "Boca Juniors",
                          "away_team": "River Plate"} for lg in ("Argentina Primera Division", "USA MLS")])
    IJ.reset()
    out = DL._enrich_with_api_shots(left)
    rep = IJ.reports()["af_history_shots.legacy"]
    assert rep["legacy_expansion_rows"] > 0 and rep["status"] == "FAIL"
    assert out["home_shots"].tolist() == [10.0, 7.0]            # the league-keyed join is right
