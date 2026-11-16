"""The af_history write-back must never lose a column or a row it does not own (2026-11-16).

The bug: src/data_loader's 'BANK THE ENRICHMENT' did concat + drop_duplicates(keep="last") with a
shots-only frame, so every touched provider row lost its corners, fouls, cards, full-time score and
fixture_id. These tests run the real write-back (data_loader.bank_af_enrichment) and the shared helper
(src/integrity/preserving_upsert) through a parquet round trip and require every pre-existing cell to
come back identical, value and dtype, except the intended updates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as DL
from src.integrity.preserving_upsert import union_columns, upsert

LG = "Brazil Serie A"


def _stored() -> pd.DataFrame:
    """A small af_history in the committed schema, plus PR #34's fixture_id (Int64)."""
    d = pd.DataFrame({
        "league": [LG, LG, LG, LG],
        "date": pd.to_datetime(["2024-04-13", "2024-04-14", "2024-04-14", "2024-04-20"]),
        "home_team": ["Palmeiras", "Flamengo", "Flamengo", "Santos"],
        "away_team": ["Vitoria", "Sao Paulo", "Sao Paulo", "Gremio"],
        "season": ["2024"] * 4,
        "FTHG": [2.0, 1.0, 1.0, np.nan], "FTAG": [0.0, 1.0, 1.0, np.nan],
        "FTR": ["H", "D", "D", None],
        "HS": [14.0, 9.0, 9.0, np.nan], "AS": [7.0, 11.0, 11.0, np.nan],
        "HST": [5.0, 3.0, 3.0, np.nan], "AST": [2.0, 4.0, 4.0, np.nan],
        "HC": [6.0, 4.0, 4.0, np.nan], "AC": [3.0, 5.0, 5.0, np.nan],
        "HF": [12.0, 15.0, 15.0, np.nan], "AF": [10.0, 9.0, 9.0, np.nan],
        "HY": [2.0, 3.0, 3.0, np.nan], "AY": [1.0, 2.0, 2.0, np.nan],
        "HR": [0.0, 0.0, 0.0, np.nan], "AR": [0.0, 1.0, 1.0, np.nan],
        "HXG": [np.nan, 1.2, 1.2, np.nan], "AXG": [np.nan, 0.9, 0.9, np.nan],
        "HIB": [np.nan, 6.0, 6.0, np.nan], "AIB": [np.nan, 7.0, 7.0, np.nan],
        "fixture_id": pd.array([1001, 1002, 1002, pd.NA], dtype="Int64"),
    })
    for c in ("league", "home_team", "away_team", "season", "FTR"):
        d[c] = d[c].astype("string")
    return d


def _loader_frame() -> pd.DataFrame:
    """What _enrich_with_api_shots hands the write-back: internal names, values from FD / the merge."""
    return pd.DataFrame({
        "league": [LG, LG, LG, LG],
        "season": ["2024", "2024", "2024", "2024"],
        "date": pd.to_datetime(["2024-04-13", "2024-04-14", "2024-04-20", "2024-05-01"]),
        "home_team": ["Palmeiras", "Flamengo", "Santos", "Bahia"],
        "away_team": ["Vitoria", "Sao Paulo", "Gremio", "Cuiaba"],
        # Palmeiras: a DIFFERENT (football-data) shot count must not overwrite the provider's 14;
        # its xG is new and fills the missing cell.
        "home_shots": [16.0, np.nan, 10.0, 12.0], "away_shots": [7.0, np.nan, 8.0, 6.0],
        "home_sot": [5.0, np.nan, 4.0, 5.0], "away_sot": [2.0, np.nan, 3.0, 2.0],
        "home_xg": [1.7, np.nan, np.nan, 1.1], "away_xg": [0.4, np.nan, np.nan, 0.8],
        # Flamengo: xG missing in the frame must NOT erase the stored 1.2 / 0.9.
        "home_insidebox": [np.nan, np.nan, np.nan, np.nan], "away_insidebox": [np.nan] * 4,
        "home_corners": [99.0, 99.0, 99.0, 99.0],          # not owned by the write-back: ignored
    })


def _assert_unchanged_except(before: pd.DataFrame, after: pd.DataFrame, changed: set[tuple[int, str]]):
    assert list(after.columns[: len(before.columns)]) == list(before.columns)
    for c in before.columns:
        assert after[c].dtype == before[c].dtype, c
        for i in range(len(before)):
            if (i, c) in changed:
                continue
            a, b = before.at[i, c], after.at[i, c]
            assert (pd.isna(a) and pd.isna(b)) or a == b, (i, c, a, b)


def test_bank_round_trip_keeps_corners_cards_scores_and_fixture_id(tmp_path):
    p = tmp_path / "af_history.parquet"
    before = _stored()
    before.to_parquet(p, index=False)
    before = pd.read_parquet(p)                     # compare against exactly what is on disk

    out = DL.bank_af_enrichment(_loader_frame(), hist_path=p)
    after = pd.read_parquet(p)

    assert out["written"] is True
    assert len(after) == len(before) + 1            # nothing dropped (not even the stored duplicate)
    intended = {(0, "HXG"), (0, "AXG"), (3, "HS"), (3, "AS"), (3, "HST"), (3, "AST")}
    _assert_unchanged_except(before, after.iloc[: len(before)].reset_index(drop=True), intended)
    assert after.at[0, "HXG"] == 1.7 and after.at[0, "AXG"] == 0.4
    assert after.at[0, "HS"] == 14.0                # the provider value is never overwritten
    assert after.at[3, "HS"] == 10.0                # a missing cell is filled
    for c in ("HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "FTHG", "FTAG", "fixture_id"):
        assert after.loc[:2, c].notna().all(), c    # corners / cards / score / fixture_id survive
    new = after.iloc[-1]
    assert new["home_team"] == "Bahia" and new["HS"] == 12.0
    for c in ("HC", "AC", "HY", "AY", "FTHG", "fixture_id"):
        assert pd.isna(new[c]), c                   # missing stays missing, never a default
    assert after["fixture_id"].dtype == "Int64"


def test_bank_second_run_is_a_byte_identical_no_op(tmp_path):
    p = tmp_path / "af_history.parquet"
    _stored().to_parquet(p, index=False)
    DL.bank_af_enrichment(_loader_frame(), hist_path=p)
    once = p.read_bytes()
    out = DL.bank_af_enrichment(_loader_frame(), hist_path=p)
    assert out["written"] is False
    assert p.read_bytes() == once


def test_bank_keeps_stored_misfiled_rows_and_refuses_new_ones(tmp_path):
    p = tmp_path / "af_history.parquet"
    st = _stored()
    st.loc[3, "league"] = "Denmark 1st Div"         # not in the registry: misfiled
    st.to_parquet(p, index=False)
    fr = _loader_frame()
    fr.loc[3, "league"] = "Denmark 1st Div"
    DL.bank_af_enrichment(fr, hist_path=p)
    after = pd.read_parquet(p)
    assert (after["league"] == "Denmark 1st Div").sum() == 1   # the stored row stays, the new is refused
    assert "Bahia" not in set(after["home_team"])


def test_bank_never_drops_a_column_the_frame_lacks(tmp_path):
    p = tmp_path / "af_history.parquet"
    st = _stored()
    st["home_team_id"] = [1.0, 2.0, 2.0, np.nan]
    st.to_parquet(p, index=False)
    DL.bank_af_enrichment(_loader_frame(), hist_path=p)
    after = pd.read_parquet(p)
    assert "home_team_id" in after.columns
    assert after.loc[:2, "home_team_id"].tolist() == [1.0, 2.0, 2.0]


# ── the shared helper ────────────────────────────────────────────────────────────────────────────

def test_upsert_modes():
    st = pd.DataFrame({"k": [1, 2], "a": [1.0, np.nan], "b": ["x", "y"], "c": [5.0, 6.0]})
    nw = pd.DataFrame({"k": [1.0, 2.0, 3.0], "a": [9.0, 2.0, 3.0], "c": [np.nan, 7.0, 8.0]})
    fill, r = upsert(st, nw, key=["k"], mode="fill")
    assert fill["a"].tolist()[:2] == [1.0, 2.0] and fill["c"].tolist()[:2] == [5.0, 6.0]
    assert fill["b"].tolist() == ["x", "y", None] or pd.isna(fill.at[2, "b"])
    upd, _ = upsert(st, nw, key=["k"], mode="update")
    assert upd["a"].tolist()[:2] == [9.0, 2.0] and upd["c"].tolist()[:2] == [5.0, 7.0]
    rep, _ = upsert(st, nw, key=["k"], mode="replace")
    assert pd.isna(rep.at[0, "c"]) and rep.at[1, "c"] == 7.0
    keep, _ = upsert(st, nw, key=["k"], mode="keep")
    assert keep["a"].tolist()[:2] == [1.0, np.nan] or pd.isna(keep.at[1, "a"])
    for out in (fill, upd, rep, keep):
        assert list(out.columns) == ["k", "a", "b", "c"] and len(out) == 3
        assert out["b"].tolist()[:2] == ["x", "y"]


def test_upsert_untouched_parquet_round_trip_is_identical(tmp_path):
    p = tmp_path / "s.parquet"
    _stored().to_parquet(p, index=False)
    before = pd.read_parquet(p)
    nothing = pd.DataFrame({"league": [LG], "date": ["2024-04-13"], "home_team": ["Palmeiras"],
                            "away_team": ["Vitoria"], "HS": [np.nan]})
    out, r = upsert(before, nothing, key=["league", "date", "home_team", "away_team"],
                    mode="update", date_keys=("date",))
    assert not r.changed
    pd.testing.assert_frame_equal(out, before)


def test_union_columns_never_drops():
    old = pd.DataFrame({"a": [1], "z": [2]})
    out = union_columns(old, ["a", "b"])
    assert list(out.columns) == ["a", "b", "z"] and out.at[0, "z"] == 2


@pytest.mark.parametrize("mode", ["fill", "update", "replace"])
def test_upsert_never_removes_stored_duplicates(mode):
    st = pd.DataFrame({"k": [1, 1, 2], "v": [1.0, 1.0, 2.0], "extra": ["a", "b", "c"]})
    out, _ = upsert(st, pd.DataFrame({"k": [1], "v": [5.0]}), key=["k"], mode=mode)
    assert len(out) == 3 and out["extra"].tolist() == ["a", "b", "c"]


# ── the restored rows, beside the original ───────────────────────────────────────────────────────

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
_RESTORED = _ROOT / "output" / "corrections" / "af_history_restored_provider_rows_2026-11.parquet"


@pytest.mark.skipif(not _RESTORED.exists(), reason="no restored sidecar")
def test_restored_rows_are_provider_rows_of_current_written_back_keys():
    """Every restored row is a provider row (FT score present) and its key is in the current
    af_history as a written-back row (no score) - restored beside, the original file not rewritten."""
    r = pd.read_parquet(_RESTORED)
    assert len(r) > 0 and r["FTHG"].notna().all() and r["FTAG"].notna().all()
    assert r["restored_from_commit"].str.fullmatch(r"[0-9a-f]{40}").all()
    k = ["league", "date", "home_team", "away_team"]
    assert not r.duplicated(k).any()
    af = _ROOT / "output" / "af_history.parquet"
    if af.exists():
        h = pd.read_parquet(af)
        h["date"] = pd.to_datetime(h["date"]).dt.normalize()
        m = r[k].assign(date=pd.to_datetime(r["date"]).dt.normalize()).merge(
            h[k + ["FTHG"]], on=k, how="left", indicator=True)
        assert (m["_merge"] == "both").all()
        assert m["FTHG"].isna().all()
