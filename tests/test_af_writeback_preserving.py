"""No writer of af_history or of a collect file may lose a column or a row it does not own (2026-11-16).

The bug: main's src/data_loader 'BANK THE ENRICHMENT' did concat + drop_duplicates(keep="last") with a
shots-only frame, so every touched provider row lost its corners, fouls, cards, full-time score and
fixture_id. On this branch the loader no longer writes af_history at all (fix round 9,
tests/test_fix9_bank.py); the writers that remain - backfill_af_history's merge, the results sync from
main, collect_common.append_parquet and the sidecar collectors - merge through
src/integrity/preserving_upsert. These tests run them through a parquet round trip and require every
pre-existing cell to come back identical, value and dtype, except the intended updates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
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


def test_append_parquet_keeps_columns_and_rows_it_does_not_own(tmp_path):
    import collect_common as CC
    p = tmp_path / "c.parquet"
    pd.DataFrame({"fixture_id": [1, 2], "v": [1.0, 2.0], "note": ["kept", "kept2"]}).to_parquet(p, index=False)
    n = CC.append_parquet(p, [{"fixture_id": 2, "v": 5.0}, {"fixture_id": 3, "v": 3.0}],
                          key=["fixture_id"], columns=["fixture_id", "v"])
    out = pd.read_parquet(p)
    assert n == 3 and list(out.columns) == ["fixture_id", "v", "note"]
    assert out["v"].tolist() == [1.0, 5.0, 3.0]                  # keep="last": the new answer wins
    assert out["note"].tolist()[:2] == ["kept", "kept2"] and pd.isna(out.at[2, "note"])
    CC.append_parquet(p, [{"fixture_id": 1, "v": 9.0}], key=["fixture_id"], columns=["fixture_id", "v"],
                      keep="first")
    assert pd.read_parquet(p)["v"].tolist() == [1.0, 5.0, 3.0]   # keep="first": the stored row wins


def test_backfill_merge_keeps_corners_cards_and_fixture_id(tmp_path):
    """backfill_af_history's merge, as it runs (mode "update"): a provider answer for a stored key
    fills / updates only cells it gives and never drops a stored column (here one it does not emit)."""
    st = _stored()
    st["home_team_id"] = [1.0, 2.0, 2.0, np.nan]
    p = tmp_path / "af_history.parquet"
    st.to_parquet(p, index=False)
    before = pd.read_parquet(p)
    fresh = pd.DataFrame({"league": [LG, LG], "season": ["2024", "2024"],
                          "date": pd.to_datetime(["2024-04-20", "2024-05-01"]),
                          "home_team": ["Santos", "Bahia"], "away_team": ["Gremio", "Cuiaba"],
                          "FTHG": [3, 0], "FTAG": [1, 0], "FTR": ["H", "D"], "HS": [11.0, 8.0],
                          "HC": [7.0, np.nan], "fixture_id": pd.array([2004, 2005], dtype="Int64")})
    out, rep = upsert(before, fresh, key=["league", "date", "home_team", "away_team"], mode="update",
                      date_keys=("date",))
    intended = {(3, "FTHG"), (3, "FTAG"), (3, "FTR"), (3, "HS"), (3, "HC"), (3, "fixture_id")}
    _assert_unchanged_except(before, out.iloc[: len(before)].reset_index(drop=True), intended)
    assert out.at[3, "HC"] == 7.0 and out.at[3, "fixture_id"] == 2004 and out.at[3, "AC"] != out.at[3, "AC"]
    assert len(out) == 5 and out.at[4, "home_team"] == "Bahia" and pd.isna(out.at[4, "HC"])
    assert out["fixture_id"].dtype == "Int64" and "home_team_id" in out.columns


def test_sync_from_main_keeps_every_column_main_rows_carry(tmp_path, monkeypatch):
    """A column only main's file has (fixture_id, PR #34) reaches the synced rows; this branch's rows
    are untouched and the column is missing for them, never a default."""
    import subprocess

    import sync_results_from_main as SY

    def g(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True,
                       capture_output=True)
    repo = tmp_path / "r"
    (repo / "output").mkdir(parents=True)
    g("init", "-q", "-b", "main")
    af = repo / "output" / "af_history.parquet"
    base = pd.DataFrame({"league": ["L"], "date": pd.to_datetime(["2026-10-01"]), "home_team": ["A"],
                         "away_team": ["B"], "HC": [4.0]})
    base.to_parquet(af, index=False)
    g("add", "-A"); g("commit", "-q", "-m", "base")
    g("checkout", "-q", "-b", "dt")
    pd.concat([base, pd.DataFrame({"league": ["L"], "date": pd.to_datetime(["2026-10-03"]),
                                   "home_team": ["G"], "away_team": ["H"], "HC": [2.0]})],
              ignore_index=True).to_parquet(af, index=False)
    g("commit", "-qam", "dt gap row")
    g("checkout", "-q", "main")
    pd.concat([base.assign(fixture_id=pd.array([pd.NA], dtype="Int64")),
               pd.DataFrame({"league": ["L"], "date": pd.to_datetime(["2026-11-05"]), "home_team": ["E"],
                             "away_team": ["F"], "HC": [6.0], "fixture_id": pd.array([77], dtype="Int64")})],
              ignore_index=True).to_parquet(af, index=False)
    g("commit", "-qam", "main extend with fixture_id")
    g("checkout", "-q", "dt")
    monkeypatch.setattr(SY, "ROOT", repo)
    monkeypatch.setattr(SY, "MAIN", "main")
    monkeypatch.setattr(SY, "WHOLE_FILE", [])
    monkeypatch.setattr(SY, "WHOLE_DIRS", [])
    monkeypatch.setattr(SY, "SIDECAR", repo / "output" / "corrections" / "synced.csv")
    r = SY.run()
    out = pd.read_parquet(af)
    assert r["rows_appended"]["output/af_history.parquet"] == 1
    assert "fixture_id" in out.columns
    assert out.set_index("home_team").at["E", "fixture_id"] == 77
    assert out.set_index("home_team").loc[["A", "G"], "fixture_id"].isna().all()
    assert out.set_index("home_team").loc[["A", "G", "E"], "HC"].tolist() == [4.0, 2.0, 6.0]


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
