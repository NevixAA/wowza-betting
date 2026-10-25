"""player_history.parquet month-part layout (2026-10-25).

The directory holds one part per calendar month, and _save_history rewrites only parts whose
rows changed, so git stores a new blob only for months that really moved.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_model import pipeline as P


def _frame(n_per_month=40, months=("2025-05", "2025-08", "2025-09", "2026-09", "2026-10"), seed=0):
    rng = np.random.default_rng(seed)
    rows, fid = [], 1000
    for m in months:
        for i in range(n_per_month):
            fid += 1
            rows.append({
                "fixture_id": fid, "player_id": 900_000_000 + i, "date": pd.Timestamp(f"{m}-{1 + i % 27:02d}"),
                "league": ["Championship", "J-League", "La Liga 2"][i % 3], "team": f"T{i % 7}",
                "player_name": None if i % 11 == 0 else f"P{i}",
                "goals": int(rng.integers(0, 3)), "shots_on_target": float(rng.integers(0, 5)),
                "opp_strength_index": float(rng.random()),
                "days_since_last_injury": np.nan if i % 5 == 0 else float(rng.integers(1, 365)),
            })
    return pd.DataFrame(rows)


def _cast(df):
    """The precision _save_history writes (float32 / int32, ID columns untouched)."""
    return df.astype({c: ("float32" if df[c].dtype == np.float64 else "int32")
                      for c in df.columns
                      if c not in P._ID_COLS and df[c].dtype in (np.float64, np.int64)})


def _digest(root: Path) -> dict:
    return {f.name: (f.stat().st_mtime_ns, hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(root.glob("*.parquet"))}


def _sorted(df):
    return df.sort_values(["fixture_id", "player_id"], kind="stable").reset_index(drop=True)


def test_one_part_per_month_and_directory_read_is_identical(tmp_path):
    root = tmp_path / "player_history.parquet"
    df = _frame()
    st = P._save_history(df, root)
    names = sorted(f.name for f in root.glob("*.parquet"))
    assert names == ["month_2025-05.parquet", "month_2025-08.parquet", "month_2025-09.parquet",
                     "month_2026-09.parquet", "month_2026-10.parquet"]
    assert sorted(st["written"]) == names and not st["unchanged"]
    back = pd.read_parquet(root)
    expect = _cast(df)
    assert list(back.columns) == list(expect.columns)
    assert (back.dtypes == expect.dtypes).all()
    pd.testing.assert_frame_equal(_sorted(back), _sorted(expect), check_exact=True)
    # a column projection, as the workflows and notifier use it, also reads every part
    assert len(pd.read_parquet(root, columns=["date"])) == len(df)


def test_resave_identical_frame_rewrites_nothing(tmp_path):
    root = tmp_path / "player_history.parquet"
    P._save_history(_frame(), root)
    before = _digest(root)
    st = P._save_history(pd.read_parquet(root), root)
    assert st["written"] == [] and st["removed"] == [] and len(st["unchanged"]) == 5
    assert _digest(root) == before          # byte-identical AND untouched (mtime unchanged)


def test_daily_append_rewrites_only_todays_month(tmp_path):
    root = tmp_path / "player_history.parquet"
    df = _frame()
    P._save_history(df, root)
    before = _digest(root)
    new = _frame(n_per_month=3, months=("2026-10",), seed=9)
    new["fixture_id"] += 10_000                       # fresh fixtures played today
    P._save_history(pd.concat([pd.read_parquet(root), new], ignore_index=True), root)
    after = _digest(root)
    changed = {n for n in after if before.get(n) != after[n]}
    assert changed == {"month_2026-10.parquet"}
    assert len(pd.read_parquet(root)) == len(df) + 3


def test_refresh_of_a_past_row_rewrites_only_its_month(tmp_path):
    root = tmp_path / "player_history.parquet"
    P._save_history(_frame(), root)
    before = _digest(root)
    cur = pd.read_parquet(root)
    i = cur.index[pd.to_datetime(cur["date"]).dt.strftime("%Y-%m") == "2026-09"][0]
    cur.loc[i, "opp_strength_index"] = np.float32(0.123)
    st = P._save_history(cur, root)
    assert st["written"] == ["month_2026-09.parquet"] and len(st["unchanged"]) == 4
    assert {n for n, d in _digest(root).items() if before[n] != d} == {"month_2026-09.parquet"}


def test_backfill_of_a_new_league_touches_only_its_months(tmp_path):
    root = tmp_path / "player_history.parquet"
    P._save_history(_frame(), root)
    before = _digest(root)
    bf = _frame(n_per_month=5, months=("2025-08", "2025-09"), seed=3)
    bf["league"] = "Ligue 2"; bf["fixture_id"] += 50_000
    P._save_history(pd.concat([pd.read_parquet(root), bf], ignore_index=True), root)
    after = _digest(root)
    assert {n for n in after if before.get(n) != after[n]} == {"month_2025-08.parquet",
                                                               "month_2025-09.parquet"}


def test_migrates_old_season_parts_and_removes_them(tmp_path):
    root = tmp_path / "player_history.parquet"
    root.mkdir()
    df = _cast(_frame())       # season parts as the 2026-09-27 writer stored them
    season = pd.to_datetime(df["date"]).dt.year - (pd.to_datetime(df["date"]).dt.month < 7)
    for s, g in df.groupby(season):
        g.to_parquet(root / f"season_{s}.parquet", index=False)
    old = pd.read_parquet(root)
    st = P._save_history(old, root)
    assert sorted(st["removed"]) == ["season_2024.parquet", "season_2025.parquet", "season_2026.parquet"]
    assert not list(root.glob("season_*.parquet"))
    new = pd.read_parquet(root)
    assert list(new.columns) == list(old.columns)
    pd.testing.assert_frame_equal(_sorted(new), _sorted(old), check_exact=True)
    # date-sorted reads see the same frame as before the migration
    pd.testing.assert_frame_equal(
        new.sort_values("date", kind="stable").reset_index(drop=True),
        old.sort_values("date", kind="stable").reset_index(drop=True), check_exact=True)


def test_legacy_single_file_self_migrates(tmp_path):
    root = tmp_path / "player_history.parquet"
    _frame().to_parquet(root, index=False)          # the pre-2026-09-27 single-file layout
    P._save_history(pd.read_parquet(root), root)
    assert root.is_dir() and len(list(root.glob("month_*.parquet"))) == 5


def test_unparseable_dates_kept_in_unknown_part(tmp_path):
    root = tmp_path / "player_history.parquet"
    df = _frame()
    df["date"] = df["date"].astype(object)
    df.loc[0, "date"] = None
    P._save_history(df, root)
    assert (root / "month_unknown.parquet").exists()
    assert len(pd.read_parquet(root)) == len(df)


def test_all_parts_share_one_schema_even_when_a_month_is_all_null(tmp_path):
    import pyarrow.parquet as pq
    root = tmp_path / "player_history.parquet"
    df = _frame()
    df.loc[pd.to_datetime(df["date"]).dt.strftime("%Y-%m") == "2025-05", "player_name"] = None
    P._save_history(df, root)
    schemas = {f.name: pq.read_schema(f) for f in root.glob("*.parquet")}
    first = next(iter(schemas.values()))
    assert all(s.equals(first, check_metadata=False) for s in schemas.values())
    assert str(first.field("player_name").type) in ("string", "large_string")


def test_size_guard_runs_per_part(tmp_path, capsys):
    root = tmp_path / "player_history.parquet"
    root.mkdir()
    (root / "month_2026-09.parquet").write_bytes(b"x" * 2_000_000)   # 2.0 MB
    (root / "month_2026-10.parquet").write_bytes(b"x" * 1_200_000)   # 1.2 MB
    (root / "month_2026-08.parquet").write_bytes(b"x" * 500_000)     # 0.5 MB
    msgs = P._size_guard(root, warn_mb=1.0, error_mb=1.5)
    assert len(msgs) == 2
    assert "ERROR" in msgs[0] and "month_2026-09" in msgs[0]          # judged on its own size
    assert "WARNING" in msgs[1] and "month_2026-10" in msgs[1]
    assert "month_2026-08" not in " ".join(msgs)                       # the 3.7 MB total is irrelevant
    assert "REJECT" in capsys.readouterr().out
    assert P._size_guard(root) == []                                    # real 70/100 MB limits


def test_real_history_parts_are_under_the_guard():
    root = Path(P.HISTORY_CACHE)
    if not root.is_dir():
        pytest.skip("no player_history.parquet checkout")
    assert not list(root.glob("season_*.parquet")), "old season parts still present"
    assert P._size_guard(root) == []
