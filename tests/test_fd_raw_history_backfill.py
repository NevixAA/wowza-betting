"""football-data season files as a history source + their provenance (fix/fd-ht-and-history-backfill).

THE DEFECT (2026-11-18). Standard training had half-time goals missing on 37% of rows (2021/22 ~99%):
fd_history.parquet starts at 2023/24 and the older seasons came only from the backtest archive, which
has no HT column, although football-data's raw files carry HTHG/HTAG on ~100% of rows. And the exact-key
de-duplication dropped every cell only the dropped copy held.

    python -m pytest tests/test_fd_raw_history_backfill.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data_loader as DL  # noqa: E402
from src.integrity.duplicate_matches import SOURCE_PRIORITY  # noqa: E402

OLD = (b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR\n"
       b"E1,05/08/16,Fulham,Newcastle,1,0,H,1,0,H\n"
       b"E1,06/08/16,Brighton,Derby,0,0,D,0,0,D\n"
       b"E1,07/08/16,Leeds,QPR,0,3,A,0,1,A\n")
NEW = (b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HC,AC,HY,AY\n"
       b"E1,06/08/2021,Fulham,Middlesbrough,1,1,D,0,1,A,10,8,4,2,6,3,1,2\n"
       b"E1,07/08/2021,Derby,Huddersfield,1,1,D,0,0,D,12,9,3,3,5,5,2,1\n")


def _bank(tmp_path, prov_rows=None):
    d = tmp_path / "output" / "collect" / "fd_raw_history"
    d.mkdir(parents=True)
    (d / "E1_1617.csv").write_bytes(OLD)
    (d / "E1_2122.csv").write_bytes(NEW)
    if prov_rows is not None:
        pd.DataFrame(prov_rows).to_parquet(tmp_path / DL.FD_RAW_PROVENANCE, index=False)
    return tmp_path


def _prov(fd_file, date, h, a, check, af_date=None, fid=None):
    return {"fd_file": fd_file, "league": "Championship", "date": pd.Timestamp(date), "home_team": h,
            "away_team": a, "ft_check": check, "af_date": af_date, "af_fixture_id": fid}


def test_old_files_parse_two_digit_years_and_unpublished_columns_stay_missing():
    raw = pd.read_csv(pd.io.common.BytesIO(OLD))
    f = DL.parse_fd_season_file(raw, "Championship", "2016/17")
    assert list(f["date"].dt.strftime("%Y-%m-%d")) == ["2016-08-05", "2016-08-06", "2016-08-07"]
    assert f["ht_home_goals"].tolist() == [1, 0, 0] and f["ht_away_goals"].tolist() == [0, 0, 1]
    # 2016/17 publishes no shots / corners / cards: missing, never 0
    for c in ("home_shots", "home_corners", "home_yellows", "odds_over25"):
        assert f[c].isna().all()


def test_raw_seasons_are_served_with_ht_and_bookings(tmp_path):
    out = DL.read_fd_raw_seasons(root=_bank(tmp_path))
    assert len(out) == 5 and out["ht_home_goals"].notna().all()
    assert set(out["season"]) == {"2016/17", "2021/22"}
    new = out[out["season"] == "2021/22"]
    assert new["home_yellows"].tolist() == [1, 2] and new["home_corners"].tolist() == [6, 5]
    assert "fd_file" not in out.columns


def test_a_score_the_provider_disputes_or_marks_not_played_is_not_served(tmp_path):
    root = _bank(tmp_path, [_prov("E1_1617.csv", "2016-08-05", "Fulham", "Newcastle", "mismatch"),
                            _prov("E1_1617.csv", "2016-08-06", "Brighton", "Derby", "provider_not_played"),
                            _prov("E1_1617.csv", "2016-08-07", "Leeds", "QPR", "no_af_fixture")])
    out = DL.read_fd_raw_seasons(root=root)
    assert len(out) == 3
    old = out[out["season"] == "2016/17"]
    assert old["home_team"].tolist() == ["Leeds"]
    assert ((out["home_team"] == "Leeds") & (out["season"] == "2016/17")).sum() == 1


def test_unreadable_provenance_serves_nothing(tmp_path):
    root = _bank(tmp_path)
    (tmp_path / DL.FD_RAW_PROVENANCE).write_bytes(b"not a parquet file")
    assert DL.read_fd_raw_seasons(root=root).empty


def test_provider_date_and_af_values_of_the_proven_fixture(tmp_path):
    root = _bank(tmp_path, [_prov("E1_2122.csv", "2021-08-06", "Fulham", "Middlesbrough", "equal",
                                  "2021-08-07", 11),
                            _prov("E1_2122.csv", "2021-08-07", "Derby", "Huddersfield", "mismatch", None, 12)])
    stats = tmp_path / "output" / "collect" / "af_fixture_stats.parquet"
    pd.DataFrame({"fixture_id": [11, 12], "HXG": [1.4, 0.3], "AXG": [0.9, 0.2],
                  "HIB": [7.0, 1.0], "AIB": [5.0, 1.0]}).to_parquet(stats, index=False)
    full = tmp_path / "output" / "collect" / "af_history_fullstats_fill.parquet"
    pd.DataFrame({"fixture_id": [11], "home_shots": [99.0], "home_yellow": [9.0]}).to_parquet(full, index=False)
    out = DL.read_fd_raw_seasons(root=root)
    r = out[(out["home_team"] == "Fulham") & (out["season"] == "2021/22")].iloc[0]
    assert r["date"] == pd.Timestamp("2021-08-07")               # served on the provider's date
    assert r["home_xg"] == 1.4 and r["away_insidebox"] == 5.0     # bought values attached
    assert r["home_shots"] == 10 and r["home_yellows"] == 1        # football-data value never overwritten
    assert not (out["home_team"] == "Derby").any()                 # disputed score not served


def test_exact_copies_keep_every_real_value_but_never_mix_prices():
    base = {"date": pd.Timestamp("2021-08-06"), "league": "Championship", "season": "2021/22",
            "home_team": "Fulham", "away_team": "Middlesbrough", "home_goals": 1.0, "away_goals": 1.0}
    rich = {**base, "home_yellows": 1.0, "away_yellows": 2.0, "home_corners": 6.0, "odds_over25": 2.0,
            "odds_under25": np.nan, "ht_home_goals": np.nan, "row_source": "football_data_raw"}
    thin = {**base, "home_yellows": np.nan, "away_yellows": np.nan, "home_corners": np.nan,
            "odds_over25": np.nan, "odds_under25": 1.8, "ht_home_goals": 0.0, "row_source": "fd_cache"}
    out = DL._canonicalise_clubs(pd.DataFrame([rich, thin]))
    assert len(out) == 1
    r = out.iloc[0]
    assert r["ht_home_goals"] == 0.0 and r["home_yellows"] == 1.0 and r["home_corners"] == 6.0
    assert r["odds_over25"] == 2.0 and pd.isna(r["odds_under25"])  # O/U pair stays from one copy
    assert r["row_source"] == "fd_cache"                          # best source among exact copies


def test_raw_files_rank_below_history_and_provider_above_archive():
    assert SOURCE_PRIORITY["fd_cache"] < SOURCE_PRIORITY["football_data_raw"]
    assert SOURCE_PRIORITY["provider_supplement"] < SOURCE_PRIORITY["football_data_raw"]
    assert SOURCE_PRIORITY["football_data_raw"] < SOURCE_PRIORITY["backtest_archive"]
    from src.history_corrections import HISTORY_SOURCES
    assert "football_data_raw" in HISTORY_SOURCES


def test_history_collector_plans_the_older_seasons_into_its_own_directory(tmp_path, monkeypatch):
    import collect_fd_raw as FD
    assert FD.history_seasons() == ["1617", "1718", "1819", "1920", "2021", "2122"]
    monkeypatch.setattr(FD, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(FD, "plan", lambda now, history=False: [("Championship", "E1", "1617")])

    class _R:
        status_code, content = 200, OLD

    class _S:
        def get(self, url, **k):
            return _R()
    st = FD.run(session=_S(), history=True)
    assert st["written"] == 1 and (tmp_path / "E1_1617.csv").read_bytes() == OLD
    assert (tmp_path / "manifest.json").exists()


def test_verify_pairs_rows_with_the_provider_fixture():
    import verify_fd_raw_seasons as V
    rows = DL.parse_fd_season_file(pd.read_csv(pd.io.common.BytesIO(NEW)), "Championship", "2021/22")
    rows = rows.assign(fd_file="E1_2122.csv", fd_dir="x", url="u", sha256="s", fetched_at="t")
    scores = pd.DataFrame({
        "fixture_id": [1, 2], "league": "Championship", "date": ["2021-08-06", "2021-08-07"],
        "kickoff": None, "home_name": ["Fulham", "Derby"], "away_name": ["Middlesbrough", "Huddersfield"],
        "ht_home": [0, 0], "ht_away": [1, 0], "ft_home": [1, 2], "ft_away": [1, 1], "status": "FT",
        "et_home": None, "et_away": None, "pen_home": None, "pen_away": None})
    p = V.verify(rows, scores)
    chk = dict(zip(p["home_team"], p["ft_check"]))
    assert chk == {"Fulham": "equal", "Derby": "mismatch"}
    assert p.loc[p["home_team"] == "Fulham", "ht_check"].iloc[0] == "equal"


def test_banked_files_keep_the_prematch_and_closing_pairs_apart():
    """Main's 2026-11-11 rule holds for the banked season files too: odds_over25 / odds_under25 are the
    PRE-MATCH pair (Bet365 first), the closing pair is separate (CLV only), each with its source."""
    raw = pd.DataFrame({"Div": ["E1"], "Date": ["06/08/2021"], "HomeTeam": ["Fulham"], "AwayTeam": ["Derby"],
                        "FTHG": [1], "FTAG": [1], "FTR": ["D"], "HTHG": [0], "HTAG": [1],
                        "B365>2.5": [2.0], "B365<2.5": [1.8], "AvgC>2.5": [2.2], "AvgC<2.5": [1.7]})
    f = DL.parse_fd_season_file(raw, "Championship", "2021/22")
    assert f["odds_over25"].iloc[0] == 2.0 and f["odds_under25"].iloc[0] == 1.8
    assert f["odds_ou25_source"].iloc[0] == "B365"
    assert f["odds_over25_close"].iloc[0] == 2.2 and f["odds_ou25_close_source"].iloc[0] == "AVGC"
