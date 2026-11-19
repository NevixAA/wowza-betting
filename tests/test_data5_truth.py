"""Data truth round 5 (fix5/data5, 2026-11-02): HT fill wiring, bookmaker_overround without the clip,
rule 5 gated, alias-store identities, side-bets first-half CLV quarantine.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_data5_truth.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as DL
from src.integrity import data_truth as DT

NF = "Brazil Serie A"


# ── 1 HT fill sidecar ────────────────────────────────────────────────────────────────────────────
def _hist():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-07-05", "2026-07-12", "2026-07-19", "2026-07-26"]),
        "league": NF, "home_team": ["Flamengo", "Santos", "Flamengo", "Gremio"],
        "away_team": ["Santos", "Gremio", "Gremio", "Santos"],
        "home_goals": [2.0, 1.0, 0.0, 3.0], "away_goals": [1.0, 1.0, 0.0, 2.0],
        "ht_home_goals": [1.0, np.nan, np.nan, np.nan], "ht_away_goals": [0.0, np.nan, np.nan, np.nan],
        "ht_over05": [1.0, np.nan, np.nan, np.nan]})


def _write_fill(root, rows):
    p = root / "output" / "collect" / "fd_history_ht_fill.parquet"
    p.parent.mkdir(parents=True)
    pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "HTHG", "HTAG", "fixture_id"]) \
        .to_parquet(p, index=False)


def test_ht_fill_sidecar_fills_only_missing_and_relabels(tmp_path):
    _write_fill(tmp_path, [
        (NF, "2026-07-05", "Flamengo", "Santos", 2, 1, 1),        # stored HT 1-0 must NOT be overwritten
        (NF, "2026-07-12", "Santos", "Gremio", 0, 0, 2),
        (NF, "2026-07-19", "Flamengo", "Gremio", None, None, 3),  # null HT stays missing, never 0
        ("Japan J-League", "2026-07-26", "Gremio", "Santos", 1, 1, 4)])   # other league: never joins
    out = DL._enrich_ht_scores(_hist(), root=tmp_path)
    assert out["ht_home_goals"].tolist()[:2] == [1.0, 0.0] and out["ht_away_goals"].tolist()[:2] == [0.0, 0.0]
    assert out.loc[2:, "ht_home_goals"].isna().all()
    # the labels follow the filled score (they were built BEFORE step 1b and never refreshed)
    assert out["ht_over05"].tolist()[:2] == [1.0, 0.0] and out.loc[2:, "ht_over05"].isna().all()
    cov = out.attrs["ht_coverage"]
    assert cov["before"][NF] == 0.25 and cov["after"][NF] == 0.5
    assert cov["filled_rows"]["fd_history_ht_fill"] == 1


def test_ht_fill_absent_sidecar_changes_nothing(tmp_path):
    h = _hist()
    out = DL._enrich_ht_scores(h.copy(), root=tmp_path)
    pd.testing.assert_series_equal(out["ht_home_goals"], h["ht_home_goals"])
    assert out.attrs["ht_coverage"]["after"] == out.attrs["ht_coverage"]["before"] == {NF: 0.25}


def test_both_loader_paths_read_ht_through_the_one_step():
    import inspect
    src = inspect.getsource(DL._enrich_with_api_shots)
    assert "_enrich_ht_scores(df)" in src
    assert [n for n, _ in DL.HT_SOURCES] == ["af_ht_history", "fd_history_ht_fill"]


# ── 2 bookmaker_overround: missing, never an invented 0 ──────────────────────────────────────────
def test_bookmaker_overround_negative_margin_is_missing():
    from src.feature_engineering import bookmaker_overround
    r = bookmaker_overround(pd.Series([2.10, 1.80, np.nan, 0.0]), pd.Series([2.10, 1.90, 1.9, 1.9]))
    assert np.isnan(r[0]) and r[1] == pytest.approx(1 / 1.8 + 1 / 1.9 - 1) and r[2:].isna().all()


def test_training_path_leaves_a_negative_margin_missing():
    from src.feature_engineering import build_features
    from tests.test_standard_forensic_repair import _frame
    raw = _frame()
    i = raw.index[40]
    raw.loc[i, ["odds_over25", "odds_under25"]] = [2.10, 2.10]          # margin -4.8%: not a price
    f = build_features(raw)
    row = f[(f["date"] == raw.loc[i, "date"]) & (f["home_team"] == raw.loc[i, "home_team"])
            & (f["league"] == raw.loc[i, "league"])]
    assert len(row) == 1 and np.isnan(row["bookmaker_overround"].iloc[0])
    m = 1 / f["odds_over25"] + 1 / f["odds_under25"] - 1
    assert (m < 0).sum() > 1                                            # the frame holds such pairs
    assert f.loc[m < 0, "bookmaker_overround"].isna().all()             # was exactly 0.0 (clip)
    assert np.allclose(f.loc[m >= 0, "bookmaker_overround"], m[m >= 0])


def test_serving_negative_best_pair_is_missing_not_the_one_book_margin():
    from src.feature_engineering import build_upcoming_features
    from tests.test_standard_forensic_repair import STD_A, _frame
    raw = _frame()
    nxt = raw["date"].max() + pd.Timedelta(days=7)
    st_t = sorted(raw.loc[raw.league == STD_A, "home_team"].unique())
    up = pd.DataFrame({"date": nxt, "league": [NF, STD_A], "home_team": ["N0", st_t[0]],
                       "away_team": ["N1", st_t[1]], "odds_over25": [1.80, 2.10], "odds_under25": [1.90, 2.10],
                       "best_over25": 2.10, "best_under25": 2.10})
    f = build_upcoming_features(up, raw).set_index("league")
    assert np.isnan(f.loc[NF, "bookmaker_overround"])                # was 0.0 (clip) - training reads NaN
    assert np.isnan(f.loc[STD_A, "bookmaker_overround"])             # one-book negative pair: NaN, not 0


def test_no_overround_clip_left_in_feature_engineering():
    import inspect
    from src import feature_engineering as FE
    src = inspect.getsource(FE)
    assert "- 1.0\n    ).clip(lower=0.0)" not in src and "- 1.0).clip(lower=0.0)" not in src


# ── 3 rule 5 gated ───────────────────────────────────────────────────────────────────────────────
GREEN = [{"check": f"Data truth {i}: x", "status": "GREEN", "value": "", "detail": ""} for i in (1, 2, 3, 4, 5)]


def test_rule5_red_blocks_the_stoplight():
    red5 = [dict(c, status="RED") if c["check"].startswith("Data truth 5") else c for c in GREEN]
    assert any(f.startswith("Data truth 5") for f in DT.gate(checks=red5))


def test_a_crashed_gated_check_blocks_under_its_rule_label(monkeypatch):
    def boom():
        raise RuntimeError("bank unreadable")
    for name in ("naming", "defaults", "missing", "leagues"):
        monkeypatch.setattr(DT, name, lambda *a, **k: {"check": "stub", "status": "GREEN", "value": "",
                                                         "rule": "", "detail": ""})
    monkeypatch.setattr(DT, "one_match", boom)
    checks = DT.evaluate()
    c5 = [c for c in checks if c["check"].startswith("Data truth 5")]
    assert len(c5) == 1 and c5[0]["status"] == "RED"
    assert any(f.startswith("Data truth 5") for f in DT.gate(checks=checks))


# ── 4 alias store: Verona, Montreal, Czechia, Craiova ────────────────────────────────────────────
@pytest.mark.parametrize("a,b,other", [("Verona", "Hellas Verona", "Inter"),
                                       ("Montreal Impact", "CF Montreal", "Toronto FC"),
                                       ("Czech Republic", "Czechia", "Albania"),
                                       ("CS Universitatea Craiova", "Univ. Craiova", "FC Botosani"),
                                       ("Universitatea Craiova", "Univ. Craiova", "Farul Constanta")])
def test_alias_pairs_resolve_both_ways(a, b, other):
    from src.team_names import resolve
    assert resolve(a, [b, other]) == b and resolve(b, [a, other]) == a


def test_fc_u_craiova_1948_stays_a_different_club():
    from src.team_names import resolve
    assert resolve("U Craiova 1948", ["Univ. Craiova", "CS Universitatea Craiova"]) is None
    assert resolve("CS Universitatea Craiova", ["U Craiova 1948", "FCSB"]) is None


def test_player_block_name_now_proves_the_side():
    from player_model.club_rule import side_of
    assert side_of("Verona", "Hellas Verona", "Inter") is True
    assert side_of("Montreal Impact", "Toronto FC", "CF Montreal") is False
    assert side_of("Czech Republic", "Czechia", "Albania") is True


def _r(d, lg, h, a, hg, ag):
    return {"date": pd.Timestamp(d), "league": lg, "season": None, "home_team": h, "away_team": a,
            "home_goals": float(hg), "away_goals": float(ag), "row_source": "fd_cache"}


def test_loader_unifies_an_alias_store_split_the_evidence_cannot_see():
    lg = "Serie B"
    rows = [_r("2026-08-23", lg, "Hellas Verona", "Ascoli", 1, 2), _r("2026-08-29", lg, "Padova", "Hellas Verona", 1, 1),
            _r("2026-09-13", lg, "Verona", "Bari", 2, 0), _r("2026-09-20", lg, "Palermo", "Verona", 0, 0),
            _r("2026-09-27", lg, "Bari", "Ascoli", 1, 0)]
    out = DL.one_club_one_match(pd.DataFrame(rows))
    names = set(out["home_team"]) | set(out["away_team"])
    assert "Verona" in names and "Hellas Verona" not in names
    assert DL.ALIAS_STORE_MERGES == {lg: {"Hellas Verona": "Verona"}}


def test_loader_keeps_alias_spellings_that_meet_apart():
    lg = "Serie B"
    rows = [_r("2026-08-23", lg, "Hellas Verona", "Verona", 1, 2), _r("2026-09-13", lg, "Verona", "Bari", 2, 0)]
    out = DL.one_club_one_match(pd.DataFrame(rows))
    assert {"Hellas Verona", "Verona"} <= set(out["home_team"]) | set(out["away_team"])


def test_craiova_split_unified_and_1948_untouched():
    lg = "Romanian Superliga"
    rows = [_r("2021-09-01", lg, "CS Universitatea Craiova", "FCSB", 1, 0),
            _r("2021-10-01", lg, "Rapid", "CS Universitatea Craiova", 0, 2),
            _r("2022-02-14", lg, "CS Universitatea Craiova", "U Craiova 1948", 2, 0),
            _r("2022-08-01", lg, "Univ. Craiova", "Farul Constanta", 3, 1),
            _r("2022-09-01", lg, "U Craiova 1948", "Univ. Craiova", 0, 1)]
    out = DL.one_club_one_match(pd.DataFrame(rows))
    names = set(out["home_team"]) | set(out["away_team"])
    assert "Univ. Craiova" in names and "CS Universitatea Craiova" not in names and "U Craiova 1948" in names


def test_alias_groups_in_finds_only_store_pairs():
    from src.team_names import alias_groups_in
    g = alias_groups_in(["Verona", "Hellas Verona", "Inter", "U Craiova 1948", "Univ. Craiova"])
    assert g == [{"Verona", "Hellas Verona"}] or g == [{"Hellas Verona", "Verona"}]


# ── 5 side_bets first-half CLV quarantine ────────────────────────────────────────────────────────
def test_committed_side_sidecar_lists_the_six_rows_with_archive_evidence():
    from src import clv_corrections as CC
    s = pd.read_csv(CC.sidecar(CC.SIDE_CLV_SIDECAR))
    assert len(s) == 6 and s["archive_evidence"].all()
    assert set(s["market"]) == {"btts"} and set(s["side"]) == {"YES"}
    assert (pd.to_numeric(s["clv_pct"]).abs() > 50).all() and (pd.to_numeric(s["closing_odds"]) > 3.2).all()
    a = pd.read_csv(CC.sidecar(CC.SIDE_CLV_ARCHIVE_SIDECAR))
    assert len(a) == 7 and set(a["status"]) == {"QUARANTINED"}


def test_side_ledger_clv_excluded_and_kept_beside():
    import config
    from src import clv_corrections as CC
    s = pd.read_csv(config.OUTPUT_DIR / "side_bets_ledger.csv", low_memory=False)
    c = CC.exclude_contaminated_clv(s)
    assert int(c["clv_quarantined"].sum()) == 6
    assert c.loc[c["clv_quarantined"], "clv_pct"].isna().all()
    assert (c.loc[c["clv_quarantined"], "clv_pct_stored"].abs() > 50).all()
    assert (pd.to_numeric(c["clv_pct"], errors="coerce").abs().dropna() <= 50).all()
    pd.testing.assert_series_equal(c["pnl"], s["pnl"])


def test_side_key_includes_the_market_and_bets_ledger_is_unchanged(tmp_path):
    from src import clv_corrections as CC
    (tmp_path / "corrections").mkdir()
    pd.DataFrame([{"match_date": "2026-08-22", "league": "X", "home_team": "A", "away_team": "B",
                   "market": "btts", "side": "YES"}]).to_csv(tmp_path / "corrections" / CC.SIDE_CLV_SIDECAR, index=False)
    pd.DataFrame([{"match_date": "2026-08-22", "league": "X", "home_team": "A", "away_team": "B",
                   "side": "OVER"}]).to_csv(tmp_path / "corrections" / "clv_first_half_contaminated.csv", index=False)
    side = pd.DataFrame({"match_date": "2026-08-22", "league": "X", "home_team": "A", "away_team": "B",
                         "market": ["btts", "over15"], "side": ["YES", "OVER"], "clv_pct": [-60.0, 5.0]})
    assert CC.contaminated_mask(side, tmp_path).tolist() == [True, False]
    bets = pd.DataFrame({"match_date": "2026-08-22", "league": "X", "home_team": "A", "away_team": "B",
                         "side": ["OVER"], "clv_pct": [80.0]})
    assert CC.contaminated_mask(bets, tmp_path).tolist() == [True]
    mixed = pd.concat([bets, side], ignore_index=True)
    assert CC.contaminated_mask(mixed, tmp_path).tolist() == [True, True, False]
