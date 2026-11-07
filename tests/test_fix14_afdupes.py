"""fix14/afdupes (owner 2026-11-04 duplicate rule, extended to af_history): the copy of a provider
fixture filed one day off the provider's date is removed in the READ, proven by fixture id + both
clubs + the full-time score where held; provider values of the removed copy fill only EMPTY cells of
the kept copy, conflicting provider cells become missing (never averaged), a written-back copy gives
nothing, and the sidecars keyed on a removed copy move onto the kept copy's key."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src import af_history_corrections as AC  # noqa: E402

LG = "Mexico Liga MX"
NAN = np.nan


def _row(date, home, away, **kw):
    base = {"league": LG, "date": pd.Timestamp(date), "home_team": home, "away_team": away,
            "season": "2026", "FTHG": NAN, "FTAG": NAN, "FTR": None,
            **{c: NAN for c in AC.STAT_COLS}, "fixture_id": NAN, "home_team_id": NAN, "away_team_id": NAN}
    base.update(kw)
    return base


def _corr(removed: dict, kept_date, kept_home, kept_away, fid=1551013):
    r = {c: removed.get(c) for c in AC.KEY + AC.SIGNATURE}
    r["date"] = pd.Timestamp(removed["date"]).strftime("%Y-%m-%d")
    r.update({"kept_date": kept_date, "kept_home_team": kept_home, "kept_away_team": kept_away,
              "provider_fixture_id": fid})
    return pd.DataFrame([r])


def test_written_back_copy_is_removed_and_gives_nothing():
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, FTR="D", HS=16.0, AS=14.0)
    rem = _row("2026-10-25", "Club Leon", "Toluca", HS=27.0, AS=9.0, HC=NAN, HIB=8.0)
    af = pd.DataFrame([kept, rem, _row("2026-10-25", "Pachuca", "Atlas")])
    out = AC.apply_af_history_corrections(af, _corr(rem, "2026-10-24", "Leon", "Toluca"))
    assert len(out) == 2 and "Club Leon" not in set(out["home_team"])
    k = out[out["home_team"] == "Leon"].iloc[0]
    assert k["HS"] == 16.0 and np.isnan(k["HIB"])          # the unproven 27 / 8 never reach the kept row
    assert out.attrs["af_history_corrections"]["total"]["rows_removed"] == 1
    assert out.attrs["af_history_corrections"]["total"]["cells_filled"] == 0


def test_provider_copy_fills_empty_cells_of_a_written_back_kept_row():
    kept = _row("2026-04-19", "Talleres", "Riestra", HXG=0.9, HS=99.0)      # written back: unproven
    rem = _row("2026-04-20", "Talleres", "Riestra", FTHG=2.0, FTAG=0.0, FTR="H", HS=15.0, AS=9.0, HIB=9.0)
    out = AC.apply_af_history_corrections(pd.DataFrame([kept, rem]),
                                          _corr(rem, "2026-04-19", "Talleres", "Riestra"))
    assert len(out) == 1
    k = out.iloc[0]
    assert (k["FTHG"], k["FTAG"], k["HS"], k["AS"], k["HIB"]) == (2.0, 0.0, 15.0, 9.0, 9.0)
    assert np.isnan(k["HXG"])                              # unproven written-back cell cleared, not kept
    assert pd.Timestamp(k["date"]) == pd.Timestamp("2026-04-19")


def test_conflicting_provider_cells_become_missing_never_averaged():
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=16.0, AS=14.0)
    rem = _row("2026-10-25", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=27.0, AS=14.0, HC=3.0)
    out = AC.apply_af_history_corrections(pd.DataFrame([kept, rem]),
                                          _corr(rem, "2026-10-24", "Leon", "Toluca"))
    k = out.iloc[0]
    assert np.isnan(k["HS"]) and k["AS"] == 14.0 and k["HC"] == 3.0
    t = out.attrs["af_history_corrections"]["total"]
    assert t["cells_made_missing"] == 1 and t["cells_filled"] == 1


def test_row_changed_since_listing_or_kept_row_absent_is_left_in():
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0)
    rem = _row("2026-10-25", "Club Leon", "Toluca", HS=27.0)
    corr = _corr(rem, "2026-10-24", "Leon", "Toluca")
    changed = dict(rem, HS=26.0)
    out = AC.apply_af_history_corrections(pd.DataFrame([kept, changed]), corr)
    assert len(out) == 2
    assert out.attrs["af_history_corrections"]["total"]["listed_rows_changed_since_listing"] == 1
    out2 = AC.apply_af_history_corrections(pd.DataFrame([rem]), corr)
    assert len(out2) == 1 and out2.attrs["af_history_corrections"]["total"]["kept_row_absent"] == 1


def test_empty_or_absent_list_changes_nothing(tmp_path):
    af = pd.DataFrame([_row("2026-10-24", "Leon", "Toluca")])
    assert AC.apply_af_history_corrections(af, AC.read_af_corrections(tmp_path / "none.csv")) is af
    assert AC.rekey_sidecar(af, AC.read_af_corrections(tmp_path / "none.csv")) is af


def test_sidecar_rows_move_onto_the_kept_key_same_fixture_only():
    rem = _row("2026-10-25", "Club Leon", "Toluca")
    corr = _corr(rem, "2026-10-24", "Leon", "Toluca", fid=1551013)
    side = pd.DataFrame([
        {"league": LG, "date": "2026-10-25", "home_team": "Club Leon", "away_team": "Toluca",
         "HIB": 12.0, "fixture_id": 1551013},
        {"league": LG, "date": "2026-10-26", "home_team": "Pachuca", "away_team": "Atlas",
         "HIB": 5.0, "fixture_id": 7},
    ])
    out = AC.rekey_sidecar(side, corr)
    assert set(zip(out["date"], out["home_team"])) == {("2026-10-24", "Leon"), ("2026-10-26", "Pachuca")}
    # the kept key already answered -> the moved row is dropped (one answer per fixture)
    held = pd.concat([side, pd.DataFrame([{"league": LG, "date": "2026-10-24", "home_team": "Leon",
                                           "away_team": "Toluca", "HIB": 12.0,
                                           "fixture_id": 1551013}])], ignore_index=True)
    out2 = AC.rekey_sidecar(held, corr)
    assert len(out2) == 2 and out2.attrs["af_sidecar_rekey"]["dropped_kept_key_held"] == 1
    # another fixture id on the removed key -> left where it is
    other = side.assign(fixture_id=[999, 7])
    out3 = AC.rekey_sidecar(other, corr)
    assert out3.equals(other) and out3.attrs["af_sidecar_rekey"]["left_other_fixture"] == 1


def _fixtures():
    fx = pd.DataFrame([{"fixture_id": 1551013, "league_id": 262, "season": "2026", "date": "2026-10-24",
                        "home_id": 2289, "home_name": "Leon", "away_id": 2281, "away_name": "Toluca"}])
    fx["_d"] = pd.to_datetime(fx["date"]).dt.normalize()
    prov = fx.assign(kickoff="2026-10-24T23:00:00+00:00", ft_home=1, ft_away=1, status="FT")
    return fx, prov


def test_build_proves_a_pair_and_keeps_the_provider_date_copy():
    import build_af_history_corrections as B
    fx, prov = _fixtures()
    af = pd.DataFrame([_row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, FTR="D", HS=16.0),
                       _row("2026-10-25", "Club Leon", "Toluca", HS=27.0)])
    rem, con, st = B.build(af, fx, B.provider_records(comp=pd.DataFrame(), ht=prov))
    assert con.empty and len(rem) == 1
    r = rem.iloc[0]
    assert (r["date"], r["home_team"], r["kept_date"], r["kept_home_team"]) == \
        ("2026-10-25", "Club Leon", "2026-10-24", "Leon")
    assert r["HS"] == 27.0 and bool(r["removed_is_written_back"]) and r["provider_fixture_id"] == 1551013
    assert st["total"]["proven"] == 1


def test_build_lists_a_score_conflict_and_removes_nothing():
    import build_af_history_corrections as B
    fx, prov = _fixtures()
    af = pd.DataFrame([_row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0),
                       _row("2026-10-25", "Leon", "Toluca", FTHG=2.0, FTAG=1.0)])
    rem, con, st = B.build(af, fx, B.provider_records(comp=pd.DataFrame(), ht=prov))
    assert rem.empty and len(con) == 2
    assert set(con["conflict"]) == {"score_differs_between_copies"}


def test_collector_still_refuses_unproven_two_date_claims_by_default():
    import collect_af_history_gap as G
    fx, _ = _fixtures()
    rows = pd.DataFrame([_row("2026-10-24", "Leon", "Toluca"), _row("2026-10-25", "Leon", "Toluca")])
    rows = rows.assign(canon=LG, league_id=262)
    m, st = G.match_fixtures(rows, fx)
    assert m.empty and st[LG]["refused_fixture_claimed_twice"] == 2
    m2, _ = G.match_fixtures(rows, fx, refuse_claimed_twice=False)
    assert len(m2) == 2 and m2["claimed_twice"].all()


def test_every_af_history_reader_applies_the_rule():
    for f in ("src/data_loader.py", "src/strength/results.py", "src/research/edge_replay.py",
              "scripts/collect_af_history_gap.py"):
        assert "apply_af_history_corrections" in (ROOT / f).read_text(encoding="utf-8-sig"), f
    dl = (ROOT / "src/data_loader.py").read_text(encoding="utf-8")
    assert dl.count("rekey_sidecar(fill") == 2
    wf = (ROOT / ".github/workflows/data_truth_collect.yml").read_text(encoding="utf-8")
    assert wf.count("build_af_history_corrections.py") == 2
    assert "output/corrections/af_history_removed_duplicates.csv" in wf


@pytest.mark.skipif(not AC.AF_DUP_REMOVED.exists(), reason="committed list absent")
def test_committed_list_holds_full_original_values_and_evidence():
    d = pd.read_csv(AC.AF_DUP_REMOVED, low_memory=False)
    af_cols = list(pd.read_parquet(ROOT / "output" / "af_history.parquet").columns)
    assert list(d.columns[:len(af_cols)]) == af_cols
    for c in AC.KEPT + ["provider_fixture_id", "provider_date", "provider_kickoff_utc", "found_by"]:
        assert d[c].notna().all(), c
    assert (pd.to_datetime(d["kept_date"]) - pd.to_datetime(d["date"])).abs().dt.days.eq(1).all()
    assert (d["kept_date"] == d["provider_date"]).all()


def test_join_alias_offers_the_kept_values_under_the_removed_key_for_the_loader_only():
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=16.0)
    rem = _row("2026-10-25", "Club Leon", "Toluca", HS=27.0)
    corr = _corr(rem, "2026-10-24", "Leon", "Toluca")
    out = AC.add_join_aliases(AC.apply_af_history_corrections(pd.DataFrame([kept, rem]), corr), corr)
    assert len(out) == 2
    alias = out[out["home_team"] == "Club Leon"].iloc[0]
    assert pd.Timestamp(alias["date"]) == pd.Timestamp("2026-10-25")
    assert alias["HS"] == 16.0 and alias["FTHG"] == 1.0         # the kept provider values, not the 27
    # the removed key still present (row changed since listing) -> no alias, no second row on a key
    changed = dict(rem, HS=26.0)
    out2 = AC.add_join_aliases(AC.apply_af_history_corrections(pd.DataFrame([kept, changed]), corr), corr)
    assert len(out2) == 2 and out2["HS"].tolist() == [16.0, 26.0]


def test_loader_joins_the_kept_copy_onto_a_served_row_dated_like_the_removed_copy(tmp_path, monkeypatch):
    import src.data_loader as DL
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=16.0, AS=14.0)
    rem = _row("2026-10-25", "Club Leon", "Toluca", HS=27.0, AS=9.0)
    af = pd.DataFrame([kept, rem])
    p = tmp_path / "af_history.parquet"
    af.to_parquet(p, index=False)
    corr_csv = tmp_path / "corr.csv"
    _corr(rem, "2026-10-24", "Leon", "Toluca").to_csv(corr_csv, index=False)
    monkeypatch.setattr(DL, "AF_HISTORY_PATH", p)
    monkeypatch.setattr(AC, "AF_DUP_REMOVED", corr_csv)
    monkeypatch.setattr(DL, "_enrich_ht_scores", lambda d: d)
    monkeypatch.setattr(DL, "_finish_af_enrichment", lambda d, wb, recompute_sot_ratio: d)
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    monkeypatch.setattr(DL.config, "API_KEY", "", raising=False)
    served = pd.DataFrame([{"league": LG, "date": pd.Timestamp("2026-10-25"), "home_team": "Club Leon",
                            "away_team": "Toluca", "home_goals": 1.0, "away_goals": 1.0,
                            "home_shots": NAN, "away_shots": NAN}])
    out = DL._enrich_with_api_shots(served)
    assert out["home_shots"].iloc[0] == 16.0 and out["away_shots"].iloc[0] == 14.0
