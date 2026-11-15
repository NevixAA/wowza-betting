"""fix15/dates (owner decision 2026-11-07, "use the provider's date"):

1. a SINGLE football-data history row dated one day off API-Football's own date for the same proven
   fixture is served on the provider's date (output/corrections/history_date_corrections.csv, applied
   by the loader before any join or feature; fd_history.parquet is never rewritten);
2. af_history copies of one provider fixture held on the provider's date under two spellings are
   removed through the af corrections list (found_by af_same_day_spelling), same proof as fix14.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src import af_history_corrections as AC  # noqa: E402
from src import history_corrections as HC  # noqa: E402

LG = "USA MLS"
NAN = np.nan


def _dates(rows):
    return pd.DataFrame([{"league": LG, "history_date": d, "home_team": h, "away_team": a,
                          "home_goals": hg, "away_goals": ag, "provider_date": p, "fixture_id": f}
                         for d, h, a, hg, ag, p, f in rows])


def _hist(rows, source="fd_cache"):
    return pd.DataFrame([{"league": LG, "date": pd.Timestamp(d), "home_team": h, "away_team": a,
                          "home_goals": hg, "away_goals": ag, "row_source": source}
                         for d, h, a, hg, ag in rows])


# ---- 1. the history date list -------------------------------------------------------------------

def test_listed_history_row_is_served_on_the_provider_date():
    df = _hist([("2026-10-26", "Austin", "Dallas", 2.0, 1.0), ("2026-10-26", "Miami", "Orlando", 0.0, 0.0)])
    out = HC.apply_date_corrections(df, _dates([("2026-10-26", "Austin", "Dallas", 2, 1, "2026-10-25", 11)]))
    assert out["date"].tolist() == [pd.Timestamp("2026-10-25"), pd.Timestamp("2026-10-26")]
    assert df["date"].iloc[0] == pd.Timestamp("2026-10-26")            # the input is not rewritten
    assert out.attrs["history_date_corrections"]["rows_redated"] == 1
    assert out.attrs["history_date_corrections"]["per_league"][LG]["rows_redated"] == 1


def test_row_with_another_score_or_from_another_source_is_left_alone():
    d = _dates([("2026-10-26", "Austin", "Dallas", 2, 1, "2026-10-25", 11)])
    changed = HC.apply_date_corrections(_hist([("2026-10-26", "Austin", "Dallas", 3.0, 1.0)]), d)
    assert changed["date"].iloc[0] == pd.Timestamp("2026-10-26")
    assert changed.attrs["history_date_corrections"]["listed_rows_changed_since_listing"] == 1
    af_row = _hist([("2026-10-26", "Austin", "Dallas", 2.0, 1.0)], source="api_football")
    assert HC.apply_date_corrections(af_row, d) is af_row


def test_absent_or_unreadable_date_list_changes_nothing(tmp_path):
    df = _hist([("2026-10-26", "Austin", "Dallas", 2.0, 1.0)])
    assert HC.apply_date_corrections(df, HC.read_date_corrections(tmp_path / "none.csv")) is df
    bad = tmp_path / "bad.csv"
    bad.write_text("league,date\nx,y\n", encoding="utf-8")
    assert HC.read_date_corrections(bad).empty                         # lacks columns: NOT applied


def test_sidecar_on_the_history_key_moves_with_the_redated_row():
    d = _dates([("2026-10-26", "Austin", "Dallas", 2, 1, "2026-10-25", 11)])
    ht = pd.DataFrame([{"league": LG, "date": pd.Timestamp("2026-10-26"), "home_team": "Austin",
                        "away_team": "Dallas", "HTHG": 1, "HTAG": 0},
                       {"league": LG, "date": pd.Timestamp("2026-10-20"), "home_team": "Miami",
                        "away_team": "Orlando", "HTHG": 0, "HTAG": 0}])
    out = HC.rekey_dated_sidecar(ht, d)
    assert out["date"].tolist() == [pd.Timestamp("2026-10-25"), pd.Timestamp("2026-10-20")]
    held = pd.concat([ht, ht.iloc[[0]].assign(date=pd.Timestamp("2026-10-25"))], ignore_index=True)
    out2 = HC.rekey_dated_sidecar(held, d)
    assert out2.attrs["dated_sidecar_rekey"]["left_new_key_held"] == 1
    assert HC.redated_fixture_ids(d) == {11: "2026-10-25"}


def test_loader_applies_the_date_list_after_the_removals_and_before_identity(monkeypatch):
    import src.data_loader as DL
    df = _hist([("2026-10-26", "Austin", "Dallas", 2.0, 1.0)])
    d = _dates([("2026-10-26", "Austin", "Dallas", 2, 1, "2026-10-25", 11)])
    out = DL.identity_before_one_match(df, corrections=HC.read_corrections(Path("none_a.csv"),
                                                                           Path("none_b.csv"),
                                                                           Path("none_c.csv")),
                                       scores=pd.DataFrame(columns=HC.KEY + HC.SCORE_COLS), dates=d)
    assert out["date"].iloc[0] == pd.Timestamp("2026-10-25")
    # the builder measures the loader's pairs with an empty list: no re-dating
    out2 = DL.identity_before_one_match(df, corrections=HC.read_corrections(Path("none_a.csv"),
                                                                            Path("none_b.csv"),
                                                                            Path("none_c.csv")),
                                        scores=pd.DataFrame(columns=HC.KEY + HC.SCORE_COLS),
                                        dates=pd.DataFrame())
    assert out2["date"].iloc[0] == pd.Timestamp("2026-10-26")
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8")
    assert src.count("_apply_date_corrections(out") == 2               # CI path (identity) + local path
    assert "rekey_dated_sidecar(ht" in src


def _pool(rows):
    return pd.DataFrame([{"fixture_id": f, "league": LG, "league_id": 253, "season": "2026", "date": d,
                          "kickoff": k, "home_id": 1, "home_name": h, "away_id": 2, "away_name": a,
                          "ht_home": None, "ht_away": None, "status": "FT", "ft_home": fh, "ft_away": fa,
                          "ft_source": "x", "et_home": None, "et_away": None, "pen_home": None,
                          "pen_away": None}
                         for f, d, k, h, a, fh, fa in rows])


def test_builder_proves_a_single_copy_one_day_after_the_provider_date():
    import build_history_corrections as B
    pool = _pool([(11, "2026-10-25", "2026-10-25T23:30:00+00:00", "Austin", "FC Dallas", 2, 1)])
    hist = _hist([("2026-10-26", "Austin", "FC Dallas", 2.0, 1.0)]).drop(columns="row_source")
    out, ref, st = B.date_corrections(hist, pool, raw=hist)
    assert ref.empty and len(out) == 1
    r = out.iloc[0]
    assert (r["history_date"], r["provider_date"], r["fixture_id"], r["day_offset"]) == \
        ("2026-10-26", "2026-10-25", 11, 1)
    assert r["provider_kickoff_utc"] == "2026-10-25T23:30:00+00:00" and r["found_by"] == B.DATE_FOUND_BY
    assert st["rows_redated_by_league"] == {LG: 1}


@pytest.mark.parametrize("extra, hist_ft, reason", [
    ([("2026-10-25", "Austin", "FC Dallas", 2.0, 1.0)], (2.0, 1.0), "fixture_claimed_by_another_history_row"),
    ([], (3.0, 1.0), "score_differs_from_provider"),
])
def test_builder_refuses_what_is_not_proven(extra, hist_ft, reason):
    import build_history_corrections as B
    pool = _pool([(11, "2026-10-25", "2026-10-25T23:30:00+00:00", "Austin", "FC Dallas", 2, 1)])
    hist = _hist([("2026-10-26", "Austin", "FC Dallas", *hist_ft)] + extra).drop(columns="row_source")
    out, ref, _ = B.date_corrections(hist, pool, raw=hist)
    assert out.empty and reason in set(ref["reason"])


def test_builder_refuses_when_the_stored_history_holds_the_clubs_on_the_provider_date():
    import build_history_corrections as B
    pool = _pool([(11, "2026-10-25", "2026-10-25T23:30:00+00:00", "Austin", "FC Dallas", 2, 1)])
    served = _hist([("2026-10-26", "Austin", "FC Dallas", 2.0, 1.0)]).drop(columns="row_source")
    raw = _hist([("2026-10-26", "Austin", "FC Dallas", 2.0, 1.0),
                 ("2026-10-25", "Austin", "FC Dallas", 0.0, 0.0)]).drop(columns="row_source")
    out, ref, _ = B.date_corrections(served, pool, raw=raw)
    assert out.empty and set(ref["reason"]) == {"same_league_and_clubs_on_provider_date_in_stored_history"}


def test_collection_job_rebuilds_and_commits_the_date_list():
    import build_history_corrections as B
    wf = (ROOT / ".github/workflows/data_truth_collect.yml").read_text(encoding="utf-8")
    assert "python scripts/build_history_corrections.py --fetch" in wf
    assert "git add -f output/corrections/history_*.csv" in wf
    assert B.DATE_OUT.name.startswith("history_") and B.DATE_UNPROVEN.name.startswith("history_")
    assert B.DATE_OUT == HC.DATE_CORR


@pytest.mark.skipif(not HC.DATE_CORR.exists(), reason="committed list absent")
def test_committed_date_list_is_proven_one_day_and_unique():
    d = pd.read_csv(HC.DATE_CORR)
    assert set(HC.DATE_COLS) <= set(d.columns)
    assert len(d) > 0                                   # a header-only list would serve nothing
    for c in ("provider_date", "provider_kickoff_utc", "fixture_id", "found_by"):
        assert d[c].notna().all(), c
    gap = (pd.to_datetime(d["history_date"]) - pd.to_datetime(d["provider_date"])).dt.days
    assert gap.abs().eq(1).all() and gap.eq(d["day_offset"]).all()
    assert not d.duplicated(["league", "history_date", "home_team", "away_team"]).any()
    assert not d["fixture_id"].duplicated().any()
    assert (pd.to_datetime(d["provider_kickoff_utc"], utc=True).dt.strftime("%Y-%m-%d")
            == d["provider_date"]).all()


# ---- 2. af_history same-day spelling duplicates -------------------------------------------------

def _row(date, home, away, **kw):
    base = {"league": "Mexico Liga MX", "date": pd.Timestamp(date), "home_team": home, "away_team": away,
            "season": "2026", "FTHG": NAN, "FTAG": NAN, "FTR": None,
            **{c: NAN for c in AC.STAT_COLS}, "fixture_id": NAN, "home_team_id": NAN, "away_team_id": NAN}
    base.update(kw)
    return base


def _fixtures():
    fx = pd.DataFrame([{"fixture_id": 1551013, "league_id": 262, "season": "2026", "date": "2026-10-24",
                        "home_id": 2289, "home_name": "Leon", "away_id": 2281, "away_name": "Toluca"}])
    fx["_d"] = pd.to_datetime(fx["date"]).dt.normalize()
    prov = fx.assign(kickoff="2026-10-24T23:00:00+00:00", ft_home=1, ft_away=1, status="FT")
    return fx, prov


def test_build_removes_a_same_day_spelling_copy_and_keeps_the_provider_row(tmp_path):
    import build_af_history_corrections as B
    fx, prov = _fixtures()
    af = pd.DataFrame([_row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, FTR="D", HS=16.0,
                            fixture_id=1551013, home_team_id=2289, away_team_id=2281),
                       _row("2026-10-24", "Club Leon", "Toluca", HS=27.0)])
    rem, con, st = B.build(af, fx, B.provider_records(comp=pd.DataFrame(), ht=prov))
    assert con.empty and len(rem) == 1
    r = rem.iloc[0]
    assert (r["date"], r["home_team"], r["kept_date"], r["kept_home_team"], r["found_by"]) == \
        ("2026-10-24", "Club Leon", "2026-10-24", "Leon", B.FOUND_SAME_DAY)
    assert st["total"]["same_day_proven"] == 1 and st["total"]["same_day_rows_removed"] == 1
    rem.to_csv(tmp_path / "rem.csv", index=False)
    out = AC.apply_af_history_corrections(af, AC.read_af_corrections(tmp_path / "rem.csv"))
    assert len(out) == 1 and out["HS"].iloc[0] == 16.0             # the written-back 27 gives nothing


def test_build_two_dates_and_two_spellings_keep_one_copy():
    import build_af_history_corrections as B
    fx, prov = _fixtures()
    af = pd.DataFrame([_row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=16.0),
                       _row("2026-10-24", "Club Leon", "Toluca"),
                       _row("2026-10-25", "Club Leon", "Toluca", HS=27.0)])
    rem, con, _ = B.build(af, fx, B.provider_records(comp=pd.DataFrame(), ht=prov))
    assert con.empty and len(rem) == 2
    assert set(zip(rem["date"], rem["home_team"], rem["found_by"])) == {
        ("2026-10-25", "Club Leon", B.FOUND_TWO_DATES), ("2026-10-24", "Club Leon", B.FOUND_SAME_DAY)}
    assert set(rem["kept_home_team"]) == {"Leon"}


def test_build_lists_a_same_day_score_conflict_and_removes_nothing():
    import build_af_history_corrections as B
    fx, prov = _fixtures()
    af = pd.DataFrame([_row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0),
                       _row("2026-10-24", "Club Leon", "Toluca", FTHG=2.0, FTAG=1.0)])
    rem, con, st = B.build(af, fx, B.provider_records(comp=pd.DataFrame(), ht=prov))
    assert rem.empty and len(con) == 2
    assert set(con["conflict"]) == {"score_differs_between_copies"}
    assert set(con["found_by"]) == {B.FOUND_SAME_DAY}


def test_same_day_provider_copies_with_different_values_make_the_cell_missing():
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=16.0, AS=14.0)
    rem = _row("2026-10-24", "Club Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=27.0, AS=14.0, HC=3.0)
    corr = pd.DataFrame([{**{c: rem.get(c) for c in AC.KEY + AC.SIGNATURE}, "date": "2026-10-24",
                          "kept_date": "2026-10-24", "kept_home_team": "Leon", "kept_away_team": "Toluca",
                          "provider_fixture_id": 1551013}])
    out = AC.apply_af_history_corrections(pd.DataFrame([kept, rem]), corr)
    k = out.iloc[0]
    assert len(out) == 1 and np.isnan(k["HS"]) and k["AS"] == 14.0 and k["HC"] == 3.0


def test_join_alias_is_not_added_for_a_redated_served_row_or_a_same_day_copy():
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=16.0)
    rem = _row("2026-10-25", "Club Leon", "Toluca", HS=27.0)
    corr = pd.DataFrame([{**{c: rem.get(c) for c in AC.KEY + AC.SIGNATURE}, "date": "2026-10-25",
                          "kept_date": "2026-10-24", "kept_home_team": "Leon", "kept_away_team": "Toluca",
                          "provider_fixture_id": 1551013}])
    af = AC.apply_af_history_corrections(pd.DataFrame([kept, rem]), corr)
    assert len(AC.add_join_aliases(af, corr, redated={})) == 2                  # fix14 bridge
    out = AC.add_join_aliases(af, corr, redated={1551013: "2026-10-24"})
    assert len(out) == 1                                                       # redundant: not added
    same = corr.assign(date="2026-10-24", home_team="Club Leon")
    assert len(AC.add_join_aliases(af, same, redated={})) == 1


def test_loader_joins_the_kept_copy_onto_the_redated_served_row_without_an_alias(tmp_path, monkeypatch):
    import src.data_loader as DL
    kept = _row("2026-10-24", "Leon", "Toluca", FTHG=1.0, FTAG=1.0, HS=16.0, AS=14.0,
                fixture_id=1551013)
    rem = _row("2026-10-25", "Club Leon", "Toluca", HS=27.0, AS=9.0)
    p = tmp_path / "af_history.parquet"
    pd.DataFrame([kept, rem]).to_parquet(p, index=False)
    corr = pd.DataFrame([{**{c: rem.get(c) for c in AC.KEY + AC.SIGNATURE}, "date": "2026-10-25",
                          "kept_date": "2026-10-24", "kept_home_team": "Leon", "kept_away_team": "Toluca",
                          "provider_fixture_id": 1551013}])
    corr.to_csv(tmp_path / "corr.csv", index=False)
    pd.DataFrame([{"league": "Mexico Liga MX", "history_date": "2026-10-25", "home_team": "Leon",
                   "away_team": "Toluca", "home_goals": 1, "away_goals": 1, "provider_date": "2026-10-24",
                   "fixture_id": 1551013}]).to_csv(tmp_path / "dates.csv", index=False)
    monkeypatch.setattr(DL, "AF_HISTORY_PATH", p)
    monkeypatch.setattr(AC, "AF_DUP_REMOVED", tmp_path / "corr.csv")
    monkeypatch.setattr(HC, "DATE_CORR", tmp_path / "dates.csv")
    monkeypatch.setattr(DL, "_enrich_ht_scores", lambda d: d)
    monkeypatch.setattr(DL, "_finish_af_enrichment", lambda d, wb, recompute_sot_ratio: d)
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    monkeypatch.setattr(DL.config, "API_KEY", "", raising=False)
    served = HC.apply_date_corrections(pd.DataFrame([{
        "league": "Mexico Liga MX", "date": pd.Timestamp("2026-10-25"), "home_team": "Leon",
        "away_team": "Toluca", "home_goals": 1.0, "away_goals": 1.0, "home_shots": NAN, "away_shots": NAN}]))
    assert served["date"].iloc[0] == pd.Timestamp("2026-10-24")
    out = DL._enrich_with_api_shots(served)
    assert out["home_shots"].iloc[0] == 16.0 and out["away_shots"].iloc[0] == 14.0


def test_sidecar_row_of_a_redated_fixture_follows_it_by_fixture_id_only():
    side = pd.DataFrame([
        {"league": LG, "date": "2024-10-23", "home_team": "CF Montreal", "away_team": "Atlanta Utd",
         "HIB": 7.0, "fixture_id": 1312300},
        {"league": LG, "date": "2024-10-23", "home_team": "Austin", "away_team": "Dallas",
         "HIB": 5.0, "fixture_id": 77},                                    # another fixture: stays
        {"league": LG, "date": "2024-10-25", "home_team": "CF Montreal", "away_team": "Atlanta Utd",
         "HIB": 1.0, "fixture_id": 1312300}])                              # two days off: stays
    out = AC.follow_redated_fixtures(side, redated={1312300: "2024-10-22"})
    assert out["date"].tolist() == ["2024-10-22", "2024-10-23", "2024-10-25"]
    held = pd.concat([side.iloc[[0]], side.iloc[[0]].assign(date="2024-10-22")], ignore_index=True)
    out2 = AC.follow_redated_fixtures(held, redated={1312300: "2024-10-22"})
    assert out2.attrs["af_redated_follow"]["left_key_held"] == 1
    assert AC.follow_redated_fixtures(side, redated={}) is side
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8")
    assert src.count("follow_redated_fixtures(") == 3                    # af_history + both sidecars
