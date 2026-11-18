"""Fix round 8 (nfgap, owner 2026-11-04): new-format corners / fouls gap.

Data truth check 3 put the live board AMBER on all four new-format tracks (home/away corners_pg_roll
and fouls_pg_roll 32-33% missing). Root cause: af_history.parquet rows banked by the loader's
write-back carry shots / SoT / xG / inside-box but never Corner Kicks or Fouls. The --fullstats sidecar
(output/collect/af_history_fullstats_fill.parquet) re-reads the cached /fixtures/statistics answer;
src/data_loader._enrich_af_fullstats_fill fills the history from it - nan_only, league-keyed, clubs
proven by team_names.resolve or the loader's own identity map, ambiguity refused.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_fix8_nfgap.py

Synthetic sidecars in tmp_path only; no network, nothing under output/ is written.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from src import data_loader as DL  # noqa: E402
from src import feature_engineering as FE  # noqa: E402

LG = "USA MLS"
OTHER = "Japan J-League"
STATS = ("home_corners", "away_corners", "home_fouls", "away_fouls",
         "home_shots", "away_shots", "home_sot", "away_sot")


def _m(date, h, a, league=LG, **kw):
    r = {"date": pd.Timestamp(date), "league": league, "home_team": h, "away_team": a,
         "home_goals": 1.0, "away_goals": 1.0, **{c: np.nan for c in STATS},
         "home_sot_ratio": np.nan, "away_sot_ratio": np.nan}
    r.update(kw)
    return r


def _side(root, rows):
    """A sidecar in the collector's own schema (scripts/collect_af_history_gap.FULL_COLS)."""
    import collect_af_history_gap as G
    p = root / DL.AF_FULLSTATS_FILL
    p.parent.mkdir(parents=True, exist_ok=True)
    full = []
    for r in rows:
        x = {c: np.nan for c in G.FULL_COLS}
        x.update(r)
        full.append(x)
    pd.DataFrame(full, columns=G.FULL_COLS).to_parquet(p, index=False)
    return p


def _hist():
    return pd.DataFrame([
        _m("2026-10-04", "Inter Miami", "Charlotte FC"),
        _m("2026-10-04", "Inter Miami", "Charlotte FC", league=OTHER),           # same key, other league
        _m("2026-10-11", "Charlotte FC", "Inter Miami", home_corners=6.0, home_shots=12.0,
           home_sot=5.0, home_sot_ratio=5 / 12),                                  # stored values
    ])


def test_collector_schema_carries_every_column_the_loader_reads():
    import collect_af_history_gap as G
    assert set(DL._AF_FULLSTATS_COLS) <= set(G.FULL_COLS)
    assert {"league", "date", "home_team", "away_team"} <= set(G.FULL_COLS)
    # the provider names behind them
    for typ, stem in (("Corner Kicks", "corners"), ("Fouls", "fouls"),
                      ("Total Shots", "shots"), ("Shots on Goal", "sot")):
        assert G.FULL_STATS[typ] == stem


def test_fills_missing_corners_fouls_shots_nan_only_and_league_keyed(tmp_path):
    _side(tmp_path, [
        {"league": LG, "date": "2026-10-04", "home_team": "Inter Miami", "away_team": "Charlotte FC",
         "home_corners": 7.0, "away_corners": 3.0, "home_fouls": 10.0, "away_fouls": 14.0,
         "home_shots": 15.0, "away_shots": 9.0, "home_sot": 6.0, "away_sot": 2.0, "fixture_id": 1},
        {"league": LG, "date": "2026-10-11", "home_team": "Charlotte FC", "away_team": "Inter Miami",
         "home_corners": 99.0, "away_corners": 2.0, "home_shots": 99.0, "home_sot": 99.0,
         "fixture_id": 2}])
    out = DL._enrich_af_fullstats_fill(_hist(), root=tmp_path)
    r0 = out.iloc[0]
    assert (r0.home_corners, r0.away_corners, r0.home_fouls, r0.away_fouls) == (7.0, 3.0, 10.0, 14.0)
    assert (r0.home_shots, r0.away_shots, r0.home_sot, r0.away_sot) == (15.0, 9.0, 6.0, 2.0)
    assert out.iloc[1][list(STATS)].isna().all()                  # other league never touched
    r2 = out.iloc[2]
    assert r2.home_corners == 6.0 and r2.home_shots == 12.0      # a stored value is never overwritten
    assert r2.away_corners == 2.0                                 # only the gap is filled
    assert r2.home_sot_ratio == pytest.approx(5 / 12)             # untouched row keeps its ratio
    assert r0.home_sot_ratio == pytest.approx(6 / 15) and r0.away_sot_ratio == pytest.approx(2 / 9)
    att = out.attrs["af_fullstats_fill"]
    assert att["filled_cells"]["home_corners"] == 1 and att["filled_cells"]["away_corners"] == 2
    assert len(out) == 3 and list(out.columns) == list(_hist().columns)


def test_provider_missing_value_stays_missing_never_zero(tmp_path):
    _side(tmp_path, [{"league": LG, "date": "2026-10-04", "home_team": "Inter Miami",
                      "away_team": "Charlotte FC", "home_corners": 4.0, "away_corners": None,
                      "home_fouls": "None", "fixture_id": 1}])
    out = DL._enrich_af_fullstats_fill(_hist(), root=tmp_path)
    assert out.loc[0, "home_corners"] == 4.0
    assert pd.isna(out.loc[0, "away_corners"]) and pd.isna(out.loc[0, "home_fouls"])


def test_source_spelling_is_resolved_and_an_unprovable_club_is_refused(tmp_path):
    _side(tmp_path, [
        {"league": LG, "date": "2026-10-04", "home_team": "Inter Miami CF",       # resolve() proves it
         "away_team": "Charlotte", "home_corners": 5.0, "fixture_id": 1},
        {"league": LG, "date": "2026-10-11", "home_team": "Some Other Club",       # no such club
         "away_team": "Inter Miami", "away_corners": 8.0, "fixture_id": 2}])
    out = DL._enrich_af_fullstats_fill(_hist(), root=tmp_path)
    assert out.loc[0, "home_corners"] == 5.0
    assert pd.isna(out.loc[2, "away_corners"])


def test_conflicting_sidecar_rows_for_one_match_are_refused(tmp_path):
    _side(tmp_path, [
        {"league": LG, "date": "2026-10-04", "home_team": "Inter Miami", "away_team": "Charlotte FC",
         "home_corners": 5.0, "home_fouls": 9.0, "fixture_id": 1},
        {"league": LG, "date": "2026-10-04", "home_team": "Inter Miami CF", "away_team": "Charlotte FC",
         "home_corners": 6.0, "home_fouls": 9.0, "fixture_id": 3}])
    out = DL._enrich_af_fullstats_fill(_hist(), root=tmp_path)
    assert pd.isna(out.loc[0, "home_corners"])                    # 5 vs 6: ambiguous -> missing
    assert out.loc[0, "home_fouls"] == 9.0                        # agreeing cell is kept
    assert out.attrs["af_fullstats_fill"]["ambiguous_key_groups"] == 1


def test_identity_alias_places_a_row_keyed_on_a_renamed_spelling(tmp_path, monkeypatch):
    monkeypatch.setattr(DL, "IDENTITY_ALIASES", {LG: {"Herons": "Inter Miami"}})
    _side(tmp_path, [{"league": LG, "date": "2026-10-04", "home_team": "Herons",
                      "away_team": "Charlotte FC", "home_corners": 5.0, "fixture_id": 1}])
    out = DL._enrich_af_fullstats_fill(_hist(), root=tmp_path)
    assert out.loc[0, "home_corners"] == 5.0
    # the same alias in another league's map never places it
    monkeypatch.setattr(DL, "IDENTITY_ALIASES", {OTHER: {"Herons": "Inter Miami"}})
    out = DL._enrich_af_fullstats_fill(_hist(), root=tmp_path)
    assert pd.isna(out.loc[0, "home_corners"])


def test_absent_or_keyless_sidecar_leaves_history_unchanged(tmp_path):
    h = _hist()
    pd.testing.assert_frame_equal(DL._enrich_af_fullstats_fill(h.copy(), root=tmp_path), h)
    p = tmp_path / DL.AF_FULLSTATS_FILL
    p.parent.mkdir(parents=True)
    pd.DataFrame([{"date": "2026-10-04", "home_team": "Inter Miami", "away_team": "Charlotte FC",
                   "home_corners": 5.0}]).to_parquet(p)                         # no league column
    pd.testing.assert_frame_equal(DL._enrich_af_fullstats_fill(h.copy(), root=tmp_path), h)


def test_loader_calls_the_fullstats_fill_on_both_exits_after_the_stats_fill():
    # fix round 9 (owner 2026-11-04): both exits go through _finish_af_enrichment, and the bank
    # step that wrote into af_history.parquet is gone (tests/test_fix9_bank.py)
    src = inspect.getsource(DL._enrich_with_api_shots)
    assert src.count("return _finish_af_enrichment(df, wb_blank") == 2
    fin = inspect.getsource(DL._finish_af_enrichment)
    assert fin.index("_enrich_af_stats_fill(df)") < fin.index("_enrich_af_fullstats_fill(df)")
    assert "to_parquet" not in src and "to_parquet" not in fin


def test_no_key_path_fills_from_the_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    monkeypatch.setattr(config, "API_KEY", "")
    seen = []
    real = DL._enrich_af_fullstats_fill
    monkeypatch.setattr(DL, "_enrich_af_fullstats_fill",
                        lambda df, root=None: seen.append(len(df)) or real(df, root=tmp_path))
    monkeypatch.setattr(DL, "_enrich_ht_scores", lambda df, root=None: df)
    monkeypatch.setattr(DL, "_enrich_af_stats_fill", lambda df, root=None: df)
    # 2030 dates: no committed af_history row can share the key (that merge is read-only here)
    _side(tmp_path, [{"league": LG, "date": "2030-10-04", "home_team": "Inter Miami",
                      "away_team": "Charlotte FC", "home_corners": 7.0, "fixture_id": 1}])
    h = pd.DataFrame([_m("2030-10-04", "Inter Miami", "Charlotte FC"),
                      _m("2030-10-11", "Charlotte FC", "Inter Miami")])
    out = DL._enrich_with_api_shots(h)
    assert seen == [2]
    assert out.loc[0, "home_corners"] == 7.0 and pd.isna(out.loc[1, "home_corners"])


def test_filled_corners_reach_the_served_and_trained_form_features(tmp_path):
    """No new feature: the fill feeds the existing corners_pg_roll / fouls_pg_roll window."""
    rows = [_m(f"2026-09-{d:02d}", "Inter Miami", f"Opp{d}") for d in (6, 13, 20, 27)]
    h = pd.DataFrame(rows)
    h["over25"] = 0.0
    _side(tmp_path, [{"league": LG, "date": f"2026-09-{d:02d}", "home_team": "Inter Miami",
                      "away_team": f"Opp{d}", "home_corners": float(d % 7), "home_fouls": 10.0,
                      "fixture_id": d} for d in (20, 27)])
    before = FE._latest_form(h, 5)
    assert pd.isna(before.loc[(LG, "Inter Miami"), "corners"])
    out = DL._enrich_af_fullstats_fill(h, root=tmp_path)
    after = FE._latest_form(out, 5)
    assert after.loc[(LG, "Inter Miami"), "corners"] == pytest.approx((20 % 7 + 27 % 7) / 2)
    assert after.loc[(LG, "Inter Miami"), "fouls"] == 10.0
    # only the existing stat columns are filled: no possession / blocked / passes column appears
    # (home_possession_last5 is serving-only and is not fed from the sidecar - owner question)
    assert list(out.columns) == list(h.columns)
    assert set(DL._AF_FULLSTATS_COLS.values()) == set(STATS)
