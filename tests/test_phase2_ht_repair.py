"""Phase 2 Part 3 (P0 half-time history repair), 2026-11-03.

The HT sidecar (output/collect/fd_history_ht_fill.parquet) is keyed on fd_history's OWN club
spellings. load_all_matches runs one_club_one_match BEFORE the HT step, and that pipeline renames a
club by same-day evidence ('Atletico Paranaense' -> 'Athletico-PR') that team_names.resolve cannot
spell-match. The HT step then could not place the sidecar row (measured on the committed files at
data-truth-100 e07a4591: 2,303 of 33,259 sidecar rows skipped as 'not provably a club of the
league'). The fix lets _resolve_onto_history use the identity map the loader itself applied to the
history - league-scoped, only onto a club that is in that league's pool, refused when ambiguous.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_phase2_ht_repair.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loader as DL

LG = "Brazil Serie A"
OLD, NEW = "Atletico Paranaense", "Athletico-PR"


def _r(d, h, a, hg, ag, hth=np.nan, hta=np.nan, lg=LG):
    return {"date": pd.Timestamp(d), "league": lg, "season": None, "home_team": h, "away_team": a,
            "home_goals": float(hg), "away_goals": float(ag), "ht_home_goals": hth, "ht_away_goals": hta,
            "row_source": "fd_cache"}


def _history():
    """Three same-day same-score copies under both spellings (the evidence), a later match under
    the NEW spelling (newest wins), and one match only under the OLD spelling without HT."""
    rows = []
    for d, opp in (("2024-04-14", "Santos"), ("2024-05-05", "Gremio"), ("2024-06-02", "Bahia")):
        rows += [_r(d, OLD, opp, 1, 0, 1.0, 0.0), _r(d, NEW, opp, 1, 0, 1.0, 0.0)]
    rows.append(_r("2024-07-07", "Fortaleza", OLD, 2, 2))         # target: no HT, old spelling only
    rows.append(_r("2025-03-30", NEW, "Santos", 0, 0, 0.0, 0.0))   # newest spelling = canonical
    return pd.DataFrame(rows)


def _write_fill(root, rows):
    p = root / "output" / "collect" / "fd_history_ht_fill.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "HTHG", "HTAG", "fixture_id"]) \
        .to_parquet(p, index=False)


def test_identity_pipeline_renames_the_old_spelling():
    out = DL.one_club_one_match(_history())
    names = set(out["home_team"]) | set(out["away_team"])
    assert NEW in names and OLD not in names
    assert DL.IDENTITY_ALIASES.get(LG, {}).get(OLD) == NEW


def test_ht_fill_keyed_on_the_history_spelling_reaches_the_renamed_club(tmp_path):
    df = DL.one_club_one_match(_history())
    _write_fill(tmp_path, [(LG, "2024-07-07", "Fortaleza", OLD, 1, 0, 99)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    row = out[(out["date"] == "2024-07-07") & (out["home_team"] == "Fortaleza")]
    assert len(row) == 1 and row["away_team"].iloc[0] == NEW
    assert row["ht_home_goals"].iloc[0] == 1.0 and row["ht_away_goals"].iloc[0] == 0.0
    assert out.attrs["ht_coverage"]["filled_rows"]["fd_history_ht_fill"] == 1


def test_identity_alias_is_league_scoped_and_needs_the_club_in_the_pool(tmp_path):
    df = DL.one_club_one_match(_history())
    # same spelling filed under another league: that league has no 'Athletico-PR' -> never placed
    _write_fill(tmp_path, [("Japan J-League", "2024-07-07", "Fortaleza", OLD, 1, 0, 99)])
    out = DL._enrich_ht_scores(df, root=tmp_path)
    assert out["ht_home_goals"].isna().sum() == df["ht_home_goals"].isna().sum()


def test_resolve_onto_history_without_aliases_is_unchanged():
    df = DL.one_club_one_match(_history())
    right = pd.DataFrame({"league": [LG], "home_team": ["Fortaleza"], "away_team": [OLD]})
    assert DL._resolve_onto_history(right, df, "home_team", "away_team", "t").empty
    got = DL._resolve_onto_history(right, df, "home_team", "away_team", "t", aliases={LG: {OLD: NEW}})
    assert got["_a"].tolist() == [NEW]


def test_ambiguous_identity_alias_is_refused():
    df = DL.one_club_one_match(_history())
    right = pd.DataFrame({"league": [LG], "home_team": ["Fortaleza"], "away_team": [OLD]})
    # one spelling claimed by two different clubs of the pool -> not placed
    amb = {LG: {OLD: NEW, "Atletico-Paranaense": "Santos"}}
    got = DL._resolve_onto_history(right, df, "home_team", "away_team", "t", aliases=amb)
    assert got.empty


def test_rematch_drops_a_stored_answer_that_is_now_refused(tmp_path, monkeypatch):
    """collect_ht_scores.run() concatenated the stored sidecar with the new match result and kept an
    old row unless it was re-matched now - so a row that a later run REFUSES (a second history row of
    another date now claims the same fixture) kept its stale score. A row re-evaluated this run must
    carry this run's answer, including 'no answer'."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(DL.__file__).resolve().parents[1] / "scripts"))
    import collect_ht_scores as H
    for name in ("SCORES", "FILL", "AUDIT", "JOURNAL", "FT_JOURNAL", "FT_MISMATCH"):
        monkeypatch.setattr(H, name, tmp_path / getattr(H, name).name)
    fx = {"fixture": {"id": 11, "date": "2026-08-01T19:30:00+00:00", "status": {"short": "FT"}},
          "teams": {"home": {"id": 1, "name": "Columbus Crew"}, "away": {"id": 2, "name": "Inter Miami"}},
          "score": {"halftime": {"home": 1, "away": 0}}}

    class _C:
        def get(self, endpoint, params):
            return {"response": [fx]}

        def flush(self):
            pass

    row = {"league": "USA MLS", "home_team": "Columbus Crew", "away_team": "Inter Miami",
           "ht_home_goals": np.nan, "ht_away_goals": np.nan}
    H.run(client=_C(), history=pd.DataFrame([dict(row, date="2026-08-01")]))
    assert len(pd.read_parquet(H.FILL)) == 1
    # the history now also holds the same pairing one day later: fixture 11 is claimed twice
    H.run(match_only=True, history=pd.DataFrame([dict(row, date="2026-08-01"), dict(row, date="2026-08-02")]))
    assert pd.read_parquet(H.FILL).empty


def test_rematch_keeps_rows_not_re_evaluated(tmp_path, monkeypatch):
    """A stored row whose history row was NOT re-evaluated this run (e.g. it has since gained its own
    HT score) is left as stored - the sidecar is not shrunk by rows outside this run's scope."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(DL.__file__).resolve().parents[1] / "scripts"))
    import collect_ht_scores as H
    for name in ("SCORES", "FILL", "AUDIT", "JOURNAL", "FT_JOURNAL", "FT_MISMATCH"):
        monkeypatch.setattr(H, name, tmp_path / getattr(H, name).name)
    fx = {"fixture": {"id": 11, "date": "2026-08-01T19:30:00+00:00", "status": {"short": "FT"}},
          "teams": {"home": {"id": 1, "name": "Columbus Crew"}, "away": {"id": 2, "name": "Inter Miami"}},
          "score": {"halftime": {"home": 1, "away": 0}}}

    class _C:
        def get(self, endpoint, params):
            return {"response": [fx]}

        def flush(self):
            pass

    row = {"league": "USA MLS", "date": "2026-08-01", "home_team": "Columbus Crew", "away_team": "Inter Miami"}
    H.run(client=_C(), history=pd.DataFrame([dict(row, ht_home_goals=np.nan, ht_away_goals=np.nan)]))
    H.run(match_only=True, history=pd.DataFrame([dict(row, ht_home_goals=1.0, ht_away_goals=0.0)]))
    assert len(pd.read_parquet(H.FILL)) == 1


def test_committed_fill_integrity():
    """The committed sidecar: unique key, no null/negative HT, HT never above the history's FT, no
    fixture id claimed by history rows of two different dates, and every fixture id a stored
    provider fixture of the same league. Fails if a later collection breaks any of these."""
    from pathlib import Path
    root = Path(DL.__file__).resolve().parents[1]
    p = root / "output" / "collect" / "fd_history_ht_fill.parquet"
    s = root / "output" / "collect" / "af_ht_scores.parquet"
    if not p.exists() or not s.exists():
        import pytest
        pytest.skip("HT sidecar not present in this checkout")
    f = pd.read_parquet(p)
    f["date"] = pd.to_datetime(f["date"])
    key = ["league", "date", "home_team", "away_team"]
    assert not f.duplicated(key).any()
    assert f[["HTHG", "HTAG"]].notna().all().all() and (f[["HTHG", "HTAG"]] >= 0).all().all()
    assert (f.groupby("fixture_id")["date"].nunique() == 1).all()
    sc = pd.read_parquet(s, columns=["fixture_id", "league", "ht_home", "ht_away"])
    assert not sc["fixture_id"].duplicated().any()
    j = f.merge(sc.rename(columns={"league": "provider_league"}), on="fixture_id", how="left")
    assert j["provider_league"].notna().all()                       # every id is a stored fixture
    assert (j["provider_league"] == j["league"]).all()              # never another league's fixture
    assert (j["HTHG"] == j["ht_home"]).all() and (j["HTAG"] == j["ht_away"]).all()
    h = pd.read_parquet(root / "output" / "fd_history.parquet",
                        columns=key + ["home_goals", "away_goals"])
    h["date"] = pd.to_datetime(h["date"])
    m = f.merge(h.drop_duplicates(key), on=key, how="left")
    assert m["home_goals"].notna().all()
    assert ((m["HTHG"] <= m["home_goals"]) & (m["HTAG"] <= m["away_goals"])).all()
