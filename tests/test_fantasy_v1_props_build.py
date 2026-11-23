"""Fantasy V1 scores the props payloads on the props build (owner 2026-11-23 "Fix V1, then D4"):
fantasy.v1_model_inputs = player_model.feature_engineering.build_features on the props source frame,
never the stored player_history feature columns. The full-frame proof (0 differing cells) is
scripts/certify_real_data.py fantasy_v1_parity; these tests pin the wiring."""
import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import fantasy as FA  # noqa: E402


def _projection_src() -> str:
    tree = ast.parse((ROOT / "player_model" / "fantasy.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_fantasy_projections")
    return ast.get_source_segment((ROOT / "player_model" / "fantasy.py").read_text(encoding="utf-8"), fn)


def test_v1_scores_the_props_build_not_the_stored_columns():
    src = _projection_src()
    assert "v1_model_inputs(pl, source)" in src
    assert "predict_proba(pl, payload)" not in src          # the legacy stored-column scoring is gone
    assert "predict_proba(X[ok], payload)" in src
    fn = (ROOT / "player_model" / "fantasy.py").read_text(encoding="utf-8")
    assert "build_upcoming_features(rows, _v1_context(raw, pl, rows))" in fn    # the props serving build


def test_v1_model_inputs_is_the_serving_build_on_next_fixtures(monkeypatch):
    import player_model.feature_engineering as FE
    pl = pd.DataFrame({"player_id": [1, 2], "team": ["Arsenal", "Leeds"]}, index=[7, 9])
    seen = {}

    def fake(rows, hist, n=None):
        seen["rows"] = rows
        return pd.DataFrame({"player_id": [1], "fixture_id": [500], "f": [0.5]})
    monkeypatch.setattr(FE, "build_upcoming_features", fake)
    monkeypatch.setattr(FA, "_v1_context", lambda raw, pl, rows=None: raw)
    rows = [{"player_id": 1, "fixture_id": 500, "date": "2026-11-29"}]
    X = FA.v1_model_inputs(pl, pd.DataFrame({"x": [1]}), rows)
    assert seen["rows"] == rows
    assert list(X.index) == [7, 9] and X["_v1_built"].tolist() == [True, False]
    assert X.loc[7, "f"] == 0.5 and pd.isna(X.loc[9, "f"])   # no row -> no input, never a stored value


def test_next_fixture_rows_take_each_clubs_first_unplayed_fixture():
    fx = [{"fixture": {"id": 2, "date": "2026-12-03T20:00:00+00:00", "referee": None},
           "teams": {"home": {"name": "Leeds"}, "away": {"name": "Arsenal"}}},
          {"fixture": {"id": 1, "date": "2026-11-29T15:00:00+00:00", "referee": "M. Oliver"},
           "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}}}]
    pl = pd.DataFrame({"player_id": [1, 2, 3], "team": ["Arsenal", "Leeds", "Wigan"], "player_name": ["a", "b", "c"]})
    rows = FA.next_fixture_rows(pl, fx)
    by = {r["player_id"]: r for r in rows}
    assert set(by) == {1, 2}                                  # a club with no listed fixture -> no row
    assert by[1]["fixture_id"] == 1 and by[1]["is_home"] == 1.0 and by[1]["date"] == "2026-11-29"
    assert by[2]["fixture_id"] == 2 and by[2]["is_home"] == 1.0
    assert all("goals" not in r for r in rows)                # identity + context only, no result


def test_context_keeps_league_club_player_and_referee_rows():
    raw = pd.DataFrame({
        "league": ["Premier League", "FA Cup", "La Liga", "La Liga", "La Liga"],
        "home_team": ["Arsenal", "Arsenal", "Real Madrid", "Sevilla", "Betis"],
        "away_team": ["Chelsea", "Wigan", "Barcelona", "Valencia", "Celta"],
        "player_id": [1, 1, 99, 5, 6],
        "referee": ["M. Oliver", "A. Taylor", "X", "M. Oliver", "Y"]})
    if FA.FANTASY_LEAGUE is None:
        return
    pl = pd.DataFrame({"fixture_id": [1], "player_id": [99]})
    kept = FA._v1_context(raw, pl)
    assert set(kept.index) >= {0, 1, 2}                      # league row, club's cup row, V1 player's row
    assert 4 not in kept.index                               # unrelated club, unrelated referee
