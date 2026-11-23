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


def test_v1_model_inputs_keys_rows_and_marks_unbuilt(monkeypatch):
    import player_model.feature_engineering as FE
    pl = pd.DataFrame({"fixture_id": [10, 11], "player_id": [1, 2]}, index=[7, 9])
    built = pd.DataFrame({"fixture_id": [10.0, 12.0], "player_id": [1.0, 3.0], "f": [0.5, 0.9]})
    monkeypatch.setattr(FE, "build_features", lambda df, copy=True: built.copy())
    monkeypatch.setattr(FA, "_v1_context", lambda raw, pl: raw)
    X = FA.v1_model_inputs(pl, pd.DataFrame({"x": [1]}))
    assert list(X.index) == [7, 9]
    assert X["_v1_built"].tolist() == [True, False]
    assert X.loc[7, "f"] == 0.5 and pd.isna(X.loc[9, "f"])   # no row -> no input, never a stored value


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
