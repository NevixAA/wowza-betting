"""fix12 (2026-11-07): board names meet the CANONICAL history through the APPLIED identity map.

The identity merge renames a club to its canonical spelling ('Stockport County' -> 'Stockport',
'Energie Cottbus' -> 'Cottbus'), so the served history pool no longer holds the spelling the board
(OddsAPI) uses. resolve() alone then missed 'Stockport County FC' and the fixture was served blind
(name_unresolved, NaN form). src.team_names.resolve_identity accepts a match through the applied
map's ALIAS spellings - proven identity only: one canonical club in the pool, or nothing.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import config  # noqa: E402
from src.team_names import identity_alias_hit, resolve, resolve_identity  # noqa: E402


@pytest.fixture(scope="module")
def frozen_map():
    """The identity map FROZEN in the served Standard payload (what serving_history applied)."""
    p = Path(config.MODEL_FILE_STANDARD_REPAIRED)
    if not p.exists():
        pytest.skip("repaired Standard payload not committed")
    with open(p, "rb") as f:
        pay = pickle.load(f)
    return pay["identity_alias_map"]


LEAGUE_ONE = ["Bolton", "Stockport", "Wigan", "Wycombe", "Reading"]
BUNDES_2 = ["Cottbus", "Hertha", "Karlsruhe", "Darmstadt", "Paderborn"]


def test_frozen_map_carries_the_renames_this_fix_relies_on(frozen_map):
    assert frozen_map["League One"]["Stockport County"] == "Stockport"
    assert frozen_map["Bundesliga 2"]["Energie Cottbus"] == "Cottbus"


def test_stockport_county_fc_reaches_the_canonical_stockport_through_the_frozen_map(frozen_map):
    # the regression: resolve() alone cannot see 'Stockport' in the board spelling
    assert resolve("Stockport County FC", LEAGUE_ONE) is None
    assert resolve("Stockport County FC", ["Stockport County", "Bolton"]) == "Stockport County"
    assert resolve_identity("Stockport County FC", LEAGUE_ONE, frozen_map["League One"]) == "Stockport"
    assert identity_alias_hit("Stockport County FC", LEAGUE_ONE, frozen_map["League One"]) == "Stockport"


def test_energie_cottbus_reaches_the_canonical_cottbus_through_the_frozen_map(frozen_map):
    # the map path on its own (the alias store also carries the board spelling since fix12)
    assert identity_alias_hit("FC Energie Cottbus", BUNDES_2, frozen_map["Bundesliga 2"]) == "Cottbus"
    assert resolve_identity("FC Energie Cottbus", BUNDES_2, frozen_map["Bundesliga 2"]) == "Cottbus"


def test_an_alias_whose_canonical_is_not_in_the_pool_is_refused():
    amap = {"Stockport County": "Stockport"}
    pool = ["Bolton", "Wigan", "Wycombe"]                    # no Stockport in this league's history
    assert identity_alias_hit("Stockport County FC", pool, amap) is None
    assert resolve_identity("Stockport County FC", pool, amap) is None


def test_two_canonical_clubs_are_refused():
    # two alias spellings both proven equal to the board name, pointing at two different clubs
    amap = {"Stockport County": "Stockport", "Stockport County AFC": "Stockport Town"}
    pool = ["Stockport", "Stockport Town", "Bolton"]
    assert identity_alias_hit("Stockport County FC", pool, amap) is None
    assert resolve_identity("Stockport County FC", pool, amap) is None


def test_two_aliases_of_one_canonical_still_resolve():
    amap = {"Stockport County": "Stockport", "Stockport Cty": "Stockport"}
    assert resolve_identity("Stockport County FC", LEAGUE_ONE, amap) == "Stockport"


def test_without_a_map_nothing_changes():
    assert resolve_identity("Stockport County FC", LEAGUE_ONE, None) is None
    assert resolve_identity("Stockport County FC", LEAGUE_ONE, {}) is None
    assert resolve_identity("Wigan Athletic", LEAGUE_ONE, None) == resolve("Wigan Athletic", LEAGUE_ONE)


def test_platense_refusal_is_unchanged(frozen_map):
    pool = ["River Plate", "Boca Juniors", "Racing Club", "Velez Sarsfield"]
    assert resolve("Platense", pool) is None
    # whatever identity map is applied, it never makes Platense another club
    for amap in (frozen_map.get("Argentina Primera Division", {}), {"River Plate CA": "River Plate"}):
        assert resolve_identity("Platense", pool, amap) is None


def test_map_resolution_never_overrides_a_direct_proof():
    # resolve() proves 'Wigan Athletic' == 'Wigan'; a map pointing elsewhere is never consulted
    amap = {"Wigan Athletic": "Bolton"}
    assert resolve_identity("Wigan Athletic", LEAGUE_ONE, amap) == "Wigan"


def test_data_loader_uses_the_same_shared_rule():
    from src import data_loader as dl
    hist = pd.DataFrame({"league": ["L1"] * 2, "home_team": ["Stockport", "Bolton"],
                         "away_team": ["Bolton", "Stockport"]})
    right = pd.DataFrame({"league": ["L1"], "home_team": ["Stockport County FC"], "away_team": ["Bolton"]})
    got = dl._resolve_onto_history(right, hist, "home_team", "away_team", "t",
                                   aliases={"L1": {"Stockport County": "Stockport"}})
    assert got["_h"].tolist() == ["Stockport"] and got["_a"].tolist() == ["Bolton"]
    assert dl._resolve_onto_history(right, hist, "home_team", "away_team", "t").empty


# ── end to end: the repaired Standard serving builder ──────────────────────────────────────────
def _history_with_stockport():
    from test_standard_forensic_repair import _frame, STD_A
    raw = _frame()
    for c in ("home_team", "away_team"):
        raw.loc[(raw["league"] == STD_A) & (raw[c] == "A0"), c] = "Stockport"
    return raw, STD_A


def test_build_upcoming_features_repaired_gives_stockport_real_form():
    from src.features_repaired import build_upcoming_features_repaired
    raw, lg = _history_with_stockport()
    up = pd.DataFrame({"date": [raw["date"].max() + pd.Timedelta(days=7)], "league": [lg],
                       "home_team": ["Stockport County FC"], "away_team": ["A1"]})
    blind = build_upcoming_features_repaired(up, raw)
    assert bool(blind["name_unresolved"].iloc[0]) and np.isnan(blind["home_scored_last5"].iloc[0])
    f = build_upcoming_features_repaired(up, raw, identity_aliases={lg: {"Stockport County": "Stockport"}})
    assert not bool(f["name_unresolved"].iloc[0])
    assert f["history_team_home"].iloc[0] == "Stockport"
    for c in ("home_scored_last5", "home_conceded_last5", "away_scored_last5"):
        assert np.isfinite(f[c].iloc[0]), c
    assert f["home_team"].iloc[0] == "Stockport County FC"     # board spelling preserved
    # the same numbers as a board that already used the history's own spelling
    same = build_upcoming_features_repaired(up.assign(home_team="Stockport"), raw)
    assert f["home_scored_last5"].iloc[0] == same["home_scored_last5"].iloc[0]


def test_standard_serving_passes_the_frozen_payload_map(monkeypatch):
    from src import features_repaired as FR
    from src import standard_serving as SS
    seen = {}

    def fake(ident, history, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(FR, "build_upcoming_features_repaired", fake)
    lg = sorted(config.STANDARD_FORMAT_LEAGUES)[0]
    fx = pd.DataFrame({"date": ["2026-11-10"], "league": [lg], "home_team": ["X"], "away_team": ["Y"]})
    pay = {"identity_alias_map": {lg: {"Stockport County": "Stockport"}}}
    with pytest.raises(RuntimeError):
        SS.score_fixtures(fx, pd.DataFrame(), pay)
    assert seen["identity_aliases"] is pay["identity_alias_map"]


def test_legacy_builder_resolves_through_the_loader_map():
    from src.feature_engineering import build_upcoming_features
    raw, lg = _history_with_stockport()
    up = pd.DataFrame({"date": [raw["date"].max() + pd.Timedelta(days=7)], "league": [lg],
                       "home_team": ["Stockport County FC"], "away_team": ["A1"],
                       "odds_over25": [1.9], "odds_under25": [1.9]})
    blind = build_upcoming_features(up, raw)
    f = build_upcoming_features(up, raw, identity_aliases={lg: {"Stockport County": "Stockport"}})
    assert np.isnan(blind["home_scored_last5"].iloc[0])
    assert np.isfinite(f["home_scored_last5"].iloc[0])
    assert np.isfinite(f["home_rest_days"].iloc[0])
