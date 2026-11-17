"""fix17/replay-identity: the edge replay settles a fixture on the corrected history by PROVEN identity.

After fix16 the replay reads fd_history through serve_history; the owner's duplicate removal keeps ONE
copy of a match, and that copy may carry the bank's other spelling of a club. Settlement used to
resolve a board name ONCE against every spelling the league ever used, so it picked the removed
spelling (still present on other rows) and never found the kept row. It now matches each club to the
spellings ON the rows that can settle the fixture (same league, +-1 day) that the shared resolvers
prove (src.team_names.resolve_identity / resolve_result), and demands ONE score across those rows.
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research import edge_replay as ER

ROOT = Path(__file__).resolve().parents[1]
ARG, MEX = "Argentina Primera Division", "Mexico Liga MX"


def _results(rows):
    r = pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "hg", "ag"])
    r["_d"] = pd.to_datetime(r["date"]).dt.normalize()
    return r


def _entries(rows):
    return pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team"])


def _score(s, i):
    return (s["hg"].iloc[i], s["ag"].iloc[i])


# the corrected history of the three fixtures fix16 left unsettled: the kept copy carries the other
# spelling, while the removed spelling is still the club's name on OTHER rows of the league
HISTORY = _results([
    (ARG, "2026-08-16", "Central Cordoba de Santiago", "Instituto Cordoba", 0, 1),   # kept copy
    (ARG, "2026-08-30", "Talleres Cordoba", "Central Cordoba", 0, 0),                # spelling elsewhere
    (ARG, "2026-09-07", "Instituto", "Boca Juniors", 1, 1),
    (MEX, "2026-08-29", "Atlas", "Club Queretaro", 1, 3),                            # kept copy
    (MEX, "2026-08-29", "CF Pachuca", "Guadalajara Chivas", 1, 1),                   # kept copy
    (MEX, "2026-08-18", "Pachuca", "Puebla", 2, 3),
    (MEX, "2026-09-02", "Queretaro", "Tigres UANL", 0, 0),
])


def test_the_three_fixtures_settle_on_the_kept_copy():
    e = _entries([(ARG, "2026-08-16", "Central Córdoba", "Instituto de Córdoba"),
                  (MEX, "2026-08-29", "Atlas", "Querétaro"),
                  (MEX, "2026-08-29", "Pachuca", "Guadalajara")])
    s = ER.settle(e, HISTORY)
    assert [_score(s, i) for i in range(3)] == [(0.0, 1.0), (1.0, 3.0), (1.0, 1.0)]


def test_the_other_spelling_of_the_club_still_settles_its_own_rows():
    e = _entries([(ARG, "2026-08-30", "Talleres", "Central Córdoba"),
                  (MEX, "2026-08-18", "Pachuca", "Puebla")])
    s = ER.settle(e, HISTORY)
    assert [_score(s, i) for i in range(2)] == [(0.0, 0.0), (2.0, 3.0)]


def test_two_copies_of_one_match_under_two_spellings_must_agree():
    two = _results([("League Two", "2026-08-15", "Accrington ST", "Colchester", 2, 2),     # af copy
                    ("League Two", "2026-08-15", "Accrington", "Colchester", 2, 2)])       # fd copy
    e = _entries([("League Two", "2026-08-15", "Accrington Stanley", "Colchester United")])
    assert _score(ER.settle(e, two), 0) == (2.0, 2.0)
    clash = two.copy()
    clash.loc[1, "ag"] = 3                                         # the copies disagree: never a pick
    assert np.isnan(ER.settle(e, clash)["hg"].iloc[0])


def test_an_unproven_name_stays_unsettled():
    r = _results([(ARG, "2026-08-16", "Estudiantes de Rio Cuarto", "Boca Juniors", 2, 0)])
    e = _entries([(ARG, "2026-08-16", "Estudiantes", "Boca Juniors")])   # a different club
    assert np.isnan(ER.settle(e, r)["hg"].iloc[0])


def test_outside_the_one_day_window_nothing_settles():
    e = _entries([(MEX, "2026-08-31", "Atlas", "Querétaro")])
    assert np.isnan(ER.settle(e, HISTORY)["hg"].iloc[0])


def test_the_applied_identity_map_proves_an_alias_spelling():
    r = _results([("L", "2026-10-01", "Newtown Rovers", "Oldcastle", 1, 0)])
    e = _entries([("L", "2026-10-01", "Ancient Name Club", "Oldcastle")])
    amap = {"L": {"Ancient Name Club": "Newtown Rovers"}}
    assert _score(ER.settle(e, r, identity_aliases=amap), 0) == (1.0, 0.0)
    assert np.isnan(ER.settle(e, r, identity_aliases={})["hg"].iloc[0])      # no map, no proof
    # the map is evidence alias -> canonical only; it never sends the canonical name to the alias
    r2 = _results([("L", "2026-10-01", "Ancient Name Club", "Oldcastle", 1, 0)])
    e2 = _entries([("L", "2026-10-01", "Newtown Rovers", "Oldcastle")])
    assert np.isnan(ER.settle(e2, r2, identity_aliases=amap)["hg"].iloc[0])


def test_proven_spellings_reuses_the_shared_resolvers():
    src = inspect.getsource(ER.proven_spellings)
    assert "resolve_identity" in src and "resolve_result" in src
    text = (ROOT / "src" / "research" / "edge_replay.py").read_text(encoding="utf-8")
    assert "def identity_alias_hit" not in text and "def resolve_identity" not in text   # never copied
    assert ER.proven_spellings("Central Córdoba", ["Central Cordoba de Santiago", "Central Cordoba",
                                                   "Talleres Cordoba"]) == {"Central Cordoba de Santiago",
                                                                            "Central Cordoba"}


def test_the_default_map_is_the_audited_identity_map():
    assert ER.IDENTITY_MAP == ROOT / "config" / "standard_ou25_identity_aliases.json"
    m = ER.identity_map()
    assert m[ARG]["Central Cordoba de Santiago"] == "Central Cordoba"


@pytest.mark.skipif(not (ROOT / "output" / "fd_history.parquet").exists()
                    or not (ROOT / "output" / "af_history.parquet").exists(), reason="no committed history")
def test_the_committed_history_settles_the_three_fixtures():
    r = ER.load_results(ROOT / "output")
    e = _entries([(ARG, "2026-08-16", "Central Córdoba", "Instituto de Córdoba"),
                  (MEX, "2026-08-29", "Atlas", "Querétaro"),
                  (MEX, "2026-08-29", "Pachuca", "Guadalajara")])
    s = ER.settle(e, r)
    assert [_score(s, i) for i in range(3)] == [(0.0, 1.0), (1.0, 3.0), (1.0, 1.0)]
