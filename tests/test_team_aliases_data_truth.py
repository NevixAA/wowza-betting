"""Club aliases added from the 2026-11-01 data profile (unresolved_club_name), each proven the same
club at fixture level against output/fd_history.parquet + af_history.parquet (see src/team_names.py).

Single-spelling clubs go in _ALIASES (predict + settlement); clubs the history holds under TWO
spellings go in _RESULT_ALIASES only, so predict-time form is unchanged until the identity merge.
"""
import pytest

from src.team_names import resolve, resolve_result

PREDICT = [  # (league pool from the history bank, predict spelling, history spelling)
    (["Estudiantes L.P.", "Estudiantes Rio Cuarto", "Estudiantes de Rio Cuarto"], "Estudiantes", "Estudiantes L.P."),
    (["Gimnasia L.P.", "Gimnasia M.", "Gimnasia Mendoza"], "Gimnasia La Plata", "Gimnasia L.P."),
    (["Cercle Brugge", "Club Brugge"], "Cercle Brugge KSV", "Cercle Brugge"),
    (["Oud-Heverlee Leuven", "Club Brugge"], "OH Leuven", "Oud-Heverlee Leuven"),
    (["Beijing Guoan", "Beijing Renhe"], "Beijing FC", "Beijing Guoan"),
    (["Boston Utd", "Boston"], "Boston United", "Boston Utd"),
    (["Inverness C", "Partick"], "Inverness CT", "Inverness C"),
    (["Raith Rvs", "Partick"], "Raith Rovers", "Raith Rvs"),
    (["Airdrie Utd", "Alloa"], "Airdrie United", "Airdrie Utd"),
    (["Queen of Sth", "Queens Park"], "Queen of the South", "Queen of Sth"),
    (["Sp Lisbon", "Sp Braga", "Benfica"], "Sporting CP", "Sp Lisbon"),
    (["Guimaraes", "Setubal"], "Vitória SC", "Guimaraes"),
    (["Buyuksehyr", "Besiktas"], "Başakşehir", "Buyuksehyr"),
]

SETTLE_ONLY = [
    (["Belgrano", "Belgrano Cordoba", "Talleres Cordoba"], "Belgrano de Cordoba", "Belgrano"),
    (["Instituto", "Instituto Cordoba", "Talleres Cordoba"], "Instituto de Córdoba", "Instituto"),
    (["Chapecoense-SC", "Chapecoense-sc"], "Chapecoense", "Chapecoense-SC"),
    (["Paderborn", "SC Paderborn 07"], "SC Paderborn", "Paderborn"),
    (["SHANGHAI SIPG", "Shanghai Port", "Shanghai SIPG", "Shanghai Shenhua"], "Shanghai SIPG FC", "Shanghai Port"),
    (["St Patrick's Athl.", "St. Patricks"], "St Patricks Athletic", "St. Patricks"),
    (["Kyoto", "Kyoto Sanga"], "Kyoto Purple Sanga", "Kyoto Sanga"),
    (["Andorra", "FC Andorra"], "Andorra CF", "Andorra"),
    (["Castellon", "Castellón"], "CD Castellón", "Castellon"),
    (["U.N.A.M. - Pumas", "UNAM Pumas"], "Pumas", "UNAM Pumas"),
]


@pytest.mark.parametrize("pool,name,want", PREDICT)
def test_single_spelling_clubs_resolve_at_predict_and_settlement(pool, name, want):
    assert resolve(name, pool) == want
    assert resolve_result(name, pool) == want


@pytest.mark.parametrize("pool,name,want", SETTLE_ONLY)
def test_duplicate_identity_clubs_settle_but_predict_stays_unresolved(pool, name, want):
    assert resolve_result(name, pool) == want
    assert resolve(name, pool) is None          # unchanged: no half-history form at predict time


def test_aliases_never_cross_to_the_other_club():
    assert resolve("Estudiantes", ["Estudiantes L.P.", "Estudiantes de Rio Cuarto"]) == "Estudiantes L.P."
    assert resolve("Beijing FC", ["Beijing Guoan", "Beijing Renhe"]) == "Beijing Guoan"
    assert resolve("Vitória SC", ["Setubal"]) is None
    assert resolve("Sporting CP", ["Sporting Gijon"]) is None
    # a first team labelled in La Liga 2 data is NOT its reserve side (left unresolved on purpose)
    assert resolve_result("Celta Vigo", ["Celta B", "Celta de Vigo II"]) is None
