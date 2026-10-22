"""Predict (OddsAPI) club names resolve to API-Football's spellings (aliases added 2026-10-22)."""
import pytest

from src.team_names import resolve, resolve_result

PAIRS = {
    "Brazil Serie A": [("Bragantino-SP", "RB Bragantino"), ("Atletico Mineiro", "Atletico-MG"),
                       ("Atletico Paranaense", "Atletico Paranaense")],
    "USA MLS": [("D.C. United", "DC United"), ("LA Galaxy", "Los Angeles Galaxy")],
    "Finland Veikkausliiga": [("TPS Turku", "Turku PS")],
    "Argentina Primera Division": [("Argentinos Juniors", "Argentinos JRS"), ("Estudiantes", "Estudiantes L.P."),
                                   ("Gimnasia La Plata", "Gimnasia L.P."), ("Gimnasia Mendoza", "Gimnasia M.")],
    "Austrian Bundesliga": [("Austria Wien", "Austria Vienna"), ("Rapid Wien", "Rapid Vienna"),
                            ("RB Salzburg", "Red Bull Salzburg"), ("Rheindorf Altach", "SCR Altach"),
                            ("WSG Tirol", "WSG Wattens")],
    "China Super League": [("Zhejiang", "Hangzhou Greentown"), ("Chengdu Rongcheng FC", "Chengdu Better City"),
                           ("Dalian Yingbo", "Dalian Zhixing"), ("Liaoning Tieren FC", "Shenyang Urban"),
                           ("Qingdao Hainiu FC", "Qingdao Jonoon"), ("Qingdao West Coast FC", "Qingdao Youth Island"),
                           ("Shenzhen Peng City FC", "Sichuan Jiuniu"), ("Tianjin Jinmen Tiger FC", "Tianjin Teda")],
    "Championship": [("Sheffield United", "Sheffield Utd"), ("Sheffield Wednesday", "Sheffield Weds")],
    "League Two": [("Accrington Stanley", "Accrington ST")],
    "La Liga 2": [("Celta Fortuna", "Celta de Vigo II"), ("Real Sociedad B", "Real Sociedad II")],
}


@pytest.mark.parametrize("league", sorted(PAIRS))
def test_aliases_resolve_within_league(league):
    cands = [af for _, af in PAIRS[league]]
    if league == "Argentina Primera Division":
        cands.append("Estudiantes de Rio Cuarto")                          # the decoy must never win
    if league == "Championship":
        cands.append("Manchester Utd")
    for odds_name, af_name in PAIRS[league]:
        assert resolve_result(odds_name, cands) == af_name, (odds_name, af_name)


def test_first_team_never_resolves_to_its_reserve_side():
    assert resolve("Celta Vigo", ["Celta de Vigo II"]) is None
    assert resolve("Real Sociedad", ["Real Sociedad II"]) is None
    assert resolve_result("Celta Fortuna", ["Celta de Vigo II", "Celta Vigo"]) == "Celta de Vigo II"
    assert resolve_result("Real Sociedad B", ["Real Sociedad II", "Real Sociedad"]) == "Real Sociedad II"


def test_predict_time_resolve_is_not_changed_by_result_aliases():
    # history holds both spellings; predict-time resolve() must keep its token-rule answer
    assert resolve("Rapid Wien", ["SK Rapid", "Rapid Vienna"]) == "SK Rapid"
    assert resolve("Zhejiang", ["Zhejiang Professional", "Hangzhou Greentown"]) == "Zhejiang Professional"
