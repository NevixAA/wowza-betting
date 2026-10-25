"""Canonical league_week_id (rule GW-1): deterministic, tip-day based, documented edge cases."""
import pandas as pd

from src import gameweek as W


def test_deterministic_and_pure():
    a = [W.league_week_id("Championship", "2026-10-24T14:00:00Z") for _ in range(3)]
    assert a == ["Championship|2026-W43"] * 3
    d = pd.DataFrame({"league": ["Championship", "Mexico Liga MX"],
                      "kickoff_utc": ["2026-10-24T14:00:00Z", "2026-10-26T03:10:00Z"],
                      "match_date": ["2026-10-24", "2026-10-26"]})
    x, y = W.add_league_week_id(d), W.add_league_week_id(d.iloc[::-1])
    assert x["league_week_id"].tolist() == y["league_week_id"].tolist()[::-1]
    assert len(x) == 2 and list(d.columns) == list(x.columns[:3])


def test_sunday_night_americas_match_stays_in_its_weekend():
    # Liga MX 03:10 UTC Monday 2026-10-26 = Sunday night locally -> tip day Sunday 10-25 -> W43
    assert W.league_week_id("Mexico Liga MX", "2026-10-26T03:10:00Z", "2026-10-26") == "Mexico Liga MX|2026-W43"
    # same weekend's Saturday game
    assert W.league_week_id("Mexico Liga MX", "2026-10-25T01:00:00Z") == "Mexico Liga MX|2026-W43"


def test_documented_edge_cases():
    # Monday-evening EFL game belongs to the NEXT ISO week (tip day Monday)
    assert W.league_week_id("League Two", "2026-10-26T19:45:00Z") == "League Two|2026-W44"
    # midweek round shares the ISO week with the next weekend (Mon-Sun weeks)
    assert W.league_week_id("League Two", "2026-10-28T19:45:00Z") == W.league_week_id("League Two",
                                                                                     "2026-10-31T15:00:00Z")
    # ISO year boundary uses the ISO year
    assert W.league_week_id("Serie B", "2026-12-31T19:00:00Z") == "Serie B|2026-W53"
    assert W.league_week_id("Serie B", "2027-01-04T19:00:00Z") == "Serie B|2027-W01"


def test_official_round_wins_when_supplied_and_unknown_is_empty():
    assert W.league_week_id("Ligue 2", "2026-10-24T18:00:00Z", official_round="Regular Season -  12",
                            season="2026") == "Ligue 2|2026|R:Regular Season - 12"
    assert W.league_week_id("Ligue 2", None, None) == ""
    assert W.league_week_id("Ligue 2", None, "nan") == ""
    # no kickoff -> the match date is used
    assert W.league_week_id("Ligue 2", None, "2026-10-24") == "Ligue 2|2026-W43"
