"""Tip day = the Israeli-morning album that carries a match (owner 2026-10-25)."""
from datetime import datetime, timezone

import pandas as pd

from src import tip_day as T


def test_early_utc_kickoffs_belong_to_the_previous_album():
    # J-League 06:00 UTC Sunday = 09:00 Israel, before the 10:00 rollover -> Saturday's album
    assert T.tip_day_of("2026-10-25T06:00:00Z", "2026-10-25") == "2026-10-24"
    # Liga MX 03:10 UTC Monday -> Sunday's album (it is Sunday night in Mexico)
    assert T.tip_day_of("2026-10-26T03:10:00Z", "2026-10-26") == "2026-10-25"
    # Brazil 23:00 UTC stays on its own day
    assert T.tip_day_of("2026-10-24T23:00:00Z", "2026-10-24") == "2026-10-24"
    # an afternoon European kickoff is unchanged
    assert T.tip_day_of("2026-10-24T14:00:00Z", "2026-10-24") == "2026-10-24"


def test_rollover_is_israel_local_time_on_both_sides_of_the_clock_change():
    # summer (UTC+3): 07:00 UTC = 10:00 Israel -> today's album; 06:59 UTC -> yesterday's
    assert T.tip_day_of("2026-10-20T07:00:00Z") == "2026-10-20"
    assert T.tip_day_of("2026-10-20T06:59:00Z") == "2026-10-19"
    # winter (UTC+2, from 2026-10-25): 08:00 UTC = 10:00 Israel
    assert T.tip_day_of("2026-11-03T08:00:00Z") == "2026-11-03"
    assert T.tip_day_of("2026-11-03T07:30:00Z") == "2026-11-02"


def test_unknown_kickoff_keeps_the_match_date():
    assert T.tip_day_of(None, "2026-10-24") == "2026-10-24"
    assert T.tip_day_of("not a time", "2026-10-24") == "2026-10-24"


def test_add_tip_day_looks_kickoffs_up_and_never_drops_rows():
    ko = pd.DataFrame({"league": ["Mexico Liga MX"], "home_team": ["Tijuana"], "away_team": ["Santos Laguna"],
                       "match_date": ["2026-10-26"], "kickoff_utc": ["2026-10-26T03:10:00Z"]})
    d = pd.DataFrame({"league": ["Mexico Liga MX", "Championship"], "home_team": ["Tijuana", "Hull"],
                      "away_team": ["Santos Laguna", "Leeds"], "match_date": ["2026-10-26", "2026-10-26"]})
    out = T.add_tip_day(d, ko=ko)
    assert len(out) == 2 and out["tip_day"].tolist() == ["2026-10-25", "2026-10-26"]


def test_instant_tip_found_after_the_album_still_goes_out(monkeypatch):
    from telegram_bot import notifier as N
    monkeypatch.setattr(T, "kickoffs", lambda: pd.DataFrame({
        "league": ["Mexico Liga MX"], "home_team": ["Tijuana"], "away_team": ["Santos Laguna"],
        "match_date": ["2026-10-26"], "kickoff_utc": ["2026-10-26T03:10:00Z"]}))
    row = {"league": "Mexico Liga MX", "home_team": "Tijuana", "away_team": "Santos Laguna"}
    after_album = datetime(2026, 10, 25, 14, 0, tzinfo=timezone.utc)
    sent = {"DIGEST|2026-10-25"}
    # UTC match date is tomorrow, but its album (today's) has gone: it must go out alone
    assert N.instant_tip_allowed("2026-10-26", sent, now=after_album, row=row)
    # before today's album: it waits for the album
    before = datetime(2026, 10, 25, 4, 0, tzinfo=timezone.utc)
    assert not N.instant_tip_allowed("2026-10-26", set(), now=before, row=row)
