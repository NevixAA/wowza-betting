"""Canonical per-league GAMEWEEK id for the weekly research panel (rule GW-1, 2026-10-25).

    from src.gameweek import league_week_id, add_league_week_id
    league_week_id("Championship", "2026-10-26T19:45:00Z")        # 'Championship|2026-W44'

RESEARCH / DISPLAY ONLY. Nothing in production reads this; it groups settled evidence into weeks.

RULE GW-1 (deterministic, pure: same inputs -> same id, no clock, no file reads)
  1. OFFICIAL ROUND, when the caller supplies one: '<league>|<season>|R:<round>'. The round string is
     normalised (whitespace collapsed, case kept) so 'Regular Season - 12' and 'Regular Season -  12'
     agree. Checked 2026-10-25: NO committed v9 or Pro artifact carries an API-Football round today
     (af_history.parquet, predictions.csv, Pro fixtures/team_match_stats were all inspected), so this
     branch is dormant until a round column is collected. It is an explicit argument, never inferred.
  2. Otherwise '<league>|<ISO-year>-W<ISO-week>' of the TIP DAY (src/tip_day.tip_day_of), not of the
     UTC kickoff date. The tip day rolls over at 10:00 Israel time, so a Sunday-night match in the
     Americas that kicks off at 01:00-03:00 UTC Monday stays in its weekend's week.
  3. No kickoff and no match date -> '' (unassignable). Never a guessed week.

EDGE CASES (documented, deliberately not "fixed" by heuristics)
  * Midweek rounds (Tue/Wed): fall in the same ISO week as the PRECEDING weekend (ISO weeks run
    Mon-Sun), so 'weekend round N + midweek round N+1' share one id. The panel is weekly, not per round.
  * Two rounds in one ISO week: same as above — one id. Use the official-round branch to split them.
  * Monday-EVENING fixtures (EFL Monday night, 19:45 UTC): the tip day is Monday, so they belong to the
    NEXT ISO week, i.e. with the following midweek/weekend, not with the Saturday they follow.
  * Long split rounds (a round played over Fri-Mon, or postponed games replayed weeks later): each
    match takes the week it was actually PLAYED in. A rearranged fixture never back-fills its
    original round's week.
  * ISO year boundary: 2026-12-31 may be 2027-W53/W01 — the ISO year is used, never the calendar year.
"""
from __future__ import annotations

import re

import pandas as pd

from src.tip_day import tip_day_of

RULE_VERSION = "GW-1"


def _norm_round(r) -> str:
    return re.sub(r"\s+", " ", str(r)).strip()


def league_week_id(league: str, kickoff_utc=None, match_date: str | None = None,
                   official_round: str | None = None, season: str | None = None) -> str:
    """The canonical week id for one fixture (rule GW-1). Pure and deterministic."""
    lg = str(league).strip()
    if official_round is not None and not (isinstance(official_round, float) and pd.isna(official_round)) \
            and _norm_round(official_round):
        return f"{lg}|{str(season or '').strip()}|R:{_norm_round(official_round)}"
    day = tip_day_of(kickoff_utc, match_date)
    d = pd.to_datetime(str(day or "")[:10], errors="coerce")
    if pd.isna(d):
        return ""
    iso = d.isocalendar()
    return f"{lg}|{int(iso[0])}-W{int(iso[1]):02d}"


def add_league_week_id(df: pd.DataFrame, kickoff_col: str = "kickoff_utc", date_col: str = "match_date",
                       round_col: str | None = None, season_col: str | None = None) -> pd.DataFrame:
    """Return df with a `league_week_id` column. Never drops or reorders rows."""
    out = df.copy()
    n = len(out)
    ko = out[kickoff_col] if kickoff_col in out.columns else pd.Series([None] * n, index=out.index)
    md = out[date_col].astype(str).str[:10] if date_col in out.columns else pd.Series([None] * n, index=out.index)
    rd = out[round_col] if round_col and round_col in out.columns else pd.Series([None] * n, index=out.index)
    ss = out[season_col] if season_col and season_col in out.columns else pd.Series([None] * n, index=out.index)
    out["league_week_id"] = [league_week_id(lg, k, m, r, s)
                             for lg, k, m, r, s in zip(out["league"], ko, md, rd, ss)]
    return out
