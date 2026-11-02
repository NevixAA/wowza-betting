"""
Understat payload parsing (owner decision B, 2026-11-02). Network-free: the collector
(scripts/collect_understat_shots.py) fetches, these functions turn a payload into rows.

Understat serves the same data two ways, and both are accepted:
  * JSON endpoints  /getLeagueData/<league>/<season>  -> {"dates": [...], ...}
                    /getMatchData/<match_id>          -> {"shots": {"h": [...], "a": [...]},
                                                          "rosters": {"h": {...}, "a": {...}}}
  * HTML pages      /league/<league>/<season>, /match/<id> with
                    `var datesData = JSON.parse('<hex-escaped JSON>')` (and shotsData, rostersData).

SHOT SITUATIONS (Understat's own labels): OpenPlay, FromCorner, SetPiece, DirectFreekick, Penalty.
"""
from __future__ import annotations

import codecs
import json
import re

# The leagues Understat covers that we hold player rows for, as Understat names them in its URLs
# (the fetch list - which pages to request - not a mapping to our leagues: our fixture and league
# are found by date + both clubs resolving, see mapping.py).
UNDERSTAT_LEAGUES = ("EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1")

SET_PIECE_SITUATIONS = frozenset({"FromCorner", "SetPiece", "DirectFreekick"})
ON_TARGET_RESULTS = frozenset({"Goal", "SavedShot"})

SHOT_COLS = ["shot_id", "match_id", "h_a", "minute", "result", "X", "Y", "xG", "player",
             "us_player_id", "situation", "shotType", "lastAction", "player_assisted"]
ROSTER_COLS = ["match_id", "h_a", "us_team_id", "us_player_id", "player", "position", "time",
               "goals", "shots", "key_passes", "assists", "xG", "xA"]
MATCH_COLS = ["match_id", "us_league", "us_season", "datetime", "h_id", "h_title", "a_id", "a_title",
              "h_goals", "a_goals", "is_result"]


def _embedded(html: str, var: str):
    """The object of `var <name> = JSON.parse('...')` in an Understat page, or None."""
    m = re.search(r"var\s+" + re.escape(var) + r"\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        return None
    raw = codecs.decode(m.group(1), "unicode_escape")
    try:                                      # \xNN escapes of UTF-8 bytes -> real characters
        raw = raw.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return json.loads(raw)


def _payload(text: str) -> dict | None:
    t = (text or "").lstrip()
    if t.startswith("{"):
        try:
            return json.loads(t)
        except ValueError:
            return None
    return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_league(text: str, league: str, season: str) -> list[dict]:
    """Match rows of one Understat league-season (JSON endpoint or HTML page)."""
    body = _payload(text)
    dates = body.get("dates") if isinstance(body, dict) else _embedded(text, "datesData")
    rows = []
    for m in dates or []:
        h, a, g = m.get("h") or {}, m.get("a") or {}, m.get("goals") or {}
        rows.append({"match_id": _i(m.get("id")), "us_league": league, "us_season": str(season),
                     "datetime": m.get("datetime"), "h_id": _i(h.get("id")), "h_title": h.get("title"),
                     "a_id": _i(a.get("id")), "a_title": a.get("title"),
                     "h_goals": _f(g.get("h")), "a_goals": _f(g.get("a")),
                     "is_result": bool(m.get("isResult"))})
    return [r for r in rows if r["match_id"] is not None]


def parse_match(text: str, match_id: int) -> tuple[list[dict], list[dict]] | None:
    """(shots, roster) of one match, or None when the payload holds neither."""
    body = _payload(text)
    if isinstance(body, dict) and ("shots" in body or "rosters" in body):
        shots, rosters = body.get("shots") or {}, body.get("rosters") or {}
    else:
        shots, rosters = _embedded(text, "shotsData"), _embedded(text, "rostersData")
        if shots is None and rosters is None:
            return None
        shots, rosters = shots or {}, rosters or {}
    srows = []
    for side in ("h", "a"):
        for s in shots.get(side) or []:
            srows.append({"shot_id": _i(s.get("id")), "match_id": int(match_id), "h_a": side,
                          "minute": _f(s.get("minute")), "result": s.get("result"),
                          "X": _f(s.get("X")), "Y": _f(s.get("Y")), "xG": _f(s.get("xG")),
                          "player": s.get("player"), "us_player_id": _i(s.get("player_id")),
                          "situation": s.get("situation"), "shotType": s.get("shotType"),
                          "lastAction": s.get("lastAction"),
                          "player_assisted": s.get("player_assisted") or None})
    rrows = []
    for side in ("h", "a"):
        for r in (rosters.get(side) or {}).values():
            rrows.append({"match_id": int(match_id), "h_a": side, "us_team_id": _i(r.get("team_id")),
                          "us_player_id": _i(r.get("player_id")), "player": r.get("player"),
                          "position": r.get("position"), "time": _f(r.get("time")),
                          "goals": _f(r.get("goals")), "shots": _f(r.get("shots")),
                          "key_passes": _f(r.get("key_passes")), "assists": _f(r.get("assists")),
                          "xG": _f(r.get("xG")), "xA": _f(r.get("xA"))})
    return srows, rrows
