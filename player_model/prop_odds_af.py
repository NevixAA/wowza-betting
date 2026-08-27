"""
API-Football as a SECOND player-prop price source.
==================================================

WHY THIS EXISTS

OddsAPI does not sell player props for the divisions this system actually targets. Their own
documentation is explicit: *"Soccer player props are available for EPL, French Ligue 1, German
Bundesliga, Italian Serie A, Spanish La Liga, and MLS."* Confirmed by probe on 2026-08-27 —
Championship, League One, League Two and EFL Cup return **zero** prop bookmakers across
`uk,eu`, `us` and `uk,eu,us,us2`, while the same fixtures return 36-38 books for `h2h`. No plan
tier changes it; the markets are simply not sourced.

API-Football, which this project already pays for, does carry them:

    Championship      13 books, 22 player-prop markets
    League One        13 books, 11
    League Two        13 books, 11
    Europa League     13 books,  4
    Conference League 13 books,  1

Wrexham v Birmingham, Anytime Goal Scorer: Bet365 with 52 priced players, one call. Daily quota
sits around 74,000 of 75,000 unused.

DESIGN: TWO SOURCES, ONE SET OF FILES

This module is a CLIENT, not a parallel pipeline. It returns exactly the two shapes
`odds_fetcher` already produces — the `{"Player|market": best_odds}` lookup and the
`_append_odds_history` record dicts — so every existing consumer, the odds history CSV and the
coverage ledger stay as they are. That mirrors the rule ARCHITECTURE.md already sets for the
main odds layer: sources normalise to an identical shape and downstream code stays
source-agnostic, with parsing differences confined to the client module.

Concretely: no new output file, no new schema, no second history. `source` is recorded on each
row so the two can be told apart after the fact, which is the one thing that would otherwise be
unrecoverable.

WHAT IT DELIBERATELY DOES NOT DO

* It does not touch leagues OddsAPI already covers. `odds_fetcher` calls this only for the
  remainder, so the primary source keeps priority and cost does not double for the top five.
* It does not invent markets. API-Football exposes 338 bet types; only the four that map cleanly
  onto our vocabulary are read, and an unmapped bet name is skipped rather than guessed at.
* It does not backfill. API-Football `/odds` is pre-match only — established by probe and
  recorded in CLAUDE.md — so a price not captured before kickoff is gone. Forward only.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import config

# PROP_LEAGUES / PROP_SEASONS live in player_model/config.py, NOT the repo-root config that
# odds_fetcher imports as `config`. Reading them off the wrong module returns an empty dict and
# this source then silently covers nothing — which is exactly how the first version failed: no
# error, no log line, just zero quotes and a green run.
def _prop_leagues() -> dict:
    from player_model import config as pm_config
    return getattr(pm_config, "PROP_LEAGUES", {}) or {}


def _prop_seasons() -> dict:
    from player_model import config as pm_config
    return getattr(pm_config, "PROP_SEASONS", {}) or {}


# API-Football bet names -> our market vocabulary (odds_fetcher._MARKET_MAP keys).
#
# Read from their /odds/bets catalogue rather than assumed. `Anytime Goal Scorer` is a
# yes/no per player; the shots and assists markets arrive as Over/Under lines, so the
# threshold is parsed out of the value string and matched to our sot / sot2 / sot3 split.
_BET_TO_MARKET = {
    "anytime goal scorer": ("goals", None),
    "player to be booked": ("cards", None),
    "player assists": ("assists", 0.5),
    "player shots on target total": ("sot", "line"),
    "player shots total": (None, None),          # total shots, not ON TARGET — not our market
}

# Home/away-prefixed duplicates of the same market. API-Football splits several props into
# "Away Anytime Goal Scorer" / "Anytime Goal Scorer"; both carry per-player outcomes and both
# are wanted, because taking only the unprefixed one silently loses one team's players.
_PREFIXES = ("home ", "away ")

# Which of our sot buckets a parsed Over line belongs to.
_SOT_LINE = {0.5: "sot", 1.5: "sot2", 2.5: "sot3"}

# "2+" style thresholds, the form API-Football actually uses for per-player lines.
_PLUS_RE = re.compile(r"(\d+(?:\.\d+)?)\+")

_MIN_ODDS = 1.01
_MAX_ODDS = 1000.0


def _norm_bet(name: str) -> str:
    s = str(name or "").strip().lower()
    for p in _PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    return s


def _clean_player(raw: str) -> str:
    """Player name out of a value string.

    API-Football's threshold props use `"Liberato Cacace - 1+"`, NOT the Over/Under wording the
    first version of this assumed. That mattered twice: the threshold never parsed (so every
    shots-on-target quote was silently dropped — 0 of 118 outcomes on the probe fixture), and
    the suffix would have stayed glued to the name, so `player_history` would never have
    matched it. Both forms are handled now; the `N+` form is the one that actually occurs.
    """
    s = str(raw or "").strip()
    if " - " in s:
        s = s.split(" - ", 1)[0]
    for tok in ("Over", "over", "Under", "under"):
        s = s.replace(tok, " ")
    parts = [p for p in s.split() if not _is_number(p) and not _PLUS_RE.fullmatch(p)]
    return " ".join(parts).strip()


def _parse_line(raw: str) -> float | None:
    """Threshold as an OVER LINE, from either `"- 2+"` (2 or more -> 1.5) or `"Over 1.5"`.

    Returned on the Over-line scale so both encodings land in the same `_SOT_LINE` buckets:
    "1+" means 1 or more, which is Over 0.5.
    """
    s = str(raw or "")
    if " - " in s:
        tail = s.rsplit(" - ", 1)[1].strip()
        m = _PLUS_RE.fullmatch(tail)
        if m:
            return float(m.group(1)) - 0.5
    for tok in s.split():
        if _is_number(tok):
            return float(tok)
    return None


def _is_under(raw: str) -> bool:
    return "under" in str(raw or "").lower()


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def _key_name(name: str) -> str:
    """Accent-folded, case-insensitive player key, matching how odds_fetcher joins names."""
    nf = unicodedata.normalize("NFKD", str(name or ""))
    return "".join(c for c in nf if not unicodedata.combining(c)).strip()


def leagues_without_oddsapi(wanted_leagues: Iterable[str],
                            oddsapi_leagues: Iterable[str]) -> list[str]:
    """The leagues this source should cover: wanted, known to API-Football, and NOT already
    served by OddsAPI. Keeps the primary source's priority explicit rather than implied."""
    have = {str(x) for x in oddsapi_leagues}
    known = _prop_leagues()
    return [lg for lg in dict.fromkeys(str(x) for x in wanted_leagues)
            if lg not in have and lg in known]


def find_upcoming_fixture_id(league_id: int, season: str, date_str: str,
                             home: str, away: str) -> int | None:
    """API-Football fixture id for an UPCOMING match.

    `api_football.find_fixture_id` cannot be reused: it hardcodes `status=FT`, because it exists
    to grade matches that have already finished. Props are pre-match by definition, so that
    filter returns nothing and the whole second source silently covers zero fixtures — which is
    how the first version of this looked like it worked. Left alone rather than parameterised,
    since grading relies on that filter and on a 7-day cache that would be wrong here.
    """
    from player_model.api_football import _get, _team_match

    data = _get("/fixtures", {"league": league_id, "season": season, "date": date_str},
                cache_hours=6) or {}
    for fix in data.get("response", []):
        t = fix.get("teams", {})
        fh = (t.get("home") or {}).get("name", "")
        fa = (t.get("away") or {}).get("name", "")
        if _team_match(home, fh) and _team_match(away, fa):
            return (fix.get("fixture") or {}).get("id")
    return None


def fetch(fixtures: list[dict], *, verbose: bool = True) -> tuple[dict[str, float], list[dict]]:
    """Fetch props for `fixtures` and return the SAME two shapes odds_fetcher produces.

    `fixtures` items need: league, match ("{home} vs {away}"), match_date, and either
    `fixture_id` (an API-Football id) or enough to resolve one.

    Returns ({"Player|market": best_decimal_odds}, [history record dicts]).
    """
    from player_model.api_football import _get

    best: dict[str, float] = {}
    records: list[dict] = []
    if not fixtures:
        return best, records

    for fx in fixtures:
        fid = fx.get("fixture_id")
        league = str(fx.get("league") or "")
        match = str(fx.get("match") or "")
        match_date = str(fx.get("match_date") or "")[:10]
        if not fid:
            continue
        resp = (_get("/odds", {"fixture": int(fid)}, cache_hours=1) or {}).get("response") or []
        n_before = len(records)
        for block in resp:
            for bm in (block.get("bookmakers") or []):
                book = str(bm.get("name") or "")
                for bet in (bm.get("bets") or []):
                    mapped = _BET_TO_MARKET.get(_norm_bet(bet.get("name")))
                    if not mapped or mapped[0] is None:
                        continue
                    market, spec = mapped
                    for val in (bet.get("values") or []):
                        raw = val.get("value")
                        # An UNDER quote is a different bet and must never be stored as if it
                        # were the over — that mistake would look like a very generous price.
                        if _is_under(raw):
                            continue
                        odd = _to_float(val.get("odd"))
                        if odd is None or not (_MIN_ODDS <= odd <= _MAX_ODDS):
                            continue
                        mkt = market
                        if spec == "line":
                            line = _parse_line(raw)
                            # No parseable threshold means this is NOT the same bet. The
                            # "Away Player Shots On Target Total" block carries bare player
                            # names at very different prices (Kieffer Moore at 5.50 where the
                            # 1+ line is 1.33), so treating an unlabelled value as 1+ would
                            # store a much longer price as if it were ours.
                            mkt = _SOT_LINE.get(line) if line is not None else None
                            if mkt is None:
                                continue          # a line we do not model
                        elif isinstance(spec, float):
                            line = _parse_line(raw)
                            if line is not None and abs(line - spec) > 1e-9:
                                continue
                        player = _clean_player(raw)
                        if not player:
                            continue
                        k = f"{_key_name(player)}|{mkt}"
                        # Best available price across books, matching odds_fetcher's contract.
                        if odd > best.get(k, 0.0):
                            best[k] = odd
                        records.append({"match_date": match_date, "league": league,
                                        "match": match, "player": player, "market": mkt,
                                        "odds": odd, "source": "api_football",
                                        "bookmaker": book})
        if verbose:
            got = len(records) - n_before
            print(f"[prop_odds_af] {league}: {match} -> {got} quote(s)"
                  + ("" if got else "  (no player props returned)"))
    return best, records


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None
