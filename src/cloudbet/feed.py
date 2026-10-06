"""Cloudbet's own prices — the only prices this bot may bet into.

    from src.cloudbet.feed import price_for, find_event, discover_markets

WHY THIS IS THE CRITICAL PIECE. v9 computes edge against Bet365 and a cross-book consensus.
Cloudbet is in neither. An edge of 9% against Bet365 may be 4% at Cloudbet, or nothing — so
every selection must be RE-PRICED here before it is bettable. Until that happens the selector's
`no_cloudbet_price` filter correctly drops everything.

THE MARKET MAP BELOW IS A HYPOTHESIS, NOT A FACT. Cloudbet documents the shape
(`soccer.market/outcome?params`) but their wiki is not reachable over TLS from here, and the
exact keys for total goals and both-teams-to-score are not pinned anywhere I could verify. So
this module does NOT quietly guess:

  * `price_for` returns (None, reason) when a market cannot be found, and the reason NAMES the
    keys the payload actually contained.
  * `discover_markets` dumps those keys, so one real event fixes the map in a single edit.

That is deliberate and it is the lesson from `book_quotes`: a classifier that matched on a NAME
rather than a verified key swallowed seven unrelated bet types and produced a market probability
of 0.254 where the truth was 0.63. A mapping that silently matches nothing is better than one
that silently matches the wrong thing — but only if it SAYS so, which is what the reason string
is for.

TEAM MATCHING IS LEAGUE-SCOPED AND REFUSES AMBIGUITY (invariant 11). Club names differ between
sources — `1. FC Kaiserslautern` / `Kaiserslautern`, `QPR` / `Queens Park Rangers`. A naive
`startswith(first_word)` once mapped `Real Valladolid CF` onto any club starting "Real" and left
46% of standard fixtures with no form data. Betting the wrong fixture is worse than not betting.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: Our market name -> (cloudbet market key, outcome param name). UNVERIFIED — see module docs.
#: `line` is formatted into the query string when the market takes one.
MARKET_MAP: dict[str, tuple[str, str | None]] = {
    "ou25":   ("soccer.total_goals", "total"),
    "over15": ("soccer.total_goals", "total"),
    "over35": ("soccer.total_goals", "total"),
    "btts":   ("soccer.both_teams_to_score", None),
}

#: Our side -> Cloudbet outcome. Over/Under for totals, yes/no for BTTS.
SIDE_MAP = {"OVER": "over", "UNDER": "under", "YES": "yes", "NO": "no"}

#: The line each of our market names implies. ou25 is 2.5 by definition; the others say it.
MARKET_LINE = {"ou25": 2.5, "over15": 1.5, "over35": 3.5, "btts": None}

#: Tokens that distinguish clubs and must match. Same principle as src/team_names: a name is
#: resolved on its identity token, never on a prefix, so Manchester City never matches
#: Manchester United.
_NOISE = {"fc", "cf", "afc", "sc", "ac", "club", "de", "the", "1", "calcio", "sv", "bk", "if",
          "cd", "ud", "ca", "ss", "us", "as", "fk", "sk", "nk", "og", "sd", "rc"}


def _tokens(name: str) -> set[str]:
    t = re.sub(r"[^\w\s]", " ", str(name).lower())
    return {w for w in t.split() if w and w not in _NOISE}


def match_team(ours: str, candidates: list[str]) -> tuple[str | None, str]:
    """Resolve one of our club names against Cloudbet's, or refuse.

    Refuses on ambiguity rather than picking the best score. A wrong fixture is a bet on the
    wrong match, which no amount of edge compensates for.
    """
    ot = _tokens(ours)
    if not ot:
        return None, "our name has no identity tokens"
    scored = []
    for c in candidates:
        ct = _tokens(c)
        if not ct:
            continue
        inter = ot & ct
        if not inter:
            continue
        # Jaccard on identity tokens — symmetric, so neither a long nor a short name wins by
        # length alone.
        scored.append((len(inter) / len(ot | ct), c))
    if not scored:
        return None, f"no candidate shares an identity token with {ours!r}"
    scored.sort(reverse=True)
    best, second = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
    if best[0] < 0.34:
        return None, f"best candidate {best[1]!r} shares too little with {ours!r}"
    if second[1] is not None and best[0] - second[0] < 0.17:
        return None, (f"ambiguous: {best[1]!r} and {second[1]!r} score {best[0]:.2f} vs "
                      f"{second[0]:.2f} for {ours!r} — refusing rather than guessing")
    return best[1], ""


@dataclass(frozen=True)
class Quote:
    odds: float
    market_url: str
    market_key: str
    outcome: str
    line: float | None


def discover_markets(event: dict) -> dict:
    """Every market key in a real payload, with its outcomes. Run this once with a live key and
    the MARKET_MAP above stops being a hypothesis."""
    out: dict[str, list] = {}
    for key, m in (event.get("markets") or {}).items():
        outs = set()
        for sub in (m.get("submarkets") or {}).values():
            for sel in (sub.get("selections") or []):
                o = sel.get("outcome")
                if o:
                    outs.add(str(o))
        out[str(key)] = sorted(outs)
    return out


def _submarket_params(name: str) -> dict:
    """Cloudbet keys submarkets by their params, e.g. "total=2.5". Parsed, not assumed."""
    d = {}
    for part in str(name).split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def price_for(event: dict, market: str, side: str,
              line: float | None = None) -> tuple[Quote | None, str]:
    """The executable price for one selection, or (None, reason).

    The reason names what the payload DID contain, so an unmapped market is diagnosable from a
    log line instead of a debugging session.
    """
    if market not in MARKET_MAP:
        return None, f"{market!r} is not mapped to a Cloudbet market"
    key, param = MARKET_MAP[market]
    outcome = SIDE_MAP.get(str(side).upper())
    if outcome is None:
        return None, f"side {side!r} has no Cloudbet outcome"
    want_line = MARKET_LINE.get(market) if line is None else line

    markets = event.get("markets") or {}
    m = markets.get(key)
    if m is None:
        return None, (f"market {key!r} absent; payload had: "
                      f"{sorted(markets)[:12] or 'no markets at all'}")

    for sub_name, sub in (m.get("submarkets") or {}).items():
        params = _submarket_params(sub_name)
        if param is not None:
            got = params.get(param)
            if got is None:
                continue
            try:
                if abs(float(got) - float(want_line)) > 1e-9:
                    continue
            except (TypeError, ValueError):
                continue
        for sel in (sub.get("selections") or []):
            if str(sel.get("outcome", "")).lower() != outcome:
                continue
            # A selection that is not currently tradeable must never be treated as a price.
            if str(sel.get("status", "SELECTION_ENABLED")).upper() not in (
                    "SELECTION_ENABLED", "ENABLED", "ACTIVE", ""):
                continue
            try:
                price = float(sel.get("price"))
            except (TypeError, ValueError):
                continue
            if price <= 1.0:
                continue
            url = f"{key}/{outcome}"
            if param is not None:
                url += f"?{param}={want_line}"
            return Quote(price, url, key, outcome, want_line), ""

    lines = sorted((m.get("submarkets") or {}).keys())[:12]
    return None, (f"{key!r} present but no tradeable {outcome!r} at {param}={want_line}; "
                  f"submarkets: {lines or 'none'}")


def find_event(events: list[dict], home: str, away: str) -> tuple[dict | None, str]:
    """Our fixture -> Cloudbet event, refusing ambiguity on either club."""
    homes = [str((e.get("home") or {}).get("name", "")) for e in events]
    aways = [str((e.get("away") or {}).get("name", "")) for e in events]
    h, hr = match_team(home, homes)
    if h is None:
        return None, f"home: {hr}"
    a, ar = match_team(away, aways)
    if a is None:
        return None, f"away: {ar}"
    for e in events:
        if (str((e.get("home") or {}).get("name", "")) == h
                and str((e.get("away") or {}).get("name", "")) == a):
            return e, ""
    # Both clubs resolved but not to the SAME event — two different fixtures. Never bet that.
    return None, (f"{home!r}->{h!r} and {away!r}->{a!r} resolved to different events; "
                  f"refusing rather than betting the wrong fixture")
