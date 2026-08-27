"""
Player Prop Odds Fetcher
========================
Fetches player-level prop odds from The Odds API and returns a dict
compatible with enrich_with_odds(): {"PlayerName|market": best_decimal_odds}

Markets fetched:
  goals   → player_goal_scorer_anytime   (1+ goals)
  goals2  → player_to_score_2_or_more    (2+ goals)
  sot     → player_shots_on_target       (Over 0.5)
  sot2    → player_shots_on_target       (Over 1.5)
  sot3    → player_shots_on_target       (Over 2.5 = 3+ SOT)
  sot4    → player_shots_on_target       (Over 3.5 = 4+ SOT; research only, 0.27% base rate)
  assists → player_assists (Over 0.5)    (1+ assist; falls back to calibration if no market)
  cards   → player_to_receive_card
  (goals3 removed — base rate too low. sot4 RESTORED 2026-08-27: it rides the same
   player_shots_on_target response, so the price costs nothing, and a prop price is pre-match
   only. goals2 stays MODELLED but unpriceable — no source sells a player 2+ goals market.)

Two price sources, one set of files: OddsAPI first, then player_model/prop_odds_af for the
leagues OddsAPI does not sell props for. See that module's docstring.

Cost: 1 API call per event (events list is free).
Only fetches events where we have signals in player_tips.csv.
"""
from __future__ import annotations

import re
import unicodedata
import os
import json
from pathlib import Path
from typing import Optional

import requests

_BASE = "https://api.the-odds-api.com/v4"

# Bookmaker regions to request. EU books barely price player props outside the top
# five leagues — the deep prop market for English and German football is the UK
# books (bet365, Sky Bet, William Hill, Paddy Power). Asking for "eu" alone is why
# 9 of the 12 leagues in PROP_SPORT_KEYS had never returned a single price by
# 2026-08-16, and why the World Cup accounted for 91% of all prop odds ever fetched.
# OddsAPI bills [markets x regions], so adding a region roughly doubles per-event
# cost; at our volume (a few hundred calls in seven weeks) that is immaterial.
#
# WIDENED TO uk,eu,us,us2 ON 2026-08-27, because "uk,eu" was collecting under half of what is
# available. OddsAPI's own documentation says prop coverage is "currently limited to US
# bookmakers", and measuring one Premier League fixture confirms it:
#
#     uk,eu          ->  4 books,   459 player quotes   <- what we were asking for
#     us             ->  8 books,   478
#     uk,eu,us,us2   -> 15 books, 1,155                 <- 2.5x the prices
#
# More books is not just more rows: it is the difference between a single price and an actual
# consensus, which is what any no-vig or best-executable-price calculation needs. Cost roughly
# doubles per event and the monthly plan is at 67.9% with 4 days left, so there is headroom;
# revert via the env var if usage tightens.
_REGIONS = os.getenv("PROP_ODDS_REGIONS", "uk,eu,us,us2").strip() or "uk,eu,us,us2"

# How long to trust a cached response. An empty bookmaker list is NOT proof the
# fixture will never be priced: books post prop markets progressively as kickoff
# approaches. Hull City v Man United carried only anytime-goalscorer 7 days out,
# while Espanyol v Levante carried all 6 markets 3 days out. So an early empty
# means "not posted yet" and can be held a long time, but once the fixture is
# close an empty must be re-checked often or we miss the window entirely.
CACHE_TTL_PRICED = 7200       # 2h  — live prices move, keep it short
CACHE_TTL_EMPTY_NEAR = 7200   # 2h  — inside 48h of kickoff, markets are appearing
CACHE_TTL_EMPTY_FAR = 86400   # 24h — days out, an empty list says "too early"
_NEAR_KICKOFF_HOURS = 48


def hours_to_kickoff(commence_time: str) -> float | None:
    """Hours until kickoff, or None when the timestamp is unusable.

    Shared by the cache TTL and the coverage park streak so the two cannot disagree about how
    far out a probe was. None means UNKNOWN and callers must treat it as such — never as zero.
    """
    from datetime import datetime, timezone
    try:
        ko = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
    except Exception:
        return None
    return (ko - datetime.now(timezone.utc)).total_seconds() / 3600.0


def _empty_ttl_for(commence_time: str) -> int:
    """TTL for an empty bookmaker list, scaled by time to kickoff."""
    hours = hours_to_kickoff(commence_time)
    if hours is None:
        return CACHE_TTL_EMPTY_NEAR      # unknown kickoff -> re-check sooner
    return CACHE_TTL_EMPTY_NEAR if hours <= _NEAR_KICKOFF_HOURS else CACHE_TTL_EMPTY_FAR

# Maps our market names → Odds API market key + Over point (None = binary yes/no)
_MARKET_MAP = {
    "goals":   ("player_goal_scorer_anytime",  None),  # 1+ goals
    # goals2 / `player_to_score_2_or_more` RETIRED 2026-08-18. OddsAPI rejects it as an
    # invalid market on every soccer event — 0 prices in the entire odds history — and the
    # self-heal below then dropped it and re-requested, costing an extra call per event.
    # Its model cap (0.0878) was also below config.MIN_SIGNAL_PROB, so it could never tip.
    "sot":     ("player_shots_on_target",      0.5),   # 1+ SOT
    "sot2":    ("player_shots_on_target",      1.5),   # 2+ SOT
    "sot3":    ("player_shots_on_target",      2.5),   # 3+ SOT
    # 4+ SOT. Arrives on the SAME player_shots_on_target response as the lines above,
    # so it costs no extra API call. Research data only -- see the base rates in
    # player_model/config.py next to MARKETS.
    "sot4":    ("player_shots_on_target",      3.5),   # 4+ SOT
    "assists": ("player_assists",              0.5),   # Over 0.5 = 1+ assist (OddsAPI key is player_assists, Over/Under)
    "cards":   ("player_to_receive_card",      None),  # yellow card
}
# Unique API market keys to request in one call (avoids requesting same key multiple times)
_API_MARKETS_STR = ",".join(dict.fromkeys(v[0] for v in _MARKET_MAP.values()))

# Markets not available for WC on OddsAPI (returns 422 if requested)
_WC_EXCLUDED_API_MARKETS = {"player_to_score_2_or_more", "player_assists"}
_WC_API_MARKETS_STR = ",".join(
    k for k in dict.fromkeys(v[0] for v in _MARKET_MAP.values())
    if k not in _WC_EXCLUDED_API_MARKETS
)
_WC_SPORT_KEY = "soccer_fifa_world_cup"

# Odds API sport keys for our prop leagues
PROP_SPORT_KEYS = {
    "World Cup":        "soccer_fifa_world_cup",
    "Premier League":   "soccer_epl",
    "Bundesliga":       "soccer_germany_bundesliga",
    "La Liga":          "soccer_spain_la_liga",
    "Serie A":          "soccer_italy_serie_a",
    # FIXED 2026-08-19. Was "soccer_france_ligue_1", which OddsAPI answers with
    # 404 UNKNOWN_SPORT — so Ligue 1 player props have NEVER been fetchable, silently, since
    # the key was written. The real key is "ligue_one". Confirmed against the free /v4/sports
    # endpoint (no quota cost), which lists soccer_france_ligue_one / soccer_france_ligue_two.
    "Ligue 1":          "soccer_france_ligue_one",
    # UCL group stage has not started; the qualifiers are being played NOW and are a separate
    # sport key. Without this the competition is invisible until the group stage.
    "Champions League Qualifying": "soccer_uefa_champs_league_qualification",
    "Championship":     "soccer_efl_champ",
    "League One":       "soccer_england_league1",
    "Bundesliga 2":     "soccer_germany_bundesliga2",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League":    "soccer_uefa_europa_league",
    "Conference League":"soccer_uefa_europa_conference_league",
    # ADDED 2026-08-27. OddsAPI's documentation lists MLS as one of only six competitions it
    # sells soccer player props for -- "EPL, French Ligue 1, German Bundesliga, Italian Serie A,
    # Spanish La Liga, and MLS" -- and it was the one of those six we never asked for. A probe
    # returned 133 player quotes across 2 books on a single fixture. MLS is also our largest
    # league by settled bets, so this was the most expensive omission in the file.
    "USA MLS":          "soccer_usa_mls",
}

_CACHE_DIR = Path(__file__).resolve().parents[1] / "prop_odds_cache"
_CACHE_DIR.mkdir(exist_ok=True)

# Append-only history of every fetched player-prop odd (built forward, ~free —
# we already fetch these). Becomes the real-odds dataset for prop backtests.
_ODDS_HISTORY_FILE = Path(__file__).resolve().parents[1] / "output" / "player_prop_odds_history.csv"
SPORT_KEY_TO_LEAGUE = {v: k for k, v in PROP_SPORT_KEYS.items()}

# Learned per-league prop-market coverage. Rather than hardcoding a guess about
# which competitions offer player props, record what OddsAPI actually returns and
# park the leagues that never price anything. Committed by the props workflow so
# the knowledge survives across CI runs (the repo is the database).
_COVERAGE_FILE = Path(__file__).resolve().parents[1] / "output" / "prop_odds_coverage.json"
_COVERAGE_MIN_PROBES = 12   # consecutive events with no bookmaker before parking
_COVERAGE_RETRY_DAYS = 7    # ...and how long before we probe it again anyway


def validate_sport_keys(*, quiet: bool = False) -> dict[str, str]:
    """Check every PROP_SPORT_KEYS value against OddsAPI's own catalogue.

    Exists because a wrong key fails SILENTLY in the worst possible way: the events call
    returns 404 UNKNOWN_SPORT, the fetcher logs one line and moves on, and the league simply
    never has props. "Ligue 1" carried soccer_france_ligue_1 — a key that does not exist —
    for the entire life of the file, and nothing surfaced it. A key that is merely
    out of season is DIFFERENT and fine: it is valid with active=False, which is how the
    World Cup and the UEFA competitions look right now.

    The /v4/sports endpoint is FREE and does not consume quota, so this can run on every
    props run without cost.

    Returns {league: verdict} where verdict is VALID_ACTIVE / VALID_INACTIVE / UNKNOWN_KEY.
    """
    key = _load_odds_key()
    if not key:
        return {}
    try:
        r = requests.get(f"{_BASE}/sports", params={"apiKey": key, "all": "true"}, timeout=30)
        if r.status_code != 200:
            return {}
        catalogue = {s["key"]: s for s in r.json()}
    except Exception as e:
        if not quiet:
            print(f"[odds_fetcher] sport-key validation skipped ({e})")
        return {}

    out: dict[str, str] = {}
    bad: list[str] = []
    for league, sk in PROP_SPORT_KEYS.items():
        entry = catalogue.get(sk)
        if entry is None:
            out[league] = "UNKNOWN_KEY"
            bad.append(f"{league} -> {sk}")
        else:
            out[league] = "VALID_ACTIVE" if entry.get("active") else "VALID_INACTIVE"
    if bad and not quiet:
        print(f"[odds_fetcher] *** {len(bad)} sport key(s) DO NOT EXIST in OddsAPI — these "
              f"leagues can never return props: {bad} ***")
    return out


def _load_coverage() -> dict:
    try:
        return json.loads(_COVERAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_coverage(cov: dict) -> None:
    try:
        _COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _COVERAGE_FILE.write_text(
            json.dumps(cov, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception as e:
        print(f"[odds_fetcher] coverage save skipped: {e}")


def _coverage_parked(cov: dict, sport_key: str, today: str) -> bool:
    """True when a league has returned nothing but empty bookmaker lists for
    _COVERAGE_MIN_PROBES straight events and we last probed it under
    _COVERAGE_RETRY_DAYS ago. Parking is never permanent: a book that adds the
    market is picked up on the next weekly retry, so this answers 'does the
    Championship have props?' from evidence instead of assumption."""
    from datetime import date
    c = cov.get(sport_key)
    if not c or int(c.get("empty_streak", 0)) < _COVERAGE_MIN_PROBES:
        return False
    last = str(c.get("last_probe") or "")[:10]
    try:
        return (date.fromisoformat(today) - date.fromisoformat(last)).days < _COVERAGE_RETRY_DAYS
    except Exception:
        return False


def _coverage_fixtures(cov: dict, sport_key: str, today: str,
                       wanted: int, matched: int, unmatched: list[str]) -> None:
    """Record how many of the fixtures we wanted were even FOUND in OddsAPI's event
    list. Without this, 'no odds' has two indistinguishable causes: the book offers
    no props (empty bookmakers, an odds call is spent) or our fixture name never
    matched an event (no call is made at all, because the event loop just skips).
    Those need opposite fixes, so never let the second masquerade as the first."""
    c = cov.setdefault(sport_key, {})
    c["league"] = SPORT_KEY_TO_LEAGUE.get(sport_key, sport_key)
    c["wanted_fixtures"] = wanted
    c["matched_events"] = matched
    c["unmatched_sample"] = unmatched
    if wanted and not matched:
        c["last_all_unmatched"] = today
        print(f"[odds_fetcher] {sport_key}: NAME MISMATCH — {wanted} fixture(s) wanted, "
              f"0 found in the event list, so no odds call was made. "
              f"Unmatched: {unmatched}")


# An empty probe only counts toward PARKING if it was taken close enough to kickoff that a
# real market should already exist. See _coverage_record.
_COVERAGE_STREAK_WINDOW_H = 24.0


def _coverage_record(cov: dict, sport_key: str, today: str, priced: bool,
                     hours_to_kickoff: float | None = None) -> None:
    """Record one probe. An empty result only extends the PARK STREAK when it was taken
    inside `_COVERAGE_STREAK_WINDOW_H` of kickoff.

    WHY, because this was a self-fulfilling gap. Parking counted 12 consecutive empty probes
    with no notion of time-to-kickoff. Lower divisions post player props late, so probes taken
    three to five days out come back empty CORRECTLY — the market is not open yet. Twelve of
    those parked the league for seven days, parking skips it entirely (the `continue` in the
    fetch loop happens before the event call), and the weekly retry lands on whatever day it
    lands. Once parked, a league could never be probed on a match day again.

    Measured on 2026-08-27: Championship 24 empty probes and PARKED until 2026-09-01, with
    fixtures being played on 08-28. Premier League, which prices early, had 807 probes and had
    never parked. So the ledger was recording OUR PROBING SCHEDULE and presenting it as the
    market's behaviour — the opposite of what it exists for.

    Probes further out are still counted in `probes` and still logged; they just cannot park a
    league. `far_empty_probes` keeps them visible so "we looked 40 times, all of them early" is
    distinguishable from "we never looked".
    """
    c = cov.setdefault(sport_key, {})
    c["league"] = SPORT_KEY_TO_LEAGUE.get(sport_key, sport_key)
    c["probes"] = int(c.get("probes", 0)) + 1
    c["last_probe"] = today
    if priced:
        c["empty_streak"] = 0
        c["last_priced"] = today
        c["priced_events"] = int(c.get("priced_events", 0)) + 1
        return
    # Unknown time-to-kickoff is treated as TOO FAR: it must not be able to park a league,
    # because an unknown is not evidence that the market was open.
    near = hours_to_kickoff is not None and hours_to_kickoff <= _COVERAGE_STREAK_WINDOW_H
    if near:
        c["empty_streak"] = int(c.get("empty_streak", 0)) + 1
        c["last_near_empty"] = today
    else:
        c["far_empty_probes"] = int(c.get("far_empty_probes", 0)) + 1


def _append_odds_history(records: list[dict]) -> None:
    """Append fetched prop odds to a permanent CSV history. Keeps every distinct
    PRICE per real fixture (match_date+match+player+market), first time seen — so the
    full open->close line trajectory is preserved for closing-line-value (CLV) analysis.
    (Previously deduped per snapshot_DATE keep=last, which collapsed all intraday
    captures into one row and made CLV untestable. Fixed 2026-06-30 for forward CLV
    data collection — the hourly match-day predict run now logs each line move.)
    Callers wrap this in try/except so it can never break live odds fetching."""
    if not records:
        return
    import pandas as pd
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(records)
    df["snapshot_date"] = now.strftime("%Y-%m-%d")
    df["snapshot_ts"]   = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    # `source` and `bookmaker` are carried so the two price sources can be told apart after the
    # fact — the one thing that would otherwise be unrecoverable once the rows are merged. Rows
    # from OddsAPI arrive without them, so they default rather than becoming NaN: 38,189 existing
    # rows predate the second source and are all OddsAPI by definition.
    cols = ["snapshot_date", "snapshot_ts", "match_date", "league", "match", "player", "market",
            "odds", "source", "bookmaker"]
    if "source" not in df.columns:
        df["source"] = "oddsapi"
    else:
        df["source"] = df["source"].fillna("oddsapi")
    df = df.reindex(columns=cols)
    _ODDS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _ODDS_HISTORY_FILE.exists():
        try:
            prev = pd.read_csv(_ODDS_HISTORY_FILE)
            # Rows written before the second source existed carry no `source`. Stamped on READ
            # as oddsapi — true by construction, since API-Football was not a price source then.
            # Without this the dedup below (keep="first") lets the older blank-source row win and
            # every row in the file reads as unknown provenance, which is what happened on the
            # first attempt: 38,494 rows, all blank.
            if "source" not in prev.columns:
                prev["source"] = "oddsapi"
            else:
                prev["source"] = prev["source"].fillna("oddsapi")
            df = pd.concat([prev, df], ignore_index=True)
        except Exception:
            pass
    # Keep every distinct PRICE per real fixture (match_date disambiguates fixture-name
    # collisions across seasons) -> preserves the open->close line trajectory for CLV.
    df = df.drop_duplicates(
        subset=["match_date", "league", "match", "player", "market", "odds"], keep="first"
    )
    df.to_csv(_ODDS_HISTORY_FILE, index=False)


def _norm(name: str) -> str:
    """Normalize player name: lowercase, remove accents, collapse spaces."""
    nfkd = unicodedata.normalize("NFKD", name or "")
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_name.lower().strip())


# Club-name resolution for matching OUR fixture names onto OddsAPI events. This used to be
# plain equality on _norm(), which failed on every spelling variant and silently skipped the
# event — no odds call, no data, and indistinguishable from "no bookmaker offers props".
# Measured 2026-08-17 from prop_odds_coverage.json: 35 of 56 wanted fixtures (63%) never
# matched, including Bundesliga 2 at 0 of 8, which would have been parked as "no coverage"
# when in truth it was never asked. src/team_names.resolve is the same league-scoped,
# ambiguity-refusing matcher the team model already uses.
try:
    import sys as _sys
    if str(Path(__file__).resolve().parents[1]) not in _sys.path:
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.team_names import resolve as _resolve_club
except Exception as _e:                                      # pragma: no cover
    print(f"[odds_fetcher] team_names unavailable ({_e}); falling back to exact matching")
    _resolve_club = None


def _resolve_event(ev_home: str, ev_away: str,
                   wanted: set[tuple[str, str]]) -> Optional[tuple[str, str]]:
    """Map one OddsAPI event onto a fixture we want, or None.

    `wanted` holds ORIGINAL (home, away) names for a single league, so resolution is
    league-scoped exactly as team_names.resolve expects, and an ambiguous club is rejected
    rather than guessed. The (home, away) pair is re-checked after resolving each side so
    two clubs from *different* fixtures can never combine into a false match.
    """
    if not wanted:
        return None
    if _resolve_club is None:
        h, a = _norm(ev_home), _norm(ev_away)
        for (wh, wa) in wanted:
            if (_norm(wh), _norm(wa)) in {(h, a), (a, h)}:
                return (wh, wa)
        return None

    homes = [h for h, _ in wanted]
    aways = [a for _, a in wanted]

    rh, ra = _resolve_club(ev_home, homes), _resolve_club(ev_away, aways)
    if rh and ra and (rh, ra) in wanted:
        return (rh, ra)

    # OddsAPI occasionally lists a tie with the sides swapped relative to our fixture.
    rh2, ra2 = _resolve_club(ev_home, aways), _resolve_club(ev_away, homes)
    if rh2 and ra2 and (ra2, rh2) in wanted:
        return (ra2, rh2)
    return None


def _load_odds_key() -> str:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if key:
        return key
    # Fallback: .api_keys file in repo root (KEY=VALUE per line)
    root = Path(__file__).resolve().parents[1]
    api_keys_f = root / ".api_keys"
    if api_keys_f.exists():
        for line in api_keys_f.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _get(path: str, params: dict) -> Optional[dict | list]:
    key = _load_odds_key()
    if not key:
        return None
    params["apiKey"] = key
    r = requests.get(f"{_BASE}/{path}", params=params, timeout=15)
    remaining = r.headers.get("x-requests-remaining", "?")

    # Self-heal: if one or more requested markets aren't offered for this
    # sport/event, OddsAPI 422s the WHOLE request. Drop the invalid market(s)
    # named in the error and retry once so the valid markets still come back.
    if r.status_code == 422 and params.get("markets") and "nvalid market" in r.text:
        bad = set()
        for grp in re.findall(r"[Ii]nvalid markets?:\s*([a-z0-9_,\s]+)", r.text):
            bad.update(m.strip() for m in grp.split(",") if m.strip())
        kept = [m for m in params["markets"].split(",") if m not in bad]
        if bad and kept and len(kept) < len(params["markets"].split(",")):
            print(f"[odds_fetcher] {path}: dropping invalid markets {sorted(bad)}, retrying with {kept}")
            params["markets"] = ",".join(kept)
            r = requests.get(f"{_BASE}/{path}", params=params, timeout=15)
            remaining = r.headers.get("x-requests-remaining", "?")

    if r.status_code != 200:
        print(f"[odds_fetcher] {path}: HTTP {r.status_code} | {r.text[:120]}")
        return None
    print(f"[odds_fetcher] {path} OK | quota remaining: {remaining}")
    return r.json()


def _best_price(bookmakers: list, mkt_key: str, point: Optional[float]) -> dict[str, float]:
    """
    Extract best (highest) odds per player for a given market key.
    Returns {normalized_name: best_decimal_odds}.
    """
    best: dict[str, float] = {}
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt["key"] != mkt_key:
                continue
            for o in mkt.get("outcomes", []):
                # SOT: only take Over 0.5
                if point is not None:
                    if o.get("name") != "Over":
                        continue
                    if abs(float(o.get("point", 999)) - point) > 0.01:
                        continue
                player = _norm(o.get("description", ""))
                price = float(o.get("price", 0))
                if price > best.get(player, 0):
                    best[player] = price
    return best


def fetch_prop_odds(signals_df) -> dict[str, float]:
    """
    Fetch player prop odds for all matches in signals_df.
    signals_df must have columns: league, match ('{home} vs {away}'), date.
    Returns {"PlayerName|market": best_decimal_odds}.
    """
    if signals_df is None or signals_df.empty:
        return {}

    # Build set of (league, home, away) we need odds for
    needed: dict[str, set[tuple[str, str]]] = {}  # sport_key → set of (home_norm, away_norm)
    unmapped: dict[str, int] = {}
    for _, row in signals_df.drop_duplicates(["league", "match"]).iterrows():
        league = row.get("league", "")
        sport_key = PROP_SPORT_KEYS.get(league)
        if not sport_key:
            # No sport key = we never ask, so these players can NEVER be priced and are
            # AVOID by construction. On 2026-08-16 player_tips.csv carried Argentina
            # Primera, Austrian Bundesliga, China Super League, La Liga 2 and MLS — none
            # of them in PROP_SPORT_KEYS and none in config.PROP_LEAGUES either. They
            # arrive via the bets.csv supplement in predict, which is not restricted to
            # prop leagues. Surfacing the count makes that visible instead of silent.
            unmapped[str(league)] = unmapped.get(str(league), 0) + 1
            continue
        parts = str(row.get("match", "")).split(" vs ")
        if len(parts) != 2:
            continue
        # Keep the ORIGINAL names — _resolve_event needs them for club-name resolution.
        needed.setdefault(sport_key, set()).add((parts[0].strip(), parts[1].strip()))

    if unmapped:
        _n = sum(unmapped.values())
        print(f"[odds_fetcher] {_n} fixture(s) in leagues with no OddsAPI sport key — "
              f"never asked, so their players stay AVOID by construction: "
              f"{dict(sorted(unmapped.items(), key=lambda kv: -kv[1]))}")

    if not needed:
        return {}

    result: dict[str, float] = {}
    history_records: list[dict] = []

    from datetime import datetime as _dt, timezone as _tz
    today = _dt.now(_tz.utc).strftime("%Y-%m-%d")
    cov = _load_coverage()

    # Free check, so a typo can never again cost a league months of silence.
    _verdicts = validate_sport_keys()
    if _verdicts:
        _unknown = [lg for lg, v in _verdicts.items() if v == "UNKNOWN_KEY"]
        _inactive = [lg for lg, v in _verdicts.items() if v == "VALID_INACTIVE"]
        if _unknown:
            print(f"[odds_fetcher] BROKEN sport key(s): {_unknown}")
        if _inactive:
            print(f"[odds_fetcher] out of season (valid key, nothing to fetch): {_inactive}")

    for sport_key, match_set in needed.items():
        if _coverage_parked(cov, sport_key, today):
            print(f"[odds_fetcher] {sport_key}: parked — "
                  f"{cov[sport_key]['empty_streak']} consecutive events with no bookmaker, "
                  f"last probed {cov[sport_key]['last_probe']}; retries every "
                  f"{_COVERAGE_RETRY_DAYS}d")
            continue

        # Get event list (free, no quota cost)
        events = _get(f"sports/{sport_key}/events", {})
        if not events:
            continue

        _matched_pairs: set[tuple[str, str]] = set()

        for event in events:
            _hit = _resolve_event(event.get("home_team", ""), event.get("away_team", ""),
                                  match_set)
            if _hit is None:
                continue
            _matched_pairs.add(_hit)

            event_id = event["id"]
            cache_f = _CACHE_DIR / f"{event_id}.json"

            # Cache per event — timestamp stored inside JSON (not file mtime, which GitHub
            # Actions cache restore resets to extraction time, breaking the TTL check).
            import time
            bookmakers = None
            if cache_f.exists():
                try:
                    obj = json.loads(cache_f.read_text(encoding="utf-8"))
                    if isinstance(obj, dict) and "bookmakers" in obj:
                        ttl = (CACHE_TTL_PRICED if obj["bookmakers"]
                               else _empty_ttl_for(event.get("commence_time", "")))
                        if (time.time() - obj.get("fetched_at", 0)) < ttl:
                            bookmakers = obj["bookmakers"]
                    # Old format (plain list) — treat as expired; will re-fetch below
                except Exception:
                    pass

            if bookmakers is None:
                api_markets = _WC_API_MARKETS_STR if sport_key == _WC_SPORT_KEY else _API_MARKETS_STR
                data = _get(
                    f"sports/{sport_key}/events/{event_id}/odds",
                    {"regions": _REGIONS, "markets": api_markets, "oddsFormat": "decimal"},
                )
                if data is None:
                    continue
                bookmakers = data.get("bookmakers", [])
                _coverage_record(cov, sport_key, today, bool(bookmakers),
                                 hours_to_kickoff=hours_to_kickoff(
                                     event.get("commence_time", "")))
                if not bookmakers:
                    print(f"[odds_fetcher] {sport_key}: no bookmaker priced "
                          f'{event.get("home_team","")} v {event.get("away_team","")} '
                          f"(regions={_REGIONS})")
                cache_f.write_text(
                    json.dumps({"fetched_at": time.time(), "bookmakers": bookmakers}, ensure_ascii=False),
                    encoding="utf-8",
                )

            # Extract best odds per market (+ record every price for the history)
            _match_str  = f'{event.get("home_team", "")} vs {event.get("away_team", "")}'
            _match_date = str(event.get("commence_time", ""))[:10]
            _league     = SPORT_KEY_TO_LEAGUE.get(sport_key, sport_key)
            for our_market, (api_key, point) in _MARKET_MAP.items():
                prices = _best_price(bookmakers, api_key, point)
                for player_norm, price in prices.items():
                    key = f"{player_norm}|{our_market}"
                    if price > result.get(key, 0):
                        result[key] = price
                    history_records.append({
                        "match_date": _match_date, "league": _league, "match": _match_str,
                        "player": player_norm, "market": our_market, "odds": price,
                    })

        _coverage_fixtures(
            cov, sport_key, today, len(match_set), len(_matched_pairs),
            [f"{x} vs {y}" for x, y in sorted(match_set - _matched_pairs)][:8],
        )

    # ── SECOND SOURCE: API-Football, for the leagues OddsAPI does not sell props for ──────
    #
    # TWO SOURCES, ONE SET OF FILES. This fills the same `result` lookup and the same
    # `history_records` list, so player_prop_odds_history.csv, prop_odds_coverage.json and every
    # downstream consumer are unchanged. Same rule ARCHITECTURE.md sets for the main odds layer:
    # sources normalise to an identical shape, parsing differences live in the client module.
    #
    # OddsAPI's documented soccer prop coverage is EPL, Ligue 1, Bundesliga, Serie A, La Liga and
    # MLS — nothing else. Championship, League One, League Two and EFL Cup return zero prop books
    # on any region while returning 36-38 books for h2h, so the markets simply are not sourced.
    # API-Football carries them (Championship 22 prop markets, Bet365, 52 priced players on one
    # fixture) and its daily quota sits ~74,000/75,000 unused.
    #
    # Runs LAST and only for leagues OddsAPI left unpriced, so the primary source keeps priority
    # and the top five cost nothing extra.
    try:
        _af_added = _fill_from_api_football(signals_df, result, history_records, cov, today)
        if _af_added:
            print(f"[odds_fetcher] api-football second source: +{_af_added} quote(s)")
    except Exception as e:                                       # noqa: BLE001
        # Never take the primary source down: a failure here must leave OddsAPI's result intact.
        print(f"[odds_fetcher] api-football second source skipped ({type(e).__name__}: {e})")

    # Persist every fetched prop odd to the permanent forward-built history.
    try:
        _append_odds_history(history_records)
    except Exception as e:
        print(f"[odds_fetcher] odds-history save skipped: {e}")

    _save_coverage(cov)
    _live = sorted(k for k, c in cov.items() if c.get("priced_events"))
    print(f"[odds_fetcher] {len(result)} player-market prices | "
          f"leagues with any prop coverage: {[SPORT_KEY_TO_LEAGUE.get(k, k) for k in _live] or 'none'}")

    return result


def _fill_from_api_football(signals_df, result: dict[str, float],
                            history_records: list[dict], cov: dict, today: str) -> int:
    """Top up `result` and `history_records` from API-Football. Returns quotes added.

    Only touches leagues OddsAPI did NOT price. `result` keys stay in OddsAPI's normalised form
    (`_norm(player)|market`) so `match_odds_to_tips` keeps working untouched, and a price is only
    written when it BEATS what OddsAPI already found — the contract everywhere here is best
    available price, and a second source must not quietly lower one.
    """
    import pandas as pd

    from player_model import prop_odds_af as af

    if signals_df is None or getattr(signals_df, "empty", True):
        return 0
    if "league" not in signals_df.columns or "match" not in signals_df.columns:
        return 0

    priced_by_oddsapi = {SPORT_KEY_TO_LEAGUE.get(k, k) for k, c in cov.items()
                         if c.get("priced_events")}
    af_leagues = af.leagues_without_oddsapi(signals_df["league"].astype(str).unique(),
                                            priced_by_oddsapi)
    if not af_leagues:
        return 0

    # From player_model/config.py, not the repo-root `config` imported above — see
    # prop_odds_af._prop_leagues for why that distinction silently broke the first version.
    seasons = af._prop_seasons()
    ids = af._prop_leagues()
    wanted = signals_df[signals_df["league"].astype(str).isin(af_leagues)]
    date_col = next((c for c in ("match_date", "date") if c in wanted.columns), None)

    fixtures, seen = [], set()
    for _, row in wanted.iterrows():
        league = str(row["league"])
        match = str(row["match"])
        if " vs " not in match:
            continue
        home, away = match.split(" vs ", 1)
        md = str(row[date_col])[:10] if date_col else ""
        if (league, match, md) in seen:
            continue
        seen.add((league, match, md))
        lid = ids.get(league)
        if not lid or not md:
            continue
        # NOT api_football.find_fixture_id: that one filters status=FT for grading settled
        # matches, so it can never resolve a pre-match fixture. See the docstring there.
        fid = af.find_upcoming_fixture_id(int(lid), str(seasons.get(league, "")), md, home, away)
        if fid:
            fixtures.append({"fixture_id": fid, "league": league, "match": match,
                             "match_date": md})

    if not fixtures:
        print(f"[odds_fetcher] api-football: no fixture id resolved for {af_leagues}")
        return 0

    best, records = af.fetch(fixtures, verbose=False)
    added = 0
    for k, odd in best.items():
        player, _, market = k.rpartition("|")
        norm_key = f"{_norm(player)}|{market}"
        if odd > result.get(norm_key, 0.0):
            result[norm_key] = odd
            added += 1
    history_records.extend(records)

    # Same ledger, so coverage is answered from evidence for BOTH sources. Keyed by league name
    # rather than an OddsAPI sport key, because that is what this source is addressed by.
    for lg in af_leagues:
        got = sum(1 for r in records if r.get("league") == lg)
        c = cov.setdefault(f"apifootball:{lg}", {})
        c["league"] = lg
        c["source"] = "api_football"
        c["probes"] = int(c.get("probes", 0)) + 1
        c["last_probe"] = today
        if got:
            c["priced_events"] = int(c.get("priced_events", 0)) + 1
            c["last_priced"] = today
            c["empty_streak"] = 0
        else:
            c["empty_streak"] = int(c.get("empty_streak", 0)) + 1
    return added


def match_odds_to_tips(tips_df, odds_raw: dict[str, float]) -> dict[str, float]:
    """
    Map raw odds dict (normalized names) to tips_df player names.
    Returns {"{original_player_name}|{market}": odds} for rows in tips_df.
    """
    matched: dict[str, float] = {}
    for _, row in tips_df.iterrows():
        player = row.get("player_name", "")
        market = row.get("market", "")
        key_norm = f"{_norm(player)}|{market}"
        if key_norm in odds_raw:
            matched[f"{player}|{market}"] = odds_raw[key_norm]
    return matched
