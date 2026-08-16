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
  sot3    → player_shots_on_target       (Over 2.5)
  assists → player_assists (Over 0.5)    (1+ assist; falls back to calibration if no market)
  (goals3/sot4/cards removed — base rate too low / too noisy)

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
_REGIONS = os.getenv("PROP_ODDS_REGIONS", "uk,eu").strip() or "uk,eu"

# An empty bookmaker list means no book priced THIS event. That does not change
# within the hour, so it is cached far longer than a real price. At the old flat
# 2h TTL we re-probed — and re-paid for — the same dead leagues all day long.
CACHE_TTL_PRICED = 7200    # 2h  — live prices move, keep it short
CACHE_TTL_EMPTY = 86400    # 24h — nobody priced it; don't keep asking

# Maps our market names → Odds API market key + Over point (None = binary yes/no)
_MARKET_MAP = {
    "goals":   ("player_goal_scorer_anytime",  None),  # 1+ goals
    "goals2":  ("player_to_score_2_or_more",   None),  # 2+ goals
    "sot":     ("player_shots_on_target",      0.5),   # 1+ SOT
    "sot2":    ("player_shots_on_target",      1.5),   # 2+ SOT
    "sot3":    ("player_shots_on_target",      2.5),   # 3+ SOT
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
    "Ligue 1":          "soccer_france_ligue_1",
    "Championship":     "soccer_efl_champ",
    "League One":       "soccer_england_league1",
    "Bundesliga 2":     "soccer_germany_bundesliga2",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League":    "soccer_uefa_europa_league",
    "Conference League":"soccer_uefa_europa_conference_league",
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


def _coverage_record(cov: dict, sport_key: str, today: str, priced: bool) -> None:
    c = cov.setdefault(sport_key, {})
    c["league"] = SPORT_KEY_TO_LEAGUE.get(sport_key, sport_key)
    c["probes"] = int(c.get("probes", 0)) + 1
    c["last_probe"] = today
    if priced:
        c["empty_streak"] = 0
        c["last_priced"] = today
        c["priced_events"] = int(c.get("priced_events", 0)) + 1
    else:
        c["empty_streak"] = int(c.get("empty_streak", 0)) + 1


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
    cols = ["snapshot_date", "snapshot_ts", "match_date", "league", "match", "player", "market", "odds"]
    df = df.reindex(columns=cols)
    _ODDS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _ODDS_HISTORY_FILE.exists():
        try:
            df = pd.concat([pd.read_csv(_ODDS_HISTORY_FILE), df], ignore_index=True)
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
    for _, row in signals_df.drop_duplicates(["league", "match"]).iterrows():
        league = row.get("league", "")
        sport_key = PROP_SPORT_KEYS.get(league)
        if not sport_key:
            continue
        parts = str(row.get("match", "")).split(" vs ")
        if len(parts) != 2:
            continue
        needed.setdefault(sport_key, set()).add((_norm(parts[0]), _norm(parts[1])))

    if not needed:
        return {}

    result: dict[str, float] = {}
    history_records: list[dict] = []

    from datetime import datetime as _dt, timezone as _tz
    today = _dt.now(_tz.utc).strftime("%Y-%m-%d")
    cov = _load_coverage()

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

        for event in events:
            h = _norm(event.get("home_team", ""))
            a = _norm(event.get("away_team", ""))
            if (h, a) not in match_set and (a, h) not in match_set:
                continue

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
                        ttl = CACHE_TTL_PRICED if obj["bookmakers"] else CACHE_TTL_EMPTY
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
                _coverage_record(cov, sport_key, today, bool(bookmakers))
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
