"""
FORWARD standard side-market odds capture (open -> moving -> closing).
=====================================================================
Sibling of capture_nf_odds_forward.py, for the STANDARD 2nd-division leagues.
`output/standard_sidemarket_odds_history.csv` was BACKFILL-ONLY (static since 2026-05-18) —
no scheduled job appended it, so standard BTTS / O/U 1.5 / 3.5 had NO forward odds at all.
This fetches TODAY's odds for UPCOMING standard fixtures and APPENDS every DISTINCT price, so
each (fixture, market) accumulates a real open->moving->closing curve as the season plays.

FREE: uses API-Football /odds (Bet365, id 8) on the APIFOOTBALL_KEY quota — NO OddsAPI credits.
ISOLATION: writes ONLY output/standard_sidemarket_odds_history.csv (standard side/main odds).
Does not train/predict any model; does not touch new-format, props, or shared model code.

Markets captured (match the existing schema): btts_yes/no, over25/under25, over15/under15,
over35/under35. Dedup keeps every DISTINCT price per (match_date, match, market) -> movement.
"""
import os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
import config
from player_model.api_football import get_upcoming_fixtures

_KEY = os.getenv("APIFOOTBALL_KEY", "")
_BASE = "https://v3.football.api-sports.io"
_HEADERS = {"x-apisports-key": _KEY}
BET365 = 8
OUT = PROJ / "output" / "standard_sidemarket_odds_history.csv"
COLS = ["snapshot_date", "snapshot_ts", "match_date", "kickoff_utc", "league",
        "match", "market", "odds"]
NEXT_N = 12          # look-ahead fixtures per league
_MIN_QUOTA = int(os.getenv("MIN_QUOTA", "5000"))
# Floor below which this capture stops and leaves the rest of the day's quota alone.
#
# Was 300, sized for the old Pro tier (7,500/day) where 300 was a sensible 4% reserve. The plan
# is now Ultra (75,000/day) and this script may run in a LOOP, so 300 would let a capture drain
# the day to within 0.4% of the cap and starve predict, player_props and live_scanner — the
# consumers that actually send tips. 5,000 is ~7% of the Ultra cap and comfortably more than a
# full day of predict enrichment, so the tip path can always finish.



def _ordered(df):
    """Write COLS in their declared order, keeping any unexpected column rather than dropping it.

    pd.concat appends a NEW column at the END, so adding kickoff_utc to an existing 7-column
    archive would leave the file's order permanently out of step with COLS. No reader depends on
    position — they all select by name — but a declared schema that does not match the file is a
    trap for the next person, and dropping unknown columns would silently lose data someone else
    added. So: COLS first, extras after.
    """
    cols = [c for c in COLS if c in df.columns] + [c for c in df.columns if c not in COLS]
    return df[cols]

# Only price fixtures kicking off within this many hours. None = no filter (the WIDE run).
# Set from --max-hours / MAX_HOURS so one script serves both cadences.
MAX_HOURS: float | None = None
if "--max-hours" in sys.argv:
    try:
        MAX_HOURS = float(sys.argv[sys.argv.index("--max-hours") + 1])
    except (IndexError, ValueError):
        MAX_HOURS = None
elif os.getenv("MAX_HOURS"):
    try:
        MAX_HOURS = float(os.environ["MAX_HOURS"])
    except ValueError:
        MAX_HOURS = None


def _hours_to_kickoff(raw_ko: str, now) -> float | None:
    """Hours until kickoff, or None when the timestamp is unusable.

    None means "cannot tell", and the caller keeps such a fixture rather than dropping it: a
    wasted odds call costs one credit, a silently missing curve costs the fixture.
    """
    if not raw_ko:
        return None
    try:
        ko = datetime.fromisoformat(str(raw_ko).replace("Z", "+00:00"))
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return (ko - now).total_seconds() / 3600.0
    except Exception:
        return None


def _iso_kickoff(raw_ko: str) -> str:
    """Kickoff as a normalised UTC ISO timestamp, or "" when unusable.

    WHY THIS COLUMN EXISTS. These archives recorded only `match_date`, i.e. the DAY. So the whole
    reason the NEAR capture exists — T-1h / T-30m / T-10m resolution into kickoff — was
    UNVERIFIABLE: you could count snapshots but never measure how close to kickoff any of them
    landed. On 2026-08-19 an attempt to check near-kickoff coverage had to be abandoned because the
    only archive carrying a kickoff time was one day old (a single kicked-off fixture in it).

    Empty string, never a guess: an invented kickoff would produce a confident wrong lead time,
    which is worse than a blank one (invariant 9 — write NaN, never an invented number).
    """
    if not raw_ko:
        return ""
    try:
        ko = datetime.fromisoformat(str(raw_ko).replace("Z", "+00:00"))
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return ko.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""

_OU_EXCLUDE = ("Corner", "Card", "Team", "Player", "Shot", "Foul", "Handicap")

# FULL-MATCH BTTS ONLY — see _is_fullmatch_btts. Duplicated verbatim in
# scripts/capture_nf_odds_forward.py, src/api_football_ou.py and scripts/backfill_af_odds.py;
# all four had the same defect and must not diverge.
_BTTS_BET_ID = 8
_BTTS_DISQUALIFY = ("Half", "1st", "2nd", "/", "Total Goals", "Corner", "Card",
                    "Player", "Shot", "Foul", "Handicap", "Minute")
_BTTS_EXACT = ("Both Teams Score", "Both Teams To Score", "BTTS")


def _is_fullmatch_btts(bet: dict) -> bool:
    """True only for the 90-minute Both-Teams-To-Score market.

    THE BUG THIS REPLACES (found 2026-08-23). The test was:

        if "Both Teams Score" in name or "BTTS" in name:

    a bare substring match. Bet365 offers at least three bets containing that substring, and
    the parse loop assigns on EVERY match, so the LAST one in the payload wins:

        id  8  Both Teams Score                 Yes=1.91  No=1.80   <- what we want
        id 34  Both Teams Score - First Half    Yes=5.50  No=1.14   <- overwrote it
        id 24  Results/Both Teams Score         Home/Yes=5.00 ...    <- harmless, compound labels

    So `btts_yes` was silently the FIRST-HALF price whenever bet 34 was offered. It is hard to
    catch by eye because the pair is internally perfect — the yes/no overround on the corrupted
    rows is a textbook 1.04, identical to the clean rows — and the price is real and stable all
    the way to kickoff. It just answers a different question. 271 of 6,358 btts_yes rows (4.3%)
    sit at >= 3.0, i.e. an implied <= 33% chance both teams score across 90 minutes, which does
    not occur in professional football; first-half BTTS at ~15% is exactly that shape.

    Matching on the numeric bet id is the primary test because it cannot be broken by a rename.
    The name test is a fallback for payloads without ids, and requires an EXACT match after
    disqualifying every qualifier ("Half", "/", "Total Goals", ...) rather than a substring.
    """
    if bet.get("id") == _BTTS_BET_ID:
        return True
    name = (bet.get("name") or "").strip()
    if any(x in name for x in _BTTS_DISQUALIFY):
        return False
    return name in _BTTS_EXACT


def _sanitize_ou(out: dict) -> dict:
    """Drop FT GOAL O/U odds that are internally impossible (a non-goal O/U market leaked in,
    a line was mislabelled, or a BTTS price was copied). Keeps btts / ht_* / h2h. Guards CLV:
    the std side-market archive had 21.9% of fixtures with over25>=over35 (audit H3)."""
    for ok, bk in (("over25", "btts_yes"), ("under25", "btts_no")):
        if ok in out and bk in out and out[ok] == out[bk]:
            out.pop(ok, None)
    ov = [out[k] for k in ("over15", "over25", "over35") if k in out]
    if len(ov) >= 2 and any(ov[i] >= ov[i + 1] for i in range(len(ov) - 1)):
        for k in ("over15", "over25", "over35", "under15", "under25", "under35"):
            out.pop(k, None)
    return out


def _parse_all_odds(data: dict) -> dict:
    """Bet365 BTTS yes/no + O/U 1.5/2.5/3.5 (over+under) -> {market: odds}."""
    out = {}
    for entry in data.get("response", []):
        for bk in entry.get("bookmakers", []):
            if bk.get("id") != BET365:
                continue
            for bet in bk.get("bets", []):
                name = bet.get("name", "")
                vals = bet.get("values", [])
                if _is_fullmatch_btts(bet):
                    for v in vals:
                        try:
                            if v.get("value") == "Yes":
                                out["btts_yes"] = float(v["odd"])
                            elif v.get("value") == "No":
                                out["btts_no"] = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            pass
                elif ("Over/Under" in name and ("First Half" in name or "1st Half" in name)
                        and not any(x in name for x in _OU_EXCLUDE)):
                    # HT model: first-half GOALS O/U 0.5 / 1.5 -> open->moving->closing capture
                    for v in vals:
                        label = v.get("value", "")
                        try:
                            odd = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        for ln, key in (("0.5", "05"), ("1.5", "15")):
                            if f"Over {ln}" in label:
                                out[f"ht_over{key}"] = odd
                            elif f"Under {ln}" in label:
                                out[f"ht_under{key}"] = odd
                elif ("Over/Under" in name and "Half" not in name
                        and not any(x in name for x in _OU_EXCLUDE)):   # FT GOALS O/U only
                    for v in vals:
                        label = v.get("value", "")
                        try:
                            odd = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        for ln, key in (("1.5", "15"), ("2.5", "25"), ("3.5", "35")):
                            if f"Over {ln}" in label:
                                out[f"over{key}"] = odd
                            elif f"Under {ln}" in label:
                                out[f"under{key}"] = odd
                elif name == "Match Winner":            # 1X2 (h2h) — free, same /odds call
                    for v in vals:
                        try:
                            odd = float(v["odd"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        val = v.get("value")
                        if val == "Home":
                            out["h2h_home"] = odd
                        elif val == "Draw":
                            out["h2h_draw"] = odd
                        elif val == "Away":
                            out["h2h_away"] = odd
    return _sanitize_ou(out)


def _fetch_odds(fixture_id: int) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(f"{_BASE}/odds", headers=_HEADERS,
                             params={"fixture": fixture_id, "bookmaker": BET365}, timeout=20)
            if r.status_code == 429:
                time.sleep(15); continue
            remaining = int(r.headers.get("x-ratelimit-requests-remaining", 9999))
            if remaining < _MIN_QUOTA:
                print(f"  [!] quota low ({remaining}) — stopping"); return {"_stop": True}
            if r.status_code != 200:
                return {}
            data = r.json()
            if data.get("errors"):
                if "rateLimit" in str(data["errors"]) or "requests" in str(data["errors"]).lower():
                    time.sleep(15); continue
                return {}
            return _parse_all_odds(data)
        except Exception:
            if attempt == 2:
                return {}
            time.sleep(2)
    return {}


def run() -> int:
    if not _KEY:
        print("[std_odds] APIFOOTBALL_KEY not set — skipping"); return 0
    now = datetime.now(timezone.utc)
    snap_date = now.strftime("%Y-%m-%d")
    snap_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    leagues = sorted(set(config.STANDARD_FORMAT_LEAGUES) & set(config.API_FOOTBALL_IDS))
    rows, stop = [], False
    for league in leagues:
        if stop:
            break
        lid = config.API_FOOTBALL_IDS[league]
        season = config.API_FOOTBALL_SEASONS.get(league, str(now.year))
        try:
            fixtures = get_upcoming_fixtures(lid, season, next_n=NEXT_N)
        except Exception as e:
            print(f"  {league}: fixtures fetch failed ({e})"); continue
        n_lg = 0
        n_skipped_far = 0
        for fx in fixtures:
            fid = fx.get("fixture", {}).get("id")
            raw_ko = fx.get("fixture", {}).get("date", "") or ""
            mdate = raw_ko[:10]
            teams = fx.get("teams", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            if not (fid and home and away):
                continue

            # KICKOFF-PROXIMITY FILTER (--max-hours).
            #
            # A window costs ~350 calls because it prices NEXT_N fixtures per league regardless
            # of kickoff, and a fixture ten days out has not moved. Nearly all of that spend
            # buys nothing. The fixtures that matter are the imminent ones, and there are very
            # few of them — roughly 24 kick off per day across all leagues, so about 12 inside
            # any 12-hour window.
            #
            # So the same budget buys two different runs:
            #   WIDE  no filter, a few times a day  -> the far horizon (T-7d, T-3d, T-24h)
            #   NEAR  --max-hours 12, frequently    -> T-6h ... T-30m, T-10m, close
            # The near run is cheap precisely because it is selective, which is what makes
            # 10-minute resolution into kickoff affordable at all.
            if MAX_HOURS is not None:
                hrs = _hours_to_kickoff(raw_ko, now)
                # Unknown kickoff is KEPT: dropping it would silently lose a fixture, and a
                # wasted odds call is cheaper than a missing curve.
                if hrs is not None and not (0 <= hrs <= MAX_HOURS):
                    n_skipped_far += 1
                    continue
            odds = _fetch_odds(fid)
            if odds.get("_stop"):
                stop = True; break
            for market, odd in odds.items():
                rows.append({"snapshot_date": snap_date, "snapshot_ts": snap_ts,
                             "match_date": mdate,
                             "kickoff_utc": _iso_kickoff(raw_ko),
                             "league": league,
                             "match": f"{home} vs {away}", "market": market, "odds": odd})
            n_lg += 1
        _far = f", {n_skipped_far} beyond {MAX_HOURS}h skipped" if MAX_HOURS is not None else ""
        print(f"  {league}: {n_lg} upcoming fixtures priced{_far}")
    if not rows:
        print("[std_odds] no rows captured"); return 0
    new = pd.DataFrame(rows, columns=COLS)
    if OUT.exists():
        new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
    # keep price CHANGES (consecutive-distinct) per (fixture, market) -> open..moving..close
    # curve. A plain distinct-dedup would drop a reverted closing price (1.90->1.95->1.90 loses
    # the true 1.90 close); this keeps every change AND the true last snapshot.
    new = new.sort_values(["match_date", "match", "market", "snapshot_ts"])
    _prev = new.groupby(["match_date", "match", "market"])["odds"].shift()
    new = new[new["odds"].ne(_prev)].sort_values("snapshot_ts")
    _ordered(new).to_csv(OUT, index=False)
    print(f"[std_odds] appended -> {OUT.name} (total {len(new):,})")
    return len(rows)


if __name__ == "__main__":
    run()
