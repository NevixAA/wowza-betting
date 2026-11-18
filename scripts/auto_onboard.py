"""Auto-onboard leagues that passed every Pro League Scout gate (owner standing approval 2026-10-20).

    python scripts/auto_onboard.py            # reads Pro's queue, updates config/auto_leagues.json
    python scripts/auto_onboard.py --dry-run

"If a new league is ready to become a tips generator, go." Pro's weekly League Scout queues a league
only when it passed data quality, market-residual evidence AND the existing-league regression harness
(output/scout/auto_onboard_queue.json). This script makes each queued league a tips generator by
writing it into config/auto_leagues.json, which config.py merges into every league registry.

FAIL CLOSED, per league:
  * no APPROVED OddsAPI sport key in the queue entry (or one OddsAPI does not list) -> skipped,
    reason reported: without a price source predict cannot produce a tip, and a guessed key would
    price a different league. Word matching (match_sport_key) only SUGGESTS keys in --probe.
  * no timezone for the league's country -> skipped (never 'UTC' by default)
  * already configured (by name or API-Football id) -> skipped
  * Pro queue unreachable -> nothing changes
Stakes are not decided here: a new league has no rating, so the stake plan's league-rating sizing
fails closed to its minimum scale until the league has results. The next retrain includes the league
under the champion gate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUEUE_URL = "https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main/output/scout/auto_onboard_queue.json"
AUTO = ROOT / "config" / "auto_leagues.json"
HISTORY_SEASONS = 4

#: Display timezone per country (local match dates in tips). A country missing here is not
#: onboarded until it is added - see league_timezone().
COUNTRY_TZ = {
    "England": "Europe/London", "Scotland": "Europe/London", "Wales": "Europe/London",
    "Ireland": "Europe/Dublin", "Northern-Ireland": "Europe/London", "France": "Europe/Paris",
    "Germany": "Europe/Berlin", "Spain": "Europe/Madrid", "Italy": "Europe/Rome", "Portugal": "Europe/Lisbon",
    "Netherlands": "Europe/Amsterdam", "Belgium": "Europe/Brussels", "Switzerland": "Europe/Zurich",
    "Austria": "Europe/Vienna", "Denmark": "Europe/Copenhagen", "Norway": "Europe/Oslo",
    "Sweden": "Europe/Stockholm", "Finland": "Europe/Helsinki", "Poland": "Europe/Warsaw",
    "Czech-Republic": "Europe/Prague", "Greece": "Europe/Athens", "Turkey": "Europe/Istanbul",
    "Romania": "Europe/Bucharest", "Hungary": "Europe/Budapest", "Croatia": "Europe/Zagreb",
    "Serbia": "Europe/Belgrade", "Ukraine": "Europe/Kyiv", "Russia": "Europe/Moscow",
    "Brazil": "America/Sao_Paulo", "Argentina": "America/Argentina/Buenos_Aires", "Chile": "America/Santiago",
    "Colombia": "America/Bogota", "Uruguay": "America/Montevideo", "Peru": "America/Lima",
    "Mexico": "America/Mexico_City", "USA": "America/New_York", "Canada": "America/Toronto",
    "Japan": "Asia/Tokyo", "South-Korea": "Asia/Seoul", "China": "Asia/Shanghai", "Australia": "Australia/Sydney",
    "Saudi-Arabia": "Asia/Riyadh", "Qatar": "Asia/Qatar", "Israel": "Asia/Jerusalem",
}
_STOP = {"league", "division", "liga", "the", "de", "1", "first", "football", "soccer", "premier", "pro"}


def _toks(s: str) -> set[str]:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return {t for t in re.split(r"[^a-z0-9]+", s) if t}


def match_sport_key(country: str, league: str, sports: list[dict]) -> tuple[str | None, str]:
    """SUGGESTION for --probe only: the OddsAPI soccer key whose title/description carries the
    COUNTRY and the league's identity words. Unique match only; anything ambiguous or absent returns
    (None, reason). Never used to onboard - plan() requires an approved key (approved_sport_key)."""
    c = _toks(country.replace("-", " "))
    lw = _toks(league) - c - _STOP
    if not lw:
        # "1. Division" / "First Division" leave no identity word, and would otherwise match the
        # country's only key - which is its TOP flight. Never price a league with another's odds.
        return None, "league name has no identifying word - refusing to guess"
    cands = []
    for s in sports:
        if not str(s.get("key", "")).startswith("soccer_"):
            continue
        hay = _toks(f"{s.get('title', '')} {s.get('description', '')} {s.get('key', '').replace('_', ' ')}")
        if c and not (c & hay):
            continue
        if lw and not lw <= hay:
            continue
        cands.append(s["key"])
    if len(cands) == 1:
        return cands[0], "matched"
    return None, ("no OddsAPI sport key for this league" if not cands
                  else f"ambiguous OddsAPI keys {cands}")


def approved_sport_key(q: dict, sports: list[dict]) -> tuple[str | None, str]:
    """The OddsAPI key APPROVED for this league in the queue entry (`odds_api_sport_key`), checked
    against OddsAPI's live sports list. No approved key -> (None, reason): the league is not
    onboarded. match_sport_key's word matching is only a SUGGESTION for --probe; a price source is
    never chosen by fuzzy matching (owner rule 2026-11-01)."""
    k = str(q.get("odds_api_sport_key") or "").strip()
    if not k:
        return None, "no approved OddsAPI sport key in the queue entry"
    if k not in {str(s.get("key", "")) for s in sports}:
        return None, f"approved OddsAPI key {k!r} is not in OddsAPI's sports list"
    return k, "approved"


def league_timezone(q: dict) -> str | None:
    """The league's display timezone: the queue's own `timezone` if given, else its country's zone.
    None when neither exists - the league is then NOT onboarded (no 'UTC' stand-in)."""
    return (str(q.get("timezone") or "").strip() or None) or COUNTRY_TZ.get(str(q.get("country") or ""))


def build_entry(q: dict, sport_key: str, now: str, tz: str) -> dict:
    season = str(q.get("current_season") or "")
    hist = [str(int(season) - k) for k in range(1, HISTORY_SEASONS + 1)] if season.isdigit() else []
    return {"name": q["league"], "api_football_id": int(q["league_id"]), "current_season": season,
            "history_seasons": hist, "odds_api_sport_key": sport_key,
            "timezone": tz,
            "onboarded_at": now, "scout_week": q.get("week"), "promotion_score": q.get("promotion_score"),
            "source": "Pro League Scout auto-onboard (owner standing approval 2026-10-20)"}


def plan(queue: list[dict], current: dict, configured_names: set, configured_ids: set,
         sports: list[dict], now: str) -> tuple[list[dict], list[dict]]:
    """Pure: which queued leagues to add, and why the others were skipped."""
    have_names = configured_names | {e["name"] for e in current.get("leagues", [])}
    have_ids = configured_ids | {int(e["api_football_id"]) for e in current.get("leagues", [])}
    add, skipped = [], []
    from src.league_registry import canonical_name
    for q in queue:
        # registered under any alias counts as configured (registry lookup, not a raw ==)
        if (q["league"] in have_names or canonical_name(q["league"]) is not None
                or int(q["league_id"]) in have_ids):
            skipped.append({"league": q["league"], "reason": "already configured"})
            continue
        key, why = approved_sport_key(q, sports)
        if key is None:
            skipped.append({"league": q["league"], "reason": why})
            continue
        tz = league_timezone(q)
        if tz is None:
            skipped.append({"league": q["league"],
                            "reason": f"no timezone for country {q.get('country')!r} - refusing to assume UTC"})
            continue
        add.append(build_entry(q, key, now, tz))
    return add, skipped


def _notify(added: list[dict]) -> None:
    try:
        import requests
        from telegram_bot.notifier import _load_config, _load_notified, _save_notified
        cfg = _load_config()
        token, chat = cfg.get("token", ""), cfg.get("chat_id", "")
        if not token or token == "YOUR_BOT_TOKEN":
            return
        notified = _load_notified()
        for e in added:
            key = f"AUTO_ONBOARD|{e['name']}"
            if key in notified:
                continue
            text = (f"🆕 <b>New league added: {e['name']}</b>\n"
                    f"Passed every League Scout gate (data quality, market evidence, no harm to existing "
                    f"leagues). Tips start with the next predict run; stakes start at the minimum until the "
                    f"league earns a rating.")
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=30)
            if r.status_code == 200:
                notified[key] = datetime.now(timezone.utc).isoformat() if isinstance(notified, dict) else True
        _save_notified(notified)
    except Exception as e:                                             # noqa: BLE001 - notice never blocks
        print(f"[auto_onboard] notice skipped: {type(e).__name__}")


def _sports() -> list:
    import requests
    key = os.getenv("ODDS_API_KEY", "").strip()
    try:
        s = requests.get("https://api.the-odds-api.com/v4/sports", params={"apiKey": key, "all": "true"},
                         timeout=30).json() if key else []
    except Exception:                                                  # noqa: BLE001
        s = []
    return s if isinstance(s, list) else []


def probe() -> int:
    """Report which leagues Scout is tracking would get a price source today (no changes made)."""
    import requests
    sports = _sports()
    try:
        st = requests.get("https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main/output/scout/league_status.json",
                          timeout=30).json().get("leagues", [])
    except Exception as e:                                             # noqa: BLE001
        print(f"[probe] scout status unreachable ({type(e).__name__})")
        return 0
    ok, rows = 0, []
    for r in st:
        name = f"{r.get('country')} {r.get('league')}"
        k, why = match_sport_key(str(r.get("country") or ""), name, sports)
        ok += bool(k)
        rows.append({"league": name, "sport_key": k, "reason": None if k else why, "scout_status": r.get("status")})
        print(f"{name:55s} {k or '-':40s} {'' if k else why}")
    summary = (f"{ok} of {len(st)} scout leagues have a uniquely matching OddsAPI sport key "
               f"({len(sports)} sports listed)")
    print(f"[probe] {summary}")
    (ROOT / "output" / "auto_onboard_probe.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "summary": summary,
         "leagues": rows}, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return probe()
    import requests
    import config
    try:
        queue = requests.get(QUEUE_URL, timeout=30).json().get("leagues", [])
    except Exception as e:                                             # noqa: BLE001 - fail closed
        print(f"[auto_onboard] Pro queue unreachable ({type(e).__name__}) - nothing changes")
        return 0
    if not queue:
        print("[auto_onboard] queue empty - no league has passed every gate")
        return 0
    sports = _sports()
    if not sports:
        print("[auto_onboard] OddsAPI sports list unavailable - nothing changes (fail closed)")
        return 0
    current = json.loads(AUTO.read_text(encoding="utf-8")) if AUTO.exists() else {"leagues": []}
    configured_names = set(config.STANDARD_FORMAT_LEAGUES) | set(config.NEW_FORMAT_LEAGUES)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    add, skipped = plan(queue, current, configured_names, set(config.API_FOOTBALL_IDS.values()), sports, now)
    print(json.dumps({"added": [e["name"] for e in add], "skipped": skipped}, indent=1))
    if add and not a.dry_run:
        current.setdefault("leagues", []).extend(add)
        AUTO.write_text(json.dumps(current, indent=1), encoding="utf-8")
        _notify(add)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
