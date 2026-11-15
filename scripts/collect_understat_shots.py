"""
collect_understat_shots.py - Understat shots with their situation (OpenPlay / FromCorner / SetPiece
/ DirectFreekick / Penalty) and match rosters, for the leagues Understat covers (owner decision B,
2026-11-02). No API-Football calls.

    python scripts/collect_understat_shots.py [--max-matches N] [--map-only]

FETCH (resumable; the parquets are the journal):
  1. each Understat league-season page (UNDERSTAT_LEAGUES x seasons from the year before our first
     player_history match to this year; a season Understat does not have returns no matches and is
     simply empty) -> output/collect/understat_matches.parquet. Seasons that have no unplayed match
     left are not fetched again.
  2. each finished match not yet stored -> output/collect/understat_shots.parquet (keyed by
     Understat shot id) and understat_rosters.parquet (match_id, side, Understat player id).
     A match whose payload holds neither shots nor roster is recorded in understat_fetched.parquet
     as 'empty' so it is not asked again; a failed request records nothing.
MAP (every run, after fetching): player_model.setpiece.mapping writes
  output/collect/understat_match_map.parquet   (match_id -> fixture_id, league)
  output/collect/understat_player_map.parquet  (us_player_id -> player_id, n_votes)
  output/collect/understat_map_report.json     (what mapped and what was refused, and why)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import COLLECT_DIR, append_parquet, history_frame, read_parquet  # noqa: E402
from player_model.setpiece import mapping, understat as U  # noqa: E402

BASE = "https://understat.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (wowza data-truth collector)", "X-Requested-With": "XMLHttpRequest"}
PAUSE_S = 0.6
FLUSH_EVERY = 100

MATCHES = COLLECT_DIR / "understat_matches.parquet"
SHOTS = COLLECT_DIR / "understat_shots.parquet"
ROSTERS = COLLECT_DIR / "understat_rosters.parquet"
FETCHED = COLLECT_DIR / "understat_fetched.parquet"
MATCH_MAP = COLLECT_DIR / "understat_match_map.parquet"
PLAYER_MAP = COLLECT_DIR / "understat_player_map.parquet"
REPORT = COLLECT_DIR / "understat_map_report.json"
# SOURCE STATUS (Data Truth round 5). Run 41440689838 asked for 60 Understat URLs, every one failed with
# "Failed to resolve 'understat.com'", and the run ended "no matches collected: nothing to map" /
# "nothing new to commit" - an empty result that looked like no new data. Every run now writes what the
# source did: understat_status.json (latest) and one line per run in understat_status_log.jsonl.
STATUS = COLLECT_DIR / "understat_status.json"
STATUS_LOG = COLLECT_DIR / "understat_status_log.jsonl"

# Run-level request accounting, reset by main().
STATS: dict = {}


def _reset_stats() -> None:
    STATS.clear()
    STATS.update({"requests": 0, "ok": 0, "unreachable": 0, "timeout": 0, "http_error": 0, "other_error": 0,
                  "http_codes": {}, "pages_without_matches": 0, "sample_error": None})


def failure_kind(exc: Exception) -> str:
    """'unreachable' (DNS failure, refused or reset connection), 'timeout', or 'other_error'."""
    import requests
    if isinstance(exc, (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout)):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "unreachable"
    return "other_error"


def _get(session, urls: list[str]) -> str | None:
    if not STATS:
        _reset_stats()
    for u in urls:
        STATS["requests"] += 1
        try:
            r = session.get(u, headers=HEADERS, timeout=30)
        except Exception as e:                                           # noqa: BLE001
            STATS[failure_kind(e)] += 1
            STATS["sample_error"] = STATS["sample_error"] or f"{type(e).__name__}: {str(e)[:300]}"
            print(f"[understat] {u}: {e}")
            continue
        time.sleep(PAUSE_S)
        if r.status_code == 200 and r.text:
            STATS["ok"] += 1
            return r.text
        STATS["http_error"] += 1
        STATS["http_codes"][str(r.status_code)] = STATS["http_codes"].get(str(r.status_code), 0) + 1
    return None


def preflight(session) -> str | None:
    """One request to the site root before ~60+ page requests. Returns the failure kind when the source
    cannot be reached at all (so the run stops in seconds, not after 60 DNS time-outs), else None."""
    if not STATS:
        _reset_stats()
    STATS["requests"] += 1
    try:
        r = session.get(BASE + "/", headers=HEADERS, timeout=30)
    except Exception as e:                                               # noqa: BLE001
        kind = failure_kind(e)
        STATS[kind] += 1
        STATS["sample_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return kind if kind in ("unreachable", "timeout") else None
    if r.status_code == 200:
        STATS["ok"] += 1
        return None
    STATS["http_error"] += 1
    STATS["http_codes"][str(r.status_code)] = STATS["http_codes"].get(str(r.status_code), 0) + 1
    return None


def source_status(stats: dict, stored_matches: int, stored_shots_matches: int) -> str:
    """One word for what the source did this run:
    SOURCE_UNREACHABLE  no request got an answer and every failure was DNS / connection / time-out
    SOURCE_FAILED       no request succeeded (HTTP errors or other failures)
    SOURCE_EMPTY        pages answered but none held a match (the page format may have changed)
    PARTIAL             some requests succeeded, some failed
    OK                  every request succeeded (or there was nothing left to ask)"""
    req, ok = stats.get("requests", 0), stats.get("ok", 0)
    failed = req - ok
    if req and ok == 0:
        net = stats.get("unreachable", 0) + stats.get("timeout", 0)
        return "SOURCE_UNREACHABLE" if net == failed else "SOURCE_FAILED"
    if ok and stats.get("pages_without_matches", 0) and stored_matches == 0 and stored_shots_matches == 0:
        return "SOURCE_EMPTY"
    return "PARTIAL" if failed else "OK"


def write_status(status: str, new_matches: int, new_match_payloads: int, note: str = "") -> dict:
    doc = {"source": "understat", "host": BASE, "status": status,
           "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run_id": os.getenv("GITHUB_RUN_ID", ""),
           "new_league_matches": int(new_matches), "match_payloads_fetched": int(new_match_payloads),
           **{k: STATS.get(k) for k in ("requests", "ok", "unreachable", "timeout", "http_error",
                                         "other_error", "http_codes", "pages_without_matches",
                                         "sample_error")},
           "note": note}
    COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    with STATUS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, sort_keys=True) + "\n")
    line = (f"Understat {status}: {doc['ok']}/{doc['requests']} request(s) answered, "
            f"{new_matches} new league match(es), {new_match_payloads} match payload(s)"
            + (f" - {doc['sample_error']}" if status != "OK" and doc["sample_error"] else ""))
    if status in ("SOURCE_UNREACHABLE", "SOURCE_FAILED", "SOURCE_EMPTY"):
        print(f"::error title=Understat source::{line}")
    elif status == "PARTIAL":
        print(f"::warning title=Understat source::{line}")
    print(f"[understat] {line}")
    return doc


def seasons() -> list[str]:
    h = history_frame(["date"])
    d = pd.to_datetime(h["date"], errors="coerce") if not h.empty else pd.Series(dtype="datetime64[ns]")
    if d.notna().sum() == 0:
        return []
    first = int(d.min().year) - 1                  # Understat labels a season by its start year
    return [str(y) for y in range(first, datetime.now(timezone.utc).year + 1)]


def fetch_leagues(session) -> int:
    have = read_parquet(MATCHES)
    finished = set()
    if not have.empty:
        g = have.groupby(["us_league", "us_season"])["is_result"].all()
        finished = {k for k, v in g.items() if v}
    n = 0
    for lg in U.UNDERSTAT_LEAGUES:
        for s in seasons():
            if (lg, s) in finished:
                continue
            text = _get(session, [f"{BASE}/getLeagueData/{lg}/{s}", f"{BASE}/league/{lg}/{s}"])
            if text is None:
                continue
            rows = U.parse_league(text, lg, s)
            if not rows:
                STATS["pages_without_matches"] = STATS.get("pages_without_matches", 0) + 1
            if rows:
                append_parquet(MATCHES, rows, ["match_id"], U.MATCH_COLS)
                n += len(rows)
            print(f"  understat {lg} {s}: {len(rows)} match(es)")
    return n


def fetch_matches(session, max_matches: int | None) -> int:
    m = read_parquet(MATCHES)
    if m.empty:
        return 0
    done = set(read_parquet(FETCHED).get("match_id", pd.Series(dtype=int)).astype(int))
    todo = [int(x) for x in m.loc[m["is_result"].astype(bool), "match_id"] if int(x) not in done]
    print(f"[understat] {len(todo):,} finished match(es) to fetch")
    sb, rb, fb, n = [], [], [], 0

    def flush():
        if sb:
            append_parquet(SHOTS, sb, ["shot_id"], U.SHOT_COLS)
        if rb:
            append_parquet(ROSTERS, rb, ["match_id", "h_a", "us_player_id"], U.ROSTER_COLS)
        if fb:                                   # the journal last, after its data is stored
            append_parquet(FETCHED, fb, ["match_id"], ["match_id", "status", "fetched_at"])
        sb.clear(); rb.clear(); fb.clear()

    for mid in todo[: max_matches or None]:
        text = _get(session, [f"{BASE}/getMatchData/{mid}", f"{BASE}/match/{mid}"])
        if text is None:
            continue
        parsed = U.parse_match(text, mid)
        status = "empty"
        if parsed is not None:
            s, r = parsed
            sb.extend(s)
            rb.extend(r)
            status = "ok" if (s or r) else "empty"
        fb.append({"match_id": mid, "status": status, "fetched_at": datetime.now(timezone.utc).isoformat()})
        n += 1
        if len(fb) >= FLUSH_EVERY:
            flush()
    flush()
    return n


def build_maps() -> dict:
    m, r = read_parquet(MATCHES), read_parquet(ROSTERS)
    if m.empty:
        print("[understat] no matches collected: nothing to map")
        return {}
    h = history_frame(["fixture_id", "date", "league", "home_team", "away_team", "is_home",
                       "player_id", "player_name"])
    mm, rep_m = mapping.map_matches(m, h)
    pm, rep_p = mapping.map_players(r, mm, h)
    mm.to_parquet(MATCH_MAP, index=False)
    pm.to_parquet(PLAYER_MAP, index=False)
    rep = {"matches": rep_m, "players": rep_p, "mapped_players": int(len(pm)),
           "built_at": datetime.now(timezone.utc).isoformat()}
    REPORT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"[understat] map: {rep}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect Understat shots + rosters and map them")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--map-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.map_only:
        import requests
        _reset_stats()
        n_league = n_match = 0
        with requests.Session() as session:
            down = preflight(session)
            if down:
                write_status("SOURCE_UNREACHABLE", 0, 0,
                             note=f"preflight {BASE}/ failed ({down}); no page was requested and nothing "
                                  f"was stored. The stored Understat data is unchanged.")
            else:
                n_league = fetch_leagues(session)
                n_match = fetch_matches(session, a.max_matches)
                write_status(source_status(STATS, n_league, n_match), n_league, n_match)
    build_maps()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
