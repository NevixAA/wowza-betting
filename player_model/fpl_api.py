"""Official Fantasy Premier League API client (free, no key required).

Supplies the LIVE fantasy layer the prop model can't produce on its own:
  * current squads (transfer-accurate — fixes stale player->club, e.g. Salah)
  * availability / injury flags (status + chance_of_playing + news)
  * official positions and prices (for points-per-£ value)
  * fixtures + official FDR (fixture difficulty)

Endpoints (public, JSON):
  {BASE}/bootstrap-static/  -> elements (players), teams, element_types, events (gameweeks)
  {BASE}/fixtures/          -> fixtures with per-side difficulty

Responses are cached under output/ so off-network reruns (and the dashboard) still work.
All fetches degrade gracefully to the cache, then to empty — never raise at import or call.
"""
from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import pandas as pd

_BASE       = "https://fantasy.premierleague.com/api"
_OUT        = Path(__file__).resolve().parents[1] / "output"
_BOOT_CACHE = _OUT / "fpl_bootstrap.json"
_FIX_CACHE  = _OUT / "fpl_fixtures.json"

# element_type -> FPL position code
_POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
# status flag -> human label
_STATUS = {"a": "available", "d": "doubtful", "i": "injured",
           "s": "suspended", "u": "unavailable", "n": "not in squad"}


def _norm(s: str) -> str:
    """Accent-strip + lowercase + alnum-only key for cross-source name matching."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def _get_json(url: str, timeout: int = 20):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


_CACHE_META = _OUT / "fpl_cache_meta.json"


def _meta() -> dict:
    try:
        return json.loads(_CACHE_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_age_h(cache: Path) -> float:
    """Hours since this cache was FETCHED, from a recorded timestamp — not from file mtime.

    WHY NOT MTIME. `git checkout` on a CI runner sets every file's mtime to checkout time, so a
    freshly cloned cache always looked ~0 hours old, always passed the TTL test, and the FPL API
    was therefore NEVER called. The committed snapshot was returned forever.

    Measured 2026-09-09: output/fpl_bootstrap.json had not changed content since 2026-07-27
    despite fantasy_refresh.yml running daily and explicitly staging it. Inside it, the "next"
    gameweek was GW1 with a 2026-08-21 deadline — a pre-season snapshot being served in
    September. Two visible consequences on the Fantasy page, both from this one cause:

      * Jeremy Doku was offered as a captaincy pick while injured, because the stale payload
        says status='a' with no news. The page's injury filter was working correctly on data
        that claimed he was fine.
      * "Next 5 fixtures" listed games already played, because the snapshot's idea of "next"
        was still GW1.

    This is the third instance of the same defect class in this repo. v9's provenance._model_sha
    hashed size+mtime and so changed on every run; registry.age_hours documents the identical
    trap and avoids it by reading a recorded timestamp. Same fix here.

    Falls back to mtime only when no timestamp has been recorded yet, so a cache written by an
    older version still expires rather than being treated as ageless.
    """
    rec = _meta().get(cache.name)
    if rec:
        try:
            return (time.time() - float(rec)) / 3600.0
        except (TypeError, ValueError):
            pass
    try:
        return (time.time() - cache.stat().st_mtime) / 3600.0
    except OSError:
        return float("inf")


def _record_fetch(cache: Path) -> None:
    """Stamp the fetch time in a SIDECAR, never inside the payload.

    The payload shape is the provider's and is read directly by the dashboard — bootstrap is a
    dict, fixtures is a list — so injecting a key would break one of them and change a file other
    code parses. A sidecar keeps both intact.
    """
    m = _meta()
    m[cache.name] = time.time()
    try:
        _OUT.mkdir(parents=True, exist_ok=True)
        _CACHE_META.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _cached_fetch(url: str, cache: Path, ttl_h: float, force: bool):
    """Return fresh JSON (and refresh cache), else cache, else None — never raises."""
    if not force and cache.exists() and _cache_age_h(cache) < ttl_h:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        data = _get_json(url)
        _OUT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        _record_fetch(cache)
        return data
    except Exception:
        # A FAILED fetch still falls back to the cache, deliberately — a stale page beats a broken
        # one. But it must be VISIBLE, because this silent path is why a two-month-old snapshot
        # was served without a single error anywhere.
        if cache.exists():
            try:
                age = _cache_age_h(cache)
                print(f"[fpl_api] fetch FAILED for {url} — serving {cache.name} "
                      f"aged {age:.1f}h (ttl {ttl_h}h)")
                return json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None


def fetch_bootstrap(ttl_h: float = 12.0, force: bool = False) -> dict:
    """bootstrap-static: players (elements), teams, positions, gameweeks. Cached ttl_h hours."""
    return _cached_fetch(f"{_BASE}/bootstrap-static/", _BOOT_CACHE, ttl_h, force) or {}


def fetch_fixtures(ttl_h: float = 6.0, force: bool = False) -> list:
    """All fixtures with per-side FDR. Cached ttl_h hours."""
    return _cached_fetch(f"{_BASE}/fixtures/", _FIX_CACHE, ttl_h, force) or []


def players_df(bootstrap: dict | None = None) -> pd.DataFrame:
    """Tidy player table from bootstrap elements: LIVE team, position, price, availability,
    injury flags, and name-match keys (match_key = full name, web_key = FPL short name)."""
    b = bootstrap if bootstrap is not None else fetch_bootstrap()
    if not b or "elements" not in b:
        return pd.DataFrame()
    teams = {t["id"]: t for t in b.get("teams", [])}
    rows = []
    for e in b["elements"]:
        t      = teams.get(e.get("team"), {})
        full   = f"{e.get('first_name', '')} {e.get('second_name', '')}".strip()
        status = e.get("status", "a") or "a"
        rows.append({
            "fpl_id":            e.get("id"),
            "web_name":          e.get("web_name", ""),
            "full_name":         full,
            "team":              t.get("name", ""),
            "team_short":        t.get("short_name", ""),
            "team_id":           e.get("team"),
            "position":          _POS.get(e.get("element_type"), ""),
            "price":             round((e.get("now_cost") or 0) / 10.0, 1),
            "status":            status,
            "availability":      _STATUS.get(status, "available"),
            "chance_of_playing": e.get("chance_of_playing_next_round"),
            "news":              (e.get("news") or "").strip(),
            "injured":           status in ("i", "s", "u"),
            "doubtful":          status == "d",
            # FPL-native signals (for differentials / hybrid ranking / price-change context)
            "owned_pct":         float(e.get("selected_by_percent") or 0.0),
            "fpl_form":          float(e.get("form") or 0.0),
            "fpl_ppg":           float(e.get("points_per_game") or 0.0),
            "fpl_ep_next":       float(e.get("ep_next") or 0.0),   # FPL's own expected pts
            "ict_index":         float(e.get("ict_index") or 0.0),
            "transfers_in_gw":   int(e.get("transfers_in_event") or 0),
            "transfers_out_gw":  int(e.get("transfers_out_event") or 0),
            "total_points":      int(e.get("total_points") or 0),
            "match_key":         _norm(full),
            "web_key":           _norm(e.get("web_name", "")),
        })
    return pd.DataFrame(rows)


def upcoming_fdr(next_n: int = 5, bootstrap: dict | None = None,
                 fixtures: list | None = None) -> dict:
    """{team_name: [{'opp': short, 'fdr': 1-5, 'home': bool}, ...]} for the next N unplayed
    fixtures per club. FDR is the OFFICIAL FPL difficulty. Empty if data unavailable."""
    b  = bootstrap if bootstrap is not None else fetch_bootstrap()
    fx = fixtures  if fixtures  is not None else fetch_fixtures()
    if not b or not fx:
        return {}
    teams = {t["id"]: t for t in b.get("teams", [])}
    unplayed = [f for f in fx if not f.get("finished") and f.get("event") is not None]
    unplayed.sort(key=lambda f: (f.get("event") or 999, f.get("kickoff_time") or ""))
    out: dict[str, list] = {}
    for f in unplayed:
        h = teams.get(f.get("team_h"), {})
        a = teams.get(f.get("team_a"), {})
        if h.get("name"):
            out.setdefault(h["name"], []).append(
                {"opp": a.get("short_name", ""), "fdr": f.get("team_h_difficulty"), "home": True})
        if a.get("name"):
            out.setdefault(a["name"], []).append(
                {"opp": h.get("short_name", ""), "fdr": f.get("team_a_difficulty"), "home": False})
    return {t: v[:next_n] for t, v in out.items()}


if __name__ == "__main__":
    bp = fetch_bootstrap()
    pdf = players_df(bp)
    print(f"FPL bootstrap: {len(pdf)} players, {pdf['team'].nunique() if not pdf.empty else 0} teams")
    if not pdf.empty:
        inj = pdf[pdf["injured"] | pdf["doubtful"]]
        print(f"  flagged (injured/doubtful): {len(inj)}")
        print(pdf.head(5)[["web_name", "team", "position", "price", "availability"]].to_string(index=False))
    fdr = upcoming_fdr()
    print(f"FDR: {len(fdr)} teams with upcoming fixtures")
