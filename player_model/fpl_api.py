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
#: availability label of a player whose FPL status is missing (or one we do not know).
UNKNOWN_AVAILABILITY = "unknown"
#: labels a recommender (squad optimiser, chips, transfers) must never pick. 'unknown' is here on
#: purpose: a player whose availability nobody published is not recommendable (Data Truth 2026-11-02).
NOT_RECOMMENDABLE = frozenset({"injured", "unavailable", "suspended", "not in squad",
                               UNKNOWN_AVAILABILITY})


def availability_unknown(av: "pd.Series") -> "pd.Series":
    """Boolean mask: the availability label is missing or 'unknown'."""
    s = av.astype(object)
    return s.isna() | s.astype(str).str.strip().str.lower().isin(("", "nan", "none", UNKNOWN_AVAILABILITY))


def not_recommendable(av: "pd.Series", also: tuple = ()) -> "pd.Series":
    """Boolean mask over availability labels: True = must not be recommended (ruled out, or
    availability unknown). `also` adds labels a caller excludes on top (chips add 'doubtful')."""
    lab = av.astype(str).str.strip().str.lower()
    return availability_unknown(av) | lab.isin(NOT_RECOMMENDABLE | set(also))


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

    NO RECORDED TIMESTAMP MEANS STALE, NOT FRESH. The first version of this fix fell back to
    mtime when the sidecar was absent, and that made the fix unable to fire at all:

        no sidecar -> fall back to mtime -> CI mtime is checkout time, ~0h -> "fresh"
        -> skip the fetch -> never write the sidecar -> no sidecar next run either

    A permanent deadlock, reproducing the exact bug it was meant to cure. Verified 2026-09-10:
    output/fpl_cache_meta.json had never been committed and fpl_bootstrap.json had still not
    changed since 2026-07-27.

    So an unknown age is now treated as infinitely old. Cost of being wrong: one extra API call
    on the first run after deploy, which then writes the sidecar and settles. Cost of the other
    default: serving a two-month-old squad list forever, which is what happened.
    """
    rec = _meta().get(cache.name)
    if rec:
        try:
            return (time.time() - float(rec)) / 3600.0
        except (TypeError, ValueError):
            pass
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


def gameweek_context(bootstrap: dict | None = None) -> dict:
    """Which gameweek a projection made RIGHT NOW is actually predicting, and its deadline.

    WHY THIS EXISTS. fantasy_projection_log.csv had 8,366 rows across 31 days and every one of
    them carried an empty `gw`. fantasy_log.append() takes `gw` as a keyword defaulting to None
    and the caller never passed it, so the column was written as "" every time.

    That is the reason projections cannot be settled against real results. A projection is only
    a forecast if you know WHICH gameweek it forecast; without that tag there is no key to join
    it to the points that were actually scored, so "is Wowza better than FPL's own ep_next"
    stays unanswerable no matter how much history accumulates.

    THE TARGET IS `is_next`, NOT `is_current`. Once a deadline passes, the current gameweek is
    locked -- no transfer or captain choice can change it -- so a projection generated now is
    advice for the NEXT one. FPL marks exactly one event `is_next`, including while `is_current`
    is still being played, which is the behaviour we want in both cases.

    Returns keys: gw, deadline_utc, current_gw, last_finished_gw, season_complete. Every value
    may be None; this must never raise, because it is called from a logging path that a
    projection run should survive without.
    """
    out = {"gw": None, "deadline_utc": None, "current_gw": None,
           "last_finished_gw": None, "season_complete": False}
    try:
        b = bootstrap if bootstrap is not None else fetch_bootstrap()
        events = (b or {}).get("events") or []
        if not events:
            return out
        nxt = next((e for e in events if e.get("is_next")), None)
        cur = next((e for e in events if e.get("is_current")), None)
        finished = [e for e in events if e.get("finished")]
        out["current_gw"] = cur.get("id") if cur else None
        out["last_finished_gw"] = finished[-1].get("id") if finished else None
        if nxt:
            out["gw"] = nxt.get("id")
            out["deadline_utc"] = nxt.get("deadline_time")
        elif finished and len(finished) == len(events):
            # Season over: there is no next gameweek to predict. Say so rather than silently
            # tagging rows with the last one, which would corrupt the ledger at season end.
            out["season_complete"] = True
        elif cur:
            out["gw"] = cur.get("id")
            out["deadline_utc"] = cur.get("deadline_time")
    except Exception:                                                # noqa: BLE001
        pass
    return out


def _f_or_nan(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


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
        # Absent status is UNKNOWN (None), not "available" (Data Truth, 2026-11-01/02). Its
        # availability label is 'unknown', which every recommender treats as NOT recommendable.
        status = e.get("status") or None
        cost = _f_or_nan(e.get("now_cost"))
        rows.append({
            "fpl_id":            e.get("id"),
            "web_name":          e.get("web_name", ""),
            "full_name":         full,
            "team":              t.get("name", ""),
            "team_short":        t.get("short_name", ""),
            "team_id":           e.get("team"),
            "position":          _POS.get(e.get("element_type"), ""),
            # No now_cost -> price NaN (was 0.0, which made the player free to the optimiser).
            "price":             round(cost / 10.0, 1) if cost == cost else float("nan"),
            "status":            status,
            "availability":      _STATUS.get(status, UNKNOWN_AVAILABILITY),
            "chance_of_playing": e.get("chance_of_playing_next_round"),
            "news":              (e.get("news") or "").strip(),
            # 'n' (not in squad / ineligible) is OUT too - V2 already treats it so; V1 did not
            # (R2-D fantasy bug 'status_n_not_out_v1').
            "injured":           status in ("i", "s", "u", "n"),
            "doubtful":          status == "d",
            # FPL-native signals (for differentials / hybrid ranking / price-change context)
            # Every FPL number FPL did not send is NaN, never 0 (Data Truth, 2026-11-02).
            "owned_pct":         _f_or_nan(e.get("selected_by_percent")),
            "fpl_form":          _f_or_nan(e.get("form")),
            "fpl_ppg":           _f_or_nan(e.get("points_per_game")),
            "fpl_ep_next":       _f_or_nan(e.get("ep_next")),     # FPL's own expected pts
            "ict_index":         _f_or_nan(e.get("ict_index")),
            "transfers_in_gw":   _f_or_nan(e.get("transfers_in_event")),
            "transfers_out_gw":  _f_or_nan(e.get("transfers_out_event")),
            "total_points":      _f_or_nan(e.get("total_points")),
            # FPL's own defensive-contribution action count per 90 (the stat its DC points use)
            "dc_per90":          (float(e["defensive_contribution_per_90"])
                                  if e.get("defensive_contribution_per_90") is not None else None),
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
