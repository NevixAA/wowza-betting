"""
Warm the new-format shot-stat cache (v10 NF upgrade — UNCOMMITTED per 2026-07-13 rule).
=========================================================================================
The NF O/U model needs rolling SOT (`combined_sot_ratio`); without warm shot data it falls
back to the 0.34 constant (the "No completed fixtures" gap). The predict job runs in
AF_OU_CACHE_ONLY mode (never fetches), so the cache must be warmed by a non-cache-only job.
This fetches completed-fixture shot stats for every new-format league (current + prior season)
into apifootball_ou_cache/. Stats are cached ~1 year, so after the first warm it's incremental
(only new completed fixtures cost calls). Quota-guarded: stops if daily headroom runs low.

Pairs with the _get cache-poisoning fix (don't cache error responses) — together they resolve
the NF shot gap. Run daily in CI (has APIFOOTBALL_KEY + the 7,500/day Pro quota).
"""
import os, sys, requests
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
os.chdir(PROJ); sys.path.insert(0, str(PROJ))
import config
import src.api_football_ou as afou

_BASE = "https://v3.football.api-sports.io"
_MIN_HEADROOM = 500   # stop if fewer than this many daily requests remain


def _quota_remaining() -> int:
    """Free /status check (does not consume quota)."""
    try:
        H = {"x-apisports-key": os.getenv("APIFOOTBALL_KEY", "")}
        s = requests.get(f"{_BASE}/status", headers=H, timeout=15).json().get("response", {})
        req = s.get("requests", {})
        return int(req.get("limit_day", 0)) - int(req.get("current", 0))
    except Exception:
        return 9999


def run() -> None:
    if not afou._KEY:
        print("[warm_nf] APIFOOTBALL_KEY not set — skipping"); return
    leagues = sorted(config.NEW_FORMAT_LEAGUES & set(config.API_FOOTBALL_IDS))
    now_year = int(__import__("datetime").datetime.utcnow().year)
    total = 0
    for league in leagues:
        head = _quota_remaining()
        if head < _MIN_HEADROOM:
            print(f"[warm_nf] quota headroom {head} < {_MIN_HEADROOM} — stopping early"); break
        lid = config.API_FOOTBALL_IDS[league]
        cur = config.API_FOOTBALL_SEASONS.get(league, str(now_year))
        # warm current + prior season (rolling features need recent history)
        try:
            prior = str(int(cur) - 1)
        except ValueError:
            prior = cur
        for season in {cur, prior}:
            df = afou.fetch_shots_for_league(lid, season, league)
            n = len(df) if df is not None else 0
            total += n
            print(f"[warm_nf] {league} {season}: {n} fixtures warmed (headroom~{_quota_remaining()})")
    print(f"[warm_nf] done — {total} fixture-stats warmed into apifootball_ou_cache/")


if __name__ == "__main__":
    run()
