"""
Dense player-prop odds snapshot (open -> shifting -> closing), decoupled + always-on.
=====================================================================================
The hourly predict already logs prop odds, but it's HEAVY (collect+train+predict) and can
skip/fail. This lightweight job just SNAPSHOTS odds for ALL upcoming props (every prop-covered
league, all players/markets — not only tip candidates) and closes out CLV. Run it always-on
(not match-day-gated) so we capture the OPENING line as soon as the book posts it, through to
kickoff, building our own open->shifting->closing record for CLV.

Cost: reuses odds_fetcher (2h per-event cache), so runs offset from predict mostly hit cache
= near-free; only a genuinely-expired price costs credits. NO real money — tracking only.
ISOLATION: props-only; writes output/player_prop_odds_history.csv (distinct-price append) +
output/clv_records.csv. Standard model / team pipeline untouched.

NOTE (2026-07-09 investigation): line MOVEMENT itself carries no edge (persistence corr≈0.03;
positive CLV coexists with −67% ROI). So this captures data to MEASURE CLV cheaply — it is NOT
a movement-prediction program. Read CLV, not P/L.
"""
import os, sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
os.chdir(PROJ); sys.path.insert(0, str(PROJ))
import pandas as pd
from player_model import config
from player_model.api_football import get_upcoming_fixtures
from player_model.odds_fetcher import fetch_prop_odds, PROP_SPORT_KEYS
from player_model import clv_tracker


def build_upcoming_signals() -> pd.DataFrame:
    """All upcoming fixtures for prop-covered leagues -> (league, match, date)."""
    rows = []
    for league, lg_id in config.PROP_LEAGUES.items():
        if league not in PROP_SPORT_KEYS:      # only leagues with OddsAPI prop markets
            continue
        season = config.PROP_SEASONS.get(league, str(pd.Timestamp.utcnow().year))
        n_fix = 20 if league == "World Cup" else 8
        try:
            fixtures = get_upcoming_fixtures(lg_id, season, next_n=n_fix)
        except Exception as e:
            print(f"  {league}: fixtures fetch failed ({e})"); continue
        for fx in fixtures:
            teams = fx.get("teams", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            dt = (fx.get("fixture", {}).get("date", "") or "")[:10]
            if home and away:
                rows.append({"league": league, "match": f"{home} vs {away}", "date": dt})
    return pd.DataFrame(rows)


def run() -> None:
    if not os.getenv("APIFOOTBALL_KEY"):
        print("[snapshot] APIFOOTBALL_KEY not set — skipping"); return
    sig = build_upcoming_signals()
    if sig.empty:
        print("[snapshot] no upcoming prop fixtures"); return
    print(f"[snapshot] {len(sig)} upcoming fixtures across {sig['league'].nunique()} leagues")
    # fetch_prop_odds appends every DISTINCT price to player_prop_odds_history.csv internally
    odds = fetch_prop_odds(sig)
    print(f"[snapshot] captured {len(odds)} player/market prices this run")
    # Close out CLV for finished matches + print rolling CLV (read CLV, not P/L)
    try:
        clv_tracker.close_out()
        # Grade the RESULT of each closed-out prop from actual stats (once in the parquet) so
        # every CLV pairs with WIN/LOSS -> the analyzable open->close->CLV->result dataset.
        graded = clv_tracker.settle_results()
        print(f"[snapshot] graded {graded} prop result(s)")
        clv_tracker.report()
    except Exception as e:
        print(f"[snapshot] clv close/settle/report skipped: {e}")


if __name__ == "__main__":
    run()
