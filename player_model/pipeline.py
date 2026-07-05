"""
Player Model Pipeline v2
========================
Usage:
  python -m player_model.pipeline --mode collect   # fetch FBref training data
  python -m player_model.pipeline --mode train     # train 4 models
  python -m player_model.pipeline --mode predict   # generate today's player tips
  python -m player_model.pipeline --mode all       # collect + train + predict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Force UTF-8 stdout so team names with non-ASCII chars (ü, é, etc.) don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_v9 = Path(__file__).resolve().parents[1]
load_dotenv(_v9 / ".env")
sys.path.insert(0, str(_v9))

from player_model import config
from player_model.data_fetcher import (
    collect_history, collect_match_history, collect_national_team_history,
    fetch_all_player_season_stats, fetch_lineup, fetch_current_squad, fetch_league_teams,
    FBREF_LEAGUES, EUROPEAN_CUPS, APIFOOTBALL_LEAGUES, enrich_sp_events,
)
from player_model.league_quality import enrich_league_quality
from player_model.feature_engineering import build_features
from player_model.model import train, save_model
from player_model.predict import run_player_predictions, enrich_with_odds, enrich_no_odds_markets
from player_model.odds_fetcher import fetch_prop_odds, match_odds_to_tips
from player_model.ledger import append_player_signals

HISTORY_CACHE = config.BASE_DIR / "player_history.parquet"


# ── Phase 2: Collect ──────────────────────────────────────────────────────────

def mode_collect(extended: bool = False, last_n: int = 100) -> None:
    if extended:
        # Extended mode: add European cups on top of COLLECT_SEASONS
        print(f"[collect] Fetching per-match player stats (COLLECT_SEASONS + European cups)...")
        rows = collect_match_history()  # core multi-season
        extra = collect_match_history(leagues=EUROPEAN_CUPS, last_n=last_n)
        rows = rows + extra
    else:
        print("[collect] Fetching per-match player stats (COLLECT_SEASONS: 2024+2025 per league)...")
        rows = collect_match_history()  # uses COLLECT_SEASONS — no last_n needed
    if not rows:
        print("[collect] No data — FBref may be rate-limiting. Try again in a few minutes.")
        return

    rows = enrich_sp_events(rows)
    rows = enrich_league_quality(rows)
    df = build_features(rows)
    if df.empty:
        print("[collect] Feature engineering returned empty DataFrame.")
        return

    # MERGE, don't overwrite: replace only the leagues collected this run, KEEP all others
    # (WC/international + any league a partial/timed-out CI run didn't reach). Prevents the
    # silent degradation where a Sunday collect that only got PL before the job timeout
    # nuked the other 10 club leagues down to a PL-only parquet (bug found 2026-07-05).
    if HISTORY_CACHE.exists():
        try:
            existing = pd.read_parquet(HISTORY_CACHE)
            fresh_leagues = set(df["league"].dropna().unique())
            kept = existing[~existing["league"].isin(fresh_leagues)]
            df = pd.concat([kept, df], ignore_index=True)
            if {"fixture_id", "player_id"}.issubset(df.columns):
                df = df.drop_duplicates(subset=["fixture_id", "player_id"])
            df = df.reset_index(drop=True)
            print(f"[collect] merged: kept {len(kept)} rows from {existing['league'].nunique() - len(fresh_leagues & set(existing['league']))} untouched leagues")
        except Exception as e:
            print(f"[collect] merge with existing failed ({e}) — writing fresh collect only")
    df.to_parquet(HISTORY_CACHE, index=False)
    print(f"[collect] Saved {len(df)} player rows, {df['player_id'].nunique()} players -> {HISTORY_CACHE.name}")

    # Summary — all markets
    for market, target_col in config.MARKET_TARGETS.items():
        if target_col in df.columns:
            rate = df[target_col].mean()
            print(f"  {market} ({target_col}): {rate:.1%} positive rate")


# ── WC26 national team collect ────────────────────────────────────────────────

def mode_collect_wc(last_n: int = 10) -> None:
    print(f"[collect-wc] Fetching WC2026 national team player stats (last {last_n} fixtures/team)...")
    rows = collect_national_team_history(last_n=last_n)
    if not rows:
        print("[collect-wc] No data collected.")
        return

    rows = enrich_sp_events(rows)
    rows = enrich_league_quality(rows)
    new_df = build_features(rows)
    if new_df.empty:
        print("[collect-wc] Feature engineering returned empty DataFrame.")
        return

    # Merge with existing history (club leagues) — deduplicate on fixture+player
    if HISTORY_CACHE.exists():
        existing = pd.read_parquet(HISTORY_CACHE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["fixture_id", "player_id"])
        combined = combined.reset_index(drop=True)
    else:
        combined = new_df

    combined.to_parquet(HISTORY_CACHE, index=False)
    wc_players = new_df["player_id"].nunique()
    print(f"[collect-wc] +{len(new_df)} WC rows ({wc_players} players) merged -> {len(combined)} total rows")

    for market in ["target_goals", "target_sot", "target_cards", "target_assists"]:
        if market in new_df.columns:
            rate = new_df[market].mean()
            print(f"  {market}: {rate:.1%} positive rate (WC data)")


# ── Phase 3: Train ────────────────────────────────────────────────────────────

def mode_train() -> None:
    if not HISTORY_CACHE.exists():
        print("[train] No history. Run --mode collect first.")
        return

    df = pd.read_parquet(HISTORY_CACHE)
    print(f"[train] Training on {len(df)} rows, {df['player_id'].nunique()} players.")

    for market in config.MARKETS:
        print(f"\n  Training market: {market}")
        try:
            results = train(df, market)
            save_model(results, market)
        except ValueError as e:
            print(f"  [SKIP] {market}: {e}")
        except Exception as e:
            print(f"  [ERROR] {market}: {e}")

    print("\n[train] Done.")


# ── Match contexts (team win probs for opponent-weakness multiplier) ──────────

def _build_match_contexts() -> dict:
    """
    Build per-match context dict with de-vigged team win probabilities.
    Currently reads from worldcup_history.json (1X2 odds tracked by the WC drift tracker).
    Returns {match_key: {home_win_prob, away_win_prob}}.
    """
    import json
    contexts: dict = {}
    wc_hist_file = config.OUTPUT_DIR / "worldcup_history.json"
    if not wc_hist_file.exists():
        return contexts
    try:
        hist = json.loads(wc_hist_file.read_text(encoding="utf-8"))
        for rec in hist.values():
            if rec.get("market") != "h2h":
                continue
            home = rec["home"]
            away = rec["away"]
            dt   = rec.get("date", "")[:10]
            match_key = f"{home}|{away}|{dt}"
            snaps = rec.get("snapshots", [])
            odds  = snaps[-1]["odds"] if snaps else rec.get("opening", {})
            oh = odds.get("odds_home")
            oa = odds.get("odds_away")
            od = odds.get("odds_draw", 3.0) or 3.0
            if not oh or not oa:
                continue
            total = 1 / oh + 1 / oa + 1 / od
            contexts[match_key] = {
                "home_win_prob": round((1 / oh) / total, 3),
                "away_win_prob": round((1 / oa) / total, 3),
            }
        if contexts:
            print(f"[pipeline] Win probs loaded for {len(contexts)} WC matches")
    except Exception as e:
        print(f"[pipeline] Match context build failed: {e}")
    return contexts


# ── Referee profiles ─────────────────────────────────────────────────────────

def _build_referee_profiles() -> dict:
    """
    Fetch referee names for all upcoming prop-league fixtures and build strictness profiles.
    Returns {match_key: {yellows_per_game, strictness_score, n_games}}.
    All underlying API calls are cached in api_football._get() — safe to call every run.
    """
    from player_model.api_football import (
        get_upcoming_fixtures, get_fixture_referee, build_referee_profile,
    )
    profiles: dict = {}
    try:
        for league, lg_id in config.PROP_LEAGUES.items():
            season = config.PROP_SEASONS.get(league, "2025")
            n_fix = 20 if league == "World Cup" else 5
            fixtures = get_upcoming_fixtures(lg_id, season, next_n=n_fix)
            for fix in fixtures:
                teams    = fix.get("teams", {})
                home     = teams.get("home", {}).get("name", "")
                away     = teams.get("away", {}).get("name", "")
                dt       = fix.get("fixture", {}).get("date", "")[:10]
                fid      = fix.get("fixture", {}).get("id")
                if not fid:
                    continue
                match_key = f"{home}|{away}|{dt}"
                referee   = get_fixture_referee(fid)
                if referee:
                    profiles[match_key] = build_referee_profile(referee, lg_id, season)
    except Exception as e:
        print(f"[pipeline] Referee profile build failed: {e}")
    return profiles


# ── Match-day gating (active only AFTER the World Cup) ──────────────────────────
# WC2026 final ~Jul 26. During/before WC there are matches almost daily, so the
# gate is dormant. After WC, predict + injury-refresh only run when there are
# prop-league fixtures in the next 72h — no wasted API calls on quiet days.
_WC_END = "20260726"


def _match_day_gate_skips() -> bool:
    """True => caller should skip (post-WC and no prop fixtures within 72h).
    Fail-open: any error returns False so we never lose tips to a check bug."""
    import datetime as _dt
    try:
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
        if today <= _WC_END:
            return False  # dormant during/before WC
        from player_model.api_football import get_upcoming_fixtures
        now     = _dt.datetime.now(_dt.timezone.utc)
        horizon = now + _dt.timedelta(hours=72)
        for league, lg_id in config.PROP_LEAGUES.items():
            season = config.PROP_SEASONS.get(league, "2025")
            for fix in get_upcoming_fixtures(lg_id, season, next_n=3) or []:
                ds = fix.get("fixture", {}).get("date", "")
                if not ds:
                    continue
                try:
                    fd = _dt.datetime.fromisoformat(ds.replace("Z", "+00:00"))
                except Exception:
                    continue
                if now <= fd <= horizon:
                    return False  # a match is coming up → run
        return True  # post-WC and nothing within 72h → skip
    except Exception as e:
        print(f"[gate] fixture check failed ({e}) — running anyway")
        return False


# ── Predict ───────────────────────────────────────────────────────────────────

def mode_predict() -> None:
    if not HISTORY_CACHE.exists():
        print("[predict] No history. Run --mode collect + train first.")
        return

    if _match_day_gate_skips():
        print("[predict] Post-WC match-day gating: no prop fixtures in next 72h — skipping.")
        return

    history = pd.read_parquet(HISTORY_CACHE)

    # Try to enrich with API-Football recent form if key available
    import os
    api_key = os.getenv("APIFOOTBALL_KEY", "")
    if api_key:
        print("[predict] API-Football key found — will fetch recent match stats for rolling features.")
        _enrich_with_recent_form(history)
    else:
        print("[predict] No APIFOOTBALL_KEY — using season stats only (less precise rolling features).")

    tips = run_player_predictions(
        bets_csv=config.OUTPUT_DIR / "bets.csv",
        history_df=history,
        referee_profiles=_build_referee_profiles() if api_key else None,
        match_contexts=_build_match_contexts(),
    )

    if tips.empty:
        print("[predict] No player tips generated.")
        return

    # Enrich with live market odds from The Odds API
    from player_model.odds_fetcher import _load_odds_key
    odds_api_key = _load_odds_key()
    if odds_api_key:
        print("[predict] Fetching player prop odds from The Odds API...")
        try:
            odds_raw = fetch_prop_odds(tips)
            if odds_raw:
                odds_mapped = match_odds_to_tips(tips, odds_raw)
                tips = enrich_with_odds(tips, odds_mapped)
                # Re-save enriched CSV
                tips.to_csv(config.OUTPUT_DIR / "player_tips.csv", index=False)
                print(f"[predict] Odds enriched for {len(odds_mapped)} player/market pairs")
            else:
                print("[predict] No player prop odds returned from Odds API")
        except Exception as e:
            print(f"[predict] Odds enrichment failed: {e}")
    else:
        print("[predict] No ODDS_API_KEY — skipping odds enrichment (tier will stay AVOID)")

    # Tier markets with no live Odds API coverage (assists) via calibration base rates
    tips = enrich_no_odds_markets(tips)
    tips.to_csv(config.OUTPUT_DIR / "player_tips.csv", index=False)
    assists_tiered = tips[(tips["market"] == "assists") & (tips["tier"] != "AVOID")]
    if not assists_tiered.empty:
        print(f"[predict] calibration-implied: {len(assists_tiered)} assists tip(s) tiered (AVOID excluded)")

    # Lineup availability filter — drops confirmed bench players, flags TBC
    if api_key:
        tips = _filter_by_lineup(tips)

    sniper   = tips[tips["tier"] == "SNIPER"]
    marksman = tips[tips["tier"] == "MARKSMAN"]
    valuable = tips[tips["tier"] == "VALUABLE"]
    print(f"[predict] SNIPER:{len(sniper)}  MARKSMAN:{len(marksman)}  VALUABLE:{len(valuable)}")

    n_new = append_player_signals(tips)
    if n_new:
        print(f"[predict] player_ledger: +{n_new} new signal(s) recorded")


def _filter_by_lineup(tips: pd.DataFrame) -> pd.DataFrame:
    """
    Filter player tips by confirmed starting XI.
    - Confirmed starter    → keep, lineup_status='confirmed'
    - Confirmed benched    → drop (saves Telegram noise for DNP players)
    - Lineup not announced → keep, lineup_status='tbc'
    Requires fixture_id and player_id columns in tips.
    """
    if tips.empty or "fixture_id" not in tips.columns or "player_id" not in tips.columns:
        return tips

    fixture_ids = tips["fixture_id"].dropna().unique().astype(int)
    lineup_map: dict[int, dict] = {}
    for fid in fixture_ids:
        lineup_map[int(fid)] = fetch_lineup(int(fid))

    def _status(row) -> str:
        fid = row.get("fixture_id")
        pid = row.get("player_id")
        if not fid or not pid:
            return "tbc"
        lu = lineup_map.get(int(fid), {})
        if not lu.get("confirmed"):
            return "tbc"
        return "confirmed" if int(pid) in lu["starters"] else "benched"

    tips = tips.copy()
    tips["lineup_status"] = tips.apply(_status, axis=1)

    before = len(tips)
    tips = tips[tips["lineup_status"] != "benched"].copy()
    dropped = before - len(tips)
    if dropped:
        print(f"[lineup] Dropped {dropped} tips — players confirmed benched/not in squad")

    confirmed = (tips["lineup_status"] == "confirmed").sum()
    tbc       = (tips["lineup_status"] == "tbc").sum()
    print(f"[lineup] {confirmed} confirmed starters | {tbc} lineup TBC")
    return tips


def _enrich_with_recent_form(history: pd.DataFrame) -> None:
    """Use API-Football to update rolling stats for players in upcoming matches."""
    try:
        from player_model.api_football import get_recent_fixtures, get_fixture_player_stats
        from player_model.feature_engineering import build_rolling_features

        bets_csv = config.OUTPUT_DIR / "bets.csv"
        if not bets_csv.exists():
            return

        bets = pd.read_csv(bets_csv)
        leagues_needed = bets["league"].unique().tolist()

        for league, lg_id in config.PROP_LEAGUES.items():
            if not any(league.lower() in l.lower() for l in leagues_needed):
                continue
            season = config.PROP_SEASONS.get(league, "2025")
            fixtures = get_recent_fixtures(lg_id, season, last_n=5)
            print(f"  [{league}] {len(fixtures)} recent fixtures fetched")

            for fix in fixtures[:3]:  # limit to 3 most recent to save API calls
                fix_id = fix.get("fixture", {}).get("id")
                if not fix_id:
                    continue
                player_stats = get_fixture_player_stats(fix_id)
                # Update rolling stats in history for matched players
                for stat in player_stats:
                    pid = stat.get("player_id")
                    if pid and len(history[history["player_id"] == pid]) > 0:
                        idx = history[history["player_id"] == pid].index[-1]
                        # Update with most recent match data
                        mins = int(stat.get("minutes_played") or 0)
                        if mins > 0:
                            history.at[idx, "goals_pg"]  = stat.get("goals", 0)
                            history.at[idx, "sot_pg"]    = stat.get("shots_on_target", 0)
                            history.at[idx, "cards_pg"]  = stat.get("yellow_card", 0)

    except Exception as e:
        print(f"  [enrich] Error enriching with API-Football: {e}")


# ── Season stats enrichment ───────────────────────────────────────────────────

def mode_enrich_season() -> None:
    """
    Fetch /players/statistics for every unique player in the parquet and merge
    season-level averages as new columns: season_goals_pg, season_sot_pg,
    season_shots_pg, season_assists_pg, season_cards_pg, season_minutes_pg,
    season_appearances.

    Safe to re-run — existing values updated, missing players get NaN.
    Run after collect to enrich the parquet before retraining.
    Usage: python -m player_model.pipeline --mode enrich-season
    """
    if not HISTORY_CACHE.exists():
        print("[enrich-season] No history. Run --mode collect first.")
        return

    df = pd.read_parquet(HISTORY_CACHE)
    unique_pids = df["player_id"].dropna().unique().tolist()
    total = len(unique_pids)
    print(f"[enrich-season] Fetching season stats for {total} unique players (batches of 500)...")

    season_cols = [
        "season_appearances", "season_goals_pg", "season_assists_pg",
        "season_shots_pg", "season_sot_pg", "season_cards_pg", "season_minutes_pg",
    ]

    BATCH_SIZE = 500
    all_stats: dict = {}

    for batch_start in range(0, total, BATCH_SIZE):
        batch_pids = set(unique_pids[batch_start : batch_start + BATCH_SIZE])
        batch_df   = df[df["player_id"].isin(batch_pids)]
        batch_stats = fetch_all_player_season_stats(batch_df, leagues=APIFOOTBALL_LEAGUES)
        all_stats.update(batch_stats)

        # Checkpoint: merge what we have so far and write parquet
        for col in season_cols:
            df[col] = df["player_id"].map(
                lambda pid, c=col: all_stats.get(pid, {}).get(c, float("nan"))
            )
        df.to_parquet(HISTORY_CACHE, index=False)
        done = min(batch_start + BATCH_SIZE, total)
        enriched_so_far = df["season_goals_pg"].notna().sum()
        print(f"[enrich-season] Checkpoint {done}/{total} players | {enriched_so_far}/{len(df)} rows enriched → saved")

    enriched = df["season_goals_pg"].notna().sum()
    print(f"[enrich-season] Done. {enriched}/{len(df)} rows enriched -> {HISTORY_CACHE.name}")


# ── Player profile enrichment ─────────────────────────────────────────────────

def mode_enrich_profiles():
    """Fetch player profile data (age, height, weight) for all players in parquet."""
    from player_model.data_fetcher import fetch_all_player_profiles
    if not HISTORY_CACHE.exists():
        print("[enrich-profiles] No player_history.parquet found. Run --mode collect first.")
        return
    df = pd.read_parquet(HISTORY_CACHE)
    unique_pids = df["player_id"].dropna().unique().tolist()
    total = len(unique_pids)
    print(f"[enrich-profiles] Fetching profiles for {total} unique players (batches of 1000)...")

    BATCH_SIZE = 1000
    # Initialize profile columns if missing — accumulated across batches via map-update
    for col in ["age", "height_cm", "weight_kg"]:
        if col not in df.columns:
            df[col] = float("nan")

    for batch_start in range(0, total, BATCH_SIZE):
        batch_pids = set(unique_pids[batch_start : batch_start + BATCH_SIZE])
        batch_df   = df[df["player_id"].isin(batch_pids)]
        profiles   = fetch_all_player_profiles(batch_df)
        if not profiles:
            continue
        prof_df = pd.DataFrame.from_dict(profiles, orient="index")
        prof_df.index.name = "player_id"
        prof_df = prof_df.reset_index()
        # Map new values onto df rows without dropping the column (preserves prior batches)
        for col in ["age", "height_cm", "weight_kg"]:
            if col in prof_df.columns:
                pid_map = prof_df.set_index("player_id")[col].to_dict()
                df[col] = df["player_id"].map(pid_map).combine_first(df[col])
        # Merge-on-write: re-read latest parquet to preserve columns written by parallel processes
        latest = pd.read_parquet(HISTORY_CACHE)
        for col in ["age", "height_cm", "weight_kg"]:
            if col in df.columns:
                latest[col] = df[col]
        latest.to_parquet(HISTORY_CACHE, index=False)
        done = min(batch_start + BATCH_SIZE, total)
        enriched_so_far = df["age"].notna().sum() if "age" in df.columns else 0
        print(f"[enrich-profiles] Checkpoint {done}/{total} players | {enriched_so_far}/{len(df)} rows → saved")

    enriched = df["age"].notna().sum() if "age" in df.columns else 0
    print(f"[enrich-profiles] Done. {enriched}/{len(df)} rows have age data -> {HISTORY_CACHE}")


# ── Player sidelined enrichment ───────────────────────────────────────────────

def mode_enrich_sidelined():
    """Fetch sidelined/injury history for all players and compute chronic injury features."""
    from player_model.data_fetcher import fetch_all_player_sidelined
    import datetime as dt
    if not HISTORY_CACHE.exists():
        print("[enrich-sidelined] No player_history.parquet found.")
        return
    df = pd.read_parquet(HISTORY_CACHE)
    unique_pids = df["player_id"].dropna().unique().tolist()
    total = len(unique_pids)
    print(f"[enrich-sidelined] Fetching sidelined history for {total} players (batches of 1000)...")

    today = dt.date.today()

    def _compute_injury_features(pid: int, match_date, sidelined_map: dict) -> dict:
        entries = sidelined_map.get(int(pid), [])
        if not entries:
            return {"chronic_injury_risk": 0.0, "days_since_last_injury": 365.0, "return_from_injury_flag": 0.0}
        if hasattr(match_date, "date"):
            match_d = match_date.date()
        else:
            try:
                match_d = dt.date.fromisoformat(str(match_date)[:10])
            except Exception:
                match_d = today
        year_ago = match_d - dt.timedelta(days=365)
        recent = [e for e in entries if e.get("start") and dt.date.fromisoformat(e["start"][:10]) >= year_ago]
        past = sorted([e for e in entries if e.get("end") and dt.date.fromisoformat(e["end"][:10]) < match_d],
                      key=lambda x: x["end"], reverse=True)
        days_since = 365.0
        if past:
            last_end = dt.date.fromisoformat(past[0]["end"][:10])
            days_since = float((match_d - last_end).days)
        return_flag = float(0 < days_since <= 21)
        return {
            "chronic_injury_risk":     float(len(recent)),
            "days_since_last_injury":  days_since,
            "return_from_injury_flag": return_flag,
        }

    BATCH_SIZE = 1000
    for batch_start in range(0, total, BATCH_SIZE):
        batch_pids = set(unique_pids[batch_start : batch_start + BATCH_SIZE])
        batch_df   = df[df["player_id"].isin(batch_pids)]
        sidelined_map = fetch_all_player_sidelined(batch_df)
        if not sidelined_map:
            continue

        batch_rows = df[df["player_id"].isin(batch_pids)].index
        for idx in batch_rows:
            row   = df.loc[idx]
            feats = _compute_injury_features(row.get("player_id", 0), row.get("date"), sidelined_map)
            for col, val in feats.items():
                df.at[idx, col] = val

        # Merge-on-write: re-read latest parquet to preserve columns written by parallel processes
        latest = pd.read_parquet(HISTORY_CACHE)
        for col in ["chronic_injury_risk", "days_since_last_injury", "return_from_injury_flag"]:
            if col in df.columns:
                latest[col] = df[col]
        latest.to_parquet(HISTORY_CACHE, index=False)
        done = min(batch_start + BATCH_SIZE, total)
        print(f"[enrich-sidelined] Checkpoint {done}/{total} players → saved")

    print(f"[enrich-sidelined] Done. {len(df)} rows updated -> {HISTORY_CACHE}")


def mode_enrich_sidelined_live() -> None:
    """Refresh sidelined/injury data for players active in the last 60 days — fast pre-predict refresh."""
    from player_model.data_fetcher import fetch_all_player_sidelined
    import datetime as dt

    if not HISTORY_CACHE.exists():
        print("[enrich-sidelined-live] No player_history.parquet found.")
        return

    if _match_day_gate_skips():
        print("[enrich-sidelined-live] Post-WC match-day gating: no prop fixtures in next 72h — skipping.")
        return

    df = pd.read_parquet(HISTORY_CACHE)
    cutoff = pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=60)
    if "date" in df.columns:
        recent_mask = pd.to_datetime(df["date"], errors="coerce", utc=True) >= cutoff
        active_pids = df.loc[recent_mask, "player_id"].dropna().unique().tolist()
    else:
        active_pids = df["player_id"].dropna().unique().tolist()

    # Limit to PROP_LEAGUES players only
    prop_league_names = set(config.PROP_LEAGUES.keys())
    if "league" in df.columns and prop_league_names:
        league_mask = df["league"].isin(prop_league_names)
        prop_pids = set(df.loc[league_mask, "player_id"].dropna().unique())
        active_pids = [p for p in active_pids if p in prop_pids]

    total = len(active_pids)
    print(f"[enrich-sidelined-live] Refreshing {total} active players (last 60 days in prop leagues)...")

    today = dt.date.today()
    active_df = df[df["player_id"].isin(set(active_pids))]
    sidelined_map = fetch_all_player_sidelined(active_df)
    if not sidelined_map:
        print("[enrich-sidelined-live] No sidelined data returned.")
        return

    def _compute_injury_features(pid: int, sidelined_map: dict) -> dict:
        entries = sidelined_map.get(int(pid), [])
        if not entries:
            return {"chronic_injury_risk": 0.0, "days_since_last_injury": 365.0, "return_from_injury_flag": 0.0}
        year_ago = today - dt.timedelta(days=365)
        recent = [e for e in entries if e.get("start") and dt.date.fromisoformat(e["start"][:10]) >= year_ago]
        past = sorted([e for e in entries if e.get("end") and dt.date.fromisoformat(e["end"][:10]) < today],
                      key=lambda x: x["end"], reverse=True)
        days_since = 365.0
        if past:
            last_end = dt.date.fromisoformat(past[0]["end"][:10])
            days_since = float((today - last_end).days)
        return {
            "chronic_injury_risk":     float(len(recent)),
            "days_since_last_injury":  days_since,
            "return_from_injury_flag": float(0 < days_since <= 21),
        }

    active_pids_set = set(active_pids)
    for idx in df[df["player_id"].isin(active_pids_set)].index:
        pid = df.at[idx, "player_id"]
        feats = _compute_injury_features(pid, sidelined_map)
        for col, val in feats.items():
            df.at[idx, col] = val

    df.to_parquet(HISTORY_CACHE, index=False)
    print(f"[enrich-sidelined-live] Done. {total} players refreshed -> {HISTORY_CACHE}")


def mode_squad_sync() -> None:
    """
    Update player team assignments in player_history.parquet.
    For each league in APIFOOTBALL_LEAGUES, fetches current squads and updates
    the team field for players who have transferred since last collect.
    Run weekly (or before predict) to keep team assignments accurate.
    Usage: python -m player_model.pipeline --mode squad-sync
    """
    if not HISTORY_CACHE.exists():
        print("[squad-sync] No player_history.parquet. Run --mode collect first.")
        return

    df = pd.read_parquet(HISTORY_CACHE)
    current_teams: dict[int, str] = {}

    for league_name, (league_id, season) in APIFOOTBALL_LEAGUES.items():
        print(f"  [{league_name}] fetching teams...")
        teams = fetch_league_teams(league_id, season)
        for t in teams:
            squad = fetch_current_squad(t["team_id"], season)
            for p in squad:
                pid = p["player_id"]
                if pid and pid not in current_teams:
                    current_teams[pid] = t["team_name"]

    updates = 0
    for pid, current_team in current_teams.items():
        mask = df["player_id"] == pid
        if not mask.any():
            continue
        last_team = df.loc[mask, "team"].iloc[-1]
        if last_team != current_team:
            # Update last 30 rows only (current season games)
            idx = df[mask].tail(30).index
            df.loc[idx, "team"] = current_team
            updates += 1
            print(f"  [squad-sync] player {pid}: {last_team} → {current_team}")

    df.to_parquet(HISTORY_CACHE, index=False)
    print(f"[squad-sync] Done. {updates} players updated | {len(current_teams)} players tracked.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Player Model Pipeline")
    parser.add_argument("--mode", choices=["collect", "collect-wc", "train", "predict", "all",
                                           "enrich-season", "enrich-profiles", "enrich-sidelined",
                                           "enrich-sidelined-live", "squad-sync"], required=True)
    parser.add_argument("--extended", action="store_true",
                        help="Also collect Champions League / Europa League / Conference League")
    parser.add_argument("--last-n", type=int, default=99,
                        help="Number of recent fixtures to collect per league (max 99, default: 99)")
    args = parser.parse_args()

    if args.mode == "collect" or args.mode == "all":
        mode_collect(extended=args.extended, last_n=args.last_n)
    if args.mode == "collect-wc":
        mode_collect_wc(last_n=args.last_n)
    if args.mode == "enrich-season":
        mode_enrich_season()
    if args.mode == "enrich-profiles":
        mode_enrich_profiles()
    if args.mode == "enrich-sidelined":
        mode_enrich_sidelined()
    if args.mode == "enrich-sidelined-live":
        mode_enrich_sidelined_live()
    if args.mode == "squad-sync":
        mode_squad_sync()
    if args.mode == "train" or args.mode == "all":
        mode_train()
    if args.mode == "predict" or args.mode == "all":
        mode_predict()


if __name__ == "__main__":
    main()
