"""
Player Prop Predictions v2
===========================
Generates player prop signals for today's SNIPER/MARKSMAN matches.
Uses de-vigged edge, relative edge floors, and 5-component confidence scoring.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

from . import config

_APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY", "")
# Club identity: src.team_names.resolve (via config._v9_src), player identity: player_id. The
# api_football name matchers (_same_club / _club_name_subset / _norm_name) are no longer used here.
from .feature_engineering import build_upcoming_features, compute_ges, _build_history_index
from .model import load_model, predict_proba

import numpy as np


def _compute_market_caps(history_path: Path) -> dict[str, float]:
    """
    Compute per-market probability caps = top-1% career rate among players
    with >= 10 matches in player_history.parquet.
    No model should output a probability above the best real player's career rate.
    """
    _COL_THRESHOLD = {
        "goals":   ("goals",           1),
        "goals2":  ("goals",           2),
        "assists": ("assists",         1),
        "sot":     ("shots_on_target", 1),
        "sot2":    ("shots_on_target", 2),
        "sot3":    ("shots_on_target", 3),
        "cards":   ("yellow_cards",    1),
    }
    caps: dict[str, float] = {}
    # Data Truth (2026-11-01): no history -> no caps -> the run FAILS (raise), and a market whose
    # column is absent gets NO cap -> that market emits no tip (logged by the caller). Both used
    # to fall back to a typed 0.95.
    if not history_path.exists():
        raise FileNotFoundError(f"market caps need {history_path}; refusing to tip without them")
    hist = pd.read_parquet(history_path)
    for mkt, (col, thr) in _COL_THRESHOLD.items():
        if col not in hist.columns:
            continue
        # Rate among STARTS, and the BEST player's rate — not the 99th percentile of all
        # appearances. The old version was wrong twice over, against its own docstring:
        #
        #   1. It took rates.quantile(0.99), which by construction sits BELOW the best 1% of
        #      players. "No probability above the best real player's rate" needs the max.
        #   2. It counted every appearance including substitute cameos — 11.1% of rows are
        #      under 20 minutes, and a striker who plays 10 minutes almost never records 3
        #      shots on target, which drags his career rate down.
        #
        # Measured 2026-08-18 the two errors compounded badly:
        #     sot3    old cap 0.1288   true best starter 0.5000  (Mbappé, 3+ SOT in half his starts)
        #     goals2  old cap 0.0878   true best starter 0.2755  (Mbappé)
        #
        # Both caps landed BELOW config.MIN_SIGNAL_PROB (0.15), so every prediction in those
        # markets was discarded and neither had produced a single row in player_tips.csv or
        # player_ledger.csv, ever — while bookmakers priced sot3 311 times in five days.
        #
        # Starts only, and at least MIN_STARTS_FOR_CAP of them.
        #
        # Taking the max makes the threshold matter: at >= 10 starts the caps were being set
        # by hot streaks rather than sustained ability —
        #     goals   0.7273 from Robert Glatzel on 11 starts  (0.6762 at >= 20)
        #     assists 0.4545 from Teemu Pukki on 11 starts     (0.4167 at >= 20)
        #     cards   0.6000 from Diego González on 10 starts  (0.5517 at >= 20)
        # while the caps that actually bind (sot2/sot3/goals2, all Mbappé on 98 starts) do not
        # move at all. So 20 trims noise without tightening anything real, and every market
        # stays comfortably above config.MIN_SIGNAL_PROB.
        MIN_STARTS_FOR_CAP = 20
        starts = hist
        if "minutes" in hist.columns:
            starts = hist[pd.to_numeric(hist["minutes"], errors="coerce").fillna(0) >= 60.0]
        if starts.empty:
            starts = hist
        # Per player_id (two players sharing a name used to pool into one "player").
        rates = (starts.groupby("player_id")
                       .filter(lambda g: len(g) >= MIN_STARTS_FOR_CAP)
                       .groupby("player_id")[col]
                       .apply(lambda x: (x >= thr).mean()))
        if rates.empty:
            continue
        caps[mkt] = round(min(float(rates.max()), 0.95), 4)
    return caps


# ── Edge calculations (v2 — proper de-vig) ────────────────────────────────────

def _devig_fair_prob(market_odds: float) -> float:
    """
    Compute fair (de-vigged) implied probability.
    Uses assumed overround based on odds level. No usable price -> NaN (was 0.5).
    """
    if not market_odds or market_odds <= 1.0:
        return float("nan")
    raw = 1.0 / market_odds
    # Find assumed overround
    overround = config.OVERROUND_BY_ODDS[99.0]  # default high
    for min_odds, ovr in sorted(config.OVERROUND_BY_ODDS.items()):
        if market_odds <= min_odds:
            overround = ovr
            break
    return raw / overround


def _ev(p_model: float, market_odds: float) -> float:
    return round(p_model * market_odds - 1, 4)


def _relative_edge(p_model: float, fair_prob: float) -> float:
    if fair_prob <= 0:
        return 0.0
    return round((p_model - fair_prob) / fair_prob, 4)


def _edge_passes_floor(rel_edge: float, market_odds: float) -> bool:
    """Check relative edge meets the floor for this odds tier."""
    for min_odds in sorted(config.RELATIVE_EDGE_FLOORS.keys(), reverse=True):
        if market_odds >= min_odds:
            return rel_edge >= config.RELATIVE_EDGE_FLOORS[min_odds]
    return rel_edge >= config.RELATIVE_EDGE_FLOORS[min(config.RELATIVE_EDGE_FLOORS.keys())]


def _kelly_stake(ev: float, market_odds: float) -> float:
    if market_odds <= 1.0:
        return 0.0
    full_kelly = ev / (market_odds - 1)
    return max(0.0, round(full_kelly * config.KELLY_FRACTION, 4))


def _team_strength_mult(team_win_prob: float, market: str) -> float:
    """
    Bidirectional team-strength adjustment on model probability.

    Favourite  (win_prob > 60%): boost  — up to ×1.30 at 90%+
    Balanced   (40% – 60%)     : no change
    Underdog   (win_prob < 40%): penalty — down to ×0.40 at 5%-

    Cards exempt: card rate doesn't scale cleanly with team dominance.
    Assists: 70% of the adjustment.
    """
    if market == "cards":
        return 1.0

    if team_win_prob >= 0.60:
        raw = min(team_win_prob - 0.60, 0.30)   # 0.0 → 0.30
        adj = raw
        if market == "assists":
            adj *= 0.70
        return round(1.0 + adj, 3)

    if team_win_prob < 0.40:
        # Extended penalty for heavy underdogs (Panama vs England = ~5% win prob)
        # Linear from 0.40 (no change) → 0.05 (×0.40 max penalty)
        raw = min(0.40 - team_win_prob, 0.35)   # 0.0 → 0.35
        adj = raw * (0.60 / 0.35)               # scale so 0.35 raw → 0.60 penalty
        adj = min(adj, 0.60)                    # cap at ×0.40 floor
        if market == "assists":
            adj *= 0.70
        return round(max(1.0 - adj, 0.40), 3)

    return 1.0  # 40–60%: balanced game, no adjustment


# ── Confidence scoring ────────────────────────────────────────────────────────

def _confidence_score(row: dict, lazy_factor_count: int = 0) -> float:
    """5-component confidence score (0.0-1.0)."""
    n_games = row.get("n_games", 0)

    # MEASURED COMPONENTS ONLY (Data Truth audit finding 52, owner decision 2026-11-01). The score
    # used to add a literal recency 0.8 and model agreement 1.0 (+0.40 on every row) and a
    # p_start of 0.8 when absent - 45% of it was constant. Weights of the measured parts are kept
    # and renormalised; p_start counts only when it was actually computed.
    vol_score = min(1.0, n_games / 20.0) * 0.70 + (min(n_games, 5) / 5.0) * 0.30
    lazy_score = min(lazy_factor_count / 2.0, 1.0)
    parts = [(0.30, vol_score), (0.15, lazy_score)]
    p_start = row.get("p_start")
    if p_start is not None and p_start == p_start:
        parts.append((0.10, float(p_start)))
    return round(sum(w * v for w, v in parts) / sum(w for w, _ in parts), 3)


# ── Tier classification ───────────────────────────────────────────────────────

def _classify_tier(
    ev: float, market_odds: float, rel_edge: float,
    confidence: float, lazy_count: int, ges: float = float("nan"),
    market: str = "", position: str = ""
) -> str:
    """Classify signal into SNIPER/MARKSMAN/VALUABLE/AVOID (WATCH removed).

    Applies a real-odds-backtest role×market gate: clear money-losing combos
    (defenders→goals, multi-SOT/goals2) are capped at VALUABLE (data-only) rather
    than producing a real SNIPER/MARKSMAN tip.
    """
    is_goals_sot = market in ("goals", "sot")

    # GES gate for goals/SOT — below the floor = no signal. A GES that could not be computed (NaN)
    # is no signal either: NaN compares False, so without the explicit test it would pass every gate.
    if is_goals_sot and not (ges == ges and ges >= config.GES_SUPPRESS):
        return "AVOID"
    if not (ev == ev and rel_edge == rel_edge):
        return "AVOID"

    # Use relative edge gates instead of hard market_odds floors.
    # rel_edge = (model_prob - fair_prob) / fair_prob — odds-level-agnostic.
    _sniper_re   = getattr(config, "REL_EDGE_SNIPER",   0.20)
    _marksman_re = getattr(config, "REL_EDGE_MARKSMAN", 0.12)
    # Confidence floors apply only when the gate is ON (config.PROPS_CONFIDENCE_GATE). It is OFF:
    # the honest score is uninformative today (n_games is the window size, 5 on every row), so a
    # floor could only pass everything or nothing. Measured 2026-11-01: identical tiers with and
    # without the gate on the live board (686 AVOID / 41 VALUABLE / 2 MARKSMAN).
    if not getattr(config, "PROPS_CONFIDENCE_GATE", False):
        confidence = 1.0

    if (ev >= config.SNIPER_EV and rel_edge >= _sniper_re
            and confidence >= config.CONFIDENCE_FLOORS["SNIPER"]
            and lazy_count >= 2
            and _edge_passes_floor(rel_edge, market_odds)
            and (not is_goals_sot or ges >= config.GES_SNIPER_MIN)):
        tier = "SNIPER"
    elif (ev >= config.MARKSMAN_EV and rel_edge >= _marksman_re
            and confidence >= config.CONFIDENCE_FLOORS["MARKSMAN"]
            and lazy_count >= 1
            and _edge_passes_floor(rel_edge, market_odds)
            and (not is_goals_sot or ges >= config.GES_MARKSMAN_MIN)):
        tier = "MARKSMAN"
    elif (ev >= config.VALUABLE_EV
            and confidence >= config.CONFIDENCE_FLOORS["VALUABLE"]
            and _edge_passes_floor(rel_edge, market_odds)):
        tier = "VALUABLE"
    else:
        return "AVOID"

    # ── Role × market gate (real-odds backtest Jun 2026) ──────────────────────
    # Cap clear money-losers at VALUABLE (data-only): multi-SOT/goals2 markets,
    # and defenders for anytime-goalscorer.
    pos = (position or "").strip().upper()[:1]
    if tier in ("SNIPER", "MARKSMAN") and (
        market in getattr(config, "VALUABLE_ONLY_MARKETS", set())
        or (pos, market) in getattr(config, "VALUABLE_ONLY_ROLE_MARKET", set())
    ):
        tier = "VALUABLE"
    return tier


# ── Lazy market factor detection ──────────────────────────────────────────────

def _detect_lazy_factors(row: dict, feat_row: dict, market: str) -> list[str]:
    """Identify which lazy market factors are active for this player/match."""
    factors = []

    def _v(d, k):          # measured value or NaN; a missing input never activates a factor
        try:
            return float(d.get(k))
        except (TypeError, ValueError):
            return float("nan")

    if market == "sot":
        sp_score = _v(feat_row, "set_piece_threat_score")
        # Threshold: aerial 45% × corners at league avg (6) × 0.30 sp rate = 0.135
        if sp_score > 0.135 and (feat_row.get("pos_defender", 0) or feat_row.get("pos_midfielder", 0)):
            factors.append("SET_PIECE")

    if market == "cards":
        if _v(feat_row, "referee_strictness") > 0.5:
            factors.append("REFEREE")

    if market == "sot" and _v(feat_row, "opp_sot_conceded_pg") > 5.5:
        factors.append("KEEPER_WEAK")

    if _v(feat_row, "minutes_pg") < 65 and _v(row, "minutes_est") > 75:
        factors.append("MINUTES")

    return factors


# ── Main prediction function ──────────────────────────────────────────────────

def run_player_predictions(
    bets_csv: Path,
    history_df: pd.DataFrame,
    referee_profiles: dict | None = None,
    match_contexts: dict | None = None,
) -> pd.DataFrame:
    """
    Generate player prop signals for ALL upcoming fixtures in PROP_LEAGUES.
    Decoupled from team model — runs for Premier League, Bundesliga etc.
    even when we don't have a team O/U signal for those leagues.

    bets_csv:         output/bets.csv (used to add context for our standard leagues)
    history_df:       player season stats from FBref training data
    referee_profiles: {fixture_id: {strictness_score, yellows_per_game, lg_yellows_per_match}}
    match_contexts:   {match_key: {team_corners_per90, opp_sot_conceded_pg, ...}}
    """
    if history_df.empty:
        return pd.DataFrame()

    # Compute caps once per run from the actual history distribution
    _history_path = config.BASE_DIR / "player_history.parquet"
    try:
        _MARKET_CAPS = _compute_market_caps(_history_path)
    except FileNotFoundError as _e:
        print(f"[player_model] FAIL: {_e}")
        return pd.DataFrame()
    print(f"[player_model] market caps (top-1% real player): { {k: v for k,v in _MARKET_CAPS.items()} }")
    _uncapped = [m for m in config.MARKETS if m not in _MARKET_CAPS]
    if _uncapped:
        print(f"[player_model] market(s) with no measurable cap emit no tip this run: {_uncapped}")

    # STRUCTURALLY DEAD MARKET GUARD.
    #
    # Two independent safety mechanisms can silently cancel each other out: p_model is capped
    # at the top-1% real-player rate, and then dropped if it falls below MIN_SIGNAL_PROB. When
    # a market's cap is BELOW the floor, every prediction in that market is discarded and the
    # market can never emit a single tip — with no error anywhere.
    #
    # Measured 2026-08-18: sot3 cap 0.1288 and goals2 cap 0.0878 against a 0.15 floor. Neither
    # market has produced one row in player_tips.csv or player_ledger.csv, ever, while
    # bookmakers priced sot3 311 times in five days. This log makes that visible instead of
    # leaving it to be rediscovered by hand.
    _dead = {m: c for m, c in _MARKET_CAPS.items()
             if m in config.MARKETS and c < config.MIN_SIGNAL_PROB}
    if _dead:
        print(f"[player_model] WARNING: market(s) cannot emit ANY tip — cap below the "
              f"{config.MIN_SIGNAL_PROB} probability floor: "
              f"{ {m: round(c, 4) for m, c in _dead.items()} }. "
              f"Either the floor is wrong for them or they should be retired.")

    today = pd.Timestamp.now().normalize()
    # Only the near-term slate. next_n pulls the next N fixtures per league even if weeks away
    # (off-season leagues), which balloons the per-match lineup/feature work — the main reason
    # a club-season run crawled to ~60 min while WC (one league, few games) ran in ~2 min.
    _horizon = today + pd.Timedelta(days=4)

    # Primary: ALL upcoming fixtures in PROP_LEAGUES (decoupled from team model)
    prop_matches = []
    _n_undated = 0
    if _APIFOOTBALL_KEY:
        try:
            from player_model.api_football import get_fixtures_between
            _from, _to = today.strftime("%Y-%m-%d"), _horizon.strftime("%Y-%m-%d")
            for league, lg_id in config.PROP_LEAGUES.items():
                season = config.prop_season(league)
                # The whole window, not `next=5`: five kickoffs cut every full round in half.
                fixtures = get_fixtures_between(lg_id, season, _from, _to)
                for fix in fixtures:
                    # A date window also returns today's live and finished games; `next` never did.
                    if fix.get("fixture", {}).get("status", {}).get("short", "NS") not in ("NS", "TBD"):
                        continue
                    teams    = fix.get("teams", {})
                    dt_full  = fix.get("fixture", {}).get("date", "")
                    dt       = dt_full[:10]
                    if not dt:
                        # No kickoff date -> not a schedulable fixture (was dated "today").
                        _n_undated += 1
                        continue
                    _fd = pd.Timestamp(dt)
                    if _fd < today or _fd > _horizon:
                        continue   # skip weeks-away fixtures — only the near-term slate
                    prop_matches.append({
                        "home_team":    teams.get("home", {}).get("name", ""),
                        "away_team":    teams.get("away", {}).get("name", ""),
                        "league":       league,
                        "date":         _fd,
                        "kickoff_utc":  dt_full,
                        "signal_tier":  "PROP_LEAGUE",
                        "fixture_id":   fix.get("fixture", {}).get("id"),
                    })
        except Exception as e:
            print(f"[predict] API-Football fixtures fetch failed: {e}")

    _n_from_api = len(prop_matches)

    # Fallback / supplement: our team model matches (bets.csv)
    if bets_csv and bets_csv.exists():
        bets_df = pd.read_csv(bets_csv)
        bets_df["date"] = pd.to_datetime(bets_df["date"], errors="coerce")
        bets_df = bets_df[bets_df["date"] >= today]
        for _, r in bets_df.iterrows():
            prop_matches.append({
                "home_team":   r["home_team"],
                "away_team":   r["away_team"],
                "league":      r.get("league", ""),
                "date":        r["date"],
                "signal_tier": r.get("signal_tier", ""),
                "fixture_id":  None,
            })

    # Where the fixture slate came from. Until now an EMPTY API response was completely
    # silent — only an exception printed anything — so "no player tips generated" gave no clue
    # whether the cause was zero fixtures, zero squads or zero odds. player_props.yml reported
    # SUCCESS on every hourly run from 2026-08-15 to 08-17 while writing nothing, and this is
    # the line that would have identified it on day one.
    print(f"[predict] fixture slate: {_n_from_api} from PROP_LEAGUES via API-Football, "
          f"{len(prop_matches) - _n_from_api} from bets.csv supplement"
          + (f"; {_n_undated} API fixture(s) without a kickoff date skipped" if _n_undated else ""))

    if not prop_matches:
        print("[predict] NO FIXTURES AT ALL — API-Football returned nothing for every prop "
              "league and bets.csv contributed none. Zero tips is a data-availability "
              "failure here, not a model decision.")
        return pd.DataFrame()

    # Deduplicate. An API-Football fixture is identified by its fixture_id. A bets.csv row (no id)
    # duplicates an API fixture when BOTH clubs resolve (src.team_names.resolve) onto that
    # fixture's clubs in the same canonical league on the same date; it is otherwise kept once.
    # (Was a raw "home|away|date" string key, so a spelling variant produced a second fixture.)
    _reg = config.league_registry()
    _resolve = config._v9_src("team_names").resolve
    seen_ids, seen_rows = set(), set()
    api_by_day: dict = {}
    bets_list = []
    for m in prop_matches:
        m["league"] = _reg.canonical_name(m["league"]) or m["league"]
        day = str(m["date"])[:10]
        fid = m.get("fixture_id")
        if fid is not None and not pd.isna(fid):
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            api_by_day.setdefault((m["league"], day), []).append((m["home_team"], m["away_team"]))
            bets_list.append(m)
            continue
        api = api_by_day.get((m["league"], day), [])
        rh = _resolve(m["home_team"], [h for h, _ in api]) if api else None
        ra = _resolve(m["away_team"], [a for _, a in api]) if api else None
        if rh and ra and (rh, ra) in api:
            continue
        key = (m["league"], m["home_team"], m["away_team"], day)
        if key not in seen_rows:
            seen_rows.add(key)
            bets_list.append(m)

    import pandas as _pd
    bets = _pd.DataFrame(bets_list)

    # ── Pre-fetch injury lists once per league (4 h cache) ────────────────────
    injured_cache: dict[str, set] = {}
    if _APIFOOTBALL_KEY:
        try:
            from player_model.api_football import get_injured_players as _gip
            from datetime import date as _date
            _today_str = _date.today().isoformat()
            # Only leagues that actually have upcoming matches this run — skip the off-season
            # ones so we don't burn API calls (and time) on leagues with zero fixtures.
            _bet_leagues = set(bets["league"].astype(str).unique()) if not bets.empty else set()
            for _lg, _lg_id in config.PROP_LEAGUES.items():
                if _bet_leagues and _lg not in _bet_leagues:
                    continue
                _szn = config.prop_season(_lg)
                injured_cache[_lg] = _gip(_lg_id, _szn, _today_str)
            _n_inj = sum(len(v) for v in injured_cache.values())
            if _n_inj:
                print(f"[predict] Injury filter: {_n_inj} injured/suspended players loaded")
        except Exception as _e:
            print(f"[predict] Injury fetch failed: {_e}")

    payloads = {m: load_model(m) for m in config.MARKETS}
    missing  = [m for m, p in payloads.items() if p is None]
    if missing:
        print(f"[player_model] Models not trained: {missing}. Run --mode train first.")
        return pd.DataFrame()

    all_tips = []

    # PROSPECTIVE EVIDENCE FOR THE PROMOTION GATE (PG-PROPS, 2026-10-25). When a retrain has parked
    # a challenger for a market, score it on the same player-fixtures as the serving model and log
    # both, first-sight, before kickoff. Recorder only: it never changes a tip and never raises.
    try:
        from player_model.props_gate import FirstSightRecorder
        _first_sight = FirstSightRecorder()
    except Exception as _e:                                            # noqa: BLE001
        print(f"[predict] first-sight recorder unavailable: {_e}")
        _first_sight = None
    # R2-D repaired props recipe in SHADOW (owner decision 2026-11-01): scored on the same
    # player-fixtures, logged first-sight to output/props_r2_shadow.csv. Never changes a tip.
    try:
        from player_model.props_r2_shadow import Recorder as _R2Recorder
        _r2_shadow = _R2Recorder()
    except Exception as _e:                                            # noqa: BLE001
        print(f"[predict] props R2 shadow unavailable: {_e}")
        _r2_shadow = None

    # Perf: index the history ONCE and lowercase the team column ONCE. Previously every match
    # re-scanned the full 230k-row history per player (O(matches x players x rows)) which pushed
    # the CI predict past its 60-min timeout → cancelled → zero tips. Behaviour is unchanged.
    _hist_index = _build_history_index(history_df)

    # A player belongs to the club he plays for NOW — not to every club he has ever
    # played for. Selecting on ANY historical row put Tyrhys Dolan in both
    # "Espanyol v Levante" (correct: he moved there in 2025) and "Wolves v Blackburn"
    # (a ghost — he had left Blackburn) on the same weekend of 2026-08-16, and the
    # grader then settled the ghost row off a goal he scored for Espanyol. Villarreal's
    # roster picked up Matteo Gabbia (Milan) and Yerson Mosquera (Wolves) the same way.
    # It also defeated the end-of-run dedup, whose key includes `match`: the same
    # player+market in two different fixtures reads as two distinct tips, which is the
    # duplicate Telegram notification reported on 2026-08-15.
    # Resolve every player_id to the team on his most recent dated row instead.
    # NOTE: history currently ends 2026-07-04, so a transfer completed after that date
    # is still invisible here — that needs player_history.parquet to advance (or a
    # /players/squads overlay), not a change to this rule.
    #
    # Club rows ONLY. History also holds internationals, where `team` is the player's
    # COUNTRY, so a plain latest-row rule resolved Raúl Jiménez to "Mexico" and Rúben
    # Neves to "Portugal" and then dropped them from their clubs. (Same root cause as
    # the "Australia"/"Austria" teams that surfaced in player_tips.csv.) Players with
    # no club row at all fall back to their latest row of any kind.
    # A club row is a row of a registry CLUB competition (competition_type club_league / club_cup,
    # config/leagues.json). Was a name regex ("world cup|friendl|...|euro 20"). A competition the
    # registry does not know is not proven to be a club competition and is not used here.
    _ctype = {lg: _reg.competition_type(lg) for lg in history_df["league"].astype(str).unique()}
    _is_club_row = history_df["league"].astype(str).map(_ctype).isin(["club_league", "club_cup"])
    _club_hist = history_df[_is_club_row]
    _latest_team = (_club_hist.sort_values("date")
                    .drop_duplicates("player_id", keep="last")
                    .set_index("player_id")["team"].astype(str))
    _latest_any = (history_df.sort_values("date")
                   .drop_duplicates("player_id", keep="last")
                   .set_index("player_id")["team"].astype(str))
    _latest_team = _latest_team.reindex(_latest_any.index).fillna(_latest_any)

    # ── LIVE SQUAD OVERLAY — authoritative for CURRENT club ──────────────────────────────
    # This is the "/players/squads overlay" the note above asks for. A historical appearance
    # cannot know about a transfer completed AFTER it, so the latest-club rule is correct and
    # still wrong for anyone who moved in the window: Víctor Muñoz (id 338751) last appeared
    # for Osasuna on 2026-05-17, signed for Liverpool over the summer, and was therefore being
    # tipped as an Osasuna player against Osasuna fixtures he will not play in.
    #
    # SCALE, measured 2026-08-22: of 526 Premier League squad players who join to club history,
    # 131 (25%) have a live club that differs from their latest club appearance — Rashford shown
    # at Barcelona, Tielemans at Aston Villa, Aarons at Valencia, Andrey Santos at Chelsea. A
    # quarter of the squad, not an edge case.
    #
    # The join is on player_id, not name, so accents and abbreviations ("Víctor Muñoz" vs
    # "Victor Munoz") cannot break it — the exact failure mode that made club matching hard
    # everywhere else in this codebase. 534 of 674 squad ids (79%) are present in history.
    #
    # PRECEDENCE: live squad > latest club appearance > latest appearance of any kind > the row's
    # own team. Each step is strictly less current than the one before it.
    #
    # COVERAGE LIMIT, stated because it is not a bug: pl_squads_official.csv is PREMIER LEAGUE
    # only. A player who left the PL this summer is absent from it, so he still falls back to his
    # last PL appearance. No source we hold knows better, and inventing one would be worse than
    # being out of date. The overlay fires only where it genuinely knows.
    # COVERAGE IS NO LONGER PREMIER LEAGUE ONLY. squads_official.csv carries every league we tip
    # in (written daily by fantasy_refresh); pl_squads_official.csv is read AFTER it and only
    # fills ids the wide file does not have, so the overlay degrades to the old PL-only behaviour
    # if the wide file is missing rather than disappearing. Order matters — the wide file is the
    # more current of the two for a player who moved between leagues.
    from player_model.feature_engineering import team_scope as _team_scope
    _live_squad = {}
    _live_age: dict = {}          # real age from /players/squads (feature `age`; parquet has none)
    _squad_scopes: dict = {}      # club -> identity scope(s) of the league its squad list is for
    for _name in ("squads_official.csv", "pl_squads_official.csv"):
        try:
            _sqf = config.OUTPUT_DIR / _name
            if not _sqf.exists():
                continue
            _sq = pd.read_csv(_sqf)
            _sq = _sq.dropna(subset=["player_id", "team"])
            # Guarded here too, not only in the writer: this file arrives from CI and a zero id
            # would silently give every id-less history row the same club.
            _sq = _sq[pd.to_numeric(_sq["player_id"], errors="coerce") > 0]
            for _pid, _tm in zip(_sq["player_id"].astype("int64"), _sq["team"].astype(str)):
                _live_squad.setdefault(_pid, _tm)          # first file wins
            if "age" in _sq.columns:
                for _pid, _ag in zip(_sq["player_id"].astype("int64"),
                                     pd.to_numeric(_sq["age"], errors="coerce")):
                    if _ag == _ag and _ag > 0:
                        _live_age.setdefault(_pid, float(_ag))
            if "league" in _sq.columns:
                for _tm, _lg in zip(_sq["team"].astype(str), _sq["league"]):
                    _s = _team_scope(_lg)
                    if _s:
                        _squad_scopes.setdefault(_tm, set()).add(_s)
        except Exception as _e:
            print(f"[squad_overlay] {_name} skipped ({_e})")
    if not _live_squad:
        print("[squad_overlay] no squad file readable; falling back to appearance history")

    _from_live = history_df["player_id"].map(_live_squad)
    _from_hist = history_df["player_id"].map(_latest_team)
    _club = _from_live.fillna(_from_hist).fillna(history_df["team"].astype(str))
    if _live_squad:
        _corrected = int((_from_live.notna() & _from_hist.notna()
                          & (_from_live != _from_hist)).sum())
        _n_players = history_df.loc[_from_live.notna(), "player_id"].nunique()
        print(f"[squad_overlay] {len(_live_squad)} live squad entries; current club taken "
              f"from the live squad for {_n_players} player(s); {_corrected} history row(s) "
              f"corrected where the latest appearance was a former club")
    # Unique club names once, so club resolution below costs O(teams) per fixture instead of
    # O(rows) — history_df is ~800k rows but only a few hundred clubs.
    _uniq_teams = _club.unique()
    # team -> competitions we actually hold history for. This SCOPES club resolution: a fixture's
    # clubs are resolved only among clubs that played in the same identity scope (the league's
    # country for a club league, feature_engineering.team_scope), so "Inter" in a Conference
    # League tie cannot become Inter Milan and Lillestrom cannot become Lille.
    #
    # KEYED ON THE ROW'S OWN TEAM, NOT ON THE PLAYER'S CURRENT CLUB (fixed 2026-10-08). Keyed on
    # `_club`, a club inherited every competition its CURRENT players had ever played in for
    # OTHER clubs — one Sporting CP signing with Spanish second-division appearances made
    # "Sporting CP" look like a La Liga 2 club, so the subset rule accepted it as "Sporting Gijón"
    # and tipped Luis Suárez, Maxi Araújo, Issa Doumbia and Irankunda in Cádiz v Sporting Gijón.
    # A club's competitions are the ones IT played in.
    _team_leagues = (history_df.assign(_t=history_df["team"].astype(str))
                     .groupby("_t")["league"].agg(set).to_dict())
    _team_scopes = {t: ({_team_scope(lg) for lg in lgs} | _squad_scopes.get(t, set())) - {None}
                    for t, lgs in _team_leagues.items()}
    for _t, _s in _squad_scopes.items():
        _team_scopes.setdefault(_t, set()).update(_s)
    _cands_by_scope: dict = {}

    def _candidates(lg) -> list:
        """Current clubs in the fixture league's identity scope. A club cup (UCL/UEL/UECL) draws
        from every club-league scope; resolve() still refuses an ambiguous name."""
        sc = _team_scope(lg)
        if sc not in _cands_by_scope:
            if sc == "club_cup":
                ok = lambda ss: any(str(s).startswith("club_league:") for s in ss)
            else:
                ok = lambda ss: sc is not None and sc in ss
            _cands_by_scope[sc] = [t for t in _uniq_teams if ok(_team_scopes.get(t, ()))]
        return _cands_by_scope[sc]
    _n_unresolved = 0
    _n_quality_cap_skipped = 0

    # ── STALE-CLUB GUARD (2026-10-08) ─────────────────────────────────────────────────────────
    # The latest-club rule can only see clubs we collect. A player who left for a league we do
    # not collect keeps his last collected club forever: on 2026-10-08 the board tipped Benzema
    # for Real Madrid (last appearance 2023), Verratti for PSG, Brozović for Inter, Wijnaldum for
    # Roma — 357 rows, 14% of the board, all for players in NO current squad list and unseen for
    # 120+ days. A player is eligible only if
    #   (a) he is in a live squad list (then `_club` already IS that squad's club), or
    #   (b) his club publishes no squad list we hold AND he appeared within STALE_DAYS.
    # A club whose live squad we DO hold and that does not list him has told us he is not there.
    _squad_clubs = set(_live_squad.values())
    _last_seen = (history_df.assign(_d=pd.to_datetime(history_df["date"], errors="coerce"))
                  .groupby("player_id")["_d"].max())
    _cutoff = pd.Timestamp.now() - pd.Timedelta(days=config.STALE_DAYS)

    def _eligible(pid, club) -> bool:
        try:
            if int(pid) in _live_squad:
                return True
        except (TypeError, ValueError):
            return False
        if str(club) in _squad_clubs:
            return False
        seen = _last_seen.get(pid)
        return seen is not None and pd.notna(seen) and seen >= _cutoff
    _stale_dropped = set()

    for _, match_row in bets.iterrows():
        home      = match_row["home_team"]
        away      = match_row["away_team"]
        league    = match_row.get("league", "")
        date_str  = str(match_row["date"])[:10]
        match_key = f"{home}|{away}|{date_str}"

        # Referee profile by API-Football fixture id (a bets.csv fixture has none -> no profile).
        _fid = match_row.get("fixture_id")
        ref = ((referee_profiles or {}).get(int(_fid), {})
               if _fid is not None and not pd.isna(_fid) else {})
        ctx = (match_contexts or {}).get(match_key, {})

        # Team win probabilities for opponent-weakness multiplier. Absent -> NaN and the
        # multiplier is skipped for this fixture (was a typed 0.5 / 0.5).
        def _p(k):
            try:
                return float(ctx.get(k))
            except (TypeError, ValueError):
                return float("nan")
        home_win_prob, away_win_prob = _p("home_win_prob"), _p("away_win_prob")

        # ── Lineup check: confirmed starter IDS for this fixture ──────────────
        lineup_starters: set[int] = set()
        lineup_available = False
        fid = match_row.get("fixture_id")
        if _APIFOOTBALL_KEY and fid and not pd.isna(fid):
            try:
                from player_model.api_football import get_fixture_lineup as _gfl
                _lineup = _gfl(int(fid))
                if _lineup:
                    for _team_players in _lineup.values():
                        for _p in _team_players:
                            if _p.get("started") and _p.get("player_id") is not None:
                                lineup_starters.add(int(_p["player_id"]))
                    # Only treat lineup as confirmed when both teams have starters posted
                    # (empty startXI = lineup not yet submitted → don't drop any player)
                    if len(lineup_starters) >= 18:
                        lineup_available = True
            except Exception as _e:
                print(f"[predict] Lineup fetch failed (fixture {fid}): {_e}")

        # Find players from both teams. Each fixture club is RESOLVED (src.team_names.resolve,
        # ambiguity refused) among the current clubs of the fixture league's identity scope. The
        # old matching (lower-cased equality, then api_football._same_club / _club_name_subset
        # over every club ever collected) is gone; history lessons it encoded still hold:
        # a 5-char prefix put Lille in Lillestrom fixtures and Messi in Sarmiento v Argentinos.
        _cands = _candidates(league)
        hit_home = _resolve(str(home), _cands) if _cands else None
        hit_away = _resolve(str(away), _cands) if _cands else None
        if hit_home and hit_home == hit_away:
            hit_home = hit_away = None                      # both sides on one club: refuse
        _sides = {c for c in (hit_home, hit_away) if c}
        if not _sides:
            _n_unresolved += 1
            continue
        team_players = history_df[_club.isin(_sides)].copy()
        if team_players.empty:
            continue

        # Carry the CURRENT club, so a tip names the club the player actually plays
        # for and is_home/opponent below are decided on it rather than on whichever
        # club happened to be on the historical row.
        team_players["team"] = _club.loc[team_players.index]
        # One record per player. build_upcoming_features looks history up by
        # player_id, so every additional historical row was duplicated feature work
        # that the end-of-run drop_duplicates discarded anyway.
        team_players = (team_players.sort_values("date")
                        .drop_duplicates("player_id", keep="last"))
        _keep = [_eligible(p, c) for p, c in zip(team_players["player_id"], team_players["team"])]
        _stale_dropped.update(team_players.loc[[not k for k in _keep], "player_id"].tolist())
        team_players = team_players[_keep]
        if team_players.empty:
            continue

        # Match context: home side = the club resolved for the fixture's HOME name (same resolver,
        # so a player can never be labelled home on the strength of a shared name fragment).
        team_players["is_home"] = (team_players["team"].astype(str) == str(hit_home)).astype(float) \
            if hit_home else 0.0
        team_players["opponent"] = [away if t == hit_home else home
                                    for t in team_players["team"].astype(str)]
        # The upcoming fixture's league/date (history rows carry their OWN past league/date), and a
        # real age from the live squad list where we hold one.
        team_players["fixture_league"] = league
        team_players["fixture_date"] = date_str
        team_players["age"] = team_players["player_id"].map(
            lambda x: _live_age.get(int(x)) if x == x and x is not None else None)

        upcoming_list = team_players.to_dict("records")
        feat_df = build_upcoming_features(upcoming_list, history_df, ref, ctx, history_index=_hist_index)

        if feat_df.empty:
            continue

        # Predict all markets
        for market, payload in payloads.items():
            probs = predict_proba(feat_df, payload)
            feat_df[f"p_{market}"] = probs.values
            if _first_sight is not None:
                _ko = match_row.get("kickoff_utc", "")
                _first_sight.add(market, feat_df, probs.values, league=str(league),
                                 match_date=date_str,
                                 kickoff_utc=_ko if isinstance(_ko, str) else "",
                                 fixture_id=match_row.get("fixture_id"))
            if _r2_shadow is not None:
                _ko = match_row.get("kickoff_utc", "")
                _r2_shadow.add(market, feat_df, probs.values, league=str(league), match_date=date_str,
                               kickoff_utc=_ko if isinstance(_ko, str) else "",
                               fixture_id=match_row.get("fixture_id"))

        _is_nat = _reg.competition_type(league) == "national_teams"
        for i, feat_row in feat_df.iterrows():
            try:
                _pid = int(feat_row.get("player_id"))
            except (TypeError, ValueError):
                _pid = None

            # Skip players not in confirmed starting XI when lineup is confirmed (by player id)
            if lineup_available and _pid not in lineup_starters:
                continue

            # Starter proxy: without a confirmed lineup, only tip players who average real
            # starter minutes — avoids tipping fringe/rotation players who may not start.
            if not lineup_available:
                _mpg = feat_row.get("minutes_pg", None)
                if _mpg is not None and not pd.isna(_mpg) and float(_mpg) < config.MIN_STARTER_MINUTES:
                    continue

            # Skip injured / suspended players (API-Football ids, api_football.get_injured_players)
            if _pid is not None and _pid in injured_cache.get(league, set()):
                continue

            _is_gk = str(feat_row.get("position", "")).strip().upper().startswith("G")
            # National-team competition by the registry (was: "world cup" in the league name).
            _is_wc = _is_nat

            for market in config.MARKETS:
                if market == "cards":
                    # The cards model is well-calibrated for CLUB leagues (mean ~0.13 vs the
                    # 12.8% base rate, AUC 0.69). World Cup features get imputed to league
                    # averages, inflating WC card predictions to ~0.40 for everyone. But ~0.40
                    # is roughly CORRECT for genuinely card-prone players, so for WC we apply a
                    # higher bar: gate on the player's REAL booking history (their true rate ≈
                    # the prediction) and drop the rest. Club cards pass through (model is fine).
                    if _is_wc:
                        _rates = pd.to_numeric(pd.Series([feat_row.get("cards_pg"),
                                                          feat_row.get("season_cards_pg")]),
                                               errors="coerce").dropna()
                        if _rates.empty or float(_rates.max()) < config.WC_CARD_MIN_RATE:
                            continue
                    # GKs allowed for cards — a bunker/time-wasting GK with real booking history
                    # is a legitimate card tip; the history gate already filters the rest.
                else:
                    # Goalkeepers don't score / shoot / assist — skip them for those markets.
                    if _is_gk:
                        continue
                if market not in _MARKET_CAPS:
                    continue                                   # no measurable cap -> no tip (logged)
                p_model = pd.to_numeric(feat_row.get(f"p_{market}"), errors="coerce")
                if not (p_model == p_model):
                    continue                                   # model gave no probability
                p_model = float(p_model)
                # Team-strength multiplier: boost for heavy favourites, penalty for underdogs.
                # Skipped explicitly when the fixture has no measured win probability.
                _is_home = float(feat_row.get("is_home")) > 0.5
                _win_prob = home_win_prob if _is_home else away_win_prob
                if _win_prob == _win_prob:
                    p_model = p_model * _team_strength_mult(_win_prob, market)

                # Hard quality-mismatch cap: player career tier << opponent tier
                # e.g. Panama striker (career_quality=0.30) vs England CBs (opp_quality=1.0)
                # context_quality_discount = 0.30 → apply extra cap so model can't output
                # inflated probabilities for weak-league players vs elite opposition.
                # Needs BOTH qualities measured; otherwise the cap is skipped (no 0.65 / 1.0 priors).
                _career_q = pd.to_numeric(feat_row.get("player_career_avg_quality"), errors="coerce")
                _opp_q = pd.to_numeric(feat_row.get("opp_def_player_quality"), errors="coerce")
                if _career_q == _career_q and _opp_q == _opp_q:
                    _ctx_disc = float(_career_q) / max(float(_opp_q), 0.1)
                    if _ctx_disc < 0.60 and market in ("goals", "goals2", "sot", "sot2", "sot3"):
                        # Discount scales: ctx_disc=0.50 → ×0.50, ctx_disc=0.30 → ×0.30
                        p_model = p_model * _ctx_disc
                else:
                    _n_quality_cap_skipped += 1

                # Cap at realistic maximum then absolute ceiling of 0.95
                p_model = min(p_model, _MARKET_CAPS[market], 0.95)

                if p_model < config.MIN_SIGNAL_PROB:
                    continue

                n_games = int(feat_row.get("n_games", 0))
                if n_games < config.MIN_GAMES_SIGNAL:
                    continue

                # Fair odds (we don't have market odds yet — placeholder)
                fair_o = round(1 / p_model, 2)

                # GES for goals/SOT only (undefined elsewhere -> None). NaN when an input is
                # missing; _classify_tier then gives the tip AVOID.
                ges = compute_ges(
                    feat_row.to_dict(),
                    opp_weakness=feat_row.get("opp_goals_conceded_pg"),
                ) if market in ("goals", "sot") else None

                lazy_factors = _detect_lazy_factors(
                    feat_row.to_dict(), feat_row.to_dict(), market
                )
                # Confirmed starter adds a lazy factor (confidence boost)
                if lineup_available and _pid in lineup_starters and "STARTER" not in lazy_factors:
                    lazy_factors = ["STARTER"] + lazy_factors
                confidence = _confidence_score(feat_row.to_dict(), len(lazy_factors))
                # World Cup non-card props use imputed national-team features (live ROI ~ -33%)
                # → flag low-confidence and discount so they rank below club-league signals.
                if _is_wc and market != "cards":
                    confidence *= config.WC_PROP_CONF_PENALTY
                    lazy_factors = lazy_factors + ["WC_LOW_CONF"]

                all_tips.append({
                    "date":          date_str,
                    "kickoff_utc":   match_row.get("kickoff_utc", ""),
                    "league":        league,
                    "match":         f"{home} vs {away}",
                    "match_tier":    match_row.get("signal_tier", ""),
                    # Stable identifiers, carried so downstream files can join EXACTLY
                    # instead of on normalised names. player_history.parquet has both, but
                    # they were dropped here, so player_ledger / player_prop_odds_history /
                    # clv_records could only ever be reconciled by (name, date) — the same
                    # fuzzy-key weakness that put Real Madrid players in an MLS fixture.
                    # settle_results uses player_id first and falls back to the name.
                    "player_id":     feat_row.get("player_id"),
                    "fixture_id":    match_row.get("fixture_id"),
                    "player_name":   feat_row.get("player_name", ""),
                    "team":          feat_row.get("team", ""),
                    "position":      feat_row.get("position", ""),
                    "market":        market,
                    "model_prob":    round(p_model, 3),
                    "fair_odds":     fair_o,
                    "market_odds":   None,   # to be filled when odds available
                    "fair_implied":  None,
                    "edge_abs":      None,
                    "edge_rel":      None,
                    "ev":            None,
                    "ges":           ges if market in ("goals", "sot") else None,
                    "confidence":    confidence,
                    "lazy_factors":  "|".join(lazy_factors),
                    "n_games":       n_games,
                    "data_source":   feat_row.get("data_source", ""),
                    "tier":          "AVOID",   # updated when market odds available
                    "kelly_stake":   None,
                })

    if _stale_dropped:
        print(f"[stale_club] {len(_stale_dropped)} player(s) dropped: not in their club's current "
              f"squad list, or unseen for {config.STALE_DAYS}+ days where no list exists")
    if _n_unresolved:
        print(f"[predict] {_n_unresolved} fixture(s) skipped: neither club resolved among the "
              f"clubs we hold for that league's scope")
    if _n_quality_cap_skipped:
        print(f"[predict] quality-mismatch cap skipped on {_n_quality_cap_skipped} player-market(s): "
              f"league/opponent quality not measured")
    if _first_sight is not None:
        _first_sight.flush()
    if _r2_shadow is not None:
        _n_r2 = _r2_shadow.flush()
        print(f"[props_r2_shadow] +{_n_r2} first-sight row(s) -> output/props_r2_shadow.csv")
    if not all_tips:
        return pd.DataFrame()

    tips_df = pd.DataFrame(all_tips)
    tips_df = tips_df.sort_values("model_prob", ascending=False)
    # One tip per (fixture, player, market) on IDS (was the normalised player NAME, which merged
    # two different players sharing a short name). A bets.csv fixture has no fixture_id; its
    # `match` string (the fixture's own names) stands in for it.
    tips_df["_fx"] = tips_df["fixture_id"].where(tips_df["fixture_id"].notna(), tips_df["match"]).astype(str)
    tips_df = tips_df.drop_duplicates(subset=["_fx", "player_id", "market"])
    tips_df = tips_df.drop(columns=["_fx"])

    output = config.OUTPUT_DIR / "player_tips.csv"
    tips_df.to_csv(output, index=False)
    print(f"[player_model] {len(tips_df)} player prop rows -> {output}")
    return tips_df


def enrich_with_odds(tips_df: pd.DataFrame, odds_data: dict) -> pd.DataFrame:
    """
    Enrich player tips with market odds, compute EV, edge, tier.
    odds_data: {"{player_name}|{market}": market_odds}
    Called separately when odds are available.
    """
    if tips_df.empty:
        return tips_df

    # Fixture-scoped key ONLY (odds_fetcher.tip_price_key): two tips sharing a name in different
    # fixtures must not share a price. The plain "player|market" fallback is gone (Data Truth).
    _date_col = "date" if "date" in tips_df.columns else "match_date"
    for idx, row in tips_df.iterrows():
        key        = f"{row['player_name']}|{row['market']}"
        scoped     = f"{key}|{str(row.get(_date_col, '') or '')[:10]}|{row.get('match', '')}"
        mkt_odds   = odds_data.get(scoped)
        if not mkt_odds:
            continue

        p_model    = float(row["model_prob"])
        fair_prob  = _devig_fair_prob(mkt_odds)
        ev_val     = _ev(p_model, mkt_odds)
        rel_edge   = _relative_edge(p_model, fair_prob)
        confidence = float(row["confidence"])
        lazy_count = len([f for f in str(row.get("lazy_factors", "")).split("|")
                          if f and f != "WC_LOW_CONF"])   # WC_LOW_CONF is a PENALTY, not a positive factor
        # GES exists only for goals/SOT; elsewhere it is not read. Missing -> NaN -> AVOID there.
        ges        = pd.to_numeric(row.get("ges"), errors="coerce")
        ges        = float(ges) if ges is not None else float("nan")
        market     = row["market"]

        tier = _classify_tier(ev_val, mkt_odds, rel_edge, confidence, lazy_count, ges, market, row.get("position", ""))

        tips_df.at[idx, "market_odds"]  = mkt_odds
        tips_df.at[idx, "fair_implied"] = round(fair_prob, 4)
        tips_df.at[idx, "edge_abs"]     = round(p_model - fair_prob, 4)
        tips_df.at[idx, "edge_rel"]     = rel_edge
        tips_df.at[idx, "ev"]           = ev_val
        tips_df.at[idx, "tier"]         = tier
        tips_df.at[idx, "kelly_stake"]  = _kelly_stake(ev_val, mkt_odds) if tier != "AVOID" else 0.0

    return tips_df


def tag_paper_feed(tips_df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """PAPER (tracking-only) feed: after real tiers are set, promote the top-N remaining
    AVOID picks (that HAVE odds) by EV to tier='PAPER'. These are the model's strongest
    picks on the efficient props markets — NO proven edge (validation 2026-07-09: props
    have no bettable edge), so they are PAPER ONLY, never real money. Sending + logging
    them builds a live track record to check whether live diverges from backtest.
    config.PROPS_PAPER_FEED_N (default 5) controls how many per run; 0 disables.
    """
    if tips_df.empty:
        return tips_df
    n = int(getattr(config, "PROPS_PAPER_FEED_N", 5)) if top_n is None else int(top_n)
    if n <= 0 or "tier" not in tips_df.columns:
        return tips_df
    cand = tips_df[(tips_df["tier"] == "AVOID") & tips_df["market_odds"].notna()].copy()
    if cand.empty:
        return tips_df
    cand["_ev"] = pd.to_numeric(cand["ev"], errors="coerce")
    cand = cand[cand["_ev"].notna()]          # a PAPER pick needs a real EV - never a sentinel rank
    if cand.empty:
        return tips_df
    top_idx = cand.sort_values("_ev", ascending=False).head(n).index
    tips_df.loc[top_idx, "tier"] = "PAPER"
    print(f"[player_model] PAPER feed: promoted {len(top_idx)} top picks (tracking-only, no edge)")
    return tips_df


# Markets with no live Odds API coverage.
_NO_ODDS_MARKETS = {"assists"}


def enrich_no_odds_markets(tips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Markets with no live Odds API coverage (assists).

    DATA TRUTH (2026-11-01): this used to INVENT a bookmaker price from the calibration base rate
    (1 / (base_rate x 1.10), "assumes 10% overround") and tier the tip on it, so an assists tip
    could reach VALUABLE/MARKSMAN/SNIPER against a price no book ever offered. Now a tip without a
    real quote keeps market_odds / ev / edge unset and tier AVOID. Real assist quotes reach tips
    through odds_fetcher (OddsAPI player_assists) and player_model/prop_odds_af (API-Football).
    The count of unpriced rows is logged.
    """
    if tips_df.empty or "market" not in tips_df.columns:
        return tips_df
    unpriced = tips_df["market"].isin(_NO_ODDS_MARKETS) & tips_df["market_odds"].isna()
    if unpriced.any():
        print(f"[player_model] {int(unpriced.sum())} {sorted(_NO_ODDS_MARKETS)} tip(s) without a real "
              f"bookmaker quote stay AVOID (no synthetic price)")
    return tips_df
