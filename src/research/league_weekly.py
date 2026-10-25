"""WEEKLY PER-LEAGUE RESEARCH PANEL (owner principle 2026-10-25). RESEARCH ONLY.

    python -m src.research.league_weekly            # sync candidates, freeze locked weeks, write reports
    python -m src.research.league_weekly settle     # grade frozen rows (update_results.yml), write reports
    python -m src.research.league_weekly report     # reports only

"Every league generates evidence every week. Not every league generates bets every week. Do not break
a working production strategy." Every tracked league x market gets a ranked TOP-5 of its week's
fixtures, frozen before kickoff and graded afterwards, whether or not production tipped anything.

WHAT THIS NEVER DOES. It never writes a policy, threshold, stake, eligibility, regime, ledger, stake plan
or Telegram file, and nothing in production reads its outputs. Rows are NOT bets:
selection_mode = WEEKLY_LEAGUE_RESEARCH. `actually_bet` / `stake` only REPORT what the live stake plan
and ledgers did (tests/test_league_weekly.py checks the module source for writes to production files).

OUTPUTS (all with a recorded generated_at)
  output/league_weekly_candidates.csv      every eligible fixture x panel market, FIRST SIGHT, append-only
  output/league_weekly_research.csv        the canonical panel: top-5 per league x market x week,
                                           frozen columns append-only + grading columns filled once
  output/league_weekly_research_summary.json, output/league_market_health.json,
  output/weekly_league_report.json

SOURCE OF EVERY SCORE. output/edge_engine_shadow.csv (src/edge_engine_shadow.py, market_edge_v2): the
earliest pre-kickoff opinion per fixture x market, already frozen by the shared FirstSightStore. This
module never re-prices a fixture and never reads a post-kickoff quote.

PANEL MARKETS
  OU25    both sides: each fixture enters with the engine's better side (max ev_lower_bound, ties ->
          OVER, exactly market_edge.evaluate); the two sides are analysed separately.
  BTTS    side YES  (the registry side). OVER15: side OVER. OVER35: side OVER, OBSERVE-ONLY (blocked
          market; ranked and graded for evidence, never a production candidate).
  When the engine chose the other side (e.g. BTTS NO), the registry side's numbers are RECONSTRUCTED
  exactly from the logged row: the de-vigged market, the decision probability and every penalty except
  `long_odds` are symmetric between the two sides of a two-way market, so
      p_market_fair' = 1 - p_market_fair      p_decision' = 1 - p_decision
      uncertainty'   = uncertainty - long_odds(odds) + long_odds(odds')
  with odds' = the board price of that side (price_source = "row"). side_source = RECONSTRUCTED and
  decision_state = NO_BET ("the engine decides ONE side per market", as the stake plan join does).
  If the board price of that side is missing the row is INSUFFICIENT (not reconstructable).
  Every probability in a panel row is FOR THE ROW'S SIDE (p_model_raw included).

RANKING SCORE  RS-1 (documented, not a black box). Sort, first key first:
  1. price tier: a two-sided market price (market_source MEDIAN_TWO_SIDED_BOOKS or
     SINGLE_TWO_SIDED_BOOK) before INSUFFICIENT (no two-sided price, one-sided, stale, not
     reconstructable). An INSUFFICIENT row can be ranked but never certifies betting performance.
  2. ev_lower_bound, descending (market_edge_v2's EV at the uncertainty lower bound)
  3. ev_point, descending           4. p_lower_bound, descending
  5. n_books, descending            6. kickoff_utc, ascending           7. fixture_id, ascending
  NaN sorts last in every numeric key, so the order is total and deterministic.

GAMEWEEK  rule GW-1 (src/gameweek.league_week_id): '<league>|<ISO year>-W<ISO week>' of the TIP DAY.

WEEK MEMBERSHIP  rule WL-1 (fixtures arrive over several days):
  * Each fixture x market is FROZEN at first sight (the candidates log; its score never changes).
  * The league-week's LOCK is the earliest kickoff among that league-week's candidates.
  * The week's top-5 per market = RS-1 ranking of the candidates FIRST SEEN STRICTLY BEFORE THE LOCK.
    It is written once, at the first run at/after the lock, and never again. Because every input
    was observed before the first match of the week kicked off, no result (and no post-kickoff
    price) of any fixture of that week can influence membership or order.
  * A candidate first seen at/after the lock (a late-listed fixture) is kept in the candidates log,
    counted in health as `late_after_lock`, and is never a member.
  * Before the lock the week's ranking is PROVISIONAL (shown in weekly_league_report.json, labelled,
    never written to the canonical panel).
  * legacy_rank = rank of the same pre-lock candidates by legacy raw edge (descending, NaN last), for
    comparison; legacy values are copied, never recomputed or overwritten.

SETTLEMENT (append-only grading columns, frozen columns hashed before/after; a row is graded once)
  score   from output/v91_shadow.csv's settled rows (same fixture_id, else same date + teams), i.e. the
          ledgers' results hierarchy, never a new network call.
  close   src/shadow_settlement.closing_price over the ledgers' odds archives + heartbeats, CERTIFIED
          only: strictly pre-kickoff and within CLOSE_MAX_MIN = 30 minutes. A stale close is NaN.
  clv_pct = (odds / closing_odds - 1) * 100. closing_market_fair_prob = power de-vig of the certified
          closes of BOTH sides; clv_prob = closing_market_fair_prob - p_market_fair.
  flat_pnl = +(odds - 1) / -1 on 1u, ONLY for rows with a two-sided price; INSUFFICIENT rows carry the
          result but no P/L.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import market_edge as ME
from src.gameweek import RULE_VERSION as GW_RULE, league_week_id
from src.research.first_sight import FirstSightStore

log = logging.getLogger(__name__)

PANEL_VERSION = "LWR-1"
RANKING_SCORE_VERSION = "RS-1"
WEEK_RULE = "WL-1"
SELECTION_MODE = "WEEKLY_LEAGUE_RESEARCH"
TOP_N = 5
#: settlement waits as long as the shadow settlement does (closes land up to ~3 h after kickoff)
SETTLE_DELAY_MIN = 240
CLOSE_MAX_MIN = 30.0

OUT = config.OUTPUT_DIR
ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_FILE = OUT / "league_weekly_candidates.csv"
PANEL_FILE = OUT / "league_weekly_research.csv"
SUMMARY_FILE = OUT / "league_weekly_research_summary.json"
HEALTH_FILE = OUT / "league_market_health.json"
REPORT_FILE = OUT / "weekly_league_report.json"
REGISTRY_FILE = ROOT / "config" / "track_registry.json"

#: panel market -> (engine family in edge_engine_shadow.csv, fixed side or None = engine's better side)
PANEL_MARKETS = {"OU25": ("OU25", None), "BTTS": ("BTTS", "YES"), "OVER15": ("OU15", "OVER"),
                 "OVER35": ("OU35", "OVER")}
FAMILY_TO_PANEL = {v[0]: k for k, v in PANEL_MARKETS.items()}
OBSERVE_ONLY = frozenset({"OVER35"})
#: engine family -> board price columns in predictions.csv (side A, side B)
BOARD_ODDS = {"OU25": ("odds_over25", "odds_under25"), "BTTS": ("odds_btts", "odds_btts_no"),
              "OU15": ("odds_over15", "odds_under15"), "OU35": ("odds_over35", "odds_under35")}
VERSION_COL = {"OU25": "model_version", "BTTS": "model_version_btts", "OU15": "model_version_over15",
               "OU35": "model_version_over35"}
CHAMP_SLOT = {"BTTS": "btts", "OU15": "over15", "OU35": "over35"}
TWO_SIDED = ("MEDIAN_TWO_SIDED_BOOKS", "SINGLE_TWO_SIDED_BOOK")
PRICED_STATES = ("TWO_SIDED_MULTI_BOOK", "TWO_SIDED_SINGLE_BOOK")
REAL_STATES = ("PROBATION", "ACTIVE_LOW", "ACTIVE", "ACTIVE_STRONG", "DEGRADED")
LEGACY_TIERS = ("SNIPER", "MARKSMAN", "VALUABLE")
SETTLE_MARKET = {"OU25": "ou25", "BTTS": "btts", "OVER15": "ou15", "OVER35": "ou35"}

CAND_KEY = ["selection_engine_version", "fixture_id", "market"]
CAND_COLS = [
    "selection_engine_version", "fixture_id", "market", "engine_market", "first_seen_at", "created_at",
    "league_week_id", "season", "league", "model_type", "match_date", "home_team", "away_team",
    "kickoff_utc", "side", "side_source", "prediction_engine_version",
    "p_model_raw", "p_market_fair", "p_decision", "p_lower_bound", "market_residual", "ev_point",
    "ev_lower_bound", "odds", "opposite_side_odds", "market_overround", "market_source", "books_used",
    "n_books", "price_timestamp", "minutes_to_kickoff", "uncertainty", "decision_state", "decision_reason",
    "price_quality_state", "legacy_engine_version", "legacy_edge", "legacy_tier", "legacy_side",
    "research_regime", "production_state", "production_selected", "actually_bet", "stake", "observe_only",
]

PANEL_KEY = ["selection_engine_version", "league_week_id", "market", "league_market_rank"]
PANEL_FROZEN = [
    "research_id", "selection_mode", "panel_version", "ranking_score_version", "week_rule", "gameweek_rule",
    "season", "league_week_id", "week_lock_ts", "n_week_candidates", "league", "model_type", "fixture_id",
    "match_date", "home_team", "away_team", "kickoff_utc", "market", "engine_market", "side", "side_source",
    "league_market_rank", "legacy_rank", "prediction_engine_version", "selection_engine_version",
    "p_model_raw", "p_market_fair", "p_decision", "p_lower_bound", "market_residual", "ev_point",
    "ev_lower_bound", "odds", "opposite_side_odds", "market_overround", "market_source", "books_used",
    "n_books", "price_timestamp", "minutes_to_kickoff", "uncertainty", "decision_state", "decision_reason",
    "price_quality_state", "legacy_engine_version", "legacy_edge", "legacy_tier", "legacy_side",
    "research_regime", "production_state", "production_selected", "actually_bet", "stake", "observe_only",
    "created_at", "first_seen_at",
]
PANEL_GRADING = [
    "home_goals", "away_goals", "result_source", "outcome", "result", "won", "flat_pnl", "closing_odds",
    "closing_opposite_odds", "closing_market_fair_prob", "clv_pct", "clv_prob", "close_status",
    "close_minutes_to_kickoff", "production_selected_final", "actually_bet_final", "stake_final",
    "settle_status", "settled_at",
]
PANEL_COLS = PANEL_FROZEN + PANEL_GRADING


# ── small helpers ─────────────────────────────────────────────────────────────────────────────
def _stamp(now) -> str:
    return pd.Timestamp(now).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _now(now=None) -> pd.Timestamp:
    t = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _r(v, nd=6):
    x = _f(v)
    return "" if math.isnan(x) else round(x, nd)


def _s(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s in ("nan", "NaN", "None", "<NA>", "NaT") else s


def _ts(s) -> pd.Series:
    return pd.to_datetime(pd.Series(s, dtype=object).astype(str).where(lambda x: x.str.strip() != "", None),
                          errors="coerce", utc=True, format="mixed")


def _read(path: Path, **kw) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False, **kw)
    except Exception as e:                                             # noqa: BLE001 - a recorder never raises
        log.warning(f"[league_weekly] {p.name} unreadable ({type(e).__name__}: {e})")
        return pd.DataFrame()


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return {}


def season_of(league: str, model_type: str, kickoff_utc) -> str:
    """Standard-format (European 2nd divisions): '2026/27', rolling over on 1 July. New-format leagues
    play calendar-year seasons: '2026'. Display/grouping only."""
    t = pd.to_datetime(kickoff_utc, errors="coerce", utc=True)
    if pd.isna(t):
        return ""
    if str(model_type) == "standard":
        y = t.year if t.month >= 7 else t.year - 1
        return f"{y}/{(y + 1) % 100:02d}"
    return str(t.year)


def research_id(engine: str, week: str, market: str, fixture_id: str, side: str) -> str:
    return hashlib.sha1(f"{engine}|{week}|{market}|{fixture_id}|{side}".encode("utf-8")).hexdigest()[:16]


def registry_entries(path: Path | None = None) -> dict:
    """{'league|market|side|model_type': entry} from config/track_registry.json (read-only)."""
    reg = _json(path or REGISTRY_FILE)
    return {e["key"]: e for e in reg.get("entries", []) if "key" in e}


def tracked_leagues(path: Path | None = None) -> list[str]:
    reg = _json(path or REGISTRY_FILE)
    return sorted({e["league"] for e in reg.get("entries", []) if e.get("league")})


# ── side view: the row's side and its numbers ───────────────────────────────────────────────────
def _long_odds(odds: float, P: dict) -> float:
    return float(P["long_odds_slope"]) * max(0.0, odds - float(P["long_odds_soft"]))


def side_view(r: dict, want: str | None, P: dict, board_odds: tuple = (float("nan"), float("nan"))) -> dict:
    """The panel side and its numbers for one edge_engine_shadow row (all strings, as stored).

    want=None -> the engine's better side (OU25). want='YES'/'OVER' -> that side, reconstructed when the
    engine chose the other one (see module docstring). Returns a dict of side fields incl.
    price_quality_state and side_source. Pure."""
    fam = str(r.get("market"))
    a, b = ME.SIDES[fam]
    eng = _s(r.get("side"))
    src = _s(r.get("market_source"))
    pm_a = _f(r.get("p_model_raw"))
    ev_lb = _f(r.get("ev_lower_bound"))
    priced = src in TWO_SIDED and not math.isnan(ev_lb) and eng in (a, b)
    pq = ("TWO_SIDED_MULTI_BOOK" if src == "MEDIAN_TWO_SIDED_BOOKS" else "TWO_SIDED_SINGLE_BOOK") if priced \
        else "INSUFFICIENT"
    side = want or (eng if eng in (a, b) else (a if (not math.isnan(pm_a) and pm_a >= 0.5) else b))
    i = 0 if side == a else 1
    pm = pm_a if i == 0 else (1.0 - pm_a if not math.isnan(pm_a) else float("nan"))
    nan = float("nan")
    out = {"side": side, "p_model_raw": pm, "board_odds": board_odds[i], "opp_board_odds": board_odds[1 - i],
           "decision_state": _s(r.get("decision_state")), "decision_reason": _s(r.get("decision_reason")),
           "price_quality_state": pq}
    if not priced:
        out.update(side_source=("ENGINE_SIDE" if eng == side else "MODEL_SIDE_NO_PRICE" if want is None
                                else "FIXED_SIDE_NO_PRICE"),
                   p_market_fair=nan, p_decision=nan, p_lower_bound=nan, market_residual=nan, ev_point=nan,
                   ev_lower_bound=nan, uncertainty=nan, odds=board_odds[i])
        return out
    if eng == side:
        out.update(side_source="ENGINE_SIDE", p_market_fair=_f(r.get("p_market_fair")),
                   p_decision=_f(r.get("p_decision")), p_lower_bound=_f(r.get("p_lower_bound")),
                   market_residual=_f(r.get("market_residual_raw")), ev_point=_f(r.get("ev_point")),
                   ev_lower_bound=ev_lb, uncertainty=_f(r.get("uncertainty_penalty")), odds=_f(r.get("odds")))
        return out
    # reconstruct the other side of a two-way market (price_source "row" only)
    o_row = _f(r.get("legacy_odds")) if _s(r.get("legacy_side")) == side else board_odds[i]
    if str(P.get("price_source", "row")) != "row" or not (o_row > 1.0):
        out.update(side_source="NOT_RECONSTRUCTABLE", price_quality_state="INSUFFICIENT",
                   p_market_fair=nan, p_decision=nan, p_lower_bound=nan, market_residual=nan, ev_point=nan,
                   ev_lower_bound=nan, uncertainty=nan, odds=o_row,
                   decision_state="NO_BET",
                   decision_reason=f"NOT_ENGINE_SIDE(engine side {eng}); other side not reconstructable")
        return out
    haircut = float(P.get("odds_haircut", 0.0) or 0.0)
    odds_e = 1.0 + (o_row - 1.0) * (1.0 - haircut)
    o_eng = _f(r.get("odds"))
    u = _f(r.get("uncertainty_penalty")) - _long_odds(o_eng, P) + _long_odds(odds_e, P)
    p_dec = 1.0 - _f(r.get("p_decision"))
    p_lb = min(max(p_dec - u, 0.0), 1.0)
    out.update(side_source="RECONSTRUCTED", p_market_fair=1.0 - _f(r.get("p_market_fair")), p_decision=p_dec,
               p_lower_bound=p_lb, market_residual=-_f(r.get("market_residual_raw")),
               ev_point=p_dec * odds_e - 1.0, ev_lower_bound=p_lb * odds_e - 1.0, uncertainty=u, odds=odds_e,
               decision_state="NO_BET",
               decision_reason=f"NOT_ENGINE_SIDE(engine side {eng}: {_s(r.get('decision_state'))})")
    return out


# ── candidates ──────────────────────────────────────────────────────────────────────────────────
def _board(preds: pd.DataFrame) -> dict:
    if preds is None or preds.empty:
        return {}
    out = {}
    for r in preds.to_dict("records"):
        out.setdefault((str(r.get("date", ""))[:10], str(r.get("home_team")), str(r.get("away_team"))), r)
    return out


def _stake_lookup(plan: pd.DataFrame) -> dict:
    """(fixture, market, side) -> (actually_bet, stake) from a stake plan / stake audit frame."""
    out: dict = {}
    if plan is None or plan.empty or "fixture" not in plan.columns:
        return out
    for r in plan.to_dict("records"):
        amt = _f(r.get("stake_amount"))
        real = str(r.get("deployment_state", "")) in REAL_STATES and amt > 0
        sides = [s.strip() for s in str(r.get("side", "")).split("/") if s.strip()]
        for s in sides:
            k = (str(r.get("fixture")), str(r.get("market")), s)
            prev = out.get(k, (False, 0.0))
            out[k] = (prev[0] or real, max(prev[1], amt if real else 0.0))
    return out


def build_candidates(shadow: pd.DataFrame, preds: pd.DataFrame | None, registry: dict,
                     plan: pd.DataFrame | None, champions: dict | None, params: dict,
                     now: pd.Timestamp) -> pd.DataFrame:
    """Every edge-shadow row (strictly pre-kickoff) -> one candidate per panel market. Pure."""
    if shadow is None or shadow.empty:
        return pd.DataFrame(columns=CAND_COLS)
    board, stakes, champions = _board(preds), _stake_lookup(plan), champions or {}
    created = _stamp(now)
    rows = []
    for r in shadow.to_dict("records"):
        fam = str(r.get("market"))
        pm_name = FAMILY_TO_PANEL.get(fam)
        if pm_name is None or str(r.get("selection_engine_version")) != ME.ENGINE_VERSION:
            continue
        fs, ko = pd.to_datetime(r.get("prediction_ts"), utc=True, errors="coerce"), \
            pd.to_datetime(r.get("kickoff_utc"), utc=True, errors="coerce")
        if pd.isna(fs) or pd.isna(ko) or fs >= ko:
            continue                                         # never a post-kickoff observation
        md, home, away = str(r.get("match_date"))[:10], str(r.get("home_team")), str(r.get("away_team"))
        b = board.get((md, home, away), {})
        ca, cb = BOARD_ODDS[fam]
        sv = side_view(r, PANEL_MARKETS[pm_name][1], params, (_f(b.get(ca)), _f(b.get(cb))))
        side, league, mt = sv["side"], str(r.get("league")), str(r.get("model_type"))
        e = registry.get(f"{league}|{pm_name}|{side}|{mt}", {})
        ver = _s(b.get(VERSION_COL[fam]))
        if not ver:
            slot = CHAMP_SLOT.get(fam) or ("standard" if mt == "standard" else "newformat")
            ver = _s((champions.get(slot) or {}).get("model_version"))
        sel = _s(r.get("legacy_tier")) in LEGACY_TIERS and _s(r.get("legacy_side")) == side
        bet, stake = stakes.get((f"{md}|{home}|{away}", pm_name, side), (False, 0.0))
        rows.append({
            "selection_engine_version": ME.ENGINE_VERSION, "fixture_id": str(r.get("fixture_id")),
            "market": pm_name, "engine_market": fam, "first_seen_at": _stamp(fs), "created_at": created,
            "league_week_id": league_week_id(league, ko), "season": season_of(league, mt, ko),
            "league": league, "model_type": mt, "match_date": md, "home_team": home, "away_team": away,
            "kickoff_utc": _stamp(ko), "side": side, "side_source": sv["side_source"],
            "prediction_engine_version": ver,
            "p_model_raw": _r(sv["p_model_raw"]), "p_market_fair": _r(sv["p_market_fair"]),
            "p_decision": _r(sv["p_decision"]), "p_lower_bound": _r(sv["p_lower_bound"]),
            "market_residual": _r(sv["market_residual"]), "ev_point": _r(sv["ev_point"]),
            "ev_lower_bound": _r(sv["ev_lower_bound"]), "odds": _r(sv["odds"], 4),
            "opposite_side_odds": _r(sv["opp_board_odds"], 4),
            "market_overround": _r(r.get("market_overround")), "market_source": _s(r.get("market_source")),
            "books_used": _s(r.get("books_used")), "n_books": _s(r.get("n_books")),
            "price_timestamp": _s(r.get("price_timestamp")), "minutes_to_kickoff": _r(r.get("minutes_to_kickoff"), 1),
            "uncertainty": _r(sv["uncertainty"]), "decision_state": sv["decision_state"],
            "decision_reason": sv["decision_reason"], "price_quality_state": sv["price_quality_state"],
            "legacy_engine_version": _s(r.get("legacy_engine_version")),
            "legacy_edge": _r(r.get("legacy_raw_edge")), "legacy_tier": _s(r.get("legacy_tier")),
            "legacy_side": _s(r.get("legacy_side")),
            "research_regime": e.get("research_regime", "UNREGISTERED"),
            "production_state": e.get("production_state", "UNREGISTERED"),
            "production_selected": bool(sel), "actually_bet": bool(bet), "stake": round(float(stake), 4),
            "observe_only": pm_name in OBSERVE_ONLY,
        })
    if not rows:
        return pd.DataFrame(columns=CAND_COLS)
    d = pd.DataFrame(rows).reindex(columns=CAND_COLS)
    # one candidate per real match: two fixture ids for the same match keep the earliest first sight
    d = d.sort_values(["first_seen_at", "fixture_id"], kind="mergesort")
    return d.drop_duplicates(["market", "match_date", "home_team", "away_team"], keep="first")


def candidate_store(path: Path | None = None) -> FirstSightStore:
    return FirstSightStore(Path(path) if path else CANDIDATES_FILE, key=CAND_KEY, columns=CAND_COLS,
                           observed_col="first_seen_at", kickoff_col="kickoff_utc")


def panel_store(path: Path | None = None) -> FirstSightStore:
    return FirstSightStore(Path(path) if path else PANEL_FILE, key=PANEL_KEY, columns=PANEL_COLS,
                           observed_col="first_seen_at", kickoff_col="kickoff_utc",
                           frozen_cols=PANEL_FROZEN, digest=True)


# ── ranking (RS-1) and week membership (WL-1) ───────────────────────────────────────────────────
def rank_frame(c: pd.DataFrame) -> pd.DataFrame:
    """Candidates of ONE league-week x market, sorted by RS-1, with `league_market_rank` 1..n."""
    if c.empty:
        return c.assign(league_market_rank=pd.Series(dtype=int), legacy_rank=pd.Series(dtype=int))
    recs = c[["price_quality_state", "ev_lower_bound", "ev_point", "p_lower_bound", "n_books", "kickoff_utc",
              "fixture_id", "legacy_edge"]].to_dict("records")
    order = sorted(range(len(recs)), key=lambda i: rs1_key(recs[i]))
    d = c.iloc[order].copy()
    d["league_market_rank"] = range(1, len(d) + 1)
    lorder = sorted(range(len(recs)), key=lambda i: (_desc(recs[i]["legacy_edge"]), str(recs[i]["kickoff_utc"]),
                                                     str(recs[i]["fixture_id"])))
    lrank = {pos: k + 1 for k, pos in enumerate(lorder)}
    d["legacy_rank"] = [lrank[i] for i in order]
    return d


def _desc(v) -> tuple:
    """Descending numeric sort key with NaN last."""
    x = _f(v)
    return (1, 0.0) if math.isnan(x) else (0, -x)


def rs1_key(r: dict) -> tuple:
    """RS-1 sort key (see module docstring). Timestamps are stored normalised ('%Y-%m-%dT%H:%M:%SZ'),
    so their string order is their time order."""
    return (0 if r.get("price_quality_state") in PRICED_STATES else 1, _desc(r.get("ev_lower_bound")),
            _desc(r.get("ev_point")), _desc(r.get("p_lower_bound")), _desc(r.get("n_books")),
            str(r.get("kickoff_utc")), str(r.get("fixture_id")))


def week_locks(cands: pd.DataFrame) -> dict:
    """{league_week_id: lock Timestamp} = earliest kickoff among the league-week's candidates."""
    if cands is None or cands.empty:
        return {}
    ko = _ts(cands["kickoff_utc"])
    return {w: t for w, t in ko.groupby(cands["league_week_id"].to_numpy()).min().items() if w and pd.notna(t)}


def week_ranking(cands: pd.DataFrame, week: str, market: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """RS-1 ranking of the week's candidates first seen strictly before `cutoff` (the lock, or now)."""
    return _rank_before(cands[(cands["league_week_id"] == week) & (cands["market"] == market)], cutoff)


def _rank_before(g: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return rank_frame(g[(g["first_seen_at"].astype(str) < _stamp(cutoff)).to_numpy()])


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Object-dtype copy: per-group work on arrow-backed string columns is ~10x slower."""
    return df.astype(object) if df is not None and len(df) else df


def freeze_rows(cands: pd.DataFrame, panel: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Panel rows for every (league-week, market) whose lock has passed and that is not frozen yet."""
    if cands is None or cands.empty:
        return pd.DataFrame(columns=PANEL_COLS)
    done = set()
    if panel is not None and not panel.empty:
        done = set(zip(panel["selection_engine_version"], panel["league_week_id"], panel["market"]))
    cands = _plain(cands)
    locks = week_locks(cands)
    created, out = _stamp(now), []
    for (week, market), g in cands.groupby(["league_week_id", "market"], sort=True):
        lock = locks.get(week)
        if lock is None or now < lock or (ME.ENGINE_VERSION, week, market) in done:
            continue
        ranked = _rank_before(g, lock)
        n = len(ranked)
        top = ranked[ranked["league_market_rank"] <= TOP_N]
        for r in top.to_dict("records"):
            out.append({**{k: r.get(k, "") for k in CAND_COLS},
                        "research_id": research_id(ME.ENGINE_VERSION, week, market, r["fixture_id"], r["side"]),
                        "selection_mode": SELECTION_MODE, "panel_version": PANEL_VERSION,
                        "ranking_score_version": RANKING_SCORE_VERSION, "week_rule": WEEK_RULE,
                        "gameweek_rule": GW_RULE, "week_lock_ts": _stamp(lock), "n_week_candidates": n,
                        "league_market_rank": int(r["league_market_rank"]), "legacy_rank": int(r["legacy_rank"]),
                        "created_at": created})
    if not out:
        return pd.DataFrame(columns=PANEL_COLS)
    return pd.DataFrame(out).reindex(columns=PANEL_COLS)


def provisional(cands: pd.DataFrame, panel: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Top-5 of every league-week x market whose lock has NOT passed yet (display only, never stored)."""
    if cands is None or cands.empty:
        return pd.DataFrame()
    cands = _plain(cands)
    locks, parts = week_locks(cands), []
    for (week, market), g in cands.groupby(["league_week_id", "market"], sort=True):
        lock = locks.get(week)
        if lock is None or now >= lock:
            continue
        r = _rank_before(g, now + pd.Timedelta(seconds=1))
        parts.append(r[r["league_market_rank"] <= TOP_N].assign(week_lock_ts=_stamp(lock),
                                                                  n_week_candidates=len(r)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── settlement ──────────────────────────────────────────────────────────────────────────────────
def _scores(v91: pd.DataFrame) -> tuple[dict, dict]:
    """(by fixture_id, by (date, home, away)) -> (home_goals, away_goals, result_source) from settled rows."""
    by_id, by_key = {}, {}
    if v91 is None or v91.empty or "home_goals" not in v91.columns:
        return by_id, by_key
    s = v91[(v91["home_goals"].astype(str).str.strip() != "") & (v91["away_goals"].astype(str).str.strip() != "")]
    for r in s.to_dict("records"):
        try:
            g = (int(float(r["home_goals"])), int(float(r["away_goals"])), str(r.get("result_source", "")))
        except (TypeError, ValueError):
            continue
        by_id.setdefault(str(r.get("fixture_id", "")).strip(), g)
        by_key.setdefault((str(r.get("match_date", ""))[:10], str(r.get("home_team", "")),
                           str(r.get("away_team", ""))), g)
    return by_id, by_key


def _ledger_selected(bets: pd.DataFrame | None, side_bets: pd.DataFrame | None) -> set:
    """{(date, home, away, market, side)} tipped by production (live ledgers)."""
    out = set()
    if bets is not None and not bets.empty:
        b = bets[bets.get("source", pd.Series("live", index=bets.index)).astype(str) == "live"]
        for r in b.to_dict("records"):
            out.add((str(r.get("match_date"))[:10], str(r.get("home_team")), str(r.get("away_team")), "OU25",
                     str(r.get("side")).upper()))
    if side_bets is not None and not side_bets.empty:
        m = {"btts": ("BTTS", "YES"), "over15": ("OVER15", "OVER"), "over35": ("OVER35", "OVER")}
        b = side_bets[side_bets.get("source", pd.Series("live", index=side_bets.index)).astype(str) == "live"]
        for r in b.to_dict("records"):
            mk = m.get(str(r.get("market", "")).lower())
            if mk:
                side = str(r.get("side", "") or "").strip().upper() or mk[1]
                out.add((str(r.get("match_date"))[:10], str(r.get("home_team")), str(r.get("away_team")), mk[0], side))
    return out


def settle_panel(panel: pd.DataFrame, v91: pd.DataFrame | None, archives: list | None,
                 now: pd.Timestamp, bets: pd.DataFrame | None = None, side_bets: pd.DataFrame | None = None,
                 stake_audit: pd.DataFrame | None = None, fallback_matcher=None) -> tuple[pd.DataFrame, int, dict]:
    """Fill grading columns of finished, unsettled rows. Frozen columns are hashed before and after;
    a mismatch raises and nothing is returned. Returns (frame, n_settled, pending_reasons)."""
    from src.research.first_sight import frozen_digest
    from src.shadow_settlement import closing_price, grade, index_archives, MARKETS as SM, CLOSE_OK

    out = panel.copy().reindex(columns=list(dict.fromkeys(list(panel.columns) + PANEL_COLS))).fillna("")
    before = frozen_digest(out, PANEL_FROZEN)
    by_id, by_key = _scores(v91)
    indexed = index_archives(archives or [])
    selected = _ledger_selected(bets, side_bets)
    stakes = _stake_lookup(stake_audit)
    stamp, n, pending = _stamp(now), 0, {}
    for i in range(len(out)):
        r = out.iloc[i]
        if _s(r["settle_status"]):
            continue
        ko = pd.to_datetime(r["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(ko) or now < ko + pd.Timedelta(minutes=SETTLE_DELAY_MIN):
            pending["NOT_FINISHED"] = pending.get("NOT_FINISHED", 0) + 1
            continue
        key = (str(r["match_date"])[:10], str(r["home_team"]), str(r["away_team"]))
        g = by_id.get(str(r["fixture_id"]).strip()) or by_key.get(key)
        if g is None:
            pending["NO_RESULT_YET"] = pending.get("NO_RESULT_YET", 0) + 1
            continue
        hg, ag, src = g
        sm = SETTLE_MARKET[str(r["market"])]
        side = str(r["side"])
        outcome, result = grade(sm, side, hg, ag)
        sides = SM[sm][2]
        opp = [s for s in sides if s != side][0]
        cl = closing_price(indexed, r["home_team"], r["away_team"], r["match_date"], ko, sides[side],
                           league=str(r["league"]), fallback_matcher=fallback_matcher, max_min=CLOSE_MAX_MIN)
        co = closing_price(indexed, r["home_team"], r["away_team"], r["match_date"], ko, sides[opp],
                           league=str(r["league"]), fallback_matcher=fallback_matcher, max_min=CLOSE_MAX_MIN)

        def certified(c):
            return c["status"] == CLOSE_OK and np.isfinite(c["minutes"]) and c["minutes"] <= CLOSE_MAX_MIN

        c_side = cl["odds"] if certified(cl) else float("nan")
        c_opp = co["odds"] if certified(co) else float("nan")
        status = cl["status"] if (cl["status"] != CLOSE_OK or certified(cl)) else "STALE_CLOSE_GT_30MIN"
        odds = _f(r["odds"])
        priced = str(r["price_quality_state"]) in PRICED_STATES and odds > 1.0
        pnl = ((odds - 1.0) if result == "WIN" else -1.0) if (priced and result in ("WIN", "LOSS")) else float("nan")
        clv = (odds / c_side - 1.0) * 100.0 if (priced and c_side > 1.0) else float("nan")
        fair = float("nan")
        if c_side > 1.0 and c_opp > 1.0:
            # power de-vig is defined on (side A, side B); map back to the row's side
            a_first = side == list(sides)[0]
            pa = ME.power_devig(c_side, c_opp) if a_first else ME.power_devig(c_opp, c_side)
            if pa is not None:
                fair = pa if a_first else 1.0 - pa
        pmf = _f(r["p_market_fair"])
        fk = f"{key[0]}|{key[1]}|{key[2]}"
        bet, stake = stakes.get((fk, str(r["market"]), side), (False, 0.0))
        vals = {
            "home_goals": hg, "away_goals": ag, "result_source": src, "outcome": outcome or "",
            "result": result or "", "won": ("1" if result == "WIN" else "0" if result == "LOSS" else ""),
            "flat_pnl": _r(pnl), "closing_odds": _r(c_side, 4), "closing_opposite_odds": _r(c_opp, 4),
            "closing_market_fair_prob": _r(fair), "clv_pct": _r(clv, 3),
            "clv_prob": _r(fair - pmf) if not (math.isnan(fair) or math.isnan(pmf)) else "",
            "close_status": status, "close_minutes_to_kickoff": _r(cl["minutes"], 1),
            "production_selected_final": (*key, str(r["market"]), side) in selected,
            "actually_bet_final": bool(bet), "stake_final": round(float(stake), 4),
            "settle_status": "SETTLED" if result else "SETTLED_NO_SIDE", "settled_at": stamp,
        }
        for k, v in vals.items():
            out.iat[i, out.columns.get_loc(k)] = "" if v == "" else str(v)
        n += 1
    if frozen_digest(out, PANEL_FROZEN) != before:
        raise RuntimeError("league_weekly settlement would alter a frozen column - refused")
    return out, n, pending


# ── reports ─────────────────────────────────────────────────────────────────────────────────────
def _fixtures_by_week(kickoffs: pd.DataFrame, since: pd.Timestamp | None) -> tuple[dict, dict]:
    """({league_week_id: n fixtures kicking off at/after `since`}, {league_week_id: n before it}) from
    tip_day.kickoffs() (predictions.csv + v91_shadow.csv). `since` = the panel's first observation: a
    fixture that kicked off before the panel existed was never expected to have a row."""
    if kickoffs is None or kickoffs.empty:
        return {}, {}
    k = kickoffs.drop_duplicates(["league", "home_team", "away_team", "match_date"])
    wk = {x: league_week_id("_", x).split("|", 1)[-1] for x in set(k["kickoff_utc"].astype(str))}  # GW-1 per kickoff
    w = pd.Series([f"{lg}|{wk[str(x)]}" if wk[str(x)] else "" for lg, x in zip(k["league"], k["kickoff_utc"])],
                  index=k.index)
    ko = _ts(k["kickoff_utc"])
    after = (ko >= since) if since is not None else pd.Series(False, index=k.index)
    return w[after].value_counts().to_dict(), w[~after].value_counts().to_dict()


def health(panel: pd.DataFrame, cands: pd.DataFrame, leagues: list[str], kickoffs: pd.DataFrame | None,
           now: pd.Timestamp, lock: dict | None = None) -> dict:
    """Per league x market coverage for the current and the previous gameweek + panel totals.

    A league with fixtures in a week but no candidate rows is MISSING_ROWS (a silent disappearance);
    a league with no fixtures that week is NO_FIXTURES (nothing expected)."""
    from src.tip_day import today as tip_today
    day = pd.Timestamp(tip_today(now))
    weeks = {"current": day, "previous": day - pd.Timedelta(days=7)}
    panel = panel if panel is not None and not panel.empty else pd.DataFrame(columns=PANEL_COLS)
    cands = cands if cands is not None and not cands.empty else pd.DataFrame(columns=CAND_COLS)
    start = _ts(cands["first_seen_at"]).min() if len(cands) else None
    start = None if start is None or pd.isna(start) else start
    fx, pre = _fixtures_by_week(kickoffs, start)
    locks = week_locks(cands)
    per, missing, expected, actual, with_data = {}, {"current": [], "previous": []}, 0, 0, 0
    for lg in leagues:
        rec = {}
        for label, d in weeks.items():
            iso = d.isocalendar()
            wk = f"{lg}|{int(iso[0])}-W{int(iso[1]):02d}"
            n_fx = int(fx.get(wk, 0))
            c = cands[cands["league_week_id"] == wk]
            p = panel[panel["league_week_id"] == wk]
            n_c = int(c["fixture_id"].nunique()) if len(c) else 0
            lk = locks.get(wk)
            mk = {m: {"candidates": int((c["market"] == m).sum()), "panel_rows": int((p["market"] == m).sum())}
                  for m in PANEL_MARKETS}
            gaps = sorted(m for m, v in mk.items() if v["candidates"] < n_c)
            n_pre = int(pre.get(wk, 0))
            if n_fx == 0 and n_c == 0:
                state = "BEFORE_PANEL_START" if n_pre else "NO_FIXTURES"
            elif n_c == 0:
                state = "MISSING_ROWS"
                missing[label].append(lg)
            elif lk is not None and now >= lk:
                state = "FROZEN" if len(p) else "LOCKED_NOT_FROZEN"
                if not len(p):
                    missing[label].append(lg)
            else:
                state = "PROVISIONAL"
            late = int((_ts(c["first_seen_at"]) >= lk).sum()) if (lk is not None and len(c)) else 0
            rec[label] = {"league_week_id": wk, "state": state, "fixtures_known": max(n_fx, n_c),
                          "fixtures_before_panel_start": n_pre, "fixtures_with_candidates": n_c,
                          "fixtures_without_candidates": max(n_fx - n_c, 0),
                          "week_lock_ts": _stamp(lk) if lk is not None else None,
                          "late_after_lock": late, "market_gaps": gaps, "markets": mk}
            if label == "current":
                exp_c = max(n_fx, n_c) * len(PANEL_MARKETS)
                expected += exp_c
                actual += int(len(c))
                with_data += int(n_c > 0)
                rec[label]["expected_candidate_rows"] = exp_c
                rec[label]["actual_candidate_rows"] = int(len(c))
        per[lg] = rec
    settled = panel["settle_status"].astype(str).isin(["SETTLED", "SETTLED_NO_SIDE"]) if len(panel) else pd.Series(dtype=bool)
    n_set = int(settled.sum()) if len(panel) else 0
    clv_n = int((pd.to_numeric(panel.loc[settled, "clv_pct"], errors="coerce").notna()).sum()) if n_set else 0
    ptm = None if lock is None else lock.get("protected_tracks_modified")
    return {
        "generated_at": _stamp(now), "panel_version": PANEL_VERSION,
        "tracked_leagues": len(leagues),
        "current_tip_day": day.strftime("%Y-%m-%d"),
        "leagues_with_current_week_data": with_data,
        "missing_leagues": sorted(set(missing["current"])),
        "missing_leagues_previous_week": sorted(set(missing["previous"])),
        "no_fixture_leagues": sorted(lg for lg, r in per.items() if r["current"]["state"] == "NO_FIXTURES"),
        "panel_start": _stamp(start) if start is not None else None,
        "market_gap_leagues": sorted(lg for lg, r in per.items() if r["current"]["market_gaps"]),
        "expected_candidate_rows": int(expected), "actual_candidate_rows": int(actual),
        "panel_rows": int(len(panel)), "settled_rows": n_set, "pending_rows": int(len(panel) - n_set),
        "clv_certified_rows": clv_n, "clv_coverage": round(clv_n / n_set, 4) if n_set else None,
        "protected_tracks_modified": ptm,
        "leagues": per,
    }


def _row_view(r: dict, frozen: bool) -> dict:
    keep = ("league_market_rank", "fixture_id", "home_team", "away_team", "kickoff_utc", "side", "decision_state",
            "price_quality_state", "ev_lower_bound", "ev_point", "p_model_raw", "p_market_fair", "odds",
            "legacy_tier", "legacy_rank", "production_state", "research_regime", "production_selected",
            "actually_bet", "result", "flat_pnl", "clv_pct")
    v = {k: r.get(k, "") for k in keep}
    v["status"] = "FROZEN" if frozen else "PROVISIONAL"
    return v


def weekly_report(panel: pd.DataFrame, prov: pd.DataFrame, leagues: list[str], now: pd.Timestamp,
                  eval_tables: dict | None = None) -> dict:
    """Per league: the latest gameweek's top-5 per market (frozen, else provisional) + previous weeks'
    Top1/Top3/Top5 ROI, mean CLV and market-relative log loss (from league_weekly_eval)."""
    from src.research import league_weekly_eval as EV
    panel = panel if panel is not None and not panel.empty else pd.DataFrame(columns=PANEL_COLS)
    prov = prov if prov is not None and not prov.empty else pd.DataFrame(columns=CAND_COLS + ["league_market_rank"])
    out = {}
    for lg in leagues:
        p, q = panel[panel["league"] == lg], prov[prov["league"] == lg] if len(prov) else prov
        weeks = sorted(set(p["league_week_id"]) | set(q["league_week_id"] if len(q) else []))
        latest = weeks[-1] if weeks else None
        mk = {}
        for m in PANEL_MARKETS:
            fr = p[(p["league_week_id"] == latest) & (p["market"] == m)]
            if len(fr):
                rows = [_row_view(r, True) for r in fr.sort_values("league_market_rank", key=lambda s: s.astype(int)).to_dict("records")]
            else:
                pr = q[(q["league_week_id"] == latest) & (q["market"] == m)] if len(q) else q
                rows = [_row_view(r, False) for r in pr.to_dict("records")] if len(pr) else []
            mk[m] = rows
        prev = EV.previous_weeks_summary(p, latest) if len(p) else {}
        out[lg] = {"latest_league_week_id": latest, "top5": mk, "previous_weeks": prev}
    return {"generated_at": _stamp(now), "panel_version": PANEL_VERSION, "selection_mode": SELECTION_MODE,
            "note": ("Research panel, not bets. FROZEN rows were fixed at the week lock (first kickoff of the "
                     "league-week) from first-sight scores; PROVISIONAL rows may still change until the lock."),
            "leagues": out}


def summary(panel: pd.DataFrame, cands: pd.DataFrame, hl: dict, now: pd.Timestamp) -> dict:
    from src.research import league_weekly_eval as EV
    panel = panel if panel is not None and not panel.empty else pd.DataFrame(columns=PANEL_COLS)
    first = (panel["created_at"].min() if len(panel) else None) or None
    return {
        "generated_at": _stamp(now), "panel_version": PANEL_VERSION, "selection_mode": SELECTION_MODE,
        "evidence_type": "TRUE_FORWARD", "true_forward_start": first,
        "methodology": {"ranking_score": RANKING_SCORE_VERSION, "week_rule": WEEK_RULE, "gameweek_rule": GW_RULE,
                        "top_n": TOP_N, "markets": {k: {"engine_family": v[0], "side": v[1] or "engine better side",
                                                        "observe_only": k in OBSERVE_ONLY}
                                                    for k, v in PANEL_MARKETS.items()},
                        "close_rule": f"certified close: strictly pre-kickoff and within {CLOSE_MAX_MIN:.0f} min",
                        "doc": "src/research/league_weekly.py module docstring"},
        "counts": {"candidates": int(len(cands)) if cands is not None else 0, "panel_rows": int(len(panel)),
                   "league_weeks_frozen": int(panel[["league_week_id", "market"]].drop_duplicates().shape[0])
                   if len(panel) else 0,
                   "settled_rows": hl.get("settled_rows"), "pending_rows": hl.get("pending_rows")},
        "health": {k: v for k, v in hl.items() if k != "leagues"},
        "evaluation": EV.evaluate(panel),
    }


def _write_json(path: Path, obj: dict) -> None:
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_reports(now=None, out_dir: Path | None = None, kickoffs: pd.DataFrame | None = None,
                  lock: dict | None = None, registry_path: Path | None = None) -> dict:
    now = _now(now)
    od = Path(out_dir) if out_dir else OUT
    panel = _plain(panel_store(od / PANEL_FILE.name).read())
    cands = _plain(candidate_store(od / CANDIDATES_FILE.name).read())
    leagues = tracked_leagues(registry_path)
    if kickoffs is None:
        try:
            from src.tip_day import kickoffs as _k
            kickoffs = _k()
        except Exception:                                              # noqa: BLE001
            kickoffs = pd.DataFrame()
    if lock is None:
        try:
            from src.research_lock import check
            lock = check()
        except Exception as e:                                         # noqa: BLE001
            lock = {"error": f"{type(e).__name__}: {e}"[:200]}
    hl = health(panel, cands, leagues, kickoffs, now, lock)
    prov = provisional(cands, panel, now)
    _write_json(od / HEALTH_FILE.name, hl)
    _write_json(od / REPORT_FILE.name, weekly_report(panel, prov, leagues, now))
    sm = summary(panel, cands, hl, now)
    _write_json(od / SUMMARY_FILE.name, sm)
    return {"health": hl, "summary": sm}


# ── entrypoints ─────────────────────────────────────────────────────────────────────────────────
def run(now=None, out_dir: Path | None = None, registry_path: Path | None = None,
        kickoffs: pd.DataFrame | None = None, lock: dict | None = None, reports: bool = True) -> dict:
    """Sync candidates from the edge shadow, freeze every locked league-week, write reports.
    Never raises (a recorder must not stop predict); returns counts."""
    now = _now(now)
    od = Path(out_dir) if out_dir else OUT
    res = {"candidates_appended": 0, "panel_rows_appended": 0}
    try:
        shadow = _read(od / "edge_engine_shadow.csv")
        preds = _read(od / "predictions.csv")
        plan = _read(od / "stake_plan.csv")
        champs = _json(ROOT / "models" / "registry" / "current_champions.json")
        cands = build_candidates(shadow, preds, registry_entries(registry_path), plan, champs, ME.load_params(), now)
        cs = candidate_store(od / CANDIDATES_FILE.name)
        res["candidates_appended"] = cs.append(cands)
        ps = panel_store(od / PANEL_FILE.name)
        new = freeze_rows(cs.read(), ps.read(), now)
        res["panel_rows_appended"] = ps.append(new)
        try:                                                  # contract check: observe only
            from src.contracts.shadow import observe
            observe("league_weekly_research.csv (new rows)", new, "LEAGUE_WEEKLY_RESEARCH_ROW")
        except Exception:                                              # noqa: BLE001
            pass
        if reports:
            write_reports(now, od, kickoffs, lock, registry_path)
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[league_weekly] run skipped ({type(e).__name__}: {e}) - production unaffected")
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    return res


def settle(now=None, out_dir: Path | None = None, archives: list | None = None, fallback_matcher=None,
           reports: bool = True, kickoffs: pd.DataFrame | None = None, lock: dict | None = None,
           registry_path: Path | None = None) -> dict:
    """Grade finished panel rows (grading columns only) and rewrite the reports."""
    now = _now(now)
    od = Path(out_dir) if out_dir else OUT
    p = od / PANEL_FILE.name
    ps = panel_store(p)
    panel = ps.read()
    n, pending = 0, {}
    if len(panel):
        v = ps.verify()
        if v["ok"] is False:
            raise RuntimeError(f"panel TAMPERED ({v['reason']}) - not settling")
        out, n, pending = settle_panel(panel, _read(od / "v91_shadow.csv"), archives, now,
                                       _read(od / "bets_ledger.csv"), _read(od / "side_bets_ledger.csv"),
                                       _read(od / "stake_audit.csv"), fallback_matcher)
        if n:
            tmp = p.with_suffix(".csv.tmp")
            out.reindex(columns=PANEL_COLS).to_csv(tmp, index=False, lineterminator="\n")
            tmp.replace(p)
    if reports:
        write_reports(now, od, kickoffs, lock, registry_path)
    return {"settled": n, "pending": pending}


def _cli(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cmd = argv[0] if argv else "run"
    if cmd == "run":
        r = run()
        print(json.dumps(r))
        return 0
    if cmd == "settle":
        sys.path.insert(0, str(ROOT))
        import update_results as U                       # the ledgers' archives + name matcher
        r = settle(archives=U._archives_with_heartbeats(), fallback_matcher=U._names_match)
        print(json.dumps(r))
        return 0
    if cmd == "report":
        r = write_reports()
        print(json.dumps({k: v for k, v in r["health"].items() if k != "leagues"}, default=str))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
