"""market_edge_v2 — market-first selection engine. SHADOW ONLY (2026-10-25).

    from src.market_edge import Candidate, BookQuote, evaluate, load_params

WHY. Production O/U 2.5 selection (src/betting.py::evaluate_value) tiers on
`edge = p_model - 1/odds`: the model's probability is taken as truth and the size of its
disagreement with one book's vigged price decides SNIPER / MARKSMAN / VALUABLE. Side markets
(pipeline._generate_side_bets) do the same against `(1/odds)/1.08`. Measured evidence says the
market price beats the model (memory: Brier 0.2368 vs 0.2583 on the model's own bets, weight on
v9 ~0 in the market-anchor study), so the size of a disagreement is not evidence of value.

src/edge_engine.py documents a market-first design but was never called, and its audit
(docs/EDGE_ENGINE_AUDIT_2026_10.md) found it unusable as written. This module is the rebuilt
version. It is NOT on the served path: pipeline.py calls it through src/edge_engine_shadow.py,
which only appends a first-sight log. Legacy stays the production/reference selection.

PIPELINE PER CANDIDATE (fixture x market family; both sides evaluated, the better EV_LB kept)

    1. fail-closed gates      unsupported market / unknown track / missing model / missing or
                              post-kickoff kickoff / synthetic price            -> NO_BET
    2. per-book validation    same book, both sides, odds > 1, quote before kickoff and before
                              now, overround inside [overround_min, overround_max], not older than
                              the stale limit. One side only -> counted as one-sided, never used.
    3. consensus hierarchy    validated exchange (only when exchange_validated; today: skipped)
                              > median of >=2 valid two-sided books > single two-sided book
                              > one-sided only -> OBSERVE / ONE_SIDED_PRICE (no fair prob invented)
                              > nothing -> NO_BET / NO_PRICE
    4. de-vig                 per book, BOTH proportional and Power; `devig_method` picks the one
                              used, the other is recorded so replay can choose on evidence
    5. decision prob          logit(p_decision) = logit(p_market_fair) + w_track_family *
                              clip(logit(p_model) - logit(p_market_fair), +-residual_logit_clip)
    6. lower bound            p_lower_bound = p_decision - uncertainty_penalty, where the penalty
                              is the sum of: book dispersion / few books, de-vig method gap, the
                              unproven part of the residual (shrinks with effective sample size),
                              thin league history, extreme disagreement, long odds, stale price,
                              recent retrain, weak/absent track CLV
    7. EV                     ev_point = p_decision*odds - 1 ; ev_lower_bound = p_lower_bound*odds - 1
    8. state                  NO_BET / OBSERVE / QUALIFIED / STRONG with explicit decision_reason.
                              QUALIFIED needs ev_lower_bound > 0, >= min_books_qualified books and a
                              track CLV record (n >= clv_min_n, clipped mean > 0 AND median > 0);
                              without one, OBSERVE
                              (NO_TRACK_EVIDENCE). STRONG additionally
                              needs PROSPECTIVE CLV evidence for this engine version and track
                              (n >= strong_min_n, CI lower bound > 0). No such evidence exists
                              today, so STRONG is unreachable by construction.

EXTREME DISAGREEMENT NEVER HELPS. The residual is clipped in logit space, so past the clip a larger
disagreement adds nothing to p_decision while the disagreement penalty keeps growing; past
`disagree_hard` the state is capped at OBSERVE. STRONG never depends on the residual at all.

The presentation map (STRONG->SNIPER, QUALIFIED->MARKSMAN, OBSERVE->VALUABLE) exists only in
`display_tier` for a dashboard. Nothing in this module emits a legacy tier name as a decision.
"""
from __future__ import annotations

import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ENGINE_VERSION = "market_edge_v2"
LEGACY_ENGINE_VERSION = "legacy_raw_edge"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "market_edge_v2.json"

#: market family -> (side A, side B). p_model is always P(side A).
SIDES = {"OU25": ("OVER", "UNDER"), "OU15": ("OVER", "UNDER"), "OU35": ("OVER", "UNDER"),
         "BTTS": ("YES", "NO")}
TRACKS = ("standard", "new_format")
STATES = ("NO_BET", "OBSERVE", "QUALIFIED", "STRONG")
STATE_RANK = {s: i for i, s in enumerate(STATES)}
_DISPLAY = {"STRONG": "SNIPER", "QUALIFIED": "MARKSMAN", "OBSERVE": "VALUABLE"}

#: The constants src/backtest.py fills a missing side-market price with (`_DEFAULT_ODDS`). A row
#: carrying one of these WITHOUT a `price_synthetic` flag cannot be told apart from a filled one.
SYNTHETIC_DEFAULT_ODDS = {"BTTS": 1.85, "OU15": 1.40, "OU35": 2.60}

#: Every output column, in order. Missing values are NaN, never an invented number.
OUTPUT_COLS = [
    "selection_engine_version", "market", "track", "decision_state", "decision_reason", "side",
    "p_model_raw", "p_market_fair", "p_market_fair_power", "p_market_fair_proportional",
    "devig_method", "p_decision", "p_lower_bound", "market_residual_raw",
    "market_residual_adjusted", "residual_weight", "odds", "odds_source", "best_book_odds",
    "market_overround", "ev_point", "ev_lower_bound", "effective_sample_size",
    "uncertainty_penalty", "penalty_breakdown", "clv_track_n", "clv_track_mean",
    "clv_track_median", "clv_track_positive_pct", "clv_evidence_state", "market_source",
    "books_used", "n_books", "n_one_sided_books", "n_invalid_books", "price_timestamp",
    "minutes_to_kickoff", "price_age_minutes",
]

DEFAULTS: dict = {
    "engine_version": ENGINE_VERSION, "status": "UNVALIDATED_DEFAULTS",
    "devig_method": "power", "price_source": "row", "exchange_validated": False,
    "odds_haircut": 0.0,
    "residual_weight": {f"{t}|{f}": 0.10 for t in TRACKS for f in SIDES},
    "residual_n_eff": {f"{t}|{f}": 0 for t in TRACKS for f in SIDES},
    "residual_logit_clip": 0.35, "residual_n0": 300,
    "overround_min": 0.0, "overround_max": 0.15,
    "book_sd_floor": 0.02, "min_books_qualified": 2, "z": 1.2816,
    "league_history_n0": 200, "league_penalty_max": 0.02,
    "disagree_soft": 0.08, "disagree_slope": 0.25, "disagree_hard": 0.20,
    "long_odds_soft": 2.5, "long_odds_slope": 0.02, "max_odds": 3.0,
    "stale_soft_minutes": 60, "stale_slope_per_hour": 0.002, "stale_hard_minutes": 720,
    "stale_hard_minutes_near": 180, "near_kickoff_minutes": 360,
    "recent_retrain_days": 7, "recent_retrain_penalty": 0.01,
    "clv_min_n": 50, "clv_clip_pct": 50.0, "clv_require_positive_median": True,
    "weak_clv_penalty": 0.01,
    "strong_min_n": 150, "blocked_families": ["OU35"],
    "evidence_path": "output/edge_engine_evidence.json",
}


def load_params(path: str | Path | None = None) -> dict:
    """DEFAULTS overlaid with the config file. An unreadable file falls back to DEFAULTS (which
    are the same conservative values) and says so; it never raises."""
    p = dict(DEFAULTS)
    try:
        with open(path or CONFIG_PATH, encoding="utf-8") as fh:
            j = json.load(fh)
        for k, v in j.items():
            if isinstance(v, dict) and isinstance(p.get(k), dict):
                p[k] = {**p[k], **v}
            else:
                p[k] = v
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"[market_edge] config unreadable ({type(e).__name__}) - using DEFAULTS")
    if p.get("devig_method") not in ("power", "proportional"):
        raise ValueError(f"devig_method must be power|proportional, got {p.get('devig_method')!r}")
    return p


def display_tier(state: str) -> str:
    """PRESENTATION ONLY. Never feed this back into a decision, a stake or a ledger tier."""
    return _DISPLAY.get(str(state), "AVOID")


# ── de-vig ────────────────────────────────────────────────────────────────────────────────────
def proportional_devig(odds_a: float, odds_b: float) -> float | None:
    """Fair P(side A) = (1/a) / (1/a + 1/b). None unless BOTH sides are real prices."""
    if not _valid_price(odds_a) or not _valid_price(odds_b):
        return None
    ra, rb = 1.0 / odds_a, 1.0 / odds_b
    return ra / (ra + rb)


def power_devig(odds_a: float, odds_b: float, tol: float = 1e-12) -> float | None:
    """Fair P(side A) = (1/a)^k with k solving (1/a)^k + (1/b)^k = 1. None unless both sides are
    real prices. Unlike src/edge_engine.power_devig the bracket is checked and widened, so an
    unusual pair can never silently return the bracket endpoint."""
    if not _valid_price(odds_a) or not _valid_price(odds_b):
        return None
    ra, rb = 1.0 / odds_a, 1.0 / odds_b
    f = lambda k: ra ** k + rb ** k - 1.0                            # noqa: E731 - decreasing in k
    lo, hi = 0.05, 20.0
    if f(lo) < 0 or f(hi) > 0:                                       # pair cannot be de-vigged
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return ra ** (0.5 * (lo + hi))


def overround(odds_a: float, odds_b: float) -> float | None:
    if not _valid_price(odds_a) or not _valid_price(odds_b):
        return None
    return 1.0 / odds_a + 1.0 / odds_b - 1.0


def _valid_price(x) -> bool:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and x > 1.0


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── synthetic-price guard ─────────────────────────────────────────────────────────────────────
def exclude_synthetic_prices(df: pd.DataFrame, family: str, odds_col: str) -> pd.DataFrame:
    """Drop rows whose side-market price was FILLED, not quoted.

    src/backtest.py fills a missing BTTS / O1.5 / O3.5 price with a constant (`_DEFAULT_ODDS`:
    1.85 / 1.40 / 2.60) and flags it `price_synthetic`. Measured 2026-10-25 on the committed
    output/backtest_results_over15.csv: 22,441 of 22,516 rows are the 1.40 fill. An edge against a
    constant is not an edge, so any replay or engine input must drop them:

    * with a `price_synthetic` column: drop every flagged row (a real 1.40 quote survives);
    * without one: fail closed and drop every row priced exactly at the family's fill constant,
      because a filled row cannot be told apart from a real one.
    """
    if df is None or df.empty:
        return df
    if "price_synthetic" in df.columns:
        flag = df["price_synthetic"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        return df[~flag]
    d = SYNTHETIC_DEFAULT_ODDS.get(family)
    if d is None or odds_col not in df.columns:
        return df
    return df[~pd.to_numeric(df[odds_col], errors="coerce").eq(d)]


# ── inputs ────────────────────────────────────────────────────────────────────────────────────
@dataclass
class BookQuote:
    """One bookmaker's two-sided quote. `ts` is when BOTH prices were valid together (the later
    of the two sides' last change under change-log carry-forward, or the run snapshot)."""
    book: str
    odds_a: float | None
    odds_b: float | None
    ts: pd.Timestamp | None
    synthetic: bool = False


@dataclass
class Candidate:
    family: str                                  # OU25 | BTTS | OU15 | OU35
    track: str                                   # standard | new_format | (anything else fails)
    p_model: float | None                        # P(side A) from the track's model
    kickoff: pd.Timestamp | None
    now: pd.Timestamp
    quotes: list = field(default_factory=list)   # [BookQuote]
    row_odds_a: float | None = None              # the board price legacy decides at
    row_odds_b: float | None = None
    league: str = ""
    league_history_n: int | None = None
    model_trained_at: pd.Timestamp | None = None
    clv: dict | None = None                      # {n, mean, median, positive_pct} (retrospective)
    evidence: dict | None = None                 # PROSPECTIVE evidence for this engine/track
    price_synthetic: bool = False


def _utc(ts) -> pd.Timestamp | None:
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _nan_out(c: Candidate) -> dict:
    out = {k: float("nan") for k in OUTPUT_COLS}
    out.update(selection_engine_version=ENGINE_VERSION, market=c.family, track=c.track,
               side="", odds_source="", market_source="NONE", books_used="", devig_method="",
               penalty_breakdown="", clv_evidence_state="NONE", price_timestamp="",
               n_books=0, n_one_sided_books=0, n_invalid_books=0)
    return out


def _done(out: dict, state: str, reason: str) -> dict:
    out["decision_state"], out["decision_reason"] = state, reason
    return out


def strong_evidence_ok(evidence: dict | None, track: str, family: str, params: dict) -> bool:
    """STRONG gate. Only PROSPECTIVE evidence recorded for THIS engine version counts: n at least
    strong_min_n and a CLV confidence interval whose lower bound is above zero."""
    if not evidence or evidence.get("engine_version") != ENGINE_VERSION:
        return False
    e = (evidence.get("tracks") or {}).get(f"{track}|{family}") or {}
    try:
        return (e.get("evidence_type") == "prospective"
                and int(e.get("n", 0)) >= int(params["strong_min_n"])
                and float(e.get("ci_lo")) > 0.0)
    except (TypeError, ValueError):
        return False


def evaluate(c: Candidate, params: dict | None = None) -> dict:
    """One candidate -> every output column for the better side. Pure; never raises on bad data
    (bad data is a NO_BET with a reason). A blocked family (O3.5) is NO_BET on every path,
    including the early OBSERVE ones."""
    P = params or load_params()
    out = _evaluate(c, P)
    if c.family in set(P.get("blocked_families") or ()) and out["decision_state"] != "NO_BET":
        out["decision_reason"] = (f"MARKET_BLOCKED({c.family}); "
                                  f"engine_would_be={out['decision_state']}({out['decision_reason']})")
        out["decision_state"] = "NO_BET"
    return out


def _evaluate(c: Candidate, P: dict) -> dict:
    out = _nan_out(c)
    now, ko = _utc(c.now), _utc(c.kickoff)
    clv = c.clv or {}
    out.update(clv_track_n=clv.get("n", 0), clv_track_mean=clv.get("mean", float("nan")),
               clv_track_median=clv.get("median", float("nan")),
               clv_track_positive_pct=clv.get("positive_pct", float("nan")))
    key = f"{c.track}|{c.family}"
    strong_ok = strong_evidence_ok(c.evidence, c.track, c.family, P)
    out["clv_evidence_state"] = ("PROSPECTIVE_VALIDATED" if strong_ok else
                                 "RETROSPECTIVE_ONLY" if clv.get("n", 0) else "NONE")

    # 1. fail-closed gates ─────────────────────────────────────────────────────────────────────
    if c.family not in SIDES:
        return _done(out, "NO_BET", f"UNSUPPORTED_MARKET({c.family})")
    if c.track not in TRACKS:
        return _done(out, "NO_BET", f"UNKNOWN_TRACK({c.track})")
    try:
        pm = float(c.p_model)
    except (TypeError, ValueError):
        pm = float("nan")
    if not (0.0 < pm < 1.0):
        return _done(out, "NO_BET", "MISSING_MODEL")
    out["p_model_raw"] = pm
    if now is None or ko is None:
        return _done(out, "NO_BET", "MISSING_KICKOFF")
    mtk = (ko - now).total_seconds() / 60.0
    out["minutes_to_kickoff"] = round(mtk, 1)
    if mtk <= 0:
        return _done(out, "NO_BET", "POST_KICKOFF")
    if c.price_synthetic:
        return _done(out, "NO_BET", "SYNTHETIC_PRICE")

    # 2. per-book validation ───────────────────────────────────────────────────────────────────
    near = mtk <= float(P["near_kickoff_minutes"])
    stale_hard = float(P["stale_hard_minutes_near"] if near else P["stale_hard_minutes"])
    valid, n_one, n_invalid, n_stale = [], 0, 0, 0
    for q in c.quotes or []:
        a_ok, b_ok = _valid_price(q.odds_a), _valid_price(q.odds_b)
        ts = _utc(q.ts)
        if q.synthetic or not str(q.book or "").strip():
            n_invalid += 1
            continue
        if a_ok != b_ok:
            n_one += 1
            continue
        if not (a_ok and b_ok):
            continue
        if ts is None or ts > now or ts >= ko:                        # no future / in-play prices
            n_invalid += 1
            continue
        ovr = overround(q.odds_a, q.odds_b)
        if ovr is None or not (float(P["overround_min"]) <= ovr <= float(P["overround_max"])):
            n_invalid += 1
            continue
        age = (now - ts).total_seconds() / 60.0
        if age > stale_hard:
            n_stale += 1
            continue
        valid.append((q, ts, age, ovr))
    out.update(n_one_sided_books=n_one, n_invalid_books=n_invalid)

    # 3. consensus hierarchy ───────────────────────────────────────────────────────────────────
    # A validated exchange would sit at the top; exchange_validated is false today, so skipped.
    if not valid:
        if n_stale:
            return _done(out, "NO_BET", f"STALE_PRICE({n_stale} book(s) older than {stale_hard:.0f}m)")
        row_one = _valid_price(c.row_odds_a) != _valid_price(c.row_odds_b)
        if n_one or row_one:
            out["market_source"] = "ONE_SIDED"
            return _done(out, "OBSERVE", "ONE_SIDED_PRICE")
        return _done(out, "NO_BET", "NO_PRICE" if not n_invalid else f"NO_VALID_PRICE({n_invalid} invalid)")
    n_books = len(valid)
    out["n_books"] = n_books
    out["market_source"] = "MEDIAN_TWO_SIDED_BOOKS" if n_books >= 2 else "SINGLE_TWO_SIDED_BOOK"
    out["books_used"] = ";".join(sorted(str(q.book) for q, *_ in valid))
    oldest = min(ts for _, ts, _, _ in valid)
    out["price_timestamp"] = oldest.strftime("%Y-%m-%dT%H:%M:%SZ")
    age = max(a for _, _, a, _ in valid)
    out["price_age_minutes"] = round(age, 1)
    out["market_overround"] = round(statistics.median(o for *_, o in valid), 6)

    # 4. de-vig: both methods per book, median across books ────────────────────────────────────
    pw = [power_devig(q.odds_a, q.odds_b) for q, *_ in valid]
    pr = [proportional_devig(q.odds_a, q.odds_b) for q, *_ in valid]
    if any(v is None for v in pw + pr):
        return _done(out, "NO_BET", "DEVIG_FAILED")
    p_pow, p_prop = statistics.median(pw), statistics.median(pr)
    method = P["devig_method"]
    used = pw if method == "power" else pr
    p_mkt_a = p_pow if method == "power" else p_prop
    sd = statistics.pstdev(used) if len(used) > 1 else 0.0
    out["devig_method"] = method

    # execution prices per side
    def _exec(side_idx: int) -> tuple[float | None, str]:
        row = c.row_odds_a if side_idx == 0 else c.row_odds_b
        books = [q.odds_a if side_idx == 0 else q.odds_b for q, *_ in valid]
        if P["price_source"] == "best_valid_book":
            return max(books), "BEST_VALID_BOOK"
        if _valid_price(row):
            return float(row), "ROW"
        return float(statistics.median(books)), "MEDIAN_VALID_BOOKS"

    w = float((P["residual_weight"] or {}).get(key, 0.0))
    n_eff = int((P["residual_n_eff"] or {}).get(key, 0) or 0)
    clip = float(P["residual_logit_clip"])
    haircut = float(P.get("odds_haircut", 0.0) or 0.0)

    # retrain recency (recorded timestamp from the model version, never file mtime; unknown = recent)
    trained = _utc(c.model_trained_at)
    recent = trained is None or (now - trained).total_seconds() / 86400.0 < float(P["recent_retrain_days"])
    def _f(x) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")
    # The clipped mean alone is not enough: on the committed ledgers new_format|OU25 has mean +4.69
    # but median 0.0 and 45% positive — a tail, not a track. Both must be above zero (NaN -> weak).
    weak_clv = (int(clv.get("n", 0) or 0) < int(P["clv_min_n"]) or not (_f(clv.get("mean")) > 0)
                or (bool(P.get("clv_require_positive_median", True)) and not (_f(clv.get("median")) > 0)))

    sides = []
    for i, s in enumerate(SIDES[c.family]):
        pm_s = pm if i == 0 else 1.0 - pm
        pk_s = p_mkt_a if i == 0 else 1.0 - p_mkt_a
        pp_s = p_pow if i == 0 else 1.0 - p_pow
        pq_s = p_prop if i == 0 else 1.0 - p_prop
        odds, osrc = _exec(i)
        odds_e = 1.0 + (odds - 1.0) * (1.0 - haircut)
        raw = _logit(pm_s) - _logit(pk_s)
        adj = w * max(-clip, min(clip, raw))
        p_dec = _sigmoid(_logit(pk_s) + adj)
        dis = abs(pm_s - pk_s)
        pen = {
            "books": float(P["z"]) * max(sd, float(P["book_sd_floor"])) / math.sqrt(n_books),
            "devig": abs(pp_s - pq_s),
            "residual": abs(p_dec - pk_s) * float(P["residual_n0"]) / (float(P["residual_n0"]) + n_eff),
            "league": float(P["league_penalty_max"]) * float(P["league_history_n0"])
                      / (float(P["league_history_n0"]) + float(c.league_history_n or 0)),
            "disagreement": float(P["disagree_slope"]) * max(0.0, dis - float(P["disagree_soft"])),
            "long_odds": float(P["long_odds_slope"]) * max(0.0, odds_e - float(P["long_odds_soft"])),
            "stale": float(P["stale_slope_per_hour"]) * max(0.0, age - float(P["stale_soft_minutes"])) / 60.0,
            "retrain": float(P["recent_retrain_penalty"]) if recent else 0.0,
            "clv": float(P["weak_clv_penalty"]) if weak_clv else 0.0,
        }
        u = sum(pen.values())
        p_lb = min(max(p_dec - u, 0.0), 1.0)
        best = max(q.odds_a if i == 0 else q.odds_b for q, *_ in valid)
        sides.append(dict(side=s, p_market_fair=pk_s, p_market_fair_power=pp_s,
                          p_market_fair_proportional=pq_s, p_decision=p_dec, p_lower_bound=p_lb,
                          market_residual_raw=raw, market_residual_adjusted=adj,
                          odds=odds_e, odds_source=osrc, best_book_odds=best,
                          ev_point=p_dec * odds_e - 1.0, ev_lower_bound=p_lb * odds_e - 1.0,
                          uncertainty_penalty=u, disagreement=dis,
                          penalty_breakdown=";".join(f"{k}={v:.4f}" for k, v in pen.items())))
    ch = max(sides, key=lambda d: d["ev_lower_bound"])                # ties -> side A
    out.update({k: v for k, v in ch.items() if k in OUTPUT_COLS})
    out.update(residual_weight=w, effective_sample_size=n_eff)
    for k in ("p_market_fair", "p_market_fair_power", "p_market_fair_proportional", "p_decision",
              "p_lower_bound", "market_residual_raw", "market_residual_adjusted", "ev_point",
              "ev_lower_bound", "uncertainty_penalty"):
        out[k] = round(float(out[k]), 6)
    out["_sides"] = sides                                              # both sides, for the stake plan join

    # 5. state ─────────────────────────────────────────────────────────────────────────────────
    # The value path decides a provisional state; the caps after it can only LOWER it. Nothing
    # below can raise a state, so extreme disagreement, long odds or a blocked market never help.
    ev_lb, ev_pt = ch["ev_lower_bound"], ch["ev_point"]
    if ev_lb > 0:
        if n_books < int(P["min_books_qualified"]):
            state, reason = "OBSERVE", f"INSUFFICIENT_BOOKS({n_books}<{P['min_books_qualified']})"
        # EVIDENCE GATE. QUALIFIED needs at least a track CLV record (n >= clv_min_n, mean > 0).
        # Today that record is RETROSPECTIVE and belongs to the legacy picks, so it is weak; without
        # it a positive EV_LB stays OBSERVE. STRONG needs PROSPECTIVE evidence for this engine.
        elif weak_clv and not strong_ok:
            state, reason = "OBSERVE", "EV_LB>0 but NO_TRACK_EVIDENCE(clv n/mean)"
        elif strong_ok:
            state, reason = "STRONG", "EV_LB>0; prospective CLV evidence validated"
        else:
            state, reason = "QUALIFIED", "EV_LB>0; STRONG needs prospective CLV evidence (none)"
    elif ev_pt > 0:
        state, reason = "OBSERVE", "EV_LB_NOT_POSITIVE"
    else:
        state, reason = "NO_BET", "NO_POSITIVE_EV"

    def _cap(st: str, rs: str, ceiling: str, why: str) -> tuple[str, str]:
        if STATE_RANK[st] > STATE_RANK[ceiling]:
            return ceiling, f"{why}; engine_would_be={st}"
        return st, (rs if why in rs else f"{rs}; {why}")

    if ch["odds"] > float(P["max_odds"]):
        state, reason = _cap(state, reason, "OBSERVE", f"LONG_ODDS(>{P['max_odds']})")
    if ch["disagreement"] > float(P["disagree_hard"]):
        state, reason = _cap(state, reason, "OBSERVE",
                             f"EXTREME_DISAGREEMENT(|p_model-p_market|={ch['disagreement']:.3f})")
    if c.family in set(P.get("blocked_families") or ()):              # never staked, always NO_BET
        state, reason = "NO_BET", f"MARKET_BLOCKED({c.family}); engine_would_be={state}"
    return _done(out, state, reason)


# ── helpers the shadow uses ───────────────────────────────────────────────────────────────────
_VER_TS = re.compile(r"_(\d{8}T\d{6}Z)_")


def trained_at_from_version(model_version) -> pd.Timestamp | None:
    """Training time parsed from a model_version string (`standard_20261018T080423Z_...`). A
    recorded value, never a file mtime. Unparseable -> None, which the engine treats as recent."""
    m = _VER_TS.search(str(model_version or ""))
    if not m:
        return None
    try:
        from datetime import datetime, timezone
        return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))
    except (TypeError, ValueError):
        return None


def clv_track_stats(bets_ledger: pd.DataFrame | None, side_ledger: pd.DataFrame | None,
                    clip_pct: float = 50.0) -> dict:
    """RETROSPECTIVE CLV of the LEGACY selections per track|family: {n, mean, median, positive_pct}.

    Context for the weak-CLV penalty only — it is evidence about the legacy engine's picks, not
    about this engine, so it can never unlock STRONG. clv_pct carries garbage tails (the committed
    bets_ledger reaches +287%), so values are clipped to +-clip_pct before the mean.
    """
    out: dict = {}
    frames = []
    if bets_ledger is not None and len(bets_ledger) and "clv_pct" in bets_ledger:
        b = bets_ledger[["model_type", "clv_pct"]].copy()
        b["family"] = "OU25"
        frames.append(b)
    if side_ledger is not None and len(side_ledger) and "clv_pct" in side_ledger:
        s = side_ledger[["model_type", "market", "clv_pct"]].copy()
        s["family"] = s["market"].astype(str).str.lower().map({"btts": "BTTS", "over15": "OU15",
                                                               "over35": "OU35"})
        frames.append(s.drop(columns=["market"]))
    if not frames:
        return out
    d = pd.concat(frames, ignore_index=True)
    d["clv"] = pd.to_numeric(d["clv_pct"], errors="coerce")
    d = d.dropna(subset=["clv", "family"])
    for (mt, fam), g in d.groupby(["model_type", "family"]):
        v = g["clv"].clip(-clip_pct, clip_pct)
        out[f"{mt}|{fam}"] = {"n": int(len(v)), "mean": round(float(v.mean()), 4),
                              "median": round(float(v.median()), 4),
                              "positive_pct": round(float((v > 0).mean()), 4)}
    return out


def load_evidence(path: str | Path) -> dict | None:
    """Prospective evidence file for STRONG. Absent today -> None -> STRONG unreachable."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001 - fail closed
        return None
