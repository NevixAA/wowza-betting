"""TOP-3 PER LEAGUE PER GAMEWEEK - production tip selection (owner 2026-11-08). Behind a flag, default OFF.

OWNER 2026-11-11 ("Clean rebuild"; TOP-3 "Paper track"). Supersedes the tier / EV text below where they differ:
  * POOL: tips from the retrained models that clear their cell's ONE owner-approved bar (bars_opt cell,
    valuable_th; src/bars_opt/serve.py). A league x market without its own applied CELL has no tip
    (NO_LEAGUE_BAR) - the legacy three-tier entries are retired. No tier is computed, shown or stored.
  * RANK: T3 = R3_CAL_EV_LCB of protocol top3-ranking-paper/1 (config/top3_ranking_paper_protocol.json,
    sha256 363e7f41...), computed by THE research code (src/research/top3_ranking.rank_key, imported, not
    forked): cautious calibrated value (fair + lam_lo * (p_used - fair)) * odds - 1, lam_lo from the cell's
    own calibration (its FROZEN statistics: the research freeze record, scripts/top3_paper_freeze.py).
    The old EV ranking is DROPPED entirely - not a
    key, not a tie-break, not a reference track. A cell without a T3 statistic ranks after every cell with
    one, in kickoff order. Ties: kickoff, then fixture.
  * SLOTS: TOP first; TIER2 (a tip in its cell's VALUABLE..MARKSMAN band where that cell's TIER2 is on,
    study 12.2) only fills slots left empty. At most 3, never padded, the direction rule unchanged.
  * Each pick carries its cell status (PROMISING / UNPROVEN) and its slot (TOP / TIER2).
  * FAIL CLOSED: no frozen T3 statistics -> nothing is frozen or sent (status T3_STATS_MISSING).
  * TOP-3 is a named PAPER track in the Clean Record; stakes follow the per-cell status only.
  * OWN LIST (owner 2026-11-11 "Own list"): a LEAGUE-WIDE cell (source NONE / bar_mode LEAGUE_WIDE, e.g.
    Argentina NF BTTS) is NOT in the TOP-3 pool. Its qualifying tips form the league's own line ("League rule:
    BTTS Yes, every match"), selection_mode LEAGUE_RULE, in kickoff order, uncapped: it never takes, and is
    never pushed out of, a TOP-3 / TIER2 slot. The TOP-3 / TIER2 pool = MODEL cells only.
  * DIRECTION across both lists (production rule, src/direction_guard.py): (1) the fixture's first LIVE
    production tip (committed) stands - an own-list or TOP-3 tip opposite to it is dropped; (2) an own-list tip
    stands over a TOP-3 model tip of the opposite direction on the same fixture (the league rule is decided
    first and is the staked line - "real money wins"); the model tip yields (DIRECTION_CONFLICT) and the next
    model qualifier takes the slot (no padding).

    python -m src.top3_league run       # predict.yml: sync candidates, freeze due league-weeks, report
    python -m src.top3_league notify    # predict.yml: send newly frozen picks to Telegram (once each)
    python -m src.top3_league settle    # update_results.yml: grade finished picks (grading columns only)
    python -m src.top3_league report    # rewrite output/top3_league_report.json

Owner rules (2026-11-08, verbatim intent):
  * "3 tips every gw per league (bets value)": each league gets its 3 best VALUE tips per gameweek, across
    ALL team markets together - O/U 2.5 (either side), BTTS (Yes), Over 1.5, Over 3.5.
  * A tip qualifies only when its edge clears THAT league's OWN bar for THAT market ("its cant be global...
    every league has different numbers"). NO GLOBAL BAR, ever: a league x market without its own
    owner-applied bars yields no TOP-3 tip, with the reason recorded (NO_LEAGUE_BAR / BAR_NOT_OWNER_APPLIED).
  * Fewer than 3 qualifying -> only those (0, 1 or 2). Never padded.
  * The direction rule (src/direction_guard.py): never opposite goal directions on one fixture (BTTS Yes =
    OVER). Checked against the picks of the same week AND against the fixture's earlier live production tip.

THE BARS. A league's bars are read from the production params files that ONLY src/threshold_apply writes
(models/best_params_standard.json for O/U 2.5, models/best_params_side_markets.json[market] for the side
markets). A league x market counts as having its own bars only when
    (a) sniper_th, marksman_th AND valuable_th are all present, finite and ordered (VALUABLE <= MARKSMAN <=
        SNIPER) - the per-league VALUABLE bar exists only in the clean-retrain bars (league_bars/1), so the
        pre-repair params (no valuable_th) can never be used here; and
    (b) its entry "<market>|<league>" was applied by src/threshold_apply (config/threshold_applied.json
        history), i.e. the OWNER approved it.
Production's own `approved` flag (= statistically PROVEN under the strict rule) is carried on every pick as
bar_status (PROVEN / UNPROVEN) but does not gate TOP-3: the owner approved each league's bars himself.

EDGE AND VALUE. edge = p_model_raw - p_market_fair for the pick's side, where p_market_fair is the power
de-vig of a TWO-SIDED pre-kickoff price (market_edge_v2, the source the clean bars were fitted against).
No two-sided price -> no tip (NO_TWO_SIDED_PRICE). The tier is the league's: SNIPER / MARKSMAN / VALUABLE by
its own three bars (src/side_thresholds.tier_for); below its VALUABLE bar -> BELOW_LEAGUE_BAR.
Ranking among qualifying tips: RS-1 of the research panel (src/research/league_weekly.rs1_key:
ev_lower_bound first), across all markets of the league-week.

CANDIDATES. The same pure builder as the research panel (league_weekly.build_candidates over
output/edge_engine_shadow.csv, strictly pre-kickoff) into this module's OWN first-sight store
output/top3_league_candidates.csv. Production never reads the research panel's files.

GAMEWEEK, LOCK, FREEZE (GW-1 + WL-1, with one production change).
  * League-week = src/gameweek.league_week_id (GW-1, tip day). LOCK = the earliest kickoff among the
    league-week's candidates (WL-1).
  * The research panel freezes AT the lock. A tip must go out BEFORE its own kickoff, and the week's
    earliest fixture kicks off at the lock, so production freezes at FREEZE = LOCK - TOP3_FREEZE_LEAD_MIN.
    The lead is an owner value (env TOP3_FREEZE_LEAD_MIN, minutes). There is NO default: unset -> nothing
    is frozen or sent and the report says FREEZE_LEAD_NOT_SET. predict.yml runs every 15-30 min Fri-Sun and
    every 30-60 min Mon-Thu, so the lead must exceed the gap between two runs or a week can be frozen late.
  * The week's TOP-3 = the qualifying candidates FIRST SEEN STRICTLY BEFORE FREEZE, RS-1 ranked, written
    ONCE at the first run at/after FREEZE and never rewritten (first-sight store, digest sidecar).
  * Late fixtures: a candidate first seen at/after FREEZE is never a member of that week (the panel's rule,
    applied at FREEZE) - counted as FIRST_SEEN_AFTER_FREEZE.
  * A qualifying candidate whose kickoff is not strictly after the moment of issue (a late run) cannot be
    tipped and is skipped (STARTED_BEFORE_ISSUE); the next qualifying candidate takes its place. That is not
    padding: it also clears its league's bar.

STAKES. This module never sizes or places a stake; the stake plan is unchanged. Per league the policy switch
config/bankroll_policy.json -> top3_league.leagues.<league> in {CURRENT, PAPER, REAL} (absent = CURRENT) is
recorded on every pick; CURRENT records whether the league x market x side track is real money today.
Wiring REAL picks into the stake plan / execution is an owner decision for go-live, not done here.

FLAG. env TOP3_LEAGUE_ENABLED (true/1/yes). Off (the default): every entrypoint returns at once and writes
nothing. The flag and the lead are read HERE, not in config.py, so turning TOP-3 on does not move the
research lock's deployed_env; the go-live switch-on is an explicit owner step.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.research import league_weekly as LW
from src.research.first_sight import FirstSightStore

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = config.OUTPUT_DIR
FLAG_ENV = "TOP3_LEAGUE_ENABLED"
LEAD_ENV = "TOP3_FREEZE_LEAD_MIN"
SELECTION_MODE = "TOP3_LEAGUE_GW"
OWN_LIST_MODE = "LEAGUE_RULE"              # owner 2026-11-11 "Own list": league-wide cells, outside the 3 slots
VERSION = "TOP3-2"                         # owner 2026-11-11: cell bar + T3 ranking, no tiers
from src.research.top3_ranking import LIVE_RULE as T3_RULE, in_pool  # noqa: E402  the research rule, not forked
SLOT_TOP, SLOT_TIER2 = "TOP", "TIER2"
TOP_N = 3
BAR_SOURCE = "LEAGUE"                      # the only source this module ever uses

CANDIDATES_NAME = "top3_league_candidates.csv"
PICKS_NAME = "top3_league_ledger.csv"
WEEKS_NAME = "top3_league_weeks.csv"
REPORT_NAME = "top3_league_report.json"
NOTIFIED_NAME = "top3_league_notified.json"

STD_PARAMS = Path("models") / "best_params_standard.json"
SIDE_PARAMS = Path("models") / "best_params_side_markets.json"
APPLIED_LEDGER = Path("config") / "threshold_applied.json"

#: panel market -> (params key, direction-guard market)
MARKETS = {"OU25": ("ou25", "ou25"), "BTTS": ("btts", "btts"), "OVER15": ("over15", "over15"),
           "OVER35": ("over35", "over35")}
POLICY_SWITCH = ("CURRENT", "PAPER", "REAL")

# exclusion reasons (one per candidate, first that applies)
R_NO_PRICE = "NO_TWO_SIDED_PRICE"
R_NO_BAR = "NO_LEAGUE_BAR"
R_NOT_APPLIED = "BAR_NOT_OWNER_APPLIED"
R_BAD_BAR = "LEAGUE_BAR_INVALID"
R_BELOW = "BELOW_LEAGUE_BAR"
R_LATE = "FIRST_SEEN_AFTER_FREEZE"
R_STARTED = "STARTED_BEFORE_ISSUE"
R_DIRECTION = "DIRECTION_CONFLICT"
R_RANK = "RANKED_BELOW_TOP3"

PICK_KEY = ["selection_mode", "league_week_id", "top3_rank"]
PICK_FROZEN = [
    "pick_id", "selection_mode", "version", "league_week_id", "top3_rank", "week_lock_ts", "freeze_ts",
    "issued_at", "league", "model_type", "season", "fixture_id", "match_date", "home_team", "away_team",
    "kickoff_utc", "market", "side", "odds", "opposite_side_odds", "p_model_raw", "p_market_fair", "edge",
    "slot", "cell_status", "p_used", "t3_key", "t3_has_stat", "edge_calibrated", "bar_valuable", "bar_source",
    "bar_status",
    "bar_evidence",
    "ev_point", "ev_lower_bound", "p_lower_bound", "price_quality_state", "market_source", "n_books",
    "first_seen_at", "prediction_engine_version", "selection_engine_version", "policy_switch",
    "track_is_real", "stake_effect",
]
PICK_GRADING = ["home_goals", "away_goals", "result", "won", "flat_pnl", "settle_status", "settled_at"]
PICK_COLS = PICK_FROZEN + PICK_GRADING

WEEK_KEY = ["selection_mode", "league_week_id"]
WEEK_COLS = ["selection_mode", "version", "league_week_id", "league", "week_lock_ts", "freeze_ts", "issued_at",
             "kickoff_utc", "n_candidates", "n_qualifying", "n_picks", "excluded", "frozen_at"]


# ── switches ────────────────────────────────────────────────────────────────────────────────────
def enabled(env: dict | None = None) -> bool:
    v = (env if env is not None else os.environ).get(FLAG_ENV, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def freeze_lead_min(env: dict | None = None) -> float | None:
    """The owner's lead in minutes, or None when unset/unusable (no default)."""
    v = str((env if env is not None else os.environ).get(LEAD_ENV, "")).strip()
    if not v:
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) and x > 0 else None


# ── the league's own bars ───────────────────────────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return {}


def bars_context(root: Path | None = None) -> dict:
    """The production params files and the owner-applied entry ids, read once per run."""
    r = Path(root) if root else ROOT
    led = _read_json(r / APPLIED_LEDGER)
    applied = set()
    for h in (led.get("history") or []):
        applied.update(str(e) for e in (h.get("entries") or []))
    return {"standard": _read_json(r / STD_PARAMS), "side": _read_json(r / SIDE_PARAMS), "applied": applied}


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


#: panel side -> the bars_opt cell side of each params key (src/bars_opt/serve.py)
_CELL_SIDE = {("ou25", "OVER"): "over", ("ou25", "UNDER"): "under", ("btts", "YES"): "yes",
              ("over15", "OVER"): "yes", ("over35", "OVER"): "yes"}


def league_bars(ctx: dict, league: str, market: str, side: str | None = None) -> tuple[dict | None, str]:
    """(bars, "") or (None, reason) for one league x panel market (x side). Never a global value.
    An OPTIMIZED entry (bars_opt cells) returns its owner-applied cell for the side under "cell"."""
    key = MARKETS[market][0]
    lp = (ctx["standard"] if key == "ou25" else (ctx["side"].get(key) or {})).get(league)
    if not isinstance(lp, dict):
        return None, R_NO_BAR
    from src.bars_opt import serve as BS
    if BS.CELL_KEY in lp:
        cs = _CELL_SIDE.get((key, str(side or "").upper()))
        if cs is None:
            return None, R_NO_BAR
        cell, why = BS.served_cell(lp, key, league, cs, frozenset(ctx["applied"]))
        if cell is None:
            return None, (R_NOT_APPLIED if "not applied" in why else (R_BAD_BAR if "cell lacks" in why
                                                                      or "not ordered" in why else R_NO_BAR))
        return {"valuable": float(cell["valuable_th"]), "marksman": float(cell["marksman_th"]),
                "sniper": float(cell["sniper_th"]), "status": str(cell["verdict"]),
                "evidence": f"bars_opt {cell['config']} {cell['mode']} {cell.get('bar_basis') or ''}".strip(),
                "cell": cell}, ""
    # owner 2026-11-11: a legacy three-tier entry (no cell) is no bar any more - one bar per cell
    return None, R_NO_BAR


# ── qualification and selection ─────────────────────────────────────────────────────────────────
#: panel market x side -> the study market-side of its cell (src/bars_opt/study.MARKET_SIDES)
_MSIDE = {("OU25", "OVER"): "over25_over", ("OU25", "UNDER"): "over25_under", ("BTTS", "YES"): "btts_yes",
          ("OVER15", "OVER"): "over15_over", ("OVER35", "OVER"): "over35_over"}


def qualify(cands: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Candidates + edge / tier (QUALIFIED or "") / slot / cell status / p_used / reason ("" = qualifies). Pure.
    A candidate qualifies ONLY on its league x market x side cell's one owner-applied bar."""
    d = LW._plain(cands.copy()) if len(cands) else cands.copy()
    if d.empty:
        return d.assign(edge=pd.Series(dtype=float), tier="", reason="")
    from src.bars_opt import serve as BS
    cols = {k: [] for k in ("edge", "tier", "reason", "bar_valuable", "bar_status", "bar_evidence", "slot",
                            "cell_status", "p_used", "fair", "mside", "track", "league_wide")}
    for r in d.to_dict("records"):
        pm, pf = _num(r.get("p_model_raw")), _num(r.get("p_market_fair"))
        e = pm - pf if not (math.isnan(pm) or math.isnan(pf)) else float("nan")
        b, why = league_bars(ctx, str(r.get("league")), str(r.get("market")), str(r.get("side")))
        t, slot, pu = "", "", float("nan")
        if str(r.get("price_quality_state")) not in LW.PRICED_STATES or math.isnan(e) or not (_num(r.get("odds")) > 1):
            why = R_NO_PRICE
        elif b is not None:
            # the cell decides (src/bars_opt/serve.py): its probability source, measure, odds range, ONE bar
            res = BS.score(pd.DataFrame([{"league": str(r.get("league")), "p_side": pm, "fair": pf,
                                          "odds": _num(r.get("odds"))}]), b["cell"])
            t = str(res["tier"].iloc[0] or "")
            if not t:
                why = R_BELOW if res["reason"].iloc[0] == BS.R_BELOW else str(res["reason"].iloc[0])
            else:
                pu = _num(res["p_used"].iloc[0])
                t2 = bool(res["tier2_ok"].iloc[0]) if pd.notna(res["tier2_ok"].iloc[0]) else False
                slot = SLOT_TIER2 if (str(res["band"].iloc[0]) == "BAND_VM" and t2) else SLOT_TOP
        cols["edge"].append(e)
        cols["tier"].append(t if not why else "")
        cols["reason"].append(why)
        cols["bar_valuable"].append(b["valuable"] if b else float("nan"))
        cols["bar_status"].append(b["status"] if b else "")
        cols["bar_evidence"].append(b["evidence"] if b else "")
        cols["slot"].append(slot if not why else "")
        cols["cell_status"].append(b["status"] if b else "")
        cols["p_used"].append(pu)
        cols["fair"].append(pf)
        cols["mside"].append(_MSIDE.get((str(r.get("market")), str(r.get("side")).upper()), ""))
        cols["track"].append(str(r.get("model_type") or ""))
        # the TOP-3 pool = MODEL cells only (src.research.top3_ranking.in_pool, owner "Own list")
        cols["league_wide"].append(bool(b is not None and not in_pool(b["cell"])))
    return d.assign(**cols)


# ── T3 ranking (owner 2026-11-11; protocol top3-ranking-paper/1, the research code) ─────────────────────
def t3_keys(q: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """q + t3_key / t3_has_stat from src.research.top3_ranking.rank_key (rule R3_CAL_EV_LCB). Pure."""
    from src.research import top3_ranking as TR
    if q is None or q.empty:
        return q
    b = q.assign(odds=pd.to_numeric(q["odds"], errors="coerce"), p_used=pd.to_numeric(q["p_used"], errors="coerce"),
                 fair=pd.to_numeric(q["fair"], errors="coerce"))
    k = TR.rank_key(b[TR.CELL + ["fair", "p_used", "odds"]].reset_index(drop=True),
                    stats if stats is not None else pd.DataFrame(columns=TR.CELL + TR.STAT_COLS), T3_RULE)
    out = q.copy()
    out["t3_key"] = k["key"].to_numpy()
    out["t3_has_stat"] = k["has_stat"].to_numpy()
    # the CALIBRATED edge shown for the pick (owner 2026-11-11, src/calibrated_edge.py; same frozen statistics)
    from src import calibrated_edge as CE
    out["edge_calibrated"] = CE.calibrated(b.reset_index(drop=True), stats).to_numpy() \
        if stats is not None else np.nan
    return out


def load_t3_stats(root: Path | None = None) -> pd.DataFrame | None:
    """The FROZEN per-cell T3 statistics: THE research freeze record (scripts/top3_paper_freeze.py, named by
    config/top3_paper_tracks.json -> freeze_ledger, hash-checked by src.research.top3_paper.load_freeze) - one
    freeze for the live TOP-3 and its paper tracks, never a second builder. None when not frozen / not valid."""
    from src.research import top3_paper as TP3
    try:
        cfg = TP3.load_config(root)
        stats, _rec, _sha, why = TP3.load_freeze(cfg, root)
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[top3_league] T3 statistics unreadable ({type(e).__name__}: {e})")
        return None
    if why:
        log.warning(f"[top3_league] T3 statistics not usable: {'; '.join(why)[:300]}")
        return None
    return stats


def _direction(market: str, side: str) -> str | None:
    from src.direction_guard import goal_direction
    return goal_direction(MARKETS[str(market)][1], side)


def select_week(q: pd.DataFrame, freeze: pd.Timestamp, issue: pd.Timestamp,
                committed: dict | None = None, stats: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """TOP-3 of ONE league-week from qualified candidates `q` (qualify()). Returns (picks with
    top3_rank 1..k, k <= 3, never padded; {reason: count} for every candidate not picked). Pure."""
    committed = committed or {}
    excluded: Counter = Counter()

    def bump(k):
        excluded[k] += 1
    if q is None or q.empty:
        return q.iloc[0:0] if q is not None else pd.DataFrame(), dict(excluded)
    q = t3_keys(q.reset_index(drop=True), stats)
    fs = LW._ts(q["first_seen_at"])
    ko = LW._ts(q["kickoff_utc"])
    if "league_wide" not in q.columns:
        q["league_wide"] = False
    pool, own, dirs = [], [], {}
    for i, r in enumerate(q.to_dict("records")):
        if not (pd.notna(fs.iloc[i]) and fs.iloc[i] < freeze):
            bump(R_LATE)
        elif r.get("reason"):
            bump(r["reason"])
        elif bool(r.get("league_wide")):
            own.append(i)                       # the league's OWN line, outside the TOP-3 slots
        else:
            # owner 2026-11-11: TOP before TIER2 (TIER2 only fills empty slots); then T3 (cells with a
            # statistic first, higher key first); then kickoff, then fixture. No EV anywhere.
            key = _num(r.get("t3_key"))
            has = bool(r.get("t3_has_stat")) and not math.isnan(key)
            kick = ko.iloc[i].value if pd.notna(ko.iloc[i]) else float("inf")
            pool.append(((0 if r.get("slot") != SLOT_TIER2 else 1), (0, -key) if has else (1, 0.0), kick,
                         str(r.get("fixture_id")), i))
    pool.sort()
    # 1) the own list first (league rule, every qualifying match, kickoff order): only the fixture's committed
    #    live production tip can stop one
    own_taken = []
    for i in sorted(own, key=lambda j: (ko.iloc[j].value if pd.notna(ko.iloc[j]) else float("inf"),
                                        str(q.iloc[j]["fixture_id"]))):
        r = q.iloc[i]
        if not (pd.notna(ko.iloc[i]) and ko.iloc[i] > issue):
            bump(R_STARTED)
            continue
        fx = (str(r["match_date"])[:10], str(r["home_team"]), str(r["away_team"]))
        dr = _direction(r["market"], r["side"])
        stood = dirs.get(fx) or (committed.get(fx) or {}).get("direction")
        if dr and stood and dr != stood:
            bump(R_DIRECTION)
            continue
        if dr:
            dirs[fx] = dr
        own_taken.append(i)
    # 2) TOP-3 / TIER2 from MODEL cells only; a model tip opposite to an own-list tip yields
    taken = []
    for *_k, i in pool:
        r = q.iloc[i]
        if len(taken) >= TOP_N:
            bump(R_RANK)
            continue
        if not (pd.notna(ko.iloc[i]) and ko.iloc[i] > issue):
            bump(R_STARTED)
            continue
        fx = (str(r["match_date"])[:10], str(r["home_team"]), str(r["away_team"]))
        dr = _direction(r["market"], r["side"])
        stood = dirs.get(fx) or (committed.get(fx) or {}).get("direction")
        if dr and stood and dr != stood:
            bump(R_DIRECTION)
            continue
        if dr:
            dirs[fx] = dr
        taken.append(i)
    picks = q.iloc[taken].copy()
    picks["top3_rank"] = range(1, len(picks) + 1)
    picks["list"] = SELECTION_MODE
    ol = q.iloc[own_taken].copy()
    ol["top3_rank"] = range(1, len(ol) + 1)
    ol["slot"] = OWN_LIST_MODE
    ol["list"] = OWN_LIST_MODE
    return pd.concat([picks, ol], ignore_index=True), dict(excluded)


def _policy() -> dict:
    try:
        from src.bankroll_policy import load_policy
        return load_policy() or {}
    except Exception:                                                  # noqa: BLE001
        return {}


def policy_switch(policy: dict, league: str) -> str:
    v = str(((policy.get("top3_league") or {}).get("leagues") or {}).get(league, "CURRENT")).strip().upper()
    return v if v in POLICY_SWITCH else "CURRENT"


def freeze_due(cands: pd.DataFrame, weeks_done: set, ctx: dict, now: pd.Timestamp, lead_min: float,
               committed: dict | None = None, policy: dict | None = None,
               real_checker=None, stats: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(new pick rows, new week rows) for every league-week whose FREEZE has passed and that is not
    decided yet. A week with 0 picks is still recorded (decided once, with its reasons). Pure apart
    from the injected policy / checker."""
    if cands is None or cands.empty:
        return pd.DataFrame(columns=PICK_COLS), pd.DataFrame(columns=WEEK_COLS)
    cands = LW._plain(cands)
    policy = policy or {}
    real_checker = real_checker or (lambda *a, **k: False)
    locks = LW.week_locks(cands)
    stamp = LW._stamp(now)
    picks_out, weeks_out = [], []
    for week, g in cands.groupby("league_week_id", sort=True):
        lock = locks.get(week)
        if not week or lock is None or week in weeks_done:
            continue
        freeze = lock - pd.Timedelta(minutes=float(lead_min))
        if now < freeze:
            continue
        q = qualify(g, ctx)
        picks, excl = select_week(q, freeze, now, committed, stats)
        league = str(g["league"].iloc[0])
        n_q = int((q["reason"] == "").sum()) if len(q) else 0
        n_top = int((picks["list"] == SELECTION_MODE).sum()) if len(picks) else 0
        weeks_out.append({"selection_mode": SELECTION_MODE, "version": VERSION, "league_week_id": week,
                          "league": league, "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze),
                          "issued_at": stamp, "kickoff_utc": LW._stamp(lock), "n_candidates": int(len(g)),
                          "n_qualifying": n_q, "n_picks": n_top,
                          "excluded": json.dumps(excl, sort_keys=True), "frozen_at": stamp})
        sw = policy_switch(policy, league)
        for r in picks.to_dict("records"):
            mk = MARKETS[str(r["market"])][1]
            picks_out.append({
                **{k: r.get(k, "") for k in PICK_FROZEN if k in r},
                "pick_id": LW.research_id(VERSION, week, str(r["market"]), str(r["fixture_id"]), str(r["side"])),
                "selection_mode": r["list"], "version": VERSION, "league_week_id": week,
                "top3_rank": int(r["top3_rank"]), "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze),
                "issued_at": stamp, "edge": LW._r(r["edge"]), "slot": r["slot"], "cell_status": r["cell_status"],
                "p_used": LW._r(r["p_used"]), "t3_key": LW._r(r["t3_key"]), "t3_has_stat": bool(r["t3_has_stat"]),
                "edge_calibrated": LW._r(r["edge_calibrated"]),
                "bar_valuable": LW._r(r["bar_valuable"]), "bar_source": BAR_SOURCE,
                "bar_status": r["bar_status"], "bar_evidence": r["bar_evidence"],
                "policy_switch": sw, "track_is_real": bool(real_checker(r.get("model_type"), mk, r.get("side"),
                                                                     r.get("league"))),
                "stake_effect": "NONE (TOP-3 never sizes a stake; stake plan unchanged)",
            })
    return (pd.DataFrame(picks_out).reindex(columns=PICK_COLS) if picks_out else pd.DataFrame(columns=PICK_COLS),
            pd.DataFrame(weeks_out).reindex(columns=WEEK_COLS) if weeks_out else pd.DataFrame(columns=WEEK_COLS))


# ── stores ──────────────────────────────────────────────────────────────────────────────────────
def candidate_store(od: Path) -> FirstSightStore:
    return LW.candidate_store(Path(od) / CANDIDATES_NAME)


def picks_store(od: Path) -> FirstSightStore:
    return FirstSightStore(Path(od) / PICKS_NAME, key=PICK_KEY, columns=PICK_COLS, observed_col="issued_at",
                           kickoff_col="kickoff_utc", frozen_cols=PICK_FROZEN, digest=True)


def weeks_store(od: Path) -> FirstSightStore:
    return FirstSightStore(Path(od) / WEEKS_NAME, key=WEEK_KEY, columns=WEEK_COLS)


# ── entrypoints ─────────────────────────────────────────────────────────────────────────────────
def run(now=None, out_dir: Path | None = None, root: Path | None = None, env: dict | None = None,
        committed: dict | None = None) -> dict:
    """Flag OFF -> returns {'enabled': False} and writes nothing. Never raises."""
    if not enabled(env):
        return {"enabled": False}
    now = LW._now(now)
    od = Path(out_dir) if out_dir else OUT
    res = {"enabled": True, "candidates_appended": 0, "picks_appended": 0, "weeks_decided": 0}
    lead = freeze_lead_min(env)
    try:
        shadow = LW._read(od / "edge_engine_shadow.csv")
        preds = LW._read(od / "predictions.csv")
        plan = LW._read(od / "stake_plan.csv")
        champs = LW._json((Path(root) if root else ROOT) / "models" / "registry" / "current_champions.json")
        from src import market_edge as ME
        cands = LW.build_candidates(shadow, preds, LW.registry_entries(), plan, champs, ME.load_params(), now)
        cs = candidate_store(od)
        res["candidates_appended"] = cs.append(cands)
        if lead is None:
            res["status"] = "FREEZE_LEAD_NOT_SET"
        else:
            if committed is None:
                from src.direction_guard import committed as _committed
                committed = _committed()
            from src.direction_guard import real_track_checker
            stats = load_t3_stats(root)
            if stats is None:
                # fail closed (owner 2026-11-11): no frozen T3 statistics, no ranking, nothing frozen or sent
                res["status"] = "T3_STATS_MISSING"
            else:
                ws = weeks_store(od)
                done = set(ws.read()["league_week_id"]) if len(ws.read()) else set()
                picks, weeks = freeze_due(cs.read(), done, bars_context(root), now, lead, committed, _policy(),
                                          real_track_checker(), stats)
                res["picks_appended"] = picks_store(od).append(picks)
                res["weeks_decided"] = ws.append(weeks)
                res["status"] = "OK"
        write_report(now, od, lead)
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[top3_league] run skipped ({type(e).__name__}: {e}) - other tips unaffected")
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    return res


def settle_picks(picks: pd.DataFrame, v91: pd.DataFrame | None, now: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Fill grading columns of finished picks once; frozen columns are hashed before/after."""
    from src.research.first_sight import frozen_digest
    from src.shadow_settlement import grade
    out = picks.copy().reindex(columns=PICK_COLS).fillna("")
    before = frozen_digest(out, PICK_FROZEN)
    by_id, by_key = LW._scores(v91)
    n, stamp = 0, LW._stamp(now)
    for i in range(len(out)):
        r = out.iloc[i]
        if LW._s(r["settle_status"]):
            continue
        ko = pd.to_datetime(r["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(ko) or now < ko + pd.Timedelta(minutes=LW.SETTLE_DELAY_MIN):
            continue
        g = by_id.get(str(r["fixture_id"]).strip()) or by_key.get((str(r["match_date"])[:10], str(r["home_team"]),
                                                                   str(r["away_team"])))
        if g is None:
            continue
        hg, ag, _src = g
        _outcome, result = grade(LW.SETTLE_MARKET[str(r["market"])], str(r["side"]), hg, ag)
        odds = _num(r["odds"])
        pnl = ((odds - 1.0) if result == "WIN" else -1.0) if result in ("WIN", "LOSS") and odds > 1 else float("nan")
        vals = {"home_goals": hg, "away_goals": ag, "result": result or "",
                "won": "1" if result == "WIN" else "0" if result == "LOSS" else "", "flat_pnl": LW._r(pnl),
                "settle_status": "SETTLED" if result else "SETTLED_NO_SIDE", "settled_at": stamp}
        for k, v in vals.items():
            out.iat[i, out.columns.get_loc(k)] = "" if v == "" else str(v)
        n += 1
    if frozen_digest(out, PICK_FROZEN) != before:
        raise RuntimeError("top3_league settlement would alter a frozen column - refused")
    return out, n


def settle(now=None, out_dir: Path | None = None, env: dict | None = None) -> dict:
    if not enabled(env):
        return {"enabled": False}
    now = LW._now(now)
    od = Path(out_dir) if out_dir else OUT
    ps = picks_store(od)
    picks = ps.read()
    if not len(picks):
        return {"enabled": True, "settled": 0}
    v = ps.verify()
    if v["ok"] is False:
        raise RuntimeError(f"top3 ledger TAMPERED ({v['reason']}) - not settling")
    out, n = settle_picks(picks, LW._read(od / "v91_shadow.csv"), now)
    if n:
        p = od / PICKS_NAME
        tmp = p.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False, lineterminator="\n")
        tmp.replace(p)
    write_report(now, od, freeze_lead_min(env))
    return {"enabled": True, "settled": n}


def report(picks: pd.DataFrame, weeks: pd.DataFrame, now: pd.Timestamp, lead: float | None) -> dict:
    """Per league: the latest decided week's picks (or why there are none) + settled totals (flat 1u)."""
    picks = picks if picks is not None and len(picks) else pd.DataFrame(columns=PICK_COLS)
    weeks = weeks if weeks is not None and len(weeks) else pd.DataFrame(columns=WEEK_COLS)
    leagues = {}
    for lg in sorted(set(weeks["league"]) | set(picks["league"])):
        w = weeks[weeks["league"] == lg].sort_values("league_week_id")
        p_all = picks[picks["league"] == lg]
        own = p_all[p_all["selection_mode"].astype(str) == OWN_LIST_MODE]
        p = p_all[p_all["selection_mode"].astype(str) != OWN_LIST_MODE]       # the TOP-3 track itself
        latest = w["league_week_id"].iloc[-1] if len(w) else None
        cur = p[p["league_week_id"] == latest].sort_values("top3_rank", key=lambda s: s.astype(int))
        st = p[p["settle_status"].astype(str) == "SETTLED"]
        pnl = pd.to_numeric(st["flat_pnl"], errors="coerce")
        ost = own[own["settle_status"].astype(str) == "SETTLED"]
        opnl = pd.to_numeric(ost["flat_pnl"], errors="coerce")
        leagues[lg] = {
            "league_rule": {"picks": int(len(own)), "settled": int(len(ost)),
                            "flat_pnl_units": round(float(opnl.sum()), 3) if len(ost) else 0.0},
            "latest_league_week_id": latest,
            "latest_excluded": json.loads(w["excluded"].iloc[-1]) if len(w) and w["excluded"].iloc[-1] else {},
            "latest_picks": [{k: r.get(k) for k in ("top3_rank", "home_team", "away_team", "kickoff_utc", "market",
                                                    "side", "odds", "edge_calibrated", "slot", "cell_status", "result")}
                             for r in cur.to_dict("records")],
            "weeks_decided": int(len(w)), "picks": int(len(p)), "settled": int(len(st)),
            "won": int((st["won"].astype(str) == "1").sum()),
            "flat_pnl_units": round(float(pnl.sum()), 3) if len(st) else 0.0,
            "roi": round(float(pnl.mean()), 4) if pnl.notna().any() else None}
    return {"generated_at": LW._stamp(now), "selection_mode": SELECTION_MODE, "version": VERSION,
            "freeze_lead_min": lead, "status": "OK" if lead is not None else "FREEZE_LEAD_NOT_SET",
            "note": ("Each league's 3 best tips of the gameweek across O/U 2.5, BTTS, Over 1.5, Over 3.5 from its "
                     "MODEL cells, each cleared its cell's one owner-approved bar, ranked by T3; fewer than 3 when "
                     "fewer clear the bar. League-wide cells are the league's own line (league_rule), outside the 3. "
                     "Flat 1u; a PAPER track - TOP-3 never sizes a stake."),
            "leagues": leagues}


def write_report(now=None, out_dir: Path | None = None, lead: float | None = None) -> dict:
    od = Path(out_dir) if out_dir else OUT
    rep = report(picks_store(od).read(), weeks_store(od).read(), LW._now(now), lead)
    LW._write_json(od / REPORT_NAME, rep)
    return rep


def message(picks: pd.DataFrame) -> str:
    """Telegram text for one league-week's picks (HTML). The edge shown is the CALIBRATED one (owner 2026-11-11)."""
    from html import escape
    from src.calibrated_edge import edge_text
    r0 = picks.iloc[0]
    lines = [f"🏆 <b>TOP-3 {escape(str(r0['league']))}</b> · {escape(str(r0['league_week_id']).split('|')[-1])}"]
    lab = {"OU25": "O/U 2.5", "BTTS": "BTTS", "OVER15": "Over 1.5", "OVER35": "Over 3.5"}
    mode = picks["selection_mode"].astype(str) if "selection_mode" in picks.columns else pd.Series(SELECTION_MODE,
                                                                                                   index=picks.index)
    rule = picks[mode == OWN_LIST_MODE]
    picks = picks[mode != OWN_LIST_MODE]
    for r in picks.sort_values("top3_rank", key=lambda s: s.astype(int)).to_dict("records"):
        side = "Yes" if str(r["side"]) == "YES" else str(r["side"]).title()
        mk = lab.get(str(r["market"]), str(r["market"]))
        lines.append(f"{r['top3_rank']}. {escape(str(r['home_team']))} v {escape(str(r['away_team']))} - "
                     f"{mk} {side} @ {float(r['odds']):.2f} · {r['slot']} · {r['cell_status']} "
                     f"({edge_text(pd.to_numeric(r.get('edge_calibrated'), errors='coerce') * 100)})")
    for (mk, sd), g in rule.groupby(["market", "side"], sort=True):
        side = "Yes" if str(sd) == "YES" else str(sd).title()
        lines.append(f"📏 League rule: {lab.get(str(mk), str(mk))} {side}, every match "
                     f"({g['cell_status'].iloc[0]}, outside the 3)")
        for r in g.sort_values("top3_rank", key=lambda s: s.astype(int)).to_dict("records"):
            lines.append(f"   • {escape(str(r['home_team']))} v {escape(str(r['away_team']))} @ {float(r['odds']):.2f}")
    lines.append("<i>Paper track. Each tip cleared its league's own owner-approved bar; ranked by cautious "
                 "calibrated value (T3). Fewer than 3 = fewer cleared the bar.</i>")
    return "\n".join(lines)


def notify(out_dir: Path | None = None, env: dict | None = None, sender=None) -> dict:
    """Send each frozen league-week's picks once. Needs TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (or `sender`)."""
    if not enabled(env):
        return {"enabled": False}
    od = Path(out_dir) if out_dir else OUT
    e = env if env is not None else os.environ
    if sender is None:
        token, chat = e.get("TELEGRAM_TOKEN", ""), e.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            return {"enabled": True, "sent": 0, "status": "NO_TELEGRAM_CREDENTIALS"}
        sys.path.insert(0, str(ROOT))
        from telegram_bot.notifier import _send

        def sender(text):
            return _send(token, chat, text)
    picks = picks_store(od).read()
    nf = od / NOTIFIED_NAME
    sent = set(_read_json(nf).get("weeks", []))
    n = 0
    for week, g in (picks.groupby("league_week_id", sort=True) if len(picks) else []):
        if week in sent:
            continue
        if sender(message(g)):
            sent.add(week)
            n += 1
    if n:
        LW._write_json(nf, {"weeks": sorted(sent)})
    return {"enabled": True, "sent": n}


def _cli(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cmd = argv[0] if argv else "run"
    fn = {"run": run, "settle": settle, "notify": notify}.get(cmd)
    if fn is not None:
        print(json.dumps(fn(), default=str))
        return 0
    if cmd == "report":
        rep = write_report(lead=freeze_lead_min())
        print(json.dumps({k: v for k, v in rep.items() if k != "leagues"}, default=str))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
