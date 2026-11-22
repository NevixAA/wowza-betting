"""TOP-3 PER LEAGUE PER GAMEWEEK - production tip selection (owner 2026-11-08). Behind a flag, default OFF.

OWNER 2026-11-11 ("Clean rebuild"; TOP-3 "Paper track"). Supersedes the tier / EV text below where they differ:
  * POOL: tips from the retrained models that clear their cell's ONE owner-approved bar (bars_opt cell,
    valuable_th; src/bars_opt/serve.py). A league x market without its own applied CELL has no tip
    (NO_LEAGUE_BAR) - the legacy three-tier entries are retired. No tier is computed, shown or stored.
  * RANK: the LIVE RULE (src.research.top3_ranking.LIVE_RULE; protocol top3-ranking-paper/6, owner 2026-11-15:
    R2_CAL_EV - was R3_CAL_EV_LCB under v1-v5), computed by THE research code (src/research/top3_ranking.rank_key, imported, not
    forked): calibrated value from the cell's own calibration (its FROZEN statistics: the research freeze record, scripts/top3_paper_freeze.py).
    The old EV ranking is DROPPED entirely - not a
    key, not a tie-break, not a reference track. A cell without a live-rule statistic ranks after every cell with
    one, in kickoff order. Ties: kickoff, then fixture.
  * SLOTS: TOP first; TIER2 (a tip in its cell's VALUABLE..MARKSMAN band where that cell's TIER2 is on,
    study 12.2) only fills slots left empty. At most 3, never padded, the direction rule unchanged.
  * Each pick carries its cell status (PROMISING / UNPROVEN) and its slot (TOP / TIER2).
  * FAIL CLOSED: no frozen live-rule statistics -> nothing is frozen or sent (status RANK_STATS_MISSING).
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
    python -m src.top3_league reseed    # predict.yml, before the pipeline: the one-shot re-seed (owner 2026-11-20)

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

TOP-3 STAKE (owner 2026-11-17, PROPOSED in the go-live rehearsal): "from Nov 21 every TOP-3 tip carries a real
stake of 0.5% of bankroll, flat; I place the bets manually; exec and Cloudbet stay off". Read from
config/bankroll_policy.json -> top3_league.stake ({"rule": "FLAT", "fraction": 0.005, "execution": "MANUAL"}),
validated by top3_stake() (no default: absent / invalid = no stake, every line keeps its cell's label). When set,
every TOP-3 / TIER2 pick (selection_mode TOP3_LEAGUE_GW) records stake_effect "REAL <f>% ... MANUAL" and
track_is_real True, and its Telegram line shows the stake. The stake is NEVER written to output/stake_plan.csv:
that file feeds Cloudbet and exec, so a TOP-3 stake can only be placed by hand. League-rule (own list) lines are
not TOP-3 tips: their stake follows their cell, as before.

SHOW ALL, STAKE BY PROOF (owner 2026-11-22, "3 per league, stake by proof"; tier mode only). Supersedes the
"never padded / fewer when fewer clear the bar" text above for the P11 tier mode:
  * Every league-week shows T1 / T2 / T3, always 3 when it has at least 3 ELIGIBLE candidates. Eligible = a real
    two-sided bet-time price (NO_TWO_SIDED_PRICE stays out), inside its cell's own odds range where it has a cell
    (OUTSIDE_ODDS_RANGE stays out; a league x market without a cell has no range - no global value), first seen
    before the freeze and after the TOP-3 start, kickoff after the issue, the direction rule and the coherence gate
    (src/tip_coherence). A candidate without its league's bar (NO_LEAGUE_BAR / BAR_NOT_OWNER_APPLIED /
    LEAGUE_BAR_INVALID) or below it (BELOW_LEAGUE_BAR) is now eligible - shown, never staked. Fewer than 3 only
    when fewer eligible candidates exist; the message says so.
  * VALUE = p x odds - 1 with p the CALIBRATED probability (fair + lam (p_used - fair), the cell's frozen lam in
    config/top3_tiers_p11.json) where the candidate has a cell AND a calibration cell (value_basis CALIBRATED),
    else the model's own probability (lam = 1: its full edge vs the de-vigged fair price, value_basis MODEL_EDGE).
  * T1 = highest value. T2 = highest p of the rest. T3 = of the rest, the longest odds with a positive edge
    (p - fair > 0); none left -> the next best by value. A no-stake candidate is shown only when its model BEATS THE
    FAIR PRICE (owner 2026-11-22 "Beat the fair price": p - fair > 0, p calibrated where available - it need not
    beat the margin); at or below fair it is never shown (NOT_ABOVE_FAIR_PRICE). A tier with no such pick left
    stays EMPTY (EMPTY_SLOT_NO_PICK_ABOVE_FAIR) and the message says so. Each unstaked pick is labelled with its
    edge vs the fair price ("model +2.1pp vs fair price, no stake: below bar").
  * STAKED PICKS ARE PROTECTED (owner 2026-11-22): stakeable picks take their slots before the others. With 3 or
    more, the three tiers are chosen by their own rule among the stakeable picks only (the approved P11 rule on the
    approved picks). With 1-2, each (highest value first) goes to the tier it matches best (the tier whose order ranks
    it highest among all candidates; tie T1 > T2 > T3; a tier already held -> its next best free tier). The other
    tiers are then filled by their own rule. A stakeable pick's value is the one its league's approved gate measured (CAL: calibrated; BAR: the cell's
    p_used x odds - 1), so it is never a negative-value pick under the approved rule.
  * STAKE BY PROOF: only a pick that clears its league's own approved bar (qualify(), reason "") AND the league's
    approved P11 tier gate (CAL: calibrated value > 0; BAR: p_used x odds - 1 > 0) carries the flat TOP-3 stake
    (config/bankroll_policy.json top3_league.stake, unchanged). Every other pick: stake 0 and a stake_reason
    (NO_STAKE_BELOW_BAR / NO_STAKE_NO_APPROVED_BAR / NO_STAKE_NO_VALUE ...), labelled in Telegram and on the
    dashboard ("No stake: below bar"). All picks are frozen at first sight in the same ledger (columns stake,
    stake_reason, value_basis), settled the same way and reported apart (staked vs no stake).

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
VERSION = "TOP3-2"                         # owner 2026-11-11: cell bar + live-rule ranking, no tiers
from src.research.top3_ranking import LIVE_RULE, in_pool  # noqa: E402  the research rule, not forked
from src.research import top3_paper as TP3  # noqa: E402  the v6 re-measurement rule, shared with the paper tracks
SLOT_TOP, SLOT_TIER2 = "TOP", "TIER2"
TOP_N = 3
# P11 (owner 2026-11-18 "best optimized bars and T1, T2, T3"; PROPOSAL, config/p11_bars_tiers_protocol.json): the
# three named tiers replace the R2 top-3 ONLY when config/top3_tiers_p11.json carries the owner's approval
# (load_tiers). Absent / not approved -> the v6 R2 TOP-3 above, unchanged.
TIERS_FILE = Path("config") / "top3_tiers_p11.json"
TIER_NAMES = ("T1_BEST_VALUE", "T2_SAFEST", "T3_RISKY_VALUE")
R_TIER_GATE = "TIER_NO_VALUE"              # qualifies on its bar but fails the league's tier gate (CAL / BAR)
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
R_PRE_START = "FIRST_SEEN_BEFORE_TOP3_START"   # go-live rehearsal 2026-11-15: TOP-3 starts at go-live
R_STARTED = "STARTED_BEFORE_ISSUE"
R_DIRECTION = "DIRECTION_CONFLICT"
R_MISMATCH = "MISMATCH_WITH_SENT_TIP"          # src/tip_coherence (owner 2026-11-19): vs a tip already sent
R_RANK = "RANKED_BELOW_TOP3"
# owner 2026-11-22 "Beat the fair price": a no-stake candidate is shown only when its model beats the de-vigged fair
# price (p - fair > 0, p calibrated where available); at or below fair it is never shown
R_NOT_ABOVE_FAIR = "NOT_ABOVE_FAIR_PRICE"
R_EMPTY_SLOT = "EMPTY_SLOT_NO_PICK_ABOVE_FAIR"  # a tier left empty: no other priced, coherent pick above the fair price

# owner 2026-11-22 "3 per league, stake by proof": the stake decision of every TOP-3 pick (ledger column stake_reason)
STAKED = "STAKED"
NS_BELOW = "NO_STAKE_BELOW_BAR"                 # priced, eligible, below its league's own approved bar
NS_NO_BAR = "NO_STAKE_NO_APPROVED_BAR"          # its league x market x side has no owner-approved bar
NS_NO_VALUE = "NO_STAKE_NO_VALUE"               # cleared the bar, failed the league's approved P11 tier gate
NS_PAPER_LEAGUE = "NO_STAKE_PAPER_LEAGUE"       # config.PAPER_LEAGUES: ledgered and graded, never sent
NS_NOT_SET = "NO_STAKE_NOT_CONFIGURED"          # no valid top3_league.stake in the bankroll policy
NS_LEAGUE_RULE = "LEAGUE_RULE"                  # the league's own line: its stake follows its cell, as before
#: qualify() reason -> the no-stake reason of a candidate that is still SHOWN (tier mode)
SHOWABLE = {R_NO_BAR: NS_NO_BAR, R_NOT_APPLIED: NS_NO_BAR, R_BAD_BAR: NS_NO_BAR, R_BELOW: NS_BELOW}
#: (NS_NOT_SET has no label: with no TOP-3 stake configured no line carries a stake label, owner 2026-11-17)
NO_STAKE_LABEL = {NS_BELOW: "No stake: below bar", NS_NO_BAR: "No stake: no approved bar",
                  NS_NO_VALUE: "No stake: no value at the bar", NS_PAPER_LEAGUE: "No stake: league not staked"}
BASIS_CAL, BASIS_MODEL = "CALIBRATED", "MODEL_EDGE"

PICK_KEY = ["selection_mode", "league_week_id", "top3_rank"]
PICK_FROZEN = [
    "pick_id", "selection_mode", "version", "league_week_id", "top3_rank", "week_lock_ts", "freeze_ts",
    "issued_at", "league", "model_type", "season", "fixture_id", "match_date", "home_team", "away_team",
    "kickoff_utc", "market", "side", "odds", "opposite_side_odds", "p_model_raw", "p_market_fair", "edge",
    "slot", "cell_status", "p_used", "rank_key", "rank_has_stat", "edge_calibrated", "bar_valuable", "bar_source",
    "bar_status",
    "bar_evidence",
    "ev_point", "ev_lower_bound", "p_lower_bound", "price_quality_state", "market_source", "n_books",
    "first_seen_at", "prediction_engine_version", "selection_engine_version", "policy_switch",
    "track_is_real", "stake_effect",
    # owner 2026-11-22 "3 per league, stake by proof" (appended; older rows migrated by migrate_ledger())
    "stake", "stake_reason", "value_basis",
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


START_CONFIG = Path("config") / "top3_paper_tracks.json"


def top3_start(root: Path | None = None) -> pd.Timestamp | None:
    """TOP-3 starts at go-live (owner 2026-11-21): the start_utc of config/top3_paper_tracks.json, the one
    start the TOP-3 protocol pins (the paper tracks gate on it too). None when unset or unreadable (no
    default): run() then freezes nothing (status TOP3_START_NOT_SET)."""
    v = _read_json((Path(root) if root else ROOT) / START_CONFIG).get("start_utc")
    try:
        ts = pd.Timestamp(v) if v else None
    except (TypeError, ValueError):
        return None
    if ts is None or pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


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
            # the cell's own probability, also for a row below its bar (owner 2026-11-22: shown, never staked)
            pu = _num(res["p_used"].iloc[0])
            if not t:
                why = R_BELOW if res["reason"].iloc[0] == BS.R_BELOW else str(res["reason"].iloc[0])
            else:
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


# ── live-rule ranking (owner 2026-11-11; protocol top3-ranking-paper/1, the research code) ─────────────────────
def rank_keys(q: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """q + rank_key / rank_has_stat from src.research.top3_ranking.rank_key (rule LIVE_RULE). Pure."""
    from src.research import top3_ranking as TR
    if q is None or q.empty:
        return q
    b = q.assign(odds=pd.to_numeric(q["odds"], errors="coerce"), p_used=pd.to_numeric(q["p_used"], errors="coerce"),
                 fair=pd.to_numeric(q["fair"], errors="coerce"))
    k = TR.rank_key(b[TR.CELL + ["fair", "p_used", "odds"]].reset_index(drop=True),
                    stats if stats is not None else pd.DataFrame(columns=TR.CELL + TR.STAT_COLS), LIVE_RULE)
    out = q.copy()
    out["rank_key"] = k["key"].to_numpy()
    out["rank_has_stat"] = k["has_stat"].to_numpy()
    # the CALIBRATED edge shown for the pick (owner 2026-11-11, src/calibrated_edge.py; same frozen statistics)
    from src import calibrated_edge as CE
    out["edge_calibrated"] = CE.calibrated(b.reset_index(drop=True), stats).to_numpy() \
        if stats is not None else np.nan
    return out


def load_rank_stats(root: Path | None = None) -> pd.DataFrame | None:
    """The FROZEN per-cell live-rule statistics: THE research freeze record (scripts/top3_paper_freeze.py, named by
    config/top3_paper_tracks.json -> freeze_ledger, hash-checked by src.research.top3_paper.load_freeze) - one
    freeze for the live TOP-3 and its paper tracks, never a second builder. None when not frozen / not valid."""
    from src.research import top3_paper as TP3
    try:
        cfg = TP3.load_config(root)
        stats, _rec, _sha, why = TP3.load_freeze(cfg, root)
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"[top3_league] live-rule statistics unreadable ({type(e).__name__}: {e})")
        return None
    if why:
        log.warning(f"[top3_league] live-rule statistics not usable: {'; '.join(why)[:300]}")
        return None
    # protocol v6 re-measurement (owner 2026-11-15): which frame the statistics were frozen from, and the all-data
    # models' engine versions - a candidate of those models is ranked only by statistics from their own frame
    stats.attrs["frozen_frame"] = _rec.get("frame_hash")
    stats.attrs["all_data"] = cfg.get("all_data_models")
    return stats


def _direction(market: str, side: str) -> str | None:
    from src.direction_guard import goal_direction
    return goal_direction(MARKETS[str(market)][1], side)


def _gate_candidate(r):
    """A TOP-3 candidate row as a src/tip_coherence Candidate (None when its market has no event)."""
    from src.tip_coherence import Candidate, fixture_id, parse_event
    ev = parse_event(r["market"], r["side"])
    if ev is None:
        return None
    return Candidate(fixture_id(r["match_date"], r["home_team"], r["away_team"]), ev, "TOP-3",
                     uid=f"top3|{r.get('fixture_id')}|{r['market']}|{r['side']}")


def _gate_blocks(gate, r) -> str:
    """'' or why the fixture coherence gate refuses this candidate (a MISMATCH with a tip already SENT by any
    sender - HT, O/U, side markets, an earlier TOP-3 week). No gate -> ''."""
    if not gate:
        return ""
    c = _gate_candidate(r)
    return gate.check(c) if c is not None else ""


def _gate_take(gate, r) -> None:
    if gate:
        c = _gate_candidate(r)
        if c is not None:
            gate.commit(c)


def load_tiers(root: Path | None = None) -> dict | None:
    """The P11 tier config when the OWNER approved it (status starting "APPROVED" and owner_approved_at set),
    else None (the v6 R2 TOP-3 stays). Never raises."""
    try:
        cfg = json.loads(((Path(root) if root else ROOT) / TIERS_FILE).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return None
    if not str(cfg.get("status", "")).startswith("APPROVED") or not cfg.get("owner_approved_at"):
        return None
    if not isinstance(cfg.get("leagues"), dict) or not (cfg.get("calibration") or {}).get("cells"):
        return None
    return cfg


def tier_stats(cfg: dict) -> pd.DataFrame:
    """The tier config's frozen calibration (lam, lam_se per cell) in the live-rule statistics shape."""
    from src.research import top3_ranking as TR
    st = pd.DataFrame(cfg["calibration"]["cells"])
    for c in TR.STAT_COLS:
        if c not in st.columns:
            st[c] = np.nan
    st = st[TR.CELL + TR.STAT_COLS]
    st.attrs["frozen_frame"], st.attrs["all_data"] = cfg.get("frame_hash"), None
    return st


def _tier_metrics(r: dict) -> dict:
    """cal_ev / p_cal / ev_lo / ev_raw of one qualified candidate from its frozen lam (NaN = no statistic)."""
    pu, pf, o = _num(r.get("p_used")), _num(r.get("fair")), _num(r.get("odds"))
    lam, se = _num(r.get("lam")), _num(r.get("lam_se"))
    se = 0.0 if math.isnan(se) else se
    p_cal = pf + lam * (pu - pf)
    return {"p_cal": p_cal, "cal_ev": p_cal * o - 1.0, "ev_lo": (pf + (lam - se) * (pu - pf)) * o - 1.0,
            "ev_raw": pu * o - 1.0, "odds": o}


def select_week(q: pd.DataFrame, freeze: pd.Timestamp, issue: pd.Timestamp,
                committed: dict | None = None, stats: pd.DataFrame | None = None,
                start: pd.Timestamp | None = None, tiers: dict | None = None,
                gate=None) -> tuple[pd.DataFrame, dict]:
    """TOP-3 of ONE league-week from qualified candidates `q` (qualify()). Returns (picks with
    top3_rank 1..k, k <= 3, never padded; {reason: count} for every candidate not picked). Pure.
    `start` (the TOP-3 start, top3_start()): a candidate first seen before it was frozen by the pre-go-live
    models and is never a member (FIRST_SEEN_BEFORE_TOP3_START)."""
    committed = committed or {}
    excluded: Counter = Counter()

    def bump(k):
        excluded[k] += 1
    if q is None or q.empty:
        return q.iloc[0:0] if q is not None else pd.DataFrame(), dict(excluded)
    q = rank_keys(q.reset_index(drop=True), stats)
    fs = LW._ts(q["first_seen_at"])
    ko = LW._ts(q["kickoff_utc"])
    if "league_wide" not in q.columns:
        q["league_wide"] = False
    pool, own, dirs, nostake = [], [], {}, {}
    for i, r in enumerate(q.to_dict("records")):
        if not (pd.notna(fs.iloc[i]) and fs.iloc[i] < freeze):
            bump(R_LATE)
        elif start is not None and fs.iloc[i] < start:
            bump(R_PRE_START)
        elif (tiers is not None and r.get("reason") in SHOWABLE and not bool(r.get("league_wide"))
              and (stats is None or TP3.stats_valid_for_engine(r.get("prediction_engine_version"),
                                                              stats.attrs.get("all_data"),
                                                              stats.attrs.get("frozen_frame")))):
            # owner 2026-11-22 "3 per league, stake by proof": priced but without / below its bar -> SHOWN, no stake
            nostake[i] = SHOWABLE[r["reason"]]
            pool.append((1, (1, 0.0), 0, "", i))
        elif r.get("reason"):
            bump(r["reason"])
        elif stats is not None and not TP3.stats_valid_for_engine(
                r.get("prediction_engine_version"), stats.attrs.get("all_data"), stats.attrs.get("frozen_frame")):
            bump(TP3.R_NOT_REMEASURED)          # all-data models before their re-measurement: refused (v6)
        elif bool(r.get("league_wide")):
            own.append(i)                       # the league's OWN line, outside the TOP-3 slots
        else:
            # owner 2026-11-11: TOP before TIER2 (TIER2 only fills empty slots); then live-rule (cells with a
            # statistic first, higher key first); then kickoff, then fixture. No EV anywhere.
            key = _num(r.get("rank_key"))
            has = bool(r.get("rank_has_stat")) and not math.isnan(key)
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
        if _gate_blocks(gate, r):
            bump(R_MISMATCH)
            continue
        if dr:
            dirs[fx] = dr
        _gate_take(gate, r)
        own_taken.append(i)
    # 2) TOP-3 / TIER2 from MODEL cells only; a model tip opposite to an own-list tip yields
    taken = []
    if tiers is not None:
        return _select_tiers(q, pool, own_taken, dirs, committed, issue, ko, stats, tiers, excluded, coherence=gate,
                             nostake=nostake)
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
        if _gate_blocks(gate, r):
            bump(R_MISMATCH)
            continue
        if dr:
            dirs[fx] = dr
        _gate_take(gate, r)
        taken.append(i)
    picks = q.iloc[taken].copy()
    picks["top3_rank"] = range(1, len(picks) + 1)
    picks["list"] = SELECTION_MODE
    ol = q.iloc[own_taken].copy()
    ol["top3_rank"] = range(1, len(ol) + 1)
    ol["slot"] = OWN_LIST_MODE
    ol["list"] = OWN_LIST_MODE
    return pd.concat([picks, ol], ignore_index=True), dict(excluded)


def _select_tiers(q, pool, own_taken, dirs, committed, issue, ko, stats, tiers, excluded, coherence=None,
                  nostake=None):
    """P11 tiers, owner 2026-11-22 "3 per league, stake by proof". The pool = the bar-cleared candidates plus the
    priced candidates without / below their league's bar (`nostake` {index: no-stake reason}).
      T1 = highest value, T2 = highest p of the rest, T3 = of the rest the longest odds with a positive edge
      (p - fair > 0), none left -> the next best by value; a no-stake pick must beat the fair price, else empty.
      Stakeable picks take their slots first, each in the tier it matches best (module docstring).
    value = p x odds - 1, p = the calibrated probability (cell + calibration cell: CALIBRATED) or the model's own
    (MODEL_EDGE). Stake by proof: a bar-cleared pick that also passes the league's APPROVED P11 tier gate (CAL:
    cal_ev > 0; BAR: p_used * odds - 1 > 0, both on the frozen calibration, unchanged) is STAKED; every other pick
    carries its no-stake reason. 3 when 3 eligible exist, never a candidate without a price; distinct fixture x
    market, started-before-issue, the direction rule and the coherence gate as the R2 path."""
    from src.research import top3_ranking as TR
    nostake = nostake or {}
    idx = [i for *_k, i in pool]
    lam = {}
    if idx and stats is not None and len(stats):
        st = stats.drop_duplicates(TR.CELL).set_index(TR.CELL)
        for i in idx:
            k = tuple(str(q.iloc[i][c]) for c in TR.CELL)
            if k in st.index:
                lam[i] = (st.loc[k, "lam"], st.loc[k, "lam_se"])
    met = {}
    for i in idx:
        r = q.iloc[i].to_dict()
        lm, ls = lam.get(i, (np.nan, np.nan))
        # the stake gate: the league's approved P11 gate, exactly as before (frozen lam; NaN = no statistic)
        x = _tier_metrics(r | {"lam": lm, "lam_se": ls})
        prm = (tiers.get("leagues") or {}).get(f"{r.get('track')}|{r.get('league')}") or {}
        gate = prm.get("gate", "CAL")
        gate_ok = (bool(np.isfinite(x["cal_ev"]) and np.isfinite(x["p_cal"]))
                   and (x["cal_ev"] > 0 if gate == "CAL" else x["ev_raw"] > 0))
        reason = nostake.get(i, "")
        if not reason and not gate_ok:
            reason = NS_NO_VALUE
            excluded[R_TIER_GATE] += 1          # cleared its bar, failed the approved tier gate: never staked
        # the display value: calibrated where the cell has a calibration, else the model's own edge (lam = 1)
        if np.isfinite(x["cal_ev"]) and np.isfinite(x["p_cal"]):
            v = x | {"basis": BASIS_CAL}
        else:
            v = _tier_metrics(r | {"p_used": _num(r.get("p_model_raw")), "lam": 1.0, "lam_se": 0.0}) \
                | {"basis": BASIS_MODEL}
        if not (np.isfinite(v["cal_ev"]) and np.isfinite(v["p_cal"]) and v["odds"] > 1):
            excluded[R_NO_PRICE] += 1            # never a pick without a usable price
            continue
        staked = not reason
        # owner 2026-11-22 (2nd decision): NEVER a pick without positive value. A stakeable pick passed its league's
        # APPROVED value gate (CAL: calibrated value > 0; BAR: the cell's p_used x odds - 1 > 0), which is its value.
        if not staked and not v["p_cal"] - _num(r.get("fair")) > 0:
            excluded[R_NOT_ABOVE_FAIR] += 1      # owner 2026-11-22 "Beat the fair price": at / below fair, never shown
            continue
        met[i] = v | {"stake_reason": STAKED if staked else reason, "fair": _num(r.get("fair"))}
    used, taken, names, blocked = set(), [], [], {}

    def ok(i) -> bool:
        """Can candidate i take a slot now? (not taken, distinct fixture x market, kickoff after the issue, the
        direction rule, the coherence gate). Records why not."""
        r = q.iloc[i]
        if i in taken or (str(r["fixture_id"]), str(r["market"])) in used:
            return False
        if not (pd.notna(ko.iloc[i]) and ko.iloc[i] > issue):
            blocked[i] = R_STARTED
            return False
        fx = (str(r["match_date"])[:10], str(r["home_team"]), str(r["away_team"]))
        dr = _direction(r["market"], r["side"])
        stood = dirs.get(fx) or (committed.get(fx) or {}).get("direction")
        if dr and stood and dr != stood:
            blocked[i] = R_DIRECTION
            return False
        if _gate_blocks(coherence, r):
            blocked[i] = R_MISMATCH
            return False
        return True

    def take(i, name) -> None:
        r = q.iloc[i]
        fx = (str(r["match_date"])[:10], str(r["home_team"]), str(r["away_team"]))
        dr = _direction(r["market"], r["side"])
        if dr:
            dirs[fx] = dr
        _gate_take(coherence, r)
        used.add((str(r["fixture_id"]), str(r["market"])))
        taken.append(i)
        names.append(name)

    cand = list(met)
    kick = {i: (ko.iloc[i].value if pd.notna(ko.iloc[i]) else float("inf"), str(q.iloc[i]["fixture_id"])) for i in cand}
    # the three tier orders over every eligible candidate (T3: longest odds among positive edge, then by value)
    by_value = sorted(cand, key=lambda i: (-met[i]["cal_ev"], kick[i]))
    by_p = sorted(cand, key=lambda i: (-met[i]["p_cal"], kick[i]))
    pos = [i for i in cand if met[i]["p_cal"] - met[i]["fair"] > 0]
    by_t3 = sorted(pos, key=lambda i: (-met[i]["odds"], -met[i]["cal_ev"], kick[i])) + \
        [i for i in by_value if i not in pos]
    orders = (by_value, by_p, by_t3)
    # 1) PROTECT STAKED PICKS (owner 2026-11-22): every stakeable pick keeps a slot.
    #    * 3 or more stakeable: the three tiers are chosen by their own rule AMONG the stakeable picks only (T1 best
    #      value, T2 highest p of the rest, T3 longest odds of the rest) - the approved P11 rule on the approved
    #      picks; a 4th+ stakeable pick ranks below (only 3 slots).
    #    * fewer than 3: each (highest value first) goes to the tier it matches best = the tier whose order ranks it
    #      highest among ALL candidates (tie: T1 before T2 before T3); a tier already held -> its next best free tier.
    free = list(TIER_NAMES)
    stk = [j for j in by_value if met[j]["stake_reason"] == STAKED]
    if len(stk) >= len(TIER_NAMES):
        for t, name in enumerate(TIER_NAMES):
            for i in orders[t]:
                if met[i]["stake_reason"] == STAKED and ok(i):
                    take(i, name)
                    free.remove(name)
                    break
    else:
        for i in stk:
            if not ok(i):
                continue
            pref = sorted(range(len(TIER_NAMES)),
                          key=lambda t: (orders[t].index(i) if i in orders[t] else len(cand), t))
            tier = next(TIER_NAMES[t] for t in pref if TIER_NAMES[t] in free)
            take(i, tier)
            free.remove(tier)
    # 2) the remaining tiers by their own rule, T1 / T2 / T3, from the candidates left (all above the fair price)
    for t, name in enumerate(TIER_NAMES):
        if name not in free:
            continue
        for i in orders[t]:
            if ok(i):
                take(i, name)
                free.remove(name)
                break
    for name in free:
        excluded[R_EMPTY_SLOT] += 1              # no other priced, coherent pick above fair: the slot stays empty
    for i in cand:
        if i not in taken:
            # one reason per candidate left out: why it could not take a slot (started / direction / coherence),
            # else it simply ranked below the three
            excluded[blocked.get(i, R_RANK)] += 1
    order = sorted(range(len(taken)), key=lambda k: TIER_NAMES.index(names[k]))
    taken, names = [taken[k] for k in order], [names[k] for k in order]
    picks = q.iloc[taken].copy()
    picks["slot"] = names
    picks["top3_rank"] = [TIER_NAMES.index(n) + 1 for n in names]
    # the calibrated edge only where it IS calibrated (never the raw edge under that name)
    picks["edge_calibrated"] = [met[i]["p_cal"] - met[i]["fair"] if met[i]["basis"] == BASIS_CAL else np.nan
                                for i in taken]
    picks["rank_key"] = [met[i]["cal_ev"] for i in taken]
    picks["rank_has_stat"] = [met[i]["basis"] == BASIS_CAL for i in taken]
    picks["value_basis"] = [met[i]["basis"] for i in taken]
    picks["stake_reason"] = [met[i]["stake_reason"] for i in taken]
    picks["list"] = SELECTION_MODE
    ol = q.iloc[own_taken].copy()
    ol["top3_rank"] = range(len(TIER_NAMES) + 1, len(TIER_NAMES) + 1 + len(ol))
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


STAKE_RULES = ("FLAT",)
STAKE_EXECUTION = ("MANUAL",)


def top3_stake(policy: dict | None) -> float | None:
    """The flat TOP-3 stake as a bankroll fraction (owner 2026-11-17), or None = no TOP-3 stake.

    Only config/bankroll_policy.json -> top3_league.stake with rule FLAT, execution MANUAL and a finite fraction in
    (0, HARD single_bet_cap] counts. Anything else (absent, typo, NaN, above the hard cap, an execution mode that
    is not MANUAL) is no stake: never a default, never a guess."""
    st = ((policy or {}).get("top3_league") or {}).get("stake") or {}
    if str(st.get("rule", "")).strip().upper() not in STAKE_RULES:
        return None
    if str(st.get("execution", "")).strip().upper() not in STAKE_EXECUTION:
        return None
    try:
        f = float(st.get("fraction"))
    except (TypeError, ValueError):
        return None
    from src.bankroll_policy import HARD
    return f if math.isfinite(f) and 0.0 < f <= float(HARD["single_bet_cap"]) else None


def stake_text(fraction: float) -> str:
    """'0.5%' for 0.005 (no trailing zeros)."""
    return f"{fraction * 100:.2f}".rstrip("0").rstrip(".") + "%"


def stake_effect_text(fraction: float | None) -> str:
    if fraction is None:
        return "NONE (TOP-3 never sizes a stake; stake plan unchanged)"
    return (f"REAL {stake_text(fraction)} of bankroll, flat (owner 2026-11-17) - placed MANUALLY by the owner; "
            "never in output/stake_plan.csv, so no automatic execution (exec / Cloudbet)")


def stake_reason_of(r: dict, flat: float | None) -> str:
    """STAKED or why a pick carries no stake (owner 2026-11-22). A league-rule line keeps LEAGUE_RULE; a paper
    league is never staked; no valid flat stake configured -> NO_STAKE_NOT_CONFIGURED; else the pick's own
    stake_reason from the tier selection (absent = the R2 path, where every TOP-3 pick cleared its bar)."""
    if str(r.get("list") or r.get("selection_mode") or SELECTION_MODE) == OWN_LIST_MODE:
        return NS_LEAGUE_RULE
    if config.is_paper_league(r.get("league")):
        return NS_PAPER_LEAGUE
    if flat is None:
        return NS_NOT_SET
    own = str(r.get("stake_reason") or "").strip()
    return STAKED if own in ("", "nan", STAKED) else own


def freeze_due(cands: pd.DataFrame, weeks_done: set, ctx: dict, now: pd.Timestamp, lead_min: float,
               committed: dict | None = None, policy: dict | None = None,
               real_checker=None, stats: pd.DataFrame | None = None,
               start: pd.Timestamp | None = None, tiers: dict | None = None,
               gate=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(new pick rows, new week rows) for every league-week whose FREEZE has passed and that is not
    decided yet. A week with 0 picks is still recorded (decided once, with its reasons). Pure apart
    from the injected policy / checker.
    `start` = the TOP-3 start (top3_start(); run() always passes it, and freezes nothing without it): a
    league-week whose FREEZE is before it is never decided - TOP-3 starts at go-live, so the first run
    after the switch-on cannot back-fill past weeks with picks issued after their freeze - and a candidate
    first seen before it is never a member (go-live rehearsal 2026-11-15, 44 past weeks back to W43)."""
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
        if now < freeze or (start is not None and freeze < start):
            continue
        q = qualify(g, ctx)
        picks, excl = select_week(q, freeze, now, committed, stats, start, tiers, gate)
        league = str(g["league"].iloc[0])
        n_q = int((q["reason"] == "").sum()) if len(q) else 0
        n_top = int((picks["list"] == SELECTION_MODE).sum()) if len(picks) else 0
        weeks_out.append({"selection_mode": SELECTION_MODE, "version": VERSION, "league_week_id": week,
                          "league": league, "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze),
                          "issued_at": stamp, "kickoff_utc": LW._stamp(lock), "n_candidates": int(len(g)),
                          "n_qualifying": n_q, "n_picks": n_top,
                          "excluded": json.dumps(excl, sort_keys=True), "frozen_at": stamp})
        sw = policy_switch(policy, league)
        flat = top3_stake(policy)
        for r in picks.to_dict("records"):
            mk = MARKETS[str(r["market"])][1]
            # a league-rule line is not TOP-3; a PAPER league's pick (config.PAPER_LEAGUES, e.g. USA MLS) is never
            # sent (notify drops it), so the owner cannot place it and it never carries a stake
            # owner 2026-11-22 "stake by proof": only a pick whose stake_reason is STAKED (cleared its league's own
            # approved bar and the approved tier gate) carries the flat stake; the R2 path (no tiers) keeps its rule
            sr = stake_reason_of(r, flat)
            staked = sr == STAKED
            picks_out.append({
                **{k: r.get(k, "") for k in PICK_FROZEN if k in r},
                "pick_id": LW.research_id(VERSION, week, str(r["market"]), str(r["fixture_id"]), str(r["side"])),
                "selection_mode": r["list"], "version": VERSION, "league_week_id": week,
                "top3_rank": int(r["top3_rank"]), "week_lock_ts": LW._stamp(lock), "freeze_ts": LW._stamp(freeze),
                "issued_at": stamp, "edge": LW._r(r["edge"]), "slot": r["slot"], "cell_status": r["cell_status"],
                "p_used": LW._r(r["p_used"]), "rank_key": LW._r(r["rank_key"]), "rank_has_stat": bool(r["rank_has_stat"]),
                "edge_calibrated": LW._r(r["edge_calibrated"]),
                "bar_valuable": LW._r(r["bar_valuable"]), "bar_source": BAR_SOURCE,
                "bar_status": r["bar_status"], "bar_evidence": r["bar_evidence"],
                "policy_switch": sw,
                "track_is_real": True if staked else bool(real_checker(r.get("model_type"), mk, r.get("side"),
                                                                       r.get("league"))),
                "stake_effect": stake_effect_text(flat if staked else None),
                "stake": LW._r(flat if staked else 0.0), "stake_reason": sr,
                "value_basis": str(r.get("value_basis") or "") if str(r["list"]) == SELECTION_MODE else "",
            })
    return (pd.DataFrame(picks_out).reindex(columns=PICK_COLS) if picks_out else pd.DataFrame(columns=PICK_COLS),
            pd.DataFrame(weeks_out).reindex(columns=WEEK_COLS) if weeks_out else pd.DataFrame(columns=WEEK_COLS))


# ── stores ──────────────────────────────────────────────────────────────────────────────────────
def candidate_store(od: Path) -> FirstSightStore:
    return LW.candidate_store(Path(od) / CANDIDATES_NAME)


def picks_store(od: Path) -> FirstSightStore:
    return FirstSightStore(Path(od) / PICKS_NAME, key=PICK_KEY, columns=PICK_COLS, observed_col="issued_at",
                           kickoff_col="kickoff_utc", frozen_cols=PICK_FROZEN, digest=True)


#: columns appended to the ledger on 2026-11-22 (owner "3 per league, stake by proof")
STAKE_COLS = ("stake", "stake_reason", "value_basis")


def _legacy_stake(effect: str) -> tuple[str, str]:
    """(stake, stake_reason) of a pick written before the stake columns, read from its own frozen stake_effect
    ("REAL 0.5% ..." -> 0.005 STAKED; "NONE ..." -> 0 with no reason recorded). Never a guess."""
    import re
    m = re.match(r"REAL ([0-9]+(?:\.[0-9]+)?)%", str(effect or ""))
    if m:
        return LW._r(float(m.group(1)) / 100.0), STAKED
    return (LW._r(0.0), "") if str(effect or "").startswith("NONE") else ("", "")


def migrate_ledger(od: Path) -> dict:
    """One-shot, idempotent: give a ledger written before 2026-11-22 the stake columns. The existing frozen columns
    keep their exact values (the sidecar digest is checked first and nothing is written when it fails); stake /
    stake_reason come from each row's own stake_effect; value_basis stays blank (not recorded then)."""
    ps = picks_store(od)
    if not ps._has_rows():
        return {"migrated": False, "reason": "no ledger"}
    d = ps.read()
    if all(c in d.columns for c in STAKE_COLS):
        return {"migrated": False, "reason": "up to date"}
    v = ps.verify()
    if v["ok"] is False:
        raise RuntimeError(f"top3 ledger TAMPERED ({v['reason']}) - not migrating")
    from src.research.first_sight import frozen_digest
    old_cols = [c for c in PICK_FROZEN if c in d.columns]
    before = frozen_digest(d, old_cols)
    leg = [(_legacy_stake(e)) for e in d.get("stake_effect", pd.Series("", index=d.index)).astype(str)]
    if "stake" not in d.columns:
        d["stake"] = [a for a, _b in leg]
    if "stake_reason" not in d.columns:
        d["stake_reason"] = [b for _a, b in leg]
    if "value_basis" not in d.columns:
        d["value_basis"] = ""
    out = d.reindex(columns=PICK_COLS).fillna("")
    if frozen_digest(out, old_cols) != before:
        raise RuntimeError("top3 ledger migration would alter a frozen column - refused")
    path = od / PICKS_NAME
    tmp = path.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)
    ps._write_sidecar()
    return {"migrated": True, "rows": int(len(out))}


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
            tiers = load_tiers(root)
            stats = tier_stats(tiers) if tiers is not None else load_rank_stats(root)
            start = top3_start(root)
            if start is None:
                # fail closed: TOP-3 starts at go-live; without its pinned start nothing is frozen or sent
                res["status"] = "TOP3_START_NOT_SET"
            elif stats is None:
                # fail closed (owner 2026-11-11): no frozen live-rule statistics, no ranking, nothing frozen or sent
                res["status"] = "RANK_STATS_MISSING"
            else:
                ws = weeks_store(od)
                done = set(ws.read()["league_week_id"]) if len(ws.read()) else set()
                # the fixture coherence gate (src/tip_coherence, owner 2026-11-19): a pick that mismatches a
                # tip already SENT on its match (HT / O/U / side / an earlier TOP-3 week) is never frozen
                try:
                    from src.tip_coherence import CoherenceGate
                    gate = CoherenceGate.load(out_dir=od)
                except Exception as e:                                 # noqa: BLE001
                    log.warning(f"[top3_league] coherence gate unavailable ({type(e).__name__}: {e})")
                    gate = None
                picks, weeks = freeze_due(cs.read(), done, bars_context(root), now, lead, committed, _policy(),
                                          real_track_checker(), stats, start, tiers, gate)
                migrate_ledger(od)               # owner 2026-11-22: stake columns before the first new append
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
    migrate_ledger(od)
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
        stk = staked_mask(p)
        ost = own[own["settle_status"].astype(str) == "SETTLED"]
        opnl = pd.to_numeric(ost["flat_pnl"], errors="coerce")
        leagues[lg] = {
            "league_rule": {"picks": int(len(own)), "settled": int(len(ost)),
                            "flat_pnl_units": round(float(opnl.sum()), 3) if len(ost) else 0.0},
            "latest_league_week_id": latest,
            "latest_excluded": json.loads(w["excluded"].iloc[-1]) if len(w) and w["excluded"].iloc[-1] else {},
            "latest_picks": [{**{k: r.get(k) for k in ("top3_rank", "home_team", "away_team", "kickoff_utc", "market",
                                                       "side", "odds", "edge_calibrated", "slot", "cell_status",
                                                       "result", "stake", "stake_reason", "value_basis")},
                              "stake_label": pick_stake_label(r)}
                             for r in cur.to_dict("records")],
            # owner 2026-11-22 "stake by proof": staked picks and no-stake picks reported apart
            "staked": _totals(p[stk]), "no_stake": _totals(p[~stk]),
            "weeks_decided": int(len(w)), "picks": int(len(p)), "settled": int(len(st)),
            "won": int((st["won"].astype(str) == "1").sum()),
            "flat_pnl_units": round(float(pnl.sum()), 3) if len(st) else 0.0,
            "roi": round(float(pnl.mean()), 4) if pnl.notna().any() else None}
    return {"generated_at": LW._stamp(now), "selection_mode": SELECTION_MODE, "version": VERSION,
            "freeze_lead_min": lead, "status": "OK" if lead is not None else "FREEZE_LEAD_NOT_SET",
            "note": ("Each league's T1 best value / T2 safest / T3 risky value of the gameweek across O/U 2.5, BTTS, "
                     "Over 1.5, Over 3.5: always 3 when at least 3 priced, coherent candidates exist. Only a pick "
                     "that cleared its league's own approved bar carries the stake; the others are shown with their "
                     "no-stake reason and counted apart (staked / no_stake). League-wide cells are the league's own "
                     "line (league_rule), outside the 3. " + _report_stake_note()),
            "leagues": leagues}


def staked_mask(p: pd.DataFrame) -> pd.Series:
    """Rows whose frozen stake decision is a stake: stake_reason STAKED, or (a row without the column) a REAL
    stake_effect. Read from the ledger, never recomputed."""
    if p is None or not len(p):
        return pd.Series(False, index=p.index if p is not None else None, dtype=bool)
    eff = p.get("stake_effect", pd.Series("", index=p.index)).astype(str).str.startswith("REAL ")
    if "stake_reason" not in p.columns:
        return eff
    sr = p["stake_reason"].astype(str).str.strip()
    return (sr == STAKED) | (sr.isin(["", "nan"]) & eff)


def _totals(p: pd.DataFrame) -> dict:
    st = p[p["settle_status"].astype(str) == "SETTLED"]
    pnl = pd.to_numeric(st["flat_pnl"], errors="coerce")
    return {"picks": int(len(p)), "settled": int(len(st)), "won": int((st["won"].astype(str) == "1").sum()),
            "flat_pnl_units": round(float(pnl.sum()), 3) if len(st) else 0.0,
            "roi": round(float(pnl.mean()), 4) if pnl.notna().any() else None}


def fair_edge_pp(r: dict) -> float:
    """The pick's recorded edge vs the de-vigged fair price, in percentage points: the calibrated edge for a
    CALIBRATED pick, the model's own edge for a MODEL_EDGE pick (NaN when not recorded). Read, never recomputed."""
    col = "edge" if str(r.get("value_basis") or "") == BASIS_MODEL else "edge_calibrated"
    e = _num(r.get(col))
    return e * 100 if not math.isnan(e) else float("nan")


def pick_stake_label(r: dict, flat: float | None = None) -> str:
    """The words a pick shows: '0.5% stake' when its frozen stake > 0; an unstaked pick shows its edge vs the fair
    price and why it has no stake ('model +2.1pp vs fair price, no stake: below bar', owner 2026-11-22); a row
    without a recorded edge keeps the bare label ('No stake: below bar'); '' for a league-rule line."""
    sr = str(r.get("stake_reason") or "").strip()
    if sr in NO_STAKE_LABEL:
        e = fair_edge_pp(r)
        if math.isnan(e):
            return NO_STAKE_LABEL[sr]
        why = NO_STAKE_LABEL[sr].split(": ", 1)[-1]
        return f"model {e:+.1f}pp vs fair price, no stake: {why}"
    f = _num(r.get("stake"))
    if sr == STAKED and f > 0:
        return f"{stake_text(f)} stake"
    if sr == STAKED and flat is not None:
        return f"{stake_text(flat)} stake"
    return ""


def _report_stake_note() -> str:
    flat = top3_stake(_policy())
    if flat is None:
        return "Flat 1u; TOP-3 sizes no stake."
    return (f"A staked TOP-3 tip: {stake_text(flat)} stake of bankroll, flat, placed manually by the owner at the "
            "odds shown on the tip (never auto-executed); P/L here in flat units.")


def write_report(now=None, out_dir: Path | None = None, lead: float | None = None) -> dict:
    od = Path(out_dir) if out_dir else OUT
    rep = report(picks_store(od).read(), weeks_store(od).read(), LW._now(now), lead)
    LW._write_json(od / REPORT_NAME, rep)
    return rep


def _stake_label(real_checker, r: dict, flat: float | None = None) -> str:
    """'0.5% stake' on a TOP-3 / TIER2 pick when the policy's flat TOP-3 stake is set (owner 2026-11-17, "Show stake
    on TOP-3"); '' on everything else - a league-rule line, a paper league's pick, or no TOP-3 stake configured.
    Owner: an ordinary prediction carries NO stake label at all. `real_checker` is kept for the callers."""
    sr = str(r.get("stake_reason") or "").strip()
    if sr and sr != "nan":                       # owner 2026-11-22: the pick's own frozen stake decision
        return pick_stake_label(r, flat)
    if (flat is not None and str(r.get("selection_mode") or SELECTION_MODE) != OWN_LIST_MODE
            and not config.is_paper_league(r.get("league"))):
        return f"{stake_text(flat)} stake"
    return ""


def _with_label(text: str, label: str) -> str:
    return f"{text} · {label}" if label else text


def message(picks: pd.DataFrame, real_checker=None, policy: dict | None = None) -> str:
    """Telegram text for one league-week's picks (HTML). The edge shown is the CALIBRATED one (owner 2026-11-11).
    Each line carries its own frozen stake decision ("0.5% stake" / "No stake: below bar"); `real_checker` defaults
    to the live policy's."""
    from html import escape
    if real_checker is None:
        from src.direction_guard import real_track_checker
        real_checker = real_track_checker()
    flat = top3_stake(policy if policy is not None else _policy())
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
        cell = str(r.get("cell_status") or "").strip()
        lab_r = _stake_label(real_checker, r, flat)
        edge_w = "" if "vs fair price" in lab_r else f" ({_edge_words(r)})"     # the label already carries the edge
        lines.append(f"{r['top3_rank']}. {escape(str(r['home_team']))} v {escape(str(r['away_team']))} - "
                     f"{mk} {side} @ {float(r['odds']):.2f} · {TIER_TEXT.get(str(r['slot']), r['slot'])}"
                     f"{' · ' + escape(cell) if cell and cell != 'nan' else ''}{edge_w}")
        lines[-1] = _with_label(lines[-1], lab_r)
    if picks["slot"].astype(str).isin(TIER_NAMES).any() or (len(picks) == 0 and len(rule) == 0):
        for t, name in enumerate(TIER_NAMES):            # owner 2026-11-22: an empty tier says why
            if name not in set(picks["slot"].astype(str)):
                lines.append(f"{t + 1}. {TIER_TEXT[name]}: none - no other priced, coherent pick above the fair price")
    for (mk, sd), g in rule.groupby(["market", "side"], sort=True):
        side = "Yes" if str(sd) == "YES" else str(sd).title()
        lines.append(f"📏 League rule: {lab.get(str(mk), str(mk))} {side}, every match "
                     f"({g['cell_status'].iloc[0]}, outside the 3)")       # owner 2026-11-17: no stake label
        for r in g.sort_values("top3_rank", key=lambda s: s.astype(int)).to_dict("records"):
            lines.append(f"   • {escape(str(r['home_team']))} v {escape(str(r['away_team']))} @ {float(r['odds']):.2f}")
    n = int(len(picks))
    if picks["slot"].astype(str).isin(TIER_NAMES).any():          # P11 tier mode (owner 2026-11-22 rule)
        head = (f"Staked ({stake_text(flat)} of bankroll, flat, placed manually) only where the tip cleared its "
                "league's own approved bar; the others show why they carry no stake." if flat is not None else "")
        foot = ("T1 best value, T2 safest, T3 risky value."
                + (f" Only {n} priced, coherent pick{'s' if n != 1 else ''} above the fair price this week." if n < TOP_N else ""))
    else:
        head = (f"Every TOP-3 tip: {stake_text(flat)} stake of bankroll, flat, placed manually." if flat is not None
                else "")
        foot = "Each tip cleared its league's own owner-approved bar; ranked by calibrated value (the live rule)."
    lines.append(f"<i>{head + ' ' if head else ''}{foot}</i>")
    return "\n".join(lines)


TIER_TEXT = {"T1_BEST_VALUE": "T1 best value", "T2_SAFEST": "T2 safest", "T3_RISKY_VALUE": "T3 risky value"}


def _edge_words(r) -> str:
    """The calibrated edge where calibrated; a MODEL_EDGE pick shows the model's own edge vs the fair price,
    named as such (never as calibrated)."""
    from src.calibrated_edge import edge_text
    if str(r.get("value_basis") or "") == BASIS_MODEL:
        e = _num(r.get("edge"))
        return f"model edge {e * 100:+.1f} pp, not calibrated" if not math.isnan(e) else "edge: not calibrated"
    return edge_text(pd.to_numeric(r.get("edge_calibrated"), errors="coerce") * 100)


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
    # PAPER leagues (config.PAPER_LEAGUES, e.g. USA MLS) are never sent: ledgered and graded only, as on every
    # other send path (go-live rehearsal 2026-11-15)
    picks = config.drop_paper_leagues(picks_store(od).read())
    nf = od / NOTIFIED_NAME
    sent = set(_read_json(nf).get("weeks", []))
    n = 0
    checker = None
    gate = None
    for week, g in (picks.groupby("league_week_id", sort=True) if len(picks) else []):
        if week in sent:
            continue
        if checker is None:
            from src.direction_guard import real_track_checker
            checker = real_track_checker()
        # last check at send time (src/tip_coherence, owner 2026-11-19): a pick that now mismatches a tip SENT
        # since its freeze is left out of the message (logged); picks are checked in TOP-3 rank order
        if gate is None:
            try:
                from src.tip_coherence import CoherenceGate
                gate = CoherenceGate.load(out_dir=od)
            except Exception as e:                                     # noqa: BLE001
                log.warning(f"[top3_league] coherence gate unavailable ({type(e).__name__}: {e})")
                gate = False
        if gate:
            keep = []
            for i, r in g.sort_values("top3_rank", key=lambda x: pd.to_numeric(x, errors="coerce")).iterrows():
                why = _gate_blocks(gate, r)
                if why:
                    log.warning(f"[top3_league] SKIP {r['home_team']} v {r['away_team']} {r['market']} {r['side']} "
                                f"({week}): {why}")
                else:
                    keep.append(i)
            g = g.loc[keep]
        if g.empty:
            sent.add(week)                       # nothing left to send for this week; never retried
            continue
        if sender(message(g, checker)):
            sent.add(week)
            n += 1
            if gate:
                for _, r in g.iterrows():
                    _gate_take(gate, r)
    if n:
        LW._write_json(nf, {"weeks": sorted(sent)})
    return {"enabled": True, "sent": n}


# ── one-shot re-seed (owner 2026-11-20 "Re-seed for this weekend") ──────────────────────────────
SHADOW_NAME = "edge_engine_shadow.csv"
RESEED_KEY = "reseed"


def _archive_path(p: Path, tag: str) -> Path:
    return p.with_name(f"{p.stem}_{tag}{p.suffix}")


def _split_unplayed(raw: bytes, at: pd.Timestamp) -> tuple[bytes, int, int]:
    """(kept bytes, kept rows, dropped rows) of a first-sight CSV: drop every row first seen BEFORE `at`
    (prediction_ts) whose kickoff is AT/AFTER `at` - the unplayed fixtures scored by the pre-re-seed models.
    Every kept line keeps its exact bytes and order. Fails (ValueError) on any line it cannot read as one row."""
    import csv
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise ValueError("empty store")
    head = next(csv.reader([lines[0].decode("utf-8-sig").rstrip("\r\n")]))
    i_ts, i_ko = head.index("prediction_ts"), head.index("kickoff_utc")
    out, kept, dropped = [lines[0]], 0, 0
    for ln in lines[1:]:
        if not ln.strip():
            out.append(ln)
            continue
        rows = list(csv.reader([ln.decode("utf-8").rstrip("\r\n")]))
        if len(rows) != 1 or len(rows[0]) != len(head):
            raise ValueError("a row spans lines or has the wrong width - refusing to rewrite")
        ts = pd.to_datetime(rows[0][i_ts], utc=True, errors="coerce")
        ko = pd.to_datetime(rows[0][i_ko], utc=True, errors="coerce")
        if pd.notna(ts) and pd.notna(ko) and ts < at <= ko:
            dropped += 1
            continue
        out.append(ln)
        kept += 1
    return b"".join(out), kept, dropped


def reseed(now=None, out_dir: Path | None = None, root: Path | None = None, env: dict | None = None) -> dict:
    """ONE-SHOT re-seed of the TOP-3 candidates from the CURRENT models (owner 2026-11-20 "Re-seed for this
    weekend"). Configured by config/top3_paper_tracks.json -> reseed {at_utc, archive_tag}; at_utc MUST equal
    start_utc (the TOP-3 start moved to the re-seed instant). Run BEFORE the predict pipeline (predict.yml), so
    the same run re-scores every unplayed fixture with the served models and live prices:
      * output/top3_league_candidates.csv (+ any sidecar) -> *_<tag>.csv, bytes unchanged (archived, never deleted);
      * output/edge_engine_shadow.csv is copied byte-for-byte to edge_engine_shadow_<tag>.csv, then the live file
        keeps every line except the rows first seen before at_utc of fixtures kicking off at/after it (those were
        scored by the pre-re-seed models, and the candidates take first_seen_at from these rows). Kept lines are
        byte-identical, in order; the pipeline then logs those fixtures again at first sight from the new models.
    Idempotent: once the candidates archive exists it does nothing (ALREADY_DONE). Fails closed (nothing
    written) without the flag, without a coherent config, or before at_utc."""
    if not enabled(env):
        return {"enabled": False}
    now = LW._now(now)
    od = Path(out_dir) if out_dir else OUT
    rs = _read_json((Path(root) if root else ROOT) / START_CONFIG).get(RESEED_KEY) or {}
    if not rs:
        return {"enabled": True, "status": "NO_RESEED_CONFIGURED"}
    tag = str(rs.get("archive_tag") or "").strip()
    try:
        at = pd.Timestamp(rs.get("at_utc")) if rs.get("at_utc") else None
    except (TypeError, ValueError):
        at = None
    if at is None or pd.isna(at) or not tag or not tag.replace("_", "").replace("-", "").isalnum():
        return {"enabled": True, "status": "RESEED_CONFIG_INVALID"}
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    start = top3_start(root)
    if start is None or at != start:
        return {"enabled": True, "status": "RESEED_NOT_AT_START", "at_utc": LW._stamp(at)}
    if now < at:
        return {"enabled": True, "status": "BEFORE_RESEED_INSTANT", "at_utc": LW._stamp(at)}
    cands = od / CANDIDATES_NAME
    cands_arc = _archive_path(cands, tag)
    if cands_arc.exists():
        return {"enabled": True, "status": "ALREADY_DONE", "archive": cands_arc.name}
    shadow = od / SHADOW_NAME
    shadow_arc = _archive_path(shadow, tag)
    res = {"enabled": True, "status": "RESEEDED", "at_utc": LW._stamp(at), "tag": tag,
           "shadow_rows_kept": 0, "shadow_rows_archived_out": 0, "candidates_archived": False}
    if shadow.exists():
        raw = shadow.read_bytes()
        kept_b, kept, dropped = _split_unplayed(raw, at)          # raises before anything is written
        if shadow_arc.exists() and shadow_arc.read_bytes() != raw:
            return {"enabled": True, "status": "RESEED_PARTIAL_ARCHIVE_DIFFERS", "archive": shadow_arc.name}
        shadow_arc.write_bytes(raw)                               # the archive first, then the live file
        shadow.write_bytes(kept_b)
        res.update(shadow_rows_kept=kept, shadow_rows_archived_out=dropped, shadow_archive=shadow_arc.name)
    for p in sorted(od.glob(CANDIDATES_NAME + "?*")):             # any sidecar (e.g. .first_sight.json)
        p.replace(p.with_name(cands_arc.name + p.name[len(CANDIDATES_NAME):]))
    if cands.exists():
        cands.replace(cands_arc)
        res["candidates_archived"] = True
    else:
        cands_arc.write_bytes(b"")                                # the marker: the re-seed ran once
    res["candidates_archive"] = cands_arc.name
    return res


def _cli(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cmd = argv[0] if argv else "run"
    fn = {"run": run, "settle": settle, "notify": notify, "reseed": reseed}.get(cmd)
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
