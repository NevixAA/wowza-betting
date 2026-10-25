"""The WOWZA data contracts, version CONTRACTS-1 (2026-10-25). Descriptive, not prescriptive.

Each schema states what a row of one family of frames MUST look like: required columns, dtypes,
allowed values, ranges, and cross-column invariants. A schema describes the canonical contract;
where a committed file does not meet it, that is a FINDING to report, never a reason to bend the
schema to the file or to rewrite the file (historical rows are never rewritten).

Families requested by the 2026-10-25 platform brief, and the files they describe:

  PREDICTION_ROW                 output/predictions.csv
  MARKET_SNAPSHOT_ROW            canonical per-book quote (book, market, side, odds, timestamp,
                                 minutes_to_kickoff, overround, paired opposite price, source);
                                 output/book_odds_snapshots.csv is validated against it
  SELECTION_DECISION             market_edge_v2 output (output/edge_engine_shadow.csv)
  RESEARCH_OBSERVATION           base contract of every first-sight research log; specialised by
    V91_SHADOW_ROW                 output/v91_shadow.csv
    SHADOW_EXPERIMENT_ROW          output/model_lineage/shadow_experiments.csv
    TOP_PICKS_ROW                  output/top_picks_log.csv
    PROPS_SHOWN_ROW                output/props_shown_log.csv
    LEAGUE_WEEKLY_RESEARCH_ROW     output/league_weekly_research.csv (weekly per-league panel)
  STAKE_PLAN_ROW                 output/stake_plan.csv (and STAKE_AUDIT_ROW: output/stake_audit.csv)
  SETTLEMENT_ROW                 settled ledger rows (bets_ledger.csv, side_bets_ledger.csv)
  CLV_RECORD                     output/clv_records.csv
  MODEL_EXPERIMENT_REGISTRATION  models/registry/experiments.json entries

VOCABULARY IS NOT UNIFORM ACROSS THE ESTATE, and each schema states the vocabulary its file
actually uses: the model track is `new_format` in ledgers/predictions but `newformat` in the model
registry and the experiment ledger; the Over 3.5 market is `OU35` in the edge shadow, `OVER35` in the
stake plan and `over35` in the side ledger. Unifying names is a later, owner-reviewed step.
"""
from __future__ import annotations

import pandas as pd

from src.contracts.schema import Col, Rule, Schema, null_mask, num, text, ts

CONTRACTS_VERSION = "CONTRACTS-1"

TRACKS = ("standard", "new_format")
REGISTRY_TRACKS = ("standard", "newformat")
TIERS = ("SNIPER", "MARKSMAN", "VALUABLE")
LEGACY_TIERS = TIERS + ("AVOID",)
ENGINE_STATES = ("NO_BET", "OBSERVE", "QUALIFIED", "STRONG")
EDGE_FAMILIES = ("OU25", "OU15", "OU35", "BTTS")
PLAN_MARKETS = ("OU25", "BTTS", "OVER15", "OVER35")
SIDES = ("OVER", "UNDER", "YES", "NO")
RESULTS = ("WIN", "LOSS", "VOID", "PUSH", "HALF_WIN", "HALF_LOSS")
REAL_STATES = ("PROBATION", "ACTIVE_LOW", "ACTIVE", "ACTIVE_STRONG", "DEGRADED")
DEPLOYMENT_STATES = REAL_STATES + ("PAPER", "BLOCKED", "BLOCKED_FROM_REAL", "UNSUPPORTED")
STAKE_REASON_RX = (r"OK|NO_REAL_STAKE_[A-Z_]+|PAPER_LEAGUE|ODDS_BLOCKED|ODDS_ABOVE_TRACK_CAP|"
                   r"ODDS_BELOW_TRACK_FLOOR|TIER_NOT_STAKED|ONE_POSITION_PER_FIXTURE|[A-Z_]*CAP[A-Z_]*|"
                   r"DRAWDOWN_[A-Z_]+")
HEX64 = r"[0-9a-f]{64}"


def P(name, **kw):
    """A probability column."""
    return Col(name, "float", min=0.0, max=1.0, **kw)


def ODDS(name, **kw):
    """A decimal price: strictly above 1 (1.0 would be a certain outcome)."""
    return Col(name, "float", min=1.0001, **kw)


# ── shared rules ─────────────────────────────────────────────────────────────────────────────────
def _after_kickoff(obs_col: str, ko_col: str = "kickoff_utc") -> Rule:
    def chk(d):
        o, k = ts(d[obs_col]), ts(d[ko_col])
        return o.notna() & k.notna() & (o >= k)
    return Rule(f"{obs_col}_before_{ko_col}", chk,
                f"a first-sight observation must be written strictly before kickoff ({obs_col} < {ko_col})",
                (obs_col, ko_col))


def _side_matches_market(side_col: str, market_col: str, btts=("BTTS", "btts")) -> Rule:
    def chk(d):
        s, m = text(d[side_col]).str.upper(), text(d[market_col])
        has = ~null_mask(d[side_col])
        is_btts = m.isin(btts)
        return has & ((is_btts & ~s.isin(["YES", "NO"])) | (~is_btts & ~s.isin(["OVER", "UNDER"])))
    return Rule(f"{side_col}_matches_{market_col}", chk, "BTTS sides are YES/NO; total-goals sides OVER/UNDER",
                (side_col, market_col))


def _settled_pnl_sign(result_col="result", pnl_col="pnl") -> Rule:
    def chk(d):
        r, p = text(d[result_col]), num(d[pnl_col])
        return (((r == "WIN") & ~(p > 0)) | ((r == "LOSS") & ~(p < 0))
                | ((r.isin(["VOID", "PUSH"])) & p.notna() & (p.abs() > 1e-9)))
    return Rule("pnl_sign_matches_result", chk, "WIN => pnl > 0, LOSS => pnl < 0, VOID/PUSH => pnl 0 or blank",
                (result_col, pnl_col))


def _settled_has_pnl(result_col="result", pnl_col="pnl") -> Rule:
    def chk(d):
        return text(d[result_col]).isin(["WIN", "LOSS"]) & num(d[pnl_col]).isna()
    return Rule("settled_row_has_pnl", chk, "a WIN/LOSS row must carry its pnl", (result_col, pnl_col))


def _pnl_without_result(result_col="result", pnl_col="pnl") -> Rule:
    def chk(d):
        return null_mask(d[result_col]) & num(d[pnl_col]).notna() & (num(d[pnl_col]).abs() > 1e-9)
    return Rule("pnl_without_result", chk, "a non-zero pnl on an unsettled row", (result_col, pnl_col))


# ── PREDICTION_ROW ───────────────────────────────────────────────────────────────────────────────
PREDICTION_ROW = Schema(
    "PREDICTION_ROW", "1", "One upcoming fixture as scored by predict (output/predictions.csv).",
    kind="prediction", writers=("output/predictions.csv",),
    key=("date", "home_team", "away_team"), unique_key=True,
    columns=(
        Col("league", nullable=False), Col("date", "date", nullable=False),
        Col("kickoff_utc", "datetime"), Col("home_team", nullable=False), Col("away_team", nullable=False),
        Col("model_type", allowed=TRACKS + ("unknown",), nullable=False),
        P("p_over25"), P("p_btts", required=False), P("p_over15", required=False), P("p_over35", required=False),
        ODDS("odds_over25"), ODDS("odds_under25"),
        Col("best_side", allowed=("OVER", "UNDER")), Col("best_edge", "float", min=-1.0, max=1.0),
        Col("signal_tier", allowed=LEGACY_TIERS, nullable=False),
        Col("generated_at", "datetime", required=False), Col("model_version", required=False),
        Col("model_sha", required=False), Col("git_sha", required=False),
    ),
    rules=(
        Rule("tiered_row_has_price",
             lambda d: text(d["signal_tier"]).isin(TIERS)
             & num(d["odds_over25"].where(text(d["best_side"]) == "OVER", d["odds_under25"])).isna(),
             "a tiered fixture must carry the price of its side",
             ("signal_tier", "best_side", "odds_over25", "odds_under25")),
    ),
)

# ── MARKET_SNAPSHOT_ROW ──────────────────────────────────────────────────────────────────────────
MARKET_SNAPSHOT_ROW = Schema(
    "MARKET_SNAPSHOT_ROW", "1",
    "One bookmaker's price for one side of one market at one instant. Canonical fields per the "
    "2026-10-25 brief; change-log storage (consecutive-distinct prices) is allowed, so carry forward "
    "before aggregating.",
    kind="market_snapshot", writers=("output/book_odds_snapshots.csv",),
    columns=(
        Col("snapshot_ts", "datetime", nullable=False), Col("match_date", "date", nullable=False),
        Col("kickoff_utc", "datetime"), Col("match", nullable=False), Col("league"),
        Col("bookmaker", nullable=False, doc="book"),
        Col("market", allowed=EDGE_FAMILIES, nullable=False), Col("side", allowed=SIDES, nullable=False),
        ODDS("odds", nullable=False),
        Col("minutes_to_kickoff", "float", doc="(kickoff_utc - snapshot_ts) in minutes"),
        Col("overround", "float", min=-0.05, max=0.5,
            doc="margin 1/odds + 1/opposite_odds - 1, same book, same instant (src/market_edge.overround)"),
        ODDS("opposite_odds", doc="paired opposite price from the SAME book"),
        Col("source", nullable=False, doc="feed that produced the quote (oddsapi, api_football, ...)"),
    ),
    rules=(
        _side_matches_market("side", "market", btts=("BTTS",)),
        _after_kickoff("snapshot_ts"),
    ),
)

# ── SELECTION_DECISION ───────────────────────────────────────────────────────────────────────────
_SEL_COLS = (
    Col("selection_engine_version", allowed=("market_edge_v2",), nullable=False),
    Col("prediction_ts", "datetime", nullable=False), Col("fixture_id", nullable=False),
    Col("match_date", "date", nullable=False), Col("kickoff_utc", "datetime", nullable=False),
    Col("league", nullable=False), Col("home_team", nullable=False), Col("away_team", nullable=False),
    Col("model_type", allowed=TRACKS, nullable=False), Col("market", allowed=EDGE_FAMILIES, nullable=False),
    Col("legacy_engine_version", allowed=("legacy_raw_edge",)), Col("legacy_raw_edge", "float"),
    Col("legacy_tier", allowed=LEGACY_TIERS + ("NOT_EVALUATED",)), Col("legacy_side", allowed=SIDES),
    Col("legacy_threshold", "float"), ODDS("legacy_odds"),
    Col("new_state", allowed=ENGINE_STATES, nullable=False), Col("new_side", allowed=SIDES),
    Col("new_reason", nullable=False),
    Col("track", allowed=TRACKS), Col("decision_state", allowed=ENGINE_STATES, nullable=False),
    Col("decision_reason", nullable=False), Col("side", allowed=SIDES),
    P("p_model_raw"), P("p_market_fair"), P("p_market_fair_power"), P("p_market_fair_proportional"),
    Col("devig_method", allowed=("power", "proportional")), P("p_decision"), P("p_lower_bound"),
    Col("market_residual_raw", "float"), Col("market_residual_adjusted", "float"),
    Col("residual_weight", "float", min=0.0, max=1.0), ODDS("odds"), Col("odds_source"),
    ODDS("best_book_odds"),
    Col("market_overround", "float", min=0.0, max=0.5, doc="median per-book margin 1/a + 1/b - 1"),
    Col("ev_point", "float"), Col("ev_lower_bound", "float"), Col("effective_sample_size", "float", min=0.0),
    Col("uncertainty_penalty", "float", min=0.0), Col("penalty_breakdown"),
    Col("clv_track_n", "float", min=0.0), Col("clv_track_mean", "float"), Col("clv_track_median", "float"),
    Col("clv_track_positive_pct", "float", min=0.0, max=100.0), Col("clv_evidence_state"),
    Col("market_source", allowed=("MEDIAN_TWO_SIDED_BOOKS", "SINGLE_TWO_SIDED_BOOK", "VALIDATED_EXCHANGE",
                                  "ONE_SIDED", "NONE")),
    Col("books_used"), Col("n_books", "int", min=0), Col("n_one_sided_books", "int", min=0),
    Col("n_invalid_books", "int", min=0), Col("price_timestamp", "datetime"),
    Col("minutes_to_kickoff", "float"), Col("price_age_minutes", "float", min=0.0),
)
SELECTION_DECISION = Schema(
    "SELECTION_DECISION", "1",
    "market_edge_v2 decision for one fixture x market family, logged beside the legacy decision.",
    kind="selection_decision", writers=("output/edge_engine_shadow.csv",),
    key=("selection_engine_version", "fixture_id", "market"), unique_key=True, columns=_SEL_COLS,
    rules=(
        _after_kickoff("prediction_ts"),
        _side_matches_market("side", "market", btts=("BTTS",)),
        Rule("state_mirrors_new_state", lambda d: text(d["decision_state"]) != text(d["new_state"]),
             "new_state is the logged copy of decision_state", ("decision_state", "new_state")),
        Rule("qualified_needs_positive_ev_lb",
             lambda d: text(d["decision_state"]).isin(["QUALIFIED", "STRONG"])
             & ~(num(d["ev_lower_bound"]) > 0),
             "QUALIFIED/STRONG require ev_lower_bound > 0", ("decision_state", "ev_lower_bound")),
        Rule("selected_state_has_side",
             lambda d: text(d["decision_state"]).isin(["QUALIFIED", "STRONG"]) & null_mask(d["side"]),
             "a QUALIFIED/STRONG decision names its side", ("decision_state", "side")),
        Rule("lower_bound_le_decision", lambda d: num(d["p_lower_bound"]) > num(d["p_decision"]) + 1e-9,
             "p_lower_bound <= p_decision", ("p_lower_bound", "p_decision")),
        Rule("price_not_after_prediction",
             lambda d: ts(d["price_timestamp"]).notna() & (ts(d["price_timestamp"]) > ts(d["prediction_ts"])),
             "a quote used at prediction time cannot be timestamped after it", ("price_timestamp", "prediction_ts")),
    ),
)

# ── RESEARCH_OBSERVATION (base) and its specialisations ──────────────────────────────────────────
RESEARCH_OBSERVATION = Schema(
    "RESEARCH_OBSERVATION", "1",
    "Base contract of a first-sight research log row: an identity, the instant it was observed, the "
    "kickoff it must precede, and a market. Grading columns may be filled later; observation columns "
    "never change.",
    kind="research_observation",
    columns=(Col("fixture_id", nullable=False), Col("observed_at", "datetime", nullable=False),
             Col("kickoff_utc", "datetime"), Col("league"), Col("market", nullable=False)),
    rules=(_after_kickoff("observed_at"),),
)

V91_SHADOW_ROW = Schema(
    "V91_SHADOW_ROW", "1", "First-sight champion/challenger observation (output/v91_shadow.csv).",
    kind="research_observation", writers=("output/v91_shadow.csv",),
    key=("fixture_id", "market", "match_date"), unique_key=True,
    columns=(
        Col("fixture_id", nullable=False), Col("market", allowed=("ou25", "btts", "over15", "over35")),
        Col("logged_at", "datetime", nullable=False), Col("match_date", "date", nullable=False),
        Col("kickoff_utc", "datetime"), Col("league", nullable=False),
        Col("model_type", allowed=TRACKS + ("unknown",)),
        P("p_v9"), P("p_v91"), P("p_pro_hgb"), P("p_goal_distribution"), P("p_market"),
        Col("v9_side", allowed=("OVER", "UNDER")), Col("v9_tier", allowed=LEGACY_TIERS),
        Col("v9_edge", "float", min=-1.0, max=1.0), ODDS("entry_odds"), ODDS("closing_odds"),
        Col("clv", "float"), Col("result", allowed=RESULTS),
        Col("production_valuable_gate", "float"), Col("production_marksman_gate", "float"),
        Col("production_sniper_gate", "float"), Col("threshold_regime_id"),
        ODDS("opening_odds"), Col("v9_model_sha"), Col("dataset_id"),
        Col("home_team"), Col("away_team"),
        Col("outcome", allowed=("OVER", "UNDER", "YES", "NO")), Col("home_goals", "int", min=0),
        Col("away_goals", "int", min=0), Col("close_status"), Col("settle_status"),
    ),
    rules=(
        _after_kickoff("logged_at"),
        Rule("settled_row_has_score",
             lambda d: text(d["result"]).isin(["WIN", "LOSS"]) & (num(d["home_goals"]).isna() | num(d["away_goals"]).isna()),
             "a settled row records the score it was graded on", ("result", "home_goals", "away_goals")),
        Rule("identity_present", lambda d: null_mask(d["home_team"]) | null_mask(d["away_team"]),
             "home/away identity is needed to settle the row (P0 2026-10-20)", ("home_team", "away_team")),
    ),
)

SHADOW_EXPERIMENT_ROW = Schema(
    "SHADOW_EXPERIMENT_ROW", "1",
    "First-sight challenger-vs-champion row (output/model_lineage/shadow_experiments.csv).",
    kind="research_observation", writers=("output/model_lineage/shadow_experiments.csv",),
    key=("experiment_id", "fixture_id", "match_date"), unique_key=True,
    columns=(
        Col("experiment_id", nullable=False), Col("hypothesis_id", nullable=False),
        Col("fixture_id", nullable=False), Col("match_date", "date", nullable=False),
        Col("kickoff_utc", "datetime"), Col("league", nullable=False),
        Col("home_team", nullable=False), Col("away_team", nullable=False),
        Col("model_type", allowed=REGISTRY_TRACKS, nullable=False),
        Col("market", allowed=("over25", "over15", "over35"), nullable=False),
        Col("prediction_ts", "datetime", nullable=False),
        Col("champion_version", nullable=False), Col("champion_sha256", pattern=HEX64),
        Col("challenger_version", nullable=False), Col("challenger_sha256", pattern=HEX64, nullable=False),
        P("p_served"), P("p_champion_raw"), P("p_challenger"), P("p_market"),
        Col("dataset_id"), Col("code_commit"), Col("challenger_training_cutoff", "date"), Col("feature_schema"),
        Col("home_goals", "int", min=0), Col("away_goals", "int", min=0), Col("y", "int", allowed=("0", "1")),
        Col("result_source"), Col("settled_at", "datetime"),
    ),
    rules=(
        _after_kickoff("prediction_ts"),
        Rule("challenger_probability_logged", lambda d: num(d["p_challenger"]).isna(),
             "an experiment row exists to carry a challenger probability", ("p_challenger",)),
        Rule("label_matches_score",
             lambda d: num(d["y"]).notna() & ((num(d["home_goals"]) + num(d["away_goals"]) > 2.5).astype(float)
                                             != num(d["y"])) & text(d["market"]).eq("over25"),
             "y = 1[home_goals + away_goals > 2.5] for over25", ("y", "home_goals", "away_goals", "market")),
    ),
)

TOP_PICKS_ROW = Schema(
    "TOP_PICKS_ROW", "1", "One pick of one frozen daily list (output/top_picks_log.csv).",
    kind="research_observation", writers=("output/top_picks_log.csv",),
    key=("day", "list", "rank"), unique_key=True,
    columns=(
        Col("frozen_at", "datetime", nullable=False), Col("list", allowed=("BEST_BETS", "SAFEST", "VALUE"), nullable=False),
        Col("rank", "int", min=1, max=5, nullable=False), Col("day", "date", nullable=False),
        Col("league", nullable=False), Col("home_team", nullable=False), Col("away_team", nullable=False),
        Col("market", allowed=("OU25_OVER", "OU25_UNDER", "btts", "over15", "over35"), nullable=False),
        Col("pick"), ODDS("odds", nullable=False), P("p_win"), Col("ev", "float"),
        Col("signal_tier", allowed=TIERS), Col("model_type", allowed=TRACKS + ("unknown",)),
    ),
    rules=(Rule("frozen_on_or_before_day",
                lambda d: ts(d["frozen_at"]).dt.strftime("%Y-%m-%d") > text(d["day"]).str[:10],
                "a day's list is frozen on or before that day", ("frozen_at", "day")),),
)

PROPS_SHOWN_ROW = Schema(
    "PROPS_SHOWN_ROW", "1", "One player prop shown on the morning album (output/props_shown_log.csv).",
    kind="research_observation", writers=("output/props_shown_log.csv",),
    key=("date", "section", "rank"), unique_key=True,
    columns=(
        Col("date", "date", nullable=False), Col("shown_at", "datetime", nullable=False),
        Col("section", allowed=("priced", "model_only"), nullable=False), Col("rank", "int", min=1, nullable=False),
        Col("player_id"), Col("player_name", nullable=False), Col("match", nullable=False),
        Col("market", nullable=False), P("model_prob"), ODDS("market_odds"), Col("edge_pts", "float"),
    ),
    rules=(Rule("shown_on_or_before_day",
                lambda d: ts(d["shown_at"]).dt.strftime("%Y-%m-%d") > text(d["date"]).str[:10],
                "the album for a day is shown on or before that day", ("shown_at", "date")),
           Rule("priced_section_has_odds",
                lambda d: text(d["section"]).eq("priced") & num(d["market_odds"]).isna(),
                "a 'priced' row records the odds shown", ("section", "market_odds"))),
)

LEAGUE_WEEKLY_RESEARCH_ROW = Schema(
    "LEAGUE_WEEKLY_RESEARCH_ROW", "1",
    "One frozen top-5 member of a league x market x gameweek research panel "
    "(output/league_weekly_research.csv, src/research/league_weekly.py). A research observation, NOT a bet.",
    kind="research_observation", writers=("output/league_weekly_research.csv",),
    key=("selection_engine_version", "league_week_id", "market", "league_market_rank"), unique_key=True,
    columns=(
        Col("research_id", nullable=False), Col("selection_mode", allowed=("WEEKLY_LEAGUE_RESEARCH",), nullable=False),
        Col("league_week_id", nullable=False), Col("week_lock_ts", "datetime", nullable=False),
        Col("league", nullable=False), Col("model_type", allowed=TRACKS, nullable=False),
        Col("fixture_id", nullable=False), Col("kickoff_utc", "datetime", nullable=False),
        Col("market", allowed=PLAN_MARKETS, nullable=False), Col("side", allowed=SIDES, nullable=False),
        Col("league_market_rank", "int", min=1, max=5, nullable=False), Col("legacy_rank", "int", min=1),
        Col("selection_engine_version", allowed=("market_edge_v2",), nullable=False),
        P("p_model_raw"), P("p_market_fair"), P("p_decision"), P("p_lower_bound"),
        Col("ev_point", "float"), Col("ev_lower_bound", "float"), ODDS("odds"),
        Col("decision_state", allowed=ENGINE_STATES, nullable=False),
        Col("price_quality_state", allowed=("TWO_SIDED_MULTI_BOOK", "TWO_SIDED_SINGLE_BOOK", "INSUFFICIENT"),
            nullable=False),
        Col("first_seen_at", "datetime", nullable=False), Col("created_at", "datetime", nullable=False),
        Col("result", allowed=RESULTS), Col("flat_pnl", "float"), ODDS("closing_odds"), Col("clv_pct", "float"),
        P("closing_market_fair_prob"),
    ),
    rules=(
        _after_kickoff("first_seen_at"),
        _side_matches_market("side", "market", btts=("BTTS",)),
        Rule("first_seen_before_week_lock", lambda d: ts(d["first_seen_at"]) >= ts(d["week_lock_ts"]),
             "a week member was first seen strictly before the week lock (WL-1)", ("first_seen_at", "week_lock_ts")),
        Rule("insufficient_price_has_no_pnl",
             lambda d: text(d["price_quality_state"]).eq("INSUFFICIENT") & num(d["flat_pnl"]).notna(),
             "a row without a two-sided price never carries a P/L", ("price_quality_state", "flat_pnl")),
        Rule("qualified_needs_positive_ev_lb",
             lambda d: text(d["decision_state"]).isin(["QUALIFIED", "STRONG"]) & ~(num(d["ev_lower_bound"]) > 0),
             "QUALIFIED/STRONG require ev_lower_bound > 0", ("decision_state", "ev_lower_bound")),
    ),
)

# ── STAKE_PLAN_ROW ───────────────────────────────────────────────────────────────────────────────
_STAKE_COLS = (
    Col("date", "date", nullable=False), Col("fixture", nullable=False), Col("league", nullable=False),
    Col("model_type", allowed=TRACKS + ("unknown",), nullable=False),
    Col("market", allowed=PLAN_MARKETS, nullable=False), Col("side", allowed=SIDES, nullable=False),
    Col("signal_tier", allowed=TIERS, nullable=False), ODDS("odds", nullable=False),
    Col("policy_version", nullable=False),
    Col("track", pattern=r"(standard|new_format|unknown)\|(OU25|BTTS|OVER15|OVER35)", nullable=False),
    Col("deployment_state", allowed=DEPLOYMENT_STATES, nullable=False),
    Col("bankroll_before", "float", min=0.0), Col("base_fraction", "float", min=0.0, max=0.02),
    Col("tier_multiplier", "float", min=0.0), Col("league_multiplier", "float", min=0.0),
    Col("drawdown_multiplier", "float", min=0.0, max=1.0), Col("requested_fraction", "float", min=0.0),
    Col("capped_fraction", "float", min=0.0), Col("daily_scale", "float", min=0.0, max=1.0),
    Col("final_fraction", "float", min=0.0, max=0.02, nullable=False), Col("stake_amount", "float", min=0.0),
    Col("reason", pattern=STAKE_REASON_RX, nullable=False),
    Col("bankroll_basis"), Col("generated_at", "datetime"),
)


def _stake_rules() -> tuple:
    return (
        Rule("stake_only_on_ok", lambda d: (num(d["final_fraction"]) > 0) & (text(d["reason"]) != "OK"),
             "a non-zero stake carries reason OK", ("final_fraction", "reason")),
        Rule("stake_only_on_real_state",
             lambda d: (num(d["final_fraction"]) > 0) & ~text(d["deployment_state"]).isin(REAL_STATES),
             "only a real-money track state may carry a stake", ("final_fraction", "deployment_state")),
        Rule("track_is_model_type_market",
             lambda d: text(d["track"]) != text(d["model_type"]) + "|" + text(d["market"]),
             "track = model_type|market", ("track", "model_type", "market")),
        Rule("fixture_matches_date",
             lambda d: text(d["fixture"]).str[:10] != text(d["date"]).str[:10],
             "fixture key starts with its date", ("fixture", "date")),
        Rule("one_position_per_fixture",
             lambda d: (num(d["final_fraction"]) > 0)
             & d.assign(_s=num(d["final_fraction"]) > 0).groupby("fixture")["_s"].transform("sum").gt(1),
             "the policy allows one real position per fixture", ("final_fraction", "fixture")),
    )


STAKE_PLAN_ROW = Schema(
    "STAKE_PLAN_ROW", "1", "One tip sized by the approved bankroll policy (output/stake_plan.csv).",
    kind="stake_plan", writers=("output/stake_plan.csv",),
    key=("policy_version", "fixture", "track", "side"), unique_key=True,
    columns=_STAKE_COLS + (
        Col("league_rating", "float", required=False), Col("rating_scale", "float", min=0.0, max=1.0, required=False),
        Col("selection_engine_version", allowed=("legacy_raw_edge",), required=False),
        Col("decision_state", allowed=ENGINE_STATES, required=False),
        Col("engine_version", allowed=("market_edge_v2",), required=False),
        P("p_market_fair", required=False), P("p_decision", required=False), P("p_lower_bound", required=False),
        Col("ev_lower_bound", "float", required=False), Col("market_source", required=False),
        Col("clv_evidence_state", required=False),
    ),
    rules=_stake_rules(),
)

STAKE_AUDIT_ROW = Schema(
    "STAKE_AUDIT_ROW", "1", "Append-only first-sight stake decision (output/stake_audit.csv).",
    kind="stake_plan", writers=("output/stake_audit.csv",),
    key=("policy_version", "fixture", "track"), unique_key=True,
    columns=_STAKE_COLS, rules=_stake_rules(),
)

# ── SETTLEMENT_ROW ───────────────────────────────────────────────────────────────────────────────
SETTLEMENT_ROW = Schema(
    "SETTLEMENT_ROW", "1",
    "A ledger row's settlement fields (bets_ledger.csv / side_bets_ledger.csv). Unsettled rows are "
    "valid with blank result and pnl.",
    kind="settlement", writers=("output/bets_ledger.csv", "output/side_bets_ledger.csv"),
    columns=(
        Col("source", allowed=("live", "backtest")), Col("match_date", "date", nullable=False),
        Col("league", nullable=False), Col("home_team", nullable=False), Col("away_team", nullable=False),
        ODDS("odds"), Col("signal_tier", allowed=TIERS), Col("model_type", allowed=TRACKS + ("unknown",)),
        Col("result", allowed=RESULTS), Col("pnl", "float"), ODDS("closing_odds"), Col("clv_pct", "float"),
    ),
    rules=(_settled_has_pnl(), _settled_pnl_sign(), _pnl_without_result(),
           Rule("closing_price_has_clv",
                lambda d: num(d["closing_odds"]).notna() & num(d["clv_pct"]).isna(),
                "a recorded closing price has its CLV computed", ("closing_odds", "clv_pct"))),
)

# ── CLV_RECORD ───────────────────────────────────────────────────────────────────────────────────
CLV_RECORD = Schema(
    "CLV_RECORD", "1", "One bet's entry vs closing price (output/clv_records.csv).",
    kind="clv", writers=("output/clv_records.csv",), key=("bet_id",), unique_key=True,
    columns=(
        Col("bet_id", nullable=False), Col("ts_bet", "datetime", nullable=False), Col("market", nullable=False),
        Col("player"), Col("match", nullable=False), Col("side", nullable=False),
        ODDS("odds_bet", nullable=False), ODDS("under_odds_bet"), ODDS("odds_close"), ODDS("under_odds_close"),
        P("p_bet_novig"), P("p_close_novig"), Col("clv_pct", "float"), Col("clv_prob", "float"),
        Col("result", allowed=RESULTS), Col("pnl", "float"), Col("notes"),
    ),
    rules=(
        Rule("close_has_clv", lambda d: num(d["odds_close"]).notna() & num(d["clv_pct"]).isna(),
             "a closing price has its CLV computed", ("odds_close", "clv_pct")),
        Rule("clv_without_close", lambda d: num(d["odds_close"]).isna() & num(d["clv_pct"]).notna(),
             "a CLV value needs the closing price it was measured against", ("odds_close", "clv_pct")),
        _settled_has_pnl(), _settled_pnl_sign(),
    ),
)

# ── MODEL_EXPERIMENT_REGISTRATION ────────────────────────────────────────────────────────────────
MODEL_EXPERIMENT_REGISTRATION = Schema(
    "MODEL_EXPERIMENT_REGISTRATION", "1",
    "One frozen challenger registration (models/registry/experiments.json -> experiments[]).",
    kind="experiment_registration", writers=("models/registry/experiments.json",),
    key=("experiment_id",), unique_key=True,
    columns=(
        Col("experiment_id", nullable=False), Col("hypothesis_id", nullable=False),
        Col("status", allowed=("ACTIVE", "PAUSED", "RETIRED", "CONCLUDED"), nullable=False),
        Col("model_type", allowed=REGISTRY_TRACKS + ("btts", "over15", "over35"), nullable=False),
        Col("market", nullable=False), Col("pred_column", nullable=False),
        Col("registered_at", "datetime", nullable=False), Col("trained_at", "datetime", nullable=False),
        Col("training_cutoff", "date", nullable=False), Col("feature_schema", nullable=False),
        Col("artifact", nullable=False), Col("artifact_sha256", pattern=HEX64, nullable=False),
        Col("challenger_version", nullable=False), Col("comparator", required=False),
        Col("n_min_per_track", "int", min=1, required=False),
    ),
    rules=(
        Rule("cutoff_before_registration",
             lambda d: ts(d["training_cutoff"]) >= ts(d["registered_at"]),
             "training data must end before the experiment is registered", ("training_cutoff", "registered_at")),
        Rule("version_embeds_sha",
             lambda d: ~pd.Series([str(v).endswith(str(s)[:8]) for v, s in
                                   zip(d["challenger_version"], d["artifact_sha256"])], index=d.index),
             "challenger_version ends with artifact_sha256[:8]", ("challenger_version", "artifact_sha256")),
    ),
)

SCHEMAS = {s.name: s for s in (
    PREDICTION_ROW, MARKET_SNAPSHOT_ROW, SELECTION_DECISION, RESEARCH_OBSERVATION, V91_SHADOW_ROW,
    SHADOW_EXPERIMENT_ROW, TOP_PICKS_ROW, PROPS_SHOWN_ROW, LEAGUE_WEEKLY_RESEARCH_ROW, STAKE_PLAN_ROW, STAKE_AUDIT_ROW,
    SETTLEMENT_ROW, CLV_RECORD, MODEL_EXPERIMENT_REGISTRATION)}

#: committed file -> schema, for the daily validation of what is actually in the repo
COMMITTED_FILES = {
    "output/predictions.csv": "PREDICTION_ROW",
    "output/book_odds_snapshots.csv": "MARKET_SNAPSHOT_ROW",
    "output/edge_engine_shadow.csv": "SELECTION_DECISION",
    "output/v91_shadow.csv": "V91_SHADOW_ROW",
    "output/model_lineage/shadow_experiments.csv": "SHADOW_EXPERIMENT_ROW",
    "output/top_picks_log.csv": "TOP_PICKS_ROW",
    "output/props_shown_log.csv": "PROPS_SHOWN_ROW",
    "output/league_weekly_research.csv": "LEAGUE_WEEKLY_RESEARCH_ROW",
    "output/stake_plan.csv": "STAKE_PLAN_ROW",
    "output/stake_audit.csv": "STAKE_AUDIT_ROW",
    "output/bets_ledger.csv": "SETTLEMENT_ROW",
    "output/side_bets_ledger.csv": "SETTLEMENT_ROW",
    "output/clv_records.csv": "CLV_RECORD",
    "models/registry/experiments.json": "MODEL_EXPERIMENT_REGISTRATION",
}
