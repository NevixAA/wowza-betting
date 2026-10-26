"""
v9 Football Prediction Pipeline — Hybrid v7 + v8
==================================================
Entry point for all pipeline modes.

Usage
-----
    python pipeline.py --mode train
    python pipeline.py --mode predict
    python pipeline.py --mode backtest
    python pipeline.py --mode all

Modes
-----
    train    : load data, engineer features, train + calibrate models, save
    predict  : load trained models, fetch upcoming fixtures, output bets
    backtest : run strict walk-forward backtest, output results + summary
    all      : train → predict → backtest

v9 additions over v8
---------------------
    - Odds drift tracking  : every predict run snapshots odds → drift signal
                             adjusts SNIPER/VALUE tiers at output time
    - Both-losing guard    : leagues flagged by v7 optimizer are set to AVOID
    - Postponement detect  : fixtures missing since last run are flagged
    - 3-tier output        : SNIPER (>=0.10) / VALUE (0.04-0.10) / AVOID (<0.04)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.data_loader import load_all_matches
from src.feature_engineering import build_features
from src.model import (train as train_model, load_models,
                        get_feature_importances, chronological_split, score_payload,
                        predict_proba as model_predict_proba,
                        FEATURE_COLS as MODEL_FEATURE_COLS)
from src.betting import generate_bets
from src.backtest import (run_backtest, run_side_market_backtest,
                           optimize_side_market_thresholds, optimize_standard_thresholds)
from src.ledger import (append_tips, append_side_market_tips, append_ht_tips,
                        append_ht_observations, print_ledger)


# ── Side-market bet generation ────────────────────────────────────────────────

def _load_side_market_thresholds() -> dict:
    """Load per-league thresholds from best_params_side_markets.json if it exists."""
    import json
    th_file = config.MODELS_DIR / "best_params_side_markets.json"
    if th_file.exists():
        with open(th_file) as f:
            return json.load(f)
    return {}


def _generate_side_bets(preds: "pd.DataFrame", side_markets: dict) -> "pd.DataFrame":
    """
    Generate SNIPER/MARKSMAN/VALUABLE tips for BTTS / Over 1.5 / Over 3.5.
    Uses per-league optimized thresholds from best_params_side_markets.json when that league's
    threshold has been APPROVED out of sample, falling back to the fixed global bar otherwise.

    NO LEAGUE IS EVER SUPPRESSED. This docstring used to say "leagues marked 'drop=True' are
    suppressed entirely", which has not been true since the 2026/27 live-test policy — the code
    below has emitted tips for every league all along. Corrected 2026-09-22 rather than left to
    mislead the next reader into thinking leagues go dark.
    """
    import pandas as pd
    import numpy as np
    from src.stake_plan import SIDE_SELECTION_COLUMNS
    frames = []
    OVERROUND    = 1.08
    league_params = _load_side_market_thresholds()  # {target: {league: {sniper_th, marksman_th, drop}}}

    for target in side_markets:
        prob_col = f"p_{target}"
        odds_col = f"odds_{target}"
        if prob_col not in preds.columns or odds_col not in preds.columns:
            continue
        _src_cols = ["date", "league", "home_team", "away_team", prob_col, odds_col]
        if "model_type" in preds.columns:
            _src_cols.append("model_type")
        df = preds[_src_cols].dropna(subset=[prob_col, odds_col]).copy()
        if df.empty:
            continue
        # THE ORIGINATING PREDICTION'S MODEL TAG (P0 2026-10-22). This frame used to drop it, so
        # src/stake_plan.py read model_type="unknown" for every side tip and the bankroll policy
        # mapped all BTTS / OVER15 tips to `unknown|BTTS` -> UNSUPPORTED -> zero stake, while
        # new_format|BTTS is the ACTIVE track. Taken from the predictions row (src/predict.py tags
        # it); config.model_type_for_league only where that row carries no tag. Never forced.
        _canon = df["league"].map(config.model_type_for_league)
        if "model_type" in df.columns:
            _mt = df["model_type"]
            _mt = _mt.where(_mt.notna() & (_mt.astype(str).str.strip() != ""), _canon)
        else:
            _mt = _canon
        df["model_type"] = _mt.astype(str)

        market_params = league_params.get(target, {})

        df["model_prob"]  = df[prob_col]
        df["market_odds"] = df[odds_col]
        df["odds"]        = df["market_odds"]   # canonical name (src/stake_plan.SIDE_SELECTION_COLUMNS)
        df["fair_prob"]   = (1.0 / df["market_odds"]) / OVERROUND
        df["edge"]        = df["model_prob"] - df["fair_prob"]
        df["ev"]          = df["model_prob"] * df["market_odds"] - 1.0
        df["market"]      = target
        # EXPLICIT SIDE. This path is one-sided by construction: it reads p_{target} and
        # odds_{target}, so "btts" means BTTS-YES and "over15"/"over35" mean the OVER side.
        # Recording it removes the ambiguity from the ledger as well as the alert, and makes the
        # column exist for the day the opposite side becomes reachable.
        #
        # BTTS-NO IS DELIBERATELY NOT ENABLED. Measured 2026-08-22: the NO side shows a median
        # +2.66% edge with 61 of 138 fixtures over 3%, against YES at median -2.56% with 18. That
        # asymmetry is NOT opportunity — it is model bias. True BTTS base rate is 0.5283 (n=14,586),
        # p_btts averages 0.5134 (-1.49pp LOW) while the market sits at 0.5416 (+1.33pp high), so
        # the MARKET is closer to truth than the model. The 2.82pp model-market gap manufactures a
        # NO edge on every fixture before any fixture-specific information exists. Enabling NO
        # would systematically bet a calibration error 61 times. Recalibrate p_btts first.
        df["side"]        = {"btts": "YES", "over15": "OVER", "over35": "OVER"}.get(target, "")

        def _tier(row):
            lg = row["league"]
            lp = market_params.get(lg, {})
            # Live-test policy (2026/27): NEVER SUPPRESS A LEAGUE. Every league emits signals
            # all season so it builds a CLV/ROI record — a league you stopped tipping is a
            # league you can never learn about. Real-money gating is a separate downstream
            # decision, not a notification filter.
            #
            # What the gate decides is narrower: whether to trust THIS league's own learned
            # threshold, or fall back to the global bar. A league's threshold is used only once
            # `approved` — meaning a walk-forward pass tuned it on earlier seasons, bet the next
            # one blind, and came out positive (src/backtest.optimize_side_market_thresholds).
            #
            # Before 2026-09-22 that optimizer had no out-of-sample pass at all: it grid-searched
            # ~17 thresholds on all the data and shipped the winner. Pro's threshold study then
            # showed what that costs — of the four cells with enough data to check, THREE were
            # worse out of sample (Norway -29.7pp, China -40.9pp, Finland -61.6pp). An
            # in-sample-optimal bar is not just unproven, it lost money on fresh fixtures.
            #
            # `approved` missing (an older best_params file) falls back to the previous rule, so
            # this can never regress an environment that has not rerun the optimizer yet.
            _approved = lp.get("approved")
            _use_own = (lp.get("approved", not lp.get("drop", False))
                        if _approved is not None else not lp.get("drop", False))
            if _use_own:
                sniper_th   = lp.get("sniper_th",   0.10)
                marksman_th = lp.get("marksman_th",  0.08)
            else:
                sniper_th, marksman_th = 0.10, 0.08
            edge = row["edge"]
            if edge >= sniper_th:   return "SNIPER"
            if edge >= marksman_th: return "MARKSMAN"
            if edge >= 0.04:        return "VALUABLE"
            return "AVOID"

        df["signal_tier"] = df.apply(_tier, axis=1)
        df = df[df["signal_tier"] != "AVOID"]
        df = df.sort_values("edge", ascending=False)
        # ONE SCHEMA, defined once in src/stake_plan.py and read by the stake plan. `market_odds`
        # is kept beside the canonical `odds` for the side_bets.csv / ledger / notify readers.
        frames.append(df[list(SIDE_SELECTION_COLUMNS) + ["market_odds"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("edge", ascending=False)


# ── TRAIN ─────────────────────────────────────────────────────────────────────

TRAIN_LOG = config.OUTPUT_DIR / "retrain_log.json"


def _mean_logloss(metrics: dict) -> float:
    """Mean held-out log loss across the base models. Lower is better."""
    vals = [m["log_loss"] for m in metrics.values()
            if isinstance(m, dict) and isinstance(m.get("log_loss"), (int, float))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _log_shadow(preds: "pd.DataFrame", exp_rows: "pd.DataFrame | None" = None) -> None:
    """§3/§18 — record what every model believed about every upcoming fixture, BEFORE kickoff.

    THIS IS THE FILE THAT UNBLOCKS EVERY CHALLENGER. The promotion gate's `second_period` check
    requires an INDEPENDENT forward period: fixtures that had not been played, and whose prices
    had not formed, when the prediction was written. Nothing in the estate recorded that, so the
    check could never pass and every challenger was pinned at CHALLENGER by construction rather
    than by evidence. It is not a backlog problem -- no amount of history can create a
    prospective record retrospectively, so the clock starts when this first runs and not before.

    FROZEN AT FIRST SIGHT. predict runs every 5-15 minutes, so a fixture is scored ~100 times
    before kickoff and the later runs see more of the price curve and more team news. Keeping
    the latest would quietly convert a prospective log into a retrospective one -- the model
    would be judged on its best-informed opinion while appearing to be judged on its first.
    `append_shadow` dedups on (fixture_id, match_date) and keeps the EARLIEST, so a logged
    opinion cannot be revised once the world has moved.

    Every scored fixture is logged, not only the tipped ones. Tips fire on ~14% of the board, so
    a tips-only record would need years to reach a usable n -- and it would throw away exactly
    the fixtures that test whether the tier threshold separates anything.

    `exp_rows` (src.shadow_experiments.score) carries the FROZEN challengers' probabilities for
    the same fixtures at the same moment; an experiment that names a v91 column (H-MODEL-01 ->
    `p_pro_hgb`) fills it here, at first sight, and nowhere else. Without `exp_rows` every
    challenger column stays empty — never a copy of the champion.

    Cannot raise: this is a recorder, and a recorder must never be able to stop tips going out.
    """
    try:
        import numpy as np
        from src.shadow_compare import append_shadow

        d = preds.copy()
        # A stable identity. `_oa_event_id` is present only when the fixture matched an OddsAPI
        # event, so fall back to the natural key rather than dropping the row -- an unmatched
        # fixture is still a prediction worth grading.
        fid = d["_oa_event_id"] if "_oa_event_id" in d.columns else pd.Series(index=d.index,
                                                                             dtype=object)
        natural = (d["date"].astype(str) + "|" + d["home_team"].astype(str) + "|"
                   + d["away_team"].astype(str))
        d["fixture_id"] = fid.where(fid.notna() & (fid.astype(str) != ""), natural)

        from src.gate_resolver import resolve_effective_tier_gates, regime_id
        now = datetime.now(timezone.utc).isoformat()
        rid = regime_id()

        # THE GATES THAT WERE IN FORCE, resolved per league from what production actually
        # reads — not from config.py, which disagrees with it for any league carrying an
        # approved optimiser value. Cached per league so 150 fixtures do not re-read the
        # artifacts 150 times.
        _gcache: dict = {}

        def _gate(league, market, which):
            k = (str(league), market)
            if k not in _gcache:
                _gcache[k] = resolve_effective_tier_gates(str(league), market)
            return getattr(_gcache[k], which)

        # Challenger columns from the registered experiments, keyed by fixture_id. P0 2026-10-22:
        # these were hard-coded pd.NA, so p_pro_hgb was empty on 669/669 rows by construction.
        ch_cols = {"p_v91": pd.Series(pd.NA, index=d.index, dtype=object),
                   "p_pro_hgb": pd.Series(pd.NA, index=d.index, dtype=object)}
        if exp_rows is not None and not exp_rows.empty:
            from src.shadow_experiments import load_registry
            col_of = {e["experiment_id"]: e.get("v91_shadow_column") for e in load_registry()}
            for eid, g in exp_rows.groupby("experiment_id"):
                c = col_of.get(eid)
                if c in ch_cols:
                    mp = dict(zip(g["fixture_id"].astype(str), g["p_challenger"]))
                    hit = d["fixture_id"].astype(str).map(mp)
                    ch_cols[c] = ch_cols[c].where(hit.isna(), hit)

        rows = pd.DataFrame({
            "fixture_id": d["fixture_id"],
            "market": "ou25",
            "logged_at": now,
            "match_date": d["date"],
            "kickoff_utc": d.get("kickoff_utc", pd.NA),
            "league": d["league"],
            "model_type": d.get("model_type", pd.NA),
            # What each model believed. p_v91/p_pro_hgb/p_goal_distribution stay NaN until those
            # challengers are wired -- a column that exists and is empty is honest; a column
            # filled with the champion's number would silently fake agreement.
            "p_v9": d.get("p_over25", pd.NA),
            "p_v91": ch_cols["p_v91"],
            "p_pro_hgb": ch_cols["p_pro_hgb"],
            "p_goal_distribution": d.get("p_over25_poisson_dc", pd.NA),
            "p_market": d.get("impl_prob_over", d.get("implied_prob_over", pd.NA)),
            "v9_side": d.get("best_side", pd.NA),
            "v91_side": pd.NA,
            "v9_tier": d.get("signal_tier", pd.NA),
            "v91_tier": pd.NA,
            "v9_edge": d.get("best_edge", pd.NA),
            "v91_edge": pd.NA,
            "entry_odds": np.where(d.get("best_side", "").astype(str).str.upper() == "UNDER",
                                   d.get("odds_under25", np.nan), d.get("odds_over25", np.nan)),
            "closing_odds": pd.NA,
            "clv": pd.NA,
            "result": pd.NA,
            # Provenance. Without these a row cannot be tied to the artefact that wrote it, and
            # a forward period that cannot name its model proves nothing about that model.
            "production_valuable_gate": d["league"].map(lambda l: _gate(l, "ou25", "valuable")),
            "production_marksman_gate": d["league"].map(lambda l: _gate(l, "ou25", "marksman")),
            "production_sniper_gate": d["league"].map(lambda l: _gate(l, "ou25", "sniper")),
            "threshold_regime_id": rid,
            # Opening price, so open->current movement is reconstructable later. Grading fields
            # (closing_odds, clv, result) stay empty here by design — §6 forbids a future value
            # ever reaching a selection rule, and the only way to guarantee that is for the
            # prediction-time writer never to populate one.
            "opening_odds": d.get("first_over_odds", pd.NA),
            "book_count": pd.NA,
            "market_dispersion": pd.NA,
            "v9_model_sha": d.get("model_sha", pd.NA),
            "v91_model_sha": pd.NA,
            "dataset_id": d.get("git_sha", pd.NA),
            # Identity, so settlement can join a results source. Without these the row held
            # only an OddsAPI event id and could never be graded (P0 2026-10-20).
            "home_team": d.get("home_team", pd.NA),
            "away_team": d.get("away_team", pd.NA),
        })
        n = append_shadow(rows)
        log.info(f"shadow: logged {n} new fixture(s) of {len(rows)} scored")
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"shadow log skipped ({e}) — predictions and tips are unaffected")


H2H_FILE = config.OUTPUT_DIR / "model_lineage" / "shadow_h2h.csv"
H2H_COLS = ["logged_at", "match_date", "kickoff_utc", "league", "home_team", "away_team",
            "model_type", "market", "champion_version", "p_served", "p_champion_raw",
            "challenger_version", "p_challenger"]


def _log_h2h(preds: "pd.DataFrame", exp_rows: "pd.DataFrame | None" = None) -> None:
    """TRUE HEAD-TO-HEAD SHADOW (champion high-water mark, brief §25).

    For every model type, record what the CHAMPION said about every upcoming fixture — and, when
    a SHADOW_MORE challenger is parked in models/registry/<type>/challenger.pkl.gz, what the
    challenger said about the SAME fixture at the SAME moment. The challenger never reaches a
    tier, a tip, a stake or Telegram: it is scored here and only here.

    `p_served` is the production column; `p_champion_raw` / `p_challenger` are both raw
    `predict_proba` on the identical feature row, so the comparison is like-for-like even if
    predict ever post-processes the served number.

    FIRST SIGHT ONLY, keyed on (fixture, model_type, challenger_version): a later run, with more
    of the price curve and team news, cannot revise a logged opinion. Pro grades the results.
    Cannot raise — a recorder must never stop tips going out.

    P0 2026-10-22: the retrain slot above is filled only by a SHADOW_MORE retrain verdict, and none
    has been issued, so p_champion_raw / challenger_version / p_challenger were empty on every row.
    Registered prospective experiments (`exp_rows`, src.shadow_experiments) are now challengers
    here too, one row per (fixture, challenger_version); a model type covered by an experiment no
    longer gets the champion-only blank-challenger row.
    """
    try:
        import numpy as np
        from src import model_registry as reg
        if preds is None or preds.empty:
            return
        now = datetime.now(timezone.utc).isoformat()
        tag = (preds["model_type"].astype(str).map(reg.ou_model_for)
               if "model_type" in preds.columns else pd.Series("standard", index=preds.index))
        frames = []
        exp_types: set = set()
        if exp_rows is not None and not exp_rows.empty:
            exp_types = set(exp_rows["model_type"].astype(str))
            frames.append(pd.DataFrame({
                "logged_at": exp_rows["prediction_ts"].values,
                "match_date": exp_rows["match_date"].values,
                "kickoff_utc": exp_rows["kickoff_utc"].values,
                "league": exp_rows["league"].values, "home_team": exp_rows["home_team"].values,
                "away_team": exp_rows["away_team"].values,
                "model_type": exp_rows["model_type"].values,
                "market": exp_rows["market"].values,
                "champion_version": exp_rows["champion_version"].values,
                "p_served": np.round(pd.to_numeric(exp_rows["p_served"]).values, 5),
                "p_champion_raw": np.round(pd.to_numeric(exp_rows["p_champion_raw"]).values, 5),
                "challenger_version": exp_rows["challenger_version"].values,
                "p_challenger": np.round(pd.to_numeric(exp_rows["p_challenger"]).values, 5)}))
        for mt, col in reg.PRED_COLUMN.items():
            if col not in preds.columns:
                continue
            sub = preds[tag == mt] if mt in ("standard", "newformat") else preds
            sub = sub[pd.to_numeric(sub[col], errors="coerce").notna()]
            if mt in exp_types and not reg.challenger_path(mt).exists():
                # Fixtures an experiment already logged need no champion-only blank row.
                e = exp_rows[exp_rows["model_type"].astype(str) == mt]
                cov = set(zip(e["match_date"].astype(str), e["home_team"].astype(str),
                              e["away_team"].astype(str)))
                sub = sub[[(str(a)[:10], str(b), str(c)) not in cov for a, b, c in
                           zip(sub["date"], sub["home_team"], sub["away_team"])]]
            if sub.empty:
                continue
            champ = reg.stamp_for(mt)
            p_raw = p_ch = np.full(len(sub), np.nan)
            ch_v = ""
            try:
                ch = reg.load_challenger(mt)
                if ch is not None:
                    ch_payload, meta = ch
                    ch_v = meta.get("model_version", "")
                    p_ch = np.asarray(model_predict_proba(sub, payload=ch_payload), float)
                    p_raw = np.asarray(model_predict_proba(
                        sub, payload=load_models(model_file=reg.MODEL_FILES[mt])), float)
            except Exception as e:                                     # noqa: BLE001
                log.warning(f"h2h: challenger for {mt} not scored ({e})")
                ch_v, p_raw, p_ch = "", np.full(len(sub), np.nan), np.full(len(sub), np.nan)
            frames.append(pd.DataFrame({
                "logged_at": now, "match_date": sub["date"].astype(str).str[:10].values,
                "kickoff_utc": sub["kickoff_utc"].values if "kickoff_utc" in sub.columns else "",
                "league": sub["league"].values, "home_team": sub["home_team"].values,
                "away_team": sub["away_team"].values, "model_type": mt,
                "market": reg.TARGET[mt], "champion_version": champ["model_version"],
                "p_served": pd.to_numeric(sub[col], errors="coerce").round(5).values,
                "p_champion_raw": np.round(p_raw, 5), "challenger_version": ch_v,
                "p_challenger": np.round(p_ch, 5)}))
        if not frames:
            return
        new = pd.concat(frames, ignore_index=True)[H2H_COLS]
        key = ["match_date", "home_team", "away_team", "model_type", "challenger_version"]
        if H2H_FILE.exists():
            old = pd.read_csv(H2H_FILE, dtype=str, keep_default_na=False)
            seen = set(map(tuple, old[key].astype(str).values))
            new = new[[tuple(map(str, r)) not in seen
                       for r in new[key].astype(str).values]]
        new = new.drop_duplicates(key, keep="first")
        if new.empty:
            return
        H2H_FILE.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(H2H_FILE, mode="a", header=not H2H_FILE.exists(), index=False)
        log.info(f"h2h: +{len(new)} first-sight row(s) -> {H2H_FILE.name}")
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"h2h log skipped ({e}) — predictions and tips are unaffected")


def _behaviour_canary(label: str, old_payload, results: dict, test_df, target: str) -> dict:
    """Did the challenger merely get better, or did it start behaving like a different model?

    WHY LOG LOSS IS NOT ENOUGH. The promotion gate asks one question -- did the average get
    worse -- and on 2026-09-23 the answer was an emphatic no: log loss improved 0.68885 ->
    0.64066, so the model promoted correctly and by a wide margin. What the gate could not see
    is that the SHAPE of its output had changed completely. On the upcoming board the spread of
    p(over 2.5) doubled (std 0.109 -> 0.207), 19 of 57 fixtures crossed the Over/Under boundary,
    and a single predict run produced five real-money SNIPER tips with edges to 29.5% where the
    week before had nothing above 11%. A mean can improve while every individual number moves.

    This measures the shape, on the SAME holdout the gate already built, from two models it has
    already loaded. One extra prediction pass, no refit, no new data.

    IT NEVER BLOCKS. Promotion stays the gate's decision; this only records and warns. A
    behaviour check that can stop a retrain is a new way for production to break, and the whole
    estate's health design is that nothing gates v9. What it buys is that the next 2026-09-23
    shows up in the retrain log and the Telegram digest the moment it happens, instead of being
    found days later by reading a ledger.

    Thresholds are set from that incident, which is the only real example available: it would
    have tripped all three.
    """
    out = {"available": False}
    if old_payload is None or test_df is None or len(test_df) < 50:
        return out
    try:
        import numpy as np
        payload_new = {
            "target": target,
            "models": {k: v["model"] for k, v in results.items()},
            "feature_cols": results[next(iter(results))]["feature_cols"],
            "metrics": {k: v["metrics"] for k, v in results.items()},
        }
        p_old = np.asarray(model_predict_proba(test_df, payload=old_payload), dtype=float)
        p_new = np.asarray(model_predict_proba(test_df, payload=payload_new), dtype=float)
        m = np.isfinite(p_old) & np.isfinite(p_new)
        if m.sum() < 50:
            return out
        p_old, p_new = p_old[m], p_new[m]
        delta = p_new - p_old
        crossed = ((p_old > 0.5) & (p_new < 0.5)) | ((p_old < 0.5) & (p_new > 0.5))
        spread_ratio = float(p_new.std() / p_old.std()) if p_old.std() > 1e-9 else float("nan")
        out = {
            "available": True, "n": int(m.sum()),
            "std_old": round(float(p_old.std()), 5), "std_new": round(float(p_new.std()), 5),
            "spread_ratio": round(spread_ratio, 3) if spread_ratio == spread_ratio else None,
            "mean_old": round(float(p_old.mean()), 5), "mean_new": round(float(p_new.mean()), 5),
            "frac_crossed_half": round(float(crossed.mean()), 4),
            "frac_moved_gt_10pp": round(float((np.abs(delta) > 0.10).mean()), 4),
            "max_abs_move": round(float(np.abs(delta).max()), 4),
        }
        flags = []
        if spread_ratio == spread_ratio and (spread_ratio > 1.5 or spread_ratio < 1 / 1.5):
            flags.append(f"spread x{spread_ratio:.2f}")
        if out["frac_crossed_half"] > 0.20:
            flags.append(f"{out['frac_crossed_half']:.0%} crossed the 0.5 boundary")
        if out["frac_moved_gt_10pp"] > 0.25:
            flags.append(f"{out['frac_moved_gt_10pp']:.0%} moved >10pp")
        out["flags"] = flags
        if flags:
            log.warning(f"  [{label}] BEHAVIOUR CANARY: {'; '.join(flags)}. The average may have "
                        f"improved while the shape of the output changed — check the board "
                        f"before trusting today's tiers. (This does NOT block promotion.)")
        else:
            log.info(f"  [{label}] behaviour canary clean (spread x{spread_ratio:.2f}, "
                     f"{out['frac_crossed_half']:.0%} crossed)")
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"  [{label}] behaviour canary failed ({e}) — promotion is unaffected")
        return {"available": False, "error": str(e)[:200]}
    return out


#: One id per retrain RUN, shared by every model it trains. Set by mode_train.
_RETRAIN_BATCH: dict = {}


def _record_retrain(label: str, entry: dict) -> None:
    """Append one model's outcome to the run record the Telegram notice reads."""
    try:
        blob = json.loads(TRAIN_LOG.read_text(encoding="utf-8")) if TRAIN_LOG.exists() else {}
    except Exception:
        blob = {}
    run = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    blob.setdefault("runs", {}).setdefault(run[:10], {})[label] = {**entry, "at": run}
    blob["latest_run"] = run[:10]
    try:
        TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        TRAIN_LOG.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"  [{label}] could not write {TRAIN_LOG.name}: {e}")


def _train_one(valid: "pd.DataFrame", label: str, model_file,
               target: str = "over25", weights=None, feature_cols=None) -> dict:
    """Train one CHALLENGER. Whether anything serves is decided ONLY by src/promotion_gate.py.

    RETRAINING IS FREQUENT, PROMOTION IS RARE (owner 2026-10-25). Every run:

      1. `promotion_gate.review_parked` judges the challenger parked by an EARLIER retrain on its
         independent prospective period (first-sight shadow_h2h rows, graded). This is the only
         way a retrain challenger can become champion.
      2. a new candidate is trained and scored against the serving champion on the same untouched
         chronological holdout (`promotion_gate.submit_candidate`). It can at most QUALIFY and be
         frozen into the challenger slot (SHADOW_MORE). It is never promoted at birth.

    HISTORY. Until 2026-10-04 a candidate promoted when its log loss had not risen by more than a
    tolerance (0.030, briefly 0.005): 29 of 91 replayed decisions promoted a model that scored
    worse. 2026-10-04 added "do no harm" (any rise blocks); 2026-10-18 the champion high-water mark
    (src/champion_gate.py, material gain + CI on the served probabilities). 2026-10-25 the owner
    required one canonical gate with a prospective second period; the tolerance branch, the v9.1
    log-only gate and champion_gate.py were removed so no second definition of "promoted" exists.

    THE HOLDOUT is `chronological_split` — the newest 20% of rows, exactly the split train_model
    holds out — so the candidate never saw it, and the champion is scored on the identical rows.
    """
    from src import promotion_gate as PG
    log.info(f"  [{label}] {len(valid):,} rows — training ensemble (target={target})...")
    batch = _RETRAIN_BATCH.get("id", "")

    # 1. period 2 for the parked challenger (may install it; the incumbent is loaded AFTER this)
    review = PG.review_parked(label, batch=batch) or {}

    basis = "no_incumbent"
    _test_df = None
    _old_payload = None
    try:
        _, _test_df = chronological_split(valid, target)
        if Path(model_file).exists() and len(_test_df) >= 50:
            _old_payload = load_models(model_file=model_file)
            basis = "same_holdout"
    except Exception as e:                                           # noqa: BLE001
        basis = "incumbent_unscorable"
        log.warning(f"  [{label}] could not load/score the incumbent on today's holdout ({e}); "
                    f"the gate will fail closed")

    results = train_model(valid, target=target, sample_weight=weights, feature_cols=feature_cols)
    canary = _behaviour_canary(label, _old_payload, results, _test_df, target)

    # 2. period 1 for the new candidate
    gate, gate_extra = PG.submit_candidate(label, model_file, _old_payload, results, _test_df,
                                           target, basis, valid, batch=batch)
    # "promoted" in retrain_log means exactly one thing: the serving model changed this run, and
    # only the gate can change it — a parked challenger passing period 2, or an explicit BOOTSTRAP.
    reviewed_promote = review.get("verdict") == PG.PROMOTE
    promote = reviewed_promote or gate["verdict"] == PG.BOOTSTRAP
    why = (f"PROMOTE (parked {review.get('candidate_version')}): {review.get('reason', '')} | "
           if reviewed_promote else "") + f"{gate['verdict']}: {gate['reason']}"
    if promote:
        log.warning(f"  [{label}] {why}")
    else:
        log.info(f"  [{label}] {gate['verdict']} (qualified={gate.get('qualified')}, "
                 f"parked={gate.get('parked')}) — {gate['reason'][:300]}. "
                 f"{Path(model_file).name} left untouched; the champion keeps serving.")
    _record_retrain(label, {"promoted": promote, "why": why, "rows": int(len(valid)),
                            "verdict": gate["verdict"], "comparison_basis": basis,
                            "gate_version": PG.GATE_VERSION,
                            "qualified": bool(gate.get("qualified")),
                            "parked": bool(gate.get("parked")),
                            "evaluable_tracks": gate.get("evaluable_tracks") or [],
                            # SERVED-probability log loss on the largest evaluable track
                            "logloss_old": gate.get("ll_champion"),
                            "logloss_new": gate.get("ll_challenger"),
                            "logloss_basis": "served_ensemble_largest_track",
                            "ll_ci": gate.get("ll_ci"),
                            "champion_version": gate_extra.get("champion_version"),
                            "challenger_version": gate_extra.get("challenger_version"),
                            "retrain_batch_id": gate_extra.get("retrain_batch_id"),
                            "prospective_review": {k: review.get(k) for k in (
                                "verdict", "candidate_version", "prospective_status",
                                "prospective_n", "prospective_ll_delta", "reason")}
                            if review else None,
                            "canary": canary})

    try:
        payload = load_models(model_file=model_file)
        fi = get_feature_importances(payload)
    except FileNotFoundError:
        fi = pd.DataFrame()
    if not fi.empty:
        fi_file = config.MODELS_DIR / f"feature_importances_{label}.csv"
        fi.to_csv(fi_file, index=False)
        print(f"\nTop 10 features [{label}]:")
        print(fi.head(10).to_string(index=False))
    return results


def mode_train() -> tuple:
    log.info("=" * 60)
    log.info("MODE: TRAIN  (standard model + new-format model — separate)")
    from src import model_registry as _reg
    _RETRAIN_BATCH["id"] = _reg.batch_id(_reg.utcnow())
    log.info(f"  retrain batch {_RETRAIN_BATCH['id']} — every model trained now is a CHALLENGER")
    log.info("=" * 60)

    log.info("Loading historical data...")
    raw = load_all_matches()
    log.info(f"  {len(raw):,} matches | {raw['league'].nunique()} leagues | "
             f"{raw['date'].min().date()} → {raw['date'].max().date()}")

    # ── Exclude COVID seasons ────────────────────────────────────────────────
    if config.EXCLUDE_COVID_SEASONS:
        before = len(raw)
        raw = raw[~raw["season"].isin(config.COVID_SEASONS)]
        log.info(f"  Excluded COVID seasons {config.COVID_SEASONS}: "
                 f"{before - len(raw):,} rows removed → {len(raw):,} remaining")

    log.info("Engineering features...")
    feat  = build_features(raw)
    valid = feat.dropna(subset=["over25", "home_scored_last5"])
    log.info(f"  {len(valid):,} rows with full features")

    # ── Build time-decay sample weights ─────────────────────────────────────
    def _get_weights(df):
        weights = df["season"].map(
            lambda s: config.TRAINING_DECAY_WEIGHTS.get(s, config.DEFAULT_DECAY_WEIGHT)
        ).fillna(config.DEFAULT_DECAY_WEIGHT)
        return weights.values

    # ── Standard model — trained only on standard-format league data ────────
    log.info("\nTraining STANDARD model ...")
    std_valid = valid[valid["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    std_weights = _get_weights(std_valid)
    log.info(f"  Standard leagues: {sorted(std_valid['league'].unique())}")
    log.info(f"  Seasons in training: {sorted(std_valid['season'].unique())}")
    _train_one(std_valid, "standard", config.MODEL_FILE_STANDARD, weights=std_weights)

    # ── New-format model — trained only on new-format league data ───────────
    log.info("\nTraining NEW-FORMAT model ...")
    nf_valid = valid[valid["league"].isin(config.NEW_FORMAT_LEAGUES)]
    nf_weights = _get_weights(nf_valid)
    log.info(f"  New-format leagues: {sorted(nf_valid['league'].unique())}")
    # NF-ONLY feature drop (isolation: standard model keeps the full FEATURE_COLS).
    # These features are populated in NF TRAINING data but absent for NF UPCOMING fixtures
    # (xG/inside-box come from a cold API cache; HT rolling stats are standard-oriented). At
    # predict _prep imputes an all-NaN column to 0.0, the scaler turns 0.0 into z~-37, and the
    # logistic base model collapses -> NF P(over) crushed to ~0.36 (should be ~0.51). Removing
    # them from the NF model restores calibration (validated: fresh 27-feat model -> 0.53).
    _NF_DROP = {
        "home_xg_last5", "away_xg_last5", "home_insidebox_last5", "away_insidebox_last5",
        "home_ht_scored_last5", "away_ht_scored_last5",
        "home_ht_conceded_last5", "away_ht_conceded_last5",
        "home_ht_attack_str", "away_ht_attack_str", "combined_ht_goals_avg",
        "home_ht_over05_rate", "away_ht_over05_rate",
        "home_ht_over15_rate", "away_ht_over15_rate",
    }
    nf_features = [c for c in MODEL_FEATURE_COLS if c not in _NF_DROP]
    if len(nf_valid) >= config.BACKTEST_MIN_TRAIN:
        _train_one(nf_valid, "newformat", config.MODEL_FILE_NEWFORMAT,
                   weights=nf_weights, feature_cols=nf_features)
    else:
        log.warning(f"  Not enough new-format data ({len(nf_valid)} rows) — skipping new-format model")

    # ── HT models — all leagues with HT data (not just std_valid) ───────────
    # THE HT MODELS GET THEIR OWN FEATURE SET (added 2026-09-29). They used to train on the
    # STANDARD set, which is built for full-time Over/Under 2.5: corners, fouls, shot ratios,
    # full-time O/U implied probabilities, and exactly four genuinely half-time columns — each a
    # 5-match rolling binary rate with only eleven possible values. A 5-match rolling average of
    # a near-coin-flip is mostly noise; measured alone it reaches AUC 0.522.
    #
    # src/ht_features.py replaces that with a multiplicative Poisson rate model on FIRST-HALF
    # goals (league baseline x team attack x opponent defence), empirical-Bayes shrunk and
    # recency-weighted. Walk-forward on 14,041 out-of-sample fixtures it roughly DOUBLES the
    # Brier skill over the standard set (+0.53% vs +0.26% on the 0.5 line, +0.33% vs +0.22% on
    # 1.5) and lands calibration honestly: claimed 0.6975 against realised 0.7000.
    #
    # It does NOT create a bet and is not meant to. The bookmaker's own price still beats it
    # (market AUC 0.568 vs model 0.533) and blending the model into that price adds nothing.
    # See docs/HT_MODEL_UPGRADE.md.
    log.info("\nTraining HT models (all leagues with HTHG/HTAG data)...")
    try:
        from src.ht_features import build_ht_features, HT_FEATURE_COLS
        valid = build_ht_features(valid)
        _ht_cols = ([c for c in MODEL_FEATURE_COLS if c in valid.columns]
                    + [c for c in HT_FEATURE_COLS if c in valid.columns])
        log.info(f"  HT feature set: {len(_ht_cols)} columns "
                 f"({len(HT_FEATURE_COLS)} half-time specific)")
    except Exception as e:                                            # noqa: BLE001
        # Falling back keeps a retrain alive rather than shipping no HT model at all.
        log.warning(f"  HT feature build failed ({e}) — falling back to the standard set")
        _ht_cols = None
    ht_valid = valid.dropna(subset=["ht_over05", "home_ht_over05_rate"])
    log.info(f"  HT leagues: {sorted(ht_valid['league'].unique())}")
    if len(ht_valid) >= config.BACKTEST_MIN_TRAIN:
        log.info(f"  HT data: {len(ht_valid):,} rows with HT scores")
        _train_one(ht_valid, "ht_over05", config.HT_MODEL_FILE_05, target="ht_over05",
                   feature_cols=_ht_cols)
        ht_valid15 = valid.dropna(subset=["ht_over15", "home_ht_over15_rate"])
        _train_one(ht_valid15, "ht_over15", config.HT_MODEL_FILE_15, target="ht_over15",
                   feature_cols=_ht_cols)
    else:
        log.warning(f"  Not enough HT data ({len(ht_valid)} rows) — skipping HT models")

    # ── Side-market models (BTTS / O1.5 / O3.5) — all leagues with odds data ──
    # nf_valid rows now have odds_btts/over15/over35 when af_odds_history.parquet
    # has been backfilled; rows without odds simply drop out via dropna().
    log.info("\nTraining side-market models (BTTS / Over 1.5 / Over 3.5)...")
    _sm_all = pd.concat([std_valid, nf_valid], ignore_index=True)
    for target, model_file in config.SIDE_MARKETS.items():
        sm_valid = _sm_all.dropna(subset=[target])
        if len(sm_valid) >= config.BACKTEST_MIN_TRAIN:
            log.info(f"  [{target}] {len(sm_valid):,} rows  "
                     f"base rate={sm_valid[target].mean():.1%}")
            _train_one(sm_valid, target, model_file,
                       target=target, weights=_get_weights(sm_valid))
        else:
            log.warning(f"  [{target}] Not enough data ({len(sm_valid)}) — skipping")

    return feat, None


# ── PREDICT ───────────────────────────────────────────────────────────────────

def mode_predict(historical: "pd.DataFrame" = None) -> "pd.DataFrame":
    log.info("=" * 60)
    log.info("MODE: PREDICT  (v9 — drift + both-losing + postponement)")
    log.info("=" * 60)

    import pandas as pd
    from src.predict import predict_upcoming

    if historical is None:
        historical = load_all_matches()

    payload_std = load_models(model_file=config.MODEL_FILE_STANDARD)
    payload_nf  = load_models(model_file=config.MODEL_FILE_NEWFORMAT) \
                  if config.MODEL_FILE_NEWFORMAT.exists() else None

    # Side-market payloads (BTTS / O1.5 / O3.5) — optional, skip if not trained yet
    side_payloads = {}
    for target, model_file in config.SIDE_MARKETS.items():
        if model_file.exists():
            side_payloads[target] = load_models(model_file=model_file)

    preds, postponed = predict_upcoming(
        historical, payload_std,
        payload_newformat=payload_nf,
        side_payloads=side_payloads,
        days_ahead=7,
    )

    if preds is None or preds.empty:
        log.warning("No predictions generated.")
        return preds

    # ── Save full predictions ────────────────────────────────────────────────
    # Stamp WHEN and by WHAT this board was produced. predictions.csv is overwritten every
    # 5 minutes, so without this the model's path toward kickoff (T-7d ... T-1h) can only be
    # recovered by walking git history, and no row can say which model version made it.
    # Additive columns only — every reader accesses this file by name. Cannot raise.
    from src.provenance import stamp as _stamp_provenance
    preds = _stamp_provenance(preds)

    pred_file = config.OUTPUT_DIR / "predictions.csv"
    preds.to_csv(pred_file, index=False)
    log.info(f"Predictions saved → {pred_file}  ({len(preds)} fixtures)")

    # BEFORE tips are generated, so the record is of what was believed, not what was acted on.
    # Registered prospective experiments (H-MODEL-01 ...): champion re-scored RAW and the frozen
    # challenger scored on the identical rows. Read-only on `preds`; never served, tiered or
    # staked. Cannot raise.
    exp_rows = None
    try:
        import time as _time
        from src import shadow_experiments as _SE
        _t0 = _time.perf_counter()
        exp_rows = _SE.score(preds, {"standard": payload_std, "newformat": payload_nf})
        _n = _SE.append_first_sight(exp_rows)
        log.info(f"experiments: scored {len(exp_rows)} row(s), +{_n} first-sight "
                 f"in {_time.perf_counter() - _t0:.2f}s")
    except Exception as e:                                             # noqa: BLE001
        log.warning(f"experiments log skipped ({e}) — predictions and tips are unaffected")
    _log_shadow(preds, exp_rows)
    _log_h2h(preds, exp_rows)
    # Drift is information only since 2026-10-26 (owner decision, config.DRIFT_TIER_ADJUSTMENT):
    # record, at first sight, the tier the removed drift rule WOULD have given each tip. Never raises.
    from src.drift import log_shadow as _log_drift_shadow
    _n_drift = _log_drift_shadow(preds)
    if _n_drift:
        log.info(f"drift shadow: +{_n_drift} first-sight row(s) -> output/drift_shadow.csv")

    # ── Save postponed list ──────────────────────────────────────────────────
    if postponed:
        pst_file = config.OUTPUT_DIR / "postponed.csv"
        pd.DataFrame(postponed).to_csv(pst_file, index=False)
        log.info(f"Postponed fixtures → {pst_file}  ({len(postponed)} matches)")

    # ── Generate bets (SNIPER + VALUE) ───────────────────────────────────────
    bets = generate_bets(preds)

    # ── Side-market tips (BTTS / O1.5 / O3.5) ───────────────────────────────
    side_bets = _generate_side_bets(preds, config.SIDE_MARKETS)

    # ── ONE GOAL DIRECTION PER FIXTURE (owner rule 2026-10-25, src/direction_guard.py). The one point
    # where both the O/U 2.5 tips and the side-market tips are known and nothing is written yet. A tip
    # whose goal direction (any Over line = OVER, any Under line = UNDER; BTTS never conflicts) is
    # opposite to the fixture's FIRST tip is removed here, so it never reaches bets.csv / side_bets.csv
    # (Telegram), the ledgers (album, top picks) or the stake plan. First tip stands: an earlier live
    # ledger tip across runs; within one run the O/U 2.5 tip. Suppressed tips -> direction_conflicts.csv.
    # The shadow engine keeps the unguarded frames (research logs everything). Never raises.
    _bets_all, _side_all = bets, side_bets
    _had_bets, _had_side = not bets.empty, not side_bets.empty
    from src import direction_guard as _DG
    bets, side_bets, _dir_conflicts = _DG.apply(bets, side_bets)
    _DG.record(_dir_conflicts)

    if _had_side:   # pre-guard: a run whose only side tips were suppressed still rewrites side_bets.csv
        side_bets_file = config.OUTPUT_DIR / "side_bets.csv"
        # LEDGER GETS EVERYTHING, THE SEND SOURCE DOES NOT. side_bets.csv is what
        # notify_side_bets reads, so a paper league is stripped from it; the ledger keeps the
        # row so the tip is still settled, graded and CLV-tracked (config.PAPER_LEAGUES).
        config.drop_paper_leagues(side_bets).to_csv(side_bets_file, index=False)
        append_side_market_tips(side_bets)   # persist to side_bets_ledger.csv (all tiers)
        log.info(f"Side-market tips → {side_bets_file}  ({len(side_bets)} tips)")
        for mkt, label in config.SIDE_MARKET_LABELS.items():
            n = (side_bets["market"] == mkt).sum()
            log.info(f"  {label}: {n} tips")

    # ── market_edge_v2 SHADOW (2026-10-25). The market-first engine (src/market_edge.py) scores
    # every pre-kickoff fixture x market BESIDE the legacy decision and appends first-sight rows to
    # output/edge_engine_shadow.csv (+ edge_engine_health.json). It reads `preds`, `side_bets` and
    # the per-book quotes; it never changes a tip, tier, threshold, stake or message, and it
    # cannot raise. Its per-side output is attached to the stake plan as trailing shadow columns.
    _edge_shadow = None
    try:
        from src import edge_engine_shadow as _EES
        _edge_shadow = _EES.run(preds, _bets_all, _side_all, historical=historical)
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"edge shadow skipped ({e}) — tips are unaffected")

    # ── Stake plan (2026-10-14): the approved bankroll policy applied to this run's tips. Output
    # files only (stake_plan.csv, stake_audit.csv); no tip, tier, message or bet changes. Paper
    # leagues and PAPER tracks get a zero stake with the reason recorded. Never raises.
    try:
        from src.stake_plan import write as _write_stake_plan
        from src.ledger import drop_flipped as _drop_flipped
        # drop_flipped: an O/U 2.5 tip on the opposite side of an earlier O/U 2.5 tip is never sent or
        # ledgered (existing flip guard), so it is not staked either (direction rule 2026-10-25).
        _write_stake_plan(_drop_flipped(bets), side_bets, shadow=_edge_shadow)
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"stake plan skipped: {e}")

    # ── HT (half-time O/U) tips -> ht_ledger.csv (open->close->CLV->result) ──
    try:
        append_ht_tips(preds)
        # EVERY scored fixture, not only the tipped ones. A tip fires on ~14% of fixtures, so
        # a tips-only record reaches a thousand evaluable rows in about 20 months; logging all
        # of them takes about 12 weeks — and keeps the fixtures that test whether the tip
        # threshold means anything, which filtering first throws away.
        append_ht_observations(preds)
    except Exception as e:
        log.warning(f"append_ht_tips skipped: {e}")

    if _had_bets:   # pre-guard: a run whose only tips were suppressed still rewrites bets.csv (empty)
        bets_file = config.OUTPUT_DIR / "bets.csv"
        # bets.csv is the SEND SOURCE (notify_new_snipers, agent analysis, dashboard live tips,
        # the props supplement), so paper leagues are stripped from it. The ledger below still
        # receives every row, paper leagues included, so they settle and accrue CLV.
        _send = config.drop_paper_leagues(bets)
        # One side per fixture: a tip on the opposite side of one already sent is never sent.
        from src.ledger import drop_flipped
        _send = drop_flipped(_send)
        _send.to_csv(bets_file, index=False)
        log.info(f"Tips saved → {bets_file}  ({len(_send)} tips"
                 + (f", {len(bets) - len(_send)} paper-league tip(s) ledgered only"
                    if len(bets) != len(_send) else "") + ")")

        # Append to persistent ledger (never overwrites, deduped per fixture)
        append_tips(bets)

        display = [
            "date", "league", "home_team", "away_team",
            "best_side", "odds_over25", "odds_under25",
            "best_edge", "bet", "signal_tier",
            "drift_signal", "over_drift", "under_drift",
        ]
        display = [c for c in display if c in bets.columns]

        for tier, label, note in [
            ("SNIPER",   "SNIPER    (per-league threshold)", "full stake"),
            ("MARKSMAN", "MARKSMAN  (edge 8%-threshold)",   "3/4 stake"),
            ("VALUABLE", "VALUABLE  (edge 4-8%)",           "half stake / monitor"),
        ]:
            rows = bets[bets["signal_tier"] == tier]
            print("\n" + "=" * 90)
            print(f"  {label}  — {note}  [{len(rows)} tips]")
            print("=" * 90)
            if rows.empty:
                print("  (none this week)")
            else:
                print(rows[display].to_string(index=False))

        # Both-losing summary
        bl_count = int(preds.get("both_losing", pd.Series(False)).sum()) if "both_losing" in preds.columns else 0

        print(f"\n  SNIPER      : {(bets['signal_tier'] == 'SNIPER').sum()}")
        print(f"  VALUE       : {(bets['signal_tier'] == 'VALUE').sum()}")
        print(f"  AVOID       : {len(preds) - len(bets)} fixtures below threshold")
        if bl_count:
            print(f"  BOTH-LOSING : {bl_count} fixtures suppressed (v7 guard active)")
        if postponed:
            print(f"  POSTPONED   : {len(postponed)} fixtures disappeared since last run")

        # Drift summary
        if "drift_signal" in bets.columns:
            print(f"\n  Drift breakdown (tips only):")
            print(bets["drift_signal"].value_counts().to_string())
    else:
        print("\n  No tips this week (all fixtures below 0.04 edge).")
        if postponed:
            print(f"  {len(postponed)} fixture(s) postponed since last run.")

    return preds


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def _run_one_backtest(feat: "pd.DataFrame", leagues: set, label: str, out_prefix: str) -> tuple:
    results_df, summary, league_summary = run_backtest(feat, enabled_leagues=leagues)

    bt_file = config.OUTPUT_DIR / f"backtest_results_{out_prefix}.csv"
    results_df.to_csv(bt_file, index=False)
    lg_file = config.OUTPUT_DIR / f"backtest_by_league_{out_prefix}.csv"
    league_summary.to_csv(lg_file, index=False)

    print("\n" + "=" * 60)
    print(f"  BACKTEST SUMMARY — {label}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:35s}: {v}")
    print(f"\n  BY LEAGUE [{label}]:")
    print(league_summary.to_string(index=False))

    return results_df, summary


def mode_backtest(feat: "pd.DataFrame" = None, only: str = None) -> tuple:
    """Walk-forward backtest, standard + new-format computed SEPARATELY (already isolated).
    only=None      -> both (+ side markets) — default, unchanged (monthly cron behaviour).
    only='standard'-> standard 2.5 + side markets only (new-format skipped).
    only='newformat'-> new-format only (standard + side markets skipped; standard results
                       CSV is NOT regenerated -> zero effect on the standard model path)."""
    log.info("=" * 60)
    log.info(f"MODE: BACKTEST  (walk-forward; only={only or 'all'})")
    log.info("=" * 60)

    if feat is None:
        raw  = load_all_matches()
        feat = build_features(raw)

    std_leagues = config.STANDARD_FORMAT_LEAGUES & config.ENABLED_LEAGUES
    nf_leagues  = config.NEW_FORMAT_LEAGUES  & config.ENABLED_LEAGUES

    std_results = std_summary = nf_results = nf_summary = None
    if only in (None, "standard"):
        log.info(f"Standard leagues: {sorted(std_leagues)}")
        std_results, std_summary = _run_one_backtest(feat, std_leagues, "STANDARD", "standard")
    if only in (None, "newformat"):
        log.info(f"New-format leagues: {sorted(nf_leagues)}")
        nf_results, nf_summary = _run_one_backtest(feat, nf_leagues, "NEW-FORMAT", "newformat")

    # Combined ledger file only when BOTH ran (don't clobber it on a per-model run)
    if std_results is not None and nf_results is not None:
        import pandas as pd
        combined = pd.concat([std_results, nf_results], ignore_index=True)
        combined.to_csv(config.OUTPUT_DIR / "backtest_results.csv", index=False)

    # ── Standard O/U 2.5 per-league SNIPER-threshold optimizer ─────────────────
    # Auto-tunes per-league thresholds (the standard-model equivalent of
    # optimize_side_market_thresholds; standard historically used hand-set constants).
    #
    # PROPOSE, NEVER APPLY (owner rule 2026-10-26). The result is a CANDIDATE written under
    # output/threshold_proposals/candidates/ — models/best_params_standard.json, which production
    # reads, is written only by src/threshold_apply.py from an owner-approved proposal entry.
    if std_results is not None:
        from src.threshold_proposals import write_candidate
        std_th = optimize_standard_thresholds(std_results)
        _cf = write_candidate("standard", std_th)
        appr = [lg for lg, v in std_th.items() if v.get("approved")]
        log.info(f"Standard per-league thresholds -> CANDIDATE {_cf} ({len(appr)} optimizer-approved: "
                 f"{appr or 'none'}) — not served until the owner approves a proposal")

    # Single-model run (only='standard' or 'newformat'): skip the side-market backtests
    # entirely — they're a separate concern, run them alone via `--mode backtest-side --market X`.
    # This is what makes per-model runs fast (no side-market walk-forwards bundled in).
    if only is not None:
        return (std_results, std_summary) if only == "standard" else (nf_results, nf_summary)

    # ── Side-market backtests (BTTS / Over 1.5 / Over 3.5) ──────────────────
    log.info("\nBacktesting side markets (BTTS / Over 1.5 / Over 3.5)...")
    import json
    all_side_thresholds = {}  # {target: {league: {sniper_th, marksman_th, roi, bets, drop}}}

    for target, label in config.SIDE_MARKET_LABELS.items():
        try:
            sm_results, sm_summary, sm_league = run_side_market_backtest(
                feat, target=target, enabled_leagues=std_leagues
            )
            bt_file = config.OUTPUT_DIR / f"backtest_results_{target}.csv"
            sm_results.to_csv(bt_file, index=False)
            lg_file = config.OUTPUT_DIR / f"backtest_by_league_{target}.csv"
            sm_league.to_csv(lg_file, index=False)

            print(f"\n{'=' * 60}")
            print(f"  BACKTEST — {label}")
            print(f"{'=' * 60}")
            for k, v in sm_summary.items():
                print(f"  {k:35s}: {v}")
            if not sm_league.empty:
                print(f"\n  BY LEAGUE [{label}]:")
                print(sm_league.to_string(index=False))

            # ── Per-league threshold optimization ────────────────────────────
            log.info(f"\nOptimizing per-league thresholds for {label}...")
            league_ths = optimize_side_market_thresholds(sm_results, target=target)
            all_side_thresholds[target] = league_ths

            print(f"\n  PER-LEAGUE THRESHOLDS [{label}]:")
            print(f"  {'League':<30} {'SNIPER th':>10} {'MARKSMAN th':>12} {'ROI%':>8} {'Bets':>6} {'Drop':>6}")
            print(f"  {'-'*72}")
            for lg, v in sorted(league_ths.items()):
                roi_str = f"{v['roi']:.1f}%" if v['roi'] is not None else "—"
                print(f"  {lg:<30} {v['sniper_th']:>10.2f} {v['marksman_th']:>12.2f} "
                      f"{roi_str:>8} {v['bets']:>6} {'DROP' if v['drop'] else '':>6}")

        except Exception as e:
            log.warning(f"  [{target}] Backtest failed: {e}")

    # Save all side-market thresholds as a CANDIDATE (owner rule 2026-10-26: propose, never apply)
    if all_side_thresholds:
        from src.threshold_proposals import write_candidate
        th_file = write_candidate("side_markets", all_side_thresholds)
        log.info(f"Side-market thresholds -> CANDIDATE {th_file} (not served until approved)")

    return std_results, std_summary


# ── Entry point ───────────────────────────────────────────────────────────────

def mode_backtest_side(feat: "pd.DataFrame" = None, market: str = None) -> None:
    """Run ONLY the side-market backtests (BTTS / Over 1.5 / Over 3.5).
    Skips the standard 2.5 walk-forward backtest entirely.
    market: if set, run only that one market (used for parallel dispatch).
    """
    import json, subprocess, sys
    log.info("=" * 60)
    log.info("MODE: BACKTEST-SIDE  (BTTS / Over 1.5 / Over 3.5 only)")
    log.info("=" * 60)

    # If no specific market requested, spawn 3 parallel subprocesses (3× faster)
    if market is None:
        procs = []
        for t in config.SIDE_MARKET_LABELS:
            p = subprocess.Popen(
                [sys.executable, __file__, "--mode", "backtest-side", "--market", t],
                cwd=Path(__file__).parent,
            )
            procs.append((t, p))
            log.info(f"Spawned backtest-side --market {t} (PID {p.pid})")
        for t, p in procs:
            p.wait()
            log.info(f"[{t}] subprocess finished (exit {p.returncode})")
        # Merge the per-market CANDIDATES into one combined candidate (owner rule 2026-10-26:
        # the backtest proposes; models/best_params_side_markets.json is written only by
        # src/threshold_apply.py from an owner-approved proposal).
        from src.threshold_proposals import CANDIDATE_DIR, write_candidate
        all_side_thresholds = {}
        for t in config.SIDE_MARKET_LABELS:
            f = CANDIDATE_DIR / f"best_params_{t}.json"
            if f.exists():
                all_side_thresholds[t] = json.loads(f.read_text())
        if all_side_thresholds:
            th_file = write_candidate("side_markets", all_side_thresholds)
            log.info(f"Combined threshold CANDIDATE saved → {th_file} (not served until approved)")
        return

    if feat is None:
        raw  = load_all_matches()
        feat = build_features(raw)

    std_leagues = config.STANDARD_FORMAT_LEAGUES & config.ENABLED_LEAGUES
    all_side_thresholds = {}
    markets_to_run = {market: config.SIDE_MARKET_LABELS[market]} if market else config.SIDE_MARKET_LABELS

    for target, label in markets_to_run.items():
        try:
            sm_results, sm_summary, sm_league = run_side_market_backtest(
                feat, target=target, enabled_leagues=std_leagues
            )
            (config.OUTPUT_DIR / f"backtest_results_{target}.csv").parent.mkdir(exist_ok=True)
            sm_results.to_csv(config.OUTPUT_DIR / f"backtest_results_{target}.csv", index=False)
            sm_league.to_csv(config.OUTPUT_DIR / f"backtest_by_league_{target}.csv", index=False)

            print(f"\n{'=' * 60}")
            print(f"  BACKTEST — {label}")
            print(f"{'=' * 60}")
            for k, v in sm_summary.items():
                print(f"  {k:35s}: {v}")
            if not sm_league.empty:
                print(f"\n  BY LEAGUE [{label}]:")
                print(sm_league.to_string(index=False))

            log.info(f"\nOptimizing per-league thresholds for {label}...")
            league_ths = optimize_side_market_thresholds(sm_results, target=target)
            all_side_thresholds[target] = league_ths

            print(f"\n  PER-LEAGUE THRESHOLDS [{label}]:")
            print(f"  {'League':<30} {'SNIPER th':>10} {'MARKSMAN th':>12} {'ROI%':>8} {'Bets':>6} {'Drop':>6}")
            print(f"  {'-'*72}")
            for lg, v in sorted(league_ths.items()):
                roi_str = f"{v['roi']:.1f}%" if v['roi'] is not None else "—"
                print(f"  {lg:<30} {v['sniper_th']:>10.2f} {v['marksman_th']:>12.2f} "
                      f"{roi_str:>8} {v['bets']:>6} {'DROP' if v['drop'] else '':>6}")

        except Exception as e:
            log.warning(f"  [{target}] Backtest failed: {e}")

    if all_side_thresholds:
        # CANDIDATES only (owner rule 2026-10-26) — see src/threshold_proposals.py.
        from src.threshold_proposals import write_candidate
        if market:
            m_file = write_candidate(market, all_side_thresholds.get(market, {}))
            log.info(f"[{market}] threshold CANDIDATE saved → {m_file} (not served until approved)")
        else:
            th_file = write_candidate("side_markets", all_side_thresholds)
            log.info(f"Side-market threshold CANDIDATE saved → {th_file} (not served until approved)")


def main():
    parser = argparse.ArgumentParser(description="v9 Football Prediction Pipeline (hybrid v7+v8)")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "backtest", "backtest-side", "all"],
        default="predict",
        help="Pipeline mode (default: predict)",
    )
    parser.add_argument(
        "--market",
        choices=["btts", "over15", "over35"],
        default=None,
        help="Run backtest-side for a single market only (used internally for parallel dispatch)",
    )
    parser.add_argument(
        "--only",
        choices=["standard", "newformat"],
        default=None,
        help="Backtest ONE model family only (isolation): 'newformat' skips standard entirely "
             "(standard results not regenerated); 'standard' skips new-format. Default: both.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        mode_train()
    elif args.mode == "predict":
        mode_predict()
    elif args.mode == "backtest":
        mode_backtest(only=args.only)
    elif args.mode == "backtest-side":
        mode_backtest_side(market=args.market)
    elif args.mode == "all":
        feat, _ = mode_train()
        mode_predict(load_all_matches())
        mode_backtest(feat)



if __name__ == "__main__":
    main()
