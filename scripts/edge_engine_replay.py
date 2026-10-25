"""RETROSPECTIVE legacy-vs-market_edge_v2 replay at first sight. Research only - changes nothing served.

    python scripts/edge_engine_replay.py [--cache PATH] [--since 2026-08-14] [--boot 2000]

Writes (all labelled RETROSPECTIVE):
    output/edge_engine_backtest.csv          one row per fixture x family at first sight
    output/edge_engine_comparison.json       per track x market comparison, disagreement test, current board
    output/edge_engine_proposed_params.json  PROPOSED residual weights / de-vig (discovery fit, holdout scored)

config/market_edge_v2.json is READ, never written: the production shadow config stays as committed.
See src/research/edge_replay.py for every definition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import market_edge as ME  # noqa: E402
from src.research import edge_replay as R  # noqa: E402

OUT = config.OUTPUT_DIR
BACKTEST = OUT / "edge_engine_backtest.csv"
COMPARISON = OUT / "edge_engine_comparison.json"
PROPOSED = OUT / "edge_engine_proposed_params.json"
SHADOW = OUT / "edge_engine_shadow.csv"


def config_hash(path: Path = ME.CONFIG_PATH) -> str:
    """sha256 of the file as git stores it (LF), so a Windows CRLF checkout and the GitHub raw bytes
    hash identically. Pro registry/edge2_hypotheses_v1.json freezes this value."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current_board(params: dict) -> dict:
    """(b): legacy selections on the committed board vs the NEW engine at the board's own run time."""
    from src.edge_engine_shadow import _quote_frame, book_pairs, evaluate_board
    board = pd.read_csv(OUT / "predictions.csv", low_memory=False)
    gen = pd.to_datetime(board["generated_at"], errors="coerce", utc=True).max()
    now = gen if pd.notna(gen) else pd.Timestamp.now(tz="UTC")
    sp = json.loads((config.MODELS_DIR / "best_params_side_markets.json").read_text(encoding="utf-8"))
    side = []
    for _, r in board.iterrows():
        for fam, mkt in (("BTTS", "btts"), ("OU15", "over15")):
            s = R.FAMILY_SPEC[fam]
            tier, edge = R.legacy_side_tier(r.get(s["p"]), r.get(s["a"]), str(r["league"]), mkt, sp)
            if tier in R.TIERS:
                side.append({"date": r["date"], "home_team": r["home_team"], "away_team": r["away_team"],
                             "market": mkt, "signal_tier": tier})
    side_df = pd.DataFrame(side)
    q = _quote_frame(now, [], OUT / "book_odds_snapshots.csv")
    rows, _ = evaluate_board(board, side_df, now, book_pairs(q, now), params, {}, {}, None)
    rows = rows[rows["market"].isin(["OU25", "BTTS", "OU15"])]
    sel = rows[rows["legacy_tier"].isin(R.TIERS)]
    res = {"board_generated_at": str(now), "board_fixtures": int(board.shape[0]),
           "pre_kickoff_fixture_market_rows": int(len(rows)), "legacy_selections": int(len(sel)),
           "legacy_selections_by_market": sel["market"].value_counts().to_dict(),
           "legacy_selections_by_tier": sel["legacy_tier"].value_counts().to_dict(),
           "new_state_of_legacy_selections": sel["new_state"].value_counts().to_dict()}
    priced = sel[sel["new_side"].astype(str) != ""]
    res["side_changes"] = int((priced["new_side"] != priced["legacy_side"]).sum())
    res["become_NO_BET"] = int((sel["new_state"] == "NO_BET").sum())
    res["become_OBSERVE"] = int((sel["new_state"] == "OBSERVE").sum())
    res["stay_QUALIFIED_or_STRONG"] = int(sel["new_state"].isin(["QUALIFIED", "STRONG"]).sum())
    res["new_ev_lb_positive_among_legacy"] = int((pd.to_numeric(sel["ev_lower_bound"], errors="coerce") > 0).sum())
    res["new_ev_lb_positive_all_rows"] = int((pd.to_numeric(rows["ev_lower_bound"], errors="coerce") > 0).sum())
    res["no_bet_reasons"] = (sel.loc[sel["new_state"] == "NO_BET", "new_reason"].astype(str).str.split(";").str[0]
                             .str.replace(r"\(.*", "", regex=True).value_counts().to_dict())
    res["by_market"] = {m: {"legacy": int(len(g)), "states": g["new_state"].value_counts().to_dict(),
                            "side_changes": int(((g["new_side"] != g["legacy_side"]) & (g["new_side"].astype(str) != "")).sum())}
                        for m, g in sel.groupby("market")}
    # the committed first-sight shadow log, as a second view
    if SHADOW.exists():
        sh = pd.read_csv(SHADOW, low_memory=False)
        ko = pd.to_datetime(sh["kickoff_utc"], errors="coerce", utc=True)
        up = sh[ko > now]
        ls = up[up["legacy_tier"].isin(R.TIERS)]
        res["shadow_first_sight_view"] = {
            "rows": int(len(sh)), "upcoming_rows": int(len(up)), "legacy_selections": int(len(ls)),
            "new_state_of_legacy_selections": ls["new_state"].value_counts().to_dict(),
            "side_changes": int(((ls["new_side"] != ls["legacy_side"]) & ls["new_side"].notna()).sum()),
            "states_all_upcoming": up["new_state"].value_counts().to_dict()}
    return res


def shadow_counts() -> dict:
    if not SHADOW.exists():
        return {"rows": 0}
    sh = pd.read_csv(SHADOW, low_memory=False)
    return {"rows": int(len(sh)), "first_prediction_ts": str(sh["prediction_ts"].min()),
            "last_prediction_ts": str(sh["prediction_ts"].max()),
            "by_track_market": {f"{t}|{m}": {"n": int(len(g)), "states": g["new_state"].value_counts().to_dict(),
                                             "two_sided": int(g["market_source"].isin(
                                                 ["MEDIAN_TWO_SIDED_BOOKS", "SINGLE_TWO_SIDED_BOOK"]).sum())}
                                for (t, m), g in sh.groupby(["model_type", "market"])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "wowza_edge_replay_board.parquet"))
    ap.add_argument("--since", default="2026-08-14")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)

    params = ME.load_params()
    board = R.board_history(R.REPO, a.since, Path(a.cache))
    quotes = R.load_book_quotes(OUT)
    results = R.load_results(OUT)
    bl = pd.read_csv(OUT / "bets_ledger.csv", low_memory=False)
    sl = pd.read_csv(OUT / "side_bets_ledger.csv", low_memory=False)
    df = R.replay(board, quotes, results, bl, sl, params)
    df.to_csv(BACKTEST, index=False)

    cells, dis, prop = {}, {}, {}
    for track, fam in R.CELLS:
        g = df[(df["track"] == track) & (df["family"] == fam)]
        key = f"{track}|{fam}"
        cells[key] = R.compare_cell(g) if len(g) else {"status": "INSUFFICIENT", "reason": "no first-sight rows"}
        dis[key] = R.disagreement_cell(g, a.boot) if len(g) else {"status": "INSUFFICIENT", "n": 0}
        prop[key] = (R.fit_params_cell(g, float(params["residual_logit_clip"]),
                                       float(params["residual_weight"].get(key, 0.10)))
                     if len(g) else {"status": "INSUFFICIENT", "n": 0})
    # BH-FDR across every (cell x metric) moderate-vs-extreme test
    flat = [(k, m, t["p_one_sided"]) for k, v in dis.items() if v.get("status") == "OK"
            for m, t in v["tests"].items() if t.get("p_one_sided") is not None]
    qs = R.bh([p for *_, p in flat])
    for (k, m, _), q in zip(flat, qs):
        dis[k]["tests"][m]["bh_q"] = q
    # BH-FDR across cells for the NEW_EVLB ROI (one-sided t, >0)
    from scipy import stats
    roi_p = []
    for k, v in cells.items():
        g = df[(df["track"] + "|" + df["family"] == k) & df["two_sided"]]
        x = R.settle_bets(g, "NEW_EVLB") if len(g) else pd.DataFrame()
        p = float(stats.ttest_1samp(x["pnl"], 0, alternative="greater").pvalue) if len(x) >= 10 and x["pnl"].std() > 0 else None
        roi_p.append((k, p))
    for (k, p), q in zip(roi_p, R.bh([p for _, p in roi_p])):
        if "engines_two_sided_universe" in cells[k]:
            cells[k]["engines_two_sided_universe"]["NEW_EVLB"]["p_one_sided_roi_gt_0"] = p
            cells[k]["engines_two_sided_universe"]["NEW_EVLB"]["bh_q"] = q

    # BH-FDR across the proposed-parameter holdout tests (one per cell)
    pk = [k for k, v in prop.items() if v.get("status") == "PROPOSED_DISCOVERY_ONLY"]
    pq = R.bh([prop[k]["holdout_fitted_minus_market_ll"].get("p_one_sided_fitted_better") for k in pk])
    for k, q in zip(pk, pq):
        prop[k]["holdout_fitted_minus_market_ll"]["bh_q"] = q

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comp = {
        "label": R.LABEL,
        "note": "RETROSPECTIVE replay. Can describe, never confirm. Forward evidence: Pro registry/edge2_hypotheses_v1.json.",
        "generated_at": now, "engine_version": ME.ENGINE_VERSION,
        "config_path": "config/market_edge_v2.json", "config_sha256": config_hash(),
        "config_status": params.get("status"),
        "engine_code_sha256": config_hash(ROOT / "src" / "market_edge.py"),
        "inputs": {"board_runs": int(board["commit_sha"].nunique()), "board_rows": int(len(board)),
                   "book_quote_rows": int(len(quotes)), "result_rows": int(len(results)),
                   "replay_rows": int(len(df)), "settled_rows": int(df["y_a"].notna().sum())},
        "definitions": {
            "first_sight": "earliest committed predictions.csv run strictly before kickoff with the family's p and a real board price",
            "legacy_ou25": "tier RECORDED by production at first sight (signal_tier/best_side); reproduced from the threshold regime for validation",
            "legacy_side": "p-(1/odds)/1.08 with 0.10/0.08/0.04 bars unless league approved in best_params_side_markets.json at the time; BTTS-YES / OVER only",
            "NEW_QUALIFIED": "market_edge_v2 decision_state QUALIFIED/STRONG, committed UNVALIDATED_DEFAULTS",
            "NEW_EVLB": "market_edge_v2 value path: ev_lower_bound>0, >=2 books, odds<=max_odds, no extreme-disagreement cap (evidence gate ignored)",
            "NEW_OBSERVE_POOL": "any non-NO_BET state with a two-sided consensus (ev_point>0) - NOT a bet recommendation, shown for completeness",
            "two_sided_universe": "first-sight rows where >=1 same-book two-sided pair existed at the entry instant",
            "clv": "certified only: book write within 30 min before kickoff, LOCF per-book pairs at kickoff, power de-vig median of >=3 books; CLV% = entry_odds*p_close_fair-1",
            "roi_ci": "matchday (kickoff date) block bootstrap, 2000 resamples",
            "thresholds_in_force": "predict.yml env history: one change b82cec9f 2026-08-21T16:06:04Z (VALUABLE .04->.03, MARKSMAN .14->.08, league SNIPER cap 1.0->.12); best_params_* read from git at entry time",
        },
        "legacy_vs_ledger_same_price": R.legacy_vs_ledger(df, bl, sl),
        "cells": cells,
        "disagreement_test": dis,
        "current_board": current_board(params),
        "edge_engine_shadow_counts": shadow_counts(),
    }
    COMPARISON.write_text(json.dumps(comp, indent=2, default=str), encoding="utf-8")
    PROPOSED.write_text(json.dumps({
        "status": "PROPOSED", "label": R.LABEL,
        "warning": "NOT production config. config/market_edge_v2.json is unchanged. Discovery-fitted values, scored on a "
                   "later holdout. Applying any of them requires owner review and a PR - and they would not change the "
                   "prospective H-EDGE2 tests, which are frozen to the CURRENT config hash.",
        "generated_at": now, "current_config_sha256": config_hash(),
        "method": "chronological split per track x family (first 60% of settled two-sided first-sight rows = discovery, "
                  "rest = holdout); de-vig method chosen by discovery market log loss; residual weight w chosen on a "
                  "0..1 grid (0.05) by discovery log loss of sigmoid(logit p_mkt + w*clip(raw, +-0.35)); holdout scored once.",
        "cells": prop}, indent=2, default=str), encoding="utf-8")
    for k, v in cells.items():
        e = v.get("engines_two_sided_universe", {})
        print(k, v.get("status"), {n: (m.get("n"), m.get("roi_pct")) for n, m in e.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
