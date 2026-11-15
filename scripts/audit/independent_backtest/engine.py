"""Chronological first-sight replay over committed prediction boards.

Every committed predictions.csv board is a frozen decision point. Boards are processed in decision-
time order; for each fixture x market the independent rule decides; the FIRST board that selects a
(fixture, market, side) is that bet's single first-sight decision (price, edge, tier frozen there).
Duplicates on later boards are dropped. Two derived books:
  * `firstsight`  one row per (fixture, market, side)   -> reconciles against the ledgers' own key
  * `policy`      one position per fixture per goal DIRECTION (O/U 2.5 first, first-sight wins);
                  this is the owner rule (flip guard 2026-10-11, direction guard 2026-10-25) applied
                  to the WHOLE history so the independent ROI never double counts a fixture.
"""
from __future__ import annotations

import pandas as pd

from .regimes import RegimeTimeline
from .selection import SIDE_MARKETS, decide_ou25, decide_side, goal_direction, snapshot_time

MARKETS = ["ou25", *SIDE_MARKETS]


def replay(snaps: pd.DataFrame, timeline: RegimeTimeline, keep_board_universe: bool = True):
    snaps = snaps.copy()
    snaps["fx_date"] = snaps["date"].astype(str).str[:10]
    recs = snaps.to_dict("records")
    for r in recs:
        r["_t"] = snapshot_time(r)
    recs.sort(key=lambda r: (r["_t"], str(r["snap_sha"])))
    first: dict = {}
    universe: dict = {}     # (fixture, market) -> first priced board (placebo sampling frame)
    board_last_t: dict = {}  # fixture -> list of board times seen (for "was on a board" checks)
    decisions_count: dict = {}
    reg_cache: dict = {}
    for r in recs:
        t = r["_t"]
        sha = r["snap_sha"]
        if sha not in reg_cache:
            reg_cache[sha] = timeline.at(t)
        reg = reg_cache[sha]
        if reg is None:
            continue
        fx = (r["fx_date"], str(r["home_team"]), str(r["away_team"]))
        board_last_t.setdefault(fx, []).append(t)
        for mkt in MARKETS:
            d = decide_ou25(r, reg, t) if mkt == "ou25" else decide_side(r, mkt, reg, t)
            decisions_count[(mkt, d["reason"] if not d["selected"] else "SELECTED")] = \
                decisions_count.get((mkt, d["reason"] if not d["selected"] else "SELECTED"), 0) + 1
            if keep_board_universe and (fx, mkt) not in universe and d["reason"] not in (
                    "unpriced", "no_model_probability", "post_kickoff_quote"):
                if mkt == "ou25":
                    universe[(fx, mkt)] = {"fx_date": fx[0], "home_team": fx[1], "away_team": fx[2],
                                           "league": r["league"], "model_type": r.get("model_type"),
                                           "market": mkt, "t": t, "odds_OVER": r.get("odds_over25"),
                                           "odds_UNDER": r.get("odds_under25"), "p_over": r.get("p_over25")}
                else:
                    universe[(fx, mkt)] = {"fx_date": fx[0], "home_team": fx[1], "away_team": fx[2],
                                           "league": r["league"], "model_type": r.get("model_type"),
                                           "market": mkt, "t": t, f"odds_{SIDE_MARKETS[mkt]}": r.get(f"odds_{mkt}"),
                                           "p_yes": r.get(f"p_{mkt}")}
            if not d["selected"]:
                continue
            key = (fx, mkt, d["side"])
            if key in first:
                first[key]["n_boards_selected"] += 1
                continue
            first[key] = {"fx_date": fx[0], "home_team": fx[1], "away_team": fx[2], "league": r["league"],
                          "model_type": r.get("model_type"), "market": mkt, "side": d["side"],
                          "odds": d["odds"], "edge": d["edge"], "tier": d["tier"], "t_first": t,
                          "snap_sha": sha, "regime_sha": reg.since_sha, "kickoff_utc": r.get("kickoff_utc"),
                          "blind": d.get("blind", False), "model_version": r.get("model_version"),
                          "n_boards_selected": 1,
                          "prod_board_bet": r.get("bet") if mkt == "ou25" else None,
                          "prod_board_tier": r.get("signal_tier") if mkt == "ou25" else None,
                          "p_model": (r.get("p_over25") if d["side"] == "OVER" else 1 - float(r.get("p_over25")))
                          if mkt == "ou25" else r.get(f"p_{mkt}"),
                          "odds_opposite": (r.get("odds_under25") if d["side"] == "OVER" else r.get("odds_over25"))
                          if mkt == "ou25" else (r.get("odds_btts_no") if mkt == "btts" else None)}
    fs = pd.DataFrame(list(first.values()))
    fs["_mo"] = fs["market"].map({"ou25": 0, "btts": 1, "over15": 2, "over35": 3})
    fs = fs.sort_values(["t_first", "_mo"], kind="mergesort").drop(columns="_mo")
    # policy book: one direction per fixture, O/U 2.5 one side per fixture, first sight wins
    kept, fx_dir, fx_ou = [], {}, {}
    for r in fs.to_dict("records"):
        fx = (r["fx_date"], r["home_team"], r["away_team"])
        if r["market"] == "ou25":
            if fx in fx_ou and fx_ou[fx] != r["side"]:
                r["policy_drop"] = "flip_opposite_ou25"
                kept.append(r)
                continue
            fx_ou.setdefault(fx, r["side"])
        dirn = goal_direction(r["market"], r["side"])
        if dirn is not None:
            if fx in fx_dir and fx_dir[fx] != dirn:
                r["policy_drop"] = "opposite_goal_direction"
                kept.append(r)
                continue
            fx_dir.setdefault(fx, dirn)
        r["policy_drop"] = ""
        kept.append(r)
    fs = pd.DataFrame(kept)
    uni = pd.DataFrame(list(universe.values()))
    dec = pd.DataFrame([{"market": k[0], "outcome": k[1], "n": v} for k, v in decisions_count.items()])
    boards = {fx: (min(ts), max(ts), len(ts)) for fx, ts in board_last_t.items()}
    return fs, uni, dec, boards
