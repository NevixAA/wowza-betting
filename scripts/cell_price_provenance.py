"""PRICE PROVENANCE of every cell of a bars proposal (owner 2026-11-11 "Strict, keep Argentina") - counts only.

    python scripts/cell_price_provenance.py --proposal output/threshold_proposals/bars_oddsapi_hist/2026-11.json \
        --aliases <loader identity map json> [--priced <local priced universe dir>] --before 2026-11-01

For each cell (track x market-side x league) the priced rows its bar selection / verdict evidence could read
(the committed own-price study universes, rows dated before --before), split by HOW the price's capture time is
known:
  football_data_prematch  football-data pre-match same-book pair (published before kickoff)       PROVEN
  own_forward             our own live capture, snapshot_ts < kickoff                               PROVEN
  own_backfill            snapshot stamped 00:00 on match day (price_provenance BACKFILL_HISTORICAL):
                          the real capture time is NOT known                                        UNPROVEN
  oddsapi_hist            bought history priced the row (fresh-holdout cells; B1 = "close" ~9 min
                          before kickoff, i.e. a closing price used as the bet price)               NOT BET-TIME
plus `unproven_share` = (own_backfill + oddsapi_hist) / priced rows. Writes the JSON beside the proposal
(statistics only). The data-truth gate (rule 8) reads it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import league_bars as LB  # noqa: E402
from src.bars_opt import study as S  # noqa: E402
from src.bars_opt import universe as U  # noqa: E402

FRESH = "FRESH_HOLDOUT_SELECTION"


def proposal_cells(p: dict) -> list[dict]:
    out = []
    for e in p["entries"]:
        for side, c in ((e.get("proposed") or {}).get("bars_opt") or {}).items():
            out.append({"entry_id": e["entry_id"], "side": side, "track": c["track"], "mside": c["mside"],
                        "league": c["league"], "verdict": c["verdict"], "basis": c.get("config_basis"),
                        "mode": c.get("mode")})
    return out


def provenance(cells: list[dict], aliases: dict, before: str, priced_dir: Path | None) -> list[dict]:
    own = {t: pd.read_parquet(ROOT / "output" / "bars_opt" / f"universe_{t}.parquet") for t in ("standard", "new_format")}
    hist = {}
    if priced_dir is not None:
        hist = {t: pd.read_parquet(Path(priced_dir) / f"universe_{t}.parquet",
                                   columns=["row_id", "market", "league", "price_source"]) for t in own}
    snap, rows = {}, []
    for c in cells:
        tr, ms, lg = c["track"], c["mside"], c["league"]
        mk = S.MARKET_SIDES[ms][0]
        u = own[tr]
        g = u[(u["market"] == mk) & (u["league"] == lg) & u["yes"].notna() & u["no"].notna() & (u["date"] < before)]
        rec = {"football_data_prematch": 0, "own_forward": 0, "own_backfill": 0, "unmatched": 0, "oddsapi_hist": 0}
        if tr == "standard" and mk == "over25":
            rec["football_data_prematch"] = int(len(g))
        elif len(g):
            if (tr, mk) not in snap:
                snap[(tr, mk)] = U._snapshot(tr, mk, "first")
            pr = snap[(tr, mk)]
            pr = pr[pr["league"] == lg].reset_index(drop=True)
            att, _ = LB.attach_prices(g.drop(columns=["yes", "no"]),
                                      pr.assign(yes=pr.index.astype(float) + 2.0, no=2.0)[
                                          ["date", "league", "home", "away", "yes", "no"]], aliases)
            for i in (att["yes"] - 2.0).round():
                if pd.isna(i):
                    rec["unmatched"] += 1
                elif pr.loc[int(i), "price_origin"] == "BACKFILL_HISTORICAL":
                    rec["own_backfill"] += 1
                else:
                    rec["own_forward"] += 1
        if c["basis"] == FRESH and hist:
            h = hist[tr]
            rec["oddsapi_hist"] = int(((h["market"] == mk) & (h["league"] == lg)
                                       & (h["price_source"] == "oddsapi_hist")).sum())
        priced = rec["football_data_prematch"] + rec["own_forward"] + rec["own_backfill"] + rec["unmatched"] \
            + rec["oddsapi_hist"]
        unproven = rec["own_backfill"] + rec["unmatched"] + rec["oddsapi_hist"]
        rows.append(c | rec | {"priced_rows": priced,
                               "unproven_share": round(unproven / priced, 4) if priced else None})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--aliases", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--priced", help="local priced universes of a fresh-holdout run (counts of oddsapi_hist rows)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = json.loads(Path(a.proposal).read_text(encoding="utf-8"))
    rows = provenance(proposal_cells(p), json.loads(Path(a.aliases).read_text(encoding="utf-8")), a.before,
                      Path(a.priced) if a.priced else None)
    doc = {"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"], "before": a.before,
           "rule": "unproven_share = (own_backfill + unmatched + oddsapi_hist) / priced rows; PROVEN capture time = "
                   "football-data pre-match or our own live capture before kickoff", "cells": rows}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    over = [r for r in rows if r["unproven_share"] is not None and r["unproven_share"] > 0.5]
    print(json.dumps({"cells": len(rows), "unproven_gt_half": len(over)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
