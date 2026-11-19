"""Stake-plan replay: the CURRENT per-track bankroll policy against the per-CELL policy (owner 2026-11-09
"Paper + PROMISING small"), in memory only.

    python scripts/stake_cells_replay.py --audit output/stake_audit.csv --results output/v91_shadow.csv \
        --cells-replay output/threshold_proposals/bars_opt_serving/2026-11_replay.json --out <json> [--days 30]

(A) SAME SELECTIONS: every selection the live stake plan sized in the last N days (output/stake_audit.csv, the
    last sizing per date x fixture x market x side), re-sized day by day with src/bankroll_policy.size_day under
    the committed policy with cell_staking OFF (= today's per-track staking) and ON (per cell). Isolates the
    staking change; the tips themselves are what the old bars produced.
(B) CELL TIPS: the tips the proposed bars_opt cells would have produced (scripts/bars_serving_replay.py), sized
    under the per-cell policy.
P/L = final_fraction x (odds - 1) when won, - final_fraction when lost, on a notional bankroll of 1 (results
from output/v91_shadow.csv; unsettled rows carry no P/L). Cloudbet stays PAPER_ONLY: nothing is placed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import bankroll_policy as BP  # noqa: E402

LINE = {"OU25": 2.5, "OVER15": 1.5, "OVER35": 3.5}


def _won(market, side, hg, ag):
    if hg is None or ag is None or not (np.isfinite(hg) and np.isfinite(ag)):
        return np.nan
    m, s = str(market).upper(), str(side).upper()
    if m == "BTTS":
        y = float(hg > 0 and ag > 0)
        return y if s == "YES" else 1.0 - y
    if m in LINE:
        y = float(hg + ag > LINE[m])
        return y if s == "OVER" else 1.0 - y
    return np.nan


def _results(path: Path) -> dict:
    v = pd.read_csv(path, low_memory=False)
    v = v[["match_date", "home_team", "away_team", "home_goals", "away_goals"]].dropna()
    return {(str(r.match_date)[:10], str(r.home_team), str(r.away_team)): (float(r.home_goals), float(r.away_goals))
            for r in v.itertuples(index=False)}


def size(sel: pd.DataFrame, policy: dict) -> pd.DataFrame:
    parts = [BP.size_day(g.reset_index(drop=True), policy, bankroll=100.0, peak=100.0) for _, g in sel.groupby("date")]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summary(sized: pd.DataFrame, res: dict) -> dict:
    if sized.empty:
        return {"selections": 0}
    st = sized[sized["final_fraction"] > 0].copy()
    won = []
    for r in st.itertuples(index=False):
        d, _, rest = str(r.fixture).partition("|")
        h, _, a = rest.partition("|")
        hg, ag = res.get((d, h, a), (None, None))
        won.append(_won(r.market, r.side, hg, ag))
    st["won"] = won
    settled = st[st["won"].notna()]
    pnl = np.where(settled["won"] == 1, settled["final_fraction"] * (settled["odds"].astype(float) - 1),
                   -settled["final_fraction"])
    cells = st.assign(cell=st["track"] + "|" + st["league"]).groupby("cell").agg(
        stakes=("final_fraction", "size"), total_fraction=("final_fraction", "sum"))
    return {"selections": int(len(sized)), "staked": int(len(st)),
            "total_fraction": round(float(st["final_fraction"].sum()), 4),
            "settled_staked": int(len(settled)), "pnl_units_of_bankroll": round(float(pnl.sum()), 5) if len(settled) else None,
            "roi_on_staked": round(float(pnl.sum() / settled["final_fraction"].sum()), 4) if len(settled) else None,
            "reasons": sized["reason"].value_counts().to_dict(),
            "staked_by_cell": {k: {"stakes": int(v["stakes"]), "total_fraction": round(float(v["total_fraction"]), 4)}
                               for k, v in cells.iterrows()}}


def run(audit: Path, results: Path, cells_replay: Path | None, days: int) -> dict:
    a = pd.read_csv(audit, low_memory=False)
    a["date"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d")
    end = pd.Timestamp(a["date"].max())
    a = a[pd.to_datetime(a["date"]) > end - pd.Timedelta(days=days)]
    a = a.sort_values("generated_at").drop_duplicates(["date", "fixture", "market", "side"], keep="last")
    sel = a[["date", "fixture", "league", "model_type", "market", "side", "signal_tier", "odds"]].reset_index(drop=True)
    pol_cell = BP.load_policy()
    pol_track = copy.deepcopy(pol_cell)
    pol_track["cell_staking"]["enabled"] = False
    res = _results(results)
    out = {"window": [str((end - pd.Timedelta(days=days - 1)).date()), str(end.date())],
           "policy_version": pol_cell.get("policy_version"),
           "A_same_selections": {"current_per_track": summary(size(sel, pol_track), res),
                                 "new_per_cell": summary(size(sel, pol_cell), res),
                                 "as_recorded_live": {"staked": int((a["final_fraction"] > 0).sum()),
                                                      "total_fraction": round(float(a["final_fraction"].sum()), 4)}}}
    if cells_replay is not None and Path(cells_replay).exists():
        rp = json.loads(Path(cells_replay).read_text(encoding="utf-8"))
        tips = pd.DataFrame(rp.get("tipped_rows") or [])
        out["B_cell_tips"] = ({"note": "the cells replay file carries no tipped_rows; re-run bars_serving_replay.py"}
                              if tips.empty else {"new_per_cell": summary(size(tips, pol_cell), res)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--cells-replay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args(argv)
    out = run(Path(a.audit), Path(a.results), Path(a.cells_replay) if a.cells_replay else None, a.days)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:7000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
