"""DRY RUN of production TOP-3 (owner 2026-11-11: cell bar + T3 ranking, no tiers) on a past board - writes nothing
in output/, prints counts per league.

    python scripts/top3_t3_dry_run.py --proposal output/threshold_proposals/bars_oddsapi_hist/2026-11.json \
        --week-from 2026-11-03 --week-to 2026-11-09 --lead 120

The cells are the proposal's (ASSUMED owner-applied: the dry run reads them as if threshold_apply had run); the
candidates are production's own builder (src.research.league_weekly.build_candidates over output/
edge_engine_shadow.csv) restricted to the league-weeks locking in the window; the T3 statistics are built from
the committed own-price universes with rows dated BEFORE --week-from (no later data). Every week is frozen at
its own FREEZE (lock - lead), the moment production issues it. CAVEAT: the board's probabilities are whatever models produced output/ at the time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import top3_league as T  # noqa: E402
from src.research import league_weekly as LW  # noqa: E402


def ctx_from_proposal(p: dict) -> dict:
    std, side, ids = {}, {}, set()
    for e in p["entries"]:
        if not e.get("proposed"):
            continue
        ids.add(e["entry_id"])
        if e["market"] == "ou25":
            std[e["league"]] = e["proposed"]
        else:
            side.setdefault(e["market"], {})[e["league"]] = e["proposed"]
    return {"standard": std, "side": side, "applied": ids}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--week-from", required=True)
    ap.add_argument("--week-to", required=True)
    ap.add_argument("--lead", type=float, required=True, help="freeze lead in minutes (production: TOP3_FREEZE_LEAD_MIN)")
    a = ap.parse_args(argv)
    ctx = ctx_from_proposal(json.loads(Path(a.proposal).read_text(encoding="utf-8")))
    stats_doc = T.build_t3_stats(ROOT / "output" / "bars_opt", ctx, a.week_from)
    stats = pd.DataFrame(stats_doc["cells"])
    od = T.OUT
    shadow = LW._read(od / "edge_engine_shadow.csv")
    preds = LW._read(od / "predictions.csv")
    plan = LW._read(od / "stake_plan.csv")
    champs = LW._json(ROOT / "models" / "registry" / "current_champions.json")
    from src import market_edge as ME
    end = pd.Timestamp(a.week_to) + pd.Timedelta(days=1)
    cands = LW.build_candidates(shadow, preds, LW.registry_entries(), plan, champs, ME.load_params(),
                                pd.Timestamp(end, tz="UTC"))
    cands = LW._plain(cands)
    locks = LW.week_locks(cands)
    lo, hi = pd.Timestamp(a.week_from, tz="UTC"), pd.Timestamp(end, tz="UTC")
    weeks = {w for w, lk in locks.items() if lk is not None and lo <= lk < hi}
    cands = cands[cands["league_week_id"].isin(weeks)]
    # each week is frozen at its own FREEZE (= lock - lead), the moment production would issue it
    pk, wks = [], []
    for w in sorted(weeks):
        g = cands[cands["league_week_id"] == w]
        now_w = locks[w] - pd.Timedelta(minutes=a.lead)
        p_, w_ = T.freeze_due(g, set(), ctx, now_w, a.lead, {}, {}, None, stats)
        pk.append(p_)
        wks.append(w_)
    picks = pd.concat(pk, ignore_index=True) if pk else pd.DataFrame(columns=T.PICK_COLS)
    wk = pd.concat(wks, ignore_index=True) if wks else pd.DataFrame(columns=T.WEEK_COLS)
    out = {"window": [a.week_from, a.week_to], "league_weeks": len(weeks), "candidates": int(len(cands)),
           "t3_cells_with_stat": stats_doc["cells_with_stat"], "picks": int(len(picks)),
           "per_league": picks.groupby("league").size().to_dict() if len(picks) else {},
           "slots": picks["slot"].value_counts().to_dict() if len(picks) else {},
           "cell_status": picks["cell_status"].value_counts().to_dict() if len(picks) else {},
           "picks_with_t3_stat": int(picks["t3_has_stat"].astype(bool).sum()) if len(picks) else 0,
           "weeks_with_0_1_2_3": wk["n_picks"].astype(int).value_counts().sort_index().to_dict() if len(wk) else {},
           "excluded": {}}
    from collections import Counter
    ex: Counter = Counter()
    for e in wk["excluded"]:
        ex.update({k: int(v) for k, v in json.loads(e).items()})
    out["excluded"] = dict(ex)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
