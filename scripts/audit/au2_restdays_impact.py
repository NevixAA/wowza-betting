"""AU2 - probability impact of the legacy serving rest-days defect on the audited board. READ-ONLY.

Re-scores every legacy-frame model on the persisted board features twice: as served, and with ONLY
home_rest_days / away_rest_days replaced by the per-fixture value (league-scoped name resolution,
full history - what training computes). Same batch composition as production. Reports the change in
probability and how many NF OU2.5 rows cross a production tier floor (VALUABLE 0.03 raw edge).
Counterfactual on ONE board only; it is not an ROI estimate.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, heavy_lock, load_history_offline  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402


def main() -> int:
    from src.model import predict_proba
    from src.team_names import resolve
    board = pd.read_csv(ROOT / "output/predictions.csv", low_memory=False)
    h = load_history_offline().sort_values("date")
    last = {}
    for r in h.itertuples():
        last[(r.league, r.home_team)] = r.date
        last[(r.league, r.away_team)] = r.date
    pools = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in h.groupby("league")}

    def rd(lg, t, d):
        pool = pools.get(lg, [])
        name = t if t in pool else (resolve(t, pool) or t)
        ld = last.get((lg, name))
        return float((pd.Timestamp(d) - ld).days) if ld is not None else np.nan
    fixed = board.copy()
    fixed["home_rest_days"] = [rd(r.league, r.home_team, r.date) for r in board.itertuples()]
    fixed["away_rest_days"] = [rd(r.league, r.away_team, r.date) for r in board.itertuples()]
    std = board["model_type"] == "standard"
    nf = board["model_type"] == "new_format"
    out = {"board_generated_at": str(board["generated_at"].iloc[0])}
    for name, f, batch in (("NF_OU25", config.MODEL_FILE_NEWFORMAT, nf), ("BTTS", config.MODEL_FILE_BTTS, std | nf),
                           ("O15", config.MODEL_FILE_OVER15, std | nf), ("O35", config.MODEL_FILE_OVER35, std | nf),
                           ("STD_legacy_shadow", config.MODEL_FILE_STANDARD, std)):
        with open(f, "rb") as fh:
            pl = pickle.load(fh)
        a = predict_proba(board[batch], payload=pl).to_numpy()
        b = predict_proba(fixed[batch], payload=pl).to_numpy()
        d = b - a
        rec = {"n": int(batch.sum()), "median_abs_pp": round(float(np.median(np.abs(d))) * 100, 3),
               "p90_abs_pp": round(float(np.quantile(np.abs(d), 0.9)) * 100, 3),
               "max_abs_pp": round(float(np.abs(d).max()) * 100, 3), "mean_signed_pp": round(float(d.mean()) * 100, 3)}
        if name == "NF_OU25":
            sub = board[batch]
            oo = pd.to_numeric(sub["odds_over25"], errors="coerce").to_numpy()
            ou = pd.to_numeric(sub["odds_under25"], errors="coerce").to_numpy()
            ea = np.maximum(a - 1 / oo, (1 - a) - 1 / ou)
            eb = np.maximum(b - 1 / oo, (1 - b) - 1 / ou)
            ok = np.isfinite(ea) & np.isfinite(eb)
            rec["rows_crossing_0.03_edge_floor"] = int(((ea[ok] >= 0.03) != (eb[ok] >= 0.03)).sum())
            rec["rows_with_best_side_flip"] = int((((a - 1 / oo) >= ((1 - a) - 1 / ou)) != ((b - 1 / oo) >= ((1 - b) - 1 / ou)))[ok].sum())
        out[name] = rec
    out["fixed_rest_days_missing"] = {"standard": round(float(fixed.loc[std, "home_rest_days"].isna().mean()), 3),
                                      "new_format": round(float(fixed.loc[nf, "home_rest_days"].isna().mean()), 3)}
    (DATA / "restdays_impact.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    with heavy_lock("AU2 rest-days impact"):
        raise SystemExit(main())
