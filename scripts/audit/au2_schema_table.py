"""AU2 Phase 2B - per-feature train-vs-production schema table for every team model family.
READ-ONLY. Joins (a) the payload's own feature order and training-time scaler mean/sd
(payload_inventory.json), (b) the canary builder agreement (canary_feature_agreement.csv),
(c) the live board coverage (output/predictions.csv on the audited commit; for the repaired Standard
model the repaired board rebuild is used, because its inputs are not persisted).

    python scripts/audit/au2_schema_table.py -> reports/au2_data/schema_table.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAM = {  # payload -> (canary family, board rows, imputation)
    "standard_legacy": ("STD_OU25_legacy", "standard", "batch-median (legacy _prep)"),
    "standard_repaired": ("STD_OU25_repaired", "standard", "persisted train-only imputer"),
    "newformat": ("NF_OU25", "new_format", "batch-median (legacy _prep)"),
    "btts": ("BTTS_std", "standard", "batch-median over Standard+NF board"),
    "over15": ("O15_std", "standard", "batch-median over Standard+NF board"),
    "over35": ("O35_std", "standard", "batch-median over Standard+NF board"),
    "ht_over05": (None, "standard", "batch-median over Standard board"),
    "ht_over15": (None, "standard", "batch-median over Standard board"),
}
PRICE = {"bookmaker_overround"}


def main() -> int:
    inv = json.loads((DATA / "payload_inventory.json").read_text(encoding="utf-8"))["team"]
    can = pd.read_csv(DATA / "canary_feature_agreement.csv")
    board = pd.read_csv(ROOT / "output/predictions.csv", low_memory=False)
    rows = []
    for pay, (fam, mt, imp) in FAM.items():
        r = inv[pay]
        sub = board[board["model_type"] == mt]
        c = can[can["family"] == fam].set_index("feature") if fam else pd.DataFrame()
        for i, f in enumerate(r["feature_cols"]):
            sc = (r.get("train_scaler") or {}).get(f, {})
            live = pd.to_numeric(sub[f], errors="coerce") if f in sub.columns else pd.Series(dtype=float)
            rec = {"payload": pay, "position": i, "feature": f, "imputation": imp,
                   "price_derived": f in PRICE,
                   "train_scaler_mean": sc.get("mean"), "train_scaler_sd": sc.get("sd"),
                   "constant_in_training": (sc.get("sd") == 1.0 and abs(sc.get("mean", 1)) < 1e-12) if sc else None,
                   "persisted_on_board": f in sub.columns and pay != "standard_repaired",
                   "live_missing": round(float(live.isna().mean()), 3) if len(live) and pay != "standard_repaired" else None,
                   "live_median": round(float(live.median()), 4) if len(live) and live.notna().any() and pay != "standard_repaired" else None,
                   "live_constant": bool(live.nunique(dropna=True) <= 1) if len(live) and pay != "standard_repaired" else None,
                   "live_shift_train_sd": (round(abs(float(live.mean()) - sc["mean"]) / sc["sd"], 2)
                                           if len(live) and live.notna().any() and sc and sc.get("sd") and pay != "standard_repaired" else None)}
            if fam and f in c.index:
                rec.update({"canary_close_rate": round(float(c.at[f, "close_rate"]), 3),
                            "canary_train_missing": round(float(c.at[f, "train_missing"]), 3),
                            "canary_serve_missing": round(float(c.at[f, "serve_missing"]), 3),
                            "canary_verdict": c.at[f, "verdict"]})
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "schema_table.csv", index=False)
    s = df.groupby("payload").agg(n=("feature", "size"),
                                  canary_agree=("canary_verdict", lambda v: int((v == "AGREE").sum())),
                                  live_missing_gt_25pct=("live_missing", lambda v: int((v > 0.25).sum())),
                                  live_constant=("live_constant", lambda v: int(v.fillna(False).sum())),
                                  shift_gt_1sd=("live_shift_train_sd", lambda v: int((v > 1).sum())))
    print(s.to_string())
    print(df[(df["live_shift_train_sd"] > 1) | (df["live_constant"] == True)][["payload", "feature", "live_missing", "live_median", "train_scaler_mean", "train_scaler_sd", "live_shift_train_sd", "live_constant"]].to_string())  # noqa: E712
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
