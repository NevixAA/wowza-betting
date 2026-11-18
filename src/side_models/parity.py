"""Train/serve parity + as-of (no leakage) check for the side models.
=====================================================================

    python -m src.side_models.parity        # exit 1 on any difference

1. PARITY: for every verified fixture the training frame holds, the serving path
   (src.side_models.ledger.score on the same fixture presented as an upcoming target) gives the SAME
   features and the SAME probability per market x side x line as the training path. Max |diff| must be 0.
2. AS-OF: for a deterministic sample of fixtures per league x family, features built from a table cut to
   matches strictly BEFORE the fixture's date equal the features built from the full table - nothing on
   or after the match day leaks in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.side_models import data as D  # noqa: E402
from src.side_models import features as F  # noqa: E402
from src.side_models import model as M  # noqa: E402
from src.side_models import train as T  # noqa: E402

FEATS = ["home_for", "home_against", "away_for", "away_against"]


def run(sample_per_cell: int = 60, art: dict | None = None) -> dict:
    from src.side_models import ledger as L
    cfg = T.load_config()
    art = art or L.load_artifact()
    covered = sorted({lg for fam in cfg["families"] for lg, c in art["coverage"][fam].items() if c.get("covered")})
    table, _ = D.build(covered)
    res = {"fixtures": 0, "probabilities": 0, "max_abs_feature_diff": 0.0, "max_abs_prob_diff": 0.0,
           "asof_checked": 0, "asof_max_abs_diff": 0.0, "mismatches": []}
    for fam, fc in cfg["families"].items():
        for lg, c in art["coverage"][fam].items():
            if not c.get("covered"):
                continue
            fr = T.training_frame(table, lg, fam, cfg["features"], c["window_from"])
            if fr.empty:
                continue
            tg = fr[["fixture_id", "league", "date", "home_id", "away_id"]].copy()
            tg["kickoff_utc"] = pd.to_datetime(tg["date"], utc=True)
            served = L.score(table, tg, art, {**cfg, "families": {fam: fc}})[fam]
            served = served.set_index("fixture_id").loc[fr["fixture_id"]].reset_index()
            fd = np.nanmax(np.abs(served[FEATS].to_numpy(float) - fr[FEATS].to_numpy(float)))
            res["max_abs_feature_diff"] = max(res["max_abs_feature_diff"], float(fd))
            prm = art["models"][fam][lg]["params"]
            for market, mc in fc["markets"].items():
                for side in (["total"] if mc["part"] == "total" else ["home", "away"]):
                    mu = M.mu_of(prm[side], fr, side)
                    for line in mc["lines"]:
                        p_train = M.p_over(mu, prm[side]["alpha"], line)
                        p_serve = served[f"p|{market}|{side}|{line:g}"].to_numpy(float)
                        d = float(np.nanmax(np.abs(p_train - p_serve)))
                        if np.isnan(p_serve).any() or d > 0:
                            res["mismatches"].append(f"{fam}/{lg}/{market}/{side}/{line}: max diff {d}")
                        res["max_abs_prob_diff"] = max(res["max_abs_prob_diff"], d)
                        res["probabilities"] += len(p_train)
            res["fixtures"] += len(fr)
            # as-of: deterministic sample (every k-th fixture)
            k = max(1, len(fr) // sample_per_cell)
            for r in fr.iloc[::k].itertuples(index=False):
                cut = table[(table["league"] == lg) & (table["date"] < r.date)]
                one = pd.DataFrame([{"fixture_id": r.fixture_id, "league": lg, "date": r.date,
                                     "home_id": r.home_id, "away_id": r.away_id}])
                f1 = F.build(cut, one, fam, **cfg["features"])
                d = float(np.nanmax(np.abs(f1[FEATS].to_numpy(float)[0] -
                                           np.array([getattr(r, c) for c in FEATS], float))))
                res["asof_max_abs_diff"] = max(res["asof_max_abs_diff"], d)
                res["asof_checked"] += 1
                if d > 0 or f1["not_scored_reason"].iloc[0]:
                    res["mismatches"].append(f"as-of {fam}/{lg}/{r.fixture_id}: diff {d}")
    res["ok"] = (not res["mismatches"] and res["max_abs_feature_diff"] == 0.0
                 and res["max_abs_prob_diff"] == 0.0 and res["asof_max_abs_diff"] == 0.0)
    return res


def main(argv=None) -> int:
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "mismatches"}, indent=1))
    for m in r["mismatches"][:20]:
        print("MISMATCH", m)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
