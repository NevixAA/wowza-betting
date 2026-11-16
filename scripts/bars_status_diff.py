"""Cell-by-cell diff of two bars proposals (status tables) with a one-line reason per changed cell - COUNTS and
STATISTICS ONLY (no price is printed or written).

    python scripts/bars_status_diff.py --old-evidence <evidence A.json> --new-evidence <evidence B.json> \
        --old-universe <A own_bettime dir> --new-universe <B own_bettime dir> \
        [--old-priced <A priced dir> --new-priced <B priced dir>] --out <json>

Used for the clean-model re-run (TP-2026-11-15-bade2c0b -> the clean-model proposal) and for the quick
confirm-diff after the 2026-11-19/20 final refit (this run -> the refit run): the same cells, the model
probability change on the rows both universes hold, and the price-coverage change, per cell.

Reason per changed cell (first that applies):
  PRICE_COVERAGE   the cell's priced rows changed by >= 10% (row set / prices), so the evidence base moved
  MODEL_PROB       otherwise: the model probabilities moved (mean |dp| on common priced rows is reported)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bars_opt import study as S  # noqa: E402

TRACKS = ("standard", "new_format")
COVERAGE_SHARE = 0.10


def _fam(udir: Path | None) -> dict:
    if not udir:
        return {}
    out = {}
    for t in TRACKS:
        f = Path(udir) / f"universe_{t}.parquet"
        if f.exists():
            u = pd.read_parquet(f)
            for ms in S.MARKET_SIDES:
                out[(t, ms)] = S.family_frame(u, ms)
    return out


def _key(r):
    return (r["track"], r["mside"], r["league"])


def cell_metrics(fa: dict, fb: dict, k: tuple) -> dict:
    a, b = fa.get(k[:2]), fb.get(k[:2])
    a = a[a["league"] == k[2]] if a is not None else pd.DataFrame(columns=["row_id", "p_side"])
    b = b[b["league"] == k[2]] if b is not None else pd.DataFrame(columns=["row_id", "p_side"])
    m = a[["row_id", "p_side"]].merge(b[["row_id", "p_side"]], on="row_id", suffixes=("_a", "_b"))
    d = (m["p_side_b"] - m["p_side_a"]).to_numpy(float) if len(m) else np.array([])
    return {"rows_old": int(len(a)), "rows_new": int(len(b)), "rows_common": int(len(m)),
            "mean_abs_dp": round(float(np.abs(d).mean()), 4) if len(d) else None,
            "mean_dp": round(float(d.mean()), 4) if len(d) else None}


def reason(mx: dict) -> str:
    ra, rb = mx["rows_old"], mx["rows_new"]
    if max(ra, rb) == 0:
        return "PRICE_COVERAGE"
    if abs(rb - ra) / max(ra, rb) >= COVERAGE_SHARE:
        return "PRICE_COVERAGE"
    return "MODEL_PROB"


def pred_diff(fa: dict, fb: dict) -> list[dict]:
    rows = []
    for k in sorted(set(fa) & set(fb)):
        if not k[1].endswith(("_over", "_yes")):
            continue
        m = fa[k][["row_id", "p_side"]].merge(fb[k][["row_id", "p_side"]], on="row_id", suffixes=("_a", "_b"))
        if m.empty:
            continue
        d = (m["p_side_b"] - m["p_side_a"]).to_numpy(float)
        rows.append({"track": k[0], "family": k[1], "rows_common": int(len(m)),
                     "mean_abs_dp": round(float(np.abs(d).mean()), 4), "p95_abs_dp": round(float(np.quantile(np.abs(d), .95)), 4),
                     "mean_dp": round(float(d.mean()), 4)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-evidence", required=True)
    ap.add_argument("--new-evidence", required=True)
    ap.add_argument("--old-universe")
    ap.add_argument("--new-universe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ea = json.loads(Path(a.old_evidence).read_text(encoding="utf-8"))
    eb = json.loads(Path(a.new_evidence).read_text(encoding="utf-8"))
    ta = {_key(r): r for r in ea["league_market_status"]}
    tb = {_key(r): r for r in eb["league_market_status"]}
    fa, fb = _fam(a.old_universe), _fam(a.new_universe)
    order = {None: -1, "OFF": 0, "UNPROVEN": 1, "PROMISING": 2, "PROVEN": 3}
    changes, bars = [], []
    for k in sorted(set(ta) | set(tb)):
        sa, sb = (ta.get(k) or {}).get("status"), (tb.get(k) or {}).get("status")
        if sa != sb:
            mx = cell_metrics(fa, fb, k)
            changes.append({"cell": "|".join(k), "old": sa, "new": sb,
                            "direction": "UP" if order[sb] > order[sa] else "DOWN",
                            "old_basis": (ta.get(k) or {}).get("basis") or (ta.get(k) or {}).get("reason"),
                            "new_basis": (tb.get(k) or {}).get("basis") or (tb.get(k) or {}).get("reason"),
                            "old_config": (ta.get(k) or {}).get("config"), "new_config": (tb.get(k) or {}).get("config"),
                            "reason": reason(mx)} | mx)
        elif sa != "OFF" and ((ta[k].get("bars"), ta[k].get("config")) != (tb[k].get("bars"), tb[k].get("config"))):
            bars.append({"cell": "|".join(k), "status": sa, "old": [ta[k].get("bars"), ta[k].get("config")],
                         "new": [tb[k].get("bars"), tb[k].get("config")]})
    from collections import Counter
    res = {"cells_compared": len(set(ta) | set(tb)),
           "counts_old": dict(Counter(r["status"] for r in ta.values())),
           "counts_new": dict(Counter(r["status"] for r in tb.values())),
           "status_changes": changes, "bar_or_config_changes_same_status": bars,
           "prediction_diff": pred_diff(fa, fb)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("cells_compared", "counts_old", "counts_new")}
                     | {"status_changes": len(changes), "bar_changes_same_status": len(bars)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
