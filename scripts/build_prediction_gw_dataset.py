"""Build / append the prediction success-by-gameweek dataset (owner 2026-11-23). See src/prediction_gw_dataset.py.

    python scripts/build_prediction_gw_dataset.py --pro-root DIR [--now ISO] [--out DIR]   # update_results.yml
    python scripts/build_prediction_gw_dataset.py --verify [--out DIR]                     # exit 1 on a finding

LEARNING DATA, NOT BETS: no price is read or written; v11 is private and never read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import prediction_gw_dataset as G  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pro-root", type=Path, default=None,
                    help="checkout of NevixAA/wowzaV9-Pro holding output/prediction_ledger[_settled]/pro")
    ap.add_argument("--out", type=Path, default=G.OUT)
    ap.add_argument("--now", default=None, help="ISO instant (tests / replays); default the wall clock")
    ap.add_argument("--verify", action="store_true", help="reconcile the stored dataset with its link rows")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.verify:
        rows = G.read_dataset(out / G.DATASET.name)
        issues = G.verify(rows, G.read_links(out / G.LINKS_DIR.name))
        for i in issues:
            print(f"::error title=Prediction GW dataset::{i}")
        print(json.dumps({"rows": int(len(rows)), "findings": len(issues)}))
        return 1 if issues else 0
    pro = a.pro_root if a.pro_root and (a.pro_root / "output" / "prediction_ledger").exists() else None
    if a.pro_root and pro is None:
        print(f"::warning::Pro ledger not found under {a.pro_root} - Pro rows stay as they are (closed) or wait")
    print(json.dumps(G.run(pro_root=pro, out=out, now=a.now)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
