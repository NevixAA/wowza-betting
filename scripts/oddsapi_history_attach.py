"""
Attach the stored OddsAPI history (a checkout of the PRIVATE wowza-exec branch `odds-history`) to a
prediction / bars universe parquet by fixture id (src/oddsapi_history_adapter.py).

  python scripts/oddsapi_history_attach.py --store DIR --universe FILE.parquet --out DIR

Writes <out>/prices_long.parquet, <out>/prices_summary.parquet, <out>/universe_with_history.parquet
and <out>/unmatched.csv. --out must be OUTSIDE this public repo (redistribution is prohibited).
Prints counts only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import oddsapi_history_adapter as AD  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--universe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        a.out.resolve().relative_to(ROOT)
        raise SystemExit("--out is inside the public wowza-betting tree: refused (redistribution prohibited)")
    except ValueError:
        pass
    u = pd.read_parquet(a.universe)
    odds = AD.load_store(a.store)
    long, summary, unmatched = AD.attach(u, odds, AD.load_context(a.store, u))
    a.out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(a.out / "prices_long.parquet", index=False)
    summary.to_parquet(a.out / "prices_summary.parquet", index=False)
    # BET prices (tier BT) and CLOSE prices (CLV only) side by side; a bet price never comes from CLOSE
    ub = AD.attach_to_universe(u, summary, purpose="bet")
    AD.attach_to_universe(ub, summary, purpose="close").drop(columns=["fixture_id"]).to_parquet(
        a.out / "universe_with_history.parquet", index=False)
    unmatched.to_csv(a.out / "unmatched.csv", index=False)
    kinds = unmatched["kind"].value_counts().to_dict() if not unmatched.empty else {}
    print(f"[oddsapi-history-attach] dropped {long.attrs.get('dropped', {})}; fixtures priced {long['fixture_id'].nunique() if not long.empty else 0:,}; "
          f"price rows {len(long):,}; unmatched {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
