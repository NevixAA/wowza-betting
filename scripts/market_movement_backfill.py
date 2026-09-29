"""Attach opening -> moving -> closing price to EVERY ledger. Read-mostly, additive.

    python scripts/market_movement_backfill.py --dry-run    # report coverage, write nothing
    python scripts/market_movement_backfill.py              # write the mv_* columns

WHAT IT DOES. Reads the ~304,000 banked price rows through src.market_movement and adds these
columns to each ledger, prefixed `mv_` so nothing existing is touched:

    mv_open  mv_close  mv_n_snaps  mv_drift_pct  mv_signal  mv_clv_pct  mv_source

WHAT IT DOES NOT DO. It changes no tier, no stake, no threshold and no tip. Every column is
additive and every writer downstream addresses this file by column name. The point is to make
"does market movement carry information, per league x market x model" answerable from evidence.

WHY IT IS NEEDED. Before this, movement was tracked for exactly one market on one track:
drift.py watches the OU25 over/under pair and nothing else. Side markets had closing prices but
no opening and no drift; half-time had 11% coverage; player props had no movement columns at
all. Meanwhile the curve files hold 304,000 rows nobody reads.

The measured reason to care: the one movement rule in production is pointing the wrong way. On
settled bets its `Confirmed` bucket -- the one it UPGRADES on -- runs -10.8%, while `Neutral`,
which it ignores entirely, runs +8.0%. And the two model tracks share one global rule despite
invariant 1 keeping them separate everywhere else: new-format `Neutral` is +16.6% against
standard's -7.3%.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.market_movement import CurveIndex, enrich

log = logging.getLogger("mv_backfill")

#: side value -> the market key the price is captured under, per ledger.
_SIDE = {"OVER": "over25", "UNDER": "under25"}
_SIDE_MKT = {("btts", "YES"): "btts_yes", ("btts", "NO"): "btts_no",
             ("over15", "OVER"): "over15", ("over15", "UNDER"): "under15",
             ("over35", "OVER"): "over35", ("over35", "UNDER"): "under35"}


def _main_ou(r):
    """Main ledger: an implicit OU25 whose side is OVER/UNDER."""
    return _SIDE.get(str(r.get("side", "")).strip().upper())


def _side_market(r):
    """Side ledger: the market column alone.

    THERE IS NO `side` COLUMN HERE — it exists but is entirely NaN, because the side markets
    only ever back the over/yes side. An earlier version of this mapper keyed on (market, side)
    and matched 0 of 395 rows for exactly that reason.
    """
    mkt = str(r.get("market", "")).strip().lower()
    return {"btts": "btts_yes", "over15": "over15", "over35": "over35"}.get(mkt)


def _ht(r):
    """HT ledger already stores the exact market key (ht_over05 ... ht_under15)."""
    m = str(r.get("market", "")).strip().lower()
    return m if m.startswith("ht_") else None


def _props(r):
    """Props ledger: the market column matches the capture (goals, sot, cards, ...)."""
    m = str(r.get("market", "")).strip().lower()
    return m or None


# (label, file, market mapper, entry-odds column, player column or None)
LEDGERS = [
    ("main O/U", "bets_ledger.csv", _main_ou, "odds", None),
    ("side markets", "side_bets_ledger.csv", _side_market, "odds", None),
    ("half-time", "ht_ledger.csv", _ht, "entry_odds", None),
    # Props carry one curve PER PLAYER under the same market key — without player_name the
    # lookup grabs a team-mate's prices.
    ("player props", "player_ledger.csv", _props, "market_odds", "player_name"),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report coverage, write nothing")
    a = ap.parse_args()

    idx = CurveIndex()
    if not idx.n_rows:
        log.error("no captured price curves found — nothing to backfill")
        return 1
    log.info(f"curve index: {idx.n_rows:,} price rows\n")

    summary = []
    for name, fname, market_of, entry_col, player_col in LEDGERS:
        p = config.OUTPUT_DIR / fname
        if not p.exists():
            log.warning(f"{name}: {fname} absent"); continue
        d = pd.read_csv(p)
        if d.empty:
            log.warning(f"{name}: empty"); continue
        ec = entry_col if entry_col in d.columns else None
        pc = player_col if player_col and player_col in d.columns else None
        out = enrich(d, market_of, idx=idx, entry_col=ec, player_col=pc)

        got = out["mv_close"].notna()
        clv = out["mv_clv_pct"].notna()
        summary.append({"ledger": name, "rows": len(out),
                        "with_curve": int(got.sum()),
                        "coverage": f"{got.mean() * 100:.0f}%",
                        "with_clv": int(clv.sum()),
                        "mean_clv": round(float(out.loc[clv, "mv_clv_pct"].mean()), 2)
                        if clv.any() else None,
                        "mean_snaps": round(float(out.loc[got, "mv_n_snaps"].mean()), 1)
                        if got.any() else None})
        if not a.dry_run:
            out.to_csv(p, index=False)
            log.info(f"{name}: wrote {fname}")

    print("\n" + "=" * 78)
    print("MARKET-MOVEMENT COVERAGE" + ("  (DRY RUN — nothing written)" if a.dry_run else ""))
    print("=" * 78)
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nmean_clv is % against the close; positive = we beat it. `mean_snaps` under ~3 means")
    print("the curve is thin — an open and a close with little in between.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
