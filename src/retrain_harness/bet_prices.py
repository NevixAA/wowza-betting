"""BET-TIME prices for the harness's market_bet / model_market_bet candidates.

The prices themselves are PRIVATE (The Odds API history, wowza-exec branch odds-history, tier BT,
role BET: the snapshot at the measured live-tip lead time, 125 h Standard / 153 h New-Format before
kickoff). They are attached to the frame by row id ('YYYY-MM-DD|league|home|away', the harness's
own row id) from a parquet that lives OUTSIDE this public repository; only counts are recorded.

    load_bet_prices(path) -> DataFrame[row_id, market, median_yes, median_no, n_labels]
    attach_bet_prices(feat, prices, cfg) -> (feat + bet_<market>_yes / _no, counts)

Rules: a CLOSING / OPEN / MOVEMENT snapshot is never here (the attach step reads role BET only, and a
file carrying a 'role' column other than BET is refused); a fixture whose BET role holds two
snapshot labels is left unpriced; a price <= 1.01 or a one-sided pair gives no probability (devig_yes).
Missing stays missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_bet_prices(path) -> pd.DataFrame:
    p = Path(path).resolve()
    try:
        p.relative_to(ROOT)
        raise PermissionError(f"{p} is inside the public repository: bet prices stay private")
    except ValueError:
        pass
    d = pd.read_parquet(p)
    if "role" in d.columns and set(d["role"].dropna()) - {"BET"}:
        raise ValueError("bet prices must come from role BET only (a closing price is never a bet price)")
    d = d.rename(columns={"fixture_id": "row_id"})
    need = {"row_id", "market", "median_yes", "median_no"}
    if need - set(d.columns):
        raise KeyError(f"bet price file lacks {sorted(need - set(d.columns))}")
    return d


def attach_bet_prices(feat: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    rid = (pd.to_datetime(feat["date"]).dt.strftime("%Y-%m-%d") + "|" + feat["league"].astype(str) + "|"
           + feat["home_team"].astype(str) + "|" + feat["away_team"].astype(str))
    out = feat.copy()
    counts = {}
    pr = prices
    if "n_labels" in pr.columns:
        pr = pr[pd.to_numeric(pr["n_labels"], errors="coerce") == 1]
    for m, mc in cfg["markets"].items():
        bp = mc.get("bet_price")
        if not bp:
            continue
        g = pr[pr["market"] == m].drop_duplicates("row_id", keep=False).set_index("row_id")
        y = rid.map(g["median_yes"]).to_numpy(dtype=float)
        n = rid.map(g["median_no"]).to_numpy(dtype=float)
        out[bp["yes"]] = y
        out[bp["no"]] = n
        counts[m] = int((np.isfinite(y) & np.isfinite(n)).sum())
    return out, {"rows_with_two_sided_bet_price": counts, "source": "private BET-role file (path not recorded)"}
