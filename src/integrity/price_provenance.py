"""Price provenance labels and the price contract (R2-C findings 14, 15, 24 and the 2026-11-01
newformat_odds_history finding). LABELS ONLY: raw files are never rewritten; derived readers call
these functions and act on the labels.

SYNTHETIC SIDE PRICES. When a historical row has no side-market price, src/backtest.py fills
BTTS 1.85 / O1.5 1.40 / O3.5 2.60 so the model can still score the row. A filled price is not a
market: `label_side_prices` marks it SYNTHETIC_PRICE_DIAGNOSTIC_ONLY with profitability_valid =
False (INVALID_FOR_PROFITABILITY), and `assert_profitability_inputs` refuses any frame that would
put one into ROI / EV / Kelly / promotion / certification / stakes.

MARKET MAPPING (`label_market_mapping`), measured 2026-11-01 against realised outcomes
(scripts/integrity_price_provenance.py writes the evidence table):
  * output/newformat_odds_history.csv
      BACKFILL rows (snapshot_ts ...T00:00:00Z, 2025-06..2026-06): valid full-match lines. Implied
      vs realised: O1.5 0.590/0.621, O2.5 0.517/0.530, O3.5 0.438/0.442, BTTS 0.533/0.561.
      The "NF over15 median 1.66" is a SELECTION effect, not a mapping error: the historical
      OddsAPI totals carry only each book's MAIN line, so an O1.5 price exists only where 1.5 was
      the main line (393 of 2,820 fixtures, the low-scoring ones). line_selection=MAIN_LINE_ONLY.
      FORWARD rows captured before the O/U parser fix (ac6406cb4, 2026-08-10): INVALID. The parser
      took any bet whose name contained "Over/Under" (first-half, corners, team totals), so the
      last matching market won: O2.5 median 3.50, implied 0.318 vs realised 0.490.
      FORWARD BTTS captured before the BTTS fix (8d9962c68, 2026-08-23): INVALID (first-half BTTS,
      median Yes 4.00, implied 0.23-0.26 vs realised 0.52-0.59).
  * output/standard_sidemarket_odds_history.csv: btts_yes > 3.20 is first-half BTTS mislabelled as
    full-match (the gap rule data_loader already applies; the two populations do not overlap).
Any market_mapping_valid=False row must not enter a full-match analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROVENANCE_VERSION = "price-provenance-1.0"
SYNTHETIC_DEFAULT_ODDS = {"btts": 1.85, "over15": 1.40, "over35": 2.60}
SYNTHETIC_LABEL = "SYNTHETIC_PRICE_DIAGNOSTIC_ONLY"
INVALID_PROFIT = "INVALID_FOR_PROFITABILITY"
REAL_LABEL = "REAL_QUOTE"

#: (file, family) -> first snapshot time (UTC) at which the forward parser is known correct.
PARSER_FIX_UTC = {
    ("newformat_odds_history", "ou"): pd.Timestamp("2026-08-10T08:00:00Z"),     # ac6406cb4
}
# BTTS needs no window: before 8d9962c68 the parser's last match won, so a fixture whose book also
# offered first-half BTTS carries the first-half price (always > 3.40) and one that did not carries
# the real full-match price (<= 2.97). The 3.20 gap rule separates them exactly in both files.
BTTS_YES_FULLMATCH_MAX = 3.20
OU_MARKETS = {"over15", "under15", "over25", "under25", "over35", "under35"}
BTTS_MARKETS = {"btts_yes", "btts_no"}


class ProvenanceContractError(AssertionError):
    pass


def label_side_prices(df: pd.DataFrame, market: str, odds_col: str | None = None) -> pd.DataFrame:
    """Add price_provenance / profitability_valid. Uses the explicit `price_synthetic` flag when
    present (a real quote that happens to equal the constant survives); otherwise fails closed and
    treats every row priced exactly at the fill constant as synthetic."""
    odds_col = odds_col or f"odds_{market}"
    d = df.copy()
    if "price_synthetic" in d.columns:
        syn = d["price_synthetic"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        basis = "price_synthetic flag"
    else:
        c = SYNTHETIC_DEFAULT_ODDS.get(market)
        px = pd.to_numeric(d.get(odds_col), errors="coerce") if odds_col in d.columns else pd.Series(np.nan, index=d.index)
        syn = px.eq(c) if c is not None else pd.Series(False, index=d.index)
        syn = syn | px.isna()
        basis = f"fail-closed: price == {c} or missing"
    d["price_provenance"] = np.where(syn, SYNTHETIC_LABEL, REAL_LABEL)
    d["profitability_valid"] = ~syn.to_numpy()
    d["profitability_label"] = np.where(syn, INVALID_PROFIT, "VALID")
    d.attrs["provenance_basis"] = basis
    return d


def assert_profitability_inputs(df: pd.DataFrame, what: str = "ROI/EV") -> None:
    """Contract: no synthetic or mis-mapped price may enter ROI / EV / Kelly / promotion / stakes."""
    bad = pd.Series(False, index=df.index)
    if "price_provenance" in df.columns:
        bad |= df["price_provenance"].astype(str).eq(SYNTHETIC_LABEL)
    if "price_synthetic" in df.columns:
        bad |= df["price_synthetic"].astype(str).str.lower().isin(("true", "1", "yes"))
    if "market_mapping_valid" in df.columns:
        bad |= ~df["market_mapping_valid"].astype(bool)
    if bad.any():
        raise ProvenanceContractError(f"{int(bad.sum())} row(s) with a synthetic or mis-mapped "
                                      f"price would enter {what}")


def label_market_mapping(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add market_mapping_valid / mapping_reason / line_selection / price_origin to a raw odds
    capture frame (columns snapshot_ts, market, odds). Returns a NEW frame; `raw` is untouched."""
    d = raw.copy()
    ts = pd.to_datetime(d["snapshot_ts"], errors="coerce", utc=True, format="mixed")
    mk = d["market"].astype(str)
    odds = pd.to_numeric(d["odds"], errors="coerce")
    backfill = d["snapshot_ts"].astype(str).str.endswith("T00:00:00Z")
    d["price_origin"] = np.where(backfill, "BACKFILL_HISTORICAL", "FORWARD_CAPTURE")
    valid = pd.Series(True, index=d.index)
    reason = pd.Series("", index=d.index, dtype=object)

    for fam, mkts in (("ou", OU_MARKETS), ("btts", BTTS_MARKETS)):
        fix = PARSER_FIX_UTC.get((source, fam))
        if fix is None:
            continue
        pre = mk.isin(mkts) & ~backfill & (ts < fix)
        valid &= ~pre
        reason = reason.where(~pre, f"PARSER_PRE_FIX_{fam.upper()}_{fix.date()}")
    fh = (mk == "btts_yes") & (odds > BTTS_YES_FULLMATCH_MAX)
    valid &= ~fh
    reason = reason.where(~fh, "FIRST_HALF_BTTS_PRICE_GT_3.20")
    d["market_mapping_valid"] = valid.to_numpy()
    d["mapping_reason"] = reason.to_numpy()
    d["line_selection"] = np.where(backfill & mk.isin(OU_MARKETS) & (source == "newformat_odds_history"),
                                   "MAIN_LINE_ONLY", "AS_CAPTURED")
    return d


# ── price contract helpers used by tests and by the integrity checks ─────────────────────────────
def is_devigged(state: str) -> bool:
    """Only a two-sided pair may be called de-vigged."""
    return str(state) == "DEVIGGED_TWO_SIDED"


def entry_price_valid(quote_ts, kickoff) -> bool:
    """A quote at or after kickoff is never an entry price. Unknown times fail closed."""
    q = pd.to_datetime(quote_ts, errors="coerce", utc=True)
    k = pd.to_datetime(kickoff, errors="coerce", utc=True)
    if pd.isna(q) or pd.isna(k):
        return False
    return bool(q < k)


PRICE_DERIVED_FEATURES = {"bookmaker_overround", "implied_prob_over", "implied_prob_under",
                          "api_implied_over25", "api_overround", "api_implied_btts",
                          "api_implied_over35", "api_implied_over15", "api_implied_draw"}


def price_features_in(feature_cols) -> list:
    return sorted(set(feature_cols) & PRICE_DERIVED_FEATURES)
