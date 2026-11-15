"""The CALIBRATED edge of a tip - the only edge a person is shown from 2026-11-21 (owner 2026-11-11, "Show
calibrated edge").

    stats = load()                                             # the frozen per-cell statistics, or None
    e = calibrated(frame, stats)                               # Series: probability points over fair, NaN = none
    edge_text(e_pct)                                           # "+3.2 pp (calibrated)" / "edge: not calibrated"

WHAT IT IS. The cell's own calibrated probability minus the de-vigged fair price, in probability points:
    p_cal - fair,  p_cal = fair + lam_lo * (p_used - fair)
with lam_lo the cell's OWN cautious calibration slope of the live TOP-3 rule (src.research.top3_ranking.LIVE_RULE
= R3_CAL_EV_LCB) from THE one frozen statistics record (research freeze, runbook U3e). It is computed through the
research code itself (top3_ranking.rank_key: key = p_cal * odds - 1, so p_cal = (key + 1) / odds) - one formula,
not a copy. No global value: a cell without its own calibration statistic has NO calibrated edge (NaN) and is
shown "edge: not calibrated" - never a raw number dressed up as calibrated.

WHAT IT IS NOT. The raw model edge (p_model - fair) stays stored in the records (edge, edge_pct) for study and is
never displayed as the edge of a future tip. Qualification is unchanged: the cell's approved bar decides.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

#: production market x side -> study market-side of the cell (src/bars_opt/study.MARKET_SIDES)
MSIDE = {("ou25", "OVER"): "over25_over", ("ou25", "UNDER"): "over25_under", ("btts", "YES"): "btts_yes",
         ("over15", "OVER"): "over15_over", ("over15", "YES"): "over15_over", ("over35", "OVER"): "over35_over",
         ("over35", "YES"): "over35_over"}
NOT_CALIBRATED = "edge: not calibrated"


@lru_cache(maxsize=1)
def _load_cached():
    from src import top3_league as T
    return T.load_t3_stats()


def load(refresh: bool = False) -> pd.DataFrame | None:
    """The frozen per-cell statistics (the research freeze record), or None when not frozen / not valid."""
    if refresh:
        _load_cached.cache_clear()
    return _load_cached()


def calibrated(frame: pd.DataFrame, stats: pd.DataFrame | None) -> pd.Series:
    """frame: track, league, mside, p_used, fair, odds. Returns the calibrated edge (probability points, decimal)
    on the same index; NaN where the cell has no calibration statistic or an input is missing."""
    from src.research import top3_ranking as TR
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    if frame is None or frame.empty or stats is None or len(stats) == 0:
        return out
    b = frame[["track", "league", "mside", "p_used", "fair", "odds"]].copy()
    for c in ("p_used", "fair", "odds"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    k = TR.rank_key(b.reset_index(drop=True), stats, TR.LIVE_RULE)
    key = pd.to_numeric(k["key"], errors="coerce").to_numpy(float)
    odds = b["odds"].to_numpy(float)
    fair = b["fair"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (key + 1.0) / odds - fair
    e[~np.isfinite(e)] = np.nan
    out[:] = e
    return out


def edge_text(edge_pct) -> str:
    """The one way an edge is shown: '+3.2 pp (calibrated)', or 'edge: not calibrated'."""
    try:
        v = float(edge_pct)
    except (TypeError, ValueError):
        return NOT_CALIBRATED
    if not math.isfinite(v):
        return NOT_CALIBRATED
    return f"{v:+.1f} pp (calibrated)"


def display_edge_pct(tier, calibrated_pct, raw_pct):
    """The edge a display may show for a record, in percent: a future tip (QUALIFIED) shows ONLY its calibrated
    edge (NaN = not calibrated - never the raw one); a historical row (legacy tier code) keeps its recorded edge."""
    if str(tier).strip().upper() == "QUALIFIED":
        try:
            v = float(calibrated_pct)
        except (TypeError, ValueError):
            return float("nan")
        return v if math.isfinite(v) else float("nan")
    return raw_pct
