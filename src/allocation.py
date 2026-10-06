"""How much bankroll does each `league × market × model` cell deserve? Usually: none yet.

    from src.allocation import allocate, Epoch

A LONG-HORIZON ACCUMULATOR, NOT A DECISION TOOL. The system has been running without known
bugs for about two weeks. Thresholds were refitted on 2026-10-06 after five weeks frozen, and
two league approvals flipped in that single refit. Nothing here should move money yet, and the
output is designed so that it visibly says so: almost every cell returns 0.0% and a reason.

What it is for is the question that becomes answerable in about six months — "which cells have
earned a stake, and how big?" — and for making the WAIT legible: every cell reports how many
more settled bets it needs and, at its own observed rate, roughly how long that takes.

THE FRACTION IS A LOWER-CONFIDENCE-BOUND KELLY, NOT A POINT ESTIMATE.

Plain Kelly on a measured ROI is the fastest way to ruin a bankroll on noise: it is linear in
the edge with zero intercept, so a cell that got lucky is sized as if it were skilled. Full
Kelly on this estate's own numbers ruins 98.9% of simulated paths.

So the edge used is the LOWER bound of a matchday-block bootstrap CI, not the mean. A cell whose
interval includes zero therefore gets exactly 0.0% — not a small number, zero — because the
honest reading of "could be negative" is "do not bet it". Everything else is shrunk again by a
global fraction, because the estate's measured growth-optimal flat stake is 2.2% and log-growth
turns negative above 4.5%.

THREE THINGS THAT WOULD OTHERWISE CORRUPT THIS, AND ARE HANDLED:

1. REGIME. Thresholds changed on 2026-10-06. Bets taken under a different gate are a different
   strategy, and pooling them silently answers a question nobody asked. Every cell records the
   `threshold_regime_id` its evidence came from, and mixing regimes is flagged.

2. SYSTEM EPOCH. Performance before the data fixes landed is not evidence about the current
   system. `RELIABLE_FROM` is separate from `PERFORMANCE_CUTOFF_DATE` on purpose: the cutoff
   excludes pre-fix TIPS, the epoch excludes a period when the PIPELINE itself was broken.

3. CORRELATION. Twenty bets on one Saturday are not twenty observations. Every interval is
   block-bootstrapped by matchday.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: The system has run without a KNOWN pipeline defect only since the September data fixes
#: landed and the retrain began completing. Deliberately conservative, deliberately separate
#: from PERFORMANCE_CUTOFF_DATE, and deliberately a date rather than "recent".
RELIABLE_FROM = "2026-09-24"

#: Minimum settled bets before a cell may be sized at all. Below this the interval is so wide
#: that its lower bound is meaningless rather than merely negative.
MIN_N = 40
#: Minimum distinct matchdays — n=60 across 3 Saturdays is 3 observations, not 60.
MIN_MATCHDAYS = 15
#: Target for a cell to be considered decided rather than merely promising.
TARGET_N = 150

#: Global shrink applied on top of the lower-bound edge. The measured growth-optimal flat
#: fraction on this estate is 2.2%; log-growth turns negative above 4.5%. Quarter-Kelly on an
#: already-conservative edge keeps the total well inside that even when several cells qualify.
KELLY_SHRINK = 0.25
#: No single cell may ever take more than this, however good it looks.
MAX_CELL_FRACTION = 0.02
#: Nor may everything together. Slates correlate; a day where every cell fires must not be a
#: bankroll event.
MAX_TOTAL_FRACTION = 0.20

N_BOOT = 2000

NO_DATA = "NO_DATA"
ACCUMULATING = "ACCUMULATING"
INCONCLUSIVE = "INCONCLUSIVE"
NEGATIVE = "NEGATIVE"
ALLOCATABLE = "ALLOCATABLE"


@dataclass
class Cell:
    league: str
    market: str
    model_type: str
    n: int
    matchdays: int
    pnl: float
    roi: float | None
    hit: float | None
    break_even: float | None
    mean_odds: float | None
    ci_lo: float | None
    ci_hi: float | None
    fraction: float
    state: str
    reason: str
    n_to_target: int
    bets_per_week: float | None
    weeks_to_target: float | None
    regimes: tuple = ()
    first_bet: str = ""
    last_bet: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _block_ci(pnl: np.ndarray, days: np.ndarray, seed: int = 0) -> tuple:
    if len(pnl) == 0:
        return None, None
    uniq = np.unique(days)
    if len(uniq) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    by = {d: pnl[days == d] for d in uniq}
    m = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = np.concatenate([by[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        m[i] = s.mean() if len(s) else np.nan
    return float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5))


def _kelly_fraction(edge_per_unit: float, mean_odds: float) -> float:
    """Kelly on a FLAT-STAKE edge expressed as profit per unit staked.

    f* = edge / (odds - 1). Linear in the edge with zero intercept, which is exactly why the
    edge fed in here must already be a conservative lower bound — scaling a point estimate is
    how a lucky cell gets sized like a skilled one.
    """
    b = float(mean_odds) - 1.0
    if b <= 0 or edge_per_unit <= 0:
        return 0.0
    return max(0.0, edge_per_unit / b)


def _cell(d: pd.DataFrame, league: str, market: str, track: str,
          shrink: float = KELLY_SHRINK) -> Cell:
    pnl = d["pnl"].to_numpy(dtype=float)
    days = d["match_date"].dt.floor("D").astype("int64").to_numpy()
    nd = int(len(np.unique(days)))
    o = pd.to_numeric(d["odds"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(o) & (o > 1)
    mean_odds = float(np.mean(o[ok])) if ok.any() else None
    be = float(np.mean(1.0 / o[ok])) if ok.any() else None
    hit = float((d["result"].astype(str).str.upper() == "WIN").mean()) if len(d) else None
    roi = float(pnl.mean()) if len(pnl) else None
    lo, hi = _block_ci(pnl, days)

    span_days = max(1.0, (d["match_date"].max() - d["match_date"].min()).days)
    rate = len(d) / (span_days / 7.0) if span_days else None
    need = max(0, TARGET_N - len(d))
    weeks = (need / rate) if (rate and rate > 0 and need) else (0.0 if not need else None)

    base = dict(league=league, market=market, model_type=track, n=int(len(d)), matchdays=nd,
                pnl=round(float(pnl.sum()), 2),
                roi=round(roi, 4) if roi is not None else None,
                hit=round(hit, 4) if hit is not None else None,
                break_even=round(be, 4) if be is not None else None,
                mean_odds=round(mean_odds, 3) if mean_odds is not None else None,
                ci_lo=round(lo, 4) if lo is not None else None,
                ci_hi=round(hi, 4) if hi is not None else None,
                n_to_target=need,
                bets_per_week=round(rate, 2) if rate else None,
                weeks_to_target=round(weeks, 1) if weeks is not None else None,
                regimes=tuple(sorted(set(d.get("threshold_regime_id", pd.Series(dtype=str))
                                         .dropna().astype(str)))) or (),
                first_bet=str(d["match_date"].min())[:10],
                last_bet=str(d["match_date"].max())[:10])

    if len(d) < MIN_N:
        return Cell(**base, fraction=0.0, state=ACCUMULATING,
                    reason=f"{len(d)} settled bets, need {MIN_N} before sizing is meaningful")
    if nd < MIN_MATCHDAYS:
        return Cell(**base, fraction=0.0, state=ACCUMULATING,
                    reason=(f"{len(d)} bets but only {nd} matchdays — correlated within days, "
                            f"so this is nearer {nd} observations than {len(d)}"))
    if lo is None:
        return Cell(**base, fraction=0.0, state=ACCUMULATING,
                    reason="too few matchdays for a block bootstrap")
    if hi is not None and hi <= 0:
        return Cell(**base, fraction=0.0, state=NEGATIVE,
                    reason=f"entire CI below zero (upper bound {hi:+.4f}) — do not bet this cell")
    if lo <= 0:
        return Cell(**base, fraction=0.0, state=INCONCLUSIVE,
                    reason=(f"CI includes zero [{lo:+.4f}, {hi:+.4f}] — the honest reading of "
                            f"'could be negative' is zero stake, not a small one"))

    f = _kelly_fraction(lo, mean_odds or 2.0) * shrink
    f = min(f, MAX_CELL_FRACTION)
    return Cell(**base, fraction=round(f, 5), state=ALLOCATABLE,
                reason=(f"lower-bound edge {lo:+.4f} at mean odds {mean_odds:.2f} -> "
                        f"{f:.2%} of bankroll (quarter-Kelly on the CI floor, capped at "
                        f"{MAX_CELL_FRACTION:.0%})"))


def allocate(bets: pd.DataFrame, reliable_from: str = RELIABLE_FROM,
             staked_tiers_only: bool = True) -> tuple[list[Cell], dict]:
    """Every cell, its evidence, and the bankroll fraction it currently justifies.

    `bets` needs: league, market, model_type, match_date, odds, pnl, result, signal_tier.
    """
    d = bets.copy()
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce", utc=True)
    d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
    d = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"]) & d["pnl"].notna()
          & d["match_date"].notna()]
    if staked_tiers_only and "signal_tier" in d.columns:
        # VALUABLE is a collection layer, not money. Sizing a stake from it would allocate real
        # bankroll on the strength of bets nobody ever intended to place.
        d = d[d["signal_tier"].astype(str).str.upper().isin(("SNIPER", "MARKSMAN"))]
    epoch = pd.Timestamp(reliable_from, tz="UTC")
    before = len(d)
    d = d[d["match_date"] >= epoch]

    cells = []
    for (lg, mk, tr), g in d.groupby(["league", "market", "model_type"]):
        cells.append(_cell(g, str(lg), str(mk), str(tr)))
    cells.sort(key=lambda c: (-c.fraction, -(c.roi or -9)))

    total = sum(c.fraction for c in cells)
    scaled = 1.0
    if total > MAX_TOTAL_FRACTION and total > 0:
        # Scale down proportionally rather than dropping cells: the cap is about total exposure
        # on a correlated slate, not about any one cell being wrong.
        scaled = MAX_TOTAL_FRACTION / total
        for c in cells:
            c.fraction = round(c.fraction * scaled, 5)

    summary = {
        "reliable_from": reliable_from,
        "rows_before_epoch_filter": int(before),
        "rows_used": int(len(d)),
        "cells": len(cells),
        "allocatable": sum(1 for c in cells if c.state == ALLOCATABLE),
        "total_fraction": round(sum(c.fraction for c in cells), 5),
        "total_scaled_by": round(scaled, 4),
        "states": {s: sum(1 for c in cells if c.state == s)
                   for s in (ALLOCATABLE, INCONCLUSIVE, NEGATIVE, ACCUMULATING, NO_DATA)},
    }
    return cells, summary


# ── why is this cell losing, and what would fix it ────────────────────────────────────────────
#
# A losing cell is not one problem. These have different fixes and are worth telling apart
# BEFORE anyone turns a league off: "stop betting it" and "it is miscalibrated" lead to opposite
# actions, and only one of them is recoverable.

CAUSE_CALIBRATION = "MODEL_OVERCONFIDENT"
CAUSE_SIDE_BIAS = "ONE_SIDED"
CAUSE_ODDS_BAND = "LONGSHOT_DRAG"
CAUSE_MARKET_BETTER = "MARKET_PRICES_IT_BETTER"
CAUSE_EDGE_NOT_RANKING = "EDGE_NOT_RANKING"
CAUSE_VARIANCE = "LOOKS_LIKE_VARIANCE"
CAUSE_UNKNOWN = "UNDIAGNOSED"


def diagnose(d: pd.DataFrame, min_n: int = 25) -> list[dict]:
    """Ranked causes for a cell's losses, each with the evidence and the fix it implies.

    Returns every cause that fires, worst first, rather than a single verdict — a league can be
    both one-sided AND dragged by longshots, and fixing one would leave the other.

    `d` is one cell's settled bets: odds, pnl, result, side, and model probability if present.
    """
    out: list[dict] = []
    if len(d) < min_n:
        return [{"cause": CAUSE_UNKNOWN, "severity": 0.0,
                 "evidence": f"{len(d)} bets — too few to diagnose",
                 "fix": "accumulate"}]

    pnl = d["pnl"].to_numpy(dtype=float)
    o = pd.to_numeric(d["odds"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(o) & (o > 1)
    won = (d["result"].astype(str).str.upper() == "WIN").to_numpy()
    total = float(pnl.sum())

    # 1. LONGSHOT DRAG — the loss concentrated in one price band. Cheapest thing to fix, because
    #    it needs a price bound rather than a model change.
    if ok.any():
        band = pd.cut(pd.Series(o[ok]), [0, 1.8, 2.2, 2.8, 3.5, 99],
                      labels=["<1.80", "1.80-2.20", "2.20-2.80", "2.80-3.50", ">3.50"])
        g = pd.DataFrame({"band": band, "pnl": pnl[ok]}).groupby("band", observed=True)["pnl"]
        agg = g.agg(["sum", "size"]).sort_values("sum")
        if len(agg) and total < 0:
            worst = agg.iloc[0]
            share = float(worst["sum"]) / total if total else 0
            if share > 0.6 and worst["size"] >= 8:
                out.append({
                    "cause": CAUSE_ODDS_BAND, "severity": round(share, 3),
                    "evidence": (f"{agg.index[0]} is {share:.0%} of the loss on "
                                 f"{int(worst['size'])} of {int(ok.sum())} bets "
                                 f"({worst['sum']:+.2f}u)"),
                    "fix": "a price bound on this cell — no model change needed"})

    # 2. ONE-SIDED — the model only ever takes one side here, so it is a directional position
    #    sized N times, not N independent bets.
    if "side" in d.columns:
        vc = d["side"].astype(str).str.upper().value_counts(normalize=True)
        if len(vc) and vc.iloc[0] > 0.85:
            side = vc.index[0]
            sub = d[d["side"].astype(str).str.upper() == side]
            out.append({
                "cause": CAUSE_SIDE_BIAS, "severity": round(float(vc.iloc[0]), 3),
                "evidence": (f"{vc.iloc[0]:.0%} of bets are {side} "
                             f"({pd.to_numeric(sub['pnl'], errors='coerce').sum():+.2f}u) — "
                             f"one opinion applied {len(sub)} times, not {len(sub)} opinions"),
                "fix": "check the model's mean probability against the league base rate"})

    # 3. OVERCONFIDENCE — claimed win rate far above realised. The estate's measured headline
    #    defect (+13.64pp overall), and it is a calibration fix, not a threshold one.
    pcol = next((c for c in ("model_prob", "p_model", "prob") if c in d.columns), None)
    if pcol:
        p = pd.to_numeric(d[pcol], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(p)
        if m.sum() >= min_n:
            gap = float(p[m].mean() - won[m].mean())
            if gap > 0.05:
                out.append({
                    "cause": CAUSE_CALIBRATION, "severity": round(gap, 4),
                    "evidence": (f"claims {p[m].mean():.1%}, realises {won[m].mean():.1%} "
                                 f"— overconfident by {gap:+.1%} on {int(m.sum())} bets"),
                    "fix": "recalibrate this cell; a higher threshold does not fix a biased "
                           "probability, it just bets fewer of the same wrong numbers"})

    # 4. MARKET PRICES IT BETTER — our hit rate below the break-even the prices imply. Then the
    #    book's number is the better forecast and no threshold rescues the cell.
    if ok.any():
        be = float(np.mean(1.0 / o[ok]))
        hit = float(won[ok].mean())
        if hit < be - 0.03:
            out.append({
                "cause": CAUSE_MARKET_BETTER, "severity": round(be - hit, 4),
                "evidence": (f"hit {hit:.1%} against a {be:.1%} break-even — "
                             f"{be - hit:.1%} short of merely matching the price"),
                "fix": "the market's probability beats ours here; stop betting the cell or "
                       "use the market price as the anchor instead of the model"})

    # 5. VARIANCE — nothing above fired and the interval still spans zero. Saying so is more
    #    useful than inventing a cause.
    if not out:
        days = pd.to_datetime(d["match_date"], errors="coerce", utc=True
                              ).dt.floor("D").astype("int64").to_numpy()
        lo, hi = _block_ci(pnl, days)
        if lo is not None and lo <= 0 <= (hi or 0):
            out.append({
                "cause": CAUSE_VARIANCE, "severity": 0.0,
                "evidence": f"CI [{lo:+.3f}, {hi:+.3f}] spans zero on {len(d)} bets",
                "fix": "no identified defect — keep collecting before acting"})
        else:
            out.append({"cause": CAUSE_UNKNOWN, "severity": 0.0,
                        "evidence": f"{total:+.2f}u over {len(d)} bets, no single cause found",
                        "fix": "inspect manually"})

    return sorted(out, key=lambda x: -x["severity"])
