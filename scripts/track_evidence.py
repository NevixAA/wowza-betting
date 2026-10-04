"""§4 — standard and new-format are different deployment risks. Measure them as such.

    python scripts/track_evidence.py

WHAT THIS IS NOT. It is not a league switch. A cell going NEGATIVE_SIGNAL does not disable
anything, and nothing here writes a threshold, a stake or a config. Disabling a league on a
small sample is how a system fits itself to noise, and this estate has already measured that the
per-league threshold optimiser held out-of-sample in 0 of 23 cells — 13 of which were
INSUFFICIENT_DATA, i.e. never really tested at all. That is the failure this file is designed
not to repeat.

WHAT IT IS. An evidence LADDER, so a cell's status says how much is known rather than how the
last few bets went:

    INSUFFICIENT       too few settled bets to say anything
    NEUTRAL            enough bets, CI spans zero
    NEGATIVE_SIGNAL    CI entirely below zero
    POSITIVE_SIGNAL    CI entirely above zero
    VALIDATABLE        a signal large and well-sampled enough to be worth preregistering
    CONFIRMED          replicated on an independent forward period  (never set here —
                       only a preregistered forward test can award it)

CONFIRMED is deliberately unreachable from this script. Every status below it is retrospective,
and retrospective evidence cannot confirm itself; that is the whole point of the separation.

THE INTERVAL. Bootstrap over bets, resampled in BLOCKS because bets on the same matchday share
weather, scheduling and the same model artefact, so i.i.d. resampling understates the spread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

OUT = config.OUTPUT_DIR / "track_evidence.json"

INSUFFICIENT = "INSUFFICIENT"
NEUTRAL = "NEUTRAL"
NEGATIVE = "NEGATIVE_SIGNAL"
POSITIVE = "POSITIVE_SIGNAL"
VALIDATABLE = "VALIDATABLE"
CONFIRMED = "CONFIRMED"        # intentionally never assigned here

#: Below this a cell is INSUFFICIENT regardless of what the mean looks like. 30 settled bets at
#: these hit rates has a standard error of roughly 9 percentage points — wide enough that almost
#: any mean is compatible with zero.
MIN_N = 30
#: A cell must be this far from break-even, AND have a CI excluding zero, to be VALIDATABLE.
#: Chosen to sit above the noise floor of a 100-bet sample rather than from any observed result.
VALIDATABLE_ROI = 0.05
VALIDATABLE_N = 100
N_BOOT = 2000
BLOCK_DAYS = 1

TODAY = pd.Timestamp("2026-10-04")
WINDOWS = {"since_cutoff": None, "last_60d": 60, "last_30d": 30, "last_14d": 14, "last_7d": 7}


def block_bootstrap_roi(pnl: np.ndarray, day_index: np.ndarray,
                        n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% CI for mean P/L per bet, resampling whole MATCHDAYS rather than individual bets.

    Bets on one matchday share a model artefact, a slate and often a league, so they are not
    independent draws. Resampling bets individually would make every interval look tighter than
    it is, and a too-tight interval is what turns noise into a finding.
    """
    if len(pnl) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    days = np.unique(day_index)
    if len(days) < 2:
        return float("nan"), float("nan")
    by_day = {d: pnl[day_index == d] for d in days}
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        s = np.concatenate([by_day[d] for d in pick])
        means[i] = s.mean() if len(s) else np.nan
    return float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def classify(n: int, roi: float, lo: float, hi: float) -> str:
    if n < MIN_N or not np.isfinite(lo) or not np.isfinite(hi):
        return INSUFFICIENT
    if lo > 0:
        return VALIDATABLE if (n >= VALIDATABLE_N and roi >= VALIDATABLE_ROI) else POSITIVE
    if hi < 0:
        return NEGATIVE
    return NEUTRAL


def _load() -> pd.DataFrame:
    """Main O/U and side markets in one frame, each row tagged with its market."""
    frames = []
    main = config.OUTPUT_DIR / "bets_ledger.csv"
    if main.exists():
        d = pd.read_csv(main)
        d["market"] = "ou25"
        frames.append(d)
    side = config.OUTPUT_DIR / "side_bets_ledger.csv"
    if side.exists():
        frames.append(pd.read_csv(side))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")
    d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
    d["odds"] = pd.to_numeric(d.get("odds"), errors="coerce")
    d = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"]) & d["pnl"].notna()]
    cutoff = pd.Timestamp(str(getattr(config, "PERFORMANCE_CUTOFF_DATE", "2026-08-10")))
    return d[d["match_date"] >= cutoff].copy()


def _odds_band(o):
    if not np.isfinite(o):
        return "unknown"
    for hi, lbl in ((1.80, "<1.80"), (2.00, "1.80-2.00"), (2.50, "2.00-2.50"),
                    (3.00, "2.50-3.00")):
        if o < hi:
            return lbl
    return ">3.00"


def cell(d: pd.DataFrame, seed: int = 0) -> dict:
    pnl = d["pnl"].to_numpy(dtype=float)
    day = d["match_date"].dt.floor(f"{BLOCK_DAYS}D").astype("int64").to_numpy()
    roi = float(pnl.mean()) if len(pnl) else float("nan")
    lo, hi = block_bootstrap_roi(pnl, day, seed=seed)
    wins = int((d["result"].astype(str).str.upper() == "WIN").sum())
    # Break-even hit rate implied by the prices actually taken. mean(1/odds), NOT 1/mean(odds):
    # the two differ by Jensen's inequality and the wrong one flattered SNIPER by 1.5pp here.
    o = d["odds"].to_numpy(dtype=float)
    be = float(np.nanmean(1.0 / o[np.isfinite(o) & (o > 1)])) if np.isfinite(o).any() else float("nan")
    return {"n": int(len(d)), "pnl": round(float(pnl.sum()), 2),
            "roi_per_bet": round(roi, 4) if np.isfinite(roi) else None,
            "hit_rate": round(wins / len(d), 4) if len(d) else None,
            "break_even": round(be, 4) if np.isfinite(be) else None,
            "ci_lo": round(lo, 4) if np.isfinite(lo) else None,
            "ci_hi": round(hi, 4) if np.isfinite(hi) else None,
            "clv_available": int(d["clv_pct"].notna().sum()) if "clv_pct" in d else 0,
            "status": classify(len(d), roi, lo, hi)}


def main() -> int:
    d = _load()
    if d.empty:
        print("no settled bets since the cutoff")
        return 0
    print(f"{len(d):,} settled bets since cutoff, "
          f"{d['match_date'].min().date()} -> {d['match_date'].max().date()}\n")

    report: dict = {"generated_for": str(TODAY.date()), "min_n": MIN_N,
                    "note": ("Retrospective only. CONFIRMED is never assigned here — it requires "
                             "a preregistered forward period. Nothing in this file changes a "
                             "threshold, a stake or a league."),
                    "windows": {}}

    for wname, days in WINDOWS.items():
        w = d if days is None else d[d["match_date"] >= TODAY - pd.Timedelta(days=days)]
        if w.empty:
            continue
        # NEVER POOLED ACROSS MARKET OR TIER. The unit of analysis in this estate is
        # model x tier x league x market, and pooling is not a presentation choice — it changes
        # the answer. Pooling main O/U with the side markets over this window flips the headline
        # sign: pooled, standard reads -42.51u NEGATIVE and new-format +8.89u NEUTRAL; split by
        # market and restricted to staked tiers, standard main O/U is POSITIVE recently and
        # new-format main O/U is the loss. Both are arithmetically correct; only the split one
        # answers a deployment question.
        block: dict = {}
        print(f"=== {wname}  (n={len(w)}) ===")
        staked = w[w["signal_tier"].astype(str).str.upper().isin(("SNIPER", "MARKSMAN"))]
        for mkt in sorted(w["market"].astype(str).unique()):
            for mt in ("standard", "new_format"):
                sub = staked[(staked["market"].astype(str) == mkt)
                             & (staked["model_type"].astype(str) == mt)]
                if sub.empty:
                    continue
                c = cell(sub)
                block[f"{mkt}|{mt}|staked"] = c
                print(f"  {mkt:<7} {mt:<11} n={c['n']:<5} pnl={c['pnl']:+8.2f} "
                      f"roi/bet={c['roi_per_bet']:+.4f}  CI[{c['ci_lo']}, {c['ci_hi']}]"
                      f"  {c['status']}")
        report["windows"][wname] = block
    print()

    # Breakdowns, on the full window only — the short windows cannot support them.
    # STAKED TIERS ONLY. VALUABLE is half-stake paper-ish exposure with a much lower bar, so
    # including it changes what question is being asked.
    full = d[d["signal_tier"].astype(str).str.upper().isin(("SNIPER", "MARKSMAN"))].copy()
    report["breakdowns"] = {}
    for dim, col in (("market", "market"), ("tier", "signal_tier"), ("side", "side"),
                     ("league", "league")):
        if col not in full.columns:
            continue
        g = {}
        for key, sub in full.groupby(full[col].astype(str)):
            g[key] = cell(sub)
        report["breakdowns"][dim] = g

    full = full.assign(odds_band=full["odds"].map(_odds_band))
    report["breakdowns"]["odds_band"] = {k: cell(s) for k, s in full.groupby("odds_band")}
    # Market kept in the key: an odds band means something different in BTTS than in O/U 2.5,
    # and pooling them is the same error that flipped the headline above.
    # The same split per TRACK, because invariant 1 forbids pooling them and the odds>3.00
    # problem was previously shown to be new-format-only.
    report["breakdowns"]["odds_band_by_track"] = {
        f"{mkt}|{mt}|{band}": cell(s)
        for (mkt, mt, band), s in full.groupby(
            [full["market"].astype(str), full["model_type"].astype(str), "odds_band"])}

    print("market x track x odds band, staked tiers (pooling any of the three hides it):")
    for k, c in sorted(report["breakdowns"]["odds_band_by_track"].items()):
        if c["n"] >= 10:
            print(f"  {k:<24} n={c['n']:<5} pnl={c['pnl']:+8.2f} roi/bet={c['roi_per_bet']:+.4f}"
                  f"  {c['status']}")

    flagged = {k: v for k, v in report["breakdowns"]["league"].items()
               if v["status"] in (NEGATIVE, POSITIVE, VALIDATABLE)}
    print(f"\nleagues with a CI excluding zero: {len(flagged)} of "
          f"{len(report['breakdowns']['league'])}")
    for k, v in sorted(flagged.items(), key=lambda kv: kv[1]["roi_per_bet"]):
        print(f"  {k:<30} n={v['n']:<5} roi/bet={v['roi_per_bet']:+.4f}  {v['status']}")

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
