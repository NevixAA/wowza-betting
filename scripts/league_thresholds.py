"""Find the edge threshold for every league, fitted on the backtest and TESTED on live bets.

    python scripts/league_thresholds.py

THE TRAP THIS IS BUILT TO AVOID. Sweeping a threshold per league and reporting the best ROI is
guaranteed to produce a table of winners — with 30 leagues and ~28 candidate thresholds that is
840 cells, and the best cell in a random matrix of that size looks excellent. This estate has
already paid for it once: a previous per-league tuning held out-of-sample in 0 of 23 cells, and
13 of those were INSUFFICIENT_DATA and had never really been tested at all.

So nothing here is fitted and scored on the same rows. Three separate windows:

    FIT      the earlier portion of the backtest          -> pick the threshold
    TEST     the later portion of the backtest            -> does it survive in-sample-adjacent?
    LIVE     actual settled bets since the cutoff          -> genuinely independent, real prices

A threshold is only reported as USE when it clears all three. Anything else is reported with the
reason it failed, because "we could not establish one" is a result and the honest default is the
global threshold the league already uses.

WHY ROI AND NOT HIT RATE. The break-even hit rate varies with price: 60% is excellent at 2.10
and a disaster at 1.40. ROI per bet already carries the price. Break-even is reported alongside
as mean(1/odds) — NOT 1/mean(odds), which differs by Jensen's inequality and previously flattered
the SNIPER tier by 1.5 percentage points on this estate.

WHAT A THRESHOLD IS HERE. The minimum `best_edge` at which a fixture becomes bettable in that
league. Raising it bets less and should raise ROI if the edge measure means anything; if ROI does
NOT rise with the threshold, that league's edge number carries no information and the correct
answer is not a better threshold, it is no bet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

OUT = config.OUTPUT_DIR / "league_thresholds.json"

#: Candidate thresholds. Coarse on purpose — a threshold that only wins at 0.005 resolution is
#: fitted to the sample, not a finding.
GRID = np.round(np.arange(0.02, 0.31, 0.01), 3)
#: Below this many bets at a threshold, the cell is INSUFFICIENT whatever the ROI looks like.
MIN_BETS_FIT = 40
MIN_BETS_TEST = 15
MIN_BETS_LIVE = 10
#: Fraction of each league's backtest history used to FIT. Chronological, never random.
FIT_FRACTION = 0.70
N_BOOT = 2000
CUTOFF = "2026-08-10"


def _roi(pnl: np.ndarray) -> float:
    return float(pnl.mean()) if len(pnl) else float("nan")


def _block_ci(pnl: np.ndarray, days: np.ndarray, n_boot: int = N_BOOT,
              seed: int = 0) -> tuple[float, float]:
    """Resampled by MATCHDAY — bets on one slate share conditions and a model artefact."""
    if len(pnl) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    if len(uniq) < 3:
        return float("nan"), float("nan")
    by = {d: pnl[days == d] for d in uniq}
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        s = np.concatenate([by[d] for d in pick])
        out[i] = s.mean() if len(s) else np.nan
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def _bets_at(d: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Fixtures that would be bet at this threshold."""
    return d[pd.to_numeric(d["best_edge"], errors="coerce") >= thr]


def _load_backtest() -> pd.DataFrame:
    frames = []
    for f, track in (("backtest_results_standard.csv", "standard"),
                     ("backtest_results_newformat.csv", "new_format")):
        p = config.OUTPUT_DIR / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["model_type"] = track
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
    d["best_edge"] = pd.to_numeric(d["best_edge"], errors="coerce")
    # A row with no bet has no pnl to learn from.
    return d[d["pnl"].notna() & d["best_edge"].notna() & d["date"].notna()]


def _load_live() -> pd.DataFrame:
    """Settled live bets since the cutoff — the only genuinely independent evidence here."""
    p = config.OUTPUT_DIR / "bets_ledger.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if "source" in d.columns:
        d = d[d["source"].astype(str) == "live"]
    d["date"] = pd.to_datetime(d["match_date"], errors="coerce")
    d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
    # edge_pct is in PERCENT in the ledger.
    d["best_edge"] = pd.to_numeric(d["edge_pct"], errors="coerce") / 100.0
    d = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
    return d[(d["date"] >= pd.Timestamp(CUTOFF)) & d["pnl"].notna() & d["best_edge"].notna()]


def study_league(bt: pd.DataFrame, live: pd.DataFrame, league: str) -> dict:
    b = bt[bt["league"] == league].sort_values("date")
    lv = live[live["league"] == league]
    track = str(b["model_type"].iloc[0]) if len(b) else (
        config.model_type_for_league(league))
    res = {"league": league, "model_type": track,
           "n_backtest": int(len(b)), "n_live": int(len(lv)),
           "current_threshold": float(config.LEAGUE_SNIPER_THRESHOLDS.get(
               league, config.SNIPER_THRESHOLD))}

    if len(b) < MIN_BETS_FIT + MIN_BETS_TEST:
        res.update(status="INSUFFICIENT_BACKTEST",
                   reason=f"{len(b)} backtest rows, need >= {MIN_BETS_FIT + MIN_BETS_TEST}")
        return res

    cut = b["date"].quantile(FIT_FRACTION)
    fit, test = b[b["date"] <= cut], b[b["date"] > cut]
    res["fit_span"] = [str(fit["date"].min())[:10], str(fit["date"].max())[:10]]
    res["test_span"] = [str(test["date"].min())[:10], str(test["date"].max())[:10]]

    # FIT: best ROI on the training window, subject to a minimum bet count.
    rows = []
    for thr in GRID:
        f = _bets_at(fit, thr)
        if len(f) < MIN_BETS_FIT:
            continue
        rows.append({"thr": float(thr), "n": len(f), "roi": _roi(f["pnl"].to_numpy())})
    if not rows:
        res.update(status="INSUFFICIENT_AT_EVERY_THRESHOLD",
                   reason=f"no threshold reaches {MIN_BETS_FIT} bets in the fit window")
        return res

    grid = pd.DataFrame(rows)
    best = grid.loc[grid["roi"].idxmax()]
    thr = float(best["thr"])
    res["fitted_threshold"] = thr
    res["fit_roi"] = round(float(best["roi"]), 4)
    res["fit_n"] = int(best["n"])

    # Does ROI actually RISE with the threshold? If not, the edge number carries no information
    # in this league and no threshold fixes that.
    res["roi_edge_correlation"] = round(float(grid["thr"].corr(grid["roi"])), 3)

    # TEST on the held-out tail of the backtest.
    t = _bets_at(test, thr)
    res["test_n"] = int(len(t))
    if len(t) >= MIN_BETS_TEST:
        res["test_roi"] = round(_roi(t["pnl"].to_numpy()), 4)
    else:
        res["test_roi"] = None

    # LIVE — independent, real prices, after the cutoff.
    l = _bets_at(lv, thr)
    res["live_n"] = int(len(l))
    if len(l) >= MIN_BETS_LIVE:
        p = l["pnl"].to_numpy(dtype=float)
        days = l["date"].dt.floor("D").astype("int64").to_numpy()
        res["live_roi"] = round(_roi(p), 4)
        lo, hi = _block_ci(p, days)
        res["live_ci"] = [round(lo, 4) if np.isfinite(lo) else None,
                          round(hi, 4) if np.isfinite(hi) else None]
        o = pd.to_numeric(l.get("odds"), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(o) & (o > 1)
        res["live_break_even"] = round(float(np.mean(1.0 / o[ok])), 4) if ok.any() else None
        res["live_hit"] = round(float((l["result"].astype(str).str.upper() == "WIN").mean()), 4)
    else:
        res["live_roi"] = res["live_ci"] = None

    # VERDICT — deliberately hard to pass.
    if res["test_roi"] is None and res["live_roi"] is None:
        res.update(status="UNTESTED", reason="fitted, but no out-of-sample window has enough bets")
    elif res["roi_edge_correlation"] is not None and res["roi_edge_correlation"] < 0:
        res.update(status="NO_SIGNAL",
                   reason=(f"ROI falls as the threshold rises (corr "
                           f"{res['roi_edge_correlation']:+.2f}) — the edge measure carries no "
                           f"information in this league, so no threshold fixes it"))
    elif (res["test_roi"] is not None and res["test_roi"] <= 0) and \
         (res["live_roi"] is None or res["live_roi"] <= 0):
        res.update(status="FAILED_OOS",
                   reason="the fitted threshold does not hold out of sample")
    elif res["live_roi"] is not None and res["live_ci"] and res["live_ci"][0] is not None \
            and res["live_ci"][0] > 0:
        res.update(status="USE", reason="positive on live bets with a CI excluding zero")
    else:
        res.update(status="CANDIDATE",
                   reason="survives out of sample but its live CI still spans zero")
    return res


def main() -> int:
    bt, live = _load_backtest(), _load_live()
    if bt.empty:
        print("no backtest results on disk — run `python pipeline.py --mode backtest` first")
        return 1
    leagues = sorted(set(bt["league"].dropna()) | set(live["league"].dropna()))
    print(f"backtest {len(bt):,} rows / {bt['league'].nunique()} leagues "
          f"({str(bt['date'].min())[:10]} -> {str(bt['date'].max())[:10]})")
    print(f"live     {len(live):,} settled bets since {CUTOFF}\n")

    out = [study_league(bt, live, lg) for lg in leagues]
    out.sort(key=lambda r: (r.get("status", ""), -(r.get("live_roi") or -9)))

    hdr = (f"{'league':<28}{'track':<12}{'now':>6}{'fit':>6}{'fitROI':>8}"
           f"{'testROI':>9}{'liveROI':>9}{'liveN':>7}  status")
    print(hdr); print("-" * len(hdr))
    for r in out:
        f = r.get("fitted_threshold")
        print(f"{r['league'][:27]:<28}{r['model_type']:<12}{r['current_threshold']:>6.2f}"
              f"{(f if f else float('nan')):>6.2f}"
              f"{(r.get('fit_roi') or float('nan')):>8.3f}"
              f"{(r.get('test_roi') if r.get('test_roi') is not None else float('nan')):>9.3f}"
              f"{(r.get('live_roi') if r.get('live_roi') is not None else float('nan')):>9.3f}"
              f"{r.get('live_n', 0):>7}  {r.get('status')}")

    by = {}
    for r in out:
        by.setdefault(r["status"], []).append(r["league"])
    print("\nsummary:")
    for k, v in sorted(by.items()):
        print(f"  {k:<32} {len(v)}")

    use = [r for r in out if r["status"] == "USE"]
    print(f"\n{len(use)} league(s) cleared all three windows:")
    for r in use:
        print(f"  {r['league']:<28} threshold {r['fitted_threshold']:.2f} "
              f"(now {r['current_threshold']:.2f})  live ROI {r['live_roi']:+.3f} "
              f"CI{r['live_ci']}  n={r['live_n']}")
    if not use:
        print("  none — the honest answer is to keep the global threshold.")

    OUT.write_text(json.dumps({
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": ("fitted on the first 70% of each league's backtest by date, tested on the "
                   "remaining 30%, and tested again on live settled bets since " + CUTOFF
                   + ". A threshold is USE only when the live CI excludes zero."),
        "caveat": ("840 league x threshold cells were examined. Without the out-of-sample "
                   "requirement a table of apparent winners is guaranteed; a previous per-league "
                   "tuning on this estate held out-of-sample in 0 of 23 cells."),
        "grid": [float(x) for x in GRID], "leagues": out}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
