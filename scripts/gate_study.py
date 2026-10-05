"""§7/§17 — threshold evidence per `league x market x model_type`, with windows that cannot leak.

    python scripts/gate_study.py

THE UNIT IS league x market x model_type AND NOTHING COARSER. Main O/U and BTTS in one league
are different questions with different evidence, and in production they already have different
gates — Championship runs SNIPER 0.07 on O/U and 0.12 on BTTS. Pooling them would average two
unrelated signals and call the result a league.

WINDOWS, chronological, never random, never reused:

    FIT          earliest 50% of that cell's backtest      choose a candidate threshold
    VALIDATION   next 20%                                  does it survive the next period?
    HOLDOUT      final 30%                                 untouched until the end
    LIVE         settled bets since the performance cutoff  independent, real prices

NOTHING HERE CAN ASSIGN `CONFIRMED`. The top grade a retrospective script may award is
FORWARD_TEST. CONFIRMED requires a preregistered future window, which by definition cannot
exist in a file that is scoring the past.

WHAT THE LIVE LEDGER CAN AND CANNOT ANSWER (§5). It is selection-biased: only fixtures that
cleared the LIVE gate are in it. So it answers "what if I had raised the bar?" honestly, and it
cannot answer "what if I had lowered it?" at all — the fixtures below the production floor were
never recorded. Any candidate threshold BELOW the cell's effective production gate is therefore
reported as LEDGER_CENSORED rather than given a number, and only the backtest can speak to it.

EDGE THAT DOES NOT RANK IS NOT A THRESHOLD PROBLEM. If ROI falls as the threshold rises, no
threshold fixes it; the cell is returned as EDGE_NOT_RANKING_OUTCOMES. Reporting the argmax of a
declining curve is how a search produces a confident wrong answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.gate_resolver import resolve_effective_tier_gates, regime_id  # noqa: E402

OUT_JSON = config.OUTPUT_DIR / "league_market_gate_registry.json"
OUT_CSV = config.OUTPUT_DIR / "league_market_gate_research.csv"

GRID = np.round(np.arange(0.02, 0.31, 0.01), 3)
CUTOFF = "2026-08-10"
MIN_FIT, MIN_VAL, MIN_HOLD, MIN_LIVE = 40, 15, 20, 12
N_BOOT = 2000

# Evidence ladder. Retrospective code may never exceed FORWARD_TEST.
NO_DATA, INSUFFICIENT = "NO_DATA", "INSUFFICIENT"
NO_SIGNAL, NEGATIVE = "NO_SIGNAL", "NEGATIVE_SIGNAL"
DISCOVERY, CANDIDATE, VALIDATABLE, FORWARD = "DISCOVERY", "CANDIDATE", "VALIDATABLE", "FORWARD_TEST"
NOT_RANKING = "EDGE_NOT_RANKING_OUTCOMES"
CENSORED = "LEDGER_CENSORED"


def block_ci(pnl: np.ndarray, days: np.ndarray, seed: int = 0) -> tuple:
    """Bootstrap by MATCHDAY. Twenty bets on one Saturday are not twenty observations."""
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
    return round(float(np.nanpercentile(m, 2.5)), 4), round(float(np.nanpercentile(m, 97.5)), 4)


def _metrics(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"n": 0}
    pnl = d["pnl"].to_numpy(dtype=float)
    o = pd.to_numeric(d["odds"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(o) & (o > 1)
    hit = float((d["result"].astype(str).str.upper() == "WIN").mean())
    # mean(1/odds) — NOT 1/mean(odds). Jensen's inequality; the wrong one flattered SNIPER by
    # 1.5pp on this estate.
    be = float(np.mean(1.0 / o[ok])) if ok.any() else float("nan")
    days = d["date"].dt.floor("D").astype("int64").to_numpy()
    lo, hi = block_ci(pnl, days)
    out = {"n": int(len(d)), "matchdays": int(len(np.unique(days))),
           "pnl": round(float(pnl.sum()), 2), "roi": round(float(pnl.mean()), 4),
           "hit": round(hit, 4),
           "break_even": round(be, 4) if np.isfinite(be) else None,
           "excess_hit": round(hit - be, 4) if np.isfinite(be) else None,
           "mean_price": round(float(np.mean(o[ok])), 3) if ok.any() else None,
           "ci_lo": lo, "ci_hi": hi}
    if "clv_pct" in d.columns:
        c = pd.to_numeric(d["clv_pct"], errors="coerce").dropna()
        if len(c) >= 5:
            cd = d.loc[c.index, "date"].dt.floor("D").astype("int64").to_numpy()
            clo, chi = block_ci(c.to_numpy(dtype=float), cd)
            out.update(clv_n=int(len(c)), mean_clv=round(float(c.mean()), 4),
                       median_clv=round(float(c.median()), 4),
                       positive_clv_rate=round(float((c > 0).mean()), 4),
                       clv_ci_lo=clo, clv_ci_hi=chi)
    return out


def _concentration(d: pd.DataFrame) -> dict:
    """§16. A gate must not qualify because three long-odds wins carried it."""
    if d.empty or d["pnl"].sum() == 0:
        return {}
    tot = float(d["pnl"].sum())
    out = {}
    for name, col in (("month", d["date"].dt.to_period("M").astype(str)),
                      ("side", d.get("side")), ("league", d.get("league"))):
        if col is None:
            continue
        g = d.groupby(col.astype(str))["pnl"].sum().sort_values(ascending=False)
        if len(g):
            out[f"top_{name}"] = str(g.index[0])
            out[f"top_{name}_share"] = round(float(g.iloc[0] / tot), 3) if tot else None
    top5 = d.nlargest(min(5, len(d)), "pnl")["pnl"].sum()
    out["top5_winners_share"] = round(float(top5 / tot), 3) if tot else None
    out["roi_excl_top5"] = round(float(d.drop(d.nlargest(min(5, len(d)), "pnl").index)["pnl"]
                                       .mean()), 4) if len(d) > 5 else None
    return out


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest (fit/val/holdout) and live settled bets, both tagged with market + track."""
    bt = []
    for f, mkt, track in (("backtest_results_standard.csv", "ou25", "standard"),
                          ("backtest_results_newformat.csv", "ou25", "new_format"),
                          ("backtest_results_btts.csv", "btts", None),
                          ("backtest_results_over15.csv", "over15", None),
                          ("backtest_results_over35.csv", "over35", None)):
        p = config.OUTPUT_DIR / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        # The side-market backtests name the column `edge`; the main O/U ones name it
        # `best_edge`. The first version required `best_edge` and silently dropped all three
        # side-market files, which is why 20 cells came back NO_DATA on the first run.
        if "best_edge" not in d.columns and "edge" in d.columns:
            d = d.assign(best_edge=d["edge"])
        if "best_edge" not in d.columns or "pnl" not in d.columns:
            print(f"  [load] {f}: no usable edge column — skipped")
            continue
        d = d.assign(market=mkt)
        d["model_type"] = track if track else d["league"].map(config.model_type_for_league)
        bt.append(d[["date", "league", "market", "model_type", "best_edge", "pnl",
                     "odds_over25", "odds_under25"] if "odds_over25" in d.columns
                    else ["date", "league", "market", "model_type", "best_edge", "pnl"]])
    bt = pd.concat(bt, ignore_index=True) if bt else pd.DataFrame()
    if len(bt):
        bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
        bt["pnl"] = pd.to_numeric(bt["pnl"], errors="coerce")
        bt["best_edge"] = pd.to_numeric(bt["best_edge"], errors="coerce")
        bt = bt[bt["date"].notna() & bt["pnl"].notna() & bt["best_edge"].notna()]

    lv = []
    for f, mkt in (("bets_ledger.csv", "ou25"), ("side_bets_ledger.csv", None)):
        p = config.OUTPUT_DIR / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "source" in d.columns:
            d = d[d["source"].astype(str) == "live"]
        d["market"] = mkt if mkt else d["market"]
        d["date"] = pd.to_datetime(d["match_date"], errors="coerce")
        d["pnl"] = pd.to_numeric(d["pnl"], errors="coerce")
        d["best_edge"] = pd.to_numeric(d["edge_pct"], errors="coerce") / 100.0
        d = d[d["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
        if "model_type" not in d.columns:
            d["model_type"] = d["league"].map(config.model_type_for_league)
        blank = d["model_type"].isna() | d["model_type"].astype(str).str.strip().isin(["", "nan"])
        d.loc[blank, "model_type"] = d.loc[blank, "league"].map(config.model_type_for_league)
        lv.append(d)
    lv = pd.concat(lv, ignore_index=True) if lv else pd.DataFrame()
    if len(lv):
        lv = lv[(lv["date"] >= pd.Timestamp(CUTOFF)) & lv["pnl"].notna()
                & lv["best_edge"].notna()]
    return bt, lv


def study_cell(bt: pd.DataFrame, lv: pd.DataFrame, league: str, market: str,
               track: str) -> dict:
    gates = resolve_effective_tier_gates(league, market, track)
    r = {"league": league, "market": market, "model_type": track,
         "production_valuable": gates.valuable, "production_marksman": gates.marksman,
         "production_sniper": gates.sniper, "production_sniper_source": gates.sniper_source,
         "approved_optimizer_applied": gates.approved_optimizer_applied,
         "league_cap_applied": gates.league_cap_applied,
         "regime_id": gates.regime_id,
         "marksman_disabled": bool(gates.marksman >= gates.sniper)}

    b = bt[(bt.league == league) & (bt.market == market) & (bt.model_type == track)] \
        .sort_values("date") if len(bt) else pd.DataFrame()
    l = lv[(lv.league == league) & (lv.market == market) & (lv.model_type == track)] \
        if len(lv) else pd.DataFrame()
    r["backtest_n"], r["live_n"] = int(len(b)), int(len(l))

    if len(b) < MIN_FIT + MIN_VAL + MIN_HOLD:
        # NO BACKTEST BUT REAL LIVE BETS. This is the whole new-format side-market estate —
        # the side-market backtests cover 7 standard leagues only, so Argentina BTTS (the
        # estate's strongest live cell, 82 settled bets) has no historical rows at all.
        # Reporting NO_DATA there would discard the only evidence that exists.
        #
        # A LIVE-ONLY cell can be graded, but ONLY upward from the production gate: the ledger
        # contains no fixture that failed the live floor, so nothing below it is observable.
        if len(l) >= MIN_LIVE:
            at_gate = l[l.best_edge >= gates.sniper]
            r["live_only"] = True
            r["live_evaluable_from"] = gates.sniper
            r["live_note"] = ("no backtest for this cell; graded on live bets only, and only "
                              "at or above the production gate — the ledger cannot see below it")
            m = _metrics(at_gate)
            r.update({f"live_{k}": v for k, v in m.items()})
            r["roi_live"] = m.get("roi")
            if m.get("n", 0) >= MIN_LIVE:
                r["concentration"] = _concentration(at_gate)
                lo = m.get("ci_lo")
                if lo is not None and lo > 0:
                    r.update(evidence_grade=FORWARD,
                             reason=("live-only: positive with a matchday-block CI excluding "
                                     "zero. Not CONFIRMED — no preregistered forward window."))
                elif m.get("roi", 0) <= 0:
                    r.update(evidence_grade=NEGATIVE,
                             reason="live-only: negative at the production gate")
                else:
                    r.update(evidence_grade=CANDIDATE,
                             reason="live-only: positive but the CI spans zero")
            else:
                r.update(evidence_grade=INSUFFICIENT,
                         reason=f"only {m.get('n', 0)} live bets at or above the production gate")
            return r
        r.update(evidence_grade=NO_DATA if len(b) == 0 else INSUFFICIENT,
                 reason=f"{len(b)} backtest rows and {len(l)} live bets — neither is enough")
        return r

    q1, q2 = b["date"].quantile(0.50), b["date"].quantile(0.70)
    fit, val, hold = b[b.date <= q1], b[(b.date > q1) & (b.date <= q2)], b[b.date > q2]
    r["windows"] = {"fit": [str(fit.date.min())[:10], str(fit.date.max())[:10], len(fit)],
                    "validation": [str(val.date.min())[:10], str(val.date.max())[:10], len(val)],
                    "holdout": [str(hold.date.min())[:10], str(hold.date.max())[:10], len(hold)]}

    # Full curve on FIT — kept whole, not just the winner (§14).
    curve = []
    for t in GRID:
        s = fit[fit.best_edge >= t]
        if len(s) < MIN_FIT:
            continue
        curve.append({"threshold": float(t), "n": int(len(s)),
                      "roi": round(float(s.pnl.mean()), 4)})
    if not curve:
        r.update(evidence_grade=INSUFFICIENT,
                 reason=f"no threshold reaches {MIN_FIT} bets in the fit window")
        return r
    r["fit_curve"] = curve
    cdf = pd.DataFrame(curve)
    mono = float(cdf.threshold.corr(cdf.roi)) if len(cdf) > 2 else float("nan")
    r["edge_roi_monotonicity"] = round(mono, 3) if np.isfinite(mono) else None

    if np.isfinite(mono) and mono < 0:
        r.update(evidence_grade=NOT_RANKING,
                 reason=(f"ROI falls as the threshold rises (corr {mono:+.2f}). No threshold "
                         f"repairs an edge measure that does not rank outcomes."))
        return r

    best = cdf.loc[cdf.roi.idxmax()]
    thr = float(best.threshold)
    r.update(candidate_sniper=thr, fit_roi=float(best.roi), fit_n=int(best.n))
    # The ladder is derived, not independently optimised (§4/§8): three free parameters per cell
    # across hundreds of cells is a guaranteed overfit.
    r["candidate_marksman"] = round(max(GRID[0], thr * 0.75), 3)
    r["candidate_valuable"] = round(max(GRID[0], thr * 0.40), 3)

    for name, w, minn in (("validation", val, MIN_VAL), ("holdout", hold, MIN_HOLD)):
        s = w[w.best_edge >= thr]
        r[f"{name}_n"] = int(len(s))
        r[f"roi_{name}"] = round(float(s.pnl.mean()), 4) if len(s) >= minn else None

    # LIVE — censored below the production gate (§5).
    if thr < gates.sniper - 1e-9:
        r["live_status"] = CENSORED
        r["live_note"] = (f"candidate {thr:.2f} is BELOW the production gate {gates.sniper:.2f}; "
                          f"fixtures under the live floor never entered the ledger, so the "
                          f"ledger cannot evaluate lowering the bar")
        r["roi_live"] = None
    else:
        s = l[l.best_edge >= thr]
        r["live_n_at_candidate"] = int(len(s))
        if len(s) >= MIN_LIVE:
            m = _metrics(s)
            r.update({f"live_{k}": v for k, v in m.items()})
            r["roi_live"] = m["roi"]
            r["concentration"] = _concentration(s)
        else:
            r["roi_live"] = None
            r["live_status"] = INSUFFICIENT

    # GRADE — deliberately hard, and capped below CONFIRMED.
    rv, rh, rl = r.get("roi_validation"), r.get("roi_holdout"), r.get("roi_live")
    oos = [x for x in (rv, rh) if x is not None]
    if not oos and rl is None:
        r.update(evidence_grade=DISCOVERY, reason="fitted; no out-of-sample window has enough bets")
    elif oos and all(x <= 0 for x in oos) and (rl is None or rl <= 0):
        r.update(evidence_grade=NEGATIVE, reason="negative in every out-of-sample window")
    elif rl is not None and r.get("live_ci_lo") is not None and r["live_ci_lo"] > 0:
        r.update(evidence_grade=FORWARD,
                 reason=("positive on live bets with a matchday-block CI excluding zero — the "
                         "highest grade retrospective code may award; CONFIRMED needs a "
                         "preregistered forward window"))
    elif oos and any(x > 0 for x in oos):
        r.update(evidence_grade=CANDIDATE,
                 reason="survives at least one out-of-sample window; live CI spans zero or is thin")
    else:
        r.update(evidence_grade=NO_SIGNAL, reason="no window supports the fitted threshold")
    return r


def main() -> int:
    bt, lv = _load()
    if bt.empty and lv.empty:
        print("no backtest or ledger data found")
        return 1
    cells = set()
    for d in (bt, lv):
        if len(d):
            cells |= set(map(tuple, d[["league", "market", "model_type"]]
                             .dropna().drop_duplicates().to_numpy()))
    print(f"regime {regime_id()} | backtest {len(bt):,} rows | live {len(lv):,} settled")
    print(f"{len(cells)} league x market x model_type cells\n")

    rows = [study_cell(bt, lv, lg, mk, tr) for lg, mk, tr in sorted(cells)]

    hdr = (f"{'league':<26}{'market':<8}{'track':<11}{'prodS':>6}{'cand':>6}"
           f"{'fitROI':>8}{'valROI':>8}{'holdROI':>9}{'liveROI':>9}{'liveN':>6}  grade")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x.get("evidence_grade", ""), x["league"])):
        f = lambda k: (r.get(k) if r.get(k) is not None else float("nan"))  # noqa: E731
        print(f"{r['league'][:25]:<26}{r['market']:<8}{r['model_type']:<11}"
              f"{r['production_sniper']:>6.2f}{f('candidate_sniper'):>6.2f}"
              f"{f('fit_roi'):>8.3f}{f('roi_validation'):>8.3f}{f('roi_holdout'):>9.3f}"
              f"{f('roi_live'):>9.3f}{r.get('live_n', 0):>6}  {r.get('evidence_grade')}")

    by = {}
    for r in rows:
        by.setdefault(r.get("evidence_grade"), []).append(f"{r['league']}|{r['market']}")
    print("\nevidence grades:")
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:<28}{len(v)}")
    print(f"\nCONFIRMED cells: 0 — retrospective code cannot award it by construction.")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "regime_id": regime_id(),
        "unit": "league x market x model_type",
        "windows": "fit 50% / validation 20% / holdout 30% by date, then live since " + CUTOFF,
        "hypotheses_examined": len(cells) * len(GRID),
        "caveats": [
            "The live ledger is selection-biased: only fixtures clearing the production gate "
            "are in it, so any candidate BELOW that gate is LEDGER_CENSORED and unevaluable.",
            "CONFIRMED is unreachable here. FORWARD_TEST is the ceiling for retrospective code.",
            "Markets and model types are never pooled.",
        ],
        "cells": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_CSV.name} and {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
