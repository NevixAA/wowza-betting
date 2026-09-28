"""Walk-forward backtest for the half-time O/U models. READ-ONLY research.

    python scripts/ht_backtest.py                 # both lines
    python scripts/ht_backtest.py --line 05       # just HT O/U 0.5
    python scripts/ht_backtest.py --walk 2000     # bigger windows, faster

WHY THIS EXISTS. The HT models (model_ht_over05.pkl / model_ht_over15.pkl) are trained and
retrained daily like every other model, but there was NO BACKTEST MODE FOR THEM ANYWHERE. So
the only evidence about them was 38 graded live tips, which showed +15.0pp of overconfidence
and could not say whether that came from a bad model or from a bad SELECTION rule.

That distinction is the whole question, because HT is the only track in v9 that tips on raw
confidence (`p >= 0.75` -> tip OVER) with no reference to the market price. Every other track
computes `edge = model_prob - 1/odds` first.

WHAT IT MEASURES, in two deliberately separate parts:

  PART A — discrimination and calibration, on the FULL HT history.
      Needs no odds, so it runs on every fixture that has a half-time score. Answers: does the
      model rank fixtures correctly (AUC), and does it mean what it says (calibration), and
      SPECIFICALLY does it still mean what it says in the p>=0.75 tail where tips are actually
      sent? A model can be well calibrated on average and badly calibrated exactly where it is
      used, and the average is what a training metric reports.

  PART B — is the model better than the price, on the priced subset only.
      Only ~551 fixtures carry a captured closing HT pair (capture began 2026-08-21), so this
      part is small and is reported with its n every time.

A FILLED PRICE IS NOT A PRICE. This follows run_side_market_backtest's hard-won rule: over15
was once certified at +13.5% ROI on 12,186 of 12,187 rows priced at a CONSTANT 1.40, which made
"edge" a bare model-probability threshold with no market in it. Here, a fixture with no captured
HT pair is scored and contributes to calibration, but can never be a bet and never enters ROI.

NO LEAKAGE. Same protocol as the existing backtests: sort by date, train only on rows strictly
before the test window, retrain each window. Nothing about a fixture's own result, or any later
fixture, is available when it is scored.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.model import train as train_model, predict_proba

log = logging.getLogger("ht_backtest")

#: Tips are sent above these. Calibration is reported here specifically, not just overall.
LIVE_TIP_BANDS = {
    "05": [("OVER  p>=0.75", "over", 0.75), ("UNDER p<=0.30", "under", 0.30)],
    "15": [("OVER  p>=0.60", "over", 0.60), ("UNDER p<=0.25", "under", 0.25)],
}


def _load_features() -> pd.DataFrame:
    """Build the feature frame the HT models train on, WITHOUT touching production files.

    `src.data_loader.load_all_matches()` is the normal path, but it is also a WRITE: it runs
    API-Football enrichment and banks the result into af_history.parquet and
    training_coverage.json. A research script must not mutate training inputs, so this reads
    the same committed history and runs only feature engineering over it.
    """
    import os
    # RUN WITH THE API KEY UNSET. _enrich_with_api_shots skips its live fetch silently when
    # APIFOOTBALL_KEY is absent, but still merges everything already banked in
    # af_history.parquet -- so the frame is identical in content while costing zero API calls
    # and, crucially, not re-banking anything. The first run of this script without that guard
    # spent several minutes fetching shots league by league and modified af_history.parquet.
    _key = os.environ.pop("APIFOOTBALL_KEY", None)
    _cfgkey = getattr(config, "API_KEY", "")
    config.API_KEY = ""
    import src.data_loader as _dl
    from src.feature_engineering import build_features

    # _report_training_coverage writes output/training_coverage.json. That is a banking step,
    # not part of producing the frame; a research script must not mutate a training artifact.
    _saved = getattr(_dl, "_report_training_coverage", None)
    if _saved is not None:
        _dl._report_training_coverage = lambda *a, **k: None
    try:
        raw = _dl.load_all_matches()
        return build_features(raw)
    finally:
        if _saved is not None:
            _dl._report_training_coverage = _saved
        config.API_KEY = _cfgkey
        if _key is not None:
            os.environ["APIFOOTBALL_KEY"] = _key


def _ht_odds_table() -> pd.DataFrame:
    """Closing HT pair per fixture+line, resolved league-scoped (invariant 11)."""
    from src.team_names import resolve
    p = config.OUTPUT_DIR / "standard_sidemarket_odds_history.csv"
    if not p.exists():
        return pd.DataFrame()
    oh = pd.read_csv(p).sort_values("snapshot_ts")
    oh = oh[oh["market"].astype(str).str.startswith("ht_")].copy()
    if oh.empty:
        return pd.DataFrame()
    mm = oh["match"].astype(str)
    oh["_h"] = mm.apply(lambda m: m.split(" vs ")[0].strip() if " vs " in m else "")
    oh["_a"] = mm.apply(lambda m: m.split(" vs ")[1].strip() if " vs " in m else "")
    oh["_dk"] = oh["match_date"].astype(str).str[:10]
    rows = []
    for (dk, lg, h, a), g in oh.groupby(["_dk", "league", "_h", "_a"], sort=False):
        rec = {"_dk": dk, "league": lg, "_h": h, "_a": a}
        for line in ("05", "15"):
            o = g[g["market"] == f"ht_over{line}"]
            u = g[g["market"] == f"ht_under{line}"]
            if o.empty or u.empty:
                continue
            oo, uo = float(o.iloc[-1]["odds"]), float(u.iloc[-1]["odds"])
            if oo <= 1 or uo <= 1:
                continue
            ov = 1 / oo + 1 / uo
            rec[f"o{line}_over_odds"] = oo
            rec[f"o{line}_under_odds"] = uo
            rec[f"o{line}_p_devig"] = (1 / oo) / ov          # proportional de-vig
            rec[f"o{line}_overround"] = ov
        rows.append(rec)
    return pd.DataFrame(rows)


def _attach_odds(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join captured HT prices onto the scored frame, resolving club names per league+day."""
    from src.team_names import resolve
    if odds.empty:
        for line in ("05", "15"):
            for c in ("over_odds", "under_odds", "p_devig", "overround"):
                df[f"o{line}_{c}"] = np.nan
        return df
    df = df.copy()
    df["_dk"] = df["date"].astype(str).str[:10]
    by = {k: g for k, g in odds.groupby(["_dk", "league"], sort=False)}
    cols = [c for c in odds.columns if c.startswith("o05_") or c.startswith("o15_")]
    for c in cols:
        df[c] = np.nan
    for i, r in df.iterrows():
        g = by.get((r["_dk"], r["league"]))
        if g is None or g.empty:
            continue
        h = resolve(str(r["home_team"]), list(g["_h"]))
        a = resolve(str(r["away_team"]), list(g["_a"]))
        if not h or not a:
            continue
        m = g[(g["_h"] == h) & (g["_a"] == a)]
        if len(m) != 1:
            continue
        for c in cols:
            df.at[i, c] = m.iloc[0][c]
    return df


def _calib_report(p: pd.Series, y: pd.Series, label: str, bins=None) -> pd.DataFrame:
    bins = bins if bins is not None else [0, .2, .3, .4, .5, .6, .7, .75, .8, .9, 1.0]
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    if d.empty:
        return pd.DataFrame()
    d["bin"] = pd.cut(d["p"], bins, include_lowest=True)
    g = (d.groupby("bin", observed=True)
           .agg(n=("y", "size"), claimed=("p", "mean"), realised=("y", "mean"))
           .reset_index())
    g["gap_pp"] = ((g["claimed"] - g["realised"]) * 100).round(1)
    g["reliable"] = g["n"] >= 30
    return g


def run(line: str, df: pd.DataFrame, walk: int, min_train: int) -> pd.DataFrame:
    """Walk forward over one HT line and return the scored out-of-sample frame."""
    target = f"ht_over{line}"
    need_feat = f"home_ht_over{line}_rate"
    d = df.dropna(subset=[target, need_feat]).sort_values("date").reset_index(drop=True)
    log.info(f"[ht_over{line}] {len(d):,} rows with a HT result  "
             f"({d['date'].min().date()} -> {d['date'].max().date()}, "
             f"{d['league'].nunique()} leagues)")
    if len(d) < min_train + walk:
        log.error(f"[ht_over{line}] not enough data: {len(d)} < {min_train + walk}")
        return pd.DataFrame()

    out = []
    for w, start in enumerate(range(min_train, len(d), walk)):
        train_df = d.iloc[:start]
        test_df = d.iloc[start:start + walk].copy()
        if test_df.empty:
            break
        try:
            res = train_model(train_df, target=target, train_ratio=0.85)
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"[ht_over{line}] window {w}: training failed — {e}")
            continue
        payload = {"models": {k: v["model"] for k, v in res.items()},
                   "feature_cols": res[next(iter(res))]["feature_cols"]}
        try:
            test_df["p"] = predict_proba(test_df, payload=payload).values
        except Exception as e:                                        # noqa: BLE001
            log.warning(f"[ht_over{line}] window {w}: scoring failed — {e}")
            continue
        out.append(test_df)
        log.info(f"  window {w:>3}: train {len(train_df):>6,} -> test {len(test_df):>4,}  "
                 f"(through {test_df['date'].max().date()})")
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def report(line: str, s: pd.DataFrame) -> dict:
    target = f"ht_over{line}"
    y = s[target].astype(float)
    p = s["p"].astype(float)
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    auc = roc_auc_score(y, p) if y.nunique() > 1 else float("nan")
    brier = brier_score_loss(y, p)
    base = float(((y.mean() - y) ** 2).mean())
    ll = log_loss(y, p.clip(1e-6, 1 - 1e-6))

    print(f"\n{'=' * 78}\nHT OVER {line[0]}.{line[1]} — OUT-OF-SAMPLE, WALK-FORWARD  n={len(s):,}")
    print(f"{'=' * 78}")
    print("\nPART A — does the model discriminate, and does it mean what it says?")
    print(f"  AUC            {auc:.4f}   (0.50 = no ranking ability)")
    print(f"  Brier          {brier:.4f}   vs base-rate-only {base:.4f}  "
          f"-> skill {(1 - brier / base) * 100:+.1f}%")
    print(f"  log loss       {ll:.4f}")
    print(f"  claimed {p.mean():.4f}  realised {y.mean():.4f}  "
          f"overall gap {(p.mean() - y.mean()) * 100:+.2f}pp")
    print("\n  calibration by confidence band (thin bands flagged, never dropped):")
    print(_calib_report(p, y, target).to_string(index=False))

    print("\n  THE BAND THAT ACTUALLY SHIPS — where live tips are sent:")
    tip_rows = []
    for name, side, thr in LIVE_TIP_BANDS[line]:
        m = (p >= thr) if side == "over" else (p <= thr)
        n = int(m.sum())
        if n == 0:
            tip_rows.append({"band": name, "n": 0, "claimed": np.nan,
                             "realised": np.nan, "gap_pp": np.nan})
            continue
        claimed = float(p[m].mean()) if side == "over" else float(1 - p[m].mean())
        realised = float(y[m].mean()) if side == "over" else float(1 - y[m].mean())
        tip_rows.append({"band": name, "n": n, "claimed": round(claimed, 4),
                         "realised": round(realised, 4),
                         "gap_pp": round((claimed - realised) * 100, 1)})
    tip = pd.DataFrame(tip_rows)
    print(tip.to_string(index=False))

    # ── PART B — against the market, priced rows only ────────────────────────────────────
    pcol, ocol, ucol, vcol = (f"o{line}_p_devig", f"o{line}_over_odds",
                              f"o{line}_under_odds", f"o{line}_overround")
    res = {"line": line, "n": int(len(s)), "auc": round(float(auc), 4),
           "brier": round(float(brier), 4), "skill_pct": round((1 - brier / base) * 100, 2),
           "gap_pp": round(float((p.mean() - y.mean()) * 100), 2),
           "tip_bands": tip.to_dict("records")}
    if pcol not in s.columns or s[pcol].notna().sum() == 0:
        print("\nPART B — no captured HT prices overlap this history. "
              "Cannot compare to the market, and no ROI is reported.")
        res["priced_n"] = 0
        return res

    q = s.dropna(subset=[pcol]).copy()
    print(f"\nPART B — is the model better than the PRICE?   priced n={len(q):,} "
          f"({q['date'].min().date()} -> {q['date'].max().date()})")
    mb = brier_score_loss(q[target], q[pcol])
    wb = brier_score_loss(q[target], q["p"])
    # Blend, since "better than the market" and "adds to the market" are different claims.
    blend = 0.5 * q[pcol] + 0.5 * q["p"]
    bb = brier_score_loss(q[target], blend)
    print(f"  Brier — market alone   {mb:.4f}")
    print(f"  Brier — model alone    {wb:.4f}   ({'model' if wb < mb else 'MARKET'} wins)")
    print(f"  Brier — 50/50 blend    {bb:.4f}   "
          f"(adds {mb - bb:+.5f} over the market alone)")
    vig = float(q[vcol].mean())
    be_over = float((1 / q[ocol]).mean())
    print(f"  bookmaker margin {(vig - 1) * 100:.2f}% on the pair "
          f"({(vig - 1) / 2 * 100:.2f}% per side)")
    print(f"  BREAK-EVEN: beat the de-vigged market by "
          f"{(be_over - q[pcol].mean()) * 100:+.2f}pp on OVER to clear the vig.")

    # Edge-based selection at several thresholds, flat 1u, settled at the CLOSING price.
    print("\n  If we had selected on EDGE vs the de-vigged close (flat 1u, closing price):")
    tbl = []
    for thr in (0.02, 0.03, 0.04, 0.05, 0.08):
        eo = q["p"] - q[pcol]
        eu = (1 - q["p"]) - (1 - q[pcol])
        bo = q[eo >= thr]
        bu = q[eu >= thr]
        n = len(bo) + len(bu)
        if n == 0:
            tbl.append({"edge>=": f"{thr:.0%}", "bets": 0, "roi": np.nan, "hit": np.nan})
            continue
        pnl = (np.where(bo[target] == 1, bo[ocol] - 1, -1).sum()
               + np.where(bu[target] == 0, bu[ucol] - 1, -1).sum())
        wins = int((bo[target] == 1).sum() + (bu[target] == 0).sum())
        tbl.append({"edge>=": f"{thr:.0%}", "bets": n, "roi": round(pnl / n, 4),
                    "hit": round(wins / n, 3), "over": len(bo), "under": len(bu)})
    e = pd.DataFrame(tbl)
    print(e.to_string(index=False))
    print("  (small bet counts are noise — read the n column, not the ROI)")

    res.update({"priced_n": int(len(q)), "market_brier": round(float(mb), 4),
                "model_brier": round(float(wb), 4), "blend_brier": round(float(bb), 4),
                "overround": round(vig, 4),
                "breakeven_edge_pp": round(float((be_over - q[pcol].mean()) * 100), 2),
                "edge_selection": e.to_dict("records")})
    return res


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=["05", "15", "both"], default="both")
    ap.add_argument("--walk", type=int, default=1500)
    ap.add_argument("--min-train", type=int, default=8000)
    a = ap.parse_args()

    log.info("Loading history + features (read-only)...")
    df = _load_features()
    log.info(f"  {len(df):,} matches")
    odds = _ht_odds_table()
    log.info(f"  {len(odds):,} fixtures with a captured HT pair")

    lines = ["05", "15"] if a.line == "both" else [a.line]
    results = []
    for line in lines:
        s = run(line, df, a.walk, a.min_train)
        if s.empty:
            continue
        s = _attach_odds(s, odds)
        results.append(report(line, s))
        out = config.OUTPUT_DIR / f"ht_backtest_over{line}.csv"
        keep = [c for c in ("date", "league", "home_team", "away_team", f"ht_over{line}", "p",
                            f"o{line}_over_odds", f"o{line}_under_odds", f"o{line}_p_devig",
                            f"o{line}_overround") if c in s.columns]
        s[keep].to_csv(out, index=False)
        log.info(f"\n  wrote {out}")

    if results:
        import json
        p = config.OUTPUT_DIR / "ht_backtest_summary.json"
        p.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        log.info(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
