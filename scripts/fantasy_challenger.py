"""§17 — is Wowza's player model a better FPL points projection than FPL's own `ep_next`?

    python scripts/fantasy_challenger.py

THE QUESTION, STATED SO IT CAN FAIL. Invariant 2 says the props model is accurate but has no
betting edge, and that the accuracy is monetised through Fantasy instead. That claim is only
worth anything if the model actually projects FPL points better than the number FPL gives away
for free. `ep_next` is the incumbent and it costs nothing, so the model has to beat it — not
beat "nothing".

THE BRIEF ALSO ALLOWS A THIRD ANSWER: a blend may be better than either source alone, even when
neither wins outright. Two signals with uncorrelated errors usually blend well. So this does not
test "Wowza vs FPL" as a binary; it sweeps the blend weight from 0.0 to 1.0 and reports the
argmin. A weight of 0.0 is a real finding and not a failed test — it says Wowza's information is
already contained in `ep_next`, or is noise.

WHY MAE AND NOT ACCURACY. FPL points are a count with a long right tail (a hauling captain is
15+, a benched defender is 1). Rank correlation and top-10 overlap matter for the actual decision
— who to captain, who to transfer — so all four are reported. Bias is reported separately because
a projection can have good correlation and still be systematically high, which is exactly the
failure mode that makes a captaincy pick look safe when it isn't.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

LEDGER = config.OUTPUT_DIR / "fantasy_ledger.csv"
#: Blend sweep resolution. Coarse on purpose — a weight that only wins at 0.05 resolution is
#: fitted to this sample, not a finding.
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def _metrics(actual: np.ndarray, pred: np.ndarray, top_n: int = 10) -> dict:
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    rho = float(pd.Series(pred).corr(pd.Series(actual), method="spearman"))
    # top-N overlap: of the N players the projection ranks highest, how many were actually in the
    # real top N. This is the metric closest to the decision a manager makes.
    k = min(top_n, len(actual))
    top_pred = set(np.argsort(-pred)[:k])
    top_true = set(np.argsort(-actual)[:k])
    return {"mae": mae, "rmse": rmse, "rho": rho,
            "top_overlap": len(top_pred & top_true), "top_n": k,
            "bias": float(np.mean(pred - actual))}


def main() -> int:
    if not LEDGER.exists():
        print(f"no ledger at {LEDGER} — nothing to measure")
        return 0

    d = pd.read_csv(LEDGER)
    # Column names as the ledger actually writes them.
    WOWZA_COL, FPL_COL, GW_COL, START_COL = ("proj_fixture_adj", "fpl_ep_next", "gw", "p_start")
    need = {"actual_points", WOWZA_COL, FPL_COL}
    missing = need - set(d.columns)
    if missing:
        print(f"ledger missing columns: {sorted(missing)}")
        return 1

    # Settled only. An unplayed gameweek has no actual, and imputing one would be inventing a
    # label — invariant 9.
    d = d[d["actual_points"].notna() & d[WOWZA_COL].notna() & d[FPL_COL].notna()]
    if d.empty:
        print("no settled player-gameweeks yet")
        return 0

    actual = d["actual_points"].astype(float).to_numpy()
    # LIKE FOR LIKE. `proj_fixture_adj` is Wowza's projection CONDITIONAL on the player
    # starting. `ep_next` is UNCONDITIONAL — it already discounts for rotation risk. Scoring the
    # conditional number against actual points (which are ~0 for anyone benched) charges Wowza
    # for a question it was not asked, so the headline comparison multiplies by p_start. The
    # conditional number is still printed, because the gap between the two is itself the
    # diagnostic: if conditional does well and unconditional badly, the points model is fine and
    # the START model is what's broken.
    cond = d[WOWZA_COL].astype(float).to_numpy()
    pstart = (d[START_COL].astype(float).to_numpy() if START_COL in d.columns
              else np.ones(len(d)))
    wowza = cond * pstart
    fpl = d[FPL_COL].astype(float).to_numpy()

    gws = [int(g) for g in sorted(d[GW_COL].dropna().unique())] if GW_COL in d.columns else []
    print(f"settled player-gameweeks: {len(d)}" + (f" across GW {gws}" if gws else ""))

    mw, mf, mc = _metrics(actual, wowza), _metrics(actual, fpl), _metrics(actual, cond)
    for name, m in (("Wowza x p_start", mw), ("FPL ep_next", mf),
                    ("Wowza conditional (ref)", mc)):
        print(f"  {name:<28} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  rho {m['rho']:.3f}  "
              f"top{m['top_n']} {m['top_overlap']}/{m['top_n']}  bias {m['bias']:+.3f}")

    # blend sweep
    maes = [(w, float(np.mean(np.abs(actual - (w * wowza + (1 - w) * fpl))))) for w in WEIGHTS]
    best_w, best_mae = min(maes, key=lambda t: t[1])
    print(f"  best blend weight on Wowza: {best_w}  -> MAE {best_mae:.3f}")

    # A blend only counts as a win if it beats BOTH singles by a margin that is not rounding.
    # 0.01 FPL points is far below the granularity of a points projection.
    improves = (best_mae < min(mw["mae"], mf["mae"]) - 0.01) and 0.0 < best_w < 1.0
    if improves:
        print(f"  VERDICT: blend at w={best_w} beats both singles — candidate, needs a second "
              f"independent period before it replaces anything")
    elif best_w == 0.0:
        print("  VERDICT: FPL ep_next alone is best — do not replace it, and do not blend. "
              "Wowza's information is already in ep_next, or is noise.")
    else:
        print(f"  VERDICT: Wowza-weighted at {best_w} — report, do not deploy on one period")

    # The decision-relevant warning, separate from the headline.
    if mc["rho"] > mw["rho"] + 0.05:
        print(f"  NOTE: the CONDITIONAL projection ranks better (rho {mc['rho']:.3f}) than the "
              f"start-weighted one ({mw['rho']:.3f}) — the points model is not the weak part, "
              f"p_start is.")
    if mw["rho"] > mc["rho"] + 0.05:
        print(f"  NOTE: start-weighting lifts rank correlation {mc['rho']:.3f} -> "
              f"{mw['rho']:.3f}, so most of what Wowza knows here is WHO PLAYS, not how many "
              f"points he scores when he does. p_start is doing the work; the points head is "
              f"close to noise on this sample.")
    if abs(mw["bias"]) > 0.5:
        print(f"  NOTE: Wowza projects {mw['bias']:+.2f} points per player — a systematic bias "
              f"that makes every pick look better (or worse) than it is, independently of rank.")
    if len(gws) < 4:
        print(f"  NOTE: {len(gws)} gameweek(s) only. This is a reading, not a conclusion; the "
              f"same sweep on 10+ GWs is what would settle it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
