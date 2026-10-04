"""§1 replay — run every historical retrain decision through the strict gate.

    python scripts/replay_promotions.py

WHAT THIS CAN AND CANNOT ANSWER, STATED FIRST BECAUSE IT BOUNDS EVERY NUMBER BELOW.

`output/retrain_log.json` stores AGGREGATES — one log loss per side, plus the canary's summary
statistics. It does not store per-row probabilities. So of the strict gate's ten checks, exactly
three can be replayed:

    log_loss_not_worse     stored
    material_improvement   stored
    prediction_canary      stored (as summary stats, not raw probabilities)

and seven cannot:

    same_dataset, brier_not_worse, ci_excludes_zero, calibration_held,
    no_league_regression, second_period, min_sample(partial — `rows` is the TRAINING
    row count, not the holdout n the gate actually tests)

Every unreplayable check can only ever REJECT. So the strict count here is an UPPER BOUND: the
real gate would have promoted at most this many, probably fewer. A decision this replay marks
PASS might still have failed on a CI that spans zero — which, given typical deltas here, is
likely for most of them.

That asymmetry is the point. The replay cannot prove the strict gate is well-calibrated; it can
only show how many of the live gate's promotions fail on grounds that ARE recorded. Anything
stronger needs per-row probabilities, which is what the Stage A log-only wiring now stores going
forward (`v91_gate` in each retrain record).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.promotion_gate import MIN_EFFECT_LOG_LOSS  # noqa: E402

LOG = config.OUTPUT_DIR / "retrain_log.json"
OUT = config.OUTPUT_DIR / "promotion_replay.json"


def _canary_ok(c: dict) -> tuple[bool, str]:
    """The canary as the live pipeline already computes it. Flags are recorded, so reusable.

    A COLLAPSE is what this must catch: a model whose spread shrinks toward a constant looks
    fine on log loss and stops discriminating. Expansion is NOT penalised — an earlier version
    rejected a model for sharpening (sd 0.029 -> 0.146), which is an improvement, not a fault.
    """
    if not c.get("available"):
        return True, "canary unavailable — cannot reject on it"
    flags = c.get("flags") or []
    sr = c.get("spread_ratio")
    if isinstance(sr, (int, float)) and sr < 1 / 1.5:
        return False, f"probability spread COLLAPSED (ratio {sr:.3f})"
    return (not flags), (f"canary flags: {flags}" if flags else "canary clean")


def strict_replayable(old_ll: float, new_ll: float, canary: dict) -> tuple[bool, list[str]]:
    """The subset of the strict gate that the stored aggregates support."""
    reasons = []
    delta = new_ll - old_ll
    if not (delta < 0):
        reasons.append(f"log_loss_not_worse: delta {delta:+.5f} — candidate is not better")
    if not (delta <= -MIN_EFFECT_LOG_LOSS):
        reasons.append(f"material_improvement: delta {delta:+.5f} vs required "
                       f"<= -{MIN_EFFECT_LOG_LOSS:.5f}")
    ok, why = _canary_ok(canary)
    if not ok:
        reasons.append(f"prediction_canary: {why}")
    return (not reasons), reasons


def main() -> int:
    if not LOG.exists():
        print(f"no retrain log at {LOG}")
        return 0
    runs = json.loads(LOG.read_text(encoding="utf-8")).get("runs", {})

    rows = []
    for day in sorted(runs):
        for track, r in (runs[day] or {}).items():
            if not isinstance(r, dict):
                continue
            o, n = r.get("logloss_old"), r.get("logloss_new")
            if not (isinstance(o, (int, float)) and isinstance(n, (int, float))):
                continue
            passed, reasons = strict_replayable(o, n, r.get("canary") or {})
            rows.append({"day": day, "track": track, "old": o, "new": n,
                         "delta": round(n - o, 6), "live_promoted": bool(r.get("promoted")),
                         "strict_pass_replayable": passed, "strict_reasons": reasons,
                         "basis": r.get("comparison_basis")})

    n = len(rows)
    live_yes = [r for r in rows if r["live_promoted"]]
    worse = [r for r in rows if r["delta"] > 0]
    live_yes_worse = [r for r in live_yes if r["delta"] > 0]
    strict_yes = [r for r in rows if r["strict_pass_replayable"]]
    would_block = [r for r in live_yes if not r["strict_pass_replayable"]]
    would_allow_but_live_blocked = [r for r in rows
                                    if r["strict_pass_replayable"] and not r["live_promoted"]]

    print(f"decisions replayed                         {n}")
    print(f"  live gate promoted                       {len(live_yes)}")
    print(f"  candidate was WORSE than incumbent       {len(worse)}")
    print(f"    ...and promoted anyway                 {len(live_yes_worse)}")
    print(f"  strict gate PASSES replayable checks     {len(strict_yes)}   <- UPPER BOUND")
    print(f"  strict would block a live promotion      {len(would_block)}")
    print(f"  strict would allow a live block          {len(would_allow_but_live_blocked)}")

    print("\nper market:")
    tracks = sorted({r["track"] for r in rows})
    print(f"  {'track':<14}{'n':>4}{'live':>6}{'worse':>7}{'strict':>8}  worst delta")
    for t in tracks:
        g = [r for r in rows if r["track"] == t]
        wd = max((r["delta"] for r in g), default=0.0)
        print(f"  {t:<14}{len(g):>4}{sum(r['live_promoted'] for r in g):>6}"
              f"{sum(r['delta'] > 0 for r in g):>7}"
              f"{sum(r['strict_pass_replayable'] for r in g):>8}  {wd:+.5f}")

    if live_yes_worse:
        print("\npromoted while measurably worse (live gate), biggest regression first:")
        for r in sorted(live_yes_worse, key=lambda x: -x["delta"])[:10]:
            print(f"  {r['day']}  {r['track']:<12} {r['old']:.5f} -> {r['new']:.5f} "
                  f"{r['delta']:+.5f}")

    weak = [r for r in rows if r["basis"] and r["basis"] != "same_holdout"]
    if weak:
        print(f"\n{len(weak)} decision(s) rested on a WEAK comparison basis (not same_holdout) — "
              f"those cannot be replayed meaningfully at all:")
        for r in weak[:5]:
            print(f"  {r['day']}  {r['track']:<12} basis={r['basis']}")

    OUT.write_text(json.dumps({
        "generated_from": LOG.name,
        "caveat": ("UPPER BOUND. Only log_loss_not_worse, material_improvement and the canary "
                   "are replayable from stored aggregates; the other seven strict checks need "
                   "per-row probabilities and can only ever reject, so the true strict-promote "
                   "count is at most this one."),
        "n_decisions": n, "live_promoted": len(live_yes),
        "candidate_worse": len(worse), "promoted_while_worse": len(live_yes_worse),
        "strict_pass_replayable_upper_bound": len(strict_yes),
        "strict_would_block_live_promotion": len(would_block),
        "decisions": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
