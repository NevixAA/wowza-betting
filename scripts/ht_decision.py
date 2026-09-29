"""Evaluate the FROZEN half-time decision rule. Read-only.

    python scripts/ht_decision.py            # status + verdict when the trigger is reached
    python scripts/ht_decision.py --force    # run the test early, clearly marked PROVISIONAL

It reads `registry/ht_decision_rule.yaml` and applies it literally. The rule was written on
2026-09-29, before any qualifying data existed, because the half-time question has already
produced two results that looked convincing and were not:

  * an AUC of 0.663 that was the model learning which leagues had MISSING DATA, after 73.9% of
    its training labels turned out to be fabricated;
  * a "blend beats the market, +8.8% ROI over 226 bets" finding that failed split-half, its
    bootstrap CI and a shuffled placebo.

THIS SCRIPT DOES NOT DECIDE ANYTHING. It computes the numbers the rule names and prints the
verdict the rule already specified. If you find yourself wanting to adjust a threshold after
reading the output, that is the exact failure the freeze exists to prevent — say so out loud
and supersede the rule with a new versioned file instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

RULE = Path(__file__).resolve().parents[1] / "registry" / "ht_decision_rule.yaml"
RNG = np.random.default_rng(20260929)


def _load_rule() -> dict:
    try:
        import yaml
        return yaml.safe_load(RULE.read_text(encoding="utf-8"))
    except Exception as e:                                            # noqa: BLE001
        print(f"cannot read {RULE.name}: {e}")
        raise SystemExit(2)


def _devig(o_over: pd.Series, o_under: pd.Series) -> pd.Series:
    """Proportional de-vig, exactly as the rule specifies."""
    io, iu = 1.0 / o_over, 1.0 / o_under
    return io / (io + iu)


def _boot_ci(x: np.ndarray, n: int = 10000) -> tuple[float, float]:
    """Percentile bootstrap on the mean. Simple and honest about being simple."""
    if len(x) < 2:
        return (np.nan, np.nan)
    m = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def evaluate(line: str, d: pd.DataFrame, rule: dict, force: bool) -> dict:
    tgt, pcol = f"ht_over{line}", f"p_ht_over{line}"
    oc, uc = f"odds_ht_over{line}", f"odds_ht_under{line}"
    need = [c for c in (tgt, pcol, oc, uc) if c not in d.columns]
    if need:
        return {"line": line, "status": "NO DATA", "why": f"missing columns {need}"}

    q = d.dropna(subset=[tgt, pcol, oc, uc]).copy()
    q = q[(q[oc] > 1.0) & (q[uc] > 1.0)]
    trig = int(rule["sample"]["trigger_n"])
    print(f"\n{'=' * 78}\nHT OVER {line[0]}.{line[1]}\n{'=' * 78}")
    print(f"  priced AND settled fixtures: {len(q):,}   (rule triggers at {trig:,})")
    if q.empty:
        return {"line": line, "status": "NO DATA", "n": 0}
    if len(q) < trig and not force:
        wk = (trig - len(q)) / 83.0
        print(f"  NOT YET — {trig - len(q):,} more needed, roughly {wk:.0f} weeks "
              f"at the current rate.")
        print("  (run with --force to see a PROVISIONAL result; it does not count)")
        return {"line": line, "status": "PENDING", "n": int(len(q)),
                "needed": trig - int(len(q))}

    prov = len(q) < trig
    y = q[tgt].to_numpy(float)
    model = q[pcol].to_numpy(float)
    market = _devig(q[oc], q[uc]).to_numpy(float)

    # PRIMARY: paired Brier. Positive difference = model better.
    diff = (market - y) ** 2 - (model - y) ** 2
    mean = float(diff.mean())
    lo, hi = _boot_ci(diff)
    passed = bool(mean > 0 and lo > 0)

    print(f"\n  PRIMARY — paired Brier vs the de-vigged close  (n={len(q):,})")
    print(f"    market Brier          {((market - y) ** 2).mean():.5f}")
    print(f"    model  Brier          {((model - y) ** 2).mean():.5f}")
    print(f"    paired difference     {mean:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"    -> {'PASS' if passed else 'FAIL'}"
          + ("" if passed else "  (needs mean > 0 AND the CI to exclude zero)"))

    # SECONDARY — cannot rescue a failed primary, can only sink a passing one.
    #
    # AND THEY ARE MEANINGLESS WHEN THE PRIMARY FAILS, which is easy to misread. On the
    # 332-fixture dry run the placebo came back p=0.0175, "beats placebo" — while the paired
    # difference was NEGATIVE. All that says is that a shuffled model is even worse than a bad
    # one. Printed without the caveat, it hands a future reader an encouraging-looking number
    # attached to a failing result.
    print("\n  SECONDARY" + ("" if passed else
          "   [primary FAILED — diagnostic only, cannot change the verdict]"))
    mid = len(q) // 2
    h1, h2 = diff[:mid], diff[mid:]
    same_sign = bool(len(h1) and len(h2) and np.sign(h1.mean()) == np.sign(h2.mean()))
    print(f"    split-half            {h1.mean():+.5f} then {h2.mean():+.5f}"
          f"   -> {'same sign' if same_sign else 'SIGN FLIP'}")

    sims = np.array([((market - y) ** 2 - (RNG.permutation(model) - y) ** 2).mean()
                     for _ in range(400)])
    pval = float((sims >= mean).mean())
    print(f"    shuffled placebo      mean {sims.mean():+.5f}, p = {pval:.4f}"
          f"   -> {'beats placebo' if pval < 0.05 else 'INDISTINGUISHABLE'}")

    bands = ((f"p{line}>=0.75", model >= 0.75) if line == "05" else (f"p{line}>=0.60", model >= 0.60),
             (f"p{line}<=0.30", model <= 0.30) if line == "05" else (f"p{line}<=0.25", model <= 0.25))
    worst = 0.0
    for name, m in bands:
        if m.sum() < 20:
            print(f"    calibration {name:<12} n={int(m.sum())} — too few to judge")
            continue
        gap = abs(model[m].mean() - y[m].mean()) * 100
        worst = max(worst, gap)
        print(f"    calibration {name:<12} n={int(m.sum()):<5} |claimed-realised| {gap:.1f}pp"
              f"   -> {'ok' if gap <= 5 else 'OUT OF TOLERANCE'}")

    verdict = "PASS" if passed else "FAIL"
    print(f"\n  {'*** PROVISIONAL — below trigger_n, does NOT count ***' if prov else ''}")
    print(f"  VERDICT: {verdict}")
    out = rule["outcomes"][verdict]
    print(f"    {out['verdict']}")
    for a in out.get("actions", []):
        print(f"      - {a}")
    return {"line": line, "status": "PROVISIONAL" if prov else "DECIDED",
            "verdict": verdict, "n": int(len(q)), "paired_diff": round(mean, 6),
            "ci": [round(lo, 6), round(hi, 6)], "split_half_same_sign": same_sign,
            "placebo_p": round(pval, 4), "worst_band_gap_pp": round(worst, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="evaluate before the trigger; result is PROVISIONAL and does not count")
    a = ap.parse_args()
    rule = _load_rule()

    print(f"FROZEN RULE: {rule['decision_name']} v{rule['version']}")
    print(f"  frozen at {rule['frozen_at_utc']} — before any qualifying data existed")
    print(f"  trigger: {rule['sample']['trigger_n']:,} priced+settled fixtures per line")
    print(f"  primary: {rule['primary_test']['name']}")

    src = config.OUTPUT_DIR / Path(rule["sample"]["source"]).name
    if not src.exists():
        print(f"\n{src.name} does not exist yet — collection starts with the next predict run.")
        return 0
    d = pd.read_csv(src)
    print(f"\n{src.name}: {len(d):,} rows logged")

    res = [evaluate(line, d, rule, a.force) for line in ("05", "15")]
    import json
    p = config.OUTPUT_DIR / "ht_decision_status.json"
    p.write_text(json.dumps({"rule": rule["decision_name"], "results": res},
                            indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
