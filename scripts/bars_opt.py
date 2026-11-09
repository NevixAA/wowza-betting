"""Bars optimization study (reports/P4_BARS_OPTIMIZATION.md) - PROPOSALS ONLY.

    python scripts/bars_opt.py universe --track standard --predictions <run>/predictions.parquet \
        --fd-raw <Pro>/experiments/ou_info_inputs_2026_10/fd_raw --aliases <identity_aliases.json> --out <dir>
    python scripts/bars_opt.py universe --track new_format --predictions <run>/predictions.parquet \
        --aliases <identity_aliases.json> --out <dir>
    python scripts/bars_opt.py select --universe <dir> --out <dir>     # phase 1: choices.json (commit it)
    python scripts/bars_opt.py final --universe <dir> --out <dir>      # phase 2: FINAL once, verdicts, TOP-3
    python scripts/bars_opt.py monthly-sim --universe <dir> --out <dir>  # walk-forward monthly vs static + FINAL rolling
    python scripts/bars_opt.py monthly --universe <dir> --month YYYY-MM --out <dir>   # one monthly re-selection

Never writes models/ or config/threshold_approvals.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cmd_universe(a) -> int:
    from src.bars_opt import universe as U
    pred = pd.read_parquet(a.predictions)
    pred = pred[(pred["track"] == a.track) & (pred["candidate"] == a.candidate) & pred["deployable"]]
    aliases = json.loads(Path(a.aliases).read_text(encoding="utf-8")) if a.aliases else {}
    rows, stats = U.build(pred, a.track, aliases, a.fd_raw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / f"universe_{a.track}.parquet", index=False)
    (out / f"universe_{a.track}_match.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    pr = rows["yes"].notna()
    print(a.track, "rows", len(rows), "priced", int(pr.sum()),
          "closing pair", int((pr & rows["yes_c"].notna()).sum()))
    return 0


def cmd_select(a) -> int:
    from src.bars_opt import study as S
    print(json.dumps(S.run_select(Path(a.universe), Path(a.out), seed=0), indent=1, default=str))
    return 0


def cmd_final(a) -> int:
    from src.bars_opt import study as S
    print(json.dumps(S.run_final(Path(a.universe), Path(a.out))["summary"], indent=1, default=str))
    return 0


def cmd_monthly_sim(a) -> int:
    from src.bars_opt import study as S
    res = S.run_monthly(Path(a.universe), Path(a.out))
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=str)[:6000])
    return 0


def cmd_monthly(a) -> int:
    from src.bars_opt import study as S
    res = S.monthly_proposal(Path(a.universe), a.month, Path(a.out))
    print(json.dumps(res, indent=1, default=str)[:4000])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("universe")
    u.add_argument("--track", required=True, choices=["standard", "new_format"])
    u.add_argument("--predictions", required=True)
    u.add_argument("--candidate", default="champion_recipe")
    u.add_argument("--fd-raw")
    u.add_argument("--aliases")
    u.add_argument("--out", required=True)
    for name in ("select", "final"):
        s = sub.add_parser(name)
        s.add_argument("--universe", required=True)
        s.add_argument("--out", required=True)
    ms = sub.add_parser("monthly-sim")
    ms.add_argument("--universe", required=True)
    ms.add_argument("--out", required=True)
    m = sub.add_parser("monthly")
    m.add_argument("--universe", required=True)
    m.add_argument("--month", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"universe": cmd_universe, "select": cmd_select, "final": cmd_final, "monthly-sim": cmd_monthly_sim,
            "monthly": cmd_monthly}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
