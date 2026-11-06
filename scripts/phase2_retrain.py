"""Phase 2 part 4 - clean retrain harness CLI (src/retrain_harness, config/phase2_retrain.json).

    python scripts/phase2_retrain.py --dry-run [--json-out FILE]
        Loads the COMMITTED data offline (no download, network blocked), builds the point-in-time
        features, and prints the plan: holdout, folds, row counts per split and per league, price
        coverage and the feature list per track. Fits nothing; writes nothing unless --json-out.

    python scripts/phase2_retrain.py --out DIR [--stage walk_forward|holdout|all] [--only a,b]
                                     [--tracks standard|new_format] [--fd-raw DIR]
        Fits and scores into DIR/<run_id>/. Refused unless the config's holdout is approved by the
        owner AND PHASE2_RETRAIN_ENABLED=true. Meant to run only in .github/workflows/phase2_retrain.yml.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrain_harness import protocol as P  # noqa: E402

log = logging.getLogger("phase2_retrain")

#: --fd-raw columns (src.standard_repair.attach_entry_prices) -> the CLV columns of market over25
FD_RAW_CLV = {"odds_over25_entry": "odds_over25_over_entry", "odds_under25_entry": "odds_over25_under_entry",
              "odds_over25_close": "odds_over25_over_close", "odds_under25_close": "odds_over25_under_close"}


def print_plan(plan: dict, info: dict) -> None:
    h = plan["holdout"]
    print(f"Phase 2 clean retrain - DRY RUN (no model fitted)   harness {plan['harness_version']}")
    print(f"config sha256 {plan['config_sha256'][:16]}   holdout {h['start']}..{h['end']}   "
          f"approved: {'YES' if h['approved'] else 'NO (fitting refused)'}")
    print(f"frame: {plan['frame']['rows']:,} feature rows, {plan['frame']['date_min']}..{plan['frame']['date_max']}")
    for k, v in info.items():
        print(f"  data.{k}: {v}")
    for track, t in plan["tracks"].items():
        print(f"\n== track {track}: {t['n_features']} features")
        print("   " + ", ".join(t["features"]))
        if t["excluded_features"]:
            print("   excluded: " + "; ".join(f"{c} ({r})" for c, r in t["excluded_features"].items()))
        print(f"   {'market':10s} {'eligible':>9s} {'wf region':>9s} {'holdout':>8s} {'unused':>7s} "
              f"{'wf days':>7s} {'ho days':>7s} {'folds':>5s} {'wf scored':>9s} {'2-sided px wf/ho':>17s}")
        for m, s in t["markets"].items():
            pw, ph = (f"{s['price'][k]['two_sided']:,}" if s["price"][k] else "-"
                      for k in ("walk_forward_region", "holdout"))
            print(f"   {m:10s} {s['eligible_rows']:>9,} {s['walk_forward_region']:>9,} {s['holdout']:>8,} "
                  f"{s['after_holdout_unused']:>7,} {s['walk_forward_matchdays']:>7,} {s['holdout_matchdays']:>7,} "
                  f"{s['n_folds']:>5} {s['rows_scored_by_walk_forward']:>9,} {pw:>8}/{ph:<8}")
        for m in ("over25", "ht_over05"):
            s = t["markets"].get(m)
            if not s or not s["per_league"]:
                continue
            print(f"   per league, {m}: walk-forward region / holdout (holdout >= min_league_n?)")
            for lg, c in s["per_league"].items():
                ok = c["holdout_meets_min_league_n"]
                print(f"     {lg:32s} {c['walk_forward_region']:>7,} / {c['holdout']:>5,}  "
                      f"{'yes' if ok else 'no'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(P.CONFIG_PATH))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out", default=None, help="dry run: also write the plan JSON here")
    ap.add_argument("--out", default=None, help="run: parent folder of the run folder")
    ap.add_argument("--stage", default="all", choices=["walk_forward", "holdout", "all"])
    ap.add_argument("--only", default=None, help="comma-separated candidate names")
    ap.add_argument("--tracks", default=None, help="comma-separated subset of the configured tracks")
    ap.add_argument("--fd-raw", default=None, help="folder of raw football-data CSVs (entry + closing O/U 2.5)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = P.load_config(a.config)

    if not a.dry_run:
        P.assert_fit_allowed(cfg)                       # before any data is even loaded
        if not a.out:
            ap.error("--out is required for a run")
        P.assert_run_folder(a.out)

    feat, info = P.load_committed_frame(cfg)
    if a.fd_raw:
        from src.standard_repair import attach_entry_prices, load_fd_entry_prices
        feat = attach_entry_prices(feat, load_fd_entry_prices(a.fd_raw)).rename(columns=FD_RAW_CLV)
        info["fd_raw"] = str(a.fd_raw)

    if a.dry_run:
        plan = P.build_plan(feat, cfg)
        plan["data"] = info
        print_plan(plan, info)
        if a.json_out:
            out = P.assert_run_folder(Path(a.json_out).parent) / Path(a.json_out).name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
            print(f"\nplan written to {out}")
        return 0

    from src.retrain_harness import runner
    stages = runner.STAGES if a.stage == "all" else (a.stage,)
    folder = runner.run(feat, cfg, a.out, stages=stages,
                        only=a.only.split(",") if a.only else None,
                        tracks=a.tracks.split(",") if a.tracks else None, data_info=info)
    print(f"run folder: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
