"""Bars study on the bought OddsAPI history - prices stay LOCAL (the provider forbids redistribution and this
repo is PUBLIC). Prints counts and aggregate statistics only.

  # 1) priced universes + coverage + own-vs-history agreement + join integrity (safe to run any time)
  python scripts/bars_opt_oddsapi.py prepare --store <odds-history checkout> --out <dir outside this repo>

  # 2) the study re-run -> a NEW proposal id (FLAGGED OFF: config/bars_oddsapi_hist.json + B1 complete)
  python scripts/bars_opt_oddsapi.py study --store <odds-history checkout> --priced <out>/priced \
      --out <out>/study --month YYYY-MM

`prepare` writes <out>/priced/universe_{standard,new_format}.parquet (the study's universe schema +
price_source / close_source + oh_* provenance), <out>/coverage.json / coverage_families.json /
agreement.json / prepare_stats.json and <out>/coverage.md. Nothing is written inside this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.bars_opt import oddsapi_prices as P  # noqa: E402

UNIVERSE_DIR = ROOT / "output" / "bars_opt"
TRACKS = ("standard", "new_format")


def _table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def cmd_prepare(a) -> int:
    out = P.assert_outside_repo(a.out)
    (out / "priced").mkdir(parents=True, exist_ok=True)
    from src.bars_opt import fresh_holdout as FH
    tiers = FH.price_tiers()
    (out / "priced" / "price_tiers.json").write_text(json.dumps(tiers), encoding="utf-8")
    cov, fams, agree, stats = [], [], [], {"price_tiers": tiers}
    for t in TRACKS:
        u = pd.read_parquet(Path(a.universe) / f"universe_{t}.parquet")
        r = P.price_track(u, Path(a.store), tiers)
        r["priced"].to_parquet(out / "priced" / f"universe_{t}.parquet", index=False)
        c, f = P.coverage(r["priced"])
        cov += c
        fams += f
        agree += P.agreement(r["overlap"])
        stats[t] = r["stats"] | {"price_source": r["priced"]["price_source"].value_counts().to_dict(),
                                 "close_source": r["priced"]["close_source"].value_counts().to_dict()}
    ok, why = P.tiers_complete(Path(a.store), tiers)
    stats["tiers_complete"] = {"tiers": tiers, "complete": ok, "reason": why}
    for name, obj in (("coverage.json", cov), ("coverage_families.json", fams), ("agreement.json", agree),
                      ("prepare_stats.json", stats)):
        (out / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    changed = [r for r in cov if r["priced_after"] != r["priced_before"]]
    md = ["# Priced universes: coverage (counts only)", "",
          f"Price tiers {tiers} complete: {ok} ({why})", "",
          "## League x market rows that gain history prices", "",
          _table(changed, ["track", "market", "league", "rows", "priced_before", "priced_after", "after_own",
                           "after_oddsapi_hist", "select_before", "select_after", "select_months_before",
                           "select_months_after", "final_before", "final_after"]), "",
          "## Families: study windows before / after", "",
          _table(fams, list(fams[0]) if fams else []), "",
          "## Own vs history on rows priced by both", "",
          _table(agree, list(agree[0]) if agree else []), ""]
    (out / "coverage.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    for t in TRACKS:
        s = stats[t]
        print(f"[{t}] fixtures matched {s['fixtures_matched']:,}; fixture-markets with a history pair "
              f"{s['fixture_markets_with_history_pair']:,}; rows by source {s['price_source']}; "
              f"integrity {s['integrity']['rows_checked']:,} rows checked, failures {sum(s['integrity']['failures'].values())}; "
              f"pair drops {s['pairs']}")
    return 0


def cmd_study(a) -> int:
    from src.bars_opt import hist_study as H
    res = H.run(Path(a.store), Path(a.priced), Path(a.out), workers=a.workers)
    fresh = res.pop("_fresh")                     # selection frames carry prices: never printed
    print(json.dumps(res, indent=1, default=str))
    if a.month:
        from src.bars_opt import cells as C
        study_cells = json.loads(Path(a.study_cells).read_text(encoding="utf-8"))
        fams = C.build(Path(a.serving_universe), a.month, workers=a.workers, verdict_path=Path(a.study_cells))
        meta = {"fresh_verdicts_proposal": res["proposal_id"], "protocol_sha256": res["protocol_sha256"],
                "choices_sha256": res["choices_sha256"], "fresh_holm_family": res["holm_family"],
                "serving_universe": str(a.serving_universe), "study_cells": str(a.study_cells)}
        out = H.combine(fresh, fams, study_cells, a.month, meta)
        print(json.dumps(out, indent=1, default=str))
    return 0


def cmd_holdout(a) -> int:
    """Fresh-holdout coverage per family / league (counts and dates only), from the priced universes."""
    from src.bars_opt import fresh_holdout as FH
    out = P.assert_outside_repo(a.out)
    priced = {t: pd.read_parquet(Path(a.priced) / f"universe_{t}.parquet") for t in TRACKS}
    seen, runs = FH.earlier_runs()
    fam, lg = FH.coverage(FH.families(priced, seen))
    (out / "fresh_holdout_coverage.json").write_text(json.dumps({"families": fam, "leagues": lg, "earlier_runs": runs},
                                                                indent=1, default=str), encoding="utf-8")
    md = ["# Fresh holdout coverage (counts and dates only)", "",
          f"Earlier runs proven from git: {len(runs)} entries; seen priced keys {len(seen):,}", "",
          _table(fam, ["track", "mside", "status", "holdout_from", "holdout_to", "holdout_rows", "selection_rows",
                       "holdout_leagues", "cells_ge_150", "fresh_rows", "eligible_rows", "reason"]), "",
          "## Leagues in the holdouts", "",
          _table(lg, ["track", "mside", "league", "holdout_rows", "selection_rows", "ge_150"]), ""]
    (out / "fresh_holdout_coverage.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


def cmd_recheck(a) -> int:
    """Go-live R7: the approved status table against one rebuilt on new universes / a new selection study.
    Exit 1 when any cell's status would change (it goes back to the owner). Writes nothing."""
    from src.bars_opt import cells as C
    from src.bars_opt import hist_study as H
    approved = json.loads(Path(a.approved_evidence).read_text(encoding="utf-8"))
    fams = C.build(Path(a.universe), a.month, workers=a.workers, verdict_path=Path(a.study_cells))
    res = H.recheck(approved, fams, json.loads(Path(a.study_cells).read_text(encoding="utf-8")))
    print(json.dumps(res, indent=1, default=str))
    return 1 if res["back_to_owner"] else 0


def cmd_gate(a) -> int:
    from src.bars_opt import hist_study as H
    ok, why = H.gate(Path(a.store))
    print("OPEN" if ok else "CLOSED:\n  " + "\n  ".join(why))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--store", required=True, help="checkout of wowza-exec branch odds-history (read-only)")
    p.add_argument("--universe", default=str(UNIVERSE_DIR), help="dir with the study's universe_*.parquet")
    p.add_argument("--out", required=True, type=Path, help="LOCAL dir outside this repo")
    s = sub.add_parser("study")
    s.add_argument("--store", required=True)
    s.add_argument("--priced", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--month", help="also write the owner proposal: serving cells of this month + fresh cells")
    s.add_argument("--serving-universe", default=str(UNIVERSE_DIR), help="own-priced universes (serving cells)")
    s.add_argument("--study-cells", default=str(UNIVERSE_DIR / "study" / "cells.json"),
                   help="selection study FINAL cells (verdicts of the cells not tested on the fresh holdout)")
    h = sub.add_parser("holdout")
    h.add_argument("--priced", required=True)
    h.add_argument("--out", required=True, type=Path)
    rc = sub.add_parser("recheck")
    rc.add_argument("--approved-evidence", required=True, help="the approved proposal's evidence_<month>.json")
    rc.add_argument("--universe", required=True, help="dir with the rebuilt universe_*.parquet (own prices)")
    rc.add_argument("--study-cells", required=True, help="the selection study re-run on those universes")
    rc.add_argument("--month", required=True)
    rc.add_argument("--workers", type=int, default=4)
    g = sub.add_parser("gate")
    g.add_argument("--store", required=True)
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "study": cmd_study, "holdout": cmd_holdout, "recheck": cmd_recheck,
            "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
