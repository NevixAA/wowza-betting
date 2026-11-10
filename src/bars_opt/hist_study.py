"""The bars study re-run on PRICED universes (own prices + the bought OddsAPI history where ours are missing),
under the same pre-registered protocol (reports/P4_BARS_OPTIMIZATION.md) - PROPOSALS ONLY.

    python scripts/bars_opt_oddsapi.py prepare --store <odds-history checkout> --out <local dir>
    python scripts/bars_opt_oddsapi.py study --store <odds-history checkout> --priced <local dir>/priced \
        --out <local dir>/study --month YYYY-MM

FLAGGED OFF. `study` refuses unless (a) config/bars_oddsapi_hist.json has run_enabled true and an owner_ok
note (the owner said "run"), and (b) the store's run ledger shows tier B1 finished (oddsapi_prices.
b1_complete). Data-truth rule: no study on unclean / partial data.

PROTOCOL (unchanged): study.run_select (SELECT < 2026-09-01, 54 configurations per family, inner blocks,
White reality check), study.run_monthly, study.run_final (FINAL 2026-09-01..2026-11-08 scored once with the
frozen choices), then the month's serving cells (src/bars_opt/cells.py) with the FINAL verdicts of THIS run.
The ONLY change: the Holm family of the FINAL verdicts counts every configuration searched and every cell
tested in the earlier runs (src/bars_opt/search_ledger, from git history) and in this one.

LICENCE. The priced universes and the study's per-bet outputs (final_bets.parquet, top3 csvs) carry prices:
both directories must be outside this public repo. The proposal (bars, verdicts, ROI / CI / CLV statistics,
no prices) goes to output/threshold_proposals/bars_oddsapi_hist/ - its own folder, so it never overwrites
TP-2026-11-09-abfb5dc4 or the main latest.json pointer; it gets a NEW proposal id.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.bars_opt import oddsapi_prices as P

ROOT = Path(__file__).resolve().parents[2]
FLAG_FILE = ROOT / "config" / "bars_oddsapi_hist.json"
NAMESPACE = "bars_oddsapi_hist"


def gate(store_root: Path, flag_file: Path = FLAG_FILE) -> tuple[bool, list[str]]:
    why = []
    flag = json.loads(Path(flag_file).read_text(encoding="utf-8")) if Path(flag_file).exists() else {}
    if flag.get("run_enabled") is not True:
        why.append(f"{Path(flag_file).name}: run_enabled is not true (the owner has not said 'run')")
    if not str(flag.get("owner_ok") or "").strip():
        why.append(f"{Path(flag_file).name}: owner_ok is empty (quote the owner's 'run' and its date)")
    ok, reason = P.b1_complete(store_root)
    if not ok:
        why.append(f"B1 not complete: {reason}")
    return not why, why


def this_run_configs(out_dir: Path, served_families: int) -> dict:
    """Configurations THIS run searched, by the search ledger's own formulas."""
    from src.bars_opt import search_ledger as L
    from src.bars_opt import study as S
    per = len(S.configs())
    ch = json.loads((out_dir / "choices.json").read_text(encoding="utf-8"))
    mo = json.loads((out_dir / "monthly.json").read_text(encoding="utf-8"))["sim_select"]["families"]
    months = sum(len(f.get("months") or []) for f in mo)
    return {"select": int(sum(int(x["configs_searched"]) for x in ch)),
            "monthly": int((months + L.ROLLING_MONTHS * len(mo)) * per),
            "serving": int(served_families * per)}


def run(store_root: Path, priced_dir: Path, out_dir: Path, month: str, workers: int = 4) -> dict:
    from src import threshold_proposals as TP
    from src.bars_opt import cells as C
    from src.bars_opt import search_ledger as L
    from src.bars_opt import study as S
    ok, why = gate(store_root)
    if not ok:
        raise SystemExit("bars study on the OddsAPI-priced universes is FLAGGED OFF:\n  " + "\n  ".join(why))
    priced_dir, out_dir = P.assert_outside_repo(priced_dir), P.assert_outside_repo(out_dir)
    for t in ("standard", "new_format"):
        if not (priced_dir / f"universe_{t}.parquet").exists():
            raise SystemExit(f"{priced_dir}: universe_{t}.parquet missing (run `prepare` first)")
    ledger = L.build()
    out_dir.mkdir(parents=True, exist_ok=True)
    S.run_select(priced_dir, out_dir, seed=0, workers=workers)
    S.run_monthly(priced_dir, out_dir, workers=workers)
    served = [x for x in S.load_families(priced_dir) if x[2] in C.PROD_OF]
    now = this_run_configs(out_dir, len(served))
    prior = ledger["hypotheses"] + sum(now.values())
    fin = S.run_final(priced_dir, out_dir, workers=workers, prior_hypotheses=prior)
    fams = C.build(priced_dir, month, workers=workers, verdict_path=out_dir / "cells.json")
    cand_dir = TP.CANDIDATE_DIR / NAMESPACE / month
    prop_dir = TP.PROPOSAL_DIR / NAMESPACE
    prop_dir.mkdir(parents=True, exist_ok=True)
    for m, data in C.production_candidates(fams).items():
        TP.write_candidate("standard" if m == "ou25" else m, data, cand_dir)
    search = {"prior_ledger": ledger, "this_run": now, "holm_family_extra": prior,
              "final_tested": sum(1 for c in fin["cells"] if c.get("p_holm") is not None)}
    TP._safe_write(prop_dir / f"evidence_{month}.json",
                   json.dumps({"month": month, "search": search, "families": fams}, indent=1, default=str))
    p = TP.build_proposal(TP.load_candidates(cand_dir), month=month, resolve=True,
                          source={"workflow": "bars_opt on OddsAPI-priced universes (src/bars_opt/hist_study)",
                                  "candidates_dir": str(cand_dir.relative_to(ROOT)).replace("\\", "/"),
                                  "evidence": f"output/threshold_proposals/{NAMESPACE}/evidence_{month}.json",
                                  "supersedes_for_review": "TP-2026-11-09-abfb5dc4",
                                  "holm_family": prior + search["final_tested"]})
    path = TP.write_proposal(p, prop_dir)
    return {"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"],
            "proposal": str(path.relative_to(ROOT)), "entries": len(p["entries"]), "status": p["status"],
            "verdicts": fin["summary"]["verdict_counts"], "search": {k: v for k, v in search.items()
                                                                    if k != "prior_ledger"},
            "prior_hypotheses_ledger": ledger["hypotheses"], "p_redrawn": fin["summary"]["p_redrawn"]}
