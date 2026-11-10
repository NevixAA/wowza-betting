"""The bars study re-run on PRICED universes (own prices + the bought OddsAPI history where ours are missing),
tested on the owner's FRESH HOLDOUT - VERDICTS ONLY.

    python scripts/bars_opt_oddsapi.py prepare --store <odds-history checkout> --out <local dir>
    python scripts/bars_opt_oddsapi.py study --store <odds-history checkout> --priced <local dir>/priced \
        --out <local dir>/study

FLAGGED OFF. `study` refuses unless (a) config/bars_oddsapi_hist.json has run_enabled true and an owner_ok
note (the owner said "run"), and (b) the store's run ledger shows tier B1 finished (oddsapi_prices.
b1_complete). Data-truth rule: no study on unclean / partial data.

PROTOCOL: the owner's FRESH HOLDOUT (2026-11-10), pre-registered in config/bars_fresh_holdout_protocol.json
(must be committed and unchanged; its sha256 goes into the proposal). Per family: selection with the study's
own select_core on rows dated before the fresh holdout (src/bars_opt/fresh_holdout), choices frozen and hashed,
then each cell scored ONCE on its fresh holdout rows; Holm over the cells counted before the test. No monthly
simulation (it would read past the cut). Live serving is unchanged: this run writes verdicts only.

LICENCE. The priced universes and the study's per-bet outputs (final_bets.parquet, top3 csvs) carry prices:
both directories must be outside this public repo. The proposal (bars, verdicts, ROI / CI / CLV statistics,
no prices) goes to output/threshold_proposals/bars_oddsapi_hist/ - its own folder, so it never overwrites
TP-2026-11-09-abfb5dc4 or the main latest.json pointer; it gets a NEW proposal id (TP-FH-...).
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


def _select(args) -> dict:
    """Selection for one family on its pre-holdout rows only (picklable worker)."""
    from src.bars_opt import study as S
    sel, ho_ids, track, mside = args
    if set(sel["row_id"]) & ho_ids:
        raise AssertionError(f"{track}/{mside}: a holdout row reached selection")
    core = S.select_core(sel, seed=0)
    return {"track": track, "mside": mside, "select_rows": int(len(sel)),
            "select_leagues": sorted(sel["league"].unique()), "configs_searched": len(S.configs())} | core


def _cells_to_test(choice: dict, ho) -> list[str]:
    """Leagues of the family with frozen ON bars and >= 1 holdout row: counted BEFORE the test."""
    if not choice.get("chosen"):
        return []
    on = {lg for lg, b in choice["bars"].items() if "off" not in b}
    return sorted(on & set(ho["league"]))


def run(store_root: Path, priced_dir: Path, out_dir: Path, workers: int = 4) -> dict:
    import hashlib
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from src import threshold_proposals as TP
    from src.bars_opt import fresh_holdout as FH
    from src.bars_opt import study as S
    ok, why = gate(store_root)
    if not ok:
        raise SystemExit("bars study on the OddsAPI-priced universes is FLAGGED OFF:\n  " + "\n  ".join(why))
    okp, whyp = FH.protocol_committed()
    if not okp:
        raise SystemExit(f"fresh-holdout protocol not pre-registered: {whyp}")
    proto, proto_sha = FH.protocol()
    if proto["proven_rule"]["P_RESOLUTION_HITS"] != S.P_RESOLUTION_HITS:
        raise SystemExit("protocol P_RESOLUTION_HITS differs from src/bars_opt/study.py")
    priced_dir, out_dir = P.assert_outside_repo(priced_dir), P.assert_outside_repo(out_dir)
    if (out_dir / "fresh_result.json").exists() or (out_dir / "fresh_choices.json").exists():
        raise SystemExit(f"{out_dir}: a fresh-holdout test already ran here - each cell is tested ONCE")
    priced = {}
    for t in ("standard", "new_format"):
        f = priced_dir / f"universe_{t}.parquet"
        if not f.exists():
            raise SystemExit(f"{priced_dir}: universe_{t}.parquet missing (run `prepare` first)")
        priced[t] = pd.read_parquet(f)
    seen, provenance = FH.earlier_runs()
    fams = FH.families(priced, seen)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) SELECTION on pre-holdout rows only; choices frozen and hashed before any holdout row is scored
    parts = {(x["track"], x["mside"]): FH.split_family(x["frame"], x["window"]) for x in fams}
    todo = [(parts[(x["track"], x["mside"])][0], set(parts[(x["track"], x["mside"])][1]["row_id"]),
             x["track"], x["mside"]) for x in fams if x["window"]["status"] == "OK"]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            choices = list(ex.map(_select, todo))
    else:
        choices = [_select(a) for a in todo]
    tested = {(c["track"], c["mside"]): _cells_to_test(c, parts[(c["track"], c["mside"])][1]) for c in choices}
    m_pre = sum(len(v) for v in tested.values())
    frozen = json.dumps({"protocol_sha256": proto_sha,
                         "choices": [{k: v for k, v in c.items() if k != "inner_table"} for c in choices],
                         "cells_to_test": {f"{t}|{m}": v for (t, m), v in tested.items()}, "holm_family": m_pre},
                        indent=1, default=str, sort_keys=True)
    (out_dir / "fresh_choices.json").write_text(frozen, encoding="utf-8")
    choices_sha = hashlib.sha256(frozen.encode("utf-8")).hexdigest()
    # 2) the SINGLE test on the fresh holdout
    results, cells = [], []
    for c in choices:
        sel, ho = parts[(c["track"], c["mside"])]
        lgs = tested[(c["track"], c["mside"])]
        e = ho[ho["league"].isin(lgs)]
        fb, rebuilt = S.bet_with(sel, e, c) if lgs else (pd.DataFrame(), {})
        if lgs and S._norm({lg: rebuilt.get(lg) for lg in lgs}) != S._norm({lg: c["bars"][lg] for lg in lgs}):
            raise RuntimeError(f"{c['track']}/{c['mside']}: refit bars differ from the frozen choice")
        r = {"track": c["track"], "mside": c["mside"], "chosen": c["chosen"], "off": c["off"], "bars": c["bars"],
             "select_leagues": lgs, "final_leagues": lgs, "final_bets": fb}
        results.append(r)
        cells += S._cells(r)
    if len(cells) != m_pre:
        raise RuntimeError(f"{len(cells)} cells scored, {m_pre} counted before the test")
    redrawn = S.refine_floor_p(cells, results, m_pre) if m_pre else []
    finite = [c for c in cells if c.get("verdict") is None and c["final"]["bets"] > 0
              and c["final"]["p_one_sided"] is not None]
    S.verdicts(cells, prior_hypotheses=m_pre - len(finite))
    for c in cells:
        c["family_size"] = m_pre
    cov_f, cov_l = FH.coverage(fams)
    result = {"protocol": proto["protocol"], "protocol_sha256": proto_sha, "choices_sha256": choices_sha,
              "holm_family": m_pre, "p_redrawn": redrawn, "families": cov_f, "cells": cells,
              "provenance": provenance}
    (out_dir / "fresh_result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    # 3) a verdicts-only proposal (no served bar changes; live serving unchanged)
    created = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {"schema": "bars-fresh-holdout-verdicts/1", "created_at": created, "status": "AWAITING_OWNER",
            "serving_change": False, "protocol_file": "config/bars_fresh_holdout_protocol.json",
            "protocol_sha256": proto_sha, "choices_sha256": choices_sha, "holm_family": m_pre,
            "families": cov_f, "family_leagues": cov_l,
            "cells": [{k: c.get(k) for k in ("track", "mside", "league", "config", "mode", "valuable_th",
                                             "marksman_th", "sniper_th", "verdict", "reason", "p_holm",
                                             "family_size", "final")} for c in cells],
            "no_fresh_holdout": [{"track": f["track"], "mside": f["mside"], "reason": f["reason"],
                                  "note": "no historical PROVEN test; waits for the live Clean Record from 2026-11-21"}
                                 for f in cov_f if f["status"] != "OK"],
            "earlier_runs_proof": provenance}
    h = TP.sha(body)
    body["proposal_hash"], body["proposal_id"] = h, f"TP-FH-{created[:10]}-{h[:8]}"
    path = TP._safe_write(TP.PROPOSAL_DIR / NAMESPACE / f"fresh_{created[:10]}_{h[:8]}.json",
                          json.dumps(body, indent=1, default=str, ensure_ascii=False) + "\n")
    return {"proposal_id": body["proposal_id"], "proposal": str(path), "protocol_sha256": proto_sha,
            "holm_family": m_pre,
            "verdicts": pd.Series([c["verdict"] for c in cells]).value_counts().to_dict() if cells else {},
            "families_without_fresh_holdout": len(body["no_fresh_holdout"])}
