"""The bars study re-run on PRICED universes (own prices + the bought OddsAPI history where ours are missing),
tested on the owner's FRESH HOLDOUT - VERDICTS ONLY.

    python scripts/bars_opt_oddsapi.py prepare --store <odds-history checkout> --out <local dir>
    python scripts/bars_opt_oddsapi.py study --store <odds-history checkout> --priced <local dir>/priced \
        --out <local dir>/study

FLAGGED OFF. `study` refuses unless (a) config/bars_oddsapi_hist.json has run_enabled true and an owner_ok
note (the owner said "run"), and (b) the store's run ledger shows every price tier the protocol pins
finished (oddsapi_prices.tiers_complete). Data-truth rule: no study on unclean / partial data.

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
    from src.bars_opt import fresh_holdout as FH
    tiers = FH.price_tiers()
    ok, reason = P.tiers_complete(store_root, tiers)
    if not ok:
        why.append(f"price tiers {tiers} not complete: {reason}")
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
    stamp = priced_dir / "price_tiers.json"
    got = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else None
    want = FH.price_stamp()
    if got != want:
        raise SystemExit(f"{priced_dir}: priced with {got}, the protocol pins {want} - re-run prepare")
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
        if FH.evidence_label():
            c["evidence_label"] = FH.evidence_label()       # v4: re-test after a price-data fix
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
            "evidence_label": FH.evidence_label() or None,
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
            "choices_sha256": choices_sha, "holm_family": m_pre,
            "verdicts": pd.Series([c["verdict"] for c in cells]).value_counts().to_dict() if cells else {},
            "families_without_fresh_holdout": len(body["no_fresh_holdout"]),
            "_fresh": {"choices": choices, "selection": {k: v[0] for k, v in parts.items()}, "cells": cells}}


FRESH_BASIS = "FRESH_HOLDOUT_SELECTION"


def fresh_serving_cell(choice: dict, sel, league: str, tested: dict, month: str) -> tuple[dict | None, str]:
    """The serving cell of a cell tested on the fresh holdout: the FROZEN selection bars that were tested, the
    fitted parameters of their probability source on the same selection rows (study.source_params - one
    implementation for study and serving), and the fresh verdict. (cell, "") or (None, reason)."""
    from src.bars_opt import cells as C
    from src.bars_opt import serve as BS
    from src.bars_opt import study as S
    b = choice["bars"][league]
    chosen = S.parse_cfg(choice["chosen"])
    cfg = chosen if b["mode"] == S.MODE_MODEL else S.lw_of(chosen)
    params = S.source_params(cfg[0], sel) if cfg[0] != S.LEAGUE_WIDE_SOURCE else {}
    if cfg[0] == "SHRINK":
        if league not in params.get("weights", {}):
            return None, "no SHRINK weight for the league"
        params = {"weights": {league: params["weights"][league]}}
    prod_market, side = C.PROD_OF[choice["mside"]]
    cell = {"mside": choice["mside"], "track": choice["track"], "league": league, "month": month,
            "config": S.cfg_name(cfg), "config_basis": FRESH_BASIS, "source": cfg[0], "measure": cfg[1],
            "odds_max": cfg[2], "odds_min": cfg[3], "mode": b["mode"], "valuable_th": b["valuable_th"],
            "marksman_th": b["marksman_th"], "sniper_th": b["sniper_th"], "bar_basis": b.get("bar_basis"),
            "shrink_weight_at_valuable": b.get("shrink_weight_at_valuable"), "wf_bets": b.get("wf_bets"),
            "wf_roi": b.get("wf_roi"), "wf_roi_shrunk": b.get("wf_roi_shrunk"), "tier2": b.get("tier2"),
            "tier2_bets": b.get("tier2_bets"), "tier2_roi": b.get("tier2_roi"), "source_params": params,
            "verdict": tested["verdict"], "study_final": tested.get("final"), "study_verdict": tested["verdict"],
            "study_reason": tested.get("reason"), "fresh_holdout_p_holm": tested.get("p_holm"),
            "fresh_holdout_family_size": tested.get("family_size"), "prod_market": prod_market, "side": side}
    why = BS.validate_cell(cell)
    return (None, why) if why else (cell, "")


def status_table(families: list[dict], study_cells: list[dict]) -> list[dict]:
    """EVERY track x market-side x league with its own bar and status (never a global value): the serving
    cells (PROVEN / PROMISING / UNPROVEN, own or learned-blend bars), every OFF league with its reason, and the
    study-only families (no serving market, e.g. HT) from the selection study's cells."""
    rows, served = [], set()
    for fam in families:
        served.add((fam["track"], fam["mside"]))
        for c in fam["cells"]:
            rows.append({"track": c["track"], "mside": c["mside"], "league": c["league"], "status": c["verdict"],
                         "bars": [c["valuable_th"], c["marksman_th"], c["sniper_th"]], "config": c["config"],
                         "basis": c.get("config_basis"), "bar_basis": c.get("bar_basis"), "mode": c["mode"]})
        for o in fam["off"]:
            rows.append({"track": fam["track"], "mside": fam["mside"], "league": o["league"], "status": "OFF",
                         "bars": None, "reason": o["reason"], "detail": o.get("detail")})
    for c in study_cells:
        if (c["track"], c["mside"]) in served:
            continue
        has = c.get("valuable_th") is not None and c.get("verdict") != "OFF"
        rows.append({"track": c["track"], "mside": c["mside"], "league": c["league"],
                     "status": c["verdict"] if has else "OFF", "study_only": True,
                     "bars": [c.get("valuable_th"), c.get("marksman_th"), c.get("sniper_th")] if has else None,
                     "reason": c.get("reason"), "config": c.get("config")})
    return sorted(rows, key=lambda r: (r["track"], r["mside"], r["league"]))


def combine(fresh: dict, families: list[dict], study_cells: list[dict], month: str, meta: dict) -> dict:
    """The owner proposal: the month's serving families (selection-based bars, verdicts as in
    TP-2026-11-09-abfb5dc4) with every cell tested on the fresh holdout replaced by its fresh cell (frozen
    bars + fresh verdict). Candidates -> src/threshold_proposals (id + hash); nothing is applied."""
    from src import threshold_proposals as TP
    from src.bars_opt import cells as C
    ch = {(c["track"], c["mside"]): c for c in fresh["choices"]}
    fams = [dict(f, cells=list(f["cells"]), off=list(f["off"])) for f in families]
    by = {(f["track"], f["mside"]): f for f in fams}
    replaced, refused = [], []
    for t in fresh["cells"]:
        if t.get("verdict") in (None, "OFF"):
            continue
        key = (t["track"], t["mside"])
        if key not in by or t["mside"] not in C.PROD_OF:
            continue
        cell, why = fresh_serving_cell(ch[key], fresh["selection"][key], t["league"], t, month)
        if cell is None:
            refused.append({"track": key[0], "mside": key[1], "league": t["league"], "reason": why})
            continue
        f = by[key]
        f["cells"] = [c for c in f["cells"] if c["league"] != t["league"]] + [cell]
        f["off"] = [o for o in f["off"] if o["league"] != t["league"]]
        replaced.append({"track": key[0], "mside": key[1], "league": t["league"], "verdict": cell["verdict"]})
    cand_dir = TP.CANDIDATE_DIR / NAMESPACE / month
    for m, data in C.production_candidates(fams).items():
        TP.write_candidate("standard" if m == "ou25" else m, data, cand_dir)
    table = status_table(fams, study_cells)
    from collections import Counter
    counts = dict(Counter(r["status"] for r in table))
    prop_dir = TP.PROPOSAL_DIR / NAMESPACE
    TP._safe_write(prop_dir / f"evidence_{month}.json", json.dumps(
        {"month": month, "meta": meta, "fresh_cells_replaced": replaced, "fresh_cells_refused": refused,
         "status_counts": counts, "league_market_status": table, "families": fams}, indent=1, default=str))
    rel = cand_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    p = TP.build_proposal(TP.load_candidates(cand_dir), month=month, resolve=True,
                          source={"workflow": "bars fresh holdout + selection-based serving cells (src/bars_opt/hist_study)",
                                  "candidates_dir": rel,
                                  "evidence": f"output/threshold_proposals/{NAMESPACE}/evidence_{month}.json"} | meta)
    path = TP.write_proposal(p, prop_dir)
    return {"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"],
            "proposal": Path(path).resolve().relative_to(ROOT.resolve()).as_posix(),
            "entries": len(p["entries"]), "status": p["status"], "status_counts": counts,
            "fresh_cells_replaced": replaced, "fresh_cells_refused": refused}


def status_diff(approved: list[dict], new: list[dict]) -> dict:
    """R7 (go-live runbook): every track x market-side x league of the APPROVED status table against a rebuilt
    one. A status change (PROVEN / PROMISING / UNPROVEN / OFF), a cell that appears or disappears, goes back to
    the owner; a bar or configuration change under the same status is listed for information."""
    key = (lambda r: (r["track"], r["mside"], r["league"]))
    a, b = {key(r): r for r in approved}, {key(r): r for r in new}
    status = [{"cell": "|".join(k), "approved": a[k]["status"] if k in a else None,
               "rebuilt": b[k]["status"] if k in b else None}
              for k in sorted(set(a) | set(b)) if (a.get(k) or {}).get("status") != (b.get(k) or {}).get("status")]
    bars = [{"cell": "|".join(k), "approved": [a[k].get("bars"), a[k].get("config")],
             "rebuilt": [b[k].get("bars"), b[k].get("config")]}
            for k in sorted(set(a) & set(b)) if a[k]["status"] == b[k]["status"]
            and (a[k].get("bars"), a[k].get("config")) != (b[k].get("bars"), b[k].get("config"))]
    return {"cells_compared": len(set(a) | set(b)), "status_changes": status, "bar_changes_same_status": bars,
            "back_to_owner": bool(status)}


def recheck(approved_evidence: dict, families: list[dict], study_cells: list[dict]) -> dict:
    """Rebuild the status table from new serving families + new selection-study cells, with the APPROVED fresh
    cells carried unchanged (the fresh holdout is spent: those cells are never re-tested), and diff it."""
    fresh = {(c["track"], c["mside"], c["league"]): c
             for f in approved_evidence["families"] for c in f["cells"] if c.get("config_basis") == FRESH_BASIS}
    fams = []
    for f in families:
        mine = {k: c for k, c in fresh.items() if k[:2] == (f["track"], f["mside"])}
        lgs = {k[2] for k in mine}
        fams.append(dict(f, cells=[c for c in f["cells"] if c["league"] not in lgs] + list(mine.values()),
                         off=[o for o in f["off"] if o["league"] not in lgs]))
    return status_diff(approved_evidence["league_market_status"], status_table(fams, study_cells))
