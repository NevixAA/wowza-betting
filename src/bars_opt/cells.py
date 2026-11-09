"""The monthly SERVING CELLS of the optimized bars (owner 2026-11-09: "Learned blend, own number"; "Yes, build
the adapter") and their proposal.

    python scripts/bars_opt.py serving-cells --month 2026-11 [--universe output/bars_opt] [--workers 4]

FOR EACH family (track x market-side): the pre-registered choice (study.select_core) on every priced row
before the month, with allow_blend: a league whose own data is thin keeps its LEARNED-BLEND bars (its own ROI
curve pulled toward the all-league curve by a weight learned per bar, reported as shrink_weight) instead of
OFF. The cell carries the FITTED parameters of its probability source (study.source_params), so serving
(src/bars_opt/serve.py) rebuilds exactly the study's function.

A family the protocol cannot choose a configuration for (fewer than study.MIN_INNER_BETS inner bets in every
configuration) uses the configuration production already serves - RAW|EDGE, no odds range (p_model - de-vigged
fair, the edge src/betting and pipeline._generate_side_bets compute) - with the same per-league bar fit,
labelled config_basis NO_SELECTION_EVIDENCE so the owner sees that no configuration search backs it.

STAYS OFF (no cell, listed with the reason):
  * OFF_NO_OWN_ESTIMATE  a LEARNED_BLEND cell with fewer than OWN_VARIANCE_MIN own bets at its VALUABLE bar: the
                    league has no own estimate there (src.league_bars.choose_bars gives it weight 0), so its bars
                    would be the pooled curve served as a stand-in - never allowed (owner: no global vars);
  * OFF_NEGATIVE    the study's FINAL check put the cell's whole 95% CI below zero (negative evidence);
  * OFF_NO_VALUE    no bar has a positive (shrunk) selection ROI - there is no value to tip;
  * OFF_THIN        fewer than study.MIN_OWN_BETS bets at every bar, own AND pooled;
  * NO_PRICE_DATA   no priced row before the month (no real two-sided bet-time price).

VERDICT per cell: the study's FINAL verdict for that track x market-side x league (PROVEN / PROMISING /
UNPROVEN) when it served there; a league served only through the learned blend, or under NO_SELECTION_EVIDENCE,
is UNPROVEN. Nothing here is applied: the cells go into a proposal (src/threshold_proposals) the owner
approves; src/threshold_apply writes them.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.bars_opt import serve as BS
from src.bars_opt import study as S

ROOT = Path(__file__).resolve().parents[2]
STUDY_CELLS = ROOT / "output" / "bars_opt" / "study" / "cells.json"
FALLBACK_CONFIG = ("RAW", "EDGE", None, None)          # what production computes today (see module docstring)
BASIS_NO_SELECTION = "NO_SELECTION_EVIDENCE"
OFF_NEGATIVE, OFF_NO_VALUE, OFF_THIN_ALL, NO_PRICE = "OFF_NEGATIVE", "OFF_NO_VALUE", "OFF_THIN", "NO_PRICE_DATA"
OFF_NO_OWN = "OFF_NO_OWN_ESTIMATE"
#: own bets at a bar below which a league has no own variance there: the SAME rule src.league_bars.choose_bars
#: uses ("fewer than 2 own bets at a bar = no own variance there: the league's weight at that bar is 0")
OWN_VARIANCE_MIN = 2
#: production market and cell side of each study market-side (serve.MSIDE inverted)
PROD_OF = {v: k for k, v in BS.MSIDE.items()}


def _has_own_estimate(b: dict) -> bool:
    """True when the league has at least OWN_VARIANCE_MIN own bets at its VALUABLE bar. A missing count is
    no own estimate (fail closed, never read as a number)."""
    n = b.get("wf_bets")
    return n is not None and int(n) >= OWN_VARIANCE_MIN


def _no_own_detail(b: dict) -> str:
    n = b.get("wf_bets")
    have = "no own-bet count" if n is None else f"{int(n)} own bet(s)"
    return f"{have} at the VALUABLE bar: the bars would be the pooled curve (weight 0), not the league's own"


def study_verdicts(path: Path = STUDY_CELLS) -> dict:
    """{(track, mside, league): study FINAL row} of the committed study."""
    rows = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else []
    return {(r["track"], r["mside"], r["league"]): r for r in rows if r.get("league")}


def _negative(row: dict | None) -> bool:
    hi = ((row or {}).get("final") or {}).get("ci95", [None, None])[1]
    return hi is not None and float(hi) < 0


def family_cells(args) -> dict:
    """One family's cells for the month. Picklable worker (args = (frame, track, mside, month, verdicts))."""
    f, track, mside, month, verdicts = args
    m = pd.Timestamp(f"{month}-01")
    sel = f[f["date"] < m]
    out = {"track": track, "mside": mside, "month": month, "rows": int(len(sel)), "cells": [], "off": []}
    leagues_priced = sorted(sel["league"].unique())
    leagues_all = sorted(f["league"].unique())
    for lg in leagues_all:
        if lg not in leagues_priced:
            out["off"].append({"league": lg, "reason": NO_PRICE, "detail": f"no priced row before {month}-01"})
    if sel.empty:
        out["chosen"] = None
        return out
    ch = S.select_core(sel, allow_blend=True)
    basis = "SELECTED"
    if ch["chosen"]:
        chosen = S.parse_cfg(ch["chosen"])
        bars = ch["bars"]
    else:
        chosen, basis = FALLBACK_CONFIG, BASIS_NO_SELECTION
        mb, _ = S.fit_cfg(sel, chosen, allow_blend=True)
        bars = {lg: b | {"mode": S.MODE_MODEL} for lg, b in mb.items()}
    out["chosen"], out["config_basis"] = S.cfg_name(chosen), basis
    out["reality_check"] = ch.get("reality_check")
    params = {S.MODE_MODEL: S.source_params(chosen[0], sel) if chosen[0] != S.LEAGUE_WIDE_SOURCE else {},
              S.MODE_LW: {}}
    for lg in leagues_priced:
        b = bars.get(lg)
        vrow = verdicts.get((track, mside, lg))
        if b is None:
            out["off"].append({"league": lg, "reason": OFF_THIN_ALL, "detail": "no bars for this league"})
            continue
        if "off" in b:
            reason = {S.OFF_NO_VALUABLE: OFF_NO_VALUE, S.OFF_THIN: OFF_THIN_ALL}.get(b["off"], b["off"])
            out["off"].append({"league": lg, "reason": reason, "detail": b["off"]})
            continue
        if _negative(vrow):
            out["off"].append({"league": lg, "reason": OFF_NEGATIVE,
                               "detail": f"study FINAL ROI CI95 {vrow['final']['ci95']} entirely below 0"})
            continue
        mode = b["mode"]
        cfg = chosen if mode == S.MODE_MODEL else S.lw_of(chosen)
        blend = b.get("bar_basis") == S.BASIS_BLEND
        if blend and not _has_own_estimate(b):
            out["off"].append({"league": lg, "reason": OFF_NO_OWN, "detail": _no_own_detail(b)})
            continue
        served_in_study = vrow is not None and vrow.get("verdict") in ("PROVEN", "PROMISING", "UNPROVEN")
        verdict = vrow["verdict"] if (served_in_study and not blend and basis == "SELECTED") else "UNPROVEN"
        src_params = params[mode]
        if cfg[0] == "SHRINK":                          # keep only this league's learned weight
            src_params = {"weights": {lg: src_params["weights"][lg]}} if lg in src_params.get("weights", {}) else None
            if src_params is None:
                out["off"].append({"league": lg, "reason": OFF_THIN_ALL, "detail": "no SHRINK weight for the league"})
                continue
        prod_market, side = PROD_OF[mside]
        cell = {"mside": mside, "track": track, "league": lg, "month": month, "config": S.cfg_name(cfg),
                "config_basis": basis, "source": cfg[0], "measure": cfg[1], "odds_max": cfg[2], "odds_min": cfg[3],
                "mode": mode, "valuable_th": b["valuable_th"], "marksman_th": b["marksman_th"],
                "sniper_th": b["sniper_th"], "bar_basis": b.get("bar_basis"),
                "shrink_weight_at_valuable": b.get("shrink_weight_at_valuable"), "wf_bets": b.get("wf_bets"),
                "wf_roi": b.get("wf_roi"), "wf_roi_shrunk": b.get("wf_roi_shrunk"), "tier2": b.get("tier2"),
                "tier2_bets": b.get("tier2_bets"), "tier2_roi": b.get("tier2_roi"),
                "source_params": src_params, "verdict": verdict,
                "study_final": (vrow or {}).get("final"), "study_verdict": (vrow or {}).get("verdict"),
                "study_reason": (vrow or {}).get("reason"), "prod_market": prod_market, "side": side}
        why = BS.validate_cell(cell)
        if why:
            out["off"].append({"league": lg, "reason": "INVALID_CELL", "detail": why})
            continue
        out["cells"].append(cell)
    return out


def not_served_families(universe_dir: Path) -> list[tuple[str, str]]:
    """(track, mside) families the study scores but production does not tip (no serving market:
    the HT Over 0.5 / 1.5 families joined 2026-11-09, P7 item 2). They get study bars and verdicts in
    the report only; they never enter a serving proposal."""
    return sorted({(track, mside) for _f, track, mside in S.load_families(Path(universe_dir))
                   if mside not in PROD_OF})


def build(universe_dir: Path, month: str, workers: int = 4, verdict_path: Path = STUDY_CELLS) -> list[dict]:
    # only families production can serve (PROD_OF); the rest are study-only (not_served_families)
    fams = [x for x in S.load_families(Path(universe_dir)) if x[2] in PROD_OF]
    v = study_verdicts(verdict_path)
    args = [(f, track, mside, month, v) for f, track, mside in fams]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(family_cells, args))
    return [family_cells(a) for a in args]


def production_candidates(families: list[dict]) -> dict:
    """{market: {league: entry}} in the production params shape (src/threshold_apply writes them):
    ou25 entries hold {"bars_opt": {"over": cell, "under": cell}}, side markets {"bars_opt": {"yes": cell}}.
    Informational fields beside bars_opt (status, evidence) are for the proposal reader; serving reads the
    cells only."""
    out: dict = {}
    for fam in families:
        for c in fam["cells"]:
            e = out.setdefault(c["prod_market"], {}).setdefault(c["league"], {"bars_opt": {}, "drop": False,
                                                                               "source": "bars_opt",
                                                                               "month": c["month"]})
            e["bars_opt"][c["side"]] = c
    for m, lgs in out.items():
        for lg, e in lgs.items():
            vs = sorted({c["verdict"] for c in e["bars_opt"].values()})
            e["status"] = "/".join(vs)
            e["approved"] = all(c["verdict"] == "PROVEN" for c in e["bars_opt"].values())
            first = next(iter(e["bars_opt"].values()))
            e["bets"], e["roi"] = first.get("wf_bets"), first.get("wf_roi")
    return out


def write_proposal(families: list[dict], month: str, resolve: bool = True) -> dict:
    from src import threshold_proposals as TP
    cand_dir = TP.CANDIDATE_DIR / "bars_opt_serving" / month
    cands = production_candidates(families)
    for m, data in cands.items():
        TP.write_candidate("standard" if m == "ou25" else m, data, cand_dir)
    ev = TP.PROPOSAL_DIR / "bars_opt_serving"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / f"{month}.json").write_text(json.dumps({"month": month, "families": families}, indent=1, default=str),
                                      encoding="utf-8")
    p = TP.build_proposal(TP.load_candidates(cand_dir), month=month, resolve=resolve,
                          source={"workflow": "bars_opt serving-cells", "candidates_dir": str(cand_dir.relative_to(ROOT)),
                                  "evidence": f"output/threshold_proposals/bars_opt_serving/{month}.json"})
    path = TP.write_proposal(p)
    return {"proposal": str(path.relative_to(ROOT)), "proposal_id": p["proposal_id"],
            "proposal_hash": p["proposal_hash"], "entries": len(p["entries"]), "status": p["status"]}


def refilter(families: list[dict]) -> list[dict]:
    """Apply the OFF_NO_OWN_ESTIMATE rule to families built before it existed (same rule as family_cells)."""
    out = []
    for src in families:
        fam = dict(src, cells=[], off=list(src["off"]))
        for c in src["cells"]:
            if c["bar_basis"] == S.BASIS_BLEND and not _has_own_estimate(c):
                fam["off"].append({"league": c["league"], "reason": OFF_NO_OWN, "detail": _no_own_detail(c)})
            else:
                fam["cells"].append(c)
        out.append(fam)
    return out


def counts(families: list[dict]) -> list[dict]:
    rows = []
    for fam in families:
        c = {"track": fam["track"], "mside": fam["mside"], "chosen": fam.get("chosen"),
             "config_basis": fam.get("config_basis"), "PROVEN": 0, "PROMISING": 0, "UNPROVEN_OWN": 0,
             "UNPROVEN_BLEND": 0, OFF_NEGATIVE: 0, OFF_NO_VALUE: 0, OFF_NO_OWN: 0, OFF_THIN_ALL: 0, NO_PRICE: 0,
             "OTHER_OFF": 0}
        for cell in fam["cells"]:
            if cell["verdict"] in ("PROVEN", "PROMISING"):
                c[cell["verdict"]] += 1
            elif cell["bar_basis"] == S.BASIS_BLEND:
                c["UNPROVEN_BLEND"] += 1
            else:
                c["UNPROVEN_OWN"] += 1
        for o in fam["off"]:
            c[o["reason"] if o["reason"] in c else "OTHER_OFF"] += 1
        rows.append(c)
    return rows
