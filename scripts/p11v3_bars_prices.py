"""P11 v3 - the P11 v2 bars protocol UNCHANGED, re-measured on the enlarged priced universes (owner 2026-11-23:
HT prices = "Wire in + re-measure (Recommended)"; fallback books = "Extend to all markets (Recommended)").
RESEARCH / PROPOSAL ONLY (protocol config/p11_bars_tiers_protocol_v3.json). Nothing is installed or approved:
the served bars (TP-2026-11-18-1ddfc9f1) and the served tier config (config/top3_tiers_p11.json) stay as they are.

    # 1) priced universes, v2 price rules (BEFORE) and v3 price rules (AFTER) + the coverage counts
    python scripts/p11v3_bars_prices.py prepare --universe <u> --store <odds-history checkout> --aliases <map> --work <W>
    # 2) bars: selection on the fit window, FINAL once on validation (scripts/p11_bars_tiers.py cmd_bars)
    python scripts/p11v3_bars_prices.py bars --work <W>
    # 3) tier parameters on the fit window (scripts/p11_bars_tiers.py cmd_tiers_fit; validation never read)
    python scripts/p11v3_bars_prices.py tiers-fit --work <W>
    # 4) the PROPOSAL (bars cells, month 2026-11) + evidence + the PROPOSED tier config (proposal folder only)
    python scripts/p11v3_bars_prices.py propose --work <W>
    # 5) old (P11 v2) vs new (v3) per cell: reports/P11V3_PRICE_GAPS_REMEASURE.md (EFL first)
    python scripts/p11v3_bars_prices.py compare

Every v2 step is the code of scripts/p11_bars_tiers.py with its module constants pointed at v3; only the price
inputs (BET tiers + HTB, the owner's O1 fallback books per track x market, 00:00-stamped history pairs dropped)
and the prediction universe (the models served now) differ. Prices stay in --work (outside this PUBLIC repo).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bars_opt import oddsapi_prices as P  # noqa: E402
from src.bars_opt import serve as BS  # noqa: E402
from src.bars_opt import study as S  # noqa: E402
from src.research import top3_ranking as R  # noqa: E402

_spec = importlib.util.spec_from_file_location("p11_bars_tiers", ROOT / "scripts" / "p11_bars_tiers.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

PROTOCOL = ROOT / "config" / "p11_bars_tiers_protocol_v3.json"
#: addendum (committed before the re-run fit): Standard predictions = the D1 serving models (PR #96)
ADDENDUM = ROOT / "config" / "p11_bars_tiers_protocol_v3_addendum_d1.json"
#: addendum 2 (owner 2026-11-24, committed before any fit): O/U 2.5 also on the "median real price" basis
MEDIAN_ADDENDUM = ROOT / "config" / "p11_bars_tiers_protocol_v3_addendum_median.json"
OUT = ROOT / "output" / "research" / "p11_bars_tiers_v3"
NAMESPACE = "bars_p11v3"
TAG = "p11v3"
MONTH = M.MONTH
PROP_DIR_REL = Path("output") / "threshold_proposals" / NAMESPACE
TIER_CONFIG_PROPOSED = ROOT / PROP_DIR_REL / "top3_tiers_p11v3_PROPOSED.json"
REPORT = ROOT / "reports" / "P11V3_PRICE_GAPS_REMEASURE.md"
SERVED_TIER_CONFIG = ROOT / "config" / "top3_tiers_p11.json"

#: v2 price rules (BEFORE) - exactly what TP-2026-11-18-1ddfc9f1 was fitted on
V2_BET_TIERS = ["BT", "BTU", "OUU"]
V2_FALLBACK = {"btts": P.O1_FALLBACK_BOOKS}
#: v3 price rules (AFTER) - owner 2026-11-23
V3_BET_TIERS = ["BT", "BTU", "OUU", "HTB"]
O1 = P.O1_FALLBACK_BOOKS
V3_FALLBACK = {
    "standard": {"btts": O1, "over15": O1, "over35": O1, "ht_over05": O1, "ht_over15": O1},
    "new_format": {"btts": O1, "over15": O1, "over35": O1, "ht_over05": O1, "ht_over15": O1, "over25": O1},
}
CLOSE_TIERS = M.CLOSE_TIERS
TRACKS = M.TRACKS
OWNER_QUOTES = ('HT prices = "Wire in + re-measure (Recommended)"; fallback books = "Extend to all markets '
                '(Recommended)" (owner 2026-11-23)')
EFL = ("Championship", "League One", "League Two", "National League")
OLD_EVIDENCE = ROOT / "output" / "threshold_proposals" / "bars_p11v2" / "evidence_2026-11_p11v2.json"
OLD_STUDY_REF = "origin/bars/v6-tiers-clean-1118:output/research/p11_bars_tiers_v2/bars_study.json"
OLD_PROPOSAL_ID = "TP-2026-11-18-1ddfc9f1"


def predictions_rule() -> tuple[dict, dict]:
    """(predictions section in force, addendum record): the addendum's when it exists, else the protocol's."""
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not ADDENDUM.exists():
        return proto["predictions"], {}
    add = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    return add["predictions"], {"addendum": add["addendum"], "file": ADDENDUM.relative_to(ROOT).as_posix(),
                                "sha256_lf": M.sha_lf(ADDENDUM)}


def _median_addendum_record() -> dict:
    if not MEDIAN_ADDENDUM.exists():
        return {}
    return {"file": MEDIAN_ADDENDUM.relative_to(ROOT).as_posix(), "sha256_lf": M.sha_lf(MEDIAN_ADDENDUM)}


MED_NAMESPACE, MED_TAG = "bars_p11v3med", "p11v3med"
MED_OUT = ROOT / "output" / "research" / "p11_bars_tiers_v3_median"


def point_at_median_basis() -> None:
    """Addendum 2/3: the median-basis run writes to its own namespace / research folder (never the fixed-order
    proposal's, never config/top3_tiers_p11.json)."""
    global NAMESPACE, TAG, OUT, TIER_CONFIG_PROPOSED
    NAMESPACE, TAG, OUT = MED_NAMESPACE, MED_TAG, MED_OUT
    TIER_CONFIG_PROPOSED = ROOT / "output" / "threshold_proposals" / MED_NAMESPACE / "top3_tiers_p11v3med_PROPOSED.json"
    M.OUT, M.NAMESPACE, M.TAG, M.TIER_CONFIG = OUT, NAMESPACE, TAG, TIER_CONFIG_PROPOSED


def _point_m_at_v3() -> None:
    """scripts/p11_bars_tiers.py's steps run unchanged; only its output / protocol constants point at v3."""
    M.PROTOCOL, M.OUT, M.NAMESPACE, M.TAG = PROTOCOL, OUT, NAMESPACE, TAG
    M.BET_TIERS = list(V3_BET_TIERS)
    M.TIER_CONFIG = TIER_CONFIG_PROPOSED            # never the served config/top3_tiers_p11.json
    P.BOOK_FALLBACK_BY_MARKET.clear()               # v3 passes its fallback per track (price_track(fallback=...))


_point_m_at_v3()


# ── 1) prepare: BEFORE (v2 rules) and AFTER (v3 rules) ─────────────────────────────────────────────

def _cov_rows(priced: pd.DataFrame, track: str, label: str) -> list[dict]:
    u = priced.copy()
    u["date"] = pd.to_datetime(u["date"])
    u["win"] = np.where(u["date"] < M.FIT_END, "fit",
                        np.where((u["date"] >= M.VAL_START) & (u["date"] <= M.VAL_END), "validation", "other"))
    u = u[(u["win"] != "other") & u["y"].notna()]
    pr = u["yes"].notna() & u["no"].notna()
    book = u["oh_book"].where(pr & (u["price_source"] == P.SRC_HIST))
    out = []
    for (mk, lg, w), x in u.assign(priced=pr, book=book).groupby(["market", "league", "win"]):
        r = {"rules": label, "track": track, "market": mk, "league": lg, "window": w, "rows": int(len(x)),
             "priced": int(x["priced"].sum()), "own": int((x["priced"] & (x["price_source"] == P.SRC_OWN)).sum()),
             "history": int((x["priced"] & (x["price_source"] == P.SRC_HIST)).sum())}
        for b in P.BOOK_ORDER + P.O1_FALLBACK_BOOKS:
            r["hist_" + b] = int((x["book"] == b).sum())
        out.append(r)
    return out


def cmd_prepare(a) -> int:
    work = P.assert_outside_repo(a.work)
    (work / "priced").mkdir(parents=True, exist_ok=True)
    (work / "priced_v2rules").mkdir(parents=True, exist_ok=True)
    aliases = json.loads(Path(a.aliases).read_text(encoding="utf-8"))
    st = {"protocol": json.loads(PROTOCOL.read_text(encoding="utf-8"))["protocol"],
          "protocol_sha256_lf": M.sha_lf(PROTOCOL), "owner": OWNER_QUOTES,
          "predictions": predictions_rule()[0], "addendum": predictions_rule()[1],
          "before": {"bet_tiers": V2_BET_TIERS, "fallback": {k: list(v) for k, v in V2_FALLBACK.items()},
                     "drop_midnight": False},
          "after": {"bet_tiers": V3_BET_TIERS, "fallback": {t: {k: list(v) for k, v in f.items()}
                                                             for t, f in V3_FALLBACK.items()}, "drop_midnight": True},
          "close_tiers": CLOSE_TIERS, "own_backfill_excluded": True}
    cov = []
    for t in TRACKS:
        u = pd.read_parquet(Path(a.universe) / f"universe_{t}.parquet")
        u, sc = P.strip_unproven_own(u, t, aliases)
        before = P.price_track(u, Path(a.store), V2_BET_TIERS, CLOSE_TIERS, fallback=dict(V2_FALLBACK))
        after = P.price_track(u, Path(a.store), V3_BET_TIERS, CLOSE_TIERS, fallback=dict(V3_FALLBACK[t]),
                              drop_midnight=True)
        before["priced"].to_parquet(work / "priced_v2rules" / f"universe_{t}.parquet", index=False)
        after["priced"].to_parquet(work / "priced" / f"universe_{t}.parquet", index=False)
        st[t] = {"universe_rows": int(len(u)), "own_strip": sc}
        for lab, r in (("before", before), ("after", after)):
            st[t][lab] = {"price_source": r["priced"]["price_source"].value_counts().to_dict(),
                          "close_source": r["priced"]["close_source"].value_counts().to_dict(),
                          "books_used": r["stats"].get("books_used"), "integrity": r["stats"].get("integrity"),
                          "pairs": r["stats"].get("pairs")}
            cov += _cov_rows(r["priced"], t, lab)
    ok, why = P.tiers_complete(Path(a.store), V3_BET_TIERS)
    st["tiers_complete"] = {"complete": ok, "reason": why}
    try:
        st["store_commit"] = subprocess.run(["git", "-C", str(a.store), "rev-parse", "HEAD"], capture_output=True,
                                            text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - a store copy without git: recorded as unknown
        st["store_commit"] = None
    (work / "prepare_stats.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps({"price_tiers": st, "coverage": cov}, indent=1, default=str) + "\n",
                                       encoding="utf-8")
    print(json.dumps({t: {lab: st[t][lab]["price_source"] for lab in ("before", "after")} for t in TRACKS},
                     indent=1, default=str))
    return 0


# ── 4) the proposal (v2's cmd_propose with v3 meta; the tier config goes to the proposal folder) ─────

def cmd_propose(a) -> int:
    from src import threshold_proposals as TP
    from src.bars_opt import cells as C
    from src.bars_opt import hist_study as H
    served_before = SERVED_TIER_CONFIG.read_bytes()
    work = P.assert_outside_repo(a.work)
    study_cells_path = work / "study" / "cells.json"
    fams = C.build(work / "priced", MONTH, workers=a.workers, verdict_path=study_cells_path)
    study_cells = json.loads(study_cells_path.read_text(encoding="utf-8"))
    key_m = f"{MONTH}_{TAG}"
    cand_dir = TP.CANDIDATE_DIR / NAMESPACE / key_m
    for m, data in C.production_candidates(fams).items():
        TP.write_candidate("standard" if m == "ou25" else m, data, cand_dir)
    table = H.status_table(fams, study_cells)
    counts = dict(Counter(r["status"] for r in table))
    prop_dir = TP.PROPOSAL_DIR / NAMESPACE
    prop_dir.mkdir(parents=True, exist_ok=True)
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    meta = {"protocol": proto["protocol"], "protocol_sha256_lf": M.sha_lf(PROTOCOL),
            "status": "PROPOSAL - AWAITING OWNER APPROVAL; not installed, serving unchanged "
                      f"({OLD_PROPOSAL_ID} stays served)",
            "owner": OWNER_QUOTES, "predictions": predictions_rule()[0], "addendum": predictions_rule()[1],
            "price_tiers": {"bet": V3_BET_TIERS + ["own (backfill stripped)", "FD pre-match certified join"],
                            "fallback_books_by_track_market": {t: {k: list(v) for k, v in f.items()}
                                                               for t, f in V3_FALLBACK.items()},
                            "history_midnight_stamps": "dropped", "close": CLOSE_TIERS},
            "validation_window": [str(M.VAL_START.date()), str(M.VAL_END.date())]}
    ev = prop_dir / f"evidence_{key_m}.json"
    TP._safe_write(ev, json.dumps({"month": MONTH, "meta": meta, "status_counts": counts,
                                   "league_market_status": table, "families": fams}, indent=1, default=str))
    rel = cand_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    p = TP.build_proposal(TP.load_candidates(cand_dir), month=MONTH, resolve=not a.no_resolve,
                          source={"workflow": "P11 v3 price-gap re-measure (scripts/p11v3_bars_prices.py)",
                                  "candidates_dir": rel,
                                  "evidence": f"output/threshold_proposals/{NAMESPACE}/evidence_{key_m}.json"} | meta)
    path = TP.write_proposal(p, prop_dir)

    frames = {(t, m): f for f, t, m in S.load_families(work / "priced")}
    rows = []
    for fam in fams:
        for c in fam["cells"]:
            if c.get("mode") != S.MODE_MODEL:
                continue
            f = frames[(c["track"], c["mside"])]
            g = f[(f["league"] == c["league"]) & (f["date"] < M.SERVING_LAM_BEFORE)]
            if g.empty:
                continue
            s = BS.score(g[["league", "p_side", "fair", "odds"]], c)
            pu = pd.to_numeric(s["p_used"], errors="coerce")
            rows.append(g.assign(p_used=pu)[pu.notna() & s["in_range"].astype(bool)])
    lam = M.lam_from(pd.concat(rows, ignore_index=True)) if rows else R._lams(None)
    fit = json.loads((OUT / "tier_fit.json").read_text(encoding="utf-8"))
    leagues = {k: {kk: v[kk] for kk in ("gate", "t3_floor", "gate_basis", "t3_basis")}
               for k, v in sorted(fit["params"].items())}
    for fam in fams:
        for c in fam["cells"]:
            k = f"{c['track']}|{c['league']}"
            if c.get("mode") == S.MODE_MODEL and k not in leagues:
                leagues[k] = {"gate": "CAL", "t3_floor": "REL",
                              "gate_basis": "NO_FIT_POOL (cell switched on by the serving refit only): CAL, unvalidated",
                              "t3_basis": "NO_FIT_POOL: REL, unvalidated"}
    cfg = {"protocol": proto["protocol"],
           "status": "PROPOSED ONLY - AWAITING OWNER APPROVAL. Not installed: the served tier config is "
                     "config/top3_tiers_p11.json (P11 v2).",
           "bars_proposal_id": p["proposal_id"], "bars_proposal_hash": p["proposal_hash"],
           "tiers": {"T1_BEST_VALUE": "highest cal_ev", "T2_SAFEST": "highest p_cal of the rest",
                     "T3_RISKY_VALUE": "of the rest, odds >= t3_floor, highest ev_lo (REL: the rest's longest odds)"},
           "gates": {"CAL": "cal_ev > 0", "BAR": "cleared bar and p_used * odds - 1 > 0"},
           "leagues": dict(sorted(leagues.items())),
           "calibration": {"rule": "R2: p_cal = fair + lam * (p_used - fair); ev_lo with lam - lam_se",
                           "fitted_on": f"priced rows < {M.SERVING_LAM_BEFORE.date()}",
                           "cells": lam.round(6).to_dict("records")},
           "stake": "0.5% flat on every TOP-3 tip (owner 2026-11-17); unchanged"}
    TIER_CONFIG_PROPOSED.write_text(json.dumps(cfg, indent=1, default=str) + "\n", encoding="utf-8")
    if SERVED_TIER_CONFIG.read_bytes() != served_before:
        raise SystemExit("the served tier config changed during propose: refused")
    print(json.dumps({"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"], "entries": len(p["entries"]),
                      "proposal": str(Path(path).resolve().relative_to(ROOT.resolve())), "status_counts": counts},
                     indent=1, default=str))
    return 0


# ── 5) old vs new per cell ────────────────────────────────────────────────────────────────────

def _cells(evidence: dict, study: dict | None) -> dict:
    """{(track, league, mside): row} from an evidence file (serving cells + OFF table) and its bars study."""
    st = {}
    for c in (study or {}).get("cells", []):
        st[(c["track"], c["league"], c["mside"])] = c
    out = {}
    for r in evidence["league_market_status"]:
        k = (r["track"], r["league"], r["mside"])
        out[k] = {"status": r["status"], "reason": r.get("reason"), "bar": (r.get("bars") or [None])[0],
                  "wf_bets": None, "study": None}
    for fam in evidence["families"]:
        for c in fam["cells"]:
            k = (c["track"], c["league"], c["mside"])
            o = out.setdefault(k, {"status": c.get("verdict"), "reason": None, "bar": None})
            o.update({"bar": c.get("valuable_th"), "wf_bets": c.get("wf_bets"), "study": c.get("study_final"),
                      "study_verdict": c.get("study_verdict"), "mode": c.get("mode")})
    for k, c in st.items():
        o = out.setdefault(k, {"status": "OFF", "reason": c.get("reason"), "bar": None})
        if not o.get("study") and c.get("validation"):
            o["study"] = c["validation"]
        if o.get("wf_bets") is None:
            o["wf_bets_study"] = c.get("wf_bets")
        o["study_cell_verdict"] = c.get("verdict")
        o["study_cell_reason"] = c.get("reason")
    return out


def _fmt_study(s: dict | None) -> tuple[str, str, str]:
    if not s or not s.get("bets"):
        return "0", "-", "-"
    ci = s.get("ci95") or [None, None]
    roi = s.get("roi")
    ci_s = "-" if ci[0] is None else f"[{ci[0]:+.1%}, {ci[1]:+.1%}]"
    clv = s.get("clv")
    return (str(s.get("bets")), ("-" if roi is None else f"{roi:+.1%}") + " " + ci_s,
            "-" if clv is None else f"{clv:+.2%} (n {s.get('clv_n')})")


def _efl_rank(league: str) -> int:
    """EFL leagues first, in pyramid order; every other league after them (report order only)."""
    return EFL.index(league) if league in EFL else len(EFL)


def compare_rows(old: dict, new: dict) -> list[dict]:
    rows = []
    for k in sorted(set(old) | set(new), key=lambda k: (_efl_rank(k[1]), k[0], k[1], k[2])):
        o, n = old.get(k, {}), new.get(k, {})
        rows.append({"track": k[0], "league": k[1], "mside": k[2], "old": o, "new": n})
    return rows


def cmd_compare(a) -> int:
    from src import threshold_proposals as TP
    old_ev = json.loads(OLD_EVIDENCE.read_text(encoding="utf-8"))
    try:
        old_st = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", OLD_STUDY_REF], capture_output=True,
                                           text=True, check=True, encoding="utf-8").stdout)
    except Exception:  # noqa: BLE001
        old_st = None
    new_ev_path = TP.PROPOSAL_DIR / NAMESPACE / f"evidence_{MONTH}_{TAG}.json"
    new_ev = json.loads(new_ev_path.read_text(encoding="utf-8"))
    new_st = json.loads((OUT / "bars_study.json").read_text(encoding="utf-8"))
    cov = json.loads((OUT / "coverage.json").read_text(encoding="utf-8"))
    latest = json.loads((TP.PROPOSAL_DIR / NAMESPACE / "latest.json").read_text(encoding="utf-8"))
    rows = compare_rows(_cells(old_ev, old_st), _cells(new_ev, new_st))
    (OUT / "old_vs_new.json").write_text(json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8")
    text = render(rows, cov, old_ev, new_ev, latest)
    med_ev_path = TP.PROPOSAL_DIR / MED_NAMESPACE / f"evidence_{MONTH}_{MED_TAG}.json"
    if med_ev_path.exists():
        med_ev = json.loads(med_ev_path.read_text(encoding="utf-8"))
        med_st = json.loads((MED_OUT / "bars_study.json").read_text(encoding="utf-8"))
        med_latest = json.loads((TP.PROPOSAL_DIR / MED_NAMESPACE / "latest.json").read_text(encoding="utf-8"))
        text += render_median(_cells(old_ev, old_st), _cells(new_ev, new_st), _cells(med_ev, med_st), med_ev, med_latest)
    REPORT.write_text(text, encoding="utf-8")
    print(json.dumps({"cells": len(rows), "old": old_ev["status_counts"], "new": new_ev["status_counts"],
                      "report": str(REPORT.relative_to(ROOT))}, indent=1))
    return 0


def render_median(old: dict, fixed: dict, med: dict, med_ev: dict, med_latest: dict) -> str:
    """O/U 2.5 cells side by side: old (P11 v2) / fixed-order D1 / median real price basis (a). EFL first."""
    keys = sorted({k for k in set(old) | set(fixed) | set(med) if k[2].startswith("over25_")},
                  key=lambda k: (_efl_rank(k[1]), k[0], k[1], k[2]))
    L = ["", "## O/U 2.5 on two bases: fixed-order (D1) vs median real price (addenda 2 + 3)", "",
         "Median basis (owner 2026-11-24, option (a)): per side the median of the real (vigged) prices of every eu + "
         "uk book with a bet-time two-sided 2.5 pair whose overround is inside config/market_edge_v2.json and that "
         "de-vigs; >= 3 books, else unpriced. No single-book own / football-data price on this basis, and nothing "
         "before 2023-05-03 (no per-book history), so the fit window is much shorter.",
         f"Median-basis proposal: `{med_latest.get('proposal_id')}` (hash `{str(med_latest.get('proposal_hash'))[:12]}`), "
         f"{med_latest.get('n_entries')} entries, namespace `{MED_NAMESPACE}` - a separate PROPOSAL, not installed. "
         f"Statuses (all cells of that run): {med_ev['status_counts']}. Only its O/U 2.5 cells differ from the fixed run.",
         "The validation window (2026-09-01..11-08) was seen before: re-validation numbers, never proof.", "",
         "| Track | League | Cell | Old v2 status / bar / study bets / ROI [CI] | Fixed D1 status / bar / WF / study bets / "
         "ROI [CI] / CLV | Median status / bar / WF / study bets / ROI [CI] / CLV |", "|---|---|---|---|---|---|"]

    def cell(d, short=False):
        if not d:
            return "-"
        b, roi, clv = _fmt_study(d.get("study"))
        st = d.get("status") or "-"
        if st == "OFF" and (d.get("reason") or d.get("study_cell_reason")):
            st += f" ({d.get('reason') or d.get('study_cell_reason')})"
        bar = "-" if d.get("bar") is None else f"{d['bar']:.2f}"
        wf = d.get("wf_bets") if d.get("wf_bets") is not None else d.get("wf_bets_study")
        if short:
            return f"{st} / {bar} / {b} / {roi}"
        return f"{st} / {bar} / {'-' if wf is None else wf} / {b} / {roi} / {clv}"
    for k in keys:
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {cell(old.get(k), True)} | {cell(fixed.get(k))} | {cell(med.get(k))} |")
    return "\n".join(L) + "\n"


def _cov_table(cov: dict, leagues=None, markets=None) -> list[str]:
    df = pd.DataFrame(cov["coverage"])
    if df.empty:
        return ["(no coverage rows)"]
    g = df.groupby(["track", "league", "market", "rules"])[["rows", "priced"]].sum().unstack("rules")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    if leagues is not None:
        g = g[g["league"].isin(leagues)]
    if markets is not None:
        g = g[g["market"].isin(markets)]
    a_hist = df[df["rules"] == "after"].groupby(["track", "league", "market"])[
        ["hist_bet365", "hist_pinnacle", "hist_williamhill", "hist_onexbet", "hist_ladbrokes_uk"]].sum()
    lines = ["| Track | League | Market | Rows | Priced before (v2 rules) | Priced after (v3 rules) | Added | "
             "After: Pinnacle / WH / 1xBet / Ladbrokes (history rows) |", "|---|---|---|---|---|---|---|---|"]
    g = g.assign(_o=g["league"].map(_efl_rank)).sort_values(["_o", "track", "league", "market"])
    for r in g.itertuples(index=False):
        h = a_hist.loc[(r.track, r.league, r.market)] if (r.track, r.league, r.market) in a_hist.index else None
        hs = "-" if h is None else f"{int(h['hist_pinnacle'])} / {int(h['hist_williamhill'])} / " \
                                   f"{int(h['hist_onexbet'])} / {int(h['hist_ladbrokes_uk'])}"
        lines.append(f"| {r.track} | {r.league} | {r.market} | {int(r.rows_after):,} | {int(r.priced_before):,} | "
                     f"{int(r.priced_after):,} | {int(r.priced_after - r.priced_before):+,} | {hs} |")
    return lines


def _n_proven(ev: dict) -> int:
    """PROVEN cells of an evidence file (a status absent from status_counts has no cell)."""
    return sum(1 for r in ev["league_market_status"] if r["status"] == "PROVEN")


def render(rows: list[dict], cov: dict, old_ev: dict, new_ev: dict, latest: dict) -> str:
    pt = cov["price_tiers"]
    L = ["# P11 v3: bars re-measured on the enlarged priced universes (PROPOSAL, awaiting owner approval)", "",
         f"Owner 2026-11-23: {OWNER_QUOTES}. Serving is unchanged: {OLD_PROPOSAL_ID} (P11 v2) stays served and "
         "`config/top3_tiers_p11.json` is untouched. Counts and statistics only; no prices, keys or club names.", "",
         "## What changed (price inputs only; the P11 v2 bars protocol is unchanged)", "",
         f"- Protocol `{pt['protocol']}` (`config/p11_bars_tiers_protocol_v3.json`, sha256 LF "
         f"`{pt['protocol_sha256_lf'][:16]}`), committed before the re-measure.",
         "- BET tiers BT, BTU, OUU **+ HTB** (first-half totals at the bet time). `ht_over05` / `ht_over15` take only the "
         "first-half 0.5 / 1.5 line.",
         "- The owner's O1 fallback order (Bet365 > Pinnacle > William Hill > 1xBet > Ladbrokes) on BTTS, O1.5, O3.5, "
         "HT 0.5 / 1.5 (both tracks) and New-Format O/U 2.5. Standard O/U 2.5 keeps football-data / Bet365 > Pinnacle.",
         "- History pairs stamped exactly 00:00:00 UTC are dropped (counted). Own 00:00 backfill stays stripped.",
         f"- Predictions (models served now): Standard: {pt['predictions']['standard']}",
         f"- New-Format: {pt['predictions']['new_format']}",
         (f"- Addendum `{pt['addendum']['file']}` (sha256 LF `{pt['addendum']['sha256_lf'][:16]}`), committed before "
          "this re-run's fit." if pt.get("addendum") else "- No addendum."),
         "- **Not used:** the pending Standard refit D1 (another agent's draft PR, awaiting owner approval).",
         f"- Store: wowza-exec `odds-history` @ `{(pt.get('store_commit') or '?')[:8]}`. No credits spent.", "",
         "## Result in brief", "",
         f"- Old (P11 v2) cell statuses: {old_ev['status_counts']}.",
         f"- New (v3) cell statuses: {new_ev['status_counts']}.",
         f"- New proposal: `{latest.get('proposal_id')}` (hash `{str(latest.get('proposal_hash'))[:12]}`), "
         f"{latest.get('n_entries')} entries, {latest.get('n_serving_changes')} would change serving IF approved.",
         "- PROVEN still needs >= 150 validation bets + Holm + CI lower bound > 0. The validation window was scored "
         "before (P9, P11 v1, v2): these are re-validation numbers, never proof (only live tips prove a cell).",
         f"- PROVEN cells: old {_n_proven(old_ev)}, new {_n_proven(new_ev)}.",
         "- **Read the Standard differences with care.** Old vs new on the Standard track moves two things at once: the "
         "price inputs AND the predictions (P11 v2 used the refit_1118b frame `9ed237eb`, 2021/22 start; the models "
         "served now train from 2016/17 (see Predictions), so their out-of-sample universe has "
         f"{pt['standard']['universe_rows']:,} rows). New-Format predictions are identical, so New-Format differences "
         "come from the prices only.",
         "- Tier parameters (T1/T2/T3) were re-fitted on the fit window only (`tier_fit.json`); the tier validation was "
         "NOT re-run. The proposed tier config is `output/threshold_proposals/bars_p11v3/top3_tiers_p11v3_PROPOSED.json`; "
         "the served `config/top3_tiers_p11.json` is unchanged.", ""]
    efl_on = [(r["league"], r["mside"], (r["old"] or {}).get("status"), (r["new"] or {}).get("status")) for r in rows
              if r["league"] in EFL and ((r["old"] or {}).get("status") != (r["new"] or {}).get("status"))]
    L += ["- EFL status changes (old -> new): " + ("; ".join(f"{lg} {ms}: {o} -> {n}" for lg, ms, o, n in efl_on)
                                                  if efl_on else "none") + ".", ""]
    for t in TRACKS:
        L.append(f"- {t}: price source before {pt[t]['before']['price_source']}, after {pt[t]['after']['price_source']}; "
                 f"integrity failures after: {sum((pt[t]['after']['integrity'] or {}).get('failures', {}).values())}; "
                 f"00:00-stamped history pairs dropped: {(pt[t]['after']['pairs'] or {}).get('pairs_midnight_stamp_dropped')}.")
    L += ["", "## Old (P11 v2) vs new (v3) per cell - EFL first", "",
          "Bar = VALUABLE bar. WF = walk-forward bets at the bar. Study = validation (2026-09-01..11-08) bets, ROI "
          "[matchday-block 95% CI], CLV.", "",
          "| Track | League | Cell | Old status | Old bar | Old WF | Old study bets | Old ROI [CI] | Old CLV | "
          "New status | New bar | New WF | New study bets | New ROI [CI] | New CLV |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def side(d):
        if not d:
            return ["-"] * 6
        b, roi, clv = _fmt_study(d.get("study"))
        stt = d.get("status") or "-"
        if stt == "OFF" and (d.get("reason") or d.get("study_cell_reason")):
            stt += f" ({d.get('reason') or d.get('study_cell_reason')})"
        bar = "-" if d.get("bar") is None else f"{d['bar']:.2f}"
        wf = d.get("wf_bets") if d.get("wf_bets") is not None else d.get("wf_bets_study")
        return [stt, bar, "-" if wf is None else str(wf), b, roi, clv]
    for r in rows:
        L.append("| " + " | ".join([r["track"], r["league"], r["mside"]] + side(r["old"]) + side(r["new"])) + " |")
    L += ["", "## Coverage before / after per league x market (fit + validation rows with a result)", "",
          "Before = v2 price rules (what the served bars were fitted on, on the same prediction rows). After = v3 rules.",
          ""]
    L += _cov_table(cov)
    L += ["", "## Reproduce", "",
          "1. Standard predictions: `src.retrain_harness.runner.run` (champion_recipe, walk_forward + holdout, track "
          "standard) on main with the saved serving frame named under Predictions (hash re-checked). New-Format: the P11 v2 harness "
          "run (code 694995f90, frame 9ed237ebc04aee09).",
          "2. `scripts/bars_opt.py universe` (Standard with `--certified-fd`); New-Format universe of P11 v2.",
          "3. `scripts/p11v3_bars_prices.py prepare`, `bars`, `tiers-fit`, `propose`, `compare`.",
          "4. Install = owner approval only (`config/threshold_approvals.json`); nothing here applies it.", ""]
    return "\n".join(L)


def cmd_coverage_report(a) -> int:
    """Counts-only coverage report (markdown) from a prepare run's coverage.json, e.g. for the code PR."""
    cov = json.loads(Path(a.coverage).read_text(encoding="utf-8"))
    pt = cov["price_tiers"]
    L = [f"# {a.title}", "", f"Owner 2026-11-23: {OWNER_QUOTES}. Counts only: no prices, keys or club names. "
         "No credits spent. Serving bars and tier config unchanged.", "",
         f"- Universe: {a.universe_note}",
         f"- Store: wowza-exec `odds-history` @ `{(pt.get('store_commit') or '?')[:8]}`.",
         f"- Before = P11 v2 price rules: BET tiers {', '.join(pt['before']['bet_tiers'])}; fallback books on BTTS only.",
         f"- After = v3 rules: BET tiers {', '.join(pt['after']['bet_tiers'])}; O1 fallback (William Hill > 1xBet > "
         "Ladbrokes after Bet365 > Pinnacle) on BTTS, O1.5, O3.5, HT 0.5/1.5 (both tracks) + New-Format O/U 2.5; "
         "00:00:00-stamped history pairs dropped.", ""]
    for t in TRACKS:
        a_ = pt[t]["after"]
        L.append(f"- {t}: price source before {pt[t]['before']['price_source']}, after {a_['price_source']}; join "
                 f"integrity failures after {sum((a_['integrity'] or {}).get('failures', {}).values())} of "
                 f"{(a_['integrity'] or {}).get('rows_checked')} history rows; 00:00-stamped history pairs dropped "
                 f"{(a_['pairs'] or {}).get('pairs_midnight_stamp_dropped')}.")
    L += ["", "Rows = fit (< 2026-09-01) + validation (2026-09-01..11-08) rows with a result. EFL first.", ""]
    L += _cov_table(cov)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(a.out)
    return 0


# ── addendum 2 (owner 2026-11-24): "median real price" basis for O/U 2.5 ───────────────────────────

MEDIAN_MARKET = "over25"


def _hist_pairs(u: pd.DataFrame, store: Path, tiers) -> tuple[pd.DataFrame, pd.Series, dict, pd.DataFrame]:
    """(two-sided exact-line pairs of the BET tiers, fixture dates, clean fixture->event map, odds) for a universe."""
    from src import oddsapi_history_adapter as AD
    st = AD.load_store(store)
    odds = st[st["tier"].isin(list(tiers)) & st["league"].isin(set(u["league"]))]
    ctx = AD.load_context(store, u)
    fmap, _ = AD.match_universe(u, odds, ctx)
    fmap, _ = P.clean_fixture_map(fmap, odds)
    long, _s, _u = AD.attach(u, odds, ctx)
    long = long[long["fixture_id"].isin(set(fmap))] if len(long) else long
    pairs, _st = P.history_pairs(long)
    fx = u.assign(fixture_id=u["row_id"].astype(str)).drop_duplicates("fixture_id")
    return pairs, fx.set_index("fixture_id")["date"], fmap, odds


def cmd_median_coverage(a) -> int:
    """Counts only: per track x league, O/U 2.5 fit + validation rows with a result, and how many have >= 1 / >= 3
    two-sided books at bet time in the store (BET tiers BT + OUU, eu + uk)."""
    rows = []
    for t in TRACKS:
        u = pd.read_parquet(Path(a.universe) / f"universe_{t}.parquet")
        u = u[u["market"] == MEDIAN_MARKET].copy()
        u["date"] = pd.to_datetime(u["date"])
        u = u[((u["date"] < M.FIT_END) | ((u["date"] >= M.VAL_START) & (u["date"] <= M.VAL_END))) & u["y"].notna()]
        pairs, dates, _f, _o = _hist_pairs(u, Path(a.store), ["BT", "OUU"])
        bc = P.book_counts(pairs[pairs["market"] == MEDIAN_MARKET], dates)
        n = u.assign(fixture_id=u["row_id"].astype(str)).merge(bc[["fixture_id", "n_books"]], on="fixture_id", how="left")
        n["win"] = np.where(n["date"] < M.FIT_END, "fit", "validation")
        for (lg, w), x in n.groupby(["league", "win"]):
            k = x["n_books"]
            rows.append({"track": t, "league": lg, "window": w, "rows": int(len(x)),
                         "rows_since_2023_05_03": int((x["date"] >= "2023-05-03").sum()),
                         "books_ge1": int((k >= 1).sum()), "books_ge3": int((k >= P.MEDIAN_MIN_BOOKS).sum()),
                         "median_books": float(k.median()) if k.notna().any() else None})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "median_coverage.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    print(df.groupby(["track", "window"])[["rows", "rows_since_2023_05_03", "books_ge1", "books_ge3"]].sum().to_string())
    return 0


SRC_MEDIAN = "oddsapi_median"
#: addendum 3 (coordinator decision 2026-11-24, option (a)): the serving rule's overround bounds (PR #104)
MEDIAN_ADDENDUM_3 = ROOT / "config" / "p11_bars_tiers_protocol_v3_addendum_median_a.json"


def serving_overround() -> tuple[float, float]:
    """The overround bounds the serving reference price uses (config/market_edge_v2.json overround_min / max, the
    file PR #104 src/reference_price.py reads); a missing key stops the run (no stand-in bounds)."""
    cfg = json.loads((ROOT / "config" / "market_edge_v2.json").read_text(encoding="utf-8"))
    if "overround_min" not in cfg or "overround_max" not in cfg:
        raise SystemExit("config/market_edge_v2.json has no overround_min / overround_max: refused")
    return float(cfg["overround_min"]), float(cfg["overround_max"])


def _median_close(u: pd.DataFrame, store: Path) -> pd.DataFrame:
    """CLV pair on the same basis: per fixture the median of each book's LATEST pre-kickoff close-role (B1) pair,
    >= MEDIAN_MIN_BOOKS books (never a bet price)."""
    pairs, dates, _f, _o = _hist_pairs(u, store, CLOSE_TIERS)
    pairs = pairs[pairs["market"] == MEDIAN_MARKET]
    if pairs.empty:
        return pd.DataFrame(columns=["fixture_id", "yes_c", "no_c"])
    v = P._valid_pairs(pairs, dates, {}, False, True)
    last = v.sort_values("ts").groupby(["fixture_id", "bookmaker"], as_index=False).last()
    g = last.groupby("fixture_id").agg(yes_c=("yes", "median"), no_c=("no", "median"), n=("bookmaker", "nunique"))
    return g[g["n"] >= P.MEDIAN_MIN_BOOKS].reset_index()[["fixture_id", "yes_c", "no_c"]]


def cmd_prepare_median(a) -> int:
    """The median-basis priced universes: a copy of --base-work's v3 priced universes where every O/U 2.5 row is
    re-priced ONLY by the median real price (>= 3 two-sided books at bet time; own / football-data single-book
    pairs no longer price O/U 2.5 on this basis); every other market is byte-identical."""
    base, work = P.assert_outside_repo(a.base_work), P.assert_outside_repo(a.work)
    (work / "priced").mkdir(parents=True, exist_ok=True)
    st = {"addendum": predictions_rule()[1], "median_addendum": _median_addendum_record(),
          "median_addendum_3": {"file": MEDIAN_ADDENDUM_3.relative_to(ROOT).as_posix(), "sha256_lf": M.sha_lf(MEDIAN_ADDENDUM_3)},
          "overround": serving_overround(), "tracks": {}}
    for t in TRACKS:
        pu = pd.read_parquet(base / "priced" / f"universe_{t}.parquet")
        m = pu["market"] == MEDIAN_MARKET
        u = pu[m]
        pairs, dates, fmap, odds = _hist_pairs(u, Path(a.store), ["BT", "OUU"])
        ps = {}
        med = P.median_pick(pairs[pairs["market"] == MEDIAN_MARKET], dates, ps, overround=serving_overround())
        cl = _median_close(u, Path(a.store))
        med = med.merge(cl, on="fixture_id", how="left", suffixes=("", "_cl"))
        for c in ("yes_c", "no_c"):
            med[c] = med[c + "_cl"]
        mm = med.set_index("fixture_id")
        fid = pu["row_id"].astype(str)
        hit = m & fid.isin(mm.index)
        for c in ("yes", "no", "yes_c", "no_c"):
            pu.loc[m, c] = np.nan
            pu.loc[hit, c] = fid[hit].map(mm[c]).astype(float).to_numpy()
        pu.loc[m, "price_source"] = np.where(hit[m], SRC_MEDIAN, None)
        pu.loc[m, "close_source"] = np.where(hit[m] & pu.loc[m, "yes_c"].notna(), SRC_MEDIAN, None)
        pu.loc[m, "oh_book"] = np.where(hit[m], fid[m].map(mm["bookmaker"]), None)
        pu.loc[m, "oh_event_id"] = np.where(hit[m], fid[m].map(mm["event_id"]), None)
        pu.loc[m, "oh_snapshot_ts"] = np.where(hit[m], fid[m].map(mm["ts"]).astype(str), None)
        pu.loc[m, "oh_commence_time"] = np.where(hit[m], fid[m].map(mm["ko"]).astype(str), None)
        pu.loc[m, "h_league"] = np.where(hit[m], fid[m].map(mm["league"]), None)
        chk = pu[hit].assign(fixture_id=fid[hit], price_source=P.SRC_HIST)
        es = odds.drop_duplicates("event_id").set_index("event_id")["sport_key"].to_dict()
        integ = P.verify_join(chk, fmap, es)
        pu.to_parquet(work / "priced" / f"universe_{t}.parquet", index=False)
        st["tracks"][t] = {"ou25_rows": int(m.sum()), "ou25_priced_median": int(hit.sum()),
                           "ou25_close_median": int((hit & pu["yes_c"].notna()).sum()), "pairs": ps, "integrity": integ}
    (work / "prepare_stats.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    print(json.dumps({t: {k: v for k, v in x.items() if k != "pairs"} for t, x in st["tracks"].items()}, indent=1,
                     default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    for x in ("--universe", "--store", "--aliases", "--work"):
        p.add_argument(x, required=True)
    for name in ("bars", "tiers-fit", "propose"):
        s = sub.add_parser(name)
        s.add_argument("--work", required=True)
        s.add_argument("--basis", choices=["fixed", "median"], default="fixed",
                       help="median = addendum 2/3 O/U 2.5 median real price basis (namespace bars_p11v3med)")
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--no-resolve", action="store_true")
    sub.add_parser("compare")
    pm = sub.add_parser("prepare-median")
    for x in ("--base-work", "--work", "--store"):
        pm.add_argument(x, required=True)
    mc = sub.add_parser("median-coverage")
    mc.add_argument("--universe", required=True)
    mc.add_argument("--store", required=True)
    c = sub.add_parser("coverage-report")
    c.add_argument("--coverage", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--title", default="P11 v3 price coverage before / after")
    c.add_argument("--universe-note", default="")
    a = ap.parse_args(argv)
    if getattr(a, "basis", "fixed") == "median":
        point_at_median_basis()
    return {"prepare": cmd_prepare, "bars": M.cmd_bars, "tiers-fit": M.cmd_tiers_fit, "propose": cmd_propose,
            "compare": cmd_compare,
            "coverage-report": cmd_coverage_report, "median-coverage": cmd_median_coverage,
            "prepare-median": cmd_prepare_median}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
