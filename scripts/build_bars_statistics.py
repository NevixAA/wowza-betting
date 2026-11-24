"""Build reports/dashboard/bars_statistics.json - the per-league bars that PRODUCTION SERVES, statistics only.

The serving proposal is read from config/threshold_applied.json (`last_applied`, written by src/threshold_apply.py
when the owner's approval is applied), never from a hard-coded id. Its approval record comes from
config/threshold_approvals.json (matched by proposal_id AND proposal_hash), the proposal file is the one under
output/threshold_proposals/ whose proposal_id + proposal_hash match, and the per-cell evidence is the file named in the
proposal's `source.evidence`. A new apply therefore changes the page on the next rebuild, and `--check` (CI) fails as
soon as the committed file no longer matches what serves.

KEPT per league x market cell: status (PROVEN / PROMISING / UNPROVEN / OFF), the qualifying bar (valuable_th) and how
it is measured (EDGE / EV / PROB / ALL), the odds RANGE rule if the cell has one (a filter, not a price), mode, the
walk-forward bet count and ROI, the study bets / ROI / 95% CI / matchdays / mean CLV %, and for an OFF cell its
reason and the evidence's own detail text. A field the evidence does not record stays null and the cell says so
(`counts_recorded` false) - nothing is filled. The stake rule per cell comes from config/bankroll_policy.json.
History = the owner's approval chain from config/threshold_approvals.json (ids, dates, entry counts; no cells).
NEVER KEPT: any bookmaker price, odds snapshot, closing line or price file path (OddsAPI / API-Football licences).

    python scripts/build_bars_statistics.py            # write
    python scripts/build_bars_statistics.py --check    # fail if the committed file differs from a rebuild
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dashboard" / "bars_statistics.json"
SCHEMA = "wowza.bars_statistics/2"
APPLIED = "config/threshold_applied.json"
APPROVALS = "config/threshold_approvals.json"
POLICY = "config/bankroll_policy.json"
PROPOSALS_DIR = "output/threshold_proposals"

MSIDE_LABEL = {"over25_over": "O/U 2.5 Over", "over25_under": "O/U 2.5 Under", "btts_yes": "BTTS Yes",
               "over15_over": "Over 1.5", "over35_over": "Over 3.5", "ht_over05_over": "HT Over 0.5",
               "ht_over15_over": "HT Over 1.5"}
#: study market-side -> the policy track's market part (src/bankroll_policy.track_key)
MSIDE_TRACK = {"over25_over": "OU25_OVER", "over25_under": "OU25_UNDER", "btts_yes": "BTTS",
               "over15_over": "OVER15", "over35_over": "OVER35"}
TRACK_LABEL = {"standard": "Standard", "new_format": "New-Format"}

#: Owner stake decision shown on the page, with its recorded source (config/bankroll_policy.json top3_league.stake).
STAKE_PLAN = {
    "decided": "2026-11-17",
    "rule": "From 2026-11-21 every TOP-3 tip carries a real stake of 0.5% of bankroll, flat, placed manually by the "
            "owner at the odds shown on the tip (nothing is executed automatically). No cell and no track stakes on "
            "its own; every other prediction carries no stake.",
    "source": "owner decisions 2026-11-17 (\"Yes, adopt\", \"Show stake on TOP-3\", \"Use tip's odds\"); "
              "config/bankroll_policy.json top3_league.stake (rule FLAT, 0.005, execution MANUAL), every cell and "
              "track at 0",
    "cells": [],
}

NOT_RECORDED = ("the bars study stores no walk-forward or study counts for an "
                "OFF cell, only its reason and detail")


class SourceError(RuntimeError):
    pass


def _load(rel: str):
    p = ROOT / rel
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SourceError(f"{rel} cannot be read ({type(e).__name__})") from e


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def serving() -> dict:
    """The proposal production serves: config/threshold_applied.json last_applied (id + hash + dates)."""
    la = (_load(APPLIED) or {}).get("last_applied") or {}
    if not la.get("proposal_id") or not la.get("proposal_hash"):
        raise SourceError(f"{APPLIED} has no last_applied proposal_id / proposal_hash")
    return {k: la.get(k) for k in ("proposal_id", "proposal_hash", "applied_at", "approved_at", "approved_by")}


def find_proposal(pid: str, phash: str) -> tuple[str, dict]:
    """The proposal file whose proposal_id AND proposal_hash match (latest.json pointers first, then a scan)."""
    base = ROOT / PROPOSALS_DIR
    cands = []
    for ptr in sorted(base.glob("*/latest.json")) + [base / "latest.json"]:
        try:
            d = json.loads(ptr.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("proposal_id") == pid and d.get("path"):
            cands.append(d["path"])
    cands += [p.relative_to(ROOT).as_posix() for p in sorted(base.glob("**/*.json"))
              if "candidates" not in p.parts and not p.name.startswith(("evidence", "cell_price", "latest"))]
    for rel in dict.fromkeys(cands):
        try:
            d = _load(rel)
        except SourceError:
            continue
        if isinstance(d, dict) and d.get("proposal_id") == pid and d.get("proposal_hash") == phash \
                and d.get("entries") is not None:
            return rel, d
    raise SourceError(f"no proposal file under {PROPOSALS_DIR}/ has proposal_id {pid} with the applied hash")


def approval_record(pid: str, phash: str) -> dict:
    for a in _load(APPROVALS).get("approvals") or []:
        if a.get("proposal_id") == pid and a.get("proposal_hash") == phash:
            return a
    raise SourceError(f"{APPROVALS} has no approval for the applied proposal {pid}")


def history(pid: str) -> list[dict]:
    """The owner's approval chain, newest first; the serving one marked SERVING, the rest SUPERSEDED."""
    rows = []
    for a in _load(APPROVALS).get("approvals") or []:
        rows.append({"proposal_id": a.get("proposal_id"), "approved_at": a.get("approved_at"),
                     "approved_by": a.get("approved_by"),
                     "supersedes": a.get("supersedes"), "entries": len(a.get("entries") or []),
                     "state": "SERVING" if a.get("proposal_id") == pid else "SUPERSEDED"})
    return sorted(rows, key=lambda r: str(r["approved_at"]), reverse=True)


def _stake(cells: dict, track: str, mside: str, league: str) -> dict:
    top3 = "no stake of its own; a TOP-3 pick of this cell: 0.5% flat"
    if mside not in MSIDE_TRACK:
        return {"state": "NO_STAKE", "fraction": None, "rule": "no stake (no policy track for this market)"}
    e = cells.get(f"{track}|{MSIDE_TRACK[mside]}|{league}")
    if not e or e.get("base_fraction") is None or float(e["base_fraction"]) <= 0:
        return {"state": "NO_STAKE", "fraction": None, "rule": top3}
    f = float(e["base_fraction"])
    return {"state": str(e.get("state")), "fraction": f,
            "rule": f"{e.get('state')} {100 * f:.1f}% of bank, flat per tip that clears the bar"}


def build_cells(ev: dict, policy_cells: dict) -> tuple[list[dict], list[dict]]:
    stats, offs, fams = {}, {}, []
    for fam in ev.get("families") or []:
        for c in fam.get("cells") or []:
            stats[(fam["track"], fam["mside"], c["league"])] = c
        for o in fam.get("off") or []:
            offs[(fam["track"], fam["mside"], o["league"])] = o
        rc = fam.get("reality_check") or {}
        fams.append({"track": fam["track"], "track_label": TRACK_LABEL.get(fam["track"], fam["track"]),
                     "mside": fam["mside"], "market": MSIDE_LABEL.get(fam["mside"], fam["mside"]),
                     "rows": fam.get("rows"), "chosen": fam.get("chosen"), "configs_tested": rc.get("configs"),
                     "cells_on": len(fam.get("cells") or []), "cells_off": len(fam.get("off") or [])})
    rows = []
    for x in ev["league_market_status"]:
        k = (x["track"], x["mside"], x["league"])
        c = stats.get(k)
        off = x["status"] == "OFF"
        sf = (c or {}).get("study_final") or {}
        ci = sf.get("ci95") or [None, None]
        bars = x.get("bars")
        st = _stake(policy_cells, *k) if not off else {"state": "OFF", "fraction": None, "rule": "OFF: no bar, no tip"}
        detail = x.get("detail") or (offs.get(k) or {}).get("detail")
        if off and x.get("study_only") and not detail:
            detail = ("the family never reached a final study: too few selection-window bets for the study "
                      "(no family record in the evidence)")
        rows.append({
            "track": x["track"], "track_label": TRACK_LABEL.get(x["track"], x["track"]),
            "mside": x["mside"], "market": MSIDE_LABEL.get(x["mside"], x["mside"]), "league": x["league"],
            "status": x["status"],
            "bar": None if not bars else _r(bars[0]),
            "measure": (c or {}).get("measure"), "mode": x.get("mode") or (c or {}).get("mode"),
            "odds_min": _r((c or {}).get("odds_min"), 2), "odds_max": _r((c or {}).get("odds_max"), 2),
            "basis": x.get("basis"), "off_reason": x.get("reason"), "off_detail": detail if off else None,
            "counts_recorded": c is not None,
            "wf_bets": (c or {}).get("wf_bets"), "wf_roi": _r((c or {}).get("wf_roi")),
            "study_bets": sf.get("bets"), "study_roi": _r(sf.get("roi")), "study_matchdays": sf.get("matchdays"),
            "study_ci_lo": _r(ci[0]), "study_ci_hi": _r(ci[1]),
            "study_clv": _r(sf.get("clv")), "study_clv_n": sf.get("clv_n"),
            "stake_state": st["state"], "stake_fraction": st["fraction"], "stake_rule": st["rule"],
        })
    return rows, fams


def build() -> dict:
    sv = serving()
    pid, phash = sv["proposal_id"], sv["proposal_hash"]
    prop_rel, pr = find_proposal(pid, phash)
    ap = approval_record(pid, phash)
    src = pr.get("source") or {}
    ev_rel = src.get("evidence")
    if not ev_rel:
        raise SourceError(f"{prop_rel} names no source.evidence")
    ev = _load(ev_rel)
    pol = _load(POLICY)
    cells, fams = build_cells(ev, ((pol.get("cell_staking") or {}).get("cells") or {}))
    counts = {s: sum(1 for r in cells if r["status"] == s) for s in ("PROVEN", "PROMISING", "UNPROVEN", "OFF")}
    meta = ev.get("meta") or {}
    window = meta.get("validation_window") or src.get("validation_window")
    plan = {"key": "serving", "proposal_id": pid, "proposal_hash": phash, "state": "SERVING",
            "label": f"Serving bars {pid}",
            "approved_by": ap.get("approved_by"), "approved_at": ap.get("approved_at"),
            "applied_at": sv.get("applied_at"), "owner_words": ap.get("owner_words"),
            "supersedes": ap.get("supersedes"), "created_at": pr.get("created_at"),
            "entries": len(pr.get("entries") or []), "approved_entries": len(ap.get("entries") or []),
            "protocol": meta.get("protocol") or src.get("protocol"), "validation_window": window,
            "source": {"serving_config": APPLIED, "approval": APPROVALS, "proposal": prop_rel, "evidence": ev_rel,
                       "workflow": src.get("workflow")},
            "counts": counts, "families": fams, "cells": cells}
    win = f" ({window[0]} to {window[1]})" if window and len(window) == 2 else ""
    return {"schema": SCHEMA,
            "about": "The per-league bars production serves, statistics only (no bookmaker prices). Built by "
                     "scripts/build_bars_statistics.py from config/threshold_applied.json (the applied proposal), "
                     "its approval and its evidence file. Rebuild after every apply; CI --check fails when stale.",
            "serving": sv,
            "stake_plan": STAKE_PLAN,
            "not_recorded": NOT_RECORDED,
            "evidence_note": "WF = walk-forward: the inner out-of-sample bets of the fit window. Study = the bets at "
                             f"the final bar in the validation window{win}, scored once on outcomes previously seen "
                             "(an earlier study had looked at them), so it is not untouched proof. Study ROI 95% CI: "
                             "matchday block bootstrap. CLV = mean closing-line value of the study bets.",
            "history": history(pid),
            "proposals": [plan]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    try:
        text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    except SourceError as e:
        print(f"bars statistics: {e}")
        return 2
    if a.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            print(f"{OUT.relative_to(ROOT)} differs from a rebuild (the serving proposal or its evidence changed)")
            return 1
        print("bars statistics: up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
