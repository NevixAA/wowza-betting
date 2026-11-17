"""Build reports/dashboard/bars_statistics.json - the per-league bars, STATISTICS ONLY, for the Bars page.

Owner 2026-11-17: "update the dashboard with the current bars". The bars proposals live on branches that go
live with the go-live merge (2026-11-21), and their files carry private price provenance. This script reads
the two proposals straight from git (by commit, recorded in the output) and keeps only statistics:

  (a) TP-2026-11-15-bade2c0b  owner-APPROVED 2026-11-15 (protocol v5), branch prep/bars-oddsapi-hist
  (b) TP-2026-11-16-500964af  the clean-model re-run, branch bars/v5-clean-models, AWAITING OWNER (Nov 20)

KEPT per league x market cell: status (PROVEN / PROMISING / UNPROVEN / OFF), the qualifying bar
(valuable_th) and how it is measured (EDGE / EV / PROB / ALL), the odds RANGE rule if the cell has one (a
filter, not a price), mode, the walk-forward bet count and ROI, the study bets / ROI / 95% CI / matchdays /
mean CLV % and the OFF reason. The stake rule per cell comes from config/bankroll_policy.json cell_staking on
the approved branch.
NEVER KEPT: any bookmaker price, odds snapshot, closing line or price file path (OddsAPI / API-Football
licences: no redistribution).

    python scripts/build_bars_statistics.py            # write
    python scripts/build_bars_statistics.py --check    # fail if the committed file differs from a rebuild
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dashboard" / "bars_statistics.json"
SCHEMA = "wowza.bars_statistics/1"

#: (key, label, commit, proposal path, owner state). Commits are FIXED so the file is reproducible.
SOURCES = [
    {"key": "approved", "proposal_id": "TP-2026-11-15-bade2c0b",
     "label": "Approved bars (owner 2026-11-15, protocol v5)",
     "state": "APPROVED",
     "state_text": "Owner-approved 2026-11-15 (\"Approve, PROMISING small\"); supersedes TP-2026-11-10-d999073a. "
                   "Served from the go-live merge on 2026-11-21 unless the owner picks (b) on 2026-11-20.",
     "branch": "prep/bars-oddsapi-hist", "commit": "d7bf522ef55158b21ebe784b374e194442580fd6",
     "proposal": "output/threshold_proposals/bars_oddsapi_hist/2026-11_bade2c0b.json"},
    {"key": "clean_rerun", "proposal_id": "TP-2026-11-16-500964af",
     "label": "Clean-model re-run (awaiting owner approval on Nov 20)",
     "state": "AWAITING_OWNER",
     "state_text": "The same protocol v5 re-run on the clean retrained models (reports/P9_BARS_V5_CLEAN_MODELS.md). "
                   "A proposal only: nothing serves until the owner approves it on 2026-11-20.",
     "branch": "bars/v5-clean-models", "commit": "5f481d3b3ebefaae029ab24434a1f2053bfebb83",
     "proposal": "output/threshold_proposals/bars_oddsapi_hist/2026-11_500964af.json",
     "report": "reports/P9_BARS_V5_CLEAN_MODELS.md"},
]
POLICY = {"branch": "prep/bars-oddsapi-hist", "commit": "d7bf522ef55158b21ebe784b374e194442580fd6",
          "path": "config/bankroll_policy.json"}

MSIDE_LABEL = {"over25_over": "O/U 2.5 Over", "over25_under": "O/U 2.5 Under", "btts_yes": "BTTS Yes",
               "over15_over": "Over 1.5", "over35_over": "Over 3.5", "ht_over05_over": "HT Over 0.5",
               "ht_over15_over": "HT Over 1.5"}
#: study market-side -> the policy track's market part (src/bankroll_policy.track_key)
MSIDE_TRACK = {"over25_over": "OU25_OVER", "over25_under": "OU25_UNDER", "btts_yes": "BTTS",
               "over15_over": "OVER15", "over35_over": "OVER35"}
TRACK_LABEL = {"standard": "Standard", "new_format": "New-Format"}

#: Owner stake decisions shown on the page, each with its recorded source.
STAKE_PLAN = {
    "decided": "2026-11-16",
    "rule": "Real stakes on 2026-11-21 ONLY on the 3 PROMISING cells, small (0.5% of bank, flat per cell). "
            "Every other cell and every other TOP-3 tip is PAPER (shown and recorded, stake 0).",
    "source": "owner decision 2026-11-16 (\"3 proven cells only\"); stakes recorded in config/bankroll_policy.json "
              "cell_staking on prep/bars-oddsapi-hist",
    "cells": [
        {"track": "standard", "mside": "over25_over", "league": "La Liga 2"},
        {"track": "new_format", "mside": "over25_over", "league": "Finland Veikkausliiga"},
        {"track": "new_format", "mside": "over35_over", "league": "Finland Veikkausliiga"},
    ],
    "overrides": [
        {"track": "new_format", "mside": "over35_over", "league": "Finland Veikkausliiga", "fraction": 0.005,
         "text": "Finland NF Over 3.5 at 0.5%: owner override 2026-11-15 \"Lift for Finland only\" (the New-Format "
                 "Over 3.5 track stays blocked for every other league)."},
        {"track": "new_format", "mside": "btts_yes", "league": "Argentina Primera Division", "fraction": 0.005,
         "text": "Argentina NF BTTS at 0.5%: owner 2026-11-15 \"Reduce to 0.5%\" (UNPROVEN on real bet-time prices, "
                 "kept small as a league rule). The 2026-11-16 decision names only the 3 PROMISING cells for real "
                 "stakes, so whether Argentina stakes real money or paper from 2026-11-21 is confirmed with the "
                 "2026-11-20 approval."},
    ],
}


#: Tested and NOT adopted (owner 2026-11-16 asked for the run; the report recommends keeping (b)). Every figure is
#: copied from the report at the recorded commit (its section 2 table); none is recomputed here.
NOT_ADOPTED = {
    "title": "Per-league calibration variants (P10): tested, not adopted",
    "proposal_id": "TP-2026-11-16-9cef8a02",
    "state": "NOT_ADOPTED",
    "branch": "bars/v5-league-calib", "commit": "25f3eb98af2b9d4157b5db84bacbae0a5679bc7c",
    "report": "reports/P10_BARS_LEAGUE_CALIB.md",
    "text": "The same protocol v5 bars study on two calibration variants of the clean team models (A = league "
            "level + logit calibration, B = logit calibration alone). Neither creates a PROVEN cell in any "
            "league, the same as the clean champion (0); every combined-Holm p is 1.0. Both variants drop La "
            "Liga 2 O2.5 (a staked cell) to UNPROVEN. The report recommends keeping (b) as the basis; the "
            "variant-A proposal TP-2026-11-16-9cef8a02 is filed as evidence only and is not approved.",
    "counts": [
        {"variant": "Clean champion (b) TP-2026-11-16-500964af", "PROVEN": 0, "PROMISING": 3, "UNPROVEN": 49,
         "OFF": 142},
        {"variant": "A: league level + logit calibration, TP-2026-11-16-9cef8a02", "PROVEN": 0, "PROMISING": 5,
         "UNPROVEN": 70, "OFF": 119},
        {"variant": "B: logit calibration alone (research, no proposal filed)", "PROVEN": 0, "PROMISING": 7,
         "UNPROVEN": 56, "OFF": 131},
    ],
}


def _git_json(commit: str, path: str):
    out = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, check=True)
    return json.loads(out.stdout.decode("utf-8"))


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _stake(cells: dict, track: str, mside: str, league: str) -> dict:
    if mside not in MSIDE_TRACK:
        return {"state": "PAPER", "fraction": None, "rule": "PAPER (no policy track for this market)"}
    e = cells.get(f"{track}|{MSIDE_TRACK[mside]}|{league}")
    if not e or e.get("base_fraction") is None or float(e["base_fraction"]) <= 0:
        return {"state": "PAPER", "fraction": None, "rule": "PAPER (no cell entry: stake 0, recorded)"}
    f = float(e["base_fraction"])
    return {"state": str(e.get("state")), "fraction": f,
            "rule": f"{e.get('state')} {100 * f:.1f}% of bank, flat per tip that clears the bar"}


def build_one(src: dict, policy_cells: dict) -> dict:
    pr = _git_json(src["commit"], src["proposal"])
    assert pr["proposal_id"] == src["proposal_id"], (pr["proposal_id"], src["proposal_id"])
    ev_path = pr["source"]["evidence"]
    ev = _git_json(src["commit"], ev_path)
    stats = {}
    for fam in ev.get("families") or []:
        for c in fam.get("cells") or []:
            stats[(c["track"], c["mside"], c["league"])] = c
    rows = []
    for x in ev["league_market_status"]:
        k = (x["track"], x["mside"], x["league"])
        c = stats.get(k) or {}
        sf = c.get("study_final") or {}
        ci = sf.get("ci95") or [None, None]
        bars = x.get("bars")
        st = _stake(policy_cells, *k) if x["status"] != "OFF" else \
            {"state": "OFF", "fraction": None, "rule": "OFF: no bar, no tip"}
        rows.append({
            "track": x["track"], "track_label": TRACK_LABEL.get(x["track"], x["track"]),
            "mside": x["mside"], "market": MSIDE_LABEL.get(x["mside"], x["mside"]), "league": x["league"],
            "status": x["status"],
            "bar": None if not bars else _r(bars[0]),
            "measure": c.get("measure"), "mode": x.get("mode") or c.get("mode"),
            "odds_min": _r(c.get("odds_min"), 2), "odds_max": _r(c.get("odds_max"), 2),
            "basis": x.get("basis"), "off_reason": x.get("reason"),
            "wf_bets": c.get("wf_bets"), "wf_roi": _r(c.get("wf_roi")),
            "study_bets": sf.get("bets"), "study_roi": _r(sf.get("roi")), "study_matchdays": sf.get("matchdays"),
            "study_ci_lo": _r(ci[0]), "study_ci_hi": _r(ci[1]),
            "study_clv": _r(sf.get("clv")), "study_clv_n": sf.get("clv_n"),
            "stake_state": st["state"], "stake_fraction": st["fraction"], "stake_rule": st["rule"],
        })
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("PROVEN", "PROMISING", "UNPROVEN", "OFF")}
    return {"key": src["key"], "proposal_id": pr["proposal_id"], "proposal_hash": pr["proposal_hash"],
            "label": src["label"], "state": src["state"], "state_text": src["state_text"],
            "created_at": pr.get("created_at"), "entries": len(pr.get("entries") or []),
            "source": {"branch": src["branch"], "commit": src["commit"], "proposal": src["proposal"],
                       "evidence": ev_path, "report": src.get("report")},
            "counts": counts, "cells": rows}


def build() -> dict:
    pol = _git_json(POLICY["commit"], POLICY["path"])
    cs = pol.get("cell_staking") or {}
    cells = cs.get("cells") or {}
    plans = [build_one(s, cells) for s in SOURCES]
    return {"schema": SCHEMA,
            "about": "Per-league bars, statistics only (no bookmaker prices). Built by scripts/build_bars_statistics.py "
                     "from the commits named in each proposal's `source`. Rebuild after a new proposal or approval.",
            "policy_source": {**POLICY, "approved": (cs.get("approved") or {}).get("proposal_id")},
            "stake_plan": STAKE_PLAN,
            "not_adopted": [NOT_ADOPTED],
            "evidence_note": "Study ROI / CI: the selection study on own bet-time prices (matchday block bootstrap, 95%). "
                             "Fresh-holdout verdicts are labelled \"re-test after a price-data fix (outcomes previously "
                             "seen)\" - never untouched proof. CLV = mean closing-line value of the study bets.",
            "proposals": plans}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if a.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            print(f"{OUT.relative_to(ROOT)} differs from a rebuild")
            return 1
        print("bars statistics: up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
