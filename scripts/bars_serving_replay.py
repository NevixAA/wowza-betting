"""Replay a bars proposal on the live board, AS IF the owner had approved every entry - in memory only.

    python scripts/bars_serving_replay.py --proposal output/threshold_proposals/2026-11_<hash>.json \
        --candidates output/league_weekly_candidates.csv --results output/v91_shadow.csv --out <json>

Nothing is applied and no production file is read for the bars: the proposed cells are taken from the proposal
file and scored with src/bars_opt/serve.py (the function production calls). Rows are the live first-sight
candidates (one per fixture x market; O/U 2.5 rows are expanded to BOTH sides from the row's real two-sided
pair). The fair price is the power de-vig of that pair; a row without one gets no tip.

CAVEAT, printed in the output: the live candidates come from the engines serving when they were logged (the
PRE-RETRAIN models until go-live), while the cells were fitted on the clean retrained models' predictions. The
replay therefore shows how the cells filter the board (volume per league x market, the TOP-3 mechanics), not
how the clean models will perform.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bars_opt import serve as BS  # noqa: E402
from src.bars_opt import study as S  # noqa: E402

#: panel market x side -> (production market, cell side, study market-side)
SIDE_MAP = {("OU25", "OVER"): ("ou25", "over", "over25_over"), ("OU25", "UNDER"): ("ou25", "under", "over25_under"),
            ("BTTS", "YES"): ("btts", "yes", "btts_yes"), ("OVER15", "OVER"): ("over15", "yes", "over15_over"),
            ("OVER35", "OVER"): ("over35", "yes", "over35_over")}
LINE = {"OU25": 2.5, "OVER15": 1.5, "OVER35": 3.5}


def proposed_cells(proposal: dict) -> dict:
    """{(market, league, side): cell} of every entry's PROPOSED bars_opt cells."""
    out = {}
    for e in proposal.get("entries", []):
        cells = ((e.get("proposed") or {}).get("bars_opt") or {})
        for side, c in cells.items():
            if not BS.validate_cell(c):
                out[(e["market"], e["league"], side)] = c
    return out


def expand(c: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture x market x side with p_side / odds / opposite odds of that side."""
    rows = []
    for r in c.to_dict("records"):
        mk, sd = str(r["market"]), str(r["side"]).upper()
        p, o, oo = r.get("p_model_raw"), r.get("odds"), r.get("opposite_side_odds")
        rows.append(dict(r, panel_side=sd, p_side=p, odds_side=o, odds_other=oo))
        if mk == "OU25":
            other = "UNDER" if sd == "OVER" else "OVER"
            rows.append(dict(r, panel_side=other, p_side=(1 - float(p)) if pd.notna(p) else np.nan,
                             odds_side=oo, odds_other=o))
    d = pd.DataFrame(rows)
    d = d[[(m, s) in SIDE_MAP for m, s in zip(d["market"], d["panel_side"])]].copy()
    d["fair"] = [BS.fair_yes(a, b) for a, b in zip(d["odds_side"], d["odds_other"])]
    return d


def outcome(r, res: dict):
    g = res.get(r["fixture_id"])
    if g is None:
        return np.nan
    hg, ag = g
    if r["market"] == "BTTS":
        return float(hg > 0 and ag > 0)
    over = float(hg + ag > LINE[r["market"]])
    return over if r["panel_side"] == "OVER" else 1.0 - over


def run(proposal_path: Path, cand_path: Path, results_path: Path, days: int, gws: int) -> dict:
    proposal = json.loads(Path(proposal_path).read_text(encoding="utf-8"))
    cells = proposed_cells(proposal)
    cand = pd.read_csv(cand_path, low_memory=False)
    v = pd.read_csv(results_path, low_memory=False)
    res = {r["fixture_id"]: (float(r["home_goals"]), float(r["away_goals"]))
           for r in v[["fixture_id", "home_goals", "away_goals"]].dropna().drop_duplicates("fixture_id").to_dict("records")}
    d = expand(cand)
    d["date"] = pd.to_datetime(d["match_date"]).dt.normalize()
    asof = pd.to_datetime(cand["first_seen_at"], utc=True).max()
    tiers, evs, reasons, msides = [], [], [], []
    for r in d.to_dict("records"):
        mk, side, mside = SIDE_MAP[(r["market"], r["panel_side"])]
        msides.append(mside)
        cell = cells.get((mk, r["league"], side))
        if cell is None:
            tiers.append("")
            evs.append(np.nan)
            reasons.append("NO_PROPOSED_CELL")
            continue
        sc = BS.score(pd.DataFrame([{"league": r["league"], "p_side": r["p_side"], "fair": r["fair"],
                                     "odds": r["odds_side"]}]), cell)
        tiers.append(str(sc["tier"].iloc[0] or ""))
        evs.append(sc["ev_est"].iloc[0])
        reasons.append(str(sc["reason"].iloc[0] or ""))
    d["tier"], d["ev_est"], d["reason"], d["mside"] = tiers, pd.to_numeric(pd.Series(evs), errors="coerce").to_numpy(), reasons, msides
    d["y"] = [outcome(r, res) for r in d.to_dict("records")]
    d["won"] = d["y"] == 1
    d["pnl"] = np.where(d["y"].notna(), np.where(d["won"], d["odds_side"].astype(float) - 1.0, -1.0), np.nan)
    d["track"] = d["model_type"]
    d["row_id"] = d["match_date"].astype(str) + "|" + d["league"] + "|" + d["home_team"] + "|" + d["away_team"]

    tipped = d[d["tier"] != ""]
    win = d[d["date"] >= (asof.tz_localize(None).normalize() - pd.Timedelta(days=days))]
    board = d[pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce") > asof]

    def table(x):
        g = x.groupby(["track", "league", "mside"])
        t = g.size().rename("rows").to_frame()
        t["tips"] = g["tier"].apply(lambda s: int((s != "").sum()))
        return t.reset_index().to_dict("records")

    settled = tipped[tipped["pnl"].notna()].copy()
    weeks = sorted(settled["date"].dt.strftime("%G-W%V").unique())[-gws:]
    t3 = S.top3(settled[settled["date"].dt.strftime("%G-W%V").isin(weeks)].assign(band="TOP_M", tier2_ok=False),
                "TOP_ONLY_V")
    top3 = {"weeks": weeks, "picks": int(len(t3))}
    if len(t3):
        top3 |= {"roi": round(float(t3["pnl"].mean()), 4), "won": int(t3["won"].sum()),
                 "by_week": {w: {"picks": int(len(g)), "roi": round(float(g["pnl"].mean()), 4)}
                             for w, g in t3.groupby(t3["date"].dt.strftime("%G-W%V"))},
                 "list": t3[["date", "track", "league", "home_team", "away_team", "mside", "odds_side", "tier",
                             "ev_est", "won", "pnl"]].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
                 .to_dict("records")}
    return {"caveat": __doc__.split("CAVEAT, printed in the output: ", 1)[1].strip(),
            "proposal_id": proposal.get("proposal_id"), "asof": str(asof), "cells": len(cells),
            "candidate_rows": int(len(d)), "engines": cand["prediction_engine_version"].value_counts().to_dict(),
            "last_days": days, "window_rows": int(len(win)), "window_tips": int((win["tier"] != "").sum()),
            "window_tips_settled": int(((win["tier"] != "") & win["pnl"].notna()).sum()),
            "window_roi_all_tips": (round(float(win.loc[(win["tier"] != "") & win["pnl"].notna(), "pnl"].mean()), 4)
                                    if ((win["tier"] != "") & win["pnl"].notna()).any() else None),
            "reasons": d["reason"].replace("", "TIPPED").value_counts().to_dict(),
            "window_by_league_market": table(win),
            "board_rows": int(len(board)), "board_tips": int((board["tier"] != "").sum()),
            "board_by_league_market": table(board) if len(board) else [],
            "top3_last_gws": top3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--gws", type=int, default=4)
    a = ap.parse_args(argv)
    out = run(Path(a.proposal), Path(a.candidates), Path(a.results), a.days, a.gws)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("window_by_league_market", "board_by_league_market")},
                     indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
