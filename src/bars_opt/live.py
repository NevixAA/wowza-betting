"""Live rows and the monthly proposal job for the bars study (pre-registration 10.3).

LIVE ROWS. From go-live on, each month's evidence is the live FIRST-SIGHT candidates of the CLEAN engines:
output/league_weekly_candidates.csv (append-only first sight, research/league_weekly.py), one row per
fixture x market with the row's side, p_model_raw FOR THAT SIDE, the bet-time price of that side and the
opposite side's price (a real two-sided pair from the same source). Results come from output/v91_shadow.csv
(home_goals / away_goals per fixture_id). Only rows whose prediction_engine_version is listed in
config/bars_opt.json "clean_engine_versions" count (owner R5: recount only on the clean models); the list is
empty until the retrained models serve, so no live row counts before then. A row without both prices, a
result or a probability is excluded and counted - never filled. There is no closing pair in this source, so
CLV for live rows is missing (counted).

THE MONTHLY JOB. Every family is re-selected on all rows before the month (study.select_core), and the
per-league bars are written as PROPOSALS: the full evidence file plus production-shape candidates for the
markets whose served file has one bar set per league (btts, over15, over35: the Over/Yes side). O/U 2.5 has
separate Over and Under bars in this study while production serves one set per league, so O/U 2.5 is written
to the evidence file only and listed as needing a serving decision. src.threshold_proposals then builds the
month's proposal; nothing is applied without the owner (src/threshold_apply).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.bars_opt import study as S

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "bars_opt.json"
BASE_UNIVERSE_DIR = ROOT / "output" / "bars_opt"
LIVE_CANDIDATES = ROOT / "output" / "league_weekly_candidates.csv"
RESULTS = ROOT / "output" / "v91_shadow.csv"
MARKET_MAP = {"OU25": ("over25", 2.5), "BTTS": ("btts", None), "OVER15": ("over15", 1.5), "OVER35": ("over35", 3.5)}
YES_SIDES = {"OVER", "YES"}
PRODUCTION_SHAPE = {"btts_yes": "btts", "over15_over": "over15", "over35_over": "over35"}
#: the only configuration production's edge bar reads faithfully (edge = p_model_raw - fair, no odds range)
FAITHFUL_CONFIG = "RAW|EDGE|max-|min-"


def clean_engine_versions(path: Path = CONFIG) -> list[str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    v = cfg["clean_engine_versions"]
    if not isinstance(v, list):
        raise ValueError("config/bars_opt.json clean_engine_versions must be a list")
    return v


def live_rows(candidates: pd.DataFrame, results: pd.DataFrame, engines: list[str]) -> tuple[pd.DataFrame, dict]:
    """Universe-shaped rows (track, market, league, row_id, date, y, p, yes, no, yes_c, no_c) from the live
    first-sight candidates of the listed clean engines. Returns (rows, exclusion counts)."""
    cols = ["track", "market", "league", "row_id", "date", "y", "p", "yes", "no", "yes_c", "no_c", "stage"]
    counts = {"candidates": int(len(candidates))}
    c = candidates[candidates["prediction_engine_version"].isin(engines)].copy()
    counts["clean_engine"] = int(len(c))
    c = c[c["market"].isin(list(MARKET_MAP))]
    two = c["odds"].notna() & c["opposite_side_odds"].notna() & c["p_model_raw"].notna()
    counts["no_two_sided_price_or_p"] = int((~two).sum())
    c = c[two].copy()
    g = results[["fixture_id", "home_goals", "away_goals"]].dropna().drop_duplicates("fixture_id")
    c = c.merge(g, on="fixture_id", how="left")
    counts["no_result"] = int(c["home_goals"].isna().sum())
    c = c[c["home_goals"].notna()].copy()
    if c.empty:
        return pd.DataFrame(columns=cols), counts
    hg, ag = c["home_goals"].astype(float), c["away_goals"].astype(float)
    yes_side = c["side"].str.upper().isin(YES_SIDES).to_numpy()
    out = pd.DataFrame({
        "track": c["model_type"].to_numpy(),
        "market": c["market"].map(lambda m: MARKET_MAP[m][0]).to_numpy(),
        "league": c["league"].to_numpy(),
        "row_id": (c["match_date"].astype(str) + "|" + c["league"] + "|" + c["home_team"] + "|" + c["away_team"]).to_numpy(),
        "date": pd.to_datetime(c["match_date"]).to_numpy(),
        "p": np.where(yes_side, c["p_model_raw"].astype(float), 1 - c["p_model_raw"].astype(float)),
        "yes": np.where(yes_side, c["odds"].astype(float), c["opposite_side_odds"].astype(float)),
        "no": np.where(yes_side, c["opposite_side_odds"].astype(float), c["odds"].astype(float)),
        "yes_c": np.nan, "no_c": np.nan, "stage": "live"})
    tot = (hg + ag).to_numpy()
    line = c["market"].map(lambda m: MARKET_MAP[m][1]).to_numpy()
    is_btts = (c["market"] == "BTTS").to_numpy()
    y_tot = np.array([float(t > ln) if ln is not None else np.nan for t, ln in zip(tot, line)])
    out["y"] = np.where(is_btts, ((hg > 0) & (ag > 0)).astype(float).to_numpy(), y_tot)
    return out[cols], counts


def build_universe_dir(month: str, work: Path, base_dir: Path = BASE_UNIVERSE_DIR) -> dict:
    """base universe (the clean retrain's priced rows, committed) + live rows before the month -> work/."""
    work.mkdir(parents=True, exist_ok=True)
    engines = clean_engine_versions()
    cand = pd.read_csv(LIVE_CANDIDATES, low_memory=False) if LIVE_CANDIDATES.exists() else pd.DataFrame()
    res = pd.read_csv(RESULTS, low_memory=False) if RESULTS.exists() else pd.DataFrame()
    live, counts = (live_rows(cand, res, engines) if len(cand) and len(res) else (pd.DataFrame(), {"candidates": 0}))
    if len(live):
        live = live[live["date"] < pd.Timestamp(f"{month}-01")]
    stats = {"engines": engines, "live": counts, "live_rows_used": int(len(live))}
    for track in ("standard", "new_format"):
        base = pd.read_parquet(base_dir / f"universe_{track}.parquet")
        add = live[live["track"] == track] if len(live) else pd.DataFrame()
        u = pd.concat([base, add], ignore_index=True) if len(add) else base
        u.to_parquet(work / f"universe_{track}.parquet", index=False)
        stats[track] = {"base_rows": int(len(base)), "live_rows": int(len(add))}
    return stats


def production_candidates(entries: list[dict], current: dict) -> tuple[dict, list[dict]]:
    """({market: {league: params}} for btts / over15 / over35 (Over/Yes side), [OFF leagues]).

    Serving (src/side_thresholds) uses a league's own bars only when `approved` is true and serves the GLOBAL
    bar otherwise - also for a dropped or absent league. So, owner rule "no global vars":
      * an ON league carries its own bars with approved = true (the study's own-evidence choice; the
        approved_basis says so) - the owner still approves the proposal entry before anything serves;
      * an OFF league is NOT written as drop/unapproved (that would push it onto the global bar or revoke
        an existing approval): its CURRENT production entry is carried unchanged, and it is returned in the
        OFF list, which needs the serving change "OFF = no tip" (the owner decides)."""
    out: dict = {}
    off: list[dict] = []
    for e in entries:
        m = PRODUCTION_SHAPE.get(e["mside"])
        if not m or not e.get("league"):
            continue
        reason = e.get("off")
        if reason is None and (e.get("mode") != "MODEL" or e.get("config") != FAITHFUL_CONFIG):
            # production computes edge = p_model_raw - fair with no odds range and bets on the edge bar; any
            # other configuration (EV bars, BLEND / SHRINK / LEARNED probability, odds caps, LEAGUE_WIDE)
            # would be MISREAD by it - so it is never written in the production shape
            reason = "NEEDS_BARS_OPT_SERVING"
        if reason is not None:
            off.append({"market": m, "league": e["league"], "reason": reason, "config": e.get("config"),
                        "mode": e.get("mode")})
            cur = (current.get(m) or {}).get(e["league"])
            if cur is not None:
                out.setdefault(m, {})[e["league"]] = cur
            continue
        out.setdefault(m, {})[e["league"]] = {
            "valuable_th": e["valuable_th"], "marksman_th": e["marksman_th"], "sniper_th": e["sniper_th"],
            "roi": e["wf_roi"], "bets": e["wf_bets"], "drop": False, "approved": True,
            "approved_basis": "bars_opt: own-evidence bars chosen by the pre-registered nested protocol on rows "
                              "before the month (reports/P4_BARS_OPTIMIZATION.md); the owner approves the entry",
            "mode": e["mode"], "tier2": e.get("tier2"), "config": e["config"],
            "source": "bars_opt", "month": e["month"]}
    return out, off


def run_month(month: str, work: Path, proposal_dir: Path | None = None) -> dict:
    from src import threshold_proposals as TP
    stats = build_universe_dir(month, work / "universe")
    res = S.monthly_proposal(work / "universe", month, work / "evidence")
    entries = json.loads(Path(res["path"]).read_text(encoding="utf-8"))
    cand_dir = TP.CANDIDATE_DIR / "bars_opt" / month
    cands, off = production_candidates(entries, TP.production_params())
    for m, data in cands.items():
        TP.write_candidate(m, data, cand_dir)
    ev_dir = (proposal_dir or TP.PROPOSAL_DIR) / "bars_opt"
    ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / f"{month}.json").write_text(json.dumps({"month": month, "data": stats, "entries": entries,
                                                      "off_leagues_need_serving_change": off,
                                                      "needs_owner_decision": [
                                                          "O/U 2.5: separate Over and Under bars per league; "
                                                          "production serves one set per league",
                                                          "OFF = no tip: serving today falls back to the global "
                                                          "bar for any league without approved own bars"]},
                                                     indent=1, default=str), encoding="utf-8")
    p = TP.build_proposal(TP.load_candidates(cand_dir), month=month,
                          source={"workflow": "bars_monthly", "candidates_dir": str(cand_dir)})
    path = TP.write_proposal(p, proposal_dir)
    return {"month": month, "data": stats, "entries": len(entries), "proposal": str(path)}
