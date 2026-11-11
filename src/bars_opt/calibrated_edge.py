"""BARS ON THE CALIBRATED EDGE - pre-registered study, NOT RUN (protocol config/bars_calibrated_edge_protocol_v2.json).

Owner 2026-11-11: "Bars on calibrated edge" at the DECEMBER re-optimization, "No cap". Each cell's bars are chosen
on the CALIBRATED edge instead of the served edge, with the approved study's machinery (src/bars_opt/study.py:
nested inner validation, per-league bars by the src.league_bars.choose_bars rule on CAL_GRID (owner 2026-11-11
"Start at 1 point": 0.01 then the study GRID's own steps 0.02 .. 0.30), MODEL vs LEAGUE_WIDE switch,
TIER2, Holm over every configuration searched). The output is a PROPOSAL for the owner (src/threshold_proposals);
nothing is served until the owner approves and src/threshold_apply applies it.

THE MEASURE. CAL_EDGE = lam * (p_used - fair), p_used = the configuration's probability source, fair = the
power de-vig of the real two-sided pair, lam = the cell's calibration slope: least squares of (won - fair) on
(p_used - fair), clipped to [0, 1], on >= MIN_OWN_BETS rows.
  * inside the nested selection (stage A) lam is the league's OWN slope on the inner training block's rows of
    that family (scored by the configuration's source fitted on the same block); a league without it has no
    calibrated edge in that configuration (no bet, nothing filled in);
  * for the frozen bars and for serving (stage B) lam is recomputed exactly as the TOP-3 R2/R3 statistic
    (src/research/top3_ranking.cell_stats: own slope shrunk toward the SAME league's other market-sides,
    DerSimonian-Laird inside the league, clipped to [0, 1]) on every selection row of the chosen cells, the
    bars are refitted on that CAL_EDGE and lam is stored in the cell (cal_lam, cal_rows, cal_record_sha256).

CONFIGURATIONS per family: 4 sources x CAL_EDGE x 6 odds ranges = 24 MODEL + 6 LEAGUE_WIDE (measure ALL,
unchanged) = 30. League-wide cells stay the owner's own list outside the TOP-3 (owner "Own list").

This module is the SKELETON: the pure pieces below are implemented and tested; run() refuses until the
protocol is committed and unchanged, the flag config/bars_calibrated_edge.json is on with the owner's 'run',
the month is the pre-registered first month (2026-12) or later, and the U3e calibration record reproduces.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import league_bars as LB
from src.bars_opt import study as S

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_FILE = Path("config") / "bars_calibrated_edge_protocol_v2.json"
FLAG_FILE = Path("config") / "bars_calibrated_edge.json"
CAL_MEASURE = "CAL_EDGE"
FIRST_MONTH = "2026-12"
MIN_OWN_ROWS = S.MIN_OWN_BETS            # 40, the study's own-data minimum
#: owner 2026-11-11 "Start at 1 point": the bar grid of the calibrated edge starts at 1 calibrated point, then the
#: approved study's own grid steps above it (src.league_bars.GRID = 0.02 .. 0.30 by 0.01) -> 30 points
CAL_GRID = np.round(np.concatenate([[0.01], np.asarray(LB.GRID, float)]), 3)


def configs() -> list[tuple]:
    """The pre-registered configuration space: every source x CAL_EDGE x odds range, plus the league-wide ones."""
    model = [(s, CAL_MEASURE, hi, lo) for s in S.SOURCES for hi in S.MAX_ODDS for lo in S.MIN_ODDS]
    lw = [(S.LEAGUE_WIDE_SOURCE, S.LEAGUE_WIDE_MEASURE, hi, lo) for hi in S.MAX_ODDS for lo in S.MIN_ODDS]
    return model + lw


def own_lambda(rows: pd.DataFrame) -> float:
    """Stage A: the league's own calibration slope on rows (fair, p_used, won), clipped to [0, 1];
    NaN with fewer than MIN_OWN_ROWS rows or no spread (no calibrated edge - never filled)."""
    if rows is None or len(rows) < MIN_OWN_ROWS:
        return float("nan")
    dd = (rows["p_used"] - rows["fair"]).to_numpy(float)
    r = rows["won"].to_numpy(float) - rows["fair"].to_numpy(float)
    ok = np.isfinite(dd) & np.isfinite(r)
    dd, r = dd[ok], r[ok]
    sxx = float((dd ** 2).sum())
    if len(dd) < MIN_OWN_ROWS or sxx <= 0:
        return float("nan")
    return float(np.clip((dd * r).sum() / sxx, 0.0, 1.0))


def league_lambdas(rows: pd.DataFrame) -> dict:
    """Stage B: {(track, league, mside): lam} - each cell's slope shrunk toward the SAME league's other
    market-sides (study.shrink_weights with the market-side as the group inside one league), the estimator of
    the TOP-3 R2/R3 statistic. Cells with < MIN_OWN_ROWS rows have none."""
    out = {}
    cnt = rows.groupby(["track", "league", "mside"]).size()
    for (tr, lg), g in rows.groupby(["track", "league"]):
        big = [m for m in g["mside"].unique() if cnt.get((tr, lg, m), 0) >= MIN_OWN_ROWS]
        own = g[g["mside"].isin(big)]
        if own.empty:
            continue
        for m, v in S.shrink_weights(own.assign(league=own["mside"], p_side=own["p_used"])).items():
            out[(tr, lg, m)] = float(v)
    return out


def cal_edge(p_used, fair, lam) -> np.ndarray:
    """CAL_EDGE = lam * (p_used - fair); NaN where any input is missing (never filled)."""
    p, f, lm = (np.asarray(x, float) for x in (p_used, fair, lam))
    return lm * (p - f)


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gate(month: str, root: Path | None = None) -> list[str]:
    """Why the study may NOT run now ([] = it may). Fail-closed."""
    r = Path(root) if root else ROOT
    why = []
    flag = {}
    try:
        flag = json.loads((r / FLAG_FILE).read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        why.append(f"{FLAG_FILE} missing or unreadable")
    pf = r / PROTOCOL_FILE
    if not pf.is_file():
        why.append(f"{PROTOCOL_FILE} missing")
    elif sha256_lf(pf) != flag.get("protocol_sha256"):
        why.append("protocol file is not the pinned version")
    if flag.get("run_enabled") is not True or not str(flag.get("owner_ok") or "").strip():
        why.append("not enabled: run_enabled true and the owner's 'run' (owner_ok) are both required")
    if not month or month < FIRST_MONTH:
        why.append(f"month {month!r} is before the pre-registered first month {FIRST_MONTH}")
    rec = r / str(flag.get("u3e_ledger") or "")
    if not flag.get("u3e_ledger") or not rec.is_file():
        why.append("the U3e calibration ledger (TOP-3 statistics freeze) is missing")
    return why


def run(month: str, root: Path | None = None) -> dict:
    """The study. Refuses (SystemExit) while gate() objects. The body is implemented to this protocol's
    specification and dry-run before the first run (monthly checklist M1); the pre-registered pieces above
    (configs, own_lambda, league_lambdas, cal_edge) never change, and the implementation's code sha256 goes into
    the run record."""
    why = gate(month, root)
    if why:
        raise SystemExit("bars on the calibrated edge: NOT RUN\n  " + "\n  ".join(why))
    raise NotImplementedError("pre-registered skeleton: implement + dry-run per runbook M1 before 2026-12-01")
