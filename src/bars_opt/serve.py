"""Serving of the OPTIMIZED per-league bars (owner 2026-11-09: "Yes, build the adapter").

    cells, why = entry_cells(lp, market)                  # {side: cell} of a params entry
    cell, why  = served_cell(lp, market, league, side)    # the owner-applied cell or (None, reason)
    res        = score(frame, cell)                       # p_used / measure / in_range / tier / ev_est / reason

WHAT A CELL IS. One league x market x side, chosen by the pre-registered bars study (src/bars_opt/study.py,
reports/P4_BARS_OPTIMIZATION.md) and written by the monthly job (src/bars_opt/cells.py) as a proposal. The owner
approves it and src/threshold_apply writes it into the production params entry under "bars_opt":

    models/best_params_standard.json      {league: {"bars_opt": {"over": cell, "under": cell}}}     (O/U 2.5)
    models/best_params_side_markets.json  {market: {league: {"bars_opt": {"yes": cell}}}}          (btts/o1.5/o3.5)

A cell carries everything the decision needs and nothing else is read: the probability source and its FITTED
parameters (source_params), the edge measure (EDGE = p_used - fair, EV = p_used x odds - 1, or ALL for a
LEAGUE_WIDE cell), the league's own odds range (odds_min / odds_max, None = no bound chosen), the mode
(MODEL or LEAGUE_WIDE) and the league's own VALUABLE <= MARKSMAN <= SNIPER bars.

ONE IMPLEMENTATION. score() calls the study's own functions (source_fn, measure, in_odds_range, apply_bars),
imported, never re-implemented, so a served tier is the tier the study would give the same row
(tests/test_bars_serving.py proves it on real rows).

NO GLOBAL VALUE (owner rule 6). A league x market x side without its own owner-applied cell has no tier
(NO_LEAGUE_BAR). The fair price is the power de-vig of a real two-sided pair (src.market_edge.power_devig);
without one there is no fair price and no tier. Nothing here reads a config constant.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.bars_opt import study as S

CELL_KEY = "bars_opt"
#: production market x side -> study market-side
MSIDE = {("ou25", "over"): "over25_over", ("ou25", "under"): "over25_under", ("btts", "yes"): "btts_yes",
         ("over15", "yes"): "over15_over", ("over35", "yes"): "over35_over"}
SIDES = {"ou25": ("over", "under"), "btts": ("yes",), "over15": ("yes",), "over35": ("yes",)}
NO_LEAGUE_BAR = "NO_LEAGUE_BAR"
R_RANGE = "OUTSIDE_ODDS_RANGE"
R_NO_VALUE = "NO_SOURCE_VALUE"
R_BELOW = "BELOW_LEAGUE_BAR"
R_NO_FAIR = "NO_TWO_SIDED_PRICE"
#: tier label of a LEAGUE_WIDE bet: the contract knows SNIPER / MARKSMAN / VALUABLE only, so a league-wide bet
#: is served as the LOWEST tier and carries bar_mode = LEAGUE_WIDE beside it (listed for the owner)
LEAGUE_WIDE_TIER = "VALUABLE"
REQUIRED = ("mside", "league", "config", "source", "measure", "mode", "valuable_th", "marksman_th", "sniper_th",
            "source_params", "verdict")


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def validate_cell(cell) -> str:
    """"" when the cell is servable, else the reason."""
    if not isinstance(cell, dict):
        return "cell is not an object"
    miss = [k for k in REQUIRED if k not in cell]
    if miss:
        return "cell lacks " + ", ".join(miss)
    if cell["mode"] not in (S.MODE_MODEL, S.MODE_LW):
        return f"unknown mode {cell['mode']!r}"
    if cell["source"] not in S.SOURCES + (S.LEAGUE_WIDE_SOURCE,):
        return f"unknown source {cell['source']!r}"
    if cell["measure"] not in S.MEASURES + (S.LEAGUE_WIDE_MEASURE,):
        return f"unknown measure {cell['measure']!r}"
    if (cell["mode"] == S.MODE_LW) != (cell["source"] == S.LEAGUE_WIDE_SOURCE):
        return "mode and source disagree"
    v, m, s = _num(cell["valuable_th"]), _num(cell["marksman_th"]), _num(cell["sniper_th"])
    if None in (v, m, s):
        return "bars not finite"
    if not (v <= m <= s):
        return "bars not ordered (valuable<=marksman<=sniper)"
    for k in ("odds_min", "odds_max"):
        if cell.get(k) is not None and _num(cell[k]) is None:
            return f"{k} not finite"
    sp = cell["source_params"]
    need = {"BLEND": ("coef", "intercept"), "SHRINK": ("weights",), "LEARNED": ("leagues", "coef", "intercept")}
    for k in need.get(cell["source"], ()):
        if not isinstance(sp, dict) or k not in sp:
            return f"source_params lacks {k}"
    if cell["source"] == "SHRINK" and cell["league"] not in (sp.get("weights") or {}):
        return "no SHRINK weight for the league"
    return ""


def entry_cells(lp, market: str) -> tuple[dict, dict]:
    """({side: cell} valid cells, {side: reason} invalid ones) of a params entry."""
    good, bad = {}, {}
    cells = (lp or {}).get(CELL_KEY) if isinstance(lp, dict) else None
    if not isinstance(cells, dict):
        return good, bad
    for side in SIDES.get(market, ()):
        if side not in cells:
            continue
        why = validate_cell(cells[side])
        if not why and cells[side].get("mside") != MSIDE[(market, side)]:
            why = f"cell market-side {cells[side].get('mside')!r} is not {MSIDE[(market, side)]}"
        (bad.__setitem__(side, why) if why else good.__setitem__(side, cells[side]))
    return good, bad


def served_cell(lp, market: str, league: str, side: str, applied: frozenset | None = None):
    """(cell, "") when `league` x `market` x `side` has its own OWNER-APPLIED cell, else (None, reason)."""
    from src.side_thresholds import applied_entries
    if not isinstance(lp, dict) or not lp:
        return None, f"{NO_LEAGUE_BAR}: no {market} params entry for {league}"
    if lp.get("drop") is True:
        return None, f"{NO_LEAGUE_BAR}: {market} switched OFF for {league} (drop)"
    good, bad = entry_cells(lp, market)
    if side in bad:
        return None, f"{NO_LEAGUE_BAR}: {market} {side} {league}: {bad[side]}"
    if side not in good:
        return None, f"{NO_LEAGUE_BAR}: {market} {side} {league} has no bars_opt cell"
    if good[side]["league"] != league:
        return None, f"{NO_LEAGUE_BAR}: {market} {side} cell is for {good[side]['league']}, not {league}"
    ids = applied if applied is not None else applied_entries()
    if f"{market}|{league}" not in ids:
        return None, f"{NO_LEAGUE_BAR}: {market}|{league} not applied by src/threshold_apply (owner)"
    return good[side], ""


def bars_of(cell: dict) -> dict:
    """The study's bars dict for apply_bars (one league)."""
    return {"valuable_th": float(cell["valuable_th"]), "marksman_th": float(cell["marksman_th"]),
            "sniper_th": float(cell["sniper_th"]), "tier2": bool(cell.get("tier2") or False),
            "wf_roi_shrunk": _num(cell.get("wf_roi_shrunk"))}


def score(frame: pd.DataFrame, cell: dict) -> pd.DataFrame:
    """frame: rows of ONE league x market-side with p_side (the model probability of the cell's side), fair
    (the de-vigged fair probability of that side from a real two-sided pair) and odds (the bet-time price of
    that side). Returns a frame on the same index: p_used, measure, in_range, tier ("" = no tip), ev_est,
    bar_mode, reason ("" = tipped). The decision is the study's (apply_bars), not a re-implementation."""
    cols = ["p_used", "measure", "in_range", "tier", "ev_est", "bar_mode", "reason"]
    out = pd.DataFrame(index=frame.index, columns=cols)
    if frame.empty:
        return out
    e = pd.DataFrame({"league": frame["league"].astype(str).to_numpy(),
                      "p_side": pd.to_numeric(frame["p_side"], errors="coerce").to_numpy(float),
                      "fair": pd.to_numeric(frame["fair"], errors="coerce").to_numpy(float),
                      "odds": pd.to_numeric(frame["odds"], errors="coerce").to_numpy(float)}, index=frame.index)
    out["bar_mode"] = cell["mode"]
    out["tier"] = ""
    out["in_range"] = False
    priced = np.isfinite(e["fair"].to_numpy()) & np.isfinite(e["odds"].to_numpy()) & (e["odds"].to_numpy() > 1) \
        & np.isfinite(e["p_side"].to_numpy())
    out.loc[~priced, "reason"] = R_NO_FAIR
    e = e[priced]
    if e.empty:
        return out
    rng = S.in_odds_range(e, cell.get("odds_max"), cell.get("odds_min"))
    out.loc[e.index, "in_range"] = rng
    out.loc[e.index[~rng], "reason"] = R_RANGE
    ee = e[rng].copy()
    if ee.empty:
        return out
    fn = S.source_fn(cell["source"], cell["source_params"])
    ee["p_used"] = fn(ee)
    ee["edge"] = S.measure(ee, ee["p_used"].to_numpy(float), cell["measure"])
    out.loc[ee.index, "p_used"] = ee["p_used"].to_numpy()
    out.loc[ee.index, "measure"] = ee["edge"].to_numpy()
    nov = np.isnan(ee["edge"].to_numpy(float))
    out.loc[ee.index[nov], "reason"] = R_NO_VALUE
    ee = ee[~nov]
    kept = S.apply_bars(ee, {cell["league"]: bars_of(cell)}, cell["mode"])
    out.loc[ee.index, "reason"] = R_BELOW
    if len(kept):
        tiers = kept["tier"].where(kept["tier"] != S.MODE_LW, LEAGUE_WIDE_TIER)
        out.loc[kept.index, "tier"] = tiers.to_numpy()
        out.loc[kept.index, "ev_est"] = kept["ev_est"].to_numpy()
        out.loc[kept.index, "reason"] = ""
    return out


def fair_yes(yes_odds, no_odds) -> float:
    """The de-vigged fair probability of the yes / over side of a REAL two-sided pair (power method,
    src.market_edge.power_devig, the study's fair price); NaN when the pair is missing or not a price."""
    from src.market_edge import power_devig
    a, b = _num(yes_odds), _num(no_odds)
    if a is None or b is None:
        return float("nan")
    f = power_devig(a, b)
    return float("nan") if f is None else float(f)


def ev_point(p_used, odds) -> float:
    """p_used x odds - 1 (NaN when either is missing): the expected value per unit of a served bet."""
    p, o = _num(p_used), _num(odds)
    return float("nan") if p is None or o is None else p * o - 1.0
