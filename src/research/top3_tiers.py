"""TOP-3 v7 - three NAMED tiers per league per gameweek (protocol config/top3_tiers_protocol_v7.json).

RESEARCH / PROPOSAL ONLY. Nothing here is served: the live TOP-3 stays protocol v6 (src.research.top3_ranking
LIVE_RULE R2_CAL_EV) until the owner decides.

Owner 2026-11-16 (relayed by the coordinator): "TOP-3 becomes three named tiers per league per gameweek";
"Show all, stake by proof"; "if we don't have 3 bets for every league, something is wrong" (never pad - diagnose).

PUBLIC API (pure functions, pandas in / pandas out):
    cell_rows(fam, cell, dev_end)       one cell's rows scored by the cell's own source fitted on DEV rows only
    calibrate(rows, lam_stats)          + p_cal, cal_ev (R2 of v6: fair + lam * (p_used - fair))
    pick_tiers(pool, t3_thr)            T1 / T2 / T3 per (track, league, ISO week); distinct; never padded
    fit_t3(dev_pool)                    per-league T3 thresholds on DEV only (grid + MIN_OWN_BETS); no fallback
    shortfall(...)                      cause of every league-week with fewer than 3 tips

Every statistic is per cell (track, league, market-side) or per league. No global constant is fitted; the T3
grid and MIN_OWN_BETS are declared in the protocol before any holdout row is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bars_opt import study as S
from src.research import top3_ranking as R

PROTOCOL_FILE = "config/top3_tiers_protocol_v7.json"
TIERS = ("T1_BEST_VALUE", "T2_SAFEST", "T3_RISKY_VALUE")
T3_ODDS_GRID = (2.25, 2.50, 2.75, 3.00)
T3_MARGIN_GRID = (0.00, 0.03, 0.05, 0.08, 0.10, 0.15)
MIN_T3_DEV_PICKS = S.MIN_OWN_BETS                    # 40, the study's own-bets minimum
STAKE_STATUSES = ("PROVEN", "PROMISING")
CAUSES = ("NO_FIXTURES", "ALL_CELLS_OFF", "NO_MODEL", "NO_PRICE", "NO_CALIBRATION", "NO_POSITIVE_EV",
          "T3_NO_THRESHOLD", "T3_BAND_EMPTY", "RULE_CONFLICT")


def gw(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%G-W%V")


def cell_rows(fam: pd.DataFrame, cell: dict, dev_end: pd.Timestamp) -> pd.DataFrame:
    """Rows of ONE league of a family frame (study.family_frame) with p_used from the cell's own source
    (cell['source']) whose parameters are fitted on the family's rows dated < dev_end ONLY, inside the
    cell's odds range. A row without a probability stays out (missing stays missing)."""
    lg = cell["league"]
    dev = fam[fam["date"] < dev_end]
    f = fam[fam["league"] == lg]
    if f.empty:
        return f.assign(p_used=np.nan).iloc[0:0]
    src = cell["source"]
    params = S.source_params(src, dev) if src != S.LEAGUE_WIDE_SOURCE else {}
    if src == "SHRINK" and lg not in params.get("weights", {}):
        return f.assign(p_used=np.nan).iloc[0:0]
    rng = S.in_odds_range(f, cell.get("odds_max"), cell.get("odds_min"))
    f = f[rng].copy()
    if f.empty:
        return f.assign(p_used=np.nan)
    f["p_used"] = np.asarray(S.source_fn(src, params)(f), dtype=float)
    return f[np.isfinite(f["p_used"].to_numpy(float))]


def calibrate(rows: pd.DataFrame, lam: pd.DataFrame) -> pd.DataFrame:
    """rows + lam (per cell; NaN = no statistic, the row cannot be valued), p_cal and cal_ev (R2 of v6)."""
    b = rows.drop(columns=[c for c in ("lam",) if c in rows.columns]).merge(
        lam[R.CELL + ["lam"]], on=R.CELL, how="left")
    b["p_cal"] = b["fair"] + b["lam"] * (b["p_used"] - b["fair"])
    b["cal_ev"] = b["p_cal"] * b["odds"] - 1.0
    return b


def _direction(mside: str) -> str:
    return "OVER" if mside in S.OVER_DIRECTION else "UNDER"


def _market(mside: str) -> str:
    return S.MARKET_SIDES[mside][0]


def pick_tiers(pool: pd.DataFrame, t3_thr: dict) -> pd.DataFrame:
    """pool: value rows (cal_ev > 0) with track, league, mside, row_id, date, odds, p_cal, cal_ev.
    Per (track, league, ISO week): T1 = highest cal_ev; T2 = highest p_cal of the rest; T3 = highest cal_ev
    of the rest with odds >= the league's odds_min and cal_ev >= its margin (no threshold -> no T3).
    Distinct fixture x market; one direction per fixture. A tier with no candidate stays EMPTY."""
    if pool is None or pool.empty:
        return pd.DataFrame()
    p = pool.copy()
    p["gw"] = gw(p["date"])
    p["market"] = p["mside"].map(_market)
    p["direction"] = p["mside"].map(_direction)
    out = []
    for (tr, lg, w), g in p.groupby(["track", "league", "gw"], sort=True):
        g = g.sort_values(["date", "row_id", "mside"], kind="mergesort")
        taken, used, dirs = [], set(), {}

        def ok(r):
            return (r.row_id, r.market) not in used and dirs.get(r.row_id, r.direction) == r.direction

        def take(r, tier):
            used.add((r.row_id, r.market))
            dirs[r.row_id] = r.direction
            taken.append(r._asdict() | {"tier": tier})

        orders = {"T1_BEST_VALUE": g.sort_values("cal_ev", ascending=False, kind="mergesort"),
                  "T2_SAFEST": g.sort_values("p_cal", ascending=False, kind="mergesort")}
        thr = t3_thr.get(f"{tr}|{lg}")
        if thr:
            band = g[(g["odds"] >= thr["odds_min"]) & (g["cal_ev"] >= thr["margin"])]
            orders["T3_RISKY_VALUE"] = band.sort_values("cal_ev", ascending=False, kind="mergesort")
        for tier in TIERS:
            o = orders.get(tier)
            if o is None:
                continue
            for r in o.itertuples(index=False):
                if ok(r):
                    take(r, tier)
                    break
        out += taken
    return pd.DataFrame(out)


def fit_t3(dev_pool: pd.DataFrame) -> dict:
    """Per (track, league) T3 threshold on DEV league-weeks only: every (odds_min, margin) of the declared grid
    is simulated with T1 / T2 taken first; kept when it has >= MIN_T3_DEV_PICKS T3 picks; the highest dev ROI
    wins (ties: smaller margin, then lower odds). No candidate -> the league has NO T3 threshold (no fallback)."""
    res = {}
    if dev_pool is None or dev_pool.empty:
        return res
    for (tr, lg), g in dev_pool.groupby(["track", "league"]):
        key = f"{tr}|{lg}"
        best, tried = None, []
        for o in T3_ODDS_GRID:
            for m in T3_MARGIN_GRID:
                t = pick_tiers(g, {key: {"odds_min": o, "margin": m}})
                t3 = t[t["tier"] == "T3_RISKY_VALUE"] if len(t) else t
                n = int(len(t3))
                roi = float(t3["pnl"].mean()) if n else None
                tried.append({"odds_min": o, "margin": m, "dev_picks": n, "dev_roi": None if roi is None else round(roi, 4)})
                if n >= MIN_T3_DEV_PICKS and (best is None or roi > best["dev_roi_raw"] + 1e-12):
                    best = {"odds_min": o, "margin": m, "dev_picks": n, "dev_roi_raw": roi}
        if best is None:
            res[key] = {"threshold": None, "reason": f"no grid point with >= {MIN_T3_DEV_PICKS} dev T3 picks",
                        "grid": tried}
        else:
            res[key] = {"threshold": {"odds_min": best["odds_min"], "margin": best["margin"]},
                        "dev_picks": best["dev_picks"], "dev_roi": round(best["dev_roi_raw"], 4), "grid": tried}
    return res


def thresholds_of(fit: dict) -> dict:
    return {k: v["threshold"] for k, v in fit.items() if v.get("threshold")}


def stake_label(status: str | None) -> str:
    return "STAKE" if status in STAKE_STATUSES else "PAPER"


def shortfall(fixtures: pd.DataFrame, slots: pd.DataFrame, picks: pd.DataFrame, cells_on: dict,
              weeks: list[str], t3_thr: dict) -> pd.DataFrame:
    """One row per (track, league, ISO week) of the window: tips (0..3) and, when < 3, the CAUSE.
    fixtures: (track, league, row_id, date, has_form) - every played match of the frame.
    slots: one row per fixture x non-OFF MODEL cell with flags model, price, lam, value (bool).
    cells_on: {(track, league): number of non-OFF MODEL cells}.
    Precedence: NO_FIXTURES > ALL_CELLS_OFF > (the largest blocking count among NO_MODEL / NO_PRICE /
    NO_CALIBRATION / NO_POSITIVE_EV); 3+ value bets but only T3 empty -> T3_NO_THRESHOLD (the league has no
    threshold) or T3_BAND_EMPTY (no value bet at or above the league's odds / margin); otherwise RULE_CONFLICT
    (distinct fixture x market / one direction per fixture)."""
    fx = fixtures.assign(gw=gw(fixtures["date"]))
    sl = slots.assign(gw=gw(slots["date"])) if len(slots) else slots.assign(gw=[])
    pk = picks.groupby(["track", "league", "gw"]).size() if len(picks) else pd.Series(dtype=int)
    t3 = set(map(tuple, picks.loc[picks["tier"] == "T3_RISKY_VALUE", ["track", "league", "gw"]].values))         if len(picks) else set()
    rows = []
    for (tr, lg) in sorted(set(zip(fixtures["track"], fixtures["league"]))):
        for w in weeks:
            f = fx[(fx["track"] == tr) & (fx["league"] == lg) & (fx["gw"] == w)]
            s = sl[(sl["track"] == tr) & (sl["league"] == lg) & (sl["gw"] == w)] if len(sl) else sl
            n = int(pk.get((tr, lg, w), 0))
            r = {"track": tr, "league": lg, "gw": w, "fixtures": int(len(f)), "tips": n,
                 "slots": int(len(s)), "slots_no_model": int((~s["model"]).sum()) if len(s) else 0,
                 "slots_no_price": int((s["model"] & ~s["price"]).sum()) if len(s) else 0,
                 "slots_no_lam": int((s["model"] & s["price"] & ~s["lam"]).sum()) if len(s) else 0,
                 "slots_no_value": int((s["model"] & s["price"] & s["lam"] & ~s["value"]).sum()) if len(s) else 0,
                 "value_slots": int(s["value"].sum()) if len(s) else 0}
            if n >= 3:
                r["cause"] = ""
            elif len(f) == 0:
                r["cause"] = "NO_FIXTURES"
            elif cells_on.get((tr, lg), 0) == 0:
                r["cause"] = "ALL_CELLS_OFF"
            elif r["value_slots"] >= 3:
                if n == 2 and (tr, lg, w) not in t3:
                    r["cause"] = "T3_BAND_EMPTY" if f"{tr}|{lg}" in t3_thr else "T3_NO_THRESHOLD"
                else:
                    r["cause"] = "RULE_CONFLICT"
            else:
                c = {"NO_MODEL": r["slots_no_model"], "NO_PRICE": r["slots_no_price"],
                     "NO_CALIBRATION": r["slots_no_lam"], "NO_POSITIVE_EV": r["slots_no_value"]}
                r["cause"] = max(c, key=lambda k: (c[k], -list(c).index(k)))
            rows.append(r)
    return pd.DataFrame(rows)
