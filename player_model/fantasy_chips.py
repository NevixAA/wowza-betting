"""Chips priced in points, not in adjectives.

    from player_model.fantasy_chips import chip_values
    v = chip_values(proj_df, squad=my_squad_df)

WHAT THE OLD ADVISOR DID. It counted how many teams had a double gameweek and how many had a
blank, then said "Bench Boost / Triple Captain window" when the count passed four. That is a
calendar observation, not advice: it is the same recommendation for a manager whose bench is
four £4.0m non-playing defenders as for one whose bench is stacked, and those are opposite
situations.

WHAT A CHIP IS ACTUALLY WORTH. Every chip has an incremental value that can be written down:

    Triple Captain   3x instead of 2x, so the gain is exactly ONE MORE captain.
                     Worth most where the captain plays twice.
    Bench Boost      the points your bench would otherwise not score. That is a property of
                     YOUR bench and cannot be estimated from the fixture list at all.
    Free Hit         best legal one-gameweek squad, minus what your current squad would score.
    Wildcard         same comparison over a multi-gameweek horizon, since the squad persists.

WHAT IT REFUSES TO GUESS. Three of the four need your actual squad. Without one this returns
`available: false` and says which input is missing, rather than producing a number that looks
like advice. A chip is used once a season; a confidently wrong recommendation is expensive.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from player_model.fantasy import fpl_ids

log = logging.getLogger(__name__)


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return (pd.to_numeric(df[c], errors="coerce") if c in df.columns
            else pd.Series(np.nan, index=df.index))


# Unknown points stay NaN (never 0, never P(start)=1, never "one notional fixture"); every
# consumer below skips or reports NaN players instead of valuing them.
def _uncond(df: pd.DataFrame) -> pd.Series:
    for c in ("xpts_uncond", "xpts_rot"):
        if c in df.columns:
            return _col(df, c)
    return _col(df, "fantasy_pts") * _col(df, "p_start").clip(0, 1)


def _window(df: pd.DataFrame) -> pd.Series:
    for c in ("total_xpts_rot", "total_xpts_next"):
        if c in df.columns:
            return _col(df, c)
    return _uncond(df) * _col(df, "n_fixtures_next")


#: Legal starting-XI shape: exactly one keeper, then DEF/MID/FWD within these bounds.
_XI_BOUNDS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def legal_xi(squad: pd.DataFrame, col: str) -> pd.DataFrame:
    """The best LEGAL XI from a 15 — every formation tried, highest total kept.

    This exists because comparing an optimiser's legal XI against `nlargest(11)` of a squad is
    not a comparison at all: the top 11 by points is usually an illegal shape (no keeper, six
    defenders), so it scores higher than any legal team and every chip came out NEGATIVE. Free
    Hit read -1.82 and Wildcard -9.30, which would have told a manager to never use either.
    """
    by = {p: squad[squad.get("position") == p].sort_values(col, ascending=False)
          for p in _XI_BOUNDS}
    if by["GKP"].empty:
        return squad.nlargest(0, col)
    best, best_pts = None, -1e9
    for d in range(_XI_BOUNDS["DEF"][0], _XI_BOUNDS["DEF"][1] + 1):
        for m in range(_XI_BOUNDS["MID"][0], _XI_BOUNDS["MID"][1] + 1):
            f = 11 - 1 - d - m
            if not (_XI_BOUNDS["FWD"][0] <= f <= _XI_BOUNDS["FWD"][1]):
                continue
            if len(by["DEF"]) < d or len(by["MID"]) < m or len(by["FWD"]) < f:
                continue
            xi = pd.concat([by["GKP"].head(1), by["DEF"].head(d),
                            by["MID"].head(m), by["FWD"].head(f)])
            tot = float(pd.to_numeric(xi[col], errors="coerce").sum())
            if tot > best_pts:
                best, best_pts = xi, tot
    return best if best is not None else squad.nlargest(0, col)


def chip_values(proj: pd.DataFrame, squad: pd.DataFrame | None = None,
                budget: float = 100.0) -> dict:
    """Incremental expected points for each chip. Says so when it cannot tell."""
    out: dict = {"triple_captain": {}, "bench_boost": {}, "free_hit": {}, "wildcard": {}}
    if proj is None or proj.empty:
        return {"error": "no projections"}
    p = proj.copy()
    p["_gw"] = _uncond(p)
    p["_win"] = _window(p)

    # ── Triple Captain: exactly one more captain ─────────────────────────────
    elig = p
    if "availability" in p.columns:
        # Ruled out, doubtful, or availability UNKNOWN: none can be the Triple Captain.
        from player_model.fpl_api import availability_unknown, not_recommendable
        n_unknown = int(availability_unknown(p["availability"]).sum())
        if n_unknown:
            log.info("chips: %d player(s) with unknown availability not considered for Triple Captain",
                     n_unknown)
        elig = p[~not_recommendable(p["availability"], also=("doubtful",))]
    if "p_start" in elig.columns:
        # Unknown P(start) is not eligible (NaN >= 0.60 is False) - a captain must be known to start.
        elig = elig[pd.to_numeric(elig["p_start"], errors="coerce") >= 0.60]
    elig = elig[elig["_gw"].notna()]
    if elig.empty:
        out["triple_captain"] = {"available": False, "why": "no eligible captain"}
    else:
        best = elig.nlargest(1, "_gw").iloc[0]
        nfix = float(pd.to_numeric(pd.Series([best.get("n_fixtures_next")], dtype="object"),
                                   errors="coerce")[0])
        out["triple_captain"] = {
            "available": True,
            "player": best.get("player_name"), "team": best.get("team"),
            "incremental_points": round(float(best["_gw"]), 2),
            "basis": "3x instead of 2x is exactly one extra captain score",
            # n_fixtures_next counts the whole WINDOW, not one gameweek, so it cannot be used
            # to claim a double. Saying "5 fixtures in the window, so this is a DGW" was simply
            # wrong; a real DGW check needs the per-gameweek fixture calendar.
            "window_fixtures": int(nfix) if nfix == nfix else None,
            "note": ("Triple Captain is worth one extra captain score. It pays properly only in "
                     "a DOUBLE gameweek, which this cannot confirm — n_fixtures_next counts the "
                     "whole window, not a single gameweek. Check the fixture ticker before "
                     "burning it."),
        }

    if squad is None or squad.empty:
        for k in ("bench_boost", "free_hit", "wildcard"):
            out[k] = {"available": False,
                      "why": "needs your actual squad — a bench cannot be valued from the "
                             "fixture list, and a swap cannot be valued without knowing what "
                             "you already own"}
        out["squad_supplied"] = False
        return out

    out["squad_supplied"] = True
    sq = squad.copy()
    # Owned squad joined to projections on the FPL element id (not a lower-cased name). An owned
    # player with no projection is NOT valued at 0: the squad-dependent chips are withheld, since
    # any number would silently understate (or, for Free Hit/Wildcard, overstate) the chip.
    sq["_key"] = fpl_ids(sq)
    look = p.assign(_key=fpl_ids(p)).dropna(subset=["_key"]).drop_duplicates("_key").set_index("_key")
    sq["_gw"] = sq["_key"].map(look["_gw"])
    sq["_win"] = sq["_key"].map(look["_win"])
    missing = sq["_gw"].isna() | sq["_win"].isna()
    if missing.any():
        names = (", ".join(sq.loc[missing, "player_name"].astype(str))
                 if "player_name" in sq.columns else "")
        log.info("chip_values: %d owned player(s) without a projection (%s) - squad chips withheld",
                 int(missing.sum()), names)
        for k in ("bench_boost", "free_hit", "wildcard"):
            out[k] = {"available": False,
                      "why": f"{int(missing.sum())} owned player(s) have no projection "
                             f"({names}); the chip cannot be valued without them"}
        return out

    # ── Bench Boost: what the bench would otherwise not score ────────────────
    # The bench is whoever is NOT in the best legal XI. Using best_xi() here failed silently
    # on a squad it did not like, which left `starters` empty and made the whole squad count as
    # bench -- Bench Boost then rose to 59 points on a squad with four non-playing reserves,
    # the exact opposite of the truth.
    _xi = legal_xi(sq, "_gw")
    starters = set(_xi["player_name"]) if len(_xi) else set(sq.nlargest(11, "_gw")["player_name"])
    bench = sq[~sq["player_name"].isin(starters)]
    out["bench_boost"] = {
        "available": True,
        "incremental_points": round(float(bench["_gw"].sum()), 2),
        "bench": bench[["player_name", "position", "_gw"]].to_dict("records"),
        "basis": "points your four bench players are expected to score but would not count",
        "note": ("your bench is doing nothing — this chip is near-worthless right now"
                 if float(bench["_gw"].sum()) < 6 else
                 "a bench worth boosting"),
    }

    # ── Free Hit and Wildcard: the optimiser against what you own ────────────
    try:
        from player_model.fantasy_optimizer import optimal_squad
        for chip, col, horizon in (("free_hit", "_gw", "this gameweek"),
                                   ("wildcard", "_win", "the fixture window")):
            best = optimal_squad(p.assign(xpts_uncond=p[col]), budget=budget)
            if not best.get("feasible"):
                out[chip] = {"available": False, "why": best.get("reason", "no legal squad")}
                continue
            # Compare like with like: a LEGAL XI from the squad, not its top 11 by points.
            _mine_xi = legal_xi(sq, col)
            mine = float(pd.to_numeric(_mine_xi[col], errors="coerce").sum())
            out[chip] = {
                "available": True,
                "incremental_points": round(best["xi_points"] - mine, 2),
                "optimal_xi_points": round(best["xi_points"], 2),
                "your_xi_points": round(mine, 2),
                "basis": f"best legal squad over {horizon}, minus your current best XI",
                "caveat": ("the optimal squad ignores what selling your players would actually "
                           "realise — treat this as an upper bound on the gain"),
            }
    except Exception as e:                                           # noqa: BLE001
        for chip in ("free_hit", "wildcard"):
            out[chip] = {"available": False, "why": f"optimiser unavailable ({type(e).__name__})"}
    return out
