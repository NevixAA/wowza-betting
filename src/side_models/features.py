"""As-of team features for the side count models - ONE function for training and serving (parity).
=================================================================================================

For a target fixture (league, date, home team id, away team id) and a family ('corners' | 'cards'):

  * WINDOW: the team's last `window` finished fixtures IN THIS LEAGUE strictly before the target's date
    (home or away), and only those within `max_age_days` of it. The sequence comes from the full
    API-Football fixture list, so a match whose statistics are missing or not verified still occupies
    its place in the window - it is never skipped to reach an older verified one.
  * FEATURES: over the VERIFIED matches of that window: mean count FOR the team, mean count AGAINST it.
  * COVERAGE RULE: fewer than `min_verified` verified matches in the window -> the side is NOT scored,
    with the reason (missing stays missing; nothing is filled from a league mean or a prior).

Only pre-match information enters: every match used kicked off on an earlier date than the target.
No bookmaker price is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILIES = ("corners", "cards")
FEATURES = ("home_for", "home_against", "away_for", "away_against")


def _long(table: pd.DataFrame, family: str) -> pd.DataFrame:
    """One row per (fixture, team): date, count for, count against, verified flag."""
    ok = table[f"{family}_verified"].astype(bool)
    h = pd.DataFrame({"league": table["league"], "team_id": table["home_id"], "date": table["date"],
                      "fixture_id": table["fixture_id"], "for": table[f"home_{family}"],
                      "against": table[f"away_{family}"], "ok": ok})
    a = pd.DataFrame({"league": table["league"], "team_id": table["away_id"], "date": table["date"],
                      "fixture_id": table["fixture_id"], "for": table[f"away_{family}"],
                      "against": table[f"home_{family}"], "ok": ok})
    lo = pd.concat([h, a], ignore_index=True)
    lo["d"] = pd.to_datetime(lo["date"]).values.astype("datetime64[D]").astype("int64")
    return lo.sort_values(["league", "team_id", "d", "fixture_id"]).reset_index(drop=True)


def team_window(g: tuple, day: int, *, window: int, max_age_days: int, min_verified: int) -> tuple:
    """(mean for, mean against, n verified, n in window, reason) for one team before `day`.
    `g` = (days, count for, count against, verified flag) arrays of the team, sorted by day."""
    d, cf, ca, ok = g
    i = int(np.searchsorted(d, day, side="left"))          # strictly earlier dates only
    lo = max(0, i - window)
    dw, fw, aw, ow = d[lo:i], cf[lo:i], ca[lo:i], ok[lo:i]
    keep = dw >= day - max_age_days
    fw, aw, ow = fw[keep], aw[keep], ow[keep]
    n_w, n_v = int(keep.sum()), int(ow.sum())
    if n_v < min_verified:
        return (np.nan, np.nan, n_v, n_w,
                f"{n_v} verified match(es) in the team's last {n_w} league fixture(s) "
                f"(window {window}, max age {max_age_days} d); needs >= {min_verified}")
    return float(fw[ow].mean()), float(aw[ow].mean()), n_v, n_w, ""


_EMPTY = (np.zeros(0, "int64"), np.zeros(0), np.zeros(0), np.zeros(0, bool))


def build(table: pd.DataFrame, targets: pd.DataFrame, family: str, *, window: int, max_age_days: int,
          min_verified: int) -> pd.DataFrame:
    """Features for every target row (columns: fixture_id, league, date, home_id, away_id).
    Returns the targets with home_for/home_against/away_for/away_against, n_verified per side and
    `not_scored_reason` ('' when both sides are covered)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    lo = _long(table, family)
    groups = {k: (g["d"].to_numpy("int64"), g["for"].to_numpy(float), g["against"].to_numpy(float),
                  g["ok"].to_numpy(bool))
              for k, g in lo.groupby(["league", "team_id"], sort=False)}
    out = []
    for r in targets[["fixture_id", "league", "date", "home_id", "away_id"]].itertuples(index=False):
        day = int(np.datetime64(pd.Timestamp(r.date).strftime("%Y-%m-%d"), "D").astype("int64"))
        hf, ha, hn, hw, hr = team_window(groups.get((r.league, int(r.home_id)), _EMPTY), day, window=window,
                                         max_age_days=max_age_days, min_verified=min_verified)
        af, aa, an, aw, ar = team_window(groups.get((r.league, int(r.away_id)), _EMPTY), day, window=window,
                                         max_age_days=max_age_days, min_verified=min_verified)
        reason = "; ".join(x for x in (f"home: {hr}" if hr else "", f"away: {ar}" if ar else "") if x)
        out.append({"fixture_id": r.fixture_id, "home_for": hf, "home_against": ha, "away_for": af,
                    "away_against": aa, "home_n_verified": hn, "away_n_verified": an,
                    "home_n_window": hw, "away_n_window": aw, "not_scored_reason": reason})
    f = pd.DataFrame(out)
    base = targets.reset_index(drop=True).drop(columns=[c for c in f.columns if c != "fixture_id" and c in targets.columns])
    return base.merge(f, on="fixture_id", how="left", validate="one_to_one")
