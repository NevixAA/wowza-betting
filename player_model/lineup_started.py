"""
lineup_started.py - `started` read from the collected lineups, not from the provider's flag
(owner 2026-11-15, props feature audit fix 1).

WHY. player_history.started is the provider's games.substitute flag (/fixtures/players). Measured on
the props frame (props_harness.source_frame) against the collected API-Football lineups
(output/collect/lineups/players_<league>_<season>.parquet, role 'start' / 'sub'): the two agreed on
100.0% of rows every month 2022-08..2026-07 (starter share ~0.75 both). From 2026-08 the provider
flag drifts: share 0.83 in 2026-09, 0.96 in 2026-10, 0.97 in 2026-11 while the lineups stay at
0.75-0.77; agreement fell to 92% / 81% / 81%, and 87% of players with under 45 minutes in 2026-11
were flagged as starters. Every feature reading `started` (starter_rate, season_start_rate,
prev_season_start_rate) inherited it.

THE RULE. started = 1 when the collected lineup of the fixture lists the player in the starting XI,
0 when it lists him on the bench. Where no lineup proves it - the fixture has no collected lineup, the
player is not listed, or the lineups list him under both roles - started is MISSING (never the
provider flag, never a guess from minutes). The provider's value stays beside in `started_provider`;
`started_source` says which rule gave the value. props_harness.source_frame applies it, so training
and serving read the same `started`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LINEUP_DIR = config.OUTPUT_DIR / "collect" / "lineups"
ORIG_COL = "started_provider"
SOURCE_COL = "started_source"
FROM_LINEUP_START = "lineup_start"
FROM_LINEUP_SUB = "lineup_sub"
NO_PROOF = "no_lineup_proof"
_CACHE: dict = {}


def lineup_roles(directory=None) -> pd.DataFrame:
    """(fixture_id, player_id, started) from every collected lineup file: 1.0 for role 'start', 0.0 for
    'sub'. A (fixture, player) listed under both roles is left out (no proof either way)."""
    d = LINEUP_DIR if directory is None else directory
    files = sorted(d.glob("players_*.parquet")) if d.exists() else []
    key = (str(d), tuple((f.name, f.stat().st_mtime_ns) for f in files))
    if _CACHE.get("key") == key:
        return _CACHE["frame"]
    parts = [pd.read_parquet(f, columns=["fixture_id", "player_id", "role"]) for f in files]
    lp = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=["fixture_id", "player_id", "role"]))
    lp["fixture_id"] = pd.to_numeric(lp["fixture_id"], errors="coerce")
    lp["player_id"] = pd.to_numeric(lp["player_id"], errors="coerce")
    lp["started"] = lp["role"].map({"start": 1.0, "sub": 0.0})
    lp = lp.dropna(subset=["fixture_id", "player_id", "started"]).drop_duplicates(
        ["fixture_id", "player_id", "started"])
    both = lp.duplicated(["fixture_id", "player_id"], keep=False)
    out = lp.loc[~both, ["fixture_id", "player_id", "started"]].reset_index(drop=True)
    _CACHE.clear()
    _CACHE.update(key=key, frame=out)
    return out


def with_lineup_started(raw: pd.DataFrame, roles: pd.DataFrame | None = None) -> pd.DataFrame:
    """`raw` with `started` from the collected lineups (1 / 0, missing without a lineup proof), the
    provider's value kept in started_provider and the rule in started_source."""
    if raw is None or raw.empty or not {"fixture_id", "player_id"} <= set(raw.columns):
        return raw
    roles = lineup_roles() if roles is None else roles
    out = raw.copy()
    prov = (pd.to_numeric(out["started"], errors="coerce") if "started" in out.columns
            else pd.Series(np.nan, index=out.index))
    k = pd.DataFrame({"fixture_id": pd.to_numeric(out["fixture_id"], errors="coerce").to_numpy(),
                      "player_id": pd.to_numeric(out["player_id"], errors="coerce").to_numpy()})
    got = k.merge(roles, on=["fixture_id", "player_id"], how="left")["started"].to_numpy(dtype=float)
    out[ORIG_COL] = prov.astype(float)
    out["started"] = got
    out[SOURCE_COL] = np.where(got == 1.0, FROM_LINEUP_START, np.where(got == 0.0, FROM_LINEUP_SUB, NO_PROOF))
    n = len(out)
    known = ~np.isnan(got)
    disagree = int((known & prov.notna().to_numpy() & (prov.to_numpy() != got)).sum())
    print(f"[started] {int(known.sum()):,} of {n:,} row(s) carry started from the collected lineups; "
          f"{n - int(known.sum()):,} missing (no lineup proof); {disagree:,} provider flag(s) differed "
          f"(kept in {ORIG_COL})")
    return out
