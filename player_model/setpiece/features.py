"""
As-of-date set-piece features from Understat shots (owner decision B, 2026-11-02).

Per (player_id, date D), over the player's MAPPED Understat appearances on days strictly before D
and no older than WINDOW_DAYS (set-piece duties change with clubs and coaches):

  us_sp_taker_share     set-piece shots (FromCorner / SetPiece) of his team that HE assisted, over
                        the team's assisted set-piece shots in the matches he played.
                        NaN unless the denominator >= MIN_TEAM_SP_EVENTS.
  us_sp_receiver_share  his own FromCorner / SetPiece shots over his team's, same matches.
                        NaN unless the team had >= MIN_TEAM_SP_EVENTS such shots.
  us_sp_xg_p90          his xG from FromCorner / SetPiece / DirectFreekick shots per 90 minutes.
  us_sp_sot_p90         his on-target (Goal / SavedShot) FromCorner / SetPiece shots per 90.
  us_dfk_shots_p90      his direct free-kick shots per 90 (the direct free-kick taker).
                        The per-90 features are NaN below MIN_US_MINUTES minutes.

The assister of a shot is Understat's `player_assisted` name, identified inside the SAME match and
side roster (one provider, one roster); a name carried by two roster players is refused.
Appearance dates are OUR fixture dates (mapping.map_matches), so a kick-off that Understat dates a
day earlier can never leak its own match. Unmapped players / matches and leagues Understat does not
cover: NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .understat import ON_TARGET_RESULTS, SET_PIECE_SITUATIONS

DIR = Path(__file__).resolve().parents[2] / "output" / "collect"
RECEIVE_SITUATIONS = frozenset({"FromCorner", "SetPiece"})

WINDOW_DAYS = 365
MIN_TEAM_SP_EVENTS = 10      # team set-piece shots (or assisted ones) behind a share
MIN_US_MINUTES = 450         # Understat minutes behind a per-90 rate

#: The Understat set-piece features this module can build.
UNDERSTAT_SETPIECE_COLUMNS = ("us_sp_taker_share", "us_sp_receiver_share", "us_sp_xg_p90", "us_sp_sot_p90",
                              "us_dfk_shots_p90")
#: LICENCE GATE (owner 2026-11-02): Understat data is free for NON-commercial use only (its support,
#: per the understatr README); Wowza is commercial (paid tiers), so these features are NOT used in
#: training or serving until Understat grants written permission for commercial use. Set
#: UNDERSTAT_COMMERCIAL_LICENCE to the written permission's reference to enable them.
UNDERSTAT_COMMERCIAL_LICENCE: str | None = None
SETPIECE_COLUMNS = UNDERSTAT_SETPIECE_COLUMNS if UNDERSTAT_COMMERCIAL_LICENCE else ()
_SUMS = ("minutes", "sp_assists", "team_recv_assisted", "recv_shots", "team_recv_shots", "sp_xg",
         "recv_sot", "dfk_shots")


def appearances(shots: pd.DataFrame, rosters: pd.DataFrame, match_map: pd.DataFrame,
                player_map: pd.DataFrame, history_dates: pd.DataFrame) -> pd.DataFrame:
    """One row per mapped (player_id, fixture): date and the per-match counts in _SUMS."""
    cols = ["player_id", "date", *_SUMS]
    if rosters.empty or match_map.empty or player_map.empty:
        return pd.DataFrame(columns=cols)
    r = rosters.merge(match_map[["match_id", "fixture_id"]], on="match_id")
    r = r.merge(player_map[["us_player_id", "player_id"]], on="us_player_id")
    if r.empty:
        return pd.DataFrame(columns=cols)
    s = shots[shots["match_id"].isin(set(r["match_id"]))].copy()
    s["_recv"] = s["situation"].isin(RECEIVE_SITUATIONS)
    s["_sp"] = s["situation"].isin(SET_PIECE_SITUATIONS)
    s["_on"] = s["result"].isin(ON_TARGET_RESULTS)
    s["_dfk"] = s["situation"] == "DirectFreekick"           # Understat's own label
    s["_xg_sp"] = pd.to_numeric(s["xG"], errors="coerce").where(s["_sp"])
    s["_recv_on"] = s["_recv"] & s["_on"]

    per_player = s.groupby(["match_id", "us_player_id"]).agg(
        recv_shots=("_recv", "sum"), sp_xg=("_xg_sp", "sum"), recv_sot=("_recv_on", "sum"),
        dfk_shots=("_dfk", "sum"))
    recv = s[s["_recv"]]
    team = recv.groupby(["match_id", "h_a"]).agg(team_recv_shots=("shot_id", "count"),
                                                team_recv_assisted=("player_assisted", "count"))
    # Assister name -> roster player of the same match and side; a duplicated name is refused.
    ros = rosters[["match_id", "h_a", "us_player_id", "player"]]
    dupe = ros.duplicated(["match_id", "h_a", "player"], keep=False)
    ros_u = ros[~dupe]
    ast = recv.dropna(subset=["player_assisted"]).merge(
        ros_u.rename(columns={"player": "player_assisted", "us_player_id": "_assister"}),
        on=["match_id", "h_a", "player_assisted"], how="inner")       # same-provider roster key
    sp_ast = ast.groupby(["match_id", "_assister"]).size().rename("sp_assists")
    sp_ast.index = sp_ast.index.set_names(["match_id", "us_player_id"])

    a = r.rename(columns={"time": "minutes"})[["match_id", "h_a", "us_player_id", "player_id",
                                               "fixture_id", "minutes"]]
    a = a.merge(per_player.reset_index(), on=["match_id", "us_player_id"], how="left")
    a = a.merge(sp_ast.reset_index(), on=["match_id", "us_player_id"], how="left")
    a = a.merge(team.reset_index(), on=["match_id", "h_a"], how="left")
    # A REAL zero, not a fill: the roster row exists only because the match payload (shots and
    # roster together) was collected, and no shot of that kind is in it.
    for c in ("recv_shots", "sp_xg", "recv_sot", "dfk_shots", "sp_assists", "team_recv_shots",
              "team_recv_assisted"):
        a[c] = pd.to_numeric(a[c], errors="coerce").fillna(0.0)
    d = history_dates.drop_duplicates("fixture_id")[["fixture_id", "date"]]
    a = a.merge(d, on="fixture_id", how="inner")
    a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.normalize()
    a["minutes"] = pd.to_numeric(a["minutes"], errors="coerce")
    a = a.dropna(subset=["date", "minutes"])
    return a[cols].sort_values(["player_id", "date"]).reset_index(drop=True)


def asof(app: pd.DataFrame, player_ids, dates) -> pd.DataFrame:
    """SETPIECE_COLUMNS for each (player_id, date), aligned with the inputs."""
    q = pd.DataFrame({"player_id": pd.to_numeric(pd.Series(list(player_ids)), errors="coerce"),
                      "date": pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()})
    out = pd.DataFrame(np.nan, index=q.index, columns=list(SETPIECE_COLUMNS))
    if app.empty:
        return out
    groups = {int(p): g for p, g in app.groupby("player_id")}
    win = np.timedelta64(WINDOW_DAYS, "D")
    for pid, qi in q.dropna().groupby("player_id").groups.items():
        g = groups.get(int(pid))
        if g is None:
            continue
        dts = g["date"].to_numpy("datetime64[ns]")
        cum = {c: np.r_[0.0, np.cumsum(g[c].to_numpy(float))] for c in _SUMS}
        qd = q.loc[qi, "date"].to_numpy("datetime64[ns]")
        hi = np.searchsorted(dts, qd, side="left")           # strictly earlier days
        lo = np.searchsorted(dts, qd - win, side="left")
        tot = {c: cum[c][hi] - cum[c][lo] for c in _SUMS}
        mins = tot["minutes"]
        with np.errstate(divide="ignore", invalid="ignore"):
            out.loc[qi, "us_sp_taker_share"] = np.where(
                tot["team_recv_assisted"] >= MIN_TEAM_SP_EVENTS, tot["sp_assists"] / tot["team_recv_assisted"], np.nan)
            out.loc[qi, "us_sp_receiver_share"] = np.where(
                tot["team_recv_shots"] >= MIN_TEAM_SP_EVENTS, tot["recv_shots"] / tot["team_recv_shots"], np.nan)
            ok = mins >= MIN_US_MINUTES
            out.loc[qi, "us_sp_xg_p90"] = np.where(ok, tot["sp_xg"] / mins * 90, np.nan)
            out.loc[qi, "us_sp_sot_p90"] = np.where(ok, tot["recv_sot"] / mins * 90, np.nan)
            out.loc[qi, "us_dfk_shots_p90"] = np.where(ok, tot["dfk_shots"] / mins * 90, np.nan)
    return out


_LOADED: dict = {}


def load(directory: Path | None = None, history_dates: pd.DataFrame | None = None) -> pd.DataFrame:
    """The appearance table built from the collected files (cached on file mtimes). Missing files ->
    empty table (every feature NaN), logged once."""
    d = Path(directory or DIR)
    files = {k: d / f"understat_{k}.parquet" for k in ("shots", "rosters", "match_map", "player_map")}
    stamp = tuple(f.stat().st_mtime if f.exists() else None for f in files.values())
    key = (str(d), None if history_dates is None else
           (len(history_dates), float(pd.to_numeric(history_dates["fixture_id"], errors="coerce").sum())))
    if key in _LOADED and _LOADED[key][0] == stamp:
        return _LOADED[key][1]
    if not all(f.exists() for f in files.values()) or history_dates is None:
        print(f"[setpiece] no mapped Understat data in {d}: set-piece features are NaN")
        app = pd.DataFrame(columns=["player_id", "date", *_SUMS])
    else:
        t = {k: pd.read_parquet(f) for k, f in files.items()}
        app = appearances(t["shots"], t["rosters"], t["match_map"], t["player_map"], history_dates)
    _LOADED[key] = (stamp, app)
    return app
