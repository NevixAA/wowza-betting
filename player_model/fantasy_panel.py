"""Fantasy V2 panel: one row per (gameweek, player) with ONLY pre-deadline information + what happened.

    from player_model.fantasy_panel import build_panel
    panel = build_panel()          # reads the snapshot archive, fixtures cache and projection log

INFORMATION SET (all knowable before the gameweek's deadline):
  * the archived pre-deadline FPL snapshot for that gameweek (ep_next, price, status, chance of
    playing, ownership, transfers, set-piece orders, cumulative season counters);
  * deltas against the previous gameweek's snapshot (minutes / start / points LAST gameweek);
  * fixtures that had finished before the deadline (team goals, team xG built from player xG);
  * the gameweek's own fixture list (opponent, home/away, official FDR, blank/double);
  * Wowza's own published projection: the last projection-log row stamped before the deadline.

TARGET (after the gameweek): next snapshot minus this snapshot, on cumulative counters. The next
snapshot is taken before the NEXT deadline, so bonus is included and no later points leak in.

KNOWN, STATED LEAKAGE RISKS (small): the fixtures cache is the latest file, so a fixture
postponed or re-rated AFTER a deadline shows its final event/FDR rather than the one shown at the
time. FPL rarely re-rates FDR in-season; postponements are visible as a blank (n_fix = 0).

A gameweek is FINAL when FPL marks it finished + data_checked and the following snapshot exists.
Otherwise it is PROVISIONAL, and only players whose team had finished all of its gameweek
fixtures before the following snapshot are kept — an unplayed match is unknown, never a zero.
"""
from __future__ import annotations

import json
import logging
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from player_model import fantasy_snapshots as fs

log = logging.getLogger(__name__)

OUT = Path(__file__).resolve().parents[1] / "output"
FIXTURES = OUT / "fpl_fixtures.json"
BOOTSTRAP = OUT / "fpl_bootstrap.json"
PROJ_LOG = OUT / "fantasy_projection_log.csv"

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
#: minutes buckets of the mixture model
BUCKETS = ["m0", "m1_59", "m60"]


def norm_name(s) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def load_fixtures(path: Path | None = None) -> pd.DataFrame:
    p = path or FIXTURES
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                           # noqa: BLE001
        log.warning("fantasy_panel: no FPL fixtures (%s: %s) - fixture counts unknown",
                    type(e).__name__, e)
        return pd.DataFrame()
    rows = [{"fixture": f.get("id"), "event": f.get("event"), "kickoff": f.get("kickoff_time"),
             "team_h": f.get("team_h"), "team_a": f.get("team_a"),
             "h_score": f.get("team_h_score"), "a_score": f.get("team_a_score"),
             "h_fdr": f.get("team_h_difficulty"), "a_fdr": f.get("team_a_difficulty"),
             "finished": bool(f.get("finished") or f.get("finished_provisional"))}
            for f in raw]
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["kickoff"] = pd.to_datetime(d["kickoff"], utc=True, errors="coerce")
    return d


def team_fixture_long(fx: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, fixture) with the team's own perspective."""
    if fx.empty:
        return pd.DataFrame(columns=["team_id", "opp_id", "event", "kickoff", "home", "gf", "ga",
                                     "fdr", "finished", "fixture"])
    h = fx.rename(columns={"team_h": "team_id", "team_a": "opp_id", "h_score": "gf",
                           "a_score": "ga", "h_fdr": "fdr"}).assign(home=True)
    a = fx.rename(columns={"team_a": "team_id", "team_h": "opp_id", "a_score": "gf",
                           "h_score": "ga", "a_fdr": "fdr"}).assign(home=False)
    cols = ["team_id", "opp_id", "event", "kickoff", "home", "gf", "ga", "fdr", "finished", "fixture"]
    return pd.concat([h[cols], a[cols]], ignore_index=True)


def events_table(path: Path | None = None) -> pd.DataFrame:
    try:
        b = json.loads((path or BOOTSTRAP).read_text(encoding="utf-8"))
    except Exception as e:                                           # noqa: BLE001
        log.warning("fantasy_panel: no FPL bootstrap (%s: %s) - no gameweek events",
                    type(e).__name__, e)
        return pd.DataFrame(columns=["gw", "deadline", "finished", "checked"])
    ev = pd.DataFrame([{"gw": e["id"], "deadline": e["deadline_time"],
                        "finished": bool(e.get("finished")),
                        "checked": bool(e.get("data_checked"))} for e in b.get("events", [])])
    if not ev.empty:
        ev["deadline"] = pd.to_datetime(ev["deadline"], utc=True)
    return ev


# ── team strength, point in time ────────────────────────────────────────────────────────────
def team_strength_asof(snap: pd.DataFrame, tfl: pd.DataFrame, deadline) -> pd.DataFrame:
    """Per team, using only fixtures finished before `deadline` and this snapshot's player xG.

    xg_for_pg      team xG per game (sum of its players' cumulative xG / games played)
    xga_pg         team xG conceded per game (minutes-weighted goalkeeper xGC per 90)
    gf_pg, ga_pg   actual goals for/against per game
    Each is shrunk toward the league mean with a 4-game prior, because after three gameweeks a
    raw per-game rate is mostly noise.
    """
    done = tfl[(tfl["finished"]) & (tfl["kickoff"] < deadline)]
    gp = done.groupby("team_id").agg(games=("fixture", "size"), gf=("gf", "sum"), ga=("ga", "sum"))
    s = snap.copy()
    # A player with no xG / xGC / minutes figure makes his TEAM's total unknown (NaN), rather than
    # being counted as 0 and quietly deflating the team's xG.
    s["xg"] = pd.to_numeric(s["xg"], errors="coerce")
    xg_for = s.groupby("team_id")["xg"].agg(_strict_sum)
    gk = s[s["position"] == "GKP"].copy()
    gk["minutes"] = pd.to_numeric(gk["minutes"], errors="coerce")
    gk["xgc"] = pd.to_numeric(gk["xgc"], errors="coerce")
    gkt = gk.groupby("team_id").agg(xgc=("xgc", _strict_sum), mins=("minutes", _strict_sum))
    t = pd.DataFrame(index=sorted(set(s["team_id"].dropna().astype(int))))
    t["games"] = gp["games"].reindex(t.index).fillna(0)
    g = t["games"].replace(0, np.nan)
    t["gf_pg_raw"] = gp["gf"].reindex(t.index) / g
    t["ga_pg_raw"] = gp["ga"].reindex(t.index) / g
    t["xg_for_pg_raw"] = xg_for.reindex(t.index) / g
    t["xga_pg_raw"] = (gkt["xgc"] / (gkt["mins"] / 90.0).replace(0, np.nan)).reindex(t.index)
    prior = 4.0
    for k in ("gf_pg", "ga_pg", "xg_for_pg", "xga_pg"):
        raw = t[f"{k}_raw"]
        # The league mean is measured from this league's own finished fixtures; with none yet it
        # is unknown (NaN) - it used to be a typed 1.35.
        mu = float(np.nanmean(raw)) if raw.notna().any() else float("nan")
        t[k] = ((raw.fillna(mu) * t["games"] + mu * prior) / (t["games"] + prior))
    # Home advantage MEASURED from the same finished fixtures (goals per game at home / overall,
    # and away / overall); it was an unfitted 1.10 / 0.91. NaN with no finished fixture.
    gf = pd.to_numeric(done["gf"], errors="coerce")
    allm = float(gf.mean()) if gf.notna().any() else float("nan")
    hm = done["home"].astype(bool)
    t["ha_home"] = float(gf[hm].mean()) / allm if allm and allm == allm and hm.any() else np.nan
    t["ha_away"] = float(gf[~hm].mean()) / allm if allm and allm == allm and (~hm).any() else np.nan
    t.index.name = "team_id"
    return t[["games", "gf_pg", "ga_pg", "xg_for_pg", "xga_pg", "ha_home", "ha_away"]].reset_index()


def _strict_sum(v: pd.Series) -> float:
    """Sum that is NaN when any member is unknown (pandas' sum would count a NaN as 0)."""
    return float(v.sum()) if len(v) and v.notna().all() else float("nan")


def fixture_lambdas(team_id, opp_id, home, ts: pd.DataFrame) -> tuple[float, float]:
    """(team goal expectancy, opponent goal expectancy) for one fixture, multiplicative model.
    (NaN, NaN) when either club or the league reference is unknown - never the league mean."""
    nan = (float("nan"), float("nan"))
    t = ts.set_index("team_id")
    lg = float(t["xg_for_pg"].mean()) if len(t) else float("nan")
    if not lg == lg or lg <= 0:
        return nan
    try:
        att = 0.5 * t.at[team_id, "xg_for_pg"] + 0.5 * t.at[team_id, "gf_pg"]
        dfn = 0.5 * t.at[team_id, "xga_pg"] + 0.5 * t.at[team_id, "ga_pg"]
        o_att = 0.5 * t.at[opp_id, "xg_for_pg"] + 0.5 * t.at[opp_id, "gf_pg"]
        o_dfn = 0.5 * t.at[opp_id, "xga_pg"] + 0.5 * t.at[opp_id, "ga_pg"]
    except KeyError:
        return nan
    ha_for = t["ha_home" if home else "ha_away"].iloc[0] if "ha_home" in t.columns else np.nan
    ha_against = t["ha_away" if home else "ha_home"].iloc[0] if "ha_home" in t.columns else np.nan
    lam_for = att * o_dfn / lg * ha_for
    lam_against = o_att * dfn / lg * ha_against
    if not (lam_for == lam_for and lam_against == lam_against):
        return nan
    return float(np.clip(lam_for, 0.2, 4.5)), float(np.clip(lam_against, 0.2, 4.5))


def gw_fixture_context(gw: int, tfl: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """Per team for gameweek `gw`: n_fix, home share, mean FDR, lambda for/against, CS prob."""
    f = tfl[tfl["event"] == gw]
    rows = []
    for tid, g in f.groupby("team_id"):
        lams = [fixture_lambdas(tid, r.opp_id, bool(r.home), ts) for r in g.itertuples()]
        rows.append({"team_id": int(tid), "n_fix": int(len(g)),
                     "home": float(g["home"].mean()),
                     "fdr": float(pd.to_numeric(g["fdr"], errors="coerce").mean()),
                     "opp_ids": ",".join(str(int(x)) for x in g["opp_id"]),
                     "lam_for": float(sum(l[0] for l in lams)),
                     "lam_against": float(sum(l[1] for l in lams)),
                     "cs_prob": float(np.mean([math.exp(-l[1]) for l in lams]))})
    return pd.DataFrame(rows, columns=["team_id", "n_fix", "home", "fdr", "opp_ids", "lam_for",
                                       "lam_against", "cs_prob"])


# ── Wowza's own published projection, as of each deadline ───────────────────────────────────
_WOWZA_COLS = ["key", "fpl_id", "wowza_cond", "wowza_p_start", "wowza_uncond"]


def wowza_asof(log: pd.DataFrame, deadline) -> pd.DataFrame:
    """Wowza's last projection per player before `deadline`. Grouped by the FPL element id where
    the log row carries one (fantasy_log writes it from 2026-11-01), else by name (legacy rows)."""
    if log is None or log.empty:
        return pd.DataFrame(columns=_WOWZA_COLS)
    lg = log[log["snapshot_ts"] < deadline].copy()
    if lg.empty:
        return pd.DataFrame(columns=_WOWZA_COLS)
    fid = (pd.to_numeric(lg["fpl_id"], errors="coerce") if "fpl_id" in lg.columns
           else pd.Series(np.nan, index=lg.index))
    lg["fpl_id"] = fid
    lg["_g"] = ("id:" + fid.astype("Int64").astype(str)).where(fid.notna(),
                                                               "name:" + lg["player_name"].astype(str))
    last = lg.sort_values("snapshot_ts").groupby("_g", as_index=False).last()
    # Only a projection made for THIS window counts: one older than 14 days is not "as of".
    last = last[last["snapshot_ts"] >= deadline - pd.Timedelta(days=14)]
    base = pd.to_numeric(last["proj_fixture_adj"], errors="coerce")
    ps = pd.to_numeric(last["p_start"], errors="coerce").clip(0, 1)
    return pd.DataFrame({"key": last["player_name"].map(norm_name), "fpl_id": last["fpl_id"],
                         "team_log": last["team"],
                         "wowza_cond": base, "wowza_p_start": ps, "wowza_uncond": base * ps})


def _load_log(path: Path | None = None) -> pd.DataFrame:
    p = path or PROJ_LOG
    if not p.exists():
        return pd.DataFrame()
    lg = pd.read_csv(p, usecols=lambda c: c in {"snapshot_ts", "fpl_id", "player_name", "team",
                                                "proj_fixture_adj", "p_start"})
    lg["snapshot_ts"] = pd.to_datetime(lg["snapshot_ts"], utc=True, errors="coerce")
    return lg.dropna(subset=["snapshot_ts"])


def _num(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)


def _position_rate(d: pd.DataFrame, col: str, per: str) -> dict:
    """{position: sum(col) / sum(per)} over the snapshot's players with a known value and a
    positive denominator; per = 'n90' (minutes / 90) or 'starts'. Missing position -> absent."""
    num = _num(d, col)
    den = _num(d, "minutes") / 90.0 if per == "n90" else _num(d, "starts")
    ok = num.notna() & (den > 0)
    g = pd.DataFrame({"p": d["position"], "n": num, "d": den})[ok].groupby("p")
    s = g["n"].sum() / g["d"].sum()
    return {k: float(v) for k, v in s.items() if np.isfinite(v)}


def features_for(snap: pd.DataFrame, prev: pd.DataFrame | None, tfl: pd.DataFrame,
                 gw: int, deadline, wowza: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pre-deadline feature rows for one gameweek."""
    d = snap.copy()
    d["gw"] = gw
    d["price"] = _num(d, "now_cost") / 10.0
    ts = team_strength_asof(d, tfl, deadline)
    d = d.merge(ts.rename(columns={"games": "team_games"}), on="team_id", how="left")
    ctx = gw_fixture_context(gw, tfl, ts)
    d = d.merge(ctx, on="team_id", how="left")
    # A club with no fixture in a LOADED fixture list has a blank gameweek (0 fixtures, real);
    # with no fixture list at all the count is unknown (NaN), not a league-wide blank.
    d["n_fix"] = d["n_fix"].fillna(0).astype(int) if not tfl.empty else np.nan
    tg = d["team_games"].replace(0, np.nan)
    # Unknown season counters stay NaN (they were 0, i.e. "never played").
    mins = _num(d, "minutes")
    d["min_per_team_game"] = (mins / tg).clip(0, 90)
    d["start_rate"] = (_num(d, "starts") / tg).clip(0, 1)
    d["pts_per_team_game"] = _num(d, "total_points") / tg
    n90 = (mins / 90.0)
    prior90 = 3.0    # shrink per-90 rates with three full games of league-average prior
    # The priors are the POSITION's own per-90 rates MEASURED on this same pre-deadline snapshot
    # (they were typed: xG FWD 0.38 / MID 0.17 / DEF 0.05 / GKP 0.0, xA 0.12 / 0.14 / 0.06 / 0.01,
    # BPS 15 per 90, bonus +0.3 per start). NaN while a position has no minutes yet.
    pos_xg, pos_xa, pos_bps, pos_bonus = (_position_rate(d, c, den)
                                          for c, den in (("xg", "n90"), ("xa", "n90"),
                                                         ("bps", "n90"), ("bonus", "starts")))
    d["xg90"] = (_num(d, "xg") + d["position"].map(pos_xg) * prior90) / (n90 + prior90)
    d["xa90"] = (_num(d, "xa") + d["position"].map(pos_xa) * prior90) / (n90 + prior90)
    d["xgi90"] = d["xg90"] + d["xa90"]
    d["bps90"] = (_num(d, "bps") + d["position"].map(pos_bps) * prior90) / (n90 + prior90)
    d["bonus_per_start"] = ((_num(d, "bonus") + d["position"].map(pos_bonus))
                            / (_num(d, "starts") + 1))
    # No defensive-contribution column in the snapshot -> unknown, not 0.
    d["dc90"] = (_num(d, "def_contrib") / (n90 + prior90) if "def_contrib" in d.columns
                 else np.nan)
    d["saves90"] = _num(d, "saves") / (n90 + prior90)
    d["threat_pg"] = _num(d, "threat") / tg
    d["creativity_pg"] = _num(d, "creativity") / tg
    st = d["status"].astype(str)
    ch = _num(d, "chance_next")
    # FPL publishes no chance for a fully available ('a') player: that convention means 100.
    # A null chance on any other status is unknown (NaN) - it used to be read as 0%.
    d["chance_next_f"] = ch.where(ch.notna(), np.where(st == "a", 100.0, np.nan)) / 100.0
    d["is_out"] = st.isin(["i", "s", "u", "n"]).astype(int)
    d["is_doubt"] = (st == "d").astype(int)
    d["has_news"] = d["news"].fillna("").astype(str).str.len().gt(0).astype(int)
    d["pen_first"] = (_num(d, "pen_order") == 1).astype(int)
    d["pen_any"] = _num(d, "pen_order").notna().astype(int)
    d["set_piece"] = (_num(d, "corner_order").le(2) | _num(d, "dfk_order").le(2)).astype(int)
    d["ownership"] = _num(d, "selected_pct")
    d["net_transfers_k"] = (_num(d, "transfers_in_event")
                            - _num(d, "transfers_out_event")) / 1000.0
    d["ppm"] = d["pts_per_team_game"] / d["price"].replace(0, np.nan)
    for p in POSITIONS:
        d[f"pos_{p}"] = (d["position"] == p).astype(int)
    d["is_defensive"] = d["position"].isin(["GKP", "DEF"]).astype(int)
    # Expected CS points: 0 for a blank gameweek (no fixture, no clean sheet - real), NaN when the
    # fixture's CS probability is unknown (it was 0, which reads as "certain to concede").
    blank = d["n_fix"] == 0
    d["cs_exp"] = (d["cs_prob"] * d["position"].map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
                   ).where(~blank, 0.0)
    # GKP 10: FPL game_config.scoring.goals_scored (was 6, R2-D fantasy bug #2).
    d["goal_pts"] = d["position"].map({"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}).astype(float)
    # Attack fixture factor vs the league's measured xG/game; NaN when either is unknown or the
    # club blanks (it was a neutral 1.0, and the league reference a typed 1.35).
    lg_avg = float(ts["xg_for_pg"].mean()) if len(ts) else float("nan")
    lg_avg = lg_avg if lg_avg == lg_avg and lg_avg > 0 else np.nan
    d["attack_fixture"] = (d["lam_for"] / d["n_fix"].replace(0, np.nan)) / lg_avg
    d["ep_next"] = _num(d, "ep_next")
    # last gameweek deltas (NaN when no previous snapshot exists)
    if prev is not None and not prev.empty:
        p = prev.set_index("element")
        for c, new in (("minutes", "min_last"), ("starts", "started_last"),
                       ("total_points", "pts_last")):
            d[new] = _num(d, c).values - pd.to_numeric(d["element"].map(p[c]), errors="coerce").values
        d["ep_prev"] = pd.to_numeric(d["element"].map(p["ep_next"]), errors="coerce")
    else:
        d["min_last"] = np.nan
        d["started_last"] = np.nan
        d["pts_last"] = np.nan
        d["ep_prev"] = np.nan
    return attach_wowza(d, wowza)


def attach_wowza(d: pd.DataFrame, wowza: pd.DataFrame | None) -> pd.DataFrame:
    """Attach Wowza's as-of projection (wowza_cond / wowza_p_start / wowza_uncond, in_ledger)."""
    d = d.copy()
    d["key"] = d["player_name"].map(norm_name)
    d["key_web"] = d["web_name"].map(norm_name)
    wc = ["wowza_cond", "wowza_p_start", "wowza_uncond"]
    if wowza is not None and not wowza.empty:
        fid = (pd.to_numeric(wowza["fpl_id"], errors="coerce") if "fpl_id" in wowza.columns
               else pd.Series(np.nan, index=wowza.index))
        # 1) the FPL element id, wherever the log row has one (both sides then carry an id);
        by_id = wowza[fid.notna()].assign(_id=fid[fid.notna()].astype(int)).drop_duplicates("_id")
        m = pd.DataFrame({c: d["element"].map(by_id.set_index("_id")[c]).values for c in wc})
        # 2) legacy log rows without an id: the name bridge (full name, then web_name) - no id
        #    crosswalk exists for them. A name key held by two players on either side is refused,
        #    never resolved to one of them.
        w = wowza[fid.isna()]
        miss = m["wowza_cond"].isna().values
        if miss.any() and not w.empty:
            dup_w = set(w["key"][w["key"].duplicated(keep=False)])
            w = w[~w["key"].isin(dup_w)]
            look = w[w["key"].map(lambda k: isinstance(k, str) and k != "")].set_index("key")
            for col in ("key", "key_web"):
                dup_d = set(d[col][d[col].duplicated(keep=False)])
                hit = d[col].where(~d[col].isin(dup_d))
                for c in wc:
                    vals = hit.map(lambda k, c=c: look.at[k, c] if isinstance(k, str)
                                   and k in look.index else np.nan).values
                    m.loc[miss, c] = vals[miss]
                miss = m["wowza_cond"].isna().values
        for c in wc:
            d[c] = m[c].values
    else:
        for c in wc:
            d[c] = np.nan
    d["in_ledger"] = d["wowza_uncond"].notna().astype(int)
    return d


def _team_done_by(tfl: pd.DataFrame, gw: int, ts) -> set:
    """Teams whose every gameweek-`gw` fixture had finished (with 3h for scoring) before ts."""
    f = tfl[tfl["event"] == gw]
    done = set()
    for tid, g in f.groupby("team_id"):
        if bool((g["finished"] & (g["kickoff"] + pd.Timedelta(hours=3) < ts)).all()):
            done.add(int(tid))
    return done


def build_panel(archive: pd.DataFrame | None = None, fixtures: pd.DataFrame | None = None,
                events: pd.DataFrame | None = None, log: pd.DataFrame | None = None,
                include_current: bool = True) -> pd.DataFrame:
    """Every archived gameweek with features; settled ones carry actual_* targets.

    status column: FINAL (settled, use for training/eval), PROVISIONAL (partly played, only
    finished teams kept), PENDING (deadline not passed, the gameweek to forecast).
    """
    a = fs.load() if archive is None else archive.copy()
    if a.empty:
        return pd.DataFrame()
    fx = load_fixtures() if fixtures is None else fixtures
    tfl = team_fixture_long(fx)
    ev = events_table() if events is None else events
    lg = _load_log() if log is None else log
    a["fetched_at"] = pd.to_datetime(a["fetched_at"], utc=True)
    a["deadline"] = pd.to_datetime(a["deadline"], utc=True)
    gws = sorted(int(g) for g in a["target_gw"].unique())
    out = []
    for i, gw in enumerate(gws):
        snap = a[a["target_gw"] == gw]
        prev = a[a["target_gw"] == gws[i - 1]] if i > 0 and gws[i - 1] == gw - 1 else None
        deadline = snap["deadline"].iloc[0]
        feats = features_for(snap, prev, tfl, gw, deadline, wowza_asof(lg, deadline))
        nxt = a[a["target_gw"] == gw + 1]
        if nxt.empty:
            feats["status"] = "PENDING"
            if include_current:
                out.append(feats)
            continue
        n = nxt.set_index("element")
        for c in ("total_points", "minutes", "starts", "goals", "assists", "clean_sheets",
                  "bonus", "bps", "saves", "goals_conceded"):
            feats[f"actual_{c}" if c != "total_points" else "actual_points"] = (
                pd.to_numeric(feats["element"].map(n[c]), errors="coerce").values
                - pd.to_numeric(feats[c], errors="coerce").values)
        e = ev[ev["gw"] == gw]
        final = bool(len(e) and e["finished"].iloc[0] and e["checked"].iloc[0])
        nxt_ts = nxt["fetched_at"].max()
        if not final:
            # the in-progress gameweek: keep only teams that have finished all of it
            done = _team_done_by(tfl, gw, nxt_ts)
            blank = set(feats.loc[feats["n_fix"] == 0, "team_id"].astype(int))
            feats = feats[feats["team_id"].astype(int).isin(done | blank)].copy()
            feats["status"] = "PROVISIONAL"
        else:
            feats["status"] = "FINAL"
        feats = feats.dropna(subset=["actual_points"])
        # cumulative counters can only rise inside a season; a fall is a data fault, not a score
        feats = feats[(feats["actual_minutes"] >= 0) & (feats["actual_points"] > -10)]
        am = feats["actual_minutes"]
        feats["bucket"] = np.where(am <= 0, "m0", np.where(am < 60, "m1_59", "m60"))
        out.append(feats)
        if include_current and gw == gws[-1]:
            pass
    if not out:
        return pd.DataFrame()
    # the newest archived gameweek with no successor is the one to forecast
    return add_own_minutes_history(pd.concat(out, ignore_index=True))


#: the player's own minutes history, attached to every panel row (fix9/defaults 2026-11-04)
OWN_HISTORY_COLS = ["hist_gws", "hist_play_share", "hist_starts", "hist_p60_given_start"]


def add_own_minutes_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Each row's PLAYER-OWN minutes record from the FINAL gameweeks BEFORE its own (pre-deadline:
    a FINAL gameweek < g finished before g's deadline). Only single-fixture gameweeks count (a blank
    has no match to play, a double gameweek's 2 starts / 180 minutes cannot be split per match):

      hist_gws               single-fixture FINAL gameweeks before this one with a known minutes count
      hist_play_share        share of those in which he played (minutes > 0)
      hist_starts            those in which he started (starts == 1)
      hist_p60_given_start   share of those starts that reached 60 minutes

    WHY. The FPL minutes baseline (fantasy_residual.heuristic_minutes) multiplied by a typed 0.92
    ("a start reaches 60'"), floored usage at a typed 0.35 and read a missing chance / start rate as 0.
    These are the player's own real rates instead; with no qualifying history they are NaN (missing),
    never a number."""
    d = panel
    if d is None or d.empty or not {"element", "gw", "status"} <= set(d.columns):
        return d
    d = d.copy()
    am = pd.to_numeric(d["actual_minutes"], errors="coerce") if "actual_minutes" in d.columns \
        else pd.Series(np.nan, index=d.index)
    ast = pd.to_numeric(d["actual_starts"], errors="coerce") if "actual_starts" in d.columns \
        else pd.Series(np.nan, index=d.index)
    nfx = pd.to_numeric(d["n_fix"], errors="coerce") if "n_fix" in d.columns \
        else pd.Series(np.nan, index=d.index)
    use = (d["status"] == "FINAL") & (nfx == 1) & am.notna()
    ev = pd.DataFrame({"element": d["element"], "gw": pd.to_numeric(d["gw"], errors="coerce"),
                       "g": use.astype(int), "play": (use & (am > 0)).astype(int),
                       "start": (use & (ast == 1)).astype(int),
                       "s60": (use & (ast == 1) & (am >= 60)).astype(int)})
    # one record per (player, gameweek); the cumulative sum of the EARLIER gameweeks only
    per = ev.groupby(["element", "gw"], as_index=False)[["g", "play", "start", "s60"]].max()
    per = per.sort_values(["element", "gw"])
    cum = per.groupby("element")[["g", "play", "start", "s60"]].cumsum() - per[["g", "play", "start", "s60"]]
    per = per[["element", "gw"]].join(cum)
    m = ev[["element", "gw"]].merge(per, on=["element", "gw"], how="left")
    g, st = m["g"].to_numpy(dtype=float), m["start"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["hist_gws"] = g
        d["hist_play_share"] = np.where(g > 0, m["play"].to_numpy(dtype=float) / g, np.nan)
        d["hist_starts"] = st
        d["hist_p60_given_start"] = np.where(st > 0, m["s60"].to_numpy(dtype=float) / st, np.nan)
    return d


def active_mask(d: pd.DataFrame) -> pd.Series:
    """The evaluation universe: players with season minutes before the deadline OR a non-trivial
    FPL expectation. Scoring hundreds of never-used reserves at 0 vs 0 would flatter every arm."""
    # Unknown minutes / ep_next simply do not qualify (NaN comparisons are False).
    return (pd.to_numeric(d["minutes"], errors="coerce") > 0) | (d["ep_next"] >= 0.5)
