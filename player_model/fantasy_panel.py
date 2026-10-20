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
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from player_model import fantasy_snapshots as fs

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
    except Exception:                                                # noqa: BLE001
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
    except Exception:                                                # noqa: BLE001
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
    s["xg"] = pd.to_numeric(s["xg"], errors="coerce").fillna(0)
    xg_for = s.groupby("team_id")["xg"].sum()
    gk = s[s["position"] == "GKP"].copy()
    gk["minutes"] = pd.to_numeric(gk["minutes"], errors="coerce").fillna(0)
    gk["xgc"] = pd.to_numeric(gk["xgc"], errors="coerce").fillna(0)
    gkt = gk.groupby("team_id").agg(xgc=("xgc", "sum"), mins=("minutes", "sum"))
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
        mu = float(np.nanmean(raw)) if raw.notna().any() else 1.35
        if not np.isfinite(mu):
            mu = 1.35
        t[k] = ((raw.fillna(mu) * t["games"] + mu * prior) / (t["games"] + prior))
    t.index.name = "team_id"
    return t[["games", "gf_pg", "ga_pg", "xg_for_pg", "xga_pg"]].reset_index()


def fixture_lambdas(team_id, opp_id, home, ts: pd.DataFrame) -> tuple[float, float]:
    """(team goal expectancy, opponent goal expectancy) for one fixture, multiplicative model."""
    t = ts.set_index("team_id")
    lg = float(t["xg_for_pg"].mean()) if len(t) else 1.35
    try:
        att = 0.5 * t.at[team_id, "xg_for_pg"] + 0.5 * t.at[team_id, "gf_pg"]
        dfn = 0.5 * t.at[team_id, "xga_pg"] + 0.5 * t.at[team_id, "ga_pg"]
        o_att = 0.5 * t.at[opp_id, "xg_for_pg"] + 0.5 * t.at[opp_id, "gf_pg"]
        o_dfn = 0.5 * t.at[opp_id, "xga_pg"] + 0.5 * t.at[opp_id, "ga_pg"]
    except KeyError:
        return lg, lg
    ha = 1.10 if home else 0.91
    lam_for = att * o_dfn / lg * ha
    lam_against = o_att * dfn / lg / ha
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
def wowza_asof(log: pd.DataFrame, deadline) -> pd.DataFrame:
    if log is None or log.empty:
        return pd.DataFrame(columns=["key", "wowza_cond", "wowza_p_start", "wowza_uncond"])
    lg = log[log["snapshot_ts"] < deadline]
    if lg.empty:
        return pd.DataFrame(columns=["key", "wowza_cond", "wowza_p_start", "wowza_uncond"])
    last = lg.sort_values("snapshot_ts").groupby("player_name", as_index=False).last()
    # Only a projection made for THIS window counts: one older than 14 days is not "as of".
    last = last[last["snapshot_ts"] >= deadline - pd.Timedelta(days=14)]
    base = pd.to_numeric(last["proj_fixture_adj"], errors="coerce")
    ps = pd.to_numeric(last["p_start"], errors="coerce").clip(0, 1)
    return pd.DataFrame({"key": last["player_name"].map(norm_name), "team_log": last["team"],
                         "wowza_cond": base, "wowza_p_start": ps, "wowza_uncond": base * ps})


def _load_log(path: Path | None = None) -> pd.DataFrame:
    p = path or PROJ_LOG
    if not p.exists():
        return pd.DataFrame()
    lg = pd.read_csv(p, usecols=lambda c: c in {"snapshot_ts", "player_name", "team",
                                                "proj_fixture_adj", "p_start"})
    lg["snapshot_ts"] = pd.to_datetime(lg["snapshot_ts"], utc=True, errors="coerce")
    return lg.dropna(subset=["snapshot_ts"])


def _num(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)


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
    d["n_fix"] = d["n_fix"].fillna(0).astype(int)
    tg = d["team_games"].replace(0, np.nan)
    mins = _num(d, "minutes").fillna(0)
    d["min_per_team_game"] = (mins / tg).clip(0, 90)
    d["start_rate"] = (_num(d, "starts").fillna(0) / tg).clip(0, 1)
    d["pts_per_team_game"] = _num(d, "total_points").fillna(0) / tg
    n90 = (mins / 90.0)
    prior90 = 3.0    # shrink per-90 rates with three full games of league-average prior
    pos_xg = {"FWD": 0.38, "MID": 0.17, "DEF": 0.05, "GKP": 0.0}
    pos_xa = {"FWD": 0.12, "MID": 0.14, "DEF": 0.06, "GKP": 0.01}
    d["xg90"] = (_num(d, "xg").fillna(0) + d["position"].map(pos_xg) * prior90) / (n90 + prior90)
    d["xa90"] = (_num(d, "xa").fillna(0) + d["position"].map(pos_xa) * prior90) / (n90 + prior90)
    d["xgi90"] = d["xg90"] + d["xa90"]
    d["bps90"] = (_num(d, "bps").fillna(0) + 15 * prior90) / (n90 + prior90)
    d["bonus_per_start"] = (_num(d, "bonus").fillna(0) + 0.3) / (_num(d, "starts").fillna(0) + 1)
    d["dc90"] = (_num(d, "def_contrib").fillna(0)) / (n90 + prior90) if "def_contrib" in d else 0.0
    d["saves90"] = _num(d, "saves").fillna(0) / (n90 + prior90)
    d["threat_pg"] = _num(d, "threat").fillna(0) / tg
    d["creativity_pg"] = _num(d, "creativity").fillna(0) / tg
    st = d["status"].astype(str)
    ch = _num(d, "chance_next")
    d["chance_next_f"] = ch.where(ch.notna(), np.where(st == "a", 100.0, 0.0)) / 100.0
    d["is_out"] = st.isin(["i", "s", "u", "n"]).astype(int)
    d["is_doubt"] = (st == "d").astype(int)
    d["has_news"] = d["news"].fillna("").astype(str).str.len().gt(0).astype(int)
    d["pen_first"] = (_num(d, "pen_order") == 1).astype(int)
    d["pen_any"] = _num(d, "pen_order").notna().astype(int)
    d["set_piece"] = (_num(d, "corner_order").le(2) | _num(d, "dfk_order").le(2)).astype(int)
    d["ownership"] = _num(d, "selected_pct")
    d["net_transfers_k"] = (_num(d, "transfers_in_event").fillna(0)
                            - _num(d, "transfers_out_event").fillna(0)) / 1000.0
    d["ppm"] = d["pts_per_team_game"] / d["price"].replace(0, np.nan)
    for p in POSITIONS:
        d[f"pos_{p}"] = (d["position"] == p).astype(int)
    d["is_defensive"] = d["position"].isin(["GKP", "DEF"]).astype(int)
    d["cs_exp"] = d["cs_prob"].fillna(0) * d["position"].map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
    d["goal_pts"] = d["position"].map({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}).astype(float)
    lg_avg = float(ts["xg_for_pg"].mean()) if len(ts) else 1.35
    lam_rel = (d["lam_for"] / d["n_fix"].replace(0, np.nan)) / (lg_avg or 1.35)
    d["attack_fixture"] = lam_rel.fillna(1.0)
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
    d["key"] = d["player_name"].map(norm_name)
    d["key_web"] = d["web_name"].map(norm_name)
    if wowza is not None and not wowza.empty:
        w = wowza.drop_duplicates("key")
        m = d[["key"]].merge(w, on="key", how="left")
        miss = m["wowza_cond"].isna().values
        if miss.any():
            m2 = d[["key_web"]].merge(w.rename(columns={"key": "key_web"}), on="key_web", how="left")
            for c in ("wowza_cond", "wowza_p_start", "wowza_uncond"):
                m.loc[miss, c] = m2.loc[miss, c].values
        for c in ("wowza_cond", "wowza_p_start", "wowza_uncond"):
            d[c] = m[c].values
    else:
        d["wowza_cond"] = d["wowza_p_start"] = d["wowza_uncond"] = np.nan
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
    return pd.concat(out, ignore_index=True)


def active_mask(d: pd.DataFrame) -> pd.Series:
    """The evaluation universe: players with season minutes before the deadline OR a non-trivial
    FPL expectation. Scoring hundreds of never-used reserves at 0 vs 0 would flatter every arm."""
    return (pd.to_numeric(d["minutes"], errors="coerce").fillna(0) > 0) | (d["ep_next"].fillna(0) >= 0.5)
