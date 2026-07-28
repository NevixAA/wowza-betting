"""FPL fantasy FEATURES built on the calibrated prop models + the official FPL API.

All PREDICTION content (no betting, no odds) — the props model is efficient-market-priced
for BETTING but its calibrated accuracy converts to value in the fantasy contest. Everything
here is dashboard-only and isolated from the SNIPER/MARKSMAN betting pipeline.

Functions:
  differentials()       high projected points + low ownership (mini-league movers)
  best_xi()             points-maximising starting XI across legal formations (+ total cost)
  market_leaderboards() per-market calibrated probability boards (goals/brace/assist/SOT/cards)
  fixture_ticker()      per-club next-N fixtures + official FDR (easy/hard runs)
  transfer_suggestions()given an FPL team id -> sell/buy ideas (needs the live FPL entry API)
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from player_model import fpl_api
from player_model.fantasy import build_fantasy_projections, FANTASY_LEAGUE, _PARQUET
from player_model.model import load_model, predict_proba

# Legal FPL starting-XI formations: (DEF, MID, FWD) — always 1 GKP.
_FORMATIONS = [(3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 4, 1), (5, 3, 2)]
_MARKET_LABEL = {"goals": "Anytime scorer", "goals2": "2+ goals (brace)", "assists": "Assist",
                 "sot2": "2+ shots on target", "sot3": "3+ shots on target", "cards": "Booked"}


def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def _proj(proj: pd.DataFrame | None) -> pd.DataFrame:
    return proj if proj is not None else build_fantasy_projections()


# ── Differentials ───────────────────────────────────────────────────────────────

def differentials(proj: pd.DataFrame | None = None, max_owned: float = 10.0,
                  top_n: int = 20) -> pd.DataFrame:
    """High projected points AND low ownership. Needs FPL ownership (empty if unavailable)."""
    df = _proj(proj)
    if df.empty or "player_name" not in df.columns:
        return pd.DataFrame()
    fpl = fpl_api.players_df()
    if fpl.empty or "owned_pct" not in fpl.columns:
        return pd.DataFrame()
    own_full = fpl.set_index("match_key")["owned_pct"].to_dict()
    own_web  = fpl.set_index("web_key")["owned_pct"].to_dict()
    df = df.copy()
    keys = df["player_name"].map(_norm)
    df["owned_pct"] = [own_full.get(k, own_web.get(k, np.nan)) for k in keys]
    d = df[df["owned_pct"].notna() & (df["owned_pct"] <= max_owned)]
    return d.sort_values("fantasy_pts", ascending=False).head(top_n).reset_index(drop=True)


# ── Optimal starting XI ─────────────────────────────────────────────────────────

def best_xi(proj: pd.DataFrame | None = None, points_col: str = "fantasy_pts") -> dict:
    """Points-maximising legal starting XI (1 GKP + a legal DEF/MID/FWD split).

    Tries every legal formation, takes the top-N per position by projected points, keeps the
    highest-scoring formation. Returns {formation, players(df), total_pts, total_cost}. Budget
    is reported (not hard-constrained) — a hard-budget ILP is a heavier follow-up.
    """
    df = _proj(proj)
    if df.empty or "position" not in df.columns or points_col not in df.columns:
        return {}
    df = df.copy()
    # exclude injured/unavailable from a STARTING XI
    if "injured" in df.columns:
        df = df[~df["injured"].astype(bool)]
    by_pos = {p: df[df["position"] == p].sort_values(points_col, ascending=False)
              for p in ("GKP", "DEF", "MID", "FWD")}
    if any(by_pos[p].empty for p in ("GKP", "DEF", "MID", "FWD")):
        return {}
    gk = by_pos["GKP"].head(1)
    best = None
    for d, m, f in _FORMATIONS:
        if len(by_pos["DEF"]) < d or len(by_pos["MID"]) < m or len(by_pos["FWD"]) < f:
            continue
        xi = pd.concat([gk, by_pos["DEF"].head(d), by_pos["MID"].head(m), by_pos["FWD"].head(f)])
        tot = float(xi[points_col].sum())
        if best is None or tot > best["total_pts"]:
            cost = float(xi["price"].sum()) if "price" in xi.columns else float("nan")
            best = {"formation": f"{d}-{m}-{f}", "players": xi.reset_index(drop=True),
                    "total_pts": round(tot, 2), "total_cost": round(cost, 1)}
    return best or {}


# ── Per-market calibrated probability leaderboards ───────────────────────────────

def market_leaderboards(markets=("goals", "goals2", "assists", "sot2", "cards"),
                        top_n: int = 15, min_minutes: float = 45.0,
                        parquet_path: Path | None = None) -> dict:
    """{market: DataFrame[player_name, team, position, prob]} from the CALIBRATED prop models,
    on each PL player's latest form row. Only markets whose model exists are returned."""
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = df[df["league"] == FANTASY_LEAGUE].copy()
    if pl.empty:
        return {}
    pl = pl.sort_values("date").groupby("player_id", as_index=False).tail(1)
    mins = pd.to_numeric(pl.get("minutes_pg", 0), errors="coerce").fillna(0)
    pl = pl[mins >= min_minutes].copy()
    if pl.empty:
        return {}
    out = {}
    for mkt in markets:
        payload = load_model(mkt)
        if payload is None:
            continue
        try:
            probs = predict_proba(pl, payload).values
        except Exception:
            continue
        cols = [c for c in ("player_name", "team", "position") if c in pl.columns]
        board = pl[cols].copy()
        board["prob"] = np.round(probs, 3)
        out[mkt] = board.sort_values("prob", ascending=False).head(top_n).reset_index(drop=True)
    return out


# ── Fixture ticker ───────────────────────────────────────────────────────────────

def fixture_ticker(next_n: int = 5) -> pd.DataFrame:
    """One row per club with its next-N opponents + official FDR, plus mean FDR (easy run =
    low). Empty when FPL has no upcoming fixtures."""
    fdr = fpl_api.upcoming_fdr(next_n=next_n)
    if not fdr:
        return pd.DataFrame()
    rows = []
    for team, fx in fdr.items():
        row = {"team": team}
        fdrs = []
        for i, f in enumerate(fx[:next_n], 1):
            row[f"GW+{i}"] = f"{f.get('opp','')} ({'H' if f.get('home') else 'A'},{f.get('fdr','')})"
            if f.get("fdr"):
                fdrs.append(f["fdr"])
        row["avg_fdr"] = round(float(np.mean(fdrs)), 2) if fdrs else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("avg_fdr").reset_index(drop=True)


# ── Transfer suggestions (needs the live FPL entry API) ─────────────────────────

def _fetch_entry_squad(team_id: int) -> pd.DataFrame:
    """Current 15-man squad for an FPL manager (entry). Uses the current-event picks.
    Returns DataFrame[fpl_id, web_name, position, team, price, is_captain] or empty."""
    try:
        entry = fpl_api._get_json(f"{fpl_api._BASE}/entry/{int(team_id)}/")
        gw = (entry or {}).get("current_event")
        if not gw:
            boot = fpl_api.fetch_bootstrap()
            gw = next((e["id"] for e in boot.get("events", []) if e.get("is_current")), None)
        if not gw:
            return pd.DataFrame()
        picks = fpl_api._get_json(f"{fpl_api._BASE}/entry/{int(team_id)}/event/{int(gw)}/picks/")
        ids = [(p["element"], p.get("is_captain", False)) for p in (picks or {}).get("picks", [])]
    except Exception:
        return pd.DataFrame()
    if not ids:
        return pd.DataFrame()
    fpl = fpl_api.players_df()
    if fpl.empty:
        return pd.DataFrame()
    by_id = fpl.set_index("fpl_id")
    rows = []
    for eid, is_cap in ids:
        if eid in by_id.index:
            r = by_id.loc[eid]
            rows.append({"fpl_id": eid, "web_name": r["web_name"], "position": r["position"],
                         "team": r["team"], "price": r["price"], "is_captain": bool(is_cap)})
    return pd.DataFrame(rows)


def set_piece_penalty_takers(parquet_path: Path | None = None, top_per_team: int = 3) -> pd.DataFrame:
    """Per current-FPL club: likely penalty taker + set-piece goal threats, from historical
    involvement (penalties won/scored + set-piece/free-kick/headed goals). Current squad only
    (joined to live FPL). A top-3 FPL edge that's otherwise invisible."""
    df = pd.read_parquet(parquet_path or _PARQUET)
    pl = df[df["league"] == FANTASY_LEAGUE]
    if pl.empty:
        return pd.DataFrame()
    pen_cols = [c for c in ("penalty_scored", "penalty_won") if c in pl.columns]
    sp_cols  = [c for c in ("sp_goal", "fk_goal", "headed_goal") if c in pl.columns]
    g = pl.groupby("player_id")
    tot = pd.DataFrame({"player_name": g["player_name"].last()})
    tot["pens"]     = g[pen_cols].sum().sum(axis=1) if pen_cols else 0.0
    tot["sp_goals"] = g[sp_cols].sum().sum(axis=1) if sp_cols else 0.0
    tot = tot.reset_index()

    fpl = fpl_api.players_df()
    if fpl.empty:
        return pd.DataFrame()
    own_full = {r["match_key"]: (r["team"], r["position"]) for _, r in fpl.iterrows()}
    own_web  = {r["web_key"]:  (r["team"], r["position"]) for _, r in fpl.iterrows()}
    rows = []
    for _, r in tot.iterrows():
        k = _norm(r["player_name"])
        tp = own_full.get(k) or own_web.get(k)
        if not tp:
            continue
        rows.append({"player_name": r["player_name"], "team": tp[0], "position": tp[1],
                     "pens": float(r["pens"]), "sp_goals": float(r["sp_goals"])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[(out["pens"] > 0) | (out["sp_goals"] > 0)]
    return (out.sort_values(["team", "pens", "sp_goals"], ascending=[True, False, False])
               .groupby("team").head(top_per_team).reset_index(drop=True))


def transfer_suggestions(team_id: int, proj: pd.DataFrame | None = None,
                         n: int = 3) -> dict:
    """Given an FPL manager id, suggest the N weakest sells and best same-position buys.

    Sell score = low projected pts / injured / bad fixtures. Buy = highest projected pts in the
    same position not already owned. Needs the live FPL entry API (network). Returns
    {squad, sells:[{out, in_options:[...]}], note}.
    """
    squad = _fetch_entry_squad(team_id)
    if squad.empty:
        return {"error": "Could not fetch that FPL team (check the ID / FPL API reachable)."}
    df = _proj(proj)
    if df.empty:
        return {"error": "No projections available."}
    df = df.copy()
    df["_key"] = df["player_name"].map(_norm)
    proj_by_key = {k: r for k, r in df.set_index("_key").iterrows()}

    fpl = fpl_api.players_df()
    web_to_key = {}
    if not fpl.empty:
        web_to_key = {int(r["fpl_id"]): r["match_key"] for _, r in fpl.iterrows()}

    # attach projected pts to each owned player
    owned_keys = set()
    squad = squad.copy()
    pts, inj = [], []
    for _, r in squad.iterrows():
        k = web_to_key.get(int(r["fpl_id"]))
        rec = proj_by_key.get(k) if k else None
        owned_keys.add(k)
        pts.append(float(rec["fantasy_pts"]) if rec is not None else 0.0)
        inj.append(bool(rec["injured"]) if (rec is not None and "injured" in rec) else False)
    squad["proj_pts"] = pts
    squad["injured"]  = inj
    # weakest = injured first, then lowest projected pts
    squad["_sell_score"] = squad["injured"].astype(int) * 100 - squad["proj_pts"]
    sells = squad.sort_values("_sell_score", ascending=False).head(n)

    suggestions = []
    for _, s in sells.iterrows():
        pos = s["position"]
        cands = df[(df["position"] == pos) & (~df["_key"].isin(owned_keys))]
        cands = cands.sort_values("fantasy_pts", ascending=False).head(3)
        suggestions.append({
            "out": f"{s['web_name']} ({s['team']}, {pos}) — {s['proj_pts']:.2f} pts"
                   + ("  ⚠️ injured" if s["injured"] else ""),
            "in_options": [f"{r['player_name']} ({r.get('team','')}) — {r['fantasy_pts']:.2f} pts"
                           for _, r in cands.iterrows()],
        })
    return {"squad": squad, "sells": suggestions,
            "note": "Prediction-based suggestions; verify price/availability in the FPL app."}


# ── Chip strategy advisor ────────────────────────────────────────────────────

def chip_advisor(proj: pd.DataFrame | None = None, horizon: int = 8) -> dict:
    """Suggest Wildcard / Bench Boost / Triple Captain / Free Hit timing from the fixture
    calendar: per upcoming GW count DOUBLE (2 fixtures) and BLANK (0) teams + avg FDR, then
    flag chip windows. Triple-captain shortlist = top healthy projected players (easy fixtures)."""
    boot = fpl_api.fetch_bootstrap()
    fx   = fpl_api.fetch_fixtures()
    if not boot or not fx:
        return {}
    teams  = {t["id"]: t.get("short_name", "") for t in boot.get("teams", [])}
    events = boot.get("events", [])
    cur = next((e["id"] for e in events if e.get("is_current")), None)
    if cur is None:
        nxt = next((e["id"] for e in events if e.get("is_next")), 1)
        cur = nxt - 1

    gw_team: dict = {}
    for f in fx:
        ev = f.get("event")
        if ev is None or f.get("finished") or ev <= cur:
            continue
        gw_team.setdefault(ev, {}).setdefault(f.get("team_h"), []).append(f.get("team_h_difficulty"))
        gw_team[ev].setdefault(f.get("team_a"), []).append(f.get("team_a_difficulty"))

    rows, recs = [], []
    for gw in sorted(gw_team)[:horizon]:
        tm = gw_team[gw]
        dgw = sorted(teams.get(t, "") for t, fl in tm.items() if len(fl) >= 2)
        bgw = sorted(teams.get(t, "") for t in teams if t not in tm)
        all_fdr = [d for fl in tm.values() for d in fl if d]
        rows.append({"gw": gw, "n_dgw": len(dgw), "dgw_teams": dgw,
                     "n_bgw": len(bgw), "bgw_teams": bgw,
                     "avg_fdr": round(float(np.mean(all_fdr)), 2) if all_fdr else None})
        if len(dgw) >= 4:
            recs.append(f"GW{gw}: **Bench Boost / Triple Captain** window — {len(dgw)} teams double ({', '.join(dgw[:8])})")
        if len(bgw) >= 6:
            recs.append(f"GW{gw}: **Blank** — {len(bgw)} teams idle; Free Hit or navigate with a Wildcard")

    tc = []
    if proj is not None and not proj.empty:
        p = proj.copy()
        if "injured" in p.columns:
            p = p[~p["injured"].astype(bool)]
        for _, r in p.sort_values("fantasy_pts", ascending=False).head(5).iterrows():
            tc.append(f"{r['player_name']} ({r.get('team','')}) — {r['fantasy_pts']:.1f} pts"
                      + (f", FDR {r['avg_fdr']}" if "avg_fdr" in r and r["avg_fdr"] == r["avg_fdr"] else ""))
    return {"gameweeks": rows, "recommendations": recs or ["No standout chip windows in range."],
            "triple_captain": tc}


# ── Auto lineup + bench order ────────────────────────────────────────────────

def auto_lineup(team_id: int, proj: pd.DataFrame | None = None) -> dict:
    """Given an FPL team id, pick the optimal starting XI + bench order + (vice-)captain from
    the manager's 15 by projected points. Needs the live FPL entry API."""
    squad = _fetch_entry_squad(team_id)
    if squad.empty:
        return {"error": "Could not fetch that FPL team (check the ID / FPL reachable)."}
    df = _proj(proj)
    if df.empty:
        return {"error": "No projections available."}
    df = df.copy()
    df["_key"] = df["player_name"].map(_norm)
    proj_by = {r["_key"]: r for _, r in df.set_index("_key").iterrows()}
    fpl = fpl_api.players_df()
    web2key = {int(r["fpl_id"]): r["match_key"] for _, r in fpl.iterrows()} if not fpl.empty else {}

    rows = []
    for _, s in squad.iterrows():
        k = web2key.get(int(s["fpl_id"]))
        rec = proj_by.get(k) if k else None
        rows.append({"player_name": s["web_name"], "position": s["position"], "team": s["team"],
                     "fantasy_pts": float(rec["fantasy_pts"]) if rec is not None else 0.0,
                     "injured": bool(rec["injured"]) if (rec is not None and "injured" in rec) else False,
                     "price": s.get("price", np.nan)})
    sq = pd.DataFrame(rows)
    xi = best_xi(sq)                       # points-max legal XI from the 15
    if not xi:
        return {"error": "Could not form a legal XI from that squad."}
    xi_names = set(xi["players"]["player_name"])
    bench = (sq[~sq["player_name"].isin(xi_names)]
             .assign(_gk=lambda d: (d["position"] == "GKP").astype(int))   # bench GK last
             .sort_values(["_gk", "fantasy_pts"], ascending=[True, False]))
    starters = xi["players"].sort_values("fantasy_pts", ascending=False)
    cap = starters.iloc[0]["player_name"] if len(starters) else None
    vc  = starters.iloc[1]["player_name"] if len(starters) > 1 else None
    return {"formation": xi["formation"], "xi": xi["players"], "bench": bench,
            "captain": cap, "vice_captain": vc,
            "xi_pts": xi["total_pts"], "note": "Projected-points optimal; check price/availability."}


# ── Mini-league mode (rivals + effective ownership + differentials) ──────────

def mini_league_analysis(league_id: int, proj: pd.DataFrame | None = None,
                         top_managers: int = 30) -> dict:
    """Pull a classic mini-league's top managers, aggregate LEAGUE ownership, and surface the
    template (most-owned) + differentials (high projected pts, low league ownership). Live FPL."""
    from collections import Counter
    try:
        st = fpl_api._get_json(f"{fpl_api._BASE}/leagues-classic/{int(league_id)}/standings/")
        entries = [e["entry"] for e in st.get("standings", {}).get("results", [])[:top_managers]]
    except Exception:
        return {"error": "Could not fetch that league (check the ID / FPL reachable)."}
    if not entries:
        return {"error": "No managers found in that league."}
    boot = fpl_api.fetch_bootstrap()
    gw = next((e["id"] for e in boot.get("events", []) if e.get("is_current")), None)
    if gw is None:
        gw = max(1, next((e["id"] for e in boot.get("events", []) if e.get("is_next")), 2) - 1)
    own = Counter()
    for eid in entries:
        try:
            picks = fpl_api._get_json(f"{fpl_api._BASE}/entry/{eid}/event/{gw}/picks/")
            for p in picks.get("picks", []):
                own[p["element"]] += 1
        except Exception:
            continue
    n = len(entries)
    if not own:
        return {"error": "Could not read any squads in that league."}
    fpl = fpl_api.players_df()
    id2name = {int(r["fpl_id"]): r["match_key"] for _, r in fpl.iterrows()} if not fpl.empty else {}
    id2disp = {int(r["fpl_id"]): (r["web_name"], r["team"]) for _, r in fpl.iterrows()} if not fpl.empty else {}
    df = _proj(proj).copy()
    df["_key"] = df["player_name"].map(_norm)
    pj = {r["_key"]: float(r["fantasy_pts"]) for _, r in df.set_index("_key").iterrows()}

    recs = []
    for eid, cnt in own.items():
        nm, tm = id2disp.get(eid, ("", ""))
        key = id2name.get(eid)
        xp = pj.get(key, np.nan)
        recs.append({"player": nm, "team": tm, "league_own_pct": round(100 * cnt / n, 1),
                     "xpts": xp})
    r = pd.DataFrame(recs)
    template = r.sort_values("league_own_pct", ascending=False).head(15)
    diffs = r[(r["league_own_pct"] <= 25) & r["xpts"].notna()].sort_values("xpts", ascending=False).head(15)
    return {"n_managers": n, "gw": gw, "template": template, "differentials": diffs}


# ── Fantasy alerts / watchlist ───────────────────────────────────────────────

def fantasy_alerts(proj: pd.DataFrame | None = None, watchlist: list | None = None) -> dict:
    """Actionable movers: injuries/doubts, price-change candidates (net FPL transfers), and top
    projected players. `watchlist` (list of names) filters to tracked players if given."""
    df = _proj(proj).copy()
    fpl = fpl_api.players_df()
    wl = {_norm(w) for w in (watchlist or [])}

    def _filt(d, col="player_name"):
        if not wl or col not in d.columns:
            return d
        return d[d[col].map(_norm).isin(wl)]

    injuries = pd.DataFrame()
    if "injured" in df.columns:
        inj = df[df["injured"].astype(bool) | df.get("doubtful", False).astype(bool)]
        injuries = _filt(inj)[[c for c in ["player_name", "team", "position", "availability",
                                           "chance_of_playing"] if c in inj.columns]]
    risers = fallers = pd.DataFrame()
    if not fpl.empty and "transfers_in_gw" in fpl.columns:
        fpl = fpl.assign(net=fpl["transfers_in_gw"] - fpl["transfers_out_gw"])
        cols = ["web_name", "team", "position", "price", "owned_pct", "net"]
        risers = _filt(fpl.sort_values("net", ascending=False).head(12), "web_name")[cols]
        fallers = _filt(fpl.sort_values("net").head(12), "web_name")[cols]
    return {"injuries": injuries, "price_risers": risers, "price_fallers": fallers}
