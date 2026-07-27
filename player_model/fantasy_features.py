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
