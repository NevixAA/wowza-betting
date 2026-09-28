"""Fantasy adapter — FPL points, the no-vig track. Lives in v9 but is NOT betting."""
from __future__ import annotations

import pandas as pd

from dashboard_data.core import (V9_DIR, Freshness, freshness_from_timestamp,
                                 newest_date, read_csv, read_json)

#: Fantasy is scored in POINTS, never in units or ROI. Mixing the two scales is the fastest way
#: to make a Fantasy number look like a betting result, which it is not.
UNIT = "FPL points"

OUT = V9_DIR / "output"


def tips() -> pd.DataFrame:
    """The current ranked board. 38 columns, of which the page shows the few that decide."""
    return read_csv(OUT / "fantasy_tips.csv")


def projection_log() -> pd.DataFrame:
    """Every projection ever published, stamped with the gameweek it was FOR.

    The gameweek stamp is the keystone. Without it a projection cannot be matched to the result
    it was predicting, and settlement silently compares the wrong things.
    """
    return read_csv(OUT / "fantasy_projection_log.csv")


def ledger() -> pd.DataFrame:
    """Settled projections joined to actual points."""
    return read_csv(OUT / "fantasy_ledger.csv")


def benchmark() -> dict:
    """Wowza's projection against FPL's own ep_next, per gameweek, on the same players.

    This is the honest headline and it is currently a LOSS: FPL's published expectation beats
    ours on MAE and on rank correlation in every settled gameweek. It stays on the page, because
    a benchmark shown only when it is won is not a benchmark. What Wowza adds is the things
    ep_next does not carry — start probability, the fixture split, and an explicit bias number.
    """
    j = read_json(OUT / "fantasy_performance.json")
    rows = j.get("by_gameweek") or []
    if not rows:
        return {"available": False, "why": "no settled gameweeks compared yet"}
    flat = []
    for r in rows:
        w, f = r.get("wowza") or {}, r.get("fpl_ep_next") or {}
        flat.append({"gw": r.get("gw"), "n": r.get("n"),
                     "wowza_mae": w.get("mae"), "fpl_mae": f.get("mae"),
                     "wowza_spearman": w.get("spearman"), "fpl_spearman": f.get("spearman"),
                     "wowza_bias": w.get("bias"), "fpl_bias": f.get("bias"),
                     "wowza_better": bool(r.get("wowza_better_mae")),
                     "top10_overlap_wowza": r.get("top10_overlap_wowza"),
                     "top10_overlap_fpl": r.get("top10_overlap_fpl"),
                     "top10_of": r.get("top10_of")})
    d = pd.DataFrame(flat)
    won = int(d["wowza_better"].sum())
    return {"available": True, "unit": UNIT,
            "comparable": bool(j.get("comparable")),
            "gameweeks": int(len(d)),
            "gameweeks_wowza_better": won,
            "verdict": ("Wowza does NOT beat FPL's own ep_next"
                        if won * 2 <= len(d) else "Wowza ahead of ep_next"),
            "wowza_mae": float(d["wowza_mae"].mean()),
            "fpl_mae": float(d["fpl_mae"].mean()),
            "wowza_bias": float(d["wowza_bias"].mean()),
            "excluded": j.get("excluded_gameweeks") or j.get("excluded") or [],
            "table": d.to_dict("records")}


def settlement() -> dict:
    """Projected vs actual by gameweek, straight off the ledger.

    IT SCORES THE UNCONDITIONAL NUMBER — projection × P(start) — because that is the number the
    product actually shows. Scoring the conditional projection instead is not a fair test: 46% of
    real gameweek scores are zero because the player did not play, and a conditional projection
    never claimed to predict those. The difference is large and in the scorer's favour if you get
    it backwards (MAE 2.16 unconditional vs 3.43 conditional), so this reproduces exactly what
    fantasy_settle.py does rather than picking its own column — one bias per gameweek across the
    whole dashboard, not two that disagree.

    Any gameweek the settler rejected carries its reason rather than disappearing. Two have been
    rejected so far and both were instrument faults, not model faults: one took its "actual" from
    the first post-deadline snapshot instead of a settled one (+4.03 points of pure bias), and one
    read a season-rollover total that made a gameweek average −73. A settlement chart that quietly
    drops those reads as "no data"; one that keeps them reads as a broken model. Neither is true.
    """
    d = ledger()
    if d.empty or "actual_points" not in d.columns:
        return {"available": False, "why": "no settled gameweeks in fantasy_ledger.csv"}
    d = d[pd.to_numeric(d["actual_points"], errors="coerce").notna()].copy()
    if d.empty:
        return {"available": False, "why": "ledger has rows but no settled actuals yet"}
    d["actual_points"] = pd.to_numeric(d["actual_points"], errors="coerce")
    base = "proj_fixture_adj" if "proj_fixture_adj" in d.columns else "proj_pts_per_game"
    d[base] = pd.to_numeric(d[base], errors="coerce")
    if "p_start" in d.columns:
        d["proj_uncond"] = d[base] * pd.to_numeric(d["p_start"], errors="coerce").clip(0, 1)
        proj_col = "proj_uncond"
    else:
        proj_col = base
    d["err"] = d[proj_col] - d["actual_points"]
    d["err_cond"] = d[base] - d["actual_points"]
    g = d.groupby("gw").agg(n=("actual_points", "size"),
                            projected=(proj_col, "mean"),
                            actual=("actual_points", "mean"),
                            mae=("err", lambda s: s.abs().mean()),
                            bias=("err", "mean"),
                            mae_conditional=("err_cond", lambda s: s.abs().mean()),
                            ).round(3).reset_index()
    perf = read_json(OUT / "fantasy_performance.json")
    return {"available": True, "unit": UNIT,
            "projection_column": proj_col,
            "conditional_column": base,
            "scoring_note": ("scored on projection × P(start), the number the product shows; "
                             "the conditional MAE is carried alongside, never instead"),
            "gameweeks_settled": int(len(g)),
            "players_settled": int(len(d)),
            "exclusions": perf.get("excluded_gameweeks") or perf.get("excluded") or [],
            "table": g.to_dict("records")}


def health() -> dict:
    """Feed health, as the Fantasy pipeline itself recorded it.

    fantasy_health.json is written by the run, so its ages come from recorded fetch times rather
    than from stat(). That distinction is not pedantry: fpl_api._cached_fetch aged its cache by
    file mtime and served a six-week-old squad list through it, offering an injured player as a
    captaincy pick, because `git checkout` resets mtime on every CI run. A feed whose age cannot
    be established is reported UNKNOWN and treated as old, never as fresh.
    """
    j = read_json(OUT / "fantasy_health.json")
    if not j:
        return {"available": False, "unit": UNIT, "why": "fantasy_health.json not written yet"}
    feeds = {}
    for name, f in (j.get("feeds") or {}).items():
        age = f.get("age_hours")
        state = str(f.get("state") or "UNKNOWN")
        feeds[name] = Freshness(
            "UNKNOWN" if state.startswith("UNKNOWN") else state,
            age,
            (f"{age:.0f}h ago" if isinstance(age, (int, float)) else "age unknown"),
            f.get("file", name))
    return {"available": True, "unit": UNIT,
            "generated_at": j.get("generated_at"),
            "run": freshness_from_timestamp(j.get("generated_at"), "daily",
                                            "fantasy_health.json"),
            "feeds": feeds,
            "coverage": j.get("coverage") or {},
            "alerts": j.get("alerts") or j.get("warnings") or []}


def availability_split() -> dict:
    """How many ranked players are actually startable.

    The board used to put eight unplayable players in a top twenty — injured, suspended, or no
    longer at a Premier League club. Start probability now gates the ranking, and the split is
    published rather than assumed, because "nobody is unavailable" and "the availability check
    returned nothing" look identical on a page that only shows the survivors.
    """
    p = tips()
    if p.empty or "p_start" not in p.columns:
        return {"available": False, "why": "no start probabilities in fantasy_tips.csv"}
    s = pd.to_numeric(p["p_start"], errors="coerce").fillna(0.0)
    top = s.iloc[:20] if len(s) >= 20 else s
    return {"available": True, "total": int(len(p)),
            "nailed": int((s >= 0.85).sum()),
            "rotation_risk": int(((s >= 0.60) & (s < 0.85)).sum()),
            "doubtful": int(((s > 0.0) & (s < 0.60)).sum()),
            "unavailable": int((s <= 0.0).sum()),
            "unavailable_in_top20": int((top <= 0.0).sum())}


def board(n: int = 25) -> pd.DataFrame:
    """Top n as the model actually ranks them, with the columns a human decides on.

    IT SORTS BY overall_rank, NEVER BY A POINTS COLUMN. overall_rank is the availability-gated
    order (unconditional points, hard zero for anyone ruled out); fixture_adj_pts and fantasy_pts
    are CONDITIONAL — what the player scores *if he plays*. Sorting the board on either of those
    is precisely the bug that put eight unplayable players in a top twenty, and it comes back the
    moment anyone re-sorts on the number that looks biggest. Hugo Ekitiké is rank 105 on a
    conditional 7.26 because he is injured; a points sort puts him second.
    """
    p = tips()
    if p.empty:
        return p
    cols = [c for c in ("overall_rank", "player_name", "team", "position", "price",
                        "xpts_uncond", "fixture_adj_pts", "p_start", "availability",
                        "start_confidence", "avg_fdr", "n_fixtures_next", "fpl_ep_next",
                        "owned_pct", "value", "captain_pick") if c in p.columns]
    if not cols:
        return p.head(n)
    if "overall_rank" in p.columns:
        return p.sort_values("overall_rank").head(n)[cols].reset_index(drop=True)
    # No published rank — fall back to the UNCONDITIONAL number, never the conditional one.
    sort_col = "xpts_uncond" if "xpts_uncond" in p.columns else cols[0]
    return p.sort_values(sort_col, ascending=False).head(n)[cols].reset_index(drop=True)


def team_history() -> dict:
    """Premier League team-match history — Fantasy's own source, read by nothing else.

    It is a separate file on purpose. af_history feeds the TEAM MODEL'S TRAINING, so adding the
    Premier League there would change what v9 trains on — a training-source change that has
    nothing to do with Fantasy and would need its own authorisation. The betting universe is
    unchanged by this file's existence.
    """
    p = OUT / "pl_team_history.parquet"
    if not p.exists():
        return {"available": False, "why": "pl_team_history.parquet not built"}
    try:
        d = pd.read_parquet(p)
    except Exception as e:                                            # noqa: BLE001
        return {"available": False, "why": f"unreadable ({type(e).__name__})"}
    return {"available": True,
            "rows": int(len(d)), "fixtures": int(d["fixture_id"].nunique()),
            "teams": int(d["team"].nunique()),
            "seasons": sorted(str(x) for x in d["season"].unique()),
            "freshness": freshness_from_timestamp(newest_date(d, "date"), "weekly",
                                                  "pl_team_history.parquet"),
            "coverage": {c: float(d[c].notna().mean())
                         for c in ("shots", "corners", "possession") if c in d.columns}}
