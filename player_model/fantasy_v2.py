"""Fantasy V2 — FPL-anchored projections, benchmark, captaincy, transfers and fixture matrix.

    python -m player_model.fantasy_v2              # run after the FPL bootstrap/fixtures refresh

WHAT IT IS. A batch job (fantasy_refresh.yml) that turns the FPL caches already on disk into
small artifacts the dashboard only READS:

    output/fantasy_fpl_snapshots.csv      point-in-time FPL archive (one row per gameweek x player)
    output/fantasy_v2_projections.csv     next-gameweek board: mean, distribution, minutes, roles,
                                          3-GW / 5-GW projections, decomposition
    output/fantasy_v2_benchmark.json      walk-forward benchmark of every arm vs FPL ep_next,
                                          acceptance decision, bias diagnosis, calibration
    output/fantasy_v2_oos.csv             every out-of-sample prediction (audit trail)
    output/fantasy_v2_scoreboard.csv      chronological GW x arm scoreboard (accuracy, top-10/15,
                                          captain, bias, minutes calibration)
    output/fantasy_v2_scoreboard_reliability.csv   binned P(60+ minutes) reliability per GW
    output/fantasy_v2_summary.json        command-centre header + captaincy (SAFE/BALANCED/UPSIDE)
    output/fantasy_v2_transfers.csv       OUT -> IN moves ranked by projected gain
    output/fantasy_v2_fixture_matrix.csv  20 teams x next 5 gameweeks

THE MEAN FORECAST IS FPL's ep_next UNTIL THE RESIDUAL EARNS ITS PLACE. The residual is computed and
published every run (column `wowza_residual`) but only added to `final_xpts` when (a) the
pre-registered acceptance gate passes AND (b) FANTASY_V2_RESIDUAL_MODE=auto. Default is `shadow`,
so a gate flipping to GO is reported for the owner to approve rather than silently changing the
board.

No API calls are made here, and no file mtime is consulted: freshness comes from the recorded
fetch timestamp in fpl_cache_meta.json.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from player_model import fantasy_panel as fp
from player_model import fantasy_residual as R
from player_model import fantasy_snapshots as fs

log = logging.getLogger(__name__)

OUT = Path(__file__).resolve().parents[1] / "output"
PROJECTIONS = OUT / "fantasy_v2_projections.csv"
BENCHMARK = OUT / "fantasy_v2_benchmark.json"
OOS = OUT / "fantasy_v2_oos.csv"
SCOREBOARD = OUT / "fantasy_v2_scoreboard.csv"
SCOREBOARD_RELIABILITY = OUT / "fantasy_v2_scoreboard_reliability.csv"
SUMMARY = OUT / "fantasy_v2_summary.json"
TRANSFERS = OUT / "fantasy_v2_transfers.csv"
FIXTURE_MATRIX = OUT / "fantasy_v2_fixture_matrix.csv"
LEDGER = OUT / "fantasy_ledger.csv"

CALC_VERSION = "fantasy_v2-1.0.0"
#: the residual arm chosen to run in production. Declared, not re-selected each run.
PRODUCTION_ARM = "LINEAR"
HORIZON = 5
#: transfer planner: a realistic swap may cost up to this much more than the outgoing player
BUDGET_SLACK = 0.5

#: fixture multiplier convention — share of a position's points that scales with team attack /
#: with clean-sheet chance. A CONVENTION (stated), not fitted: three gameweeks cannot fit it.
ATTACK_SHARE = {"FWD": 0.55, "MID": 0.45, "DEF": 0.20, "GKP": 0.05}
DEFENCE_SHARE = {"FWD": 0.00, "MID": 0.08, "DEF": 0.40, "GKP": 0.45}


def residual_mode() -> str:
    m = os.getenv("FANTASY_V2_RESIDUAL_MODE", "shadow").strip().lower()
    return m if m in {"shadow", "auto"} else "shadow"


# ── benchmark ─────────────────────────────────────────────────────────────────────────────────
def published_ledger_benchmark(path: Path | None = None) -> dict:
    """Recompute the published head-to-head (fantasy_ledger.csv) exactly as Pro scored it."""
    p = path or LEDGER
    if not p.exists():
        return {"available": False}
    L = pd.read_csv(p).dropna(subset=["actual_points"])
    w = pd.to_numeric(L["proj_fixture_adj"], errors="coerce") * \
        pd.to_numeric(L["p_start"], errors="coerce").clip(0, 1)
    y = L["actual_points"].to_numpy(float)
    by = []
    for gw, g in L.assign(_w=w).groupby("gw"):
        by.append({"gw": int(gw), "n": int(len(g)),
                   "WOWZA": R.metrics(g["actual_points"], g["_w"]),
                   "FPL": R.metrics(g["actual_points"], g["fpl_ep_next"])})
    return {"available": True, "source": "output/fantasy_ledger.csv",
            "gameweeks": sorted(int(g) for g in L["gw"].unique()),
            "WOWZA": R.metrics(y, w.to_numpy(float)),
            "FPL": R.metrics(y, L["fpl_ep_next"].to_numpy(float)),
            "fpl_won_every_gw": all(r["FPL"]["mae"] < r["WOWZA"]["mae"] for r in by),
            "by_gw": by}


def _examples(oos: pd.DataFrame, k: int = 8) -> dict:
    d = oos.dropna(subset=[PRODUCTION_ARM]).copy()
    d["gain"] = (d["FPL"] - d["actual_points"]).abs() - (d[PRODUCTION_ARM] - d["actual_points"]).abs()
    cols = ["gw", "status", "player_name", "team", "position", "ep_next", PRODUCTION_ARM,
            "actual_points", "actual_minutes", "chance_next_f", "start_rate", "min_last", "gain"]
    fmt = lambda f: f[cols].round(2).rename(columns={PRODUCTION_ARM: "v2"}).to_dict("records")
    return {"helps": fmt(d.nlargest(k, "gain")), "hurts": fmt(d.nsmallest(k, "gain"))}


def run_benchmark(panel: pd.DataFrame, extras: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Walk-forward benchmark. If `extras` is a dict it also receives the GW scoreboard frames
    ("scoreboard", "reliability") so run() can write them as their own small artifacts."""
    oos = R.walk_forward(panel)
    if oos.empty:
        return {"available": False, "why": "fewer than two settled gameweeks"}, oos
    res = {"available": True, "calculation_version": CALC_VERSION,
           "production_arm": PRODUCTION_ARM,
           "arms": {"FPL": "FPL ep_next (champion)",
                    "WOWZA": "existing standalone Wowza projection x P(start)",
                    "LINEAR": "ep_next + OLS residual on 9 features",
                    "RIDGE": "ep_next + ridge residual on all features (alpha by leave-one-GW-out)",
                    "HGB": "ep_next + shallow regularised gradient-boosting residual",
                    "RANK": "not built — 3 gameweeks are too few query groups for learning-to-rank"},
           "populations": {
               "ACTIVE": "players with season minutes before the deadline or ep_next >= 0.5",
               "LEDGER": "players the existing Wowza projection covered (the published head-to-head)"}}
    res["scores"] = {pop: R.score_arms(oos, pop) for pop in ("ACTIVE", "LEDGER")}
    res["acceptance"] = {arm: R.acceptance(res["scores"]["ACTIVE"], arm) for arm in R.RESIDUAL_ARMS}
    led_v = res["scores"]["LEDGER"]["vs_fpl_final_only"].get(PRODUCTION_ARM, {})
    prod = dict(res["acceptance"][PRODUCTION_ARM])
    # second condition, fixed in advance: the residual must not lose materially on the players the
    # old projection covered (the ones managers actually own).
    if prod["decision"] == "GO" and (led_v.get("mae_minus_fpl") or 0) > 0.05:
        prod["decision"] = "NO-GO"
        prod["reasons"].append("loses > 0.05 MAE to FPL on the LEDGER population")
    res["decision"] = {
        "replace_old_wowza_projection_with_fpl_anchored_v2": "GO",
        "apply_residual_to_mean": prod["decision"],
        "detail": prod,
        "residual_mode": residual_mode(),
        "applied": bool(prod["decision"] == "GO" and residual_mode() == "auto"),
    }
    res["published_ledger"] = published_ledger_benchmark()
    res["wowza_bias"] = R.wowza_bias_breakdown(panel)
    # calibration of the minutes mixture + distribution, walk-forward
    prep = R.prepare(panel)
    settled = prep[prep["status"].isin(["FINAL", "PROVISIONAL"]) & fp.active_mask(prep)]
    mm_for = R.minutes_models_for(panel, oos["gw"].unique())
    res["distribution"] = {
        "mean_fpl": R.evaluate_distribution(oos, mm_for, "FPL"),
        f"mean_{PRODUCTION_ARM.lower()}": R.evaluate_distribution(oos, mm_for, PRODUCTION_ARM)}
    # feature importance: permutation on the latest fold, and coefficients on all settled data
    last = int(oos["gw"].max())
    tr = settled[(settled["gw"] < last) & (settled["status"] == "FINAL")]
    te = settled[settled["gw"] == last]
    res["importance"] = {
        "permutation_test_gw": last,
        PRODUCTION_ARM: R.permutation_importance_groups(tr, te, PRODUCTION_ARM),
        "HGB": R.permutation_importance_groups(tr, te, "HGB"),
        "RIDGE": R.permutation_importance_groups(tr, te, "RIDGE")}
    full = settled
    lin = R.LinearResidual().fit(full, (full["actual_points"] - full["ep_next"]).to_numpy(float))
    res["linear_coefficients_all_settled"] = lin.coefficients().round(4).to_dict("records")
    rid = R.RidgeResidual().fit(full, (full["actual_points"] - full["ep_next"]).to_numpy(float),
                                groups=full["gw"].to_numpy())
    res["ridge_top_coefficients"] = rid.coefficients().head(15).round(4).to_dict("records")
    res["ridge_alpha_all_settled"] = rid.alpha_
    res["examples"] = _examples(oos)
    res["fpl_overdispersion"] = _overdispersion(oos)
    if extras is not None:
        try:
            extras["scoreboard"], extras["reliability"] = R.gw_scoreboard(oos, mm_for)
        except Exception as e:                                       # noqa: BLE001
            log.warning(f"[fantasy_v2] scoreboard failed ({e})")
    return res, oos


def _overdispersion(oos: pd.DataFrame) -> list:
    d = oos.copy()
    d["band"] = pd.cut(d["ep_next"], [-1, 0.5, 2, 4, 6, 99], labels=["<=0.5", "0.5-2", "2-4",
                                                                     "4-6", ">6"])
    g = d.groupby("band", observed=True).agg(
        n=("actual_points", "size"), ep_next=("ep_next", "mean"),
        actual=("actual_points", "mean"), v2=(PRODUCTION_ARM, "mean"))
    return g.round(3).reset_index().to_dict("records")


# ── production projection for the pending gameweek ───────────────────────────────────────────
def _future_context(tfl: pd.DataFrame, ts: pd.DataFrame, gws: list) -> pd.DataFrame:
    rows = []
    for gw in gws:
        f = tfl[tfl["event"] == gw]
        for r in f.itertuples():
            lf, la = fp.fixture_lambdas(r.team_id, r.opp_id, bool(r.home), ts)
            rows.append({"gw": gw, "team_id": int(r.team_id), "opp_id": int(r.opp_id),
                         "home": bool(r.home), "fdr": r.fdr, "lam_for": lf, "lam_against": la,
                         "cs_prob": math.exp(-la)})
    return pd.DataFrame(rows, columns=["gw", "team_id", "opp_id", "home", "fdr", "lam_for",
                                       "lam_against", "cs_prob"])


def _fixture_mult(position: str, lam_for: float, cs: float, lam_ref: float, cs_ref: float) -> float:
    a, d = ATTACK_SHARE.get(position, 0.3), DEFENCE_SHARE.get(position, 0.2)
    m = (1 - a - d) + a * (lam_for / lam_ref) + d * (cs / cs_ref)
    return float(np.clip(m, 0.55, 1.6))


def project(panel: pd.DataFrame, bench: dict) -> pd.DataFrame:
    """The pending gameweek's board."""
    prep = R.prepare(panel)
    pend = prep[prep["status"] == "PENDING"].copy()
    if pend.empty:
        return pd.DataFrame()
    pend = pend[fp.active_mask(pend)].copy()
    train = prep[prep["status"].isin(["FINAL", "PROVISIONAL"]) & fp.active_mask(prep)]
    r = (train["actual_points"] - train["ep_next"]).to_numpy(float)
    lin = R.LinearResidual().fit(train, r)
    mm = R.MinutesModel().fit(train)
    gw = int(pend["gw"].iloc[0])

    pend["fpl_ep_next"] = pend["ep_next"]
    # floored so FPL + residual >= 0 (same rule as the walk-forward that validated it)
    pend["wowza_residual"] = np.round(np.maximum(lin.predict(pend), -pend["ep_next"].to_numpy(float)), 3)
    contrib = lin.contributions(pend)
    applied = bool(bench.get("decision", {}).get("applied"))
    pend["residual_applied"] = applied
    pend["final_xpts"] = (pend["ep_next"] + (pend["wowza_residual"] if applied else 0.0)).round(3)
    pend["v2_candidate_xpts"] = (pend["ep_next"] + pend["wowza_residual"]).round(3)
    for c in contrib.columns:
        pend[f"adj_{c}"] = contrib[c].round(3)

    P = mm.predict_proba(pend)
    pend["p_0"], pend["p_1_59"], pend["p_60"] = P["m0"].round(3), P["m1_59"].round(3), P["m60"].round(3)
    ev = R.event_probs(pend, P, mm.cond_minutes_)
    for c in ev.columns:
        pend[c] = ev[c]
    dists = [R.dist_summary(R.point_distribution(float(mu), P.loc[i].to_dict(), pos, mm))
             for i, mu, pos in zip(pend.index, pend["final_xpts"], pend["position"])]
    D = pd.DataFrame(dists, index=pend.index)
    for c in D.columns:
        pend[c] = D[c]
    pend["ceiling"] = pend["p90"]
    # the same distribution around the SHADOW mean (FPL + residual), for the side-by-side view
    D2 = pd.DataFrame([R.dist_summary(R.point_distribution(float(mu), P.loc[i].to_dict(), pos, mm))
                       for i, mu, pos in zip(pend.index, pend["v2_candidate_xpts"], pend["position"])],
                      index=pend.index)
    for c in ("p6", "p10", "p15", "p90", "median"):
        pend[f"cand_{c}"] = D2[c]
    rel = pend["sd"] / pend["final_xpts"].clip(lower=0.5)
    pend["uncertainty_score"] = (100 * (1 - np.exp(-rel))).round(0)
    pend["uncertainty"] = pd.cut(pend["uncertainty_score"], [-1, 45, 60, 101],
                                 labels=["low", "medium", "high"]).astype(str)
    pend["minutes_disagreement"] = ((pend["ep_next"] >= 3) & (pend["p_60"] < 0.4)) | \
                                   ((pend["ep_next"] < 1) & (pend["p_60"] > 0.7))
    pend = _multi_gw(pend, panel, gw)
    pend["value_per_m"] = (pend["final_xpts"] / pend["price"]).round(3)
    pend["value_5gw_per_m"] = (pend["xpts_5gw"] / pend["price"]).round(3)
    pend["confidence"] = np.select(
        [(pend["p_start"] >= 0.85) & (pend["uncertainty"] != "high") & ~pend["minutes_disagreement"],
         (pend["p_start"] >= 0.6) & ~pend["minutes_disagreement"]],
        ["High", "Medium"], "Low")
    pend["risk"] = np.select(
        [pend["is_out"] == 1, pend["is_doubt"] == 1, pend["p_start"] < 0.6, pend["uncertainty"] == "high"],
        ["Unavailable", "Doubtful", "Rotation", "Volatile"], "Low")
    keep = ["gw", "element", "player_name", "web_name", "team", "team_id", "position", "price",
            "status", "chance_next_f", "news", "ownership", "net_transfers_k", "fpl_ep_next",
            "wowza_residual", "residual_applied", "final_xpts", "v2_candidate_xpts",
            "adj_calibration", "adj_availability", "adj_minutes", "adj_attacking_role",
            "adj_fixture", "adj_wowza_projection", "adj_base",
            "p_0", "p_1_59", "p_60", "p_start", "p_play", "exp_minutes", "goal_prob",
            "assist_prob", "cs_prob", "team_cs_prob", "exp_bonus", "team_goal_exp",
            "median", "p25", "p75", "p90", "ceiling", "sd", "p_blank", "p6", "p10", "p15",
            "uncertainty_score", "uncertainty", "minutes_disagreement", "confidence", "risk",
            "xg90", "xa90", "pen_first", "pen_order", "set_piece", "start_rate", "min_last",
            "n_fix", "fdr", "next_fixture", "fixture_run", "fdr_run", "xpts_3gw", "xpts_5gw",
            "value_per_m", "value_5gw_per_m", "wowza_uncond",
            "cand_p6", "cand_p10", "cand_p15", "cand_p90", "cand_median"]
    out = pend[[c for c in keep if c in pend.columns]].copy()
    out = out.sort_values("final_xpts", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["pos_rank"] = out.groupby("position")["final_xpts"].rank(ascending=False,
                                                                 method="first").astype(int)
    return out


def _team_meta() -> dict:
    try:
        b = json.loads(fp.BOOTSTRAP.read_text(encoding="utf-8"))
        return {t["id"]: {"name": t.get("name"), "short": t.get("short_name")} for t in b["teams"]}
    except Exception:                                                # noqa: BLE001
        return {}


def _multi_gw(pend: pd.DataFrame, panel: pd.DataFrame, gw: int) -> pd.DataFrame:
    """3-GW and 5-GW projections from a per-fixture rate x fixture multiplier x availability path.

    rate      = points per fixture IF he plays, backed out of the next-gameweek mean
    fixture   = position-weighted attack / clean-sheet multiplier (convention, see top)
    available = next-GW play probability relaxing toward his long-run play rate (half-life 1 GW)
    """
    tfl = fp.team_fixture_long(fp.load_fixtures())
    snap = panel[panel["status"] == "PENDING"]
    ts = fp.team_strength_asof(snap, tfl, pd.Timestamp(snap["deadline"].iloc[0]))
    gws = list(range(gw, gw + HORIZON))
    fut = _future_context(tfl, ts, gws)
    meta = _team_meta()
    lam_ref = float(fut["lam_for"].mean()) if len(fut) else 1.35
    cs_ref = float(fut["cs_prob"].mean()) if len(fut) else 0.28
    by_team = {t: g for t, g in fut.groupby("team_id")}
    x3, x5, nxt, run, fdr_run = [], [], [], [], []
    for r in pend.itertuples():
        g = by_team.get(int(r.team_id), pd.DataFrame(columns=fut.columns))
        cur = g[g["gw"] == gw]
        mults_now = [_fixture_mult(r.position, f.lam_for, f.cs_prob, lam_ref, cs_ref)
                     for f in cur.itertuples()]
        p_play = float(r.p_play)
        long_play = float(np.clip((r.start_rate or 0) + 0.25, 0, 1)) if (r.minutes or 0) > 0 else 0.2
        if p_play >= 0.15 and mults_now:
            rate = float(r.final_xpts) / (p_play * sum(mults_now))
        else:
            ppg = pd.to_numeric(getattr(r, "ppg", np.nan), errors="coerce")
            rate = float(ppg) if np.isfinite(ppg) and ppg > 0 else 2.0
        tot3 = tot5 = 0.0
        for k, wk in enumerate(gws):
            fx = g[g["gw"] == wk]
            a = p_play if k == 0 else p_play * 0.5 ** k + long_play * (1 - 0.5 ** k)
            pts = sum(rate * a * _fixture_mult(r.position, f.lam_for, f.cs_prob, lam_ref, cs_ref)
                      for f in fx.itertuples())
            if k == 0:
                pts = float(r.final_xpts)        # the next gameweek IS the headline number
            tot5 += pts
            if k < 3:
                tot3 += pts
        x3.append(round(tot3, 2))
        x5.append(round(tot5, 2))
        labs, fdrs = [], []
        for wk in gws:
            fx = g[g["gw"] == wk]
            if fx.empty:
                labs.append("BLANK")
                fdrs.append("")
                continue
            labs.append("+".join(f"{meta.get(int(f.opp_id), {}).get('short', '?')}"
                                 f"({'H' if f.home else 'A'})" for f in fx.itertuples()))
            fdrs.append("+".join(str(int(f.fdr)) if pd.notna(f.fdr) else "?" for f in fx.itertuples()))
        nxt.append(labs[0])
        run.append(" | ".join(labs))
        fdr_run.append(" | ".join(fdrs))
    pend["xpts_3gw"], pend["xpts_5gw"] = x3, x5
    pend["next_fixture"], pend["fixture_run"], pend["fdr_run"] = nxt, run, fdr_run
    return pend


def fixture_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    snap = panel[panel["status"] == "PENDING"]
    if snap.empty:
        return pd.DataFrame()
    gw = int(snap["gw"].iloc[0])
    tfl = fp.team_fixture_long(fp.load_fixtures())
    ts = fp.team_strength_asof(snap, tfl, pd.Timestamp(snap["deadline"].iloc[0]))
    fut = _future_context(tfl, ts, list(range(gw, gw + HORIZON)))
    if fut.empty:
        return fut
    meta = _team_meta()
    fut["team"] = fut["team_id"].map(lambda t: meta.get(t, {}).get("name"))
    fut["team_short"] = fut["team_id"].map(lambda t: meta.get(t, {}).get("short"))
    fut["opp_short"] = fut["opp_id"].map(lambda t: meta.get(t, {}).get("short"))
    # ease ratings 1 (hard) .. 5 (easy) by quintile across every fixture in the window
    fut["attack_ease"] = pd.qcut(fut["lam_for"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    fut["defence_ease"] = pd.qcut(fut["cs_prob"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    fut["label"] = fut["opp_short"] + np.where(fut["home"], " (H)", " (A)")
    for c in ("lam_for", "lam_against", "cs_prob"):
        fut[c] = fut[c].round(3)
    return fut.sort_values(["team", "gw"]).reset_index(drop=True)


# ── captaincy ─────────────────────────────────────────────────────────────────────────────────
def _reason(r, kind: str) -> str:
    bits = [f"{r['final_xpts']:.1f} xPts (FPL {r['fpl_ep_next']:.1f}"
            f"{', Wowza residual %+.1f in shadow' % r['wowza_residual'] if not r['residual_applied'] else ', Wowza residual %+.1f included' % r['wowza_residual']})",
            f"P(start) {r['p_start']:.0%}, ~{r['exp_minutes']:.0f} min",
            f"vs {r['next_fixture']} (team goal expectancy {r['team_goal_exp']:.2f})"]
    if r.get("pen_first") == 1:
        bits.append("on penalties")
    if r.get("set_piece") == 1:
        bits.append("takes set pieces")
    bits.append(f"P(10+) {r['p10']:.0%}, ceiling {r['ceiling']:.0f}")
    lead = {"SAFE": "Highest floor among nailed starters",
            "BALANCED": "Best expected captain score",
            "HIGH_UPSIDE": "Biggest haul chance within reach of the best EV"}[kind]
    return f"{lead}: " + "; ".join(bits) + "."


def captaincy(proj: pd.DataFrame, view: str = "headline") -> dict:
    """SAFE / BALANCED / HIGH_UPSIDE captains. view='shadow' ranks on FPL + residual instead of
    the headline mean, so the page can show where the shadow model would disagree."""
    if proj.empty:
        return {}
    d = proj[~proj["risk"].isin(["Unavailable", "Doubtful"])].copy()
    if view == "shadow" and "v2_candidate_xpts" in d:
        d["final_xpts"] = d["v2_candidate_xpts"]
        d["residual_applied"] = True
        for c in ("p6", "p10", "p15", "p90"):
            d[c] = d[f"cand_{c}"]
        d["ceiling"] = d["p90"]
    nailed = d[d["p_start"] >= 0.85]
    likely = d[d["p_start"] >= 0.75]
    picks, notes = {}, {}
    if len(likely):
        bal = likely.nlargest(1, "final_xpts").iloc[0]
        picks["BALANCED"] = bal
        best = float(likely["final_xpts"].max())
        taken = {bal["element"]}
        # SAFE: lowest chance of a blank (<= 2 pts) among nailed starters within 70% of the best
        # EV. HIGH_UPSIDE: highest P(15+) within 80% of the best EV. Each is a DIFFERENT player
        # from the BALANCED pick when one exists, so the three cards are three real options;
        # when the best-EV player would also have won a category, that is recorded, not hidden.
        safe_pool = nailed[nailed["final_xpts"] >= 0.7 * best]
        if len(safe_pool):
            raw = safe_pool.nsmallest(1, "p_blank").iloc[0]
            alt_ = safe_pool[~safe_pool["element"].isin(taken)]
            pick = alt_.nsmallest(1, "p_blank").iloc[0] if len(alt_) else raw
            if raw["element"] in taken:
                notes["SAFE"] = f"{raw['web_name']} (the BALANCED pick) also has the best floor"
            picks["SAFE"] = pick
            taken.add(pick["element"])
        up_pool = likely[likely["final_xpts"] >= 0.8 * best]
        key = "p15" if up_pool["p15"].max() > 0 else "p90"
        raw = up_pool.nlargest(1, key).iloc[0]
        alt_ = up_pool[~up_pool["element"].isin(taken)]
        pick = alt_.nlargest(1, key).iloc[0] if len(alt_) else raw
        if raw["element"] in taken:
            notes["HIGH_UPSIDE"] = f"{raw['web_name']} also has the biggest haul chance"
        picks["HIGH_UPSIDE"] = pick
    out = {}
    for k, r in picks.items():
        out[k] = {"player": r["player_name"], "web_name": r["web_name"], "team": r["team"],
                  "position": r["position"], "price": float(r["price"]),
                  "final_xpts": float(r["final_xpts"]), "captain_ev": round(2 * float(r["final_xpts"]), 2),
                  "fpl_ep_next": float(r["fpl_ep_next"]), "wowza_residual": float(r["wowza_residual"]),
                  "p6": float(r["p6"]), "p10": float(r["p10"]), "p15": float(r["p15"]),
                  "p_blank": float(r["p_blank"]),
                  "p_start": float(r["p_start"]), "exp_minutes": float(r["exp_minutes"]),
                  "ceiling": float(r["ceiling"]), "next_fixture": r["next_fixture"],
                  "opponent_weakness": float(r["team_goal_exp"]),
                  "pen_first": int(r["pen_first"]), "set_piece": int(r["set_piece"]),
                  "reasoning": _reason(r, k), "note": notes.get(k, "")}
    top = d.nlargest(12, "final_xpts")
    out["candidates"] = top[["player_name", "web_name", "team", "position", "final_xpts",
                             "p6", "p10", "p15", "p_blank", "p_start", "ceiling", "sd",
                             "next_fixture"]].round(3).to_dict("records")
    return out


# ── transfers ─────────────────────────────────────────────────────────────────────────────────
def transfers(proj: pd.DataFrame, n_out: int = 120, per_out: int = 3) -> pd.DataFrame:
    """OUT -> IN by DELTA, same position, IN price <= OUT price + BUDGET_SLACK.

    OUT pool = the most-owned players (the players managers realistically hold). Each move is
    scored on 3-GW and 5-GW projected points, never on a single number."""
    if proj.empty:
        return pd.DataFrame()
    pool_out = proj.nlargest(n_out, "ownership")
    cand = proj[~proj["risk"].isin(["Unavailable"])]
    rows = []
    for o in pool_out.itertuples():
        c = cand[(cand["position"] == o.position) & (cand["element"] != o.element)
                 & (cand["price"] <= o.price + BUDGET_SLACK)].copy()
        if c.empty:
            continue
        c["gain_5gw"] = c["xpts_5gw"] - o.xpts_5gw
        c["gain_3gw"] = c["xpts_3gw"] - o.xpts_3gw
        c["gain_next"] = c["final_xpts"] - o.final_xpts
        for i in c.nlargest(per_out, "gain_5gw").itertuples():
            if i.gain_5gw <= 0:
                continue
            rows.append({
                "out_player": o.player_name, "out_web": o.web_name, "out_team": o.team,
                "position": o.position, "out_price": o.price, "out_xpts_next": o.final_xpts,
                "out_xpts_3gw": o.xpts_3gw, "out_xpts_5gw": o.xpts_5gw, "out_risk": o.risk,
                "out_ownership": o.ownership, "out_fixture_run": o.fixture_run,
                "in_player": i.player_name, "in_web": i.web_name, "in_team": i.team,
                "in_price": i.price, "in_fpl_ep_next": i.fpl_ep_next,
                "in_wowza_residual": i.wowza_residual, "in_final_xpts": i.final_xpts,
                "in_xpts_3gw": i.xpts_3gw, "in_xpts_5gw": i.xpts_5gw, "in_p_start": i.p_start,
                "in_ceiling": i.ceiling, "in_value_per_m": i.value_per_m,
                "in_fixture_run": i.fixture_run, "in_ownership": i.ownership, "in_risk": i.risk,
                "price_delta": round(i.price - o.price, 1),
                "gain_next": round(i.gain_next, 2), "gain_3gw": round(i.gain_3gw, 2),
                "gain_5gw": round(i.gain_5gw, 2),
                "net_gain_5gw_with_hit": round(i.gain_5gw - 4.0, 2)})
    t = pd.DataFrame(rows)
    return t.sort_values("gain_5gw", ascending=False).reset_index(drop=True) if len(t) else t


# ── orchestration ────────────────────────────────────────────────────────────────────────────
def _projection_health(proj: pd.DataFrame, panel: pd.DataFrame) -> dict:
    if proj.empty:
        return {"state": "EMPTY"}
    return {"players": int(len(proj)),
            "with_fixture": int((proj["n_fix"] > 0).sum()),
            "blank_gw_players": int((proj["n_fix"] == 0).sum()),
            "double_gw_players": int((proj["n_fix"] >= 2).sum()),
            "minutes_disagreements": int(proj["minutes_disagreement"].sum()),
            "archived_gameweeks": sorted(int(g) for g in panel["gw"].unique()),
            "final_gameweeks": sorted(int(g) for g in panel.loc[panel["status"] == "FINAL", "gw"].unique()),
            "provisional_gameweeks": sorted(int(g) for g in panel.loc[panel["status"] == "PROVISIONAL", "gw"].unique()),
            "state": "OK" if (proj["n_fix"] > 0).mean() > 0.8 else "DEGRADED"}


def summary(proj: pd.DataFrame, panel: pd.DataFrame, bench: dict, fetched_at: float | None) -> dict:
    pend = panel[panel["status"] == "PENDING"]
    now = pd.Timestamp.now(tz="UTC")
    fetched = pd.Timestamp(fetched_at, unit="s", tz="UTC") if fetched_at else None
    sc = (bench.get("scores") or {}).get("ACTIVE", {})
    pooled = sc.get("pooled", {})
    return {
        "calculation_version": CALC_VERSION,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gameweek": int(pend["gw"].iloc[0]) if len(pend) else None,
        "deadline": (pd.Timestamp(pend["deadline"].iloc[0]).strftime("%Y-%m-%dT%H:%M:%SZ")
                     if len(pend) else None),
        "fpl_fetched_at": fetched.strftime("%Y-%m-%dT%H:%M:%SZ") if fetched is not None else None,
        "fpl_age_hours_at_build": (round((now - fetched).total_seconds() / 3600, 1)
                                   if fetched is not None else None),
        "projection_health": _projection_health(proj, panel),
        "benchmark_status": {
            "decision_residual": (bench.get("decision") or {}).get("apply_residual_to_mean"),
            "residual_mode": residual_mode(),
            "residual_applied": (bench.get("decision") or {}).get("applied", False),
            "mean_forecast": ("FPL ep_next + Wowza residual"
                              if (bench.get("decision") or {}).get("applied") else "FPL ep_next"),
            "gws_evaluated": [r["gw"] for r in sc.get("by_gw", [])],
            "fpl": pooled.get("FPL"), "v2": pooled.get(PRODUCTION_ARM),
            "wowza_old": pooled.get("WOWZA")},
        "captaincy": captaincy(proj),
        "captaincy_shadow": captaincy(proj, view="shadow"),
    }


def run(write: bool = True) -> dict:
    boot = {}
    try:
        boot = json.loads(fp.BOOTSTRAP.read_text(encoding="utf-8"))
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"[fantasy_v2] no bootstrap ({e})")
    fetched_at = fs.recorded_fetch_time()
    n_arch = fs.update_archive(boot, fetched_at) if boot else 0
    panel = fp.build_panel()
    if panel.empty:
        return {"ok": False, "why": "empty panel"}
    extras: dict = {}
    bench, oos = run_benchmark(panel, extras)
    proj = project(panel, bench)
    summ = summary(proj, panel, bench, fetched_at)
    tr = transfers(proj)
    fm = fixture_matrix(panel)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        if not proj.empty:
            proj.to_csv(PROJECTIONS, index=False, encoding="utf-8", lineterminator="\n")
        BENCHMARK.write_text(json.dumps(bench, indent=1, default=str), encoding="utf-8")
        if not oos.empty:
            cols = ["gw", "status", "element", "player_name", "team", "position", "price",
                    "in_ledger", "ep_next", "WOWZA", "LINEAR", "RIDGE", "HGB", "actual_points",
                    "actual_minutes", "train_gws"]
            oos[[c for c in cols if c in oos.columns]].round(3).to_csv(OOS, index=False,
                                                                       encoding="utf-8",
                                                                       lineterminator="\n")
        board = extras.get("scoreboard")
        if board is not None and not board.empty:
            board.to_csv(SCOREBOARD, index=False, encoding="utf-8", lineterminator="\n")
            rel = extras.get("reliability")
            if rel is not None and not rel.empty:
                rel.to_csv(SCOREBOARD_RELIABILITY, index=False, encoding="utf-8",
                           lineterminator="\n")
        SUMMARY.write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
        tr.to_csv(TRANSFERS, index=False, encoding="utf-8", lineterminator="\n")
        if not fm.empty:
            fm.to_csv(FIXTURE_MATRIX, index=False, encoding="utf-8", lineterminator="\n")
    return {"ok": True, "archived_rows": n_arch, "players": int(len(proj)),
            "decision": bench.get("decision"), "gameweek": summ.get("gameweek")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = run()
    print(json.dumps(r, indent=1, default=str))
