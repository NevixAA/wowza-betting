"""Read-only views for the Market Intelligence page.

Sources (all small, all written by CI):
  * output/edge_engine_health.json   market_edge_v2 shadow run: two-sided price coverage + states per
                                     market x track (src/edge_engine_shadow.py, predict.yml)
  * output/clv_collection_health.json certified close coverage: a close observed <= 30 / <= 10 min
                                     before kickoff, by league and market (src/clv_collection_health.py)
  * output/direction_conflicts.csv   tips suppressed because they pointed the opposite goal direction
                                     to a tip that already stood (src/direction_guard.py)
  * output/edge_engine_shadow.csv    first-sight rows: LEGACY RAW EDGE next to the market-first decision.
                                     Column-projected; never more than the columns listed below.
  * output/edge_engine_comparison.json + output/edge_engine_backtest.csv
                                     RETROSPECTIVE legacy vs market-first replay (src/research/edge_replay.py);
                                     migration_comparison() splits LEGACY into DEPLOYABLE / PAPER
  * Pro output/edge_reassessment/edge2_forward_progress.json (via dashboard_data.pro.edge2_forward)
                                     PROSPECTIVE H-EDGE2 forward status; edge2_status() never shows CONFIRMED

Nothing here de-vigs, prices, scores or decides. Every number is what those writers recorded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
EDGE_HEALTH = OUT / "edge_engine_health.json"
CLV_HEALTH = OUT / "clv_collection_health.json"
CONFLICTS = OUT / "direction_conflicts.csv"
SHADOW = OUT / "edge_engine_shadow.csv"

SHADOW_COLS = ["prediction_ts", "match_date", "league", "home_team", "away_team", "track", "market", "side",
               "legacy_tier", "legacy_side", "legacy_raw_edge", "legacy_odds", "p_model_raw", "p_market_fair",
               "p_decision", "p_lower_bound", "ev_lower_bound", "decision_state", "decision_reason", "n_books",
               "clv_evidence_state"]
CONFLICT_COLS = ["generated_at", "match_date", "league", "home_team", "away_team", "suppressed_market",
                 "suppressed_side", "suppressed_tier", "stood_market", "stood_side", "reason"]
LEGACY_TIPPED = ("SNIPER", "MARKSMAN", "VALUABLE")


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def _csv(p: Path, cols: list[str]) -> pd.DataFrame:
    if not Path(p).exists():
        return pd.DataFrame(columns=cols)
    try:
        return pd.read_csv(p, usecols=lambda c: c in cols, dtype=str).reindex(columns=cols)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame(columns=cols)


def edge_health(path: Path | None = None) -> tuple[dict, pd.DataFrame]:
    h = _json(path or EDGE_HEALTH)
    rows = []
    for key, v in (h.get("coverage") or {}).items():
        market, _, track = key.partition("|")
        rows.append({"market": market, "track": track, "fixtures": v.get("fixtures"),
                     "two_sided": v.get("two_sided"), "two_sided_share": v.get("two_sided_share"),
                     "multi_book": v.get("multi_book"), "median_books": v.get("median_books"),
                     "legacy_tipped": v.get("legacy_tipped"),
                     "market_first_states": ", ".join(f"{k} {n}" for k, n in (v.get("states") or {}).items())})
    meta = {k: h.get(k) for k in ("generated_at", "selection_engine_version", "config_status", "devig_method",
                                  "shadow_only", "rows_scored", "current_run_book_quotes",
                                  "prospective_evidence_loaded")}
    return meta, pd.DataFrame(rows)


def _league_rows(d: dict, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"name": k, **{c: v.get(c) for c in cols}} for k, v in (d or {}).items()])


def clv_health(path: Path | None = None) -> dict:
    c = _json(path or CLV_HEALTH)
    cert_cols = ["eligible", "with_t30_close", "with_t10_close", "stale_close", "no_close",
                 "certified_t30_pct", "certified_t10_pct"]
    coll_cols = ["fixtures", "fixture_markets", "t60_pct", "t30_pct", "t10_pct", "median_last_obs_minutes"]
    since = c.get("shadow_since_measure_from") or {}
    allt = c.get("shadow_all_time") or {}
    coll = c.get("collection") or {}
    return {"generated_at": c.get("generated_at"), "close_rule": c.get("close_rule"),
            "measure_from": c.get("measure_from"), "check": c.get("check") or {},
            "since": {k: since.get(k) for k in cert_cols}, "all_time": {k: allt.get(k) for k in cert_cols},
            "since_by_league": _league_rows(since.get("by_league"), cert_cols),
            "since_by_market": _league_rows(since.get("by_market"), cert_cols),
            "collection": {k: coll.get(k) for k in ("window_days", "from", "fixtures", "fixture_markets",
                                                     "t60_pct", "t30_pct", "t10_pct")},
            "collection_by_track": _league_rows(coll.get("by_track"), coll_cols),
            "collection_by_market": _league_rows(coll.get("by_market"), coll_cols),
            "collection_by_league": _league_rows(coll.get("by_league"), coll_cols)}


def conflicts(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, summary by suppressed market x reason)."""
    d = _csv(path or CONFLICTS, CONFLICT_COLS)
    if d.empty:
        return d, pd.DataFrame(columns=["suppressed_market", "reason", "suppressed"])
    s = d.groupby(["suppressed_market", "reason"], dropna=False).size().rename("suppressed").reset_index()
    return d.sort_values("generated_at", ascending=False, kind="mergesort"), s.sort_values("suppressed",
                                                                                              ascending=False)


def legacy_vs_market(path: Path | None = None, today: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(upcoming first-sight rows from `today`, all-time summary per market x track).

    LEGACY = production's raw edge (p_model - 1/odds) and tier. MARKET-FIRST = market_edge_v2's fair
    price, decision probability, lower bound, EV lower bound and state. Shadow only."""
    d = _csv(path or SHADOW, SHADOW_COLS)
    if d.empty:
        return d, pd.DataFrame()
    for c in ("legacy_raw_edge", "legacy_odds", "p_model_raw", "p_market_fair", "p_decision", "p_lower_bound",
              "ev_lower_bound"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["legacy_tipped"] = d["legacy_tier"].isin(LEGACY_TIPPED)
    summ = (d.groupby(["market", "track"], dropna=False)
             .agg(rows=("market", "size"), legacy_tipped=("legacy_tipped", "sum"),
                  market_first_bet=("decision_state", lambda s: int((s == "BET").sum())),
                  market_first_paper=("decision_state", lambda s: int((s == "PAPER").sum())),
                  market_first_no_bet=("decision_state", lambda s: int((s == "NO_BET").sum())),
                  median_ev_lb=("ev_lower_bound", "median"))
             .reset_index())
    up = d[d["match_date"].astype(str) >= str(today)] if today else d
    up = up.sort_values(["match_date", "league", "home_team", "market"], kind="mergesort")
    return up.drop(columns=["legacy_tipped"]), summ


# ══ LEGACY vs MARKET-FIRST migration comparison (market-first brief, 2026-10-26) ══════════════════
# RETROSPECTIVE side: output/edge_engine_comparison.json + output/edge_engine_backtest.csv, both written
# by src/research/edge_replay.py (first-sight replay 2026-08-15 -> 10-22). PROSPECTIVE side:
# output/edge_engine_shadow.csv (live first-sight log) + Pro's H-EDGE2 forward progress.
# Every figure is per track x market; tracks, markets, PAPER and DEPLOYABLE are never pooled.
COMPARISON = OUT / "edge_engine_comparison.json"
BACKTEST = OUT / "edge_engine_backtest.csv"
POLICY = ROOT / "config" / "bankroll_policy.json"
#: (track, replay family) cells in display order; OU15 is the OVER 1.5 market
CELLS = (("new_format", "OU25"), ("new_format", "BTTS"), ("new_format", "OU15"),
         ("standard", "OU25"), ("standard", "BTTS"), ("standard", "OU15"))
TRACK_LABEL = {"new_format": "NF", "standard": "STD"}
MARKET_LABEL = {"OU25": "OU25", "BTTS": "BTTS", "OU15": "OVER15", "OU35": "OVER35 (blocked)"}
#: replay family -> bankroll-policy track market
POLICY_MARKET = {"OU25": "OU25", "BTTS": "BTTS", "OU15": "OVER15"}
BACKTEST_COLS = ["family", "track", "league", "entry_ts", "kickoff", "legacy_tier", "legacy_side", "legacy_odds",
                 "new_side", "new_odds", "legacy_bet", "two_sided", "y_a", "close_status", "p_close_fair"]
#: market-first engines as the comparison JSON names them; every one is PAPER (shadow only)
NEW_ENGINES = (("NEW_QUALIFIED", "NEW · QUALIFIED/STRONG"), ("NEW_EVLB", "NEW · EV lower bound > 0"),
               ("NEW_OBSERVE_POOL", "NEW · observe pool (not a bet)"))
PROB_SOURCES = (("model", "model"), ("market_power", "market (power de-vig)"), ("decision", "decision"))
SHADOW_SAMPLE_COLS = ["prediction_ts", "kickoff_utc", "model_type", "market", "decision_state", "market_source"]
TWO_SIDED_SOURCES = ("MEDIAN_TWO_SIDED_BOOKS", "SINGLE_TWO_SIDED_BOOK")
#: statuses Pro's H-EDGE2 evaluator may assign (src/studies/edge2_forward.py); CONFIRMED never is
EDGE2_ALLOWED = ("REGISTERED", "ACCRUING", "ELIGIBLE_FOR_REVIEW", "BLOCKED_CONFIG_CHANGED")


def cell_label(track: str, family: str) -> str:
    return f"{TRACK_LABEL.get(track, track)} {MARKET_LABEL.get(family, family)}"


def comparison_meta(path: Path | None = None) -> dict:
    c = _json(path or COMPARISON)
    keep = ("label", "note", "generated_at", "engine_version", "config_status", "inputs")
    out = {k: c.get(k) for k in keep}
    out["definitions"] = c.get("definitions") or {}
    return out


def deployable(league: str, track: str, family: str, tier: str, odds: float, policy: dict) -> tuple[bool, str]:
    """Would TODAY's bankroll policy put real money on this legacy selection? The same per-row rules as
    src/bankroll_policy.py (track state + base fraction, paper league, hard and track odds caps, tier
    multiplier), WITHOUT the portfolio caps / drawdown throttle, which depend on the day's book. A
    RETROSPECTIVE classification: the policy in force at the time differed (e.g. VALUABLE was staked on
    OU25 only from 2026-10-22)."""
    from src.bankroll_policy import HARD, REAL_STATES
    t = (policy.get("tracks") or {}).get(f"{track}|{POLICY_MARKET.get(family, family)}",
                                         {"state": "UNSUPPORTED", "base_fraction": 0.0})
    if t.get("state") not in REAL_STATES or float(t.get("base_fraction", 0.0)) <= 0:
        return False, f"track {t.get('state')}"
    if league in set(policy.get("paper_leagues") or []):
        return False, "paper league"
    hard = min(float((policy.get("portfolio") or {}).get("hard_max_odds", HARD["blocked_odds"])), HARD["blocked_odds"])
    if not (odds == odds) or odds >= hard:
        return False, "odds blocked"
    if odds > min(float(t.get("max_odds", HARD["max_odds"])), HARD["max_odds"]):
        return False, "odds above track cap"
    if odds < float(t.get("min_odds", 1.0)):
        return False, "odds below track floor"
    tt = (policy.get("track_tier_multipliers") or {}).get(f"{track}|{POLICY_MARKET.get(family, family)}")
    if float((tt if tt is not None else policy.get("tier_multipliers", {})).get(str(tier), 0.0)) <= 0:
        return False, "tier not staked"
    return True, "staked under today's policy"


def _backtest(path: Path | None = None) -> pd.DataFrame:
    d = _csv(path or BACKTEST, BACKTEST_COLS)
    if d.empty:
        return d
    for c in ("legacy_bet", "two_sided"):
        d[c] = d[c].astype(str).str.strip().str.lower() == "true"
    for c in ("legacy_odds", "new_odds", "y_a", "p_close_fair"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("entry_ts", "kickoff"):
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    return d


def _metric_row(cell: str, track: str, family: str, engine: str, basis: str, status: str, m: dict) -> dict:
    ci = m.get("roi_ci95") or [None, None]
    return {"cell": cell, "track": track, "market": MARKET_LABEL.get(family, family), "engine": engine,
            "basis": basis, "cell_status": status, "n": int(m.get("n") or 0), "matchdays": m.get("matchdays"),
            "hit_pct": m.get("hit_pct"), "roi_pct": m.get("roi_pct"), "roi_ci_lo": ci[0], "roi_ci_hi": ci[1],
            "pnl_u": m.get("pnl_u"), "max_drawdown_u": m.get("max_drawdown_u"), "clv_n": m.get("clv_n"),
            "clv_coverage_pct": m.get("clv_coverage_pct"), "clv_mean_pct": m.get("clv_mean_pct")}


def migration_comparison(cmp_path: Path | None = None, bt_path: Path | None = None,
                         policy_path: Path | None = None) -> dict:
    """RETROSPECTIVE legacy vs market-first comparison, per track x market (never pooled).

    engines   one row per cell x engine on the TWO-SIDED universe (rows where the market-first engine could
              form a market at first sight, the only universe both engines are judged on). LEGACY is split
              into DEPLOYABLE (today's bankroll policy would stake it) and PAPER (it would not) and computed
              from the replay rows with the replay's own settle_bets/bet_metrics; NEW_* rows are copied from
              the comparison JSON and are all PAPER (market_edge_v2 is shadow only).
    probability  Brier / log loss / calibration slope / ECE of model, de-vigged market and decision
              probability (side A = OVER/YES) + the model-minus-market Brier gain with its CI.
    agreement LEGACY (all tiers) vs NEW_EVLB and NEW_QUALIFIED selection counts on the two-sided universe.
    reconcile per cell: DEPLOYABLE + PAPER n and P/L against the JSON's (pooled) LEGACY figure."""
    c = _json(cmp_path or COMPARISON)
    cells = c.get("cells") or {}
    policy = _json(policy_path or POLICY)
    bt = _backtest(bt_path)
    eng, prob, agree, gain, rec = [], [], [], [], []
    if cells and not bt.empty:
        from src.research import edge_replay as R                    # lazy: 1 s import, cached by the page
    for track, fam in CELLS:
        key = f"{track}|{fam}"
        cj = cells.get(key)
        if not cj:
            continue
        lab, status = cell_label(track, fam), str(cj.get("status"))
        two_n = int(cj.get("two_sided_at_entry") or 0)
        e2 = cj.get("engines_two_sided_universe") or {}
        # LEGACY split DEPLOYABLE / PAPER, recomputed from the replay rows of this cell only
        if not bt.empty:
            s = bt[(bt["track"] == track) & (bt["family"] == fam) & bt["y_a"].notna()]
            two = s[s["two_sided"]]
            days = max((s["entry_ts"].max() - s["entry_ts"].min()).total_seconds() / 86400.0, 1.0) if len(s) else 1.0
            x = R.settle_bets(two, "LEGACY") if len(two) else two
            if len(x):
                dep = [deployable(lg, track, fam, tier, o, policy)[0]
                       for lg, tier, o in zip(x["league"], x["legacy_tier"], x["bet_odds"])]
                x = x.assign(_dep=dep)
            n_all, pnl_all = 0, 0.0
            for basis, part in (("DEPLOYABLE", x[x["_dep"]] if len(x) else x),
                                ("PAPER", x[~x["_dep"]] if len(x) else x)):
                m = R.bet_metrics(part, days) if len(part) else {"n": 0}
                eng.append(_metric_row(lab, track, fam, "LEGACY raw edge", basis, status, m))
                n_all += int(m.get("n") or 0)
                pnl_all += float(m.get("pnl_u") or 0.0)
            lj = e2.get("LEGACY") or {}
            rec.append({"cell": lab, "legacy_json_n": int(lj.get("n") or 0), "split_n": n_all,
                        "legacy_json_pnl_u": lj.get("pnl_u"), "split_pnl_u": round(pnl_all, 2),
                        "reconciled": int(lj.get("n") or 0) == n_all
                        and abs(float(lj.get("pnl_u") or 0.0) - pnl_all) < 0.02})
        for k, name in NEW_ENGINES:
            eng.append(_metric_row(lab, track, fam, name, "PAPER", status, e2.get(k) or {"n": 0}))
        pq = cj.get("probability_quality_side_a") or {}
        for k, name in PROB_SOURCES:
            v = pq.get(k) or {}
            prob.append({"cell": lab, "track": track, "market": MARKET_LABEL[fam], "source": name,
                         "n": v.get("n"), "brier": v.get("brier"), "logloss": v.get("logloss"),
                         "calibration_slope": v.get("calibration_slope"), "ece": v.get("ece")})
        g = cj.get("model_minus_market_brier_gain") or {}
        gci = g.get("ci95") or [None, None]
        gain.append({"cell": lab, "track": track, "market": MARKET_LABEL[fam], "cell_status": status,
                     "n": int((pq.get("model") or {}).get("n") or 0), "brier_gain": g.get("mean"),
                     "ci_lo": gci[0], "ci_hi": gci[1]})
        for k in ("NEW_EVLB", "NEW_QUALIFIED"):
            a = (cj.get("agreement_two_sided") or {}).get(k) or {}
            tot = sum(int(a.get(f, 0) or 0) for f in ("legacy_only", "new_only", "both_same_side",
                                                      "disagree_side", "neither"))
            sel = tot - int(a.get("neither", 0) or 0)
            agree.append({"cell": lab, "track": track, "market": MARKET_LABEL[fam], "new_engine": k,
                          "two_sided_rows": tot or two_n, "legacy_only": a.get("legacy_only"),
                          "new_only": a.get("new_only"), "both_same_side": a.get("both_same_side"),
                          "disagree_side": a.get("disagree_side"), "neither": a.get("neither"),
                          "agreement_rate": round((int(a.get("both_same_side", 0) or 0)
                                                   + int(a.get("neither", 0) or 0)) / tot, 4) if tot else None,
                          "selection_overlap": round(int(a.get("both_same_side", 0) or 0) / sel, 4) if sel else None})
    return {"meta": comparison_meta(cmp_path), "engines": pd.DataFrame(eng), "probability": pd.DataFrame(prob),
            "brier_gain": pd.DataFrame(gain), "agreement": pd.DataFrame(agree), "reconcile": pd.DataFrame(rec),
            "policy_version": policy.get("policy_version")}


def shadow_sample(path: Path | None = None, forward_start: str | None = None,
                  now: pd.Timestamp | None = None) -> tuple[dict, pd.DataFrame]:
    """PROSPECTIVE side: current market-first first-sight sample per track x market (never pooled).

    rows, two-sided rows, rows since the H-EDGE2 forward start, kicked-off rows, and the decision-state
    distribution. The shadow log carries no result columns: settlement happens in Pro's evaluator."""
    d = _csv(path or SHADOW, SHADOW_SAMPLE_COLS)
    if d.empty:
        return {"rows": 0}, pd.DataFrame()
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    ts = pd.to_datetime(d["prediction_ts"], utc=True, errors="coerce", format="mixed")
    ko = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce", format="mixed")
    fs = pd.to_datetime(forward_start, utc=True, errors="coerce") if forward_start else pd.NaT
    d = d.assign(_two=d["market_source"].isin(TWO_SIDED_SOURCES), _ko=(ko <= now),
                 _fwd=(ts >= fs) if pd.notna(fs) else False)
    rows = []
    for (mt, mk), g in d.groupby(["model_type", "market"], sort=True):
        st_counts = g["decision_state"].fillna("?").value_counts()
        rows.append({"cell": cell_label(mt, mk), "track": mt, "market": MARKET_LABEL.get(mk, mk),
                     "rows": int(len(g)), "two_sided": int(g["_two"].sum()),
                     "since_forward_start": int(g["_fwd"].sum()) if pd.notna(fs) else None,
                     "kicked_off": int(g["_ko"].sum()),
                     "states": ", ".join(f"{k} {int(v)}" for k, v in st_counts.items()),
                     **{f"state_{k}": int(v) for k, v in st_counts.items()}})
    meta = {"rows": int(len(d)), "first_prediction_ts": str(ts.min()) if ts.notna().any() else None,
            "last_prediction_ts": str(ts.max()) if ts.notna().any() else None,
            "runs": int(ts.nunique()), "forward_start": forward_start}
    return meta, pd.DataFrame(rows).fillna({c: 0 for c in rows[0] if c.startswith("state_")}) if rows else pd.DataFrame()


def edge2_status(progress: dict | None) -> tuple[dict, pd.DataFrame]:
    """Pro H-EDGE2 forward progress -> (meta, one row per hypothesis). A status outside EDGE2_ALLOWED
    (e.g. CONFIRMED, which the evaluator never assigns) is reported as INVALID, never shown as a result."""
    p = progress or {}
    rows = []
    for h in p.get("hypotheses") or []:
        f = h.get("forward") or {}
        s = str(h.get("status"))
        rows.append({"id": h.get("id"), "track": h.get("track"),
                     "market": MARKET_LABEL.get(str(h.get("market")), h.get("market")),
                     "status": s if s in EDGE2_ALLOWED else f"INVALID ({s})",
                     "logged": f.get("logged"), "two_sided": f.get("two_sided"),
                     "settled_two_sided": f.get("settled_two_sided"), "min_settled": h.get("min_n_settled_two_sided"),
                     "gameweeks": f.get("gameweeks"), "min_gameweeks": h.get("min_gameweeks"),
                     "progress_pct": h.get("progress_pct"),
                     "pre_registration_not_counted": h.get("PRE_REGISTRATION_NOT_COUNTED")})
    meta = {k: p.get(k) for k in ("generated_at", "forward_start", "registration_intact",
                                  "engine_hashes_match_registration", "confirmed_assignable_here",
                                  "production_changed", "status_counts", "registry")}
    meta["invalid_status"] = [r["id"] for r in rows if str(r["status"]).startswith("INVALID")]
    return meta, pd.DataFrame(rows)
