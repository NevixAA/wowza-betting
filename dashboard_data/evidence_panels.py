"""Evidence panels (owner brief 2026-10-26, part D) — DISPLAY ONLY. No policy, model or selection change.

Four read-only computations, one per panel. Nothing here writes a file, sizes a stake, settles a bet,
promotes a model or imports a module that does (tests/test_evidence_panels.py pins that). Every input is
a committed artifact the dashboard already mirrors:

  1. since_review()      Bankroll   — Pro's latest formal bankroll review (output/bankroll/track_evidence.json
                                      + current_policy_recommendation.json) vs v9's live ledgers since its
                                      evidence cutoff, per production track (O/U 2.5 split by side).
  2. hmodel_rows()       Model Lab  — H-MODEL-01 paired champion-vs-HGB numbers exactly as
                                      src/shadow_experiments.evaluate wrote them (eval JSON); never recomputed.
  3. edge2_report()      Market Intelligence — Pro's H-EDGE2 forward report per registered hypothesis +
                                      the counts-only description of v9's first-sight shadow log.
  4. over15_real_price() Market Intelligence — NF Over 1.5 live tips on GENUINE quoted prices only.

Light by construction: the ledgers are ~6k rows; the only resampling is a matchday block bootstrap with a
fixed seed (deterministic, < 50 ms per cell). Pages cache the results (st.cache_data).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
POLICY_FILE = ROOT / "config" / "bankroll_policy.json"          # read only — json.loads, never written
BETS = OUT / "bets_ledger.csv"
SIDE_BETS = OUT / "side_bets_ledger.csv"
SHADOW = OUT / "edge_engine_shadow.csv"
EXPERIMENTS = ROOT / "models" / "registry" / "experiments.json"
GATE_CFG = ROOT / "config" / "promotion_gate.json"
SHADOW_EVAL = OUT / "model_lineage" / "shadow_experiments_eval.json"

NO_POLICY_CHANGE = ("NO POLICY CHANGE — next formal policy review follows the registered monthly process.")

# ── shared ───────────────────────────────────────────────────────────────────────────────────────
#: Evidence epoch of the Pro bankroll review (Pro src/bankroll/__init__.PERFORMANCE_CUTOFF) = v9
#: config.PERFORMANCE_CUTOFF_DATE. Tips before it predate fixes.
PERFORMANCE_CUTOFF = "2026-08-10"
#: The policy key rule, mirrored from src/bankroll_policy.track_key (NOT imported: that module also
#: writes the stake audit, and the bankroll page must not import a writer). Parity is a unit test.
SIDE_SPLIT_MARKETS = {"OU25": ("OVER", "UNDER")}
UNKNOWN_SIDE = "UNKNOWN_SIDE"
SIDE_MARKET = {"btts": "BTTS", "over15": "OVER15", "over35": "OVER35"}
N_BOOT = 2000


def track_key(model_type, market, side=None) -> str:
    mt, mk = str(model_type), str(market)
    sides = SIDE_SPLIT_MARKETS.get(mk.strip().upper())
    if sides is None:
        return f"{mt}|{mk}"
    s = "" if side is None or (isinstance(side, float) and side != side) else str(side).strip().upper()
    return f"{mt}|{mk}_{s if s in sides else UNKNOWN_SIDE}"


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def _read(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, low_memory=False) if Path(p).exists() else pd.DataFrame()
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame()


def block_ci(values, days, level: float = 0.90, n_boot: int = N_BOOT, seed: int = 0):
    """(lo, hi) of the MEAN, resampling whole matchdays (the unit Pro's review uses). None when < 3 days."""
    v = np.asarray(values, float)
    d = np.asarray(days)
    if len(v) < 2:
        return None, None
    u = pd.DataFrame({"x": v, "g": d}).groupby("g")["x"].agg(["sum", "count"])
    if len(u) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    m = u["sum"].to_numpy()[idx].sum(1) / u["count"].to_numpy()[idx].sum(1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 1. BANKROLL — since the last formal review
# ═════════════════════════════════════════════════════════════════════════════════════════════════
#: Directional rule (fixed 2026-10-26, before any number was looked at):
#:   INSUFFICIENT SAMPLE  since-review n < MIN_N or < MIN_DAYS matchdays, or no review ROI to compare
#:   IMPROVING            the since-review ROI's 90% matchday-block-bootstrap interval lies wholly ABOVE
#:                        the review-date ROI
#:   DETERIORATING        the interval lies wholly BELOW the review-date ROI
#:   STABLE               the interval contains the review-date ROI
DIRECTION_MIN_N = 30
DIRECTION_MIN_DAYS = 5
DIRECTION_LEVEL = 0.90
DIRECTION_RULE = (f"Since-review ROI vs the review-date ROI. Needs n ≥ {DIRECTION_MIN_N} settled bets on "
                  f"≥ {DIRECTION_MIN_DAYS} matchdays since the review, else INSUFFICIENT SAMPLE. IMPROVING when "
                  f"the since-review ROI's {int(DIRECTION_LEVEL * 100)}% matchday-block-bootstrap interval is "
                  "wholly above the review-date ROI, DETERIORATING when wholly below, otherwise STABLE. "
                  "Descriptive only — it never moves a stake.")


def _paper_leagues() -> set[str]:
    """config.PAPER_LEAGUES read as text (the same parse Pro's review uses) — no config import."""
    import re
    try:
        txt = (ROOT / "config.py").read_text(encoding="utf-8")
        m = re.search(r"PAPER_LEAGUES\s*(?::\s*\w+)?\s*=\s*(\{[^}]*\}|\[[^\]]*\]|frozenset\([^)]*\))", txt)
        if m:
            return set(re.findall(r"[\"']([^\"']+)[\"']", m.group(1)))
    except Exception:                                                 # noqa: BLE001
        pass
    return {"USA MLS"}


def load_bets(bets: pd.DataFrame | None = None, side_bets: pd.DataFrame | None = None,
              from_date: str = PERFORMANCE_CUTOFF) -> pd.DataFrame:
    """The prospective staked-tier record exactly as Pro's review builds it (Pro src/bankroll/data.load_bets):
    live source, paper leagues removed, SNIPER/MARKSMAN/VALUABLE, settled, from the evidence epoch, keyed
    with the policy track key. unit = flat-1u P/L from the ledger; clv = clv_pct / 100."""
    m = _read(BETS) if bets is None else bets
    s = _read(SIDE_BETS) if side_bets is None else side_bets
    parts = []
    if len(m):
        parts.append(pd.DataFrame({
            "source": m.get("source"), "date": m["match_date"], "league": m["league"], "home": m["home_team"],
            "away": m["away_team"], "model_type": m.get("model_type"), "market": "OU25",
            "side": m["side"].astype(str).str.upper(), "signal_tier": m["signal_tier"], "odds": m["odds"],
            "result": m["result"], "pnl": m["pnl"], "clv_pct": m.get("clv_pct")}))
    if len(s):
        parts.append(pd.DataFrame({
            "source": s.get("source"), "date": s["match_date"], "league": s["league"], "home": s["home_team"],
            "away": s["away_team"], "model_type": s.get("model_type"), "market": s["market"].map(SIDE_MARKET),
            "side": np.where(s["market"].eq("btts"), "YES", "OVER"), "signal_tier": s["signal_tier"],
            "odds": s["odds"], "result": s["result"], "pnl": s["pnl"], "clv_pct": s.get("clv_pct")}))
    if not parts:
        return pd.DataFrame(columns=["date", "track", "unit", "clv", "signal_tier", "league"])
    d = pd.concat(parts, ignore_index=True)
    d = d[d["source"].fillna("live").astype(str).eq("live")]
    d = d[~d["league"].isin(_paper_leagues())]
    d = d[d["signal_tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"]) & d["market"].notna()]
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10], errors="coerce")
    d = d[d["date"] >= pd.Timestamp(from_date)]
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d["unit"] = pd.to_numeric(d["pnl"], errors="coerce")
    d["clv"] = pd.to_numeric(d["clv_pct"], errors="coerce") / 100.0
    d["model_type"] = d["model_type"].fillna("unknown").astype(str)
    d = d[d["unit"].notna()]
    d["track"] = [track_key(mt, mk, sd) for mt, mk, sd in zip(d["model_type"], d["market"], d["side"])]
    return d.drop(columns=["source", "pnl", "clv_pct"]).sort_values(["date", "track"], kind="mergesort") \
            .reset_index(drop=True)


def tier_multiplier(policy: dict, track: str, tier: str) -> float:
    """The multiplier v9's reader applies: the track override if present, else the global ladder."""
    tt = (policy.get("track_tier_multipliers") or {}).get(track)
    return float((tt if tt is not None else policy.get("tier_multipliers", {})).get(str(tier), 0.0))


def staked(bets: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Only the tiers the policy stakes (multiplier > 0) — the evidence for a track, as in the review."""
    if bets.empty:
        return bets
    keep = [tier_multiplier(policy, tr, t) > 0 for tr, t in zip(bets["track"], bets["signal_tier"])]
    return bets[keep]


EQUAL_TIERS = {"SNIPER": 1.0, "MARKSMAN": 1.0, "VALUABLE": 1.0}


def review_policy(policy: dict, tier_rules: pd.DataFrame | None) -> dict:
    """The tier set the review's evidence used: production policy + the review's own per-track tier rule
    (Pro output/bankroll/tier_rules.csv; Pro src/bankroll/allocator.derive_tier_rules): EQUAL -> every
    tier counts, LADDER -> the global ladder (override removed), KEEP -> whatever is in force. Applied to
    both periods so since-review is measured on exactly the bets the review measured. A copy — the
    production policy dict is never mutated."""
    import copy
    p = copy.deepcopy(policy or {})
    if tier_rules is None or tier_rules.empty or "rule" not in tier_rules.columns:
        return p
    tt = dict(p.get("track_tier_multipliers") or {})
    for _, r in tier_rules.iterrows():
        rule = str(r["rule"]).upper()
        if rule.startswith("EQUAL"):
            tt[str(r["track"])] = dict(EQUAL_TIERS)
        elif rule.startswith("LADDER"):
            tt.pop(str(r["track"]), None)
    p["track_tier_multipliers"] = tt
    return p


def _window(x: pd.DataFrame) -> dict:
    """n / matchdays / ROI / median CLV / CLV coverage of a set of settled bets."""
    n = int(len(x))
    clv = x["clv"].dropna() if n else pd.Series(dtype=float)
    return {"n": n, "matchdays": int(x["date"].nunique()) if n else 0,
            "roi": round(float(x["unit"].mean()), 4) if n else None,
            "units": round(float(x["unit"].sum()), 2) if n else 0.0,
            "clv_median": round(float(clv.median()), 4) if len(clv) else None,
            "clv_mean": round(float(clv.mean()), 4) if len(clv) else None,
            "clv_coverage": round(len(clv) / n, 3) if n else None}


def direction(since: pd.DataFrame, review_roi) -> dict:
    """The explicit rule above. Returns {state, lo, hi, reason}."""
    n, days = len(since), (since["date"].nunique() if len(since) else 0)
    if review_roi is None or pd.isna(review_roi):
        return {"state": "INSUFFICIENT", "lo": None, "hi": None, "reason": "no review-date ROI for this track"}
    if n < DIRECTION_MIN_N or days < DIRECTION_MIN_DAYS:
        return {"state": "INSUFFICIENT", "lo": None, "hi": None,
                "reason": f"n={n}, {days} matchdays since the review (needs {DIRECTION_MIN_N} / {DIRECTION_MIN_DAYS})"}
    lo, hi = block_ci(since["unit"].to_numpy(float), since["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                      level=DIRECTION_LEVEL)
    if lo is None:
        return {"state": "INSUFFICIENT", "lo": None, "hi": None, "reason": "fewer than 3 matchdays to resample"}
    state = "IMPROVING" if lo > review_roi else "DETERIORATING" if hi < review_roi else "STABLE"
    return {"state": state, "lo": round(lo, 4), "hi": round(hi, 4),
            "reason": f"90% interval [{lo:+.1%}, {hi:+.1%}] vs review ROI {review_roi:+.1%}"}


def review_meta(recommendation: dict, track_evidence: dict) -> dict:
    """When the formal review looked, and what it decided."""
    rv = (recommendation or {}).get("review") or {}
    return {"as_of": rv.get("as_of") or (track_evidence or {}).get("as_of"),
            "evidence_cutoff": rv.get("evidence_cutoff") or (track_evidence or {}).get("evidence_cutoff")
            or (recommendation or {}).get("evidence_cutoff"),
            "generated_at": (recommendation or {}).get("generated_at") or (track_evidence or {}).get("generated_at"),
            "policy_change_recommended": rv.get("POLICY_CHANGE_RECOMMENDED"),
            "recommended_policy": rv.get("RECOMMENDED_POLICY"),
            "policy_version": (recommendation or {}).get("policy_version")}


def since_review(track_evidence: dict, recommendation: dict, policy: dict | None = None,
                 bets: pd.DataFrame | None = None, today=None,
                 tier_rules: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """(meta, one row per production track). Review-date figures are Pro's own (track_evidence.json),
    never recomputed; since-review and rolling figures come from v9's ledgers, on the tiers the review's
    evidence counted (review_policy)."""
    policy = _json(POLICY_FILE) if policy is None else policy
    meta = review_meta(recommendation, track_evidence)
    b = staked(load_bets() if bets is None else bets, review_policy(policy, tier_rules))
    cutoff = pd.Timestamp(meta["evidence_cutoff"]) if meta.get("evidence_cutoff") else None
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    ev = (track_evidence or {}).get("tracks") or {}
    rows = []
    for track, t in (policy.get("tracks") or {}).items():
        x = b[b["track"] == track] if len(b) else b
        r = ev.get(track) or {}
        since = x[x["date"] > cutoff] if (cutoff is not None and len(x)) else x.iloc[0:0]
        w = _window(since)
        d = direction(since, r.get("roi"))
        r14 = _window(x[x["date"] > today - pd.Timedelta(days=14)]) if len(x) else _window(x)
        r30 = _window(x[x["date"] > today - pd.Timedelta(days=30)]) if len(x) else _window(x)
        rev_n = r.get("n")
        rows.append({
            "track": track, "state": t.get("state"),
            "review_n": rev_n, "review_matchdays": r.get("matchdays"), "review_roi": r.get("roi"),
            "review_clv_median": r.get("clv_median"),
            "review_clv_coverage": (round(r["clv_n"] / rev_n, 3) if rev_n and r.get("clv_n") is not None else None),
            "since_n": w["n"], "since_matchdays": w["matchdays"], "since_roi": w["roi"], "since_units": w["units"],
            "since_clv_median": w["clv_median"], "since_clv_coverage": w["clv_coverage"],
            "roll14_n": r14["n"], "roll14_roi": r14["roi"], "roll30_n": r30["n"], "roll30_roi": r30["roi"],
            "direction": d["state"], "direction_lo": d["lo"], "direction_hi": d["hi"], "direction_reason": d["reason"],
            "in_review": bool(r)})
    meta.update(today=today.strftime("%Y-%m-%d"), last_settled=(b["date"].max().strftime("%Y-%m-%d") if len(b) else None),
                rule=DIRECTION_RULE)
    return meta, pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 2. MODEL LAB — H-MODEL-01 per track
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def hmodel_rows(eval_path: Path | None = None, exp_path: Path | None = None, gate_path: Path | None = None,
                today=None) -> tuple[dict, pd.DataFrame]:
    """One row per H-MODEL-01 track, every number exactly as src/shadow_experiments.evaluate wrote it
    (CIs are present only once that evaluator computed them). Requirements are PG-1.0's period-2 rule."""
    ev = _json(eval_path or SHADOW_EVAL)
    exps = {e.get("experiment_id"): e for e in (_json(exp_path or EXPERIMENTS).get("experiments") or [])}
    gate = _json(gate_path or GATE_CFG)
    p = gate.get("prospective") or {}
    h = gate.get("holdout") or {}
    req = {"min_n": int(p.get("min_n_per_track") or 1000), "max_ll_delta_exclusive": float(p.get("max_ll_delta_exclusive", 0.0)),
           "max_brier_worsening": float(p.get("max_brier_worsening", 0.0001)),
           "early_fail_min_n": int(p.get("early_fail_min_n") or 300), "max_age_days": int(p.get("max_age_days") or 180),
           "min_bootstrap_blocks": int(h.get("min_bootstrap_blocks") or 20), "gate_version": gate.get("gate_version")}
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC")
    if today.tzinfo is None:
        today = today.tz_localize("UTC")
    rows = []
    for key, r in (ev.get("results") or {}).items():
        exp_id, _, track = key.partition("|")
        e = exps.get(exp_id, {})
        need = int(e.get("n_min_per_track") or req["min_n"])
        n = int(r.get("settled_paired") or 0)
        ci_ll = r.get("delta_log_loss_ci95") or [None, None]
        ci_br = r.get("delta_brier_ci95") or [None, None]
        reg = pd.to_datetime(e.get("registered_at"), utc=True, errors="coerce")
        expires = (reg + pd.Timedelta(days=req["max_age_days"])) if pd.notna(reg) else None
        dll, dbr = r.get("delta_log_loss"), r.get("delta_brier")
        rows.append({
            "experiment": exp_id, "track": track, "hypothesis": e.get("hypothesis_id", exp_id.split("__")[0]),
            "status": "SHADOW — not promoted", "experiment_status": e.get("status", ""),
            "settled_paired_n": n, "required_n": need, "progress": min(1.0, n / need) if need else 0.0,
            "matchdays": r.get("n_matchdays"),
            "ll_champion": r.get("ll_champion_raw"), "ll_challenger": r.get("ll_challenger"),
            "brier_champion": r.get("brier_champion_raw"), "brier_challenger": r.get("brier_challenger"),
            "delta_log_loss": dll, "delta_ll_lo": ci_ll[0], "delta_ll_hi": ci_ll[1],
            "delta_brier": dbr, "delta_brier_lo": ci_br[0], "delta_brier_hi": ci_br[1],
            "req_n_met": n >= need,
            "req_ll_met": dll is not None and dll < req["max_ll_delta_exclusive"],
            "req_brier_met": dbr is not None and dbr <= req["max_brier_worsening"],
            "early_fail_zone": bool(n >= req["early_fail_min_n"] and ci_ll[0] is not None and ci_ll[0] > 0),
            "registered_at": e.get("registered_at"),
            "window_expires": expires.strftime("%Y-%m-%d") if expires is not None else None,
            "days_left": int((expires - today).days) if expires is not None else None,
            "challenger_version": e.get("challenger_version")})
    meta = {"computed_at": ev.get("computed_at"), **req}
    return meta, pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 3. MARKET INTELLIGENCE — H-EDGE2 forward experiment report
# ═════════════════════════════════════════════════════════════════════════════════════════════════
EDGE2_ALLOWED = ("REGISTERED", "ACCRUING", "ELIGIBLE_FOR_REVIEW", "BLOCKED_CONFIG_CHANGED")
TWO_SIDED = ("MEDIAN_TWO_SIDED_BOOKS", "SINGLE_TWO_SIDED_BOOK")
SELECTION_STATES = ("QUALIFIED", "STRONG")
ODDS_BANDS = [1.0, 1.6, 1.9, 2.2, 2.5, 3.0, 99.0]


def _bands(odds: pd.Series) -> dict:
    c = pd.cut(pd.to_numeric(odds, errors="coerce"), ODDS_BANDS)
    return {str(k): int(v) for k, v in c.value_counts(sort=False).items()}


def shadow_forward(shadow: pd.DataFrame | None, track: str, market: str, forward_start,
                   engine_version: str = "market_edge_v2") -> dict:
    """Counts-only description of v9's first-sight shadow rows for one hypothesis since forward_start —
    the same row filter as Pro's evaluator (engine version, pre-kickoff, first sight per fixture x market).
    No outcome is read: the interim policy publishes no effect size before ELIGIBLE_FOR_REVIEW."""
    s = shadow if shadow is not None else _read(SHADOW)
    empty = {"logged": 0, "two_sided": 0, "two_sided_share": None, "selections": 0, "selection_rate": None,
             "states": {}, "odds_bands_selected": {}, "leagues": 0, "top_league": None, "top_league_share": None,
             "sides_selected": {}, "kickoff_days": 0}
    if s is None or s.empty or "selection_engine_version" not in s.columns:
        return empty
    s = s[s["selection_engine_version"].astype(str) == engine_version].copy()
    s["_pts"] = pd.to_datetime(s["prediction_ts"], errors="coerce", utc=True)
    s["_ko"] = pd.to_datetime(s["kickoff_utc"], errors="coerce", utc=True)
    s = s[s["_pts"].notna() & s["_ko"].notna() & (s["_pts"] < s["_ko"])]
    s = s.sort_values("_pts", kind="mergesort").drop_duplicates(
        ["selection_engine_version", "fixture_id", "market"], keep="first")
    fs = pd.Timestamp(forward_start) if forward_start else None
    if fs is not None and fs.tzinfo is None:
        fs = fs.tz_localize("UTC")
    g = s[(s["model_type"] == track) & (s["market"] == market)]
    if fs is not None:
        g = g[g["_pts"] >= fs]
    if g.empty:
        return empty
    sel = g[g["decision_state"].isin(SELECTION_STATES)]
    lg = g["league"].value_counts(normalize=True)
    n = int(len(g))
    return {"logged": n, "two_sided": int(g["market_source"].isin(TWO_SIDED).sum()),
            "two_sided_share": round(float(g["market_source"].isin(TWO_SIDED).mean()), 3),
            "selections": int(len(sel)), "selection_rate": round(len(sel) / n, 3),
            "states": {k: int(v) for k, v in g["decision_state"].value_counts().items()},
            "odds_bands_selected": _bands(sel["odds"]) if len(sel) else {},
            "leagues": int(g["league"].nunique()), "top_league": lg.index[0],
            "top_league_share": round(float(lg.iloc[0]), 3),
            "sides_selected": {k: int(v) for k, v in sel["side"].value_counts().items()} if len(sel) else {},
            "kickoff_days": int(g["_ko"].dt.normalize().nunique())}


def edge2_report(progress: dict | None, shadow: pd.DataFrame | None = None) -> tuple[dict, list[dict]]:
    """Per registered hypothesis: Pro's forward counts/status (+ its metrics ONLY once ELIGIBLE_FOR_REVIEW)
    and the live counts-only description from v9's shadow log. A status outside EDGE2_ALLOWED is INVALID."""
    p = progress or {}
    out = []
    for h in p.get("hypotheses") or []:
        f = h.get("forward") or {}
        s = str(h.get("status"))
        ok = s in EDGE2_ALLOWED
        eligible = ok and s == "ELIGIBLE_FOR_REVIEW"
        out.append({
            "id": h.get("id"), "track": h.get("track"), "market": h.get("market"),
            "status": s if ok else f"INVALID ({s})",
            "settled": f.get("settled_two_sided"), "min_settled": h.get("min_n_settled_two_sided"),
            "gameweeks": f.get("gameweeks"), "min_gameweeks": h.get("min_gameweeks"),
            "matchdays": (h.get("descriptive") or {}).get("matchdays"),
            "logged": f.get("logged"), "two_sided": f.get("two_sided"),
            "selections": f.get("selections_QUALIFIED_STRONG"), "states": f.get("states") or {},
            "progress_pct": h.get("progress_pct"), "pre_registration": h.get("PRE_REGISTRATION_NOT_COUNTED"),
            "descriptive": h.get("descriptive") or {},
            # outcome metrics are published by Pro ONLY at ELIGIBLE_FOR_REVIEW (registry interim policy)
            "metrics": (h.get("metrics") or {}) if eligible else {},
            "secondary": (h.get("secondary") or {}) if eligible else {},
            "criteria_met": (h.get("criteria_met") or {}) if eligible else {},
            "shadow": shadow_forward(shadow, h.get("track"), h.get("market"), p.get("forward_start"),
                                     p.get("engine_version") or "market_edge_v2")})
    meta = {k: p.get(k) for k in ("generated_at", "forward_start", "registration_intact", "engine_version",
                                  "engine_hashes_match_registration", "confirmed_assignable_here",
                                  "production_changed", "settlement_source_available")}
    return meta, out


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 4. NF OVER 1.5 — REAL PRICE ONLY
# ═════════════════════════════════════════════════════════════════════════════════════════════════
#: The constant src/backtest.py fills a missing O1.5 price with (_DEFAULT_ODDS["over15"] ==
#: src/market_edge.SYNTHETIC_DEFAULT_ODDS["OU15"]; parity is a unit test). The live ledger carries no
#: `price_synthetic` flag, so — like market_edge.exclude_synthetic_prices — a row priced exactly at the
#: fill constant is dropped (fail closed): it cannot be told apart from a filled one.
O15_FILL_PRICE = 1.40
GAP_BANDS = ("SNIPER", "MARKSMAN", "VALUABLE")          # shown with src/tier_labels words (BIG/MEDIUM/SMALL GAP)
O15_WINDOWS = (("30d", 30), ("60d", 60), ("season", None))
SMALL_N = 30


def over15_genuine(side_bets: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Settled live new-format Over 1.5 tips on a genuine quoted price, plus how many rows each rule removed."""
    s = _read(SIDE_BETS) if side_bets is None else side_bets
    if s.empty:
        return pd.DataFrame(), {}
    x = s[s["market"].astype(str).eq("over15") & s["model_type"].astype(str).eq("new_format")
          & s["source"].fillna("live").astype(str).eq("live")].copy()
    x["date"] = pd.to_datetime(x["match_date"].astype(str).str[:10], errors="coerce")
    x = x[x["date"] >= pd.Timestamp(PERFORMANCE_CUTOFF)]
    x = x[x["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
    n0 = len(x)
    x["odds"] = pd.to_numeric(x["odds"], errors="coerce")
    no_price = x["odds"].isna() | (x["odds"] <= 1.0)
    fill = x["odds"].round(4).eq(O15_FILL_PRICE)
    notes = x.get("notes", pd.Series("", index=x.index)).astype(str).str.contains("fair_odds|no_market_price", na=False)
    keep = ~(no_price | fill | notes)
    excl = {"settled_rows": int(n0), "no_price": int(no_price.sum()), "fill_constant_1.40": int((fill & ~no_price).sum()),
            "settled_at_fair_odds": int((notes & ~no_price & ~fill).sum()), "kept": int(keep.sum())}
    x = x[keep]
    x["unit"] = pd.to_numeric(x["pnl"], errors="coerce")
    x["clv"] = pd.to_numeric(x["clv_pct"], errors="coerce")
    return x[x["unit"].notna()], excl


def over15_real_price(side_bets: pd.DataFrame | None = None, today=None) -> tuple[dict, pd.DataFrame]:
    """One row per gap band x window: n, matchdays, ROI with a 95% matchday-block-bootstrap CI, ledger CLV
    mean + coverage, and the CERTIFIED CLV coverage (0 — no certified close exists for ledger O1.5 rows)."""
    x, excl = over15_genuine(side_bets)
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows = []
    for wname, days in O15_WINDOWS:
        w = x[x["date"] > today - pd.Timedelta(days=days)] if (days and len(x)) else x
        for band in GAP_BANDS + ("ALL",):
            g = w if band == "ALL" else (w[w["signal_tier"] == band] if len(w) else w)
            n = int(len(g))
            lo, hi = (block_ci(g["unit"].to_numpy(float), g["date"].dt.strftime("%Y-%m-%d").to_numpy(), level=0.95)
                      if n else (None, None))
            clv = g["clv"].dropna() if n else pd.Series(dtype=float)
            rows.append({"window": wname, "gap_band": band, "n": n,
                         "matchdays": int(g["date"].nunique()) if n else 0,
                         "roi": round(float(g["unit"].mean()), 4) if n else None,
                         "roi_lo95": None if lo is None else round(lo, 4), "roi_hi95": None if hi is None else round(hi, 4),
                         "units": round(float(g["unit"].sum()), 2) if n else 0.0,
                         "mean_odds": round(float(g["odds"].mean()), 3) if n else None,
                         "ledger_clv_mean_pct": round(float(clv.mean()), 2) if len(clv) else None,
                         "ledger_clv_coverage": round(len(clv) / n, 3) if n else None,
                         "certified_clv_mean_pct": None, "certified_clv_coverage": 0.0 if n else None,
                         "small_sample": n < SMALL_N})
    meta = {"exclusions": excl, "today": today.strftime("%Y-%m-%d"), "season_from": PERFORMANCE_CUTOFF,
            "first": x["date"].min().strftime("%Y-%m-%d") if len(x) else None,
            "last": x["date"].max().strftime("%Y-%m-%d") if len(x) else None}
    return meta, pd.DataFrame(rows)
