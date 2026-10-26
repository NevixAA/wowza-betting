"""Build config/league_market_regimes.json — the per-league x market RESEARCH regime registry.

    python scripts/build_league_regimes.py                    # PRO_LOCAL or ../pro auto-detected
    python scripts/build_league_regimes.py --pro <pro dir> --as-of 2026-10-25

RESEARCH ONLY (owner rule 2026-10-25). This file never changes a threshold, a stake, a market's
eligibility, an ACTIVE/PAPER state or a selection rule. It READS production state so the registry can
say what production does; it writes exactly one file: config/league_market_regimes.json.
`research_regime` is a label for where research effort goes — production never reads it.

ENTRIES: one per league x market x side x model_type, for every league in config.ENABLED_LEAGUES
(the tipped/tracked set; paper leagues included and classified, never staked):
    OU25 OVER, OU25 UNDER, BTTS YES, OVER15 OVER, OVER35 OVER (OVER35 kept so its BLOCKED state is
    on record; side markets are one-sided by construction in pipeline.py).

EVIDENCE (never pooled across model_type, market or side — invariant 1):
  * PRIMARY  Pro experiments/valuable_2026_10/all_fixture_settled.parquet (ALL_FIXTURE): every fixture
    v9 priced, FIRST SIGHT (immune to src/ledger.py upgrading tier/edge in place), settled, with a
    strict pre-kickoff close. "Signals" = rows whose first-sight production tier is not AVOID. On
    new-format OU25 the deployed NF_MAX_OU_ODDS cap is applied (a signal priced above it is no longer
    tipped), so the evidence describes the selection production runs today. Side-market tiers in that
    table are Pro's approximation (raw edge vs 0.10/0.08/0.04 at a 1.08 overround), not the per-league
    approved side gates — stated, not corrected.
  * CLV      CERTIFIED ONLY: close observed <= 30 min before kickoff (gap_h <= 0.5). Coverage reported.
  * MARKET   model vs de-vigged first-sight market, log loss and Brier, on EVERY settled fixture of the
    league x market that has both sides priced (OU25, BTTS). One fixture-level number, shared by the
    OU25 OVER and UNDER entries. OVER15/OVER35 have no opposite price -> UNAVAILABLE.
    Standard OU25 also reports a long horizon from Pro experiments/ou_info_inputs_2026_10/
    market_anchor_rows.parquet (2024-01 ->); it is REPORTED, NOT CLASSIFYING (frame price timing is not
    certified). New-format long horizon is not used: Pro docs/WOWZA_EDGE_REASSESSMENT_2026_10.md marks
    new-format historical prices invalid (best-of-books, 11% with margin < 0).
  * LEDGER CROSS-CHECK  v9 output/bets_ledger.csv + side_bets_ledger.csv, settled WIN/LOSS since
    config.PERFORMANCE_CUTOFF_DATE, all tiers. Informational: tiers/edges are upgraded in place and
    the side-market settlement backlog was only fixed 2026-10-25, so it may still be catching up.

CLASSIFICATION — RULES_VERSION "REGIME-1" (frozen before results were looked at; change = new version)
  Adequate evidence  = n_signals >= 30 AND independent matchdays >= 8.
  Adequate market    = n_fixtures >= 100 AND matchdays >= 8 with both-side prices.
  CI                 = 90% block bootstrap, blocks = matchdays (kickoff date), 2,000 resamples, seed =
                       crc32 of the entry key (deterministic).
  1 PROTECTED  production_state is a real-money state (PROBATION/ACTIVE_LOW/ACTIVE/ACTIVE_STRONG/
               DEGRADED), the league is not a paper league, evidence is adequate, AND
               (ROI CI lower bound > 0  OR  adequate market with model-minus-market log-loss CI upper < 0).
               owner_locked = true. Checked first: a credible staked track is protected even when a
               repair signal also fires (the conflict is written into `reason`).
  2 REPAIR     (adequate evidence AND ROI CI upper bound < 0)  OR
               (adequate market AND model-minus-market log-loss CI lower bound > 0, i.e. the market
               clearly beats the model).
  3 LEARNING   everything else (including no evidence). One losing week can never move an entry:
               every rule needs >= 8 matchdays.

repair_priority_score (REPAIR-1 weights; each term in [0, 1], shown separately in the entry):
    neg_roi              0.25 * clip(-ROI / 0.25)
    neg_clv              0.15 * clip(-mean certified CLV% / 5)        (0 if < 10 certified closes)
    market_beats_model   0.20 * clip(dLogLoss(model - market) / 0.03)  (0 if unavailable)
    drawdown             0.15 * clip(max drawdown units / 20)
    calibration_error    0.15 * clip(|mean p_model - hit rate| / 0.15)
    volume_poor_results  0.10 * clip(n_signals / 100) if ROI < 0 else 0
  Each term is multiplied by the reliability of ITS OWN evidence and the terms are summed:
    signal terms (all but market_beats_model) x min(1, n_signals/30) * min(1, matchdays/8)
    market_beats_model                         x min(1, n_fixtures/100) * min(1, matchdays/8)
  (The first draft applied the signal-count factor to the market term too, which discounted a
  100-fixture market comparison by an unrelated signal count; corrected before the first commit.)
  REPAIR_PRIORITY_HIGH = research_regime == REPAIR AND score >= 0.40.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (read-only: league lists, model_type, paper leagues, NF cap)

OUT_PATH = ROOT / "config" / "league_market_regimes.json"
POLICY_PATH = ROOT / "config" / "bankroll_policy.json"
RULES_VERSION = "REGIME-1"
SCORE_VERSION = "REPAIR-1"
MIN_N, MIN_MD, MKT_MIN_N = 30, 8, 100
N_BOOT, CI_Q = 2000, (5.0, 95.0)
CERTIFIED_MAX_GAP_H = 0.5
MIN_CERTIFIED_FOR_CLV_TERM = 10
HIGH_SCORE = 0.40
REAL_STATES = ("PROBATION", "ACTIVE_LOW", "ACTIVE", "ACTIVE_STRONG", "DEGRADED")
WEIGHTS = {"neg_roi": 0.25, "neg_clv": 0.15, "market_beats_model": 0.20, "drawdown": 0.15,
           "calibration_error": 0.15, "volume_poor_results": 0.10}
SCALES = {"neg_roi": 0.25, "neg_clv": 5.0, "market_beats_model": 0.03, "drawdown": 20.0,
          "calibration_error": 0.15, "volume_poor_results": 100.0}
# (registry market, side, all_fixture market key, ledger market key)
SPECS = [("OU25", "OVER", "ou25", "ou25"), ("OU25", "UNDER", "ou25", "ou25"), ("BTTS", "YES", "btts", "btts"),
         ("OVER15", "OVER", "over15", "over15"), ("OVER35", "OVER", "over35", "over35")]
TIPPED = ("SNIPER", "MARKSMAN", "VALUABLE")
EPS = 1e-6
#: Production policy/threshold files this script must never write, whatever --out says.
PRODUCTION_NAMES = {"bankroll_policy.json", "best_params_standard.json", "best_params_side_markets.json",
                    "league_thresholds.json", "auto_leagues.json", "research_lock_snapshot.json", "predict.yml"}

RULES_TEXT = {
    "version": RULES_VERSION,
    "adequate_evidence": f"n_signals >= {MIN_N} AND independent_matchdays >= {MIN_MD}",
    "adequate_market": f"n_fixtures >= {MKT_MIN_N} AND matchdays >= {MIN_MD}, both sides priced",
    "ci": f"90% block bootstrap by matchday, {N_BOOT} resamples, seed crc32(entry key)",
    "PROTECTED": "real-money production_state (PROBATION/ACTIVE_LOW/ACTIVE/ACTIVE_STRONG/DEGRADED), not a "
                 "paper league, adequate evidence, AND (ROI CI lower > 0 OR adequate market with "
                 "dLogLoss(model-market) CI upper < 0). owner_locked=true. Evaluated first.",
    "REPAIR": "(adequate evidence AND ROI CI upper < 0) OR (adequate market AND dLogLoss(model-market) "
              "CI lower > 0)",
    "LEARNING": "everything else, including no evidence",
    "never": "no rule fires on fewer than 8 matchdays, so one losing week cannot move an entry",
}
SCORE_TEXT = {"version": SCORE_VERSION, "weights": WEIGHTS, "scales": SCALES,
              "formula": "score = sum(evidence_factor_of_term * weight * clip(term/scale, 0, 1)); signal terms "
                         "use min(1, n_signals/30)*min(1, matchdays/8), market_beats_model uses "
                         "min(1, n_fixtures/100)*min(1, matchdays/8); neg_clv needs >= 10 certified closes; "
                         "volume_poor_results only when ROI < 0",
              "high": f"REPAIR_PRIORITY_HIGH = research_regime == REPAIR AND score >= {HIGH_SCORE}"}


# ───────────────────────────── helpers ─────────────────────────────

def _f(x, nd=4):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def _seed(key: str) -> int:
    return zlib.crc32(key.encode("utf-8"))


def block_boot(values: np.ndarray, blocks: np.ndarray, key: str, stat=np.mean) -> tuple:
    """90% CI of `stat` by resampling whole blocks (matchdays) with replacement. Deterministic."""
    if len(values) == 0:
        return (None, None)
    ub = np.unique(blocks)
    groups = [values[blocks == b] for b in ub]
    if len(ub) < 2:
        return (None, None)
    rng = np.random.default_rng(_seed(key))
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(ub), len(ub))
        out[i] = stat(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.percentile(out, CI_Q)
    return (float(lo), float(hi))


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


def _ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def production_state(policy: dict, league: str, model_type: str, market: str, side: str | None = None) -> dict:
    """Exactly what production applies: bankroll_policy track state, then the paper-league override
    (src/bankroll_policy.size_day: PAPER_LEAGUE -> no stake; config.PAPER_LEAGUES -> never sent).
    The track key is src/bankroll_policy.track_key: O/U 2.5 is per side since 2026-10-26."""
    from src.bankroll_policy import track_key
    track = track_key(model_type, market, side)
    t = (policy.get("tracks") or {}).get(track)
    track_state = t.get("state", "UNSUPPORTED") if t else "UNSUPPORTED"
    paper = league in set(policy.get("paper_leagues", [])) or config.is_paper_league(league)
    return {"track": track, "track_state": track_state,
            "base_fraction": _f((t or {}).get("base_fraction", 0.0), 6),
            "paper_league": bool(paper),
            "production_state": "PAPER_LEAGUE" if paper else track_state,
            "provider_supported": bool(config.provider_supported(league))}


# ───────────────────────────── evidence ─────────────────────────────

def signal_metrics(sig: pd.DataFrame, key: str) -> dict:
    n = len(sig)
    if n == 0:
        return {"evidence_n": 0, "independent_matchdays": 0, "recent_roi": {"roi": None, "ci90": [None, None],
                "units": None}, "max_drawdown_units": None, "calibration_error": None, "hit_rate": None,
                "mean_p_model": None, "mean_odds": None, "tier_mix": {}}
    sig = sig.sort_values(["kick", "fx"])
    pnl = sig["pnl"].to_numpy(float)
    md = sig["md"].astype(str).to_numpy()
    lo, hi = block_boot(pnl, md, key + "|roi")
    return {"evidence_n": int(n), "independent_matchdays": int(len(np.unique(md))),
            "recent_roi": {"roi": _f(pnl.mean()), "ci90": [_f(lo), _f(hi)], "units": _f(pnl.sum(), 2)},
            "max_drawdown_units": _f(max_drawdown(pnl), 2),
            "calibration_error": _f(sig["p_model"].mean() - sig["won"].mean()),
            "hit_rate": _f(sig["won"].mean()), "mean_p_model": _f(sig["p_model"].mean()),
            "mean_odds": _f(sig["odds"].mean(), 3),
            "tier_mix": {k: int(v) for k, v in sorted(sig["prod_tier"].value_counts().items())}}


def clv_metrics(sig: pd.DataFrame) -> dict:
    n = len(sig)
    cert = sig[(sig.get("close_status") == "VALID_PRE_KICKOFF_CLOSE") & (sig["gap_h"] <= CERTIFIED_MAX_GAP_H)
               & sig["clv"].notna()] if n else sig
    c = len(cert)
    return {"certified_definition": "close observed <= 30 min before kickoff", "n_certified": int(c),
            "coverage": _f(c / n) if n else None,
            "mean_clv_pct": _f(cert["clv"].mean(), 3) if c else None,
            "median_clv_pct": _f(cert["clv"].median(), 3) if c else None,
            "share_positive": _f((cert["clv"] > 0).mean()) if c else None}


def market_metrics(fx: pd.DataFrame, key: str) -> dict:
    """Model vs de-vigged market on every settled fixture with both sides priced (all tiers)."""
    x = fx[fx["p_market_fair"].notna()] if len(fx) else fx
    if len(x) == 0:
        return {"status": "UNAVAILABLE", "why": "no opposite-side price in the first-sight snapshots"
                if len(fx) else "no settled fixtures", "n_fixtures": 0}
    under = x["side"].isin(["UNDER", "NO"]).to_numpy()
    pm = np.where(under, 1 - x["p_model"], x["p_model"]).astype(float)
    pk = np.where(under, 1 - x["p_market_fair"], x["p_market_fair"]).astype(float)
    y = x["outcome_yes"].to_numpy(float)
    d = _ll(pm, y) - _ll(pk, y)
    md = x["md"].astype(str).to_numpy()
    lo, hi = block_boot(d, md, key + "|dll")
    return {"status": "OK", "basis": "first-sight de-vigged two-way price, every settled fixture",
            "n_fixtures": int(len(x)), "matchdays": int(len(np.unique(md))),
            "logloss_model": _f(_ll(pm, y).mean()), "logloss_market": _f(_ll(pk, y).mean()),
            "dlogloss_model_minus_market": _f(d.mean(), 5), "dlogloss_ci90": [_f(lo, 5), _f(hi, 5)],
            "brier_model": _f(((pm - y) ** 2).mean()), "brier_market": _f(((pk - y) ** 2).mean())}


def long_horizon(anchor: pd.DataFrame, league: str) -> dict:
    a = anchor[(anchor["track"] == "standard") & (anchor["league"] == league)] if len(anchor) else anchor
    a = a.dropna(subset=["p_mkt", "p_base", "y"]) if len(a) else a
    if len(a) == 0:
        return {"status": "UNAVAILABLE", "why": "no standard market-anchor rows for this league"}
    y = a["y"].to_numpy(float)
    d = _ll(a["p_base"].to_numpy(float), y) - _ll(a["p_mkt"].to_numpy(float), y)
    return {"status": "REPORTED_NOT_CLASSIFYING", "source": "pro:experiments/ou_info_inputs_2026_10/"
            "market_anchor_rows.parquet", "n": int(len(a)),
            "from": str(pd.Timestamp(a["date"].min()).date()), "to": str(pd.Timestamp(a["date"].max()).date()),
            "dlogloss_model_minus_market": _f(d.mean(), 5),
            "brier_model": _f(((a["p_base"] - y) ** 2).mean()), "brier_market": _f(((a["p_mkt"] - y) ** 2).mean())}


def ledger_crosscheck(bets: pd.DataFrame, side: pd.DataFrame, league: str, mkt: str, sd: str) -> dict:
    if mkt == "ou25":
        g = bets[(bets["league"] == league) & (bets["side"].astype(str).str.upper() == sd)] if len(bets) else bets
    else:
        g = side[(side["league"] == league) & (side["market"].astype(str).str.lower() == mkt)] if len(side) else side
    if len(g) == 0:
        return {"n_settled": 0}
    g = g[g["match_date"].astype(str).str[:10] >= str(config.PERFORMANCE_CUTOFF_DATE)]
    g = g[g["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
    pnl = pd.to_numeric(g["pnl"], errors="coerce").dropna()
    if len(pnl) == 0:
        return {"n_settled": 0}
    return {"n_settled": int(len(pnl)), "roi": _f(pnl.mean()), "units": _f(pnl.sum(), 2),
            "as_of_last_settled_match": str(g["match_date"].astype(str).str[:10].max()),
            "caveat": "all tiers, tier/edge upgraded in place by src/ledger.py; side settlement backlog "
                      "fixed 2026-10-25 and may still be catching up"}


# ───────────────────────────── classification ─────────────────────────────

def classify(ps: dict, sm: dict, mk: dict) -> tuple[str, str]:
    n, md = sm["evidence_n"], sm["independent_matchdays"]
    lo, hi = sm["recent_roi"]["ci90"]
    adequate = n >= MIN_N and md >= MIN_MD
    mkt_ok = mk.get("status") == "OK" and mk["n_fixtures"] >= MKT_MIN_N and mk["matchdays"] >= MIN_MD
    mlo, mhi = (mk.get("dlogloss_ci90") or [None, None]) if mkt_ok else (None, None)
    pos_roi = adequate and lo is not None and lo > 0
    neg_roi = adequate and hi is not None and hi < 0
    mkt_beats_model = mkt_ok and mlo is not None and mlo > 0
    model_beats_mkt = mkt_ok and mhi is not None and mhi < 0
    real = ps["production_state"] in REAL_STATES
    if real and adequate and (pos_roi or model_beats_mkt):
        why = ("ROI 90% CI lower bound > 0" if pos_roi else "model beats de-vigged market (dLL CI upper < 0)")
        if neg_roi or mkt_beats_model:
            why += "; CONFLICT: repair signal also fires (" + (
                "market beats model" if mkt_beats_model else "ROI CI upper < 0") + ")"
        return "PROTECTED", f"staked track {ps['track']} ({ps['production_state']}) with credible positive " \
                            f"evidence: {why}"
    if neg_roi or mkt_beats_model:
        parts = []
        if neg_roi:
            parts.append(f"ROI 90% CI upper bound < 0 on n={n}, {md} matchdays")
        if mkt_beats_model:
            parts.append(f"market beats model: dLL CI lower {mlo} > 0 on {mk['n_fixtures']} fixtures")
        return "REPAIR", "; ".join(parts)
    if n == 0 and mk.get("status") != "OK":
        return "LEARNING", "no settled evidence in the first-sight record"
    if not adequate:
        return "LEARNING", f"evidence below the adequacy bar (n={n}, matchdays={md}; need {MIN_N}/{MIN_MD})"
    return "LEARNING", "adequate evidence but neither a credible positive nor a credible negative result"


def repair_score(regime: str, sm: dict, clv: dict, mk: dict) -> dict:
    n, md = sm["evidence_n"], sm["independent_matchdays"]
    roi = sm["recent_roi"]["roi"]
    raw = {
        "neg_roi": -roi if roi is not None else 0.0,
        "neg_clv": -(clv["mean_clv_pct"] or 0.0) if clv["n_certified"] >= MIN_CERTIFIED_FOR_CLV_TERM else 0.0,
        "market_beats_model": (mk.get("dlogloss_model_minus_market") or 0.0) if mk.get("status") == "OK" else 0.0,
        "drawdown": sm["max_drawdown_units"] or 0.0,
        "calibration_error": abs(sm["calibration_error"] or 0.0),
        "volume_poor_results": float(n) if (roi is not None and roi < 0) else 0.0,
    }
    ef = min(1.0, n / MIN_N) * min(1.0, md / MIN_MD)
    mf = (min(1.0, mk["n_fixtures"] / MKT_MIN_N) * min(1.0, mk["matchdays"] / MIN_MD)
          if mk.get("status") == "OK" else 0.0)
    terms = {k: _f((mf if k == "market_beats_model" else ef) * WEIGHTS[k]
                   * float(np.clip(raw[k] / SCALES[k], 0.0, 1.0))) for k in WEIGHTS}
    score = _f(sum(terms.values()))
    return {"repair_priority_score": score, "repair_priority_terms": terms,
            "evidence_factor": {"signals": _f(ef), "market": _f(mf)},
            "REPAIR_PRIORITY_HIGH": bool(regime == "REPAIR" and score is not None and score >= HIGH_SCORE)}


# ───────────────────────────── build ─────────────────────────────

def build_registry(af: pd.DataFrame, anchor: pd.DataFrame, bets: pd.DataFrame, side: pd.DataFrame,
                   policy: dict, as_of: str, leagues=None, sources: dict | None = None) -> dict:
    """Pure: same inputs -> byte-identical output (no clock, seeded bootstrap, sorted keys)."""
    leagues = sorted(leagues if leagues is not None else config.ENABLED_LEAGUES)
    af = af.copy()
    af["kick"] = pd.to_datetime(af["kick"], utc=True)
    af["md"] = pd.to_datetime(af["md"]).dt.strftime("%Y-%m-%d")
    cap = float(getattr(config, "NF_MAX_OU_ODDS", 0) or 0)
    cutoff = str(af["kick"].max().date()) if len(af) else None
    entries = []
    for lg in leagues:
        mt = config.model_type_for_league(lg)
        for market, sd, akey, lkey in SPECS:
            key = f"{lg}|{market}|{sd}|{mt}"
            ps = production_state(policy, lg, mt, market, sd)
            fx = af[(af["league"] == lg) & (af["market"] == akey) & (af["model_type"] == mt)]
            sig = fx[(fx["side"] == sd) & fx["prod_tier"].isin(TIPPED)]
            if akey == "ou25" and mt == "new_format" and cap > 0:
                sig = sig[sig["odds"] <= cap]
            sm = signal_metrics(sig, key)
            clv = clv_metrics(sig)
            mk = market_metrics(fx, f"{lg}|{akey}|{mt}")
            if mk.get("status") == "OK" and akey == "ou25":
                mk["note"] = "fixture-level, shared by the OU25 OVER and UNDER entries"
            if akey == "ou25" and mt == "standard":
                mk["long_horizon"] = long_horizon(anchor, lg)
            elif akey == "ou25":
                mk["long_horizon"] = {"status": "UNAVAILABLE", "why": "new-format historical prices invalid "
                                      "(Pro docs/WOWZA_EDGE_REASSESSMENT_2026_10.md)"}
            regime, reason = classify(ps, sm, mk)
            if not ps["provider_supported"]:
                reason += f"; provider does not carry this league ({config.unsupported_reason(lg)})"
            rs = repair_score(regime, sm, clv, mk)
            entries.append({
                "key": key, "league": lg, "market": market, "side": sd, "model_type": mt,
                "research_regime": regime, "owner_locked": regime == "PROTECTED",
                **{k: ps[k] for k in ("production_state", "track", "track_state", "base_fraction",
                                      "paper_league", "provider_supported")},
                "reason": reason, "evidence_cutoff": cutoff, **sm, "clv": clv,
                "market_relative_metric": mk, "ledger_crosscheck": ledger_crosscheck(bets, side, lg, lkey, sd),
                **rs, "last_updated": as_of})
    counts = {r: sum(e["research_regime"] == r for e in entries) for r in ("PROTECTED", "REPAIR", "LEARNING")}
    high = sorted([e for e in entries if e["REPAIR_PRIORITY_HIGH"]],
                  key=lambda e: (-e["repair_priority_score"], e["key"]))
    return {
        "about": "RESEARCH regime registry (owner rule 2026-10-25): labels where research effort goes. "
                 "Production never reads this file; nothing here changes a threshold, stake, eligibility, "
                 "ACTIVE/PAPER state or selection rule. Generated by scripts/build_league_regimes.py - "
                 "do not hand-edit.",
        "generated_for": as_of, "evidence_cutoff": cutoff, "rules": RULES_TEXT, "repair_priority": SCORE_TEXT,
        "policy_version": policy.get("policy_version"), "sources": sources or {},
        "global_protections": {
            "player_props": (policy.get("tracks") or {}).get("player_props", {}).get("state"),
            "standard|OU25_OVER": (policy.get("tracks") or {}).get("standard|OU25_OVER", {}).get("state"),
            "standard|OU25_UNDER": (policy.get("tracks") or {}).get("standard|OU25_UNDER", {}).get("state"),
            "new_format|OVER35": (policy.get("tracks") or {}).get("new_format|OVER35", {}).get("state"),
            "standard|OVER35": (policy.get("tracks") or {}).get("standard|OVER35", {}).get("state"),
            "cloudbet": "PAPER_ONLY (src/cloudbet/client.py MODE defaults to DRY; no workflow sets CLOUDBET_MODE)"},
        "tracked_leagues": leagues, "n_entries": len(entries), "regime_counts": counts,
        "repair_priority_high": [e["key"] for e in high],
        "entries": entries,
    }


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"


def find_pro(arg: str | None) -> Path:
    cands = [arg, os.getenv("PRO_LOCAL"), ROOT.parent / "pro", ROOT.parents[3] / "pro" if len(ROOT.parents) > 3 else None,
             Path.home() / "dev" / "wowza" / "pro"]
    for c in cands:
        if c and (Path(c) / "experiments" / "valuable_2026_10" / "all_fixture_settled.parquet").exists():
            return Path(c)
    raise SystemExit("Pro checkout not found (pass --pro or set PRO_LOCAL); the registry is built from Pro evidence")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro")
    ap.add_argument("--as-of", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.resolve() != OUT_PATH.resolve() and out.resolve().parent == (ROOT / "config").resolve():
        raise SystemExit("refusing to write any other file under config/")
    if out.name in PRODUCTION_NAMES or out.suffix not in (".json",):
        raise SystemExit(f"refusing to write {out.name}: the registry is a .json research file only")
    pro = find_pro(a.pro)
    afp = pro / "experiments" / "valuable_2026_10" / "all_fixture_settled.parquet"
    anp = pro / "experiments" / "ou_info_inputs_2026_10" / "market_anchor_rows.parquet"
    bp, sp = ROOT / "output" / "bets_ledger.csv", ROOT / "output" / "side_bets_ledger.csv"
    af = pd.read_parquet(afp)
    anchor = pd.read_parquet(anp) if anp.exists() else pd.DataFrame()
    bets = pd.read_csv(bp, low_memory=False) if bp.exists() else pd.DataFrame()
    side = pd.read_csv(sp, low_memory=False) if sp.exists() else pd.DataFrame()
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    sources = {"all_fixture_settled": {"path": "pro:experiments/valuable_2026_10/all_fixture_settled.parquet",
                                       "sha256_16": _sha(afp)},
               "market_anchor_rows": {"path": "pro:experiments/ou_info_inputs_2026_10/market_anchor_rows.parquet",
                                      "sha256_16": _sha(anp)},
               "bets_ledger": {"path": "output/bets_ledger.csv", "sha256_16": _sha(bp)},
               "side_bets_ledger": {"path": "output/side_bets_ledger.csv", "sha256_16": _sha(sp)},
               "bankroll_policy": {"path": "config/bankroll_policy.json", "sha256_16": _sha(POLICY_PATH)}}
    reg = build_registry(af, anchor, bets, side, policy, a.as_of, sources=sources)
    out.write_text(json.dumps(reg, indent=1, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{out}: {reg['n_entries']} entries {reg['regime_counts']}; HIGH: {reg['repair_priority_high']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
