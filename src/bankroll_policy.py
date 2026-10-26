"""Deterministic bankroll-policy reader: qualifying selections in, stake fractions + audit out.

    from src.bankroll_policy import load_policy, size_day
    rows = size_day(selections, load_policy(), bankroll=100.0, peak=100.0)

STAKING IS DOWNSTREAM OF PREDICTION. This module never decides whether a tip exists and never
reads model output beyond what a selection already carries. It only answers "given this
qualifying selection, how much of the CURRENT bankroll, under the approved policy?". The policy
file (config/bankroll_policy.json) is produced by the monthly review in wowzaV9-Pro and changes
only through a reviewed PR — this file never fetches anything and never decides evidence.

WIRED (2026-10-14) into predict via src/stake_plan.py -> output/stake_plan.csv; PAPER stakes only, never
notify/execution. It is the single sizing implementation:
Pro's bankroll simulator imports this exact file (read-only, via V9_LOCAL), so every backtest
sizes bets with the code production would use.

Order of operations (owner brief, section 32):
 1 track = track_key(model_type, market, side): model_type|market, except O/U 2.5, which is split by
   SIDE (model_type|OU25_OVER / model_type|OU25_UNDER, owner 2026-10-26). An O/U 2.5 tip without a valid
   side maps to model_type|OU25_UNKNOWN_SIDE, which no policy defines -> no stake (fail closed)
 2 state/base (0 -> no real stake)   3 paper league -> 0
 4 odds caps (track min/max, hard 3.00)   5 tier multiplier   6 convergence multiplier
 7 ONE position per fixture, chosen by the track's evidence_score (never by raw edge)
 8 single-bet / fixture cap   9 drawdown throttle   10 league and league x market daily caps
 11 daily exposure cap: every stake scaled by cap / total   -> final fraction, stake amount

selections: one row per qualifying tip with columns
    fixture, league, model_type, market (OU25/BTTS/OVER15/OVER35), side, signal_tier, odds
    (`side` is required for OU25: without it a tip cannot be routed to a side track and gets no stake)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "bankroll_policy.json"

REAL_STATES = ("PROBATION", "ACTIVE_LOW", "ACTIVE", "ACTIVE_STRONG", "DEGRADED")
# Absolute ceilings the reader enforces whatever a policy file says (owner brief 4, 7, 9).
HARD = {"single_bet_cap": 0.02, "fixture_cap": 0.02, "daily_cap": 0.075, "league_daily_cap": 0.04,
        "league_market_daily_cap": 0.03, "max_odds": 2.75, "blocked_odds": 3.0, "base_fraction": 0.015}

#: Markets whose policy track is split by SIDE (owner 2026-10-26). Each side is its own track with its
#: own state, base fraction, odds caps, tier multipliers and evidence, and Pro reviews it separately.
SIDE_SPLIT_MARKETS = {"OU25": ("OVER", "UNDER")}
UNKNOWN_SIDE = "UNKNOWN_SIDE"


def track_key(model_type, market, side=None) -> str:
    """THE policy track key. `model_type|market` (new_format|BTTS), except a side-split market, which is
    `model_type|market_SIDE` (new_format|OU25_OVER, standard|OU25_UNDER). A side-split market with a
    missing or unrecognised side -> `model_type|market_UNKNOWN_SIDE`, a key no policy defines, so the tip
    can never be staked on the other side's settings (fail closed). Self-contained on purpose: Pro
    imports this file alone and keys its track evidence with this same function."""
    mt, mk = str(model_type), str(market)
    sides = SIDE_SPLIT_MARKETS.get(mk.strip().upper())
    if sides is None:
        return f"{mt}|{mk}"
    s = "" if side is None or (isinstance(side, float) and side != side) else str(side).strip().upper()
    return f"{mt}|{mk}_{s if s in sides else UNKNOWN_SIDE}"


def split_track(track) -> tuple[str, str, str | None]:
    """Inverse of track_key: 'new_format|OU25_OVER' -> ('new_format', 'OU25', 'OVER');
    'new_format|BTTS' -> ('new_format', 'BTTS', None); 'player_props' -> ('player_props', '', None)."""
    mt, _, mk = str(track).partition("|")
    for base, sides in SIDE_SPLIT_MARKETS.items():
        for s in sides + (UNKNOWN_SIDE,):
            if mk == f"{base}_{s}":
                return mt, base, s
    return mt, mk, None


AUDIT_COLS = ["policy_version", "fixture", "league", "model_type", "market", "track",
              "signal_tier", "odds", "deployment_state", "bankroll_before", "base_fraction",
              "tier_multiplier", "league_multiplier", "drawdown_multiplier", "requested_fraction",
              "capped_fraction", "daily_scale", "final_fraction", "stake_amount", "reason"]


def load_policy(path: str | Path | None = None) -> dict:
    with open(path or POLICY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def drawdown_multiplier(policy: dict, bankroll: float, peak: float,
                        floor: float | None = None) -> float:
    """Throttle from the high-water mark. `floor` (from the caller's state) keeps a reduced level
    until it is lifted at a review — recovery is gradual, never an instant snap back."""
    dd = 0.0 if peak <= 0 else max(0.0, 1.0 - bankroll / peak)
    m = 1.0
    for thr, mult in sorted(policy.get("drawdown_throttle", []), key=lambda x: x[0]):
        if dd >= thr:
            m = float(mult)
    return m if floor is None else min(m, float(floor))


def rating_scale(rating: int | None, min_scale: float) -> float:
    """League rating 1-10 -> stake scale (owner 2026-10-18): 10 -> 1.0 (today's stake), 1 -> min_scale,
    linear between. Never above 1.0, so a rating can only reduce a stake."""
    if rating is None:
        return min_scale
    r = min(10, max(1, int(rating)))
    return min_scale + (r - 1) / 9.0 * (1.0 - min_scale)


def goal_direction(market, side=None) -> str | None:
    """OVER / UNDER for a goal-total position (OU25 by its side, OVER15/OVER35/OVERxx -> OVER,
    UNDERxx -> UNDER); None for BTTS and anything else, which never conflicts. Self-contained on
    purpose: Pro's simulator imports this file alone. Mirrors src/direction_guard.goal_direction."""
    m = str(market or "").strip().upper()
    s = str(side or "").strip().upper()
    if m.startswith("BTTS"):
        return None
    if m.startswith("OVER"):
        return "OVER"
    if m.startswith("UNDER"):
        return "UNDER"
    if m in ("OU25", "OU", ""):
        return s if s in ("OVER", "UNDER") else None
    return None


def size_day(selections: pd.DataFrame, policy: dict, bankroll: float, peak: float,
             dd_floor: float | None = None, league_ratings: dict | None = None,
             prior_directions: dict | None = None) -> pd.DataFrame:
    """Stake fractions for one day's qualifying selections. Pure: same input -> same output.

    league_ratings ({league: 1-10}, Pro league_ratings.json) is applied only when the policy's
    league_rating_sizing.enabled is true: each stake x rating_scale; an unrated league gets the minimum.
    Callers that pass nothing (Pro's simulator) are unaffected.

    prior_directions ({fixture: "OVER"|"UNDER"}) = the goal direction of positions ALREADY staked on a
    fixture by earlier runs (src/stake_plan reads them from stake_audit.csv). See step 6b."""
    cols = list(selections.columns)
    if selections.empty:
        return pd.DataFrame(columns=cols + [c for c in AUDIT_COLS if c not in cols])
    pf = policy["portfolio"]
    caps = {k: min(float(pf.get(k, HARD[k])), HARD[k]) for k in
            ("single_bet_cap", "fixture_cap", "daily_cap", "league_daily_cap", "league_market_daily_cap")}
    tiers = policy.get("tier_multipliers", {})
    conv = policy.get("convergence", {})
    conv_leagues = set(conv.get("leagues", []))
    conv_mult = min(float(conv.get("multiplier", 1.0)), 1.20)
    paper = set(policy.get("paper_leagues", []))
    hard_odds = min(float(pf.get("hard_max_odds", HARD["blocked_odds"])), HARD["blocked_odds"])
    ddm = drawdown_multiplier(policy, bankroll, peak, dd_floor)
    lrs = policy.get("league_rating_sizing") or {}
    use_rating = bool(lrs.get("enabled")) and league_ratings is not None
    min_scale = min(max(float(lrs.get("min_scale", 0.2)), 0.0), 1.0)

    d = selections.copy().reset_index(drop=True)
    side_in = d["side"] if "side" in d.columns else pd.Series(None, index=d.index, dtype=object)
    d["track"] = [track_key(mt, mk, sd) for mt, mk, sd in zip(d["model_type"], d["market"], side_in)]
    out = []
    for r in d.itertuples(index=False):
        t = policy["tracks"].get(r.track, {"state": "UNSUPPORTED", "base_fraction": 0.0})
        base = min(float(t.get("base_fraction", 0.0)), HARD["base_fraction"])
        state = t.get("state", "UNSUPPORTED")
        odds = float(r.odds) if pd.notna(r.odds) else float("nan")
        # PER-TRACK TIER MULTIPLIERS (owner 2026-10-22): a track whose edge does not rank outcomes
        # (OU25: SNIPER -9.9% < MARKSMAN -5.8% < VALUABLE -0.8% since Aug 26) stakes every tier the
        # same instead of paying most on the biggest raw edge. Absent override -> the global ladder.
        tt = (policy.get("track_tier_multipliers") or {}).get(r.track)
        tm = float((tt if tt is not None else tiers).get(str(r.signal_tier), 0.0))
        lm = conv_mult if r.league in conv_leagues else 1.0
        reason = "OK"
        if str(r.track).endswith(f"_{UNKNOWN_SIDE}"):
            reason = "NO_REAL_STAKE_UNKNOWN_SIDE"           # side-split market without a usable side
        elif state not in REAL_STATES or base <= 0:
            reason = f"NO_REAL_STAKE_{state}"
        elif r.league in paper:
            reason = "PAPER_LEAGUE"
        elif not (odds == odds) or odds >= hard_odds:
            reason = "ODDS_BLOCKED"
        elif odds > min(float(t.get("max_odds", HARD["max_odds"])), HARD["max_odds"]):
            reason = "ODDS_ABOVE_TRACK_CAP"
        elif odds < float(t.get("min_odds", 1.0)):
            reason = "ODDS_BELOW_TRACK_FLOOR"
        elif tm <= 0:
            reason = "TIER_NOT_STAKED"
        rt = league_ratings.get(r.league) if use_rating else None
        rs = rating_scale(rt, min_scale) if use_rating else 1.0
        req = base * tm * lm * rs if reason == "OK" else 0.0
        out.append({**r._asdict(), "deployment_state": state, "base_fraction": base,
                    "tier_multiplier": tm, "league_multiplier": lm, "requested_fraction": req,
                    "league_rating": rt, "rating_scale": round(rs, 3),
                    "_score": float(t.get("evidence_score", 0.0)), "reason": reason})
    a = pd.DataFrame(out)

    # 6b. ONE GOAL DIRECTION PER FIXTURE (owner rule 2026-10-25). Defence in depth behind
    # src/direction_guard.py, which already removes such tips before they reach this reader: never
    # stake Over and Under goal-total positions on one match (O/U 2.5 Over v Under, Over 1.5 v Under
    # 2.5, Over 3.5 v Under 2.5, ...). BTTS has no direction and never conflicts. The FIRST position
    # stands: a direction already staked by an earlier run (prior_directions), else today's staked O/U
    # 2.5 position, else the first staked one in input order. Only the opposite-direction stake is
    # zeroed; nothing else about sizing changes.
    if len(a):
        live = a["requested_fraction"] > 0
        sides = a["side"] if "side" in a.columns else pd.Series(None, index=a.index, dtype=object)
        dirs = pd.Series([goal_direction(m, s) for m, s in zip(a["market"], sides)], index=a.index,
                         dtype=object)
        fixed = dict(prior_directions or {})
        for pri in (True, False):                        # OU25 first, then any other staked position
            for i in a.index[live & dirs.notna()]:
                fx = a.at[i, "fixture"]
                if fx not in fixed and (not pri or str(a.at[i, "market"]).upper() == "OU25"):
                    fixed[fx] = dirs[i]
        opp = [i for i in a.index[live & dirs.notna()]
               if fixed.get(a.at[i, "fixture"]) not in (None, dirs[i])]
        if opp:
            a.loc[opp, "requested_fraction"] = 0.0
            a.loc[opp, "reason"] = "OPPOSITE_DIRECTION_ON_FIXTURE"

    # 7. one real position per fixture: highest evidence_score, then tier, then base; never edge
    live = a["requested_fraction"] > 0
    if pf.get("one_position_per_fixture", True) and live.any():
        rank = a[live].sort_values(["fixture", "_score", "tier_multiplier", "base_fraction"],
                                   ascending=[True, False, False, False], kind="mergesort")
        keep = set(rank.drop_duplicates("fixture", keep="first").index)
        drop = [i for i in rank.index if i not in keep]
        a.loc[drop, "requested_fraction"] = 0.0
        a.loc[drop, "reason"] = "ONE_POSITION_PER_FIXTURE"

    # 8. per-bet / per-fixture cap, 9. drawdown throttle
    a["capped_fraction"] = a["requested_fraction"].clip(upper=min(caps["single_bet_cap"], caps["fixture_cap"]))
    a["drawdown_multiplier"] = ddm
    f = a["capped_fraction"] * ddm
    if ddm == 0.0:
        a.loc[a["requested_fraction"] > 0, "reason"] = "DRAWDOWN_STOP_PAPER_REVIEW_REQUIRED"

    # 10. league/day and league x market/day caps: scale inside the group down to the cap
    for keys, cap in ((["league"], caps["league_daily_cap"]),
                      (["league", "market"], caps["league_market_daily_cap"])):
        tot = f.groupby([a[k] for k in keys]).transform("sum")
        scale = (cap / tot).where(tot > cap, 1.0).fillna(1.0)
        f = f * scale

    # 11. daily cap: scale every stake proportionally
    total = float(f.sum())
    daily_scale = caps["daily_cap"] / total if total > caps["daily_cap"] else 1.0
    a["daily_scale"] = daily_scale
    a["final_fraction"] = (f * daily_scale).round(6)
    a["bankroll_before"] = float(bankroll)
    a["stake_amount"] = (a["final_fraction"] * float(bankroll)).round(2)
    a["policy_version"] = policy.get("policy_version", "?")
    extra = [c for c in ("league_rating", "rating_scale") if use_rating and c not in cols]
    return a.drop(columns=["_score"])[cols + [c for c in AUDIT_COLS if c not in cols] + extra]


def log_decisions(rows: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Append every stake decision (all AUDIT_COLS, zero stakes included with their reason) to
    output/stake_audit.csv, so any stake can be reproduced from policy_version + inputs."""
    p = Path(path) if path else POLICY_PATH.parents[1] / "output" / "stake_audit.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.assign(logged_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"))
    rows.to_csv(p, mode="a", header=not p.exists(), index=False)
    return p


if __name__ == "__main__":
    # Example / smoke: python -m src.bankroll_policy
    demo = pd.DataFrame([
        {"fixture": "A v B", "league": "Argentina Primera Division", "model_type": "new_format",
         "market": "BTTS", "side": "YES", "signal_tier": "SNIPER", "odds": 2.10},
        {"fixture": "A v B", "league": "Argentina Primera Division", "model_type": "new_format",
         "market": "OU25", "side": "OVER", "signal_tier": "SNIPER", "odds": 2.05},
        {"fixture": "C v D", "league": "Championship", "model_type": "standard",
         "market": "OU25", "side": "UNDER", "signal_tier": "SNIPER", "odds": 1.95},
    ])
    print(size_day(demo, load_policy(), bankroll=1000.0, peak=1000.0)[
        ["fixture", "track", "signal_tier", "odds", "final_fraction", "stake_amount", "reason"]].to_string())
