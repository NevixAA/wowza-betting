"""One goal DIRECTION per fixture: never tip Over and Under goal-total lines on the same match.

OWNER RULE (2026-10-25, verbatim): "what i meant to deny its over 2.5 and under 2.5 for the same
match, or over 1.5 and under 2.5, or under 3.5 and over 2.5". Several tips on one match in the SAME
direction are fine ("thats ok" about Over 1.5 + BTTS on San Lorenzo v Newell's).

The case that triggered it: 2026-10-25 Viking FK v Tromso went out as O/U 2.5 UNDER SNIPER (tipped
2026-10-18) and Over 3.5 VALUABLE (tipped 2026-10-20). If the match has 4+ goals the Under loses and
the Over 3.5 wins; 3 goals and BOTH lose. Two tips that hedge each other are a vig donation, and the
fixture becomes unreadable in the ledger.

DIRECTION
    any Over goal-total line (O/U 2.5 OVER, over15, over35, any future overXX)  -> OVER
    any Under goal-total line (O/U 2.5 UNDER, any future underXX)               -> UNDER
    BTTS (and anything else that is not a goal-total line)                     -> None, never conflicts

WHICH TIP STANDS: THE FIRST ONE (same principle as the O/U 2.5 flip guard, src/ledger.drop_flipped /
committed_sides and the notifier's "both-sides" skip).
    across runs   the earliest live tip already in bets_ledger.csv / side_bets_ledger.csv for the
                  fixture fixes its direction; a later opposite-direction tip is suppressed.
    within a run  (no earlier tip) the O/U 2.5 tip stands, because O/U 2.5 is the main market, and an
                  opposite-direction side-market tip is suppressed. Between two side markets of
                  opposite direction in one run (none exist today: every side market is OVER/YES) the
                  highest-edge one stands, being first in _generate_side_bets' edge-descending order.

WHERE IT SITS. pipeline.mode_predict calls apply() right after generate_bets + _generate_side_bets,
before anything is written: so a suppressed tip never reaches bets.csv / side_bets.csv (the Telegram
send sources), bets_ledger.csv / side_bets_ledger.csv (and therefore the morning album, tips_table and
top picks, which read the ledgers), or the stake plan. It is recorded instead in
output/direction_conflicts.csv (first sight) as evidence.

WHAT IT LEAVES ALONE. An O/U 2.5 tip opposite to an earlier O/U 2.5 tip is the EXISTING flip guard's
case (it is not sent, not ledgered, and noted FLIP_SEEN on the original ledger row): it is passed
through untouched here so that guard behaves exactly as before. The market-first shadow engine still
sees every tip (research logs everything). Thresholds, tiers and policy states are not read or changed.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

import config

log = logging.getLogger(__name__)

CONFLICTS_NAME = "direction_conflicts.csv"      # under config.OUTPUT_DIR, resolved at call time
CONFLICT_COLS = [
    "generated_at", "match_date", "league", "home_team", "away_team",
    "suppressed_market", "suppressed_side", "suppressed_direction", "suppressed_tier",
    "suppressed_odds", "suppressed_edge_pct",
    "stood_market", "stood_side", "stood_direction", "stood_tier", "stood_generated_at",
    "stood_source", "reason",
]
#: first-sight key: one row per suppressed (fixture, market, side), ever
CONFLICT_KEY = ["match_date", "home_team", "away_team", "suppressed_market", "suppressed_side"]
OU25 = "ou25"


def goal_direction(market, side=None) -> str | None:
    """OVER / UNDER for a goal-total position, None for anything else (BTTS never conflicts)."""
    m = str(market or "").strip().lower().replace(" ", "").replace("/", "").replace(".", "")
    s = str(side or "").strip().upper()
    if m.startswith("btts"):
        return None
    if m.startswith("over"):
        return "OVER"
    if m.startswith("under"):
        return "UNDER"
    if m in ("", OU25, "ou"):
        return s if s in ("OVER", "UNDER") else None
    return None


def _ou_side(row) -> str:
    """The side of an O/U 2.5 tip, read the way src/ledger.append_tips reads it."""
    s = row.get("bet") if hasattr(row, "get") else None
    if s is None or (isinstance(s, float) and s != s) or not str(s).strip() or str(s) == "nan":
        s = row.get("best_side", "")
    return str(s).strip().upper()


def _fx(date, home, away) -> tuple:
    return (str(date)[:10], str(home), str(away))


def committed(source: str = "live") -> dict:
    """Fixture -> the FIRST live tip with a goal direction, from both ledgers:
    {(match_date, home, away): {direction, market, side, tier, generated_at, ledger}}.
    Ties on generated_at (one run wrote both) resolve to O/U 2.5, the main market."""
    parts = []
    try:
        from src import ledger as _L         # the exact files the ledger writers write
        p = _L.LEDGER_FILE
        if p.exists():
            b = pd.read_csv(p, dtype=str)
            src = b["source"].fillna("live") if "source" in b.columns else pd.Series("live", index=b.index)
            b = b[src.astype(str) == source]
            parts.append(pd.DataFrame({
                "match_date": b["match_date"].astype(str).str[:10], "home_team": b["home_team"].astype(str),
                "away_team": b["away_team"].astype(str), "market": OU25,
                "side": b["side"].astype(str).str.strip().str.upper(), "tier": b.get("signal_tier"),
                "generated_at": b["generated_at"].astype(str), "ledger": "bets_ledger", "_pri": 0}))
        p = _L.SIDE_LEDGER_FILE
        if p.exists():
            s = pd.read_csv(p, dtype=str)
            src = s["source"].fillna("live") if "source" in s.columns else pd.Series("live", index=s.index)
            s = s[src.astype(str) == source]
            side = s["side"] if "side" in s.columns else pd.Series("", index=s.index)
            parts.append(pd.DataFrame({
                "match_date": s["match_date"].astype(str).str[:10], "home_team": s["home_team"].astype(str),
                "away_team": s["away_team"].astype(str), "market": s["market"].astype(str).str.lower(),
                "side": side.fillna("").astype(str), "tier": s.get("signal_tier"),
                "generated_at": s["generated_at"].astype(str), "ledger": "side_bets_ledger", "_pri": 1}))
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"direction guard: ledgers unreadable ({type(e).__name__}: {e}) — within-run rule only")
        return {}
    if not parts:
        return {}
    d = pd.concat(parts, ignore_index=True)
    if d.empty:
        return {}
    d["direction"] = [goal_direction(m, s) for m, s in zip(d["market"], d["side"])]
    d = d[d["direction"].notna()]
    d = d.sort_values(["generated_at", "_pri"], kind="mergesort")
    d = d.drop_duplicates(["match_date", "home_team", "away_team"], keep="first")
    return {(r.match_date, r.home_team, r.away_team): {
                "direction": r.direction, "market": r.market, "side": r.side, "tier": r.tier,
                "generated_at": r.generated_at, "ledger": r.ledger}
            for r in d.itertuples(index=False)}


def apply(bets: pd.DataFrame | None, side_bets: pd.DataFrame | None,
          source: str = "live", first: dict | None = None):
    """(bets, side_bets) -> (bets_kept, side_bets_kept, conflicts). Never raises: on any internal
    error the inputs are returned unchanged (fail open to the pre-rule behaviour) and logged loudly.

    `first` (tests) overrides the ledger lookup with a committed() style mapping."""
    empty = pd.DataFrame(columns=CONFLICT_COLS)
    try:
        if first is None:
            first = committed(source)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        has_b = bets is not None and not bets.empty
        has_s = side_bets is not None and not side_bets.empty
        if not has_b and not has_s:
            return bets, side_bets, empty

        conflicts = []

        def _note(r, mkt, side, dirn, tier, odds, edge, stood, reason):
            conflicts.append({
                "generated_at": now, "match_date": str(r["date"])[:10], "league": str(r.get("league", "")),
                "home_team": str(r["home_team"]), "away_team": str(r["away_team"]),
                "suppressed_market": mkt, "suppressed_side": side, "suppressed_direction": dirn,
                "suppressed_tier": tier, "suppressed_odds": odds, "suppressed_edge_pct": edge,
                "stood_market": stood["market"], "stood_side": stood["side"],
                "stood_direction": stood["direction"], "stood_tier": stood["tier"],
                "stood_generated_at": stood["generated_at"], "stood_source": stood["ledger"],
                "reason": reason})

        # 1. fixture -> the direction that stands: the ledger's first tip, else this run's O/U 2.5 tip
        stands = dict(first)
        keep_b = []
        if has_b:
            for _, r in bets.iterrows():
                fx = _fx(r["date"], r["home_team"], r["away_team"])
                side = _ou_side(r)
                dirn = goal_direction(OU25, side)
                fixed = stands.get(fx)
                if fixed is None or dirn is None or fixed["direction"] == dirn:
                    keep_b.append(True)
                    if fixed is None and dirn is not None:
                        stands[fx] = {"direction": dirn, "market": OU25, "side": side,
                                      "tier": str(r.get("signal_tier", "")), "generated_at": now,
                                      "ledger": "this_run"}
                    continue
                if fixed["market"] == OU25:
                    # O/U 2.5 vs an earlier O/U 2.5 tip: the existing flip guard's case, untouched.
                    keep_b.append(True)
                    continue
                odds = r.get("odds_under25") if side == "UNDER" else r.get("odds_over25")
                edge = r.get("best_edge")
                _note(r, OU25, side, dirn, str(r.get("signal_tier", "")),
                      odds, round(float(edge) * 100, 2) if pd.notna(edge) else None, fixed,
                      "OPPOSITE_TO_EARLIER_TIP")
                keep_b.append(False)

        keep_s = []
        if has_s:
            for _, r in side_bets.iterrows():
                fx = _fx(r["date"], r["home_team"], r["away_team"])
                mkt = str(r.get("market", "")).lower()
                side = str(r.get("side", "") or "")
                dirn = goal_direction(mkt, side)
                fixed = stands.get(fx)
                if fixed is None or dirn is None or fixed["direction"] == dirn:
                    keep_s.append(True)
                    if fixed is None and dirn is not None:
                        stands[fx] = {"direction": dirn, "market": mkt, "side": side,
                                      "tier": str(r.get("signal_tier", "")), "generated_at": now,
                                      "ledger": "this_run"}
                    continue
                reason = ("OPPOSITE_TO_OU25_SAME_RUN" if fixed["ledger"] == "this_run" and fixed["market"] == OU25
                          else "OPPOSITE_TO_SAME_RUN_TIP" if fixed["ledger"] == "this_run"
                          else "OPPOSITE_TO_EARLIER_TIP")
                edge = r.get("edge")
                _note(r, mkt, side, dirn, str(r.get("signal_tier", "")), r.get("odds", r.get("market_odds")),
                      round(float(edge) * 100, 2) if pd.notna(edge) else None, fixed, reason)
                keep_s.append(False)

        out_b = bets[keep_b] if has_b else bets
        out_s = side_bets[keep_s] if has_s else side_bets
        c = pd.DataFrame(conflicts, columns=CONFLICT_COLS)
        if len(c):
            log.warning(f"Direction guard: {len(c)} opposite-direction tip(s) suppressed "
                        f"(not sent, not ledgered, not staked) → {CONFLICTS_NAME}")
            for x in c.itertuples(index=False):
                log.warning(f"  {x.match_date} {x.home_team} v {x.away_team}: {x.suppressed_market} "
                            f"{x.suppressed_direction} suppressed, {x.stood_market} {x.stood_direction} "
                            f"stands ({x.stood_source} {x.stood_generated_at})")
        return out_b, out_s, c
    except Exception as e:                                           # noqa: BLE001
        log.error(f"DIRECTION GUARD FAILED ({type(e).__name__}: {e}) — tips passed through unguarded")
        return bets, side_bets, empty


def record(conflicts: pd.DataFrame | None) -> int:
    """Append FIRST-SIGHT conflict rows to output/direction_conflicts.csv. Returns rows added. Never raises."""
    try:
        if conflicts is None or conflicts.empty:
            return 0
        f = config.OUTPUT_DIR / CONFLICTS_NAME
        new = conflicts.drop_duplicates(CONFLICT_KEY, keep="first")
        if f.exists():
            prev = pd.read_csv(f, dtype=str)
            seen = set(map(tuple, prev.reindex(columns=CONFLICT_KEY).fillna("").astype(str).values))
            new = new[[tuple(x) not in seen
                       for x in new[CONFLICT_KEY].fillna("").astype(str).values]]
        if new.empty:
            return 0
        new = new.reindex(columns=CONFLICT_COLS)
        new.to_csv(f, mode="a", header=not f.exists(), index=False)
        return len(new)
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"direction_conflicts.csv not written ({type(e).__name__}: {e})")
        return 0
