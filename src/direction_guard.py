"""One goal DIRECTION per fixture: never tip Over and Under goal-total lines on the same match.

OWNER RULE (2026-10-25, verbatim): "what i meant to deny its over 2.5 and under 2.5 for the same
match, or over 1.5 and under 2.5, or under 3.5 and over 2.5". Several tips on one match in the SAME
direction are fine ("thats ok" about Over 1.5 + BTTS on San Lorenzo v Newell's).

The case that triggered it: 2026-10-25 Viking FK v Tromso went out as O/U 2.5 UNDER SNIPER (tipped
2026-10-18) and Over 3.5 VALUABLE (tipped 2026-10-20). If the match has 4+ goals the Under loses and
the Over 3.5 wins; 3 goals and BOTH lose. Two tips that hedge each other are a vig donation, and the
fixture becomes unreadable in the ledger.

OWNER RULE (2026-11-04, verbatim): "Several bets on one match are fine ... if its not the opposites ...
u cant tipped over 2.5 and under 3.5 / u cant tipped under 2.5 and over 1.5 / u cant tipped under1.5
and btts Y". So BTTS now HAS a direction: Yes needs goals from both sides (OVER), No (UNDER).

DIRECTION
    any Over goal-total line (O/U 2.5 OVER, over15, over35, any future overXX)  -> OVER
    any Under goal-total line (O/U 2.5 UNDER, any future underXX)               -> UNDER
    BTTS YES -> OVER, BTTS NO -> UNDER (since 2026-11-04); any other side or market -> None

REAL MONEY WINS (owner 2026-11-04). A tip on a PAPER track never blocks a tip on a real-money track
(config/bankroll_policy.json: state in REAL_STATES and base_fraction > 0). Measured when decided: 99
live fixtures carried both an O/U 2.5 UNDER tip (new-format UNDER is PAPER since 2026-11-02) and a
BTTS YES tip (the only ACTIVE track); the Under came first on 73. So:
    the opposite tip is real and the standing one paper, same run -> the paper tip is suppressed and
        the real one stands (reason PAPER_YIELDS_TO_REAL);
    the opposite tip is real and the standing one is an EARLIER paper tip (already sent and ledgered,
        which cannot be unsent) -> the real tip goes out; the clash is logged (REAL_AFTER_EARLIER_PAPER)
        and from then on the real tip's direction stands;
    otherwise (both real, both paper, or real standing) -> the FIRST tip stands, as below.
An unreadable policy makes every tip count as paper: the rule falls back to first-tip-stands.

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
    """OVER / UNDER for a goal-total position or BTTS (YES -> OVER, NO -> UNDER, owner 2026-11-04);
    None for anything else."""
    m = str(market or "").strip().lower().replace(" ", "").replace("/", "").replace(".", "")
    s = str(side or "").strip().upper()
    if m.startswith("btts"):
        return {"YES": "OVER", "NO": "UNDER"}.get(s)
    if m.startswith("over"):
        return "OVER"
    if m.startswith("under"):
        return "UNDER"
    if m in ("", OU25, "ou"):
        return s if s in ("OVER", "UNDER") else None
    return None


#: guard market spelling -> bankroll-policy market (src/bankroll_policy.track_key)
_POLICY_MARKET = {OU25: "OU25", "btts": "BTTS", "over15": "OVER15", "over35": "OVER35"}


def real_track_checker():
    """f(model_type, market, side) -> True when the tip's bankroll-policy track is real money
    (state in REAL_STATES and base_fraction > 0). Policy unreadable -> always False (every tip
    counts as paper, so the guard falls back to first-tip-stands; logged)."""
    try:
        from src.bankroll_policy import REAL_STATES, load_policy, track_key
        tracks = load_policy().get("tracks") or {}
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"direction guard: bankroll policy unreadable ({type(e).__name__}: {e}) — "
                    "every tip treated as paper (first tip stands)")
        return lambda *a: False

    def _real(model_type, market, side) -> bool:
        mk = _POLICY_MARKET.get(str(market or "").strip().lower())
        mt = str(model_type or "").strip()
        if mk is None or not mt or mt == "nan":
            return False
        t = tracks.get(track_key(mt, mk, str(side or "").strip().upper())) or {}
        return t.get("state") in REAL_STATES and float(t.get("base_fraction") or 0.0) > 0.0
    return _real


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
                "model_type": b.get("model_type"),
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
                "model_type": s.get("model_type"),
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
    is_real = real_track_checker()
    return {(r.match_date, r.home_team, r.away_team): {
                "direction": r.direction, "market": r.market, "side": r.side, "tier": r.tier,
                "generated_at": r.generated_at, "ledger": r.ledger,
                "real": bool(is_real(r.model_type, r.market, r.side))}
            for r in d.itertuples(index=False)}


def apply(bets: pd.DataFrame | None, side_bets: pd.DataFrame | None,
          source: str = "live", first: dict | None = None, is_real=None):
    """(bets, side_bets) -> (bets_kept, side_bets_kept, conflicts). Never raises: on any internal
    error the inputs are returned unchanged (fail open to the pre-rule behaviour) and logged loudly.

    `first` (tests) overrides the ledger lookup with a committed() style mapping (an entry without
    'real' counts as paper). `is_real` (tests) overrides the policy lookup: f(model_type, market, side)."""
    empty = pd.DataFrame(columns=CONFLICT_COLS)
    try:
        if first is None:
            first = committed(source)
        if is_real is None:
            is_real = real_track_checker()
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

        # This run's tips as candidates. Real-money tips are considered FIRST (owner 2026-11-04: a
        # paper tip never blocks a real one); inside each group the old order holds - O/U 2.5 tips,
        # then side markets in _generate_side_bets' edge-descending order.
        cands = []
        if has_b:
            for i, (_, r) in enumerate(bets.iterrows()):
                side = _ou_side(r)
                edge = r.get("best_edge")
                cands.append(("b", i, r, OU25, side,
                              r.get("odds_under25") if side == "UNDER" else r.get("odds_over25"),
                              round(float(edge) * 100, 2) if pd.notna(edge) else None))
        if has_s:
            for i, (_, r) in enumerate(side_bets.iterrows()):
                edge = r.get("edge")
                cands.append(("s", i, r, str(r.get("market", "")).lower(), str(r.get("side", "") or ""),
                              r.get("odds", r.get("market_odds")),
                              round(float(edge) * 100, 2) if pd.notna(edge) else None))
        real = [bool(is_real(c[2].get("model_type"), c[3], c[4])) for c in cands]
        order = sorted(range(len(cands)), key=lambda k: (not real[k], k))

        # fixture -> the direction that stands: the ledger's first tip, else this run's first claim
        stands = dict(first)
        keep = {"b": [True] * (len(bets) if has_b else 0), "s": [True] * (len(side_bets) if has_s else 0)}
        for k in order:
            kind, i, r, mkt, side, odds, edge = cands[k]
            fx = _fx(r["date"], r["home_team"], r["away_team"])
            dirn = goal_direction(mkt, side)
            tier = str(r.get("signal_tier", ""))
            mine = {"direction": dirn, "market": mkt, "side": side, "tier": tier, "generated_at": now,
                    "ledger": "this_run", "real": real[k]}
            fixed = stands.get(fx)
            if fixed is None or dirn is None or fixed["direction"] == dirn:
                if fixed is None and dirn is not None:
                    stands[fx] = mine
                continue
            if kind == "b" and fixed["market"] == OU25:
                # O/U 2.5 vs an earlier O/U 2.5 tip: the existing flip guard's case, untouched.
                continue
            if real[k] and not fixed.get("real", False):
                # only an EARLIER paper tip can still stand against a real one here (real tips of
                # this run are considered first): it was sent already, so the real tip goes out and
                # its direction stands from now on; the clash is logged.
                conflicts.append({
                    "generated_at": now, "match_date": str(r["date"])[:10], "league": str(r.get("league", "")),
                    "home_team": str(r["home_team"]), "away_team": str(r["away_team"]),
                    "suppressed_market": fixed["market"], "suppressed_side": fixed["side"],
                    "suppressed_direction": fixed["direction"], "suppressed_tier": fixed["tier"],
                    "suppressed_odds": None, "suppressed_edge_pct": None,
                    "stood_market": mkt, "stood_side": side, "stood_direction": dirn, "stood_tier": tier,
                    "stood_generated_at": now, "stood_source": "this_run",
                    "reason": "REAL_AFTER_EARLIER_PAPER"})
                stands[fx] = mine
                continue
            reason = ("PAPER_YIELDS_TO_REAL" if fixed.get("real", False) and not real[k]
                      and fixed["ledger"] == "this_run"
                      else "OPPOSITE_TO_OU25_SAME_RUN" if fixed["ledger"] == "this_run" and fixed["market"] == OU25
                      else "OPPOSITE_TO_SAME_RUN_TIP" if fixed["ledger"] == "this_run"
                      else "OPPOSITE_TO_EARLIER_TIP")
            _note(r, mkt, side, dirn, tier, odds, edge, fixed, reason)
            keep[kind][i] = False

        out_b = bets[keep["b"]] if has_b else bets
        out_s = side_bets[keep["s"]] if has_s else side_bets
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
