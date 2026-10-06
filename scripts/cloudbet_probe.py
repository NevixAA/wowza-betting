"""Read-only probe: what does Cloudbet actually return, and does our map match it?

    CLOUDBET_API_KEY=... python scripts/cloudbet_probe.py
    CLOUDBET_API_KEY=... python scripts/cloudbet_probe.py --competition soccer-argentina-primera

PLACES NO BETS. It never imports the placement path, and it runs regardless of CLOUDBET_MODE.

WHAT IT IS FOR. `src/cloudbet/feed.MARKET_MAP` is a HYPOTHESIS — Cloudbet documents the shape
(`soccer.market/outcome?params`) but the exact keys for total goals and both-teams-to-score are
not pinned in anything reachable from here. This prints the keys a real event carries, so the
map is corrected from evidence in one edit rather than guessed.

It also answers the question that decides whether the bot can work at all: for OUR fixtures, in
OUR leagues, does Cloudbet price the markets we bet, and at what prices relative to the ones our
edge was computed against? An edge of 9% against Bet365 is not an edge if Cloudbet is 4% worse.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.cloudbet.client import CloudbetClient  # noqa: E402
from src.cloudbet.feed import (MARKET_MAP, discover_markets, find_event,  # noqa: E402
                               price_for)
from src.cloudbet.candidates import build_candidates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", help="Cloudbet competition key; omit to list them")
    ap.add_argument("--days", type=int, default=3)
    a = ap.parse_args()

    if not os.getenv("CLOUDBET_API_KEY", "").strip():
        print("CLOUDBET_API_KEY is not set in the environment.")
        print("Set it in a gitignored .env or as a repo secret — never on the command line,")
        print("where it lands in shell history.")
        return 1

    # DRY so that even an accidental import of the placement path cannot send anything.
    c = CloudbetClient(mode="DRY")

    if not a.competition:
        payload = c.competitions("soccer")
        comps = []
        for cat in (payload.get("categories") or []):
            for comp in (cat.get("competitions") or []):
                comps.append((comp.get("key"), comp.get("name"),
                              str(cat.get("name", ""))))
        if not comps:
            print("no competitions returned — check the key's permissions")
            return 1
        print(f"{len(comps)} soccer competitions. Ones that look like our leagues:\n")
        want = [w.lower() for w in config.ENABLED_LEAGUES]
        for key, name, cat in sorted(comps, key=lambda x: str(x[1])):
            n = f"{cat} {name}".lower()
            if any(tok in n for tok in
                   ("argentina", "brazil", "serie b", "segunda", "championship",
                    "league one", "league two", "ligue 2", "2. bundesliga", "j1",
                    "liga mx", "allsvenskan", "eliteserien", "superliga", "veikkaus")):
                print(f"  {str(key):<44} {cat} — {name}")
        print("\nRe-run with --competition <key> to probe one.")
        return 0

    fx = c.fixtures(a.competition)
    events = (fx.get("events") or
              [e for comp in (fx.get("competitions") or []) for e in (comp.get("events") or [])])
    print(f"{len(events)} event(s) in {a.competition}\n")
    if not events:
        return 0

    # 1. What market keys exist, really?
    seen: dict[str, set] = {}
    for e in events[:10]:
        full = c.event_odds(str(e.get("id"))) or e
        for k, outs in discover_markets(full).items():
            seen.setdefault(k, set()).update(outs)
    print("MARKET KEYS ACTUALLY RETURNED (this is what MARKET_MAP must match):")
    for k in sorted(seen):
        print(f"  {k:<44} outcomes: {sorted(seen[k])}")
    mapped = {v[0] for v in MARKET_MAP.values()}
    missing = mapped - set(seen)
    print(f"\n  our map expects: {sorted(mapped)}")
    if missing:
        print(f"  *** NOT PRESENT: {sorted(missing)} — MARKET_MAP needs correcting ***")
    else:
        print("  all mapped keys are present")

    # 2. Can we price OUR board, and how do their prices compare to ours?
    cands = build_candidates(days_ahead=a.days)
    if cands.empty:
        print("\nno candidates on our board to price")
        return 0
    print(f"\nPRICING OUR BOARD ({len(cands)} candidates):")
    hit = miss = 0
    for _, r in cands.iterrows():
        ev, why = find_event(events, str(r.get("home_team")), str(r.get("away_team")))
        if ev is None:
            miss += 1
            print(f"  NO EVENT   {r['selection_label'][:38]:<40} {why[:70]}")
            continue
        full = c.event_odds(str(ev.get("id"))) or ev
        q, why = price_for(full, str(r["market"]), str(r["side"]))
        if q is None:
            miss += 1
            print(f"  NO PRICE   {r['selection_label'][:38]:<40} {why[:70]}")
            continue
        hit += 1
        ours = float(r.get("our_odds") or 0)
        delta = (1 / ours - 1 / q.odds) if ours > 1 else float("nan")
        print(f"  PRICED     {r['selection_label'][:38]:<40} ours {ours:>5.2f} -> "
              f"cloudbet {q.odds:>5.2f}   edge shift {delta:+.3f}")
    print(f"\npriced {hit} of {hit + miss}. An edge computed against our book is not an edge "
          f"at theirs — the shift column is how much of it survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
