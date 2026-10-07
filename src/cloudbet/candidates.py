"""Build one candidate frame from team tips and player props, for the Cloudbet selector.

    from src.cloudbet.candidates import build_candidates
    cands = build_candidates()

TEAM MARKETS and PLAYER PROPS have different bars, by the owner's instruction:

    team (ou25, btts, over15, over35)   edge strictly above 5%
    player props                        any POSITIVE EV

They are merged into one frame and then compete for the same 20-bet daily cap, sorted by edge
descending. So a prop only displaces a team bet when its edge is larger.

TWO THINGS RECORDED ON EVERY PROP ROW, because they change how the number should be read:

`settlement_status` — from registry/settlement_alignment.json. All six prop markets are
UNVERIFIED: we have never checked that our outcome label matches how the bookmaker settles.
`goals` is the lowest-risk of them (anytime-scorer is the least ambiguous prop; own goals are
the main edge case and are rare); `sot`/`sot2`/`sot3` are the highest, because whether a blocked
shot counts as on target is the largest source of provider disagreement in player props. An EV
computed against the wrong event is not EV, so the status travels with the bet into the ledger
and can be audited after the fact.

`implied_prob` and `prob_ratio` — the market's probability and how many times our model's
probability exceeds it. Measured on the current board, the +EV props sit at ratios near 4x on
13.0 and 19.5 shots. Prior OOS research on this estate found that "model disagrees with the
book" is a longshot machine rather than an edge (props -41% to -57%, AUC about 0.5), and the
signature of that is exactly a large ratio at long odds. The ratio is not used to filter here —
it is recorded so the pattern is visible in the ledger rather than discovered later.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

TEAM_MARKETS = ("ou25", "btts", "over15", "over35")
#: Prop markets the owner has enabled. Keyed to player_tips.csv's `market` column.
PROP_MARKETS = ("goals", "assists", "sot", "sot2", "sot3", "cards")

_REG = Path(__file__).resolve().parents[2] / "registry" / "settlement_alignment.json"

#: The side each market implies when the ledger does not record one. These paths only ever
#: produce one side by construction, which is why the column was left blank.
_IMPLIED_SIDE = {"btts": "YES", "over15": "OVER", "over35": "OVER", "ou25": "OVER"}


def _settlement() -> dict:
    """market -> alignment_status. Unknown markets are UNVERIFIED, not ALIGNED."""
    try:
        j = json.loads(_REG.read_text(encoding="utf-8"))
        return {m["market"]: m.get("alignment_status", "UNVERIFIED") for m in j["markets"]}
    except Exception as e:                                            # noqa: BLE001
        log.warning(f"settlement registry unreadable ({e}) — treating every market as UNVERIFIED")
        return {}


def team_candidates(days_ahead: int = 0) -> pd.DataFrame:
    """Team-market tips from the LEDGER, which is what the Telegram digest reports.

    READ THE LEDGER, NOT bets.csv. This was wrong in the first version and the discrepancy was
    caught by the owner comparing a dry run against his own digest. The two sources answer
    different questions:

        bets.csv          the CURRENT board — every fixture in the 7-day look-ahead, rewritten
                          every predict run. On 2026-10-05 it held 76 tips dated Oct 7-11 and
                          ZERO for today.
        bets_ledger.csv   what was actually tipped and persisted, deduped per fixture. The
                          digest sends `source == "live"` filtered to match_date == today, and
                          on the same morning that was 3 tips, all OVER.

    Both were correct and they described different days. A bot reading bets.csv would place
    bets on fixtures up to six days out that the owner has never been shown in any digest —
    so "what I see in Telegram" and "what the bot bets" could diverge silently, which is the
    one property an automated placer must not have.

    `days_ahead=0` means today only, matching the digest exactly. Raise it deliberately.
    """
    f = config.OUTPUT_DIR / "bets_ledger.csv"
    s = config.OUTPUT_DIR / "side_bets_ledger.csv"
    rows = []
    today = pd.Timestamp.now(tz="UTC").normalize()
    horizon = today + pd.Timedelta(days=max(0, int(days_ahead)))

    def _window(d: pd.DataFrame) -> pd.DataFrame:
        """Live rows, not yet settled, kicking off between today and the horizon."""
        if "source" in d.columns:
            d = d[d["source"].astype(str) == "live"]
        # An already-settled row is history, not a candidate.
        d = d[~d["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
        md = pd.to_datetime(d["match_date"], errors="coerce", utc=True).dt.normalize()
        return d[(md >= today) & (md <= horizon)]

    for path, market in ((f, "ou25"), (s, None)):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if d.empty:
            continue
        d = _window(d)
        if d.empty:
            continue
        # edge_pct is in PERCENT in the ledger; the team bar and the prop bar are both in
        # probability points, so it is converted here rather than at the comparison.
        edge = pd.to_numeric(d.get("edge_pct"), errors="coerce") / 100.0
        rows.append(pd.DataFrame({
            "fixture_id": (d["match_date"].astype(str).str[:10] + "|"
                           + d["home_team"].astype(str) + "|" + d["away_team"].astype(str)
                           + "|" + (market or d["market"]).astype(str)
                           if market is None else
                           d["match_date"].astype(str).str[:10] + "|"
                           + d["home_team"].astype(str) + "|" + d["away_team"].astype(str)),
            "date": d["match_date"].astype(str).str[:10], "league": d["league"],
            "home_team": d["home_team"], "away_team": d["away_team"],
            "kickoff_utc": d.get("kickoff_utc"),
            "market": market if market else d["market"],
            # SIDE IS NaN ON EVERY SIDE-MARKET LEDGER ROW (435 of 435). The ledger never
            # stores it because those paths only ever produce one side, so for reporting it is
            # implied. A bot cannot work from an implication: a NaN side builds a broken
            # marketUrl and would either fail or, worse, place the wrong side. Derived here
            # from the market, which is the same rule the tip generator applies.
            "side": (pd.Series(d.get("side")).astype(str).str.upper()
                     .where(lambda x: x.isin(["OVER", "UNDER", "YES", "NO"]),
                            (market if market else d["market"]).map(_IMPLIED_SIDE)
                            if market is None else _IMPLIED_SIDE.get(market, "OVER"))),
            "edge": edge,
            "signal_tier": d.get("signal_tier"),
            "model_type": d.get("model_type"),
            "our_odds": pd.to_numeric(d.get("odds"), errors="coerce"),
            "selection_label": d["home_team"].astype(str) + " v " + d["away_team"].astype(str),
            # The MATCH, independent of market — what the correlation cap groups on.
            "match_key": (d["match_date"].astype(str).str[:10] + "|"
                          + d["home_team"].astype(str) + "|" + d["away_team"].astype(str)),
        }))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["bet_kind"] = "team"
    out["settlement_status"] = out["market"].map(
        {"ou25": "ALIGNED", "btts": "ALIGNED"}).fillna("UNVERIFIED")
    return out[out["market"].isin(TEAM_MARKETS)]


def prop_candidates(min_ev: float = 0.0) -> pd.DataFrame:
    """Player props with a REAL price and positive EV.

    An unpriced prop is dropped rather than treated as a rejection. Measured on the current
    board, 1,263 of 1,296 prop rows carry no market price at all — `enrich_with_odds` skips a
    row it cannot price, so its default tier survives untouched. Counting those as "the model
    said no" would badly misread what the model actually did: it was never asked.
    """
    f = config.OUTPUT_DIR / "player_tips.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    if d.empty:
        return pd.DataFrame()

    d = d[d["market"].astype(str).isin(PROP_MARKETS)]
    odds = pd.to_numeric(d.get("market_odds"), errors="coerce")
    ev = pd.to_numeric(d.get("ev"), errors="coerce")
    p = pd.to_numeric(d.get("model_prob"), errors="coerce")
    d = d[odds.notna() & (odds > 1.0) & ev.notna() & (ev > min_ev)]
    if d.empty:
        return d

    odds = pd.to_numeric(d["market_odds"], errors="coerce")
    p = pd.to_numeric(d["model_prob"], errors="coerce")
    implied = 1.0 / odds
    align = _settlement()

    out = pd.DataFrame({
        # One prop per player per market — NOT per fixture, or the fixture-level dedup would
        # silently drop every prop after the first on a match.
        "fixture_id": ("prop|" + d["player_id"].astype(str) + "|" + d["market"].astype(str)
                       + "|" + d["date"].astype(str)),
        "date": d["date"], "league": d["league"],
        "home_team": d.get("match", ""), "away_team": "",
        "kickoff_utc": d.get("kickoff_utc"),
        "market": d["market"],
        "side": "YES",
        # Edge on the same scale as the team markets (probability points), so the two compete
        # fairly for the daily cap. `ev` is a different scale and would dominate the sort.
        "edge": (p - implied),
        "signal_tier": d.get("tier"),
        "model_type": "player_prop",
        "our_odds": odds,
        "selection_label": d["player_name"].astype(str) + " — " + d["market"].astype(str),
        "player_name": d.get("player_name"),
        "ev": ev,
        "implied_prob": implied.round(4),
        "prob_ratio": (p / implied).round(2),
        # A prop's "match" is the fixture it is in, so several props on one game are capped the
        # same way several markets on one game are.
        "match_key": "prop|" + d.get("fixture_id", d["date"]).astype(str),
    })
    out["bet_kind"] = "prop"
    out["settlement_status"] = out["market"].map(align).fillna("UNVERIFIED")

    longshot = out[out["our_odds"] >= 8.0]
    if len(longshot):
        log.warning(
            f"{len(longshot)} of {len(out)} +EV prop(s) are longshots at odds >= 8.0 with a "
            f"median model/market probability ratio of {longshot['prob_ratio'].median():.1f}x. "
            f"Prior OOS work on this estate found 'model disagrees with the book' to be a "
            f"longshot machine rather than an edge. Recorded, not filtered.")
    return out


def build_candidates(team_min_edge: float = 0.05, prop_min_ev: float = 0.0,
                     days_ahead: int = 0) -> pd.DataFrame:
    """Everything bettable today, on one scale, ready for the selector.

    The edge filters live in the selector, not here — this builds the universe and the selector
    decides. Keeping the two apart is what lets the selector be tested without any files.
    """
    t, p = team_candidates(days_ahead=days_ahead), prop_candidates(min_ev=prop_min_ev)
    frames = [f for f in (t, p) if len(f)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # A PAPER LEAGUE IS NEVER BET. It is collected and its tips are ledgered precisely so it
    # can be judged later — and the bot reads the ledger, so without this line a paper league
    # would reach real money by the one path that bypasses the Telegram filters.
    out = config.drop_paper_leagues(out)
    if out.empty:
        return out
    out["edge"] = pd.to_numeric(out["edge"], errors="coerce")
    out = out[out["edge"].notna()]
    log.info(f"candidates: {len(t)} team, {len(p)} prop ({len(out)} total)")
    return out.sort_values("edge", ascending=False, kind="mergesort").reset_index(drop=True)
