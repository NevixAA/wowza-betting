"""Opening -> moving -> closing price, for EVERY market and every model. Read-only.

    from src.market_movement import CurveIndex
    idx = CurveIndex()                       # loads every captured odds source once
    c = idx.curve("League One", "Plymouth Argyle", "Burton Albion", "2026-09-26", "ht_under15")
    # -> {"open": 1.44, "close": 1.40, "n": 7, "drift_pct": +2.78, ...}

WHY THIS EXISTS. About 304,000 rows of price curve are banked across six capture files, and
almost nothing reads them. Only ONE consumer existed: `drift.py`, which tracks the OU25
over/under pair alone and feeds a three-rule tier nudge on the main Over/Under track. Measured
on settled bets that rule is pointing the wrong way -- the `Confirmed` bucket it UPGRADES on
runs -10.8% while the `Neutral` bucket it ignores runs +8.0% -- and `grep drift src/backtest.py`
returns 0, so it has never been walk-forward tested.

Meanwhile side markets, half-time and player props get no movement signal at all, and the two
model tracks that the estate otherwise keeps strictly isolated (invariant 1) share one global
rule with one set of constants.

THIS MODULE MEASURES. IT DOES NOT DECIDE. Nothing here changes a tier, a stake, a threshold or
a tip. It attaches opening, closing, drift and CLV to rows that already exist, so the question
"does market movement carry information, per league x market x model" can be answered from
evidence rather than argued. Acting on the answer is a separate decision that needs its own
walk-forward -- the per-league threshold work is the warning: 10 of 10 genuinely-tested cells
came back UNCHANGED and `holds_oos` was False on all 23.

A FILLED PRICE IS NOT A PRICE. A fixture with no captured curve returns None, never a default.
The side-market backtest records what the alternative costs: over15 was once certified at +13.5%
ROI on 12,186 of 12,187 rows priced at a CONSTANT 1.40, which made "edge" a bare probability
threshold with no market in it.

CLUB NAMES DIFFER BETWEEN SOURCES (invariant 11). The captures carry the OddsAPI event name and
the ledgers carry the predict-side name: `Accrington ST` against `Accrington Stanley`,
`Cheltenham` against `Cheltenham Town`, `Plymouth` against `Plymouth Argyle`. Matching on a bare
alphanumeric key finds almost nothing -- on the half-time ledger it found 2 of 47 where
league-scoped resolution finds 38. Every lookup here is league-scoped through
`src.team_names.resolve`, which refuses an ambiguous match rather than guessing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

#: Every captured curve, and which track it belongs to. All six share one schema:
#: snapshot_date, snapshot_ts, match_date, league, match, market, odds.
SOURCES = [
    ("standard_sidemarket_odds_history.csv", "standard"),
    ("standard_odds_history.csv", "standard"),
    ("newformat_odds_history.csv", "new_format"),
    ("newformat_odds_dense.csv", "new_format"),
    ("player_prop_odds_history.csv", "props"),
]

#: Movement smaller than this is noise, not a signal. Matches drift.py's existing constants so
#: the new per-market signal cannot silently disagree with the one already in production.
CONFIRM_PCT = 3.0
CONFLICT_PCT = 3.0


def _split_match(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    t = s.astype(str)
    return (t.apply(lambda m: m.split(" vs ")[0].strip() if " vs " in m else ""),
            t.apply(lambda m: m.split(" vs ")[1].strip() if " vs " in m else ""))


class CurveIndex:
    """Every captured price curve, indexed by (match_date, league) for O(1) lookup.

    Loaded once and reused: the props file alone is 200k+ rows, and a per-row scan of it would
    turn a ledger enrichment into an hour of work.
    """

    def __init__(self, sources=None, out_dir: Path | None = None):
        out = out_dir or config.OUTPUT_DIR
        frames = []
        for fname, track in (sources or SOURCES):
            p = out / fname
            if not p.exists():
                log.info(f"[curve] {fname} absent — skipped")
                continue
            try:
                d = pd.read_csv(p, usecols=lambda c: c in {
                    "snapshot_ts", "snapshot_date", "match_date", "league", "match",
                    "market", "odds", "player"})
            except Exception as e:                                    # noqa: BLE001
                log.warning(f"[curve] {fname} unreadable ({e})")
                continue
            if d.empty or "market" not in d.columns:
                continue
            d["_track"] = track
            d["_src"] = fname
            frames.append(d)
        if not frames:
            self.by_day = {}
            self.n_rows = 0
            return

        c = pd.concat(frames, ignore_index=True)
        c["odds"] = pd.to_numeric(c["odds"], errors="coerce")
        c = c[c["odds"] > 1.0].dropna(subset=["market"])
        # Sort ONCE, stably, so "first row = opening" and "last row = closing" hold everywhere.
        # mergesort because the default sort is not stable and same-timestamp rows would
        # otherwise permute between runs, making the opening price non-deterministic.
        ts = c["snapshot_ts"] if "snapshot_ts" in c.columns else c.get("snapshot_date")
        c["_ts"] = pd.to_datetime(ts, errors="coerce", utc=True)
        c = c.sort_values("_ts", kind="mergesort")
        c["_h"], c["_a"] = _split_match(c["match"])
        c["_dk"] = c["match_date"].astype(str).str[:10]
        self.n_rows = len(c)
        self.by_day = {k: g for k, g in c.groupby(["_dk", "league"], sort=False)}
        log.info(f"[curve] indexed {len(c):,} price rows across {len(self.by_day):,} "
                 f"(date, league) groups")

    def fixture(self, league: str, home: str, away: str, day: str):
        """Every captured price for one fixture, or None. League-scoped name resolution."""
        from src.team_names import resolve
        g = self.by_day.get((str(day)[:10], str(league)))
        if g is None or g.empty:
            return None
        h = resolve(str(home), list(g["_h"]))
        a = resolve(str(away), list(g["_a"]))
        if not h or not a:
            return None
        m = g[(g["_h"] == h) & (g["_a"] == a)]
        return m if not m.empty else None

    def curve(self, league: str, home: str, away: str, day: str, market: str,
              player: str | None = None) -> dict | None:
        """Opening, closing and movement for one fixture+market. None when never priced.

        `player` is REQUIRED for prop markets and ignored elsewhere. A prop fixture carries a
        separate price curve per player — dozens of them under the same market key — so a
        lookup that omits the player attaches an arbitrary team-mate's curve to every row. That
        is not a small error: it produced a mean CLV of +18.6% out of nothing.
        """
        fx = self.fixture(league, home, away, day)
        if fx is None:
            return None
        m = fx[fx["market"].astype(str) == str(market)]
        if player is not None and "player" in m.columns:
            want = str(player).strip().lower()
            m = m[m["player"].astype(str).str.strip().str.lower() == want]
        elif "player" in m.columns and m["player"].notna().any():
            # A prop market reached without a player: refuse rather than pick one at random.
            return None
        if m.empty:
            return None
        o, c = float(m.iloc[0]["odds"]), float(m.iloc[-1]["odds"])
        # Positive drift = the price SHORTENED between open and close = money came in on it.
        drift = (o - c) / o * 100.0 if o else np.nan
        return {"open": round(o, 3), "close": round(c, 3), "n": int(len(m)),
                "first_ts": m.iloc[0]["_ts"], "last_ts": m.iloc[-1]["_ts"],
                "drift_pct": round(drift, 2), "track": m.iloc[0]["_track"],
                "source": m.iloc[0]["_src"]}


def movement_signal(drift_pct: float | None) -> str:
    """CONFIRMED (our side shortened), CONFLICTED (drifted), NEUTRAL, or NO_CURVE.

    NO_CURVE is deliberately NOT the same as NEUTRAL. Neutral means the price was watched and
    barely moved; NO_CURVE means nothing was ever captured. Collapsing the two is how a missing
    feed reads as a measured non-event — the same mistake that let `drift_signal` sit on "New"
    for 100% of rows for months while looking like a working signal.
    """
    if drift_pct is None or (isinstance(drift_pct, float) and np.isnan(drift_pct)):
        return "NO_CURVE"
    if drift_pct >= CONFIRM_PCT:
        return "CONFIRMED"
    if drift_pct <= -CONFLICT_PCT:
        return "CONFLICTED"
    return "NEUTRAL"


def clv_pct(entry: float | None, close: float | None) -> float | None:
    """Percent, positive = we beat the close. Same convention as bets_ledger and ht_ledger."""
    if not entry or not close or close <= 1.0:
        return None
    return round((entry / close - 1.0) * 100.0, 2)


def enrich(df: pd.DataFrame, market_of, idx: CurveIndex | None = None,
           league_col="league", home_col="home_team", away_col="away_team",
           date_col="match_date", entry_col=None, player_col=None,
           prefix="mv_") -> pd.DataFrame:
    """Attach opening / closing / drift / signal / CLV to any ledger.

    `market_of` maps a row to the market key its price lives under, because every ledger names
    its market differently — the HT ledger stores `ht_under15` directly, the side ledger stores
    `btts` and needs the side to become `btts_yes`, the main ledger stores a side of OVER/UNDER
    on an implicit OU25. Passing a function keeps that per-ledger knowledge at the call site
    instead of hard-coding a mapping that silently rots.

    Columns added are prefixed so nothing existing is overwritten: an enrichment must never
    clobber a settled figure.
    """
    idx = idx or CurveIndex()
    out = df.copy()
    # NUMERIC AND TEXT COLUMNS ARE SEPARATED ON PURPOSE. An all-NaN column reads back as
    # float64, and writing a string into it raises TypeError mid-loop — the enrichment then
    # dies after doing its work and before saving, which presents as "nothing to backfill".
    # The same trap cost a full grading run on ht_ledger's `notes` column.
    for c in ("open", "close", "n_snaps", "drift_pct", "clv_pct"):
        out[f"{prefix}{c}"] = np.nan
    for c in ("signal", "source"):
        out[f"{prefix}{c}"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out[f"{prefix}signal"] = "NO_CURVE"

    hits = 0
    for i, row in out.iterrows():
        mkt = market_of(row)
        if not mkt:
            continue
        c = idx.curve(row.get(league_col), row.get(home_col), row.get(away_col),
                      row.get(date_col), mkt,
                      player=row.get(player_col) if player_col else None)
        if c is None:
            continue
        hits += 1
        out.at[i, f"{prefix}open"] = c["open"]
        out.at[i, f"{prefix}close"] = c["close"]
        out.at[i, f"{prefix}n_snaps"] = c["n"]
        out.at[i, f"{prefix}drift_pct"] = c["drift_pct"]
        out.at[i, f"{prefix}signal"] = movement_signal(c["drift_pct"])
        out.at[i, f"{prefix}source"] = c["source"]
        entry = pd.to_numeric(row.get(entry_col), errors="coerce") if entry_col else np.nan
        if pd.isna(entry):
            entry = c["open"]          # no recorded entry -> the opening price is our best proxy
        v = clv_pct(float(entry) if pd.notna(entry) else None, c["close"])
        if v is not None:
            out.at[i, f"{prefix}clv_pct"] = v
    log.info(f"[curve] enriched {hits:,} of {len(out):,} rows with a real price curve")
    return out
