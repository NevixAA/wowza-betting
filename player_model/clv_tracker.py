"""
CLV tracker (props) — wires the paper feed into src/clv_capture.
=================================================================
Year-long PAPER tracking of player props (no real money). For every tip we log the ENTRY
price the FIRST time we see it (≈ opening), then after kickoff fill the CLOSING price (the
last snapshot before kickoff) and compute CLV. Positive persistent CLV would be the only
signal that live diverges from the (efficient-market) backtest.

DATA REALITY: our prop odds are single-book, OVER-PRICE ONLY. So `clv_prob` (no-vig) can't be
computed (needs two-sided), but `clv_pct = odds_entry/odds_close − 1` DOES work one-sided and
is the metric we read. ISOLATION: props-only; writes only output/clv_records.csv. Standard
model / team pipeline untouched.
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime, timezone

from src import clv_capture
from player_model import config
from player_model.odds_fetcher import _norm

_TRACK_TIERS = {"PAPER", "SNIPER", "MARKSMAN"}


def _bet_id(date, player, market) -> str:
    return f"PLAYER|{str(date)[:10]}|{player}|{market}"


def log_new_tips(tips_df: pd.DataFrame) -> int:
    """Log the ENTRY (first-seen) price for each tracked tip with odds. Idempotent: a bet_id
    already logged is NOT overwritten, so we keep the opening price, not the latest snapshot."""
    if tips_df is None or tips_df.empty or "tier" not in tips_df.columns:
        return 0
    existing = {r["bet_id"] for r in clv_capture._load()}
    n = 0
    for _, row in tips_df.iterrows():
        if row.get("tier") not in _TRACK_TIERS:
            continue
        odds = row.get("market_odds")
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            continue
        if not odds or odds <= 1.0:
            continue
        bid = _bet_id(row.get("date"), row.get("player_name", ""), row.get("market", ""))
        if bid in existing:
            continue                      # keep first-seen entry price
        clv_capture.log_bet(bid, row.get("market", ""), row.get("player_name", ""),
                            row.get("match", ""), "over", odds, None,
                            notes=str(row.get("tier", "")))
        existing.add(bid); n += 1
    if n:
        print(f"[clv_tracker] logged {n} new tip entry prices")
    return n


def close_out(odds_history_path=None, kickoff_map: dict | None = None) -> int:
    """For open CLV records whose kickoff has passed, set the CLOSING price = the latest odds
    snapshot for that (player, market, date) in the prop-odds history, then compute CLV."""
    path = odds_history_path or (config.OUTPUT_DIR / "player_prop_odds_history.csv")
    try:
        oh = pd.read_csv(path)
    except Exception:
        print(f"[clv_tracker] no odds history at {path}"); return 0
    if oh.empty:
        return 0
    oh["_pk"] = oh["player"].astype(str).map(_norm)
    oh["_d"] = oh["match_date"].astype(str).str[:10]
    now = datetime.now(timezone.utc)
    rows = clv_capture._load()
    closed = 0
    for r in rows:
        if r.get("odds_close") not in ("", None):
            continue                      # already closed
        # bet_id = PLAYER|date|player|market
        try:
            _, d, player, market = r["bet_id"].split("|", 3)
        except ValueError:
            continue
        # only close after kickoff if we know it; else close once history has moved on
        if kickoff_map:
            ko = kickoff_map.get(r["bet_id"])
            if ko:
                try:
                    if datetime.fromisoformat(ko.replace("Z", "+00:00")) > now:
                        continue          # not kicked off yet
                except Exception:
                    pass
            else:
                if d >= now.strftime("%Y-%m-%d"):
                    continue              # no kickoff info + match not in the past -> wait
        elif d >= now.strftime("%Y-%m-%d"):
            continue                      # no kickoff map + match today/future -> don't close early
        pk = _norm(player)
        m = oh[(oh["_pk"] == pk) & (oh["market"].astype(str) == market) & (oh["_d"] == d)]
        if m.empty:
            continue
        closing = float(m.sort_values("snapshot_ts").iloc[-1]["odds"])   # last snapshot = close
        if closing > 1.0:
            clv_capture.capture_close(r["bet_id"], closing, None)
            closed += 1
    if closed:
        print(f"[clv_tracker] closed out {closed} records")
    return closed


def report() -> dict:
    rep = clv_capture.clv_report()
    print(f"[clv_tracker] CLV report: {rep}")
    return rep
