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

# Track CLV for EVERY tier that has odds (2026/27: keep the full open->close->CLV record for
# the whole test slate, not just the sent feed). Rows without odds are skipped inside log_new_tips.
_TRACK_TIERS = {"PAPER", "SNIPER", "MARKSMAN", "VALUABLE", "WATCH", "AVOID"}


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


# ── Result grading — turns captured CLV into an open→close→CLV→RESULT dataset ────
_MARKET_HIT = {
    "goals":   lambda s: s["goals"] >= 1,
    "goals2":  lambda s: s["goals"] >= 2,
    "goals3":  lambda s: s["goals"] >= 3,
    "assists": lambda s: s["assists"] >= 1,
    "sot":     lambda s: s["sot"] >= 1,
    "sot2":    lambda s: s["sot"] >= 2,
    "sot3":    lambda s: s["sot"] >= 3,
    "sot4":    lambda s: s["sot"] >= 4,
    "cards":   lambda s: (s["yc"] >= 1) or (s["rc"] >= 1),
}


def settle_results(parquet_path=None, void_after_days: int = 3) -> int:
    """Grade pending clv_records against ACTUAL player stats (player_history.parquet).

    Fills `result` (WIN/LOSS/VOID) + `pnl` (flat 1u on the entry price — PAPER only). A player
    not found for a match played more than `void_after_days` ago = DNP/uncovered → VOID.
    Props-only; touches only output/clv_records.csv. Returns the number graded.
    """
    from datetime import datetime as _dt
    from pathlib import Path
    f = config.OUTPUT_DIR / "clv_records.csv"
    if not f.exists():
        return 0
    clv = pd.read_csv(f)
    if clv.empty:
        return 0
    for c in ("result", "pnl"):
        if c not in clv.columns:
            clv[c] = ""
    # avoid float64/arrow-string columns rejecting mixed str/float writes
    clv["result"] = clv["result"].astype(object)
    clv["pnl"]    = clv["pnl"].astype(object)
    pending = clv["result"].isna() | (clv["result"].astype(str).str.strip() == "")
    if not pending.any():
        print("[clv_tracker] settle: no pending records")
        return 0

    pqp = Path(parquet_path) if parquet_path else (config.OUTPUT_DIR.parent / "player_history.parquet")
    if not pqp.exists():
        print(f"[clv_tracker] settle: parquet not found ({pqp})")
        return 0
    _cols = ["player_name", "date", "goals", "assists",
             "shots_on_target", "yellow_cards", "red_cards"]
    try:                       # player_id landed 2026-08-15; tolerate older parquets
        pq = pd.read_parquet(pqp, columns=_cols + ["player_id"])
    except Exception:
        pq = pd.read_parquet(pqp, columns=_cols)
        pq["player_id"] = None
    pq["pk"]   = pq["player_name"].astype(str).map(_norm)
    pq["dkey"] = pq["date"].astype(str).str[:10]
    idx = {(r.pk, r.dkey): r for r in pq.itertuples()}
    # Exact index. Joining players by NAME is the weak link that has kept props ungraded:
    # any spelling variance between the tip and the stats row silently drops the record.
    # player_ledger now carries player_id, so resolve it and match on (id, date) first.
    idx_by_id = {(str(r.player_id), r.dkey): r for r in pq.itertuples()
                 if r.player_id is not None and str(r.player_id) not in ("", "nan", "None")}

    pid_lookup: dict = {}
    try:
        _led = pd.read_csv(config.OUTPUT_DIR / "player_ledger.csv", dtype=str)
        if "player_id" in _led.columns:
            for _, _r in _led.iterrows():
                _pid = str(_r.get("player_id", "")).strip()
                if _pid and _pid not in ("nan", "None"):
                    pid_lookup[(_norm(_r.get("player_name", "")),
                                str(_r.get("market", "")),
                                str(_r.get("match_date", ""))[:10])] = _pid
    except Exception:
        pass

    pq_max = str(pq["dkey"].max()) if len(pq) else "(empty)"
    today = str(_dt.utcnow().date())
    graded = 0
    no_stats: list[str] = []   # played, but the parquet has no stats row -> can't grade
    for i in clv[pending].index:
        try:
            _, d, player, market = str(clv.at[i, "bet_id"]).split("|", 3)
        except ValueError:
            continue
        d = d[:10]
        if d >= today:                      # not played yet
            continue
        # Prefer the exact (player_id, date) match; fall back to the name for rows written
        # before player_id existed.
        rec = None
        _pid = pid_lookup.get((_norm(player), market, d))
        if _pid:
            rec = idx_by_id.get((_pid, d))
        if rec is None:
            rec = idx.get((_norm(player), d))
        hit = _MARKET_HIT.get(market)
        if hit is None:
            continue
        if rec is None:
            # Match not in our player-stats parquet yet (data gap / not collected) -> leave
            # PENDING; it grades once the fixture's stats land in the parquet. Never guess a VOID.
            no_stats.append(d)
            continue
        s = {"goals":   float(rec.goals or 0),      "assists": float(rec.assists or 0),
             "sot":     float(rec.shots_on_target or 0),
             "yc":      float(rec.yellow_cards or 0), "rc":     float(rec.red_cards or 0)}
        won = bool(hit(s))
        try:
            odds = float(clv.at[i, "odds_bet"])
        except (TypeError, ValueError):
            odds = 0.0
        clv.at[i, "result"] = "WIN" if won else "LOSS"
        clv.at[i, "pnl"] = round(odds - 1.0, 3) if won else -1.0
        graded += 1

    if graded:
        clv.to_csv(f, index=False)
    print(f"[clv_tracker] settle: {graded} record(s) graded")
    if no_stats:
        # Be LOUD about this. A silent "0 graded" looks like "nothing to do", but the real
        # cause is upstream: player_history.parquet has no rows for those match dates, so the
        # collect step is behind. That is what kept clv_records.result 100% empty while CLV
        # itself was being captured fine (found 2026-08-15: parquet ended 2026-07-04 while
        # bets ran to 2026-08-17 — 26 played bets, 0 gradeable).
        print(f"[clv_tracker] settle: WARNING — {len(no_stats)} played bet(s) have NO player "
              f"stats and cannot grade. player_history.parquet ends {pq_max}; earliest "
              f"ungradeable match is {min(no_stats)}. Run `python -m player_model.pipeline "
              f"--mode collect` (or check the Sunday collect step) — the grader is fine, "
              f"its input is stale.")
    return graded
