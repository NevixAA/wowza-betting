"""
CLV tracker (props) — wires the paper feed into src/clv_capture.
=================================================================
Year-long PAPER tracking of player props (no real money). For every tip we log the ENTRY
price the FIRST time we see it (≈ opening), then after kickoff fill the CLOSING price (the
last snapshot before kickoff) and compute CLV. Positive persistent CLV would be the only
signal that live diverges from the (efficient-market) backtest.

BASIS (2026-11-02): entry and close are both BEST-OF-BOOKS (see close_best_of_books); every
entry logged since is labelled `clv_basis=best_vs_best` in its notes.

INTEGRITY (2026-11-06, Phase 2 part 7): a close is written only for a record that passes the checks of
player_model.clv_integrity - the entry price was quoted for this player / market / fixture by the tip
time, the kickoff is recorded, the tip was logged before it, and the close is taken from quotes
strictly BEFORE kickoff (in-play rows used to be eligible, AU4-P8-03). Every record gets `clv_status`;
one that fails gets its status (IDENTITY_UNPROVEN, ENTRY_AFTER_KICKOFF, TIMING_UNPROVEN, NO_CLOSE)
and no CLV value. The entry's kickoff is written with it (`kickoff_utc`).

DATA REALITY: our prop odds are OVER-PRICE ONLY. So `clv_prob` (no-vig) can't be
computed (needs two-sided), but `clv_pct = odds_entry/odds_close − 1` DOES work one-sided and
is the metric we read. ISOLATION: props-only; writes only output/clv_records.csv. Standard
model / team pipeline untouched.
"""
from __future__ import annotations
from collections import Counter

import pandas as pd
from datetime import datetime, timezone

from src import clv_capture
from player_model import config
from player_model import clv_integrity as CI
from player_model.odds_fetcher import _norm
from player_model.club_rule import same_fixture_sides

# Track CLV for EVERY tier that has odds (2026/27: keep the full open->close->CLV record for
# the whole test slate, not just the sent feed). Rows without odds are skipped inside log_new_tips.
_TRACK_TIERS = {"QUALIFIED", "PAPER", "SNIPER", "MARKSMAN", "VALUABLE", "WATCH", "AVOID"}


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
        ko = row.get("kickoff_utc")
        clv_capture.log_bet(bid, row.get("market", ""), row.get("player_name", ""),
                            row.get("match", ""), "over", odds, None,
                            notes=f"{row.get('tier', '')};{CLV_BASIS}",
                            kickoff_utc=ko if isinstance(ko, str) else "")
        existing.add(bid); n += 1
    if n:
        print(f"[clv_tracker] logged {n} new tip entry prices")
    return n


def _tip_kickoffs() -> dict:
    """{(date, match): kickoff_utc} from the current board (output/player_tips.csv), for records
    written before entries carried their kickoff. A (date, match) with two kickoffs is left out."""
    try:
        t = pd.read_csv(config.OUTPUT_DIR / "player_tips.csv", usecols=["date", "match", "kickoff_utc"],
                        dtype=str)
    except Exception:                                                    # noqa: BLE001 - no board
        return {}
    t = t.dropna().drop_duplicates()
    t["date"] = t["date"].str[:10]
    one = t.groupby(["date", "match"])["kickoff_utc"].nunique() == 1
    t = t.set_index(["date", "match"])
    return {k: t.loc[[k], "kickoff_utc"].iloc[0] for k in one[one].index}


def _player_ids() -> dict:
    """player_ledger provider ids per (name, market, date) - a name with two ids is a namesake."""
    try:
        led = pd.read_csv(config.OUTPUT_DIR / "player_ledger.csv", dtype=str,
                          usecols=lambda c: c in ("player_name", "market", "match_date", "player_id"))
    except Exception:                                                    # noqa: BLE001
        return {}
    return CI.ledger_ids(led)


def close_out(odds_history_path=None, kickoff_map: dict | None = None) -> int:
    """For open CLV records whose kickoff has passed, set the CLOSING price = the best-of-books close
    over quotes strictly before kickoff, for the same player / market / fixture, then compute CLV.
    A record that fails identity or timing gets its clv_status instead (clv_integrity.decide_close).
    `kickoff_map` = {bet_id: kickoff_utc}; else the record's own kickoff_utc, else the board's."""
    path = odds_history_path or (config.OUTPUT_DIR / "player_prop_odds_history.csv")
    rows = clv_capture._load()
    # The history is a ROLLING file (src/rolling_store): matches already played live in
    # player_prop_odds_history_archive/<match_date>.csv. Read only the archive days that still
    # have open records, plus the live file -- so a record closed late still finds its close.
    open_days = set()
    for r in rows:
        if r.get("odds_close") in ("", None) and r.get("clv_status", "") not in CI.FINAL:
            parts = str(r.get("bet_id", "")).split("|", 3)
            if len(parts) == 4:
                open_days.add(parts[1][:10])
    try:
        from src.rolling_store import read_all
        oh = read_all(path, dates=open_days)
    except Exception:
        print(f"[clv_tracker] no odds history at {path}"); return 0
    if oh.empty or "player" not in oh.columns:
        return 0
    idx = CI.index_history(CI.prepare_history(oh))
    board_ko = None
    ids = _player_ids()
    now = datetime.now(timezone.utc)
    closed, labelled = 0, Counter()
    for r in rows:
        if r.get("odds_close") not in ("", None) or r.get("clv_status", "") in CI.FINAL:
            continue                      # already closed, or already given a final status
        parsed = CI.parse_bet_id(r.get("bet_id"))
        if parsed is None:
            continue
        d, player, market = parsed
        ko = (kickoff_map or {}).get(r["bet_id"]) or r.get("kickoff_utc") or None
        if not ko:
            if board_ko is None:
                board_ko = _tip_kickoffs()
            ko = board_ko.get((d, str(r.get("match", "") or "")))
        ko_t = CI.utc(ko)
        if pd.notna(ko_t):
            if ko_t > now:
                continue                  # not kicked off yet
        elif d >= now.strftime("%Y-%m-%d"):
            continue                      # no kickoff + match today/future -> wait
        pids = ids.get((_norm(player), market, d), set())
        # Same FIXTURE only (2026-10-25): a namesake playing elsewhere that day (Matheuzinho in
        # Vitoria v Atletico-PR vs Vasco v Corinthians) must not supply this record's close.
        m = CI.identity_rows(idx, d, player, market, str(r.get("match", "") or ""),
                             next(iter(pids)) if len(pids) == 1 else None)
        dec = CI.decide_close(r, m, ko_t, namesake=len(pids) > 1)
        if dec["status"] in CI.VALID:
            clv_capture.capture_close(r["bet_id"], dec["close"], None, dec["status"], dec["reason"])
            closed += 1
        else:
            clv_capture.mark_status(r["bet_id"], dec["status"], dec["reason"])
            labelled[dec["status"]] += 1
    if closed or labelled:
        print(f"[clv_tracker] closed out {closed} records (basis: {CLV_BASIS}); "
              f"no CLV (status): {dict(labelled) or 'none'}")
    return closed


#: Label written into clv_records.notes when a tip's entry is logged (log_new_tips): entry and
#: close are both best-of-books. Records logged before 2026-11-02 carry no label; any of them
#: closed before then used the last logged row of ANY single book as the close.
CLV_BASIS = "clv_basis=best_vs_best"


def close_best_of_books(m: pd.DataFrame) -> float | None:
    """The CLOSE on the same basis as the ENTRY.

    The entry (`market_odds` on the tip) is the BEST price across books (odds_fetcher._best_price,
    and a second source only ever raises it). The close used to be the last history row, which
    for API-Football is ONE bookmaker's quote (rows are per book), so CLV compared best-of-books
    against a single book. Now: each book's LAST logged price, then the best of those. OddsAPI
    rows carry no bookmaker - each is already the best across OddsAPI books at its snapshot - so
    they form one series of their own.
    """
    if m is None or m.empty:
        return None
    d = m.copy()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d = d[d["odds"] > 1.0]
    if d.empty:
        return None
    src = d["source"].astype(str) if "source" in d.columns else pd.Series("oddsapi", index=d.index)
    book = (d["bookmaker"].astype(str).where(d["bookmaker"].notna(), "")
            if "bookmaker" in d.columns else pd.Series("", index=d.index))
    d["_book"] = src.str.strip().str.lower() + "|" + book.str.strip().str.lower().replace("nan", "")
    last = d.sort_values("snapshot_ts", kind="mergesort").groupby("_book", sort=False).tail(1)
    return float(last["odds"].max())




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
    "cards":   lambda s: _cards_result(s) == "WIN",
}


def _cards_result(s: dict) -> str:
    """THE one card rule, update_results.card_market_result (a yellow OR a red card wins; LOSS only
    when both are known to be 0; '' = pending). This grader used to keep its own lambda and held
    'yellow known >= 1, red unknown' pending where the ledger settles it as a WIN (round 5)."""
    from update_results import card_market_result
    return card_market_result(s.get("yc"), s.get("rc"))
#: stats each market reads; a None among them leaves the record PENDING (Data Truth 2026-11-02).
_MARKET_NEEDS = {m: (("yc", "rc") if m == "cards" else
                     ("goals",) if m.startswith("goals") else
                     ("assists",) if m == "assists" else ("sot",))
                 for m in _MARKET_HIT}


def _id_str(v) -> str | None:
    """A provider id as a canonical string ('1635606.0' -> '1635606'), or None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return str(int(f)) if f == f and f > 0 else None


def _stat(v) -> float | None:
    """A recorded count as float, or None when the stats row has no value (never 0)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _settle_stats(rec) -> dict:
    """The grading stats of one player_history row; a missing stat is None, not 0 (was `or 0`)."""
    return {"goals": _stat(rec.goals), "assists": _stat(rec.assists),
            "sot": _stat(rec.shots_on_target),
            "yc": _stat(rec.yellow_cards), "rc": _stat(rec.red_cards)}


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
             "shots_on_target", "yellow_cards", "red_cards", "home_team", "away_team"]
    try:                       # player_id landed 2026-08-15; tolerate older parquets
        pq = pd.read_parquet(pqp, columns=_cols + ["player_id", "fixture_id"])
    except Exception:
        try:
            pq = pd.read_parquet(pqp, columns=_cols + ["player_id"])
        except Exception:
            pq = pd.read_parquet(pqp, columns=_cols)
            pq["player_id"] = None
        pq["fixture_id"] = None
    pq["dkey"] = pq["date"].astype(str).str[:10]
    # Exact index. Joining players by NAME is the weak link that has kept props ungraded:
    # any spelling variance between the tip and the stats row silently drops the record, and a
    # namesake's row grades the wrong player. player_ledger carries player_id (and, since issue #1
    # step 1, fixture_id): (player_id, fixture_id) when both sides have the fixture, else
    # (player_id, date) for legacy rows.
    idx_by_id: dict = {}
    idx_by_fx: dict = {}
    for r in pq.itertuples():
        pid = _id_str(r.player_id)
        if pid is None:
            continue
        idx_by_id[(pid, r.dkey)] = r
        fid = _id_str(r.fixture_id)
        if fid is not None:
            idx_by_fx[(pid, fid)] = r

    # clv bet_ids carry the tip's player NAME (the same string player_ledger wrote for that tip);
    # the ledger row supplies its ids. This is an exact key between two files of one pipeline run,
    # never a match against another source's spelling.
    pid_lookup: dict = {}
    try:
        _led = pd.read_csv(config.OUTPUT_DIR / "player_ledger.csv", dtype=str)
        if "player_id" in _led.columns:
            for _, _r in _led.iterrows():
                _pid = _id_str(_r.get("player_id"))
                if _pid is not None:
                    pid_lookup[(_norm(_r.get("player_name", "")),
                                str(_r.get("market", "")),
                                str(_r.get("match_date", ""))[:10])] = (
                        _pid, _id_str(_r.get("fixture_id")), _r.get("home_team"), _r.get("away_team"))
    except Exception as e:
        print(f"[clv_tracker] settle: player_ledger.csv unreadable ({type(e).__name__}: {e}) - "
              f"no player ids, nothing can be graded")

    pq_max = str(pq["dkey"].max()) if len(pq) else "(empty)"
    today = str(_dt.utcnow().date())
    graded = 0
    no_id = 0                  # no player_id for the tip -> cannot be graded honestly
    no_stats: list[str] = []   # played, but the parquet has no stats row -> can't grade
    unknown_stat = 0           # stats row found, but the market's stat is missing -> pending
    ghost = 0                  # the player's row that day is another fixture -> pending
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
        # By player_id ONLY (Data Truth, 2026-11-01). The id comes from the ledger row the same tip
        # wrote; the old fallback joined the stats parquet on the normalised NAME, which graded a
        # namesake's match. No id -> the record stays PENDING (counted below).
        ids = pid_lookup.get((_norm(player), market, d))
        if ids is None:
            no_id += 1
            continue
        _pid, _fid, _home, _away = ids
        rec = idx_by_fx.get((_pid, _fid)) if _fid is not None else idx_by_id.get((_pid, d))
        hit = _MARKET_HIT.get(market)
        if hit is None:
            continue
        if rec is not None and _fid is None and not same_fixture_sides(
                _home, _away, getattr(rec, "home_team", None), getattr(rec, "away_team", None)):
            # CURRENT-CLUB RULE (invariant 12): the player's stats row that day is ANOTHER fixture
            # (he played for another club) - never graded off it. PENDING, counted.
            ghost += 1
            continue
        if rec is None:
            # Match not in our player-stats parquet yet (data gap / not collected) -> leave
            # PENDING; it grades once the fixture's stats land in the parquet. Never guess a VOID.
            no_stats.append(d)
            continue
        s = _settle_stats(rec)
        if market == "cards":
            # One card rule (update_results.card_market_result): a known card settles even when
            # the other colour is missing; '' = unknown -> PENDING.
            if not _cards_result(s):
                unknown_stat += 1
                continue
        elif any(s[k] is None for k in _MARKET_NEEDS[market]):
            # A stat the market needs is missing (provider null): unknown, NOT zero. PENDING.
            unknown_stat += 1
            continue
        won = bool(hit(s))
        try:
            odds = float(clv.at[i, "odds_bet"])
        except (TypeError, ValueError):
            odds = float("nan")
        clv.at[i, "result"] = "WIN" if won else "LOSS"
        # A WIN without a recorded bet price has unknown P/L, left blank (was odds 0.0 -> -1.0).
        if not won:
            clv.at[i, "pnl"] = -1.0
        elif odds == odds and odds > 1.0:
            clv.at[i, "pnl"] = round(odds - 1.0, 3)
        else:
            clv.at[i, "pnl"] = float("nan")
        graded += 1

    if graded:
        clv.to_csv(f, index=False)
    print(f"[clv_tracker] settle: {graded} record(s) graded"
          + (f"; {no_id} left PENDING (no player_id on the tip)" if no_id else "")
          + (f"; {unknown_stat} left PENDING (the market's stat is missing in the stats row)"
             if unknown_stat else "")
          + (f"; {ghost} left PENDING (the player's stats row that day is another fixture - "
             f"current-club rule)" if ghost else ""))
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
