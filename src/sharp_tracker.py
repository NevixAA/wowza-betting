"""
Sharp Money Tracker v2
======================
Tracks odds drift across all enabled leagues.

Signals (owner 2026-11-15: per-league thresholds, no global constant; definition from research
sharp-filter/1). A side's MOVE is its fair-probability change from the snapshot closest to 72 h before
kickoff ([60, 84] h) to the current one; it is judged against the same league's moves (same market,
side for 1X2, lead-time bucket) of fixtures dated strictly earlier. Fewer than 100 such fixtures: no
signal.
  STRONG       — move toward the side above the league's 90th percentile
  SHARP        — above the league's 75th percentile
  FADING       — move against the side above the 75th percentile
  STEAM_ prefix — the last window's move above the league's 90th percentile of single-window moves
                  (at least 4 books)

Output:
  output/sharp_history.json  — full snapshot history per match/market
  output/sharp_tips.csv      — flagged signals with consensus scores
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from player_model.ledger import append_sharp_signals

log = logging.getLogger(__name__)

SHARP_HISTORY_FILE = config.OUTPUT_DIR / "sharp_history.json"
SHARP_TIPS_FILE    = config.OUTPUT_DIR / "sharp_tips.csv"

# The global DRIFT_SHARP_PCT / DRIFT_STRONG_PCT / STEAM_PCT constants are GONE (owner 2026-11-15): every
# signal threshold is now learned per league from strictly earlier fixtures (see "Signals" below).
DAYS_AHEAD       = 14

# SNAPSHOT CADENCE (Data Truth round 5, owner: Pro 1X2 keeps its 12h max price age, so snapshot more often).
# Measured on main's sharp_history (2026-10-19..11-02): ~3 snapshots a day, at roughly 13-15, 18-21 and
# 22-01 UTC, and NONE between ~01:00 and ~13:00 UTC, so the newest h2h price was 12-14h old every
# morning. The workflow now asks every hour around the clock (.github/workflows/sharp_tracker.yml); this
# guard turns a fire that lands within MIN_INTERVAL_H of the newest stored snapshot into a no-op, so the
# credit cost is capped at one fetch per ~2h however many slots GitHub delivers.
MIN_INTERVAL_H = float(os.getenv("SHARP_MIN_INTERVAL_H", "1.75"))
# QUOTA GUARD: stop fetching when the account has fewer than this many credits left - production
# (predict's odds) comes first. Same default as the exchange capture (EXCHANGE_CREDIT_FLOOR).
CREDIT_FLOOR = int(os.getenv("SHARP_CREDIT_FLOOR", "15000"))
REGIONS = "eu"
MARKETS = ("totals", "h2h")


def new_usage() -> dict:
    """Credit accounting for one run (logged as an [oddsapi_usage] line for src/integrity/oddsapi_quota)."""
    return {"calls": 0, "credits": 0, "credits_by_league": {}, "credits_by_market": {},
            "requests_remaining": None, "stopped": None, "errors": 0}


def max_credits_per_snapshot(n_leagues: int) -> int:
    """Upper bound of one snapshot: OddsAPI bills markets x regions per /odds call (calls that return no
    events are not billed), one call per market per league."""
    return n_leagues * len(MARKETS) * len([r for r in REGIONS.split(",") if r.strip()])


# ── OddsAPI fetch — median across all bookmakers ──────────────────────────────

def _fetch_odds_for_league(sport_key: str, league: str, usage: dict | None = None) -> list[dict]:
    """Fetch totals + h2h odds — median across ALL bookmakers + consensus count."""
    results = []
    cutoff  = datetime.utcnow() + timedelta(days=DAYS_AHEAD)
    usage = usage if usage is not None else new_usage()

    for market in MARKETS:
        rem = usage.get("requests_remaining")
        if rem is not None and rem < CREDIT_FLOOR:
            usage["stopped"] = (f"account below the {CREDIT_FLOOR}-credit floor ({rem} left) - "
                                f"production comes first")
            break
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    "apiKey":     config.ODDS_API_KEY,
                    "regions":    REGIONS,
                    "markets":    market,
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
        except Exception as e:
            usage["errors"] += 1
            log.warning(f"  {league} ({market}): request failed: {type(e).__name__}")
            continue

        usage["calls"] += 1
        hdr = getattr(r, "headers", None) or {}
        try:
            spent = int(float(hdr.get("x-requests-last") or 0))
        except (TypeError, ValueError):
            spent = 0
        usage["credits"] += spent
        usage["credits_by_league"][league] = usage["credits_by_league"].get(league, 0) + spent
        usage["credits_by_market"][market] = usage["credits_by_market"].get(market, 0) + spent
        try:
            if hdr.get("x-requests-remaining") not in (None, ""):
                usage["requests_remaining"] = int(float(hdr["x-requests-remaining"]))
        except (TypeError, ValueError):
            pass

        if r.status_code != 200:
            usage["errors"] += 1
            log.warning(f"  {league} ({market}): HTTP {r.status_code}")
            continue

        for event in r.json():
            dt = pd.to_datetime(event.get("commence_time", ""), errors="coerce", utc=True)
            if pd.isna(dt):
                continue
            dt_naive = dt.tz_localize(None)
            if dt_naive > cutoff or dt_naive < datetime.utcnow():
                continue

            # Collect odds from ALL bookmakers
            book_odds: dict[str, list[float]] = {
                "odds_over": [], "odds_under": [],
                "odds_home": [], "odds_away": [], "odds_draw": [],
            }

            for bm in event.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if market == "totals" and mkt["key"] == "totals":
                        for o in mkt["outcomes"]:
                            if o.get("point") == 2.5:
                                if o["name"] == "Over":
                                    book_odds["odds_over"].append(o["price"])
                                elif o["name"] == "Under":
                                    book_odds["odds_under"].append(o["price"])
                    elif market == "h2h" and mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            if o["name"] == event["home_team"]:
                                book_odds["odds_home"].append(o["price"])
                            elif o["name"] == event["away_team"]:
                                book_odds["odds_away"].append(o["price"])
                            elif o["name"] == "Draw":
                                book_odds["odds_draw"].append(o["price"])

            # Build entry with median odds + book count
            entry = {
                "id":         event["id"],
                "league":     league,
                "home":       event["home_team"],
                "away":       event["away_team"],
                "date":       str(dt_naive)[:16],
                "market":     market,
                "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            }

            n_books = 0
            for k, vals in book_odds.items():
                if vals:
                    entry[k] = round(statistics.median(vals), 3)
                    n_books = max(n_books, len(vals))

            entry["n_books"] = n_books

            if any(k in entry for k in ("odds_over", "odds_home")):
                results.append(entry)

    return results


# ── History management ────────────────────────────────────────────────────────

# Month-partitioned history.
#
# The single monolithic sharp_history.json had reached 12.7 MB and was rewritten in FULL on
# every run — 8 runs/day, ~91 MB/day of git blobs, ~2,009 MB of history across 514 commits.
# Combined with player_history.parquet that pushed .git to 1,299 MB and GitHub began showing
# "Cannot retrieve latest commit at this time".
#
# The fix uses a property of the data: a settled fixture's snapshots NEVER change again. Only
# the current month is still being written, so partitioning by month means the other months
# stop being rewritten and stop generating blobs. Expected steady state is roughly one ~1 MB
# partition churning instead of 12.7 MB.
#
# The in-memory shape is unchanged — callers still see one flat dict keyed by
# "{event_id}_{market}" — so nothing downstream needs to know about partitions.
SHARP_HISTORY_DIR = config.OUTPUT_DIR / "sharp_history"


def _partition_key(entry: dict) -> str:
    """YYYY-MM from the fixture date, or 'unknown' when absent. Never guess a date: a wrong
    partition would silently split one fixture's snapshots across two files."""
    d = str((entry or {}).get("date") or "")[:7]
    return d if len(d) == 7 and d[4] == "-" else "unknown"


def _load_history() -> dict:
    """Merge every month partition, plus the legacy monolith if it is still present.

    Reading both means the migration needs no flag day: the legacy file is consumed here and
    simply stops being written by _save_history below.
    """
    merged: dict = {}
    if SHARP_HISTORY_FILE.exists():
        try:
            merged.update(json.loads(SHARP_HISTORY_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    if SHARP_HISTORY_DIR.exists():
        for p in sorted(SHARP_HISTORY_DIR.glob("*.json")):
            try:
                merged.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                log.warning(f"[sharp] unreadable partition {p.name} — skipped, not deleted")
    return merged


def _save_history(history: dict) -> None:
    """Write each month partition only when its content actually changed.

    An unchanged partition is left byte-identical so git sees no diff and stores no blob —
    that is the entire saving. Nothing is ever deleted.
    """
    SHARP_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, dict] = {}
    for k, v in history.items():
        buckets.setdefault(_partition_key(v), {})[k] = v

    written = 0
    for month, entries in buckets.items():
        p = SHARP_HISTORY_DIR / f"{month}.json"
        payload = json.dumps(dict(sorted(entries.items())), indent=2, ensure_ascii=False)
        if p.exists():
            try:
                if p.read_text(encoding="utf-8") == payload:
                    continue          # byte-identical -> no write, no git diff, no blob
            except Exception:
                pass
        p.write_text(payload, encoding="utf-8")
        written += 1

    # Retire the monolith once its contents are safely partitioned. Renamed rather than
    # deleted — Prompt 2 section 3 / invariant: legacy outputs are research assets.
    if SHARP_HISTORY_FILE.exists():
        legacy = SHARP_HISTORY_FILE.with_suffix(".json.migrated")
        try:
            SHARP_HISTORY_FILE.replace(legacy)
            log.info(f"[sharp] migrated monolith -> {len(buckets)} month partition(s); "
                     f"old file kept as {legacy.name}")
        except Exception as e:
            log.warning(f"[sharp] could not retire monolith: {e}")

    log.info(f"[sharp] history: {len(history)} entries across {len(buckets)} month(s); "
             f"{written} partition(s) rewritten, {len(buckets) - written} unchanged")


def _extract_odds(ev: dict) -> dict:
    keys = ("odds_over", "odds_under", "odds_home", "odds_away", "odds_draw")
    return {k: ev[k] for k in keys if k in ev}


def _update_history(history: dict, events: list[dict]) -> dict:
    for ev in events:
        key = f"{ev['id']}_{ev['market']}"
        if key not in history:
            history[key] = {
                "league":    ev["league"],
                "home":      ev["home"],
                "away":      ev["away"],
                "date":      ev["date"],
                "market":    ev["market"],
                "opening":   _extract_odds(ev),
                "snapshots": [],
            }
        history[key]["snapshots"].append({
            "at":      ev["fetched_at"],
            "odds":    _extract_odds(ev),
            "n_books": ev.get("n_books"),       # unknown stays None, never an invented 1
        })
        # EVERY snapshot is kept (owner, 2026-11-02). The old `[-50:]` cap silently dropped the
        # EARLIEST prices of 1,430 fixtures (Jun-Oct 2026 partitions; Aug: 906 of 1,396 at the cap),
        # i.e. the price history nearest the opening was lost. Growth stays bounded per file by the
        # month partitions below: a settled month is never rewritten, only the current one churns.
    return history


# ── Signals: per-league thresholds learned from strictly earlier fixtures ─────
#
# OWNER 2026-11-15 ("Paper track + capture", item 3). The old rule labelled a move with GLOBAL constants
# (5% / 10% decimal drift since the first snapshot ever seen, 3% in one window = steam), the same number
# for every league and market, on decimal odds (a 5% drift at 1.5 is not the move a 5% drift at 4.0 is).
# Research sharp-filter/1 (research/sharp-filter, reports/SHARP_FILTER_RESEARCH.md) defined a move instead:
#
#   * the move of a side = its FAIR probability now minus at the OPEN snapshot, the open being the
#     snapshot closest to 72 h before kickoff inside [60, 84] h (the tracker's median pair / triple,
#     power de-vig for O/U 2.5, proportional for 1X2);
#   * a move is judged against the SAME league's moves (same market, side for 1X2, and lead-time bucket,
#     because a move keeps growing toward kickoff) of fixtures dated STRICTLY EARLIER and already
#     kicked off - learned from this tracker's own history every run, never a typed number;
#   * fewer than MIN_HISTORY such earlier fixtures -> NO signal for that row (missing, counted in the
#     log), never a global fallback.
#
# Labels keep their names for every reader (sharp_tips.csv, sharp_ledger.csv, Telegram):
#   SHARP  move toward the side above the league's SHARP_Q quantile of |move|
#   STRONG move toward the side above the league's STRONG_Q quantile
#   FADING move against the side above the SHARP_Q quantile
#   STEAM_ prefix: the LAST window's move toward the side above the league's STEAM_Q quantile of
#                  single-window moves (thin-market guard: at least 4 books), learned the same way.
OPEN_LEAD_H = (60.0, 84.0)
OPEN_TARGET_H = 72.0
#: lead-time buckets (h, lower exclusive, upper inclusive); a fixture outside them has no signal
LEAD_BUCKETS_H = ((36.0, 60.0), (18.0, 36.0), (10.0, 18.0), (4.0, 10.0), (0.0, 4.0))
SHARP_Q = 0.75          # sharp-filter/1 protocol quantile
STRONG_Q = 0.90
STEAM_Q = 0.90
MIN_HISTORY = 100       # sharp-filter/1 protocol min_history
MIN_STEAM_BOOKS = 4
SIDES = {"totals": (("OVER", "odds_over"), ("UNDER", "odds_under")),
         "h2h": (("HOME", "odds_home"), ("AWAY", "odds_away"), ("DRAW", "odds_draw"))}


def _ts(s) -> datetime | None:
    try:
        return datetime.strptime(str(s)[:16], "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None


def _lead_h(ko: datetime, at) -> float | None:
    t = _ts(at)
    return None if t is None else (ko - t).total_seconds() / 3600.0


def _bucket(lead: float | None):
    if lead is None:
        return None
    for lo, hi in LEAD_BUCKETS_H:
        if lo < lead <= hi:
            return (lo, hi)
    return None


def _fair(odds: dict, market: str) -> dict | None:
    """{side: fair probability} from one snapshot's median prices; None when a side is missing."""
    from src.market_edge import power_devig
    try:
        if market == "totals":
            o, u = float(odds["odds_over"]), float(odds["odds_under"])
            p = power_devig(o, u)
            return None if p is None else {"OVER": float(p), "UNDER": 1.0 - float(p)}
        h, d, a = float(odds["odds_home"]), float(odds["odds_draw"]), float(odds["odds_away"])
        if min(h, d, a) <= 1.0:
            return None
        inv = {"HOME": 1 / h, "DRAW": 1 / d, "AWAY": 1 / a}
        s = sum(inv.values())
        return {k: v / s for k, v in inv.items()}
    except (KeyError, TypeError, ValueError):
        return None


def _open_index(snaps: list, ko: datetime) -> int | None:
    best = None
    for i, s in enumerate(snaps):
        ld = _lead_h(ko, s.get("at"))
        if ld is None or not (OPEN_LEAD_H[0] <= ld <= OPEN_LEAD_H[1]):
            continue
        key = (abs(ld - OPEN_TARGET_H), -i)
        if best is None or key < best[0]:
            best = (key, i)
    return None if best is None else best[1]


def _thr_side(market: str, side: str) -> str:
    """O/U is symmetric (|move over| = |move under|): one distribution; 1X2 per side."""
    return "OU" if market == "totals" else side


def fixture_moves(rec: dict, ko: datetime) -> dict:
    """A KICKED-OFF fixture's moves for learning: {(side_key, bucket): |move| at the latest snapshot in
    that bucket}, and {side_key: [|single-window move| toward either side, after the open]}."""
    snaps = rec.get("snapshots") or []
    oi = _open_index(snaps, ko)
    out, steps = {}, {}
    if oi is None:
        return {"moves": out, "steps": steps}
    f0 = _fair(snaps[oi]["odds"], rec["market"])
    if f0 is None:
        return {"moves": out, "steps": steps}
    prev = f0
    for s in snaps[oi + 1:]:
        ld = _lead_h(ko, s.get("at"))
        if ld is None or ld <= 0:
            continue
        f = _fair(s["odds"], rec["market"])
        if f is None:
            continue
        b = _bucket(ld)
        for side, _ in SIDES[rec["market"]]:
            k = _thr_side(rec["market"], side)
            if b is not None:
                out[(k, b)] = abs(f[side] - f0[side])           # the latest snapshot in the bucket wins
            if rec["market"] == "h2h" or side == "OVER":
                steps.setdefault(k, []).append(abs(f[side] - prev[side]))
        prev = f
    return {"moves": out, "steps": steps}


class LeagueThresholds:
    """Per (league, market, side key, bucket): the quantiles of |move| over fixtures dated STRICTLY
    before a given date that have kicked off by `now`. A query with fewer than MIN_HISTORY earlier
    fixtures returns None (no signal)."""

    def __init__(self, history: dict, now: datetime):
        self.moves: dict = {}
        self.steps: dict = {}
        for rec in history.values():
            ko = _ts(rec.get("date"))
            if ko is None or ko >= now or rec.get("market") not in SIDES:
                continue
            m = fixture_moves(rec, ko)
            day = ko.date()
            for (k, b), v in m["moves"].items():
                self.moves.setdefault((rec.get("league"), rec["market"], k, b), []).append((day, v))
            for k, vs in m["steps"].items():
                # one value per fixture (its largest single-window move) so a busy fixture counts once
                self.steps.setdefault((rec.get("league"), rec["market"], k), []).append((day, max(vs)))

    @staticmethod
    def _q(pairs: list, before, qs: tuple) -> tuple | None:
        vals = [v for d, v in pairs if d < before]
        if len(vals) < MIN_HISTORY:
            return None
        s = pd.Series(vals, dtype=float)
        return tuple(float(s.quantile(q)) for q in qs) + (len(vals),)

    def move(self, league, market, side, bucket, before):
        return self._q(self.moves.get((league, market, _thr_side(market, side), bucket), []), before,
                       (SHARP_Q, STRONG_Q))

    def steam(self, league, market, side, before):
        return self._q(self.steps.get((league, market, _thr_side(market, side)), []), before, (STEAM_Q,))


def _detect_steam(snapshots: list[dict], odds_key: str) -> bool:
    """Kept for readers of the old helper: True only when the caller's league threshold says so; the
    signal itself uses LeagueThresholds.steam (no global percentage)."""
    return False


def _drift_pct(opening: float, current: float) -> float:
    """Informational decimal drift (reported, never a decision)."""
    if not opening or not current:
        return 0.0
    return (current - opening) / opening


def _consensus_pct(snapshots: list[dict], odds_key: str, direction: float) -> int:
    """Informational: time-decay weighted share of snapshot-to-snapshot moves in the move's direction."""
    if len(snapshots) < 2:
        return 0
    import math as _math
    _ALPHA = 0.5
    t_first = _ts(snapshots[0].get("at"))
    total_w = agree_w = 0.0
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]["odds"].get(odds_key)
        curr = snapshots[i]["odds"].get(odds_key)
        if not prev or not curr:
            continue
        t_curr = _ts(snapshots[i].get("at"))
        w = _math.exp(_ALPHA * max(0.0, (t_curr - t_first).total_seconds() / 3600)) if (t_first and t_curr) else float(i)
        total_w += w
        if (curr - prev < 0) == (direction < 0):
            agree_w += w
    return 0 if total_w <= 0 else round(agree_w / total_w * 100)


TIP_COLS = ["date", "league", "match", "market", "opening_odds", "current_odds", "drift_pct", "signal", "steam",
            "consensus_pct", "n_books", "snapshots", "updated_at", "move_pp", "threshold_pp", "threshold_n",
            "lead_h", "open_lead_h"]


def _build_tips(history: dict, now: datetime | None = None, counts: dict | None = None) -> pd.DataFrame:
    rows = []
    now = now or datetime.utcnow()
    from collections import Counter
    counts = counts if counts is not None else Counter()
    if not isinstance(counts, Counter):
        raise TypeError("counts must be a collections.Counter (a tally that starts at zero)")
    th = LeagueThresholds(history, now)
    for key, rec in history.items():
        snaps = rec.get("snapshots") or []
        ko = _ts(rec.get("date"))
        if len(snaps) < 2 or ko is None or ko < now or rec.get("market") not in SIDES:
            continue
        oi = _open_index(snaps, ko)
        cur = snaps[-1]
        lead = _lead_h(ko, cur.get("at"))
        b = _bucket(lead)
        if oi is None or b is None or oi == len(snaps) - 1:
            counts["no_window"] += 1
            continue
        f0, f1 = _fair(snaps[oi]["odds"], rec["market"]), _fair(cur["odds"], rec["market"])
        if f0 is None or f1 is None:
            counts["no_fair_price"] += 1
            continue
        f_prev = _fair(snaps[-2]["odds"], rec["market"])
        n_books = cur.get("n_books")
        for side, ok in SIDES[rec["market"]]:
            op, cu = snaps[oi]["odds"].get(ok), cur["odds"].get(ok)
            if not op or not cu:
                continue
            if side == "DRAW" and not (1.3 <= op <= 20.0 and 1.3 <= cu <= 20.0):
                continue
            if side in ("OVER", "UNDER") and not (1.05 <= op <= 15.0 and 1.05 <= cu <= 15.0):
                continue
            if abs(op - cu) / op > 0.70:
                continue
            q = th.move(rec.get("league"), rec["market"], side, b, ko.date())
            if q is None:
                counts["no_league_history"] += 1
                continue
            q_sharp, q_strong, n = q
            mv = f1[side] - f0[side]
            if mv > q_strong:
                sig = "STRONG"
            elif mv > q_sharp:
                sig = "SHARP"
            elif mv < -q_sharp:
                sig = "FADING"
            else:
                continue
            steam = False
            if sig in ("STRONG", "SHARP") and f_prev is not None and n_books is not None and n_books >= MIN_STEAM_BOOKS:
                qs = th.steam(rec.get("league"), rec["market"], side, ko.date())
                steam = bool(qs is not None and (f1[side] - f_prev[side]) > qs[0])
            if steam:
                sig = "STEAM_" + sig
            d = _drift_pct(op, cu)
            label = (f"O/U 2.5 {side}" if rec["market"] == "totals"
                     else f"1X2 {side} ({rec['home'] if side == 'HOME' else rec['away'] if side == 'AWAY' else 'Draw'})")
            rows.append({
                "date": rec["date"][:10], "league": rec.get("league", ""), "match": f"{rec['home']} vs {rec['away']}",
                "market": label, "opening_odds": round(op, 3), "current_odds": round(cu, 3),
                "drift_pct": round(d * 100, 1), "signal": sig, "steam": steam,
                "consensus_pct": _consensus_pct(snaps[oi:], ok, mv * -1), "n_books": n_books,
                "snapshots": len(snaps), "updated_at": cur["at"], "move_pp": round(mv * 100, 2),
                "threshold_pp": round((q_strong if sig.endswith("STRONG") else q_sharp) * 100, 2),
                "threshold_n": n, "lead_h": round(lead, 1),
                "open_lead_h": round(_lead_h(ko, snaps[oi].get("at")), 1)})
    if not rows:
        return pd.DataFrame(columns=TIP_COLS)
    df = pd.DataFrame(rows)[TIP_COLS]
    order = {"STEAM_STRONG": 0, "STEAM_SHARP": 1, "STRONG": 2, "SHARP": 3, "FADING": 4}
    df["_order"] = df["signal"].map(order)
    df = df.sort_values(["_order", "move_pp"], ascending=[True, False])
    return df.drop("_order", axis=1).reset_index(drop=True)


def last_snapshot_at(history: dict) -> datetime | None:
    """Newest snapshot time (UTC, naive) anywhere in the history; None when there is none."""
    best = None
    for rec in (history or {}).values():
        for s in (rec.get("snapshots") or [])[-1:]:
            t = pd.to_datetime(s.get("at"), errors="coerce")
            if pd.notna(t) and (best is None or t > best):
                best = t
    return None if best is None else best.to_pydatetime()


def price_freshness(history: dict, now: datetime, market: str = "h2h") -> dict:
    """Age of the newest snapshot of every upcoming fixture in `market` (what Pro 1X2 re-prices from):
    fixtures, max age, and how many are older than 12h."""
    ages = []
    for rec in (history or {}).values():
        if rec.get("market") != market or not rec.get("snapshots"):
            continue
        ko = pd.to_datetime(rec.get("date"), errors="coerce")
        t = pd.to_datetime(rec["snapshots"][-1].get("at"), errors="coerce")
        if pd.isna(ko) or pd.isna(t) or ko.to_pydatetime() <= now:
            continue
        ages.append((now - t.to_pydatetime()).total_seconds() / 3600.0)
    return {"market": market, "upcoming_fixtures": len(ages),
            "max_age_h": round(max(ages), 2) if ages else None,
            "older_than_12h": sum(a > 12 for a in ages)}


def _log_usage(usage: dict, status: str, n_leagues: int, fresh: dict) -> None:
    """One machine-readable line per run for src/integrity/oddsapi_quota.py. Never raises, never the key."""
    try:
        log.info("[oddsapi_usage] " + json.dumps({
            "source": "sharp_tracker", "workflow": os.getenv("GITHUB_WORKFLOW", ""),
            "run_id": os.getenv("GITHUB_RUN_ID", ""), "status": status, "regions": REGIONS,
            "calls": usage["calls"], "credits": usage["credits"],
            "credits_by_market": usage["credits_by_market"], "credits_by_league": usage["credits_by_league"],
            "max_credits_per_snapshot": max_credits_per_snapshot(n_leagues), "errors": usage["errors"],
            "stopped": usage["stopped"], "requests_remaining": usage["requests_remaining"],
            "h2h_freshness": fresh}, sort_keys=True))
    except Exception:                                                   # noqa: BLE001
        pass


def run(now: datetime | None = None):
    log.info("Sharp Money Tracker v2 — volume-weighted + steam detection")
    now = now or datetime.utcnow()
    history = _load_history()
    total_events = 0
    leagues = [lg for lg in config.ENABLED_LEAGUES if config.ODDS_API_SPORT_KEYS.get(lg)]
    usage = new_usage()

    last = last_snapshot_at(history)
    if last is not None and (now - last).total_seconds() / 3600.0 < MIN_INTERVAL_H:
        log.info(f"[sharp] newest snapshot {last:%Y-%m-%d %H:%M} UTC is under {MIN_INTERVAL_H}h old - "
                 f"no fetch this slot (0 credits)")
        _log_usage(usage, "SKIPPED_FRESH", len(leagues), price_freshness(history, now))
        return None

    for league in leagues:
        sport_key = config.ODDS_API_SPORT_KEYS.get(league)
        events = _fetch_odds_for_league(sport_key, league, usage)
        if events:
            history = _update_history(history, events)
            total_events += len(events)
            log.info(f"  {league}: {len(events)} snapshots")
        if usage["stopped"]:
            log.warning(f"[sharp] stopped early: {usage['stopped']}")
            break

    fresh = price_freshness(history, now)
    _log_usage(usage, "STOPPED_QUOTA_FLOOR" if usage["stopped"] else "FETCHED", len(leagues), fresh)
    log.info(f"[sharp] {usage['calls']} call(s), {usage['credits']} credit(s); h2h prices: "
             f"{fresh['upcoming_fixtures']} upcoming fixtures, newest-price max age {fresh['max_age_h']}h, "
             f"{fresh['older_than_12h']} older than 12h")

    _save_history(history)
    from collections import Counter
    counts: Counter = Counter()
    tips = _build_tips(history, now, counts)
    log.info(f"[sharp] per-league thresholds: rows without a signal - {dict(counts) or 'none'} "
             f"(no_league_history = fewer than {MIN_HISTORY} earlier fixtures: no signal, never a global value)")
    tips.to_csv(SHARP_TIPS_FILE, index=False)
    append_sharp_signals(tips)

    steam_strong = tips[tips["signal"] == "STEAM_STRONG"]
    steam_sharp  = tips[tips["signal"] == "STEAM_SHARP"]
    strong       = tips[tips["signal"] == "STRONG"]
    sharp        = tips[tips["signal"] == "SHARP"]
    fading       = tips[tips["signal"] == "FADING"]

    log.info(f"\n{'='*55}")
    log.info(f"  SHARP MONEY v2 — {total_events} snapshots processed")
    log.info(f"{'='*55}")
    log.info(f"  STEAM_STRONG : {len(steam_strong)}")
    log.info(f"  STEAM_SHARP  : {len(steam_sharp)}")
    log.info(f"  STRONG       : {len(strong)}")
    log.info(f"  SHARP        : {len(sharp)}")
    log.info(f"  FADING       : {len(fading)}")

    top = tips[tips["signal"].isin(["STEAM_STRONG", "STEAM_SHARP", "STRONG"])]
    if not top.empty:
        log.info("  Top signals:")
        for _, r in top.iterrows():
            steam_tag = " 🔥STEAM" if r["steam"] else ""
            log.info(f"    [{r['signal']}]{steam_tag} {r['date']} | {r['league']} | "
                     f"{r['match']} | {r['market']} | "
                     f"move {r['move_pp']:+.2f} pp vs league threshold {r['threshold_pp']:.2f} pp "
                     f"(n={r['threshold_n']}) | {r['n_books']} books")

    log.info(f"  Tips saved → {SHARP_TIPS_FILE}")
    return tips


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")
    run()
