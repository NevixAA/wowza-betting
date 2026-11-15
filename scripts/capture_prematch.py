"""
capture_prematch.py - PRE-KICKOFF capture of injuries, confirmed lineups and player-prop odds from
API-Football, each row stamped with the real fetch time (owner approval 2026-11-04; SWITCHED OFF
until the big data collection ends on 2026-11-08).

    python scripts/capture_prematch.py --dry-run [--days 28]   # call estimate, no call, no key
    python scripts/capture_prematch.py                         # ONE pass (the workflow loops it)

WHY. The props matchup research in wowzaV9-Pro (research/props-matchup, R_PROPS_MATCHUP_SPEC.md)
asks whether bookmakers fail to price key defender / goalkeeper absences in the player SoT and
anytime-goalscorer markets. That test needs data whose timing before kickoff is PROVEN: Pro measured
that 18.8% of a fixture's final injury list is added AFTER kickoff, so any list downloaded after
the match carries hindsight. On 2026-11-04 Pro's readiness gate measured 0% lineups with player ids
before kickoff, and props prices without fixture_id, player_id or kickoff time. This job closes
those gaps going forward; it cannot be backfilled (API-Football /odds is pre-match only).

WHAT ONE PASS DOES (API-Football only, through collect_common.AFClient)
  1. The fixture list: /fixtures?date=<UTC day> (ALL leagues in one call, no paging), kept for the
     registry leagues (src.league_registry api_football_id, the same 48 ids the collectors use) and
     status NS. Re-read at most once per LIST_REFRESH_MIN per date, so a postponement or a moved
     kickoff is seen within the hour (the stored competition_fixtures.parquet holds finished
     fixtures only, so it cannot be the live source; it is the dry-run's source instead).
  2. For every listed fixture, the SLOTS below that are due now and not yet in the journal:
        injuries_120  /injuries?fixture          due when 90 <= minutes before kickoff <= 120
        odds_75       /odds?fixture              55 .. 75   (before the team news)
        lineups_45    /fixtures/lineups?fixture  25 .. 45   (confirmed XI + subs with PLAYER IDS)
        odds_15       /odds?fixture              2 .. 15    (after the team news)
     A pass fetches a slot the first time it finds the fixture inside the window, so with the
     workflow's 5-minute passes the capture lands within ~5 minutes of the target. A fixture first
     seen after a window closed is counted 'missed' and never fetched for that slot.
  3. Player ids for the odds' player names, from the SAME fixture's captured lineup only
     (prop_player_ids, below). Nothing else is used as identity.

NO DOUBLE FETCH. The journal (table `slots`) holds one row per (fixture_id, endpoint, slot) that
returned an answer - an EMPTY answer included, so it is never asked again. A failed request (network,
HTTP, provider error) records nothing and is retried on the next pass while the window is open.
Within one workflow run the in-memory journal guards; across runs the stored one does (read back
over JOURNAL_DAYS). Data rows are written BEFORE the journal, so a crash between the two re-asks a
slot rather than losing it.

TIMING. captured_at_utc is stamped AFTER the provider's answer arrived (requested_at_utc before the
request) - the conservative end: whatever the answer says was known by then. minutes_before_kickoff
= (kickoff_utc - captured_at_utc) in minutes, kickoff_utc being the list's fixture.timestamp as known
at capture time. /odds and /injuries also carry the provider's fixture date, stored as
kickoff_utc_response so a moved kickoff is visible. Nothing is backdated, nothing is defaulted.

STORAGE (output/prematch/ only; reader: read_table()). One parquet CHUNK per table per month per
workflow run: <table>/<YYYY-MM>/<run_tag>.parquet (month of captured_at_utc). A run only appends to
its own chunk, a chunk past ROLL_BYTES rolls to _p2, _p3 ..., so no file nears GitHub's 100 MB limit
and no commit rewrites an older run's file. Every chunk has an explicit schema (an all-missing
column stays a typed null, so the month directory reads as one dataset). Raw provider answers are
kept for audit in table `raw` (compact JSON, zstd-compressed by parquet); for the date lists only the
registry-league items are kept (raw_scope says so), the rest of the world's fixtures are irrelevant.

BUDGET (its own, never the big collection's):
  * DAILY_CAP calls per UTC day, journaled in output/prematch/af_budget.json - NOT
    output/collect/af_budget.json, so data_truth_collect's own cap counts only its own calls and
    this job's cap only these;
  * FLOOR = collect_common.FLOOR - DAILY_CAP: this job stops when the account's remaining daily
    quota falls below it. The big job stops at collect_common.FLOOR, so when it has drained the
    account to its floor this job still runs (it never has to wait for, or race, the big job), and
    even a full day of this job leaves >= FLOOR - DAILY_CAP for normal production use (~23,300/day).
  Either limit raises BudgetStop; the pass saves what it has and exits 0.

SWITCH. A pass makes NO call unless the environment says PREMATCH_CAPTURE_ENABLED=true (the
workflow passes the repository variable of the same name). --dry-run never calls the API.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_af_lists as AL  # noqa: E402
import collect_common as CC  # noqa: E402
from collect_common import AFClient, BudgetStop  # noqa: E402

PREMATCH_DIR = ROOT / "output" / "prematch"
BUDGET_FILE = PREMATCH_DIR / "af_budget.json"
ENABLE_VAR = "PREMATCH_CAPTURE_ENABLED"

#: Calls per UTC day (hard). Measured 2026-11-04 by --dry-run on the stored registry fixture lists of
#: the 28 days 2026-10-07 .. 2026-11-03 (finished fixtures of the 48 registry ids): peak day 185
#: fixtures -> 4 x 185 + 27 list reads = 767 calls; mean 276, median 179, p90 688. 1,000 = the peak
#: x 1.3, so a busier Saturday (postponed fixtures are not in the finished lists, an odds answer may
#: be split into pages) still completes; reaching it is a BudgetStop, not a failure.
DAILY_CAP = 1_000
#: Stop below the shared account floor by this job's own cap (see BUDGET above).
FLOOR = CC.FLOOR - DAILY_CAP
#: Re-read a day's fixture list at most this often (minutes): a postponement / moved kickoff is seen
#: within the hour at 24 calls a day per date.
LIST_REFRESH_MIN = 60
#: Stored journals are read back over this many days (a slot is at most 2 h before a kickoff; a
#: fixture postponed by more than this would be captured again for its NEW kickoff, which is wanted).
JOURNAL_DAYS = 40
#: Chunk roll size (bytes on disk): far below GitHub's 100 MB hard limit.
ROLL_BYTES = 40_000_000
#: Workflow pass interval (seconds); used by the dry-run's list-read simulation only.
PASS_SECONDS = 300


@dataclass(frozen=True)
class Slot:
    name: str
    endpoint: str
    target: int        # minutes before kickoff the capture aims at (the window's upper edge)
    latest: int        # the window's lower edge: below it the slot is missed

    def due(self, mbk: float) -> bool:
        return self.latest <= mbk <= self.target


#: Owner approval 2026-11-04: injuries ~2 h, odds ~75 min (before the team news), lineups ~45 min,
#: odds ~15 min (after the team news). Windows do not overlap for the same endpoint.
SLOTS = (
    Slot("injuries_120", "injuries", 120, 90),
    Slot("odds_75", "odds", 75, 55),
    Slot("lineups_45", "fixtures/lineups", 45, 25),
    Slot("odds_15", "odds", 15, 2),
)
SLOT_BY_NAME = {s.name: s for s in SLOTS}
LIST_HORIZON_MIN = max(s.target for s in SLOTS) + 10

# -- tables -----------------------------------------------------------------------------------------
I, F, S, B = "int64", "float64", "string", "bool"
COMMON = [("fixture_id", I), ("league_id", I), ("league", S), ("season", S), ("kickoff_utc", S),
          ("slot", S), ("endpoint", S), ("captured_at_utc", S), ("minutes_before_kickoff", F)]
TABLES: dict[str, tuple[list[tuple[str, str]], list[str]]] = {
    "slots": (COMMON + [("home_id", I), ("away_id", I), ("requested_at_utc", S), ("status", S),
                        ("n_items", I), ("n_pages", I), ("pages_total", I),
                        ("kickoff_utc_response", S), ("run_tag", S)],
              ["fixture_id", "endpoint", "slot"]),
    "lists": ([("list_date", S), ("requested_at_utc", S), ("captured_at_utc", S),
               ("n_fixtures_all", I), ("n_registry", I), ("run_tag", S)],
              ["list_date", "captured_at_utc"]),
    "fixtures": ([("list_date", S), ("captured_at_utc", S), ("fixture_id", I), ("league_id", I),
                  ("league", S), ("season", S), ("kickoff_utc", S), ("status_short", S),
                  ("home_id", I), ("home_name", S), ("away_id", I), ("away_name", S)],
                 ["fixture_id", "captured_at_utc"]),
    "injuries": (COMMON + [("row", I), ("team_id", I), ("team_name", S), ("player_id", I),
                           ("player_name", S), ("type", S), ("reason", S)],
                 ["fixture_id", "slot", "row"]),
    "lineup_teams": (COMMON + [("team_id", I), ("team_name", S), ("formation", S), ("coach_id", I),
                               ("coach_name", S), ("n_start", I), ("n_subs", I)],
                     ["fixture_id", "slot", "team_id"]),
    "lineup_players": (COMMON + [("team_id", I), ("role", S), ("order", I), ("player_id", I),
                                 ("player_name", S), ("number", I), ("pos", S), ("grid", S)],
                       ["fixture_id", "slot", "team_id", "role", "order"]),
    "prop_odds": (COMMON + [("row", I), ("page", I), ("odds_update_utc", S), ("bookmaker_id", I),
                            ("bookmaker_name", S), ("bet_id", I), ("bet_name", S), ("bet_prefix", S),
                            ("market_v9", S), ("value_raw", S), ("odd_raw", S), ("odd", F),
                            ("player_name", S), ("line", F), ("side_word", S)],
                  ["fixture_id", "slot", "row"]),
    "odds_bets": (COMMON + [("bet_id", I), ("bet_name", S), ("is_player_prop", B),
                            ("n_bookmakers", I), ("bookmaker_ids", S), ("n_values", I)],
                  ["fixture_id", "slot", "bet_id", "bet_name"]),
    "prop_player_ids": ([("fixture_id", I), ("player_name", S), ("norm_name", S), ("player_id", I),
                         ("team_id", I), ("status", S), ("match_method", S), ("roster_slot", S),
                         ("roster_captured_at_utc", S), ("n_roster", I), ("mapped_at_utc", S)],
                        ["fixture_id", "player_name"]),
    "raw": ([("fixture_id", I), ("endpoint", S), ("slot", S), ("page", I), ("params_json", S),
             ("captured_at_utc", S), ("raw_scope", S), ("raw_json", S)],
            ["endpoint", "slot", "fixture_id", "page"]),
}
#: written LAST in a flush: they decide what is asked / mapped again
JOURNAL_TABLES = ("lists", "slots")

# -- player props --------------------------------------------------------------------------------
#: A bet is a PLAYER PROP when v9's API-Football prop reader already maps it
#: (player_model/prop_odds_af._BET_TO_MARKET) or its name says so. Every bet name seen is kept in
#: `odds_bets` with this flag, so an unclassified player market is visible, never silently dropped.
_PLAYER_PROP_RE = re.compile(r"\b(player|players|scorer|goalscorer|goalscorers)\b", re.IGNORECASE)


def _iso(ts: datetime | None) -> str | None:
    return None if ts is None else ts.astimezone(timezone.utc).isoformat(timespec="seconds")


def _tally(d: dict, key) -> None:
    """Count one event under `key` (a counter of what happened, never a data value)."""
    d[key] = d[key] + 1 if key in d else 1


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(v) -> datetime | None:
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if t is pd.NaT or pd.isna(t):
        return None
    return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).to_pydatetime()


def _from_epoch(v) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(v), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def enabled() -> bool:
    """The switch: only the exact string 'true' turns the capture on (owner 2026-11-04: off until the
    big collection ends 2026-11-08)."""
    return os.getenv(ENABLE_VAR, "").strip() == "true"


def registry_leagues() -> dict[int, str]:
    """{api_football_id: registry league name} for every registry league that has an id."""
    from src import league_registry as LR
    return {int(e["api_football_id"]): str(e["name"]) for e in LR.leagues()
            if e.get("api_football_id") is not None}


_TAG_RE = re.compile(r"^(\d{8}T\d{4}Z)[-\w]*$")


def run_tag(now: datetime | None = None) -> str:
    """Chunk name for this workflow run: PREMATCH_RUN_TAG (set once per run by the workflow,
    '<YYYYMMDDTHHMM>Z-<run id>'), else the pass start time. The leading timestamp lets a reader skip
    old chunks by name."""
    t = os.getenv("PREMATCH_RUN_TAG", "").strip()
    if t and _TAG_RE.match(t):
        return t
    return (now or _utcnow()).strftime("%Y%m%dT%H%MZ") + "-local"


# -- the store: chunked, schema-typed, append-only -------------------------------------------------

def _schema(table: str):
    import pyarrow as pa
    m = {I: pa.int64(), F: pa.float64(), S: pa.string(), B: pa.bool_()}
    return pa.schema([(c, m[t]) for c, t in TABLES[table][0]])


def _frame(table: str, rows: list[dict]) -> pd.DataFrame:
    cols = TABLES[table][0]
    df = pd.DataFrame(rows, columns=[c for c, _ in cols])
    for c, t in cols:
        if t == I:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif t == F:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        elif t == B:
            df[c] = df[c].astype("boolean")
        else:
            df[c] = df[c].astype("string")
    return df


def _write(path: Path, df: pd.DataFrame, table: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(df, schema=_schema(table), preserve_index=False)
    tmp = path.with_suffix(".tmp")
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, path)


def _chunk_paths(root: Path, table: str, month: str, tag: str) -> list[Path]:
    d = root / table / month
    base = d / f"{tag}.parquet"
    out = [base] if base.exists() else []
    out += sorted(d.glob(f"{tag}_p*.parquet"), key=lambda p: int(p.stem.rsplit("_p", 1)[1]))
    return out


def _append_chunk(root: Path, table: str, month: str, tag: str, rows: list[dict]) -> None:
    """Append rows to this run's chunk (the newest part); a row whose key is already in that part is
    not rewritten (the stored row wins). A part past ROLL_BYTES is left as it is and a new one starts."""
    if not rows:
        return
    parts = _chunk_paths(root, table, month, tag)
    if parts and parts[-1].stat().st_size >= ROLL_BYTES:
        path = root / table / month / f"{tag}_p{len(parts) + 1}.parquet"
    else:
        path = parts[-1] if parts else root / table / month / f"{tag}.parquet"
    new = _frame(table, rows)
    if path.exists():
        new = pd.concat([_frame(table, pd.read_parquet(path).to_dict("records")), new], ignore_index=True)
    new = new.drop_duplicates(TABLES[table][1], keep="first").reset_index(drop=True)
    _write(path, new, table)


def _month(ts: str | None) -> str:
    return str(ts or "")[:7] or "unknown"


def chunk_files(table: str, root: Path | None = None, since: datetime | None = None) -> list[Path]:
    """Every chunk of `table`, optionally only those whose run started on/after `since`'s day."""
    d = (root or PREMATCH_DIR) / table
    if not d.exists():
        return []
    files = sorted(d.glob("*/*.parquet"))
    if since is None:
        return files
    cut = (since - timedelta(days=1)).strftime("%Y%m%d")
    return [p for p in files if p.name[:8] >= cut or not p.name[:8].isdigit()]


def read_table(table: str, root: Path | None = None, since: datetime | None = None,
               columns: list[str] | None = None) -> pd.DataFrame:
    """The stored rows of `table` (every chunk, or the chunks of runs since `since`), de-duplicated
    on the table key with the FIRST stored row winning. Missing values stay missing."""
    cols = [c for c, _ in TABLES[table][0]]
    want = cols if columns is None else [c for c in cols if c in set(columns) | set(TABLES[table][1])]
    parts = [pd.read_parquet(p, columns=want) for p in chunk_files(table, root, since)]
    if not parts:
        return pd.DataFrame(columns=want)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(TABLES[table][1], keep="first").reset_index(drop=True)


class Store:
    """Buffers one pass's rows and writes them as this run's chunks (journals last)."""

    def __init__(self, root: Path, tag: str):
        self.root, self.tag = Path(root), tag
        self.buf: dict[str, list[dict]] = {t: [] for t in TABLES}

    def add(self, table: str, rows: list[dict]) -> None:
        self.buf[table].extend(rows)

    def flush(self) -> None:
        order = [t for t in TABLES if t not in JOURNAL_TABLES] + list(JOURNAL_TABLES)
        for t in order:
            rows, self.buf[t] = self.buf[t], []
            stamp = "mapped_at_utc" if t == "prop_player_ids" else "captured_at_utc"
            by_month: dict[str, list[dict]] = {}
            for r in rows:
                by_month.setdefault(_month(r.get(stamp)), []).append(r)
            for m, rs in sorted(by_month.items()):
                _append_chunk(self.root, t, m, self.tag, rs)


# -- parsers (provider answer -> rows; a field the provider does not give stays None) ---------------

def _common(fx: dict, slot: Slot, captured: datetime) -> dict:
    ko = fx["kickoff"]
    return {"fixture_id": fx["fixture_id"], "league_id": fx["league_id"], "league": fx.get("league"),
            "season": fx.get("season"), "kickoff_utc": _iso(ko), "slot": slot.name,
            "endpoint": slot.endpoint, "captured_at_utc": _iso(captured),
            "minutes_before_kickoff": round((ko - captured).total_seconds() / 60.0, 2)}


def parse_list(body: dict | None, leagues: dict[int, str], list_date: str, captured: datetime
               ) -> tuple[list[dict], list[dict]]:
    """/fixtures?date -> (registry fixture rows, their raw items)."""
    rows, items = [], []
    for it in (body or {}).get("response") or []:
        lg = it.get("league") or {}
        lid = AL.int_or_none(lg.get("id"))
        if lid is None or lid not in leagues:
            continue
        fx, teams = it.get("fixture") or {}, it.get("teams") or {}
        ko = _from_epoch(fx.get("timestamp")) or _parse_ts(fx.get("date"))
        rows.append({"list_date": list_date, "captured_at_utc": _iso(captured),
                     "fixture_id": AL.int_or_none(fx.get("id")), "league_id": lid,
                     "league": leagues[lid],
                     "season": None if lg.get("season") is None else str(lg.get("season")),
                     "kickoff_utc": _iso(ko), "status_short": (fx.get("status") or {}).get("short"),
                     "home_id": AL.int_or_none((teams.get("home") or {}).get("id")),
                     "home_name": (teams.get("home") or {}).get("name"),
                     "away_id": AL.int_or_none((teams.get("away") or {}).get("id")),
                     "away_name": (teams.get("away") or {}).get("name")})
        items.append(it)
    return rows, items


def _response_kickoff(body: dict | None) -> str | None:
    for it in (body or {}).get("response") or []:
        fx = it.get("fixture") or {}
        ko = _from_epoch(fx.get("timestamp")) or _parse_ts(fx.get("date"))
        if ko is not None:
            return _iso(ko)
    return None


def parse_injuries(body: dict | None, base: dict) -> list[dict]:
    out = []
    for i, it in enumerate((body or {}).get("response") or []):
        p, t = it.get("player") or {}, it.get("team") or {}
        out.append({**base, "row": i, "team_id": AL.int_or_none(t.get("id")), "team_name": t.get("name"),
                    "player_id": AL.int_or_none(p.get("id")), "player_name": p.get("name"),
                    "type": p.get("type"), "reason": p.get("reason")})
    return out


def parse_lineups(body: dict | None, base: dict) -> tuple[list[dict], list[dict]]:
    teams, players = [], []
    for t in (body or {}).get("response") or []:
        team = t.get("team") or {}
        tid = AL.int_or_none(team.get("id"))
        coach = t.get("coach") or {}
        start, subs = t.get("startXI") or [], t.get("substitutes") or []
        teams.append({**base, "team_id": tid, "team_name": team.get("name"),
                      "formation": t.get("formation"), "coach_id": AL.int_or_none(coach.get("id")),
                      "coach_name": coach.get("name"), "n_start": len(start), "n_subs": len(subs)})
        for role, lst in (("start", start), ("sub", subs)):
            for k, e in enumerate(lst):
                p = (e or {}).get("player") or {}
                players.append({**base, "team_id": tid, "role": role, "order": k,
                                "player_id": AL.int_or_none(p.get("id")), "player_name": p.get("name"),
                                "number": AL.int_or_none(p.get("number")), "pos": p.get("pos"),
                                "grid": p.get("grid")})
    return teams, players


def is_player_prop(bet_name) -> bool:
    from player_model import prop_odds_af as PA
    n = PA._norm_bet(bet_name)
    return n in PA._BET_TO_MARKET or bool(_PLAYER_PROP_RE.search(n))


def _bet_prefix(bet_name) -> str | None:
    s = str(bet_name or "").strip().lower()
    return next((p for p in ("home", "away") if s.startswith(p + " ")), None)


def parse_odds(body: dict | None, base: dict, page: int, row0: int = 0
               ) -> tuple[list[dict], dict[tuple, dict]]:
    """/odds?fixture -> (player-prop value rows, {(bet_id, bet_name): catalogue entry}). Every
    player-prop value the provider returns is kept with its raw strings; the parsed player name and
    line use v9's own readers (player_model/prop_odds_af), and stay None when they do not parse."""
    from player_model import prop_odds_af as PA
    rows, bets = [], {}
    k = row0
    for block in (body or {}).get("response") or []:
        upd = _parse_ts(block.get("update"))
        for bm in block.get("bookmakers") or []:
            bid = AL.int_or_none(bm.get("id"))
            for bet in bm.get("bets") or []:
                name = bet.get("name")
                prop = is_player_prop(name)
                key = (AL.int_or_none(bet.get("id")), name)
                cat = bets.setdefault(key, {"is_player_prop": prop, "books": set(), "n_values": 0})
                cat["books"].add(bid)
                vals = bet.get("values") or []
                cat["n_values"] += len(vals)
                if not prop:
                    continue
                mapped = PA._BET_TO_MARKET.get(PA._norm_bet(name))
                for v in vals:
                    raw, odd_raw = v.get("value"), v.get("odd")
                    player = PA._clean_player(raw) if raw is not None else ""
                    rows.append({**base, "row": k, "page": page, "odds_update_utc": _iso(upd),
                                 "bookmaker_id": bid, "bookmaker_name": bm.get("name"),
                                 "bet_id": key[0], "bet_name": name, "bet_prefix": _bet_prefix(name),
                                 "market_v9": mapped[0] if mapped else None,
                                 "value_raw": None if raw is None else str(raw),
                                 "odd_raw": None if odd_raw is None else str(odd_raw),
                                 "odd": PA._to_float(odd_raw), "player_name": player or None,
                                 "line": PA._parse_line(raw) if raw is not None else None,
                                 # every side is kept (each value is its own row); this NAMES
                                 # it: over / under / yes / no, None when the text names none
                                 "side_word": PA.side_word(raw) if raw is not None else None})
                    k += 1
    return rows, bets


# -- player id mapping: the fixture's own captured lineup is the only identity source --------------

def map_player_names(names: list[str], roster: list[dict]) -> dict[str, dict]:
    """{odds player name: mapping} against `roster` (the fixture's lineup rows: player_id, team_id,
    player_name). v9's rule (player_model/odds_fetcher.attach_player_ids, 2026-11-02):
      exact            the normalised name equals exactly ONE roster player;
      initial_surname  first initial + surname equals exactly one roster player AND exactly one
                       quoted name of this fixture.
    Anything else keeps player_id missing: 'ambiguous', 'unmatched', or 'roster_entry_without_id'
    (the one roster player the name matches has no provider id). There is no player alias store in
    v9 (src.team_names is clubs only), so no other route is proven."""
    from player_model.odds_fetcher import _initial_surname, _norm
    ros = [(AL.int_or_none(r.get("player_id")), AL.int_or_none(r.get("team_id")),
            _norm(str(r.get("player_name") or "")), _initial_surname(r.get("player_name")))
           for r in roster if r.get("player_name")]
    quoted: dict[tuple, set] = {}
    for q in names:
        ik = _initial_surname(q)
        if ik:
            quoted.setdefault(ik, set()).add(_norm(q))
    out = {}
    for name in names:
        n = _norm(name)
        res = {"norm_name": n, "player_id": None, "team_id": None, "status": "unmatched",
               "match_method": None}
        exact = {(p, t) for p, t, rn, _ in ros if rn == n}
        method, cands = None, set()
        if len(exact) == 1:
            method, cands = "exact", exact
        elif len(exact) > 1:
            res["status"] = "ambiguous"
        else:
            ik = _initial_surname(name)
            short = {(p, t) for p, t, _, rk in ros if ik and rk == ik}
            if len(short) == 1 and len(quoted.get(ik, ())) == 1:
                method, cands = "initial_surname", short
            elif short:
                res["status"] = "ambiguous"
        if method:
            pid, tid = next(iter(cands))
            if pid is None:
                res["status"] = "roster_entry_without_id"
            else:
                res.update(player_id=pid, team_id=tid, status="matched", match_method=method)
        out[name] = res
    return out


# -- planning --------------------------------------------------------------------------------------

def _latest_lists(lists: pd.DataFrame, fixtures: pd.DataFrame) -> dict[str, tuple[datetime, list[dict]]]:
    """{list_date: (captured_at, registry fixture rows of the newest read of that date)}."""
    out = {}
    if lists.empty:
        return out
    for d, g in lists.groupby("list_date"):
        cap = str(g["captured_at_utc"].max())
        rows = fixtures[(fixtures["list_date"] == d) & (fixtures["captured_at_utc"] == cap)] \
            if not fixtures.empty else fixtures
        out[str(d)] = (_parse_ts(cap), rows.to_dict("records") if len(rows) else [])
    return out


def due_slots(fixtures: list[dict], done: set[tuple], now: datetime) -> tuple[list[tuple[dict, Slot]], dict]:
    """(fixture, slot) pairs due at `now`, most urgent first, + counts of what is not due and why.
    Only fixtures with a known kickoff and provider status NS are planned."""
    plan, why = [], {}
    for fx in fixtures:
        if fx.get("kickoff") is None:
            _tally(why, "kickoff_unknown")
            continue
        if fx.get("status_short") != "NS":
            _tally(why, f"status_{fx.get('status_short')}")
            continue
        mbk = (fx["kickoff"] - now).total_seconds() / 60.0
        for s in SLOTS:
            if (str(fx["fixture_id"]), s.endpoint, s.name) in done:
                continue
            if s.due(mbk):
                plan.append((fx, s))
            elif mbk < s.latest:
                _tally(why, f"missed_{s.name}")
    plan.sort(key=lambda p: ((p[0]["kickoff"] - now).total_seconds() / 60.0 - p[1].latest,
                             p[0]["fixture_id"]))
    return plan, why


# -- one pass --------------------------------------------------------------------------------------

def _fixture_dict(r: dict) -> dict:
    return {"fixture_id": int(r["fixture_id"]), "league_id": int(r["league_id"]), "league": r.get("league"),
            "season": r.get("season"), "kickoff": _parse_ts(r.get("kickoff_utc")),
            "status_short": r.get("status_short"), "home_id": AL.int_or_none(r.get("home_id")),
            "away_id": AL.int_or_none(r.get("away_id")),
            "home_name": _na(r.get("home_name")), "away_name": _na(r.get("away_name"))}


def _na(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA else v


def run_pass(client: AFClient | None = None, root: Path | None = None, now_fn=_utcnow,
             leagues: dict[int, str] | None = None, tag: str | None = None,
             is_enabled: bool | None = None) -> dict:
    """One capture pass. Returns its stats. Makes no call while the switch is off."""
    stats = {"calls": 0, "slots_fetched": {}, "failed": 0, "not_due": {}, "fixtures_listed": 0,
             "specials_rows": 0}
    if not (enabled() if is_enabled is None else is_enabled):
        print(f"[prematch] switched off ({ENABLE_VAR} != 'true'): no call made")
        stats["switched_off"] = True
        return stats
    root = Path(root or PREMATCH_DIR)
    now = now_fn()
    tag = tag or run_tag(now)
    leagues = registry_leagues() if leagues is None else leagues
    client = client or AFClient(budget_file=root / "af_budget.json", daily_cap=DAILY_CAP, floor=FLOOR)
    store = Store(root, tag)
    since = now - timedelta(days=JOURNAL_DAYS)
    j = read_table("slots", root, since, ["fixture_id", "endpoint", "slot"])
    done = {(str(_na(a)), str(b), str(c)) for a, b, c in zip(j["fixture_id"], j["endpoint"], j["slot"])}
    try:
        lists = _latest_lists(read_table("lists", root, now - timedelta(days=2)),
                              read_table("fixtures", root, now - timedelta(days=2)))
        dates = sorted({now.strftime("%Y-%m-%d"),
                        (now + timedelta(minutes=LIST_HORIZON_MIN)).strftime("%Y-%m-%d")})
        fixtures: list[dict] = []
        for d in dates:
            have = lists.get(d)
            if have is None or have[0] is None or (now - have[0]) >= timedelta(minutes=LIST_REFRESH_MIN):
                req = now_fn()
                body = client.get("fixtures", {"date": d})
                cap = now_fn()
                if body is not None:
                    stats["calls"] += 1
                    rows, items = parse_list(body, leagues, d, cap)
                    store.add("fixtures", rows)
                    store.add("raw", [{"fixture_id": None, "endpoint": "fixtures", "slot": f"list:{d}:{_iso(cap)}",
                                       "page": 1, "params_json": json.dumps({"date": d}),
                                       "captured_at_utc": _iso(cap), "raw_scope": "registry_items",
                                       "raw_json": json.dumps({"get": body.get("get"),
                                                               "parameters": body.get("parameters"),
                                                               "results_all": body.get("results"),
                                                               "response": items},
                                                              separators=(",", ":"), ensure_ascii=False)}])
                    store.add("lists", [{"list_date": d, "requested_at_utc": _iso(req), "captured_at_utc": _iso(cap),
                                         "n_fixtures_all": len(body.get("response") or []),
                                         "n_registry": len(rows), "run_tag": tag}])
                    have = (cap, rows)
                elif have is None:
                    _tally(stats["not_due"], "list_unavailable")
                    continue
            fixtures += [_fixture_dict(r) for r in have[1] if _na(r.get("fixture_id")) is not None]
        stats["fixtures_listed"] = len(fixtures)
        plan, why = due_slots(fixtures, done, now)
        stats["not_due"].update(why)
        for fx, slot in plan:
            key = (str(fx["fixture_id"]), slot.endpoint, slot.name)
            if key in done:
                continue
            # the window is re-checked at call time: a long pass must not fetch a closed slot
            if not slot.due((fx["kickoff"] - now_fn()).total_seconds() / 60.0):
                _tally(stats["not_due"], f"missed_{slot.name}")
                continue
            if _capture(client, store, fx, slot, now_fn, tag, stats):
                done.add(key)
    except BudgetStop as e:
        print(f"[prematch] budget stop: {e} (clean exit; the next pass continues)")
        stats["budget_stop"] = str(e)
    finally:
        try:
            _map_players(store, root, now_fn(), stats)
        finally:
            store.flush()
            client.flush()
    print(f"[prematch] {json.dumps(stats, default=str, sort_keys=True)}")
    return stats


def _capture(client: AFClient, store: Store, fx: dict, slot: Slot, now_fn, tag: str, stats: dict) -> bool:
    params = {"fixture": int(fx["fixture_id"])}
    req = now_fn()
    body = client.get(slot.endpoint, params)
    cap = now_fn()
    if body is None:
        stats["failed"] += 1
        return False                          # nothing recorded: asked again while the window is open
    stats["calls"] += 1
    base = _common(fx, slot, cap)
    bodies = [(1, body)]
    pages_total, stop = None, None
    if slot.endpoint == "odds":
        pg = body.get("paging") or {}
        pages_total = AL.int_or_none(pg.get("total"))
        for p in range(2, (pages_total or 1) + 1):
            try:
                b = client.get(slot.endpoint, {**params, "page": p})
            except BudgetStop as e:           # store page 1.. and journal the slot, THEN stop
                stop = e
                break
            if b is None:
                break                         # the journal records the pages actually stored
            stats["calls"] += 1
            bodies.append((p, b))
    n_items = sum(len(b.get("response") or []) for _, b in bodies)
    if slot.endpoint == "injuries":
        store.add("injuries", parse_injuries(body, base))
    elif slot.endpoint == "fixtures/lineups":
        teams, players = parse_lineups(body, base)
        store.add("lineup_teams", teams)
        store.add("lineup_players", players)
    else:
        row0, cat = 0, {}
        for p, b in bodies:
            rows, bets = parse_odds(b, base, p, row0)
            row0 += len(rows)
            store.add("prop_odds", rows)
            for k, v in bets.items():
                c = cat.setdefault(k, {"is_player_prop": v["is_player_prop"], "books": set(), "n_values": 0})
                c["books"] |= v["books"]
                c["n_values"] += v["n_values"]
        store.add("odds_bets", [{**base, "bet_id": k[0], "bet_name": k[1], "is_player_prop": v["is_player_prop"],
                                 "n_bookmakers": len(v["books"]),
                                 "bookmaker_ids": ",".join(str(x) for x in sorted(b for b in v["books"] if b is not None)),
                                 "n_values": v["n_values"]} for k, v in sorted(cat.items(), key=lambda kv: str(kv[0]))])
        _store_specials(store, fx, slot, bodies, cap, stats)
    for p, b in bodies:
        store.add("raw", [{"fixture_id": fx["fixture_id"], "endpoint": slot.endpoint, "slot": slot.name,
                           "page": p, "params_json": json.dumps({**params, **({"page": p} if p > 1 else {})}),
                           "captured_at_utc": _iso(cap), "raw_scope": "full",
                           "raw_json": json.dumps(b, separators=(",", ":"), ensure_ascii=False)}])
    store.add("slots", [{**base, "home_id": fx.get("home_id"), "away_id": fx.get("away_id"),
                         "requested_at_utc": _iso(req), "status": "ok" if n_items else "empty",
                         "n_items": n_items, "n_pages": len(bodies), "pages_total": pages_total,
                         "kickoff_utc_response": _response_kickoff(body)
                         if slot.endpoint in ("odds", "injuries") else None, "run_tag": tag}])
    _tally(stats["slots_fetched"], slot.name)
    if stop is not None:
        raise stop
    return True


def _store_specials(store: Store, fx: dict, slot: Slot, bodies: list, cap: datetime, stats: dict) -> None:
    """Correct score / high-low totals / a side scoring 4+ from the SAME odds answer (src/specials_quotes,
    owner 2026-11-10): zero extra calls. Written beside output/prematch as output/specials_quotes
    (producer 'prematch', capture prematch_75 / prematch_15). Never takes the slot down with it."""
    try:
        from src import specials_quotes as SQ
        capture = "prematch_" + slot.name.split("_", 1)[1]
        rows = []
        for _p, b in bodies:
            rows += SQ.parse_specials(b, capture=capture, af_fixture_id=fx["fixture_id"],
                                      league=fx.get("league") or "", kickoff_utc=_iso(fx["kickoff"]) or "",
                                      home=fx.get("home_name") or "", away=fx.get("away_name") or "",
                                      captured_at=cap)
        n = SQ.append_specials(rows, base=store.root.parent / "specials_quotes")
        stats["specials_rows"] += n
    except Exception as e:                                            # noqa: BLE001
        print(f"  [specials_quotes] skipped fixture {fx.get('fixture_id')}: {e}")


def _map_players(store: Store, root: Path, now: datetime, stats: dict) -> None:
    """prop_player_ids for every odds player name of a fixture once BOTH are final: the quoted names
    (the last odds window has closed, so the initial+surname uniqueness test sees every name) and the
    roster (the lineups_45 answer; an empty or missed lineup slot = no roster, ids stay missing)."""
    since = now - timedelta(days=2)
    odds = pd.concat([read_table("prop_odds", root, since, ["fixture_id", "player_name", "kickoff_utc"]),
                      _frame("prop_odds", store.buf["prop_odds"])[["fixture_id", "player_name", "kickoff_utc", "slot", "row"]]
                      if store.buf["prop_odds"] else pd.DataFrame()], ignore_index=True)
    odds = odds.dropna(subset=["fixture_id", "player_name"])
    if odds.empty:
        return
    done = read_table("prop_player_ids", root, since, ["fixture_id", "player_name"])
    done_k = {(int(a), str(b)) for a, b in zip(done["fixture_id"], done["player_name"])}
    jl = pd.concat([read_table("slots", root, since), _frame("slots", store.buf["slots"])], ignore_index=True)
    jl = jl[jl["slot"] == "lineups_45"].drop_duplicates(["fixture_id"], keep="first")
    roster_slot = {int(r.fixture_id): r for r in jl.itertuples(index=False)}
    players = pd.concat([read_table("lineup_players", root, since),
                         _frame("lineup_players", store.buf["lineup_players"])], ignore_index=True)
    last = min(s.latest for s in SLOTS)
    rows, counts = [], {}
    for fid, g in odds.groupby("fixture_id"):
        fid = int(fid)
        names = sorted({str(x) for x in g["player_name"]})
        todo = [n for n in names if (fid, n) not in done_k]
        if not todo:
            continue
        ko = _parse_ts(g["kickoff_utc"].dropna().iloc[0]) if g["kickoff_utc"].notna().any() else None
        if ko is None or (ko - now).total_seconds() / 60.0 >= last:
            continue                          # a slot can still add names or the roster: wait
        js = roster_slot.get(fid)
        if js is None:
            roster, rslot, rcap = [], None, None
        else:
            roster = players[players["fixture_id"] == fid].to_dict("records")
            rslot, rcap = "lineups_45", js.captured_at_utc
        res = map_player_names(names, roster)
        for n in todo:
            m = dict(res[n])
            if not roster:
                m.update(status="no_roster", player_id=None, team_id=None, match_method=None)
            _tally(counts, m["status"])
            rows.append({"fixture_id": fid, "player_name": n, "roster_slot": rslot,
                         "roster_captured_at_utc": rcap, "n_roster": len(roster),
                         "mapped_at_utc": _iso(now), **m})
    store.add("prop_player_ids", rows)
    if counts:
        stats["player_ids"] = counts


# -- dry run ---------------------------------------------------------------------------------------

def list_calls_per_day(refresh_min: int = LIST_REFRESH_MIN, pass_s: int = PASS_SECONDS) -> int:
    """List reads in one UTC day of uninterrupted passes (a simulation of run_pass's refresh rule)."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    last: dict[str, datetime] = {}
    n = 0
    for k in range(int(86400 / pass_s)):
        now = t0 + timedelta(seconds=k * pass_s)
        for d in {now.strftime("%Y-%m-%d"), (now + timedelta(minutes=LIST_HORIZON_MIN)).strftime("%Y-%m-%d")}:
            if d not in last or now - last[d] >= timedelta(minutes=refresh_min):
                last[d] = now
                n += 1
    return n


def estimate(days: int = 28, end: str | None = None, fx: pd.DataFrame | None = None) -> dict:
    """Calls per day from REAL fixture counts: the stored registry fixture lists of the `days` full
    UTC days before `end` (default: today). Assumes every fixture gets all four slots (one odds page
    each) - the maximum; a fixture first seen late or postponed costs less."""
    rf = AL.registry_fixtures(fx)
    end_d = pd.Timestamp(end or _utcnow().strftime("%Y-%m-%d"))
    start_d = end_d - pd.Timedelta(days=days)
    dts = pd.to_datetime(rf["date"], errors="coerce")
    sel = rf[(dts >= start_d) & (dts < end_d)]
    per_day = sel.groupby(pd.to_datetime(sel["date"]).dt.strftime("%Y-%m-%d")).size()
    per_day = per_day.reindex([(start_d + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)],
                              fill_value=0)
    lc = list_calls_per_day()
    calls = per_day * len(SLOTS) + lc
    peak = int(calls.max()) if len(calls) else 0
    out = {"window": f"{start_d.date()} .. {(end_d - pd.Timedelta(days=1)).date()}",
           "registry_leagues": len(AL.registry_af_ids()), "days": days,
           "fixtures_per_day_mean": round(float(per_day.mean()), 1) if len(per_day) else None,
           "fixtures_per_day_max": int(per_day.max()) if len(per_day) else None,
           "calls_per_fixture": len(SLOTS), "list_calls_per_day": lc,
           "calls_per_day_mean": round(float(calls.mean()), 1) if len(calls) else None,
           "calls_per_day_median": float(calls.median()) if len(calls) else None,
           "calls_per_day_p90": float(calls.quantile(0.9)) if len(calls) else None,
           "calls_per_day_max": peak, "daily_cap": DAILY_CAP, "floor": FLOOR,
           "cap_over_peak": round(DAILY_CAP / peak, 2) if peak else None,
           "per_day": {k: int(v) for k, v in per_day.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pre-kickoff capture (API-Football): one pass")
    ap.add_argument("--dry-run", action="store_true", help="print the call estimate; no call")
    ap.add_argument("--days", type=int, default=28, help="dry-run: full days of fixture counts")
    ap.add_argument("--end", default=None, help="dry-run: first day NOT counted (YYYY-MM-DD)")
    a = ap.parse_args(argv)
    if a.dry_run:
        print(json.dumps(estimate(a.days, a.end), indent=1))
        return 0
    run_pass()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
