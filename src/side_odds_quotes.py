"""Corners and cards bookmaker quotes kept from /odds answers the odds captures ALREADY fetch
(owner 2026-11-16, audit gap G1 "Keep corners/cards odds"; reports/R_SIDE_DATA_TRUTH_LIVE_2026-11-16.md).

ZERO EXTRA API CALLS. v9's std/nf captures (CAPTURE_BOOK_QUOTES=1) and Pro's league scout (sweep and
near loop) receive every bookmaker and every bet in each /odds answer. Until now they dropped the corners
and cards bets (`_OU_EXCLUDE` in the v9 capture scripts). This module reads them from the SAME parsed
answer; it never makes a request.

PRIVATE STORE ONLY (licence). API-Football's terms (https://www.api-football.com/terms, "Last updated:
Mai 21, 2025") say: "We do not provide a "license" for the use and publication of the data provided by
our services ... Any license or permission to publish the data must be requested by the user from the
competent authorities", and "it is prohibited to resell this data to third parties". wowza-betting and
wowzaV9-Pro are PUBLIC, so these prices are NEVER written into them. The store is the private repo
NevixAA/wowza-exec, orphan branch `af-side-odds`, cloned by the workflow into $RUNNER_TEMP. Writing
needs `SIDE_ODDS_DIR` pointing at that clone, which must carry the marker file `.side_odds_store` and
must not lie inside this code checkout. Anything else -> nothing is written (no fallback location).
Logs print counts only, never a price.

KEPT BETS. Read from the provider on 2026-11-16 (`/odds/bets` catalogue, and /odds answers for the
Championship, Premier League and Bundesliga). A bet is kept only when its numeric id AND its exact name
both match; a rename or a re-used id is dropped and reported by name, never guessed:

    id 45   Corners Over Under          match corners      (seen in the answers: 3 books)
    id 57   Home Corners Over/Under     home corners       (seen)
    id 58   Away Corners Over/Under     away corners       (seen)
    id 80   Cards Over/Under            match cards        (catalogue; no book quoted it on 11-16)
    id 82   Home Team Total Cards       home cards         (catalogue)
    id 83   Away Team Total Cards       away cards         (catalogue)
    id 150  Home Team Yellow Cards      home yellow cards  (catalogue)
    id 151  Away Team Yellow Cards      away yellow cards  (catalogue)
    id 335  Red Cards Over/Under        match red cards    (catalogue)

Full-time over/under only. First/second-half, handicap, 1x2, race-to, range and time-window bets are not
kept. A selection is kept only when it reads exactly "Over <line>" or "Under <line>".

REAL QUOTES ONLY. A row exists only when the provider returned that bookmaker, bet, selection and a
numeric decimal price above 1.0, observed STRICTLY BEFORE the kickoff the answer states (else the
capture's own kickoff). Unknown kickoff or fixture -> no row. Club names come from the capture; when it
has none they stay missing. Nothing is filled, carried forward or invented.

APPEND-ONLY. <store>/data/<YYYY-MM>/<YYYY-MM-DD>_<producer>[_pN].csv, one file per day per producer
(std, nf, sweep, near: one writer per file, since each workflow pushes independently). Per (fixture,
bookmaker, bet, selection, capture) a row is appended at first sight ("open"), at every price change
("change"), and once per near rung T-1h / T-30m / T-10m when it did not move ("heartbeat"), the
src/specials_quotes rule. Every row carries snapshot_ts (UTC). A written row is never rewritten.

THIS FILE IS SHARED VERBATIM. wowza-betting src/side_odds_quotes.py == wowzaV9-Pro src/scout/side_odds.py
(byte-identical; Pro's tests check it when V9_LOCAL is checked out). It imports nothing from either repo.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

#: bet id -> (exact provider name, family, team)
BETS = {
    45: ("Corners Over Under", "corners", "match"),
    57: ("Home Corners Over/Under", "corners", "home"),
    58: ("Away Corners Over/Under", "corners", "away"),
    80: ("Cards Over/Under", "cards", "match"),
    82: ("Home Team Total Cards", "cards", "home"),
    83: ("Away Team Total Cards", "cards", "away"),
    150: ("Home Team Yellow Cards", "yellow_cards", "home"),
    151: ("Away Team Yellow Cards", "yellow_cards", "away"),
    335: ("Red Cards Over/Under", "red_cards", "match"),
}
LADDER = [("T-10m", 10), ("T-30m", 30), ("T-1h", 60), ("T-3h", 180), ("T-6h", 360)]
HEARTBEAT_BANDS = frozenset({"T-1h", "T-30m", "T-10m"})
CAPTURES = frozenset({"std", "nf", "scout_sweep", "scout_near"})
PRODUCER = {"std": "std", "nf": "nf", "scout_sweep": "sweep", "scout_near": "near"}
MARKER = ".side_odds_store"
ENV = "SIDE_ODDS_DIR"
MAX_PART_MB = 80

COLUMNS = ["snapshot_ts", "provider_update", "af_fixture_id", "league_id", "league", "country",
           "match_date", "kickoff_utc", "home_team", "away_team", "bookmaker_id", "bookmaker", "bet_id",
           "bet", "market", "team", "line", "side", "selection", "odds", "capture",
           "minutes_to_kickoff", "ladder_band", "obs_reason"]

_SEL = re.compile(r"^(Over|Under) (\d{1,3}(?:\.\d{1,2})?)$")
_SIDE_WORDS = re.compile(r"corner|card|booking", re.I)

#: corners/cards-looking bets seen but NOT kept, {(id, name): count}, for the run log (names only)
UNKEPT: Counter = Counter()


def market_of(bet_id) -> str | None:
    b = BETS.get(bet_id)
    return f"{b[1]}_ou_{b[2]}" if b else None


def ladder_band(minutes) -> str:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "UNKNOWN"
    if minutes < 0:
        return "POST"
    for name, edge in LADDER:
        if minutes <= edge:
            return name
    return "FAR"


def _utc(ts) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(ts)
    except Exception:                                                 # noqa: BLE001
        return None
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(t) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t is not None else ""


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _kickoff(fx: dict, fallback) -> pd.Timestamp | None:
    if fx.get("timestamp") is not None:
        try:
            return pd.Timestamp(int(fx["timestamp"]), unit="s", tz="UTC")
        except (TypeError, ValueError):
            pass
    if fx.get("date"):
        t = _utc(fx.get("date"))
        if t is not None:
            return t
    return _utc(fallback) if fallback else None


def parse(resp, *, capture: str, captured_at, names: dict | None = None, leagues: dict | None = None,
          kickoffs: dict | None = None, expect_fixture_id=None) -> list[dict]:
    """Kept corners/cards quotes of one /odds answer (its `response` list, any number of fixtures).

    names:    {fixture_id: (home, away)} from the capture; a fixture not in it keeps missing names.
    leagues:  {league_id: (league name, country)}; else the answer's own league block. A None country
              is taken from the answer's league block.
    kickoffs: {fixture_id: kickoff ISO} used only when the answer states no kickoff.
    expect_fixture_id: an /odds?fixture answer must be for this fixture; any other entry is dropped.
    """
    if capture not in CAPTURES:
        raise ValueError(f"unknown capture {capture!r}; expected one of {sorted(CAPTURES)}")
    cap = _utc(captured_at)
    if cap is None:
        return []
    names, leagues, kickoffs = names or {}, leagues or {}, kickoffs or {}
    want = _int(expect_fixture_id)
    rows: list[dict] = []
    for entry in resp or []:
        fx = entry.get("fixture") or {}
        fid = _int(fx.get("id"))
        if fid is None or (want is not None and fid != want):
            continue
        ko = _kickoff(fx, kickoffs.get(fid))
        if ko is None or cap >= ko:
            continue                                  # unknown kickoff, or not a pre-match price
        lg = entry.get("league") or {}
        lid = _int(lg.get("id"))
        lname, country = leagues.get(lid, (lg.get("name"), lg.get("country")))
        if country is None:
            country = lg.get("country")
        home, away = (x.strip() if isinstance(x, str) else x for x in names.get(fid, (None, None)))
        upd = _utc(entry.get("update")) if entry.get("update") else None
        ko_iso, cap_iso = _iso(ko), _iso(cap)
        mins = (ko - cap).total_seconds() / 60.0
        band = ladder_band(mins)
        for bk in entry.get("bookmakers") or []:
            bid = _int(bk.get("id"))
            if bid is None:
                continue
            bname = (bk.get("name") or "").strip()
            for bet in bk.get("bets") or []:
                bet_id, name = _int(bet.get("id")), bet.get("name")
                spec = BETS.get(bet_id)
                if spec is None or spec[0] != name:
                    if isinstance(name, str) and _SIDE_WORDS.search(name):
                        UNKEPT[(bet_id, name)] += 1
                    continue
                for v in bet.get("values") or []:
                    sel = str(v.get("value") if v.get("value") is not None else "").strip()
                    m = _SEL.match(sel)
                    if not m:
                        continue
                    try:
                        odds = float(v.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(odds) or odds <= 1.0:
                        continue                      # not a decimal price
                    rows.append({
                        "snapshot_ts": cap_iso, "provider_update": _iso(upd) or None,
                        "af_fixture_id": fid, "league_id": lid, "league": lname, "country": country,
                        "match_date": ko_iso[:10], "kickoff_utc": ko_iso,
                        "home_team": home, "away_team": away,
                        "bookmaker_id": bid, "bookmaker": bname, "bet_id": bet_id, "bet": name,
                        "market": market_of(bet_id), "team": spec[2], "line": float(m.group(2)),
                        "side": m.group(1).lower(), "selection": sel, "odds": odds, "capture": capture,
                        "minutes_to_kickoff": round(mins, 1), "ladder_band": band})
    return rows


# ── private store ──────────────────────────────────────────────────────────────────────────────

class StoreRefused(RuntimeError):
    """The target is not the private side-odds store: nothing is written."""


def store_dir(base=None) -> Path | None:
    """The private store root, or None when the store is not configured (the feature is off).
    Raises StoreRefused when configured but unsafe (no marker, or inside this code checkout)."""
    raw = base if base is not None else os.environ.get(ENV, "").strip()
    if not raw:
        return None
    root = Path(raw).resolve()
    if not (root / MARKER).is_file():
        raise StoreRefused(f"{ENV} has no {MARKER} marker: not the private store")
    code = Path(__file__).resolve().parent
    for parent in [code, *code.parents]:
        if (parent / ".git").exists():
            if root == parent or parent in root.parents:
                raise StoreRefused(f"{ENV} lies inside the public code checkout")
            break
    return root


def _files(base: Path, producer: str) -> list[Path]:
    d = base / "data"
    if not d.exists():
        return []

    def order(p: Path):
        m = re.search(r"_p(\d+)$", p.stem)
        return (p.name[:10], int(m.group(1)) if m else 1)
    return sorted((p for p in d.glob(f"*/*_{producer}*.csv")
                   if re.fullmatch(rf"\d{{4}}-\d{{2}}-\d{{2}}_{producer}(_p\d+)?", p.stem)), key=order)


def _target(base: Path, day: str, producer: str) -> Path:
    month = base / "data" / day[:7]
    parts = [p for p in _files(base, producer) if p.name[:10] == day]
    p = parts[-1] if parts else month / f"{day}_{producer}.csv"
    if p.exists() and p.stat().st_size / 1e6 >= MAX_PART_MB:
        p = month / f"{day}_{producer}_p{len(parts) + 1}.csv"
    return p


def _k(df: pd.DataFrame) -> pd.Series:
    return (df["af_fixture_id"].astype("int64").astype(str) + "|"
            + df["bookmaker_id"].astype("int64").astype(str) + "|" + df["bet_id"].astype("int64").astype(str)
            + "|" + df["selection"].astype(str) + "|" + df["capture"].astype(str))


#: per-process memory of what each producer stored: {(base, producer): (day, last_odds, seen_bands)}
_STATE: dict = {}


def _state(base: Path, producer: str, day: str):
    key = (str(base), producer)
    st = _STATE.get(key)
    if st is not None and st[0] == day:
        return st
    files = _files(base, producer)
    days = sorted({p.name[:10] for p in files if p.name[:10] <= day})[-2:]
    frames = [pd.read_csv(p, low_memory=False) for p in files if p.name[:10] in days]
    frames = [f for f in frames if len(f)]
    last, seen = {}, set()
    if frames:
        old = pd.concat(frames, ignore_index=True).sort_values("snapshot_ts", kind="stable")
        ok = _k(old)
        last = dict(zip(ok, old["odds"].astype(float)))
        seen = set(zip(ok, old["ladder_band"].astype(str)))
    st = (day, last, seen)
    _STATE[key] = st
    return st


def append(rows: list[dict], base=None, dry: bool = False) -> int:
    """Append open / change / near-rung heartbeat rows to the private store. Returns rows kept.
    Store not configured -> 0 and nothing written."""
    if not rows:
        return 0
    root = store_dir(base)
    if root is None:
        return 0
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    bad = set(df["capture"]) - CAPTURES
    if bad:
        raise ValueError(f"unknown capture(s) {sorted(bad)}")
    df["_day"] = df["snapshot_ts"].astype(str).str[:10]
    df["_prod"] = df["capture"].map(PRODUCER)
    total = 0
    for (day, prod), chunk in df.groupby(["_day", "_prod"], sort=True):
        total += _append_one(chunk.drop(columns=["_day", "_prod"]), root, day, prod, dry)
    return total


def _append_one(df: pd.DataFrame, base: Path, day: str, producer: str, dry: bool) -> int:
    df = df.sort_values("snapshot_ts", kind="stable").copy()
    df["_k"] = _k(df)
    df = df.drop_duplicates(["_k", "snapshot_ts"], keep="last")
    _, last, seen = _state(base, producer, day)
    if dry:
        last, seen = dict(last), set(seen)
    keep, reason = [], []
    for k, o, b in zip(df["_k"], df["odds"].astype(float), df["ladder_band"].astype(str)):
        prev = last.get(k)
        if prev is None:
            r = "open"
        elif not math.isclose(prev, o, rel_tol=0, abs_tol=1e-9):
            r = "change"
        elif b in HEARTBEAT_BANDS and (k, b) not in seen:
            r = "heartbeat"
        else:
            r = ""
        keep.append(bool(r))
        reason.append(r)
        if r:
            last[k] = o
            seen.add((k, b))
    df["obs_reason"] = reason
    out = df[pd.Series(keep, index=df.index)].drop(columns="_k")[COLUMNS]
    if out.empty or dry:
        return int(len(out))
    p = _target(base, day, producer)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    out.to_csv(p, mode="w" if new else "a", header=new, index=False)
    return int(len(out))


def summary(rows: list[dict], kept: int) -> str:
    """One log line: counts only, never a price."""
    by_m = Counter(r["market"] for r in rows)
    fx = len({r["af_fixture_id"] for r in rows})
    books = len({r["bookmaker_id"] for r in rows})
    ms = ", ".join(f"{m}={n}" for m, n in sorted(by_m.items())) or "none"
    unk = ", ".join(f"{i}:{n}" for (i, n), _ in UNKEPT.most_common(6))
    return (f"[side_odds] +{kept:,} row(s) kept from {len(rows):,} parsed; {fx} fixture(s), {books} "
            f"bookmaker(s); markets {ms}; 0 extra API calls"
            + (f"; not kept (names only): {unk}" if unk else ""))
