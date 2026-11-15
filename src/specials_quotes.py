"""Special-market quotes (correct score, high/low totals, a side scoring 4+) kept from /odds answers
the odds captures ALREADY make. Owner 2026-11-10: "start to collect correct score odds with the live
collection for all the leagues we cover".

    from src.specials_quotes import parse_specials, append_specials

ZERO EXTRA API CALLS. std_odds_capture / nf_odds_capture run with CAPTURE_BOOK_QUOTES=1, so every
/odds?fixture answer already carries every bookmaker and every bet. src/book_quotes keeps only the
modelled lines (O/U 1.5 / 2.5 / 3.5, BTTS, first-half O/U); everything else was thrown away. This module
keeps, from the SAME answer:

    Exact Score                    every score quoted ("h:a")
    Goals Over/Under               Over/Under 0.5, 4.5, 5.5, 6.5     (1.5/2.5/3.5 live in book_quotes)
    Total - Home / Total - Away    Over/Under 3.5                    (a side scoring 4+)

EXACT FULL-TIME BET NAMES. A bet is kept only when its name EQUALS one of the four names above. The
provider also offers "Exact Score - First Half", "Goals Over/Under First Half", "Goals Over/Under -
Second Half", "Total - Home First Half", ... and a substring match once let a first-half BTTS overwrite
the full-match one (scripts/capture_std_sidemarket_odds_forward._is_fullmatch_btts). Equality cannot.

REAL QUOTES ONLY. A row exists only when the provider returned that bookmaker, bet, selection and a
numeric decimal price above 1.0. Nothing is filled, carried forward or invented: a score a bookmaker did
not quote has no row. A quote is stored only BEFORE kickoff (the kickoff the provider's own answer
states, else the capture's): an in-play price is not a pre-match price, and a fixture whose kickoff is
unknown is not stored at all.

APPEND-ONLY SIDECAR. output/specials_quotes/<YYYY-MM>/<YYYY-MM-DD>_<producer>.csv, one file per day per
producing workflow (std, nf, prematch). Rows are only ever appended; a written row is never rewritten.
One file per producer because the std and nf workflows push to main independently with
`git pull --rebase -X theirs`: two writers on one file could drop each other's rows on a conflict.
Daily (inside a month folder) because a correct-score grid is ~25 scores x ~10 books per fixture, and a
monthly file would approach GitHub's 100 MB per-file limit.

WHAT IS APPENDED. Per (fixture, bookmaker, bet, selection, capture): the first sighting ("open"), every
change of price ("change"), and one observation per near rung T-1h / T-30m / T-10m when the price did
not move ("heartbeat", the same rule as src/book_quotes.HEARTBEAT_BANDS), so the last pre-kickoff row
can be certified as a close instead of a stale price.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from src.book_quotes import HEARTBEAT_BANDS, ladder_band, minutes_to_kickoff

SPECIALS_DIR = config.OUTPUT_DIR / "specials_quotes"

BET_EXACT, BET_OU, BET_HOME, BET_AWAY = "Exact Score", "Goals Over/Under", "Total - Home", "Total - Away"
#: Goals Over/Under lines kept here. 1.5 / 2.5 / 3.5 are NOT repeated: src/book_quotes keeps them
#: (MODELLED_LINES) from the same answer, every bookmaker.
OU_LINES = frozenset({"0.5", "4.5", "5.5", "6.5"})
TEAM_LINES = frozenset({"3.5"})
#: Which capture produced a row. prematch_* come from scripts/capture_prematch.py's odds slots.
CAPTURES = frozenset({"std", "nf", "prematch_75", "prematch_15"})
#: Producer = the workflow that writes the file (one writer per file).
PRODUCER = {"std": "std", "nf": "nf", "prematch_75": "prematch", "prematch_15": "prematch"}

COLUMNS = ["captured_at", "provider_update", "af_fixture_id", "fixture_key", "league", "match_date",
           "kickoff_utc", "home_team", "away_team", "bookmaker_id", "bookmaker", "bet_id", "bet",
           "selection", "odds", "capture", "minutes_to_kickoff", "ladder_band", "obs_reason"]
KEY = ["af_fixture_id", "bookmaker_id", "bet", "selection", "capture"]

_SCORE = re.compile(r"^\d{1,2}:\d{1,2}$")


def keep_selection(bet: str, value: str) -> bool:
    """True when (bet, selection) is one of the kept special quotes. `bet` must match EXACTLY."""
    if bet == BET_EXACT:
        return bool(_SCORE.match(value))
    parts = value.split(" ")
    if len(parts) != 2 or parts[0] not in ("Over", "Under"):
        return False
    if bet == BET_OU:
        return parts[1] in OU_LINES
    if bet in (BET_HOME, BET_AWAY):
        return parts[1] in TEAM_LINES
    return False


def _utc(ts) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(ts)
    except Exception:                                                 # noqa: BLE001
        return None
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(t: pd.Timestamp | None) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t is not None else ""


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def fixture_key(league: str, match_date: str, home: str, away: str) -> str:
    """v9's fixture identity as the odds archives write it: league | kickoff date (UTC) | home | away,
    with the provider's club names exactly as the capture recorded them (the archives' `match` column is
    "home vs away" from the same names)."""
    return f"{league}|{match_date}|{home}|{away}"


def parse_specials(payload: dict, *, capture: str, af_fixture_id, league: str, kickoff_utc: str,
                   home: str, away: str, captured_at: datetime | str) -> list[dict]:
    """Kept special quotes of one /odds?fixture answer, one row per (bookmaker, bet, selection).

    `kickoff_utc` is the capture's kickoff; the answer's own fixture timestamp takes precedence when
    present. No kickoff, or a capture at/after kickoff -> no rows.
    """
    if capture not in CAPTURES:
        raise ValueError(f"unknown capture {capture!r}; expected one of {sorted(CAPTURES)}")
    cap = _utc(captured_at)
    if cap is None:
        return []
    rows: list[dict] = []
    for entry in (payload or {}).get("response") or []:
        fx = entry.get("fixture") or {}
        ko = None
        if fx.get("timestamp") is not None:
            try:
                ko = pd.Timestamp(int(fx["timestamp"]), unit="s", tz="UTC")
            except (TypeError, ValueError):
                ko = None
        if ko is None:
            ko = _utc(fx.get("date")) if fx.get("date") else None
        if ko is None:
            ko = _utc(kickoff_utc) if kickoff_utc else None
        if ko is None or cap >= ko:
            continue                                  # unknown kickoff, or not a pre-match price
        fid = _int_or_none(fx.get("id")) or _int_or_none(af_fixture_id)
        if fid is None or (_int_or_none(af_fixture_id) is not None and fid != _int_or_none(af_fixture_id)):
            continue                                  # an answer for some other fixture
        upd = _utc(entry.get("update")) if entry.get("update") else None
        ko_iso, cap_iso = _iso(ko), _iso(cap)
        mdate = ko_iso[:10]
        mins = minutes_to_kickoff(ko_iso, cap_iso)
        band = ladder_band(mins)
        for bk in entry.get("bookmakers") or []:
            bid = _int_or_none(bk.get("id"))
            bname = (bk.get("name") or "").strip()
            if bid is None:
                continue
            for bet in bk.get("bets") or []:
                name = bet.get("name")
                if name not in (BET_EXACT, BET_OU, BET_HOME, BET_AWAY):
                    continue
                for v in bet.get("values") or []:
                    sel = str(v.get("value") if v.get("value") is not None else "").strip()
                    if not keep_selection(name, sel):
                        continue
                    try:
                        odds = float(v.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(odds) or odds <= 1.0:
                        continue                      # not a decimal price
                    rows.append({
                        "captured_at": cap_iso, "provider_update": _iso(upd),
                        "af_fixture_id": fid,
                        "fixture_key": fixture_key(league, mdate, home, away),
                        "league": league, "match_date": mdate, "kickoff_utc": ko_iso,
                        "home_team": home, "away_team": away,
                        "bookmaker_id": bid, "bookmaker": bname,
                        "bet_id": _int_or_none(bet.get("id")), "bet": name, "selection": sel,
                        "odds": odds, "capture": capture,
                        "minutes_to_kickoff": round(mins, 1) if mins is not None else None,
                        "ladder_band": band,
                    })
    return rows


def part_path(captured_at: str, producer: str, base: Path | None = None) -> Path:
    t = _utc(captured_at)
    if t is None:
        raise ValueError(f"unparseable captured_at {captured_at!r}")
    base = Path(base) if base is not None else SPECIALS_DIR
    return base / f"{t.year:04d}-{t.month:02d}" / f"{t.year:04d}-{t.month:02d}-{t.day:02d}_{producer}.csv"


def producer_files(producer: str, base: Path | None = None) -> list[Path]:
    base = Path(base) if base is not None else SPECIALS_DIR
    if not base.exists():
        return []
    return sorted(base.glob(f"*/*_{producer}.csv"), key=lambda p: p.name)


def _baseline(target: Path, producer: str, base: Path | None) -> pd.DataFrame:
    """This producer's target day file and the day file before it: enough to know the last stored
    price of every quote still being captured (a pass looks at most ~a week ahead, and a far quote
    that has not moved for over a day is written again once, which is harmless)."""
    files = [p for p in producer_files(producer, base) if p.name <= target.name][-2:]
    frames = [pd.read_csv(p, low_memory=False) for p in files if p.exists()]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def _k(df: pd.DataFrame) -> pd.Series:
    return (df["af_fixture_id"].astype("int64").astype(str) + "|" + df["bookmaker_id"].astype("int64").astype(str)
            + "|" + df["bet"].astype(str) + "|" + df["selection"].astype(str) + "|" + df["capture"].astype(str))


def append_specials(rows: list[dict], base: Path | None = None) -> int:
    """Append rows (open / change / near-rung heartbeat) to each producer's day file. Returns rows written."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS]
    bad = set(df["capture"]) - CAPTURES
    if bad:
        raise ValueError(f"unknown capture(s) {sorted(bad)}")
    df["_producer"] = df["capture"].map(PRODUCER)
    df["_part"] = [part_path(c, p, base) for c, p in zip(df["captured_at"], df["_producer"])]
    written = 0
    for (part, producer), chunk in df.groupby(["_part", "_producer"], sort=True):
        written += _append_one(chunk.drop(columns=["_part", "_producer"]), Path(part), producer, base)
    return written


def _append_one(df: pd.DataFrame, path: Path, producer: str, base: Path | None) -> int:
    df = df.sort_values("captured_at", kind="stable").copy()
    df["_k"] = _k(df)
    # one row per key per pass: the same answer cannot quote one selection twice for one book
    df = df.drop_duplicates(["_k", "captured_at"], keep="last")
    old = _baseline(path, producer, base)
    last_odds: dict = {}
    seen_bands: set = set()
    if len(old):
        old = old.sort_values("captured_at", kind="stable")
        ok = _k(old)
        last_odds = dict(zip(ok, old["odds"].astype(float)))
        seen_bands = set(zip(ok, old["ladder_band"].astype(str)))
    keep, reason = [], []
    for k, o, b in zip(df["_k"], df["odds"].astype(float), df["ladder_band"].astype(str)):
        prev = last_odds.get(k)
        if prev is None:
            keep.append(True); reason.append("open")
        elif not math.isclose(prev, o, rel_tol=0, abs_tol=1e-9):
            keep.append(True); reason.append("change")
        elif b in HEARTBEAT_BANDS and (k, b) not in seen_bands:
            keep.append(True); reason.append("heartbeat")
        else:
            keep.append(False); reason.append("")
        if keep[-1]:
            last_odds[k] = o
            seen_bands.add((k, b))
    df["obs_reason"] = reason
    out = df[pd.Series(keep, index=df.index)].drop(columns="_k")[COLUMNS]
    if out.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    out.to_csv(path, mode="w" if new_file else "a", header=new_file, index=False)
    return int(len(out))


def load(base: Path | None = None, producers=("std", "nf", "prematch")) -> pd.DataFrame:
    """Every stored special quote."""
    frames = []
    for p in producers:
        frames += [pd.read_csv(f, low_memory=False) for f in producer_files(p, base)]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
