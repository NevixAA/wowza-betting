"""§10 — per-bookmaker quotes on a kickoff-relative ladder.

    from src.book_quotes import parse_books, ladder_band, append_quotes

WHAT IS WRONG TODAY. The side-market capture asks API-Football for ONE bookmaker
(`params={"fixture": fid, "bookmaker": 8}`) and stores one price per market. Everything needed
for market microstructure — who moved first, how wide the books are, what the best executable
price was — is discarded at the request, before it is ever seen.

Dropping that filter costs **the same number of API calls**. The response simply carries every
book instead of one. The only extra cost is parsing and disk.

AND THE CLOSE IS NOT THE LAST SNAPSHOT. CLV is currently computed against whatever row happened
to be written last, which may be six hours before kickoff. A price six hours out is not a
closing line, and a "CLV" measured against it is measuring something else. `ladder_band` tags
every quote by how long before kickoff it was taken, so a close can be defined as the last quote
inside T-30m and a row that never got near kickoff can be excluded rather than silently treated
as one.

THE LADDER. T-6h, T-3h, T-1h, T-30m, T-10m, as §10 specifies, plus FAR for anything earlier and
POST for anything after kickoff (which must never be used as a close — a price after kick-off is
an in-play price, and treating one as a closing line manufactures enormous fake CLV).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

#: A DIRECTORY of one CSV per calendar month, not a single file.
#:
#: WHY, BEFORE IT IS A PROBLEM. The existing one-bookmaker archive is 4.4 MB / 42,699 rows over
#: 46 days. This file keeps NINE books across four markets, so it accrues roughly an order of
#: magnitude faster — about 1 MB/day, which reaches GitHub's hard 100 MB per-file limit around
#: mid-January. That limit rejects the push outright (GH001) and is identical on Free, Pro, Team
#: and Enterprise; `player_history.parquet` already hit it once and had to be split after the
#: fact, with `.git` at 3.4 GB by then. Monthly parts cost nothing now and make that impossible.
#:
#: A finished month is never rewritten, so git stores each part once.
QUOTES_DIR = config.OUTPUT_DIR / "book_quotes"
#: Pre-partition single file. Still READ if present so no captured row is orphaned; never written.
LEGACY_FILE = config.OUTPUT_DIR / "book_quotes.csv"
#: Per-part size guard, mirroring the player-history guard. A monthly part should land near
#: 30 MB; 70 means the growth model is wrong and 100 is the limit that rejects the push.
WARN_MB, MAX_MB = 70, 100

#: §10's required fields. `line` is separate from `market` so Over 2.5 and Over 3.5 are the same
#: market at different lines rather than two unrelated strings — which is what lets a future
#: study ask about the line itself.
COLUMNS = [
    "snapshot_ts", "fixture_id", "kickoff_utc", "minutes_to_kickoff", "ladder_band",
    "league", "model_type", "home_team", "away_team",
    "bookmaker_id", "bookmaker", "market", "side", "line", "odds", "source",
    # §6. WHY a row exists: "change" = the price moved, "heartbeat" = the price did NOT move but
    # this rung was observed. Without this column the two are indistinguishable and the archive
    # cannot certify observation, only movement.
    "obs_reason",
]

#: Upper edge of each band, in minutes before kickoff. Ordered tightest-first.
LADDER = [("T-10m", 10), ("T-30m", 30), ("T-1h", 60), ("T-3h", 180), ("T-6h", 360)]

#: A quote inside this window is eligible to be the closing line.
CLOSING_WINDOW_MIN = 30

#: §6 — rungs that get an OBSERVATION HEARTBEAT as well as change events.
#:
#: THE PROBLEM THIS SOLVES. Storing only consecutive-distinct prices is right for movement and
#: wrong for coverage. A book quoting 1.90 at T-3h and still 1.90 at T-10m writes ONE row, at
#: T-3h, so the archive cannot distinguish "the price never moved" from "we never looked again".
#: A CLV measured against that row is measured against a three-hour-old price while appearing to
#: be a close.
#:
#: WHY ONLY THE NEAR RUNGS. A heartbeat per (fixture, book, market, side, line, band) multiplies
#: storage by the number of bands. With ~9 books and ~15 market/side/line combinations that is
#: ~135 keys per fixture per band; across all seven bands it would dominate the file and push a
#: monthly part past the 100 MB limit the partitioning exists to avoid. The far rungs do not need
#: it -- nothing is certified against a T-6h price -- so they keep change-only semantics, and the
#: three rungs where proof-of-close actually matters get the heartbeat.
HEARTBEAT_BANDS = {"T-1h", "T-30m", "T-10m"}

#: Bands kept as a SINGLE first observation per entity, never change-tracked.
#:
#: MEASURED 2026-10-04 on the first real CI runs: 97.8% of captured rows were FAR (more than six
#: hours out), and nothing is ever certified against a FAR price. What FAR is needed for is the
#: OPENING line, which is one row per entity, not a running log of slow drift across a week.
#: Tracking FAR changes is what turned a projected 1 MB/day into a measured 142 MB/day.
OPEN_ONLY_BANDS = {"FAR"}


def ladder_band(minutes_to_kickoff: float | None) -> str:
    """Which rung of the ladder a quote sits on.

    POST is deliberately its own band and never folded into T-10m: a price taken after kickoff
    is an in-play price. Treating one as a close would produce large positive CLV out of nothing,
    which is the most flattering possible error and therefore the one most likely to be believed.
    """
    if minutes_to_kickoff is None or (isinstance(minutes_to_kickoff, float)
                                      and np.isnan(minutes_to_kickoff)):
        return "UNKNOWN"
    if minutes_to_kickoff < 0:
        return "POST"
    for name, edge in LADDER:
        if minutes_to_kickoff <= edge:
            return name
    return "FAR"


def minutes_to_kickoff(kickoff_utc, snapshot_ts) -> float | None:
    try:
        ko = pd.Timestamp(kickoff_utc)
        ts = pd.Timestamp(snapshot_ts)
        if ko.tzinfo is None:
            ko = ko.tz_localize("UTC")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return float((ko - ts).total_seconds() / 60.0)
    except Exception:                                                 # noqa: BLE001
        return None


# ── parsing ───────────────────────────────────────────────────────────────────────────────────
#: BET IDS, NOT NAMES. API-Football returns several bets whose NAME contains "over/under" and
#: which are not full-match goals:
#:
#:     id 5    Goals Over/Under                  <- the one we want
#:     id 6    Goals Over/Under First Half       <- first-half goals
#:     id 26   Goals Over/Under - Second Half    <- NOT full match
#:     id 57   Home Corners Over/Under           <- not goals at all
#:     id 58   Away Corners Over/Under
#:     id 197  Over/Under 15m-30m                <- a time window
#:     id 198  Over/Under 30m-45m
#:
#: A first version matched on the name containing "over/under" and swallowed all seven. On a
#: live fixture that produced "O/U 2.5" quotes ranging 1.25-3.50 and a cross-book anchor of
#: p=0.254 where the real market was ~0.63. The capture script already learned this the hard
#: way -- 21.9% of its archive once had over25 >= over35, which is impossible -- and its own
#: comment says the numeric id is the primary test "because it cannot be broken by a rename".
BET_IDS = {5: "ou", 6: "ht_ou", 8: "btts"}

#: Lines we actually model. MEASURED 2026-10-04: the API returns THIRTY distinct O/U lines
#: (0.5 through 6.5 in quarter steps) and we model three. Keeping all of them made 72% of the
#: archive markets nothing reads -- 40 MB/day against a 100 MB per-file limit.
#:
#: h2h was dropped from BET_IDS for the same reason: 1X2 is a research track with no bet and no
#: consumer, and it was 9% of rows. Data no model reads is cost, not value.
MODELLED_LINES = {"ou": {1.5, 2.5, 3.5}, "ht_ou": {0.5, 1.5}}


def _classify(bet_id, bet_name: str, value: str) -> tuple[str, str, float | None] | None:
    """(market, side, line) for a quote, or None when the bet is not one we model.

    Keyed on the numeric bet id. An unrecognised id is dropped silently and deliberately: a
    market we cannot name confidently is worse than a market we do not store.
    """
    market = BET_IDS.get(bet_id)
    if market is None:
        return None
    v = (value or "").strip()
    vl = v.lower()

    if market == "btts":
        return ("btts", vl, None) if vl in ("yes", "no") else None

    if market == "h2h":
        m = {"home": "home", "draw": "draw", "away": "away"}.get(vl)
        return ("h2h", m, None) if m else None

    # ou / ht_ou -> "Over 2.5" / "Under 2.5"
    parts = v.split()
    if len(parts) != 2 or parts[0].lower() not in ("over", "under"):
        return None
    try:
        line = float(parts[1])
    except ValueError:
        return None
    if line not in MODELLED_LINES.get(market, set()):
        return None
    return (market, parts[0].lower(), line)


def parse_books(payload: dict, fixture_id: int, kickoff_utc: str, league: str,
                home: str, away: str, snapshot_ts: str, model_type: str = "",
                source: str = "api_football") -> list[dict]:
    """Every bookmaker's quote for one fixture, flattened to one row per (book, market, side).

    Consensus is NOT computed here and nothing is discarded. The whole point is to keep what the
    current capture throws away; a consensus can always be derived later, while a book that was
    never stored is gone.
    """
    rows: list[dict] = []
    mins = minutes_to_kickoff(kickoff_utc, snapshot_ts)
    band = ladder_band(mins)
    for entry in (payload or {}).get("response", []) or []:
        for bk in entry.get("bookmakers", []) or []:
            bid, bname = bk.get("id"), (bk.get("name") or "").strip()
            for bet in bk.get("bets", []) or []:
                for val in bet.get("values", []) or []:
                    cls = _classify(bet.get("id"), bet.get("name", ""),
                                    str(val.get("value", "")))
                    if cls is None:
                        continue
                    try:
                        odds = float(val.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if odds <= 1.0:
                        continue
                    market, side, line = cls
                    rows.append({
                        "snapshot_ts": snapshot_ts, "fixture_id": fixture_id,
                        "kickoff_utc": kickoff_utc,
                        "minutes_to_kickoff": round(mins, 1) if mins is not None else None,
                        "ladder_band": band, "league": league, "model_type": model_type,
                        "home_team": home, "away_team": away,
                        "bookmaker_id": bid, "bookmaker": bname,
                        "market": market, "side": side, "line": line,
                        "odds": odds, "source": source,
                    })
    return rows


def part_path(snapshot_ts) -> Path:
    """The monthly part a quote belongs in, from its OWN timestamp — not from today.

    Keyed on the row rather than the clock so a backfill or a run straddling midnight on the 1st
    files each row where it belongs instead of dumping the lot into the current month.
    """
    try:
        ts = pd.Timestamp(snapshot_ts)
        if pd.isna(ts):
            raise ValueError
    except Exception:                                                 # noqa: BLE001
        return QUOTES_DIR / "unknown.csv"
    # DAILY, not monthly. Measured volume put a monthly part far past the 100 MB hard limit even
    # after filtering; a daily part is a few MB and a finished day is never rewritten, so git
    # stores each one once instead of re-storing a growing monolith on every run.
    return QUOTES_DIR / f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}.csv"


def load_quotes(path: Path | None = None) -> pd.DataFrame:
    """Every captured quote, across all monthly parts plus the pre-partition file."""
    if path is not None:
        return pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=COLUMNS)
    frames = []
    if LEGACY_FILE.exists():
        frames.append(pd.read_csv(LEGACY_FILE))
    if QUOTES_DIR.exists():
        frames += [pd.read_csv(f) for f in sorted(QUOTES_DIR.glob("*.csv"))]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def _dedup_baseline(target: Path) -> pd.DataFrame:
    """Rows to compare against for consecutive-distinct dedup: this part and the one before it.

    Not every part. Reading the whole archive to decide one append would grow without bound, and
    reading only the current part would rewrite every unchanged price on the 1st of each month.
    Two parts bounds the work and spans the boundary. The residual cost is that a price which has
    not moved in over a month is written once more — which is harmless and arguably informative.
    """
    parts = sorted(QUOTES_DIR.glob("*.csv")) if QUOTES_DIR.exists() else []
    keep = [f for f in parts if f.name <= target.name][-2:]
    if target.exists() and target not in keep:
        keep.append(target)
    frames = [pd.read_csv(f) for f in keep if f.exists()]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def append_quotes(rows: list[dict], path: Path | None = None) -> int:
    """Append, keeping only DISTINCT consecutive prices per (fixture, book, market, side, line).

    Same discipline as the existing archives: a price that has not changed is not new
    information, and storing it every run would make the file enormous for nothing. What it
    costs is that row spacing is not evidence of sampling frequency — a fact already recorded
    against this estate's NEAR-loop analysis, and the reason `ladder_band` is stored explicitly
    rather than inferred from gaps.
    """
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]

    # An explicit path keeps the whole frame in one file (tests, ad-hoc use). Otherwise split by
    # the month each row belongs to and append to each part independently.
    if path is None:
        total = 0
        for part, chunk in df.groupby(df["snapshot_ts"].map(part_path)):
            total += _append_one(chunk, Path(part))
        return total
    return _append_one(df, Path(path))


def _append_one(df: pd.DataFrame, p: Path) -> int:
    """Append one frame to one file, keeping only consecutive-DISTINCT prices."""

    key = ["fixture_id", "bookmaker_id", "market", "side", "line"]

    def _k(frame):
        """Dedup key as a single string.

        `line` is NaN for BTTS and 1X2, and NaN NEVER EQUALS ITSELF — so a tuple key containing
        one can never match its own earlier row, and every unchanged BTTS price would be written
        again on every run. Found by the test that asserts a re-append writes nothing: it let 2
        of 8 rows through, and those 2 were exactly the line-less markets.
        """
        return (frame["fixture_id"].astype(str) + "|"
                + frame["bookmaker_id"].astype(str) + "|"
                + frame["market"].astype(str) + "|"
                + frame["side"].astype(str) + "|"
                + frame["line"].fillna(-1).astype(str))

    old = _dedup_baseline(p) if p.parent == QUOTES_DIR else (
        pd.read_csv(p) if p.exists() else pd.DataFrame(columns=COLUMNS))
    if len(old) or p.exists():
        if len(old):
            prior = old.sort_values("snapshot_ts").copy()
            prior["_k"] = _k(prior)
            last = prior.groupby("_k")["odds"].last()
        else:
            # An existing-but-empty file. `prior` must still be a frame with the right columns,
            # or the heartbeat lookup below raises NameError on the first write into it.
            prior = pd.DataFrame(columns=list(COLUMNS) + ["_k"])
            last = pd.Series(dtype=float)
        df = df.assign(_k=_k(df))
        prev = df["_k"].map(last)
        changed = prev.isna() | ~np.isclose(prev.fillna(-1).astype(float),
                                            df["odds"].astype(float))

        # §6 HEARTBEAT: an unchanged price on a near rung is still kept, ONCE per rung, so the
        # archive records that the market was observed there. Keyed on (entity, band), so a run
        # that samples the same rung five times adds one row, not five -- the loop's density is
        # for catching movement, not for proving the rung twice.
        seen_bands = set(zip(prior["_k"], prior["ladder_band"])) if len(prior) else set()
        band_key = list(zip(df["_k"], df["ladder_band"]))
        heartbeat = pd.Series(
            [b in HEARTBEAT_BANDS and (k, b) not in seen_bands
             for k, b in band_key], index=df.index)
        # A row already being written as a change is not also a heartbeat.
        heartbeat &= ~changed

        # FAR is kept once per entity -- the OPENING price -- and never change-tracked.
        # Without this the far horizon dominates the file: a week-long look-ahead across nine
        # books drifts constantly, and none of that drift is used for anything.
        seen_ents = set(prior["_k"]) if len(prior) else set()
        far = df["ladder_band"].isin(OPEN_ONLY_BANDS)
        far_first = far & ~df["_k"].isin(seen_ents)
        keep = (changed & ~far) | heartbeat | far_first
        # A FAR row that survives `keep` is ALWAYS an opening price, because FAR is never
        # change-tracked. The first version tested `far_first & ~changed`, but a first sighting
        # has no prior price, so `changed` is True for exactly the rows that are opens — every
        # one was mislabelled "change". Caught on the first real capture under the new rules:
        # 582 of 582 FAR rows said "change".
        df = df.assign(obs_reason=np.where(far, "open",
                                           np.where(changed, "change", "heartbeat")))
        df = df[keep].drop(columns="_k")
        if df.empty:
            return 0
        # Append to THIS part only. `old` may span two parts for dedup purposes; writing it back
        # would duplicate the previous month into this one.
        existing = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=COLUMNS)
        out = pd.concat([existing, df], ignore_index=True) if len(existing) else df
    else:
        # FIRST EVER WRITE to this part. Every row is new information, but the LABEL still has
        # to match the band: a first FAR sighting is an opening price, not a change. Labelling
        # it "change" made the very first row of each part misreport its own reason.
        out = df.assign(obs_reason=np.where(df["ladder_band"].isin(OPEN_ONLY_BANDS),
                                            "open", "change"))
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    mb = p.stat().st_size / 1e6
    if mb >= MAX_MB:
        log.error(f"{p.name} is {mb:.1f} MB — at or past GitHub's {MAX_MB} MB hard limit; "
                  f"the next push of this file will be REJECTED. Split it.")
    elif mb >= WARN_MB:
        log.warning(f"{p.name} is {mb:.1f} MB — approaching the {MAX_MB} MB limit.")
    return int(len(df))


def closing_quote(df: pd.DataFrame, fixture_id, market: str, side: str,
                  line: float | None = None) -> dict | None:
    """The LAST pre-kickoff quote inside the closing window, or None.

    Returns None rather than falling back to an earlier price. A six-hour-old quote is not a
    close, and the honest answer when no close was captured is that there isn't one — which is
    what lets coverage be measured instead of assumed.
    """
    m = df[(df["fixture_id"] == fixture_id) & (df["market"] == market)
           & (df["side"] == side)]
    if line is not None:
        m = m[m["line"] == line]
    m = m[(m["minutes_to_kickoff"] >= 0) & (m["minutes_to_kickoff"] <= CLOSING_WINDOW_MIN)]
    if m.empty:
        return None
    r = m.sort_values("minutes_to_kickoff").iloc[0]      # smallest minutes = closest to kickoff
    return {"odds": float(r["odds"]), "bookmaker": r["bookmaker"],
            "minutes_to_kickoff": float(r["minutes_to_kickoff"]),
            "band": r["ladder_band"]}


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many fixture-markets reached each rung. The measurement §10 exists to enable."""
    if df.empty:
        return pd.DataFrame()
    g = (df.groupby("ladder_band")
           .agg(rows=("odds", "size"),
                fixtures=("fixture_id", "nunique"),
                books=("bookmaker_id", "nunique")).reset_index())
    order = {b: i for i, (b, _) in enumerate(LADDER)}
    order.update({"FAR": 98, "POST": 99, "UNKNOWN": 100})
    return g.sort_values("ladder_band", key=lambda s: s.map(order))
