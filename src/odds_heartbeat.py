"""Last-seen odds heartbeat — proof that a price was still in force near kickoff (P1 2026-10-22).

    from src.odds_heartbeat import record, load, merge_into_archives

THE DEFECT. Every odds archive here is a change-log: a row is written only when a price MOVES.
So when the NEAR capture priced a fixture at T-8m and the price had not moved since T-3h, the
archive's last row still said T-3h, and the 30-minute close rule (shadow_settlement.CLOSE_MAX_MIN)
correctly refused it. Measured 2026-10-22 over 3,613 capture fixture-markets: a capture RAN inside
T-30m for 30.1% of them, but only 2.8% have a recorded observation there.

THE FIX, WITHOUT BLOATING THE CHANGE-LOG. One extra row per (fixture, market) per writer holding
the LATEST strictly-pre-kickoff observation: when it was seen and at what price. It is overwritten
in place, so it never grows with run count. It holds only observations within HB_MAX_H of kickoff
because it exists for the close, nothing else.

Layout: output/odds_heartbeat/<writer>/<match_date>.csv — one file per kickoff day, so a commit
rewrites only the days that are live (a single growing file would store a new full copy in git on
every push, the failure that took player_history.parquet's .git to 3.4 GB). Writers never share a
file, so two workflows can never conflict on it.

Rules (each one is tested):
* an observation with no parseable kickoff is NOT recorded — nothing could prove it pre-kickoff;
* an observation at or after kickoff is NOT recorded — never a post-kickoff price;
* a newer observation replaces an older one; an older one never replaces a newer one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

import config

HB_DIR = config.OUTPUT_DIR / "odds_heartbeat"
HB_MAX_H = 6.0
COLS = ["snapshot_ts", "match_date", "kickoff_utc", "league", "match", "market", "odds",
        "writer"]
KEY = ["match_date", "match", "market"]

#: heartbeat writer -> the change-log it shadows (same source, same names, same markets). The
#: certifier merges the two so a close is chosen from one curve. If the parent is renamed the
#: heartbeat still loads as its own archive, just later in priority.
PARENTS = {
    "std_capture": "standard_sidemarket_odds_history.csv",
    "nf_capture": "newformat_odds_history.csv",
}


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="coerce", utc=True)


def record(writer: str, rows: Iterable[dict], out_dir: Path | None = None) -> int:
    """Upsert the latest pre-kickoff observation per (match_date, match, market). Returns the
    number of keys written or refreshed. Never raises into the caller's capture."""
    base = Path(out_dir or HB_DIR) / writer
    new = pd.DataFrame(list(rows))
    if new.empty or not set(COLS[:-1]) <= set(new.columns):
        return 0
    new = new[COLS[:-1]].copy()
    new["writer"] = writer
    ts, ko = _utc(new["snapshot_ts"]), _utc(new["kickoff_utc"])
    odds = pd.to_numeric(new["odds"], errors="coerce")
    lead_h = (ko - ts).dt.total_seconds() / 3600.0
    keep = ts.notna() & ko.notna() & odds.gt(1.0) & (ts < ko) & lead_h.le(HB_MAX_H)
    new = new[keep]
    if new.empty:
        return 0
    n = 0
    for md, g in new.groupby(new["match_date"].astype(str).str[:10]):
        p = base / f"{md}.csv"
        cur = pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame(columns=COLS)
        both = pd.concat([cur, g.astype(str)], ignore_index=True)
        both["_t"] = _utc(both["snapshot_ts"])
        both = (both.sort_values(KEY + ["_t"]).drop_duplicates(KEY, keep="last")
                .drop(columns="_t").sort_values(KEY))
        p.parent.mkdir(parents=True, exist_ok=True)
        both[COLS].to_csv(p, index=False)
        n += len(g)
    return n


def load(out_dir: Path | None = None, min_match_date: str | None = None
         ) -> dict[str, pd.DataFrame]:
    """{writer: frame} for every heartbeat on disk (optionally only kickoff days >= a date)."""
    base = Path(out_dir or HB_DIR)
    out: dict[str, pd.DataFrame] = {}
    if not base.exists():
        return out
    for wdir in sorted(p for p in base.iterdir() if p.is_dir()):
        parts = []
        for f in sorted(wdir.glob("*.csv")):
            if min_match_date and f.stem < min_match_date:
                continue
            try:
                parts.append(pd.read_csv(f, dtype={"odds": float}))
            except Exception:                                         # noqa: BLE001
                continue
        if parts:
            out[wdir.name] = pd.concat(parts, ignore_index=True)
    return out


def merge_into_archives(archives: list[tuple[str, pd.DataFrame | None]],
                        heartbeats: dict[str, pd.DataFrame],
                        parents: dict[str, str] | None = None
                        ) -> list[tuple[str, pd.DataFrame | None]]:
    """Fold each heartbeat into the change-log it shadows; append orphans at the end.

    The heartbeat rows are ordinary observations (snapshot_ts, odds), so after the merge the last
    pre-kickoff row of a fixture is simply the latest time the capture actually SAW that price."""
    parents = PARENTS if parents is None else parents
    out = list(archives)
    names = [n for n, _ in out]
    for writer, hb in heartbeats.items():
        if hb is None or hb.empty:
            continue
        hb = hb.drop(columns=[c for c in ("writer",) if c in hb.columns])
        parent = parents.get(writer)
        if parent in names:
            i = names.index(parent)
            frame = out[i][1]
            merged = hb if frame is None or frame.empty else pd.concat([frame, hb],
                                                                       ignore_index=True)
            out[i] = (parent, merged)
        else:
            out.append((f"odds_heartbeat/{writer}", hb))
    return out
