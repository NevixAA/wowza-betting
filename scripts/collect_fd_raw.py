"""
football-data.co.uk raw season files for the Standard leagues (P7 item 4, 2026-11-09).
====================================================================================
The bars study and src.standard_repair.load_fd_entry_prices read football-data's PRE-MATCH (entry)
O/U 2.5 prices from raw `{CODE}_{YYZZ}.csv` files. The only copy was Pro's
experiments/ou_info_inputs_2026_10/fd_raw: it has no League One 2023-24 file (E2_2324.csv, 602 rows
unpriced) and every current-season file stops about 2026-10-10.

This downloads the raw files (free; the loader already downloads the same files in CI) into
output/collect/fd_raw/ with a provenance manifest (url, sha256, rows, last match date, fetched_at).
A file is written only when its content changed; a failed or empty download never replaces a stored
file. Codes come from the league registry (data_loader._fd_download_tables, Standard = mmz4281),
seasons from 2022-23 to the current season (the loader's July roll, UTC). No API key, no quota.

    python scripts/collect_fd_raw.py            # download + manifest
    python scripts/collect_fd_raw.py --dry-run  # list the planned files only
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "output" / "collect" / "fd_raw"
MANIFEST = OUT_DIR / "manifest.json"
URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
#: first season of the bars study's walk-forward window (the Pro fd_raw copy starts here too)
FIRST_SEASON_START = 2022


def seasons(now: datetime) -> list[str]:
    start = now.year if now.month >= 7 else now.year - 1
    return [f"{str(y)[2:]}{str(y + 1)[2:]}" for y in range(FIRST_SEASON_START, start + 1)]


def plan(now: datetime) -> list[tuple[str, str, str]]:
    from src.data_loader import _fd_download_tables
    std, _new = _fd_download_tables()
    return [(lg, code, s) for lg, code in sorted(std.items()) for s in seasons(now)]


def _summary(raw: bytes) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="cp1252", encoding_errors="replace",
                         on_bad_lines="skip", low_memory=False)
    if not {"Date", "HomeTeam", "AwayTeam"} <= set(df.columns):
        return {"rows": 0, "valid": False}
    d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return {"rows": int(df["HomeTeam"].notna().sum()), "valid": True,
            "last_match": str(d.max().date()) if d.notna().any() else None,
            "has_entry_ou25": bool({"Avg>2.5", "B365>2.5", "P>2.5", "Max>2.5"} & set(df.columns))}


def run(dry_run: bool = False, session=None) -> dict:
    import requests
    now = datetime.now(timezone.utc)
    jobs = plan(now)
    if dry_run:
        return {"planned": [f"{c}_{s}.csv ({lg})" for lg, c, s in jobs]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"files": {}}
    s = session or requests.Session()
    counts = {"written": 0, "unchanged": 0, "failed": 0, "not_published": 0}
    for lg, code, season in jobs:
        name = f"{code}_{season}.csv"
        url = URL.format(season=season, code=code)
        try:
            r = s.get(url, timeout=60, headers={"User-Agent": "wowza-data-truth-collector"})
        except Exception as e:                                      # noqa: BLE001
            counts["failed"] += 1
            print(f"  [fd_raw] {name}: request failed ({type(e).__name__}) - stored copy kept")
            continue
        if r.status_code == 404:
            counts["not_published"] += 1
            continue
        if r.status_code != 200 or not r.content:
            counts["failed"] += 1
            print(f"  [fd_raw] {name}: HTTP {r.status_code} - stored copy kept")
            continue
        info = _summary(r.content)
        if not info["valid"] or info["rows"] == 0:
            counts["failed"] += 1
            print(f"  [fd_raw] {name}: not a football-data table - stored copy kept")
            continue
        sha = hashlib.sha256(r.content).hexdigest()
        prev = man["files"].get(name)
        if prev and prev.get("sha256") == sha:
            counts["unchanged"] += 1
            continue
        (OUT_DIR / name).write_bytes(r.content)
        man["files"][name] = {"league": lg, "code": code, "season": season, "url": url, "sha256": sha,
                              "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), **info}
        counts["written"] += 1
        time.sleep(0.5)                                             # gentle on a free server
    man["source"] = "football-data.co.uk (free CSV files; the loader downloads the same files in CI)"
    man["updated_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    man["last_run"] = counts
    MANIFEST.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[fd_raw] {counts}")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = run(dry_run=a.dry_run)
    if a.dry_run:
        print(f"[fd_raw] {len(out['planned'])} planned file(s): {out['planned'][:6]} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
