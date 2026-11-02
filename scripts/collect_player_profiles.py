"""
collect_player_profiles.py - API-Football /players?id&season for every (player_id, season) in
player_history.parquet without a stored profile (owner decision C/D, 2026-11-02).

    python scripts/collect_player_profiles.py [--limit N] [--dry-run]

OUTPUT output/collect/player_profiles.parquet, one row per (player_id, season):
  status            'ok' (the provider returned the player) or 'empty' (it returned nothing: a real
                    answer, recorded so it is not asked again)
  height_cm, weight_kg, birth_date   player.height / weight / birth.date as the provider gives
                    them; unparseable or absent -> NaN / None (never a placeholder)
  penalty_scored, penalty_missed     the season's statistics[].penalty, summed over the entries
                    (competitions / clubs) where BOTH are reported; NaN when no entry reports both
  pen_entries       how many entries carried both (0 -> the season's penalty record is unknown)
The parquet is the journal: a stored key is never fetched again; a failed request stores nothing.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_common import (COLLECT_DIR, FLUSH_EVERY, AFClient, BudgetStop,  # noqa: E402
                            append_parquet, history_frame, read_parquet)

OUT = COLLECT_DIR / "player_profiles.parquet"
COLS = ["player_id", "season", "status", "height_cm", "weight_kg", "birth_date",
        "penalty_scored", "penalty_missed", "pen_entries", "fetched_at"]
_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def _measure(v) -> float:
    """'185 cm' / '185' / 185 -> 185.0; None / '' / no number -> NaN."""
    if v is None:
        return float("nan")
    m = _NUM.search(str(v))
    return float(m.group(1)) if m else float("nan")


def parse(body: dict | None, player_id: int, season: str) -> dict:
    resp = (body or {}).get("response") or []
    row = {"player_id": int(player_id), "season": str(season), "status": "empty",
           "height_cm": float("nan"), "weight_kg": float("nan"), "birth_date": None,
           "penalty_scored": float("nan"), "penalty_missed": float("nan"), "pen_entries": 0,
           "fetched_at": datetime.now(timezone.utc).isoformat()}
    if not resp:
        return row
    p = resp[0].get("player") or {}
    row["status"] = "ok"
    row["height_cm"] = _measure(p.get("height"))
    row["weight_kg"] = _measure(p.get("weight"))
    bd = (p.get("birth") or {}).get("date")
    row["birth_date"] = str(bd)[:10] if bd else None
    sc = ms = 0
    n = 0
    for st in resp[0].get("statistics") or []:
        pen = st.get("penalty") or {}
        s, m = pen.get("scored"), pen.get("missed")
        if isinstance(s, (int, float)) and isinstance(m, (int, float)):
            sc += s
            ms += m
            n += 1
    if n:
        row["penalty_scored"], row["penalty_missed"] = float(sc), float(ms)
    row["pen_entries"] = n
    return row


def todo_keys() -> list[tuple[int, str]]:
    h = history_frame(["player_id", "season"]).dropna()
    if h.empty:
        return []
    h["player_id"] = pd.to_numeric(h["player_id"], errors="coerce")
    h = h.dropna().drop_duplicates()
    keys = {(int(p), str(s)) for p, s in zip(h["player_id"], h["season"])}
    have = read_parquet(OUT)
    if not have.empty:
        keys -= {(int(p), str(s)) for p, s in zip(have["player_id"], have["season"])}
    # newest seasons first: they feed the live features
    return sorted(keys, key=lambda k: (k[1], k[0]), reverse=True)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None) -> dict:
    keys = todo_keys()
    print(f"[profiles] {len(keys):,} (player, season) pair(s) without a stored profile")
    if dry_run or not keys:
        return {"todo": len(keys), "fetched": 0}
    client = client or AFClient()
    buf, fetched = [], 0
    try:
        for pid, season in keys[: limit or None]:
            body = client.get("players", {"id": pid, "season": season})
            if body is None:
                continue
            buf.append(parse(body, pid, season))
            fetched += 1
            if len(buf) >= FLUSH_EVERY:
                append_parquet(OUT, buf, ["player_id", "season"], COLS)
                buf = []
    except BudgetStop as e:
        print(f"[profiles] budget stop: {e} (resumable)")
    finally:
        if buf:
            append_parquet(OUT, buf, ["player_id", "season"], COLS)
        client.flush()
    print(f"[profiles] fetched {fetched:,}; {len(keys) - fetched:,} left")
    return {"todo": len(keys), "fetched": fetched}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football player profiles per season")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
