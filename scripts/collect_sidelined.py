"""
collect_sidelined.py - API-Football /sidelined?player for every player in player_history.parquet
without a stored answer (owner decision C/D, 2026-11-02).

    python scripts/collect_sidelined.py [--limit N] [--dry-run]

OUTPUT
  output/collect/sidelined_players.parquet  one row per player asked: player_id, n_spells,
        fetched_at. A player with n_spells 0 has a REAL empty answer ("no recorded spell"); a failed
        request stores nothing and is asked again next run. This file is the journal.
  output/collect/sidelined_spells.parquet   one row per spell: player_id, type, start, end (ISO
        dates as the provider gives them; an open spell has end None).
The injury features read these through player_model/history_integrity.injury_features_asof
(only spells that STARTED before the match).
"""
from __future__ import annotations

import argparse
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

PLAYERS = COLLECT_DIR / "sidelined_players.parquet"
SPELLS = COLLECT_DIR / "sidelined_spells.parquet"
P_COLS = ["player_id", "n_spells", "fetched_at"]
S_COLS = ["player_id", "type", "start", "end"]


def parse(body: dict | None, player_id: int) -> tuple[dict, list[dict]]:
    spells = []
    for e in (body or {}).get("response") or []:
        st = e.get("start")
        if not st:
            continue                                  # a spell without a start cannot be placed
        en = e.get("end")
        spells.append({"player_id": int(player_id), "type": e.get("type"),
                       "start": str(st)[:10], "end": str(en)[:10] if en else None})
    return ({"player_id": int(player_id), "n_spells": len(spells),
             "fetched_at": datetime.now(timezone.utc).isoformat()}, spells)


def todo_players() -> list[int]:
    h = history_frame(["player_id", "date"])
    if h.empty:
        return []
    h["player_id"] = pd.to_numeric(h["player_id"], errors="coerce")
    h = h.dropna(subset=["player_id"])
    last = h.groupby("player_id")["date"].max().sort_values(ascending=False)   # active players first
    have = read_parquet(PLAYERS)
    done = set(pd.to_numeric(have["player_id"], errors="coerce").dropna().astype(int)) if not have.empty else set()
    return [int(p) for p in last.index if int(p) not in done]


def _flush(pbuf, sbuf):
    if sbuf:                                          # spells first: the players file is the journal
        append_parquet(SPELLS, sbuf, ["player_id", "type", "start", "end"], S_COLS)
    if pbuf:
        append_parquet(PLAYERS, pbuf, ["player_id"], P_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None) -> dict:
    todo = todo_players()
    print(f"[sidelined] {len(todo):,} player(s) without a stored /sidelined answer")
    if dry_run or not todo:
        return {"todo": len(todo), "fetched": 0}
    client = client or AFClient()
    pbuf, sbuf, fetched = [], [], 0
    try:
        for pid in todo[: limit or None]:
            body = client.get("sidelined", {"player": pid})
            if body is None:
                continue
            prow, spells = parse(body, pid)
            pbuf.append(prow)
            sbuf.extend(spells)
            fetched += 1
            if len(pbuf) >= FLUSH_EVERY:
                _flush(pbuf, sbuf)
                pbuf, sbuf = [], []
    except BudgetStop as e:
        print(f"[sidelined] budget stop: {e} (resumable)")
    finally:
        _flush(pbuf, sbuf)
        client.flush()
    print(f"[sidelined] fetched {fetched:,}; {len(todo) - fetched:,} left")
    return {"todo": len(todo), "fetched": fetched}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect API-Football /sidelined per player")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
