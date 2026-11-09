"""Player-id conflicts: one API-Football player id that carries TWO real players (owner 2026-11-09).

RULE (provable, nothing guessed). A player id is CONFLICTED when both hold:
  1. the history holds two different fixtures of the id on the same UTC day for two different teams
     (one person cannot play both), and
  2. API-Football's own transfer records under that id (output/collect/transfers.parquet) carry two
     or more different player names.
Condition 1 alone is not enough: Facundo Torres (51620) has a Uruguay friendly and a Palmeiras match
on 2025-11-19 UTC and one name in every transfer record - one player, two real matches.

WHAT IS DONE. The provider gives the second player no id of his own, so the rows cannot be split
into two players without inventing an identity. Every row of a conflicted id is EXCLUDED from the
props history (training and serving read props_harness.source_frame, which applies this), so no
feature, label or tip is built from a mixed record. The original rows stay in player_history.parquet
untouched; the decision and its evidence are the correction file beside them:
    output/corrections/player_id_conflicts.csv  (one row per conflicted id)

    python -m player_model.identity_conflicts --write     recompute and write the correction file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config

CORRECTIONS_FILE = config.OUTPUT_DIR / "corrections" / "player_id_conflicts.csv"
TRANSFERS_FILE = config.OUTPUT_DIR / "collect" / "transfers.parquet"
COLS = ["player_id", "rule", "same_day_pairs", "history_teams", "history_names", "transfer_names",
        "history_rows", "decision"]
DECISION = "EXCLUDE_ALL_ROWS"


def find(history: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """The conflicted ids with their evidence (COLS)."""
    h = history.copy()
    h["player_id"] = pd.to_numeric(h["player_id"], errors="coerce")
    h = h[h["player_id"] > 0]
    h["_day"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    g = h.groupby(["player_id", "_day"]).agg(n_fx=("fixture_id", "nunique"), n_team=("team", "nunique"))
    cand = sorted(set(g[(g["n_fx"] > 1) & (g["n_team"] > 1)].index.get_level_values(0).astype("int64")))
    tr = transfers.copy()
    tr["player_id"] = pd.to_numeric(tr["player_id"], errors="coerce")
    out = []
    for pid in cand:
        names = sorted(set(tr.loc[tr["player_id"] == pid, "player_name"].dropna().astype(str).str.strip()) - {""})
        if len(names) < 2:
            continue
        rows = h[h["player_id"] == pid]
        days = g.loc[pid]
        pairs = []
        for day in days[(days["n_fx"] > 1) & (days["n_team"] > 1)].index:
            r = rows[rows["_day"] == day]
            pairs.append({"day": str(day.date()), "fixtures": [int(x) for x in r["fixture_id"]],
                          "teams": sorted(r["team"].astype(str).unique())})
        out.append({"player_id": int(pid),
                    "rule": "same-day fixtures for two teams AND >= 2 names in the provider's transfer records",
                    "same_day_pairs": json.dumps(pairs, ensure_ascii=False),
                    "history_teams": json.dumps(rows.groupby("team").size().to_dict(), ensure_ascii=False),
                    "history_names": json.dumps(sorted(rows["player_name"].dropna().astype(str).unique()),
                                                ensure_ascii=False),
                    "transfer_names": json.dumps(names, ensure_ascii=False),
                    "history_rows": int(len(rows)), "decision": DECISION})
    return pd.DataFrame(out, columns=COLS)


def conflicted_ids(path: Path | None = None) -> set:
    p = CORRECTIONS_FILE if path is None else Path(path)
    if not p.exists():
        return set()
    c = pd.read_csv(p)
    return set(pd.to_numeric(c.loc[c["decision"] == DECISION, "player_id"], errors="coerce").dropna().astype("int64"))


def apply(frame: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """The frame without the rows of conflicted ids (the correction file decides); prints the count."""
    ids = conflicted_ids(path)
    if not ids or frame is None or frame.empty or "player_id" not in frame.columns:
        return frame
    bad = pd.to_numeric(frame["player_id"], errors="coerce").isin(list(ids))
    if bad.any():
        print(f"[identity] {int(bad.sum())} row(s) of {len(ids)} conflicted player id(s) excluded "
              f"({sorted(ids)}; output/corrections/player_id_conflicts.csv)")
    return frame[~bad.to_numpy()].copy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    from .props_harness import _read_history
    from .cup_history import cup_rows, new_cup_rows
    h = _read_history(None)
    h = pd.concat([h, new_cup_rows(h, cup_rows())], ignore_index=True, sort=False)
    found = find(h, pd.read_parquet(TRANSFERS_FILE))
    print(found[["player_id", "history_rows", "transfer_names", "same_day_pairs"]].to_string())
    if a.write:
        CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        found.to_csv(CORRECTIONS_FILE, index=False)
        print(f"wrote {CORRECTIONS_FILE} ({len(found)} conflicted id(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
