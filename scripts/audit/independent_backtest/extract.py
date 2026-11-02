"""Extract frozen inputs from git history (read-only).

1. Every committed version of output/predictions.csv  -> _cache/pred_snapshots.parquet
   (one row per fixture per committed board, raw model probabilities and quoted prices,
   plus the production decision columns kept ONLY for a later selection-layer comparison).
2. First committed appearance of every ledger key in output/bets_ledger.csv and
   output/side_bets_ledger.csv -> _cache/ledger_first_seen_{ou25,side}.parquet
   (the ledger rows were overwritten in place before 2026-10-26 - "best tier wins" - so the
   current file is not a first-sight record; the git history is).

Usage:  python -m scripts.audit.independent_backtest.extract
"""
from __future__ import annotations

import sys

import pandas as pd

from .common import CACHE, REF, file_commits, heavy_lock, iter_blobs, read_csv_bytes

PRED_KEEP = [
    "league", "date", "kickoff_utc", "home_team", "away_team", "model_type",
    "odds_over25", "odds_under25", "p_over25",
    "odds_btts", "odds_btts_no", "p_btts", "odds_over15", "p_over15", "odds_over35", "p_over35",
    "home_scored_last5", "away_scored_last5", "home_conceded_last5", "away_conceded_last5",
    "serving_no_form", "serving_recipe", "drift_signal",
    # production decision columns - comparison only, never used to select
    "edge_over", "edge_under", "bet", "signal_tier", "best_edge", "best_side",
    "no_form_data", "min_odds_blocked", "max_odds_blocked",
    "generated_at", "git_sha", "model_version",
]


def extract_predictions() -> pd.DataFrame:
    commits = file_commits("output/predictions.csv")
    frames, bad = [], 0
    for (sha, ts), (_, data) in zip(commits, iter_blobs([f"{s}:output/predictions.csv" for s, _ in commits])):
        if not data:
            bad += 1
            continue
        try:
            d = read_csv_bytes(data)
        except Exception:
            bad += 1
            continue
        keep = [c for c in PRED_KEEP if c in d.columns]
        d = d[keep].copy()
        d["snap_sha"] = sha
        d["snap_commit_ts"] = ts
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string")
    print(f"predictions: {len(commits)} commits, {bad} unreadable, {len(out)} fixture-rows")
    return out


def extract_ledger_first_seen(path: str, key_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (first-seen rows, per-key version count) across the committed history of a ledger."""
    commits = file_commits(path)
    seen: dict = {}
    changes: dict = {}
    for (sha, ts), (_, data) in zip(commits, iter_blobs([f"{s}:{path}" for s, _ in commits])):
        if not data:
            continue
        try:
            d = read_csv_bytes(data, dtype=str)
        except Exception:
            continue
        if "source" in d.columns:
            d = d[d["source"].fillna("live") == "live"]
        if not set(key_cols) <= set(d.columns):
            continue
        d["match_date"] = d["match_date"].str[:10]
        for rec in d.to_dict("records"):
            k = tuple(str(rec.get(c, "")).strip() for c in key_cols)
            sig = (rec.get("odds"), rec.get("signal_tier"), rec.get("edge_pct"))
            if k not in seen:
                r = dict(rec)
                r["first_commit_sha"] = sha
                r["first_commit_ts"] = ts
                seen[k] = r
                changes[k] = [sig]
            elif changes[k][-1] != sig:
                changes[k].append(sig)
    fs = pd.DataFrame(list(seen.values()))
    fs["n_versions_odds_tier_edge"] = [len(changes[k]) for k in seen]
    print(f"{path}: {len(commits)} commits, {len(fs)} distinct live keys")
    return fs, pd.DataFrame()


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    with heavy_lock("AU5 extract predictions/ledger history"):
        preds = extract_predictions()
        preds.to_parquet(CACHE / "pred_snapshots.parquet", index=False)
        ou, _ = extract_ledger_first_seen("output/bets_ledger.csv",
                                          ["match_date", "home_team", "away_team", "side"])
        ou.to_parquet(CACHE / "ledger_first_seen_ou25.parquet", index=False)
        sd, _ = extract_ledger_first_seen("output/side_bets_ledger.csv",
                                          ["match_date", "home_team", "away_team", "market"])
        sd.to_parquet(CACHE / "ledger_first_seen_side.parquet", index=False)
    print(f"ref={REF} cache={CACHE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
