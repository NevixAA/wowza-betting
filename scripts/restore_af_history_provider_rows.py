"""
restore_af_history_provider_rows.py - give back the af_history provider rows the loader write-back
destroyed, from git history, BESIDE the original file (2026-11-16).
=====================================================================================================
THE DAMAGE. src/data_loader's 'BANK THE ENRICHMENT' (2026-09-22 .. 2026-11-01 commits of retrain.yml)
did concat + drop_duplicates(keep="last") with a shots-only frame, so a provider row of
output/af_history.parquet (written by scripts/backfill_af_history.py: full-time score, shots, corners,
fouls, cards) whose key the loader touched was REPLACED by a row holding only shots / SoT / xG /
inside-box. Its score, corners, fouls and cards went from present to missing.

WHAT THIS DOES. Walks every committed version of output/af_history.parquet on HEAD's history, oldest
first. For each row of the CURRENT file that is a write-back row (no full-time score) it takes the last
committed version in which the same key (league, date, home_team, away_team) was still a PROVIDER row
(full-time score present - the write-back never wrote one, so such a row is untouched provider data)
and writes that row exactly as committed, with the commit it came from, to
    output/corrections/af_history_restored_provider_rows_2026-11.parquet
plus an audit file with the counts (…_audit.json). Cells the write-back erased from rows that were
never provider rows (an xG / inside-box value an earlier write-back had banked, wiped by a later one
whose frame held a null) are listed cell by cell in …_writeback_cells_2026-11.parquet (key, column,
last committed value, commit): they are loader copies, unproven, kept so nothing committed is lost. af_history.parquet itself is NOT modified (it is an
original, and on data-truth-100 the no-score rule in src/af_history_provenance is what marks a row as
written back - rewriting the score in place would silently re-label 28,412 rows). Rows the owner's
corrections removed on purpose (misfiled leagues, proven duplicates) are not in the current file and
are not restored. Read-only on git; no API call.

    python scripts/restore_af_history_provider_rows.py [--ref HEAD] [--check]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from collections import Counter
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AF = "output/af_history.parquet"
OUT = ROOT / "output" / "corrections" / "af_history_restored_provider_rows_2026-11.parquet"
AUDIT = ROOT / "output" / "corrections" / "af_history_restored_provider_rows_2026-11_audit.json"
CELLS = ROOT / "output" / "corrections" / "af_history_restored_writeback_cells_2026-11.parquet"
KEY = ["league", "date", "home_team", "away_team"]
STAT = ["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
        "HXG", "AXG", "HIB", "AIB"]


def _git(*a, binary=False):
    r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True)
    r.check_returncode()
    return r.stdout if binary else r.stdout.decode()


def _keys(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.Series(list(zip(df["league"].astype(str), d, df["home_team"].astype(str),
                              df["away_team"].astype(str))), index=df.index, dtype=object)


def _provider(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["FTHG"], errors="coerce").notna()
            & pd.to_numeric(df["FTAG"], errors="coerce").notna())


def build(ref: str = "HEAD") -> tuple[pd.DataFrame, dict]:
    commits = _git("log", "--format=%H|%ad", "--date=short", "--reverse", ref, "--", AF).split()
    head_sha = commits[-1].split("|")[0]
    head = pd.read_parquet(io.BytesIO(_git("show", f"{head_sha}:{AF}", binary=True)))
    head_k = _keys(head)
    wb = set(head_k[~_provider(head)])
    last: dict = {}
    lost_by_commit: Counter = Counter()
    cell_cols = STAT + ["fixture_id", "home_team_id", "away_team_id"]
    seen_v = seen_c = None                     # last non-missing value / its commit, per key x column
    for line in commits:
        sha, day = line.split("|")
        v = pd.read_parquet(io.BytesIO(_git("show", f"{sha}:{AF}", binary=True)))
        k = _keys(v)
        sel = _provider(v) & k.isin(wb)
        for key, row in zip(k[sel], v[sel].to_dict("records")):
            last[key] = (row, sha, day)
        vv = v.reindex(columns=cell_cols).astype(object)
        vv.index = pd.Index(k.to_numpy(), dtype=object)
        vv = vv[~vv.index.duplicated(keep="last")]
        vc = pd.DataFrame(pd.notna(vv).to_numpy(), index=vv.index, columns=cell_cols).replace(
            {True: sha, False: None}).astype(object)
        vc = vc.where(pd.notna(vv))
        seen_v = vv if seen_v is None else vv.combine_first(seen_v)
        seen_c = vc if seen_c is None else vc.combine_first(seen_c)
    rows, cols = [], []
    for key in sorted(last):
        row, sha, day = last[key]
        for c in row:
            if c not in cols:
                cols.append(c)
        lost_by_commit[sha[:9]] += 1
        rows.append({**row, "restored_from_commit": sha, "restored_from_commit_date": day})
    out = pd.DataFrame(rows, columns=cols + ["restored_from_commit", "restored_from_commit_date"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ("league", "home_team", "away_team", "season", "FTR"):
        if c in out.columns:
            out[c] = out[c].astype("string")
    # what the current file holds for the same keys (all write-back rows: no score)
    cur = head[head_k.isin(set(last))]
    # cell level: present in an earlier committed version, missing now, and NOT given back by a
    # restored provider row above
    hv = head.reindex(columns=cell_cols).astype(object)
    hv.index = pd.Index(head_k.to_numpy(), dtype=object)
    hv = hv[~hv.index.duplicated(keep="last")]
    sv, sc = seen_v.reindex(hv.index), seen_c.reindex(hv.index)
    rest = out.copy()
    rest.index = pd.Index(_keys(rest).to_numpy(), dtype=object) if len(rest) else rest.index
    rv = rest.reindex(columns=cell_cols).reindex(hv.index)
    lost = pd.isna(hv) & pd.notna(sv) & pd.isna(rv)
    cl = []
    for c in cell_cols:
        idx = lost.index[lost[c].to_numpy()]
        for key in idx:
            cl.append({"league": key[0], "date": key[1], "home_team": key[2], "away_team": key[3],
                       "column": c, "value": float(sv.at[key, c]), "source_commit": sc.at[key, c]})
    cells = pd.DataFrame(cl, columns=["league", "date", "home_team", "away_team", "column", "value",
                                      "source_commit"])
    audit = {
        "what": "af_history provider rows replaced by the loader write-back, restored from git beside "
                "the original; af_history.parquet is not modified",
        "af_history_head_commit": head_sha,
        "committed_versions_walked": len(commits),
        "current_rows": int(len(head)),
        "current_written_back_rows_no_score": len(wb),
        "restored_provider_rows": int(len(out)),
        "restored_cells_present": {c: int(out[c].notna().sum()) for c in STAT + ["fixture_id"]
                                   if c in out.columns},
        "same_cells_present_in_current_file": {c: int(cur[c].notna().sum()) for c in STAT + ["fixture_id"]
                                               if c in cur.columns},
        "writeback_cells_listed": {c: int(v) for c, v in cells["column"].value_counts().sort_index().items()},
        "writeback_cells_note": "erased loader copies on rows that were never provider rows; listed in "
                                "af_history_restored_writeback_cells_2026-11.parquet, not served",
        "restored_rows_by_league": {k: int(v) for k, v in out["league"].value_counts().items()},
        "restored_rows_by_source_commit": dict(lost_by_commit),
        "not_restored": "rows removed on purpose (misfiled leagues 2026-11-01, proven duplicates) - "
                        "not in the current file; write-back rows that were never provider rows",
    }
    return out, cells, audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed file")
    a = ap.parse_args(argv)
    out, cells, audit = build(a.ref)
    if a.check:
        old = pd.read_parquet(OUT)
        same = old.drop(columns=["restored_from_commit", "restored_from_commit_date"]).equals(
            out.drop(columns=["restored_from_commit", "restored_from_commit_date"]))
        print(f"[restore] rebuilt {len(out):,} rows; identical to committed: {same}")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    cells.to_parquet(CELLS, index=False)
    AUDIT.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("restored_provider_rows", "restored_cells_present",
                                            "writeback_cells_listed")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
