"""Profile EVERY git-tracked data file in the 5 Wowza repos (read-only).

Per tabular file: rows, columns, duplicate rows, duplicate natural keys, all-empty / mostly-empty
columns, league values not in the v9 registry, club names that cannot be resolved to a club the
history banks know for that league, unparseable / impossible dates, odds outside (1.01, 1000],
probabilities outside [0, 1], result labels outside the known set. JSON: parses, generated_at age.
Writes data_profile.csv (one row per file x issue) and prints a summary.
"""
import json, os, subprocess, sys, warnings
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# usage: python scripts/audit_data_files.py <out.csv>   (sibling checkouts of the other repos are
# profiled when present next to this repo; WOWZA_DEV overrides the parent folder, GIT the git binary)
V9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.environ.get("WOWZA_DEV", os.path.dirname(V9))
GIT = os.environ.get("GIT", "git")
REPOS = [r for r in ["wowza-betting", "wowzaV9-Pro", "wowza_v11", "wowza-dashboard", "wowza-exec"]
         if os.path.isdir(os.path.join(DEV, r))]
OUT = sys.argv[1]
sys.path.insert(0, V9); os.chdir(V9)
from src.league_registry import canonical_name   # noqa: E402
from src import team_names as TN                  # noqa: E402

# clubs each league's history banks know (the canonical spellings)
bank = []
for f in ("fd_history.parquet", "af_history.parquet"):
    p = os.path.join(V9, "output", f)
    if os.path.exists(p):
        d = pd.read_parquet(p, columns=["league", "home_team", "away_team"])
        bank.append(d)
bank = pd.concat(bank)
bank["league"] = bank["league"].map(lambda x: canonical_name(str(x)) or str(x))
CLUBS = defaultdict(set)
for lg, h, a in bank[["league", "home_team", "away_team"]].itertuples(index=False):
    CLUBS[lg].update((str(h), str(a)))
CLUBS = {k: sorted(v) for k, v in CLUBS.items()}
_res_cache = {}


def unresolved_clubs(df):
    if "league" not in df.columns:
        return None
    cols = [c for c in ("home_team", "away_team") if c in df.columns]
    if not cols:
        return None
    bad = set()
    for c in cols:
        for lg, nm in df[["league", c]].dropna().drop_duplicates().itertuples(index=False):
            clg = canonical_name(str(lg)) or str(lg)
            cand = CLUBS.get(clg)
            if not cand:
                continue                      # league without a history bank: cannot judge
            k = (clg, str(nm))
            if k not in _res_cache:
                _res_cache[k] = str(nm) in cand or TN.resolve(str(nm), cand) is not None
            if not _res_cache[k]:
                bad.add(f"{clg}:{nm}")
    return bad


KEYS = [["fixture_id", "market", "player_id"], ["fixture_id", "market"], ["match_date", "home_team", "away_team", "market", "side", "player_name"],
        ["match_date", "home_team", "away_team", "market", "player_name"], ["match_date", "home_team", "away_team", "market", "side"],
        ["match_date", "home_team", "away_team", "market"], ["date", "home_team", "away_team", "market"],
        ["match_date", "home_team", "away_team", "side"], ["date", "home_team", "away_team"], ["match_date", "home_team", "away_team"]]
RESULTS = {"WIN", "LOSS", "VOID", "PUSH", "HALF_WIN", "HALF_LOSS", "PENDING", ""}
rows = []
now = pd.Timestamp.now(tz="UTC")


def add(repo, f, n, issue, detail, severity):
    rows.append({"repo": repo, "file": f, "rows": n, "issue": issue, "detail": str(detail)[:300], "severity": severity})


for repo in REPOS:
    root = os.path.join(DEV, repo)
    files = subprocess.run([GIT, "-C", root, "ls-files"], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    files = [f for f in files if f.lower().endswith((".csv", ".parquet", ".json", ".jsonl")) and not f.startswith(("tests/", ".github/"))
             and "/fixtures/" not in f and not f.endswith(("package.json", "package-lock.json"))]
    for f in files:
        p = os.path.join(root, f)
        try:
            if f.endswith(".json"):
                doc = json.load(open(p, encoding="utf-8"))
                ga = doc.get("generated_at") if isinstance(doc, dict) else None
                if ga:
                    age = (now - pd.to_datetime(ga, utc=True, errors="coerce")).days
                    if age is not None and age > 14:
                        add(repo, f, None, "stale_json", f"generated_at {ga} ({age} d)", "AMBER")
                add(repo, f, None, "ok_json", "", "INFO")
                continue
            if f.endswith(".jsonl"):
                n = sum(1 for _ in open(p, encoding="utf-8"))
                add(repo, f, n, "ok_jsonl", "", "INFO")
                continue
            df = pd.read_parquet(p) if f.endswith(".parquet") else pd.read_csv(p, low_memory=False)
        except Exception as e:  # noqa: BLE001
            add(repo, f, None, "unreadable", f"{type(e).__name__}: {e}", "RED")
            continue
        n = len(df)
        add(repo, f, n, "profiled", f"{df.shape[1]} cols", "INFO")
        if n == 0:
            add(repo, f, 0, "empty_file", "", "AMBER")
            continue
        dup = int(df.duplicated().sum())
        if dup:
            add(repo, f, n, "duplicate_rows", dup, "RED")
        for k in KEYS:
            if set(k) <= set(df.columns):
                dk = int(df.duplicated(k).sum())
                if dk:
                    add(repo, f, n, "duplicate_key", f"{dk} on {k}", "RED")
                break
        empty = [c for c in df.columns if df[c].isna().all()]
        if empty:
            add(repo, f, n, "all_empty_columns", empty[:25], "AMBER")
        if "league" in df.columns:
            bad = sorted({str(l) for l in df["league"].dropna().unique() if not canonical_name(str(l))})
            if bad:
                add(repo, f, n, "unknown_league", bad[:20], "RED")
        uc = unresolved_clubs(df)
        if uc:
            add(repo, f, n, "unresolved_club_name", f"{len(uc)}: {sorted(uc)[:15]}", "RED")
        for c in df.columns:
            lc = c.lower()
            s = df[c]
            if ("odds" in lc or lc.startswith("best_")) and s.dtype != object:
                v = pd.to_numeric(s, errors="coerce")
                bad = int(((v <= 1.01) | (v > 1000)).sum())
                if bad:
                    add(repo, f, n, "odds_out_of_range", f"{c}: {bad}", "RED")
            if (lc.startswith("p_") or lc.endswith("_prob") or lc in ("model_prob", "p_win", "implied_prob_over")) and s.dtype != object:
                v = pd.to_numeric(s, errors="coerce")
                bad = int(((v < 0) | (v > 1)).sum())
                if bad:
                    add(repo, f, n, "prob_out_of_range", f"{c}: {bad}", "RED")
            if lc in ("date", "match_date", "kickoff_utc"):
                d = pd.to_datetime(s, errors="coerce", utc=True)
                unp = int((d.isna() & s.notna()).sum())
                if unp:
                    add(repo, f, n, "unparseable_date", f"{c}: {unp}", "RED")
                if d.notna().any() and (d.min() < pd.Timestamp("1990-01-01", tz="UTC") or d.max() > now + pd.Timedelta(days=400)):
                    add(repo, f, n, "impossible_date", f"{c}: {d.min()} .. {d.max()}", "RED")
            if lc == "result" and s.dtype == object:
                bad = sorted(set(s.dropna().astype(str).str.upper().str.strip()) - RESULTS)
                if bad:
                    add(repo, f, n, "unknown_result_label", bad[:10], "AMBER")
        if {"result", "match_date"} <= set(df.columns):
            md = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
            pend = df["result"].isna() | df["result"].astype(str).str.strip().isin(["", "PENDING", "nan"])
            old = int((pend & (md < now - pd.Timedelta(days=7))).sum())
            if old:
                add(repo, f, n, "unsettled_over_7_days", old, "AMBER")
        if {"result", "pnl"} <= set(df.columns):
            settled = df["result"].astype(str).str.upper().isin(["WIN", "LOSS"])
            nopnl = int((settled & pd.to_numeric(df["pnl"], errors="coerce").isna()).sum())
            if nopnl:
                add(repo, f, n, "settled_without_pnl", nopnl, "AMBER")

rep = pd.DataFrame(rows)
rep.to_csv(OUT, index=False)
files = rep[rep["issue"].isin(["profiled", "ok_json", "ok_jsonl", "unreadable"])]
print("files profiled per repo:", files.groupby("repo").size().to_dict())
iss = rep[~rep["issue"].isin(["profiled", "ok_json", "ok_jsonl"])]
print("issues by type x severity:\n", iss.groupby(["issue", "severity"]).size().to_string())
print("\nRED issues:")
print(iss[iss["severity"] == "RED"][["repo", "file", "issue", "detail"]].to_string(index=False, max_colwidth=140))
