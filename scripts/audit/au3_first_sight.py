"""AU3 Phase 6 step 1: reconstruct IMMUTABLE FIRST-SIGHT selections from the git history of the ledgers.

Until 2026-10-26 a later predict run could overwrite a logged tip's tier, odds and edge in place
("best tier wins", src/ledger.py docstring). The committed ledger therefore is NOT first-sight before
that date. This script walks every committed version of

    output/bets_ledger.csv        key (match_date, home, away, side, source)
    output/side_bets_ledger.csv   key (match_date, home, away, market, source)

(origin/main @ AUDIT_REF, oldest first) and records, per key, the row as it was in the FIRST commit
that contains it, plus a change log of every later change to the decision fields, the settlement
fields, and whether the row ever disappeared.

Writes reports/au3_data/first_sight_{ou25,side}.csv and reports/au3_data/ledger_mutations_{ou25,side}.csv.
Read-only: touches nothing outside reports/au3_data.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au3_common import DATA, git_versions, heavy_lock, read_csv_text  # noqa: E402

SPECS = {
    "ou25": ("output/bets_ledger.csv", ["match_date", "home_team", "away_team", "side", "source"]),
    "side": ("output/side_bets_ledger.csv", ["match_date", "home_team", "away_team", "market", "source"]),
}
DECISION = ["odds", "edge_pct", "signal_tier", "kelly_pct", "drift_signal", "generated_at", "ev_pct", "model_prob",
            "model_version", "league", "model_type"]
SETTLE = ["result", "pnl", "closing_odds", "clv_pct"]


def run(name: str):
    path, key = SPECS[name]
    first: dict = {}
    last: dict = {}
    present_last: set = set()
    muts = []
    n_ver = 0
    t0 = time.time()
    for sha, ct, txt in git_versions(path):
        n_ver += 1
        try:
            d = read_csv_text(txt)
        except Exception as e:                                   # a malformed historical version is a finding
            muts.append({"key": None, "commit": sha, "ct": ct, "field": "PARSE_ERROR", "old": "", "new": str(e)[:200]})
            continue
        if "source" not in d.columns:
            d["source"] = "live"
        d["source"] = d["source"].fillna("live")
        d = d[d["source"] == "live"]
        for c in key:
            if c not in d.columns:
                d[c] = ""
        d["match_date"] = d["match_date"].astype(str).str[:10]
        cols = [c for c in DECISION + SETTLE if c in d.columns]
        keys_now = set()
        dup_in_version = d.duplicated(key, keep=False)
        if dup_in_version.any():
            muts.append({"key": None, "commit": sha, "ct": ct, "field": "DUPLICATE_KEYS_IN_VERSION",
                         "old": "", "new": int(dup_in_version.sum())})
        d = d.drop_duplicates(key, keep="first")
        for rec in d[key + cols].to_dict("records"):
            k = tuple(str(rec[c]) for c in key)
            keys_now.add(k)
            if k not in first:
                r = dict(rec)
                r.update({"fs_commit": sha, "fs_commit_ct": ct, "fs_version": n_ver})
                first[k] = r
                last[k] = rec
                continue
            prev = last[k]
            for c in cols:
                a, b = prev.get(c), rec.get(c)
                if (pd.isna(a) and pd.isna(b)) or str(a) == str(b):
                    continue
                muts.append({"key": "|".join(k), "commit": sha, "ct": ct, "field": c,
                             "old": a, "new": b})
            last[k] = rec
        vanished = present_last - keys_now
        for k in vanished:
            muts.append({"key": "|".join(k), "commit": sha, "ct": ct, "field": "ROW_VANISHED", "old": "", "new": ""})
        reappeared = (keys_now - present_last) & set(first) - {k for k, v in first.items() if v["fs_commit"] == sha}
        for k in reappeared:
            muts.append({"key": "|".join(k), "commit": sha, "ct": ct, "field": "ROW_REAPPEARED", "old": "", "new": ""})
        present_last = keys_now
    fs = pd.DataFrame(first.values())
    fs["in_final"] = [k in present_last for k in first]
    fin = pd.DataFrame([{**{f"final_{c}": v for c, v in last[k].items() if c not in key}, **dict(zip(key, k))}
                        for k in first])
    fs = fs.merge(fin, on=key, how="left")
    DATA.mkdir(parents=True, exist_ok=True)
    fs.to_csv(DATA / f"first_sight_{name}.csv", index=False)
    pd.DataFrame(muts).to_csv(DATA / f"ledger_mutations_{name}.csv", index=False)
    print(f"{name}: {n_ver} versions, {len(fs)} live keys, {len(muts)} mutation records, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    with heavy_lock("AU3 first-sight ledger history walk"):
        for nm in (sys.argv[1:] or list(SPECS)):
            run(nm)
