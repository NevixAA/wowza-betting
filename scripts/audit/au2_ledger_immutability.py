"""AU2 Phase 2C - is ledger history ever rewritten? READ-ONLY.

Walks every committed version of output/bets_ledger.csv since 2026-10-18 on the audited commit and
checks, for each live row key (source, match_date, league, home_team, away_team, side), that the
decision fields written at first sight never change in a later version and that no row disappears.
Decision fields: generated_at, odds, edge_pct, signal_tier, model_version, model_type.
Settlement fields (result, pnl, closing_odds, clv_pct) are EXPECTED to fill in later and are only
checked for being overwritten once set.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, AUDITED_COMMIT  # noqa: E402

import pandas as pd  # noqa: E402

KEY = ["source", "match_date", "league", "home_team", "away_team", "side"]
DECISION = ["generated_at", "odds", "edge_pct", "signal_tier", "model_version", "model_type"]
SETTLE = ["result", "pnl"]


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def main() -> int:
    revs = [l.split() for l in git("log", "--reverse", "--since=2026-10-18", "--format=%H %cI",
                                   AUDITED_COMMIT, "--", "output/bets_ledger.csv").split("\n") if l]
    prev = None
    changes = {"versions": len(revs), "rows_disappeared": 0, "decision_field_changes": 0,
               "settlement_overwrites": 0, "examples": []}
    for sha, ts in revs:
        d = pd.read_csv(io.StringIO(git("show", f"{sha}:output/bets_ledger.csv")), low_memory=False)
        d = d[d["source"] == "live"]
        for c in DECISION + SETTLE:
            if c not in d.columns:
                d[c] = pd.NA
        d = d.drop_duplicates(KEY, keep="first").set_index(KEY)
        if prev is not None:
            gone = prev.index.difference(d.index)
            changes["rows_disappeared"] += len(gone)
            if len(gone):
                changes["examples"].append({"commit": sha[:9], "disappeared": [list(map(str, k)) for k in gone[:3]]})
            common = prev.index.intersection(d.index)
            a, b = prev.loc[common], d.loc[common]
            for c in DECISION:
                na_x, na_y = a[c].isna(), b[c].isna()
                x, y = a[c].astype(str), b[c].astype(str)
                ch = (na_x != na_y) | (~na_x & ~na_y & (x != y))
                if ch.any():
                    changes["decision_field_changes"] += int(ch.sum())
                    changes.setdefault("by_commit", []).append({"commit": sha[:9], "commit_ts": ts, "field": c,
                                                                "rows": int(ch.sum())})
                    k = ch[ch].index[0]
                    changes["examples"].append({"commit": sha[:9], "field": c, "key": list(map(str, k)),
                                                "before": x.loc[k], "after": y.loc[k]})
            for c in SETTLE:
                na_x, na_y = a[c].isna(), b[c].isna()
                x, y = a[c].astype(str), b[c].astype(str)
                ow = ~na_x & (na_y | (x != y))
                if ow.any():
                    changes["settlement_overwrites"] += int(ow.sum())
                    k = ow[ow].index[0]
                    changes["examples"].append({"commit": sha[:9], "settle_field": c, "key": list(map(str, k)),
                                                "before": x.loc[k], "after": y.loc[k]})
        prev = d
    changes["examples"] = changes["examples"][:40]
    (DATA / "ledger_immutability.json").write_text(json.dumps(changes, indent=1, default=str), encoding="utf-8")
    print(json.dumps(changes, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
