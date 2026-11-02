"""Walk the git history of the ledgers and list rows that existed in one commit and vanished later.

    python scripts/check_ledger_history.py [--ref origin/main] [--max-commits 120] [--json out.json]

READ-ONLY. Built for the ledger-race finding (estate scan 2026-11-02, P1): predict, update_results,
daily_summary, player_props, sharp_tracker and worldcup published ledgers with
`git pull --rebase -X theirs`, which drops another run's rows in a conflicting hunk. This measures
how often that actually cost a row.

How (light by design - a laptop must survive it): the commits that touched each ledger are SAMPLED
evenly (first and last always in); between two consecutive samples the set of row keys that
disappeared is bisected down to the exact commit that removed them, so only lossy intervals cost
extra reads. A row that appears AND disappears between two samples is not seen (raise
--max-commits to look closer).

Rows are identified by the SAME natural keys the ledger sync merges on (src.ledger.SYNC_LEDGERS).
Each vanished row is classified:
  * intentional_backtest_purge - source=backtest removed on/after 2026-11-01 (the owner's removal
                                 of 4,115 backtest rows; intentional);
  * rekeyed     - the removing commit holds a row with the same fingerprint under another key
                  (a name repair, a player_id back-fill): the row moved, it was not lost;
  * reappeared  - the key is present again at the tip of --ref: lost for a while, back now;
  * lost        - none of the above. Split by who removed it: an `auto:` bot commit (the race) or
                  a hand-made commit (a repair - read its subject before calling it a loss).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ledger import SYNC_LEDGERS, ledger_row_key  # noqa: E402

LEDGERS = ["output/bets_ledger.csv", "output/side_bets_ledger.csv",
           "output/player_ledger.csv", "output/ht_ledger.csv"]
# A row's identity apart from the names a repair may rewrite: when the commit that removed a key
# holds a row with the same fingerprint under another key, the row was re-keyed, not lost.
FINGERPRINT = {
    "output/bets_ledger.csv":      ("source", "generated_at", "match_date", "side", "odds"),
    "output/side_bets_ledger.csv": ("source", "generated_at", "match_date", "market", "odds"),
    "output/ht_ledger.csv":        ("source", "generated_at", "match_date", "market", "model_prob"),
    "output/sharp_ledger.csv":     ("signal_date", "match_date", "market_label", "opening_odds"),
    "output/player_ledger.csv":    ("signal_date", "match_date", "market", "model_prob", "market_odds"),
}
PURGE_FROM = "2026-11-01"


class Blobs:
    """`git cat-file --batch` reader: one process for every blob read."""

    def __init__(self, repo: Path):
        self.p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def read(self, sha: str, path: str) -> str | None:
        self.p.stdin.write(f"{sha}:{path}\n".encode()); self.p.stdin.flush()
        head = self.p.stdout.readline().decode().split()
        if len(head) < 3 or head[1] == "missing":
            return None
        data = self.p.stdout.read(int(head[2])); self.p.stdout.read(1)
        return data.decode("utf-8", errors="replace")

    def close(self):
        self.p.stdin.close(); self.p.wait()


def parse(text: str | None, key: tuple, fp: tuple) -> dict:
    """{key: (fingerprint, source)} for one version of a ledger (first occurrence of a key)."""
    out = {}
    if not text:
        return out
    for rec in csv.DictReader(io.StringIO(text)):
        k = ledger_row_key(rec, key)
        if k not in out:
            out[k] = (tuple((rec.get(c) or "").strip() for c in fp), (rec.get("source") or "").strip())
    return out


def history(repo: Path, ref: str, path: str) -> list[tuple[str, str, str]]:
    r = subprocess.run(["git", "-C", str(repo), "log", "--reverse", "--format=%H%x09%cI%x09%s",
                        ref, "--", path], check=True, capture_output=True, text=True, encoding="utf-8")
    return [tuple(l.split("\t", 2)) for l in r.stdout.splitlines() if l.strip()]


def sample(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def check_ledger(repo: Path, ref: str, path: str, max_commits: int, blobs: Blobs) -> dict:
    key, fp = SYNC_LEDGERS[path], FINGERPRINT[path]
    commits = history(repo, ref, path)
    if not commits:
        return {"ledger": path, "commits": 0}
    idx = sample(len(commits), max_commits)
    cache: dict[int, dict] = {}

    def rows(i: int) -> dict:
        if i not in cache:
            cache[i] = parse(blobs.read(commits[i][0], path), key, fp)
        return cache[i]

    tip = parse(blobs.read(ref, path), key, fp)
    events: list[tuple[int, set]] = []

    def bisect(i: int, j: int, gone: set) -> None:
        if not gone:
            return
        if j == i + 1:
            events.append((j, gone))
            return
        m = (i + j) // 2
        at_m = rows(m)
        bisect(i, m, {k for k in gone if k not in at_m})
        bisect(m, j, {k for k in gone if k in at_m})

    for a, b in zip(idx, idx[1:]):
        ra, rb = rows(a), rows(b)
        bisect(a, b, {k for k in ra if k not in rb})
        for i in list(cache):                 # keep only the next interval's start in memory
            if i != b:
                del cache[i]

    cats = Counter()
    lost_by_commit = defaultdict(lambda: Counter())
    examples = defaultdict(list)
    for j, gone in events:
        sha, date, subj = commits[j]
        before = parse(blobs.read(commits[j - 1][0], path), key, fp)
        after = parse(blobs.read(sha, path), key, fp)
        new_fps = Counter(v[0] for k, v in after.items() if k not in before)
        bot = subj.startswith("auto:")
        for k in gone:
            fpv, src = before.get(k, ((), ""))
            if src == "backtest" and date[:10] >= PURGE_FROM:
                c = "intentional_backtest_purge"
            elif new_fps.get(fpv, 0) > 0:
                new_fps[fpv] -= 1
                c = "rekeyed"
            elif k in tip:
                c = "reappeared"
            else:
                c = "lost_in_bot_commit" if bot else "lost_in_manual_commit"
            cats[c] += 1
            if c.startswith("lost") or c == "reappeared":
                lost_by_commit[(sha[:10], date[:16], subj[:90])][c] += 1
                if len(examples[c]) < 8:
                    examples[c].append({"commit": sha[:10], "date": date[:10], "key": list(k)})
    return {"ledger": path, "commits": len(commits), "sampled": len(idx),
            "removal_commits": len(events), "vanished": dict(cats),
            "by_commit": [{"commit": c[0], "date": c[1], "subject": c[2], **dict(v)}
                          for c, v in sorted(lost_by_commit.items(), key=lambda kv: -sum(kv[1].values()))],
            "examples": dict(examples)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--max-commits", type=int, default=120,
                    help="commits sampled per ledger (default 120; more = closer look, slower)")
    ap.add_argument("--ledger", action="append", help="limit to these paths (repeatable)")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--repo", default=str(ROOT))
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    blobs = Blobs(repo)
    out = []
    try:
        for path in a.ledger or LEDGERS:
            r = check_ledger(repo, a.ref, path, a.max_commits, blobs)
            out.append(r)
            print(f"{path}: {r.get('commits', 0)} commits, {r.get('sampled', 0)} sampled, "
                  f"{r.get('removal_commits', 0)} removing commit(s); vanished rows: {r.get('vanished', {})}")
            for row in r.get("by_commit", [])[:10]:
                n = {k: v for k, v in row.items() if k not in ("commit", "date", "subject")}
                print(f"    {row['commit']} {row['date']} {n}  {row['subject']}")
    finally:
        blobs.close()
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
