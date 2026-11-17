"""AU3 audit helpers (read-only). Nothing here imports production selection or settlement code.

Shared by scripts/audit/au3_*.py:
  * heavy_lock()      - the laptop-safety lock from the audit brief (exclusive create, stale after 45 min)
  * git_versions()    - every committed version of a file, oldest first, via `git log --raw` + `cat-file --batch`
  * roi_ci()          - flat-stake ROI with an i.i.d. bootstrap and a date-block bootstrap
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LOCK = Path(r"C:\Users\nevo.APOLLO\dev\wowza\r2_brief\HEAVY.lock")
AUDIT_REF = "fb0e9697ec8556d4dd3df57f31f8ed5415a5012d"
DATA = REPO / "reports" / "au3_data"
LIVE_FROM = "2026-09-01"


@contextlib.contextmanager
def heavy_lock(name: str = "AU3", stale_min: int = 45, retry_s: int = 60):
    while True:
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{name} {datetime.now(timezone.utc).isoformat(timespec='seconds')}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = (time.time() - LOCK.stat().st_mtime) / 60
                if age > stale_min:
                    LOCK.unlink()
                    continue
            except FileNotFoundError:
                continue
            print(f"[lock] held by {LOCK.read_text(errors='replace')!r}; retry in {retry_s}s", flush=True)
            time.sleep(retry_s)
    try:
        yield
    finally:
        try:
            if LOCK.read_text(errors="replace").startswith(name):
                LOCK.unlink()
        except FileNotFoundError:
            pass


def git_versions(path: str, ref: str = AUDIT_REF):
    """Yield (commit_sha, commit_unix_time, text) for every commit that changed `path`, oldest first.
    Merge commits are skipped by `git log --raw` (the change is already on the parent branch)."""
    out = subprocess.run(["git", "-C", str(REPO), "log", "--no-abbrev", "--format=C %H %ct", "--raw", ref, "--", path],
                         capture_output=True, text=True, check=True).stdout
    items, cur = [], None
    for line in out.splitlines():
        if line.startswith("C "):
            _, sha, ct = line.split()
            cur = (sha, int(ct))
        elif line.startswith(":") and cur:
            parts = line.split()
            blob, status = parts[3], parts[4]
            if status.startswith("D") or set(blob) == {"0"}:
                continue
            items.append((cur[0], cur[1], blob))
    items.sort(key=lambda x: x[1])
    p = subprocess.Popen(["git", "-C", str(REPO), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for sha, ct, blob in items:
            p.stdin.write((blob + "\n").encode()); p.stdin.flush()
            header = p.stdout.readline().decode().split()
            size = int(header[2])
            data = p.stdout.read(size); p.stdout.read(1)
            yield sha, ct, data.decode("utf-8", errors="replace")
    finally:
        p.stdin.close(); p.wait()


def git_show(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"], capture_output=True)
    return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else None


def read_csv_text(txt: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(txt), dtype=str, keep_default_na=False, na_values=[""])


def roi_ci(pnl, dates=None, n_boot: int = 4000, seed: int = 7, alpha: float = 0.05) -> dict:
    """Flat 1u ROI = sum(pnl)/n on settled (WIN/LOSS) rows. CI: i.i.d. bootstrap and, when `dates`
    is given, a block bootstrap that resamples whole match DATES (correlated same-day bets move together)."""
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    if n == 0:
        return {"n": 0, "pnl": 0.0, "roi": np.nan, "ci_iid": (np.nan, np.nan), "ci_day": (np.nan, np.nan)}
    rng = np.random.default_rng(seed)
    iid = [pnl[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(iid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out = {"n": n, "pnl": round(float(pnl.sum()), 2), "roi": float(pnl.mean()),
           "ci_iid": (float(lo), float(hi)), "ci_day": (np.nan, np.nan)}
    if dates is not None:
        d = pd.Series(np.asarray(dates)).astype(str).values
        u = np.unique(d)
        groups = [pnl[d == x] for x in u]
        sums = np.array([g.sum() for g in groups]); cnts = np.array([len(g) for g in groups])
        bs = []
        for _ in range(n_boot):
            k = rng.integers(0, len(u), len(u))
            bs.append(sums[k].sum() / cnts[k].sum())
        lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["ci_day"] = (float(lo), float(hi))
        out["n_days"] = int(len(u))
    return out


def fmt_roi(r: dict) -> str:
    if not r["n"]:
        return "n 0"
    s = f"n {r['n']}, {r['pnl']:+.2f}u, ROI {100*r['roi']:+.1f}%"
    if not np.isnan(r["ci_day"][0]):
        s += f" [day-block 95% {100*r['ci_day'][0]:+.1f}, {100*r['ci_day'][1]:+.1f}]"
    else:
        s += f" [iid 95% {100*r['ci_iid'][0]:+.1f}, {100*r['ci_iid'][1]:+.1f}]"
    return s
