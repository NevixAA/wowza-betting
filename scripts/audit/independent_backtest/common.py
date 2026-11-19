"""Shared helpers: git blob streaming, HEAVY.lock, cache paths. No production imports."""
from __future__ import annotations

import io
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REF = os.environ.get("AU5_REF", "fb0e9697ec8556d4dd3df57f31f8ed5415a5012d")
CACHE = Path(os.environ.get("AU5_CACHE", str(REPO / "scripts" / "audit" / "independent_backtest" / "_cache")))
HEAVY_LOCK = Path(os.environ.get("AU5_HEAVY_LOCK", r"C:\Users\nevo.APOLLO\dev\wowza\r2_brief\HEAVY.lock"))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def file_commits(path: str, ref: str = REF) -> list[tuple[str, str]]:
    """(sha, committer ISO time) oldest first, for every commit reachable from ref touching path."""
    out = git("log", "--reverse", "--format=%H %cI", ref, "--", path)
    rows = []
    for line in out.splitlines():
        sha, ts = line.split(" ", 1)
        rows.append((sha, ts.strip()))
    return rows


def iter_blobs(specs: list[str]):
    """Yield (spec, bytes|None) for each '<sha>:<path>' using one git cat-file --batch process."""
    p = subprocess.Popen(["git", "-C", str(REPO), "cat-file", "--batch"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    try:
        for spec in specs:
            p.stdin.write((spec + "\n").encode())
            p.stdin.flush()
            header = p.stdout.readline().decode().strip()
            if header.endswith("missing"):
                yield spec, None
                continue
            size = int(header.split()[2])
            data = p.stdout.read(size)
            p.stdout.read(1)  # trailing LF
            yield spec, data
    finally:
        p.stdin.close()
        p.wait()


def show(sha: str, path: str) -> bytes | None:
    for _, data in iter_blobs([f"{sha}:{path}"]):
        return data
    return None


@contextmanager
def heavy_lock(name: str = "AU5 independent_backtest", poll_s: int = 60, stale_min: int = 45):
    """Exclusive HEAVY.lock per the audit brief: open(path,'x'), retry every 60 s, stale after 45 min."""
    while True:
        try:
            with open(HEAVY_LOCK, "x", encoding="utf-8") as fh:
                fh.write(f"{name} {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
            break
        except FileExistsError:
            try:
                age = time.time() - HEAVY_LOCK.stat().st_mtime
                if age > stale_min * 60:
                    HEAVY_LOCK.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            print(f"[lock] HEAVY.lock held ({HEAVY_LOCK.read_text(errors='replace')!r}); retry in {poll_s}s",
                  flush=True)
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            HEAVY_LOCK.unlink()
        except FileNotFoundError:
            pass


def read_csv_bytes(data: bytes, **kw):
    import pandas as pd
    return pd.read_csv(io.BytesIO(data), low_memory=False, **kw)
