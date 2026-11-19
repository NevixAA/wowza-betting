"""AU4 forensic audit helpers (read-only; imports NO production selection/settlement code).

Shared by scripts/audit/au4_*.py:
  * heavy_lock()        exclusive laptop-safety lock (r2_brief/HEAVY.lock, 45-min stale rule)
  * read_rolling(name)  live file + <stem>_archive/*.csv, exact duplicates dropped
  * read_dir(name)      every CSV in an output sub-directory
  * block_bootstrap()   cluster (fixture / day) bootstrap CI for a mean
  * summarise()         mean / median / trimmed / winsorized / %>0 / %==0 / %<0
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
DATA = ROOT / "reports" / "au4_data"
LOCK = Path(r"C:\Users\nevo.APOLLO\dev\wowza\r2_brief\HEAVY.lock")
STALE_S = 45 * 60


@contextlib.contextmanager
def heavy_lock(name: str = "AU4"):
    """Acquire HEAVY.lock exclusively (open 'x'); retry every 60 s; a lock older than 45 min is
    stale and removed. Always deleted in finally."""
    while True:
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"{name} {_dt.datetime.now(_dt.timezone.utc).isoformat()}\n")
            break
        except FileExistsError:
            try:
                age = time.time() - LOCK.stat().st_mtime
                if age > STALE_S:
                    LOCK.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            print(f"[au4] HEAVY.lock held ({LOCK.read_text(errors='ignore').strip()}); retry in 60s",
                  flush=True)
            time.sleep(60)
    try:
        yield
    finally:
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass


def _read(p: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False, **kw)


def read_rolling(name: str, **kw) -> pd.DataFrame:
    """output/<name> plus output/<stem>_archive/*.csv; exact duplicate rows dropped."""
    live = OUT / name
    arch = OUT / f"{Path(name).stem}_archive"
    parts = []
    if arch.is_dir():
        parts += [_read(p, **kw).assign(_part=p.name) for p in sorted(arch.glob("*.csv"))]
    if live.exists():
        parts.append(_read(live, **kw).assign(_part="live"))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    cols = [c for c in d.columns if c != "_part"]
    return d.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def read_dir(sub: str, **kw) -> pd.DataFrame:
    parts = [_read(p, **kw).assign(_part=p.name) for p in sorted((OUT / sub).glob("*.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def utc(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


def summarise(x, trim: float = 0.05) -> dict:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0}
    lo, hi = np.quantile(x, [trim, 1 - trim])
    xs = np.sort(x)
    k = int(np.floor(n * trim))
    trimmed = xs[k:n - k].mean() if n - 2 * k > 0 else np.nan
    return {"n": n, "mean": x.mean(), "median": float(np.median(x)), f"trim{int(trim*100)}": trimmed,
            f"wins{int(trim*100)}": np.clip(x, lo, hi).mean(),
            "pct_pos": (x > 0).mean(), "pct_zero": (x == 0).mean(), "pct_neg": (x < 0).mean(),
            "p01": np.quantile(x, .01), "p99": np.quantile(x, .99), "min": x.min(), "max": x.max()}


def block_bootstrap(values, clusters, stat=np.mean, n_boot: int = 2000, seed: int = 7,
                    alpha: float = 0.05):
    """Percentile CI resampling whole clusters (fixture or day) with replacement."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    c = pd.Series(clusters).astype(str).to_numpy()
    ok = ~np.isnan(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0)
    codes, uniq = pd.factorize(c)
    groups = [v[codes == i] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    G = len(groups)
    est = []
    for _ in range(n_boot):
        idx = rng.integers(0, G, G)
        est.append(stat(np.concatenate([groups[i] for i in idx])))
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return (float(stat(v)), float(lo), float(hi), G)


def save_json(obj, name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / name

    def _d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)
    p.write_text(json.dumps(obj, indent=1, default=_d, ensure_ascii=False), encoding="utf-8")
    return p


def save_csv(df: pd.DataFrame, name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / name
    df.to_csv(p, index=False, lineterminator="\n")
    return p
