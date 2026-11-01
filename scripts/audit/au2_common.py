"""AU2 audit helpers: HEAVY.lock, offline history load (committed caches only), paths. READ-ONLY."""
from __future__ import annotations

import contextlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "reports" / "au2_data"
LOCK = Path(r"C:\Users\nevo.APOLLO\dev\wowza\r2_brief\HEAVY.lock")
CACHE = Path(os.getenv("AU2_CACHE", str(Path(os.getenv("TEMP", "/tmp")) / "au2_cache")))
AUDITED_COMMIT = "fb0e9697ec8556d4dd3df57f31f8ed5415a5012d"

# No network enrichment, no keys: the audit must reproduce from committed files only.
for _k in ("APIFOOTBALL_KEY", "API_FOOTBALL_KEY", "ODDS_API_KEY", "RAPIDAPI_KEY"):
    os.environ[_k] = ""


@contextlib.contextmanager
def heavy_lock(name: str = "AU2", stale_min: int = 45, retry_s: int = 60):
    """Exclusive laptop lock per the audit brief (open(path,'x'), retry 60 s, stale after 45 min)."""
    while True:
        try:
            with open(LOCK, "x", encoding="utf-8") as f:
                f.write(f"{name} {datetime.now(timezone.utc).isoformat()}\n")
            break
        except FileExistsError:
            try:
                age = (time.time() - LOCK.stat().st_mtime) / 60
                if age > stale_min:
                    LOCK.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            print(f"[lock] HEAVY.lock held ({LOCK.read_text(errors='ignore').strip()}); retry in {retry_s}s",
                  flush=True)
            time.sleep(retry_s)
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


def load_history_offline():
    """v9's own loader with the football-data download disabled: committed fd_history cache +
    committed backtest archive / af_history, exactly the inputs on the audited commit. The raw
    frame is what predict receives (NOT COVID-filtered); training filters COVID itself."""
    import pandas as pd
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / "history_raw.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    from src import data_loader as dl
    dl._ci_download_all = lambda have=None: []          # noqa: E731 - offline: committed cache only
    raw = dl.load_all_matches()
    raw.to_parquet(cp)
    return raw
