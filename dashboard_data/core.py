"""Shared plumbing for the dashboard adapters: repo location, safe reads, freshness."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

V9_DIR = Path(__file__).resolve().parents[1]
_ROOT = V9_DIR.parent

#: Where each repo lives. Pro and v11 are siblings and may simply not be checked out.
REPOS: dict[str, dict] = {
    "v9": {"label": "V9 Production", "colour": "#2E6BE6",
           "candidates": [V9_DIR]},
    "pro": {"label": "Wowza Pro", "colour": "#8B5CF6",
            "candidates": [_ROOT / "wowzaV9-Pro", _ROOT / "v10"]},
    "v11": {"label": "V11 Market Lab", "colour": "#F59E0B",
            "candidates": [_ROOT / "wowza-v11", _ROOT / "wowza_v11", _ROOT / "v11"]},
}

#: Hours after which a feed stops being CURRENT / starts being STALE, by cadence class.
AGE_BANDS = {
    "fast": (2, 12),        # things that run many times a day
    "daily": (30, 72),      # once-a-day jobs, allowing for GitHub's documented lateness
    "weekly": (192, 360),   # weekly research
}


@dataclass(frozen=True)
class Freshness:
    """How old something is, and whether that is acceptable for its cadence."""
    state: str                      # CURRENT | AGING | STALE | FAILED | UNKNOWN
    age_hours: float | None
    label: str
    source: str = ""

    @property
    def emoji(self) -> str:
        return {"CURRENT": "🟢", "AGING": "🟡", "STALE": "🔴",
                "FAILED": "⛔", "UNKNOWN": "⚪"}.get(self.state, "⚪")

    def badge(self) -> str:
        """Colour is never the only cue -- the state word is always present."""
        return f"{self.emoji} {self.state}" + (f" · {self.label}" if self.label else "")


def repo_path(repo: str) -> Path | None:
    """First existing candidate directory for a repo, or None if not checked out."""
    for p in REPOS.get(repo, {}).get("candidates", []):
        if p.exists():
            return p
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def freshness_from_timestamp(ts, cadence: str = "daily", source: str = "") -> Freshness:
    """Classify an age. An UNKNOWN age is treated as old, never as fresh."""
    if ts is None:
        return Freshness("UNKNOWN", None, "no timestamp", source)
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        age = (pd.Timestamp(_now()) - t).total_seconds() / 3600.0
    except Exception:
        return Freshness("UNKNOWN", None, "unparseable timestamp", source)
    if age < 0:
        age = 0.0
    aging, stale = AGE_BANDS.get(cadence, AGE_BANDS["daily"])
    state = "CURRENT" if age <= aging else "AGING" if age <= stale else "STALE"
    label = (f"{age:.0f}h ago" if age >= 1 else f"{age * 60:.0f}m ago")
    if age > 48:
        label = f"{age / 24:.1f}d ago"
    return Freshness(state, round(age, 2), label, source)


def read_csv(path: Path | None, **kw) -> pd.DataFrame:
    """Never raises. A missing or unreadable file is an empty frame, not an exception."""
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()


def read_json(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def newest_date(df: pd.DataFrame, *cols: str):
    """Newest timestamp found in any of `cols` — the content-derived age.

    This is the whole reason the adapters do not stat files. A parquet written by `git checkout`
    five minutes ago can hold data that stopped six weeks back, and mtime says it is fresh.
    """
    if df is None or df.empty:
        return None
    best = None
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_datetime(df[c], errors="coerce", utc=True).dropna()
        if s.empty:
            continue
        m = s.max()
        best = m if best is None or m > best else best
    return best


# ── REMOTE FALLBACK ───────────────────────────────────────────────────────────────────────────
# The dashboard deploys from the v9 repo ALONE, so ../wowzaV9-Pro and ../wowza-v11 exist on a
# developer machine and nowhere else. Deployed, the Pro and V11 pages showed "Not connected" —
# correct behaviour, but it meant two thirds of the estate were invisible wherever the dashboard
# is actually read.
#
# All three repos are public, and Pro already reads v9's committed output over raw HTTP
# (v10/src/data/v9_source.py). This is the same pattern pointed the other way: when a sibling
# checkout is absent, fetch the file from GitHub instead. Local always wins when present, so a
# developer sees their working tree and never a stale copy of it.
RAW_BASE = {
    "pro": "https://raw.githubusercontent.com/NevixAA/wowzaV9-Pro/main",
    "v11": "https://raw.githubusercontent.com/NevixAA/wowza_v11/main",
}
API_BASE = {
    "pro": "https://api.github.com/repos/NevixAA/wowzaV9-Pro",
    "v11": "https://api.github.com/repos/NevixAA/wowza_v11",
}

#: Downloads land here. raw.githubusercontent rate-limits unauthenticated requests, so a fetched
#: file is reused for this long rather than re-pulled on every Streamlit rerun — and a rerun
#: happens on every widget interaction.
_REMOTE_TTL_SECONDS = 30 * 60
_CACHE_DIR = Path(__file__).resolve().parent / "_remote_cache"


def _cache_path(repo: str, rel: str) -> Path:
    return _CACHE_DIR / repo / rel.replace("\\", "/")


def resolve_file(repo: str, rel: str) -> Path | None:
    """Local path for `rel` inside `repo`, fetching it over HTTP if the repo is not checked out.

    Returns None when the file cannot be reached either way — never a placeholder, so callers
    keep reporting "not connected" rather than rendering an empty chart that reads as
    "measured, found nothing".
    """
    local = repo_path(repo)
    if local is not None:
        p = local / rel
        return p if p.exists() else None
    base = RAW_BASE.get(repo)
    if not base:
        return None

    cached = _cache_path(repo, rel)
    if cached.exists():
        import time
        if (time.time() - cached.stat().st_mtime) < _REMOTE_TTL_SECONDS:
            return cached
    try:
        import urllib.request
        req = urllib.request.Request(f"{base}/{rel}",
                                     headers={"User-Agent": "wowza-dashboard"})
        with urllib.request.urlopen(req, timeout=20) as r:
            if r.status != 200:
                return cached if cached.exists() else None
            body = r.read()
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(body)
        return cached
    except Exception:
        # A stale cached copy beats nothing, and the freshness shown to the reader still comes
        # from the file's CONTENT, so an old fetch cannot masquerade as current data.
        return cached if cached.exists() else None


def remote_listing(repo: str, rel_dir: str) -> list[str]:
    """Directory entry names inside a remote repo, via the GitHub contents API.

    Needed because the Pro season store is a directory of date partitions and raw HTTP cannot
    list a directory. Returns [] on any failure; callers treat that as "cannot enumerate",
    not as "empty".
    """
    local = repo_path(repo)
    if local is not None:
        d = local / rel_dir
        return sorted(x.name for x in d.iterdir()) if d.exists() else []
    base = API_BASE.get(repo)
    if not base:
        return []
    try:
        import json as _json
        import urllib.request
        req = urllib.request.Request(f"{base}/contents/{rel_dir}",
                                     headers={"User-Agent": "wowza-dashboard"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return sorted(e["name"] for e in _json.loads(r.read()))
    except Exception:
        return []


def reachable(repo: str) -> bool:
    """True when the repo can be read at all — checked out locally OR fetchable over HTTP."""
    if repo_path(repo) is not None:
        return True
    return repo in RAW_BASE
