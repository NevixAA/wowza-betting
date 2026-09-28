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
