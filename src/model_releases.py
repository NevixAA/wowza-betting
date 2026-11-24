"""Readable model-version names: "<release name> · <install date> · <status>" (owner 2026-11-23).

    from src.model_releases import Releases
    R = Releases()                                                     # config/model_release_registry.json
    R.display("v9_standard", "btts", "btts_20261119T134019Z_44c67049", kickoff)
    # 'Go-live refit · 19 Nov · serving at kickoff'

The registry is GENERATED from the install history (scripts/build_model_release_registry.py); nothing here is typed
by hand. Matching a ledger version to an install record:
  * exact version id (v9 team, side models, Pro), else
  * props / props challenger: the same market and the same 8-hex ARTIFACT digest (the ledger writer names the
    version with its own recipe word; the digest names the bytes).
No record -> "unknown release · <raw id>" (never a guess). Status at the fixture's kickoff:
  serving at kickoff   a champion window of the version holds the kickoff
  retired              the version served before, but not at this kickoff
  candidate            never served at or before this kickoff (a challenger, or not installed yet)
Display only: the raw model_version stays the key everywhere. Pure (one JSON read), reload-safe.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "model_release_registry.json"
UNKNOWN = "unknown release"
SERVING, RETIRED, CANDIDATE = "serving at kickoff", "retired", "candidate"
_PROPS_MODELS = {"v9_props": "props:", "v9_props_challenger": "props_challenger:"}


def _day(iso) -> str:
    t = pd.Timestamp(iso)
    return f"{t.day} {t.strftime('%b')}"


class Releases:
    def __init__(self, path: Path | None = None):
        p = Path(path or REGISTRY)
        doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"versions": []}
        self.rows = doc.get("versions", [])
        self._memo: dict = {}

    def records(self, model_id: str, market: str, version: str) -> list[dict]:
        """Install records (one per serving window) of a ledger version; [] when none."""
        k = (str(model_id), str(market), str(version))
        if k in self._memo:
            return self._memo[k]
        v = str(version)
        hit = [r for r in self.rows if r["version_id"] == v]
        pref = _PROPS_MODELS.get(str(model_id))
        if pref:
            m = re.search(r"_([0-9a-f]{8})$", v)
            slot = f"{pref}{market}"
            hit = [r for r in hit if r["slot"] == slot] or (
                [r for r in self.rows if r["slot"] == slot and m and r.get("artifact8") == m.group(1)] if m else [])
        self._memo[k] = hit
        return hit

    def name(self, model_id: str, market: str, version: str) -> tuple[str, str]:
        """(release name, install date 'D Mon'); ('unknown release', '') without a record or a name."""
        rec = self.records(model_id, market, version)
        if not rec or not rec[0].get("release_name"):
            return UNKNOWN, ""
        return rec[0]["release_name"], _day(min(r["installed_at"] for r in rec))

    def status(self, model_id: str, market: str, version: str, kickoff) -> str:
        rec = self.records(model_id, market, version)
        ko = pd.to_datetime(kickoff, utc=True, errors="coerce")
        if not rec or pd.isna(ko):
            return ""
        champ = [r for r in rec if r.get("role", "champion") == "champion"]
        for r in champ:
            a = pd.Timestamp(r["served_from"])
            b = pd.Timestamp(r["served_to"]) if r.get("served_to") else None
            if a <= ko and (b is None or ko < b):
                return SERVING
        if any(r.get("served_to") and pd.Timestamp(r["served_to"]) <= ko for r in champ):
            return RETIRED
        return CANDIDATE

    def display(self, model_id: str, market: str, version: str, kickoff=None) -> str:
        name, day = self.name(model_id, market, version)
        if name == UNKNOWN:
            return f"{UNKNOWN} · {version}"
        st = self.status(model_id, market, version, kickoff) if kickoff is not None else ""
        return " · ".join(x for x in (name, day, st) if x)
