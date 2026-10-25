"""Read-only views over output/system_health.json and output/artifact_manifest.json (System Health page).

Both files are SMALL (~16 KB / ~8 KB) and are written by CI: system_health.json by evidence_refresh.yml
(src/system_health.build_artifact), the manifest by src/artifact_freshness.py. This module only
reshapes what they recorded. It never runs a check, never calls git, never re-derives a status: every
status shown comes from the rule stored beside it in the artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HEALTH = ROOT / "output" / "system_health.json"
MANIFEST = ROOT / "output" / "artifact_manifest.json"

STALE_WORDS = ("STALE", "UNKNOWN", "MISSING")
STALE_LABEL = "STALE — NOT VALID FOR CURRENT DECISIONS"
STATUS_ICON = {"GREEN": "🟢 GREEN", "AMBER": "🟡 AMBER", "RED": "🔴 RED", "NO_DATA": "⚪ NO_DATA",
               "OK": "🟢 OK", "EMPTY": "⚪ EMPTY", "TAMPERED": "🔴 TAMPERED"}


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def load(health: Path | None = None, manifest: Path | None = None) -> tuple[dict, dict]:
    return _json(health or HEALTH), _json(manifest or MANIFEST)


def checks(sh: dict) -> pd.DataFrame:
    """Every check with its value and its explicit rule, RED first."""
    d = pd.DataFrame(sh.get("checks") or [], columns=["status", "check", "value", "rule", "detail"])
    if d.empty:
        return d
    order = {"RED": 0, "NO_DATA": 1, "AMBER": 2, "GREEN": 3}
    d = d.assign(_o=d["status"].map(order).fillna(9)).sort_values(["_o", "check"], kind="mergesort")
    d["status"] = d["status"].map(lambda s: STATUS_ICON.get(s, str(s)))
    return d.drop(columns="_o").reset_index(drop=True)


def check(sh: dict, name: str) -> dict:
    for c in sh.get("checks") or []:
        if c.get("check") == name:
            return c
    return {}


def manifest_table(m: dict) -> pd.DataFrame:
    """One row per tracked artifact; anything STALE / UNKNOWN / MISSING says so in words."""
    cols = ["artifact", "lifecycle", "freshness", "decision_facing", "age_hours", "max_age_hours",
            "generated_at", "owner", "reason"]
    d = pd.DataFrame(m.get("artifacts") or []).reindex(columns=cols)
    if d.empty:
        return d
    stale = d["freshness"].isin(STALE_WORDS)
    d.insert(0, "verdict", d["freshness"].where(~stale, STALE_LABEL))
    d["_ok"] = ~stale                                               # stale rows first
    return d.sort_values(["_ok", "artifact"], kind="mergesort").drop(columns="_ok").reset_index(drop=True)


def first_sight(sh: dict) -> pd.DataFrame:
    rows = [{"store": k, "status": STATUS_ICON.get(v.get("status"), v.get("status")), "rows": v.get("rows"),
             "prior_rows": v.get("prior_rows"), "prior_match": v.get("prior_match"),
             "duplicate_keys": v.get("duplicate_keys"), "post_kickoff_rows": v.get("post_kickoff_rows"),
             "reasons": "; ".join(map(str, v.get("reasons") or []))}
            for k, v in (sh.get("first_sight") or {}).items()]
    return pd.DataFrame(rows)


def contract_violations(sh: dict) -> pd.DataFrame:
    rows = []
    for f, v in ((sh.get("contracts") or {}).get("files") or {}).items():
        for x in v.get("violations") or []:
            rows.append({"file": f, "schema": v.get("schema"), "kind": x.get("kind"), "column": x.get("column"),
                         "rows": x.get("n_rows"), "detail": x.get("detail")})
    return pd.DataFrame(rows)
