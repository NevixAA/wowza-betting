"""Synthetic ROI quarantine (owner P0 deep-repair list 2026-11-22, item 6).

output/player_props_calibration.json (2026-06-24, commit 9e4251705) holds per-market ROI figures
computed against SYNTHETIC implied prices derived from each market's base rate (implied odds ~
1 / base rate), not against any bookmaker quote (reports/MODEL_DATA_AUDIT_2026-08-10.md, M9). The file
is kept for audit and carries an in-file marker; this module is the only reader and refuses it for any
purpose that could turn it into a betting-performance claim.

    from src.integrity.synthetic_roi import load_for_audit, load_quarantined, assert_real_roi_rows
    load_for_audit("output/player_props_calibration.json")       # the content + marker, audit only
    load_quarantined(path, purpose="staking")                     # raises SyntheticROIError

A REAL ROI row must carry every field of REAL_ROI_FIELDS: the real entry odds, the bookmaker, the
capture time (before kickoff when the kickoff is known), the market, the line and the settlement.
assert_real_roi_rows refuses a frame that misses one, or that carries a synthetic price label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARKER = "INVALID_FOR_REAL_BETTING_ROI"
MARKER_KEY = "_quarantine"
#: quarantined file (repo-relative) -> why
QUARANTINED = {
    "output/player_props_calibration.json":
        "ROI computed against synthetic implied prices derived from base rates (implied odds ~ 1 / base rate), "
        "never a bookmaker quote; in-sample; reports/MODEL_DATA_AUDIT_2026-08-10.md M9",
}
#: purposes a quarantined file may never feed
REFUSED_PURPOSES = ("betting_performance", "roi", "staking", "bankroll", "promotion", "edge_validation",
                    "dashboard_profitability", "dashboard_ranking", "telegram", "pro_research_roi")
AUDIT = "audit"
#: every field a real ROI row needs (owner item 6)
REAL_ROI_FIELDS = ("entry_odds", "bookmaker", "capture_time", "market", "line", "settlement")
SETTLED = ("WIN", "LOSS", "PUSH", "HALF_WIN", "HALF_LOSS", "VOID")
SYNTHETIC_LABELS = ("SYNTHETIC_PRICE_DIAGNOSTIC_ONLY", MARKER, "SYNTHETIC")


class SyntheticROIError(PermissionError):
    """A quarantined synthetic-ROI file, or a row without a real price record, would reach a real ROI."""


def _rel(path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(ROOT.resolve())
        except ValueError:
            pass
    return str(p).replace("\\", "/")


def is_quarantined(path) -> bool:
    return _rel(path) in QUARANTINED


def quarantine_block(path: str) -> dict:
    return {"status": MARKER, "reason": QUARANTINED[path], "usable_for": [AUDIT],
            "refused_for": list(REFUSED_PURPOSES), "decided": "owner P0 deep-repair list 2026-11-22, item 6",
            "reader": "src/integrity/synthetic_roi.py (load_for_audit only)"}


def load_for_audit(path) -> dict:
    """The file's content for audit display / investigation. Requires the in-file marker."""
    p = ROOT / _rel(path) if not Path(path).is_absolute() else Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if (d.get(MARKER_KEY) or {}).get("status") != MARKER:
        raise SyntheticROIError(f"{_rel(path)}: quarantine marker {MARKER} missing - refused")
    return d


def load_quarantined(path, purpose: str):
    """Refuses every purpose except 'audit' (then = load_for_audit)."""
    if purpose == AUDIT:
        return load_for_audit(path)
    raise SyntheticROIError(f"{_rel(path)} is {MARKER}: refused for '{purpose}' "
                            f"({QUARANTINED.get(_rel(path), 'quarantined')})")


def assert_not_quarantined(path, purpose: str) -> None:
    """For any loader that might be pointed at an arbitrary file (dashboards, notifiers, gates)."""
    if is_quarantined(path):
        load_quarantined(path, purpose)


def real_roi_problems(df: pd.DataFrame, kickoff_col: str | None = "kickoff_utc") -> pd.Series:
    """Per row, '' when it may enter a real ROI, else the reasons (';'-joined)."""
    probs = pd.Series("", index=df.index, dtype=object)

    def add(mask, why):
        nonlocal probs
        mask = pd.Series(mask, index=df.index).fillna(True).astype(bool)
        probs = probs.where(~mask, probs + np.where(probs == "", "", ";") + why)

    for f in REAL_ROI_FIELDS:
        if f not in df.columns:
            add(pd.Series(True, index=df.index), f"no {f}")
    if "entry_odds" in df.columns:
        o = pd.to_numeric(df["entry_odds"], errors="coerce")
        add(~(o > 1.0), "entry_odds not a price > 1")
    for f in ("bookmaker", "market"):
        if f in df.columns:
            add(df[f].isna() | df[f].astype(str).str.strip().isin(("", "nan", "None")), f"{f} missing")
    if "line" in df.columns:
        add(df["line"].isna() | df["line"].astype(str).str.strip().isin(("", "nan", "None")), "line missing")
    if "settlement" in df.columns:
        add(~df["settlement"].astype(str).str.upper().isin(SETTLED), "not settled")
    if "capture_time" in df.columns:
        ct = pd.to_datetime(df["capture_time"], errors="coerce", utc=True)
        add(ct.isna(), "capture_time missing")
        if kickoff_col and kickoff_col in df.columns:
            ko = pd.to_datetime(df[kickoff_col], errors="coerce", utc=True)
            add(ct.notna() & ko.notna() & (ct >= ko), "captured at/after kickoff")
    for col in ("price_provenance", "profitability_label", "roi_status"):
        if col in df.columns:
            add(df[col].astype(str).isin(SYNTHETIC_LABELS) | (df[col].astype(str) == "INVALID_FOR_PROFITABILITY"),
                f"synthetic ({col})")
    if "price_synthetic" in df.columns:
        add(df["price_synthetic"].astype(str).str.lower().isin(("true", "1", "yes")), "price_synthetic")
    return probs


def assert_real_roi_rows(df: pd.DataFrame, what: str = "real ROI", kickoff_col: str | None = "kickoff_utc") -> None:
    p = real_roi_problems(df, kickoff_col)
    bad = p[p != ""]
    if len(bad):
        raise SyntheticROIError(f"{len(bad)} row(s) cannot enter {what}: {sorted(set(';'.join(bad).split(';')))[:8]}")


def calibration_rows_as_roi_frame(d: dict) -> pd.DataFrame:
    """The quarantined calibration file's per-market rows as a frame (for the refusal test/audit)."""
    return pd.DataFrame([{"market": k, **v} for k, v in d.items() if not k.startswith("_") and isinstance(v, dict)])
