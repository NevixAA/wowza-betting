"""Row-level integrity validators for the independent backtest. Every check RAISES AuditViolation on a
broken row; nothing is defaulted, imputed or swallowed. No production imports.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .results import LINES


class AuditViolation(ValueError):
    """A row that must not enter a profitability calculation."""


def _fin(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def check_priced(bet: dict) -> None:
    o = _fin(bet.get("odds"))
    if not (math.isfinite(o) and o > 1.0):
        raise AuditViolation(f"missing/invalid quoted odds: {bet.get('odds')!r}")
    if bet.get("price_source") in ("synthetic", "imputed", "default", "consensus_estimate"):
        raise AuditViolation(f"non-executable price source {bet.get('price_source')!r}")


def check_pre_kickoff(quote_ts, kickoff_utc) -> None:
    if kickoff_utc is None or str(kickoff_utc) in ("", "nan", "None"):
        raise AuditViolation("kickoff unknown - cannot prove quote is pre-match")
    q, k = pd.Timestamp(str(quote_ts)), pd.Timestamp(str(kickoff_utc))
    q = q.tz_localize("UTC") if q.tzinfo is None else q.tz_convert("UTC")
    k = k.tz_localize("UTC") if k.tzinfo is None else k.tz_convert("UTC")
    if q >= k:
        raise AuditViolation(f"post-kickoff quote {q} >= kickoff {k}")


def check_under_probability(p_over, p_under, tol: float = 1e-6) -> None:
    po, pu = _fin(p_over), _fin(p_under)
    if not (0 < po < 1 and 0 < pu < 1):
        raise AuditViolation("probability outside (0,1)")
    if abs(po + pu - 1.0) > tol:
        raise AuditViolation(f"p_over + p_under = {po + pu:.6f} != 1 (inverted or mis-mapped under)")


def check_market_line(market: str, line) -> None:
    m = str(market).lower()
    if m not in LINES:
        if m == "btts" and (line is None or str(line) in ("", "nan")):
            return
        raise AuditViolation(f"unknown goal-line market {market!r}")
    if _fin(line) != LINES[m]:
        raise AuditViolation(f"market {market} carries line {line}, expected {LINES[m]}")


def check_side(market: str, side: str) -> None:
    m, s = str(market).lower(), str(side).upper()
    ok = {"ou25": {"OVER", "UNDER"}, "over15": {"OVER", "UNDER"}, "over35": {"OVER", "UNDER"},
          "btts": {"YES", "NO"}}.get(m)
    if ok is None or s not in ok:
        raise AuditViolation(f"invalid side {side!r} for market {market!r}")


def check_no_duplicates(bets: pd.DataFrame, key=("fixture_id", "market", "side")) -> None:
    dup = bets.duplicated(list(key), keep=False)
    if dup.any():
        raise AuditViolation(f"{int(dup.sum())} duplicate selections on {key}")


def check_one_direction_per_fixture(bets: pd.DataFrame) -> None:
    gl = bets[bets["market"].isin(["ou25", "over15", "over35"])]
    n = gl.groupby("fixture_id")["side"].nunique()
    if (n > 1).any():
        raise AuditViolation(f"opposite goal directions on {int((n > 1).sum())} fixture(s)")


def check_result_orientation(status: str) -> None:
    if status == "SWAPPED":
        raise AuditViolation("result found only with home/away swapped - orientation unverified")
    if status in ("AMBIGUOUS", "CONFLICT"):
        raise AuditViolation(f"result {status}")


def check_model_artifact(path) -> None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise AuditViolation(f"model artifact missing: {p}")


def check_feature_order(model_features, frame_columns) -> None:
    mf, fc = list(model_features), list(frame_columns)
    missing = [c for c in mf if c not in fc]
    if missing:
        raise AuditViolation(f"serving frame lacks {len(missing)} model feature(s): {missing[:5]}")
    sub = [c for c in fc if c in set(mf)]
    if sub != mf:
        raise AuditViolation("feature ORDER differs between model and serving frame")


def settle_player_prop(bet: dict, stats: pd.DataFrame) -> str | None:
    """Settle a player prop by (player_id, fixture_id) - never by name alone.
    Returns WIN / LOSS / VOID (did not play) / None (stats missing -> unsettled, never a loss)."""
    pid, fid = bet.get("player_id"), bet.get("fixture_id")
    if pid is None or str(pid) in ("", "nan") or fid is None or str(fid) in ("", "nan"):
        raise AuditViolation("player prop without player_id + fixture_id cannot be settled by name")
    rows = stats[(stats["player_id"].astype(str) == str(pid)) & (stats["fixture_id"].astype(str) == str(fid))]
    if rows.empty:
        return None
    if len(rows) > 1:
        raise AuditViolation("duplicate player stat rows for one player-fixture")
    r = rows.iloc[0]
    mins = _fin(r.get("minutes"))
    if not math.isfinite(mins):
        return None
    if mins <= 0:
        return "VOID"
    stat = _fin(r.get(bet["stat"]))
    if not math.isfinite(stat):
        return None
    return "WIN" if stat > float(bet["line"]) else "LOSS"


def check_player_identity(bet: dict, stats: pd.DataFrame) -> None:
    """A name that appears on more than one fixture on the bet's date is not an identity."""
    same = stats[(stats["player_name"] == bet.get("player")) & (stats["date"] == bet.get("date"))]
    if same["fixture_id"].nunique() > 1 and not bet.get("player_id"):
        raise AuditViolation(f"player name {bet.get('player')!r} on {same['fixture_id'].nunique()} fixtures")
