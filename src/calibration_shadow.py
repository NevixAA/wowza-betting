"""Calibration SHADOW log — v9's probability next to the log-odds-calibrated one. Logging only.

WHY (owner-approved PR, 2026-10-14). Pro's calibration-form study (wowzaV9-Pro
output/studies/CALIBRATION_FORM.md) found that fitting Platt on the LOG-ODDS of a model's raw
probability beats v9's Platt-on-the-raw-probability on 12 of 12 models, team and player. That is a
retrospective result, and changing how every probability is calibrated is an architecture change,
so it runs as a shadow first: every retrain fits both calibrators, production keeps using the old
one, and this module records both numbers at first sight so they can be graded on settled matches.

Nothing here feeds a tier, a stake, a Telegram message or predictions.csv. Every function swallows
its own errors: a shadow log must never be the reason a predict run fails.

Files (append-only, first sight per key, one file per month so no single file grows without bound):
    output/calibration_shadow/team_<YYYY-MM>.csv    fixture x market
    output/calibration_shadow/props_<YYYY-MM>.csv   prop tip (player x market x fixture)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

import config

log = logging.getLogger(__name__)

SHADOW_DIR = config.OUTPUT_DIR / "calibration_shadow"
TEAM_KEY = ["date", "home_team", "away_team", "market"]
PROPS_KEY = ["date", "player_id", "player_name", "market"]


def _append_first_sight(kind: str, rows: pd.DataFrame, key: list[str]) -> int:
    """Append rows whose key has not been logged before (this month's file). Returns rows added."""
    if rows is None or rows.empty:
        return 0
    now = datetime.now(timezone.utc)
    rows = rows.copy()
    rows.insert(0, "logged_at", now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    path = SHADOW_DIR / f"{kind}_{now:%Y-%m}.csv"
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    k = [c for c in key if c in rows.columns]
    rows = rows.drop_duplicates(subset=k, keep="first")
    if path.exists():
        prev = pd.read_csv(path, dtype=str)
        seen = set(map(tuple, prev[k].astype(str).values)) if set(k) <= set(prev.columns) else set()
        rows = rows[[tuple(r) not in seen for r in rows[k].astype(str).values]]
        if rows.empty:
            return 0
        rows = rows.reindex(columns=prev.columns.union(rows.columns, sort=False))
        rows.to_csv(path, mode="a", header=False, index=False, columns=list(prev.columns))
    else:
        rows.to_csv(path, index=False)
    return int(len(rows))


def log_team(feat: pd.DataFrame, payload: dict, payload_newformat: dict | None,
             side_payloads: dict | None) -> None:
    """O/U 2.5 (routed exactly as predict routes it) and the side markets, prod vs shadow."""
    try:
        from src.model import predict_proba_logit_shadow
        std = feat["league"].isin(config.STANDARD_FORMAT_LEAGUES)
        nf = feat["league"].isin(config.NEW_FORMAT_LEAGUES)
        unk = ~std & ~nf
        out = []

        def _add(mask, market, pl, prod_col):
            if pl is None or not mask.any() or prod_col not in feat.columns:
                return
            s = predict_proba_logit_shadow(feat[mask], pl)
            if s is None:                       # model trained before the shadow existed
                return
            d = feat.loc[mask, ["date", "league", "home_team", "away_team", "model_type"]].copy()
            d["market"] = market
            d["p_prod"] = feat.loc[mask, prod_col].round(5).values
            d["p_logit_shadow"] = s.round(5).values
            out.append(d)

        _add(std, "ou25", payload, "p_over25")
        _add(nf, "ou25", payload_newformat if payload_newformat is not None else payload, "p_over25")
        _add(unk, "ou25", payload, "p_over25")
        for target, sp in (side_payloads or {}).items():
            _add(std | nf, target, sp, f"p_{target}")
        if out:
            d = pd.concat(out, ignore_index=True)
            d["date"] = d["date"].astype(str).str[:10]
            n = _append_first_sight("team", d, TEAM_KEY)
            log.info(f"  calibration shadow: {n} new team row(s)")
        else:
            log.info("  calibration shadow: no shadow-capable team model yet (first retrain after merge)")
    except Exception as e:                                           # noqa: BLE001 - never fail predict
        log.warning(f"  calibration shadow (team) skipped: {e}")


def log_props(tips: pd.DataFrame) -> None:
    """Prop tips that carry both raw probabilities (p_raw_prod, p_logit_shadow)."""
    try:
        if tips is None or tips.empty or "p_logit_shadow" not in tips.columns:
            return
        cols = [c for c in ("date", "league", "match", "fixture_id", "player_id", "player_name", "team",
                            "market", "p_raw_prod", "p_logit_shadow", "model_prob") if c in tips.columns]
        d = tips.loc[tips["p_logit_shadow"].notna(), cols].copy()
        if "date" in d.columns:
            d["date"] = d["date"].astype(str).str[:10]
        n = _append_first_sight("props", d, PROPS_KEY)
        print(f"  calibration shadow: {n} new prop row(s)")
    except Exception as e:                                           # noqa: BLE001 - never fail props
        print(f"  calibration shadow (props) skipped: {e}")
