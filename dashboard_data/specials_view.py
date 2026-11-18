"""Specials (research) page helpers - pure frame functions over Pro's committed specials files.

Pro writes (src/specials, owner 2026-11-09):
  specials_log.csv       one frozen row per fixture: p_<key> / base_<key> for the exact-score grid
                         0-0..6-6 + other, the goal/margin bands and the specials, cand_<special> flags
  specials_settled.csv   the 90-minute result beside each logged row (y_<key>)
  study_*                the walk-forward history study (skill, candidate lift, calibration)
Nothing here computes a probability: every number is read from those files.
"""
from __future__ import annotations

import pandas as pd

GRID_MAX = 6
SPECIAL_LABELS = {
    "ge5": "5+ goals", "ge6": "6+ goals", "ge7": "7+ goals", "team_ge4": "a side scores 4+",
    "margin_ge4": "win by 4+", "nil_nil": "0-0", "tot_le1": "0-1 goals", "cs_unusual": "unusual exact score",
}
HIGH = ("ge5", "ge6", "ge7", "team_ge4", "margin_ge4")
LOW = ("nil_nil", "tot_le1")
BAND_LABELS = {
    "tot_0_1": "0-1 goals", "tot_2_3": "2-3 goals", "tot_4_5": "4-5 goals", "tot_6p": "6+ goals",
    "draw": "draw", "home_by1": "home by 1", "home_by2_3": "home by 2-3", "home_by4p": "home by 4+",
    "away_by1": "away by 1", "away_by2_3": "away by 2-3", "away_by4p": "away by 4+",
}


def _score_label(key: str) -> str:
    return key[3:].replace("_", "-") if isinstance(key, str) and key.startswith("cs_") else str(key)


def match_label(r) -> str:
    return f"{r.get('home_team')} v {r.get('away_team')}"


def upcoming(log: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if log is None or log.empty or "kickoff_utc" not in log:
        return pd.DataFrame()
    ko = pd.to_datetime(log["kickoff_utc"], utc=True, errors="coerce")
    return log[ko > now].assign(_ko=ko[ko > now]).sort_values("_ko")


def flagged_frame(log: pd.DataFrame, now: pd.Timestamp, group: tuple[str, ...]) -> pd.DataFrame:
    """One row per (upcoming fixture, flagged special in `group`): the model's probability, the league's
    own base rate and the ratio, exactly as logged."""
    up = upcoming(log, now)
    cols = ["kickoff", "league", "match", "special", "model (raw)", "calibrated", "league base", "ratio",
            "best price", "books", "score"]
    rows = []
    for _, r in up.iterrows():
        for k in group:
            if r.get(f"cand_{k}") != 1.0:
                continue
            if k == "cs_unusual":
                p, b, ratio = r.get("p_cs_unusual"), r.get("base_cs_unusual"), r.get("ratio_cs_unusual")
                score = _score_label(r.get("cs_unusual_pick"))
            else:
                p, b, ratio, score = r.get(f"p_{k}"), r.get(f"base_{k}"), r.get(f"ratio_{k}"), ""
            rows.append({"kickoff": str(r.get("kickoff_utc"))[:16].replace("T", " "), "league": r.get("league"),
                         "match": match_label(r), "special": SPECIAL_LABELS.get(k, k), "model (raw)": p,
                         "calibrated": r.get(f"p_cal_{k}"), "league base": b, "ratio": ratio,
                         "best price": r.get(f"price_{k}_best"), "books": r.get(f"price_{k}_books"),
                         "score": score})
    return pd.DataFrame(rows, columns=cols)


def grid_frame(row: pd.Series) -> pd.DataFrame:
    """The 0-0..6-6 grid for one logged fixture: model probability and league base per cell."""
    out = []
    for i in range(GRID_MAX + 1):
        for j in range(GRID_MAX + 1):
            k = f"cs_{i}_{j}"
            if f"p_{k}" not in row:
                continue
            p, b = row.get(f"p_{k}"), row.get(f"base_{k}")
            out.append({"home": i, "away": j, "score": f"{i}-{j}", "model": p, "league base": b,
                        "ratio": (p / b) if (pd.notna(p) and pd.notna(b) and b > 0) else None})
    return pd.DataFrame(out)


def bands_frame(row: pd.Series) -> pd.DataFrame:
    out = []
    # a band that IS a special (6+ goals, 0-1 goals) is shown once, with the special's calibration
    same = {"tot_6p": "ge6", "tot_0_1": "tot_le1"}
    specials = [(k, SPECIAL_LABELS[k]) for k in HIGH + LOW if k not in same.values()]
    for k, label in list(BAND_LABELS.items()) + specials:
        if f"p_{k}" not in row:
            continue
        p, b = row.get(f"p_{k}"), row.get(f"base_{k}")
        out.append({"outcome": label, "model (raw)": p, "calibrated": row.get(f"p_cal_{same.get(k, k)}"),
                    "league base": b,
                    "ratio": (p / b) if (pd.notna(p) and pd.notna(b) and b > 0) else None})
    if "p_cs_other" in row:
        out.append({"outcome": "other score (a side 7+)", "model (raw)": row.get("p_cs_other"),
                    "calibrated": None, "league base": row.get("base_cs_other"), "ratio": None})
    return pd.DataFrame(out)


def graded_frame(log: pd.DataFrame, settled: pd.DataFrame, min_events: int | None) -> pd.DataFrame:
    """Live graded record per league x special: flagged fixtures, how many happened, against the league
    base rate of those same fixtures. INSUFFICIENT below `min_events` events, whatever the numbers;
    NO_FLOOR when Pro has not published its rule (the floor is never assumed here)."""
    cols = ["league", "special", "flagged", "settled", "events", "hit rate", "league base", "status"]
    if log is None or log.empty or settled is None or settled.empty:
        return pd.DataFrame(columns=cols)
    d = log.merge(settled, on="row_id", how="inner")
    rows = []
    for k in HIGH + LOW + ("cs_unusual",):
        if f"cand_{k}" not in d or f"y_{k}" not in d:
            continue
        c = d[d[f"cand_{k}"] == 1.0]
        for lg, g in c.groupby("league"):
            y = g[f"y_{k}"].dropna()
            ev = int(y.sum())
            rows.append({"league": lg, "special": SPECIAL_LABELS.get(k, k), "flagged": int(len(g)),
                         "settled": int(len(y)), "events": ev,
                         "hit rate": float(y.mean()) if len(y) else None,
                         "league base": float(g[f"base_{k}"].mean()) if len(g) else None,
                         "status": ("NO_FLOOR" if min_events is None else
                                    "INSUFFICIENT" if ev < min_events else "MEASURABLE")})
    return pd.DataFrame(rows, columns=cols)


def study_table(study: dict) -> pd.DataFrame:
    """The history study's pooled answer per special: skill vs the league base rate, and the candidates'
    hit rate vs their base rate with the Holm-corrected verdict."""
    if not study:
        return pd.DataFrame()
    q = (study.get("rule") or {}).get("candidate_quantile")
    rows = []
    for k, label in SPECIAL_LABELS.items():
        sk = (study.get("skill") or {}).get(k, {})
        ce = (study.get("candidates") or {}).get(f"{k}@{q}", {})
        rows.append({"special": label, "matches": sk.get("n"), "events": sk.get("events"),
                     "Brier skill vs base": sk.get("brier_skill"),
                     "candidates": ce.get("candidates"), "candidate events": ce.get("events"),
                     "candidate hit": ce.get("hit_rate"), "their base": ce.get("mean_base"),
                     "lift": ce.get("lift"), "p (Holm)": ce.get("p_holm"), "verdict": ce.get("verdict")})
    return pd.DataFrame(rows)


def calibration_before_after(study: dict) -> pd.DataFrame:
    """CAL-1 before/after per special, as Pro's history study wrote it (raw vs calibrated vs base)."""
    ba = ((study or {}).get("calibration") or {}).get("before_after") or {}
    rows = []
    for k, v in ba.items():
        f = v.get("flagged") or {}
        rows.append({"special": SPECIAL_LABELS.get(k, k), "rows": v.get("n"), "actual": v.get("actual"),
                     "raw mean": v.get("mean_raw"), "calibrated mean": v.get("mean_cal"),
                     "skill raw": v.get("brier_skill_raw"), "skill calibrated": v.get("brier_skill_cal"),
                     "flagged actual": f.get("actual"), "flagged raw": f.get("mean_raw"),
                     "flagged calibrated": f.get("mean_cal")})
    return pd.DataFrame(rows)


def calibration_frame(grid: pd.DataFrame, league: str) -> pd.DataFrame:
    if grid is None or grid.empty:
        return pd.DataFrame()
    g = grid[grid["league"] == league].copy()
    if g.empty:
        return g
    g["gap"] = g["actual"] - g["mean_p"]
    return g[["score", "n", "events", "mean_p", "actual", "base_mean", "gap"]]
