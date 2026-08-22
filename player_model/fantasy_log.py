"""
Append-only projection log — so "projected vs actual" becomes answerable.
========================================================================
`output/fantasy_tips.csv` is OVERWRITTEN on every run and carries no timestamp or gameweek column,
so there was no record of what was projected WHEN. Without that, "projected points vs real score"
cannot be computed for any past gameweek — the same shape as the predictions.csv problem, where the
only history was 766 git commits.

This writes one row per (snapshot_date, player) and never deletes. It captures our projection
alongside two things that make it auditable later:

  * `fpl_ep_next` — FPL's OWN expected points, published free. This is the benchmark that matters.
    A projection that does not beat the number anyone can read off the FPL website adds nothing,
    however well calibrated it looks in isolation.
  * `fpl_ppg` and `fpl_total_points` — actual scoring to date, so the log is self-contained and a
    later reader does not need to reconstruct the FPL state of that day.

DEDUP IS BY (snapshot_date, player), not by timestamp: the projections job can run many times a day
and each run would otherwise add a near-identical row. Keeping the LAST of each day means the log
grows by one row per player per day, which is ~280/day for the Premier League — small enough to
commit, which is the point, because an uncommitted log has no history either.

WHAT THIS DOES NOT DO. It does not attribute a projection to a gameweek's ACTUAL result — that needs
per-GW actuals from the FPL element-summary endpoint, and it needs the log to exist first. This is
the prerequisite, not the analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

LOG_COLS = [
    "snapshot_date", "snapshot_ts", "gw", "player_name", "team", "position", "price",
    # ours
    "proj_pts_per_game", "proj_fixture_adj", "proj_total_next", "n_fixtures_next", "avg_fdr",
    "p_start", "minutes_pg",
    # FPL's own, for benchmarking and for making the row self-contained
    "fpl_ep_next", "fpl_ppg", "fpl_total_points", "fpl_form", "owned_pct",
    "availability", "injured",
]


def append(proj: pd.DataFrame, *, gw: int | None = None,
           path: Path | None = None) -> int:
    """Append today's projections. Returns rows written. Never raises."""
    try:
        if proj is None or proj.empty:
            return 0
        import config
        path = path or (config.OUTPUT_DIR / "fantasy_projection_log.csv")
        now = pd.Timestamp.now(tz="UTC")
        d = pd.DataFrame({
            "snapshot_date": now.strftime("%Y-%m-%d"),
            "snapshot_ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gw": gw if gw is not None else "",
            "player_name": proj.get("player_name"),
            "team": proj.get("team"),
            "position": proj.get("position"),
            "price": proj.get("price"),
            "proj_pts_per_game": proj.get("fantasy_pts"),
            "proj_fixture_adj": proj.get("fixture_adj_pts"),
            "proj_total_next": proj.get("total_xpts_next"),
            "n_fixtures_next": proj.get("n_fixtures_next"),
            "avg_fdr": proj.get("avg_fdr"),
            "p_start": proj.get("p_start"),
            "minutes_pg": proj.get("minutes_pg"),
            "fpl_ep_next": proj.get("fpl_ep_next"),
            "fpl_ppg": proj.get("fpl_ppg"),
            "fpl_total_points": proj.get("total_points"),
            "fpl_form": proj.get("fpl_form"),
            "owned_pct": proj.get("owned_pct"),
            "availability": proj.get("availability"),
            "injured": proj.get("injured"),
        })
        for c in LOG_COLS:
            if c not in d.columns:
                d[c] = pd.NA
        d = d[LOG_COLS]
        if path.exists():
            try:
                d = pd.concat([pd.read_csv(path), d], ignore_index=True)
            except Exception as e:
                log.warning(f"[fantasy_log] could not read existing log ({e}); starting fresh "
                            f"would DESTROY history, so appending is skipped this run")
                return 0
        # One row per player per day: the job may run many times and each would otherwise add a
        # near-duplicate. Last write of the day wins.
        d = d.drop_duplicates(subset=["snapshot_date", "player_name"], keep="last")
        d.to_csv(path, index=False)
        log.info(f"[fantasy_log] {len(proj)} projection(s) logged -> {path.name} "
                 f"(total {len(d):,} rows, {d['snapshot_date'].nunique()} day(s))")
        return len(proj)
    except Exception as e:
        log.warning(f"[fantasy_log] append skipped ({e})")
        return 0


def calibration(log_path: Path | None = None) -> pd.DataFrame:
    """Per snapshot_date: how our projection compares to actual PPG and to FPL's own ep_next.

    This is a CALIBRATION view, not a settled-result view. `fpl_ppg` is season-to-date actual
    points per game, so comparing it to a forward projection is only fair in aggregate and only
    once enough gameweeks exist. Stated here because the numbers look more precise than they are.
    """
    try:
        import config
        p = log_path or (config.OUTPUT_DIR / "fantasy_projection_log.csv")
        if not p.exists():
            return pd.DataFrame()
        d = pd.read_csv(p)
        for c in ("proj_pts_per_game", "fpl_ep_next", "fpl_ppg", "minutes_pg"):
            d[c] = pd.to_numeric(d.get(c), errors="coerce")
        # Only players who actually play: a projection for someone with no minutes is untestable,
        # and including them drags both error terms toward zero for the wrong reason.
        d = d[d["minutes_pg"].fillna(0) >= 30]
        if d.empty:
            return pd.DataFrame()
        d["err_ours"] = (d["proj_pts_per_game"] - d["fpl_ppg"]).abs()
        d["err_fpl"] = (d["fpl_ep_next"] - d["fpl_ppg"]).abs()
        g = d.groupby("snapshot_date").agg(
            players=("player_name", "size"),
            mae_ours=("err_ours", "mean"),
            mae_fpl=("err_fpl", "mean"),
            bias_ours=("proj_pts_per_game", "mean"),
            actual_ppg=("fpl_ppg", "mean"),
            fpl_ep=("fpl_ep_next", "mean"),
        ).reset_index()
        g["ours_beats_fpl"] = g["mae_fpl"] - g["mae_ours"]      # positive = we are closer
        return g.round(3)
    except Exception:
        return pd.DataFrame()
