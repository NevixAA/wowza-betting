"""
Rebuild the DERIVED columns of the stored player_history from its own stored rows - no network.

WHY (props_truth, round-2 findings, measured on data-truth-100 40d7dda9, 876,130 rows):
  * profile placeholders: age 25.0 / height_cm 180 / weight_kg 75 / age_peak_delta 0 on 100% of
    rows - never collected. -> NaN, and the age/height composites recomputed (NaN);
  * legacy HAND league tiers (league_tier & co., league_quality.LEGACY_COLUMNS): the code writes NaN
    since owner decision A (2026-11-02), the stored history still carries tiers (league_tier = 1 on
    129,434 La Liga 2 / Bundesliga 2 / Brazil Serie A / Austrian Bundesliga rows). -> NaN;
  * player_id 0: 256 rows of 153 different players pooled into one history. -> n_prev_games NaN
    (no history; the row leaves training through the n_prev_games filter). The raw id is kept;
  * own club as the opponent: 2,478 rows of a HOME club stored as away (the fixture list and the
    player sheet spell the club differently, "CF Pachuca" / "Pachuca"). -> is_home / opponent
    from club identity (player_model.club_rule.side_of); a name that proves no side is left as is;
  * n_prev_games batch-truncated on 84k rows (9.6%). -> history_integrity.prev_games_full.

RAW COLLECTED STATS ARE NEVER REWRITTEN (minutes, goals, shots, cards, ids, names, dates). Rows
with minutes above config.MAX_MATCH_MINUTES are counted only (build_features now reads them as
missing). Rolling / venue / opponent features that depend on the corrected is_home or on the
truncated batches are NOT recomputed here: that is a full build_features pass (heavy), listed in
the audit as 'not_recomputed'.

DRY RUN BY DEFAULT: reads with column projection, writes the per-row change sidecar
(output/corrections/player_history_rebuild_changes.csv.gz) and the audit
(output/corrections/player_history_rebuild_audit.json); player_history is not touched.
--apply rewrites the month parts through pipeline._save_history (the originals stay in git history;
the audit records the commit they came from).

    python scripts/rebuild_player_history.py            # dry run
    python scripts/rebuild_player_history.py --apply    # owner-approved rewrite
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model.club_rule import side_of  # noqa: E402
from player_model.history_integrity import prev_games_full  # noqa: E402
from player_model.league_quality import LEGACY_COLUMNS  # noqa: E402

PROFILE_RAWISH = ("age", "height_cm", "weight_kg")
PROFILE_COMPOSITES = ("height_diff_vs_opp_def", "height_advantage_score", "height_aerial_interaction",
                      "age_peak_delta")
NEEDED = ["fixture_id", "player_id", "date", "league", "team", "home_team", "away_team", "is_home",
          "opponent", "minutes", "n_prev_games", "aerial_won_rate", "opp_def_mean_height",
          *PROFILE_RAWISH, *PROFILE_COMPOSITES, *LEGACY_COLUMNS]
NOT_RECOMPUTED = ["venue splits (goals/shots/sot _home/_away_pg) on is_home-corrected rows",
                  "opponent / measured-strength features on is_home-corrected rows",
                  "rolling and career features on batch-truncated rows (full build_features pass)",
                  "minutes_pg / per-90 features of rows after a minutes > MAX_MATCH_MINUTES row"]


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def rebuild(df: pd.DataFrame, collected_heights: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """(rebuilt frame, counts, per-row changes). Pure function of the stored rows."""
    out = df.copy()
    counts: dict = {"rows": int(len(df))}
    changes = []

    def _log(col, mask, before, after):
        if mask.any():
            changes.append(pd.DataFrame({"fixture_id": df.loc[mask, "fixture_id"].to_numpy(),
                                         "player_id": df.loc[mask, "player_id"].to_numpy(),
                                         "date": df.loc[mask, "date"].astype(str).str[:10].to_numpy(),
                                         "column": col, "before": before[mask].to_numpy(),
                                         "after": after[mask].to_numpy()}))

    # 1. profile placeholders -> NaN (FE._profile is the one rule), composites recomputed.
    for c in PROFILE_RAWISH:
        if c in out.columns:
            new = FE._profile(out, c)
            counts[f"{c}_placeholder_to_nan"] = int((~_same(new, pd.to_numeric(df[c], errors="coerce"))).sum())
            out[c] = new
    if "opp_def_mean_height" in out.columns and not collected_heights:
        # Built from the 180 cm placeholder only (no collected height exists): not a measurement.
        before = pd.to_numeric(df["opp_def_mean_height"], errors="coerce")
        counts["opp_def_mean_height_to_nan"] = int(before.notna().sum())
        out["opp_def_mean_height"] = np.nan
    if {"height_cm", "age", "aerial_won_rate", "opp_def_mean_height"} <= set(out.columns):
        for c in ("height_cm", "age", "aerial_won_rate", "opp_def_mean_height"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        FE.profile_composites(out)
        for c in PROFILE_COMPOSITES:
            if c in df.columns:
                counts[f"{c}_changed"] = int((~_same(out[c], pd.to_numeric(df[c], errors="coerce"))).sum())

    # 2. legacy hand league tiers -> NaN (what the code writes since owner decision A).
    for c in LEGACY_COLUMNS:
        if c in out.columns:
            counts[f"{c}_to_nan"] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
            out[c] = np.nan
    if "league_tier" in df.columns:
        lt = pd.to_numeric(df["league_tier"], errors="coerce")
        wrong = df["league"].isin(["La Liga 2", "Bundesliga 2", "Brazil Serie A", "Austrian Bundesliga"])
        counts["league_tier_1_on_second_tier_or_non_top5_before"] = int((wrong & (lt == 1)).sum())

    # 3. is_home / opponent from club identity.
    combos = df[["team", "home_team", "away_team"]].astype(str).drop_duplicates()
    side = {tuple(r): side_of(*r) for r in combos.itertuples(index=False, name=None)}
    proved = pd.Series([side[t] for t in zip(df["team"].astype(str), df["home_team"].astype(str),
                                             df["away_team"].astype(str))], index=df.index, dtype=object)
    stored_home = pd.to_numeric(df["is_home"], errors="coerce") == 1
    fix = proved.notna() & (proved.astype(object) != stored_home.astype(object))
    new_home = pd.to_numeric(df["is_home"], errors="coerce").copy()
    new_home[fix] = proved[fix].astype(bool).astype(int)
    new_opp = df["opponent"].copy()
    new_opp[fix] = np.where(new_home[fix] == 1, df.loc[fix, "away_team"], df.loc[fix, "home_team"])
    counts["is_home_corrected"] = int(fix.sum())
    counts["is_home_corrected_home_was_away"] = int((fix & (new_home == 1)).sum())
    counts["team_proves_neither_side_left_as_stored"] = int(proved.isna().sum())
    _log("is_home", fix, pd.to_numeric(df["is_home"], errors="coerce"), new_home)
    _log("opponent", fix, df["opponent"], new_opp)
    out["is_home"] = new_home.astype(df["is_home"].dtype) if fix.any() else df["is_home"]
    out["opponent"] = new_opp
    own_before = _own_opponent(df)
    counts["own_club_as_opponent_before"] = int(own_before.sum())
    counts["own_club_as_opponent_after"] = int(_own_opponent(out).sum())

    # 4. player_id 0 / missing -> no history; 5. n_prev_games over the whole stored history.
    pid = pd.to_numeric(df["player_id"], errors="coerce")
    counts["player_id_not_positive_rows"] = int((~(pid > 0)).sum())
    counts["player_id_0_distinct_names"] = int(df.loc[pid == 0, "player_name"].nunique()) \
        if "player_name" in df.columns else None
    full = prev_games_full(df)
    before_n = pd.to_numeric(df["n_prev_games"], errors="coerce")
    ch = ~_same(full, before_n)
    counts["n_prev_games_changed"] = int(ch.sum())
    counts["n_prev_games_increased"] = int((full > before_n).sum())
    counts["n_prev_games_decreased"] = int((full < before_n).sum())
    counts["n_prev_games_to_nan_no_player_id"] = int((full.isna() & before_n.notna()).sum())
    _log("n_prev_games", ch, before_n, full)
    out["n_prev_games"] = full.astype(float)

    mins = pd.to_numeric(df["minutes"], errors="coerce")
    counts["minutes_above_120"] = int((mins > 120).sum())
    counts["minutes_above_MAX_MATCH_MINUTES"] = int((mins > config.MAX_MATCH_MINUTES).sum())
    chg = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame(
        columns=["fixture_id", "player_id", "date", "column", "before", "after"])
    return out, counts, chg


def _own_opponent(df: pd.DataFrame) -> pd.Series:
    """Rows whose stored opponent is the player's own club (club identity)."""
    from player_model.club_rule import same_club
    pairs = df[["team", "opponent"]].astype(str).drop_duplicates()
    hit = {(t, o): same_club(t, o) is True for t, o in pairs.itertuples(index=False, name=None)}
    return pd.Series([hit[(t, o)] for t, o in zip(df["team"].astype(str), df["opponent"].astype(str))],
                     index=df.index)


def _head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                                    # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=str(ROOT / "player_history.parquet"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "corrections"))
    ap.add_argument("--apply", action="store_true", help="rewrite player_history (owner-approved)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    parts = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
    if a.apply:
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        cols = set(pd.read_parquet(parts[-1]).columns)
        use = [c for c in dict.fromkeys(NEEDED + ["player_name"]) if c in cols]
        df = pd.concat([pd.read_parquet(p, columns=use) for p in parts], ignore_index=True)
    prof = FE.collected_profiles()
    rebuilt, counts, chg = rebuild(df, collected_heights=bool(len(prof)) and prof["height_cm"].notna().any())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chg.to_csv(out / "player_history_rebuild_changes.csv.gz", index=False, compression="gzip")
    audit = {"source_commit": _head(), "mode": "apply" if a.apply else "dry_run",
             "rule": "derived columns recomputed from stored rows; raw collected stats untouched",
             "counts": counts, "per_row_changes_file": "player_history_rebuild_changes.csv.gz",
             "per_row_columns": sorted(chg["column"].unique().tolist()) if len(chg) else [],
             "whole_column_changes_not_listed_per_row": {
                 "to_nan": [*PROFILE_RAWISH, "opp_def_mean_height", *LEGACY_COLUMNS],
                 "recomputed": list(PROFILE_COMPOSITES),
                 "original_values": "placeholder constants (age 25.0, height 180, weight 75) and hand "
                                    "tiers; recoverable from git at source_commit"},
             "not_recomputed": NOT_RECOMPUTED}
    (out / "player_history_rebuild_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n",
                                                           encoding="utf-8")
    print(json.dumps(counts, indent=2, default=str))
    if a.apply:
        from player_model.pipeline import _save_history
        _save_history(rebuilt, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
