"""
af_history_provenance.py - which output/af_history.parquet rows are PROVIDER rows and which were
written back by the loader (owner decisions 2026-11-04, fix round 9, KEY=bank).

WHY. Until 2026-11-04 src/data_loader._enrich_with_api_shots ended with 'BANK THE ENRICHMENT': it
appended every merged-history row that carried shots / SoT / xG / inside-box to af_history.parquet
- whatever the source of the value (football-data, a live API fetch joined by name, or a value the
bank itself had put there on an earlier run) - keyed only by (date, league, home, away), newest row
winning. Fix round 8 (fix8/nfgap) measured 90,876 of the 93,381 rows as such write-backs; where a
fixture-proven provider answer exists their total shots agree with it in only 14.3% of 1,583 pairs,
and the write-back's column map dropped corners and fouls.

THE RULE (owner 2026-11-04). A row is WRITTEN BACK when it has NO fixture_id AND NO full-time score.
Rechecked on the committed file (93,381 rows, data-truth-100 c32b4c8f):
  * no fixture_id ............................ 91,656
  * no FT score (FTHG or FTAG null) .......... 90,897
  * both (= written back) .................... 90,897  (90,876 carry shots: round 8's count; 21 only
                                                         xG / inside-box)
  * fixture_id but no FT score ............... 0
  * FT score but no fixture_id ............... 759     (backfill_af_history rows from before it stored
                                                         provider ids: provider rows, kept)
scripts/backfill_af_history.parse_fixture never emits a row without a score, and the write-back
never wrote FTHG/FTAG or fixture_id, so "no FT score" alone already separates them; fixture_id is
kept in the rule as the owner stated it. A written-back row's statistics are UNPROVEN: the loader
serves none of them (`WRITTEN_BACK_STAT_COLS`); a value only reaches the history from a provider
source matched to the fixture (the --fullstats / --insidebox sidecars, the provider rows, a live
provider fetch). The rows stay in af_history.parquet, which is an original and is never rewritten.
"""
from __future__ import annotations

import pandas as pd

#: af_history statistic columns a written-back row may carry: the bank's own map (HS..AIB) plus
#: corners, fouls and cards should any be present. None of them is served from such a row.
WRITTEN_BACK_STAT_COLS = ("HS", "AS", "HST", "AST", "HXG", "AXG", "HIB", "AIB",
                          "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR")


def written_back_mask(hist: pd.DataFrame) -> pd.Series:
    """True for af_history rows the loader wrote back: no fixture_id AND no full-time score."""
    def _col(c):
        return hist[c] if c in hist.columns else pd.Series(float("nan"), index=hist.index)
    no_fixture = pd.to_numeric(_col("fixture_id"), errors="coerce").isna()
    no_score = (pd.to_numeric(_col("FTHG"), errors="coerce").isna()
                | pd.to_numeric(_col("FTAG"), errors="coerce").isna())
    return no_fixture & no_score
