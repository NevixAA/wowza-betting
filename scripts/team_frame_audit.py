"""Duplicate-match audit of the team training frame (team_frame finding, 2026-11-02). READ-ONLY.

    python scripts/team_frame_audit.py [--json out.json]

Builds the CI-path frame OFFLINE from the committed banks (output/fd_history.parquet + the backtest
archive; no football-data download, no API-Football key, the coverage report is not written) and
measures, with src/integrity/duplicate_matches:
  * duplicate-match pairs (same league, canonical clubs, same score, dates <= 3 d apart; or same day,
    same score, one side the same club), same-day vs 1-3 days apart, by league;
  * rows whose last-5 form window holds their own match;
  * the over-2.5 form AUC (src/feature_engineering's own rolling) on all / self-containing / clean rows.
No retrain, no backtest, nothing written to output/.

Measured 2026-11-02:  data-truth-100 HEAD 84e06800   75,855 rows, 3,884 pairs (2,927 same day,
957 1-3 d), 3,882 self-containing rows, AUC 0.710 self-containing vs 0.544 clean (0.553 all);
fix4/team_frame   71,147 rows, 0 pairs, 0 self-containing rows, AUC 0.546.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_offline_frame():
    os.environ["APIFOOTBALL_KEY"] = ""
    os.environ["API_KEY"] = ""
    import config
    config.API_KEY = ""
    from src import data_loader as DL
    DL._ci_download_all = lambda have=None: []          # banks only, no network
    DL._report_training_coverage = lambda out: None    # never write output/training_coverage.json
    if config.SUMMARY_XLSX.exists():
        raise SystemExit(f"{config.SUMMARY_XLSX} exists: this audit measures the CI (bank) path only")
    return DL.load_all_matches(force=True)


def measure(df) -> dict:
    from src.integrity import duplicate_matches as DM
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    pairs = DM.duplicate_pairs(df)
    sc = DM.self_containing_rows(df, pairs)
    return {"rows": int(len(df)), "pairs": int(len(pairs)),
            "same_day_pairs": int((pairs["gap_days"] == 0).sum()),
            "near_date_pairs_1_3d": int((pairs["gap_days"] > 0).sum()),
            "by_kind": pairs["kind"].value_counts().to_dict(),
            "near_by_league": pairs[pairs["gap_days"] > 0]["league"].value_counts().to_dict(),
            "same_day_by_league": pairs[pairs["gap_days"] == 0]["league"].value_counts().to_dict(),
            "self_containing_rows": int(sc.sum()),
            "auc_over25_form_all": DM.over25_form_auc(df),
            "auc_over25_form_self_containing": DM.over25_form_auc(df, sc),
            "auc_over25_form_clean": DM.over25_form_auc(df, ~sc),
            "one_match_report": df.attrs.get("one_match_report"),
            "impossible_stats": df.attrs.get("impossible_stats"),
            "ou25_pairs_not_a_price": int((df["ou25_pair_is_price"] == False).sum())  # noqa: E712
            if "ou25_pair_is_price" in df.columns else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    res = measure(build_offline_frame())
    txt = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
