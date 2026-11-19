"""Integrity report of the fixture-level enrichment joins AS SERVED (R2-C finding 12). READ-ONLY, offline.

    python scripts/integrity_af_join_replay.py [--out output/join_integrity_replay.json]

1. Builds the training/serving history the way the CI path of src.data_loader.load_all_matches
   does, from the COMMITTED caches only (output/fd_history.parquet + the backtest archive), with
   the API-Football key removed so no live fetch can run and nothing is banked.
2. Runs the three enrichment steps once, as they serve, and captures every join's integrity report.

The legacy-vs-safe replay is retired: the league-free legacy joins, their positional copy-back and
the WOWZA_SAFE_ENRICH_JOINS switch were removed from src/data_loader on 2026-11-01 (Data Truth), so
there is no second join mode left to replay and this script no longer sets that variable.
`--serving` is accepted for the existing workflow call and ignored.
Writes one JSON. Nothing under models/ or telegram_bot/ is touched; output/ only gets --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

def base_history():
    """The CI construction of load_all_matches, up to (not including) the enrichment calls."""
    import numpy as np
    import pandas as pd
    from src import data_loader as DL
    from src.history_corrections import serve_history
    # the owner's history corrections, as load_all_matches applies them to the fd_cache rows
    # (fix16/readers): removal lists, 90-minute scores, provider dates - on fd_history alone, the
    # only bank they are eligible on, so the archive rows are never touched
    cached = serve_history(pd.read_parquet(ROOT / "output" / "fd_history.parquet"),
                           tag="join_replay.fd_history")
    frames = [cached]
    arch = DL._load_backtest_archive()
    if len(arch):
        frames.append(arch)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["home_team"].notna() & out["away_team"].notna()]
    out = out.sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["over25"] = (out["total_goals"] > 2.5).astype(float)
    out["over15"] = (out["total_goals"] > 1.5).astype(float)
    out["over35"] = (out["total_goals"] > 3.5).astype(float)
    out["btts"] = ((out["home_goals"] > 0) & (out["away_goals"] > 0)).astype(float)
    if "ht_home_goals" in out.columns and out["ht_home_goals"].notna().any():
        out["ht_total_goals"] = out["ht_home_goals"] + out["ht_away_goals"]
        _htn = out["ht_total_goals"].notna()
        out["ht_over05"] = (out["ht_total_goals"] >= 1).astype(float).where(_htn)
        out["ht_over15"] = (out["ht_total_goals"] >= 2).astype(float).where(_htn)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["home_sot_ratio"] = np.where(out["home_shots"] > 0, out["home_sot"] / out["home_shots"], np.nan)
        out["away_sot_ratio"] = np.where(out["away_shots"] > 0, out["away_sot"] / out["away_shots"], np.nan)
    return DL._canonicalise_clubs(out)


def enrich(base):
    """The three enrichment steps as they serve: league-keyed joins, always (the legacy
    league-free path and its WOWZA_SAFE_ENRICH_JOINS switch were removed 2026-11-01)."""
    from src import data_loader as DL
    from src.integrity import joins as IJ
    IJ.reset()
    d = base.copy()
    d = DL._enrich_with_api_shots(d)
    d = DL._enrich_with_standard_sidemarket_odds(d)
    d = DL._enrich_ou25_from_captures(d)
    return d, IJ.reports()


def write_health(doc: dict, path: Path) -> dict:
    """output/join_integrity_health.json: the joins AS SERVED (league-keyed, the only mode)."""
    from src.integrity import joins as IJ
    IJ.reset()
    for k, v in doc["join_reports"].items():
        IJ._REPORTS[k] = v
    extra = {
        "serving_join_mode": "safe",
        "flag": "none - league-keyed joins always serve (WOWZA_SAFE_ENRICH_JOINS removed 2026-11-01)",
        "fixture_level_enrichment_joins_in_v9": JOIN_INVENTORY,
    }
    return IJ.write_join_health(path, extra)


#: Every fixture-level enrichment join found in v9 (2026-11-01 sweep). `owner` = who may change it.
JOIN_INVENTORY = [
    {"site": "src/data_loader.py::_enrich_with_api_shots step 1 (af_history.parquet)",
     "key_legacy": "home_norm+away_norm+date (no league)", "copy_back": "positional zip",
     "defect": "EXPANDS + SHIFTS (proven)", "fix": "safe_left_enrich on league+keys, flag", "owner": "R2-C"},
    {"site": "src/data_loader.py::_enrich_with_api_shots step 1b (af_ht_history.parquet)",
     "key_legacy": "home_norm+away_norm+date (no league)", "copy_back": "positional zip",
     "defect": "same shape; file not committed, CI cache only", "fix": "safe_left_enrich, flag", "owner": "R2-C"},
    {"site": "src/data_loader.py::_enrich_with_api_shots live per-league shots merge",
     "key_legacy": "home_norm+away_norm+date within one league", "copy_back": "positional",
     "defect": "expands if the fetched frame holds a duplicate key", "fix": "safe_left_enrich, flag", "owner": "R2-C"},
    {"site": "src/data_loader.py::_enrich_with_standard_sidemarket_odds",
     "key_legacy": "home_norm+away_norm+date (no league), pivot last", "copy_back": "boolean mask across two indexes",
     "defect": "frame index is not a RangeIndex after _canonicalise_clubs, so mask & .values misalign (proven)",
     "fix": "safe_left_enrich on league+keys, flag", "owner": "R2-C"},
    {"site": "src/data_loader.py::_enrich_ou25_from_captures first pass",
     "key_legacy": "home_norm+away_norm+date (no league), pivot last", "copy_back": "boolean mask across two indexes",
     "defect": "same index misalignment (proven)", "fix": "safe_left_enrich, flag; resolver pass already index-safe", "owner": "R2-C"},
    {"site": "src/data_loader.py::_enrich_with_af_odds (af_odds_history.parquet)",
     "key_legacy": "home+away+date (no league), reindex", "copy_back": "boolean mask",
     "defect": "permanent no-op (file can never exist, CLAUDE.md); audited only", "fix": "none needed", "owner": "R2-C"},
    {"site": "src/data_loader.py::_ci_download_all NF odds (_nf_real_odds dict)",
     "key_legacy": "(snapshot_date, league, 'Home vs Away') dict", "copy_back": "per row lookup",
     "defect": "league-scoped, no expansion; reads mis-mapped pre-2026-08-10 O/U rows",
     "fix": "WOWZA_PRICE_MAPPING_FILTER, flag", "owner": "R2-C"},
    {"site": "src/feature_engineering.py (_merge_sofascore, season-stat / upcoming lookups)",
     "key_legacy": "league-scoped team lookups", "copy_back": "by index / dict",
     "defect": "not audited by R2-C beyond reading", "fix": "owner: feature workstream", "owner": "other"},
    {"site": "src/predict.py (odds/board merges)", "key_legacy": "various", "copy_back": "-",
     "defect": "not edited (owned by another workstream)", "fix": "-", "owner": "other"},
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/join_integrity_replay.json")
    ap.add_argument("--serving", action="store_true",
                    help="ignored: the legacy-vs-safe serving replay is retired (one join mode left)")
    ap.add_argument("--health", default="output/join_integrity_health.json",
                    help="also write the join integrity health artifact here ('' to skip)")
    a = ap.parse_args(argv)
    os.environ.pop("APIFOOTBALL_KEY", None)
    import logging
    logging.basicConfig(level=logging.WARNING)
    import config
    config.API_KEY = ""
    hist, rep = enrich(base_history())
    doc = {"artifact": "join_integrity_replay", "label": "AS_SERVED", "join_mode": "league_keyed",
           "inputs": ["output/fd_history.parquet", "output/backtest_all_leagues.csv",
                      "output/af_history.parquet", "output/standard_sidemarket_odds_history.csv",
                      "output/standard_odds_history.csv", "output/newformat_odds_history.csv"],
           "rows_total": int(len(hist)), "join_reports": rep}
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    if a.health:
        write_health(doc, ROOT / a.health)
    print(f"rows {doc['rows_total']:,}; {len(rep)} join report(s) written to {a.out}")
    if a.serving:
        print("--serving ignored: the legacy-vs-safe serving replay is retired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
