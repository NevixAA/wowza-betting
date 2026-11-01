"""RETROSPECTIVE replay of the fixture-level enrichment joins (R2-C finding 12). READ-ONLY, offline.

    python scripts/integrity_af_join_replay.py [--serving] [--out output/join_integrity_replay.json]

1. Builds the training/serving history the way the CI path of src.data_loader.load_all_matches
   does, from the COMMITTED caches only (output/fd_history.parquet + the backtest archive), with
   the API-Football key removed so no live fetch can run and nothing is banked.
2. Runs the three enrichment steps twice: legacy joins (WOWZA_SAFE_ENRICH_JOINS=0, what serves) and
   the safe league-keyed joins (=1). Every join's integrity report is captured both times.
3. Compares the two histories cell by cell, and classifies each legacy cell that differs:
     SHIFTED      legacy value present, safe (key-correct) value present and different
     PHANTOM      legacy value present, no key-correct value exists (shift onto an unmatched row, or
                  a value taken from another league / a conflicting duplicate)
     DROPPED      legacy left NaN where a key-correct value exists
4. --serving: runs the deployed predict pass (scripts/standard_switch_preview.run_mode, network
   blocked, deployed env) on the committed board with each history and diffs every serving feature,
   probability and tip.
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

ENRICHED = ["home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
            "home_fouls", "away_fouls", "home_xg", "away_xg", "home_insidebox", "away_insidebox",
            "ht_home_goals", "ht_away_goals", "odds_btts", "odds_over15", "odds_over35",
            "odds_over25", "odds_under25"]


def base_history():
    """The CI construction of load_all_matches, up to (not including) the enrichment calls."""
    import numpy as np
    import pandas as pd
    from src import data_loader as DL
    cached = pd.read_parquet(ROOT / "output" / "fd_history.parquet")
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


def enrich(base, safe: bool):
    from src import data_loader as DL
    from src.integrity import joins as IJ
    os.environ["WOWZA_SAFE_ENRICH_JOINS"] = "1" if safe else "0"
    IJ.reset()
    d = base.copy()
    d = DL._enrich_with_api_shots(d)
    d = DL._enrich_with_standard_sidemarket_odds(d)
    d = DL._enrich_ou25_from_captures(d)
    return d, IJ.reports()


def compare(L, S):
    import numpy as np
    import pandas as pd
    assert len(L) == len(S) and (L.index == S.index).all(), "row sets differ"
    out, by_league = {}, {}
    for c in ENRICHED:
        if c not in L.columns and c not in S.columns:
            continue
        a = pd.to_numeric(L.get(c), errors="coerce")
        b = pd.to_numeric(S.get(c), errors="coerce")
        an, bn = a.notna(), b.notna()
        diff_val = an & bn & ~np.isclose(a.fillna(0), b.fillna(0))
        phantom = an & ~bn
        dropped = ~an & bn
        out[c] = {"legacy_non_null": int(an.sum()), "safe_non_null": int(bn.sum()),
                  "shifted": int(diff_val.sum()), "phantom": int(phantom.sum()),
                  "dropped": int(dropped.sum()),
                  "rows_changed": int((diff_val | phantom | dropped).sum())}
        ch = diff_val | phantom | dropped
        if ch.any():
            for lg, n in L.loc[ch, "league"].value_counts().items():
                by_league.setdefault(str(lg), {})[c] = int(n)
    any_ch = None
    for c in out:
        a = pd.to_numeric(L.get(c), errors="coerce")
        b = pd.to_numeric(S.get(c), errors="coerce")
        ch = (a.notna() != b.notna()) | (a.notna() & b.notna() & ~np.isclose(a.fillna(0), b.fillna(0)))
        any_ch = ch if any_ch is None else (any_ch | ch)
    total = int(any_ch.sum()) if any_ch is not None else 0
    first = None
    if total:
        i = any_ch[any_ch].index[0]
        first = {k: str(L.at[i, k]) for k in ("date", "league", "home_team", "away_team")}
    examples = []
    if total:
        cols = [c for c in ("home_shots", "away_shots", "home_sot", "away_sot") if c in L.columns]
        ex = L.loc[any_ch, ["date", "league", "home_team", "away_team"] + cols].tail(6)
        for i, r in ex.iterrows():
            examples.append({"fixture": f"{str(r['date'])[:10]} {r['league']}: {r['home_team']} v {r['away_team']}",
                             "legacy": {c: (None if pd.isna(r[c]) else float(r[c])) for c in cols},
                             "safe": {c: (None if pd.isna(S.at[i, c]) else float(S.at[i, c])) for c in cols}})
    return {"rows_total": int(len(L)), "rows_with_any_change": total, "first_changed_row": first,
            "columns": out, "by_league": by_league, "examples_latest": examples}


def serving(histL, histS):
    import numpy as np
    import pandas as pd
    import standard_switch_preview as SP
    env = SP.deployed_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ.pop("APIFOOTBALL_KEY", None)
    os.environ.pop("ODDS_API_KEY", None)
    import config
    from src.model import load_models
    import src.drift as D
    import src.health_check as H
    D.snapshot = lambda *a_, **k_: None
    D.purge_old = lambda *a_, **k_: None
    H.record_fetch = lambda *a_, **k_: None
    H.record_feature_health = lambda *a_, **k_: None
    SP._block_network()
    upcoming, board = SP.board_fixtures(ROOT / "output" / "predictions.csv")
    for c in ("best_over25", "best_under25"):
        if c in board.columns:
            upcoming[c] = upcoming.merge(board.drop_duplicates(["league", "home_team", "away_team"])
                                         [["league", "home_team", "away_team", c]],
                                         on=["league", "home_team", "away_team"], how="left")[c].to_numpy()
    payloads = {"std": load_models(model_file=config.MODEL_FILE_STANDARD),
                "nf": load_models(model_file=config.MODEL_FILE_NEWFORMAT),
                "side": {t: load_models(model_file=f) for t, f in config.SIDE_MARKETS.items() if f.exists()}}
    recipe = env.get("STANDARD_OU25_SERVING_RECIPE", config.STANDARD_OU25_SERVING_RECIPE)
    RL = SP.run_mode(recipe, upcoming, histL, payloads)
    RS = SP.run_mode(recipe, upcoming, histS, payloads)
    k = ["league", "home_team", "away_team"]
    a = RL["preds"].drop_duplicates(k).set_index(k)
    b = RS["preds"].drop_duplicates(k).set_index(k).reindex(a.index)
    feats = sorted(set(payloads["std"]["feature_cols"]) | set(payloads["nf"]["feature_cols"])
                   | {f for p in payloads["side"].values() for f in p["feature_cols"]})
    fdiff = {}
    for f in feats + ["p_over25", "p_btts", "p_over15", "p_over35", "p_ht_over05", "p_ht_over15"]:
        if f not in a.columns:
            continue
        x = pd.to_numeric(a[f], errors="coerce")
        y = pd.to_numeric(b[f], errors="coerce")
        ch = (x.notna() != y.notna()) | (x.notna() & y.notna() & ~np.isclose(x.fillna(0), y.fillna(0), atol=1e-9))
        if ch.any():
            d = (y - x)[ch]
            mt = a.loc[ch, "model_type"].value_counts().to_dict() if "model_type" in a else {}
            fdiff[f] = {"fixtures_changed": int(ch.sum()), "by_model_type": {str(kk): int(v) for kk, v in mt.items()},
                        "mean_abs_delta": float(d.abs().mean()) if d.notna().any() else None,
                        "max_abs_delta": float(d.abs().max()) if d.notna().any() else None}

    def tips(R):
        rows = []
        for nm in ("send_bets", "send_side"):
            t = R[nm]
            if t is None or t.empty:
                continue
            side = t["side"] if "side" in t.columns else t.get("bet")
            mk = t["market"] if "market" in t.columns else "OU25"
            rows += list(zip(t["league"], t["home_team"], t["away_team"],
                             pd.Series(mk, index=t.index).astype(str), side.astype(str), t["signal_tier"]))
        return set(rows)
    tl, ts = tips(RL), tips(RS)
    return {"board_fixtures": int(len(upcoming)), "recipe": recipe,
            "serving_features_changed": fdiff,
            "tips_legacy": len(tl), "tips_safe": len(ts),
            "tips_only_legacy": sorted(map(list, tl - ts)), "tips_only_safe": sorted(map(list, ts - tl))}


def write_health(doc: dict, path: Path) -> dict:
    """output/join_integrity_health.json: the joins AS SERVED (legacy audit reports — what production
    runs with WOWZA_SAFE_ENRICH_JOINS unset), the safe joins measured beside them, and the replay."""
    from src.integrity import joins as IJ
    IJ.reset()
    serving_safe = doc.get("serving_flag") == "1"
    for k, v in (doc["join_reports_safe"] if serving_safe else doc["join_reports_legacy"]).items():
        IJ._REPORTS[k] = v
    hd = doc["history_diff"]
    extra = {
        "serving_join_mode": "safe" if serving_safe else "legacy",
        "flag": "WOWZA_SAFE_ENRICH_JOINS (default 0 = legacy joins serve; owner review pending)",
        "safe_join_reports": doc["join_reports_safe"],
        "replay": {"label": "RETROSPECTIVE", "rows_total": hd["rows_total"],
                   "rows_with_any_enriched_value_changed": hd["rows_with_any_change"],
                   "first_changed_row": hd["first_changed_row"],
                   "columns": hd["columns"]},
        "fixture_level_enrichment_joins_in_v9": JOIN_INVENTORY,
    }
    if "serving_diff" in doc:
        sd = doc["serving_diff"]
        extra["serving_replay"] = {k: sd[k] for k in ("board_fixtures", "recipe", "tips_legacy", "tips_safe",
                                                       "tips_only_legacy", "tips_only_safe")}
        extra["serving_replay"]["features_changed"] = {k: v["fixtures_changed"] for k, v in
                                                        sd["serving_features_changed"].items()}
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
    ap.add_argument("--serving", action="store_true")
    ap.add_argument("--health", default="output/join_integrity_health.json",
                    help="also write the join integrity health artifact here ('' to skip)")
    a = ap.parse_args(argv)
    serving_flag = os.getenv("WOWZA_SAFE_ENRICH_JOINS", "0").strip()
    os.environ.pop("APIFOOTBALL_KEY", None)
    import logging
    logging.basicConfig(level=logging.WARNING)
    import config
    config.API_KEY = ""
    base = base_history()
    L, repL = enrich(base, safe=False)
    S, repS = enrich(base, safe=True)
    os.environ["WOWZA_SAFE_ENRICH_JOINS"] = serving_flag
    doc = {"artifact": "join_integrity_replay", "label": "RETROSPECTIVE", "serving_flag": serving_flag,
           "inputs": ["output/fd_history.parquet", "output/backtest_all_leagues.csv",
                      "output/af_history.parquet", "output/standard_sidemarket_odds_history.csv",
                      "output/standard_odds_history.csv", "output/newformat_odds_history.csv"],
           "join_reports_legacy": repL, "join_reports_safe": repS,
           "history_diff": compare(L, S)}
    if a.serving:
        doc["serving_diff"] = serving(L, S)
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    if a.health:
        write_health(doc, ROOT / a.health)
    hd = doc["history_diff"]
    print(f"rows {hd['rows_total']:,}; rows with any enriched value changed {hd['rows_with_any_change']:,}")
    for c, v in hd["columns"].items():
        if v["rows_changed"]:
            print(f"  {c:18s} shifted {v['shifted']:6d} phantom {v['phantom']:6d} dropped {v['dropped']:6d}")
    if a.serving:
        sd = doc["serving_diff"]
        print(f"serving: {len(sd['serving_features_changed'])} feature/prob columns change; "
              f"tips legacy {sd['tips_legacy']} safe {sd['tips_safe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
