"""Train/serve feature parity health for every serving model (R2-C finding 18). READ-ONLY inputs.

    python scripts/integrity_feature_parity.py            # writes the two health files below

    output/feature_parity_health.json      all tracks / all serving models
    output/nf_feature_parity_health.json   the new-format slice (NF OU2.5 + side models on NF rows)

TRAIN side: the training frame rebuilt OFFLINE from the committed caches exactly as the CI loader
builds it (scripts/integrity_af_join_replay.base_history + the serving enrichment joins, which are
always league-keyed), COVID seasons excluded per league (config.covid_season_mask), then the feature
builder the model was trained with:
  standard OU2.5     v9_repaired recipe (src/features_repaired), standard-format leagues
  new-format OU2.5   legacy_v9 recipe, new-format leagues (27-feature NF payload)
  BTTS/O1.5/O3.5     legacy_v9 recipe, pooled training, reported per track
  HT O0.5/O1.5       legacy_v9 recipe, standard track (the only track HT serves)
Rows = the training filter (over25 and home_scored_last5 present). Time-decay weights are NOT applied
to the statistics, so a feature whose distribution drifted across seasons can show a PSI the
weighted training distribution would not; the verdict notes the training window used.

LIVE side: output/predictions.csv as committed (the latest predict board), split by model_type.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)


def _models():
    import config
    from src.model import load_models
    m = {"standard|OU25": config.MODEL_FILE_STANDARD_REPAIRED,
         "new_format|OU25": config.MODEL_FILE_NEWFORMAT}
    for t, f in config.SIDE_MARKETS.items():
        m[f"side|{t}"] = f
    for nm in ("ht_over05", "ht_over15"):
        m[f"ht|{nm}"] = config.MODELS_DIR / f"model_{nm}.pkl"
    out = {}
    for k, f in m.items():
        f = Path(f)
        if f.exists():
            try:
                out[k] = (f, load_models(model_file=f)["feature_cols"])
            except Exception as e:                                # noqa: BLE001
                out[k] = (f, None)
                print(f"[parity] could not load {f.name}: {e}")
    return out


def train_frames():
    import config
    from integrity_af_join_replay import base_history
    from src import data_loader as DL
    from src.feature_engineering import build_features
    os.environ.pop("APIFOOTBALL_KEY", None)
    config.API_KEY = ""
    raw = base_history()
    raw = DL._enrich_with_api_shots(raw)
    raw = DL._enrich_with_standard_sidemarket_odds(raw)
    raw = DL._enrich_ou25_from_captures(raw)
    if config.EXCLUDE_COVID_SEASONS:
        raw = raw[~config.covid_season_mask(raw)]
    legacy = build_features(raw)
    legacy = legacy.dropna(subset=["over25", "home_scored_last5"])
    std_raw = raw[raw["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    rep = build_features(std_raw, recipe="v9_repaired")
    rep = rep.dropna(subset=[c for c in ("over25", "home_scored_last5") if c in rep.columns])
    return legacy, rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="output/predictions.csv")
    ap.add_argument("--out-all", default="output/feature_parity_health.json")
    ap.add_argument("--out-nf", default="output/nf_feature_parity_health.json")
    a = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.WARNING)
    import pandas as pd
    import config
    from src.integrity.parity import parity_report, PARITY_RULE_VERSION

    board = pd.read_csv(ROOT / a.board, low_memory=False)
    live_std = board[board["model_type"] == "standard"]
    live_nf = board[board["model_type"] == "new_format"]
    legacy, rep = train_frames()
    tr_std = legacy[legacy["league"].isin(config.STANDARD_FORMAT_LEAGUES)]
    tr_nf = legacy[legacy["league"].isin(config.NEW_FORMAT_LEAGUES)]
    models = _models()

    reports = {}
    for key, (f, feats) in models.items():
        if not feats:
            reports[key] = {"status": "UNKNOWN", "reason": f"payload {Path(f).name} unreadable"}
            continue
        kind, name = key.split("|")
        if kind == "standard":
            reports[key] = parity_report(rep, live_std, feats, "standard", f"{Path(f).name}")
        elif kind == "new_format":
            reports[key] = parity_report(tr_nf, live_nf, feats, "new_format", f"{Path(f).name}")
        elif kind == "side":
            reports[f"standard|{name}"] = parity_report(tr_std, live_std, feats, "standard", Path(f).name)
            reports[f"new_format|{name}"] = parity_report(tr_nf, live_nf, feats, "new_format", Path(f).name)
        elif kind == "ht":
            reports[f"standard|{name}"] = parity_report(tr_std, live_std, feats, "standard", Path(f).name)

    gen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    board_meta = {"path": a.board, "rows": int(len(board)),
                  "generated_at": str(board["generated_at"].iloc[0]) if "generated_at" in board and len(board) else None,
                  "git_sha": str(board["git_sha"].iloc[0]) if "git_sha" in board and len(board) else None,
                  "rows_standard": int(len(live_std)), "rows_new_format": int(len(live_nf))}

    def _overall(rs):
        st = [r.get("status") for r in rs.values()]
        return "FAIL" if "FAIL" in st else "WARN" if "WARN" in st else "PASS" if st else "UNKNOWN"

    common = {"generated_at": gen, "rule_version": PARITY_RULE_VERSION, "live_board": board_meta,
              "train_source": "offline rebuild from committed caches (fd_history + backtest archive + "
                              "af_history + odds captures), serving enrichment joins "
                              "(league-keyed), unweighted, COVID seasons excluded",
              "nf_feature_parity_flag": os.getenv("NF_FEATURE_PARITY", "1")}
    all_doc = {"artifact": "feature_parity_health", **common, "status": _overall(reports),
               "models": reports}
    nf = {k: v for k, v in reports.items() if k.startswith("new_format|")}
    nf_doc = {"artifact": "nf_feature_parity_health", **common, "status": _overall(nf), "models": nf}
    for path, doc in ((a.out_all, all_doc), (a.out_nf, nf_doc)):
        p = ROOT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    for k, r in reports.items():
        bad = [f"{f}:{s['status']}" for f, s in r.get("features", {}).items() if s["status"] != "PASS"]
        print(f"{k:24s} {r.get('status'):5s} {r.get('counts')} {bad[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
