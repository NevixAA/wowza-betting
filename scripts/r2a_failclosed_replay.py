"""R2-A offline replay of the committed board: BEFORE (any v9 tree) vs AFTER (this branch). READ-ONLY.

    # 1. once: load history with one tree's loader and freeze it (both trees then score the SAME frame)
    python scripts/r2a_failclosed_replay.py --root <tree> --dump-history H.pkl
    # 2. per tree (run with that tree's python path; --root decides which code is imported)
    python scripts/r2a_failclosed_replay.py --root <base tree>   --history H.pkl --out OUT/before
    python scripts/r2a_failclosed_replay.py --root <branch tree> --history H.pkl --out OUT/after
    # 3. compare
    python scripts/r2a_failclosed_replay.py --compare OUT/before OUT/after

Each tree runs src.predict.predict_upcoming + the production tip pipeline (scripts/standard_switch_preview
run_mode: generate_bets, side bets, direction guard, paper strip, flip guard) on the fixtures and prices of
the committed output/predictions.csv, under the env predict.yml deploys, with OddsAPI stubbed, every writer
stubbed and the network blocked. Scenarios:

  board            the committed board as is (production today)
  near_names       the board + copies of real fixtures relabelled with near-name / alias leagues
                   ("Austria Bundesliga", "championship", "Serie B ", "Premier League")
  repaired_missing the board with models/model_v9_standard_repaired.pkl pointed at a missing file
  nf_missing       the board with the New Format artifact not loaded (payload_newformat=None)

Nothing under output/, models/ or telegram_bot/ is written; results go to --out only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

NEAR = ["Austria Bundesliga", "championship", "Serie B ", "Premier League"]
KEY = ["league", "home_team", "away_team"]
PRED_COLS = ["league", "home_team", "away_team", "model_type", "p_over25", "bet", "signal_tier", "bet_stake",
             "best_edge", "p_btts", "p_over15", "p_over35", "p_ht_over05", "p_ht_over15", "serving_recipe",
             "model_version"]


def _setup(root: Path):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    os.chdir(root)
    import standard_switch_preview as SP
    for k, v in SP.deployed_env().items():
        os.environ[k] = v
    for k in ("APIFOOTBALL_KEY", "ODDS_API_KEY", "API_KEY"):
        os.environ.pop(k, None)
    os.environ["AF_OU_CACHE_ONLY"] = "1"
    return SP


def dump_history(root: Path, out: Path) -> None:
    SP = _setup(root)
    from src.data_loader import load_all_matches
    h = load_all_matches()
    out.write_bytes(pickle.dumps(h))
    print(f"history {len(h)} rows -> {out}")


def _frame_hash(df) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


def run_tree(root: Path, hist_pkl: Path, out: Path) -> None:
    SP = _setup(root)
    import logging
    logging.basicConfig(level=logging.WARNING)
    import pandas as pd
    import config
    from src.model import load_models
    import src.drift as D
    import src.health_check as H
    D.snapshot = lambda *a_, **k_: None
    D.purge_old = lambda *a_, **k_: None
    H.record_fetch = lambda *a_, **k_: None
    H.record_feature_health = lambda *a_, **k_: None
    SP._block_network()
    historical = pickle.loads(hist_pkl.read_bytes())
    upcoming, board = SP.board_fixtures(root / "output" / "predictions.csv")
    payloads = {"std": load_models(model_file=config.MODEL_FILE_STANDARD),
                "nf": load_models(model_file=config.MODEL_FILE_NEWFORMAT),
                "side": {t: load_models(model_file=f) for t, f in config.SIDE_MARKETS.items() if f.exists()}}
    recipe = config.STANDARD_OU25_SERVING_RECIPE
    out.mkdir(parents=True, exist_ok=True)
    real_repaired = config.MODEL_FILE_STANDARD_REPAIRED

    # near-name copies: two real Standard fixtures and one NF fixture, relabelled
    src_rows = upcoming[upcoming["league"].isin(config.STANDARD_FORMAT_LEAGUES)].head(2)
    src_rows = pd.concat([src_rows, upcoming[upcoming["league"].isin(config.NEW_FORMAT_LEAGUES)].head(2)])
    near = []
    for i, name in enumerate(NEAR):
        r = src_rows.iloc[i % len(src_rows)].copy()
        r["league"] = name
        near.append(r)
    up_near = pd.concat([upcoming, pd.DataFrame(near)], ignore_index=True)

    scen = {"board": (upcoming, payloads, real_repaired),
            "near_names": (up_near, payloads, real_repaired),
            "repaired_missing": (upcoming, payloads, root / "models" / "__r2a_missing__.pkl"),
            "nf_missing": (upcoming, {**payloads, "nf": None}, real_repaired)}
    summary = {"root": str(root), "recipe": recipe, "board_rows": int(len(board)), "fixtures": int(len(upcoming))}
    for name, (up, pl, rep_file) in scen.items():
        config.MODEL_FILE_STANDARD_REPAIRED = rep_file
        try:
            R = SP.run_mode(recipe, up, historical, pl)
        except Exception as e:                                         # noqa: BLE001
            summary[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        finally:
            config.MODEL_FILE_STANDARD_REPAIRED = real_repaired
        preds = R["preds"]
        cols = [c for c in PRED_COLS if c in preds.columns]
        p = preds[cols].sort_values(KEY).reset_index(drop=True)
        p.to_csv(out / f"{name}_preds.csv", index=False)
        full = preds.sort_values(KEY).reset_index(drop=True)
        full.to_csv(out / f"{name}_preds_full.csv", index=False)
        tips = []
        for kind in ("send_bets", "send_side"):
            d = R[kind]
            if d is None or d.empty:
                continue
            d = d.copy()
            d["kind"] = "OU25" if kind == "send_bets" else d.get("market", "side")
            side = d["side"] if "side" in d.columns else d.get("bet")
            d["side_"] = side.astype(str).str.upper()
            tips.append(d[["league", "home_team", "away_team", "kind", "side_", "signal_tier"]
                          + [c for c in ("bet_stake", "stake", "edge", "odds") if c in d.columns]])
        T = (pd.concat(tips, ignore_index=True).sort_values(KEY + ["kind"]).reset_index(drop=True)
             if tips else pd.DataFrame(columns=KEY + ["kind", "side_", "signal_tier"]))
        T.to_csv(out / f"{name}_tips.csv", index=False)
        summary[name] = {"rows": int(len(preds)), "tips": int(len(T)), "preds_hash": _frame_hash(p),
                         "tips_hash": _frame_hash(T),
                         "model_type_counts": preds["model_type"].value_counts().to_dict(),
                         "serving_recipe_counts": (preds["serving_recipe"].value_counts().to_dict()
                                                   if "serving_recipe" in preds.columns else {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))


def compare(before: Path, after: Path) -> int:
    import pandas as pd
    res = {}
    for name in ("board", "near_names", "repaired_missing", "nf_missing"):
        r = {}
        for kind in ("preds", "tips"):
            fb, fa = before / f"{name}_{kind}.csv", after / f"{name}_{kind}.csv"
            if not (fb.exists() and fa.exists()):
                r[kind] = "missing"
                continue
            b, a = pd.read_csv(fb), pd.read_csv(fa)
            r[f"{kind}_identical_bytes"] = fb.read_bytes() == fa.read_bytes()
            if kind == "tips":
                m = b.merge(a, how="outer", indicator=True, on=list(b.columns.intersection(a.columns)))
                r["tips_only_before"] = m[m["_merge"] == "left_only"][KEY + ["kind", "side_", "signal_tier"]] \
                    .to_dict("records")
                r["tips_only_after"] = m[m["_merge"] == "right_only"][KEY + ["kind", "side_", "signal_tier"]] \
                    .to_dict("records")
            else:
                cols = [c for c in b.columns if c in a.columns and c != "model_version"]
                m = b[cols].merge(a[cols], on=KEY, how="outer", suffixes=("_b", "_a"), indicator=True)
                diff_rows = []
                for _, x in m.iterrows():
                    d = {c: (x.get(f"{c}_b"), x.get(f"{c}_a")) for c in cols if c not in KEY
                         and not (pd.isna(x.get(f"{c}_b")) and pd.isna(x.get(f"{c}_a")))
                         and str(x.get(f"{c}_b")) != str(x.get(f"{c}_a"))}
                    if d or x["_merge"] != "both":
                        diff_rows.append({**{k: x[k] for k in KEY}, "merge": x["_merge"], **{k: list(map(str, v))
                                                                                          for k, v in d.items()}})
                r["pred_rows_changed"] = diff_rows
                mv = b.merge(a, on=KEY, suffixes=("_b", "_a"))
                if "model_version_b" in mv and "model_version_a" in mv:
                    ch = mv[mv["model_version_b"].astype(str) != mv["model_version_a"].astype(str)]
                    r["model_version_changed"] = ch[KEY + ["model_version_b", "model_version_a"]].to_dict("records")
        res[name] = r
    print(json.dumps(res, indent=1, default=str))
    (after.parent / "compare.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--dump-history")
    ap.add_argument("--history")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args(argv)
    if a.compare:
        return compare(Path(a.compare[0]), Path(a.compare[1]))
    root = Path(a.root).resolve()
    if a.dump_history:
        dump_history(root, Path(a.dump_history))
        return 0
    run_tree(root, Path(a.history), Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
