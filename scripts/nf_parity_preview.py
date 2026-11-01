"""BEFORE/AFTER of NF_FEATURE_PARITY (2026-11-01) on the committed board. READ-ONLY, offline.

    python scripts/nf_parity_preview.py --out _nf_preview

Runs the production predict pass twice on the fixtures of output/predictions.csv (serving recipe as
deployed), NF_FEATURE_PARITY=0 (old serving features) then =1 (home_advantage and new-format
overround computed as training does), with the same tip pipeline as scripts/standard_switch_preview.
Best-of-books O/U 2.5 prices come from the committed output/book_odds_snapshots.csv (last pre-kickoff
quote per book), because the board predates the best_over25/best_under25 columns.
Nothing under output/, models/ or telegram_bot/ is written.
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

import standard_switch_preview as SP  # noqa: E402


def attach_best(upcoming):
    import pandas as pd
    b = pd.read_csv(ROOT / "output" / "book_odds_snapshots.csv", low_memory=False)
    b = b[(b["market"] == "OU25") & b["side"].isin(["OVER", "UNDER"])]
    b = b.sort_values("snapshot_ts").drop_duplicates(["match", "bookmaker", "side"], keep="last")
    best = b.pivot_table(index="match", columns="side", values="odds", aggfunc="max")
    key = upcoming["home_team"].astype(str) + " vs " + upcoming["away_team"].astype(str)
    upcoming = upcoming.copy()
    upcoming["best_over25"] = key.map(best.get("OVER", pd.Series(dtype=float)))
    upcoming["best_under25"] = key.map(best.get("UNDER", pd.Series(dtype=float)))
    return upcoming


def tips(frames, nf_leagues):
    import pandas as pd
    out = []
    for name in ("send_bets", "send_side"):
        d = frames[name]
        if d is None or d.empty:
            continue
        d = d.copy()
        d["kind"] = "OU25" if name == "send_bets" else d.get("market", "side")
        d["track"] = d["league"].map(lambda l: "new_format" if l in nf_leagues else "standard")
        side = d["side"] if "side" in d.columns else d.get("bet")
        d["side_"] = side.astype(str).str.upper()
        out.append(d[["league", "home_team", "away_team", "kind", "track", "side_", "signal_tier"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    env = SP.deployed_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ.pop("APIFOOTBALL_KEY", None)
    os.environ.pop("ODDS_API_KEY", None)
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
    from src.data_loader import load_all_matches
    SP._block_network()
    historical = load_all_matches()

    upcoming, board = SP.board_fixtures(ROOT / "output" / "predictions.csv")
    upcoming = attach_best(upcoming)
    payloads = {"std": load_models(model_file=config.MODEL_FILE_STANDARD),
                "nf": load_models(model_file=config.MODEL_FILE_NEWFORMAT),
                "side": {t: load_models(model_file=f) for t, f in config.SIDE_MARKETS.items() if f.exists()}}
    recipe = env.get("STANDARD_OU25_SERVING_RECIPE", config.STANDARD_OU25_SERVING_RECIPE)
    res = {}
    for flag in ("0", "1"):
        os.environ["NF_FEATURE_PARITY"] = flag
        res[flag] = SP.run_mode(recipe, upcoming, historical, payloads)
    nf = set(config.NEW_FORMAT_LEAGUES)

    P0, P1 = res["0"]["preds"], res["1"]["preds"]
    k = ["league", "home_team", "away_team"]
    m = P0[k + ["model_type", "p_over25", "odds_over25", "odds_under25"] + [c for c in ("p_btts", "p_over15", "p_over35") if c in P0]].merge(
        P1[k + ["p_over25"] + [c for c in ("p_btts", "p_over15", "p_over35") if c in P1]], on=k, suffixes=("_old", "_new"))
    io_, iu_ = 1 / m["odds_over25"], 1 / m["odds_under25"]
    m["market_p_over"] = io_ / (io_ + iu_)
    summary = {"board_rows": int(len(board)), "recipe": recipe, "best_price_coverage": float(upcoming["best_over25"].notna().mean())}
    for mt, g in m.groupby("model_type"):
        s = {"fixtures": int(len(g)), "market_p_over": round(float(g["market_p_over"].mean()), 4),
             "p_over25_old": round(float(g["p_over25_old"].mean()), 4), "p_over25_new": round(float(g["p_over25_new"].mean()), 4),
             "league_sd_old": round(float(g.groupby("league")["p_over25_old"].mean().std()), 4),
             "league_sd_new": round(float(g.groupby("league")["p_over25_new"].mean().std()), 4),
             "league_sd_market": round(float(g.groupby("league")["market_p_over"].mean().std()), 4)}
        for c in ("p_btts", "p_over15", "p_over35"):
            if f"{c}_old" in g:
                s[f"{c}_mean_shift_pp"] = round(float((g[f"{c}_new"] - g[f"{c}_old"]).mean() * 100), 2)
        summary[mt] = s
    T0, T1 = tips(res["0"], nf), tips(res["1"], nf)
    def count(T):
        if T.empty:
            return {}
        return {f"{t}|{kd}|{sd}": int(n) for (t, kd, sd), n in T.groupby(["track", "kind", "side_"]).size().items()}
    summary["tips_old"], summary["tips_new"] = count(T0), count(T1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "nf_parity_fixtures.csv", index=False)
    T0.assign(version="old").pipe(lambda x: pd.concat([x, T1.assign(version="new")])).to_csv(out / "nf_parity_tips.csv", index=False)
    (out / "nf_parity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
