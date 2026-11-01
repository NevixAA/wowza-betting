"""AU2 forensic audit - PLAYER PROPS canary train-vs-serve test (Phase 2B). READ-ONLY.

Training rows are the rows of player_history.parquet/ (written by player_model.feature_engineering
.build_features at collect time; mode_train reads them as-is). For a sample of recent player-matches
we rebuild the SERVING row exactly as player_model.predict.run_player_predictions does:

  * history  = player_history rows dated STRICTLY BEFORE the match (identical history),
  * upcoming = the player's latest history row before the match (`team_players ... drop_duplicates
               ("player_id", keep="last")`, then .to_dict("records")), with is_home / opponent set
               from the actual fixture, as predict sets them from the fixture,
  * referee_profile = {} and match_context = {} (production: referee profiles only with an
    API-Football key; match contexts come only from worldcup_history.json h2h odds, i.e. none for
    club football),
  * player_model.feature_engineering.build_upcoming_features(...).

Then every one of the 138 payload features is compared (same tolerances as au2_canary.py) and each
of the 8 market payloads scores both rows in ONE batch per builder.

    python scripts/audit/au2_props_canary.py [--n 400]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, AUDITED_COMMIT, heavy_lock  # noqa: E402
from au2_canary import compare, P_TOL  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
WINDOW = (pd.Timestamp("2026-10-10"), pd.Timestamp("2026-10-30"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    from player_model import config as pcfg
    from player_model.feature_engineering import build_upcoming_features
    from player_model.model import load_model, predict_proba

    hist = pd.read_parquet(ROOT / "player_history.parquet")
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None) if getattr(pd.to_datetime(hist["date"]).dt, "tz", None) else pd.to_datetime(hist["date"])
    cand = hist[(hist["date"] >= WINDOW[0]) & (hist["date"] <= WINDOW[1])]
    # clubs only (internationals carry a country as team - see CLAUDE.md invariant 12)
    if "is_international" in cand.columns:
        cand = cand[~cand["is_international"].astype(bool)]
    rng = np.random.default_rng(20261101)
    pick = cand.iloc[rng.choice(len(cand), size=min(a.n, len(cand)), replace=False)].sort_values("date")
    pay = {m: load_model(m) for m in pcfg.MARKETS}
    cols = pay[pcfg.MARKETS[0]]["feature_cols"]
    tr_rows, sv_rows, skipped = [], [], {"no_prior_history": 0}
    for d, g in pick.groupby("date"):
        h = hist[hist["date"] < d]
        by_pid = {pid: x for pid, x in h[h["player_id"].isin(g["player_id"])].sort_values("date").groupby("player_id")}
        for r in g.itertuples(index=False):
            r = r._asdict()
            ph = by_pid.get(r["player_id"])
            if ph is None or ph.empty:
                skipped["no_prior_history"] += 1
                continue
            up = ph.iloc[-1].to_dict()
            up["is_home"] = r.get("is_home")
            up["opponent"] = r.get("opponent")
            up["team"] = r.get("team")
            f = build_upcoming_features([up], h, {}, {}, history_index=(by_pid, {}))
            if f.empty:
                skipped["no_prior_history"] += 1
                continue
            sv_rows.append(f.iloc[0])
            tr_rows.append(pd.Series(r))
    tr = pd.DataFrame(tr_rows).reset_index(drop=True)
    sv = pd.DataFrame(sv_rows).reset_index(drop=True)
    agr = compare(tr, sv, cols, "PROPS_all_markets_shared_138")
    agr.to_csv(DATA / "props_canary_feature_agreement.csv", index=False)
    summ = {"audited_commit": AUDITED_COMMIT, "window": [str(WINDOW[0].date()), str(WINDOW[1].date())],
            "sampled": int(len(pick)), "compared": int(len(tr)), "skipped": skipped,
            "feature_verdicts": agr["verdict"].value_counts().to_dict(), "markets": {}}
    for m, p in pay.items():
        pt = predict_proba(tr, p).to_numpy()
        ps = predict_proba(sv, p).to_numpy()
        d = np.abs(pt - ps)
        summ["markets"][m] = {"n": int(len(d)), "pass_rate_0.5pp": round(float((d <= P_TOL).mean()), 4),
                              "median_abs_diff_pp": round(float(np.median(d)) * 100, 3),
                              "p90_abs_diff_pp": round(float(np.quantile(d, 0.9)) * 100, 3),
                              "mean_p_train": round(float(pt.mean()), 4), "mean_p_serve": round(float(ps.mean()), 4)}
    (DATA / "props_canary_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))
    print(agr[agr["verdict"] != "AGREE"].sort_values("close_rate")[["feature", "close_rate", "train_median", "serve_median", "train_missing", "serve_missing"]].to_string())
    return 0


if __name__ == "__main__":
    with heavy_lock("AU2 props canary"):
        raise SystemExit(main())
