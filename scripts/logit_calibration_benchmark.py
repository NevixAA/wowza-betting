"""Champion ("prob") vs challenger ("logit") calibration space, identical rows. RESEARCH ONLY.

    python scripts/logit_calibration_benchmark.py --task std_ou25 --out _out

Tasks (the serving structure of each model is kept; results are always reported PER TRACK):
  std_ou25     Standard O/U 2.5, v9_repaired recipe (identity-canonical, as-of features) - as served
  nf_ou25      New Format O/U 2.5, legacy features (as served), NF rows only
  side_btts / side_over15 / side_over35
               side markets, legacy features, trained POOLED on Standard + New Format rows exactly as
               pipeline.mode_train does; every metric is reported per track, never pooled

Both arms are trained by src.model.train_repaired on the SAME four chronological blocks (base ->
calibration -> meta -> untouched holdout = newest 20%), with the serving payload's own feature
list and production's weights. The only difference between the arms is calibration_space.
Outputs: <out>/<task>_holdout.csv (per-row predictions) and <out>/<task>.json (metrics).
Nothing under models/ or output/ is written.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import config  # noqa: E402
from src import model as M  # noqa: E402
from src import standard_repair as SR  # noqa: E402

LIVE_FROM = "2026-09-01"                  # August 2026 had a production bug; live evidence from here
ALIAS_MAP = ROOT / "config" / "standard_ou25_identity_aliases.json"
SIDE_FILE = {"btts": config.MODEL_FILE_BTTS, "over15": config.MODEL_FILE_OVER15, "over35": config.MODEL_FILE_OVER35}
SIDE_ODDS = {"btts": "odds_btts", "over15": "odds_over15", "over35": "odds_over35"}
RNG = np.random.default_rng(20261101)


def _feats(path) -> list:
    with open(path, "rb") as f:
        return list(pickle.load(f)["feature_cols"])


def frame(task: str):
    raw = SR.load_history()
    if task == "std_ou25":
        hist, _ = SR.prepare_history(raw, "v9_repaired", str(ALIAS_MAP), True)
        feat = SR.feature_frame(hist, "v9_repaired", "none", None, "league")
        rows = SR.track_rows(feat, "standard", "over25")
        return rows, "over25", _feats(config.MODEL_FILE_STANDARD_REPAIRED), SR.derived_weights(rows), "OU25"
    feat = SR.feature_frame(raw, "legacy_v9")
    if task == "nf_ou25":
        rows = SR.track_rows(feat, "newformat", "over25")
        return rows, "over25", _feats(config.MODEL_FILE_NEWFORMAT), SR.legacy_weights(rows), "OU25"
    target = task.split("_", 1)[1]
    rows = SR.track_rows(feat, "pooled", target)
    return rows, target, _feats(SIDE_FILE[target]), SR.legacy_weights(rows), target


def market_p(d: pd.DataFrame, task: str, target: str) -> tuple[pd.Series, str]:
    """Market probability of the modelled outcome, and its class."""
    if target == "over25" and {"odds_over25", "odds_under25"} <= set(d.columns):
        o, u = pd.to_numeric(d["odds_over25"], errors="coerce"), pd.to_numeric(d["odds_under25"], errors="coerce")
        io, iu = 1 / o, 1 / u
        return io / (io + iu), "TWO_SIDED_DEVIG"
    col = SIDE_ODDS.get(target)
    if col and col in d.columns:
        o = pd.to_numeric(d[col], errors="coerce")
        return (1 / o / 1.06).clip(0.01, 0.99), "ONE_SIDED_PROXY_6PCT_MARGIN"
    return pd.Series(np.nan, index=d.index), "NONE"


def _lg(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _boot_mean(x, groups, n=2000):
    d = pd.DataFrame({"x": np.asarray(x, float), "g": np.asarray(groups)}).dropna()
    s = d.groupby("g")["x"].agg(["sum", "count"])
    if len(s) < 5:
        return [None, None]
    i = RNG.integers(0, len(s), (n, len(s)))
    m = s["sum"].to_numpy()[i].sum(1) / s["count"].to_numpy()[i].sum(1)
    return [round(float(np.percentile(m, 2.5)), 5), round(float(np.percentile(m, 97.5)), 5)]


def cal_line(y, p) -> dict:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_lg(p).reshape(-1, 1), y)
    return {"slope": round(float(lr.coef_[0][0]), 3), "intercept": round(float(lr.intercept_[0]), 3)}


def league_view(h: pd.DataFrame) -> tuple[dict, list]:
    g = h.groupby("league").agg(n=("y", "size"), actual=("y", "mean"), prob=("p_prob", "mean"),
                                logit=("p_logit", "mean"), market=("p_mkt", "mean"))
    g = g[g["n"] >= 30]
    out = {"leagues": int(len(g)), "sd_actual": round(float(g["actual"].std()), 4),
           "sd_prob": round(float(g["prob"].std()), 4), "sd_logit": round(float(g["logit"].std()), 4),
           "sd_market": round(float(g["market"].std()), 4) if g["market"].notna().any() else None}
    for arm in ("prob", "logit"):
        if len(g) >= 3:
            b = np.polyfit(g[arm], g["actual"], 1, w=np.sqrt(g["n"]))[0]
            out[f"league_slope_{arm}"] = round(float(b), 3)    # 1.0 = league levels right; >1 = squashed
        out[f"mean_abs_league_error_{arm}_pp"] = round(float((g[arm] - g["actual"]).abs().mean() * 100), 2)
    rows = [{"league": lg, "n": int(r.n), "actual": round(r.actual, 4), "prob": round(r.prob, 4),
             "logit": round(r.logit, 4), "market": None if pd.isna(r.market) else round(r.market, 4)}
            for lg, r in g.iterrows()]
    return out, rows


def residual(h: pd.DataFrame, arm: str) -> dict:
    """Does market + model beat market alone? Fit on the first half of the holdout (by date),
    score on the second half. Gain > 0 = the model adds information to the price."""
    from sklearn.linear_model import LogisticRegression
    d = h.dropna(subset=["p_mkt"]).sort_values("date")
    if len(d) < 200:
        return {"n": int(len(d)), "status": "TOO_FEW"}
    cut = d["date"].iloc[len(d) // 2]
    a, b = d[d["date"] < cut], d[d["date"] >= cut]
    if len(a) < 100 or len(b) < 100:
        return {"n": int(len(d)), "status": "TOO_FEW"}
    m0 = LogisticRegression(C=1e6, max_iter=1000).fit(_lg(a["p_mkt"]).reshape(-1, 1), a["y"])
    X = lambda x: np.c_[_lg(x["p_mkt"]), _lg(x[f"p_{arm}"])]
    m1 = LogisticRegression(C=1e6, max_iter=1000).fit(X(a), a["y"])
    l0 = _ll(b["y"].to_numpy(), m0.predict_proba(_lg(b["p_mkt"]).reshape(-1, 1))[:, 1])
    l1 = _ll(b["y"].to_numpy(), m1.predict_proba(X(b))[:, 1])
    gain = l0 - l1
    return {"n_fit": int(len(a)), "n_score": int(len(b)), "ll_market": round(float(l0.mean()), 5),
            "ll_market_plus_model": round(float(l1.mean()), 5), "gain": round(float(gain.mean()), 5),
            "gain_ci95": _boot_mean(gain, b["day"]), "weight_on_model": round(float(m1.coef_[0][1]), 3)}


def evaluate(h: pd.DataFrame) -> dict:
    y = h["y"].to_numpy()
    out = {"n": int(len(h)), "matchdays": int(h["day"].nunique()),
           "dates": [str(h["date"].min().date()), str(h["date"].max().date())], "base_rate": round(float(y.mean()), 4)}
    for arm in ("prob", "logit"):
        p = h[f"p_{arm}"].to_numpy()
        out[arm] = {"log_loss": round(float(_ll(y, p).mean()), 5), "brier": round(float(np.mean((p - y) ** 2)), 5),
                    "mean_p": round(float(p.mean()), 4), "sd_p": round(float(p.std()), 4), "calibration": cal_line(y, p),
                    "residual_vs_market": residual(h, arm)}
    d_ll = _ll(y, h["p_logit"]) - _ll(y, h["p_prob"])
    out["delta_log_loss_logit_minus_prob"] = round(float(d_ll.mean()), 5)
    out["delta_log_loss_ci95"] = _boot_mean(d_ll, h["day"])
    d_br = (h["p_logit"] - y) ** 2 - (h["p_prob"] - y) ** 2
    out["delta_brier_logit_minus_prob"] = round(float(d_br.mean()), 5)
    out["delta_brier_ci95"] = _boot_mean(d_br, h["day"])
    if h["p_mkt"].notna().any():
        mk = h.dropna(subset=["p_mkt"])
        out["market"] = {"n": int(len(mk)), "log_loss": round(float(_ll(mk["y"].to_numpy(), mk["p_mkt"]).mean()), 5),
                         "brier": round(float(np.mean((mk["p_mkt"] - mk["y"]) ** 2)), 5)}
    lv, rows = league_view(h)
    out["league_level"] = lv
    out["per_league"] = rows
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["std_ou25", "nf_ou25", "side_btts", "side_over15", "side_over35"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, target, cols, w, label = frame(a.task)
    rows = rows.copy()
    rows["_w_in"] = np.asarray(w, float)
    blocks = M.four_block_split(rows, target, rows["_w_in"].to_numpy())
    pays = {}
    for space in M.CALIBRATION_SPACES:
        res = M.train_repaired(rows, target=target, sample_weight=rows["_w_in"].to_numpy(), feature_cols=cols,
                               calibration_space=space)
        pays[space] = M.build_payload(res, target)
    hold = blocks["holdout"].copy()
    h = pd.DataFrame({"date": pd.to_datetime(hold["date"]).to_numpy(), "league": hold["league"].astype(str).to_numpy(),
                      "home_team": hold["home_team"].to_numpy(), "away_team": hold["away_team"].to_numpy(),
                      "y": pd.to_numeric(hold[target], errors="coerce").to_numpy()})
    for space, pay in pays.items():
        h[f"p_{space}"] = M.predict_proba(hold, payload=pay).to_numpy()
    mp, mclass = market_p(hold, a.task, target)
    h["p_mkt"] = mp.to_numpy()
    h["track"] = h["league"].map(config.model_type_for_league)
    h["day"] = h["date"].dt.strftime("%Y-%m-%d")
    h = h.dropna(subset=["y"])
    h.to_csv(out / f"{a.task}_holdout.csv", index=False)
    tracks = ["standard"] if a.task == "std_ou25" else ["new_format"] if a.task == "nf_ou25" else ["standard", "new_format"]
    res = {"task": a.task, "target": target, "market_class": mclass, "n_features": len(cols),
           "training_rows": int(len(rows)), "blocks": {k: [int(len(v)), str(pd.Timestamp(v["date"].min()).date()),
                                                            str(pd.Timestamp(v["date"].max()).date())] for k, v in blocks.items()},
           "pooled_training": a.task.startswith("side_"),
           "coefficients": {space: {"platt": {k: [round(float(m._calibrator.coef_[0][0]), 3), round(float(m._calibrator.intercept_[0]), 3)]
                                              for k, m in pay["models"].items() if k != "__meta__"},
                                    "meta": [round(float(c), 3) for c in pay["models"]["__meta__"].coef_[0]]}
                            for space, pay in pays.items()},
           "tracks": {}}
    for t in tracks:
        ht = h[h["track"] == t]
        if len(ht) < 100:
            res["tracks"][t] = {"n": int(len(ht)), "status": "TOO_FEW"}
            continue
        res["tracks"][t] = {"holdout": evaluate(ht)}
        live = ht[ht["date"] >= LIVE_FROM]
        if len(live) >= 100:
            res["tracks"][t]["live_from_2026_09_01"] = evaluate(live)
    (out / f"{a.task}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({t: {k: v.get("holdout", v).get(k) for k in ("n", "delta_log_loss_logit_minus_prob", "delta_log_loss_ci95")}
                      for t, v in res["tracks"].items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
