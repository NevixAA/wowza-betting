"""RESEARCH (no serving change): are the tips' edges realistic? Owner 2026-11-11: "edges are supposed to be close to
the bookmakers; a 20% edge doesn't sound real (maybe once a month); 1-10% sounds realistic".

EDGE here = probability points over the bookmaker's de-vigged fair probability (power de-vig of the real two-sided
pair): raw = p_model - fair; served = p_used - fair (the cell's own probability source, the number a tip clears
its league's bar with when the measure is EDGE); calibrated = lam * (p_used - fair) with the cell's R2/R3 slope
learned from rows dated before the week (src/research/top3_ranking, the frozen procedure).

Data: the SELECTION window walk-forward bets of the frozen study_58f10476 choices (clean champion_recipe frame
58f10476f4c82d7b, own prices only; scripts/top3_ranking_build.py) = development; FINAL 2026-09-01..11-08 = SPENT,
descriptive only (its calibration uses the last SELECT fold's rows). League-wide cells: none in these frames.

Part 4 develops a PER-CELL upper-edge rule on the selection window only (every week uses only earlier data);
every configuration tried is listed. Writes aggregates only to output/research/top3_ranking/edge_realism.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import league_bars as LB  # noqa: E402
from src.research import top3_ranking as R  # noqa: E402

OUT = ROOT / "output" / "research" / "top3_ranking"
SEED = 20261111
N_BOOT = 2000
BANDS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, np.inf]
BAND_NAMES = ["0-2", "2-5", "5-10", "10-15", "15-20", ">20"]
CALIBRATED = {"BLEND", "SHRINK", "LEARNED"}


def cell_sources() -> dict:
    ch = json.loads((ROOT / "output/bars_opt/study_58f10476/choices.json").read_text(encoding="utf-8"))
    return {(c["track"], c["mside"]): c["chosen"].split("|")[0] for c in ch if c.get("chosen")}


def rows_before(rows: pd.DataFrame, w: pd.Timestamp) -> pd.DataFrame:
    return rows[(rows["fold_start"] <= w) & (rows["fold_end"] > w) & (rows["date"] < w)]


def last_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows["fold"] == rows.groupby(["track", "mside"])["fold"].transform("max")]


def add_edges(b: pd.DataFrame, lam_by_week) -> pd.DataFrame:
    b = b.copy()
    b["e_raw"] = b["p_side"] - b["fair"]
    b["e_srv"] = b["p_used"] - b["fair"]
    b["ev"] = b["p_used"] * b["odds"] - 1
    b["lam"] = lam_by_week(b)
    b["e_cal"] = b["lam"] * b["e_srv"]
    b["src"] = [SRC.get((t, m), "?") for t, m in zip(b["track"], b["mside"])]
    b["cell_type"] = np.where(b["src"] == "RAW", "RAW", np.where(b["src"].isin(CALIBRATED), "CALIBRATED", "OTHER"))
    b["lw"] = b["track"] + "|" + b["league"] + "|" + b["date"].dt.strftime("%G-W%V")
    return b


def lam_weekly(sel_rows: pd.DataFrame):
    def f(b):
        wk = b["date"].dt.to_period("W-SUN").dt.start_time
        lam = pd.Series(np.nan, index=b.index)
        for w in sorted(wk.unique()):
            st = R.cell_stats(None, rows_before(sel_rows, w))
            idx = b.index[wk == w]
            m = b.loc[idx, R.CELL].merge(st[R.CELL + ["lam"]], on=R.CELL, how="left")
            lam.loc[idx] = m["lam"].to_numpy()
        return lam
    return f


def lam_fixed(st: pd.DataFrame):
    def f(b):
        return b[R.CELL].merge(st[R.CELL + ["lam"]], on=R.CELL, how="left")["lam"].to_numpy()
    return f


def dist(b: pd.DataFrame) -> list[dict]:
    out = []
    for (mk, ct), g in [(("ALL", "ALL"), b)] + list(b.groupby(["mside", "cell_type"])):
        r = {"market": mk, "cell_type": ct, "tips": int(len(g))}
        for col in ("e_raw", "e_srv", "e_cal", "ev"):
            x = g[col].dropna()
            if not len(x):
                continue
            r[col] = {"p50": round(float(x.median()), 4), "p90": round(float(x.quantile(.9)), 4),
                      "max": round(float(x.max()), 4), "share_gt_10": round(float((x > .10).mean()), 3),
                      "share_gt_20": round(float((x > .20).mean()), 3), "n": int(len(x))}
        out.append(r)
    return out


def boot_roi(g: pd.DataFrame, rng) -> list:
    s = g.groupby("lw")["pnl"].agg(["sum", "size"])
    S, N = s["sum"].to_numpy(), s["size"].to_numpy()
    if len(S) < 3:
        return [None, None]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, len(S), len(S))
        d[i] = S[j].sum() / N[j].sum()
    return [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)]


def bands(b: pd.DataFrame, col: str, rng) -> list[dict]:
    b = b.assign(band=pd.cut(b[col], BANDS, labels=BAND_NAMES, right=False))
    out = []
    for mk, gm in [("ALL", b)] + list(b.groupby("mside")):
        for bd, g in gm.groupby("band", observed=False):
            if not len(g):
                continue
            out.append({"market": mk, "band_pp": str(bd), "tips": int(len(g)), "roi": round(float(g["pnl"].mean()), 4),
                        "roi_ci95_lw": boot_roi(g, rng), "win_minus_fair": round(float(g["won"].mean() - g["fair"].mean()), 4),
                        "win_minus_implied": round(float(g["won"].mean() - (1 / g["odds"]).mean()), 4),
                        "mean_edge_claimed": round(float(g[col].mean()), 4)})
    return out


def cal_shift(b: pd.DataFrame) -> dict:
    hi = b[b["e_srv"] > .10]
    ok = hi["e_cal"].notna()
    return {"served_gt_10": int(len(hi)), "with_calibration": int(ok.sum()),
            "calibrated_le_10": int((hi.loc[ok, "e_cal"] <= .10).sum()),
            "share_calibrated_le_10": round(float((hi.loc[ok, "e_cal"] <= .10).mean()), 3) if ok.any() else None,
            "calibrated_in_1_10_all_tips": round(float(b["e_cal"].between(.01, .10).mean()), 3),
            "calibrated_le_0_all_tips": round(float((b["e_cal"] <= 0).mean()), 3),
            "lam_by_cell_type": {k: {"median": round(float(g["lam"].median()), 3), "n": int(g["lam"].notna().sum())}
                                 for k, g in b.groupby("cell_type")}}


# ── part 4: per-cell upper-edge cap, learned from the cell's own earlier bets, shrunk within the league ──
def _curve(e: np.ndarray, pnl: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in LB.GRID:
        m = e <= t
        n = int(m.sum())
        rows.append({"t": float(t), "n": n, "roi": float(pnl[m].mean()) if n else np.nan,
                     "var": float(pnl[m].var(ddof=1)) if n > 1 else np.nan})
    return pd.DataFrame(rows)


def caps(prior: pd.DataFrame, objective: str) -> dict:
    """{cell: cap or None}. Per league: each cell's ROI curve of bets with served edge <= t, shrunk toward the
    league's own all-market curve (DL tau^2 across the league's cells at t, as src.league_bars does across
    leagues); cap = argmax over t with >= MIN_OWN_BETS own bets of the shrunk ROI (objective ROI) or of
    shrunk ROI x bets (objective PROFIT, the MARKSMAN rule). A cap at or above the cell's largest edge = none."""
    out = {}
    for (tr, lg), gl in prior.groupby(["track", "league"]):
        curves = {m: _curve(g["e_srv"].to_numpy(float), g["pnl"].to_numpy(float)) for m, g in gl.groupby("mside")}
        pooled = _curve(gl["e_srv"].to_numpy(float), gl["pnl"].to_numpy(float))
        tau2 = {t: LB.between_league_variance(curves, t) for t in map(float, LB.GRID)}
        for m, c in curves.items():
            se2 = c["var"] / c["n"]
            t2 = c["t"].map(tau2)
            w = np.where((c["n"] >= 2) & np.isfinite(se2) & (se2 > 0), t2 / (t2 + se2), 0.0)
            shr = w * c["roi"].fillna(pooled["roi"]) + (1 - w) * pooled["roi"]
            el = (c["n"] >= R.MIN_OWN_BETS) & np.isfinite(shr)
            if not el.any():
                out[(tr, lg, m)] = None
                continue
            score = shr[el] if objective == "ROI" else shr[el] * c.loc[el, "n"]
            t = float(c.loc[score.idxmax() if hasattr(score, "idxmax") else el.idxmax(), "t"])
            emax = float(gl.loc[gl["mside"] == m, "e_srv"].max())
            out[(tr, lg, m)] = None if t >= emax else t
    return out


def apply_cap(sel: pd.DataFrame, config: str) -> pd.DataFrame:
    """Weekly expanding: week W's tips flagged with caps learned from bets dated before W."""
    wk = sel["date"].dt.to_period("W-SUN").dt.start_time
    flag = pd.Series(False, index=sel.index)
    capv = pd.Series(np.nan, index=sel.index)
    for w in sorted(wk.unique()):
        idx = sel.index[wk == w]
        if config == "DIAGNOSTIC_GLOBAL_10PP":
            flag.loc[idx] = sel.loc[idx, "e_srv"] > .10
            continue
        cp = caps(sel[sel["date"] < w], config.split("_")[-1])
        for i in idx:
            c = cp.get((sel.at[i, "track"], sel.at[i, "league"], sel.at[i, "mside"]))
            if c is not None:
                capv.at[i] = c
                flag.at[i] = sel.at[i, "e_srv"] > c
    return sel.assign(capped=flag, cap=capv)


def cap_eval(s: pd.DataFrame, rng) -> dict:
    keep, drop = s[~s["capped"]], s[s["capped"]]
    g = s.groupby("lw").agg(sa=("pnl", "sum"), na=("pnl", "size"))
    gk = keep.groupby("lw")["pnl"].agg(["sum", "size"]).reindex(g.index).fillna(0)
    Sa, Na, Sk, Nk = g["sa"].to_numpy(), g["na"].to_numpy(), gk["sum"].to_numpy(), gk["size"].to_numpy()
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, len(Sa), len(Sa))
        d[i] = Sk[j].sum() / max(Nk[j].sum(), 1) - Sa[j].sum() / Na[j].sum()
    return {"tips_all": int(len(s)), "tips_removed": int(len(drop)), "share_removed": round(len(drop) / len(s), 3),
            "roi_all": round(float(s["pnl"].mean()), 4), "roi_kept": round(float(keep["pnl"].mean()), 4),
            "roi_removed": round(float(drop["pnl"].mean()), 4) if len(drop) else None,
            "kept_minus_all": round(float(keep["pnl"].mean() - s["pnl"].mean()), 4),
            "kept_minus_all_ci95_lw": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)],
            "cells_with_a_cap_last_week": int(s.loc[s["date"] == s["date"].max(), "cap"].notna().sum()),
            "removed_by_market": drop["mside"].value_counts().to_dict(),
            "mean_cap_pp": round(float(s["cap"].mean()), 4) if s["cap"].notna().any() else None}


SRC = cell_sources()
CONFIGS = ("CELL_CAP_ROI", "CELL_CAP_PROFIT", "DIAGNOSTIC_GLOBAL_10PP")


def main() -> None:
    rng = np.random.default_rng(SEED)
    sel = pd.read_parquet(OUT / "sel_bets.parquet")
    rows = pd.read_parquet(OUT / "sel_rows.parquet")
    fin = pd.read_parquet(ROOT / "output/bars_opt/study_58f10476/final_bets.parquet")
    S_ = add_edges(sel, lam_weekly(rows))
    F_ = add_edges(fin, lam_fixed(R.cell_stats(None, last_rows(rows))))
    res = {"definition": __doc__.split("Data:")[0].strip(), "frame": "58f10476f4c82d7b (clean champion_recipe, own prices)",
           "cell_sources": {f"{k[0]}|{k[1]}": v for k, v in SRC.items()}, "league_wide_cells": 0}
    for name, b in (("SELECTION", S_), ("FINAL_SPENT_DESCRIPTIVE_ONLY", F_)):
        res[name] = {"distribution": dist(b), "bands_served_edge": bands(b, "e_srv", rng),
                     "bands_calibrated_edge": bands(b.dropna(subset=["e_cal"]), "e_cal", rng),
                     "calibration": cal_shift(b)}
    res["cap_rule_development_selection_only"] = {
        "configurations_tried": list(CONFIGS), "n_configurations": len(CONFIGS),
        "note": "every week capped with caps learned only from bets dated before it; the global 10pp row is a "
                "DIAGNOSTIC for the owner's intuition, never a candidate (owner: no global vars)",
        "results": {c: cap_eval(apply_cap(S_, c), rng) for c in CONFIGS}}
    (OUT / "edge_realism.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
