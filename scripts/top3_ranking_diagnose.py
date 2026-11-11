"""RESEARCH (no serving change): why TOP-3 per league (TOP_ONLY_V, ranked by ev_est) replays below all bets.

Inputs: output/research/top3_ranking/sel_bets.parquet (scripts/top3_ranking_build.py; SELECTION window,
walk-forward) and output/bars_opt/study_58f10476/final_bets.parquet (FINAL 2026-09-01..11-08, SPENT:
descriptive only, never evidence). Writes aggregates only to output/research/top3_ranking/diagnosis.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bars_opt import study as S  # noqa: E402

OUT = ROOT / "output" / "research" / "top3_ranking"
SEED = 20261111
N_BOOT = 5000
N_NULL = 2000


def load() -> dict:
    sel = pd.read_parquet(OUT / "sel_bets.parquet")
    fin = pd.read_parquet(ROOT / "output" / "bars_opt" / "study_58f10476" / "final_bets.parquet")
    return {"SELECTION": sel, "FINAL_SPENT_DESCRIPTIVE": fin}


def mark_top3(b: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    b["gw"] = b["date"].dt.strftime("%G-W%V")
    key = ["track", "mside", "league", "row_id"]
    pk = picks[key].copy()
    pk["picked"] = True
    pk["rank"] = picks.groupby(["track", "league", "gw"]).cumcount().to_numpy() + 1
    b = b.merge(pk, on=key, how="left")
    b["picked"] = b["picked"].eq(True)
    b["lw_n"] = b.groupby(["track", "league", "gw"])["row_id"].transform("size")
    return b


def roi(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {"n": 0}
    return {"n": int(len(g)), "roi": round(float(g["pnl"].mean()), 4),
            "mean_odds": round(float(g["odds"].mean()), 3),
            "mean_p_used": round(float(g["p_used"].mean()), 4), "win_rate": round(float(g["won"].mean()), 4),
            "mean_fair": round(float(g["fair"].mean()), 4), "mean_ev_est": round(float(g["ev_est"].mean()), 4),
            "realised_minus_p_used": round(float(g["won"].mean() - g["p_used"].mean()), 4)}


def deciles(b: pd.DataFrame, col: str, q: int = 10) -> list[dict]:
    x = b.copy()
    x["bin"] = pd.qcut(x[col].rank(method="first"), q, labels=False)
    return [{"bin": int(k) + 1, f"{col}_lo": round(float(g[col].min()), 4), f"{col}_hi": round(float(g[col].max()), 4)}
            | roi(g) for k, g in x.groupby("bin")]


def within_lw_spearman(b: pd.DataFrame) -> dict:
    rs = []
    for _, g in b.groupby(["track", "league", "gw"]):
        if len(g) >= 3 and g["ev_est"].nunique() > 1 and g["pnl"].nunique() > 1:
            rs.append(g["ev_est"].rank().corr(g["pnl"].rank()))
    rs = np.array([r for r in rs if np.isfinite(r)])
    return {"league_weeks": int(len(rs)), "mean_spearman_ev_vs_pnl": round(float(rs.mean()), 4) if len(rs) else None}


def mix_counterfactual(b: pd.DataFrame, by: str) -> dict:
    """ROI of ALL bets re-weighted to the TOP-3 share of `by` (what the mix alone would give)."""
    t = b[b["picked"]]
    w = t[by].value_counts(normalize=True)
    r_all = b.groupby(by)["pnl"].mean()
    common = w.index.intersection(r_all.index)
    return {"all_roi_reweighted_to_top3_mix": round(float((w[common] * r_all[common]).sum() / w[common].sum()), 4),
            "top3_roi": round(float(t["pnl"].mean()), 4), "all_roi": round(float(b["pnl"].mean()), 4)}


def by_group(b: pd.DataFrame, by: str) -> list[dict]:
    rows = []
    for k, g in b.groupby(by):
        rows.append({by: k, "all": roi(g), "top3": roi(g[g["picked"]]), "not_picked": roi(g[~g["picked"]]),
                     "share_all": round(len(g) / len(b), 3),
                     "share_top3": round(int(g["picked"].sum()) / max(int(b["picked"].sum()), 1), 3)})
    return rows


def odds_buckets(b: pd.DataFrame) -> list[dict]:
    x = b.copy()
    x["ob"] = pd.qcut(x["odds"].rank(method="first"), 5, labels=False)
    return [{"odds_quintile": int(k) + 1, "odds_lo": float(g["odds"].min()), "odds_hi": float(g["odds"].max()),
             "all": roi(g), "top3": roi(g[g["picked"]]), "pick_rate": round(float(g["picked"].mean()), 3)}
            for k, g in x.groupby("ob")]


def top3_no_direction(b: pd.DataFrame) -> pd.DataFrame:
    o = b.sort_values(["ev_est", "date"], ascending=[False, True])
    return o.groupby(["track", "league", "gw"], sort=False).head(3)


def boot_diff(b: pd.DataFrame, rng) -> dict:
    """Paired cluster bootstrap over league-weeks of ROI(TOP-3) - ROI(all bets) and - ROI(not picked)."""
    b = b.assign(lw=b["track"] + "|" + b["league"] + "|" + b["gw"])
    g = b.groupby("lw").agg(s_all=("pnl", "sum"), n_all=("pnl", "size"))
    gt = b[b["picked"]].groupby("lw")["pnl"].agg(["sum", "size"])
    g["s_t"], g["n_t"] = gt["sum"].reindex(g.index).fillna(0), gt["size"].reindex(g.index).fillna(0)
    S_a, N_a, S_t, N_t = (g[c].to_numpy(float) for c in ("s_all", "n_all", "s_t", "n_t"))
    obs = S_t.sum() / N_t.sum() - S_a.sum() / N_a.sum()
    obs_np = S_t.sum() / N_t.sum() - (S_a.sum() - S_t.sum()) / (N_a.sum() - N_t.sum())
    k = len(g)
    d, dn = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, k, k)
        st, nt, sa, na = S_t[idx].sum(), N_t[idx].sum(), S_a[idx].sum(), N_a[idx].sum()
        d[i] = st / nt - sa / na
        dn[i] = st / nt - (sa - st) / (na - nt) if na > nt else np.nan
    dn = dn[np.isfinite(dn)]
    return {"league_weeks": k, "diff_top3_minus_all": round(float(obs), 4),
            "diff_ci95": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)],
            "share_boot_diff_ge_0": round(float((d >= 0).mean()), 4),
            "diff_top3_minus_not_picked": round(float(obs_np), 4),
            "diff_np_ci95": [round(float(np.quantile(dn, .025)), 4), round(float(np.quantile(dn, .975)), 4)]}


def random3_null(b: pd.DataFrame, rng) -> dict:
    """Random TOP-k: in each league-week draw the SAME number of picks the EV rule took, uniformly at random
    among the eligible bets, honouring the direction rule. Where does the EV rule's ROI fall?"""
    groups = []
    for _, g in b.groupby(["track", "league", "gw"]):
        k = int(g["picked"].sum())
        if k:
            groups.append((g["row_id"].to_numpy(), np.array([m in S.OVER_DIRECTION for m in g["mside"]]),
                           g["pnl"].to_numpy(float), k))
    obs = float(b.loc[b["picked"], "pnl"].mean())
    sims = np.empty(N_NULL)
    for i in range(N_NULL):
        tot, n = 0.0, 0
        for rid, over, pnl, k in groups:
            taken, dirs = 0, {}
            for j in rng.permutation(len(pnl)):
                d = over[j]
                if dirs.get(rid[j], d) != d:
                    continue
                dirs[rid[j]] = d
                tot += pnl[j]
                n += 1
                taken += 1
                if taken == k:
                    break
        sims[i] = tot / n
    return {"ev_rule_roi": round(obs, 4), "random_mean_roi": round(float(sims.mean()), 4),
            "random_ci95": [round(float(np.quantile(sims, .025)), 4), round(float(np.quantile(sims, .975)), 4)],
            "share_random_at_or_below_ev_rule": round(float((sims <= obs).mean()), 4)}


def diagnose(name: str, bets: pd.DataFrame, rng) -> dict:
    picks = S.top3(bets, "TOP_ONLY_V")
    b = mark_top3(bets, picks)
    b["edge_pp"] = b["p_used"] - b["fair"]
    nd = top3_no_direction(b)
    picked_keys = set(b.loc[b["picked"], "row_id"] + "|" + b.loc[b["picked"], "mside"])
    return {"window": name, "all": roi(b), "top3": roi(b[b["picked"]]), "not_picked": roi(b[~b["picked"]]),
            "league_weeks": int(b.groupby(["track", "league", "gw"]).ngroups),
            "by_rank": [{"rank": int(r)} | roi(g) for r, g in b[b["picked"]].groupby("rank")],
            "by_league_week_size": [{"lw_bets": k} | roi(g[g["picked"]]) | {"all_roi": round(float(g["pnl"].mean()), 4)}
                                    for k, g in b.assign(k=np.minimum(b["lw_n"], 7)).groupby("k")],
            "ev_est_deciles": deciles(b, "ev_est"), "edge_pp_deciles": deciles(b, "edge_pp"),
            "within_league_week": within_lw_spearman(b),
            "market": by_group(b, "mside"), "market_mix": mix_counterfactual(b, "mside"),
            "league": by_group(b.assign(lg=b["track"] + "|" + b["league"]), "lg"),
            "league_mix": mix_counterfactual(b.assign(lg=b["track"] + "|" + b["league"]), "lg"),
            "odds_quintiles": odds_buckets(b),
            "direction_rule": {"top3_without_direction_rule": roi(nd),
                               "picks_changed": int(len(set(nd["row_id"] + "|" + nd["mside"]) ^ picked_keys))},
            "bootstrap": boot_diff(b, rng), "random_top_k_null": random3_null(b, rng)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {k: diagnose(k, v, rng) for k, v in load().items()}
    (OUT / "diagnosis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
