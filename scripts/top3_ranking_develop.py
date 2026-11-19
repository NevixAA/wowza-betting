"""RESEARCH (no serving change): develop the pre-registered TOP-3 ranking rules on the SELECTION window ONLY.

Nested walk-forward inside SELECT: the bets of inner fold k (scripts/top3_ranking_build.py) are ranked with
cell statistics learned from folds < k only (src.research.top3_ranking.cell_stats). Folds 1..3 are scored;
fold 0 has no prior bets. Writes aggregates only to output/research/top3_ranking/develop_selection.json.

    python scripts/top3_ranking_develop.py              # SELECTION window (development)
    python scripts/top3_ranking_develop.py --final      # FINAL 2026-09-01..11-08: SPENT, descriptive only,
                                                        # statistics learned on every SELECT fold; run only
                                                        # AFTER the protocol is committed
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
from src.research import top3_ranking as R  # noqa: E402

OUT = ROOT / "output" / "research" / "top3_ranking"
ALL_RULES = R.DIAGNOSTIC_RULES[:1] + R.RULES + R.DIAGNOSTIC_RULES[1:]
SEED = 20261111
N_BOOT = 5000


def summ(t: pd.DataFrame, all_bets: pd.DataFrame) -> dict:
    s = S.score(t)
    return s | {"mean_odds": round(float(t["odds"].mean()), 3),
                "realised_minus_p_used": round(float(t["won"].mean() - t["p_used"].mean()), 4),
                "no_cell_history_picks": int((~t["has_stat"]).sum()) if "has_stat" in t else 0,
                "market_share": {m: round(len(g) / len(t), 3) for m, g in t.groupby("mside")},
                "all_bets_roi_same_rows": round(float(all_bets["pnl"].mean()), 4)}


def paired(t: pd.DataFrame, ref: pd.DataFrame, rng) -> dict:
    """Cluster bootstrap over league-weeks of ROI(rule) - ROI(ref) (ref: all bets or the EV rule)."""
    def agg(x):
        x = x.assign(lw=x["track"] + "|" + x["league"] + "|" + x["date"].dt.strftime("%G-W%V"))
        return x.groupby("lw")["pnl"].agg(["sum", "size"])
    a, r = agg(t), agg(ref)
    idx = a.index.union(r.index)
    a, r = a.reindex(idx).fillna(0), r.reindex(idx).fillna(0)
    Sa, Na, Sr, Nr = a["sum"].to_numpy(), a["size"].to_numpy(), r["sum"].to_numpy(), r["size"].to_numpy()
    obs = Sa.sum() / Na.sum() - Sr.sum() / Nr.sum()
    k, d = len(idx), np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, k, k)
        d[i] = Sa[j].sum() / Na[j].sum() - Sr[j].sum() / Nr[j].sum()
    return {"diff": round(float(obs), 4), "ci95": [round(float(np.quantile(d, .025)), 4),
                                                   round(float(np.quantile(d, .975)), 4)]}


def run(folds: list[tuple], rng) -> dict:
    """folds: [(prior bets, prior rows, bets to rank)]. Returns per rule the pooled picks' summary + diffs."""
    picks = {r: [] for r in ALL_RULES}
    evald = []
    for prior, rows, e in folds:
        st = R.cell_stats(prior, rows)
        evald.append(e)
        for r in ALL_RULES:
            picks[r].append(R.select(e, st, r))
    allb = pd.concat(evald, ignore_index=True)
    out = {"all_bets": S.score(allb)}
    ev = pd.concat(picks["EV_PRODUCTION"], ignore_index=True)
    for r in ALL_RULES:
        t = pd.concat(picks[r], ignore_index=True)
        out[r] = summ(t, allb) | {"minus_all_bets": paired(t, allb, rng)}
        if r != "EV_PRODUCTION":
            out[r]["minus_ev_rule"] = paired(t, ev, rng)
    return out


def rows_before(rows: pd.DataFrame, w: pd.Timestamp) -> pd.DataFrame:
    """R2 calibration rows for week w: per family the fold whose evaluation block contains w, rows < w."""
    m = (rows["fold_start"] <= w) & (rows["fold_end"] > w) & (rows["date"] < w)
    return rows[m]


def last_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Every family's last SELECT fold (its source fitted on the most data): the rows for a later window."""
    last = rows.groupby(["track", "mside"])["fold"].transform("max")
    return rows[rows["fold"] == last]


def main() -> None:
    rng = np.random.default_rng(SEED)
    sel = pd.read_parquet(OUT / "sel_bets.parquet")
    rows = pd.read_parquet(OUT / "sel_rows.parquet")
    if "--final" in sys.argv:
        fin = pd.read_parquet(ROOT / "output" / "bars_opt" / "study_58f10476" / "final_bets.parquet")
        res = {"window": "FINAL 2026-09-01..2026-11-08 - SPENT: DESCRIPTIVE ONLY, NOT EVIDENCE",
               "statistics_learned_on": "every SELECTION fold (walk-forward bets of the frozen choices)",
               "cell_stats": R.cell_stats(sel, last_rows(rows)).round(4).to_dict("records")} \
            | run([(sel, last_rows(rows), fin)], rng)
        name = "descriptive_final_spent.json"
    else:
        # expanding by ISO week: week W is ranked with statistics of every walk-forward bet dated before W's Monday
        wk = sel["date"].dt.to_period("W-SUN").dt.start_time
        folds = [(sel[sel["date"] < w], rows_before(rows, w), sel[wk == w]) for w in sorted(wk.unique())]
        n_stat = [int(R.cell_stats(p)["se"].notna().sum()) for p, _, _ in folds]
        n_lam = [int(R.cell_stats(p, r)["lam"].notna().sum()) for p, r, _ in folds]
        res = {"window": "SELECTION (rows < 2026-09-01): every week ranked with statistics learned from the "
                         "walk-forward bets dated before that week (expanding)",
               "weeks": len(folds), "weeks_with_any_cell_statistic": int(sum(x > 0 for x in n_stat)),
               "max_cells_with_bet_statistic": int(max(n_stat)),
               "weeks_with_any_lam": int(sum(x > 0 for x in n_lam))} | run(folds, rng)
        name = "develop_selection.json"
    (OUT / name).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "cell_stats"}, indent=1, default=str))


if __name__ == "__main__":
    main()
