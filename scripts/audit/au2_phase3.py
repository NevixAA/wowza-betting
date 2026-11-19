"""AU2 Phase 3 - Standard OU2.5 legacy vs repaired vs shadow populations. READ-ONLY. Never pooled.

Populations (each evaluated on its own; nothing is averaged across them):
  L1  ledger, live Standard OU25 tips, generated 2026-09-01 .. 2026-10-18 08:04Z  (legacy, unversioned rows)
  L2  ledger, live Standard OU25 tips, standard_20261018T080423Z_64313878          (legacy champion)
  R   ledger, live Standard OU25 tips, standard_repaired_20261028T123040Z_1f5805e6  (repaired, serving)
  B_leg / B_rep   full-board FIRST-SIGHT probabilities (output/model_lineage/shadow_h2h.csv), every
                  Standard fixture, model_type standard (legacy) / standard_repaired
  P   Pro forward shadow (output/standard_repair/forward_shadow.csv, Pro origin/main), OU25 first sight
      per fixture: p_CURRENT_V9_SERVED, p_V9_REPAIRED, p_MARKET on the SAME fixtures (paired).

Outcomes come only from committed sources, each recorded per fixture with its source:
  ledger result (tipped side W/L -> over25 truth), v9 lineage settlement (shadow_experiments y),
  Pro settled file, v9's offline history (committed fd cache). Conflicting sources -> fixture
  excluded and counted. Missing outcome -> UNSETTLED (never a loss).

    python scripts/audit/au2_phase3.py --pro-dir <dir with Pro forward_shadow*.csv>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au2_common import DATA, ROOT, AUDITED_COMMIT, load_history_offline  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.team_names import resolve, norm  # noqa: E402

SWITCH = pd.Timestamp("2026-10-28 12:48:14", tz="UTC")     # first v9_repaired board (switch_boards.csv)
LIVE_FROM = pd.Timestamp("2026-09-01", tz="UTC")
V_L2 = "standard_20261018T080423Z_64313878"
V_R = "standard_repaired_20261028T123040Z_1f5805e6"
INSUFFICIENT = "INSUFFICIENT FORWARD EVIDENCE — CANNOT DECLARE REPAIRED MODEL UNPROFITABLE."
MIN_SETTLED = 150        # same bar the estate uses for a CLV/ROI decision (MIN_CLV_N); stated, not tuned
RNG = np.random.default_rng(20261101)


def fkey(date, home, away) -> str:
    return f"{str(date)[:10]}|{norm(str(home))}|{norm(str(away))}"


# ── outcomes ───────────────────────────────────────────────────────────────────────────
def outcomes(ledger: pd.DataFrame, pro_settled: pd.DataFrame | None, pro_shadow: pd.DataFrame | None) -> pd.DataFrame:
    rec = []
    # 1) ledger OU2.5 results (both Standard and NF, live + backtest excluded)
    lg = ledger[(ledger["source"] == "live") & ledger["result"].isin(["WIN", "LOSS"])
                & ledger["side"].isin(["OVER", "UNDER"])]
    for r in lg.itertuples():
        y = 1.0 if (r.side == "OVER") == (r.result == "WIN") else 0.0
        rec.append((fkey(r.match_date, r.home_team, r.away_team), y, "v9_ledger"))
    # 2) v9 lineage settlement
    se = pd.read_csv(ROOT / "output/model_lineage/shadow_experiments.csv")
    se = se[se["y"].notna() & (se["market"] == "over25")]
    for r in se.itertuples():
        rec.append((fkey(r.match_date, r.home_team, r.away_team), float(r.y), "v9_shadow_experiments"))
    # 3) Pro settled
    if pro_settled is not None and pro_shadow is not None:
        ps = pro_settled[(pro_settled["market"] == "OU25") & (pro_settled["settle_status"] == "SETTLED")]
        meta = pro_shadow.drop_duplicates("fixture_key").set_index("fixture_key")
        for r in ps.itertuples():
            if r.fixture_key in meta.index:
                m = meta.loc[r.fixture_key]
                rec.append((fkey(m["match_date"], m["home_team"], m["away_team"]), float(r.y), "pro_forward_settled"))
    # 4) offline history (football-data names, resolved from OddsAPI names on demand by the caller)
    out = pd.DataFrame(rec, columns=["fkey", "y", "source"])
    g = out.groupby("fkey")["y"].agg(["nunique", "first", "size"])
    conflicts = g[g["nunique"] > 1].index
    res = g[g["nunique"] == 1]["first"].rename("y").to_frame()
    res["sources"] = out.groupby("fkey")["source"].agg(lambda s: ";".join(sorted(set(s))))
    res.attrs["conflicts"] = list(conflicts)
    return res


def add_history_outcomes(res: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining fixtures from v9's committed history (resolved league-scoped names)."""
    h = load_history_offline()
    h = h[h["league"].isin(config.STANDARD_FORMAT_LEAGUES) & h["home_goals"].notna()].copy()
    h["d"] = pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d")
    pools = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in h.groupby("league")}
    idx = {(r.d, r.league, r.home_team, r.away_team): (r.home_goals + r.away_goals) for r in h.itertuples()}
    add, conflicts = [], 0
    for r in fixtures.drop_duplicates("fkey").itertuples():
        pool = pools.get(r.league, [])
        hh = r.home_team if r.home_team in pool else resolve(r.home_team, pool)
        aa = r.away_team if r.away_team in pool else resolve(r.away_team, pool)
        tg = idx.get((str(r.match_date)[:10], r.league, hh, aa))
        if tg is None:
            continue
        y = float(tg > 2.5)
        if r.fkey in res.index:
            if res.at[r.fkey, "y"] != y:
                conflicts += 1
            else:
                res.at[r.fkey, "sources"] += ";v9_history"
        else:
            add.append({"fkey": r.fkey, "y": y, "sources": "v9_history"})
    if add:
        res = pd.concat([res, pd.DataFrame(add).set_index("fkey")])
    res.attrs["history_conflicts"] = conflicts
    return res


# ── metrics ────────────────────────────────────────────────────────────────────────────
def prob_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    if len(y) == 0:
        return {"n": 0}
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    out = {"n": int(len(y)), "logloss": round(ll, 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
           "mean_p": round(float(p.mean()), 4), "base_rate": round(float(y.mean()), 4),
           "calib_gap_pp": round(float(p.mean() - y.mean()) * 100, 2)}
    out["auc"] = round(float(roc_auc_score(y, p)), 4) if len(set(y)) == 2 else None
    return out


def boot_ci(values: np.ndarray, groups: np.ndarray, stat=np.mean, b: int = 4000):
    """Cluster bootstrap by group (match date) - a matchday is the correlated unit."""
    if len(values) < 2:
        return [None, None]
    ug = np.unique(groups)
    by = {g: values[groups == g] for g in ug}
    s = []
    for _ in range(b):
        pick = RNG.choice(ug, size=len(ug), replace=True)
        s.append(stat(np.concatenate([by[g] for g in pick])))
    return [round(float(np.quantile(s, 0.025)), 4), round(float(np.quantile(s, 0.975)), 4)]


def ledger_metrics(d: pd.DataFrame) -> dict:
    s = d[d["result"].isin(["WIN", "LOSS"])].copy()
    voids = int((d["result"] == "VOID").sum())
    out = {"tips": int(len(d)), "settled": int(len(s)), "void": voids,
           "unsettled": int(d["result"].isna().sum()),
           "first_generated": str(d["generated_at"].min()), "last_generated": str(d["generated_at"].max())}
    if len(s) == 0:
        return out
    s["win"] = (s["result"] == "WIN").astype(float)
    s["pnl_recomputed"] = np.where(s["win"] == 1, s["odds"] - 1.0, -1.0)
    s["p_side"] = s["edge_pct"] / 100.0 + 1.0 / s["odds"]
    out.update({
        "wins": int(s["win"].sum()), "hit_rate": round(float(s["win"].mean()), 4),
        "flat_profit_u": round(float(s["pnl_recomputed"].sum()), 2),
        "flat_roi": round(float(s["pnl_recomputed"].mean()), 4),
        "flat_roi_ci95_matchday_boot": boot_ci(s["pnl_recomputed"].to_numpy(), s["match_date"].astype(str).to_numpy()),
        "ledger_pnl_matches_recomputed": bool(np.allclose(s["pnl"].astype(float), s["pnl_recomputed"], atol=0.011)),
        "mean_odds": round(float(s["odds"].mean()), 3),
        "side_prob_metrics": prob_metrics(s["p_side"].to_numpy(), s["win"].to_numpy()),
    })
    c = s[s["clv_pct"].notna()]
    if len(c):
        out["clv"] = {"n": int(len(c)), "mean_pct": round(float(c["clv_pct"].mean()), 3),
                      "median_pct": round(float(c["clv_pct"].median()), 3),
                      "pct_positive": round(float((c["clv_pct"] > 0).mean()), 3),
                      "ci95_matchday_boot": boot_ci(c["clv_pct"].to_numpy(), c["match_date"].astype(str).to_numpy())}
    out["by_direction"] = {k: {"n": int(len(g)), "hit": round(float(g["win"].mean()), 3),
                               "roi": round(float(g["pnl_recomputed"].mean()), 4)}
                           for k, g in s.groupby("side")}
    bands = pd.cut(s["odds"], [0, 2.0, 2.5, 3.0, 99], right=False, labels=["<2", "2-2.5", "2.5-3", ">=3"])
    out["by_odds_band"] = {str(k): {"n": int(len(g)), "hit": round(float(g["win"].mean()), 3) if len(g) else None,
                                    "roi": round(float(g["pnl_recomputed"].mean()), 4) if len(g) else None}
                           for k, g in s.groupby(bands, observed=False)}
    out["by_tier"] = {k: {"n": int(len(g)), "roi": round(float(g["pnl_recomputed"].mean()), 4)}
                      for k, g in s.groupby("signal_tier")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro-dir", required=True)
    a = ap.parse_args()
    pro = Path(a.pro_dir)
    ledger = pd.read_csv(ROOT / "output/bets_ledger.csv", low_memory=False)
    ledger["gen"] = pd.to_datetime(ledger["generated_at"], format="mixed", utc=True)
    pro_shadow = pd.read_csv(pro / "forward_shadow.csv")
    pro_settled = pd.read_csv(pro / "forward_shadow_settled.csv")

    res = outcomes(ledger, pro_settled, pro_shadow)
    out = {"audited_v9_commit": AUDITED_COMMIT, "switch_first_repaired_board_utc": str(SWITCH),
           "min_settled_for_decision": MIN_SETTLED, "outcome_conflicts_excluded": len(res.attrs["conflicts"])}

    # ── ledger populations ──
    st = ledger[(ledger["source"] == "live") & (ledger["model_type"] == "standard")
                & ledger["side"].isin(["OVER", "UNDER"]) & (ledger["gen"] >= LIVE_FROM)].copy()
    dup = st.duplicated(["match_date", "home_team", "away_team"], keep=False)
    out["ledger_duplicate_fixture_rows"] = int(dup.sum())
    pops = {
        "L1_legacy_unversioned_2026-09-01_to_2026-10-18": st[st["model_version"].isna() & (st["gen"] < SWITCH)],
        "L2_legacy_standard_20261018": st[st["model_version"] == V_L2],
        "R_repaired_standard_repaired_20261028": st[st["model_version"] == V_R],
    }
    # sanity: no repaired-version row before the switch, no legacy-version row after it
    out["attribution_checks"] = {
        # ledger generated_at has MINUTE precision ("2026-10-28 12:48"), so compare at the minute
        "repaired_rows_before_switch": int(((st["model_version"] == V_R) & (st["gen"] < SWITCH.floor("min"))).sum()),
        "legacy_rows_after_switch": int(((st["model_version"] != V_R) & (st["gen"] >= SWITCH.floor("min"))).sum()),
        "unattributed_rows_after_2026-10-18": int((st["model_version"].isna() & (st["gen"] >= pd.Timestamp("2026-10-18 08:05", tz="UTC"))).sum()),
    }
    out["ledger"] = {k: ledger_metrics(v) for k, v in pops.items()}
    rset = out["ledger"]["R_repaired_standard_repaired_20261028"].get("settled", 0)
    out["repaired_verdict_sentence"] = INSUFFICIENT if rset < MIN_SETTLED else "SUFFICIENT - see metrics"

    # ── full-board first sight (v9 lineage) ──
    h2h = pd.read_csv(ROOT / "output/model_lineage/shadow_h2h.csv")
    h2h = h2h[h2h["model_type"].isin(["standard", "standard_repaired"])].copy()
    h2h["ts"] = pd.to_datetime(h2h["logged_at"], utc=True, format="mixed")
    # LINEAGE ATTRIBUTION DEFECT (AU2 finding): after the switch, rows logged as model_type
    # "standard" with champion_version = the LEGACY version carry p_served = the REPAIRED served
    # probability. They are excluded here; the legacy board population is pre-switch rows only.
    post_leg = (h2h["model_type"] == "standard") & (h2h["ts"] >= SWITCH.floor("min"))
    out_mis = int(post_leg.sum())
    h2h = h2h[~post_leg]
    h2h["fkey"] = [fkey(d, a, b) for d, a, b in zip(h2h["match_date"], h2h["home_team"], h2h["away_team"])]
    ee = pd.read_csv(ROOT / "output/edge_engine_shadow.csv", low_memory=False)
    ee = ee[(ee["market"] == "OU25") & (ee["model_type"] == "standard")]
    ee["fkey"] = [fkey(d, a, b) for d, a, b in zip(ee["match_date"], ee["home_team"], ee["away_team"])]
    # market fair P(over) at the edge engine's first sight; p_market_fair is for the row's `side`
    ee["p_mkt_over"] = np.where(ee["side"] == "OVER", ee["p_market_fair"], 1 - ee["p_market_fair"])
    mkt = ee.drop_duplicates("fkey").set_index("fkey")["p_mkt_over"]

    allfx = pd.concat([
        h2h[["fkey", "match_date", "league", "home_team", "away_team"]],
        pro_shadow.rename(columns={})[["match_date", "league", "home_team", "away_team"]].assign(
            fkey=[fkey(d, a, b) for d, a, b in zip(pro_shadow["match_date"], pro_shadow["home_team"], pro_shadow["away_team"])]),
    ]).drop_duplicates("fkey")
    res = add_history_outcomes(res, allfx)
    out["outcome_history_conflicts"] = res.attrs.get("history_conflicts", 0)
    out["outcome_sources"] = res["sources"].value_counts().to_dict()

    board = {}
    for mt, name in (("standard", "B_leg_first_sight_standard_legacy"), ("standard_repaired", "B_rep_first_sight_standard_repaired")):
        g = h2h[h2h["model_type"] == mt].sort_values("logged_at").drop_duplicates("fkey")
        g = g.join(res["y"], on="fkey")
        s = g[g["y"].notna()]
        rec = {"fixtures_first_seen": int(len(g)), "settled": int(len(s)),
               "versions": g["champion_version"].value_counts().to_dict(),
               "logged_from": str(g["logged_at"].min()), "logged_to": str(g["logged_at"].max()),
               "model": prob_metrics(s["p_served"].to_numpy(), s["y"].to_numpy())}
        m = s.join(mkt.rename("p_mkt"), on="fkey")
        m = m[m["p_mkt"].notna()]
        if len(m):
            rec["same_rows_with_market"] = {"model": prob_metrics(m["p_served"].to_numpy(), m["y"].to_numpy()),
                                            "market_devig": prob_metrics(m["p_mkt"].to_numpy(), m["y"].to_numpy())}
        board[name] = rec
    # paired: fixtures first seen by BOTH legacy and repaired in v9's own lineage log
    leg = h2h[h2h["model_type"] == "standard"].sort_values("logged_at").drop_duplicates("fkey").set_index("fkey")
    rep = h2h[h2h["model_type"] == "standard_repaired"].sort_values("logged_at").drop_duplicates("fkey").set_index("fkey")
    both = leg[["p_served"]].join(rep[["p_served"]], lsuffix="_leg", rsuffix="_rep", how="inner").join(res["y"])
    bs = both[both["y"].notna()]
    board["paired_v9_lineage_same_fixture"] = {
        "fixtures": int(len(both)), "settled": int(len(bs)),
        "legacy": prob_metrics(bs["p_served_leg"].to_numpy(), bs["y"].to_numpy()) if len(bs) else {"n": 0},
        "repaired": prob_metrics(bs["p_served_rep"].to_numpy(), bs["y"].to_numpy()) if len(bs) else {"n": 0},
        "mean_abs_prob_diff_pp_all": round(float((both["p_served_leg"] - both["p_served_rep"]).abs().mean()) * 100, 2) if len(both) else None,
        "note": "legacy p is from an EARLIER board than repaired p (legacy first sight precedes the switch); not the same scoring time"}
    board["excluded_post_switch_legacy_labelled_rows"] = out_mis
    out["board_first_sight"] = board

    # ── Pro forward shadow, paired at the same scoring instant ──
    ps = pro_shadow[pro_shadow["market"] == "OU25"].sort_values("scored_at").drop_duplicates("fixture_key").copy()
    ps["fkey"] = [fkey(d, a, b) for d, a, b in zip(ps["match_date"], ps["home_team"], ps["away_team"])]
    ps = ps.join(res["y"], on="fkey")
    s = ps[ps["y"].notna()]
    pro_out = {"fixtures_first_sight": int(len(ps)), "settled": int(len(s)),
               "served_versions_at_first_sight": ps["v9_served_model_version"].value_counts().to_dict(),
               "scored_from": str(ps["scored_at"].min()), "scored_to": str(ps["scored_at"].max())}
    for col in ("p_CURRENT_V9_SERVED", "p_V9_REPAIRED", "p_MARKET", "p_MKT_RESID_LR", "p_MKT_RESID_HGB"):
        ok = s[col].notna()
        pro_out[col] = prob_metrics(s.loc[ok, col].to_numpy(), s.loc[ok, "y"].to_numpy())
    post = ps[ps["v9_served_model_version"] == V_R]
    pro_out["post_switch_rows_served_equals_repaired_maxdiff"] = (
        round(float((post["p_CURRENT_V9_SERVED"] - post["p_V9_REPAIRED"]).abs().max()), 6) if len(post) else None)
    pre = ps[ps["v9_served_model_version"] == V_L2]
    pro_out["pre_switch_rows_mean_abs_diff_served_vs_repaired_pp"] = (
        round(float((pre["p_CURRENT_V9_SERVED"] - pre["p_V9_REPAIRED"]).abs().mean()) * 100, 2) if len(pre) else None)
    out["pro_forward_shadow"] = pro_out

    (DATA / "phase3_populations.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "--paired"):
    raise SystemExit(main())


def paired_dll(pro_dir: str) -> dict:
    """Paired log-loss differences on the Pro shadow first-sight OU25 rows (same fixture, same
    scoring instant), matchday-block bootstrap. Negative = first arm better. Descriptive only."""
    pro = Path(pro_dir)
    ledger = pd.read_csv(ROOT / "output/bets_ledger.csv", low_memory=False)
    ps = pd.read_csv(pro / "forward_shadow.csv")
    res = outcomes(ledger, pd.read_csv(pro / "forward_shadow_settled.csv"), ps)
    ps = ps[ps["market"] == "OU25"].sort_values("scored_at").drop_duplicates("fixture_key").copy()
    ps["fkey"] = [fkey(d, a, b) for d, a, b in zip(ps["match_date"], ps["home_team"], ps["away_team"])]
    res = add_history_outcomes(res, ps[["fkey", "match_date", "league", "home_team", "away_team"]])
    s = ps.join(res["y"], on="fkey")
    s = s[s["y"].notna()]
    y = s["y"].to_numpy()

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    md = s["match_date"].astype(str).to_numpy()
    out = {"n": int(len(s)), "matchdays": int(len(set(md)))}
    for a, b in (("p_V9_REPAIRED", "p_CURRENT_V9_SERVED"), ("p_V9_REPAIRED", "p_MARKET"),
                 ("p_CURRENT_V9_SERVED", "p_MARKET")):
        d = ll(s[a].to_numpy()) - ll(s[b].to_numpy())
        out[f"{a}_minus_{b}"] = {"mean_dLL": round(float(d.mean()), 5), "ci95_matchday_boot": boot_ci(d, md)}
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--paired":
    print(json.dumps(paired_dll(sys.argv[2]), indent=1))
