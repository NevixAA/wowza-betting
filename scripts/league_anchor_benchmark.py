"""Champion vs A (league-level inputs) vs B (market-anchored) on identical rows. RESEARCH ONLY.

    python scripts/league_anchor_benchmark.py --task std_ou25 --out _out [--fd-dir _fd]

Tasks and frames are those of scripts/logit_calibration_benchmark.py (serving structure kept, side
markets trained POOLED as production does, every metric reported PER TRACK, never pooled).

  CHAMPION   src.model.train_repaired, calibration_space "prob", the serving payload's features
  A_LEAGUE   the same trainer + as-of league-level features (src/league_level.py)
  B_ANCHOR   logistic on [logit(p_market_entry), logit(p_model)], refitted at each holdout month on
             EARLIER priced rows only (calibration + meta blocks, then earlier holdout months; the base
             block is excluded because base models are in-sample there). B_CHAMP uses the champion's
             p, B_LEAGUE uses A's p. MARKET is the de-vigged entry price alone.

REAL ENTRY PRICES ONLY, two-sided, de-vigged proportionally (no closing prices, no one-sided proxies):
  Standard O/U 2.5   football-data NON-C columns (Avg>2.5, B365>2.5, P>2.5, Max>2.5) - --fd-dir
  New Format O/U     output/newformat_odds_history.csv over25/under25 (best-of-books; rows with a
                     negative two-way margin are flagged and EXCLUDED from market work)
  Standard side      output/standard_sidemarket_odds_history.csv first pre-kickoff snapshot; rows
                     2026-06-01..2026-08-22 dropped (forward capture trusted from 08-23); BTTS Yes > 3.2
                     dropped (first-half mislabel)
  New Format side    output/newformat_odds_history.csv, same BTTS rule
Tip simulation: production's edge rule and the DEPLOYED thresholds parsed from predict.yml (CLAUDE.md
trap 1), at the same entry prices. Holdout = newest 20% (untouched); live rows = date >= 2026-09-01.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import config  # noqa: E402
from src import model as M  # noqa: E402
from src import standard_repair as SR  # noqa: E402
from src import league_level as LL  # noqa: E402
import logit_calibration_benchmark as LB  # noqa: E402

LIVE_FROM = "2026-09-01"
RNG = np.random.default_rng(20261102)
ARMS = ("CHAMPION", "A_LEAGUE")
SIDE_MKTS = {"btts": ("btts_yes", "btts_no"), "over15": ("over15", "under15"), "over35": ("over35", "under35"),
             "over25": ("over25", "under25")}


# ── deployed thresholds ───────────────────────────────────────────────────────────────────────
def deployed() -> dict:
    text = (ROOT / ".github" / "workflows" / "predict.yml").read_text(encoding="utf-8")
    env = {m.group(1): m.group(2).split(" #")[0].strip().strip('"').strip("'")
           for m in re.finditer(r'^\s+([A-Z0-9_]+):\s*(.*?)\s*$', text, flags=re.M) if "${{" not in m.group(2)}
    f = lambda k, d: float(env.get(k, os.getenv(k, d)))
    return {"VALUABLE": f("VALUABLE_THRESHOLD", "0.04"), "MARKSMAN": f("MARKSMAN_THRESHOLD", "0.14"),
            "SNIPER": min(f("SNIPER_THRESHOLD", "0.12"), f("LEAGUE_SNIPER_CAP", "1.0")),
            "MIN_OVER": f("MIN_OVER_ODDS", "1.75"), "MIN_UNDER": f("MIN_UNDER_ODDS", "1.75"),
            "NF_MAX_OU": f("NF_MAX_OU_ODDS", "3.00"),
            "SIDE_VALUABLE": 0.04, "SIDE_MARKSMAN": 0.08, "SIDE_SNIPER": 0.10, "SIDE_OVERROUND": 1.08,
            "source": {k: env[k] for k in ("VALUABLE_THRESHOLD", "MARKSMAN_THRESHOLD", "LEAGUE_SNIPER_CAP",
                                           "NF_MAX_OU_ODDS") if k in env}}


# ── real entry prices ─────────────────────────────────────────────────────────────────────────
def _odds_file(name: str) -> pd.DataFrame:
    p = ROOT / "output" / name
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    d["snapshot_ts"] = pd.to_datetime(d["snapshot_ts"], utc=True, errors="coerce")
    d["kickoff_utc"] = pd.to_datetime(d.get("kickoff_utc"), utc=True, errors="coerce")
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce").dt.normalize()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d = d[d["kickoff_utc"].isna() | (d["snapshot_ts"] < d["kickoff_utc"])]
    return d


def entry_pairs(target: str, track: str) -> pd.DataFrame:
    """(league, match_date, home, away, a, b): FIRST pre-kickoff two-sided quote of the target market."""
    yes, no = SIDE_MKTS[target]
    if track == "standard":
        d = _odds_file("standard_sidemarket_odds_history.csv")
        if len(d):
            d = d[(d["match_date"] < "2026-06-01") | (d["match_date"] >= "2026-08-23")]
    else:
        d = _odds_file("newformat_odds_history.csv")
    if d.empty:
        return pd.DataFrame()
    d = d[d["market"].isin([yes, no])]
    if target == "btts":
        d = d[~((d["market"] == "btts_yes") & (d["odds"] > 3.2))]
    w = d.pivot_table(index=["league", "match", "match_date", "snapshot_ts"], columns="market",
                      values="odds", aggfunc="median").reset_index()
    if yes not in w.columns or no not in w.columns:
        return pd.DataFrame()
    w = w.dropna(subset=[yes, no]).sort_values("snapshot_ts")
    w = w.drop_duplicates(["league", "match", "match_date"], keep="first")
    w[["home", "away"]] = w["match"].str.split(" vs ", n=1, expand=True)
    return w.rename(columns={yes: "odds_a", no: "odds_b"})[["league", "match_date", "home", "away", "odds_a", "odds_b"]]


def join_prices(rows: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Attach odds_a/odds_b to rows by league-scoped name resolution (src.team_names.resolve), date +-1."""
    from src.team_names import resolve
    out = pd.DataFrame(index=rows.index, columns=["odds_a", "odds_b"], dtype=float)
    if px.empty:
        return out
    r = rows.assign(_d=pd.to_datetime(rows["date"]).dt.normalize())
    for lg, g in px.groupby("league"):
        rr = r[r["league"] == lg]
        if rr.empty:
            continue
        teams = sorted(set(rr["home_team"].astype(str)) | set(rr["away_team"].astype(str)))
        key = {(d, h, a): i for i, d, h, a in zip(rr.index, rr["_d"], rr["home_team"].astype(str), rr["away_team"].astype(str))}
        cache = {}
        for x in g.itertuples(index=False):
            for nm in (x.home, x.away):
                if nm not in cache:
                    cache[nm] = resolve(str(nm), teams)
            h, a = cache[x.home], cache[x.away]
            if not h or not a:
                continue
            for off in (0, -1, 1):
                i = key.get((x.match_date + pd.Timedelta(days=off), h, a))
                if i is not None:
                    out.loc[i, ["odds_a", "odds_b"]] = [x.odds_a, x.odds_b]
                    break
    return out


def fd_prices(rows: pd.DataFrame, fd_dir: str | None) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index, columns=["odds_a", "odds_b"], dtype=float)
    if not fd_dir:
        return out
    px = SR.load_fd_entry_prices(fd_dir)
    f = SR.attach_entry_prices(rows.reset_index(drop=True), px)
    out["odds_a"] = pd.to_numeric(f["odds_over25_entry"], errors="coerce").to_numpy()
    out["odds_b"] = pd.to_numeric(f["odds_under25_entry"], errors="coerce").to_numpy()
    return out


def real_prices(rows: pd.DataFrame, task: str, target: str, fd_dir: str | None) -> tuple[pd.DataFrame, dict]:
    track = rows["league"].map(config.model_type_for_league)
    out = pd.DataFrame(index=rows.index, columns=["odds_a", "odds_b"], dtype=float)
    src = {}
    if task == "std_ou25":
        out = fd_prices(rows, fd_dir)
        src["standard"] = "FD_ENTRY_NON_C (Avg>2.5 > B365>2.5 > P>2.5 > Max>2.5)"
    else:
        for t in ("standard", "new_format"):
            m = track == t
            if not m.any() or (task == "nf_ou25" and t == "standard"):
                continue
            if t == "standard" and target == "over25":
                continue
            j = join_prices(rows[m], entry_pairs(target, t))
            out.loc[m, ["odds_a", "odds_b"]] = j.to_numpy()
            src[t] = ("standard_sidemarket_odds_history.csv first pre-kickoff" if t == "standard"
                      else "newformat_odds_history.csv first pre-kickoff (best-of-books)")
    a, b = pd.to_numeric(out["odds_a"], errors="coerce"), pd.to_numeric(out["odds_b"], errors="coerce")
    margin = 1 / a + 1 / b - 1
    out["margin"] = margin
    out["neg_margin"] = margin < 0
    ia, ib = 1 / a, 1 / b
    out["p_mkt"] = (ia / (ia + ib)).where(a.notna() & b.notna() & (margin >= 0))
    return out, src


# ── arms ──────────────────────────────────────────────────────────────────────────────────────
def anchor(fit_pool: pd.DataFrame, hold: pd.DataFrame, pcol: str) -> pd.Series:
    """Expanding monthly refit of y ~ logit(p_mkt) + logit(p_model) on strictly earlier priced rows."""
    from sklearn.linear_model import LogisticRegression
    out = pd.Series(np.nan, index=hold.index)
    hp = hold.dropna(subset=["p_mkt"])
    if hp.empty:
        return out
    pool = pd.concat([fit_pool.dropna(subset=["p_mkt"]), hp], ignore_index=False)
    months = hp["date"].dt.to_period("M").unique()
    for mo in sorted(months):
        start = mo.start_time
        tr = pool[pool["date"] < start]
        if len(tr) < 300:
            continue
        X = lambda d: np.c_[LB._lg(d["p_mkt"]), LB._lg(d[pcol])]
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(X(tr), tr["y"])
        te = hp[hp["date"].dt.to_period("M") == mo]
        out.loc[te.index] = lr.predict_proba(X(te))[:, 1]
    return out


# ── evaluation ───────────────────────────────────────────────────────────────────────────────
def _metrics(h: pd.DataFrame, col: str) -> dict:
    d = h.dropna(subset=[col])
    if len(d) < 50:
        return {"n": int(len(d)), "status": "TOO_FEW"}
    y, p = d["y"].to_numpy(), d[col].to_numpy()
    return {"n": int(len(d)), "log_loss": round(float(LB._ll(y, p).mean()), 5),
            "brier": round(float(np.mean((p - y) ** 2)), 5), "calibration": LB.cal_line(y, p),
            "mean_p": round(float(p.mean()), 4)}


def _spread(h: pd.DataFrame, cols: list) -> dict:
    g = h.groupby("league").agg(n=("y", "size"), actual=("y", "mean"), **{c: (c, "mean") for c in cols})
    g = g[g["n"] >= 30]
    out = {"leagues": int(len(g)), "sd_actual": round(float(g["actual"].std()), 4)}
    for c in cols:
        if g[c].notna().sum() >= 3:
            out[f"sd_{c}"] = round(float(g[c].std()), 4)
            out[f"mean_abs_league_error_{c}_pp"] = round(float((g[c] - g["actual"]).abs().mean() * 100), 2)
    return out


def _delta(h: pd.DataFrame, a: str, b: str) -> dict:
    d = h.dropna(subset=[a, b])
    if len(d) < 50:
        return {"n": int(len(d))}
    x = LB._ll(d["y"].to_numpy(), d[a]) - LB._ll(d["y"].to_numpy(), d[b])
    return {"n": int(len(d)), "delta_ll": round(float(x.mean()), 5), "ci95": LB._boot_mean(x, d["day"])}


def _residual(h: pd.DataFrame, col: str) -> dict:
    d = h.dropna(subset=["p_mkt", col]).rename(columns={col: "p_arm"})
    return LB.residual(d.assign(p_prob=d["p_arm"]), "prob") if len(d) else {"n": 0}


def tips(h: pd.DataFrame, col: str, target: str, track: str, th: dict) -> pd.DataFrame:
    d = h.dropna(subset=[col, "odds_a", "odds_b", "p_mkt"]).copy()
    if d.empty:
        return d
    p = d[col]
    if target == "over25":
        eo, eu = p - 1 / d["odds_a"], (1 - p) - 1 / d["odds_b"]
        over = eo >= eu
        d["side"] = np.where(over, "OVER", "UNDER")
        d["edge"] = np.where(over, eo, eu)
        d["odds"] = np.where(over, d["odds_a"], d["odds_b"])
        ok = (d["edge"] >= th["VALUABLE"]) & np.where(over, d["odds"] >= th["MIN_OVER"], d["odds"] >= th["MIN_UNDER"])
        if track == "new_format":
            ok &= d["odds"] <= th["NF_MAX_OU"]
        d["win"] = np.where(over, d["y"] == 1, d["y"] == 0)
        d["tier"] = np.select([d["edge"] >= th["SNIPER"], d["edge"] >= th["MARKSMAN"]], ["SNIPER", "MARKSMAN"], "VALUABLE")
    else:
        d["side"] = "YES" if target == "btts" else "OVER"
        d["edge"] = p - (1 / d["odds_a"]) / th["SIDE_OVERROUND"]
        d["odds"] = d["odds_a"]
        ok = d["edge"] >= th["SIDE_VALUABLE"]
        d["win"] = d["y"] == 1
        d["tier"] = np.select([d["edge"] >= th["SIDE_SNIPER"], d["edge"] >= th["SIDE_MARKSMAN"]], ["SNIPER", "MARKSMAN"], "VALUABLE")
    t = d[ok].copy()
    t["pnl"] = np.where(t["win"], t["odds"] - 1.0, -1.0)
    return t


def tip_summary(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"tips": 0}
    ci = LB._boot_mean(t["pnl"], t["day"]) if t["day"].nunique() >= 5 else [None, None]
    return {"tips": int(len(t)), "split": t["side"].value_counts().to_dict(),
            "tiers": t["tier"].value_counts().to_dict(), "hit": round(float(t["win"].mean()), 4),
            "breakeven": round(float((1 / t["odds"]).mean()), 4), "units": round(float(t["pnl"].sum()), 2),
            "roi": round(float(t["pnl"].mean()), 4), "roi_ci95": ci}


def evaluate(h: pd.DataFrame, target: str, track: str, th: dict) -> dict:
    arms_all = ["p_CHAMPION", "p_A_LEAGUE"]
    priced = h.dropna(subset=["p_mkt"])
    out = {"n": int(len(h)), "n_priced": int(len(priced)), "matchdays": int(h["day"].nunique()),
           "dates": [str(h["date"].min().date()), str(h["date"].max().date())],
           "base_rate": round(float(h["y"].mean()), 4),
           "all_rows": {c[2:]: _metrics(h, c) for c in arms_all},
           "league_level_all_rows": _spread(h, arms_all),
           "delta_A_minus_CHAMPION_all_rows": _delta(h, "p_A_LEAGUE", "p_CHAMPION")}
    if len(priced) >= 100:
        cols = arms_all + ["p_B_CHAMP", "p_B_LEAGUE", "p_mkt"]
        out["priced_rows"] = {c[2:]: _metrics(priced, c) for c in cols}
        out["league_level_priced"] = _spread(priced, cols)
        out["residual_vs_market"] = {c[2:]: _residual(priced, c) for c in arms_all}
        out["delta_vs_market"] = {c[2:]: _delta(priced, c, "p_mkt") for c in ["p_CHAMPION", "p_A_LEAGUE", "p_B_CHAMP", "p_B_LEAGUE"]}
        out["tips"] = {c[2:]: tip_summary(tips(priced, c, target, track, th)) for c in ["p_CHAMPION", "p_A_LEAGUE", "p_B_CHAMP", "p_B_LEAGUE"]}
    else:
        out["priced_rows"] = {"status": "NO_REAL_TWO_SIDED_PRICE" if len(priced) == 0 else "TOO_FEW_PRICED", "n": int(len(priced))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["std_ou25", "nf_ou25", "side_btts", "side_over15", "side_over35"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fd-dir", default=os.getenv("FD_DIR"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    th = deployed()

    rows, target, cols, w, label = LB.frame(a.task)
    rows = LL.add_league_level(rows).copy()            # as-of: computed per league from prior dates only
    # league features from the FULL frame would see more matches per league than the track rows do,
    # but frame() already returns every played league match of the track, so the track rows are the
    # league's own history; side tasks are pooled over both tracks and leagues are track-pure.
    rows["_w_in"] = np.asarray(w, float)
    blocks = M.four_block_split(rows, target, rows["_w_in"].to_numpy())
    cols_a = list(cols) + [c for c in LL.feature_names() if c not in cols]
    pays = {"CHAMPION": M.build_payload(M.train_repaired(rows, target=target, sample_weight=rows["_w_in"].to_numpy(),
                                                          feature_cols=cols, calibration_space="prob"), target),
            "A_LEAGUE": M.build_payload(M.train_repaired(rows, target=target, sample_weight=rows["_w_in"].to_numpy(),
                                                          feature_cols=cols_a, calibration_space="prob"), target)}

    def table(b: pd.DataFrame) -> pd.DataFrame:
        t = pd.DataFrame({"date": pd.to_datetime(b["date"]).to_numpy(), "league": b["league"].astype(str).to_numpy(),
                          "home_team": b["home_team"].to_numpy(), "away_team": b["away_team"].to_numpy(),
                          "y": pd.to_numeric(b[target], errors="coerce").to_numpy()}, index=b.index)
        for arm, pay in pays.items():
            t[f"p_{arm}"] = M.predict_proba(b, payload=pay).to_numpy()
        px, src = real_prices(b, a.task, target, a.fd_dir)
        for c in ("odds_a", "odds_b", "margin", "neg_margin", "p_mkt"):
            t[c] = px[c].to_numpy()
        t["track"] = t["league"].map(config.model_type_for_league)
        t["day"] = t["date"].dt.strftime("%Y-%m-%d")
        return t.dropna(subset=["y"]), src

    hold, src = table(blocks["holdout"])
    pool = pd.concat([table(blocks["calibration"])[0], table(blocks["meta"])[0]])
    res = {"task": a.task, "target": target, "n_features_champion": len(cols), "n_features_A": len(cols_a),
           "league_features": LL.feature_names(), "price_sources": src, "thresholds": th,
           "blocks": {k: [int(len(v)), str(pd.Timestamp(v["date"].min()).date()), str(pd.Timestamp(v["date"].max()).date())]
                      for k, v in blocks.items()},
           "pooled_training": a.task.startswith("side_"), "tracks": {}}
    tracks = ["standard"] if a.task == "std_ou25" else ["new_format"] if a.task == "nf_ou25" else ["standard", "new_format"]
    allh = []
    for t in tracks:
        ht, pt = hold[hold["track"] == t].copy(), pool[pool["track"] == t]
        ht["p_B_CHAMP"] = anchor(pt, ht, "p_CHAMPION")
        ht["p_B_LEAGUE"] = anchor(pt, ht, "p_A_LEAGUE")
        allh.append(ht)
        if len(ht) < 100:
            res["tracks"][t] = {"n": int(len(ht)), "status": "TOO_FEW"}
            continue
        priced_any = ht["odds_a"].notna() & ht["odds_b"].notna()
        res["tracks"][t] = {"price_coverage": round(float(priced_any.mean()), 4),
                            "negative_margin_rows_excluded": int(ht["neg_margin"].fillna(False).sum()),
                            "holdout": evaluate(ht, target, t, th)}
        live = ht[ht["date"] >= LIVE_FROM]
        if len(live) >= 50:
            res["tracks"][t]["live_from_2026_09_01"] = evaluate(live, target, t, th)
    pd.concat(allh).to_csv(out / f"{a.task}_holdout.csv", index=False)
    (out / f"{a.task}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({t: {k: v.get("holdout", {}).get(k) if isinstance(v, dict) else v
                          for k in ("n", "n_priced", "delta_A_minus_CHAMPION_all_rows")}
                      for t, v in res["tracks"].items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
