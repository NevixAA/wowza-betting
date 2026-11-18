"""Variant: long-window team ratings (EWMA, half-life 8 / 20 matches, shift(1) = strictly before the
match, per league x team) of goals for/against, shots, SoT, inside-box shots and xG, added to the
Standard feature list; logistic + league FE; same holdout protocol. Metrics only (no prices)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.retrain_harness import protocol as P
from research.efl_diag.experiments import (LogisticFE, metrics, boot_diff, row_ids, market_prob, EFL)

out = Path(sys.argv[1]); uni = pd.read_parquet(sys.argv[2]); p11 = pd.read_parquet(sys.argv[3])
feat = pd.read_parquet(out / "feat.parquet").sort_values("date").reset_index(drop=True)
cfg = P.load_config()
feats, _ = P.feature_list(cfg, "standard", feat.columns)
feat["_i"] = np.arange(len(feat))
stats = {"gf": ("home_goals", "away_goals"), "ga": ("away_goals", "home_goals"), "sh": ("home_shots", "away_shots"),
         "sot": ("home_sot", "away_sot"), "ib": ("home_insidebox", "away_insidebox"), "xg": ("home_xg", "away_xg")}
h = pd.DataFrame({"_i": feat._i, "league": feat.league, "team": feat.home_team, "date": feat.date, "side": "home"})
a = pd.DataFrame({"_i": feat._i, "league": feat.league, "team": feat.away_team, "date": feat.date, "side": "away"})
for k, (hc, ac) in stats.items():
    h[k] = pd.to_numeric(feat[hc], errors="coerce").to_numpy(); a[k] = pd.to_numeric(feat[ac], errors="coerce").to_numpy()
tc = pd.concat([h, a]).sort_values(["date", "_i"], kind="mergesort").reset_index(drop=True)
new = []
for hl in (8, 20):
    for k in stats:
        col = f"ew{hl}_{k}"
        # same-day rows are strictly earlier only if on earlier dates: shift by one match per team (one match/day)
        tc[col] = tc.groupby(["league", "team"])[k].transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=3, ignore_na=True).mean())
        new.append(col)
H = tc[tc.side == "home"].set_index("_i")[new].add_prefix("home_")
A = tc[tc.side == "away"].set_index("_i")[new].add_prefix("away_")
feat = feat.join(H, on="_i").join(A, on="_i")
extra = list(H.columns) + list(A.columns)
for hl in (8, 20):
    feat[f"exp_total_{hl}"] = (feat[f"home_ew{hl}_gf"] + feat[f"away_ew{hl}_ga"] + feat[f"away_ew{hl}_gf"] + feat[f"home_ew{hl}_ga"]) / 2
    feat[f"exp_xt_{hl}"] = feat[f"home_ew{hl}_ib"] + feat[f"away_ew{hl}_ib"]
    extra += [f"exp_total_{hl}", f"exp_xt_{hl}"]
rows_out, cmp_out = [], []
for market in ["over25", "over15", "over35", "btts"]:
    mcfg = cfg["markets"][market]
    rows = P.eligible_rows(feat, cfg, "standard", market)
    pre, hold, _ = P.split_holdout(rows, cfg)
    fr = P.final_frames(pre, hold, cfg); test = fr["test"]
    y = test[mcfg["target"]].to_numpy(float); ids = row_ids(test); lg = test.league.to_numpy(); dates = test.date.to_numpy()
    leagues = sorted(rows.league.unique())
    preds = {"market_bettime": market_prob(uni, market, ids)}
    ch = p11[(p11.stage == "holdout") & (p11.market == market)].drop_duplicates("row_id").set_index("row_id")
    preds["champion_p11"] = ch["p"].reindex(ids).to_numpy(float)
    for name, fl in [("logFE_longwin", feats + extra), ("logFE_longwin_only", extra)]:
        c = LogisticFE(cfg["candidates"]["logistic"], mcfg, fl, "standard", market, leagues=leagues).fit(fr)
        preds[name] = c.predict(test)
    for grp, m in [("EFL", np.isin(lg, EFL))] + [(l, lg == l) for l in EFL] + [("ALL_STD", np.ones(len(lg), bool))]:
        for n, p in preds.items():
            rows_out.append({"market": market, "group": grp, "model": n, **metrics(y[m], p[m])})
        for n in ("logFE_longwin", "logFE_longwin_only"):
            for ref in ("market_bettime", "champion_p11"):
                b = boot_diff(y[m], preds[n][m], preds[ref][m], dates[m])
                if b: cmp_out.append({"market": market, "group": grp, "model": n, "vs": ref, **b})
pd.DataFrame(rows_out).to_csv(out / "exp" / "longwin_metrics.csv", index=False)
c = pd.DataFrame(cmp_out); c.to_csv(out / "exp" / "longwin_compare.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows_out).query("group in ['EFL','ALL_STD']").to_string(index=False))
print(c[c.group == "EFL"].to_string(index=False))
