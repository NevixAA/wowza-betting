"""Champion EV selection in EFL vs bet-time prices (P11 OOS rows): n bets, ROI, bootstrap CI by
date, at EV thresholds. Prints counts and ROI only, never a price."""
import sys
import numpy as np, pandas as pd
u = pd.read_parquet(sys.argv[1]); out = sys.argv[2]
E = ["Championship", "League One", "League Two"]
res = []
for mk in ["over25", "over15", "over35", "btts"]:
    g0 = u[(u.market == mk) & u.league.isin(E + ["ALL"]) ]
    for side in (["over", "under"] if mk == "over25" else ["yes"]):
        for grp in E + ["EFL"]:
            g = g0 if grp == "EFL" else g0[g0.league == grp]
            if side == "under":
                p, o, win = 1 - g.p.to_numpy(), g["no"].to_numpy(float), 1 - g.y.to_numpy()
            else:
                p, o, win = g.p.to_numpy(), g["yes"].to_numpy(float), g.y.to_numpy()
            for th in (0.0, 0.03, 0.05, 0.10):
                s = (o > 1) & (p * o > 1 + th)
                if s.sum() < 5:
                    res.append({"market": mk, "side": side, "group": grp, "ev_min": th, "bets": int(s.sum())}); continue
                pnl = np.where(win[s] == 1, o[s] - 1, -1.0)
                dts = pd.to_datetime(g.date.to_numpy()[s]).normalize()
                gs = pd.Series(pnl).groupby(dts.values); S, C = gs.sum().to_numpy(), gs.size().to_numpy()
                rng = np.random.default_rng(1); idx = rng.integers(0, len(S), (2000, len(S)))
                bs = S[idx].sum(1) / C[idx].sum(1)
                res.append({"market": mk, "side": side, "group": grp, "ev_min": th, "bets": int(s.sum()),
                            "roi": round(float(pnl.mean()), 4), "ci95": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)]})
r = pd.DataFrame(res); r.to_csv(out, index=False)
pd.set_option("display.width", 200); print(r[r.group == "EFL"].to_string(index=False))
