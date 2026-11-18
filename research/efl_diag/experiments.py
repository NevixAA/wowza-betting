"""EFL diagnosis experiments on the P11 holdout protocol (research only, nothing served).

Same rows and splits as the phase2 harness (protocol.eligible_rows / split_holdout / final_frames):
models fitted on dates < 2026-04-01, scored once on 2026-04-01..2026-11-08. Also re-scores the
P11 champion walk-forward OOS predictions. Outputs metrics only (no prices) to <out>/exp/.

    python research/efl_diag/experiments.py <out_dir> <p11_predictions.parquet> <priced_universe.parquet>
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from src.retrain_harness import candidates as C  # noqa: E402
from src.retrain_harness import protocol as P  # noqa: E402
from src.market_anchor import power_devig  # noqa: E402

EFL = ["Championship", "League One", "League Two"]
MARKETS = ["over25", "over15", "over35", "btts", "ht_over05", "ht_over15"]
EPS = 1e-6
EXTRA = ["home_shots_last5", "away_shots_last5", "home_sot_last5", "away_sot_last5", "league_avg_goals"]


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll_rows(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(y, p):
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 20:
        return {"n": int(len(y))}
    return {"n": int(len(y)), "auc": round(float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan, 4),
            "ll": round(float(ll_rows(y, p).mean()), 5), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "mean_p": round(float(p.mean()), 4), "obs": round(float(y.mean()), 4),
            "p_sd": round(float(p.std()), 4)}


def boot_diff(y, pa, pb, dates, B=2000, seed=7):
    """mean LL(a) - LL(b) per row, bootstrap over match dates. Negative = a better."""
    ok = np.isfinite(pa) & np.isfinite(pb)
    if ok.sum() < 20:
        return None
    d = ll_rows(y[ok], pa[ok]) - ll_rows(y[ok], pb[ok])
    dt = pd.Series(d).groupby(pd.to_datetime(dates[ok]).normalize().values)
    s, c = dt.sum().to_numpy(), dt.size().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (B, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return {"n": int(ok.sum()), "diff": round(float(d.mean()), 5),
            "ci95": [round(float(np.quantile(bs, .025)), 5), round(float(np.quantile(bs, .975)), 5)],
            "p_a_better": round(float((bs < 0).mean()), 3)}


def row_ids(df):
    return (pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") + "|" + df["league"].astype(str) + "|"
            + df["home_team"].astype(str) + "|" + df["away_team"].astype(str)).to_numpy()


def market_prob(uni: pd.DataFrame, market: str, ids, col_yes="yes", col_no="no"):
    """De-vigged (power) two-sided price -> P(yes); NaN when not two-sided. Prices never leave."""
    u = uni[(uni.market == market)].drop_duplicates("row_id").set_index("row_id")
    u = u.reindex(ids)
    y, n = u[col_yes].to_numpy(float), u[col_no].to_numpy(float)
    out = np.full(len(ids), np.nan)
    for i in np.flatnonzero((y > 1) & (n > 1)):
        pr = power_devig([y[i], n[i]])
        if len(pr) == 2:
            out[i] = pr[0]
    return out


class LogisticFE(C.Logistic):
    """Logistic + league one-hot (fixed effects) + extra columns, train-only imputer as harness."""
    def __init__(self, *a, leagues=(), **k):
        super().__init__(*a, **k)
        self.leagues = list(leagues)

    def _aug(self, df):
        X = C._X(df, self.features).copy()
        for lg in self.leagues:
            X["lg_" + lg] = (df["league"].astype(str).to_numpy() == lg).astype(float)
        return X

    def fit(self, frames):
        from sklearn.impute import MissingIndicator, SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import StandardScaler
        tr = frames["train"]
        X = self._aug(tr)
        self.used = [c for c in X.columns if X[c].notna().any()]
        self.pipe = Pipeline([("inputs", FeatureUnion([("imp", SimpleImputer(strategy="median")),
                                                       ("miss", MissingIndicator(features="all"))])),
                              ("scale", StandardScaler()),
                              ("clf", LogisticRegression(C=float(self.params["C"]), max_iter=3000))])
        self.pipe.fit(X[self.used], C._y(tr, self.target), clf__sample_weight=C._weights(tr, "derived"))
        self.fitted = True
        return self

    def predict(self, df):
        return self.pipe.predict_proba(self._aug(df)[self.used])[:, 1]


def subset_frames(fr, leagues):
    return {k: v[v["league"].isin(leagues)] for k, v in fr.items()}


def trailing_base_rate(train, test, target, days=730):
    end = pd.to_datetime(train["date"]).max()
    t = train[pd.to_datetime(train["date"]) > end - pd.Timedelta(days=days)]
    r = t.groupby("league")[target].mean()
    return test["league"].map(r).to_numpy(float), float(t[target].mean())


def main():
    out = Path(sys.argv[1])
    p11 = pd.read_parquet(sys.argv[2])
    uni = pd.read_parquet(sys.argv[3])
    exp = out / "exp"
    exp.mkdir(exist_ok=True)
    feat = pd.read_parquet(out / "feat.parquet")
    cfg = P.load_config()
    feats, _ = P.feature_list(cfg, "standard", feat.columns)
    feats_plus = feats + [c for c in EXTRA if c in feat.columns and c not in feats]
    rows_out, cmp_out, store = [], [], {}
    for market in MARKETS:
        mcfg = cfg["markets"][market]
        rows = P.eligible_rows(feat, cfg, "standard", market)
        pre, hold, _ = P.split_holdout(rows, cfg)
        fr = P.final_frames(pre, hold, cfg)
        test = fr["test"]
        y = test[mcfg["target"]].to_numpy(float)
        ids = row_ids(test)
        lg = test["league"].astype(str).to_numpy()
        dates = test["date"].to_numpy()
        preds = {}
        ch = p11[(p11.stage == "holdout") & (p11.market == market)].drop_duplicates("row_id").set_index("row_id")
        preds["champion_p11"] = ch["p"].reindex(ids).to_numpy(float)
        br, glob = trailing_base_rate(fr["train"], test, mcfg["target"])
        preds["base_league_2y"] = br
        preds["base_global_2y"] = np.full(len(test), glob)
        preds["market_bettime"] = market_prob(uni, market, ids, "yes", "no")
        preds["market_close"] = market_prob(uni, market, ids, "yes_c", "no_c")
        leagues = sorted(rows["league"].astype(str).unique())

        def run(name, cand, frames, score_df=test):
            try:
                cand.fit(frames)
                p = np.full(len(test), np.nan)
                m = test["league"].isin(score_df["league"].unique()).to_numpy()
                p[m] = cand.predict(test[m])
                preds[name] = p
            except Exception as e:  # noqa: BLE001
                print(f"  {market}/{name} failed: {type(e).__name__}: {e}")
        lp = cfg["candidates"]["logistic"]
        hp = cfg["candidates"]["hgb"]
        run("logistic_all", C.Logistic(lp, mcfg, feats, "standard", market), fr)
        run("hgb_all", C.HGB(hp, mcfg, feats, "standard", market), fr)
        run("logistic_leagueFE", LogisticFE(lp, mcfg, feats, "standard", market, leagues=leagues), fr)
        run("logistic_leagueFE_shots", LogisticFE(lp, mcfg, feats_plus, "standard", market, leagues=leagues), fr)
        efr = subset_frames(fr, EFL)
        run("logistic_eflonly", LogisticFE(lp, mcfg, feats, "standard", market, leagues=EFL), efr, efr["test"])
        run("hgb_eflonly", C.HGB(hp, mcfg, feats, "standard", market), efr, efr["test"])
        dcp = dict(cfg["candidates"]["dixon_coles"])
        run("dixon_coles", C.DixonColesCandidate(dcp, mcfg, feats, "standard", market), fr)
        # stack DC + logistic_FE on the validation block (fit block = train minus validation)
        try:
            ffit = {"train": fr["fit"], "test": fr["validation"]}
            dc = C.DixonColesCandidate(dcp, mcfg, feats, "standard", market).fit(ffit)
            lf = LogisticFE(lp, mcfg, feats_plus, "standard", market, leagues=leagues).fit(ffit)
            v = fr["validation"]
            Z = np.column_stack([logit(lf.predict(v)), logit(dc.predict(v))])
            yv = v[mcfg["target"]].to_numpy(float)
            ok = np.isfinite(Z).all(1)
            from sklearn.linear_model import LogisticRegression
            st = LogisticRegression(C=1e6, max_iter=1000).fit(Z[ok], yv[ok])
            Zt = np.column_stack([logit(lf.predict(test)), logit(dc.predict(test))])
            p = np.full(len(test), np.nan)
            okt = np.isfinite(Zt).all(1)
            p[okt] = st.predict_proba(Zt[okt])[:, 1]
            preds["stack_dc_logFE"] = p
        except Exception as e:  # noqa: BLE001
            print(f"  {market}/stack failed: {e}")
        # model+market: stack [logit champion, logit bet-time market] fitted on P11 WF OOS rows (EFL+all)
        wf = p11[(p11.stage == "walk_forward") & (p11.market == market)].drop_duplicates("row_id")
        wf_m = market_prob(uni[uni.stage == "walk_forward"], market, wf["row_id"].to_numpy())
        okw = np.isfinite(wf_m)
        if okw.sum() > 200:
            from sklearn.linear_model import LogisticRegression
            Zw = np.column_stack([logit(wf["p"].to_numpy()[okw]), logit(wf_m[okw])])
            st = LogisticRegression(C=1e6, max_iter=1000).fit(Zw, wf["y"].to_numpy()[okw])
            Zt = np.column_stack([logit(preds["champion_p11"]), logit(preds["market_bettime"])])
            p = np.full(len(test), np.nan)
            okt = np.isfinite(Zt).all(1)
            p[okt] = st.predict_proba(Zt[okt])[:, 1]
            preds["champ+market_stack"] = p
            store[f"{market}_stack_coef"] = [round(float(x), 4) for x in st.coef_[0]] + [int(okw.sum())]
        for grp, mask in [("EFL", np.isin(lg, EFL))] + [(l, lg == l) for l in EFL] + [("ALL_STD", np.ones(len(lg), bool))]:
            for name, p in preds.items():
                rows_out.append({"market": market, "group": grp, "model": name, **metrics(y[mask], p[mask])})
            for name, p in preds.items():
                for ref in ("base_league_2y", "market_bettime", "champion_p11"):
                    if name == ref:
                        continue
                    b = boot_diff(y[mask], p[mask], preds[ref][mask], dates[mask])
                    if b:
                        cmp_out.append({"market": market, "group": grp, "model": name, "vs": ref, **b})
        print(f"{market}: done ({len(test)} holdout rows)", flush=True)
    pd.DataFrame(rows_out).to_csv(exp / "holdout_metrics.csv", index=False)
    pd.DataFrame(cmp_out).to_csv(exp / "holdout_compare.csv", index=False)
    (exp / "notes.json").write_text(json.dumps(store, indent=1))


if __name__ == "__main__":
    main()
