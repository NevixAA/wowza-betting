"""Fantasy V2 models: Wowza as a RESIDUAL on FPL's own ep_next, evaluated gameweek by gameweek.

    final = FPL ep_next + Wowza residual        (residual applied only once it has earned it)

WHY A RESIDUAL. FPL's free `ep_next` beat the standalone Wowza projection in every settled
gameweek (MAE 1.36 vs 2.04, Spearman 0.73 vs 0.58, Wowza bias +1.0). Another standalone xPts
model would be the same mistake again. The only defensible question is whether Wowza's evidence
— minutes and availability, attacking role, fixture context, value signals, its own projection —
explains what ep_next gets WRONG.

ARMS (all scored on the same player-gameweeks):
  FPL         ep_next alone (the champion)
  WOWZA       the existing standalone projection x P(start) (only where it was published)
  LINEAR      OLS of (actual - ep_next) on nine hand-picked features
  RIDGE       ridge regression of the residual on every feature, alpha chosen by
              leave-one-gameweek-out inside the training window
  HGB         shallow, heavily regularised HistGradientBoosting on the residual
  RANK        not built: learning-to-rank needs many query groups (gameweeks); there are three.

WALK-FORWARD ONLY. Test gameweek g is predicted by models fitted on gameweeks < g. No random
split; every feature is pre-deadline (see fantasy_panel).

Heavy imports (scikit-learn, scipy) live inside functions: this module is batch-only, and the
dashboard never imports it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from player_model.fantasy_panel import POSITIONS, active_mask

# ── feature groups (used for decomposition and importance) ───────────────────────────────────
GROUPS = {
    "minutes": ["chance_next_f", "is_out", "is_doubt", "has_news", "start_rate",
                "min_per_team_game", "min_last_imp", "started_last_imp", "wowza_p_start_imp"],
    "attack": ["att_pts90", "xg90", "xa90", "pen_first", "set_piece", "bps90", "threat_pg",
               "creativity_pg"],
    "fixture": ["n_fix", "home", "fdr", "lam_for", "lam_against", "cs_exp", "attack_fixture"],
    "value": ["price", "log_own", "net_transfers_s", "ppm", "pts_per_team_game", "form_n",
              "pts_last_imp"],
    "wowza": ["wowza_gap", "has_wowza"],
    "baseline": ["ep_next"] + [f"pos_{p}" for p in POSITIONS],
}
FEATURES = [c for g in GROUPS.values() for c in g]
LINEAR_FEATURES = ["ep_next", "chance_next_f", "start_rate", "min_last_imp", "att_pts90_x_start",
                   "cs_exp_x_start", "n_fix", "wowza_gap", "has_wowza"]
ARMS = ["FPL", "WOWZA", "LINEAR", "RIDGE", "HGB"]
RESIDUAL_ARMS = ["LINEAR", "RIDGE", "HGB"]


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    """Derived/imputed model columns. Pure and row-wise, so train and test see the same thing."""
    x = d.copy()
    x["min_last_imp"] = x["min_last"].fillna(x["min_per_team_game"]).fillna(0) / 90.0
    x["started_last_imp"] = x["started_last"].fillna(x["start_rate"]).fillna(0).clip(0, 2)
    x["wowza_p_start_imp"] = x["wowza_p_start"].fillna(x["start_rate"]).fillna(0)
    x["att_pts90"] = x["xg90"] * x["goal_pts"] + x["xa90"] * 3.0
    x["att_pts90_x_start"] = x["att_pts90"] * x["start_rate"].fillna(0)
    x["cs_exp_x_start"] = x["cs_exp"].fillna(0) * x["start_rate"].fillna(0)
    x["log_own"] = np.log1p(x["ownership"].fillna(0))
    x["net_transfers_s"] = np.sign(x["net_transfers_k"]) * np.log1p(x["net_transfers_k"].abs())
    x["form_n"] = pd.to_numeric(x.get("form"), errors="coerce")
    x["pts_last_imp"] = x["pts_last"].fillna(x["pts_per_team_game"]).fillna(0)
    x["has_wowza"] = x["wowza_uncond"].notna().astype(int)
    x["wowza_gap"] = (x["wowza_uncond"] - x["ep_next"]).fillna(0.0)
    for c in ("home", "fdr", "lam_for", "lam_against", "cs_exp", "attack_fixture"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["home"] = x["home"].fillna(0.5)
    x["fdr"] = x["fdr"].fillna(3.0)
    x["lam_for"] = x["lam_for"].fillna(0.0)
    x["lam_against"] = x["lam_against"].fillna(0.0)
    x["cs_exp"] = x["cs_exp"].fillna(0.0)
    x["attack_fixture"] = x["attack_fixture"].fillna(1.0)
    for c in FEATURES + LINEAR_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def _matrix(x: pd.DataFrame, cols) -> np.ndarray:
    m = x[cols].to_numpy(float)
    return np.where(np.isfinite(m), m, 0.0)


# ── residual models ───────────────────────────────────────────────────────────────────────────
#: how the LINEAR residual's terms map onto the decomposition the dashboard shows
LINEAR_GROUPS = {
    "calibration": ["ep_next"],                          # FPL's own number is over-dispersed
    "availability": ["chance_next_f"],
    "minutes": ["start_rate", "min_last_imp"],
    "attacking_role": ["att_pts90_x_start"],
    "fixture": ["cs_exp_x_start", "n_fix"],
    "wowza_projection": ["wowza_gap", "has_wowza"],
}


class LinearResidual:
    cols = LINEAR_FEATURES

    def fit(self, x, r, groups=None):
        X = _matrix(x, self.cols)
        self.mean_ = X.mean(0)
        X1 = np.column_stack([np.ones(len(x)), X])
        self.coef_ = np.linalg.lstsq(X1, r, rcond=None)[0]
        return self

    def predict(self, x):
        return np.column_stack([np.ones(len(x)), _matrix(x, self.cols)]) @ self.coef_

    def contributions(self, x) -> pd.DataFrame:
        """Exact additive split: residual = base + sum of group terms, each term measured from
        the training-average player, so 'minutes +0.8' means 'his minutes evidence is worth 0.8
        more than an average player's'."""
        X = _matrix(x, self.cols)
        terms = (X - self.mean_) * self.coef_[1:]
        out = pd.DataFrame(index=x.index)
        for g, cols in LINEAR_GROUPS.items():
            out[g] = terms[:, [self.cols.index(c) for c in cols]].sum(1)
        out["base"] = self.coef_[0] + float(self.mean_ @ self.coef_[1:])
        return out

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": ["intercept"] + list(self.cols), "coef": self.coef_})


class RidgeResidual:
    """Standardised ridge on all features; alpha by leave-one-gameweek-out inside training."""
    cols = FEATURES
    grid = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

    def __init__(self, alpha: float | None = None):
        self.alpha = alpha

    def _fit(self, X, r, alpha):
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        b0 = r.mean()
        k = Z.shape[1]
        w = np.linalg.solve(Z.T @ Z + alpha * np.eye(k), Z.T @ (r - b0))
        return mu, sd, b0, w

    def fit(self, x, r, groups=None):
        X = _matrix(x, self.cols)
        alpha = self.alpha
        if alpha is None:
            g = np.asarray(groups) if groups is not None else None
            if g is not None and len(np.unique(g)) >= 2:
                best = None
                for a in self.grid:
                    err = []
                    for gg in np.unique(g):
                        tr, te = g != gg, g == gg
                        mu, sd, b0, w = self._fit(X[tr], r[tr], a)
                        err.append(np.mean((((X[te] - mu) / sd) @ w + b0 - r[te]) ** 2))
                    s = float(np.mean(err))
                    if best is None or s < best[0]:
                        best = (s, a)
                alpha = best[1]
            else:
                alpha = 100.0      # one training gameweek: no inner validation possible
        self.alpha_ = alpha
        self.mu_, self.sd_, self.b0_, self.w_ = self._fit(X, r, alpha)
        return self

    def predict(self, x):
        return ((_matrix(x, self.cols) - self.mu_) / self.sd_) @ self.w_ + self.b0_

    def contributions(self, x) -> pd.DataFrame:
        """Exact additive split of the residual by feature group (relative to training means)."""
        Z = (_matrix(x, self.cols) - self.mu_) / self.sd_
        parts = Z * self.w_
        out = pd.DataFrame(index=x.index)
        for g, cols in GROUPS.items():
            idx = [self.cols.index(c) for c in cols]
            out[g] = parts[:, idx].sum(1)
        out["intercept"] = self.b0_
        return out

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.cols, "std_coef": self.w_,
                             "group": [next(g for g, cs in GROUPS.items() if c in cs)
                                       for c in self.cols]}).sort_values(
            "std_coef", key=np.abs, ascending=False)


class HGBResidual:
    cols = FEATURES

    def fit(self, x, r, groups=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m_ = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.04, max_iter=150, min_samples_leaf=40,
            l2_regularization=5.0, random_state=7)
        self.m_.fit(x[self.cols].to_numpy(float), r)
        return self

    def predict(self, x):
        return self.m_.predict(x[self.cols].to_numpy(float))


def make(arm: str):
    return {"LINEAR": LinearResidual, "RIDGE": RidgeResidual, "HGB": HGBResidual}[arm]()


# ── metrics ───────────────────────────────────────────────────────────────────────────────────
def metrics(y, f) -> dict:
    y, f = np.asarray(y, float), np.asarray(f, float)
    ok = np.isfinite(y) & np.isfinite(f)
    y, f = y[ok], f[ok]
    if len(y) < 10:
        return {"n": int(len(y))}
    e = f - y
    rho = pd.Series(f).rank().corr(pd.Series(y).rank())
    return {"n": int(len(y)), "mae": round(float(np.abs(e).mean()), 4),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
            "spearman": round(float(rho), 4), "bias": round(float(e.mean()), 4)}


def top_overlap(g: pd.DataFrame, col: str, k: int = 10) -> int:
    k = min(k, len(g))
    return len(set(g.nlargest(k, col).index) & set(g.nlargest(k, "actual_points").index))


def decision_value(g: pd.DataFrame, col: str, k: int = 15) -> float:
    """Mean actual points of the k players a projection ranks highest — what picking by it earns."""
    return float(g.nlargest(min(k, len(g)), col)["actual_points"].mean())


def cluster_boot_delta(df: pd.DataFrame, a: str, b: str, kind: str = "mae", n: int = 2000,
                       seed: int = 20261020) -> list:
    """90% CI of metric(a) - metric(b), resampling PLAYERS (rows of one player move together)."""
    d = df.dropna(subset=[a, b, "actual_points"])
    if len(d) < 30:
        return [None, None]
    y = d["actual_points"].to_numpy(float)
    if kind == "mae":
        e = np.abs(d[a].to_numpy(float) - y) - np.abs(d[b].to_numpy(float) - y)
    else:
        e = (d[a].to_numpy(float) - y) ** 2 - (d[b].to_numpy(float) - y) ** 2
    pid = d["element"].to_numpy()
    uniq, inv = np.unique(pid, return_inverse=True)
    s = np.bincount(inv, weights=e)
    c = np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n, len(uniq)))
    m = s[pick].sum(1) / c[pick].sum(1)
    if kind == "mse":
        base_b = float(np.mean((d[b].to_numpy(float) - y) ** 2))
        # report as RMSE difference
        m = np.sqrt(np.maximum(base_b + m, 0)) - math.sqrt(base_b)
    return [round(float(np.quantile(m, 0.05)), 4), round(float(np.quantile(m, 0.95)), 4)]


# ── walk-forward ──────────────────────────────────────────────────────────────────────────────
def walk_forward(panel: pd.DataFrame, min_train_gws: int = 1) -> pd.DataFrame:
    """Out-of-sample predictions for every settled gameweek that has earlier FINAL gameweeks."""
    p = prepare(panel)
    settled = p[p["status"].isin(["FINAL", "PROVISIONAL"]) & active_mask(p)].copy()
    final_gws = sorted(settled.loc[settled["status"] == "FINAL", "gw"].unique())
    rows = []
    for gw in sorted(settled["gw"].unique()):
        train = settled[(settled["gw"] < gw) & (settled["status"] == "FINAL")]
        if train["gw"].nunique() < min_train_gws:
            continue
        test = settled[settled["gw"] == gw].copy()
        r = (train["actual_points"] - train["ep_next"]).to_numpy(float)
        test["FPL"] = test["ep_next"]
        test["WOWZA"] = test["wowza_uncond"]
        for arm in RESIDUAL_ARMS:
            m = make(arm)
            m.fit(train, r, groups=train["gw"].to_numpy())
            # Expected FPL points cannot be negative in any useful sense; an additive residual on
            # an ep_next of 0 (a player FPL already knows is out) otherwise produces -3.
            test[arm] = (test["ep_next"] + m.predict(test)).clip(lower=0.0)
            if arm == "RIDGE":
                test["ridge_alpha"] = m.alpha_
        test["train_gws"] = ",".join(str(int(g)) for g in sorted(train["gw"].unique()))
        rows.append(test)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["final_gws_available"] = len(final_gws)
    return out


def score_arms(oos: pd.DataFrame, population: str = "ACTIVE") -> dict:
    """Per-arm pooled + per-gameweek metrics, CIs vs FPL, decision metrics."""
    d = oos if population == "ACTIVE" else oos[oos["in_ledger"] == 1]
    res = {"population": population, "pooled": {}, "pooled_final_only": {}, "by_gw": [],
           "vs_fpl": {}, "vs_fpl_final_only": {}}
    fin = d[d["status"] == "FINAL"]
    for arm in ARMS:
        res["pooled"][arm] = metrics(d["actual_points"], d[arm])
        res["pooled_final_only"][arm] = metrics(fin["actual_points"], fin[arm])
    for gw, g in d.groupby("gw"):
        rec = {"gw": int(gw), "status": str(g["status"].iloc[0]), "n": int(len(g)),
               "train_gws": str(g["train_gws"].iloc[0])}
        for arm in ARMS:
            gg = g.dropna(subset=[arm])
            m = metrics(gg["actual_points"], gg[arm])
            if len(gg) >= 10:
                m["top10_overlap"] = top_overlap(gg, arm)
                m["top15_actual_mean"] = round(decision_value(gg, arm), 3)
            rec[arm] = m
        res["by_gw"].append(rec)
    for arm in ARMS[1:]:
        for key, frame in (("vs_fpl", d), ("vs_fpl_final_only", fin)):
            ff = frame.dropna(subset=[arm])
            mf = metrics(ff["actual_points"], ff["FPL"])
            ma = metrics(ff["actual_points"], ff[arm])
            if "mae" not in mf or "mae" not in ma:
                res[key][arm] = {"n": int(len(ff))}
                continue
            res[key][arm] = {
                "n": int(len(ff)),
                "mae_minus_fpl": round(ma["mae"] - mf["mae"], 4),
                "mae_ci90": cluster_boot_delta(ff, arm, "FPL", "mae"),
                "rmse_minus_fpl": round(ma["rmse"] - mf["rmse"], 4),
                "rmse_ci90": cluster_boot_delta(ff, arm, "FPL", "mse"),
                "spearman_minus_fpl": round(ma["spearman"] - mf["spearman"], 4),
            }
    return res


# ── acceptance gate (fixed before results; see docs/FANTASY_V2_MODEL_AND_PRODUCT_REPORT.md) ──
MIN_FINAL_OOS_GWS = 3


def acceptance(score: dict, arm: str) -> dict:
    """GO only if the arm beats FPL out of sample over >= 3 FINAL gameweeks on MAE, or delivers a
    ranking improvement whose CI excludes zero. Anything less keeps FPL as the mean forecast."""
    fin = [r for r in score["by_gw"] if r["status"] == "FINAL"]
    n_final = len(fin)
    v = score["vs_fpl_final_only"].get(arm, {})
    mae_ok = v.get("mae_minus_fpl") is not None and v["mae_minus_fpl"] <= 0
    won = sum(1 for r in fin if r.get(arm, {}).get("mae", 9e9) <= r["FPL"].get("mae", -1))
    rank_ok = all(r.get(arm, {}).get("spearman", -1) > r["FPL"].get("spearman", 9) for r in fin) \
        and n_final >= MIN_FINAL_OOS_GWS
    enough = n_final >= MIN_FINAL_OOS_GWS
    go = enough and (mae_ok or rank_ok)
    why = []
    if not enough:
        why.append(f"only {n_final} FINAL out-of-sample gameweek(s); the rule needs "
                   f"{MIN_FINAL_OOS_GWS}")
    if not mae_ok:
        why.append(f"MAE {v.get('mae_minus_fpl'):+.3f} vs FPL (must be <= 0)"
                   if v.get("mae_minus_fpl") is not None else "no MAE comparison")
    if not rank_ok:
        why.append("no consistent within-gameweek ranking improvement")
    return {"arm": arm, "decision": "GO" if go else "NO-GO", "final_oos_gws": n_final,
            "gameweeks_mae_won": won, "reasons": why}


# ── bias diagnosis of the existing Wowza projection ───────────────────────────────────────────
def wowza_bias_breakdown(panel: pd.DataFrame) -> dict:
    """Where the +1 point of Wowza bias comes from, on the published-projection population."""
    d = prepare(panel)
    d = d[d["status"].isin(["FINAL", "PROVISIONAL"]) & d["wowza_uncond"].notna()].copy()
    if d.empty:
        return {}
    d["err"] = d["wowza_uncond"] - d["actual_points"]
    d["err_fpl"] = d["ep_next"] - d["actual_points"]
    d["played"] = (d["actual_minutes"] > 0).astype(float)
    d["minutes_band"] = d["bucket"].map({"m0": "0 min", "m1_59": "1-59 min", "m60": "60+ min"})
    d["price_band"] = pd.cut(d["price"], [0, 5.0, 6.5, 8.5, 20], labels=["<=5.0", "5.1-6.5",
                                                                        "6.6-8.5", ">8.5"])
    d["proj_band"] = pd.cut(d["wowza_uncond"], [-1, 2, 3.5, 5, 99], labels=["<2", "2-3.5",
                                                                            "3.5-5", ">5"])
    d["fdr_band"] = d["fdr"].round().astype("Int64").astype(str)
    d["venue"] = np.where(d["home"] >= 0.5, "home", "away")
    d["season_minutes_zero"] = np.where(d["minutes"].fillna(0) <= 0, "no PL minutes yet",
                                        "has minutes")
    out = {"n": int(len(d)), "bias": round(float(d["err"].mean()), 3),
           "fpl_bias": round(float(d["err_fpl"].mean()), 3), "by": {}}
    for k in ("position", "minutes_band", "price_band", "proj_band", "fdr_band", "venue",
              "season_minutes_zero", "team", "gw"):
        g = d.groupby(k, observed=True).agg(n=("err", "size"), wowza_bias=("err", "mean"),
                                            fpl_bias=("err_fpl", "mean"),
                                            share_of_total=("err", "sum"))
        g["share_of_total"] = g["share_of_total"] / d["err"].sum()
        out["by"][k] = g.round(3).reset_index().to_dict("records")
    # Minutes channel vs conditional-points channel.
    #   E[proj] - E[actual] = E[p_start*cond - played*cond_actual]
    #   minutes channel     = E[(p_start - played) * cond]       (wrong about WHO plays)
    #   points channel      = E[played * (cond - actual_if_played)] (wrong about points when he plays)
    cond = d["wowza_cond"]
    ps = d["wowza_p_start"]
    minutes_ch = float(((ps - d["played"]) * cond).mean())
    pts_ch = float((d["played"] * cond - d["actual_points"]).mean())
    out["decomposition"] = {
        "minutes_channel": round(minutes_ch, 3),
        "conditional_points_channel": round(pts_ch, 3),
        "mean_p_start": round(float(ps.mean()), 3),
        "actual_play_rate": round(float(d["played"].mean()), 3),
        "actual_60_rate": round(float((d["bucket"] == "m60").mean()), 3),
        "mean_conditional_projection": round(float(cond.mean()), 3),
        "mean_actual_when_played": round(float(d.loc[d["played"] == 1, "actual_points"].mean()), 3),
        "mean_conditional_projection_when_played": round(float(cond[d["played"] == 1].mean()), 3),
        "mean_conditional_projection_60plus": round(float(cond[d["bucket"] == "m60"].mean()), 3),
        "mean_actual_60plus": round(float(d.loc[d["bucket"] == "m60", "actual_points"].mean()), 3),
    }
    return out


# ── minutes mixture ───────────────────────────────────────────────────────────────────────────
MIN_FEATURES = ["chance_next_f", "is_out", "is_doubt", "has_news", "start_rate",
                "min_per_team_game", "min_last_imp", "started_last_imp", "ep_next", "price",
                "n_fix", "log_own"] + [f"pos_{p}" for p in POSITIONS]


class MinutesModel:
    """Multinomial P(0) / P(1-59) / P(60+) from availability and usage, with conditional
    minutes/points per bucket estimated from the training gameweeks."""

    def fit(self, x: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression
        X = _matrix(x, MIN_FEATURES)
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ < 1e-9] = 1.0
        self.m_ = LogisticRegression(C=0.5, max_iter=2000)
        y = x["bucket"].to_numpy()
        self.m_.fit((X - self.mu_) / self.sd_, y)
        self.classes_ = list(self.m_.classes_)
        self.cond_minutes_ = x.groupby("bucket")["actual_minutes"].mean().to_dict()
        sub = x[x["bucket"] == "m1_59"]
        self.sub_pmf_ = {}
        for p in POSITIONS:
            s = sub.loc[sub["position"] == p, "actual_points"].clip(-2, 20).round().astype(int)
            if len(s) < 15:
                s = sub["actual_points"].clip(-2, 20).round().astype(int)
            self.sub_pmf_[p] = s.value_counts(normalize=True).sort_index()
        full = x[x["bucket"] == "m60"]
        self.size_ = {}
        for p in POSITIONS:
            s = (full.loc[full["position"] == p, "actual_points"] - 2).clip(lower=0)
            if len(s) < 20:
                s = (full["actual_points"] - 2).clip(lower=0)
            mu, var = float(s.mean()), float(s.var())
            self.size_[p] = float(np.clip(mu * mu / max(var - mu, 1e-6), 0.3, 50.0))
        self.fit_dispersion(x)
        return self

    SIZE_GRID = (0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 20.0)

    def fit_dispersion(self, x: pd.DataFrame, mean_col: str = "ep_next"):
        """Choose the 60+ minutes dispersion per position by LIKELIHOOD of the full mixture on the
        training gameweeks, using each player's own mean.

        The method-of-moments value above pools players with very different means, so it reads
        between-player spread as within-player noise and makes every tail too fat (P(15+) came
        out at 4x its observed rate on the first pass). Fitting the mixture with per-player means
        attributes that spread to the means, where it belongs."""
        if mean_col not in x.columns or x.empty:
            return self
        P = self.predict_proba(x)
        y = x["actual_points"].clip(-2, MAX_PTS).round().astype(int).to_numpy() + 2
        for pos in POSITIONS:
            idx = np.where(x["position"].to_numpy() == pos)[0]
            if len(idx) < 40:
                continue
            best = None
            for size in self.SIZE_GRID:
                self.size_[pos] = size
                ll = 0.0
                for i in idx:
                    pmf = point_distribution(float(x[mean_col].iloc[i]), P.iloc[i].to_dict(), pos, self)
                    ll += math.log(max(pmf[y[i]], 1e-9))
                if best is None or ll > best[0]:
                    best = (ll, size)
            self.size_[pos] = best[1]
        return self

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        X = (_matrix(x, MIN_FEATURES) - self.mu_) / self.sd_
        P = self.m_.predict_proba(X)
        out = pd.DataFrame(P, columns=self.classes_, index=x.index)
        for b in ("m0", "m1_59", "m60"):
            if b not in out:
                out[b] = 0.0
        # A blank gameweek is certain zero minutes, whatever the usage history says.
        blank = x["n_fix"].fillna(1).to_numpy() == 0
        out.loc[blank, ["m1_59", "m60"]] = 0.0
        out.loc[blank, "m0"] = 1.0
        return out[["m0", "m1_59", "m60"]]


def log_loss(y, P: pd.DataFrame) -> float:
    idx = {c: i for i, c in enumerate(P.columns)}
    p = np.array([P.iloc[i, idx[v]] for i, v in enumerate(y)])
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


def heuristic_minutes(x: pd.DataFrame) -> pd.DataFrame:
    """Baseline: P(play) = FPL chance x usage rate, P(60+) = chance x start rate."""
    ch = x["chance_next_f"].fillna(0).clip(0, 1)
    play = (ch * (x["min_per_team_game"].fillna(0) > 0).astype(float)
            * np.maximum(x["start_rate"].fillna(0), 0.35)).clip(0.01, 0.98)
    p60 = (ch * x["start_rate"].fillna(0) * 0.92).clip(0.005, 0.97)
    p60 = np.minimum(p60, play - 0.005)
    out = pd.DataFrame({"m0": 1 - play, "m1_59": play - p60, "m60": p60}, index=x.index)
    blank = x["n_fix"].fillna(1).to_numpy() == 0
    out.loc[blank, ["m1_59", "m60"]] = 0.001
    out.loc[blank, "m0"] = 0.998
    return out


# ── distribution of points ───────────────────────────────────────────────────────────────────
MAX_PTS = 30


def _negbin_pmf(mean: float, size: float, kmax: int) -> np.ndarray:
    from scipy.stats import nbinom
    mean = max(mean, 1e-3)
    p = size / (size + mean)
    k = np.arange(kmax + 1)
    v = nbinom.pmf(k, size, p)
    v[-1] += max(0.0, 1.0 - v.sum())
    return v


def point_distribution(mu: float, probs: dict, position: str, mm: MinutesModel) -> np.ndarray:
    """pmf over points -2..MAX_PTS for one player, mixture over minutes buckets, with the 60+
    component's mean solved so the mixture mean equals `mu` (the headline expectation)."""
    off = 2
    pmf = np.zeros(MAX_PTS + off + 1)
    p0, p1, p60 = probs["m0"], probs["m1_59"], probs["m60"]
    pmf[0 + off] += p0
    sub = mm.sub_pmf_.get(position)
    m1 = 0.0
    if sub is not None and p1 > 0:
        for k, v in sub.items():
            k = int(np.clip(k, -2, MAX_PTS))
            pmf[k + off] += p1 * v
            m1 += k * v
    if p60 > 1e-6:
        m60 = (mu - p1 * m1) / p60
        m60 = float(np.clip(m60, 1.5, 16.0))
        nb = _negbin_pmf(max(m60 - 2.0, 0.05), mm.size_.get(position, 2.0), MAX_PTS - 2)
        for k, v in enumerate(nb):
            pmf[k + 2 + off] += p60 * v
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def dist_summary(pmf: np.ndarray) -> dict:
    off = 2
    pts = np.arange(len(pmf)) - off
    cdf = np.cumsum(pmf)

    def q(a):
        return float(pts[np.searchsorted(cdf, a - 1e-9)])
    mean = float((pts * pmf).sum())
    sd = float(np.sqrt(((pts - mean) ** 2 * pmf).sum()))
    return {"dist_mean": round(mean, 3), "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
            "p90": q(0.90), "sd": round(sd, 3),
            "p_blank": round(float(pmf[pts <= 2].sum()), 4),
            "p6": round(float(pmf[pts >= 6].sum()), 4),
            "p10": round(float(pmf[pts >= 10].sum()), 4),
            "p15": round(float(pmf[pts >= 15].sum()), 4)}


def event_probs(x: pd.DataFrame, P: pd.DataFrame, cond_minutes: dict) -> pd.DataFrame:
    """Goal / assist / clean-sheet / bonus quantities from role x expected minutes x fixture."""
    em = P["m1_59"] * cond_minutes.get("m1_59", 25.0) + P["m60"] * cond_minutes.get("m60", 88.0)
    fx = x["attack_fixture"].fillna(1.0).clip(0.4, 2.5)
    nfx = x["n_fix"].fillna(1).clip(lower=0)
    lam_g = (x["xg90"] * em / 90.0 * fx).fillna(0)
    lam_a = (x["xa90"] * em / 90.0 * fx).fillna(0)
    out = pd.DataFrame(index=x.index)
    out["exp_minutes"] = em.round(1)
    out["p_start"] = (P["m60"] + 0.35 * P["m1_59"]).clip(0, 1).round(3)
    out["p_play"] = (1 - P["m0"]).round(3)
    out["goal_prob"] = (1 - np.exp(-lam_g)).round(3)
    out["assist_prob"] = (1 - np.exp(-lam_a)).round(3)
    cs_team = x["cs_prob"].fillna(0)
    out["team_cs_prob"] = cs_team.round(3)
    out["cs_prob"] = np.where(x["position"].isin(["GKP", "DEF", "MID"]),
                              (cs_team * P["m60"]).round(3), np.nan)
    out["exp_bonus"] = (x["bonus_per_start"].fillna(0.2) * P["m60"] * nfx.clip(upper=2)).round(3)
    out["team_goal_exp"] = x["lam_for"].round(2)
    return out


def evaluate_distribution(oos_rows: pd.DataFrame, train_for: dict, mean_col: str = "FPL") -> dict:
    """Calibration of the minutes mixture and the point distribution, out of sample."""
    rows = []
    for gw, g in oos_rows.groupby("gw"):
        mm = train_for[gw]
        x = g
        P = mm.predict_proba(x)
        H = heuristic_minutes(x)
        ev = event_probs(x, P, mm.cond_minutes_)
        dist = [dist_summary(point_distribution(float(mu), P.loc[i].to_dict(), pos, mm))
                for i, mu, pos in zip(x.index, x[mean_col], x["position"])]
        D = pd.DataFrame(dist, index=x.index)
        rows.append(pd.concat([x[["gw", "status", "element", "position", "bucket",
                                  "actual_points", "actual_minutes", "actual_goals",
                                  "actual_assists", "actual_clean_sheets", "actual_bonus",
                                  "in_ledger", "wowza_p_start", "start_rate"]],
                               P.add_prefix("P_"), H.add_prefix("H_"), ev, D], axis=1))
    if not rows:
        return {}
    d = pd.concat(rows)
    y = d["bucket"].to_list()
    res = {"n": int(len(d)), "gws": sorted(int(g) for g in d["gw"].unique()),
           "mean_column": mean_col}
    Pm = d[["P_m0", "P_m1_59", "P_m60"]].rename(columns=lambda c: c[2:])
    Ph = d[["H_m0", "H_m1_59", "H_m60"]].rename(columns=lambda c: c[2:])
    res["minutes_log_loss"] = {"mixture_model": round(log_loss(y, Pm), 4),
                               "heuristic_chance_x_rate": round(log_loss(y, Ph), 4)}
    played = (d["actual_minutes"] > 0).astype(float)
    started60 = (d["bucket"] == "m60").astype(float)
    res["brier_play"] = {"mixture_model": round(float(((1 - d["P_m0"]) - played).pow(2).mean()), 4),
                         "heuristic": round(float(((1 - d["H_m0"]) - played).pow(2).mean()), 4)}
    res["brier_60plus"] = {"mixture_model": round(float((d["P_m60"] - started60).pow(2).mean()), 4),
                           "heuristic": round(float((d["H_m60"] - started60).pow(2).mean()), 4)}
    led = d[d["in_ledger"] == 1].dropna(subset=["wowza_p_start"])
    if len(led) >= 30:
        pl = (led["actual_minutes"] > 0).astype(float)
        res["brier_play_vs_wowza_p_start"] = {
            "n": int(len(led)),
            "mixture_model": round(float(((1 - led["P_m0"]) - pl).pow(2).mean()), 4),
            "wowza_p_start": round(float((led["wowza_p_start"] - pl).pow(2).mean()), 4)}
    a = d["actual_points"]
    res["interval_coverage"] = {
        "p25_p75_inclusive": round(float(((a >= d["p25"]) & (a <= d["p75"])).mean()), 4),
        "above_p90": round(float((a > d["p90"]).mean()), 4),
        "at_or_below_median": round(float((a <= d["median"]).mean()), 4)}
    for thr, col in ((6, "p6"), (10, "p10"), (15, "p15")):
        hit = (a >= thr).astype(float)
        base = float(hit.mean())
        res[f"brier_{col}"] = {"model": round(float((d[col] - hit).pow(2).mean()), 4),
                               "climatology": round(float((base - hit).pow(2).mean()), 4),
                               "mean_pred": round(float(d[col].mean()), 4),
                               "observed": round(base, 4)}
    g_hit = (d["actual_goals"] > 0).astype(float)
    a_hit = (d["actual_assists"] > 0).astype(float)
    res["brier_goal"] = {"model": round(float((d["goal_prob"] - g_hit).pow(2).mean()), 4),
                         "climatology": round(float((g_hit.mean() - g_hit).pow(2).mean()), 4),
                         "mean_pred": round(float(d["goal_prob"].mean()), 4),
                         "observed": round(float(g_hit.mean()), 4)}
    res["brier_assist"] = {"model": round(float((d["assist_prob"] - a_hit).pow(2).mean()), 4),
                           "climatology": round(float((a_hit.mean() - a_hit).pow(2).mean()), 4),
                           "mean_pred": round(float(d["assist_prob"].mean()), 4),
                           "observed": round(float(a_hit.mean()), 4)}
    dd = d[d["position"].isin(["GKP", "DEF"])]
    if len(dd):
        cs_hit = (dd["actual_clean_sheets"] > 0).astype(float)
        res["brier_cs_def_gk"] = {"model": round(float((dd["cs_prob"] - cs_hit).pow(2).mean()), 4),
                                  "climatology": round(float((cs_hit.mean() - cs_hit).pow(2).mean()), 4),
                                  "mean_pred": round(float(dd["cs_prob"].mean()), 4),
                                  "observed": round(float(cs_hit.mean()), 4)}
    # RMSE, not MAE: bonus is zero for most players, so MAE rewards predicting zero for everyone.
    res["bonus_rmse"] = {
        "model": round(float(np.sqrt(((d["exp_bonus"] - d["actual_bonus"]) ** 2).mean())), 4),
        "constant_mean": round(float(d["actual_bonus"].std(ddof=0)), 4),
        "mean_pred": round(float(d["exp_bonus"].mean()), 4),
        "observed": round(float(d["actual_bonus"].mean()), 4)}
    # reliability of P(6+) in five bins
    d["bin"] = pd.qcut(d["p6"].rank(method="first"), 5, labels=False)
    rel = d.groupby("bin").agg(pred=("p6", "mean"), obs=("actual_points", lambda s: (s >= 6).mean()),
                               n=("p6", "size")).round(3)
    res["reliability_p6"] = rel.reset_index(drop=True).to_dict("records")
    return res


def permutation_importance_groups(train: pd.DataFrame, test: pd.DataFrame, arm: str = "HGB",
                                  n_rep: int = 5, seed: int = 3) -> list:
    """Out-of-sample RMSE increase when one feature GROUP is shuffled (positive = it helps)."""
    r = (train["actual_points"] - train["ep_next"]).to_numpy(float)
    m = make(arm)
    m.fit(train, r, groups=train["gw"].to_numpy())
    y = test["actual_points"].to_numpy(float)
    base = float(np.sqrt(np.mean((test["ep_next"].to_numpy(float) + m.predict(test) - y) ** 2)))
    rng = np.random.default_rng(seed)
    out = []
    groups = LINEAR_GROUPS if arm == "LINEAR" else {k: v for k, v in GROUPS.items()
                                                     if k != "baseline"}
    for g, cols in groups.items():
        incs = []
        for _ in range(n_rep):
            t = test.copy()
            perm = rng.permutation(len(t))
            for c in cols:
                t[c] = t[c].to_numpy()[perm]
            p = t["ep_next"].to_numpy(float) + m.predict(t)
            incs.append(float(np.sqrt(np.mean((p - y) ** 2))) - base)
        out.append({"group": g, "rmse_increase": round(float(np.mean(incs)), 4)})
    return sorted(out, key=lambda r: -r["rmse_increase"])
