"""The six candidates. Each one: fit(frames) on the frames protocol.fold_frames / final_frames give
it, predict(df) -> array of P(yes) with NaN where it has no honest probability, meta() for the run
manifest. No candidate invents a value for a missing input:

    champion_recipe  src.model.train_repaired + missing indicators + 1.1 train-only imputer
                     (persisted in the payload); a feature with no training value is dropped
    logistic         train-only median imputer + a missing indicator for EVERY feature, pickled
                     with the model; columns with no training value are dropped
    hgb              HistGradientBoosting, NaN-native - nothing filled at all
    dixon_coles      per-league score distribution; unseen club or league -> NaN (no 1.0 strength)
    market_only      de-vigged two-sided price; one-sided or flagged pair -> NaN
    model_market     logistic stack of [logit hgb, logit market], weights fitted on VALIDATION only

Frames: 'train' (whole training window), 'fit' (train minus validation), 'validation', 'test'.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger(__name__)

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _X(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Numeric feature matrix; a configured column the frame lacks is an error, never created."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"feature column(s) not in the frame: {missing}")
    return df[features].apply(pd.to_numeric, errors="coerce").astype(float)


def _weights(df: pd.DataFrame, how: str | None):
    if not how:
        return None
    if how != "derived":
        raise ValueError(f"weights={how!r}: only 'derived' (src.standard_repair.derived_weights) exists")
    from src.standard_repair import derived_weights
    return derived_weights(df).to_numpy(dtype=float)


def trainable_features(df: pd.DataFrame, features: list) -> tuple[list, list]:
    """(kept, dropped): a column with NO value in the training rows cannot be learned from and is
    dropped and recorded - the champion recipe's rule (src.imputer all_missing='drop'). It is
    never filled. Early Standard folds (2021-2022) have no xG and no half-time history at all;
    sklearn 1.9's HistGradientBoosting binning raises 'window shape cannot be larger than input
    array shape' on any column with no non-missing value (11 walk-forward fits, run 42512834177)."""
    X = _X(df, features)
    dropped = [c for c in features if not X[c].notna().any()]
    return [c for c in features if c not in dropped], dropped


def _y(df: pd.DataFrame, target: str) -> np.ndarray:
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ValueError(f"{target}: unlabelled rows reached a fit")
    return y


class NotFitted(Exception):
    """A candidate that genuinely cannot be fitted on this fold's data. The runner records it as
    NOT_FITTED with this reason (never a crash, never a silent skip)."""


class Candidate:
    name = "base"
    deployable = True

    def __init__(self, params: dict, market: dict, features: list, track: str, market_name: str):
        self.params = dict(params or {})
        self.market = market
        self.market_name = market_name
        self.target = market["target"]
        self.features = list(features)
        self.track = track
        self.notes: list = []
        self.fitted = False

    def fit(self, frames: dict) -> "Candidate":
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def meta(self) -> dict:
        return {"candidate": self.name, "fitted": self.fitted, "deployable": self.deployable,
                "notes": list(self.notes)}


# ── champion recipe ─────────────────────────────────────────────────────────────────────────

#: market -> pipeline label (src.model.RETRAIN_RECIPE key) per track
def champion_label(market_name: str, track: str) -> str:
    if market_name == "over25":
        return "standard" if track == "standard" else "newformat"
    return market_name


class ChampionRecipe(Candidate):
    """src.model.train_repaired exactly as the scheduled retrain calls it, on ONE track, with the
    fold's test block as the recipe's own holdout (holdout_start = first test date), so its base /
    calibration / meta blocks all lie inside the training window. The recipe reads the test labels
    only for the metrics it logs; nothing is fitted on them (its meta says so and a test checks it)."""
    name = "champion_recipe"

    def fit(self, frames: dict) -> "ChampionRecipe":
        from src.imputer import MISSING_SUFFIX
        from src.model import RETRAIN_RECIPE, build_payload, train_repaired
        train, test = frames["train"], frames["test"]
        df = pd.concat([train, test], ignore_index=True)
        feats = self.features + [f"{c}{MISSING_SUFFIX}" for c in self.features]
        w = _weights(df, self.params.get("weights"))
        recipe = RETRAIN_RECIPE[champion_label(self.market_name, self.track)]
        res = train_repaired(df, target=self.target,
                             sample_weight=pd.Series(w, index=df.index) if w is not None else None,
                             feature_cols=feats, recipe=recipe,
                             holdout_start=pd.to_datetime(test["date"]).min(),
                             imputer_options={"all_missing": "drop"},
                             provenance={"track": self.track, "harness": "phase2_retrain"},
                             calibration_input=self.params.get("calibration_input", "probability"))
        self.payload = build_payload(res, self.target)
        dropped = list(self.payload.get("dropped_all_missing_features") or [])
        if dropped:
            self.notes.append(f"dropped (no training value): {dropped}")
        self.recipe = recipe
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        from src.model import predict_proba
        return np.asarray(predict_proba(df, payload=self.payload), dtype=float)

    def meta(self) -> dict:
        m = super().meta()
        if self.fitted:
            m.update(recipe=self.recipe, imputer_version=self.payload.get("imputer_version"),
                     calibration_input=self.payload.get("calibration_input", "probability"),
                     n_features=len(self.payload.get("feature_cols") or []),
                     blocks=self.payload.get("blocks"))
        return m


# ── logistic ────────────────────────────────────────────────────────────────────────────────

class Logistic(Candidate):
    name = "logistic"

    def fit(self, frames: dict) -> "Logistic":
        from sklearn.impute import MissingIndicator, SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import StandardScaler
        tr = frames["train"]
        X = _X(tr, self.features)
        empty = [c for c in self.features if X[c].isna().all()]
        self.used = [c for c in self.features if c not in empty]
        if empty:
            self.notes.append(f"dropped (no training value): {empty}")
        if not self.used:
            raise ValueError("no feature has a training value")
        self.pipe = Pipeline([
            ("inputs", FeatureUnion([
                ("imputed", SimpleImputer(strategy="median")),          # fitted on TRAIN only
                ("missing", MissingIndicator(features="all"))])),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=float(self.params["C"]), max_iter=int(self.params["max_iter"])))])
        self.pipe.fit(X[self.used], _y(tr, self.target), clf__sample_weight=_weights(tr, self.params.get("weights")))
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(_X(df, self.used))[:, 1]


# ── HistGradientBoosting (NaN-native) ───────────────────────────────────────────────────────

class HGB(Candidate):
    name = "hgb"
    PARAMS = ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
              "random_state")

    def _fit_on(self, df: pd.DataFrame):
        """Fit on the columns that have at least one training value; the dropped ones are recorded
        in the notes and `predict` scores on exactly the fitted columns."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        kept, dropped = trainable_features(df, self.features)
        if not kept:
            raise NotFitted("no feature column has a training value")
        if dropped:
            self.notes.append(f"dropped (no training value): {dropped}")
        self.fit_features = kept
        y = _y(df, self.target)
        if len(set(y)) < 2:
            raise NotFitted(f"only one outcome class in the {len(y)} training rows")
        kw = {k: self.params[k] for k in self.PARAMS}
        m = HistGradientBoostingClassifier(early_stopping=False, **kw)
        m.fit(_X(df, kept), y, sample_weight=_weights(df, self.params.get("weights")))
        return m

    def fit(self, frames: dict) -> "HGB":
        self.model = self._fit_on(frames["train"])
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(_X(df, self.fit_features))[:, 1]


# ── Dixon-Coles ─────────────────────────────────────────────────────────────────────────────

def market_prob_from_matrix(m: np.ndarray, market: dict) -> float:
    k = np.arange(m.shape[0])
    i, j = np.meshgrid(k, k, indexing="ij")
    if market["kind"] == "total":
        return float(m[(i + j) > float(market["line"])].sum())
    if market["kind"] == "btts":
        return float(m[1:, 1:].sum())
    raise ValueError(f"no Dixon-Coles mapping for market kind {market['kind']!r}")


class DixonColesCandidate(Candidate):
    """Strengths per (league, club) from the training window's last `fit_window_days`, shrunk toward
    the league mean by `shrink_prior_matches`; rho by maximum likelihood on the same rows (the
    Dixon-Coles tau keeps the matrix normalised, so rho's likelihood is the sum of log tau)."""
    name = "dixon_coles"

    def _goals_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        hg, ag = self.market["goals"]
        d = df[["date", "league", "home_team", "away_team"]].copy()
        d["home_goals"] = pd.to_numeric(df[hg], errors="coerce") if hg in df.columns else np.nan
        d["away_goals"] = pd.to_numeric(df[ag], errors="coerce") if ag in df.columns else np.nan
        return d

    def fit(self, frames: dict) -> "DixonColesCandidate":
        from scipy.optimize import minimize_scalar
        from src.goal_distribution import DixonColes, _tau
        tr = self._goals_frame(frames["train"])
        end = pd.to_datetime(tr["date"]).max()
        win = int(self.params["fit_window_days"])
        tr = tr[pd.to_datetime(tr["date"]) > end - pd.Timedelta(days=win)]
        tr = tr.dropna(subset=["home_goals", "away_goals"])
        self.dc = DixonColes(prior=float(self.params["shrink_prior_matches"]),
                             max_goals=int(self.params["max_goals"])).fit(tr)
        lam = np.array([self._lambdas(r.league, r.home_team, r.away_team) for r in tr.itertuples(index=False)])
        ok = np.isfinite(lam).all(axis=1) if len(lam) else np.zeros(0, bool)
        hi, ai = tr["home_goals"].to_numpy()[ok].astype(int), tr["away_goals"].to_numpy()[ok].astype(int)
        lh, la = lam[ok, 0], lam[ok, 1]
        low = (hi <= 1) & (ai <= 1)
        lo_b, hi_b = self.params["rho_bounds"]

        def nll(rho):                       # _tau broadcasts over aligned per-row arrays
            return -float(np.sum(np.log(_tau(hi[low], ai[low], lh[low], la[low], rho))))
        if low.sum() == 0:
            raise ValueError("no low-score training rows: rho cannot be estimated")
        self.dc.rho = float(minimize_scalar(nll, bounds=(lo_b, hi_b), method="bounded").x)
        self.fit_rows = int(len(tr))
        self.notes.append(f"rho={self.dc.rho:.4f} on {int(ok.sum())} rows (window {win} days)")
        self.fitted = True
        return self

    def _lambdas(self, lg, home, away) -> tuple[float, float]:
        """NaN unless the league AND both clubs have training matches in the window."""
        mu = self.dc.league_mu.get(lg)
        keys = [(lg, home), (lg, away)]
        if mu is None or any(k not in self.dc.attack for k in keys):
            return float("nan"), float("nan")
        lh = mu[0] * self.dc.attack[(lg, home)] * self.dc.defence[(lg, away)]
        la = mu[1] * self.dc.attack[(lg, away)] * self.dc.defence[(lg, home)]
        return float(lh), float(la)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        for n, r in enumerate(df[["league", "home_team", "away_team"]].itertuples(index=False)):
            lh, la = self._lambdas(r.league, r.home_team, r.away_team)
            if np.isfinite(lh) and np.isfinite(la) and lh > 0 and la > 0:
                out[n] = market_prob_from_matrix(self.dc.matrix(lh, la), self.market)
        return out


# ── market only ─────────────────────────────────────────────────────────────────────────────

def devig_yes(df: pd.DataFrame, market: dict) -> np.ndarray:
    """Power de-vig of a two-sided pair -> P(yes). NaN where the pair is missing, one-sided,
    not > 1, or flagged as not a price (e.g. ou25_pair_is_price False)."""
    from src.market_anchor import power_devig
    pr = market.get("price")
    out = np.full(len(df), np.nan)
    if not pr or not pr.get("yes") or not pr.get("no"):
        return out
    if pr["yes"] not in df.columns or pr["no"] not in df.columns:
        return out
    yes = pd.to_numeric(df[pr["yes"]], errors="coerce").to_numpy(dtype=float)
    no = pd.to_numeric(df[pr["no"]], errors="coerce").to_numpy(dtype=float)
    ok = (yes > 1) & (no > 1)
    flag = pr.get("pair_valid_flag")
    if flag and flag in df.columns:
        ok &= df[flag].astype("boolean").fillna(False).to_numpy(dtype=bool)
    for n in np.flatnonzero(ok):
        p = power_devig([yes[n], no[n]])
        if len(p) == 2:
            out[n] = p[0]
    return out


class MarketOnly(Candidate):
    name = "market_only"

    def __init__(self, *a, price_class: str = "", **k):
        super().__init__(*a, **k)
        self.price_class = price_class
        self.deployable = "CLOSING" not in price_class

    def fit(self, frames: dict) -> "MarketOnly":
        pr = self.market.get("price")
        if not pr or not pr.get("no"):
            self.notes.append("no two-sided price for this market: no probability")
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return devig_yes(df, self.market)

    def meta(self) -> dict:
        return super().meta() | {"price_class": self.price_class}


# ── model + market ──────────────────────────────────────────────────────────────────────────

class ModelMarket(Candidate):
    """Base HGB fitted on 'fit' (training window minus validation); the stack's two weights and
    intercept fitted on 'validation' rows that have a two-sided price; nothing else."""
    name = "model_market"

    def __init__(self, *a, base_params: dict | None = None, price_class: str = "", **k):
        super().__init__(*a, **k)
        self.base_params = dict(base_params or {})
        self.price_class = price_class
        self.deployable = "CLOSING" not in price_class

    def fit(self, frames: dict) -> "ModelMarket":
        from sklearn.linear_model import LogisticRegression
        self.stack = None
        val = frames["validation"]
        pm = devig_yes(val, self.market)
        ok = np.isfinite(pm)
        y = _y(val, self.target)
        if not ok.any() or len(set(y[ok])) < 2:
            self.notes.append("NOT_FITTED: validation block has no two-sided price with both outcomes")
            self.fitted = True
            return self
        self.base = HGB(self.base_params, self.market, self.features, self.track, self.market_name)
        self.base.model = self.base._fit_on(frames["fit"])
        self.notes += [f"base: {n}" for n in self.base.notes]
        pb = self.base.predict(val)
        Z = np.column_stack([_logit(pb[ok]), _logit(pm[ok])])
        self.stack = LogisticRegression(C=1e6, max_iter=1000).fit(Z, y[ok])
        self.n_validation = int(ok.sum())
        self.notes.append(f"stack fitted on {self.n_validation} validation rows; "
                          f"coef={np.round(self.stack.coef_[0], 4).tolist()}")
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        if self.stack is None:
            return out
        pm = devig_yes(df, self.market)
        ok = np.isfinite(pm)
        if ok.any():
            pb = self.base.predict(df[ok])
            out[ok] = self.stack.predict_proba(np.column_stack([_logit(pb), _logit(pm[ok])]))[:, 1]
        return out

    def meta(self) -> dict:
        return super().meta() | {"price_class": self.price_class}


def build(name: str, cfg: dict, track: str, market_name: str, features: list,
          overrides: dict | None = None) -> Candidate:
    """`overrides`: a parameter run's changed parameters (protocol.run_params), on top of the
    candidate's config block; empty for every ordinary candidate."""
    c = {**cfg["candidates"][name], **(overrides or {})}
    market = cfg["markets"][market_name]
    price_class = (cfg.get("price_class") or {}).get("loader_served", "")
    if name == "champion_recipe":
        return ChampionRecipe(c, market, features, track, market_name)
    if name == "logistic":
        return Logistic(c, market, features, track, market_name)
    if name == "hgb":
        return HGB(c, market, features, track, market_name)
    if name == "dixon_coles":
        return DixonColesCandidate(c, market, features, track, market_name)
    if name == "market_only":
        return MarketOnly(c, market, features, track, market_name, price_class=price_class)
    if name == "model_market":
        return ModelMarket(c, market, features, track, market_name,
                           base_params=cfg["candidates"][c["base"]], price_class=price_class)
    raise ValueError(f"unknown candidate {name!r}")
