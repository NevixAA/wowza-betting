"""R2-B repaired CHALLENGERS: train + evaluate on the untouched chronological holdout. RESEARCH ONLY.

    python scripts/train_repaired_challengers.py --task nf_ou25 --group core --out _out
    python scripts/train_repaired_challengers.py --task side_btts --group regimes --track standard --out _out
    python scripts/train_repaired_challengers.py --task std_ou25 --group coldstart --out _out
    python scripts/train_repaired_challengers.py --task identity --out _out

Run by Pro `.github/workflows/pro_r2_repaired_challengers.yml` on GitHub Actions — never on the
owner's laptop. Nothing here writes models/, output/, predict.yml, thresholds or any serving file;
every artifact goes to --out. No threshold is tuned and no payload is promoted.

ONE ARCHITECTURE (src.model.train_repaired, recipes nf_ / side_ / ht_v9_repaired):
  history   v9's own loader (COVID seasons excluded as mode_train does), rows strictly before the
            cutoff (default: today UTC);
  identity  src.identity_repair with the committed evidence-backed alias map
            (config/standard_ou25_identity_aliases.json, which already carries the NF leagues,
            e.g. China Super League Shandong Luneng -> Shandong Taishan) + same-day derivation;
            the MERGED map is frozen and reported;
  features  src.features_repaired (one train/serve function, league-scoped, strictly as-of,
            same-day league aggregates excluded; NF home_advantage is the rolling training rule
            and no price enters - the Nov-1 NF parity semantics hold by construction);
            HT adds src.ht_features run PER LEAGUE (its team accumulators are name-keyed);
  blocks    holdout = every row on/after HOLDOUT_START (the newest ~20% of the task's rows, cut on
            a whole date), identical for every arm; earlier rows -> base 80% / calibration 10% /
            meta 10%, whole-date cuts, disjointness asserted;
  imputer   1.1, fitted on BASE only, persisted; features with no training values are DROPPED and
            recorded (never a silent 0);
  weights   `_sample_weight` travels with the row; recency = the frozen src.recency candidates
            (PRIMARY season_age fixed before evaluation).

CHAMPION = the legacy recipe as it trains today (pipeline.mode_train: legacy feature builder, the
serving payload's feature list, the date-derived recency weights, legacy src.model.train), trained on
the same pre-holdout rows. Its meta-logistic is fitted ON the holdout (bug 6), so its holdout
number is optimistic: CHAMPION_BASEMEAN (mean of its calibrated base models, no meta) is reported
alongside as the uncontaminated legacy reference. Rows are matched on canonical
(date, league, home, away) so every arm is scored on the IDENTICAL fixtures.

Metrics: log loss, Brier, ECE (10-bin), calibration slope/intercept first; AUC secondary; deltas
with a matchday block bootstrap 95% CI. Market metrics ONLY on genuine two-sided pre-kickoff
ENTRY prices (never synthetic side prices, never closing, never one-sided; NF history's invalid
non-full-match lines and the summer 2026 window excluded).
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import config  # noqa: E402
from src import model as M  # noqa: E402
from src import recency as R  # noqa: E402
from src import standard_repair as SR  # noqa: E402

log = logging.getLogger("r2b")
ALIAS_MAP = ROOT / "config" / "standard_ou25_identity_aliases.json"
LIVE_FROM = "2026-09-01"            # August 2026 had a known production bug
HOLDOUT_FRAC = 0.80
RNG_SEED = 20261101
TASKS = ("nf_ou25", "side_btts", "side_over15", "side_over35", "ht_over05", "ht_over15",
         "std_ou25", "identity")
GROUPS = ("core", "regimes", "coldstart")
TRACK_LEAGUES = {"standard": config.STANDARD_FORMAT_LEAGUES, "newformat": config.NEW_FORMAT_LEAGUES}
SERVING_FILE = {"nf_ou25": config.MODEL_FILE_NEWFORMAT, "btts": config.MODEL_FILE_BTTS,
                "over15": config.MODEL_FILE_OVER15, "over35": config.MODEL_FILE_OVER35,
                "ht_over05": config.HT_MODEL_FILE_05, "ht_over15": config.HT_MODEL_FILE_15}
RECIPE = {"nf_ou25": "nf_v9_repaired", "side": "side_v9_repaired", "ht": "ht_v9_repaired",
          "std_ou25": "v9_repaired"}
COLD_FEATS = ("home_advantage", "away_home_adv_factor", "combined_sot_ratio", "combined_ht_goals_avg")
NF_SUMMER = ("2026-06-01", "2026-08-23")      # invalid NF price window (measured 2026-11-01)


# ── data ────────────────────────────────────────────────────────────────────────────────────

def load_frames(cutoff: pd.Timestamp, need_modern: bool) -> dict:
    """Full history (archive on, as production), identity-repaired; optional modern-only copy
    (WOWZA_USE_BACKTEST_ARCHIVE=0) to tag each canonical fixture MODERN or ARCHIVE."""
    from src.data_loader import load_all_matches
    from src.identity_repair import canonicalise_and_dedupe
    raw = SR.load_history()
    raw = raw[pd.to_datetime(raw["date"], errors="coerce").dt.normalize() < cutoff]
    hist, idrep = SR.prepare_history(raw, "v9_repaired", str(ALIAS_MAP), True)
    frozen = {lg: dict(m) for lg, m in (idrep.get("aliases") or {}).items()}
    hist = hist.copy()
    hist["_source"] = "UNTAGGED"
    if need_modern:
        old = os.environ.get("WOWZA_USE_BACKTEST_ARCHIVE")
        os.environ["WOWZA_USE_BACKTEST_ARCHIVE"] = "0"
        try:
            mod = load_all_matches(force=True)
        finally:
            if old is None:
                os.environ.pop("WOWZA_USE_BACKTEST_ARCHIVE", None)
            else:
                os.environ["WOWZA_USE_BACKTEST_ARCHIVE"] = old
        if config.EXCLUDE_COVID_SEASONS:
            mod = mod[~config.covid_season_mask(mod)]
        mod = mod[pd.to_datetime(mod["date"], errors="coerce").dt.normalize() < cutoff]
        mod, _ = canonicalise_and_dedupe(mod, frozen, derive=False)
        keys = set(_keys(mod))
        hist["_source"] = np.where([k in keys for k in _keys(hist)], "MODERN", "ARCHIVE")
    return {"raw": raw, "hist": hist, "frozen_map": frozen,
            "identity_report": {k: v for k, v in idrep.items() if k not in ("aliases", "quarantine_rows")}}


def _keys(df: pd.DataFrame) -> list:
    d = pd.to_datetime(df["date"]).dt.normalize()
    return list(zip(d, df["league"].astype(str), df["home_team"].astype(str), df["away_team"].astype(str)))


def _canon_keys(df: pd.DataFrame, frozen: dict) -> list:
    d = pd.to_datetime(df["date"]).dt.normalize()
    lg = df["league"].astype(str)
    h = [frozen.get(l, {}).get(t, t) for l, t in zip(lg, df["home_team"].astype(str))]
    a = [frozen.get(l, {}).get(t, t) for l, t in zip(lg, df["away_team"].astype(str))]
    return list(zip(d, lg, h, a))


def ht_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """src.ht_features PER LEAGUE: its team accumulators are keyed by club name only, so running
    it over all leagues at once would let a same-named club in another competition leak in."""
    from src.ht_features import build_ht_features
    parts = [build_ht_features(g) for _, g in feat.groupby("league", sort=False)]
    return pd.concat(parts, ignore_index=True)


def _serving_cols(key: str) -> list:
    with open(SERVING_FILE[key], "rb") as f:
        return list(pickle.load(f)["feature_cols"])


# ── task definition ─────────────────────────────────────────────────────────────────────────

def task_spec(task: str) -> dict:
    if task == "nf_ou25":
        return {"target": "over25", "recipe": RECIPE["nf_ou25"], "serving": "nf_ou25",
                "champ_track": "newformat", "tracks": ["newformat"], "kind": "ou"}
    if task.startswith("side_"):
        t = task.split("_", 1)[1]
        return {"target": t, "recipe": RECIPE["side"], "serving": t, "champ_track": "pooled",
                "tracks": ["pooled", "standard", "newformat"], "kind": "side"}
    if task.startswith("ht_"):
        return {"target": task, "recipe": RECIPE["ht"], "serving": task, "champ_track": "pooled",
                "tracks": ["pooled"], "kind": "ht"}
    if task == "std_ou25":
        return {"target": "over25", "recipe": RECIPE["std_ou25"], "serving": None,
                "champ_track": None, "tracks": ["standard"], "kind": "ou"}
    raise ValueError(task)


def challenger_cols(spec: dict, track: str) -> list:
    from src.ht_features import HT_FEATURE_COLS
    if spec["kind"] == "ht":
        return SR.feature_cols("v9_repaired", "none", "standard") + list(HT_FEATURE_COLS)
    return SR.feature_cols("v9_repaired", "none", "newformat" if track == "newformat" else "standard")


def repaired_rows(feat: pd.DataFrame, spec: dict, track: str) -> pd.DataFrame:
    tgt = spec["target"]
    if spec["kind"] == "ht":
        base = SR.track_rows(feat, "pooled", "over25")
        base = ht_frame(base)
        return base.dropna(subset=[tgt])
    return SR.track_rows(feat, track, tgt)


def legacy_rows(raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    feat = SR.feature_frame(raw, "legacy_v9")
    tgt = spec["target"]
    if spec["kind"] == "ht":
        from src.ht_features import build_ht_features
        valid = feat.dropna(subset=["over25", "home_scored_last5"])
        valid = build_ht_features(valid)
        return valid.dropna(subset=[tgt, f"home_{tgt}_rate"])
    return SR.track_rows(feat, spec["champ_track"], tgt)


def holdout_start(rows: pd.DataFrame) -> pd.Timestamp:
    d = pd.to_datetime(rows["date"]).sort_values().reset_index(drop=True)
    return pd.Timestamp(d.iloc[int(len(d) * HOLDOUT_FRAC)]).normalize()


def recency_weights(rows: pd.DataFrame, hs: pd.Timestamp, function: str):
    pre = pd.to_datetime(rows["date"]) < hs
    as_of = pd.to_datetime(rows.loc[pre, "date"]).max()
    w = pd.Series(1.0, index=rows.index)
    w[pre] = R.weights(rows[pre], function, as_of)
    return w, R.summary(rows[pre], w[pre], function, as_of)


# ── training ────────────────────────────────────────────────────────────────────────────────

def train_challenger(rows, spec, cols, hs, function="season_age", imputer_options=None,
                     prov=None, recipe=None):
    w, rsum = recency_weights(rows, hs, function)
    p = dict(prov or {})
    p["recency"] = rsum
    res = M.train(rows, target=spec["target"], sample_weight=w, feature_cols=cols,
                  recipe=recipe or spec["recipe"], holdout_start=hs,
                  imputer_options=imputer_options, provenance=p)
    return M.build_payload(res, spec["target"])


def train_champion(rows: pd.DataFrame, spec: dict, hs: pd.Timestamp) -> dict:
    """Legacy src.model.train exactly as mode_train calls it, split AT the holdout boundary."""
    d = rows.dropna(subset=[spec["target"]])
    n_pre = int((pd.to_datetime(d["date"]) < hs).sum())
    ratio = (n_pre + 0.5) / len(d)
    # mode_train's weights: derived from the match date (the season-label map was deleted).
    w = None if spec["kind"] == "ht" else SR.derived_weights(d).to_numpy()
    res = M.train(d, target=spec["target"], train_ratio=ratio, sample_weight=w,
                  feature_cols=_serving_cols(spec["serving"]))
    return M.build_payload(res, spec["target"])


def basemean(payload: dict, df: pd.DataFrame) -> np.ndarray:
    X, _ = M._prep(df, payload["feature_cols"], imputer=payload.get("imputer"))
    return np.mean([m.predict_proba(X)[:, 1] for k, m in payload["models"].items()
                    if k != "__meta__"], axis=0)


# ── evaluation ──────────────────────────────────────────────────────────────────────────────

def _lg(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_ci(x, groups, n=2000):
    d = pd.DataFrame({"x": np.asarray(x, float), "g": np.asarray(groups)}).dropna()
    s = d.groupby("g")["x"].agg(["sum", "count"])
    if len(s) < 5:
        return [None, None]
    rng = np.random.default_rng(RNG_SEED)
    i = rng.integers(0, len(s), (n, len(s)))
    m = s["sum"].to_numpy()[i].sum(1) / s["count"].to_numpy()[i].sum(1)
    return [round(float(np.percentile(m, 2.5)), 5), round(float(np.percentile(m, 97.5)), 5)]


def arm_metrics(y, p) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": int(len(y)), "log_loss": round(float(_ll(y, p).mean()), 5),
           "brier": round(float(np.mean((p - y) ** 2)), 5),
           "ece": round(M.expected_calibration_error(y, p), 5),
           "mean_p": round(float(p.mean()), 4), "base_rate": round(float(y.mean()), 4),
           "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None}
    if len(set(y)) > 1 and len(y) >= 50:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_lg(p).reshape(-1, 1), y)
        out["cal_slope"] = round(float(lr.coef_[0][0]), 3)
        out["cal_intercept"] = round(float(lr.intercept_[0]), 3)
    return out


def evaluate(E: pd.DataFrame, arms: list, ref: str | None) -> dict:
    y = E["y"].to_numpy(float)
    out = {"n": int(len(E)), "matchdays": int(E["day"].nunique()),
           "dates": [str(E["date"].min().date()), str(E["date"].max().date())] if len(E) else None,
           "arms": {}}
    for a in arms:
        out["arms"][a] = arm_metrics(y, E[f"p_{a}"])
        if ref and a != ref and f"p_{ref}" in E:
            dll = _ll(y, E[f"p_{a}"]) - _ll(y, E[f"p_{ref}"])
            dbr = (E[f"p_{a}"] - y) ** 2 - (E[f"p_{ref}"] - y) ** 2
            out["arms"][a].update({f"delta_ll_vs_{ref}": round(float(dll.mean()), 5),
                                   f"delta_ll_vs_{ref}_ci95": boot_ci(dll, E["day"]),
                                   f"delta_brier_vs_{ref}": round(float(dbr.mean()), 5),
                                   f"delta_brier_vs_{ref}_ci95": boot_ci(dbr, E["day"])})
    return out


def by_season(E: pd.DataFrame, arms: list) -> dict:
    from src.features_repaired import season_key
    sk = season_key(E["date"], E["league"])
    return {s: {a: arm_metrics(g["y"], g[f"p_{a}"]) for a in arms}
            for s, g in E.groupby(sk) if len(g) >= 100}


# ── genuine two-sided ENTRY prices (ported from the 2026-11-01 league/anchor benchmark) ──────

SIDE_MKTS = {"btts": ("btts_yes", "btts_no"), "over15": ("over15", "under15"),
             "over35": ("over35", "under35"), "over25": ("over25", "under25")}


def _odds_file(name: str) -> pd.DataFrame:
    p = ROOT / "output" / name
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    d["snapshot_ts"] = pd.to_datetime(d["snapshot_ts"], utc=True, errors="coerce")
    d["kickoff_utc"] = pd.to_datetime(d.get("kickoff_utc"), utc=True, errors="coerce")
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce").dt.normalize()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    # ENTRY only: a quote taken at/after kickoff is never an entry price
    return d[d["kickoff_utc"].isna() | (d["snapshot_ts"] < d["kickoff_utc"])]


def entry_pairs(target: str, track: str) -> tuple[pd.DataFrame, str]:
    if target not in SIDE_MKTS:
        return pd.DataFrame(), "NO_GENUINE_TWO_SIDED_HISTORY"
    yes, no = SIDE_MKTS[target]
    if track == "standard":
        if target == "over25":
            return pd.DataFrame(), "NOT_USED_HERE"
        d = _odds_file("standard_sidemarket_odds_history.csv")
        src = "standard_sidemarket_odds_history.csv first pre-kickoff two-sided quote"
        if len(d):
            d = d[(d["match_date"] < NF_SUMMER[0]) | (d["match_date"] >= NF_SUMMER[1])]
    else:
        d = _odds_file("newformat_odds_history.csv")
        src = "newformat_odds_history.csv first pre-kickoff two-sided quote (summer 2026 excluded)"
        if len(d):
            # non-full-match lines (NF over15 median 1.57-1.66 vs 1.20 forward; summer over25
            # median 2.20) -> summer dropped for every market, O1.5/O3.5 only from 2026-08-23
            d = d[(d["match_date"] < NF_SUMMER[0]) | (d["match_date"] >= NF_SUMMER[1])]
            if target in ("over15", "over35"):
                d = d[d["match_date"] >= NF_SUMMER[1]]
    if d.empty:
        return pd.DataFrame(), src
    d = d[d["market"].isin([yes, no])]
    if target == "btts":
        d = d[~((d["market"] == "btts_yes") & (d["odds"] > 3.2))]   # first-half BTTS mislabel
    w = d.pivot_table(index=["league", "match", "match_date", "snapshot_ts"], columns="market",
                      values="odds", aggfunc="median").reset_index()
    if yes not in w.columns or no not in w.columns:
        return pd.DataFrame(), src
    w = w.dropna(subset=[yes, no]).sort_values("snapshot_ts")
    w = w.drop_duplicates(["league", "match", "match_date"], keep="first")
    w[["home", "away"]] = w["match"].str.split(" vs ", n=1, expand=True)
    return w.rename(columns={yes: "odds_a", no: "odds_b"})[
        ["league", "match_date", "home", "away", "odds_a", "odds_b"]], src


def join_prices(E: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    from src.team_names import resolve
    out = pd.DataFrame(index=E.index, columns=["odds_a", "odds_b"], dtype=float)
    if px.empty:
        return out
    for lg, g in px.groupby("league"):
        rr = E[E["league"] == lg]
        if rr.empty:
            continue
        teams = sorted(set(rr["home_team"].astype(str)) | set(rr["away_team"].astype(str)))
        key = {(d, h, a): i for i, d, h, a in zip(rr.index, rr["date"], rr["home_team"].astype(str),
                                                   rr["away_team"].astype(str))}
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


def market_block(E: pd.DataFrame, target: str, track: str, arms: list) -> dict:
    """Genuine two-sided entry prices only; proportional de-vig; negative-margin pairs excluded."""
    from sklearn.linear_model import LogisticRegression
    px, src = entry_pairs(target, track)
    if px.empty:
        return {"status": "NOT_EVALUABLE", "price_source": src, "n_priced": 0}
    j = join_prices(E, px)
    a, b = pd.to_numeric(j["odds_a"], errors="coerce"), pd.to_numeric(j["odds_b"], errors="coerce")
    margin = 1 / a + 1 / b - 1
    ok = a.notna() & b.notna() & (margin >= 0)
    if ok.sum() < 100:
        return {"status": "TOO_FEW_PRICED", "price_source": src, "n_priced": int(ok.sum()),
                "n_negative_margin_excluded": int((margin < 0).sum())}
    ia, ib = 1 / a[ok], 1 / b[ok]
    P = E[ok].copy()
    P["p_mkt"] = (ia / (ia + ib)).to_numpy()
    y = P["y"].to_numpy(float)
    out = {"status": "OK", "price_source": src, "price_class": "GENUINE_TWO_SIDED_ENTRY_DEVIG",
           "n_priced": int(ok.sum()), "n_negative_margin_excluded": int((margin < 0).sum()),
           "median_margin": round(float(margin[ok].median()), 4),
           "market": arm_metrics(y, P["p_mkt"]), "arms_on_priced_rows": {}}
    P = P.sort_values("date")
    cut = P["date"].iloc[len(P) // 2]
    fa, sb = P[P["date"] < cut], P[P["date"] >= cut]
    for arm in arms:
        r = {"model": arm_metrics(y, P[f"p_{arm}"])}
        if len(fa) >= 100 and len(sb) >= 100:
            m0 = LogisticRegression(C=1e6, max_iter=1000).fit(_lg(fa["p_mkt"]).reshape(-1, 1), fa["y"])
            X = lambda x: np.c_[_lg(x["p_mkt"]), _lg(x[f"p_{arm}"])]          # noqa: E731
            m1 = LogisticRegression(C=1e6, max_iter=1000).fit(X(fa), fa["y"])
            l0 = _ll(sb["y"].to_numpy(float), m0.predict_proba(_lg(sb["p_mkt"]).reshape(-1, 1))[:, 1])
            l1 = _ll(sb["y"].to_numpy(float), m1.predict_proba(X(sb))[:, 1])
            r["residual_vs_market"] = {"gain": round(float((l0 - l1).mean()), 5),
                                       "gain_ci95": boot_ci(l0 - l1, sb["day"]),
                                       "weight_on_model": round(float(m1.coef_[0][1]), 3),
                                       "n_fit": int(len(fa)), "n_score": int(len(sb))}
        out["arms_on_priced_rows"][arm] = r
    return out


# ── source composition / missingness (finding 13) ───────────────────────────────────────────

def source_report(rows: pd.DataFrame, hs, cols: list, target: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from src.features_repaired import season_key
    pre = rows[pd.to_datetime(rows["date"]) < hs]
    src = pre["_source"].astype(str)
    sk = season_key(pre["date"], pre["league"])
    comp = pd.crosstab(sk, src).to_dict(orient="index")
    present = [c for c in cols if c in pre.columns]
    miss = pre[present].isna()
    out = {"rows_by_source": src.value_counts().to_dict(),
           "rows_by_season_source": {s: {k: int(v) for k, v in d.items()} for s, d in comp.items()},
           "target_rate_by_source": pre.groupby(src)[target].mean().round(4).to_dict(),
           "missing_rate_by_source": {s: miss[src == s].mean().round(4).to_dict()
                                      for s in sorted(src.unique())}}
    yb = (src == "ARCHIVE").astype(int).to_numpy()
    if 0 < yb.sum() < len(yb) and min(yb.sum(), len(yb) - yb.sum()) >= 50:
        Xm = miss.astype(float).to_numpy()
        auc = cross_val_score(LogisticRegression(max_iter=1000), Xm, yb,
                              cv=StratifiedKFold(5, shuffle=True, random_state=RNG_SEED),
                              scoring="roc_auc")
        out["source_predictable_from_missingness_auc"] = round(float(auc.mean()), 4)
    else:
        out["source_predictable_from_missingness_auc"] = None
        out["note"] = "single source in the pre-holdout rows"
    return out


def regime_masks(rows: pd.DataFrame, hs) -> dict:
    """Frozen 2026-11-01 BEFORE evaluation. Every mask selects TRAINING rows (< hs) only; the
    holdout is appended unchanged, so every regime is scored on the identical rows."""
    from src.features_repaired import season_key
    d = pd.to_datetime(rows["date"])
    pre = d < hs
    src = rows["_source"].astype(str)
    start = season_key(d, rows["league"]).str.slice(0, 4).astype(int)
    ref = start[pre].max() if pre.any() else 0
    arch_min = d[pre & (src == "ARCHIVE")].min() if (pre & (src == "ARCHIVE")).any() else None
    return {
        "FULL": pre,
        "MODERN_ONLY": pre & (src == "MODERN"),
        "MODERN_ARCHIVE": pre & (d >= arch_min) if arch_min is not None else pre,
        "RECENT_2": pre & (start >= ref - 1),
        "RECENT_3": pre & (start >= ref - 2),
    }


# ── main task runner ────────────────────────────────────────────────────────────────────────

def run(task: str, group: str, track: str | None, out: Path, cutoff: pd.Timestamp) -> dict:
    from src.features_repaired import FEATURE_RECIPE_VERSION, build_features_repaired
    from src.identity_repair import IDENTITY_VERSION
    spec = task_spec(task)
    F = load_frames(cutoff, need_modern=(group in ("core", "regimes") and spec["kind"] != "ht"))
    prov = {"feature_recipe_version": FEATURE_RECIPE_VERSION, "identity_version": IDENTITY_VERSION,
            "training_cutoff": str(cutoff.date()), "alias_map_file": str(ALIAS_MAP.relative_to(ROOT)),
            "n_aliases": sum(len(m) for m in F["frozen_map"].values()), "task": task, "group": group,
            "price_mode": "none", "team_scope": "league"}
    feat = SR.feature_frame(F["hist"], "v9_repaired", "none", None, "league")
    target = spec["target"]
    res = {"task": task, "group": group, "target": target, "cutoff": str(cutoff.date()),
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "v9_commit": os.getenv("GITHUB_SHA", "local"), "identity_report": F["identity_report"],
           "lifecycle": "RETROSPECTIVE_REPLAY", "promotion": "NONE (research only)"}
    preds, arms_meta = {}, {}

    if group == "coldstart":
        if task != "std_ou25":
            raise SystemExit("coldstart ablation is defined for std_ou25 only")
        rows_a = SR.track_rows(feat, "standard", target)
        hs = holdout_start(rows_a)
        cols = SR.feature_cols("v9_repaired", "none", "standard")
        feat_nan = build_features_repaired(F["hist"], price_mode="none", team_scope="league",
                                           cold_start="nan")
        rows_n = SR.track_rows(feat_nan, "standard", target)
        variants = {"A_CURRENT_PRIORS": (rows_a, cols, None),
                    "B_NAN_TRAIN_MEDIAN": (rows_n, cols, None),
                    "C_TRAIN_LEAGUE_PRIOR": (rows_n, cols, {"group_col": "league"}),
                    "D_GLOBAL_PRIOR_MISSING_IND": (rows_n, cols + [f"{c}__missing" for c in COLD_FEATS
                                                                    if c in cols], None)}
        frames = {}
        for arm, (rr, cc, io) in variants.items():
            pay = train_challenger(rr, spec, cc, hs, imputer_options=io, recipe="v9_repaired",
                                   prov={**prov, "arm": arm})
            frames[arm] = (rr, pay)
            arms_meta[arm] = {k: v for k, v in pay.items() if k not in ("models", "imputer")}
        ref_arm = "A_CURRENT_PRIORS"
        E = _eval_frame({a: (r[pd.to_datetime(r["date"]) >= hs], p) for a, (r, p) in frames.items()},
                        target, F["frozen_map"], key_arm=ref_arm)
        E = E.dropna(subset=[f"p_{a}" for a in frames])
        res["holdout_start"] = str(hs.date())
        res["tracks"] = {"standard": evaluate(E, list(frames), ref_arm)}
        res["tracks"]["standard"]["by_season"] = by_season(E, list(frames))
        res["cold_start_share_in_holdout"] = {
            c: round(float(frames["B_NAN_TRAIN_MEDIAN"][0].loc[
                pd.to_datetime(frames["B_NAN_TRAIN_MEDIAN"][0]["date"]) >= hs, c].isna().mean()), 4)
            for c in COLD_FEATS if c in cols}
        _write(out, f"{task}__{group}", res, E, arms_meta)
        return res

    tracks = [track] if (group == "regimes" and track) else spec["tracks"]
    prim_track = "pooled" if spec["kind"] in ("side", "ht") else tracks[0]
    rows_by_track = {t: repaired_rows(feat, spec, t) for t in set(tracks) | {prim_track}}
    hs = holdout_start(rows_by_track[prim_track])
    res["holdout_start"] = str(hs.date())
    trained = {}
    if group == "core":
        champ_rows = legacy_rows(F["raw"], spec)
        champ = train_champion(champ_rows, spec, hs)
        ch_hold = champ_rows[pd.to_datetime(champ_rows["date"]) >= hs]
        arms_meta["CHAMPION_LEGACY_REPRO"] = {"recipe": "legacy_v9", "feature_cols": champ["feature_cols"],
                                              "metrics": champ["metrics"],
                                              "note": "meta-logistic fitted ON the holdout (bug 6)"}
        trained["CHAMPION_LEGACY_REPRO"] = (ch_hold, champ, "meta")
        trained["CHAMPION_BASEMEAN"] = (ch_hold, champ, "basemean")
        for t in tracks:
            rr = rows_by_track[t]
            arm = f"CHALLENGER_{t.upper()}"
            try:
                pay = train_challenger(rr, spec, challenger_cols(spec, t), hs,
                                       prov={**prov, "track": t, "arm": arm})
            except ValueError as e:
                if "too few rows" not in str(e):
                    raise
                res.setdefault("skipped", {})[arm] = f"TOO_FEW_ROWS: {e}"
                continue
            trained[arm] = (rr[pd.to_datetime(rr["date"]) >= hs], pay, "meta")
            arms_meta[arm] = {k: v for k, v in pay.items() if k not in ("models", "imputer")}
            _save_payload(out, f"{task}__{arm}", pay)
            if spec["kind"] != "ht":
                res.setdefault("source_report", {})[t] = source_report(rr, hs, challenger_cols(spec, t), target)
        ref = "CHAMPION_LEGACY_REPRO"
    else:                                                   # regimes
        t = tracks[0]
        rr = rows_by_track[t]
        cols = challenger_cols(spec, t)
        res["track"] = t
        res["source_report"] = {t: source_report(rr, hs, cols, target)}
        masks = regime_masks(rr, hs)
        hold_m = pd.to_datetime(rr["date"]) >= hs
        arms = {}
        for name, m in masks.items():
            arms[name] = (rr[m | hold_m], "uniform")
        arms["RECENCY_WEIGHTED"] = (rr, R.PRIMARY)
        arms["RECENCY_EXP_HALF_LIFE"] = (rr, "exp_half_life")
        arms["RECENCY_BOUNDED_LINEAR"] = (rr, "bounded_linear")
        res["regime_rows"] = {k: int((pd.to_datetime(v[0]["date"]) < hs).sum()) for k, v in arms.items()}
        for name, (sub, fn) in arms.items():
            if (pd.to_datetime(sub["date"]) < hs).sum() < config.BACKTEST_MIN_TRAIN:
                res.setdefault("skipped", {})[name] = "too few training rows"
                continue
            pay = train_challenger(sub, spec, cols, hs, function=fn,
                                   prov={**prov, "track": t, "arm": name, "regime": name})
            trained[name] = (sub[pd.to_datetime(sub["date"]) >= hs], pay, "meta")
            arms_meta[name] = {k: v for k, v in pay.items() if k not in ("models", "imputer")}
        ref = "RECENCY_WEIGHTED"

    key_arm = next(a for a in trained if a.startswith("CHALLENGER") or a == ref)
    E_all = _eval_frame({a: (h, p, how) for a, (h, p, how) in trained.items()}, target,
                        F["frozen_map"], key_arm=key_arm)
    arms = list(trained)
    res["tracks"] = {}
    for t in ("standard", "newformat"):
        if group == "regimes" and t != res.get("track") and res.get("track") != "pooled":
            continue
        E = E_all[E_all["league"].isin(TRACK_LEAGUES[t])]
        if len(E) < 100:
            res["tracks"][t] = {"n": int(len(E)), "status": "TOO_FEW"}
            continue
        # a track-specific challenger is only scored on its own track's rows
        t_arms = [a for a in arms if not (a == "CHALLENGER_STANDARD" and t != "standard")
                  and not (a == "CHALLENGER_NEWFORMAT" and t != "newformat")]
        E = E.dropna(subset=[f"p_{a}" for a in t_arms])
        ev = evaluate(E, t_arms, ref)
        if group == "core" and f"CHALLENGER_{t.upper()}" in t_arms and "CHALLENGER_POOLED" in t_arms:
            ev["track_vs_pooled"] = evaluate(E, [f"CHALLENGER_{t.upper()}", "CHALLENGER_POOLED"],
                                             "CHALLENGER_POOLED")["arms"][f"CHALLENGER_{t.upper()}"]
        live = E[E["date"] >= LIVE_FROM]
        if len(live) >= 100:
            ev["live_from_2026_09_01"] = evaluate(live, t_arms, ref)
        ev["by_season"] = by_season(E, t_arms)
        ev["market"] = market_block(E, target, t, t_arms)
        res["tracks"][t] = ev
    _write(out, f"{task}__{group}" + (f"__{res['track']}" if group == "regimes" else ""), res, E_all,
           arms_meta)
    return res


def _eval_frame(trained: dict, target: str, frozen: dict, key_arm: str) -> pd.DataFrame:
    """One row per canonical fixture in ANY arm's holdout (outer join on canonical keys); the
    caller keeps only rows every compared arm scored (`dropna` on their p_ columns), so each
    comparison is on IDENTICAL fixtures while track-specific arms can still be joined."""
    frames = []
    for arm, v in trained.items():
        h, pay, how = v if len(v) == 3 else (v[0], v[1], "meta")
        h = h.dropna(subset=[target])
        p = (M.predict_proba(h, payload=pay).to_numpy() if how == "meta" else basemean(pay, h))
        k = _canon_keys(h, frozen)
        f = pd.DataFrame({"key": k, f"p_{arm}": p, f"y_{arm}": h[target].to_numpy(float)})
        f = f.drop_duplicates("key", keep="first")
        frames.append(f.set_index("key"))
    E = frames[0].join(frames[1:], how="outer") if len(frames) > 1 else frames[0]
    E = E.reset_index()
    E[["date", "league", "home_team", "away_team"]] = pd.DataFrame(E["key"].tolist(), index=E.index)
    ycols = [c for c in E.columns if c.startswith("y_")]
    agree = (E[ycols].nunique(axis=1) == 1)
    if not agree.all():
        log.warning(f"{int((~agree).sum())} fixture(s) whose label differs between arms dropped")
    E = E[agree].copy()
    E["y"] = E[ycols].bfill(axis=1).iloc[:, 0]
    E = E.drop(columns=ycols + ["key"])
    E["date"] = pd.to_datetime(E["date"])
    E["day"] = E["date"].dt.strftime("%Y-%m-%d")
    return E.sort_values(["date", "league", "home_team"]).reset_index(drop=True)


def _save_payload(out: Path, name: str, pay: dict) -> None:
    p = out / "payloads" / f"{name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(pay, f)


def _write(out: Path, name: str, res: dict, E: pd.DataFrame, arms_meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    res["arms_metadata"] = arms_meta
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    with gzip.open(out / f"{name}_holdout.csv.gz", "wt", encoding="utf-8") as f:
        E.to_csv(f, index=False)


def run_identity(out: Path, cutoff: pd.Timestamp) -> dict:
    """NF-first identity evidence: overlap audit on the RAW frame, residual collisions after."""
    from src.identity_repair import load_alias_map, overlap_audit, split_history_check
    F = load_frames(cutoff, need_modern=False)
    explicit = load_alias_map(ALIAS_MAP)
    aud = overlap_audit(F["raw"], explicit)
    aud["track"] = aud["league"].map(lambda l: "newformat" if l in config.NEW_FORMAT_LEAGUES
                                     else "standard" if l in config.STANDARD_FORMAT_LEAGUES else "other")
    resid = split_history_check(F["hist"])
    cn = F["hist"][F["hist"]["league"] == "China Super League"]
    names = set(cn["home_team"]) | set(cn["away_team"])
    health = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "identity_version": F["identity_report"].get("identity_version"),
              "alias_map": str(ALIAS_MAP.relative_to(ROOT)),
              "identity_report": F["identity_report"],
              "decisions_by_track": aud.groupby(["track", "decision"]).size().unstack(fill_value=0)
              .to_dict(orient="index"),
              "candidate_review": aud[aud["decision"] == "CANDIDATE_REVIEW"].to_dict("records"),
              "residual_same_day_same_score_by_league": resid.groupby("league")["rows"].size().to_dict(),
              "china_checks": {"Shandong Luneng_present": "Shandong Luneng" in names,
                               "Shandong Taishan_present": "Shandong Taishan" in names,
                               "Henan Jianye_present": "Henan Jianye" in names,
                               "Henan Songshan Longmen_present": "Henan Songshan Longmen" in names}}
    ok = not health["china_checks"]["Shandong Luneng_present"] and not health["china_checks"]["Henan Jianye_present"]
    health["status"] = "PASS" if ok and not health["candidate_review"] else ("WARN" if ok else "FAIL")
    out.mkdir(parents=True, exist_ok=True)
    aud.to_csv(out / "identity_overlap_audit.csv", index=False)
    (out / "identity_health.json").write_text(json.dumps(health, indent=1, default=str), encoding="utf-8")
    return health


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--group", default="core", choices=GROUPS)
    ap.add_argument("--track", default=None, choices=("standard", "newformat", "pooled"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cutoff = pd.Timestamp(a.cutoff) if a.cutoff else pd.Timestamp(datetime.now(timezone.utc).date())
    out = Path(a.out)
    if a.task == "identity":
        h = run_identity(out, cutoff)
        print(json.dumps({k: h[k] for k in ("status", "china_checks", "decisions_by_track")}, indent=1, default=str))
        return 0 if h["status"] != "FAIL" else 1
    res = run(a.task, a.group, a.track, out, cutoff)
    print(json.dumps({t: {k: v.get(k) for k in ("n", "status")} | {
        "arms": {arm: {m: x.get(m) for m in ("log_loss", "brier", "ece", "auc")}
                 for arm, x in (v.get("arms") or {}).items()}}
        for t, v in res["tracks"].items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
