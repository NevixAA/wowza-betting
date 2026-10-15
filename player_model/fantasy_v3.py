"""Fantasy V3 inputs — Player V3's P(goal), P(assist) and expected minutes for Premier League players.

    python -m player_model.fantasy_v3        # -> output/fantasy_v3_inputs.csv + fantasy_v3_meta.json

WHY (owner-approved 2026-10-15). Pro's study (wowzaV9-Pro output/studies/FANTASY_V3.md) rebuilt 42
weeks of Premier League projections: on players who played their club's previous match, V3 picked a
top-11 worth 42.0 real FPL points per week against v9's 39.0 (90% CI +1.2 to +4.8). This module is
that V3 variant, ported: BASE_V9 + FORM_SHRUNK + TEAM_ENV + expected-minutes features, HistGradient-
Boosting, Platt on LOG-ODDS (not v9's Platt-on-probability), plus the expected-minutes model.

RUNS DAILY in its own workflow (fantasy_v3.yml, ~10-15 min: too heavy for fantasy_refresh's 20-minute
job). It retrains from scratch each run - no model file is committed. fantasy.py reads the output
only if the recorded `generated_at` is under 36 hours old (never file mtime); otherwise it falls back
to v9's own numbers. FANTASY_MODEL=v9 switches V3 off.

TIMING CONVENTION, identical to the study and to v9's Fantasy (`tail(1)` of history): P(goal) and
P(assist) come from the player's LAST appearance row; minutes come from the synthetic NEXT-fixture
row of the live squad panel (player_model/v3/minutes.py).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "output" / "fantasy_v3_inputs.csv"
OUT_META = ROOT / "output" / "fantasy_v3_meta.json"
LEAGUE = "Premier League"
VALID_DAYS = 120          # Platt calibration window = the last 120 days of appearances
MAX_AGE_HOURS = 36
ATT_HGB = dict(max_iter=300, max_depth=3, learning_rate=0.05, min_samples_leaf=20, early_stopping=False,
               random_state=42)
MIN_COLS = ["xm_start", "xm_minutes", "xm_m60", "xm_m75", "xm_m90"]


def _usable(X: pd.DataFrame) -> list[str]:
    v = X.to_numpy(float)
    return [c for j, c in enumerate(X.columns) if np.isfinite(v[:, j]).sum() > 1
            and np.nanmax(v[:, j]) > np.nanmin(v[:, j])]


def _fit_binary(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, float]:
    """HGB on train, Platt on the LOG-ODDS of the raw score on validation. Returns probs + validation AUC."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    cols = _usable(X[tr])
    m = HistGradientBoostingClassifier(**ATT_HGB).fit(X.loc[tr, cols], y[tr])
    raw = np.clip(m.predict_proba(X[cols])[:, 1], 1e-6, 1 - 1e-6)
    z = np.log(raw / (1 - raw)).reshape(-1, 1)
    pl = LogisticRegression(C=1.0, max_iter=1000).fit(z[va], y[va])
    auc = float(roc_auc_score(y[va], raw[va])) if len(np.unique(y[va])) > 1 else float("nan")
    return np.clip(pl.predict_proba(z)[:, 1], 0.001, 0.999), auc


def build(asof: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    from player_model.v3 import data as D
    from player_model.v3 import features as F
    from player_model.v3 import minutes as M
    asof = (asof or pd.Timestamp.now(tz="UTC").tz_localize(None)).normalize()

    # 1. expected minutes on the live squad panel
    h_raw = D.load_history()
    h_raw = h_raw[h_raw["date"] < asof]
    xm = M.fit_predict(M.add_features(M.build_panel(h_raw, asof=asof)))

    # 2. attack models on appearance rows
    want = list(dict.fromkeys(D.RAW + F.base_cols() + ["n_prev_games"]))
    h = D.load_history(cols=want)
    h = h[h["date"] < asof]
    h = h[h["n_prev_games"] >= 1].reset_index(drop=True) if "n_prev_games" in h.columns else h
    cut = h["date"].max() - pd.Timedelta(days=VALID_DAYS)
    X, fam = F.build(h, prior_cutoff=cut)
    played = xm[xm["fixture_id"] > 0].drop_duplicates(["player_id", "fixture_id"])
    X = X.merge(played[["player_id", "fixture_id", *MIN_COLS]], on=["player_id", "fixture_id"], how="left")
    cols = fam["BASE_V9"] + fam["FORM_SHRUNK"] + fam["TEAM_ENV"] + MIN_COLS
    tr, va = (X["date"] < cut).to_numpy(), (X["date"] >= cut).to_numpy()
    meta = {"asof": str(asof.date()), "history_to": str(h["date"].max().date()),
            "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_features": len(cols)}
    for name, col in (("goal", "goals"), ("assist", "assists")):
        y = (X[col].fillna(0) > 0).astype(int).to_numpy()
        X[f"p_{name}_v3"], meta[f"auc_valid_{name}"] = _fit_binary(X[cols], y, tr, va)
        meta[f"auc_valid_{name}"] = round(meta[f"auc_valid_{name}"], 4)

    # 3. one row per current Premier League player
    last = X.sort_values(["date", "fixture_id"], kind="mergesort").groupby("player_id").tail(1)
    last = last[last["league"] == LEAGUE]
    nxt = xm[xm["fixture_id"] < 0][["player_id", "team", *MIN_COLS]].rename(columns={"team": "next_team"})
    out = last[["player_id", "player_name", "team", "date", "p_goal_v3", "p_assist_v3"]].rename(
        columns={"date": "last_app_date"}).merge(nxt, on="player_id", how="left")
    out = out.drop_duplicates("player_id")
    out["last_app_date"] = out["last_app_date"].dt.strftime("%Y-%m-%d")
    for c in ("p_goal_v3", "p_assist_v3", *MIN_COLS):
        out[c] = out[c].round(4)
    import sklearn
    meta.update({"n_players": int(len(out)), "n_with_minutes": int(out["xm_start"].notna().sum()),
                 "sklearn": sklearn.__version__,
                 "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")})
    return out, meta


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out, meta = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


# ── read side, used by fantasy.py ─────────────────────────────────────────────────────────────
def load_inputs(max_age_hours: float = MAX_AGE_HOURS) -> pd.DataFrame | None:
    """The latest V3 inputs, or None if switched off, missing, unreadable or stale.

    Age comes from the RECORDED generated_at in the meta sidecar - never file mtime, which git
    checkout resets on every CI run. An unknown age counts as stale."""
    if os.getenv("FANTASY_MODEL", "v3").strip().lower() == "v9":
        return None
    try:
        meta = json.loads(OUT_META.read_text(encoding="utf-8"))
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(meta["generated_at"])).total_seconds() / 3600
        if not (0 <= age <= max_age_hours):
            log.warning("fantasy V3 inputs are %.0fh old - using v9", age)
            return None
        d = pd.read_csv(OUT_CSV)
        d["player_id"] = pd.to_numeric(d["player_id"], errors="coerce")
        return d.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")
    except Exception as e:                                           # noqa: BLE001 - fall back to v9
        log.warning("fantasy V3 inputs unavailable (%s) - using v9", e)
        return None


def apply_attack_and_minutes(pl: pd.DataFrame) -> pd.DataFrame:
    """Swap in V3's P(goal), P(assist) and expected-minutes appearance points where available.

    v9's own numbers are kept as *_v9 columns. Sets `_v3_start` for fantasy.py's P(start) step and
    `fantasy_model` per row ("v3" or "v9"). Conditional appearance = (2 P(60+) + (P(start) - P(60+)))
    / P(start), so that (appearance + attack + ...) x P(start) equals the study's projection."""
    pl = pl.copy()
    pl["fantasy_model"] = "v9"
    pl["_v3_start"] = np.nan
    inp = load_inputs()
    if inp is None or "player_id" not in pl.columns:
        return pl
    pid = pd.to_numeric(pl["player_id"], errors="coerce")
    g, a = pid.map(inp["p_goal_v3"]), pid.map(inp["p_assist_v3"])
    xs, x60 = pid.map(inp["xm_start"]), pid.map(inp["xm_m60"])
    ok = g.notna() & a.notna()
    pl["p_goal_v9"], pl["p_assist_v9"], pl["appearance_v9"] = pl["p_goal"], pl["p_assist"], pl["appearance"]
    pl.loc[ok, "p_goal"] = g[ok]
    pl.loc[ok, "p_assist"] = a[ok]
    mins = ok & xs.notna() & x60.notna() & (xs > 0.01)
    x60c = np.minimum(x60, xs)
    pl.loc[mins, "appearance"] = ((2.0 * x60c + (xs - x60c)) / xs)[mins].clip(1.0, 2.0)
    pl.loc[mins, "_v3_start"] = xs[mins]
    pl.loc[ok, "fantasy_model"] = "v3"
    log.info("fantasy: V3 inputs for %d/%d players (%d with expected minutes)", int(ok.sum()), len(pl), int(mins.sum()))
    return pl


if __name__ == "__main__":
    raise SystemExit(main())
