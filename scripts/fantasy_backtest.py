"""Backtest & calibrate the fantasy engine against RECONSTRUCTED actual FPL points.

We rebuild each PL player-game's ACTUAL FPL points from raw stats using the real scoring rules
(appearance / goals / assists / clean-sheet / saves / cards = EXACT; bonus = BPS-proxy, approx;
defensive-contribution = excluded, source lacks clearances/recoveries). Then two things:

  1. CALIBRATION (needs no model) — regress actual points on the actual components. The
     coefficients are the empirically-correct point-value of each driver; they confirm the
     FPL-fixed parts and reveal whether our HEURISTIC weights (the sot2 "+1" proxy, bonus) are
     right or biased.
  2. PROJECTION ACCURACY — run the calibrated prop models on the same as-of-date feature rows
     to get the ex-ante expected points, compare to actual: MAE + Spearman rank correlation,
     overall and per position. (In-sample caveat: models were trained on these rows -> accuracy
     optimistic; the calibration in (1) is the robust output.)

Usage:  python scripts/fantasy_backtest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from player_model.model import load_model, predict_proba  # noqa: E402

PARQUET  = Path(__file__).resolve().parents[1] / "player_history.parquet"
POS_MAP  = {"F": "FWD", "M": "MID", "D": "DEF", "G": "GKP",
            "FWD": "FWD", "MID": "MID", "DEF": "DEF", "GKP": "GKP", "GK": "GKP"}
GOAL_PTS = {"FWD": 4, "MID": 5, "DEF": 6, "GKP": 6}
CS_PTS   = {"DEF": 4, "GKP": 4, "MID": 1, "FWD": 0}


#: Every raw stat the reconstruction reads. A player-game missing any of them (or with a position
#: POS_MAP does not know) is DROPPED and counted - a missing stat is not a 0 and an unknown position
#: is not a midfielder.
STAT_COLS = ["minutes", "goals", "assists", "saves", "yellow_cards", "red_cards", "penalty_saved",
             "penalty_missed", "goals_conceded", "shots_on_target", "key_passes", "tackles_total",
             "interceptions", "fouls_committed"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def complete_rows(pl: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known position and every STAT_COLS value present; the rest dropped, logged."""
    pos = pl["position"].map(lambda p: POS_MAP.get(str(p)))
    absent = [c for c in STAT_COLS if c not in pl.columns]
    if absent:
        raise SystemExit(f"player_history.parquet lacks {absent} - cannot reconstruct FPL points")
    ok = pos.notna() & pl[STAT_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if (~ok).any():
        print(f"  dropped {int((~ok).sum()):,} player-game(s): unknown position "
              f"({int(pos.isna().sum()):,}) or a missing stat")
    return pl[ok].assign(pos=pos[ok])


def build_actuals(pl: pd.DataFrame) -> pd.DataFrame:
    pl = complete_rows(pl)
    # team goals conceded per fixture = the GK's value (max over the team's players that match)
    tc = pl.groupby(["fixture_id", "team"])["goals_conceded"].max().rename("team_conceded").reset_index()
    pl = pl.merge(tc, on=["fixture_id", "team"], how="left")

    m    = _num(pl["minutes"])
    gls  = _num(pl["goals"]);   ast = _num(pl["assists"])
    sv   = _num(pl["saves"]);   yc  = _num(pl["yellow_cards"]); rc = _num(pl["red_cards"])
    psv  = _num(pl["penalty_saved"]); pms = _num(pl["penalty_missed"])
    conc = _num(pl["team_conceded"])
    gp   = pl["pos"].map(GOAL_PTS).astype(float)
    csp  = pl["pos"].map(CS_PTS).astype(float)
    isgk_def = pl["pos"].isin(["DEF", "GKP"])

    pl["did_score"]  = (gls > 0).astype(int)
    pl["did_assist"] = (ast > 0).astype(int)
    pl["did_sot2"]   = (_num(pl["shots_on_target"]) >= 2).astype(int)
    pl["clean_sheet"] = ((m >= 60) & (conc == 0)).astype(int)

    appr   = np.where(m >= 60, 2.0, np.where(m >= 1, 1.0, 0.0))
    goalp  = gls * gp
    astp   = ast * 3.0
    csp_v  = np.where((m >= 60) & (conc == 0), csp, 0.0)
    concp  = np.where(isgk_def, -np.floor(conc / 2.0), 0.0)
    savep  = np.where(pl["pos"] == "GKP", np.floor(sv / 3.0), 0.0)
    penp   = psv * 5.0 - pms * 2.0
    cardp  = -(yc * 1.0 + rc * 3.0)
    pl["appearance_pts"] = appr
    pl["goal_pts_a"] = goalp
    pl["assist_pts_a"] = astp
    pl["cs_pts_a"] = csp_v
    pl["actual_core"] = appr + goalp + astp + csp_v + concp + savep + penp + cardp

    # ── bonus (BPS proxy → rank top-3 per fixture) ────────────────────────────
    bps = (np.where(m >= 60, 6.0, np.where(m >= 1, 3.0, 0.0))
           + gls * pl["pos"].map({"GKP": 12, "DEF": 12, "MID": 18, "FWD": 24}).astype(float)
           + ast * 9.0
           + np.where((m >= 60) & (conc == 0), pl["pos"].map({"GKP": 12, "DEF": 12, "MID": 0, "FWD": 0}).astype(float), 0.0)
           + np.where(pl["pos"] == "GKP", sv * 2.0, 0.0)
           + psv * 15.0
           + _num(pl["key_passes"]) * 1.0
           + _num(pl["tackles_total"]) * 2.0
           + _num(pl["interceptions"]) * 1.0
           - yc * 3.0 - rc * 9.0 - pms * 6.0 - _num(pl["fouls_committed"]) * 1.0)
    pl["_bps"] = bps
    pl["actual_bonus"] = 0.0
    played = m > 0
    for fid, grp in pl[played].groupby("fixture_id"):
        top = grp["_bps"].nlargest(3)
        for rank, (idx, _) in enumerate(top.items()):
            pl.at[idx, "actual_bonus"] = [3.0, 2.0, 1.0][rank]

    pl["actual_pts"] = pl["actual_core"] + pl["actual_bonus"]
    return pl


def main():
    # FPL scores ONE league: the registry's fpl-flagged entry, matched by canonical name.
    from src.league_registry import fpl_league, canonical_name
    league = fpl_league()
    if league is None:
        raise SystemExit("config/leagues.json flags no FPL league - nothing to backtest")
    d = pd.read_parquet(PARQUET)
    pl = d[d["league"].map(lambda x: canonical_name(x) == league)].copy()
    pl = pl[_num(pl["minutes"]) > 0].copy()   # played games only
    print(f"{league} played player-games: {len(pl):,}")

    pl = build_actuals(pl)

    # ── 1. CALIBRATION: regress actual points on actual components ────────────
    print("\n=== 1. WEIGHT CALIBRATION — value of each component (regress actual_pts on outcomes) ===")
    from numpy.linalg import lstsq
    X_cols = ["appearance_pts", "did_score", "did_assist", "did_sot2", "clean_sheet"]
    X = np.column_stack([pl[c].values.astype(float) for c in X_cols] + [np.ones(len(pl))])
    y = pl["actual_pts"].values.astype(float)
    coef, *_ = lstsq(X, y, rcond=None)
    print("  component            empirical pts   (our formula uses)")
    ours = {"appearance_pts": "1·appearance", "did_score": "4-6 by pos", "did_assist": "3",
            "did_sot2": "1 (bonus proxy)", "clean_sheet": "1-4 by pos"}
    for c, b in zip(X_cols, coef):
        print(f"  {c:20s} {b:+6.2f}          {ours.get(c,'')}")
    print(f"  (intercept {coef[-1]:+.2f})")

    # bias check per component (mean actual)
    print("\n  mean actual bonus:", round(pl["actual_bonus"].mean(), 3),
          "| our formula caps bonus at 3, drivers goals/assists/etc.")

    # ── 2. PROJECTION ACCURACY: model expected pts vs actual ──────────────────
    print("\n=== 2. PROJECTION ACCURACY (in-sample caveat) ===")
    payg, paya, pays = load_model("goals"), load_model("assists"), load_model("sot2")
    _missing = [n for n, m in (("goals", payg), ("assists", paya), ("sot2", pays)) if m is None]
    if _missing:
        # a missing model is a skipped metric, never a 0 probability
        print(f"  model(s) missing: {_missing} — skipping projection accuracy.")
    else:
        pl["p_goal"]   = predict_proba(pl, payg).values
        pl["p_assist"] = predict_proba(pl, paya).values
        pl["p_sot2"]   = predict_proba(pl, pays).values
        gp = pl["pos"].map(GOAL_PTS).astype(float)
        # ex-ante expected attacking points (the comparable, model-driven part)
        pl["exp_attack"] = pl["p_goal"] * gp + pl["p_assist"] * 3.0 + pl["p_sot2"] * 1.0
        pl["act_attack"] = pl["goal_pts_a"] + pl["assist_pts_a"]
        for label, sub in [("ALL", pl)] + [(p, pl[pl["pos"] == p]) for p in ("FWD", "MID", "DEF", "GKP")]:
            if len(sub) < 50:
                continue
            mae = float(np.abs(sub["exp_attack"] - sub["act_attack"]).mean())
            bias = float(sub["exp_attack"].mean() - sub["act_attack"].mean())
            sp = sub[["exp_attack", "act_attack"]].corr(method="spearman").iloc[0, 1]
            print(f"  {label:4s} n={len(sub):5d}  attack MAE={mae:.3f}  bias(exp-act)={bias:+.3f}  "
                  f"rank-corr={sp:+.3f}")

        # ── bonus scale calibration: our formula mean vs actual mean ──────────
        try:
            from player_model.fantasy import _expected_bonus
            if not {"tackles_pg", "interceptions_pg"}.issubset(pl.columns):
                raise KeyError("tackles_pg / interceptions_pg not in player_history")
            defpg = _num(pl["tackles_pg"]) + _num(pl["interceptions_pg"])
            if defpg.isna().any():
                raise ValueError(f"{int(defpg.isna().sum())} row(s) without defensive per-game stats")
            pl["our_bonus"] = [_expected_bonus(g, a, s, float(dd), 0.0) for g, a, s, dd in
                               zip(pl["p_goal"], pl["p_assist"], pl["p_sot2"], defpg)]
            om, am = float(pl["our_bonus"].mean()), float(pl["actual_bonus"].mean())
            print(f"\n  BONUS: our-formula mean={om:.3f} vs actual mean={am:.3f}  "
                  f"-> recommended scale x{am/max(om,1e-6):.2f}")
        except Exception as e:
            print("  bonus scale check skipped:", e)

    print("\n=== NOTES ===")
    print("  * appearance/goals/assists/CS/saves/cards = EXACT FPL rules.")
    print("  * bonus = BPS-proxy (missing recoveries/clearances/big-chances) = approximate.")
    print("  * defensive-contribution EXCLUDED (source lacks clearances/ball-recoveries).")
    print("  * projection accuracy is IN-SAMPLE (models trained on these rows) = optimistic;")
    print("    trust the calibration coefficients + rank-corr direction, not absolute MAE.")


if __name__ == "__main__":
    main()
