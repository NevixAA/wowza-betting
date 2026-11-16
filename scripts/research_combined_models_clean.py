"""
COMBINED-MODELS-DEV-2 (v9): does any model get a boost from the other models' out-of-sample output,
on 100% verified data only? Research only: nothing here is read by serving, nothing is promoted, no
API-Football or OddsAPI call is made.

Owner 2026-11-16: "for every model, try to use all the models' data to find if some model gets extra
boosts with the help of the other models" ... "but only on 100% verified data".

Protocol: config/experiments/combined_models/protocol_dev2.json (COMBINED-MODELS-DEV-2), committed
with its sha256 BEFORE any effect was computed. Report: reports/R_COMBINED_MODELS_CLEAN_2026-11-16.md.

Stages (inputs and intermediate files live in --work, OUTSIDE the repo; the repo receives counts and
estimates only - no price, no OddsAPI-derived value, no player row):

  verify     Per input: source commit / run, frame hash, provenance, parity, missing-value policy.
             Writes work/verification.json. Refuses to continue (exit 2) when a required input fails.
  props-wf   Out-of-sample props probabilities of the CLEAN-SCHEMA candidate (run 43745808793,
             hgb_nan_native, 125 features, rule all_prior_verified) refitted on the first day of every
             month 2025-01..2026-10 on every eligible row dated before it, from the run's own frame
             (hash 4f6c603d00c14567, checked). The 2026-09-01 refit must reproduce the run's own
             holdout probabilities (checked, written to work/props_wf_repro.json).
  join       Fixture table (team Standard / New-Format champion_recipe, Pro Dixon-Coles, bet-time
             market) and player table (props + helpers), keyed by the crosswalk of DEV-1.
  analyse    Every registered cell: residual tests and stacks, per league, one Holm family.
  report-data  Counts and estimates into reports/combined_models_clean_data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import research_combined_models as CM  # noqa: E402  (DEV-1 helpers: holm, bootstrap, crosswalk)

PROTOCOL_PATH = ROOT / "config" / "experiments" / "combined_models" / "protocol_dev2.json"
REPORT_DATA = ROOT / "reports" / "combined_models_clean_data"

#: the clean-schema props candidate (run 43745808793) and its frame
PROPS_RUN = "43745808793"
PROPS_FRAME_HASH = "4f6c603d00c14567"
PROPS_FEATURES_SHA = "3f80d914724df7bc891bb79dda8bb3b9dafaf1afb2f6c021036c0aa263736404"
PROPS_CANDIDATE = "hgb_nan_native"
PROPS_MARKETS = ("sot", "sot2", "goals", "assists", "cards")
PROPS_WF_MONTHS = tuple(pd.date_range("2025-01-01", "2026-10-01", freq="MS"))
PROPS_HOLDOUT_START = pd.Timestamp("2026-09-01")

#: DEV-2 windows (protocol): stacks and tercile cutoffs learn from TRAIN_START.., cells are scored on
#: EVAL_START..EVAL_END only (months no combined-model test has scored: DEV-1 stopped at 2026-03-31)
TRAIN_START = pd.Timestamp("2025-01-01")
EVAL_START = pd.Timestamp("2026-04-01")
EVAL_END = pd.Timestamp("2026-10-31")
TEAM_HOLDOUT_START = pd.Timestamp("2026-04-01")
TEAM_MARKETS = ("over15", "over25", "over35", "btts")
PRO_MARKETS = ("ge6", "tot_le1", "nil_nil")


def frame_hash(df: pd.DataFrame) -> str:
    """The props harness's frame hash: sha256 of pandas' row hashes (index excluded), first 16 hex."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── props walk-forward (clean-schema candidate) ──────────────────────────────────────────────────

def props_wf_clean(frame_path: Path, work: Path, months=PROPS_WF_MONTHS, markets=PROPS_MARKETS,
                   holdout_csv: Path | None = None, log=print) -> dict:
    """Monthly refits of the clean-schema candidate on its own verified frame. Each (market, month)
    part is saved as it finishes (resumable). Refuses a frame whose hash or feature list differs."""
    import os
    from player_model import props_harness as PH
    if os.environ.get(PH.ABLATION_ENV, "").strip():
        raise RuntimeError(f"{PH.ABLATION_ENV} is set: the clean-schema candidate has no ablation")
    f = pd.read_parquet(frame_path)
    fh = frame_hash(f)
    if fh != PROPS_FRAME_HASH:
        raise ValueError(f"props frame hash {fh} != run {PROPS_RUN}'s {PROPS_FRAME_HASH}: refused")
    dec = PH.owner_decisions(PH.load_plan())
    ok = PH.eligible(f, dec["feature_rule"]).to_numpy()
    elig = f[ok]
    del f
    features = PH.harness_features(elig)
    fsha = hashlib.sha256(json.dumps(features).encode()).hexdigest()
    if fsha != PROPS_FEATURES_SHA:
        raise ValueError(f"feature list sha {fsha} != run {PROPS_RUN}'s {PROPS_FEATURES_SHA}: refused")
    parts_dir = work / "props_wf_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    d = pd.to_datetime(elig["date"]).dt.normalize()
    keep = [c for c in ("date", "fixture_id", "player_id", "league", "home_team", "away_team", "is_home")
            if c in elig.columns]
    for market in markets:
        y_col = PH.TARGETS[market]
        lab = elig[y_col].notna().to_numpy()
        for m0 in months:
            out_p = parts_dir / f"{market}_{m0.date()}.parquet"
            if out_p.exists():
                continue
            m1 = m0 + pd.offsets.MonthBegin(1)
            tr = elig[(d < m0).to_numpy() & lab]
            te = elig[((d >= m0) & (d < m1)).to_numpy() & lab]
            if te.empty:
                continue
            fitted = PH.fit_window(tr, features, y_col, names=(PROPS_CANDIDATE,))
            p = fitted[PROPS_CANDIDATE].predict(te[features])
            out = te[keep].copy()
            out["market"] = market
            out["y"] = te[y_col].to_numpy(float)
            out["p_props"] = p
            out["fit_through"] = str((m0 - pd.Timedelta(days=1)).date())
            out["train_rows"] = len(tr)
            out.to_parquet(out_p)
            log(f"[props-wf] {market} {m0.date()}: train {len(tr)} test {len(te)}")
    res = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))], ignore_index=True)
    res.to_parquet(work / "props_wf.parquet")
    meta = {"run": PROPS_RUN, "candidate": PROPS_CANDIDATE, "frame_hash": fh, "features_n": len(features),
            "features_sha256": fsha, "feature_rule": dec["feature_rule"], "eligible_rows": int(len(elig)),
            "months": [str(m.date()) for m in months], "markets": list(markets), "rows": int(len(res))}
    if holdout_csv is not None and Path(holdout_csv).exists():
        meta["reproduction"] = reproduce_holdout(res, Path(holdout_csv))
    (work / "props_wf_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def reproduce_holdout(res: pd.DataFrame, holdout_csv: Path) -> dict:
    """The 2026-09-01 refit is the run's own holdout fit (same rows, same recipe, deterministic): its
    September probabilities must equal the run's p_hgb_nan_native on the same rows."""
    h = pd.read_csv(holdout_csv, usecols=["market", "date", "fixture_id", "player_id", "y", "p_hgb_nan_native"])
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    h = h[(h["date"] >= PROPS_HOLDOUT_START) & (h["date"] < PROPS_HOLDOUT_START + pd.offsets.MonthBegin(1))]
    r = res[res["fit_through"] == str((PROPS_HOLDOUT_START - pd.Timedelta(days=1)).date())]
    m = r.merge(h, on=["market", "fixture_id", "player_id"], how="inner", suffixes=("", "_run"))
    out = {}
    for mk, g in m.groupby("market"):
        out[mk] = {"rows_run_sep": int((h["market"] == mk).sum()), "rows_matched": int(len(g)),
                   "max_abs_diff_p": float(np.max(np.abs(g["p_props"] - g["p_hgb_nan_native"]))) if len(g) else None,
                   "label_disagreements": int((g["y"] != g["y_run"]).sum())}
    return out


# ── verification ─────────────────────────────────────────────────────────────────────────────────

#: the Standard team run's frame: every other input must come from the same repaired data
TEAM_FRAME_HASH = "974e04bc77c4d438"
TEAM_LOADED_ROWS = 71583


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def feature_classes(features: list[str], registry: dict) -> dict:
    """Worst real-data-provenance class of every feature over the registry's ACTIVE team artifacts
    (config/feature_provenance.json of data-truth-100). Missing-indicator columns are checked through
    their base column. Returns {feature: class or 'NOT_REGISTERED'} and the ineligible ones."""
    eligible = set(registry["eligible"])
    by_feat: dict = {}
    for r in registry["rows"]:
        if not str(r.get("status", "")).startswith("active"):
            continue
        by_feat.setdefault(r["feature"], set()).add(r["classification"])
    out = {}
    for f in features:
        base = f[:-len("__missing")] if f.endswith("__missing") else f
        cls = by_feat.get(base)
        if not cls:
            out[f] = "NOT_REGISTERED"
        else:
            bad = sorted(c for c in cls if c not in eligible)
            out[f] = bad[0] if bad else sorted(cls)[0]
    return {"classes": out, "ineligible": {f: c for f, c in out.items() if c not in eligible}}


def verify_team(run_dir: Path, track: str, registry: dict) -> dict:
    m = _manifest(run_dir)
    d = m["data"]
    feats = m["features"][track]["features"]
    cov = [c for c in m["coverage"] if c["candidate"] == "champion_recipe"]
    fc = feature_classes(feats, registry)
    rec = {"input": f"team {track} champion_recipe", "run_id": m["run_id"], "code_sha": m["code_sha"],
           "config_sha256": m["config_sha256"], "frame_hash": d["frame_hash"], "loaded_rows": d["loaded_rows"],
           "feature_rows": d["feature_rows"], "covid_rows_excluded": d["covid_rows_excluded"],
           "holdout": m["config"]["holdout"]["start"] + ".." + m["config"]["holdout"]["end"],
           "features": len(feats), "features_ineligible_or_unregistered": fc["ineligible"],
           "folds_error": int(sum(c["folds_error"] for c in cov)),
           "folds_not_fitted": int(sum(c["folds_not_fitted"] for c in cov)),
           "dropped_no_training_value": sorted({e["note"] for e in m["errors"]
                                                if e.get("candidate") == "champion_recipe"})[:6],
           "missing_policy": m["config"]["candidates"]["champion_recipe"]["what"],
           "require_present": m["config"]["data"]["require_present"]}
    ok = (d["frame_hash"] == TEAM_FRAME_HASH and not fc["ineligible"] and rec["folds_error"] == 0)
    rec["status"] = "PASS" if ok else "FAIL"
    rec["fail_reasons"] = ([] if d["frame_hash"] == TEAM_FRAME_HASH else
                           [f"frame {d['frame_hash']} != the latest repaired frame {TEAM_FRAME_HASH}"]) + \
        ([f"ineligible / unregistered features {fc['ineligible']}"] if fc["ineligible"] else []) + \
        ([f"{rec['folds_error']} fold error(s)"] if rec["folds_error"] else [])
    return rec


def verify_pro(work: Path) -> dict:
    info = json.loads((work / "v9_history_info.json").read_text())
    dc = pd.read_parquet(work / "dc_wf.parquet", columns=["league", "date", "lam_home", "lam_away"])
    total = info["rows"] + info["rows_dropped_incomplete"]
    rec = {"input": "Pro Dixon-Coles (SPC-1)", "pro_commit": "7344ddb1 (wowzaV9-Pro data-truth-100)",
           "v9_export_commit": info["v9_sha"], "v9_rows": info["rows"],
           "v9_rows_dropped_without_score": info["rows_dropped_incomplete"],
           "v9_history_sha256": info.get("sha256_parquet"), "predictions": int(len(dc)),
           "leagues": int(dc["league"].nunique()),
           "months": f"{pd.to_datetime(dc['date']).min().date()}..{pd.to_datetime(dc['date']).max().date()}",
           "missing_policy": "a match without a 90-minute score is dropped and counted (Pro history.clean_history); "
                             "a club / league without enough earlier matches gets no lambda (no fallback)"}
    ok = total == TEAM_LOADED_ROWS
    rec["status"] = "PASS" if ok else "FAIL"
    rec["fail_reasons"] = [] if ok else [f"export rows {total} != the Standard frame's loaded rows {TEAM_LOADED_ROWS}"]
    return rec


def verify_props(work: Path) -> dict:
    meta = json.loads((work / "props_wf_meta.json").read_text())
    rep = meta.get("reproduction") or {}
    worst = max([v["max_abs_diff_p"] for v in rep.values() if v.get("max_abs_diff_p") is not None], default=np.nan)
    eq_p = work / "props_equivalence.json"
    eq = json.loads(eq_p.read_text()) if eq_p.exists() else {}
    rec = {"input": "props clean-schema candidate (hgb_nan_native)", "run": PROPS_RUN, "code_sha": "d051ee5c7",
           "frame_hash": meta["frame_hash"], "features": meta["features_n"],
           "features_sha256": meta["features_sha256"], "feature_rule": meta["feature_rule"],
           "eligible_rows": meta["eligible_rows"], "oos_rows": meta["rows"],
           "oos_window": f"{meta['months'][0]}..2026-10-31 (monthly refits)",
           "bit_reproduction_of_run_holdout_sep (original rule, failed; amendment A1)": rep,
           "bit_reproduction_max_abs_diff": worst, "equivalence_A1": eq,
           "missing_policy": "NaN-native HistGradientBoosting: no value is filled at train or score time; "
                             "provider nulls stay missing (Data truth 7c / 7e)"}
    ok = (meta["frame_hash"] == PROPS_FRAME_HASH and meta["features_sha256"] == PROPS_FEATURES_SHA
          and set(eq) == set(PROPS_MARKETS) and all(v["pass"] for v in eq.values()))
    rec["status"] = "PASS" if ok else "FAIL"
    rec["fail_reasons"] = [] if ok else ["frame / features / A1 equivalence check failed"]
    rec["note"] = ("local monthly refits of the clean recipe on the run's verified frame: equivalent to the "
                   "runner's fit (amendment A1), not bit-identical (HGB binning subsample depends on row order)")
    return rec


def side_models_note(remote: str = "origin", branch: str = "research/side-models") -> dict:
    """The side models' cells run only when a docs/ note with the prediction paths exists on
    origin/research/side-models. Reads the remote; never substitutes anything."""
    import subprocess
    r = subprocess.run(["git", "ls-remote", remote, f"refs/heads/{branch}"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=120)
    head = r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None
    rec = {"input": "side models (corners / cards / specials)", "branch": f"{remote}/{branch}", "head": head}
    if head is None:
        rec.update(status="PENDING - NOT RUN", fail_reasons=[f"{branch} is not on {remote}: no note, no inputs"])
        return rec
    subprocess.run(["git", "fetch", "-q", remote, f"{branch}:refs/remotes/{remote}/{branch}"], cwd=str(ROOT),
                   capture_output=True, text=True, timeout=300)
    d = subprocess.run(["git", "diff", "--name-only", f"{remote}/data-truth-100", f"{remote}/{branch}"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=120).stdout.split()
    notes = [f for f in d if f.startswith("docs/")]
    rec["files_added_vs_data_truth_100"] = d
    rec["docs_notes"] = notes
    if not notes:
        rec.update(status="PENDING - NOT RUN",
                   fail_reasons=["no docs/ note with prediction paths on the branch (only " + ", ".join(d) + ")"])
    else:
        rec.update(status="PENDING - NOT RUN", fail_reasons=["docs/ note present: its inputs need verification "
                                                             "before any side cell is run"])
    return rec


# ── join ─────────────────────────────────────────────────────────────────────────────────────────

def load_team_track(pred_path: Path, track: str) -> pd.DataFrame:
    """One row per fixture: champion_recipe p_<m>, the BET-time market m_<m> (missing without a
    two-sided pair), labels y_<m>. Walk-forward rows before 2026-04-01 (a match scored by two folds
    keeps the earlier fold), holdout rows from 2026-04-01. Kept: TRAIN_START..EVAL_END."""
    d = pd.read_parquet(pred_path, columns=["stage", "fold", "market", "candidate", "row_id", "date", "league",
                                            "y", "p"])
    d = d[d["candidate"].isin(["champion_recipe", "market_bet"]) & d["market"].isin(TEAM_MARKETS)]
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    wf = d[(d["stage"] == "walk_forward") & (d["date"] < TEAM_HOLDOUT_START)]
    wf = wf.sort_values(["row_id", "market", "candidate", "fold"]).drop_duplicates(
        ["row_id", "market", "candidate"], keep="first")
    ho = d[(d["stage"] == "holdout") & (d["date"] >= TEAM_HOLDOUT_START)]
    t = pd.concat([wf, ho], ignore_index=True)
    t = t[(t["date"] >= TRAIN_START) & (t["date"] <= EVAL_END)]
    p = t[t["candidate"] == "champion_recipe"].pivot_table(index=["row_id", "date", "league"], columns="market",
                                                           values="p").add_prefix("p_")
    y = t[t["candidate"] == "champion_recipe"].pivot_table(index=["row_id", "date", "league"], columns="market",
                                                           values="y").add_prefix("y_")
    m = t[t["candidate"] == "market_bet"].pivot_table(index=["row_id", "date", "league"], columns="market",
                                                      values="p").add_prefix("m_")
    w = p.join(y, how="left").join(m, how="left").reset_index()
    for k in TEAM_MARKETS:                     # a market with no value at all is a missing column, not absent
        for pre in ("p_", "y_", "m_"):
            if pre + k not in w.columns:
                w[pre + k] = np.nan
    sp = w["row_id"].str.split("|", expand=True)
    w["home_team"], w["away_team"] = sp[2], sp[3]
    w["track"] = track
    w["team_stage"] = np.where(w["date"] < TEAM_HOLDOUT_START, "walk_forward", "holdout")
    return w


def pro_same_market(dc: pd.DataFrame) -> pd.DataFrame:
    """Pro's same-market probabilities from its own score matrix (columns written by SPC-1)."""
    o = pd.DataFrame(index=dc.index)
    o["pro_over15"] = 1 - dc["p_tot_0_1"]
    o["pro_over25"] = 1 - dc["p_tot_0_1"] - (dc["p_cs_0_2"] + dc["p_cs_1_1"] + dc["p_cs_2_0"])
    o["pro_over35"] = dc["p_tot_4_5"] + dc["p_tot_6p"]
    o["pro_btts"] = sum(dc[f"p_cs_{i}_{j}"] for i in range(1, 7) for j in range(1, 7))
    return o


def implied_from_total(p15, p25, p35) -> dict:
    """The Poisson total that best fits P(O1.5), P(O2.5), P(O3.5) (DEV-1 poisson_total_from_over: the
    same grid 0.3..6.0 step 0.01 and least squares, vectorised) and its P(total >= 6) and
    P(total = 0); P(total <= 1) = 1 - P(O1.5). NaN where any input is missing."""
    from scipy.stats import poisson
    ps = np.column_stack([np.asarray(p15, float), np.asarray(p25, float), np.asarray(p35, float)])
    grid = np.linspace(0.3, 6.0, 571)
    pred = np.stack([poisson.sf(k, grid) for k in (1, 2, 3)], axis=1)          # (571, 3)
    ok = np.isfinite(ps).all(axis=1)
    mu = np.full(len(ps), np.nan)
    for i in range(0, len(ps), 20000):
        blk = ps[i:i + 20000]
        err = ((blk[:, None, :] - pred[None, :, :]) ** 2).sum(axis=2)
        mu[i:i + 20000] = grid[np.argmin(np.where(np.isfinite(err), err, np.inf), axis=1)]
    mu[~ok] = np.nan
    return {"ge6": poisson.sf(5, mu), "nil_nil": np.exp(-mu), "tot_le1": np.where(ok, 1 - ps[:, 0], np.nan),
            "mu": mu}


def build_fixtures(work: Path, team_paths: dict) -> pd.DataFrame:
    team = pd.concat([load_team_track(Path(p), tr) for tr, p in team_paths.items()], ignore_index=True)
    dc = pd.read_parquet(work / "dc_wf.parquet")
    dc["date"] = pd.to_datetime(dc["date"]).dt.normalize()
    dc = dc[(dc["date"] >= TRAIN_START) & (dc["date"] <= EVAL_END)].reset_index(drop=True)
    pro = pd.concat([dc[["league", "date", "home_team", "away_team", "home_goals", "away_goals", "lam_home",
                         "lam_away", "p_ge6", "p_tot_le1", "p_nil_nil", "y_ge6", "y_tot_le1", "y_nil_nil"]],
                     pro_same_market(dc)], axis=1)
    keys = ["league", "date", "home_team", "away_team"]
    f = pro.merge(team, on=keys, how="outer", indicator="src")
    f["fixture_key"] = f["date"].dt.strftime("%Y-%m-%d") + "|" + f["league"] + "|" + f["home_team"] + "|" + f["away_team"]
    f["month"] = f["date"].dt.to_period("M").astype(str)
    for who, cols in (("team", ("p_over15", "p_over25", "p_over35")), ("mkt", ("m_over15", "m_over25", "m_over35"))):
        imp = implied_from_total(*(f[c].to_numpy(float) for c in cols))
        for k in PRO_MARKETS:
            f[f"{who}_{k}"] = imp[k]
    return f


def build_players(work: Path, fx: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict]:
    """Props OOS rows joined to the fixture table through DEV-1's crosswalk (v9 names <-> API-Football)."""
    v9 = pd.read_parquet(work / "v9_history.parquet")
    v9["date"] = pd.to_datetime(v9["date"]).dt.normalize()
    hist = pd.read_parquet(ROOT / "player_history.parquet",
                           columns=["fixture_id", "date", "league", "home_team", "away_team", "team", "position",
                                    "goals_conceded", "goals"])
    af = CM.af_fixtures(hist)
    del hist
    xw = CM.learn_crosswalk(v9, af)
    mp = CM.map_fixtures(v9[v9["date"] >= TRAIN_START - pd.Timedelta(days=1)], af, xw)
    mp["fixture_key"] = mp["date"].dt.strftime("%Y-%m-%d") + "|" + mp["league"] + "|" + mp["home_team"] + "|" + mp["away_team"]
    props = pd.read_parquet(work / "props_wf.parquet")
    props["date"] = pd.to_datetime(props["date"]).dt.normalize()
    props["fixture_id"] = pd.to_numeric(props["fixture_id"], errors="coerce")
    link = mp[mp["fixture_id"].notna()][["fixture_id", "fixture_key", "map_status"]].drop_duplicates("fixture_id")
    j = props.merge(link, on="fixture_id", how="left")
    cols = ["fixture_key", "track", "p_over15", "p_over25", "p_over35", "p_btts", "m_over25", "lam_home", "lam_away",
            "team_stage"]
    j = j.merge(fx[cols].drop_duplicates("fixture_key"), on="fixture_key", how="left")
    home = pd.to_numeric(j["is_home"], errors="coerce")
    j["dc_lam_side"] = np.where(home == 1, j["lam_home"], np.where(home == 0, j["lam_away"], np.nan))
    j["dc_lam_opp"] = np.where(home == 1, j["lam_away"], np.where(home == 0, j["lam_home"], np.nan))
    j["month"] = j["date"].dt.to_period("M").astype(str)
    info = {"crosswalk_pairs": int(len(xw)), "v9_fixtures_mapped": mp["map_status"].value_counts().to_dict(),
            "props_rows": int(len(props)), "props_rows_with_fixture": int(j["fixture_key"].notna().sum()),
            "props_rows_with_team": int(j["p_over25"].notna().sum()),
            "props_rows_with_pro": int(j["lam_home"].notna().sum()),
            "props_rows_with_market": int(j["m_over25"].notna().sum())}
    log(json.dumps(info))
    return j, info


def props_feature_side_check(frame_path: Path, players: pd.DataFrame) -> pd.DataFrame:
    """Does team_goals_pg_roll now read the player's OWN side? Spearman vs own / opponent DC lambda
    (DEV-1 measured -0.03 own / 0.35 opponent on the buggy feature)."""
    from scipy.stats import spearmanr
    want = ["fixture_id", "player_id", "team_goals_pg_roll", "prev_season_team_goals_pg", "opp_goals_conceded_pg",
            "team_momentum_forward_matchup"]
    import pyarrow.parquet as pq
    have = pq.ParquetFile(frame_path).schema.names
    f = pd.read_parquet(frame_path, columns=[c for c in want if c in have])
    s = players[(players["market"] == "sot") & players["dc_lam_side"].notna()][["fixture_id", "player_id", "is_home",
                                                                                 "dc_lam_side", "dc_lam_opp"]]
    s = s.merge(f, on=["fixture_id", "player_id"], how="left").drop_duplicates(["fixture_id", "is_home"])
    rows = []
    for c in want[2:]:
        if c not in s.columns:
            rows.append({"feature": c, "present_in_clean_schema": False})
            continue
        ok = s[[c, "dc_lam_side", "dc_lam_opp"]].dropna()
        rows.append({"feature": c, "present_in_clean_schema": True, "n_team_fixtures": int(len(ok)),
                     "spearman_vs_own_lambda": float(spearmanr(ok[c], ok["dc_lam_side"]).statistic),
                     "spearman_vs_opp_lambda": float(spearmanr(ok[c], ok["dc_lam_opp"]).statistic)})
    return pd.DataFrame(rows)


# ── analysis ─────────────────────────────────────────────────────────────────────────────────────

MIN_PRIOR_FIX = 200
MIN_TRAIN_FIX, MIN_EVAL_FIX, MIN_TRAIN_EVENTS, MIN_EVAL_EVENTS = 300, 150, 30, 10
CLIP = 1e-4


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def logloss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def eval_months() -> list[str]:
    return [str(m) for m in pd.period_range(EVAL_START, EVAL_END, freq="M")]


def tercile_groups(values: pd.Series, league: pd.Series, month: pd.Series,
                   min_prior: int = MIN_PRIOR_FIX) -> np.ndarray:
    """1 top / 0 bottom / -1 middle / NaN (no cutoff or no value), cutoffs from the league's
    strictly earlier months (DEV-1 prior_cutoff), learning rows from TRAIN_START only."""
    lo = CM.prior_cutoff(values, league, month, 1 / 3, min_prior)
    hi = CM.prior_cutoff(values, league, month, 2 / 3, min_prior)
    v = pd.to_numeric(values, errors="coerce").to_numpy(float)
    g = np.where(v >= hi, 1.0, np.where(v <= lo, 0.0, -1.0))
    g[~np.isfinite(lo) | ~np.isfinite(v)] = np.nan
    return g


def _fit_lr(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit((X - mu) / sd, y.astype(int))
    return lambda Z: lr.predict_proba((Z - mu) / sd)[:, 1]


def stack_predictions(df: pd.DataFrame, target_x: list[str], helper_x: list[str], gbm: bool = False) -> pd.DataFrame:
    """Per scored month M: baseline (target alone, recalibrated) and stack fitted on rows dated
    TRAIN_START..M-1, scored on month M. df: one league's rows with columns month, y, cluster and the
    feature columns (complete cases). Returns the scored rows with p_base, p_stack (and p_gbm)."""
    out = []
    for m in eval_months():
        tr = df[df["month"] < m]
        te = df[df["month"] == m]
        if te.empty or len(tr) == 0 or tr["y"].nunique() < 2:
            continue
        rec = te[["cluster", "y", "month", "p_target"]].copy()
        rec["train_fix"] = tr["cluster"].nunique()
        rec["train_events"] = float(tr["y"].sum())
        rec["p_base"] = _fit_lr(tr[target_x].to_numpy(float), tr["y"].to_numpy())(te[target_x].to_numpy(float))
        X = target_x + helper_x
        rec["p_stack"] = _fit_lr(tr[X].to_numpy(float), tr["y"].to_numpy())(te[X].to_numpy(float))
        if gbm:
            from sklearn.ensemble import HistGradientBoostingClassifier
            g = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=50,
                                               random_state=42).fit(tr[X].to_numpy(float), tr["y"].astype(int))
            rec["p_gbm"] = g.predict_proba(te[X].to_numpy(float))[:, 1]
        out.append(rec)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def stack_cell(df: pd.DataFrame, target_x: list[str], helper_x: list[str], reps: int, seed: int,
               gbm: bool = False) -> tuple[dict, pd.DataFrame]:
    sc = stack_predictions(df, target_x, helper_x, gbm=gbm)
    rec = {"rows_scored": int(len(sc)), "fix_scored": int(sc["cluster"].nunique()) if len(sc) else 0,
           "events_scored": float(sc["y"].sum()) if len(sc) else 0.0,
           "train_fix_first": int(sc["train_fix"].iloc[0]) if len(sc) else 0,
           "train_events_first": float(sc["train_events"].iloc[0]) if len(sc) else 0.0}
    rec["readable"] = bool(rec["train_fix_first"] >= MIN_TRAIN_FIX and rec["fix_scored"] >= MIN_EVAL_FIX
                           and rec["train_events_first"] >= MIN_TRAIN_EVENTS and rec["events_scored"] >= MIN_EVAL_EVENTS)
    nan = {"effect": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se": np.nan, "p": np.nan}
    if not len(sc):
        rec.update(nan)
        return rec, sc
    y = sc["y"].to_numpy(float)
    ll_b, ll_s, ll_raw = logloss(y, sc["p_base"]), logloss(y, sc["p_stack"]), logloss(y, sc["p_target"])
    rec.update(ll_base=float(ll_b.mean()), ll_stack=float(ll_s.mean()), ll_raw_target=float(ll_raw.mean()),
               brier_gain=float(((sc["p_base"] - y) ** 2 - (sc["p_stack"] - y) ** 2).mean()),
               gain_vs_raw_target=float((ll_raw - ll_s).mean()))
    if gbm and "p_gbm" in sc:
        rec["gbm_gain_vs_base_context"] = float((ll_b - logloss(y, sc["p_gbm"])).mean())
    if rec["readable"]:
        b = CM.cluster_bootstrap_mean(ll_b - ll_s, sc["cluster"].to_numpy(), reps, seed)
        rec.update(effect=b["est"], ci_lo=b["lo"], ci_hi=b["hi"], se=b["se"], p=b["p_one"])
    else:
        rec.update(nan)
    return rec, sc


def residual_cell(df: pd.DataFrame, grp: str, reps: int, seed: int) -> dict:
    h = df[df[grp].isin([0, 1])].drop(columns=["fixture_id"], errors="ignore").rename(columns={"cluster": "fixture_id"})
    rec = CM._cell(h, "res", grp, reps, seed, one_sided=False)
    return rec


def team_pro_frames(fx: pd.DataFrame) -> list[dict]:
    """The registered (target, helper) blocks of fixture-level targets: team x {pro, market} and
    pro x {team, market}. Each block: rows with y, p_target and the helper probability."""
    blocks = []
    tm = fx[fx["p_over25"].notna()]
    for k in TEAM_MARKETS:
        for hname, hcol in (("pro", f"pro_{k}"), ("market", f"m_{k}")):
            blocks.append({"group": "team", "market": k, "helper": hname, "y": f"y_{k}", "p": f"p_{k}",
                           "hcols": [hcol], "rows": tm})
    pm = fx[fx["lam_home"].notna()]
    for k in PRO_MARKETS:
        for hname, hcol in (("team", f"team_{k}"), ("market", f"mkt_{k}")):
            blocks.append({"group": "pro", "market": k, "helper": hname, "y": f"y_{k}", "p": f"p_{k}",
                           "hcols": [hcol], "rows": pm})
    return blocks


PROPS_HELPERS = {"team": (["p_over15", "p_over25", "p_over35", "p_btts"], "p_over25"),
                 "pro": (["dc_lam_side", "dc_lam_opp"], "dc_lam_side"),
                 "market": (["m_over25"], "m_over25")}


def _helper_features(d: pd.DataFrame, cols: list[str]) -> list[str]:
    names = []
    for c in cols:
        n = f"x_{c}"
        d[n] = np.log(d[c].astype(float)) if c.startswith("dc_lam") else logit(d[c])
        names.append(n)
    return names


def run_analysis(fx: pd.DataFrame, pl: pd.DataFrame, reps: int = 2000, seed: int = 7, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every registered cell, per league (in family) + pooled (context). Returns (tests, stack rows)."""
    out, stack_rows = [], []
    ev0 = str(EVAL_START.to_period("M"))

    def emit(base: dict, league: str, rec: dict):
        out.append({**base, "league": league, "in_family": league != "ALL (context)", **rec})

    # fixture-level targets (team, pro)
    for b in team_pro_frames(fx):
        d = b["rows"]
        d = d[d[b["y"]].notna() & d[b["p"]].notna()].copy()
        d["y"], d["p_target"], d["cluster"] = d[b["y"]].astype(float), d[b["p"]].astype(float), d["fixture_key"]
        d["res"] = d["y"] - d["p_target"]
        h = b["hcols"][0]
        d["x_t"] = logit(d["p_target"])
        dd = d[d[h].notna()].copy()
        dd["x_h"] = logit(dd[h])
        dd["disagree"] = dd["x_h"] - dd["x_t"]
        dd = dd.sort_values("date")
        dd["grp"] = tercile_groups(dd["disagree"], dd["league"], dd["month"])
        base = {"target_group": b["group"], "market": b["market"], "helper": b["helper"]}
        sc_ev = dd[dd["month"] >= ev0]
        for lg in sorted(sc_ev["league"].unique()):
            emit({**base, "test": "residual"}, lg, residual_cell(sc_ev[sc_ev["league"] == lg], "grp", reps, seed))
            rec, sc = stack_cell(dd[dd["league"] == lg], ["x_t"], ["x_h"], reps, seed)
            emit({**base, "test": "stack"}, lg, rec)
            if len(sc):
                stack_rows.append(sc.assign(league=lg, **base, test="stack"))
        emit({**base, "test": "residual"}, "ALL (context)", residual_cell(sc_ev, "grp", reps, seed))
        log(f"[analyse] {b['group']} {b['market']} x {b['helper']}: {sc_ev['league'].nunique()} leagues")
    # stack-all for fixture-level targets (complete cases over the block's helpers)
    for group, markets, hmap, rowsel in (
            ("team", TEAM_MARKETS, lambda k: {"pro": f"pro_{k}", "market": f"m_{k}"}, fx["p_over25"].notna()),
            ("pro", PRO_MARKETS, lambda k: {"team": f"team_{k}", "market": f"mkt_{k}"}, fx["lam_home"].notna())):
        for k in markets:
            hc = hmap(k)
            d = fx[rowsel & fx[f"y_{k}"].notna() & fx[f"p_{k}"].notna()].copy()
            d = d[d[list(hc.values())].notna().all(axis=1)]
            d["y"], d["p_target"], d["cluster"] = d[f"y_{k}"].astype(float), d[f"p_{k}"].astype(float), d["fixture_key"]
            d["x_t"] = logit(d["p_target"])
            hx = []
            for n, c in hc.items():
                d[f"x_{n}"] = logit(d[c])
                hx.append(f"x_{n}")
            base = {"target_group": group, "market": k, "helper": "stack_all", "test": "stack"}
            for lg in sorted(d.loc[d["month"] >= ev0, "league"].unique()):
                rec, sc = stack_cell(d[d["league"] == lg].sort_values("date"), ["x_t"], hx, reps, seed, gbm=True)
                emit(base, lg, rec)
                if len(sc):
                    stack_rows.append(sc.assign(league=lg, **base))
    # props targets
    for k in PROPS_MARKETS:
        m = pl[(pl["market"] == k) & pl["p_props"].notna() & pl["y"].notna() & pl["fixture_key"].notna()].copy()
        m["p_target"], m["cluster"] = m["p_props"].astype(float), m["fixture_key"]
        m["res"] = m["y"] - m["p_target"]
        m["x_t"] = logit(m["p_target"])
        m = m.sort_values("date")
        allx = []
        for hname, (cols, cond) in PROPS_HELPERS.items():
            allx += _helper_features(m, cols)
        for hname, (cols, cond) in PROPS_HELPERS.items():
            hx = [f"x_{c}" for c in cols]
            d = m[m[cols].notna().all(axis=1)].copy()
            base = {"target_group": "props", "market": k, "helper": hname}
            d["grp"] = d.pipe(_props_groups, cond)
            ev = d[d["month"] >= ev0]
            for lg in sorted(ev["league"].unique()):
                emit({**base, "test": "residual"}, lg, residual_cell(ev[ev["league"] == lg], "grp", reps, seed))
                rec, sc = stack_cell(d[d["league"] == lg], ["x_t"], hx, reps, seed)
                emit({**base, "test": "stack"}, lg, rec)
                if len(sc):
                    stack_rows.append(sc.assign(league=lg, **base, test="stack"))
            emit({**base, "test": "residual"}, "ALL (context)", residual_cell(ev, "grp", reps, seed))
            log(f"[analyse] props {k} x {hname}: {ev['league'].nunique()} leagues")
        allc = sum((c for c, _ in PROPS_HELPERS.values()), [])
        d = m[m[allc].notna().all(axis=1)]
        base = {"target_group": "props", "market": k, "helper": "stack_all", "test": "stack"}
        for lg in sorted(d.loc[d["month"] >= ev0, "league"].unique()):
            rec, sc = stack_cell(d[d["league"] == lg], ["x_t"], allx, reps, seed, gbm=True)
            emit(base, lg, rec)
            if len(sc):
                stack_rows.append(sc.assign(league=lg, **base))
    res = pd.DataFrame(out)
    # pooled stack context: every league's scored rows together (each scored by its own league's stack)
    sr = pd.concat(stack_rows, ignore_index=True) if stack_rows else pd.DataFrame()
    pooled = []
    if len(sr):
        for (g, k, h), s in sr.groupby(["target_group", "market", "helper"]):
            y = s["y"].to_numpy(float)
            dl = logloss(y, s["p_base"]) - logloss(y, s["p_stack"])
            bb = CM.cluster_bootstrap_mean(dl, s["cluster"].to_numpy(), reps, seed)
            pooled.append({"target_group": g, "market": k, "helper": h, "test": "stack", "league": "ALL (context)",
                           "in_family": False, "rows_scored": int(len(s)), "fix_scored": int(s["cluster"].nunique()),
                           "effect": bb["est"], "ci_lo": bb["lo"], "ci_hi": bb["hi"], "se": bb["se"], "p": bb["p_one"]})
    res = pd.concat([res, pd.DataFrame(pooled)], ignore_index=True)
    fam = res["in_family"].astype(bool).to_numpy()
    p = res.loc[fam, "p"].to_numpy(float)
    p = np.where(res.loc[fam, "readable"].astype(bool).to_numpy() & np.isfinite(p), p, 1.0)
    res["p_holm"] = np.nan
    res.loc[fam, "p_holm"] = CM.holm(p)
    ci_ok = np.where(res["test"] == "stack", res["ci_lo"] > 0, (res["ci_lo"] > 0) | (res["ci_hi"] < 0))
    res["signal"] = res["in_family"].astype(bool) & res["readable"].fillna(False).astype(bool) \
        & (res["p_holm"] < 0.05) & ci_ok
    return res, sr


def _props_groups(d: pd.DataFrame, cond: str) -> np.ndarray:
    """Props conditioner terciles: fixture-level (team / market P(O2.5)) or side-level (own lambda),
    cutoffs from the league's earlier fixtures / sides only (each fixture or side counted once)."""
    if cond == "dc_lam_side":
        unit = d.drop_duplicates(["fixture_key", "is_home"])[["fixture_key", "is_home", "league", "month", cond]]
        unit = unit.sort_values("month")
        unit["g"] = tercile_groups(unit[cond], unit["league"], unit["month"])
        return d.merge(unit[["fixture_key", "is_home", "g"]], on=["fixture_key", "is_home"], how="left")["g"].to_numpy()
    unit = d.drop_duplicates("fixture_key")[["fixture_key", "league", "month", cond]].sort_values("month")
    unit["g"] = tercile_groups(unit[cond], unit["league"], unit["month"])
    return d.merge(unit[["fixture_key", "g"]], on="fixture_key", how="left")["g"].to_numpy()


def dev1_replication(pl: pd.DataFrame, reps: int = 2000, seed: int = 7) -> pd.DataFrame:
    """CONTEXT ONLY (never in the family): DEV-1's Q1 side-lambda test on DEV-1's own months
    2025-01..2026-03, markets sot / sot2 / goals, with the clean candidate in place of the buggy recipe."""
    rows = []
    for k in ("sot", "sot2", "goals"):
        m = pl[(pl["market"] == k) & pl["dc_lam_side"].notna() & pl["fixture_key"].notna()].copy()
        m["res"] = m["y"] - m["p_props"]
        m["cluster"] = m["fixture_key"]
        m = m.sort_values("date")
        m["grp"] = _props_groups(m, "dc_lam_side")
        ev = m[(m["date"] >= TRAIN_START) & (m["date"] < TEAM_HOLDOUT_START)]
        for lg in sorted(ev["league"].unique()) + ["ALL (context)"]:
            h = ev if lg == "ALL (context)" else ev[ev["league"] == lg]
            rows.append({"market": k, "league": lg, "window": "2025-01..2026-03 (DEV-1 months)",
                         **residual_cell(h, "grp", reps, seed)})
    return pd.DataFrame(rows)


def write_report_data(work: Path, out: Path = REPORT_DATA) -> dict:
    """Counts and estimates only into reports/combined_models_clean_data/: no price, no market
    probability, no player row."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text((work / "verification.json").read_text())
    (out / "join_info.json").write_text((work / "join_info.json").read_text())
    t = pd.read_csv(work / "tests.csv")
    keep = [c for c in t.columns if not c.startswith(("mean_", "p_emp"))]
    t[keep].round(6).to_csv(out / "tests.csv", index=False)
    pd.read_csv(work / "dev1_replication.csv").round(6).to_csv(out / "dev1_replication.csv", index=False)
    if (work / "props_side_check.csv").exists():
        pd.read_csv(work / "props_side_check.csv").round(4).to_csv(out / "props_side_check.csv", index=False)
    fx = pd.read_parquet(work / "fixtures.parquet")
    ev = fx[(fx["date"] >= EVAL_START) & (fx["date"] <= EVAL_END)]
    inv = ev.groupby("league").agg(
        fixtures_team=("p_over25", lambda s: int(s.notna().sum())),
        fixtures_pro=("lam_home", lambda s: int(s.notna().sum())),
        fixtures_team_and_pro=("p_over25", lambda s: int((s.notna() & ev.loc[s.index, "lam_home"].notna()).sum())),
        fixtures_with_bet_time_market=("m_over25", lambda s: int(s.notna().sum())))
    pl = pd.read_parquet(work / "players.parquet", columns=["date", "league", "market", "fixture_key", "p_over25",
                                                             "lam_home"])
    pe = pl[(pl["date"] >= EVAL_START) & (pl["date"] <= EVAL_END) & (pl["market"] == "sot")]
    inv = inv.join(pe.groupby("league").agg(props_rows_sot=("market", "size"),
                                            props_rows_sot_joined=("fixture_key", lambda s: int(s.notna().sum()))),
                   how="outer")
    inv.to_csv(out / "inventory_scored_window.csv")
    fam = t[t["in_family"].astype(bool)]
    summary = {"protocol_sha256": sha256_file(PROTOCOL_PATH), "family_size": int(len(fam)),
               "readable": int(fam["readable"].astype(bool).sum()), "signals": int(fam["signal"].astype(bool).sum()),
               "signal_cells": fam.loc[fam["signal"].astype(bool), ["target_group", "market", "helper", "test",
                                                                   "league"]].to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def props_equivalence(work: Path, frame_path: Path, holdout_csv: Path, pickle_dir: Path) -> dict:
    """Protocol amendment A1: per market on the September rows, (a) the run's pickle reproduces the
    run's CSV on the cached frame, (b) every run row matched with 0 label disagreements, (c) the local
    refit vs the run: |log loss difference| <= 0.0005 and correlation >= 0.99."""
    import pickle
    from player_model import props_harness as PH
    f = pd.read_parquet(frame_path)
    e = f[PH.eligible(f, "all_prior_verified").to_numpy()]
    del f
    d = pd.to_datetime(e["date"]).dt.normalize()
    sep = e[((d >= PROPS_HOLDOUT_START) & (d < PROPS_HOLDOUT_START + pd.offsets.MonthBegin(1))).to_numpy()]
    h = pd.read_csv(holdout_csv, usecols=["market", "date", "fixture_id", "player_id", "y", "p_hgb_nan_native"])
    hd = pd.to_datetime(h["date"]).dt.normalize()
    h = h[(hd >= PROPS_HOLDOUT_START) & (hd < PROPS_HOLDOUT_START + pd.offsets.MonthBegin(1))].drop(columns="date")
    w = pd.read_parquet(work / "props_wf.parquet", columns=["market", "fixture_id", "player_id", "y", "p_props",
                                                            "fit_through"])
    w = w[w["fit_through"] == str((PROPS_HOLDOUT_START - pd.Timedelta(days=1)).date())]
    out = {}
    for mk in PROPS_MARKETS:
        y_col = PH.TARGETS[mk]
        with open(pickle_dir / f"props_clean_{mk}_hgb_nan_native.pkl", "rb") as fh:
            model = pickle.load(fh)["model"]
        s = sep[sep[y_col].notna()]
        a = s[["fixture_id", "player_id"]].assign(p_pkl=model.predict(s[model.feature_cols_]))
        hm = h[h["market"] == mk]
        b = hm.merge(a, on=["fixture_id", "player_id"], how="left").merge(
            w[w["market"] == mk], on=["market", "fixture_id", "player_id"], how="left", suffixes=("", "_refit"))
        yy = b["y"].to_numpy(float)
        rec = {"rows_run": int(len(hm)), "rows_matched": int(b["p_props"].notna().sum()),
               "label_disagreements": int((b["y"] != b["y_refit"]).sum()),
               "pickle_vs_csv_max_abs": float(np.nanmax(np.abs(b["p_pkl"] - b["p_hgb_nan_native"]))),
               "refit_vs_run_mean_abs": float(np.abs(b["p_props"] - b["p_hgb_nan_native"]).mean()),
               "refit_vs_run_max_abs": float(np.abs(b["p_props"] - b["p_hgb_nan_native"]).max()),
               "corr": float(np.corrcoef(b["p_props"], b["p_hgb_nan_native"])[0, 1]),
               "log_loss_run": float(logloss(yy, b["p_hgb_nan_native"]).mean()),
               "log_loss_refit": float(logloss(yy, b["p_props"]).mean())}
        rec["log_loss_diff"] = rec["log_loss_refit"] - rec["log_loss_run"]
        rec["pass"] = bool(rec["pickle_vs_csv_max_abs"] <= 1e-12 and rec["rows_matched"] == rec["rows_run"]
                           and rec["label_disagreements"] == 0 and abs(rec["log_loss_diff"]) <= 5e-4
                           and rec["corr"] >= 0.99)
        out[mk] = rec
    (work / "props_equivalence.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["props-wf", "props-equivalence", "verify", "join", "analyse", "report-data"])
    ap.add_argument("--props-pickles", default=None, help="folder of run 43745808793's props_clean_*.pkl")
    ap.add_argument("--work", required=True)
    ap.add_argument("--props-frame", default=None)
    ap.add_argument("--props-holdout", default=None)
    ap.add_argument("--team-standard", default=None, help="run folder of the Standard champion_recipe run")
    ap.add_argument("--team-new-format", default=None, help="run folder of the New-Format champion_recipe run")
    ap.add_argument("--registry", default=None, help="config/feature_provenance.json of data-truth-100")
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.stage == "props-wf":
        meta = props_wf_clean(Path(a.props_frame), work, holdout_csv=a.props_holdout, log=log)
        print(json.dumps(meta, indent=1, default=str))
    elif a.stage == "props-equivalence":
        print(json.dumps(props_equivalence(work, Path(a.props_frame), Path(a.props_holdout), Path(a.props_pickles)),
                         indent=1))
    elif a.stage == "verify":
        reg = json.loads(Path(a.registry).read_text(encoding="utf-8"))
        v = {"team_standard": verify_team(Path(a.team_standard), "standard", reg),
             "team_new_format": verify_team(Path(a.team_new_format), "new_format", reg),
             "pro": verify_pro(work), "props": verify_props(work), "side_models": side_models_note()}
        (work / "verification.json").write_text(json.dumps(v, indent=1, default=str))
        print(json.dumps({k: (x["status"], x.get("fail_reasons")) for k, x in v.items()}, indent=1))
        return 0 if all(x["status"] == "PASS" for k, x in v.items() if k != "side_models") else 2
    elif a.stage == "join":
        v = json.loads((work / "verification.json").read_text())
        paths = {}
        for tr, key, arg in (("standard", "team_standard", a.team_standard), ("new_format", "team_new_format", a.team_new_format)):
            if v[key]["status"] == "PASS":
                paths[tr] = Path(arg) / "predictions.parquet"
            else:
                log(f"[join] {key} EXCLUDED: {v[key]['fail_reasons']}")
        fx = build_fixtures(work, paths)
        fx.to_parquet(work / "fixtures.parquet")
        pl, info = build_players(work, fx, log=log)
        pl.to_parquet(work / "players.parquet")
        if a.props_frame:
            props_feature_side_check(Path(a.props_frame), pl).to_csv(work / "props_side_check.csv", index=False)
        (work / "join_info.json").write_text(json.dumps(info, indent=1, default=str))
    elif a.stage == "report-data":
        print(json.dumps(write_report_data(work), indent=1))
    else:
        if not PROTOCOL_PATH.exists():
            raise SystemExit("protocol_dev2.json missing: nothing is computed before the pre-registration")
        fx = pd.read_parquet(work / "fixtures.parquet")
        pl = pd.read_parquet(work / "players.parquet")
        res, sr = run_analysis(fx, pl, reps=a.reps, log=log)
        res.to_csv(work / "tests.csv", index=False)
        sr.to_parquet(work / "stack_rows.parquet")
        dev1_replication(pl, reps=a.reps).to_csv(work / "dev1_replication.csv", index=False)
        log(json.dumps({"family": int(res["in_family"].sum()),
                        "readable": int((res["in_family"] & res["readable"].fillna(False).astype(bool)).sum()),
                        "signals": int(res["signal"].sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
