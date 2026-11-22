"""Independent reconstruction of home/away_has_ht_history (owner P0 item 4, 2026-11-22). No prices.

The flag's definition (src/feature_engineering.py HT_HISTORY_FEATURES): 1.0 when the team's form window
- its last n=5 matches in the SAME league, strictly before this match - holds at least one match with a
real half-time score (both HT goals present), else 0.0 (also with no earlier match).

This script re-derives it with its own loop (no src.feature_engineering code) from the raw match columns
and compares it with
  A) the training frames: new = ebe47c041a7f595f (served 2016/17 refit), old = efl-diag 11-18 (pre-PR #59)
     -> every row; every mismatch and every flag=0 row is written with its reason;
  B) serving: src.feature_engineering.build_upcoming_features (+ src.ht_features.attach_ht_features, the
     two builders predict.py runs) on the committed history strictly before sample dates (per league: the
     latest date and one older seeded-random date), vs training (build_features on the COVID-excluded
     history, as the trainers) and vs the independent value;
  C) the live board output/predictions.csv (stored served values) vs the independent value on the history
     before each fixture's date.

    python ht_flag_evidence.py <repo_root> <new_frame.parquet> <old_frame.parquet> <out_dir>
"""
from __future__ import annotations

import json
import socket
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
NEW, OLD, OUT = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
N = 5
KEY = ["date", "league", "home_team", "away_team"]
FLAGS = {"home": "home_has_ht_history", "away": "away_has_ht_history"}


def _no_net(*a, **k):
    raise OSError("network blocked")


def reconstruct(m: pd.DataFrame) -> pd.DataFrame:
    """Per row: expected flag per side + the window's provenance. Own loop, no feature code."""
    m = m.reset_index(drop=True)
    d = pd.to_datetime(m["date"]).dt.normalize().to_numpy()
    known = (m["ht_home_goals"].notna() & m["ht_away_goals"].notna()).to_numpy()
    src = m["row_source"].astype(str).to_numpy() if "row_source" in m.columns else np.array(["?"] * len(m))
    season = m["season"].astype(str).to_numpy() if "season" in m.columns else np.array(["?"] * len(m))
    lg, ht, at = m["league"].to_numpy(), m["home_team"].to_numpy(), m["away_team"].to_numpy()
    order = np.lexsort((np.arange(len(m)), d))
    hist: dict = {}
    cols = [f"{s}_{k}" for s in ("home", "away") for k in ("exp", "win_n", "win_known", "win_src", "win_seasons")]
    out = {c: [None] * len(m) for c in cols}
    pending, last = [], None
    for i in order:
        if last is not None and d[i] != last:          # same-date rows are not each other's history
            for key, j in pending:
                hist.setdefault(key, deque(maxlen=N)).append(j)
            pending = []
        last = d[i]
        for side, team in (("home", ht[i]), ("away", at[i])):
            key = (lg[i], team)
            w = list(hist.get(key, ()))
            out[f"{side}_exp"][i] = float(any(known[j] for j in w))
            out[f"{side}_win_n"][i] = len(w)
            out[f"{side}_win_known"][i] = int(sum(known[j] for j in w))
            out[f"{side}_win_src"][i] = ";".join(f"{k}x{v}" for k, v in Counter(src[j] for j in w).items())
            out[f"{side}_win_seasons"][i] = ";".join(sorted(set(season[j] for j in w)))
            pending.append((key, i))
    return pd.DataFrame(out)


def reason(exp, tr, sv=None, check_serving=False):
    if pd.isna(tr):
        return "training value missing"
    if float(tr) != float(exp):
        return "training != independent reconstruction"
    if check_serving:
        if pd.isna(sv):
            return "serving value missing"
        if float(sv) != float(exp):
            return "serving != independent reconstruction"
    return ""


def src_text(n, k, s, se):
    return f"window {n} matches, {k} with HT; sources {s}; seasons {se}"


def frame_part(label: str, path: Path, rows_out: list, summ: dict):
    from src.model import track_of
    f = pd.read_parquet(path)
    f["date"] = pd.to_datetime(f["date"])
    f["track"] = track_of(f["league"]).to_numpy()
    f = pd.concat([f.reset_index(drop=True), reconstruct(f)], axis=1)
    elig = (f["home_scored_last5"].notna() & f["away_scored_last5"].notna()
            & f["ht_home_goals"].notna() & f["ht_away_goals"].notna())
    elig &= ~((f["track"] == "standard") & (f["date"] < "2016-07-01"))
    s = {"rows": int(len(f)), "ht_market_eligible_rows": int(elig.sum())}
    for side, col in FLAGS.items():
        tr = f[col].astype(float)
        ex = f[f"{side}_exp"].astype(float)
        mism = (tr != ex) & tr.notna()
        s[f"{side}_mismatch_vs_independent"] = int(mism.sum())
        s[f"{side}_training_missing"] = int(tr.isna().sum())
        cons = tr == f[f"{side}_ht_scored_last5"].notna().astype(float)
        s[f"{side}_flag_eq_ht_scored_last5_present"] = f"{int(cons.sum())}/{len(f)}"
        for t in ("standard", "new_format"):
            e = elig & (f["track"] == t)
            s[f"{side}_{t}_eligible_rows"] = int(e.sum())
            s[f"{side}_{t}_eligible_share_1"] = round(float((tr[e] == 1).mean()), 6) if e.any() else None
            s[f"{side}_{t}_eligible_zeros"] = int((tr[e] == 0).sum())
        for i in np.flatnonzero(((tr == 0) | mism).to_numpy()):
            x = f.iloc[i]
            why = reason(x[f"{side}_exp"], x[col])
            if not why:
                why = "flag 0 is correct: " + ("no earlier match in the league" if x[f"{side}_win_n"] == 0
                                               else "no HT score in the window")
            rows_out.append({"frame": label,
                             "fixture_id": f"{x['date'].date()}|{x['league']}|{x['home_team']}|{x['away_team']}",
                             "league": x["league"], "track": x["track"], "kickoff": str(x["date"].date()),
                             "feature": col,
                             "raw_source": src_text(x[f"{side}_win_n"], x[f"{side}_win_known"],
                                                    x[f"{side}_win_src"], x[f"{side}_win_seasons"]),
                             "expected_value": x[f"{side}_exp"], "training_value": x[col], "serving_value": np.nan,
                             "ht_market_eligible": bool(elig.iat[i]), "mismatch_reason": why})
    by = f[elig].groupby(["track", "season"])[list(FLAGS.values())].agg(lambda v: round(float((v == 1).mean()), 5))
    by["rows"] = f[elig].groupby(["track", "season"]).size()
    by.to_csv(OUT / f"flag_share_by_track_season_{label}.csv")
    summ[label] = s
    print(label, json.dumps(s), flush=True)


def main():
    rows, summ = [], {}
    frame_part("new_ebe47c041a7f595f", NEW, rows, summ)
    frame_part("old_pre_pr59_1118", OLD, rows, summ)
    pd.DataFrame(rows).to_csv(OUT / "ht_flag_frame_rows.csv", index=False)

    socket.socket.connect = _no_net
    socket.create_connection = _no_net
    import config
    from src import data_loader as DL
    from src import feature_engineering as FE
    from src.ht_features import HT_FEATURE_COLS, attach_ht_features
    from src.model import track_of
    DL._ci_download_all = lambda have=None: []
    DL._report_training_coverage = lambda out: None
    hist = DL.load_all_matches(force=True).copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist[hist["home_goals"].notna() & hist["away_goals"].notna()].reset_index(drop=True)
    covid = config.covid_season_mask(hist).to_numpy() if config.EXCLUDE_COVID_SEASONS else np.zeros(len(hist), bool)
    train = FE.build_features(hist[~covid])
    train["date"] = pd.to_datetime(train["date"])
    has_fid = "fixture_id" in hist.columns
    rec_full = pd.concat([hist[KEY + (["fixture_id"] if has_fid else [])].reset_index(drop=True),
                          reconstruct(hist)], axis=1)          # serving reads the full history
    rec_full["date"] = pd.to_datetime(rec_full["date"])
    rng = np.random.default_rng(20261122)
    hd = hist["date"].dt.normalize()
    dates = set()
    for _lg, g in hist.groupby("league"):
        ds = sorted(hd[g.index].unique())
        dates.add(ds[-1])
        old = [x for x in ds if x < ds[-1] - pd.Timedelta(days=365)]
        if old:
            dates.add(old[int(rng.integers(len(old)))])
    dates = sorted(dates)
    print("serving sample dates", len(dates), flush=True)
    tcols = KEY + list(FLAGS.values())
    srows, per_date = [], {}
    for dt in dates:
        day = hist[hd == dt]
        side = pd.concat([day[["league", "home_team"]].rename(columns={"home_team": "t"}),
                          day[["league", "away_team"]].rename(columns={"away_team": "t"})])
        twice = set(map(tuple, side[side.duplicated(keep=False)].to_numpy()))
        day = day[[(a, h) not in twice and (a, w) not in twice
                   for a, h, w in zip(day["league"], day["home_team"], day["away_team"])]]
        before = hist[hist["date"] < dt]
        keep = [c for c in KEY + ["odds_over25", "odds_under25"] if c in day.columns]
        up = day[keep].reset_index(drop=True)
        sv = FE.build_upcoming_features(up, before)
        hta = attach_ht_features(before, sv, identity_aliases=getattr(DL, "IDENTITY_ALIASES", None))
        if len(hta) == len(sv):
            for c in HT_FEATURE_COLS:
                if c in hta.columns:
                    sv[c] = hta[c].to_numpy()
        k = up[KEY].copy()
        tr = k.merge(train[tcols].drop_duplicates(KEY), on=KEY, how="left")
        ex = k.merge(rec_full.drop_duplicates(KEY), on=KEY, how="left")
        nb = 0
        for i in range(len(k)):
            for sd, col in FLAGS.items():
                e, t = ex.at[i, f"{sd}_exp"], tr.at[i, col]
                s = sv.at[i, col] if col in sv.columns else np.nan
                why = reason(e, t, s, check_serving=True)
                if why == "training value missing" and bool(np.asarray(config.covid_season_mask(day.iloc[[i]]))[0]):
                    ok = float(s) == float(e)
                    why = ("no training row: COVID season (excluded from training by design); serving == expected"
                           if ok else "serving != independent reconstruction (COVID-season row)")
                sc = f"{sd}_ht_scored_last5"
                if not why and sc in sv.columns and float(s) != float(pd.notna(sv.at[i, sc])):
                    why = "serving flag disagrees with serving HT form presence (attach_ht_features)"
                nb += bool(why)
                fid = ex.at[i, "fixture_id"] if has_fid and pd.notna(ex.at[i, "fixture_id"]) else \
                    f"{pd.Timestamp(dt).date()}|{k.at[i, 'league']}|{k.at[i, 'home_team']}|{k.at[i, 'away_team']}"
                srows.append({"fixture_id": fid, "league": k.at[i, "league"],
                              "track": track_of(pd.Series([k.at[i, "league"]])).iat[0],
                              "kickoff": str(pd.Timestamp(dt).date()), "feature": col,
                              "raw_source": src_text(ex.at[i, f"{sd}_win_n"], ex.at[i, f"{sd}_win_known"],
                                                     ex.at[i, f"{sd}_win_src"], ex.at[i, f"{sd}_win_seasons"]),
                              "expected_value": e, "training_value": t, "serving_value": s,
                              "mismatch_reason": why})
        per_date[str(pd.Timestamp(dt).date())] = {"fixtures": int(len(k)), "mismatched_cells": nb}
        print(pd.Timestamp(dt).date(), per_date[str(pd.Timestamp(dt).date())], flush=True)
    sdf = pd.DataFrame(srows)
    sdf.to_csv(OUT / "ht_flag_serving_sample.csv", index=False)
    bad = sdf["mismatch_reason"] != ""
    summ["serving_sample"] = {"dates": len(dates), "fixtures": int(sum(v["fixtures"] for v in per_date.values())),
                              "cells": int(len(sdf)), "leagues": int(sdf["league"].nunique()),
                              "leagues_by_track": sdf.drop_duplicates("league").groupby("track").size().to_dict(),
                              "mismatched_cells": int((bad & ~sdf["mismatch_reason"].str.startswith("no training row: COVID")).sum()),
                              "flagged_cells_incl_covid_no_training_row": int(bad.sum()),
                              "mismatch_reasons": sdf.loc[bad, "mismatch_reason"].value_counts().to_dict(),
                              "serving_share_1": round(float((sdf["serving_value"] == 1).mean()), 5),
                              "expected_share_1": round(float((sdf["expected_value"] == 1).mean()), 5)}

    b = pd.read_csv(ROOT / "output" / "predictions.csv")
    b["date"] = pd.to_datetime(b["date"])
    from src.ht_features import resolve_board_names
    # board names are provider spellings; map them to history spellings with the serving identity
    # layer (name mapping only - the flag below is re-derived by this script's own window rule)
    bup, _rep = resolve_board_names(hist, b[["date", "league", "home_team", "away_team"]].copy(),
                                    getattr(DL, "IDENTITY_ALIASES", None))
    brows = []
    for bi, x in b.iterrows():
        before = hist[hist["date"] < x["date"].normalize()]
        for sd, col in FLAGS.items():
            team = bup.at[bi, f"{sd}_team"]
            w = before[(before["league"] == x["league"])
                       & ((before["home_team"] == team) | (before["away_team"] == team))].sort_values("date").tail(N)
            kn = int((w["ht_home_goals"].notna() & w["ht_away_goals"].notna()).sum())
            e = float(kn > 0)
            s = x[col]
            srcs = ";".join(f"{a}x{c}" for a, c in Counter(w["row_source"].astype(str)).items()) \
                if "row_source" in w.columns else "?"
            brows.append({"fixture_id": x.get("fixture_id"), "league": x["league"], "track": x.get("model_type"),
                          "kickoff": x.get("kickoff_utc"), "feature": col,
                          "raw_source": src_text(len(w), kn, srcs, ";".join(sorted(set(w["season"].astype(str))))
                                                 if "season" in w.columns else "?"),
                          "expected_value": e, "training_value": np.nan, "serving_value": s,
                          "mismatch_reason": "" if float(s) == e else "board (served) != independent reconstruction"})
    bdf = pd.DataFrame(brows)
    bdf.to_csv(OUT / "ht_flag_live_board.csv", index=False)
    summ["live_board"] = {"fixtures": int(len(b)), "cells": int(len(bdf)),
                          "leagues_by_track": b.drop_duplicates("league").groupby("model_type").size().to_dict(),
                          "mismatched_cells": int((bdf["mismatch_reason"] != "").sum()),
                          "served_share_1": round(float((bdf["serving_value"] == 1).mean()), 5),
                          "expected_share_1": round(float((bdf["expected_value"] == 1).mean()), 5),
                          "generated_at": str(b["generated_at"].iloc[0]) if "generated_at" in b.columns else None}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
