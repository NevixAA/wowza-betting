"""Props train/serve parity on REAL fixtures (owner 2026-11-08 props promotion evidence).

    python scripts/props_train_serve_parity.py --work <dir> --artifact-dir <props-clean-retrain artifact> --out <json>

<dir> must hold raw.parquet (props_harness.source_frame()) and frame.parquet (props_harness.build_frame(raw));
they are built there when absent. For 2 fixtures per league of 2026-10-24..10-31: the serving builder
(feature_engineering.build_upcoming_features on the history strictly before the fixture date, the serving
referee profile measured from that history, the player's latest history row as the serving input) against
the training frame row of the same (fixture, player); every model feature and the promoted models'
probabilities are compared. Result 2026-11-08: reports/props_promotion_parity_2026-11-08.json.
"""
import argparse, sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from player_model import props_harness as H
from player_model.feature_engineering import (build_upcoming_features, _build_history_index,
                                              referee_profile_from_history)
ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--artifact-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-league", type=int, default=2)
ap.add_argument("--start", default="2026-10-24"); ap.add_argument("--end", default="2026-10-31")
a = ap.parse_args()
S, P, OUT = a.work, a.artifact_dir, a.out
if not (Path(S) / "frame.parquet").exists():
    _raw = H.source_frame(); _raw.to_parquet(Path(S) / "raw.parquet"); H.build_frame(_raw).to_parquet(Path(S) / "frame.parquet")
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        return super().find_class("player_model.props_harness" if mod == "__main__" else mod, name)
raw = pd.read_parquet(S + "/raw.parquet")
frame = pd.read_parquet(S + "/frame.parquet")
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.sort_values("date", kind="stable").reset_index(drop=True)
frame["date"] = pd.to_datetime(frame["date"])
models = {}
for m, c in (("sot", "hgb_nan_native"), ("sot2", "hgb_nan_native"), ("goals", "hgb_nan_native"),
             ("assists", "hgb_nan_native"), ("cards", "recipe_retrained")):
    models[m] = U(open(f"{P}/models/challengers/props_clean/props_clean_{m}_{c}.pkl", "rb")).load()["model"]
feats = models["sot"].feature_cols_
# sample: fixtures of 2026-10-24..10-31 in the training frame, up to 6 per league
cand = frame[(frame.date >= a.start) & (frame.date <= a.end)]
fx = (cand.drop_duplicates("fixture_id")[["fixture_id", "date", "league"]]
          .groupby("league", group_keys=False).head(a.per_league))
print("fixtures", len(fx), "leagues", fx.league.nunique(), flush=True)
sys.path.insert(0, str(ROOT / "scripts"))
import collect_lineups as _CL
_lp = _CL.read_lineups("players").dropna(subset=["player_id"])
_lp = _lp.assign(player_id=_lp["player_id"].astype("int64"), fixture_id=pd.to_numeric(_lp["fixture_id"]).astype("int64"))
LINEUP = {(int(f), int(p)): s for f, p, s in zip(_lp["fixture_id"], _lp["player_id"], _lp["pos"]) if isinstance(s, str) and s}
rows_s, rows_t = [], []
for d, fd in fx.groupby("date"):
    hist = raw.iloc[: int(raw["date"].searchsorted(d, side="left"))]
    idx = _build_history_index(hist)
    last = hist.drop_duplicates("player_id", keep="last").set_index("player_id")
    print("date", d.date(), "hist", len(hist), flush=True)
    for _, f in fd.iterrows():
        fr = raw[(raw.fixture_id == f.fixture_id)]
        tr = frame[frame.fixture_id == f.fixture_id]
        if fr.empty or tr.empty:
            continue
        teams = list(dict.fromkeys(fr["team"].astype(str)))
        ref_name = fr["referee"].dropna().astype(str).iloc[0] if "referee" in fr and fr["referee"].notna().any() else None
        refp = referee_profile_from_history(hist, ref_name, f.league, before=d)
        up = []
        for _, r in fr.iterrows():
            if r.player_id not in set(tr.player_id):
                continue
            if r.player_id not in last.index:
                continue
            p = last.loc[r.player_id].to_dict(); p["player_id"] = r.player_id   # serving: latest history row
            p.update(team=r["team"], opponent=[t for t in teams if t != str(r["team"])][0] if len(teams) == 2 else "",
                     is_home=float(r["is_home"]) if "is_home" in r else np.nan,
                     fixture_league=f.league, fixture_date=str(d.date()),
                     age=tr.loc[tr.player_id == r.player_id, "age"].iloc[0] if "age" in tr else None)
            _lpos = LINEUP.get((int(f.fixture_id), int(r.player_id)))      # the announced lineup, as live
            if _lpos:
                p["lineup_position"] = _lpos
            up.append(p)
        if not up:
            continue
        sv = build_upcoming_features(up, hist, refp, {}, history_index=idx)
        sv["fixture_id"] = f.fixture_id
        rows_s.append(sv)
        rows_t.append(tr)
sv = pd.concat(rows_s, ignore_index=True)
tr = pd.concat(rows_t, ignore_index=True)
sv["player_id"] = pd.to_numeric(sv["player_id"]); tr["player_id"] = pd.to_numeric(tr["player_id"])
j = sv.merge(tr, on=["fixture_id", "player_id"], suffixes=("_s", "_t"))
print("paired player-fixtures", len(j), flush=True)
j.to_parquet(OUT.replace(".json", "_pairs.parquet"))           # the paired frame, for diagnosis
rep = {"paired_rows": int(len(j)), "fixtures": int(j.fixture_id.nunique()), "features": {},
       "position_source": j["position_source"].value_counts().to_dict() if "position_source" in j else {},
       "sample": {"per_league": a.per_league, "start": a.start, "end": a.end}}
for c in feats:
    a = pd.to_numeric(j.get(c + "_s"), errors="coerce") if c + "_s" in j else pd.Series(np.nan, index=j.index)
    b = pd.to_numeric(j.get(c + "_t"), errors="coerce") if c + "_t" in j else pd.Series(np.nan, index=j.index)
    same = (a.isna() & b.isna()) | (np.abs(a - b) <= 1e-9 * np.maximum(1, np.abs(b)))
    rep["features"][c] = {"mismatch": int((~same).sum()), "nan_serve_only": int((a.isna() & b.notna()).sum()),
                          "nan_train_only": int((a.notna() & b.isna()).sum())}
bad = {k: v for k, v in rep["features"].items() if v["mismatch"]}
rep["features_mismatched"] = len(bad)
probs = {}
for m, mod in models.items():
    Xs = pd.DataFrame({c: (j[c + "_s"] if c + "_s" in j else np.nan) for c in mod.feature_cols_})
    Xt = pd.DataFrame({c: (j[c + "_t"] if c + "_t" in j else np.nan) for c in mod.feature_cols_})
    ps, pt = mod.predict(Xs), mod.predict(Xt)
    d = np.abs(ps - pt)
    probs[m] = {"max_abs_diff": float(np.nanmax(d)), "mean_abs_diff": float(np.nanmean(d)),
                "rows_diff_gt_1e-9": int((d > 1e-9).sum()), "rows_diff_gt_0.01": int((d > 0.01).sum())}
rep["probabilities"] = probs
json.dump(rep, open(OUT, "w"), indent=1)
print(json.dumps({"paired": rep["paired_rows"], "features_mismatched": len(bad), "position_source": rep["position_source"], "probabilities": probs}, indent=1))
badrows = {}
for k in bad:
    a_ = pd.to_numeric(j[k + "_s"], errors="coerce"); b_ = pd.to_numeric(j[k + "_t"], errors="coerce")
    m_ = ~((a_.isna() & b_.isna()) | (np.abs(a_ - b_) <= 1e-9 * np.maximum(1, np.abs(b_))))
    badrows[k] = j.loc[m_, ["fixture_id", "player_id", "league", k + "_s", k + "_t"] + (["position_source"] if "position_source" in j else [])].head(5).to_dict("records") if "league" in j else []
json.dump(badrows, open(OUT.replace(".json", "_examples.json"), "w"), indent=1, default=str)
for k, v in sorted(bad.items(), key=lambda kv: -kv[1]["mismatch"])[:25]:
    print(k, v)
