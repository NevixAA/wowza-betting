"""EFL feature health on the harness frame: missing rates per feature x league x season-year,
duplicates, label checks, HT coverage. Counts and rates only (no prices)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.retrain_harness import protocol as P
out = Path(sys.argv[1])
feat = pd.read_parquet(out / "feat.parquet")
cfg = P.load_config()
feats, _ = P.feature_list(cfg, "standard", feat.columns)
E = ["Championship", "League One", "League Two"]
std = feat[P.track_series(feat["league"]) == "standard"].copy()
std["yr"] = pd.to_datetime(std["date"]).dt.year
std["grp"] = np.where(std.league.isin(E), std.league, "other_std")
res = {}
miss = std.groupby("grp")[feats].apply(lambda d: d.isna().mean()).T.round(3)
res["missing_by_group"] = miss.to_dict()
print("== missing rate per feature (all Standard rows, pre-require_present)"); print(miss.to_string())
for c in ["home_xg_last5", "home_insidebox_last5", "home_ht_scored_last5", "home_corners_pg_roll", "combined_sot_ratio", "home_scored_last5"]:
    t = std[std.league.isin(E)].groupby(["league", "yr"])[c].apply(lambda s: s.isna().mean()).unstack().round(2)
    print(f"\n== {c} missing by EFL league x year"); print(t.to_string())
# raw stat source availability
for c in ["home_xg", "home_insidebox", "home_shots", "home_sot", "ht_home_goals", "home_corners"]:
    if c in std.columns:
        t = std[std.league.isin(E)].groupby(["league", "yr"])[c].apply(lambda s: s.notna().mean()).unstack().round(2)
        print(f"\n== raw {c} PRESENT by EFL league x year"); print(t.to_string())
# label checks
e = std[std.league.isin(E)]
print("\n== EFL rows per league-year"); print(e.groupby(["league", "yr"]).size().unstack().to_string())
tg = pd.to_numeric(e.home_goals, errors="coerce") + pd.to_numeric(e.away_goals, errors="coerce")
print("over25 label mismatch:", int(((tg > 2.5).astype(float) != e.over25).sum()), "of", len(e))
if "ht_home_goals" in e:
    ht = pd.to_numeric(e.ht_home_goals, errors="coerce") + pd.to_numeric(e.ht_away_goals, errors="coerce")
    print("HT > FT impossible:", int((ht > tg).sum()), " HT present:", round(ht.notna().mean(), 3))
    print("ht_over05 label mismatch:", int(((ht > 0.5).astype(float) != e.ht_over05)[ht.notna()].sum()))
print("max total goals", tg.max(), " goals>=9 rows:", int((tg >= 9).sum()))
dup = e.duplicated(["date", "league", "home_team", "away_team"]).sum()
dup2 = e.assign(d=pd.to_datetime(e.date).dt.to_period("Y").astype(str)).duplicated(["league", "home_team", "away_team", "d"]).sum()
print("exact dup fixtures:", int(dup), " same pairing same calendar year (can be legit):", int(dup2))
# season sanity: matches per team per season
e2 = e.copy(); e2["szn"] = np.where(pd.to_datetime(e2.date).dt.month >= 7, e2.yr, e2.yr - 1)
tm = pd.concat([e2[["league", "szn", "home_team"]].rename(columns={"home_team": "t"}), e2[["league", "szn", "away_team"]].rename(columns={"away_team": "t"})])
cnt = tm.groupby(["league", "szn", "t"]).size()
print("\n== teams per league-season and games per team (min/median/max)")
print(cnt.groupby(["league", "szn"]).agg(["size", "min", "median", "max"]).to_string())
# require_present drop rate
rp = e[["home_scored_last5", "away_scored_last5"]].notna().all(axis=1)
print("\nEFL rows dropped by require_present (no form):", int((~rp).sum()), "of", len(e))
