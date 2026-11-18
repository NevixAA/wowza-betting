"""Leakage / swap / join sanity on EFL: single-feature AUC for over25 (leak -> too high,
broken join -> ~0.5 everywhere), home vs away goal correlations, first-match-in-league rows."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.retrain_harness import protocol as P
out = Path(sys.argv[1]); feat = pd.read_parquet(out / "feat.parquet"); cfg = P.load_config()
feats, _ = P.feature_list(cfg, "standard", feat.columns)
rows = P.eligible_rows(feat, cfg, "standard", "over25")
E = ["Championship", "League One", "League Two"]
res = []
for grp, m in [("EFL", rows.league.isin(E)), ("other_std", ~rows.league.isin(E))]:
    r = rows[m]
    for c in feats:
        x = pd.to_numeric(r[c], errors="coerce"); ok = x.notna()
        if ok.sum() > 300 and x[ok].nunique() > 1:
            res.append({"group": grp, "feature": c, "n": int(ok.sum()), "auc_over25": round(roc_auc_score(r.over25[ok], x[ok]), 4)})
t = pd.DataFrame(res).pivot(index="feature", columns="group", values="auc_over25").sort_values("EFL")
print(t.to_string())
t.to_csv(out / "exp" / "single_feature_auc.csv")
e = rows[rows.league.isin(E)]
print("\ncorr(home_scored_last5, home_goals) =", round(e.home_scored_last5.corr(e.home_goals), 3),
      " corr(home_scored_last5, away_goals) =", round(e.home_scored_last5.corr(e.away_goals), 3))
print("corr(away_scored_last5, away_goals) =", round(e.away_scored_last5.corr(e.away_goals), 3),
      " corr(home_conceded_last5, away_goals) =", round(e.home_conceded_last5.corr(e.away_goals), 3))
o = rows[~rows.league.isin(E)]
print("other std: corr(home_scored_last5, home_goals) =", round(o.home_scored_last5.corr(o.home_goals), 3))
