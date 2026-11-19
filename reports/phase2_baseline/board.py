"""Board coverage per track (rest days, HT inputs, HT predictions) + fully-empty columns + stake plan/audit facts.
usage: python board.py <worktree>  -- read only."""
import json, sys
from pathlib import Path
import pandas as pd
wt = Path(sys.argv[1]); out = wt / "output"
p = pd.read_csv(out / "predictions.csv", low_memory=False)
res = {"board_rows": len(p), "by_model_type": p["model_type"].astype(str).value_counts().to_dict()}
cols = ["home_rest_days", "away_rest_days", "home_ht_over05_rate", "away_ht_over05_rate", "home_ht_scored_last5",
        "p_ht_over05", "p_ht_over15", "ht_lambda_total", "odds_over25"]
res["missing_pct_by_track"] = {mt: {c: round(float(g[c].isna().mean() * 100), 1) for c in cols if c in g}
                               for mt, g in p.groupby(p["model_type"].astype(str))}
empty = [c for c in p.columns if p[c].isna().all()]
res["fully_empty_cols"] = empty
res["serving_recipe"] = p.get("serving_recipe", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
dates = pd.to_datetime(p.get("date", p.get("match_date")), errors="coerce")
res["board_dates"] = [str(dates.min()), str(dates.max())]
try:
    sa = pd.read_csv(out / "stake_audit.csv", low_memory=False)
    res["stake_audit_rows"] = len(sa)
    res["stake_audit_side_counts"] = sa["side"].astype(str).value_counts().to_dict()
    res["stake_audit_null_odds"] = int(pd.to_numeric(sa.get("odds"), errors="coerce").isna().sum())
except Exception as e:
    res["stake_audit"] = str(e)
try:
    sp = pd.read_csv(out / "stake_plan.csv", low_memory=False)
    res["stake_plan_rows"] = len(sp)
    for c in ("track_key", "state", "policy_state", "reason"):
        if c in sp: res[f"stake_plan_{c}"] = sp[c].astype(str).value_counts().head(12).to_dict()
    ff = pd.to_numeric(sp.get("final_fraction"), errors="coerce")
    res["stake_plan_nonzero_stakes"] = int((ff > 0).sum()) if ff is not None else None
except Exception as e:
    res["stake_plan"] = str(e)
try:
    ta = json.load(open(out / "corrections" / "props_truth_audit.json", encoding="utf-8"))
    res["props_truth_audit"] = ta
except Exception as e:
    res["props_truth_audit"] = f"missing: {type(e).__name__}"
print(json.dumps(res, indent=1, default=str))
