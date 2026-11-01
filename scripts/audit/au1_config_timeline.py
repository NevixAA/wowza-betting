"""AU1 Phase 7 - configuration regime timeline from git history (READ-ONLY).

Walks origin/main history of the files that decide what production does and records, per commit,
the effective value of every regime-defining knob:

  * predict.yml env: LEAGUE_SNIPER_CAP, MARKSMAN_THRESHOLD, VALUABLE_THRESHOLD, REQUIRE_FORM_DATA(_UNTIL),
    STANDARD_OU25_SERVING_RECIPE, NF_MAX_OU_ODDS, MIN_*_ODDS, DRIFT_TIER_ADJUSTMENT, USE_KELLY, ...
  * config.py defaults for the same knobs + PERFORMANCE_CUTOFF_DATE, DRIFT_TIER_ADJUSTMENT, EDGE_CEILING
  * models/*.pkl and models/best_params_*.json and config/bankroll_policy.json blob changes

Writes reports/au1_data/config_timeline.csv and model_epochs.csv. Never writes outside reports/.
Usage:  python scripts/audit/au1_config_timeline.py [--ref origin/main]
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "au1_data"

ENV_KEYS = ["LEAGUE_SNIPER_CAP", "MARKSMAN_THRESHOLD", "VALUABLE_THRESHOLD", "SNIPER_THRESHOLD",
            "SNIPER_THRESHOLD_OVER", "SNIPER_THRESHOLD_UNDER", "EDGE_CEILING",
            "REQUIRE_FORM_DATA", "REQUIRE_FORM_DATA_UNTIL", "STANDARD_OU25_SERVING_RECIPE",
            "NF_MAX_OU_ODDS", "MIN_OVER_ODDS", "MIN_UNDER_ODDS", "DRIFT_TIER_ADJUSTMENT", "USE_KELLY",
            "KELLY_FRACTION", "WOWZA_MODEL_RECIPE", "WOWZA_MARKET_FEATURE_MODE", "NF_FEATURE_PARITY",
            "WOWZA_SAFE_ENRICH_JOINS", "PAPER_LEAGUES"]
CFG_PATTERNS = {
    "cfg_VALUABLE_THRESHOLD": r'VALUABLE_THRESHOLD\s*=\s*float\(os\.getenv\("VALUABLE_THRESHOLD",\s*"([^"]+)"',
    "cfg_MARKSMAN_THRESHOLD": r'MARKSMAN_THRESHOLD\s*=\s*float\(os\.getenv\("MARKSMAN_THRESHOLD",\s*"([^"]+)"',
    "cfg_SNIPER_THRESHOLD": r'SNIPER_THRESHOLD\s*=\s*float\(os\.getenv\("SNIPER_THRESHOLD",\s*"([^"]+)"',
    "cfg_EDGE_CEILING": r'EDGE_CEILING\s*=\s*float\(os\.getenv\("EDGE_CEILING",\s*"([^"]+)"',
    "cfg_MIN_OVER_ODDS": r'MIN_OVER_ODDS\s*=\s*float\(os\.getenv\("MIN_OVER_ODDS",\s*"([^"]+)"',
    "cfg_NF_MAX_OU_ODDS": r'NF_MAX_OU_ODDS\s*=\s*float\(os\.getenv\("NF_MAX_OU_ODDS",\s*"([^"]+)"',
    "cfg_PERFORMANCE_CUTOFF_DATE": r'PERFORMANCE_CUTOFF_DATE\s*=\s*"([^"]+)"',
    "cfg_DRIFT_TIER_ADJUSTMENT": r'^DRIFT_TIER_ADJUSTMENT\s*=\s*(.+?)\s*(#|$)',
    "cfg_SNIPER_CAP_default": r'_SNIPER_CAP\s*=\s*float\(os\.getenv\("LEAGUE_SNIPER_CAP",\s*"([^"]+)"',
    "cfg_STANDARD_RECIPE_default": r'STANDARD_OU25_SERVING_RECIPE_RAW\s*=\s*os\.getenv\("STANDARD_OU25_SERVING_RECIPE",\s*"([^"]+)"',
    "cfg_USE_KELLY": r'USE_KELLY\s*=\s*os\.getenv\("USE_KELLY",\s*"([^"]+)"',
}
ARTIFACTS = ["models/model_v9_standard.pkl", "models/model_v9_standard_repaired.pkl",
             "models/model_v9_newformat.pkl", "models/model_v9_btts.pkl", "models/model_v9_over15.pkl",
             "models/model_v9_over35.pkl", "models/model_ht_over05.pkl", "models/model_ht_over15.pkl",
             "models/best_params_standard.json", "models/best_params_side_markets.json",
             "config/bankroll_policy.json", "output/league_roi_config.json", "config/leagues.json",
             "config/threshold_applied.json", "config/threshold_approvals.json"]


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def env_values(text: str) -> dict:
    """Values of ENV_KEYS as set in the `Run predict` step env (first occurrence of `KEY: "v"`)."""
    out = {}
    for k in ENV_KEYS:
        m = re.search(rf'^\s+{k}:\s*"?([^"\n#]*)"?', text, re.M)
        out[k] = m.group(1).strip() if m else ""
    return out


def cfg_values(text: str) -> dict:
    out = {}
    for k, pat in CFG_PATTERNS.items():
        m = re.search(pat, text, re.M)
        out[k] = m.group(1).strip() if m else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. knob timeline: every commit touching predict.yml or config.py, values only where they CHANGE
    log = git("log", "--reverse", "--format=%H|%cI|%s", a.ref, "--", ".github/workflows/predict.yml",
              "config.py").strip().splitlines()
    rows, prev = [], None
    for line in log:
        sha, ts, subj = line.split("|", 2)
        try:
            yml = git("show", f"{sha}:.github/workflows/predict.yml")
        except subprocess.CalledProcessError:
            yml = ""
        try:
            cfg = git("show", f"{sha}:config.py")
        except subprocess.CalledProcessError:
            cfg = ""
        vals = {**{f"env_{k}": v for k, v in env_values(yml).items()}, **cfg_values(cfg)}
        if vals != prev:
            changed = [k for k in vals if prev is None or vals[k] != prev.get(k)]
            rows.append({"commit": sha[:10], "committed_at": ts, "subject": subj[:90],
                         "changed": ";".join(changed), **vals})
            prev = vals
    with open(OUT / "config_timeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 2. artifact epochs: every commit that changed an artifact blob
    ep = []
    for path in ARTIFACTS:
        for line in git("log", "--reverse", "--format=%H|%cI|%s", a.ref, "--", path).strip().splitlines():
            if not line:
                continue
            sha, ts, subj = line.split("|", 2)
            try:
                blob = git("rev-parse", f"{sha}:{path}").strip()[:10]
            except subprocess.CalledProcessError:
                blob = "DELETED"
            ep.append({"artifact": path, "commit": sha[:10], "committed_at": ts, "blob": blob,
                       "subject": subj[:90]})
    with open(OUT / "artifact_epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["artifact", "commit", "committed_at", "blob", "subject"])
        w.writeheader()
        w.writerows(ep)
    print(f"config_timeline rows={len(rows)}  artifact_epochs rows={len(ep)}")


if __name__ == "__main__":
    main()
