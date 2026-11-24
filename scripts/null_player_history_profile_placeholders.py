"""D4 (owner 2026-11-23 "Yes, do all"; reports/FINAL_DATA_TRUTH_CERTIFICATION.md section 4): the
stored player_history profile PLACEHOLDERS rewritten to MISSING, the old values kept beside them.

WHAT IS A PLACEHOLDER ROW. The old build_features wrote age 25.0 / height_cm 180 / weight_kg 75 /
age_peak_delta 0 into every row (player_model.feature_engineering._PROFILE_PLACEHOLDERS); since
2026-11-18 the writer stores the collected profile (age on the match date, a fraction of a year).
A row is a placeholder row only when ALL FOUR hold at once - age exactly 25.0, height 180, weight 75,
age_peak_delta 0 - so a real 180 cm or 75 kg player (a fractional real age beside it) is never
touched. On such a row the four values and the composites computed from them
(opp_def_mean_height, height_diff_vs_opp_def, height_advantage_score, height_aerial_interaction;
player_model.feature_engineering.profile_composites) become NaN: none of them was measured.

NEVER SILENT. Every changed row's keys and OLD values go to
output/corrections/player_history_profile_placeholders_2026-11-23.parquet (+ _audit.json); no raw
collected stat (minutes, goals, shots, cards, ids, names, dates) is read for writing or changed.
The month parts are rewritten through player_model.pipeline._save_history (only changed parts).

NO MODEL READS THEM: props training and serving take profiles from output/collect/player_profiles
through feature_engineering._attach_profiles and read a stored 25 / 180 / 75 as missing
(_PROFILE_PLACEHOLDERS); fantasy masks them (_mask_profile_placeholders). Proof is the props
train/serve parity run before and after (scripts/props_train_serve_parity.py).

    python scripts/null_player_history_profile_placeholders.py            # dry run: counts only
    python scripts/null_player_history_profile_placeholders.py --apply    # rewrite + sidecar
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLACEHOLDER = {"age": 25.0, "height_cm": 180.0, "weight_kg": 75.0, "age_peak_delta": 0.0}
COMPOSITES = ("opp_def_mean_height", "height_diff_vs_opp_def", "height_advantage_score",
              "height_aerial_interaction")
KEYS = ("fixture_id", "player_id", "date", "team")
TAG = "2026-11-23"
SIDECAR = ROOT / "output" / "corrections" / f"player_history_profile_placeholders_{TAG}.parquet"
AUDIT = ROOT / "output" / "corrections" / f"player_history_profile_placeholders_{TAG}_audit.json"


def placeholder_rows(df: pd.DataFrame) -> pd.Series:
    """True where all four placeholder values sit on the row together (the legacy signature)."""
    m = pd.Series(True, index=df.index)
    for c, v in PLACEHOLDER.items():
        if c not in df.columns:
            return pd.Series(False, index=df.index)
        m &= pd.to_numeric(df[c], errors="coerce") == v
    return m


def null_placeholders(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(rewritten frame, sidecar of old values, counts). Pure function of the stored rows."""
    out = df.copy()
    m = placeholder_rows(df)
    cols = [c for c in (*PLACEHOLDER, *COMPOSITES) if c in df.columns]
    side = df.loc[m, [k for k in KEYS if k in df.columns]].copy()
    for c in cols:
        side[f"old_{c}"] = pd.to_numeric(df.loc[m, c], errors="coerce").to_numpy()
    counts = {"rows": int(len(df)), "placeholder_rows": int(m.sum())}
    for c in cols:
        before = pd.to_numeric(df[c], errors="coerce")
        counts[f"{c}_to_nan"] = int((m & before.notna()).sum())
        out[c] = before.where(~m)
        if df[c].dtype != out[c].dtype and str(df[c].dtype).startswith("float"):
            out[c] = out[c].astype(df[c].dtype)
    for c, v in PLACEHOLDER.items():   # a real value equal to the placeholder number, kept
        counts[f"{c}_equal_to_placeholder_kept_real"] = int(((pd.to_numeric(df[c], errors="coerce") == v) & ~m).sum())
    return out, side.reset_index(drop=True), counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--path", default=str(ROOT / "player_history.parquet"))
    a = ap.parse_args(argv)
    from player_model.pipeline import _save_history
    df = pd.read_parquet(a.path)
    new, side, counts = null_placeholders(df)
    src = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    audit = {"artifact": "player_history_profile_placeholders", "decided": "owner 2026-11-23 \"Yes, do all\" (D4)",
             "source_commit": src, "mode": "apply" if a.apply else "dry_run",
             "rule": "row with age == 25.0 AND height_cm == 180 AND weight_kg == 75 AND age_peak_delta == 0 "
                     "(the legacy build_features placeholders): those four and the composites computed from "
                     "them -> NaN; old values in the sidecar; real values equal to a placeholder number kept",
             "placeholder": PLACEHOLDER, "composites": list(COMPOSITES), "counts": counts,
             "sidecar": SIDECAR.relative_to(ROOT).as_posix()}
    print(json.dumps(counts, indent=1))
    if a.apply:
        side.to_parquet(SIDECAR, index=False)
        st = _save_history(new, Path(a.path))
        audit["save_history"] = st
        AUDIT.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
