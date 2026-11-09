"""Props train/serve parity on REAL fixtures (owner 2026-11-09: identical, not within tolerance).

    python scripts/props_train_serve_parity.py --work <dir> --models <dir with props_clean_*.pkl> \
        --out <json> [--per-league 8 --start 2026-10-18 --end 2026-10-31] [--rebuild]

<dir> holds raw.parquet (props_harness.source_frame()) and frame.parquet (props_harness.build_frame);
they are (re)built with the CURRENT code when absent or with --rebuild. For the sampled fixtures:

  serving  = feature_engineering.build_upcoming_features(rows, history strictly before the fixture
             day), the rows carrying only what predict passes (player, fixture, date, league, teams,
             venue, referee name) - every result missing
  training = the frame row of the same (fixture_id, player_id)

Every feature of every promoted model and every model probability is compared for EXACT equality
(NaN equals NaN; no float tolerance). Writes <out> (summary), <out>_examples.json and
<out>_pairs.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import props_harness as H  # noqa: E402
from player_model import props_gate as PPG  # noqa: E402
from player_model.feature_engineering import build_upcoming_features  # noqa: E402

SERVED = (("sot", "hgb_nan_native"), ("sot2", "hgb_nan_native"), ("goals", "hgb_nan_native"),
          ("assists", "hgb_nan_native"), ("cards", "recipe_retrained"))
ROW_FIELDS = ("player_id", "player_name", "fixture_id", "league", "team", "home_team", "away_team",
              "is_home", "referee")


def _load_model(models: Path, market: str, cand: str):
    import io
    raw = (models / f"props_clean_{market}_{cand}.pkl").read_bytes()
    return PPG._HarnessUnpickler(io.BytesIO(raw)).load()["model"]


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return (a.isna() & b.isna()) | (a == b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-league", type=int, default=2)
    ap.add_argument("--start", default="2026-10-24")
    ap.add_argument("--end", default="2026-10-31")
    ap.add_argument("--markets", default=",".join(f"{m}:{c}" for m, c in SERVED))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    work, models = Path(a.work), Path(a.models)
    if a.rebuild or not (work / "frame.parquet").exists():
        raw = H.source_frame()
        raw.to_parquet(work / "raw.parquet")
        H.build_frame(raw).to_parquet(work / "frame.parquet")
    raw = pd.read_parquet(work / "raw.parquet")
    frame = pd.read_parquet(work / "frame.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date", kind="stable").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    served = [tuple(x.split(":")) for x in a.markets.split(",")]
    mods = {m: _load_model(models, m, c) for m, c in served}
    feats = sorted(set().union(*[set(mod.feature_cols_) for mod in mods.values()]))
    cand = frame[(frame["date"] >= a.start) & (frame["date"] <= a.end)]
    fx = (cand.drop_duplicates("fixture_id")[["fixture_id", "date", "league"]]
          .groupby("league", group_keys=False).head(a.per_league))
    print("fixtures", len(fx), "leagues", fx["league"].nunique(), flush=True)
    parts = []
    for d, fd in fx.groupby("date"):
        hist = raw.iloc[: int(raw["date"].searchsorted(d, side="left"))]
        rows = []
        for fid in fd["fixture_id"]:
            fr = raw[raw["fixture_id"] == fid]
            for _, r in fr.iterrows():
                rec = {k: r[k] for k in ROW_FIELDS if k in fr.columns}
                rec["date"] = d
                rows.append(rec)
        print("date", d.date(), "hist", len(hist), "rows", len(rows), flush=True)
        sv = build_upcoming_features(rows, hist)
        if len(sv):
            parts.append(sv)
    sv = pd.concat(parts, ignore_index=True)
    for df in (sv, frame):
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce")
    j = sv.merge(frame, on=["fixture_id", "player_id"], suffixes=("_s", "_t"))
    print("paired player-fixtures", len(j), flush=True)
    j.to_parquet(a.out.replace(".json", "_pairs.parquet"))
    rep = {"paired_rows": int(len(j)), "fixtures": int(j["fixture_id"].nunique()),
           "sample": {"per_league": a.per_league, "start": a.start, "end": a.end},
           "position_source": j["position_source_t"].value_counts().to_dict() if "position_source_t" in j else {},
           "features": {}, "probabilities": {}}
    examples = {}
    for c in feats:
        s, t = (c + "_s", c + "_t")
        if s not in j or t not in j:
            rep["features"][c] = {"mismatch": None, "note": "absent on one side"}
            continue
        ok = _same(j[s], j[t])
        rep["features"][c] = {"mismatch": int((~ok).sum())}
        if (~ok).any():
            examples[c] = j.loc[~ok, ["fixture_id", "player_id", s, t]].head(5).to_dict("records")
    rep["features_mismatched"] = sum(1 for v in rep["features"].values() if v.get("mismatch"))
    rep["features_absent"] = sorted(k for k, v in rep["features"].items() if v.get("mismatch") is None)
    for m, mod in mods.items():
        Xs = pd.DataFrame({c: pd.to_numeric(j[c + "_s"], errors="coerce") for c in mod.feature_cols_})
        Xt = pd.DataFrame({c: pd.to_numeric(j[c + "_t"], errors="coerce") for c in mod.feature_cols_})
        ps, pt = mod.predict(Xs), mod.predict(Xt)
        rep["probabilities"][m] = {"rows_not_identical": int((ps != pt).sum()),
                                   "max_abs_diff": float(np.max(np.abs(ps - pt))) if len(ps) else 0.0}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    Path(a.out.replace(".json", "_examples.json")).write_text(json.dumps(examples, indent=1, default=str),
                                                               encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("paired_rows", "features_mismatched", "features_absent",
                                          "position_source", "probabilities")}, indent=1, default=str))
    for k, v in examples.items():
        print(k, v[:2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
