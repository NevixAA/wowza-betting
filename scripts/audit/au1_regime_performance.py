"""AU1 Phase 7 - regime-scoped O/U 2.5 and side-market performance (READ-ONLY).

Every live ledger row is stamped with the regime that was IN FORCE on origin/main when it was
generated (git commit time of each change; a predict run uses the SHA it checked out, so rows within
~1 run of a boundary can belong to either side - reported as boundary_rows):

  thr_epoch   : (predict.yml env tier gates) x (models/best_params_standard.json version)
  drift       : INERT (<2026-08-15 13:54Z, history not committed) / ACTIVE / OFF (>=2026-10-26 09:43Z)
  std_model   : blob of models/model_v9_standard.pkl in force, or REPAIRED (>=2026-10-28 12:38Z, env switch)
  nf_model    : blob of models/model_v9_newformat.pkl in force
  form_guard  : ON / OFF (2026-08-21 16:11Z .. 2026-09-15)
  nf_cap      : NONE / 3.00 (>=2026-10-22 06:10Z)

Comparisons are made ONLY inside one regime cell; nothing is pooled across cells in a headline.
Settlement: WIN -> odds-1, LOSS -> -1, VOID/blank excluded (never 0-filled). Flat 1u per row.
Writes reports/au1_data/regime_ledger_rows.csv, regime_performance.csv, regime_epochs.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from au1_fallback_evidence import block_ci  # noqa: E402

OUT = ROOT / "reports" / "au1_data"
O = ROOT / "output"
T = lambda s: pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC")  # noqa: E731

ENV_EPOCHS = [("ENV_DEFAULT(M0.14,V0.04,noCap)", T("2026-06-17T11:05:39Z")),
              ("ENV_WIDE(M0.08,V0.03,Cap0.12)", T("2026-08-21T16:06:04Z"))]
DRIFT = [("INERT", T("2026-04-01T00:00:00Z")), ("ACTIVE", T("2026-08-15T13:54:01Z")), ("OFF", T("2026-10-26T09:43:49Z"))]
FORM_OFF = (T("2026-08-21T16:11:04Z"), T("2026-09-15T00:00:00Z"))
NF_CAP = T("2026-10-22T06:10:54Z")
REPAIRED = T("2026-10-28T12:38:07Z")
BOUNDARY = pd.Timedelta(minutes=30)


def epoch_of(ts: pd.Series, epochs: list[tuple[str, pd.Timestamp]], before: str) -> pd.Series:
    out = pd.Series(before, index=ts.index, dtype=object)
    for name, start in epochs:
        out[ts >= start] = name
    return out


def artifact_epochs(path: str) -> list[tuple[str, pd.Timestamp]]:
    a = pd.read_csv(OUT / "artifact_epochs.csv")
    a = a[a["artifact"] == path].copy()
    a["t"] = pd.to_datetime(a["committed_at"], utc=True)
    a = a.sort_values("t")
    return [(f"{r.blob}@{r.t:%m-%d}", r.t) for r in a.itertuples()]


def perf(g: pd.DataFrame) -> dict:
    u = g["unit"].to_numpy(float)
    lo, hi = block_ci(u, g["match_date"].astype(str).to_numpy())
    p = g["p_model"].to_numpy(float)
    y = g["won"].to_numpy(float)
    q = 1.0 / g["odds"].to_numpy(float)
    return {"n": int(len(g)), "matchdays": int(g["match_date"].nunique()), "hit": round(float(y.mean()), 4),
            "mean_p_model": round(float(np.nanmean(p)), 4), "mean_p_price_raw": round(float(q.mean()), 4),
            "brier_model": round(float(np.nanmean((p - y) ** 2)), 4), "brier_price_raw": round(float(np.mean((q - y) ** 2)), 4),
            "units": round(float(u.sum()), 2), "roi": round(float(u.mean()), 4),
            "ci_lo": round(lo, 4) if lo == lo else None, "ci_hi": round(hi, 4) if hi == hi else None,
            "mean_odds": round(float(g["odds"].mean()), 3)}


def stamp(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["gen"] = pd.to_datetime(d["generated_at"], utc=True, errors="coerce")
    d["match_date"] = d["match_date"].astype(str).str[:10]
    bp = artifact_epochs("models/best_params_standard.json")
    env = epoch_of(d["gen"], ENV_EPOCHS, "ENV_PRE0617")
    bpe = epoch_of(d["gen"], [(f"BP{i+1}({n.split('@')[1]})", t) for i, (n, t) in enumerate(bp)], "BP0(config)")
    d["thr_epoch"] = env + "|" + bpe
    d["drift"] = epoch_of(d["gen"], DRIFT, "INERT")
    std = epoch_of(d["gen"], artifact_epochs("models/model_v9_standard.pkl"), "std_pre_git")
    std[d["gen"] >= REPAIRED] = "REPAIRED"
    d["std_model"] = std
    d["nf_model"] = epoch_of(d["gen"], artifact_epochs("models/model_v9_newformat.pkl"), "nf_pre_git")
    d["form_guard"] = np.where((d["gen"] >= FORM_OFF[0]) & (d["gen"] < FORM_OFF[1]), "OFF", "ON")
    d["nf_cap"] = np.where(d["gen"] >= NF_CAP, "3.00", "NONE")
    bounds = [t for _, t in ENV_EPOCHS + DRIFT + bp] + [FORM_OFF[0], NF_CAP, REPAIRED]
    d["boundary_row"] = False
    for b in bounds:
        d.loc[(d["gen"] - b).abs() <= BOUNDARY, "boundary_row"] = True
    res = d["result"].astype(str).str.upper()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d["settled"] = res.isin(["WIN", "LOSS"]) & d["odds"].gt(1)
    d["won"] = res.eq("WIN").astype(float)
    d["unit"] = np.where(res.eq("WIN"), d["odds"] - 1, np.where(res.eq("LOSS"), -1.0, np.nan))
    return d


def main() -> None:
    b = pd.read_csv(O / "bets_ledger.csv", low_memory=False)
    b = b[b["source"].astype(str).eq("live") & b["side"].isin(["OVER", "UNDER"])]
    b = stamp(b)
    b["p_model"] = pd.to_numeric(b["edge_pct"], errors="coerce") / 100 + 1 / b["odds"]
    b["family"] = "OU25_" + b["model_type"].astype(str)
    b["model_epoch"] = np.where(b["model_type"].eq("standard"), b["std_model"], b["nf_model"])
    s = pd.read_csv(O / "side_bets_ledger.csv", low_memory=False)
    s = stamp(s)
    s["p_model"] = pd.to_numeric(s["model_prob"], errors="coerce")
    s["side"] = s["market"].map({"btts": "YES", "over15": "OVER", "over35": "OVER"})
    s["family"] = s["market"].str.upper() + "_" + s["model_type"].astype(str)
    s["model_epoch"] = "side(see artifact_epochs)"
    cols = ["family", "generated_at", "match_date", "league", "home_team", "away_team", "side", "signal_tier", "odds",
            "p_model", "result", "won", "unit", "thr_epoch", "drift", "model_epoch", "form_guard", "nf_cap", "boundary_row", "settled"]
    rows = pd.concat([b[cols], s[cols]], ignore_index=True)
    rows.to_csv(OUT / "regime_ledger_rows.csv", index=False)

    perf_rows = []
    st = rows[rows["settled"] & ~rows["boundary_row"]]
    for start in ("2026-08-10", "2026-09-01"):
        x = st[st["match_date"] >= start]
        for keys in (["family", "thr_epoch", "signal_tier"], ["family", "thr_epoch", "side"],
                     ["family", "model_epoch"], ["family", "drift"], ["family", "nf_cap", "side"],
                     ["family", "form_guard"]):
            for k, g in x.groupby(keys):
                perf_rows.append({"from": start, "cut": "+".join(keys), "cell": " | ".join(map(str, k)), **perf(g)})
    pd.DataFrame(perf_rows).to_csv(OUT / "regime_performance.csv", index=False)
    summary = {"ledger_rows": int(len(rows)), "settled": int(rows["settled"].sum()),
               "boundary_rows_excluded": int((rows["settled"] & rows["boundary_row"]).sum()),
               "rows_by_thr_epoch": {" | ".join(k): int(v) for k, v in rows.groupby(["family", "thr_epoch"]).size().items()}}
    (OUT / "regime_summary.json").write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
