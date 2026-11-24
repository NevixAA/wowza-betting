"""Predictions page - "Success by gameweek" helpers (owner 2026-11-23).

Reads the stored dataset src/prediction_gw_dataset.py writes after settlement (update_results.yml):
output/prediction_gw_dataset.parquet (one row per gameweek x league x model x model version x market x line x
selection group x snapshot x Clean Record flag) and its corrections file. No price is in either. Every rate shown is
re-derived from the stored SUMS (never an average of averages) and is never pooled across model versions: the version
is always a column. A rate with no settled question stays empty.
Reload-safe: no module state beyond constants.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "output" / "prediction_gw_dataset.parquet"
CORRECTIONS = ROOT / "output" / "corrections" / "prediction_gw_dataset_corrections.csv"

RECORD_LABEL = {True: "Clean Record", False: "Before the Clean Record"}
GROUP = ["gw", "gw_start_date", "gw_end_date", "league", "model_id", "model_version", "release", "market",
         "line", "status"]
SUMS = ["predictions", "settled", "hits", "void", "pending", "unscorable", "brier_sum", "log_loss_sum"]


def load(path: Path | None = None) -> pd.DataFrame:
    """The dataset + a readable `release` per row ('<release name> · <install date>', src.model_releases; the raw
    model_version stays beside it). Raises OSError / ValueError when it is missing or unreadable."""
    d = pd.read_parquet(path or DATASET)
    for c in ("clean_record", "version_pre_clean_record"):
        d[c] = d[c].astype(str).str.lower().isin(["true", "1"])
    from src.model_releases import Releases
    R = Releases()
    keys = d[["model_id", "market", "model_version"]].drop_duplicates()
    lab = {(m, k, v): R.display(m, k, v) for m, k, v in keys.itertuples(index=False, name=None)}
    d["release"] = [lab[(m, k, v)] for m, k, v in zip(d["model_id"], d["market"], d["model_version"])]
    return d


def load_corrections(path: Path | None = None) -> pd.DataFrame:
    p = Path(path or CORRECTIONS)
    if not p.exists():
        return pd.DataFrame(columns=["row_id", "reason", "field"])
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def rollup(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sums per `by`, rates re-derived: success % = hits / settled; Brier / log loss = sum / settled."""
    if d.empty:
        return pd.DataFrame(columns=by + SUMS + ["success", "brier", "log_loss"])
    g = d.groupby(by, sort=True, dropna=False)[SUMS].sum().reset_index()
    n = g["settled"].where(g["settled"] > 0)
    g["success"] = g["hits"] / n
    g["brier"] = g["brier_sum"] / n
    g["log_loss"] = g["log_loss_sum"] / n
    return g


def table(d: pd.DataFrame, corrected: set, labels: dict, markets: dict) -> pd.DataFrame:
    """Display table: one line per GW x league x model x version x market x line (selection groups summed)."""
    if d.empty:
        return pd.DataFrame()
    ids = d.groupby(GROUP, dropna=False)["row_id"].apply(lambda s: bool(set(s) & corrected)).reset_index()
    t = rollup(d, GROUP).merge(ids, on=GROUP, how="left")
    t = t.sort_values(["gw", "league", "model_id", "market", "line", "model_version"],
                      ascending=[False, True, True, True, True, True], kind="mergesort")
    return pd.DataFrame({
        "GW": t["gw"].values,
        "Dates": [f"{a} – {b}" for a, b in zip(t["gw_start_date"], t["gw_end_date"])],
        "League": t["league"].values,
        "Model": t["model_id"].map(lambda m: labels.get(m, m)).values,
        "Version": t["release"].values,
        "Version id": t["model_version"].values,
        "Market": t["market"].map(lambda m: markets.get(m, m)).values,
        "Line": t["line"].astype(str).values,
        "Predictions": t["predictions"].astype(int).values,
        "Settled": t["settled"].astype(int).values,
        "Hits": t["hits"].astype(int).values,
        "Success": t["success"].values,
        "Brier": t["brier"].values,
        "Log loss": t["log_loss"].values,
        "Void": t["void"].astype(int).values,
        "Pending": t["pending"].astype(int).values,
        "GW status": t["status"].map({"CLOSED": "closed", "OPEN": "open"}).values,
        "Correction": ["see corrections" if x else "" for x in t["row_id"]],
    }).reset_index(drop=True)


def trend(d: pd.DataFrame, labels: dict, markets: dict, max_series: int = 12) -> pd.DataFrame:
    """Success % per GW (rows) for each model · version · market (columns), summed over the filtered leagues; the
    `max_series` series with the most settled questions."""
    if d.empty:
        return pd.DataFrame()
    t = rollup(d, ["gw", "model_id", "release", "market"])
    t = t[t["settled"] > 0]
    if t.empty:
        return pd.DataFrame()
    t["series"] = [f"{labels.get(m, m)} · {v} · {markets.get(k, k)}"
                   for m, v, k in zip(t["model_id"], t["release"], t["market"])]
    top = t.groupby("series")["settled"].sum().sort_values(ascending=False).head(max_series).index
    t = t[t["series"].isin(top)]
    return t.pivot_table(index="gw", columns="series", values="success", aggfunc="first").sort_index() * 100
