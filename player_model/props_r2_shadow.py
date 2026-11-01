"""R2-D repaired props recipe, scored LIVE in shadow next to the serving props models (2026-11-01).

    python -m player_model.props_r2_shadow            # report: both recipes on settled props

WHY. R2-D built a repaired props recipe (persisted train-fitted imputer, feature contract that
counts every missing input; challengers in models/challengers/props_r2/) and replayed it on history,
where it was indistinguishable from legacy. The legacy recipe still serves - with batch-median
imputation (a prop's value depends on who else is in the batch) and the invented constants of Data
Truth finding #42. Owner decision 2026-11-01: do not switch blind while props perform; score the
repaired recipe on the SAME live player-fixtures, first sight, and let settled results decide.

RECORDER ONLY. It never changes a probability, a tier or a tip, and never raises into predict.
Log: output/props_r2_shadow.csv, one row per (market, fixture_id, player_id) at FIRST sight, before
kickoff. Promotion is a separate owner decision, taken on the report below.
"""
from __future__ import annotations

import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHALLENGER_DIR = ROOT / "models" / "challengers" / "props_r2"
LOG = ROOT / "output" / "props_r2_shadow.csv"
KEY = ["market", "fixture_id", "player_id"]
COLUMNS = ["logged_at", "market", "fixture_id", "player_id", "player_name", "league", "match_date",
           "kickoff_utc", "p_served", "p_r2", "r2_required_fail", "r2_version"]


def load_challengers() -> dict:
    """{market: payload} for every committed R2-D challenger. Never raises."""
    out = {}
    for p in sorted(CHALLENGER_DIR.glob("props_r2_*.pkl")):
        try:
            with open(p, "rb") as f:
                out[p.stem.replace("props_r2_", "")] = pickle.load(f)
        except Exception as e:                                         # noqa: BLE001
            print(f"[props_r2_shadow] could not load {p.name}: {e}")
    return out


class Recorder:
    def __init__(self, challengers: dict | None = None):
        self.ch = load_challengers() if challengers is None else challengers
        self.rows: list[dict] = []

    def add(self, market: str, feat_df: pd.DataFrame, p_served, *, league: str, match_date: str,
            kickoff_utc: str = "", fixture_id=None) -> None:
        payload = self.ch.get(market)
        if payload is None or feat_df is None or feat_df.empty:
            return
        try:
            from player_model.model import predict_with_diagnostics
            diag = predict_with_diagnostics(feat_df, payload)
            p_r2 = diag["p"].to_numpy()
            fail = diag["required_fail"].to_numpy() if "required_fail" in diag else np.zeros(len(feat_df), bool)
        except Exception as e:                                         # noqa: BLE001
            print(f"[props_r2_shadow] {market}: scoring failed ({e}) - nothing logged")
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ver = str(payload.get("model_recipe_version") or payload.get("recipe") or "props_r2")
        for i, (_, r) in enumerate(feat_df.iterrows()):
            self.rows.append({"logged_at": now, "market": market, "fixture_id": str(fixture_id or ""),
                              "player_id": str(r.get("player_id", "")), "player_name": r.get("player_name", ""),
                              "league": league, "match_date": match_date, "kickoff_utc": kickoff_utc,
                              "p_served": float(p_served[i]), "p_r2": float(p_r2[i]),
                              "r2_required_fail": bool(fail[i]), "r2_version": ver})

    def flush(self, path: Path = LOG) -> int:
        """Append FIRST-SIGHT rows only (a key already logged is never rewritten). Never raises."""
        if not self.rows:
            return 0
        try:
            new = pd.DataFrame(self.rows, columns=COLUMNS).drop_duplicates(KEY, keep="first")
            new = new[new["fixture_id"] != ""]
            if path.exists():
                prev = pd.read_csv(path, dtype=str, keep_default_na=False)
                seen = set(map(tuple, prev[KEY].to_numpy()))
                new = new[[tuple(k) not in seen for k in new[KEY].astype(str).to_numpy()]]
                if new.empty:
                    return 0
                new.to_csv(path, mode="a", header=False, index=False)
            else:
                new.to_csv(path, index=False)
            return int(len(new))
        except Exception as e:                                         # noqa: BLE001
            print(f"[props_r2_shadow] flush failed: {e}")
            return 0


def _ll(p, y):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def report(log_path: Path = LOG, ledger_path: Path = ROOT / "output" / "player_ledger.csv") -> dict:
    """Both recipes on the same SETTLED props: n, log loss, Brier, per market. Never raises."""
    try:
        s = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        led = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    except Exception:                                                  # noqa: BLE001
        return {"n": 0, "note": "no shadow log or ledger yet"}
    led = led[led["result"].isin(["WIN", "LOSS"])]
    led = led.assign(y=(led["result"] == "WIN").astype(int),
                     md=led["match_date"].str[:10], pid=led.get("player_id", "").astype(str).str.split(".").str[0])
    s = s.assign(md=s["match_date"].str[:10], pid=s["player_id"].astype(str).str.split(".").str[0])
    j = s.merge(led[["market", "pid", "md", "y"]], on=["market", "pid", "md"], how="inner")
    j = j[(j["r2_required_fail"] != "True")]
    j["p_served"] = pd.to_numeric(j["p_served"], errors="coerce")
    j["p_r2"] = pd.to_numeric(j["p_r2"], errors="coerce")
    j = j.dropna(subset=["p_served", "p_r2"])
    out = {"n": int(len(j)), "by_market": {}}
    for mk, g in j.groupby("market"):
        out["by_market"][mk] = {
            "n": int(len(g)),
            "logloss_served": round(float(np.mean([_ll(p, y) for p, y in zip(g["p_served"], g["y"])])), 5),
            "logloss_r2": round(float(np.mean([_ll(p, y) for p, y in zip(g["p_r2"], g["y"])])), 5),
            "brier_served": round(float(((g["p_served"] - g["y"]) ** 2).mean()), 5),
            "brier_r2": round(float(((g["p_r2"] - g["y"]) ** 2).mean()), 5)}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=1))
