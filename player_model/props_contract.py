"""
Props feature contract + train-fitted imputer (R2-D, findings 20 and 21, 2026-11-01).

WHY. Until R2 the props models imputed every missing feature with the MEDIAN OF THE DATAFRAME
BEING PROCESSED (`model._prep`): in training that frame is the whole history, holdout included,
so the holdout statistics leaked into the fit; at prediction time it is one fixture's ~20-40
players, so a player's probability depended on who else happened to be in the batch. A feature
COLUMN that was absent from the scoring frame was silently filled with 0.0 for every row.

This module is the repaired preprocessing used by CHALLENGER payloads (recipe
`props_r2_repaired`). The serving (legacy) payloads keep `model._prep` byte for byte until an
owner-approved retrain replaces them; see model.predict_proba for the dispatch.

Feature contract (one class per feature, persisted in props_feature_contract.json):

  REQUIRED_FAIL_ROW  the row has no meaningful prop probability without it (role, playing time,
                     the player's own core rates, venue). Missing -> the row gets NO probability.
  SEMANTIC_ZERO      a missing value genuinely means zero (provider omits event counts that did
                     not happen). Missing -> 0.0, counted as missing.
  IMPUTABLE          derived from the player's own history and normally present. Missing -> the
                     median of the FIT block (never calibration/holdout/live), counted as missing.
  OPTIONAL_DEGRADED  comes from an external enrichment that can be unavailable at serve time
                     (injuries, referee, season stats, physical profile, league/opponent quality,
                     set-piece roles). Missing -> FIT-block median, counted as missing AND degraded.

The contract is versioned (CONTRACT_VERSION) and hashed; every challenger payload and every
scored challenger row records imputer version, contract version, missing count, degraded count.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_VERSION = "PFC-1.0"
IMPUTER_VERSION = "PROPS-IMPUTER-1.0"
RECIPE_REPAIRED = "props_r2_repaired"
RECIPE_LEGACY = "props_legacy_batch_median"

CONTRACT_PATH = Path(__file__).resolve().parent / "props_feature_contract.json"

CLASSES = ("REQUIRED_FAIL_ROW", "SEMANTIC_ZERO", "IMPUTABLE", "OPTIONAL_DEGRADED")

_REQUIRED = {
    "is_home": "venue; every row in training has it, a fixture without it is not identified",
    "n_prev_games": "history depth; without it the player is a cold start, not a prop candidate",
    "minutes_pg": "playing time drives every count market",
    "starter_rate": "role; a missing starter rate makes every count market unscorable",
    "pos_forward": "position encoding", "pos_midfielder": "position encoding",
    "pos_defender": "position encoding",
    "goals_pg": "core rate", "assists_pg": "core rate", "shots_pg": "core rate",
    "sot_pg": "core rate", "cards_pg": "core rate",
}
_SEMANTIC_ZERO = {
    "red_cards_pg": "API-Football reports null for an event that did not happen",
    "penalties_won_pg": "API-Football reports null for an event that did not happen",
    "offsides_pg": "API-Football reports null for an event that did not happen",
}
_DEGRADED_REASON = {
    "injury": "injury feed (point-in-time since 2026-10-26; serving models trained on leaked values)",
    "referee": "referee assignment is often unknown until close to kickoff",
    "season": "API-Football /players/statistics season priors",
    "profile": "API-Football /players profile (age, height)",
    "quality": "league / opponent-quality enrichment",
    "opp_detail": "opponent lineup-level detail (GK, defenders)",
    "prev_season": "the player's / team's previous season (absent for a debut season)",
}
_DEGRADED = {
    **{c: "injury" for c in ("days_since_last_injury", "return_from_injury_flag", "chronic_injury_risk")},
    **{c: "referee" for c in ("referee_yellows_pg", "referee_strictness", "card_clash_index")},
    **{c: "season" for c in ("season_goals_pg", "season_assists_pg", "season_shots_pg", "season_sot_pg",
                             "season_cards_pg", "season_minutes_pg", "season_appearances",
                             "season_start_rate", "season_dribble_pg", "season_fouls_pg",
                             "season_fouls_drawn_pg")},
    **{c: "profile" for c in ("age", "age_peak_delta", "height_cm", "height_aerial_interaction",
                              "opp_def_mean_height", "height_diff_vs_opp_def", "height_advantage_score")},
    **{c: "quality" for c in ("league_tier", "league_quality", "player_career_avg_quality",
                              "context_quality_discount", "quality_mismatch_goals", "quality_mismatch_sot",
                              "opp_strength_index", "quality_adj_goals_pg", "quality_adj_sot_pg")},
    **{c: "opp_detail" for c in ("opp_gk_save_rate", "opp_gk_saves_pg", "opp_def_player_rating_pg",
                                 "opp_top_def_rating", "opp_def_player_quality")},
    **{c: "prev_season" for c in ("prev_season_goals_pg", "prev_season_assists_pg", "prev_season_shots_pg",
                                  "prev_season_sot_pg", "prev_season_cards_pg", "prev_season_minutes_pg",
                                  "prev_season_start_rate", "prev_season_appearances",
                                  "prev_season_shot_accuracy", "prev_season_team_goals_pg",
                                  "prev_season_team_conceded_pg", "prev_season_opp_goals_pg",
                                  "prev_season_opp_conceded_pg")},
}


def classify(feature: str) -> tuple[str, str]:
    """(class, reason) for one feature. Unknown features default to IMPUTABLE (own-history
    derived) - the contract file is the reviewed record, this is only its generator."""
    if feature in _REQUIRED:
        return "REQUIRED_FAIL_ROW", _REQUIRED[feature]
    if feature in _SEMANTIC_ZERO:
        return "SEMANTIC_ZERO", _SEMANTIC_ZERO[feature]
    if feature in _DEGRADED:
        return "OPTIONAL_DEGRADED", _DEGRADED_REASON[_DEGRADED[feature]]
    return "IMPUTABLE", "derived from the player's own match history"


def build_contract(features: list[str]) -> dict:
    feats = {f: dict(zip(("class", "reason"), classify(f))) for f in features}
    body = {"contract_version": CONTRACT_VERSION, "features": feats}
    body["contract_sha256"] = _sha(feats)
    body["counts"] = {k: sum(1 for v in feats.values() if v["class"] == k) for k in CLASSES}
    return body


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    if c.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"props feature contract version {c.get('contract_version')!r} "
                         f"!= code {CONTRACT_VERSION!r}")
    if _sha(c["features"]) != c.get("contract_sha256"):
        raise ValueError("props feature contract hash mismatch: the file was edited without "
                         "re-hashing (regenerate with python -m player_model.props_contract --write)")
    return c


class PropsImputer:
    """Train-fitted, persisted imputer. fit() sees ONLY the fit block; transform() is a pure
    per-row function of (row, fitted state) - no statistic of the scoring batch is used, so a
    row scores identically alone, in any batch, in any order."""

    def __init__(self, contract: dict | None = None):
        self.contract = contract or load_contract()
        self.fill_: dict[str, float] = {}
        self.feature_cols_: list[str] = []
        self.dropped_unfittable_: list[str] = []
        self.fit_info_: dict = {}

    # ---- fit ---------------------------------------------------------------------------
    def fit(self, df_fit: pd.DataFrame, features: list[str], date_col: str = "date") -> "PropsImputer":
        classes = self.contract["features"]
        unknown = [f for f in features if f not in classes]
        if unknown:
            raise ValueError(f"features not in props contract {CONTRACT_VERSION}: {unknown[:10]}")
        self.feature_cols_, self.fill_, self.dropped_unfittable_ = [], {}, []
        for f in features:
            cls = classes[f]["class"]
            if cls == "SEMANTIC_ZERO":
                self.fill_[f] = 0.0
                self.feature_cols_.append(f)
                continue
            col = pd.to_numeric(df_fit[f], errors="coerce") if f in df_fit.columns else pd.Series(dtype=float)
            med = float(col.median()) if col.notna().any() else float("nan")
            if np.isnan(med):
                # No training information at all: drop rather than invent a constant.
                self.dropped_unfittable_.append(f)
                continue
            self.fill_[f] = med
            self.feature_cols_.append(f)
        info = {"fit_rows": int(len(df_fit))}
        if date_col in df_fit.columns and len(df_fit):
            d = pd.to_datetime(df_fit[date_col], errors="coerce")
            info["fit_start"] = str(d.min())[:10]
            info["fit_end"] = str(d.max())[:10]
        self.fit_info_ = info
        return self

    # ---- transform ---------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(X, diag). X has exactly feature_cols_ (float, no NaN). diag per row: n_missing,
        n_degraded, n_required_missing, required_fail, imputer_version, contract_version.
        Rows with required_fail keep a filled X row but MUST NOT be given a probability."""
        classes = self.contract["features"]
        cols, mcols = {}, {}
        for f in self.feature_cols_:
            v = pd.to_numeric(df[f], errors="coerce") if f in df.columns \
                else pd.Series(np.nan, index=df.index, dtype=float)
            mcols[f] = v.isna()
            cols[f] = v.fillna(self.fill_[f]).astype(float)
        X = pd.DataFrame(cols, index=df.index)[self.feature_cols_]
        miss = pd.DataFrame(mcols, index=df.index)
        req = [f for f in self.feature_cols_ if classes[f]["class"] == "REQUIRED_FAIL_ROW"]
        deg = [f for f in self.feature_cols_ if classes[f]["class"] == "OPTIONAL_DEGRADED"]
        diag = pd.DataFrame(index=df.index)
        diag["n_missing"] = miss.sum(axis=1).astype(int)
        diag["n_degraded"] = miss[deg].sum(axis=1).astype(int) if deg else 0
        diag["n_required_missing"] = miss[req].sum(axis=1).astype(int) if req else 0
        diag["required_fail"] = diag["n_required_missing"] > 0
        diag["imputer_version"] = IMPUTER_VERSION
        diag["contract_version"] = self.contract["contract_version"]
        return X, diag

    # ---- persistence -------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "imputer_version": IMPUTER_VERSION,
            "contract_version": self.contract["contract_version"],
            "contract_sha256": self.contract["contract_sha256"],
            "contract_classes": {f: self.contract["features"][f]["class"] for f in self.feature_cols_},
            "feature_cols": list(self.feature_cols_),
            "fill_values": dict(self.fill_),
            "dropped_unfittable": list(self.dropped_unfittable_),
            "fit_info": dict(self.fit_info_),
        }

    @classmethod
    def from_state(cls, state: dict) -> "PropsImputer":
        if state.get("imputer_version") != IMPUTER_VERSION:
            raise ValueError(f"imputer version {state.get('imputer_version')!r} unsupported")
        contract = {
            "contract_version": state["contract_version"],
            "contract_sha256": state["contract_sha256"],
            "features": {f: {"class": c} for f, c in state["contract_classes"].items()},
        }
        imp = cls(contract)
        imp.feature_cols_ = list(state["feature_cols"])
        imp.fill_ = {k: float(v) for k, v in state["fill_values"].items()}
        imp.dropped_unfittable_ = list(state.get("dropped_unfittable", []))
        imp.fit_info_ = dict(state.get("fit_info", {}))
        return imp


def main() -> None:
    import argparse
    from . import config
    ap = argparse.ArgumentParser(description="Generate / check the props feature contract")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    body = build_contract(list(config.PLAYER_FEATURE_COLS))
    if a.write:
        CONTRACT_PATH.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {CONTRACT_PATH} ({body['counts']})")
    else:
        cur = load_contract()
        print("OK" if cur["contract_sha256"] == body["contract_sha256"] else "DRIFT", body["counts"])


if __name__ == "__main__":
    main()
