"""Predictions page helpers (owner 2026-11-22): every model's prediction for every market of a match, its result
once settled, and the learning-record summary since the Clean Record start.

Reads the TWO compact files scripts/build_predictions_index.py writes from the v9 and Pro prediction ledgers
(update_results.yml refreshes them after settlement): reports/dashboard/predictions_index.parquet and
reports/dashboard/predictions_summary.csv. No price is in either file. v11 is private and never in them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports" / "dashboard" / "predictions_index.parquet"
SUMMARY = ROOT / "reports" / "dashboard" / "predictions_summary.csv"

LEDGER_ONLY_LABEL = "ledger only — no tips"

MODEL_LABEL = {
    "v9_standard": "v9 Standard",
    "v9_new_format": "v9 New Format",
    "v9_props": "v9 Props",
    "v9_props_challenger": "v9 Props challenger",
    "v9_corners": "v9 Corners",
    "v9_cards": "v9 Cards",
    "pro_bb_dc": "Pro score grid",
    "pro_bb_combo": "Pro combos",
}
#: The page's tabs: (title, model ids) - every model belongs to exactly one.
FAMILIES = [
    ("Team markets", ("v9_standard", "v9_new_format")),
    ("Player props", ("v9_props", "v9_props_challenger")),
    ("Corners & cards", ("v9_corners", "v9_cards")),
    ("Pro score grid", ("pro_bb_dc",)),
    ("Pro combos", ("pro_bb_combo",)),
]
MARKET_LABEL = {
    "ou15": "Over/Under 1.5", "ou25": "Over/Under 2.5", "ou35": "Over/Under 3.5", "btts": "Both teams to score",
    "ht_ou05": "Half-time O/U 0.5", "ht_ou15": "Half-time O/U 1.5",
    "goals": "Anytime scorer", "goals2": "2+ goals", "goals3": "3+ goals", "assists": "Assist", "cards": "Booked",
    "sot": "1+ shots on target", "sot2": "2+ shots on target", "sot3": "3+ shots on target",
    "sot4": "4+ shots on target",
    "corners_total": "Total corners", "corners_team": "Team corners", "cards_total": "Total cards",
    "cards_team": "Team cards",
    "1x2": "1X2", "correct_score": "Correct score", "goals_exact_0": "0 goals", "goals_le1": "1 goal or fewer",
    "goals_ge5": "5+ goals", "goals_ge6": "6+ goals", "goals_ge7": "7+ goals", "team_ge4": "Team 4+ goals",
    "combo": "Combo",
}
SNAPSHOT_LABEL = {"first_sight": "First sight", "final": "Final (≤120 min before kickoff)"}


def model_label(m) -> str:
    return MODEL_LABEL.get(str(m), str(m))


def market_label(m) -> str:
    return MARKET_LABEL.get(str(m), str(m))


def load_index(path: Path | None = None) -> pd.DataFrame:
    """The row index. Raises OSError / ValueError when the file is missing or unreadable."""
    d = pd.read_parquet(path or INDEX)
    for c in ("kickoff_utc", "predicted_at"):
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    for c in d.columns:
        if isinstance(d[c].dtype, pd.CategoricalDtype):
            d[c] = d[c].astype(str).replace("nan", "")
    for c in ("player_name", "player_id", "home_team", "away_team", "fixture_ref"):
        if c in d:
            d[c] = d[c].fillna("").astype(str)
    return d


def load_summary(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or SUMMARY, dtype={"line": str})


def league_options(idx: pd.DataFrame) -> dict[str, str]:
    """league_slug -> display name; ledger-only leagues say so."""
    lo = idx.groupby("league_slug")["ledger_only"].any()
    names = idx.groupby("league_slug")["league"].first()
    return {s: f"{names[s]} ({LEDGER_ONLY_LABEL})" if lo.get(s, False) else str(names[s])
            for s in sorted(names.index, key=lambda s: str(names[s]))}


def matches(idx: pd.DataFrame) -> pd.DataFrame:
    """One row per match in `idx` (a fixture can carry an OddsAPI and an API-Football ref: grouped by
    league + kickoff + teams, which is what the reader sees)."""
    if idx.empty:
        return pd.DataFrame(columns=["match_key", "kickoff_utc", "league_slug", "league", "home_team", "away_team",
                                     "models", "predictions", "settled", "ledger_only"])
    d = idx.assign(match_key=idx["league_slug"] + "|" + idx["kickoff_utc"].dt.strftime("%Y-%m-%dT%H:%MZ")
                   + "|" + idx["home_team"] + "|" + idx["away_team"])
    g = d.groupby("match_key", sort=False)
    out = g.agg(kickoff_utc=("kickoff_utc", "first"), league_slug=("league_slug", "first"),
                league=("league", "first"), home_team=("home_team", "first"), away_team=("away_team", "first"),
                models=("model_id", "nunique"), predictions=("probability", "size"),
                settled=("outcome_status", lambda s: int((s == "SETTLED").sum())),
                ledger_only=("ledger_only", "any")).reset_index()
    return out.sort_values(["kickoff_utc", "league", "home_team"], kind="mergesort").reset_index(drop=True)


def match_rows(idx: pd.DataFrame, match_key: str) -> pd.DataFrame:
    k = (idx["league_slug"] + "|" + idx["kickoff_utc"].dt.strftime("%Y-%m-%dT%H:%MZ") + "|" + idx["home_team"]
         + "|" + idx["away_team"])
    return idx[k == match_key]


def outcome_text(status: str, y) -> str:
    if status == "SETTLED" and pd.notna(y):
        return "happened" if int(y) == 1 else "did not happen"
    return {"VOID_DID_NOT_PLAY": "void · did not play", "NO_RESULT": "no result", "PENDING": "pending"}.get(
        str(status), str(status).lower() or "pending")


def result_text(row) -> str:
    """The model's pick on this row: hit / miss once scored; empty on a row that is not the pick."""
    if not bool(row["pick"]):
        return ""
    h = row["hit"]
    if pd.isna(h):
        return outcome_text(row["outcome_status"], None) if row["outcome_status"] != "SETTLED" else "not scored"
    return "hit" if int(h) == 1 else "miss"


def pick_text(row) -> str:
    """What the model picked on this row (a single-selection question picks YES or NO by p >= 0.5)."""
    if not bool(row["pick"]):
        return ""
    if int(row["q_size"]) > 1:
        return str(row["selection"])
    return str(row["selection"]) if float(row["probability"]) >= 0.5 else f"not {row['selection']}"


def prediction_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Display rows for one match and one family: model, market, line, selection, probability, pick, outcome."""
    if rows.empty:
        return pd.DataFrame()
    r = rows.sort_values(["model_id", "market", "line", "player_name", "model_version", "selection"],
                         kind="mergesort")
    out = pd.DataFrame({
        "Model": r["model_id"].map(model_label),
        "Market": r["market"].map(market_label),
        "Line": r["line"].astype(str).replace("nan", ""),
    })
    if (r["player_name"] != "").any():
        out["Player"] = r["player_name"].values
        out["Team"] = r["team"].astype(str).replace("nan", "").values
    out["Selection"] = r["selection"].values
    out["Probability"] = r["probability"].astype(float).values
    out["Pick"] = [pick_text(x) for _, x in r.iterrows()]
    out["Outcome"] = [outcome_text(s, y) for s, y in zip(r["outcome_status"], r["y"])]
    out["Result"] = [result_text(x) for _, x in r.iterrows()]
    out["Model version"] = r["model_version"].values
    return out.reset_index(drop=True)


def score_grid(rows: pd.DataFrame) -> pd.DataFrame:
    """Pro correct-score probabilities as a home-goals x away-goals grid (rows = home goals)."""
    cs = rows[rows["market"] == "correct_score"]
    if cs.empty:
        return pd.DataFrame()
    cs = cs[cs["model_version"] == cs["model_version"].iloc[-1]]
    hg_ag = cs["selection"].str.extract(r"^(\d+)-(\d+)$")
    g = pd.DataFrame({"Home goals": pd.to_numeric(hg_ag[0]), "Away goals": pd.to_numeric(hg_ag[1]),
                      "p": cs["probability"].astype(float).values}).dropna()
    grid = g.pivot_table(index="Home goals", columns="Away goals", values="p", aggfunc="first").sort_index()
    grid.columns = [f"Away {int(c)}" for c in grid.columns]
    return grid.reset_index()


def summary_rollup(summary: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Hit rate / log loss / Brier re-derived from the per-league sums (never an average of averages)."""
    if summary.empty:
        return summary
    g = summary.groupby(by, sort=True, dropna=False).agg(
        predictions=("predictions", "sum"), settled=("settled", "sum"), void=("void", "sum"),
        pending=("pending", "sum"), hits=("hits", "sum"), log_loss_sum=("log_loss_sum", "sum"),
        brier_sum=("brier_sum", "sum"), pick_prob_sum=("pick_prob_sum", "sum"),
        single_settled=("single_settled", "sum"), single_p_sum=("single_p_sum", "sum"),
        single_y_sum=("single_y_sum", "sum")).reset_index()
    n = g["settled"].where(g["settled"] > 0)
    g["hit_rate"] = g["hits"] / n
    g["log_loss"] = g["log_loss_sum"] / n
    g["brier"] = g["brier_sum"] / n
    g["mean_pick_prob"] = g["pick_prob_sum"] / n
    return with_calibration(g)


def with_calibration(s: pd.DataFrame) -> pd.DataFrame:
    """Single-selection questions: mean predicted probability vs the rate the selection happened."""
    s = s.copy()
    n1 = s["single_settled"].where(s["single_settled"] > 0)
    s["single_mean_p"] = s["single_p_sum"] / n1
    s["single_rate"] = s["single_y_sum"] / n1
    return s


def summary_display(s: pd.DataFrame, *, with_league: bool) -> pd.DataFrame:
    if s.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Model": s["model_id"].map(model_label), "Market": s["market"].map(market_label)})
    if with_league:
        out["League"] = [f"{n} ({LEDGER_ONLY_LABEL})" if lo else str(n)
                         for n, lo in zip(s["league"], s["ledger_only"])]
    out["Predictions"] = s["predictions"].astype(int).values
    out["Settled"] = s["settled"].astype(int).values
    out["Void"] = s["void"].astype(int).values
    out["Pending"] = s["pending"].astype(int).values
    out["Hit rate"] = s["hit_rate"].values
    out["Avg pick probability"] = s["mean_pick_prob"].values
    out["Log loss"] = s["log_loss"].values
    out["Brier"] = s["brier"].values
    if "single_mean_p" not in s:
        s = with_calibration(s)
    out["Avg probability (YES-type)"] = s["single_mean_p"].values
    out["Happened (YES-type)"] = s["single_rate"].values
    return out.reset_index(drop=True)
