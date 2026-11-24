"""Predictions page helpers (owner 2026-11-22): every model's prediction for every market of a match, its result
once settled, and the learning-record summary since the Clean Record start.

Reads the TWO compact files scripts/build_predictions_index.py writes from the v9 and Pro prediction ledgers
(update_results.yml refreshes them after settlement): reports/dashboard/predictions_index.parquet and
reports/dashboard/predictions_summary.csv. No price is in either file. v11 is private and never in them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model_releases import Releases

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
    "pro_bb_combo": "Pro combos (no market price: hit rate only)",
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
    return with_identity(d)


def with_identity(d: pd.DataFrame) -> pd.DataFrame:
    """Canonical fixture identity columns (src.fixture_identity via the index builder). An index written before
    they existed falls back to each row's own ref / kickoff / names, i.e. the old behaviour, never a guess."""
    d = d.copy()
    if "canonical_fixture_ref" not in d.columns:
        d["canonical_fixture_ref"] = d["fixture_ref"]
        d["canonical_kickoff_utc"] = d["kickoff_utc"]
        d["kickoff_source"] = ""
        d["canonical_home"], d["canonical_away"] = d["home_team"], d["away_team"]
    d["canonical_kickoff_utc"] = pd.to_datetime(d["canonical_kickoff_utc"], utc=True, errors="coerce")
    for c in ("canonical_fixture_ref", "canonical_home", "canonical_away", "kickoff_source"):
        d[c] = d[c].fillna("").astype(str).replace("nan", "")
    # display kickoff: the authoritative (API-Football) / unanimous one; a source conflict shows the row's own
    d["match_kickoff_utc"] = d["canonical_kickoff_utc"].where(d["canonical_kickoff_utc"].notna(), d["kickoff_utc"])
    return d


def serving_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """ONE version per question for display (owner 2026-11-23 "why duplicates?"): per model x market x line x player
    x snapshot x match, the version that was serving at that snapshot = the latest predicted_at strictly before the
    kickoff (else the latest). Display only - the ledger keeps every version's rows."""
    if rows.empty:
        return rows
    d = rows.assign(_before=(rows["predicted_at"] < rows["kickoff_utc"]).astype(int))
    q = ["model_id", "market", "line", "player_id", "snapshot", "canonical_fixture_ref"]
    best = (d.sort_values(["_before", "predicted_at", "model_version"], kind="mergesort")
            .drop_duplicates(q, keep="last")[q + ["model_version"]])
    return d.merge(best, on=q + ["model_version"], how="inner").drop(columns="_before")


def load_summary(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or SUMMARY, dtype={"line": str})


def league_options(idx: pd.DataFrame) -> dict[str, str]:
    """league_slug -> display name; ledger-only leagues say so."""
    lo = idx.groupby("league_slug")["ledger_only"].any()
    names = idx.groupby("league_slug")["league"].first()
    return {s: f"{names[s]} ({LEDGER_ONLY_LABEL})" if lo.get(s, False) else str(names[s])
            for s in sorted(names.index, key=lambda s: str(names[s]))}


def matches(idx: pd.DataFrame) -> pd.DataFrame:
    """One row per match in `idx`: its CANONICAL fixture identity (src.fixture_identity - one match is one identity
    whatever ref, club spelling or kickoff minute each model's source carries; owner bug 2026-11-23)."""
    cols = ["match_key", "kickoff_utc", "league_slug", "league", "home_team", "away_team", "models", "predictions",
            "settled", "ledger_only", "kickoff_source", "refs"]
    if idx.empty:
        return pd.DataFrame(columns=cols)
    if "canonical_fixture_ref" not in idx.columns:
        idx = with_identity(idx)
    d = idx.assign(match_key=idx["canonical_fixture_ref"])
    g = d.groupby("match_key", sort=False)
    out = g.agg(kickoff_utc=("match_kickoff_utc", "first"), league_slug=("league_slug", "first"),
                league=("league", "first"), home_team=("canonical_home", "first"),
                away_team=("canonical_away", "first"), models=("model_id", "nunique"),
                predictions=("probability", "size"),
                settled=("outcome_status", lambda s: int((s == "SETTLED").sum())),
                ledger_only=("ledger_only", "any"), kickoff_source=("kickoff_source", "first"),
                refs=("fixture_ref", "nunique")).reset_index()
    return out[cols].sort_values(["kickoff_utc", "league", "home_team"], kind="mergesort").reset_index(drop=True)


def match_rows(idx: pd.DataFrame, match_key: str) -> pd.DataFrame:
    if "canonical_fixture_ref" not in idx.columns:
        idx = with_identity(idx)
    return idx[idx["canonical_fixture_ref"] == match_key]


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


def pick_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The model's pick of each question only (owner 2026-11-23: one row per market and line - the complementary
    NO / UNDER row is 1 - p and adds nothing). Display only."""
    return rows[rows["pick"].astype(bool)] if len(rows) else rows


def pick_side(row) -> tuple[str, float, object]:
    """(picked side, its probability, did it happen 1/0/None) of a pick row. A single-selection question whose
    probability is below 0.5 picks 'not <selection>' at 1 - p."""
    p, y = float(row["probability"]), row["y"]
    if int(row["q_size"]) > 1 or p >= 0.5:
        return str(row["selection"]), p, (None if pd.isna(y) else int(y))
    return f"not {row['selection']}", 1 - p, (None if pd.isna(y) else 1 - int(y))


_RELEASES: list = []


def releases() -> Releases:
    """The release registry (config/model_release_registry.json), read once per process."""
    if not _RELEASES:
        _RELEASES.append(Releases())
    return _RELEASES[0]


def version_names(rows: pd.DataFrame) -> list[str]:
    """'<release> · <install date> · <status at kickoff>' per row; unknown -> 'unknown release · <raw id>'."""
    R = releases()
    ko = rows["match_kickoff_utc"] if "match_kickoff_utc" in rows.columns else rows["kickoff_utc"]
    return [R.display(m, k, v, t) for m, k, v, t in zip(rows["model_id"], rows["market"], rows["model_version"], ko)]


def pick_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Default view: ONE row per question - the serving version's pick, its probability, outcome and result."""
    if rows.empty:
        return pd.DataFrame()
    r = rows.sort_values(["model_id", "market", "line", "player_name"], kind="mergesort")
    sides = [pick_side(x) for _, x in r.iterrows()]
    out = pd.DataFrame({"Model": r["model_id"].map(model_label).values,
                        "Market": r["market"].map(market_label).values,
                        "Line": r["line"].astype(str).replace("nan", "").values})
    if (r["player_name"] != "").any():
        out["Player"] = r["player_name"].values
        out["Team"] = r["team"].astype(str).replace("nan", "").values
    out["Pick"] = [s for s, _, _ in sides]
    out["Probability"] = [p for _, p, _ in sides]
    out["Outcome"] = [("happened" if h == 1 else "did not happen") if st == "SETTLED" and h is not None
                      else outcome_text(st, None) for (_, _, h), st in zip(sides, r["outcome_status"])]
    out["Result"] = [result_text(x) for _, x in r.iterrows()]
    out["Model version"] = version_names(r)
    return out.reset_index(drop=True)


def prediction_table(rows: pd.DataFrame) -> pd.DataFrame:
    """All versions / both sides: every stored row with its selection, probability, pick, outcome, the readable
    version name and the raw version id."""
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
    out["Model version"] = version_names(r)
    out["Version id"] = r["model_version"].values
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
