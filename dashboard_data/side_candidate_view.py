"""Admin Board - "Side-model candidate" (owner 2026-11-24: monthly corners / cards candidate refit).

The serving side model comes from config/track_registry.json (`side_models` block). The latest monthly candidate is
read from the branch side-models/candidate-latest (reports/side_models/candidate_summary.json and
candidate_heldout_summary.csv), written by .github/workflows/train_side_models.yml (mode monthly_candidate). A candidate
is NEVER installed by the workflow: the owner installs it by merging its "awaiting owner approval" PR.
Pure functions here; the page fetches and renders. No prices exist in either file.
"""
from __future__ import annotations

import io
import json

import pandas as pd

CANDIDATE_REF = "side-models/candidate-latest"
SUMMARY_PATH = "reports/side_models/candidate_summary.json"
HELDOUT_PATH = "reports/side_models/candidate_heldout_summary.csv"


def serving_lines(registry: dict | None) -> list[tuple[str, str]]:
    s = (registry or {}).get("side_models") or {}
    if not s:
        return [("Serving", "unknown (no side_models block in config/track_registry.json)")]
    return [("Serving", str(s.get("model_version"))), ("Data end", str(s.get("data_end"))),
            ("Lineage", str((s.get("lineage") or {}).get("status"))), ("Refit", str(s.get("scheduled_refit")))]


def candidate_lines(summary: dict | None) -> list[tuple[str, str]]:
    if not summary:
        return []
    h = summary.get("heldout") or {}
    return [("Candidate", str(summary.get("candidate_model_version"))),
            ("Data end", str(summary.get("candidate_data_end"))),
            ("Held-out", f"{h.get('start')}..{h.get('end')}"),
            ("Log loss better / worse", f"{summary.get('candidate_better_logloss')} / "
                                        f"{summary.get('candidate_worse_logloss')} of {summary.get('cells')}"),
            ("Brier better / worse", f"{summary.get('candidate_better_brier')} / "
                                     f"{summary.get('candidate_worse_brier')}"),
            ("Lineage", str((summary.get("lineage") or {}).get("status"))),
            ("Install", "awaiting owner approval (merge the PR)")]


def parse_summary(body: str | None) -> dict | None:
    try:
        return json.loads(body) if body else None
    except ValueError:
        return None


def heldout_frame(body: str | None) -> pd.DataFrame:
    if not body:
        return pd.DataFrame()
    d = pd.read_csv(io.StringIO(body))
    keep = ["family", "league", "market", "n_games", "logloss_serving", "logloss_candidate", "logloss_delta",
            "brier_serving", "brier_candidate", "brier_delta"]
    return d[[c for c in keep if c in d.columns]]
