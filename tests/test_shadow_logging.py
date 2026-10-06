"""§3/§18 — the prospective log must freeze at first sight, or it is not prospective.

predict runs every 5-15 minutes, so each fixture is scored ~100 times before kickoff. If a later
run could overwrite an earlier one, the recorded opinion would be the best-informed one while
appearing to be the first — which turns the whole forward period into a retrospective test.
"""
from __future__ import annotations

import pandas as pd
import pytest

import pipeline
from src import shadow_compare as sc


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SHADOW_FILE", tmp_path / "v91_shadow.csv")
    return sc


def _preds(p_over=0.55, side="OVER", tier="SNIPER", date="2026-10-09", n=1):
    return pd.DataFrame({
        "_oa_event_id": [f"evt{i}" for i in range(n)],
        "date": [date] * n, "league": ["Championship"] * n,
        "home_team": [f"H{i}" for i in range(n)], "away_team": [f"A{i}" for i in range(n)],
        "model_type": ["standard"] * n, "p_over25": [p_over] * n,
        "p_over25_poisson_dc": [0.52] * n, "impl_prob_over": [0.50] * n,
        "best_side": [side] * n, "signal_tier": [tier] * n, "best_edge": [0.05] * n,
        "odds_over25": [1.95] * n, "odds_under25": [1.90] * n,
        "model_sha": ["abc123"] * n, "git_sha": ["def456"] * n,
    })


def test_a_prediction_is_written(shadow):
    pipeline._log_shadow(_preds(n=3))
    assert len(pd.read_csv(shadow.SHADOW_FILE)) == 3


def test_a_later_run_cannot_revise_an_earlier_opinion(shadow):
    """THE central guarantee. A second run with a DIFFERENT probability must not overwrite."""
    pipeline._log_shadow(_preds(p_over=0.55, side="OVER", tier="VALUABLE"))
    pipeline._log_shadow(_preds(p_over=0.91, side="UNDER", tier="SNIPER"))
    d = pd.read_csv(shadow.SHADOW_FILE)
    assert len(d) == 1, "the fixture was logged twice"
    assert d.iloc[0]["p_v9"] == 0.55 and d.iloc[0]["v9_side"] == "OVER"
    assert d.iloc[0]["v9_tier"] == "VALUABLE", "the later, better-informed tier won"


def test_every_scored_fixture_is_logged_not_only_tipped_ones(shadow):
    """Tips fire on ~14% of the board. A tips-only record would take years to reach a usable n
    and would discard the fixtures that test whether the tier threshold separates anything."""
    pipeline._log_shadow(_preds(tier="AVOID", n=5))
    assert len(pd.read_csv(shadow.SHADOW_FILE)) == 5


def test_unwired_challengers_stay_EMPTY_not_copied_from_the_champion(shadow):
    """A column filled with v9's number would silently fake agreement between models."""
    pipeline._log_shadow(_preds())
    d = pd.read_csv(shadow.SHADOW_FILE)
    for c in ("p_v91", "p_pro_hgb", "v91_side", "v91_tier", "v91_edge"):
        assert pd.isna(d.iloc[0][c]), f"{c} was populated — it has no challenger wired yet"


def test_provenance_is_recorded(shadow):
    """A forward period that cannot name the artefact it tested proves nothing about it."""
    pipeline._log_shadow(_preds())
    d = pd.read_csv(shadow.SHADOW_FILE)
    assert d.iloc[0]["v9_model_sha"] == "abc123" and d.iloc[0]["dataset_id"] == "def456"


def test_entry_odds_follow_the_side_taken(shadow):
    pipeline._log_shadow(_preds(side="UNDER"))
    assert pd.read_csv(shadow.SHADOW_FILE).iloc[0]["entry_odds"] == 1.90


def test_a_fixture_without_an_event_id_still_gets_logged(shadow):
    """An unmatched fixture is still a prediction worth grading; dropping it would bias the
    forward sample toward exactly the fixtures the market priced."""
    p = _preds()
    p["_oa_event_id"] = [None]
    pipeline._log_shadow(p)
    d = pd.read_csv(shadow.SHADOW_FILE)
    assert len(d) == 1 and "H0" in str(d.iloc[0]["fixture_id"])


def test_logging_can_never_break_predict(shadow, monkeypatch):
    """A recorder must not be able to stop tips going out."""
    monkeypatch.setattr(sc, "append_shadow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    pipeline._log_shadow(_preds())          # must not raise


def test_the_prospective_window_filters_on_when_it_was_LOGGED(shadow):
    """A fixture played after registration but PREDICTED before it is still retrospective."""
    pipeline._log_shadow(_preds(date="2026-12-01"))
    d = pd.read_csv(shadow.SHADOW_FILE)
    d["result"] = "WIN"
    d.to_csv(shadow.SHADOW_FILE, index=False)
    assert len(shadow.prospective_window("2020-01-01")) == 1
    assert len(shadow.prospective_window("2030-01-01")) == 0, \
        "filtered on match_date instead of logged_at"


def test_the_workflow_commits_the_shadow_log():
    """Captured and never committed is the 2026-08-15 failure: every step green, nothing saved.
    A prospective log is worth nothing if each CI run starts from an empty file."""
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "predict.yml")
    assert "output/v91_shadow.csv" in wf.read_text(encoding="utf-8")


# ── §5/§6: the prospective gate dataset ───────────────────────────────────────────────────────
def test_the_gates_in_force_are_recorded_with_every_prediction(shadow):
    """Without them the log records what was decided but not the RULE it was decided under, so
    a later threshold change silently re-interprets every historical row."""
    pipeline._log_shadow(_preds(n=2))
    d = pd.read_csv(shadow.SHADOW_FILE)
    for c in ("production_valuable_gate", "production_marksman_gate",
              "production_sniper_gate", "threshold_regime_id"):
        assert d[c].notna().all(), f"{c} was not captured"


def test_gates_come_from_the_resolver_not_from_config(shadow):
    """config.py disagrees with production for any league carrying an approved optimiser
    value — Championship runs 0.07, config says 0.15."""
    import inspect
    assert "resolve_effective_tier_gates" in inspect.getsource(pipeline._log_shadow)


def test_market_is_part_of_the_dedup_key(shadow):
    """Without it the first market logged for a fixture blocks every other market on the same
    fixture, so BTTS and Over 1.5 would silently never be recorded."""
    p = _preds()
    pipeline._log_shadow(p)
    d = pd.read_csv(shadow.SHADOW_FILE)
    d2 = d.copy(); d2["market"] = "btts"
    shadow.append_shadow(d2)
    out = pd.read_csv(shadow.SHADOW_FILE)
    assert set(out["market"]) == {"ou25", "btts"} and len(out) == 2


def test_grading_fields_are_never_populated_at_prediction_time(shadow):
    """§6 is absolute: a future value must not be reachable by a selection rule, and the only
    way to guarantee that is for the prediction-time writer never to set one."""
    pipeline._log_shadow(_preds(n=3))
    d = pd.read_csv(shadow.SHADOW_FILE)
    for c in shadow.GRADING_ONLY:
        assert d[c].isna().all(), f"{c} was populated before the match was played"


def test_every_scored_fixture_is_captured_including_below_the_gate(shadow):
    """THE WHOLE POINT. The ledger holds only fixtures that cleared the live floor, so it can
    never answer 'what if the bar were lower'. This log must hold the ones that failed it."""
    rows = pd.concat([_preds(p_over=0.55, tier="AVOID", n=4),
                      _preds(p_over=0.55, tier="SNIPER", n=2)], ignore_index=True)
    rows["_oa_event_id"] = [f"e{i}" for i in range(len(rows))]
    rows["home_team"] = [f"H{i}" for i in range(len(rows))]
    pipeline._log_shadow(rows)
    d = pd.read_csv(shadow.SHADOW_FILE)
    assert (d["v9_tier"] == "AVOID").sum() == 4, "below-gate fixtures were dropped"


def test_an_old_row_without_a_market_column_still_dedups(shadow):
    """Rows written before `market` existed are main O/U by construction. If the dedup could
    not read them, every one would be re-logged and the first-sight freeze would break."""
    pipeline._log_shadow(_preds())
    d = pd.read_csv(shadow.SHADOW_FILE).drop(columns=["market"])
    d.to_csv(shadow.SHADOW_FILE, index=False)
    pipeline._log_shadow(_preds(p_over=0.91))
    assert len(pd.read_csv(shadow.SHADOW_FILE)) == 1
