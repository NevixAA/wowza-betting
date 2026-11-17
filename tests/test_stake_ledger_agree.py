"""Go-live rehearsal defect 5 (2026-11-15): the stake plan staked 7 Argentina BTTS fixtures while the side
ledger accepted 3 - the other 4 were deduplicated against pre-go-live rows of another model, so the stake
audit held stakes the Clean Record has no ledger row for. append_side_market_tips now returns the rows the
ledger holds as THESE tips, and the pipeline sizes only those."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import ledger as L  # noqa: E402
from src import stake_plan as sp  # noqa: E402

ARG = "Argentina Primera Division"
NEW_MODEL = "btts_20261109T211852Z_d7cd2762"
OLD_MODEL = "btts_20261005T101905Z_238e3419"


def _row(home, tier="QUALIFIED"):
    return {"date": "2026-11-22", "league": ARG, "home_team": home, "away_team": "Away", "market": "btts",
            "side": "YES", "odds": 2.05, "market_odds": 2.05, "edge": 0.05, "ev": 0.1, "model_prob": 0.55,
            "fair_prob": 0.5, "signal_tier": tier, "model_type": "new_format"}


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(L, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    stamp = {"model_version": NEW_MODEL, "retrain_batch_id": "", "training_cutoff": "", "prediction_role": "CHAMPION"}
    monkeypatch.setattr(L, "_version_fields", lambda *a, **k: dict(stamp))
    return tmp_path, stamp


def _seed(tmp_path, homes_models):
    rows = []
    for home, mv in homes_models:
        r = {c: "" for c in L.SIDE_LEDGER_COLS}
        r.update({"source": "live", "generated_at": "2026-11-03 15:06", "match_date": "2026-11-22", "league": ARG,
                  "home_team": home, "away_team": "Away", "market": "btts", "side": "YES", "signal_tier": "SNIPER",
                  "model_type": "new_format", "model_version": mv})
        rows.append(r)
    pd.DataFrame(rows, columns=L.SIDE_LEDGER_COLS).to_csv(tmp_path / "side_bets_ledger.csv", index=False)


def test_ledger_returns_only_the_rows_it_holds_as_these_tips(out):
    tmp_path, _ = out
    _seed(tmp_path, [("Old", OLD_MODEL), ("Same", NEW_MODEL)])
    side = pd.DataFrame([_row("Old"), _row("Same"), _row("New"), _row("New")])     # last = repeat in batch
    acc = L.append_side_market_tips(side)
    assert sorted(acc["home_team"]) == ["New", "Same"]
    led = pd.read_csv(tmp_path / "side_bets_ledger.csv")
    assert sorted(led["home_team"]) == ["New", "Old", "Same"]                       # ledger unchanged otherwise


def test_stake_plan_sizes_only_ledgered_side_tips(out, monkeypatch):
    tmp_path, _ = out
    import src.betting as betting
    monkeypatch.setattr(betting, "league_threshold_evidence", lambda league, market="ou25": True)
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    _seed(tmp_path, [("Old", OLD_MODEL)])
    acc = L.append_side_market_tips(pd.DataFrame([_row("Old"), _row("New")]))
    plan = sp.write(None, acc)
    assert list(plan["fixture"].str.split("|").str[1]) == ["New"]
    audit = pd.read_csv(tmp_path / "audit.csv")
    assert not audit["fixture"].str.contains("Old").any()


def test_pipeline_feeds_the_stake_plan_the_ledgered_side_rows():
    src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert "_side_ledgered = append_side_market_tips(side_bets)" in src
    assert "_write_stake_plan(_drop_flipped(bets), _side_ledgered" in src
