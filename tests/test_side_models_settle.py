"""Side models: single certified source, settlement from verified statistics, the capped side-stats
refresh (owner decisions 2026-11-18). Synthetic data; no network; writes only under tmp_path."""
from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import prediction_ledger as PL
from src.side_models import data as D
from src.side_models import ledger as L
from src.side_models import train as T
from tests.test_side_models import NOW, _art, _champ_table, _j, _targets

ROOT = Path(__file__).resolve().parents[1]


# ── single certified source ───────────────────────────────────────────────────────────────────────

def _af(**kw):
    base = {"fixture_id": 1, "af_hc": 5.0, "af_ac": 4.0, "af_hy": 2.0, "af_ay": 1.0, "af_hr": 0.0, "af_ar": 0.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_single_source_uses_only_sane_complete_counts():
    v = D.verify_single(_af())
    assert v["corners_verified"].iloc[0] and v["cards_verified"].iloc[0] and v["home_cards"].iloc[0] == 2
    for bad in (dict(af_hc=np.nan), dict(af_hc=-1.0), dict(af_hc=20.0, af_ac=11.0),       # missing / <0 / >30
                dict(af_hr=6.0), dict(af_hy=17.0), dict(af_hy=12.0, af_ay=11.0),          # red>5 / yellow>16 / >22
                dict(af_hc=0.0, af_ac=0.0, af_hy=0.0, af_ay=0.0)):                       # all zero
        w = D.verify_single(_af(**bad))
        assert not w["corners_verified"].iloc[0] and not w["cards_verified"].iloc[0], bad
        assert np.isnan(w["home_corners"].iloc[0])                                        # nothing filled


def test_two_source_rows_also_need_sane_af_counts():
    assert not D.verify(_j(af_hc=31.0, fd_hc=31.0, af_ac=0.0, fd_ac=0.0))["corners_verified"].iloc[0]


def test_artifact_marks_the_basis_and_rows_carry_it():
    art = L.load_artifact()
    tracked = {e["name"]: e for e in T.tracked_leagues()}
    for fam, cov in art["coverage"].items():
        for lg, c in cov.items():
            assert c["basis"] == ("two_source" if T.has_fd_stats(tracked[lg]) else "single_source")
    assert L.payload_version(art, "corners", "USA MLS") == "side/1.1;basis=single_source"
    assert L.payload_version(art, "corners", "Championship") == "side/1.1;basis=two_source"


# ── settlement ────────────────────────────────────────────────────────────────────────────────────

def test_stat_outcome_rules():
    tot = {"rule": "total_stat_over", "stat": "corners"}
    team = {"rule": "team_stat_over", "stat": "cards"}
    assert PL.match_outcome(tot, "OVER", "9.5", 1, 0, hs=6, as_=4) == (1, "corners 6-4")
    assert PL.match_outcome(tot, "UNDER", "9.5", 1, 0, hs=6, as_=4)[0] == 0
    assert PL.match_outcome(tot, "OVER", "10.5", 1, 0, hs=6, as_=4)[0] == 0
    assert PL.match_outcome(team, "HOME_OVER", "1.5", 1, 0, hs=2, as_=1)[0] == 1
    assert PL.match_outcome(team, "AWAY_OVER", "1.5", 1, 0, hs=2, as_=1)[0] == 0
    assert PL.match_outcome(team, "AWAY_UNDER", "1.5", 1, 0, hs=2, as_=1)[0] == 1
    assert PL.match_outcome(tot, "OVER", "9.5", 1, 0) == (None, "no corners statistics")   # never the score
    assert PL.match_outcome(team, "OVER", "1.5", 1, 0, hs=2, as_=1)[0] is None             # bad selection
    assert PL.match_outcome({"rule": "total_goals_over"}, "OVER", 2.5, 2, 1)[0] == 1       # goals unchanged


def _settle_table():
    return pd.DataFrame([
        {"fixture_id": 10, "date": "2025-01-10", "ft_home": 2, "ft_away": 1, "basis": "two_source",
         "stats_source": "fullstats_fill", "corners_verified": True, "corners_both": True, "home_corners": 7.0,
         "away_corners": 4.0, "cards_verified": False, "cards_both": True, "home_cards": np.nan,
         "away_cards": np.nan},
        {"fixture_id": 11, "date": "2025-01-10", "ft_home": 0, "ft_away": 0, "basis": "single_source",
         "stats_source": "side_stats", "corners_verified": False, "corners_both": False,
         "home_corners": np.nan, "away_corners": np.nan, "cards_verified": False, "cards_both": False,
         "home_cards": np.nan, "away_cards": np.nan}])


def _lrow(**kw):
    r = {"repo": "v9", "model_id": "v9_corners", "market": "corners_total", "selection": "OVER", "line": "9.5",
         "fixture_id": "10", "kickoff_utc": "2025-01-10T15:00:00Z"}
    r.update(kw)
    return r


def test_outcome_fn_settles_only_on_verified_statistics():
    fn = L.outcome_fn(_settle_table(), now=datetime(2025, 1, 11, 12, 0, tzinfo=timezone.utc))
    o = fn(_lrow())
    assert o["outcome_status"] == "SETTLED" and o["y"] == 1 and o["result_source"] == "two_source:fullstats_fill"
    assert fn(_lrow(market="corners_team", selection="AWAY_OVER", line="4.5"))["y"] == 0
    o = fn(_lrow(model_id="v9_cards", market="cards_total", line="3.5"))      # sources disagree
    assert o["outcome_status"] == "NO_RESULT" and "not verified" in o["outcome_value"]
    assert fn(_lrow(fixture_id="11"))["outcome_status"] == "PENDING"            # no statistics yet
    late = L.outcome_fn(_settle_table(), now=datetime(2025, 1, 14, 16, 0, tzinfo=timezone.utc))
    assert late(_lrow(fixture_id="11"))["outcome_status"] == "NO_RESULT"        # ... after 72 h
    assert fn(_lrow(kickoff_utc="2025-01-12T15:00:00Z"))["outcome_status"] == "PENDING"
    assert fn(_lrow(fixture_id="99"))["outcome_status"] == "PENDING"
    assert fn(_lrow(kickoff_utc="2025-01-07T15:00:00Z"))["outcome_status"] == "VOID_RESCHEDULED"


def test_settled_view_of_side_partitions_is_one_to_one(tmp_path):
    cfg = T.load_config()
    art = _art()
    rows = L.rows_for(L.score(_champ_table(), _targets(), art, cfg), art, cfg, now=NOW, slugs=PL.LeagueSlugs())
    PL.append(rows, repo="v9", writer="side_models", root=tmp_path / "raw", now=NOW)
    t = pd.DataFrame([{"fixture_id": 555, "date": "2025-01-15", "ft_home": 1, "ft_away": 1, "basis": "two_source",
                       "stats_source": "fullstats_fill", "corners_verified": True, "corners_both": True,
                       "home_corners": 6.0, "away_corners": 5.0, "cards_verified": True, "cards_both": True,
                       "home_cards": 3.0, "away_cards": 1.0}])
    fn = L.outcome_fn(t, now=datetime(2025, 1, 16, tzinfo=timezone.utc))
    st = PL.write_settled(fn, repo="v9", root=tmp_path / "raw", settled_root=tmp_path / "settled",
                          models=("v9_corners", "v9_cards"))
    assert st["rows"] == len(rows) and st["settled"] == len(rows)
    s = pd.concat([pd.read_csv(p, dtype=str) for p in (tmp_path / "settled").rglob("*.csv")])
    tot = s[(s["market"] == "corners_total") & (s["line"] == "10.5") & (s["selection"] == "OVER")]
    assert tot["y"].tolist() == ["1"]                                       # 6 + 5 = 11 > 10.5


# ── capped side-stats refresh ─────────────────────────────────────────────────────────────────────

def _collector():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("collect_side_stats")


def test_collector_targets_skip_answered_and_wait_before_reasking():
    C = _collector()
    fx = pd.DataFrame([{"fixture_id": i, "league": "Championship", "status": "FT", "date": f"2026-11-0{i}",
                        "home_id": 1, "away_id": 2} for i in range(1, 6)])
    fill = pd.DataFrame({"fixture_id": [1]})
    side = pd.DataFrame([
        {"fixture_id": 2, "answered": True, "attempts": 1, "fetched_at": "2026-11-08T00:00:00Z"},
        {"fixture_id": 3, "answered": False, "attempts": 1, "fetched_at": "2026-11-09T00:00:00Z"},
        {"fixture_id": 4, "answered": False, "attempts": C.MAX_ATTEMPTS, "fetched_at": "2026-11-01T00:00:00Z"}])
    t = C.targets(datetime(2026, 11, 10, tzinfo=timezone.utc), fixtures=fx, fill=fill, side=side)
    assert t["fixture_id"].tolist() == [5]                                  # 3 asked < 48 h ago
    t = C.targets(datetime(2026, 11, 13, tzinfo=timezone.utc), fixtures=fx, fill=fill, side=side)
    assert t["fixture_id"].tolist() == [5, 3]                               # newest first; 3 due again


def _blk(tid, c, y, r):
    return {"team": {"id": tid}, "statistics": [
        {"type": "Corner Kicks", "value": c}, {"type": "Yellow Cards", "value": y},
        {"type": "Red Cards", "value": r}, {"type": "Total Shots", "value": 9}]}


def test_collector_parses_by_team_id_and_refuses_an_uncapped_run():
    C = _collector()
    p = C.parse({"response": [_blk(2, 3, 1, None), _blk(1, 7, 2, 1)]}, 1, 2)   # away block first
    assert (p["home_corners"], p["away_corners"], p["home_red"], p["away_red"]) == (7, 3, 1, 0)
    assert C.parse({"response": [_blk(9, 1, 1, 1)]}, 1, 2) is None          # not both clubs: nothing placed
    for bad in (0, C.MAX_CALLS_CAP + 1):
        with pytest.raises(SystemExit):
            C.run(bad, dry_run=True)
    with pytest.raises(SystemExit):
        C.main([])                                                          # a real run needs --max-calls


def test_sidecar_answers_feed_the_stats_but_never_override_the_fill(tmp_path):
    fill = pd.DataFrame([{"fixture_id": 1, "home_id": 1, "away_id": 2, "home_corners": 5.0, "away_corners": 4.0,
                          "home_yellow": 1.0, "away_yellow": 1.0, "home_red": 0.0, "away_red": 0.0}])
    fp = tmp_path / "fill.parquet"
    fill.to_parquet(fp)
    side = pd.DataFrame([
        {**fill.iloc[0].to_dict(), "home_corners": 9.0, "answered": True, "fetched_at": "x"},      # overlap
        {"fixture_id": 2, "home_id": 3, "away_id": 4, "home_corners": 6.0, "away_corners": 2.0,
         "home_yellow": 2.0, "away_yellow": 0.0, "home_red": 0.0, "away_red": 0.0, "answered": True, "fetched_at": "x"},
        {"fixture_id": 3, "home_id": 5, "away_id": 6, "home_corners": None, "away_corners": None,
         "home_yellow": None, "away_yellow": None, "home_red": None, "away_red": None, "answered": False,
         "fetched_at": "x"}])
    sp = tmp_path / "side.parquet"
    side.to_parquet(sp)
    s = D.load_stats(fp, sp).set_index("fixture_id")
    assert s.loc[1, "af_hc"] == 5.0 and s.loc[1, "stats_source"] == "fullstats_fill"
    assert s.loc[2, "af_hc"] == 6.0 and s.loc[2, "stats_source"] == "side_stats"
    assert 3 not in s.index                                                 # an empty answer is not a count


def test_workflows_gate_the_refresh_and_the_settlement():
    y = (ROOT / ".github" / "workflows" / "daily_injuries_lineups.yml").read_text(encoding="utf-8")
    assert "vars.SIDE_STATS_REFRESH_ENABLED == 'true'" in y
    assert 'collect_side_stats.py --max-calls "$SIDE_STATS_MAX_CALLS"' in y
    u = (ROOT / ".github" / "workflows" / "update_results.yml").read_text(encoding="utf-8")
    assert "python -m src.side_models.ledger settle" in u
    assert "vars.SIDE_MODELS_LEDGER_ENABLED == 'true' && vars.PREDICTION_LEDGER_ENABLED == 'true'" in u
