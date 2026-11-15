"""Rare props research picks (owner 2026-11-09): goals2 / goals3 / sot3 / sot4, RESEARCH only, at most
3 per gameweek across the four markets, each market only with its own study bar (no typed or global
bar), ranked by ev_lower_bound on a real two-sided price. Synthetic data only."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config as pconfig  # noqa: E402
from player_model import rare_props as RP  # noqa: E402
from player_model.fair_basis import ASSUMED_OVERROUND, TWO_SIDED_DEVIG  # noqa: E402

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def _bars(tmp_path, markets: dict) -> Path:
    p = tmp_path / "rare_props_bars.json"
    p.write_text(json.dumps({"markets": markets}), encoding="utf-8")
    return p


def _cand(i, market="sot3", p=0.30, odds=8.0, fair=0.10, basis=TWO_SIDED_DEVIG, day="2026-11-14"):
    return {"date": day, "kickoff_utc": f"{day}T15:00:00Z", "league": "Championship",
            "match": "A vs B", "fixture_id": 1000 + i, "player_id": 10 + i, "player_name": f"P{i}",
            "team": "A", "position": "F", "market": market, "model_prob": p, "market_odds": odds,
            "fair_implied": fair, "fair_basis": basis}


BAR = {"ev_lower_bound_min": 0.5, "p_lower_margin": 0.05, "source": "bars study run X"}


def test_rare_markets_are_the_hidden_ones_and_the_cap_is_three():
    assert set(RP.RARE_MARKETS) == {"goals2", "goals3", "sot3", "sot4"} == set(pconfig.HIDDEN_PENDING_CLEAN_RETRAIN)
    assert RP.MAX_PICKS_PER_GAMEWEEK == 3


def test_no_bar_file_means_no_pick_for_any_market(tmp_path):
    cands = pd.DataFrame([_cand(i, m) for i, m in enumerate(RP.RARE_MARKETS)])
    picks, counts = RP.evaluate(cands, RP.load_bars(tmp_path / "absent.json"), now=NOW)
    assert picks.empty and counts["NO_BAR_YET"] == 4 and counts["SELECTED"] == 0


def test_the_committed_repo_has_no_bar_yet():
    assert RP.load_bars() == {} or set(RP.load_bars()) <= set(RP.RARE_MARKETS)
    if not RP.BARS_PATH.exists():
        assert RP.load_bars() == {}


@pytest.mark.parametrize("entry", [{"ev_lower_bound_min": 0.5, "p_lower_margin": 0.05},          # no source
                                   {"ev_lower_bound_min": 0.5, "source": "s"},                    # no margin
                                   {"p_lower_margin": 0.05, "source": "s"},                       # no bar
                                   {"ev_lower_bound_min": "x", "p_lower_margin": 0.05, "source": "s"},
                                   {"ev_lower_bound_min": 0.5, "p_lower_margin": -0.1, "source": "s"}])
def test_an_incomplete_bar_is_no_bar(tmp_path, entry):
    assert RP.load_bars(_bars(tmp_path, {"sot3": entry})) == {}


def test_a_bar_is_per_market_never_shared(tmp_path):
    bars = RP.load_bars(_bars(tmp_path, {"sot3": BAR}))
    cands = pd.DataFrame([_cand(0, "sot3"), _cand(1, "sot4"), _cand(2, "goals2")])
    picks, counts = RP.evaluate(cands, bars, now=NOW)
    assert list(picks["market"]) == ["sot3"]
    assert counts["NO_BAR_YET"] == 2


def test_price_rules_no_price_and_no_assumed_margin(tmp_path):
    bars = RP.load_bars(_bars(tmp_path, {"sot3": BAR}))
    cands = pd.DataFrame([_cand(0, odds=None, fair=None, basis=None),
                          _cand(1, basis=ASSUMED_OVERROUND),
                          _cand(2, fair=None, basis=None),
                          _cand(3)])
    picks, counts = RP.evaluate(cands, bars, now=NOW)
    assert counts["NO_PRICE"] == 1 and counts["NO_TWO_SIDED_PRICE"] == 2
    assert list(picks["player_id"]) == [13]
    r = picks.iloc[0]
    assert r["fair_basis"] == TWO_SIDED_DEVIG and r["fair_prob_two_sided"] == pytest.approx(0.10)
    assert r["p_lower_bound"] == pytest.approx(0.25) and r["ev_lower_bound"] == pytest.approx(0.25 * 8 - 1)
    assert r["edge_lb_vs_fair"] == pytest.approx(0.15)
    assert r["state"] == "RESEARCH" and bool(r["paper_only"]) and r["bar_source"] == "bars study run X"


def test_below_the_bar_is_not_picked(tmp_path):
    bars = RP.load_bars(_bars(tmp_path, {"sot3": {**BAR, "ev_lower_bound_min": 5.0}}))
    picks, counts = RP.evaluate(pd.DataFrame([_cand(0)]), bars, now=NOW)
    assert picks.empty and counts["BELOW_BAR"] == 1


def test_at_most_three_per_gameweek_across_markets_ranked_by_ev_lower_bound(tmp_path):
    bars = RP.load_bars(_bars(tmp_path, {m: BAR for m in RP.RARE_MARKETS}))
    cands = pd.DataFrame([_cand(0, "sot3", p=0.30), _cand(1, "sot4", p=0.40), _cand(2, "goals2", p=0.25),
                          _cand(3, "goals3", p=0.35), _cand(4, "sot3", p=0.50, day="2026-11-21")])
    picks, counts = RP.evaluate(cands, bars, now=NOW)
    wk = picks[picks["gameweek"] == "2026-W46"]
    assert list(wk["player_id"]) == [11, 13, 10]              # ev_lb order: p .40, .35, .30
    assert list(wk["rank"]) == [1, 2, 3]
    assert counts["GW_CAP_REACHED"] == 1                       # p .25 (ev_lb 0.6, above the bar) lost the last slot
    assert list(picks[picks["gameweek"] == "2026-W47"]["player_id"]) == [14]   # another week


def test_earlier_picks_count_toward_the_cap_and_are_logged_once(tmp_path):
    bars = RP.load_bars(_bars(tmp_path, {"sot3": BAR}))
    log = tmp_path / "rare.csv"
    first = pd.DataFrame([_cand(0), _cand(1)])
    c1 = RP.run(first, bars_path=_bars(tmp_path, {"sot3": BAR}), log_path=log, now=NOW)
    assert c1["appended"] == 2
    second = pd.DataFrame([_cand(0), _cand(2), _cand(3)])          # one repeat, two new
    c2 = RP.run(second, bars_path=_bars(tmp_path, {"sot3": BAR}), log_path=log, now=NOW)
    assert c2["ALREADY_LOGGED"] == 1 and c2["SELECTED"] == 1 and c2["GW_CAP_REACHED"] == 1
    out = pd.read_csv(log)
    assert len(out) == 3 and out["fixture_id"].is_unique and (out["state"] == "RESEARCH").all()


def test_attach_prices_uses_only_real_two_sided_fair():
    from player_model.odds_fetcher import tip_price_key
    c = pd.DataFrame([{k: v for k, v in _cand(0).items() if k not in ("market_odds", "fair_implied", "fair_basis")},
                      {k: v for k, v in _cand(1).items() if k not in ("market_odds", "fair_implied", "fair_basis")}])
    k0 = tip_price_key("P0", "sot3", "2026-11-14", "A vs B")
    k1 = tip_price_key("P1", "sot3", "2026-11-14", "A vs B")
    out = RP.attach_prices(c, {k0: 8.0, k1: 7.0}, {k0: 0.11})
    assert list(out["market_odds"]) == [8.0, 7.0]
    assert out["fair_basis"].iloc[0] == TWO_SIDED_DEVIG and pd.isna(out["fair_basis"].iloc[1])
    assert pd.isna(out["fair_implied"].iloc[1])


def test_predict_collects_rare_candidates_from_the_hidden_markets_without_goalkeepers():
    src = (ROOT / "player_model" / "predict.py").read_text(encoding="utf-8")
    assert "_rare_cands.append(" in src
    assert 'if _skip == config.SKIP_HIDDEN_MARKET and _pos_status != "G":' in src
    assert "_RP.write_candidates(_rare_cands)" in src
    pipe = (ROOT / "player_model" / "pipeline.py").read_text(encoding="utf-8")
    assert "_RP.run(_RP.attach_prices(" in pipe
    wf = (ROOT / ".github" / "workflows" / "player_props.yml").read_text(encoding="utf-8")
    assert "output/rare_props_research.csv" in wf
