"""New-format OU25 price cap (owner 2026-10-22): no tip above NF_MAX_OU_ODDS on new_format; standard untouched."""
import pandas as pd

import config
from src.betting import evaluate_value


def _row(league, p_over, odds_over, odds_under):
    return {"league": league, "date": "2026-10-25", "home_team": "A", "away_team": "B", "p_over25": p_over,
            "odds_over25": odds_over, "odds_under25": odds_under,
            "home_scored_last5": 7, "away_scored_last5": 6, "home_conceded_last5": 5, "away_conceded_last5": 6}


def _nf_and_std():
    nf = next(l for l in sorted(config.NEW_FORMAT_LEAGUES) if l in config.ENABLED_LEAGUES)
    std = next(l for l in sorted(config.STANDARD_FORMAT_LEAGUES) if l in config.ENABLED_LEAGUES)
    return nf, std


def test_new_format_over_priced_above_cap_is_not_a_tip_but_is_still_scored():
    nf, std = _nf_and_std()
    df = evaluate_value(pd.DataFrame([_row(nf, 0.45, 3.40, 1.33), _row(std, 0.45, 3.40, 1.33)]))
    a, b = df.iloc[0], df.iloc[1]
    assert a["edge_over"] > config.VALUE_THRESHOLD                       # still scored and recorded
    assert a["bet"] == "AVOID" and a["signal_tier"] == "AVOID" and bool(a["max_odds_blocked"])
    assert b["bet"] == "OVER" and not bool(b["max_odds_blocked"])        # standard track untouched


def test_new_format_below_cap_still_tips():
    nf, _ = _nf_and_std()
    df = evaluate_value(pd.DataFrame([_row(nf, 0.60, 2.40, 1.62)]))
    assert df.iloc[0]["bet"] == "OVER" and not bool(df.iloc[0]["max_odds_blocked"])


def test_cap_can_be_disabled(monkeypatch):
    nf, _ = _nf_and_std()
    monkeypatch.setattr(config, "NF_MAX_OU_ODDS", 0.0)
    df = evaluate_value(pd.DataFrame([_row(nf, 0.45, 3.40, 1.33)]))
    assert df.iloc[0]["bet"] == "OVER"
