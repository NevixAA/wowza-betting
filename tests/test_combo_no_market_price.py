"""Pro combos: no market price, hit rate only (owner 2026-11-24, "so fix it!").

Neither provider (API-Football /odds, OddsAPI) sells a bet-builder / same-game-multi price for Pro's combos. No combo
is ever shown with a fake or derived price (a fair price = 1 / our probability, a naive multiplied price, or a P/L
at either); every combo surface carries the label."""
from __future__ import annotations

import pandas as pd

LABEL = "no market price: hit rate only"


def _sel():
    return pd.DataFrame([{"match_date": "2026-10-18", "league": "Brazil Serie A", "match": "Santos vs Fluminense",
                          "legs": "Over 3.5 goals (29%) · Hulk to score (52%)", "n_legs": 2,
                          "joint_probability": 0.0569, "fair_odds": 17.57, "independence_fair_odds": 41.25,
                          "independence_edge": 3.99, "combo_id": "x"}])


def test_telegram_album_draws_no_combo_price(monkeypatch):
    from telegram_bot import daily_pack as dp
    seen = {}
    monkeypatch.setattr(dp, "_pro_bet_builder", _sel)
    monkeypatch.setattr(dp, "render_table", lambda title, cols, rows, **k: seen.update(rows=rows, **k) or [b"x"])
    assert dp.bet_builder_image("2026-10-18") == [b"x"]
    text = repr(seen["rows"]) + seen["subtitle"]
    assert "17.57" not in text and "41.25" not in text and "fair odds" not in text.lower()
    assert LABEL in seen["subtitle"].lower()
    assert "5.7%" in text                                            # the joint probability is still shown


def test_builder_view_never_shows_a_derived_price():
    from dashboard_data import builder_view as V
    assert V.NO_MARKET_PRICE == LABEL
    for cfg in (V.SETTLED_CONFIG, V.CANDIDATE_CONFIG):
        assert not set(cfg) & set(V.DERIVED_PRICE_COLS)
    assert list(V.shown(_sel(), ("match", "joint_probability", "fair_odds", "independence_fair_odds")).columns) == \
        ["match", "joint_probability"]
    assert "pnl" not in V.settled_summary(pd.DataFrame({"combo_result": ["WON", "LOST"], "fair_odds": [3.0, 2.0],
                                                        "joint_probability": [0.3, 0.4]}))


def test_predictions_page_labels_pro_combos():
    from dashboard_data import predictions_view as P
    assert LABEL in P.model_label("pro_bb_combo")
