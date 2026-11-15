"""Owner decisions 2026-11-05 on props and Fantasy (branch phase2/propsdecide).

1. Props tiers use the model's OWN probability. The typed team-strength / quality adjustment is
   computed and logged beside the tip (model_prob_hand_adjusted, prob_adjustment), information only.
   player_ledger.csv keeps its fixed column list (the Pro contract).
2. Both sides of a props price are collected and labelled; a REAL two-sided price (same book, same
   answer) is de-vigged for the fair probability (fair_basis "two_sided_devig"); only without one does
   the assumed overround apply (fair_basis "assumed_overround"). The label reaches Telegram, the
   props table and the dashboard props page. Props stay PAPER-only.
3. Fantasy: the convention-based figures stay, unchanged, and are labelled as such on the page.
See reports/P2_PROPS_DEFAULTS_AND_FANTASY.md section 7.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _tip(name, model_prob, hand=None, market="sot", match="X vs Y", date="2026-11-08"):
    return {"player_name": name, "market": market, "date": date, "match": match, "model_prob": model_prob,
            "model_prob_raw": model_prob, "model_prob_hand_adjusted": hand if hand is not None else model_prob,
            "prob_adjustment": (hand / model_prob) if hand is not None else 1.0,
            "confidence": 0.5, "lazy_factors": "", "ges": 0.6, "position": "F"}


def _tips(*rows):
    t = pd.DataFrame(list(rows))
    for c in ("market_odds", "fair_implied", "edge_abs", "edge_rel", "ev", "tier", "kelly_stake"):
        t[c] = pd.Series([None] * len(t), dtype=object)   # the columns run_player_predictions writes
    return t


# ── 1. the model's own probability decides ────────────────────────────────────────────────────────
def test_tip_probability_is_the_models_own_and_the_hand_adjustment_is_information_only():
    from player_model import predict as P
    src = inspect.getsource(P.run_player_predictions)
    # the hand adjustment goes to its own variable; p_model (raw) is what is capped, floored and written
    assert "_p_hand, _adj, _q_ok = hand_adjusted_prob(p_model," in src
    assert src.index("hand_adjusted_prob(") < src.index("p_model = capped_prob(p_model,")
    assert "p_model, _adj, _q_ok = hand_adjusted_prob(" not in src
    assert '"model_prob":    round(p_model, 3)' in src
    assert '"model_prob_hand_adjusted": round(_p_hand, 4)' in src
    assert '"model_prob_raw":  round(_p_raw, 4)' in src and '"prob_adjustment": round(_adj, 4)' in src


def test_tier_ev_and_kelly_ignore_the_hand_adjusted_probability():
    from player_model import predict as P
    a = _tips(_tip("A", 0.55, hand=0.20))           # hand adjustment would have cut it to 0.20
    b = _tips(_tip("A", 0.55))                      # no hand adjustment at all
    odds = {"A|sot|2026-11-08|X vs Y": 2.40}
    oa, ob = P.enrich_with_odds(a, odds), P.enrich_with_odds(b, odds)
    for c in ("tier", "ev", "edge_rel", "edge_abs", "fair_implied", "kelly_stake"):
        assert oa.loc[0, c] == ob.loc[0, c], c
    assert oa.loc[0, "ev"] == round(0.55 * 2.40 - 1, 4)          # EV on the model's own 0.55
    assert oa.loc[0, "tier"] != "AVOID"
    assert oa.loc[0, "model_prob_hand_adjusted"] == 0.20          # still logged beside the tip


def test_player_ledger_column_list_is_unchanged():
    from player_model.ledger import PLAYER_LEDGER_COLS
    assert PLAYER_LEDGER_COLS == [
        "signal_date", "match_date", "league", "home_team", "away_team", "player_id", "fixture_id",
        "player_name", "team", "position", "market", "tier", "model_prob", "market_odds", "ev",
        "edge_rel", "n_games", "is_played", "result", "pnl", "notes", "resolved_date"]


# ── 2. real two-sided price first, assumed margin only without one ───────────────────────────────
def _q(player, side, odds, book="Pinnacle", market="goals", point=None):
    return {"match_date": "2026-11-08", "league": "Premier League", "match": "X vs Y", "player": player,
            "market": market, "side": side, "point": point, "odds": odds, "bookmaker": book,
            "source": "oddsapi"}


def test_pairs_need_both_sides_from_one_book_in_one_answer():
    from player_model import odds_fetcher as O
    from src.market_edge import power_devig
    fx: dict = {}
    n = O._add_fixture_pairs(fx, "2026-11-08", "X vs Y", [
        _q("kylian mbappe", "yes", 1.77), _q("kylian mbappe", "no", 2.07),      # a real pair
        _q("erling haaland", "yes", 1.83),                                       # one side only
        _q("cole palmer", "yes", 2.96, book="A"), _q("cole palmer", "no", 1.41, book="B"),  # two books
        _q("bukayo saka", "yes", 3.0), _q("bukayo saka", "yes", 3.2), _q("bukayo saka", "no", 1.3),  # ambiguous
        _q("joao pedro", "over", 1.60, market="sot", point=0.5),
        _q("joao pedro", "under", 2.30, market="sot", point=0.5),                # an Over/Under pair
        _q("morgan rogers", "over", 1.60, market="sot", point=0.5),
        _q("morgan rogers", "under", 2.30, market="sot", point=1.5)])            # different lines
    assert n == 2
    pairs = fx[("2026-11-08", "X vs Y")]
    assert set(pairs) == {"kylian mbappe|goals", "joao pedro|sot"}
    p = pairs["kylian mbappe|goals"][0]["p_fair"]
    assert p == power_devig(1.77, 2.07)
    assert np.isclose(p + power_devig(2.07, 1.77), 1.0)            # a proper two-way split


def test_fair_probability_comes_from_the_real_pair_else_the_assumed_margin():
    from player_model import odds_fetcher as O
    from player_model import predict as P
    from src.market_edge import power_devig
    prices = {("2026-11-08", "X vs Y"): {"kylian mbappe|goals": 1.80, "erling haaland|goals": 1.83}}
    pairs: dict = {}
    O._add_fixture_pairs(pairs, "2026-11-08", "X vs Y",
                         [_q("kylian mbappe", "yes", 1.77), _q("kylian mbappe", "no", 2.07),
                          _q("kylian mbappe", "yes", 1.80, book="Bet365"),
                          _q("kylian mbappe", "no", 1.95, book="Bet365"), _q("erling haaland", "yes", 1.83)])
    tips = _tips(_tip("K. Mbappe", 0.62, market="goals"), _tip("E. Haaland", 0.62, market="goals"),
                 _tip("Nobody", 0.62, market="goals"))
    odds = O.match_odds_to_tips(tips, {}, fixture_prices=prices)
    fair = O.match_fair_to_tips(tips, fixture_pairs=pairs, fixture_prices=prices)
    k_m = O.tip_price_key("K. Mbappe", "goals", "2026-11-08", "X vs Y")
    k_h = O.tip_price_key("E. Haaland", "goals", "2026-11-08", "X vs Y")
    assert set(odds) == {k_m, k_h} and set(fair) == {k_m}           # Haaland: one side only
    want = float(np.median([power_devig(1.77, 2.07), power_devig(1.80, 1.95)]))
    assert fair[k_m] == want
    out = P.enrich_with_odds(tips, odds, fair)
    assert out.loc[0, "fair_basis"] == P.FAIR_BASIS_TWO_SIDED_DEVIG == "two_sided_devig"
    assert out.loc[0, "fair_implied"] == round(want, 4)
    assert out.loc[0, "edge_rel"] == P._relative_edge(0.62, want)
    assert out.loc[1, "fair_basis"] == P.FAIR_BASIS_ASSUMED_OVERROUND == "assumed_overround"
    assert out.loc[1, "fair_implied"] == round(P._devig_fair_prob(1.83), 4)
    assert out.loc[2, "fair_basis"] is None or pd.isna(out.loc[2, "fair_basis"])   # unpriced: no basis
    # without any pair map, every priced tip carries the assumed margin, as before
    plain = P.enrich_with_odds(_tips(_tip("K. Mbappe", 0.62, market="goals")), {k_m: 1.80})
    assert plain.loc[0, "fair_basis"] == "assumed_overround"


def test_a_bad_pair_value_falls_back_to_the_assumed_margin():
    from player_model import predict as P
    t = _tips(_tip("A", 0.4))
    k = "A|sot|2026-11-08|X vs Y"
    for bad in (None, float("nan"), 0.0, 1.0, "x"):
        out = P.enrich_with_odds(t.copy(), {k: 3.1}, {k: bad})
        assert out.loc[0, "fair_basis"] == "assumed_overround", bad


def test_api_football_no_side_is_kept_in_the_quote_log_never_as_a_tip_price(monkeypatch):
    from player_model import api_football as AF
    from player_model import prop_odds_af as PA
    assert PA.side_word("Joel Piroe") is None                 # a bare name names no side
    assert PA.side_word("Joel Piroe - No") == "no" and PA.side_word("Joel Piroe - Yes") == "yes"
    assert PA.side_word("Over 1.5") == "over" and PA.side_word("Jerry Yates - Under 1.5") == "under"
    assert PA.side_word("Noah Saunders") is None              # whole words only
    body = [{"bookmakers": [{"name": "Bet365", "bets": [{"name": "Anytime Goal Scorer", "values": [
        {"value": "Joel Piroe", "odd": "2.75"}, {"value": "Joel Piroe - No", "odd": "1.40"},
        {"value": "No", "odd": "1.10"}]}]}]}]
    monkeypatch.setattr(AF, "fixture_odds_raw", lambda fid, cache_hours=1: body)
    PA.drain_quotes()
    best, records = PA.fetch([{"fixture_id": 1, "league": "Championship", "match": "Leeds vs Derby",
                               "match_date": "2026-11-08"}], verbose=False)
    assert best == {"Joel Piroe|goals": 2.75} and len(records) == 1
    q = PA.drain_quotes()
    assert sorted((x["player"], x["side"], x["odds"]) for x in q) == [
        ("Joel Piroe", "no", 1.40), ("Joel Piroe", "yes", 2.75)]
    from player_model import odds_fetcher as O
    fx: dict = {}
    assert O._add_af_pairs(fx, q) == 1                         # and they form a real pair
    assert list(fx[("2026-11-08", "Leeds vs Derby")]) == ["joel piroe|goals"]


def test_prematch_capture_keeps_every_side_and_names_it():
    import capture_prematch as CP
    body = {"response": [{"update": "2026-11-14T13:40:00+00:00", "bookmakers": [{"id": 8, "name": "Bet365", "bets": [
        {"id": 92, "name": "Anytime Goal Scorer", "values": [
            {"value": "Joel Piroe", "odd": "2.75"}, {"value": "Joel Piroe - No", "odd": "1.40"}]},
        {"id": 215, "name": "Player Shots On Target Total", "values": [
            {"value": "Jerry Yates - Over 0.5", "odd": "1.50"},
            {"value": "Jerry Yates - Under 0.5", "odd": "2.50"}]}]}]}]}
    rows, _ = CP.parse_odds(body, {"fixture_id": 1}, page=1)
    got = {(r["player_name"], r["side_word"], r["line"], r["odd"]) for r in rows}
    assert got == {("Joel Piroe", None, None, 2.75), ("Joel Piroe", "no", None, 1.40),
                   ("Jerry Yates", "over", 0.5, 1.50), ("Jerry Yates", "under", 0.5, 2.50)}


def test_fair_basis_label_on_telegram_table_and_dashboard(monkeypatch):
    from player_model import fair_basis as FB
    assert FB.label("two_sided_devig") == "two-sided price"
    assert FB.label("assumed_overround") == "margin assumed"
    assert FB.label(None) is None and FB.label(float("nan")) is None
    # Telegram daily props picture: a "Fair price" column with the label on priced rows
    from telegram_bot import daily_pack as D
    t = pd.DataFrame([
        {"date": "2026-10-24", "kickoff_utc": "2026-10-24 16:30", "league": "Premier League", "match": "A vs B",
         "player_id": 1, "player_name": "P1", "market": "goals", "model_prob": 0.6, "market_odds": 2.0,
         "fair_basis": "two_sided_devig"},
        {"date": "2026-10-24", "kickoff_utc": "2026-10-24 16:30", "league": "Premier League", "match": "A vs B",
         "player_id": 2, "player_name": "P2", "market": "goals", "model_prob": 0.5, "market_odds": 2.5,
         "fair_basis": "assumed_overround"},
        {"date": "2026-10-24", "kickoff_utc": "2026-10-24 16:30", "league": "Premier League", "match": "A vs B",
         "player_id": 3, "player_name": "P3", "market": "goals", "model_prob": 0.4, "market_odds": None,
         "fair_basis": None}])
    monkeypatch.setattr(D, "_read", lambda name, **k: t.copy() if name == "player_tips.csv" else pd.DataFrame())
    cap = {}
    monkeypatch.setattr(D, "render_table", lambda title, cols, rows, **k: cap.update(cols=cols, rows=rows) or [b"png"])
    D.props_image("2026-10-24")
    names = [c for c, _ in cap["cols"]]
    assert names[-1] == "Fair price"
    cells = {r["cells"][1]: r["cells"][-1] for r in cap["rows"] if "cells" in r}
    assert cells == {"P1": "two-sided price", "P2": "margin assumed", "P3": "–"}
    # Telegram instant alert + briefing read the same label
    from telegram_bot import notifier as N
    src = inspect.getsource(N.notify_player_props) + inspect.getsource(N.notify_props_daily_digest)
    assert src.count("_fair_basis_label(") == 2
    # dashboard props board: a Fair price column on the priced view
    from dashboard_data import props_view as V
    d = t.iloc[:2].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["ev"], d["tier"], d["team"], d["position"] = 0.2, "VALUABLE", "T", "F"
    rows, _ = V.board(d, True, limit=10)
    assert sorted(rows["Fair price"]) == ["margin assumed", "two-sided price"]
    assert "Fair price" in V.board_config(True)


def test_props_stay_paper_only():
    import json
    pol = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
    assert "BLOCKED_FROM_REAL" in json.dumps(pol) and "prop" in json.dumps(pol).lower()


# ── 3. Fantasy: convention-based figures labelled, values unchanged ──────────────────────────────
def test_fantasy_page_labels_the_convention_based_figures():
    from dashboard_data import fantasy_view as fv
    assert fv.CONVENTION_LABEL == "convention-based, not fitted on data"
    assert fv.CONVENTION_COLS == ("xpts_3gw", "xpts_5gw", "value_5gw_per_m")
    for word in ("Real or fitted", "Convention-based, not fitted on data", "3-GW", "5-GW/£m", "V1"):
        assert word in fv.FIGURES_NOTE, word
    page = next((ROOT / "pages").glob("13_*_Fantasy.py")).read_text(encoding="utf-8")
    assert "st.caption(fv.FIGURES_NOTE)" in page                                    # the page note
    best = page[page.index('ui.section_header("Best value"'):page.index('ui.section_header("Fixture swing"')]
    assert "fv.CONVENTION_LABEL" in best                                            # Best value labelled
    for col in fv.CONVENTION_COLS:                                                  # explorer columns
        line = next(ln for ln in page.splitlines() if f'"{col}": st.column_config.NumberColumn(' in ln)
        assert "(convention)" in line, col
    assert page.count("fv.V1_NOTE") >= 5                                           # every V1 tool
    assert "Every V1 points figure" in fv.V1_NOTE and fv.CONVENTION_LABEL in fv.V1_NOTE


def test_fantasy_convention_figures_keep_their_values():
    """The label is presentation only: the selection the page shows is the artifact's own numbers."""
    from dashboard_data import fantasy_view as fv
    proj = pd.DataFrame({"player_name": list("abcd"), "p_start": [0.9, 0.9, 0.5, 0.9],
                         "risk": ["Low", "Low", "Low", "Doubtful"],
                         "value_5gw_per_m": [2.0, 3.0, 9.0, 8.0], "xpts_5gw": [10.0, 12.0, 40.0, 30.0]})
    bv = fv.best_value(proj, 3)
    assert list(bv["player_name"]) == ["b", "a"]
    assert list(bv["value_5gw_per_m"]) == [3.0, 2.0]
