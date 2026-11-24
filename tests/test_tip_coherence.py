"""The fixture coherence gate (src/tip_coherence.py) and the HT tip gate (telegram_bot/notifier.notify_ht_tips).

Owner 2026-11-19 ("so fix"):
  BUG 1  HT tips went out with no market price, on one global probability threshold, saying "Paper only".
  BUG 2  two tips on the same match that work against each other (BTTS Yes + Under 2.5, ...) could both go out.
Synthetic data only; no Telegram call is made (the sender is replaced)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src import tip_coherence as TC
from src.tip_coherence import Candidate, CoherenceGate, Event, FixtureModel, fit_ft, mismatch, parse_event

ROOT = Path(__file__).resolve().parents[1]
DAY = "2030-01-05"
FX = TC.fixture_id(DAY, "Home FC", "Away FC")
BTTS_Y, BTTS_N = Event("FT", "BTTS", "YES"), Event("FT", "BTTS", "NO")
O25, U25 = Event("FT", "TOTAL", "OVER", 2.5), Event("FT", "TOTAL", "UNDER", 2.5)
O35, U35, O15 = Event("FT", "TOTAL", "OVER", 3.5), Event("FT", "TOTAL", "UNDER", 3.5), Event("FT", "TOTAL", "OVER", 1.5)
HT_O05, HT_U15 = Event("HT", "TOTAL", "OVER", 0.5), Event("HT", "TOTAL", "UNDER", 1.5)


@pytest.fixture(scope="module")
def model():
    """A typical fixture fitted to v9-style probabilities (+ HT rates)."""
    fit = fit_ft({"p_over15": 0.75, "p_over25": 0.52, "p_over35": 0.30, "p_btts": 0.55})
    assert fit is not None and fit["sse"] < 1e-3
    return FixtureModel(fit, (0.62, 0.50))


# ── the rule ────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("a,b,bad", [
    (BTTS_Y, U25, True),          # both win only on 1-1
    (O25, BTTS_Y, False),         # positively dependent - allowed
    (U25, BTTS_N, False),         # positively dependent - allowed
    (O35, BTTS_N, True),          # both win only on 4-0 / 0-4 / 5-0 ...
    (O25, U35, True),             # owner 2026-11-04: "u cant tipped over 2.5 and under 3.5"
    (O15, U25, True),             # owner 2026-11-04: "under 2.5 and over 1.5"
    (O25, U25, True),             # the old both-sides case
    (O15, O35, False),
    (HT_O05, U25, True),          # HT vs FT
    (HT_O05, O25, False),
    (HT_U15, BTTS_Y, True),
])
def test_negative_dependence_on_the_fitted_grid_and_the_hard_list_agree(model, a, b, bad):
    v = mismatch(a, b, model)
    assert v.method in ("grid_ft", "grid_ht") and v.mismatch is bad, v
    h = mismatch(a, b, None)                     # no grid -> the hard list
    assert h.method == "hard_list" and h.mismatch is bad, h


def test_same_event_is_a_duplicate_not_a_mismatch_and_unknown_markets_never_clash(model):
    assert mismatch(O25, O25, model).mismatch is False
    other = Event("FT", "OTHER", "OVER")
    assert mismatch(other, U25, model).mismatch is False and mismatch(other, U25, None).method == "no_relation"


def test_grid_and_hard_list_agree_across_realistic_rates():
    """On a Dixon-Coles / independent-Poisson grid an Over-type and an Under-type event are always negatively
    dependent (Harris / FKG), so the fallback list is the grid rule, not an approximation of it."""
    import itertools
    evs = [O15, O25, O35, U25, U35, BTTS_Y, BTTS_N, HT_O05, HT_U15]
    for lh, la, rho in itertools.product((0.5, 1.2, 2.4), (0.4, 1.0, 2.0), (-0.15, 0.0, 0.10)):
        m = FixtureModel({"lam_home": lh, "lam_away": la, "rho": rho}, (lh * 0.45, la * 0.45))
        for a, b in itertools.combinations(evs, 2):
            assert mismatch(a, b, m).mismatch == mismatch(a, b, None).mismatch, (lh, la, rho, a, b)


def test_fit_needs_btts_and_a_total_never_guesses():
    assert fit_ft({"p_over25": 0.5}) is None                      # split unpinned
    assert fit_ft({"p_btts": 0.5}) is None
    assert fit_ft({"p_over25": float("nan"), "p_btts": 0.5}) is None


@pytest.mark.parametrize("market,side,want", [
    ("ou25", "OVER", O25), ("OU25_UNDER", None, U25), ("over15", "YES", O15), ("over35", "NO", U35),
    ("btts", "YES", BTTS_Y), ("btts", None, BTTS_Y), ("BTTS", "NO", BTTS_N), ("OVER35", "OVER", O35),
    ("ht_over05", None, HT_O05), ("HT Under 1.5", None, HT_U15), ("O/U 2.5 UNDER", None, U25),
    ("BTTS YES", None, BTTS_Y), ("Over 3.5 goals", None, O35), ("Both teams to score - No", None, BTTS_N),
    ("M. Zoma to assist", None, Event("FT", "PLAYER", "OVER")), ("corners", None, None),
])
def test_every_sender_spelling_parses(market, side, want):
    assert parse_event(market, side) == want


def test_notified_keys_parse_and_non_tip_keys_are_ignored():
    keys = [f"{DAY}|Home FC|Away FC|UNDER", f"SIDE|{DAY}|Home FC|Away FC|btts", f"HT|{DAY}|Home FC|Away FC|HTOVER0.5",
            f"SIDE|{DAY}|Home FC|Away FC|over35|NO", "DIGEST|2030-01-05", "WC|x|y|z", "PLAYER|d|n|m"]
    got = [(c.fixture, c.event) for c in TC.sent_from_notified(keys)]
    assert got == [(FX, U25), (FX, BTTS_Y), (FX, HT_O05), (FX, U35)]


# ── resolution ──────────────────────────────────────────────────────────────────────────────────
def _gate(sent_keys=(), model=None):
    return CoherenceGate(TC.sent_from_notified(sent_keys), {FX: model} if model else {})


def test_first_sent_stands_cross_sender(model):
    g = _gate([f"SIDE|{DAY}|Home FC|Away FC|btts"], model)              # BTTS Yes already sent (side sender)
    why = g.check(Candidate(FX, U25, "O/U 2.5"))
    assert why.startswith("MISMATCH with BTTS Yes [notified SIDE]") and "P(both)" in why
    assert g.check(Candidate(FX, O25, "O/U 2.5")) == ""                   # same direction: allowed
    g2 = _gate(model=model)
    g2.commit(Candidate(FX, HT_O05, "HT"))                               # an HT tip sent earlier this run
    assert g2.check(Candidate(FX, U25, "O/U 2.5")).startswith("MISMATCH with HT Over 0.5 [HT] (grid_ht")


def test_within_a_run_the_higher_ranked_stands_whatever_the_order(model):
    for order in (1, -1):
        cands = [Candidate(FX, BTTS_Y, "side", rank=0.04, uid="b"), Candidate(FX, U25, "O/U 2.5", rank=0.09, uid="u"),
                 Candidate(FX, U35, "side", rank=0.01, uid="u35")][::order]
        g = _gate(model=model)
        losers = g.plan(cands)
        assert set(losers) == {"b"} and "MISMATCH with Under 2.5 [O/U 2.5]" in losers["b"]
        assert g.check(cands[[c.uid for c in cands].index("b")]) == losers["b"]


def test_a_sent_tip_beats_a_higher_ranked_new_one(model):
    g = _gate([f"{DAY}|Home FC|Away FC|UNDER"], model)
    assert set(g.plan([Candidate(FX, BTTS_Y, "side", rank=9.0, uid="b")])) == {"b"}


# ── the senders ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def notifier_env(tmp_path, monkeypatch):
    from telegram_bot import notifier as N
    nf = tmp_path / "notified.json"
    nf.write_text(json.dumps({"keys": []}), encoding="utf-8")
    monkeypatch.setattr(N, "NOTIFIED_FILE", nf)
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_utc_today_str", lambda: DAY)
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    sent = []
    monkeypatch.setattr(N, "_send", lambda token, chat, text: sent.append(text) or True)
    monkeypatch.setattr(TC.config, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"date": DAY, "home_team": "Home FC", "away_team": "Away FC", "league": "La Liga 2",
                   "p_over15": 0.75, "p_over25": 0.52, "p_over35": 0.30, "p_btts": 0.55,
                   "p_ht_over05": 0.80, "p_ht_over15": 0.40, "ht_lambda_home": 0.62, "ht_lambda_away": 0.5}]
                 ).to_csv(tmp_path / "predictions.csv", index=False)
    return N, nf, sent, tmp_path


def _bets(tmp_path, side="UNDER"):
    pd.DataFrame([{"date": DAY, "kickoff_utc": f"{DAY}T15:00:00Z", "league": "La Liga 2", "home_team": "Home FC",
                   "away_team": "Away FC", "bet": side, "best_side": side, "signal_tier": "QUALIFIED",
                   "odds_over25": 2.0, "odds_under25": 1.9, "best_edge": 0.05, "edge_calibrated": 0.02,
                   "model_type": "standard", "cell_status": "UNPROVEN", "drift_signal": None}]
                 ).to_csv(tmp_path / "bets.csv", index=False)


def _side(tmp_path, market="btts"):
    pd.DataFrame([{"date": DAY, "league": "La Liga 2", "home_team": "Home FC", "away_team": "Away FC",
                   "market": market, "side": "YES", "signal_tier": "QUALIFIED", "edge": 0.06, "ev": 0.1,
                   "market_odds": 1.8, "model_prob": 0.6, "edge_calibrated": 0.03, "model_type": "standard",
                   "threshold_source": "LEAGUE", "cell_status": "UNPROVEN"}]).to_csv(tmp_path / "side_bets.csv",
                                                                                    index=False)


def test_ou_sender_skips_a_mismatch_with_an_already_sent_side_tip(notifier_env, capsys):
    N, nf, sent, tmp = notifier_env
    nf.write_text(json.dumps({"keys": [f"SIDE|{DAY}|Home FC|Away FC|btts"]}), encoding="utf-8")
    _bets(tmp, "UNDER")
    assert N.notify_new_snipers() == 0 and sent == []
    assert "MISMATCH with BTTS Yes" in capsys.readouterr().out
    assert f"{DAY}|Home FC|Away FC|UNDER" not in json.loads(nf.read_text())["keys"]


def test_one_run_plan_sends_only_the_higher_ranked_of_a_mismatched_pair(notifier_env, capsys):
    N, nf, sent, tmp = notifier_env
    _bets(tmp, "UNDER")                 # calibrated EV 0.02 x 1.9 = 0.038
    _side(tmp, "btts")                  # calibrated EV 0.03 x 1.8 = 0.054 -> stands
    gate = N.plan_run_gate()
    assert N.notify_new_snipers(gate) == 0
    assert N.notify_side_bets(gate) == 1 and "BTTS" in sent[0]
    assert "MISMATCH with" in capsys.readouterr().out


def test_side_sender_needs_a_price_and_a_league_bar(notifier_env, capsys):
    N, nf, sent, tmp = notifier_env
    _side(tmp)
    d = pd.read_csv(tmp / "side_bets.csv")
    d.loc[0, "market_odds"] = None
    d.to_csv(tmp / "side_bets.csv", index=False)
    assert N.notify_side_bets() == 0 and "NO_MARKET_PRICE" in capsys.readouterr().out
    d.loc[0, "market_odds"] = 1.8
    d.loc[0, "threshold_source"] = "GLOBAL"
    d.to_csv(tmp / "side_bets.csv", index=False)
    assert N.notify_side_bets() == 0 and "GLOBAL_BAR" in capsys.readouterr().out and sent == []


# ── HT ──────────────────────────────────────────────────────────────────────────────────────────
def test_ht_without_a_price_or_an_approved_bar_sends_nothing_and_says_why(notifier_env, capsys):
    N, nf, sent, tmp = notifier_env
    N._HT_PRICE_FN = None
    import src.ledger as L
    orig = L._ht_price_lookup
    try:
        L._ht_price_lookup = lambda: None                                      # no capture at all
        assert N.notify_ht_tips() == 0
        out = capsys.readouterr().out
        assert "HT skip Home FC v Away FC" in out and "NO_MARKET_PRICE" in out
        N._HT_PRICE_FN = None
        L._ht_price_lookup = lambda: (lambda *a: 1.9)                          # priced, but no HT bar exists
        assert N.notify_ht_tips() == 0
        out = capsys.readouterr().out
        assert "NO_LEAGUE_BAR" in out and sent == []
    finally:
        L._ht_price_lookup = orig
        N._HT_PRICE_FN = None


def test_ht_has_no_global_threshold_left():
    import inspect
    from telegram_bot import notifier as N
    src = inspect.getsource(N.ht_tip_candidates) + inspect.getsource(N.notify_ht_tips)
    for banned in (">= 0.75", "<= 0.30", ">= 0.60", "<= 0.25", "Paper", "paper", "no market price captured"):
        assert banned not in src, banned


def test_ht_candidate_with_price_and_cleared_cell_is_sendable_and_message_has_no_retired_word(notifier_env):
    N, nf, sent, tmp = notifier_env
    from src.bars_opt import serve as BS
    df = N._ht_board()
    cell = {"mode": "MODEL"}
    orig = BS.score
    try:
        BS.score = lambda frame, c: pd.DataFrame([{"tier": "QUALIFIED", "reason": "", "measure": 0.04}])
        cands = N.ht_tip_candidates(df, price_fn=lambda *a: 1.5, bar_fn=lambda lg, line, side: (cell, ""))
        ok = [c for c in cands if not c["reason"]]
        assert {(c["market"]) for c in ok} == {"ht_over05", "ht_under05", "ht_over15", "ht_under15"}
        no_bar = N.ht_tip_candidates(df, price_fn=lambda *a: 1.5)               # the real (absent) HT bars
        assert all(c["reason"].startswith("NO_LEAGUE_BAR") for c in no_bar)
    finally:
        BS.score = orig


# ── album, Bet Builder, TOP-3 ───────────────────────────────────────────────────────────────────
def test_album_rows_drop_the_lower_ranked_of_a_mismatch(model):
    from telegram_bot.tips_table import coherent_rows
    rows = pd.DataFrame([{"date": DAY, "home_team": "Home FC", "away_team": "Away FC", "market": "O/U 2.5 UNDER",
                          "edge": 1.0, "odds": 1.9},
                         {"date": DAY, "home_team": "Home FC", "away_team": "Away FC", "market": "BTTS YES",
                          "edge": 3.0, "odds": 1.8},
                         {"date": DAY, "home_team": "Home FC", "away_team": "Away FC", "market": "O/U 1.5 OVER",
                          "edge": 2.0, "odds": 1.3}])
    kept = coherent_rows(rows, gate=_gate(model=model))
    assert list(kept["market"]) == ["BTTS YES", "O/U 1.5 OVER"]
    kept2 = coherent_rows(rows, gate=_gate([f"{DAY}|Home FC|Away FC|UNDER"], model))   # Under already sent
    assert list(kept2["market"]) == ["O/U 2.5 UNDER"]


def test_bet_builder_combo_with_a_leg_against_a_sent_tip_is_left_out(model):
    from telegram_bot.daily_pack import coherent_combos
    d = pd.DataFrame([{"match_date": DAY, "match": "Home FC vs Away FC", "legs": "Over 3.5 goals (40%) · X to assist (28%)"},
                      {"match_date": DAY, "match": "Other vs Club", "legs": "Over 3.5 goals (40%)"},
                      {"match_date": DAY, "match": "Home FC vs Away FC", "legs": "BTTS Yes (55%) · Over 2.5 goals (44%)"}])
    kept = coherent_combos(d, gate=_gate([f"{DAY}|Home FC|Away FC|UNDER"], model))
    assert list(kept["match"]) == ["Other vs Club"]
    kept2 = coherent_combos(d.iloc[[2]], gate=_gate([f"SIDE|{DAY}|Home FC|Away FC|btts"], model))
    assert len(kept2) == 1                                                   # same direction: kept


def test_top3_respects_the_gate_against_a_sent_tip(model):
    from tests.test_top3_league import _cand, _ctx, _frame, LEAD, NOW
    from src import top3_league as T
    rows = [_cand(1, "OU25", "UNDER", pm=0.7, ev_lb=0.09, home="X", away="Y"),
            _cand(3, "OVER15", "OVER", pm=0.7, ev_lb=0.07, home="P", away="Q")]
    d = rows[0]["match_date"]
    gate = CoherenceGate(TC.sent_from_notified([f"HT|{d}|X|Y|HTOVER0.5"]), {})   # HT Over sent on X v Y
    picks, weeks = T.freeze_due(_frame(rows), set(), _ctx(), NOW, LEAD, {}, {}, None, None, gate=gate)
    assert list(picks["fixture_id"]) == ["fx3"]
    assert json.loads(weeks["excluded"].iloc[0])[T.R_MISMATCH] == 1
    picks0, _ = T.freeze_due(_frame(rows), set(), _ctx(), NOW, LEAD, {}, {}, None, None)
    assert set(picks0["fixture_id"]) == {"fx1", "fx3"}                     # without the sent HT tip: both


def test_top3_notify_leaves_out_a_pick_that_now_mismatches(tmp_path, monkeypatch):
    from tests.test_top3_league import _cand, _ctx, _frame, LEAD, NOW
    from src import top3_league as T
    rows = [_cand(1, "OU25", "UNDER", pm=0.7, ev_lb=0.09, home="X", away="Y"),
            _cand(3, "OVER15", "OVER", pm=0.7, ev_lb=0.07, home="P", away="Q")]
    picks, _ = T.freeze_due(_frame(rows), set(), _ctx(), NOW, LEAD, {}, {}, None, None)
    T.picks_store(tmp_path).append(picks)
    d = rows[0]["match_date"]
    real_load = CoherenceGate.load
    monkeypatch.setattr(CoherenceGate, "load", classmethod(
        lambda cls, **k: CoherenceGate(TC.sent_from_notified([f"SIDE|{d}|X|Y|btts"]), {})))
    sent = []
    assert T.notify(out_dir=tmp_path, env={T.FLAG_ENV: "1"}, sender=lambda t: sent.append(t) or True,
                    now=NOW)["sent"] == 1
    assert "P v Q" in sent[0] and "X v Y" not in sent[0]
    monkeypatch.setattr(CoherenceGate, "load", real_load)


# ── wording ─────────────────────────────────────────────────────────────────────────────────────
def test_telegram_builders_carry_no_retired_word():
    import importlib.util
    spec = importlib.util.spec_from_file_location("uws", ROOT / "scripts" / "ui_wording_scan.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    word = re.compile(r"\bpaper\b", re.IGNORECASE)
    assert m.scan_telegram_sources(word) == []
    assert m.TELEGRAM_ALLOW == frozenset()


def test_wording_scan_catches_a_planted_literal(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("uws", ROOT / "scripts" / "ui_wording_scan.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    (tmp_path / "telegram_bot").mkdir()
    (tmp_path / "telegram_bot" / "x.py").write_text(
        '"""paper in a docstring is fine"""\n'
        'STATE = "PAPER"\n'
        'def f(r):\n'
        '    if r.endswith(" (paper)"):\n'
        '        pass\n'
        '    return f"<b>HT</b>\\n⚠️ Paper only — tested"\n', encoding="utf-8")
    hits = m.scan_telegram_sources(re.compile(r"\bpaper\b", re.IGNORECASE), root=tmp_path)
    assert [h["text"] for h in hits] == ["<b>HT</b>\n⚠️ Paper only — tested"]


def test_top3_p11_tiers_respect_the_gate_against_a_sent_tip():
    from tests.test_top3_tiers_p11 import _cfg, _odds
    from tests.test_top3_league import _cand, _ctx, _frame, LEAD, NOW
    from src import top3_league as T
    rows = [_odds(_cand(1, "OU25", "UNDER", pm=0.62, pf=0.50, home="X", away="Y"), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30)]
    d = rows[0]["match_date"]
    gate = CoherenceGate(TC.sent_from_notified([f"SIDE|{d}|X|Y|btts"]), {})       # BTTS Yes sent on X v Y
    cfg = _cfg()
    picks, weeks = T.freeze_due(_frame(rows), set(), _ctx(), NOW, LEAD, {}, {}, None, T.tier_stats(cfg), None, cfg,
                                gate=gate)
    assert "fx1" not in set(picks["fixture_id"]) and "fx2" in set(picks["fixture_id"])
    assert json.loads(weeks["excluded"].iloc[0])[T.R_MISMATCH] >= 1
