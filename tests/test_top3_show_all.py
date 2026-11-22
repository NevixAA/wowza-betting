"""Owner 2026-11-22 "3 per league, stake by proof" (src/top3_league.py, P11 tier mode) - synthetic data only.

  * every league-week shows T1 / T2 / T3 - always 3 when at least 3 priced, coherent candidates exist;
  * a candidate without its league's bar or below it is SHOWN, never staked; only a pick that clears its league's
    own approved bar (and the approved tier gate) carries the flat 0.5% stake;
  * labels: "0.5% stake" / "No stake: below bar" / "No stake: no approved bar" - never the word "paper";
  * never a pick without a two-sided price, outside its cell's odds range, or against the coherence gate;
  * every pick lands in the ledger with stake / stake_reason / value_basis, reported staked vs no stake apart.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import config
from src import top3_league as T
from tests.test_top3_league import LEAD, LEAGUE, NOW, _cand, _ctx, _frame
from tests.test_top3_tiers_p11 import _cfg, _odds

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))


def _run(rows, ctx=None, cfg=None, gate=None, policy=POLICY, frame=None):
    cfg = cfg or _cfg()
    return T.freeze_due(frame if frame is not None else _frame(rows), set(), ctx if ctx is not None else _ctx(),
                        NOW, LEAD, {}, policy, None, T.tier_stats(cfg), None, cfg, gate)


def _top(picks):
    return picks[picks["selection_mode"] == T.SELECTION_MODE].sort_values("top3_rank")


# ── always 3 when 3 priced candidates exist ──────────────────────────────────────────────────────────
def test_three_shown_when_the_league_has_no_bar_at_all():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30),
            _odds(_cand(3, "BTTS", "YES", pm=0.58, pf=0.55), 1.85)]
    picks, weeks = _run(rows, ctx=_ctx(markets=()))                    # no owner-applied cell anywhere
    top = _top(picks)
    assert list(top["slot"]) == list(T.TIER_NAMES) and len(top) == 3
    assert set(top["stake_reason"]) == {T.NS_NO_BAR}
    assert (pd.to_numeric(top["stake"]) == 0).all() and top["stake_effect"].str.startswith("NONE").all()
    assert not top["track_is_real"].astype(bool).any()
    assert set(top["value_basis"]) == {T.BASIS_MODEL}                 # no calibration cell -> the model's own edge
    assert pd.to_numeric(top["edge_calibrated"], errors="coerce").isna().all()  # never the raw edge under the calibrated name
    assert int(weeks["n_picks"].iloc[0]) == 3 and int(weeks["n_qualifying"].iloc[0]) == 0


def test_three_shown_and_only_the_bar_clearing_picks_are_staked():
    # bar 0.02 on p - fair: fx1 (+0.12) clears, fx2 (+0.01) and fx3 (+0.005) are below the bar
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.79, pf=0.78), 1.30),
            _odds(_cand(3, "BTTS", "YES", pm=0.555, pf=0.55), 1.85)]
    picks, _ = _run(rows)
    top = _top(picks)
    assert len(top) == 3
    staked = top[top["stake_reason"] == T.STAKED]
    assert list(staked["fixture_id"]) == ["fx1"]
    assert pd.to_numeric(staked["stake"]).tolist() == [0.005]
    assert staked["stake_effect"].str.startswith("REAL 0.5%").all()
    rest = top[top["stake_reason"] != T.STAKED]
    assert set(rest["stake_reason"]) == {T.NS_BELOW} and (pd.to_numeric(rest["stake"]) == 0).all()
    assert rest["stake_effect"].str.startswith("NONE").all()
    # src/staked.py reads only REAL stake_effects: the unstaked picks never become staked bets
    from src.staked import top3_records_frame
    assert len(top3_records_frame(top)) == 1


def test_fewer_than_three_only_when_fewer_priced_candidates_and_the_message_says_so():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),
            _cand(2, "BTTS", "YES", pq="NO_PRICE"),                        # no two-sided price: never a pick
            _cand(3, "OVER15", "OVER", pm=0.80, pf=float("nan"))]          # no fair price: never a pick
    lock = _cand(99, ko=pd.Timestamp("2026-11-20T19:00:00Z").strftime("%Y-%m-%dT%H:%M:%SZ"), pq="NO_PRICE")
    picks, weeks = _run(None, frame=pd.DataFrame(rows + [lock]).astype(object))
    top = _top(picks)
    assert list(top["fixture_id"]) == ["fx1"]
    assert json.loads(weeks["excluded"].iloc[0]).get("NO_TWO_SIDED_PRICE") == 3
    msg = T.message(top, real_checker=lambda *a, **k: False, policy=POLICY)
    assert "Only 1 priced, coherent pick with value this week." in msg
    assert "2. T2 safest: none - no other priced, coherent pick with value" in msg
    assert "3. T3 risky value: none - no other priced, coherent pick with value" in msg


def test_never_a_pick_without_a_price_even_with_a_huge_model_edge():
    rows = [_cand(i, "OU25", "OVER", pm=0.95, pf=0.50, pq="NO_PRICE") for i in range(1, 6)]
    picks, _ = _run(rows)
    top = _top(picks)
    assert not top["fixture_id"].isin([f"fx{i}" for i in range(1, 6)]).any()
    assert (pd.to_numeric(top["odds"]) > 1).all()
    assert top["price_quality_state"].isin(["TWO_SIDED_MULTI_BOOK", "TWO_SIDED_SINGLE_BOOK"]).all()


def test_outside_the_cells_odds_range_stays_out():
    ctx = _ctx()
    for side_cell in ctx["standard"][LEAGUE]["bars_opt"].values():
        side_cell["odds_max"] = 1.95                                    # the league's own range (no global value)
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.40)]
    picks, weeks = _run(rows, ctx=ctx)
    assert "fx1" not in set(picks["fixture_id"])
    assert json.loads(weeks["excluded"].iloc[0]).get("OUTSIDE_ODDS_RANGE", 0) >= 1


# ── ranking ──────────────────────────────────────────────────────────────────────────────────────
def test_t1_value_t2_safest_t3_longest_odds_with_positive_edge():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),       # best value
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30),     # highest p
            _odds(_cand(3, "OVER35", "OVER", pm=0.34, pf=0.30), 3.20),     # longest odds, value > 0
            _odds(_cand(4, "BTTS", "YES", pm=0.60, pf=0.55), 1.85)]
    picks, _ = _run(rows, ctx=_ctx(markets=()))
    got = dict(zip(_top(picks)["slot"], _top(picks)["fixture_id"]))
    assert got == {"T1_BEST_VALUE": "fx1", "T2_SAFEST": "fx2", "T3_RISKY_VALUE": "fx3"}


def test_a_negative_value_pick_is_never_shown_and_the_slot_stays_empty():
    """Owner 2026-11-22 (2nd decision): every slot needs value > 0 - T2 is the highest p among picks WITH value,
    never a negative-value 'safe' pick; with none left the slot stays empty and says why."""
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),       # value +0.24
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.86), 1.15),     # highest p, value -0.034: never shown
            _odds(_cand(3, "OVER35", "OVER", pm=0.25, pf=0.30), 3.20),     # value -0.20: never shown
            _odds(_cand(4, "BTTS", "YES", pm=0.50, pf=0.52), 1.95)]        # value -0.025: never shown
    picks, weeks = _run(rows, ctx=_ctx(markets=()))
    top = _top(picks)
    assert list(top["fixture_id"]) == ["fx1"] and list(top["slot"]) == ["T1_BEST_VALUE"]
    ex = json.loads(weeks["excluded"].iloc[0])
    assert ex.get("NO_POSITIVE_VALUE") == 4                         # fx2, fx3, fx4 and the lock row (value 0)
    assert ex.get("EMPTY_SLOT_NO_PICK_WITH_VALUE") == 2
    msg = T.message(top, real_checker=lambda *a, **k: False, policy=POLICY)
    assert "T2 safest: none - no other priced, coherent pick with value" in msg


def test_every_shown_pick_has_positive_value():
    import random
    rnd = random.Random(7)
    rows = [_odds(_cand(i, ("OU25", "OVER15", "OVER35", "BTTS")[i % 4], ("OVER", "OVER", "OVER", "YES")[i % 4],
                        pm=rnd.uniform(0.2, 0.9), pf=rnd.uniform(0.2, 0.9)), rnd.uniform(1.1, 4.0))
            for i in range(1, 25)]
    for ctx in (_ctx(), _ctx(markets=())):
        top = _top(_run(rows, ctx=ctx)[0])
        for r in top.to_dict("records"):
            p = float(r["p_model_raw"]) if r["value_basis"] == T.BASIS_MODEL else float(r["p_market_fair"]) + 0.5 * (
                float(r["p_used"]) - float(r["p_market_fair"]))
            assert r["stake_reason"] == T.STAKED or p * float(r["odds"]) - 1 > 0, r


# ── staked picks are protected ───────────────────────────────────────────────────────────────────
def test_a_stakeable_pick_is_never_pushed_out_by_unstaked_ones():
    """Four no-bar picks beat the one stakeable pick on value, p and odds; it still keeps a slot (owner 2026-11-22)."""
    ctx = _ctx(markets=("ou25",))                                      # only O/U 2.5 has an approved bar
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.56, pf=0.50), 1.95),       # stakeable: edge +0.06 >= bar 0.02
            _odds(_cand(2, "BTTS", "YES", pm=0.75, pf=0.50), 2.00),        # no bar, far more value
            _odds(_cand(3, "OVER15", "OVER", pm=0.95, pf=0.80), 1.25),     # no bar, highest p
            _odds(_cand(4, "OVER35", "OVER", pm=0.45, pf=0.30), 3.40),     # no bar, longest odds
            _odds(_cand(5, "BTTS", "YES", pm=0.70, pf=0.50, home="H5b"), 1.90)]
    picks, _ = _run(rows, ctx=ctx)
    top = _top(picks)
    assert len(top) == 3 and "fx1" in set(top["fixture_id"])
    r = top.set_index("fixture_id").loc["fx1"]
    assert r["stake_reason"] == T.STAKED and float(r["stake"]) == 0.005


def test_staked_pick_goes_to_the_tier_it_matches_best():
    ctx = _ctx(markets=("over15",))                                    # only Over 1.5 has an approved bar
    rows = [_odds(_cand(1, "OVER15", "OVER", pm=0.93, pf=0.80), 1.22),     # stakeable, the highest p of all
            _odds(_cand(2, "BTTS", "YES", pm=0.75, pf=0.50), 2.00),        # no bar, best value
            _odds(_cand(3, "OVER35", "OVER", pm=0.40, pf=0.30), 3.40)]     # no bar, longest odds
    top = _top(_run(rows, ctx=ctx)[0])
    got = dict(zip(top["slot"], top["fixture_id"]))
    assert got == {"T1_BEST_VALUE": "fx2", "T2_SAFEST": "fx1", "T3_RISKY_VALUE": "fx3"}


def test_three_or_more_stakeable_fill_the_tiers_among_themselves():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30),
            _odds(_cand(3, "OVER35", "OVER", pm=0.36, pf=0.30), 3.40),
            _odds(_cand(4, "BTTS", "YES", pm=0.60, pf=0.55), 1.85)]
    top = _top(_run(rows)[0])
    assert set(top["stake_reason"]) == {T.STAKED} and len(top) == 3


# ── coherence and direction ───────────────────────────────────────────────────────────────────────
class _Gate:
    """A coherence gate that refuses one fixture (as src/tip_coherence would for a mismatch with a sent tip)."""

    def __init__(self, blocked_home):
        self.blocked, self.committed = blocked_home, []

    def check(self, c):
        return "MISMATCH with a sent tip" if self.blocked.lower() in c.fixture.lower() else ""

    def commit(self, c):
        self.committed.append(c)


def test_coherence_gate_refuses_a_candidate_and_the_next_one_takes_the_slot():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50, home="BLOCKED"), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30),
            _odds(_cand(3, "BTTS", "YES", pm=0.58, pf=0.55), 1.85),
            _odds(_cand(4, "OVER35", "OVER", pm=0.33, pf=0.30), 3.20)]
    gate = _Gate("BLOCKED")
    picks, weeks = _run(rows, ctx=_ctx(markets=()), gate=gate)
    top = _top(picks)
    assert len(top) == 3 and "fx1" not in set(top["fixture_id"])
    assert json.loads(weeks["excluded"].iloc[0]).get("MISMATCH_WITH_SENT_TIP", 0) >= 1
    assert len(gate.committed) == 3


def test_direction_rule_never_two_opposite_goal_directions_on_one_match():
    rows = [_odds(_cand(1, "OU25", "UNDER", pm=0.62, pf=0.50, home="H", away="A"), 2.00),
            _odds(_cand(2, "BTTS", "YES", pm=0.70, pf=0.55, home="H", away="A"), 1.85),
            _odds(_cand(3, "OVER15", "OVER", pm=0.84, pf=0.78), 1.30)]
    picks, _ = _run(rows, ctx=_ctx(markets=()))
    top = _top(picks)
    on_match = top[(top["home_team"] == "H") & (top["away_team"] == "A")]
    assert len(on_match) == 1


# ── labels ───────────────────────────────────────────────────────────────────────────────────────
def test_telegram_labels_staked_and_no_stake_picks_and_never_says_paper():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.79, pf=0.78), 1.30)]
    ctx = _ctx(markets=("ou25",))                                       # BTTS / Over 1.5 / 3.5: no approved bar
    rows.append(_odds(_cand(3, "BTTS", "YES", pm=0.58, pf=0.55), 1.85))
    picks, _ = _run(rows, ctx=ctx)
    top = _top(picks)
    assert len(top) == 3
    msg = T.message(top, real_checker=lambda *a, **k: False, policy=POLICY)
    lines = [ln for ln in msg.splitlines() if ln[:2] in ("1.", "2.", "3.")]
    by_fx = {fx: ln for fx, ln in zip(top["home_team"], lines)}
    assert by_fx["H1"].endswith(" · 0.5% stake")
    assert by_fx["H2"].endswith(" · No stake: no approved bar")
    assert by_fx["H3"].endswith(" · No stake: no approved bar")
    assert "T1 best value" in msg and "paper" not in msg.lower()
    assert "MODEL_EDGE" not in msg and "model edge" in msg


def test_below_bar_label_and_paper_league_never_staked():
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.51, pf=0.50), 2.00)]
    picks, _ = _run(rows)
    r = _top(picks).set_index("fixture_id").loc["fx1"]
    assert r["stake_reason"] == T.NS_BELOW and T.pick_stake_label(r.to_dict()) == "No stake: below bar"
    paper = sorted(config.PAPER_LEAGUES)[0]
    fr = _frame([_odds(_cand(5, "OU25", "OVER", pm=0.62, pf=0.50, league=paper), 2.0)])
    fr = fr.assign(league=paper, league_week_id=f"{paper}|2026-W47")
    cfg = _cfg()
    cfg["leagues"][f"standard|{paper}"] = {"gate": "CAL", "t3_floor": "REL"}
    picks, _ = _run(None, ctx=_ctx(league=paper), cfg=cfg, frame=fr)
    assert len(picks) and set(picks["stake_reason"]) == {T.NS_PAPER_LEAGUE}
    assert (pd.to_numeric(picks["stake"]) == 0).all()


def test_ui_wording_has_no_paper_and_labels_are_the_owners():
    assert T.NO_STAKE_LABEL[T.NS_BELOW] == "No stake: below bar"
    assert T.NO_STAKE_LABEL[T.NS_NO_BAR] == "No stake: no approved bar"
    assert not any("paper" in v.lower() for v in T.NO_STAKE_LABEL.values())


# ── ledger, migration, report ────────────────────────────────────────────────────────────────────
def test_every_pick_is_ledgered_with_its_stake_and_settles_like_the_staked_ones(tmp_path):
    rows = [_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50, home="H1", away="A1"), 2.00),
            _odds(_cand(2, "OVER15", "OVER", pm=0.79, pf=0.78, home="H2", away="A2"), 1.30),
            _odds(_cand(3, "BTTS", "YES", pm=0.555, pf=0.55, home="H3", away="A3"), 1.85)]
    picks, weeks = _run(rows)
    assert T.picks_store(tmp_path).append(picks) == len(picks)
    T.weeks_store(tmp_path).append(weeks)
    led = T.picks_store(tmp_path).read()
    assert {"stake", "stake_reason", "value_basis"} <= set(led.columns)
    v91 = pd.DataFrame([{"fixture_id": f"fx{i}", "match_date": "2026-11-21", "home_team": f"H{i}",
                         "away_team": f"A{i}", "home_goals": "2", "away_goals": "1", "result_source": "test"}
                        for i in (1, 2, 3)])
    out, n = T.settle_picks(led, v91, pd.Timestamp("2026-11-22T12:00:00Z"))
    assert n == 3
    rep = T.report(out, T.weeks_store(tmp_path).read(), NOW, LEAD)["leagues"][LEAGUE]
    assert rep["staked"]["picks"] == 1 and rep["no_stake"]["picks"] == 2
    assert rep["staked"]["settled"] + rep["no_stake"]["settled"] == rep["settled"] == 3
    labels = {p["home_team"]: p["stake_label"] for p in rep["latest_picks"]}
    assert labels["H1"] == "0.5% stake" and labels["H2"] == labels["H3"] == "No stake: below bar"


def test_migration_adds_the_stake_columns_without_touching_a_frozen_value(tmp_path):
    from src.research.first_sight import FirstSightStore
    old_frozen = [c for c in T.PICK_FROZEN if c not in T.STAKE_COLS]
    old = FirstSightStore(tmp_path / T.PICKS_NAME, key=T.PICK_KEY, columns=old_frozen + T.PICK_GRADING,
                          observed_col="issued_at", kickoff_col="kickoff_utc", frozen_cols=old_frozen, digest=True)
    picks, _ = _run([_odds(_cand(1, "OU25", "OVER", pm=0.62, pf=0.50), 2.00)])
    assert old.append(picks.drop(columns=list(T.STAKE_COLS)).reindex(columns=old_frozen + T.PICK_GRADING)) >= 1
    before = old.read()
    res = T.migrate_ledger(tmp_path)
    assert res["migrated"] is True
    after = T.picks_store(tmp_path).read()
    assert list(after.columns) == T.PICK_COLS
    pd.testing.assert_frame_equal(after[old_frozen], before[old_frozen])
    real = after["stake_effect"].str.startswith("REAL ")
    assert (after.loc[real, "stake_reason"] == T.STAKED).all()
    assert (pd.to_numeric(after.loc[real, "stake"]) == 0.005).all()
    assert (pd.to_numeric(after.loc[~real, "stake"]) == 0).all()
    assert T.picks_store(tmp_path).verify()["ok"] is True
    assert T.migrate_ledger(tmp_path)["migrated"] is False             # idempotent


def test_clean_record_reports_staked_and_no_stake_top3_apart():
    from dashboard_data import clean_record as CR
    led = pd.DataFrame([
        {"issued_at": "2026-11-28T10:00:00Z", "match_date": "2026-11-29", "flat_pnl": "1.0",
         "stake_reason": T.STAKED, "stake_effect": "REAL 0.5% of bankroll", "settle_status": "SETTLED"},
        {"issued_at": "2026-11-28T10:00:00Z", "match_date": "2026-11-29", "flat_pnl": "-1.0",
         "stake_reason": T.NS_BELOW, "stake_effect": "NONE", "settle_status": "SETTLED"}])
    s = CR.top3_summaries(led, staked=True)
    u = CR.top3_summaries(led, staked=False)
    assert s[CR.CLEAN]["n"] + u[CR.CLEAN]["n"] == CR.top3_summaries(led)[CR.CLEAN]["n"]
    if CR.top3_summaries(led)[CR.CLEAN]["n"]:
        assert math.isclose(s[CR.CLEAN]["units"], 1.0) and math.isclose(u[CR.CLEAN]["units"], -1.0)
