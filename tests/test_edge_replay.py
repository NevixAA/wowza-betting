"""RETROSPECTIVE edge replay (src/research/edge_replay.py): the guarantees the comparison rests on.

    no post-kickoff / post-entry prices  ·  no synthetic prices  ·  same-book same-snapshot pairs only
    chronological discovery/holdout      ·  legacy reproduction matches production / ledger tiers
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import market_edge as ME
from src.research import edge_replay as R

ROOT = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2026-10-01T12:00:00Z")
KO = pd.Timestamp("2026-10-03T15:00:00Z")


def _q(rows):
    """rows: (ts, book, side, odds) for one fixture, OU25."""
    q = pd.DataFrame([{"snapshot_ts": ts, "match_date": "2026-10-03", "match": "A vs B", "bookmaker": bk,
                       "market": "OU25", "side": sd, "odds": o} for ts, bk, sd, o in rows])
    q["_ts"] = pd.to_datetime(q["snapshot_ts"], utc=True)
    return q


# ── prices: nothing after the entry instant, nothing at/after kickoff ─────────────────────────────
def test_pairs_at_never_sees_a_quote_written_after_entry():
    q = _q([("2026-10-01T10:00:00Z", "bk1", "OVER", 1.90), ("2026-10-01T10:00:00Z", "bk1", "UNDER", 1.95),
            ("2026-10-01T13:00:00Z", "bk1", "OVER", 2.50), ("2026-10-01T13:00:00Z", "bk1", "UNDER", 1.50)])
    (bq,) = R.pairs_at(q, T0, "OU25")
    assert (bq.odds_a, bq.odds_b) == (1.90, 1.95)                     # the 13:00 move is invisible at 12:00
    assert bq.ts <= T0


def test_certified_close_excludes_post_kickoff_and_requires_window():
    P = ME.load_params()
    pre = [("2026-10-03T14:45:00Z", b, s, o) for b in ("b1", "b2", "b3") for s, o in (("OVER", 1.9), ("UNDER", 1.95))]
    inplay = [("2026-10-03T15:20:00Z", b, s, o) for b in ("b1", "b2", "b3") for s, o in (("OVER", 5.0), ("UNDER", 1.1))]
    c = R.certified_close(_q(pre + inplay), KO, "OU25", P)
    assert c["close_status"] == "OK"
    assert abs(c["p_close_fair"] - ME.power_devig(1.9, 1.95)) < 1e-9  # in-play 5.0/1.1 never used
    early = [("2026-10-03T12:00:00Z", b, s, o) for b in ("b1", "b2", "b3") for s, o in (("OVER", 1.9), ("UNDER", 1.95))]
    c2 = R.certified_close(_q(early), KO, "OU25", P)
    assert c2["close_status"] == "NO_CLOSE_IN_WINDOW" and np.isnan(c2["p_close_fair"])   # never an earlier price
    c3 = R.certified_close(_q(inplay), KO, "OU25", P)
    assert c3["close_status"] == "POST_KICKOFF_ONLY"


def test_certified_close_needs_three_two_sided_books():
    P = ME.load_params()
    rows = [("2026-10-03T14:50:00Z", b, s, o) for b in ("b1", "b2") for s, o in (("OVER", 1.9), ("UNDER", 1.95))]
    rows += [("2026-10-03T14:50:00Z", "b3", "OVER", 1.9)]           # one-sided third book
    assert R.certified_close(_q(rows), KO, "OU25", P)["close_status"] == "INSUFFICIENT_BOOKS"


# ── pairing: same book only ───────────────────────────────────────────────────────────────────────
def test_two_sided_pair_is_never_formed_across_books():
    q = _q([("2026-10-01T10:00:00Z", "bkA", "OVER", 2.20), ("2026-10-01T10:00:00Z", "bkB", "UNDER", 2.10)])
    qs = R.pairs_at(q, T0, "OU25")
    assert len(qs) == 2
    assert all((x.odds_a is None) != (x.odds_b is None) for x in qs)  # both one-sided
    c = ME.Candidate(family="OU25", track="standard", p_model=0.55, kickoff=KO, now=T0, quotes=qs)
    e = ME.evaluate(c)
    assert e["n_books"] == 0 and e["decision_state"] in ("OBSERVE", "NO_BET")
    assert e["market_source"] in ("ONE_SIDED", "NONE")


def test_pair_timestamp_is_the_later_side():
    q = _q([("2026-10-01T08:00:00Z", "bk1", "OVER", 1.9), ("2026-10-01T11:00:00Z", "bk1", "UNDER", 1.95)])
    (bq,) = R.pairs_at(q, T0, "OU25")
    assert bq.ts == pd.Timestamp("2026-10-01T11:00:00Z")


# ── synthetic prices ──────────────────────────────────────────────────────────────────────────────
def _board(**over):
    row = {"league": "Championship", "date": "2026-10-03", "kickoff_utc": "2026-10-03T15:00:00Z",
           "home_team": "A", "away_team": "B", "odds_over25": 1.9, "odds_under25": 1.95, "p_over25": 0.55,
           "p_btts": 0.5, "odds_btts": 1.85, "odds_btts_no": 1.95, "model_type": "standard",
           "generated_at": "2026-10-01T12:00:00Z", "committed_at": pd.Timestamp("2026-10-01T12:01:00Z"),
           "commit_sha": "aaa", "price_synthetic": "False"}
    row.update(over)
    return pd.DataFrame([row]).reindex(columns=R.BOARD_COLS + ["commit_sha", "committed_at"])


def test_first_sight_drops_synthetic_rows():
    b = pd.concat([_board(price_synthetic="True"),
                   _board(generated_at="2026-10-01T13:00:00Z", committed_at=pd.Timestamp("2026-10-01T13:01Z"),
                          commit_sha="bbb")], ignore_index=True)
    fs = R.first_sight(b, "OU25")
    assert len(fs) == 1 and fs["commit_sha"].iloc[0] == "bbb"        # the flagged earlier run is not first sight


def test_synthetic_book_quote_is_never_a_market():
    qs = [ME.BookQuote("bk1", 1.9, 1.95, T0 - pd.Timedelta(minutes=5), synthetic=True)]
    e = ME.evaluate(ME.Candidate(family="OU25", track="standard", p_model=0.55, kickoff=KO, now=T0, quotes=qs))
    assert e["n_books"] == 0 and e["decision_state"] == "NO_BET"


def test_first_sight_is_earliest_pre_kickoff_run_with_a_price():
    b = pd.concat([
        _board(odds_over25=np.nan, commit_sha="x0"),                                     # no price yet
        _board(generated_at="2026-10-01T14:00:00Z", committed_at=pd.Timestamp("2026-10-01T14:01Z"), commit_sha="x1"),
        _board(generated_at="2026-10-02T14:00:00Z", committed_at=pd.Timestamp("2026-10-02T14:01Z"), commit_sha="x2"),
        _board(generated_at="2026-10-03T15:05:00Z", committed_at=pd.Timestamp("2026-10-03T15:06Z"), commit_sha="x3"),
    ], ignore_index=True)
    fs = R.first_sight(b, "OU25")
    assert list(fs["commit_sha"]) == ["x1"]
    assert (fs["obs_ts"] < fs["kickoff"]).all()


def test_observation_time_never_after_commit():
    b = _board(generated_at="2026-10-01T15:00:00Z")                  # claims to be later than its commit
    assert R.observation_time(b).iloc[0] == pd.Timestamp("2026-10-01T12:01:00Z")


# ── chronology ────────────────────────────────────────────────────────────────────────────────────
def _synthetic_replay(n=200, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-08-20", periods=n, freq="7h", tz="UTC")
    pk = rng.uniform(0.35, 0.65, n)
    y = (rng.uniform(size=n) < pk).astype(float)
    return pd.DataFrame({"entry_ts": ts, "kickoff": ts + pd.Timedelta(hours=30), "y_a": y,
                         "two_sided": True, "p_mkt_a": pk, "p_mkt_a_power": pk, "p_mkt_a_prop": pk,
                         "p_model_a": np.clip(pk + rng.normal(0, 0.05, n), 0.05, 0.95),
                         "close_status": "NO_CLOSE_IN_WINDOW", "p_close_fair": np.nan,
                         "new_side_a_odds": 1 / pk * 0.95, "new_side_b_odds": 1 / (1 - pk) * 0.95})


def test_param_fit_discovery_strictly_precedes_holdout():
    r = R.fit_params_cell(_synthetic_replay(), clip=0.35, default_w=0.10)
    assert r["status"] == "PROPOSED_DISCOVERY_ONLY"
    assert pd.Timestamp(r["discovery_window"][1]) < pd.Timestamp(r["holdout_window"][0])
    assert r["n_discovery"] > r["n_holdout"] > 0


def test_disagreement_bins_fit_on_first_half_only():
    d = _synthetic_replay()
    r = R.disagreement_cell(d, n_boot=50)
    assert r["status"] == "OK"
    assert pd.Timestamp(r["discovery_window"][1]) < pd.Timestamp(r["holdout_window"][0])
    first = d.sort_values("entry_ts").iloc[: len(d) // 2]
    exp = np.quantile(np.abs(first["p_model_a"] - first["p_mkt_a"]), [1 / 3, 2 / 3])
    assert np.allclose(r["bin_edges_from_discovery"], np.round(exp, 4))


def test_small_cells_are_insufficient_not_rates():
    d = _synthetic_replay(n=20)
    assert R.disagreement_cell(d, n_boot=10)["status"] == "INSUFFICIENT"
    assert R.fit_params_cell(d, 0.35, 0.10)["status"] == "INSUFFICIENT"


def test_clv_context_is_point_in_time():
    led = pd.DataFrame({"model_type": ["standard"] * 3, "clv_pct": [10.0, 10.0, -50.0],
                        "match_date": ["2026-09-01", "2026-09-02", "2026-09-10"]})
    c = R.ClvContext(led, None, 50.0)
    assert c.at(pd.Timestamp("2026-09-05"))["standard|OU25"]["n"] == 2
    assert c.at(pd.Timestamp("2026-09-01")) == {}                    # same-day rows are not yet known


# ── legacy reproduction ───────────────────────────────────────────────────────────────────────────
def test_threshold_regime_switch_b82cec9f():
    assert R.regime_at(pd.Timestamp("2026-08-20T00:00Z"))["marksman"] == 0.14
    assert R.regime_at(pd.Timestamp("2026-08-22T00:00Z"))["marksman"] == 0.08
    assert R.legacy_ou25_base_tier(0.10, "OVER", "Serie A", pd.Timestamp("2026-08-20T00:00Z")) == "VALUABLE"
    assert R.legacy_ou25_base_tier(0.10, "OVER", "Serie A", pd.Timestamp("2026-08-22T00:00Z")) == "MARKSMAN"
    assert R.legacy_ou25_base_tier(0.035, "UNDER", "Serie A", pd.Timestamp("2026-08-20T00:00Z")) == "AVOID"
    assert R.legacy_ou25_base_tier(0.035, "UNDER", "Serie A", pd.Timestamp("2026-08-22T00:00Z")) == "VALUABLE"


@pytest.mark.skipif(not (ROOT / "output" / "predictions.csv").exists(), reason="no committed board")
def test_legacy_ou25_reproduces_the_committed_board_exactly():
    b = pd.read_csv(ROOT / "output" / "predictions.csv", low_memory=False)
    ts = pd.to_datetime(b["generated_at"], utc=True).max()
    stdp = json.loads((ROOT / "models" / "best_params_standard.json").read_text(encoding="utf-8"))
    for side, ecol, tcol in (("OVER", "edge_over", "tier_over"), ("UNDER", "edge_under", "tier_under")):
        rep = [R.legacy_ou25_base_tier(e, side, lg, ts, stdp) for e, lg in zip(b[ecol], b["league"])]
        assert (pd.Series(rep) == b[tcol].map(R._tier_str)).all()


def test_legacy_side_tier_bars():
    e = 0.70 - (1 / 1.5) / 1.08                                       # 0.0827 -> MARKSMAN
    assert R.legacy_side_tier(0.70, 1.5, "X", "btts", {}) == (
        "SNIPER" if e >= .10 else "MARKSMAN" if e >= .08 else "VALUABLE" if e >= .04 else "AVOID", e)
    assert R.legacy_side_tier(None, 1.5, "X", "btts", {})[0] == "NOT_EVALUATED"
    appr = {"btts": {"X": {"approved": True, "sniper_th": 0.05, "marksman_th": 0.045}}}
    assert R.legacy_side_tier(0.70, 1.5, "X", "btts", appr)[0] == "SNIPER"


@pytest.mark.skipif(not (ROOT / "output" / "side_bets_ledger.csv").exists(), reason="no committed ledger")
def test_legacy_side_reproduction_matches_ledger_except_in_place_upgrades():
    """The side ledger upgrades tiers in place after first sight, so the reproduced FIRST-SIGHT tier may
    be lower than the ledger's, almost never higher. Sample = every committed ledger row."""
    s = pd.read_csv(ROOT / "output" / "side_bets_ledger.csv", low_memory=False)
    s = s[s["market"].str.lower().isin(["btts", "over15"])]
    sp = json.loads((ROOT / "models" / "best_params_side_markets.json").read_text(encoding="utf-8"))
    rep = pd.Series([R.legacy_side_tier(p, o, lg, m.lower(), sp)[0]
                     for p, o, lg, m in zip(s["model_prob"], s["odds"], s["league"], s["market"])], index=s.index)
    led = s["signal_tier"].map(R._tier_str)
    rank = {"AVOID": 0, "NOT_EVALUATED": 0, "VALUABLE": 1, "MARKSMAN": 2, "SNIPER": 3}
    equal = (rep == led).mean()
    repro_higher = (rep.map(rank) > led.map(rank)).mean()
    assert equal >= 0.75
    assert repro_higher <= 0.03


def test_legacy_vs_ledger_counts_same_price_only():
    df = pd.DataFrame({"legacy_bet": [True, True], "family": ["OU25", "OU25"], "date": ["2026-10-03"] * 2,
                       "home_team": ["A", "C"], "away_team": ["B", "D"], "legacy_side": ["OVER", "UNDER"],
                       "legacy_odds": [2.0, 1.9], "legacy_tier": ["VALUABLE", "SNIPER"]})
    led = pd.DataFrame({"match_date": ["2026-10-03"] * 2, "home_team": ["A", "C"], "away_team": ["B", "D"],
                        "side": ["OVER", "UNDER"], "odds": [2.0, 2.4], "signal_tier": ["MARKSMAN", "SNIPER"]})
    r = R.legacy_vs_ledger(df, led, None)["OU25"]
    assert r["matched_same_price"] == 1 and r["tier_equal"] == 0 and r["ledger_tier_higher"] == 1


# ── settlement / outcome ──────────────────────────────────────────────────────────────────────────
def test_outcome_side_a():
    assert R.outcome_a("OU25", 2, 1) == 1.0 and R.outcome_a("OU25", 1, 1) == 0.0
    assert R.outcome_a("BTTS", 1, 0) == 0.0 and R.outcome_a("BTTS", 1, 1) == 1.0
    assert R.outcome_a("OU15", 1, 1) == 1.0 and np.isnan(R.outcome_a("OU15", np.nan, 1))


def test_settle_bets_uses_entry_price_and_certified_clv_only():
    df = pd.DataFrame({"legacy_bet": [True, True], "legacy_side": ["UNDER", "OVER"], "legacy_odds": [2.0, 2.0],
                       "y_a": [0.0, 0.0], "close_status": ["OK", "NO_CLOSE_IN_WINDOW"],
                       "p_close_fair": [0.45, 0.40], "kickoff": pd.to_datetime(["2026-10-03T15:00Z"] * 2)})
    x = R.settle_bets(df, "LEGACY")
    assert list(x["pnl"]) == [1.0, -1.0]
    assert abs(x["clv_pct"].iloc[0] - (2.0 * 0.55 - 1) * 100) < 1e-9 and np.isnan(x["clv_pct"].iloc[1])


def test_block_bootstrap_and_bh():
    lo, hi = R.block_bootstrap(np.r_[np.ones(10), -np.ones(10)], np.repeat(np.arange(4), 5), n=200)
    assert lo <= 0 <= hi
    assert R.bh([0.01, 0.04, None, 0.5]) == [0.03, 0.06, None, 0.5]


def test_replay_never_writes_production_config():
    before = ME.CONFIG_PATH.read_bytes()
    R.fit_params_cell(_synthetic_replay(), 0.35, 0.10)
    assert ME.CONFIG_PATH.read_bytes() == before
    assert json.loads(before)["status"] == "UNVALIDATED_DEFAULTS"
