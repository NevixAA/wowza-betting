"""market_edge_v2 (2026-10-25) — the market-first engine and its SHADOW wiring.

Every test here fails loudly on the regression it names. The engine is shadow-only: these tests
also pin that the legacy selection, the stake plan's sizing and the bankroll-policy track states are
untouched by it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import config
import pipeline
from src import edge_engine_shadow as S
from src import market_edge as ME
from src import stake_plan as sp
from src.bankroll_policy import load_policy, size_day

NOW = pd.Timestamp("2026-10-25T12:00:00Z")
KO = NOW + pd.Timedelta(hours=30)


def _q(book, a, b, age_min=5.0):
    return ME.BookQuote(book, a, b, NOW - pd.Timedelta(minutes=age_min))


def _cand(p=0.55, quotes=None, row=(1.95, 1.95), family="OU25", track="new_format", **kw):
    base = dict(family=family, track=track, p_model=p, kickoff=KO, now=NOW,
                quotes=quotes if quotes is not None else [_q("b1", 1.95, 1.90), _q("b2", 1.97, 1.88),
                                                          _q("b3", 1.93, 1.92)],
                row_odds_a=row[0], row_odds_b=row[1], league="X", league_history_n=2000,
                model_trained_at=NOW - pd.Timedelta(days=30),
                clv={"n": 400, "mean": 1.5, "median": 0.5, "positive_pct": 0.55})
    base.update(kw)
    return ME.Candidate(**base)


def _power_newton(a, b):
    """Independent hand computation: Newton on g(k) = (1/a)^k + (1/b)^k - 1."""
    ra, rb, k = 1 / a, 1 / b, 1.0
    for _ in range(100):
        g = ra ** k + rb ** k - 1
        dg = ra ** k * math.log(ra) + rb ** k * math.log(rb)
        k -= g / dg
    return ra ** k


# ── de-vig ────────────────────────────────────────────────────────────────────────────────────
def test_proportional_devig_matches_hand_computation():
    # (1/1.8) / (1/1.8 + 1/2.1) = 2.1 / 3.9 = 7/13
    assert ME.proportional_devig(1.80, 2.10) == pytest.approx(7 / 13, abs=1e-12)
    assert ME.proportional_devig(1.90, 1.90) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("a,b", [(1.80, 2.10), (1.25, 4.00), (1.90, 1.90), (3.10, 1.38)])
def test_power_devig_matches_hand_computation(a, b):
    p = ME.power_devig(a, b)
    assert p == pytest.approx(_power_newton(a, b), abs=1e-9)
    # the two fair probabilities sum to one with the SAME exponent
    k = math.log(p) / math.log(1 / a)
    assert p + (1 / b) ** k == pytest.approx(1.0, abs=1e-9)


def test_power_devig_shades_the_longshot_more_than_proportional():
    # favourite-longshot: Power takes more margin off the long side than proportional does
    assert ME.power_devig(1.25, 4.0) > ME.proportional_devig(1.25, 4.0)


def test_engine_records_both_devig_methods_and_uses_the_configured_one():
    P = ME.load_params()
    e_pow = ME.evaluate(_cand(), {**P, "devig_method": "power"})
    e_pro = ME.evaluate(_cand(), {**P, "devig_method": "proportional"})
    assert e_pow["devig_method"] == "power" and e_pro["devig_method"] == "proportional"
    for e in (e_pow, e_pro):
        assert not np.isnan(e["p_market_fair_power"]) and not np.isnan(e["p_market_fair_proportional"])
    side_a = e_pow["side"] == "OVER"
    med_pow = float(np.median([ME.power_devig(*x) for x in ((1.95, 1.90), (1.97, 1.88), (1.93, 1.92))]))
    assert e_pow["p_market_fair"] == pytest.approx(med_pow if side_a else 1 - med_pow, abs=1e-6)


def test_bad_devig_method_is_rejected(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"devig_method": "multiplicative"}), encoding="utf-8")
    with pytest.raises(ValueError):
        ME.load_params(f)


# ── the OU25 edge is not p_model - 1/odds ─────────────────────────────────────────────────────
def test_new_engine_ou25_value_is_not_model_minus_implied():
    """Legacy: edge = p_model - 1/odds tiers a 0.20 disagreement as SNIPER. The new engine anchors on
    the de-vigged two-sided consensus; with no residual evidence (n_eff=0) its lower bound can never
    exceed the market, whatever the model says."""
    P = ME.load_params()
    for p in np.linspace(0.05, 0.95, 37):
        e = ME.evaluate(_cand(p=float(p)), P)
        legacy_edge = max(p - 1 / 1.95, (1 - p) - 1 / 1.95)
        assert e["ev_point"] != pytest.approx(legacy_edge * 1.95, abs=1e-6) or abs(legacy_edge) < 0.02
        assert e["p_lower_bound"] <= e["p_market_fair"] + 1e-12, p
        assert e["p_decision"] != pytest.approx(p if e["side"] == "OVER" else 1 - p, abs=1e-6) \
            or abs(p - 0.5) < 0.03


def test_legacy_sniper_sized_disagreement_is_not_qualified():
    e = ME.evaluate(_cand(p=0.72))                     # legacy edge 0.72 - 1/1.95 = 0.207 -> SNIPER
    assert e["decision_state"] in ("NO_BET", "OBSERVE")


# ── one-sided prices ──────────────────────────────────────────────────────────────────────────
def test_one_sided_never_becomes_a_fair_prob():
    assert ME.proportional_devig(1.9, None) is None and ME.power_devig(1.9, None) is None
    e = ME.evaluate(_cand(quotes=[_q("b1", 1.9, None), _q("b2", 1.92, None)]))
    assert (e["decision_state"], e["decision_reason"]) == ("OBSERVE", "ONE_SIDED_PRICE")
    assert np.isnan(e["p_market_fair"]) and np.isnan(e["p_decision"]) and np.isnan(e["ev_lower_bound"])
    # a one-sided BOARD price (e.g. odds_over15 with no under) is not a market either
    e = ME.evaluate(_cand(family="OU15", quotes=[], row=(1.35, None)))
    assert (e["decision_state"], e["decision_reason"]) == ("OBSERVE", "ONE_SIDED_PRICE")
    assert e["market_source"] == "ONE_SIDED"


def test_pairs_only_within_the_same_book():
    q = pd.DataFrame([
        {"snapshot_ts": "2026-10-25T11:00:00Z", "match_date": "2026-10-26", "match": "A vs B",
         "bookmaker": "bk1", "market": "OU25", "side": "OVER", "odds": 2.30},
        {"snapshot_ts": "2026-10-25T11:00:00Z", "match_date": "2026-10-26", "match": "A vs B",
         "bookmaker": "bk2", "market": "OU25", "side": "UNDER", "odds": 2.10},
    ])
    q["_ts"] = pd.to_datetime(q["snapshot_ts"], utc=True)
    pairs = S.book_pairs(q, NOW)[("2026-10-26", "A vs B", "OU25")]
    assert all((x.odds_a is None) != (x.odds_b is None) for x in pairs)   # never stitched together
    e = ME.evaluate(_cand(quotes=pairs))
    assert e["decision_reason"] == "ONE_SIDED_PRICE"


def test_change_log_is_carried_forward_per_book():
    """book_odds_snapshots stores changes only: a later OVER-only row must pair with the earlier
    UNDER of the same book (LOCF), stamped at the later change."""
    q = pd.DataFrame([
        {"snapshot_ts": "2026-10-25T08:00:00Z", "match_date": "2026-10-26", "match": "A vs B",
         "bookmaker": "bk1", "market": "OU25", "side": s, "odds": o} for s, o in (("OVER", 1.9), ("UNDER", 1.95))
    ] + [{"snapshot_ts": "2026-10-25T10:00:00Z", "match_date": "2026-10-26", "match": "A vs B",
          "bookmaker": "bk1", "market": "OU25", "side": "OVER", "odds": 1.85}])
    q["_ts"] = pd.to_datetime(q["snapshot_ts"], utc=True)
    (x,) = S.book_pairs(q, NOW)[("2026-10-26", "A vs B", "OU25")]
    assert (x.odds_a, x.odds_b) == (1.85, 1.95)
    assert x.ts == pd.Timestamp("2026-10-25T10:00:00Z")


# ── fail closed ───────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kw,reason", [
    ({"p": None}, "MISSING_MODEL"),
    ({"p": float("nan")}, "MISSING_MODEL"),
    ({"kickoff": None}, "MISSING_KICKOFF"),
    ({"kickoff": NOW - pd.Timedelta(minutes=1)}, "POST_KICKOFF"),
    ({"quotes": [], "row": (None, None)}, "NO_PRICE"),
    ({"price_synthetic": True}, "SYNTHETIC_PRICE"),
    ({"track": "unknown"}, "UNKNOWN_TRACK"),
])
def test_missing_or_post_kickoff_fails_closed(kw, reason):
    e = ME.evaluate(_cand(**kw))
    assert e["decision_state"] == "NO_BET"
    assert e["decision_reason"].startswith(reason)


def test_stale_price_fails_closed():
    e = ME.evaluate(_cand(quotes=[_q("b1", 1.95, 1.90, age_min=800), _q("b2", 1.9, 1.95, age_min=900)]))
    assert e["decision_state"] == "NO_BET" and e["decision_reason"].startswith("STALE_PRICE")
    # near kickoff the stale limit tightens
    near = dict(kickoff=NOW + pd.Timedelta(hours=2))
    e = ME.evaluate(_cand(quotes=[_q("b1", 1.95, 1.90, age_min=200)], **near))
    assert e["decision_reason"].startswith("STALE_PRICE")


def test_post_kickoff_and_future_quotes_are_never_used():
    after = ME.BookQuote("b1", 1.95, 1.90, KO + pd.Timedelta(minutes=5))
    future = ME.BookQuote("b2", 1.95, 1.90, NOW + pd.Timedelta(minutes=5))
    e = ME.evaluate(_cand(quotes=[after, future]))
    assert e["decision_state"] == "NO_BET" and e["n_books"] == 0


def test_invalid_overround_book_is_dropped():
    arb = _q("arb", 2.20, 2.20)          # overround -0.09: an arbitrage pair is a data error
    e = ME.evaluate(_cand(quotes=[arb]))
    assert e["decision_state"] == "NO_BET" and e["n_invalid_books"] == 1


# ── states and evidence ───────────────────────────────────────────────────────────────────────
def _beating_price(**kw):
    # consensus ~50/50 at 1.95/1.90-ish, the board offers 2.40 on OVER: genuine price value
    return _cand(p=0.52, row=(2.40, 1.70), **kw)


def test_qualified_is_reachable_on_price_value_with_track_evidence():
    e = ME.evaluate(_beating_price())
    assert e["ev_lower_bound"] > 0 and e["decision_state"] == "QUALIFIED", e


def test_missing_evidence_caps_at_observe():
    for clv in (None, {"n": 10, "mean": 5.0, "median": 1.0}, {"n": 500, "mean": -1.0, "median": 1.0},
                {"n": 500, "mean": float("nan")}, {"n": 910, "mean": 4.69, "median": 0.0}):  # a tail, not a track
        e = ME.evaluate(_beating_price(clv=clv))
        assert e["ev_lower_bound"] > 0
        assert e["decision_state"] in ("NO_BET", "OBSERVE"), clv
        assert "NO_TRACK_EVIDENCE" in e["decision_reason"]


def test_strong_unreachable_without_prospective_evidence(tmp_path):
    assert ME.load_evidence(tmp_path / "absent.json") is None
    retro = {"engine_version": ME.ENGINE_VERSION, "tracks": {"new_format|OU25": {
        "evidence_type": "retrospective", "n": 5000, "ci_lo": 0.5}}}
    other = {"engine_version": "legacy_raw_edge", "tracks": {"new_format|OU25": {
        "evidence_type": "prospective", "n": 5000, "ci_lo": 0.5}}}
    thin = {"engine_version": ME.ENGINE_VERSION, "tracks": {"new_format|OU25": {
        "evidence_type": "prospective", "n": 20, "ci_lo": 0.5}}}
    ci0 = {"engine_version": ME.ENGINE_VERSION, "tracks": {"new_format|OU25": {
        "evidence_type": "prospective", "n": 5000, "ci_lo": -0.1}}}
    for ev in (None, retro, other, thin, ci0):
        assert ME.evaluate(_beating_price(evidence=ev))["decision_state"] != "STRONG", ev
    good = {"engine_version": ME.ENGINE_VERSION, "tracks": {"new_format|OU25": {
        "evidence_type": "prospective", "n": 300, "ci_lo": 0.4}}}
    assert ME.evaluate(_beating_price(evidence=good))["decision_state"] == "STRONG"


def test_committed_estate_has_no_strong_evidence_today():
    P = ME.load_params()
    assert ME.load_evidence(config.BASE_DIR / P["evidence_path"]) is None


def test_extreme_disagreement_alone_never_yields_strong_or_upgrades():
    good = {"engine_version": ME.ENGINE_VERSION, "tracks": {f"{t}|OU25": {
        "evidence_type": "prospective", "n": 300, "ci_lo": 0.4} for t in ME.TRACKS}}
    P = {**ME.load_params(), "residual_n_eff": {k: 10_000 for k in ME.DEFAULTS["residual_n_eff"]}}
    grid = np.linspace(0.30, 0.995, 140)
    for ev in (None, good):
        ranks, evs = [], []
        for p in grid:
            e = ME.evaluate(_cand(p=float(p), row=(1.95, 1.90), evidence=ev), P)
            if ev is None:
                assert e["decision_state"] != "STRONG"
            d = abs(p - e["p_market_fair"]) if e["side"] == "OVER" else abs((1 - p) - e["p_market_fair"])
            if d > P["disagree_hard"]:
                assert e["decision_state"] in ("NO_BET", "OBSERVE"), (p, e["decision_state"])
            # past the residual clip AND the soft band, more disagreement only lowers EV_LB
            if e["side"] == "OVER" and p - e["p_market_fair"] > P["disagree_soft"] + 0.02:
                ranks.append(ME.STATE_RANK[e["decision_state"]])
                evs.append(e["ev_lower_bound"])
        assert all(b <= a + 1e-12 for a, b in zip(evs, evs[1:]))
        assert all(b <= a for a, b in zip(ranks, ranks[1:]))


def test_display_map_is_presentation_only():
    assert [ME.display_tier(s) for s in ME.STATES] == ["AVOID", "VALUABLE", "MARKSMAN", "SNIPER"]
    e = ME.evaluate(_beating_price())
    assert e["decision_state"] in ME.STATES          # the engine never emits a legacy tier name


# ── market / track blocks ─────────────────────────────────────────────────────────────────────
def test_over35_is_always_no_bet_for_staking():
    e = ME.evaluate(_beating_price(family="OU35"))
    assert e["decision_state"] == "NO_BET" and e["decision_reason"].startswith("MARKET_BLOCKED(OU35)")


def test_props_are_blocked():
    e = ME.evaluate(_cand(family="player_goals"))
    assert e["decision_state"] == "NO_BET" and e["decision_reason"].startswith("UNSUPPORTED_MARKET")
    assert load_policy()["tracks"]["player_props"]["state"] == "BLOCKED_FROM_REAL"


def test_policy_track_states_unchanged():
    t = load_policy()["tracks"]
    for k in ("standard|OU25_OVER", "standard|OU25_UNDER"):        # O/U 2.5 per side since 2026-10-26
        assert t[k]["state"] == "PAPER" and float(t[k]["base_fraction"]) == 0
    assert t["standard|OVER35"]["state"] == "BLOCKED" and t["new_format|OVER35"]["state"] == "BLOCKED"


def test_standard_ou25_stays_paper_even_when_engine_qualifies():
    sel = pd.DataFrame([{"date": "2026-10-26", "fixture": "2026-10-26|A|B", "league": "Championship",
                         "model_type": "standard", "market": "OU25", "side": "OVER",
                         "signal_tier": "SNIPER", "odds": 2.0}])
    plan = size_day(sel, load_policy(), bankroll=100.0, peak=100.0)
    assert float(plan["final_fraction"].iloc[0]) == 0.0
    assert plan["reason"].iloc[0] == "NO_REAL_STAKE_PAPER"


def test_residual_weights_are_track_and_family_specific_parameters():
    P = ME.load_params()
    keys = set(P["residual_weight"])
    assert {f"{t}|{f}" for t in ME.TRACKS for f in ME.SIDES} <= keys
    assert P["status"] == "UNVALIDATED_DEFAULTS"
    assert all(0 <= w <= 0.15 for w in P["residual_weight"].values())      # conservative
    w_std = {**P, "residual_weight": {**P["residual_weight"], "standard|OU25": 0.0}}
    a = ME.evaluate(_cand(p=0.60, track="standard"), w_std)
    b = ME.evaluate(_cand(p=0.60, track="new_format"), w_std)
    assert a["market_residual_adjusted"] == 0.0 and b["market_residual_adjusted"] != 0.0


# ── synthetic side-market prices ──────────────────────────────────────────────────────────────
def test_synthetic_o15_prices_are_excluded():
    df = pd.DataFrame({"odds_over15": [1.40, 1.40, 1.33], "price_synthetic": [True, False, False]})
    assert ME.exclude_synthetic_prices(df, "OU15", "odds_over15")["odds_over15"].tolist() == [1.40, 1.33]
    no_flag = df.drop(columns="price_synthetic")                 # cannot tell -> fail closed
    assert ME.exclude_synthetic_prices(no_flag, "OU15", "odds_over15")["odds_over15"].tolist() == [1.33]
    assert ME.SYNTHETIC_DEFAULT_ODDS["OU15"] == pytest.approx(1.40)
    from src import backtest
    assert backtest._DEFAULT_ODDS["over15"] == ME.SYNTHETIC_DEFAULT_ODDS["OU15"]   # stay in sync
    assert backtest._DEFAULT_ODDS["btts"] == ME.SYNTHETIC_DEFAULT_ODDS["BTTS"]
    assert backtest._DEFAULT_ODDS["over35"] == ME.SYNTHETIC_DEFAULT_ODDS["OU35"]


def test_committed_o15_backtest_rows_with_default_price_are_dropped():
    f = config.OUTPUT_DIR / "backtest_results_over15.csv"
    if not f.exists():
        pytest.skip("backtest_results_over15.csv not committed in this checkout")
    d = pd.read_csv(f, usecols=["odds_over15", "price_synthetic"])
    kept = ME.exclude_synthetic_prices(d, "OU15", "odds_over15")
    assert not kept["price_synthetic"].astype(str).str.lower().eq("true").any()
    assert len(kept) < len(d)


def test_synthetic_book_quote_is_never_a_market():
    q = ME.BookQuote("b1", 1.40, 2.80, NOW - pd.Timedelta(minutes=5), synthetic=True)
    e = ME.evaluate(_cand(family="OU15", quotes=[q], row=(1.40, None)))
    assert e["n_books"] == 0 and e["decision_state"] in ("NO_BET", "OBSERVE")
    assert np.isnan(e["p_market_fair"])


# ── shadow log ────────────────────────────────────────────────────────────────────────────────
def _board():
    rows = []
    for i, (lg, mt) in enumerate((("Argentina Primera Division", "new_format"), ("Championship", "standard"))):
        rows.append({"league": lg, "date": "2026-10-26", "kickoff_utc": "2026-10-26 18:00:00",
                     "home_team": f"H{i}", "away_team": f"A{i}", "model_type": mt,
                     "_oa_event_id": f"ev{i}", "p_over25": 0.55, "odds_over25": 1.95,
                     "odds_under25": 1.90, "best_edge": 0.04, "best_side": "OVER",
                     "signal_tier": "VALUABLE", "p_btts": 0.60, "odds_btts": 1.80, "odds_btts_no": 2.0,
                     "p_over15": 0.75, "odds_over15": 1.30, "p_over35": 0.30, "odds_over35": 3.2,
                     "model_version": "standard_20261018T080423Z_x"})
    rows.append({**rows[0], "home_team": "Old", "away_team": "Gone", "_oa_event_id": "ev_old",
                 "kickoff_utc": "2026-10-25 11:00:00"})                 # already kicked off
    return pd.DataFrame(rows)


def _quotes(odds_shift=0.0):
    out = []
    for i in range(2):
        for bk in ("b1", "b2", "b3"):
            for mk, a, b in (("OU25", ("OVER", 1.95), ("UNDER", 1.90)), ("BTTS", ("YES", 1.80), ("NO", 2.0))):
                for side, o in (a, b):
                    out.append({"snapshot_ts": "2026-10-25T11:55:00Z", "match_date": "2026-10-26",
                                "match": f"H{i} vs A{i}", "bookmaker": bk, "market": mk,
                                "side": side, "odds": o + odds_shift})
    return out


def _run_shadow(tmp_path, board=None, shift=0.0):
    board = _board() if board is None else board
    side = pipeline._generate_side_bets(board, config.SIDE_MARKETS)
    return S.run(board, None, side, historical=None, now=NOW, shadow_path=tmp_path / "shadow.csv",
                 health_path=tmp_path / "health.json", snapshots_path=tmp_path / "none.csv",
                 live_rows=_quotes(shift))


def test_shadow_rows_are_first_sight_and_byte_identical_across_reruns(tmp_path):
    _run_shadow(tmp_path)
    first = (tmp_path / "shadow.csv").read_bytes()
    _run_shadow(tmp_path, shift=0.15)                 # prices moved: the logged opinion must not
    assert (tmp_path / "shadow.csv").read_bytes() == first
    b = _board()
    b.loc[len(b)] = {**b.iloc[0].to_dict(), "home_team": "New", "away_team": "Club", "_oa_event_id": "ev9"}
    _run_shadow(tmp_path, board=b)
    after = (tmp_path / "shadow.csv").read_bytes()
    assert after.startswith(first) and len(after) > len(first)       # appended, never rewritten
    d = pd.read_csv(tmp_path / "shadow.csv")
    assert not d.duplicated(S.KEY).any()


def test_shadow_logs_pre_kickoff_only_and_every_family(tmp_path):
    _run_shadow(tmp_path)
    d = pd.read_csv(tmp_path / "shadow.csv")
    assert "ev_old" not in set(d["fixture_id"])
    assert set(d["market"]) == {"OU25", "BTTS", "OU15", "OU35"}
    assert (d["selection_engine_version"] == ME.ENGINE_VERSION).all()
    assert (d["legacy_engine_version"] == ME.LEGACY_ENGINE_VERSION).all()
    for c in ["p_model_raw", "p_market_fair", "p_decision", "p_lower_bound", "market_residual_raw",
              "market_residual_adjusted", "odds", "market_overround", "ev_point", "ev_lower_bound",
              "effective_sample_size", "uncertainty_penalty", "clv_track_n", "clv_track_mean",
              "clv_track_median", "clv_track_positive_pct", "market_source", "price_age_minutes",
              "decision_reason", "decision_state", "legacy_raw_edge", "legacy_tier", "legacy_side",
              "legacy_threshold", "new_state", "new_side", "new_reason", "prediction_ts", "kickoff_utc",
              "fixture_id", "model_type"]:
        assert c in d.columns, c
    assert (d.loc[d["market"] == "OU35", "new_state"] == "NO_BET").all()
    ou15 = d[d["market"] == "OU15"]
    assert (ou15["new_reason"] == "ONE_SIDED_PRICE").all()          # board O1.5 has no under
    h = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    assert h["generated_at"] == "2026-10-25T12:00:00Z" and h["shadow_only"] is True
    assert h["coverage"]["OU25|new_format"]["two_sided_share"] == 1.0
    assert h["coverage"]["OU15|standard"]["one_sided"] == 1


def test_shadow_legacy_columns_mirror_production(tmp_path):
    board = _board()
    side = pipeline._generate_side_bets(board, config.SIDE_MARKETS)
    _run_shadow(tmp_path)
    d = pd.read_csv(tmp_path / "shadow.csv")
    ou = d[d["market"] == "OU25"].set_index("fixture_id")
    assert (ou["legacy_tier"] == "VALUABLE").all() and (ou["legacy_raw_edge"] == 0.04).all()
    assert len(side)
    for r in side.itertuples(index=False):
        if r.home_team == "Old":                                     # kicked off: not logged
            continue
        fam = {"btts": "BTTS", "over15": "OU15", "over35": "OU35"}[r.market]
        row = d[(d["market"] == fam) & (d["home_team"] == r.home_team)].iloc[0]
        assert row["legacy_raw_edge"] == pytest.approx(r.edge, abs=1e-12)
        assert row["legacy_tier"] == r.signal_tier and row["legacy_side"] == r.side


def test_shadow_never_raises(tmp_path):
    # a board with none of the expected columns: logged nothing, raised nothing
    S.run(pd.DataFrame({"x": [1]}), None, None, now=NOW, shadow_path=tmp_path / "s.csv",
          health_path=tmp_path / "h.json", live_rows=[])
    assert not (tmp_path / "s.csv").exists()
    # no board at all: the failure is swallowed and reported as None
    assert S.run(None, None, None, now=NOW, shadow_path=tmp_path / "s.csv",
                 health_path=tmp_path / "h.json", live_rows=[]) is None


# ── stake plan: shadow columns never change sizing ────────────────────────────────────────────
def test_stake_plan_sizing_identical_with_shadow_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "load_league_ratings", lambda *a, **k: {})
    board = _board().iloc[:2]
    side = pipeline._generate_side_bets(board, config.SIDE_MARKETS)
    bets = board.assign(bet="OVER", signal_tier="MARKSMAN")
    per_side = _run_shadow(tmp_path)
    assert per_side is not None and len(per_side)

    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan_a.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    sp.write(bets, side)
    head_before = (tmp_path / "audit.csv").read_text(encoding="utf-8").splitlines()[0]
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan_b.csv")
    sp.write(bets, side, shadow=per_side)
    a = pd.read_csv(tmp_path / "plan_a.csv")
    b = pd.read_csv(tmp_path / "plan_b.csv")
    orig = [c for c in a.columns if c not in sp.SHADOW_PLAN_COLS]
    assert orig[-2:] == ["bankroll_basis", "generated_at"]           # the pre-shadow schema
    assert list(b.columns[: len(orig)]) == orig                      # same columns, same order, first
    same = [c for c in orig if c != "generated_at"]
    pd.testing.assert_frame_equal(a[same], b[same])                  # every original value identical
    assert a["decision_state"].isna().all()                          # no shadow -> NaN, never invented
    assert list(b.columns[-len(sp.SHADOW_PLAN_COLS):]) == list(sp.SHADOW_PLAN_COLS)
    assert (b["selection_engine_version"] == "legacy_raw_edge").all()
    assert b["engine_version"].eq(ME.ENGINE_VERSION).all()           # every pre-kickoff tip joined
    assert b["decision_state"].isin(ME.STATES).all()
    assert (tmp_path / "audit.csv").read_text(encoding="utf-8").splitlines()[0] == head_before


def test_stake_planner_still_receives_canonical_contract():
    board = _board().iloc[:2]
    board["p_btts"] = 0.80                                           # force a BTTS tip
    side = pipeline._generate_side_bets(board, config.SIDE_MARKETS)
    sel = sp.selections(None, side)
    assert len(sel) and set(sel["model_type"]) <= {"standard", "new_format"}
    assert sel["odds"].notna().all() and set(sel["side"]) <= {"YES", "OVER"}
    assert set(sel["market"]) <= {"BTTS", "OVER15", "OVER35"}


def test_pipeline_wires_shadow_beside_legacy_not_into_it():
    import inspect
    src = inspect.getsource(pipeline.mode_predict)
    # the shadow sees the legacy tips BEFORE the direction guard (2026-10-25): research logs everything
    assert "_EES.run(preds, _bets_all, _side_all" in src
    assert "_bets_all, _side_all = bets, side_bets" in src
    assert src.index("bets = generate_bets(preds)") < src.index("_EES.run(")
    assert "shadow=_edge_shadow" in src
    from src import betting
    assert "market_edge" not in inspect.getsource(betting)          # legacy path untouched
