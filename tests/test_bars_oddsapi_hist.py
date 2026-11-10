"""Priced universes from the OddsAPI history (src/bars_opt/oddsapi_prices.py), the flagged-off study entry
(src/bars_opt/hist_study.py) and the search ledger. Synthetic frames only: no store, no network, no price
of the provider is read, nothing under output/ is written."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import league_bars as LB  # noqa: E402
from src.bars_opt import hist_study as H  # noqa: E402
from src.bars_opt import oddsapi_prices as P  # noqa: E402
from src.bars_opt import study as S  # noqa: E402

LG, SK, D = "USA MLS", "soccer_usa_mls", "2024-06-01"
LG2, SK2 = "Japan J-League", "soccer_japan_j_league"
KO = "2024-06-02T00:30:00Z"
FX = f"{D}|{LG}|Austin|Dallas"


def _q(book, market, name, price, point=None, ts="2024-06-02T00:25:00Z", ko=KO, ev="e1", league=LG, sk=SK,
       home="Austin FC", away="FC Dallas", ld=D):
    return {"unit_key": f"u-{ev}-{ts}", "endpoint": "event_odds", "tier": "B1", "snapshot_label": "close",
            "event_id": ev, "sport_key": sk, "league": league, "league_date": ld, "requested_at": ts,
            "snapshot_ts": ts, "commence_time": ko, "home_team": home, "away_team": away, "bookmaker": book,
            "market": market, "outcome_name": name, "outcome_description": None, "point": point, "price": price}


def _ou(book, point, over, under, **kw):
    return [_q(book, "alternate_totals", "Over", over, point, **kw), _q(book, "alternate_totals", "Under", under, point, **kw)]


def _universe(own=None, league=LG, date=D, home="Austin", away="Dallas"):
    rows = []
    for m in ("over15", "over25", "over35", "btts"):
        r = {"stage": "walk_forward", "fold": 0, "track": "new_format", "market": m, "candidate": "champion_recipe",
             "row_id": f"{date}|{league}|{home}|{away}", "date": pd.Timestamp(date), "league": league, "y": 1.0,
             "p": 0.5, "deployable": True, "home": home, "away": away, "yes": np.nan, "no": np.nan,
             "yes_c": np.nan, "no_c": np.nan}
        if own and m in own:
            r["yes"], r["no"] = own[m]
            r["yes_c"], r["no_c"] = own[m]
        rows.append(r)
    return pd.DataFrame(rows)


def _price(universe, quotes):
    return P.price_frames(universe, pd.DataFrame(quotes), None)


def _row(priced, market):
    return priced[priced["market"] == market].iloc[0]


# ── exact line, per side, one book ───────────────────────────────────────────────────────────────

def test_quarter_and_other_lines_never_stand_in_for_2_5():
    q = _ou("pinnacle", 2.25, 1.80, 2.05) + _ou("pinnacle", 2.75, 2.10, 1.75) + _ou("pinnacle", 3.0, 2.4, 1.6)
    r = _price(_universe(), q)
    assert pd.isna(_row(r["priced"], "over25")["price_source"])
    assert np.isnan(_row(r["priced"], "over25")["yes"])
    assert r["stats"]["pairs"]["other_line_total_quotes"] == 6


def test_exact_line_both_sides_from_one_book_and_one_snapshot():
    q = _ou("pinnacle", 2.5, 1.90, 1.95) + _ou("pinnacle", 1.5, 1.30, 3.50)
    q += [_q("pinnacle", "alternate_totals", "Over", 2.6, 3.5)]                     # 3.5 Over without Under
    q += [_q("suprabets", "alternate_totals", "Under", 1.4, 3.5)]                   # other book's Under
    p = _price(_universe(), q)["priced"]
    o25 = _row(p, "over25")
    assert (o25["yes"], o25["no"], o25["price_source"], o25["oh_book"]) == (1.90, 1.95, "oddsapi_hist", "pinnacle")
    assert _row(p, "over15")["yes"] == 1.30
    assert pd.isna(_row(p, "over35")["price_source"])                              # never a mixed pair


def test_same_rule_as_study_one_books_pair_never_best_or_consensus():
    q = _ou("pinnacle", 2.5, 1.90, 1.95) + _ou("suprabets", 2.5, 2.30, 2.20) + _ou("coolbet", 2.5, 2.0, 2.0)
    o25 = _row(_price(_universe(), q)["priced"], "over25")
    assert (o25["yes"], o25["no"]) == (1.90, 1.95)                                  # not max, not median


def test_a_book_outside_the_study_order_never_prices():
    q = _ou("suprabets", 2.5, 1.90, 1.95)
    assert pd.isna(_row(_price(_universe(), q)["priced"], "over25")["price_source"])
    assert P.BOOK_ORDER == ("bet365", "pinnacle")


def test_btts_per_side():
    q = [_q("pinnacle", "btts", "Yes", 1.8), _q("pinnacle", "btts", "No", 2.0)]
    b = _row(_price(_universe(), q)["priced"], "btts")
    assert (b["yes"], b["no"]) == (1.8, 2.0)


def test_ambiguous_quote_is_dropped():
    q = _ou("pinnacle", 2.5, 1.90, 1.95)
    q.append(dict(_q("pinnacle", "totals", "Over", 1.99, 2.5)))                     # same snapshot, other price
    r = _price(_universe(), q)
    assert pd.isna(_row(r["priced"], "over25")["price_source"])
    assert r["stats"]["pairs"]["ambiguous_groups"] == 1


# ── timing: the study's own bet-time rule ────────────────────────────────────────────────────────

def test_earliest_pre_kickoff_snapshot_is_bet_time_latest_is_closing():
    q = (_ou("pinnacle", 2.5, 2.00, 1.85, ts="2024-06-01T10:00:00Z")
         + _ou("pinnacle", 2.5, 1.90, 1.95, ts="2024-06-02T00:25:00Z")
         + _ou("pinnacle", 2.5, 1.50, 2.60, ts="2024-06-02T00:40:00Z")            # after kickoff: never
         + _ou("pinnacle", 2.5, 2.20, 1.70, ts="2024-05-30T23:00:00Z"))           # before match day - 1
    o25 = _row(_price(_universe(), q)["priced"], "over25")
    assert (o25["yes"], o25["no"]) == (2.00, 1.85)
    assert (o25["yes_c"], o25["no_c"]) == (1.90, 1.95)
    assert o25["close_source"] == "oddsapi_hist"


def test_snapshot_after_kickoff_never_prices():
    q = _ou("pinnacle", 2.5, 1.5, 2.6, ts="2024-06-02T00:31:00Z")
    assert pd.isna(_row(_price(_universe(), q)["priced"], "over25")["price_source"])


# ── own always wins ──────────────────────────────────────────────────────────────────────────────

def test_own_price_always_wins_and_overlap_is_reported():
    q = _ou("pinnacle", 2.5, 1.90, 1.95) + _ou("pinnacle", 1.5, 1.30, 3.50)
    r = _price(_universe(own={"over25": (2.0, 1.8)}), q)
    o25, o15 = _row(r["priced"], "over25"), _row(r["priced"], "over15")
    assert (o25["yes"], o25["no"], o25["price_source"], o25["close_source"]) == (2.0, 1.8, "own", "own")
    assert pd.isna(o25["oh_event_id"])                                              # no history metadata
    assert o15["price_source"] == "oddsapi_hist"
    assert len(r["overlap"]) == 1 and r["overlap"].iloc[0]["h_yes"] == 1.90
    ag = P.agreement(r["overlap"])
    assert ag[0]["rows_both"] == 1 and ag[0]["mad_odds_yes"] == pytest.approx(0.10)


# ── never across leagues, never another fixture (fixture id + kickoff) ──────────────────────────

def test_price_join_never_crosses_leagues():
    q = _ou("pinnacle", 2.5, 1.90, 1.95)                                            # an MLS event
    r = _price(_universe(league=LG2), q)                                            # same clubs, other league
    assert r["priced"]["price_source"].isna().all()


def test_two_leagues_same_day_each_gets_its_own_event():
    q = _ou("pinnacle", 2.5, 1.90, 1.95) + _ou("pinnacle", 2.5, 2.50, 1.55, ev="e2", league=LG2, sk=SK2,
                                                 ko="2024-06-01T10:00:00Z", ts="2024-06-01T09:55:00Z")
    u = pd.concat([_universe(), _universe(league=LG2)], ignore_index=True)
    p = _price(u, q)["priced"]
    mls = p[(p["league"] == LG) & (p["market"] == "over25")].iloc[0]
    jp = p[(p["league"] == LG2) & (p["market"] == "over25")].iloc[0]
    assert (mls["yes"], mls["oh_event_id"]) == (1.90, "e1")
    assert (jp["yes"], jp["oh_event_id"]) == (2.50, "e2")


def test_kickoff_outside_the_fixture_day_is_never_used():
    ko = "2024-06-04T00:30:00Z"                                                     # two days after our date
    q = _ou("pinnacle", 2.5, 1.90, 1.95, ko=ko, ts="2024-06-04T00:25:00Z", ld="2024-06-03")
    r = _price(_universe(), q)
    assert r["priced"]["price_source"].isna().all()


def _verified():
    r = _price(_universe(), _ou("pinnacle", 2.5, 1.90, 1.95))
    return r["priced"], {FX: "e1"}, {"e1": SK}


def test_verify_join_passes_on_the_adapter_mapping():
    priced, fmap, sport = _verified()
    out = P.verify_join(priced, fmap, sport)
    assert out["rows_checked"] == 1 and sum(out["failures"].values()) == 0


@pytest.mark.parametrize("mutate,check", [
    (lambda p, f, s: f.update({FX: "e9"}), "fixture_event"),                        # price moved to another event
    (lambda p, f, s: f.update({"2024-06-01|USA MLS|Other|Club": "e1"}), "event_to_many_fixtures"),
    (lambda p, f, s: p.__setitem__("h_league", LG2), "league"),                      # price from another league
    (lambda p, f, s: s.update({"e1": SK2}), "sport_key"),
    (lambda p, f, s: p.__setitem__("oh_commence_time", pd.Timestamp("2024-06-05T00:30:00Z")), "kickoff_window"),
    (lambda p, f, s: p.__setitem__("oh_snapshot_ts", pd.Timestamp("2024-06-02T00:45:00Z")), "snapshot_after_kickoff"),
])
def test_verify_join_refuses_any_cross_league_or_moved_price(mutate, check):
    priced, fmap, sport = _verified()
    mutate(priced, fmap, sport)
    with pytest.raises(P.JoinIntegrityError, match=check):
        P.verify_join(priced, fmap, sport)


# ── outputs stay local; coverage counts ─────────────────────────────────────────────────────────

def test_outputs_inside_the_public_repo_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        P.assert_outside_repo(ROOT / "output" / "bars_opt" / "priced")
    assert P.assert_outside_repo(tmp_path) == tmp_path.resolve()


def test_coverage_counts_before_and_after_by_source():
    q = _ou("pinnacle", 2.5, 1.90, 1.95) + _ou("pinnacle", 1.5, 1.30, 3.50)
    p = _price(_universe(own={"over25": (2.0, 1.8)}), q)["priced"]
    rows, fams = P.coverage(p)
    o15 = next(r for r in rows if r["market"] == "over15")
    o25 = next(r for r in rows if r["market"] == "over25")
    assert (o15["priced_before"], o15["priced_after"], o15["after_oddsapi_hist"]) == (0, 1, 1)
    assert (o25["priced_before"], o25["priced_after"], o25["after_own"]) == (1, 1, 1)
    assert {f["mside"] for f in fams} >= {"over25_over", "over25_under", "over15_over", "btts_yes"}


# ── multiple testing: every configuration counted ───────────────────────────────────────────────

def test_holm_counts_hypotheses_from_elsewhere():
    p = np.array([0.001, 0.02, np.nan])
    assert LB.holm(p).tolist()[:2] == pytest.approx([0.002, 0.02])
    assert LB.holm(p, extra=98).tolist()[:2] == pytest.approx([0.1, 1.0])
    with pytest.raises(ValueError):
        LB.holm(p, extra=-1)


def test_verdicts_family_includes_prior_hypotheses():
    cells = [{"verdict": None, "final": {"bets": 300, "p_one_sided": 0.0001, "ci95": [0.05, 0.3], "roi": 0.2,
                                         "clv": 0.01}}]
    S.verdicts([dict(c, final=dict(c["final"])) for c in cells])
    one = [dict(c, final=dict(c["final"])) for c in cells]
    S.verdicts(one)
    assert one[0]["verdict"] == "PROVEN"
    many = [dict(c, final=dict(c["final"])) for c in cells]
    S.verdicts(many, prior_hypotheses=27_000)
    assert many[0]["verdict"] != "PROVEN" and many[0]["family_size"] == 27_001


def test_fast_bootstrap_p_matches_the_study_bootstrap():
    rng = np.random.default_rng(1)
    days = np.repeat([f"2026-09-{d:02d}" for d in range(1, 31)], 8)
    pnl = np.where(rng.random(len(days)) < 0.5, 0.95, -1.0)
    a = LB.one_sided_p(pnl, days, n_boot=4000, seed=3)
    b = S.fast_one_sided_p(pnl, days, n_boot=4000, seed=3)
    assert abs(a - b) < 0.05


def test_floor_p_is_redrawn_at_the_resolution_the_family_needs():
    days = np.repeat(pd.date_range("2026-09-01", periods=40), 6)
    fb = pd.DataFrame({"league": "X", "date": days, "pnl": np.full(len(days), 0.9)})
    cells = [{"track": "t", "mside": "m", "league": "X", "verdict": None,
              "final": {"bets": len(fb), "p_one_sided": round(1 / (S.N_BOOT + 1), 4)}}]
    done = S.refine_floor_p(cells, [{"track": "t", "mside": "m", "final_bets": fb}], hypotheses=100)
    assert done and done[0]["resamples"] == int(np.ceil(S.P_RESOLUTION_HITS * 100 / S.HOLM_ALPHA))
    assert cells[0]["final"]["p_one_sided"] == pytest.approx(1 / (done[0]["resamples"] + 1))


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────

def _ledger(tmp_path, runs):
    d = tmp_path / "store" / "ledger" / "runs"
    d.mkdir(parents=True)
    for i, r in enumerate(runs):
        (d / f"r{i}.json").write_text(json.dumps(r), encoding="utf-8")
    return tmp_path / "store"


def test_b1_complete_only_from_the_run_ledger(tmp_path):
    st = _ledger(tmp_path, [{"run_id": "a", "exit_code": 10, "stats": {"B1": {}}, "credits_by_tier": {"B1": 5}}])
    assert P.b1_complete(st)[0] is False
    st2 = _ledger(tmp_path / "b", [{"run_id": "b", "exit_code": 10, "stats": {"B1": {}, "B2": {}}}])
    assert P.b1_complete(st2)[0] is True
    st3 = _ledger(tmp_path / "c", [{"run_id": "c", "exit_code": 0, "stats": {"B1": {}}}])
    assert P.b1_complete(st3)[0] is True


def test_the_run_flag_is_open_only_with_the_owners_words_and_refuses_when_off(tmp_path):
    flag = json.loads(H.FLAG_FILE.read_text(encoding="utf-8"))
    assert flag["run_enabled"] is not True or str(flag["owner_ok"] or "").strip()   # never open without the owner
    off = tmp_path / "off.json"
    off.write_text(json.dumps({"run_enabled": False, "owner_ok": None}), encoding="utf-8")
    st = _ledger(tmp_path, [{"run_id": "c", "exit_code": 0, "stats": {"B1": {}}}])
    ok, why = H.gate(st, off)
    assert not ok and any("run_enabled" in w for w in why)
    st2 = _ledger(tmp_path / "x", [{"run_id": "a", "exit_code": 10, "stats": {"B1": {}}}])   # B1 unfinished
    with pytest.raises(SystemExit, match="FLAGGED OFF"):
        H.run(st2, tmp_path / "priced", tmp_path / "study")


def test_gate_opens_only_with_owner_ok_and_every_pinned_tier_complete(tmp_path):
    f = tmp_path / "flag.json"
    f.write_text(json.dumps({"run_enabled": True, "owner_ok": "owner 2026-12-01: run"}), encoding="utf-8")
    st = _ledger(tmp_path, [{"run_id": "a", "exit_code": 10, "stats": {"B1": {}, "B2": {}}}])   # B2 unfinished
    ok, why = H.gate(st, f)
    assert not ok and any("not complete" in w and "B2" in w for w in why)
    st2 = _ledger(tmp_path / "b", [{"run_id": "b", "exit_code": 10, "stats": {"B1": {}, "B2": {}, "B4": {}}}])
    assert H.gate(st2, f) == (True, [])


def test_a_later_tier_alone_never_proves_an_earlier_one(tmp_path):
    st = _ledger(tmp_path, [{"run_id": "a", "exit_code": 10, "stats": {"B4": {}}}])
    assert P.tier_complete(st, "B2")[0] is False and P.b1_complete(st)[0] is False


def test_prepare_prices_only_from_the_pinned_tiers():
    q = _ou("pinnacle", 2.5, 1.90, 1.95)
    odds = pd.DataFrame(q).assign(tier="C3")
    r = P.price_frames(_universe(), odds[odds["tier"].isin(["B1", "B2"])], None)
    assert r["priced"]["price_source"].isna().all()


def test_search_ledger_counts_the_earlier_runs():
    from src.bars_opt import search_ledger as L
    try:
        d = L.build()
    except L.LedgerError as e:                                     # a checkout without the study history
        pytest.skip(str(e))
    kinds = {r["kind"] for r in d["runs"]}
    if "bars_opt_select" not in kinds:
        pytest.skip("study outputs not in this checkout's history")
    assert d["configs_searched"] >= 540 and d["hypotheses"] == d["configs_searched"] + d["cells_tested"]



def test_two_legged_tie_one_event_never_prices_the_other_leg():
    """A provider event matched to both legs of a tie (home / away swapped, 3 days apart): only the leg whose
    date holds the kickoff can be priced; the other leg keeps no price."""
    leg2 = _universe(date="2024-06-04", home="Dallas", away="Austin")
    u = pd.concat([_universe(), leg2], ignore_index=True)
    fmap = {FX: "e1", "2024-06-04|USA MLS|Dallas|Austin": "e1"}
    odds = pd.DataFrame(_ou("pinnacle", 2.5, 1.90, 1.95))
    clean, st = P.clean_fixture_map(fmap, odds)
    assert clean == {FX: "e1"} and st["map_kickoff_outside_fixture_window"] == 1
    both = {FX: "e1", "2024-06-01|USA MLS|Other|Club": "e1"}            # same day, two fixtures: price none
    clean2, st2 = P.clean_fixture_map(both, odds)
    assert clean2 == {} and st2["map_events_on_several_fixtures"] == 1
    p = _price(u, _ou("pinnacle", 2.5, 1.90, 1.95))["priced"]
    assert p[(p["row_id"] == "2024-06-04|USA MLS|Dallas|Austin")]["price_source"].isna().all()


def test_a_single_snapshot_gives_no_closing_pair():
    """One snapshot is the bet-time price; it is not also a closing price (CLV would be 0 by construction)."""
    o25 = _row(_price(_universe(), _ou("pinnacle", 2.5, 1.90, 1.95))["priced"], "over25")
    assert (o25["yes"], o25["no"]) == (1.90, 1.95)
    assert np.isnan(o25["yes_c"]) and pd.isna(o25["close_source"])
