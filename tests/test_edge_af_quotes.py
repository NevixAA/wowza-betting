"""API-Football book_quotes as a second per-book source for the market_edge_v2 SHADOW (2026-10-25).

Every test names the regression it guards: same-book same-snapshot pairing only, no cross-book or
carried-forward synthetic pair, no post-kickoff / future quote, no line mixing, league-scoped name
resolution that refuses ambiguity, one vote per physical book, per-source health numbers, first-sight
immutability, and that the weekly league panel treats the new two-sided prices as priced.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import config
import pipeline
from src import edge_af_quotes as EAF
from src import edge_engine_shadow as S
from src import market_edge as ME
from src.research import league_weekly as LW

NOW = pd.Timestamp("2026-10-25T12:00:00Z")
KO = NOW + pd.Timedelta(hours=30)
T1 = NOW - pd.Timedelta(minutes=40)
T2 = NOW - pd.Timedelta(minutes=10)
LG = "Championship"


def _z(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(book, market, side, line, odds, ts=T1, fid=9001, home="Lincoln", away="Birmingham",
         ko=KO, league=LG):
    return {"snapshot_ts": _z(ts), "fixture_id": fid, "kickoff_utc": _z(ko),
            "minutes_to_kickoff": round((ko - ts).total_seconds() / 60.0, 1), "ladder_band": "FAR",
            "league": league, "model_type": "standard", "home_team": home, "away_team": away,
            "bookmaker_id": 1, "bookmaker": book, "market": market, "side": side, "line": line,
            "odds": odds, "source": "api_football", "obs_reason": "open"}


def _ou15(book, o=1.30, u=3.40, ts=T1, **kw):
    return [_row(book, "ou", "over", 1.5, o, ts, **kw), _row(book, "ou", "under", 1.5, u, ts, **kw)]


def _board(home="Lincoln City", away="Birmingham City", league=LG, ko=KO, **extra):
    r = {"league": league, "date": str(ko.date()), "kickoff_utc": ko.strftime("%Y-%m-%d %H:%M:%S"),
         "home_team": home, "away_team": away, "model_type": "standard", "_oa_event_id": "evA",
         "p_over25": 0.55, "odds_over25": 1.95, "odds_under25": 1.90, "best_edge": 0.01,
         "best_side": "OVER", "signal_tier": "AVOID", "p_btts": 0.55, "odds_btts": 1.85,
         "odds_btts_no": 1.95, "p_over15": 0.76, "odds_over15": 1.30, "p_over35": 0.28,
         "odds_over35": 3.30, "model_version": "standard_20261018T080423Z_x"}
    r.update(extra)
    return pd.DataFrame([r])


def _pairs(rows, board=None, now=NOW):
    return EAF.af_book_pairs(_board() if board is None else board, now, raw=pd.DataFrame(rows))


KEY15 = (str(KO.date()), "Lincoln City vs Birmingham City", "OU15")


# ── pairing ────────────────────────────────────────────────────────────────────────────────────
def test_same_book_same_snapshot_pairs_and_keeps_its_timestamp():
    out, st = _pairs(_ou15("Bet365") + _ou15("Pinnacle", 1.28, 3.60, ts=T2))
    qs = {q.book: q for q in out[KEY15]}
    assert set(qs) == {"af:bet365", "af:pinnacle"}
    assert (qs["af:bet365"].odds_a, qs["af:bet365"].odds_b, qs["af:bet365"].ts) == (1.30, 3.40, T1)
    assert qs["af:pinnacle"].ts == T2
    assert st["pairs_same_snapshot"] == 2 and st["matched_both_teams"] == 1


def test_no_cross_book_synthetic_pair():
    rows = [_row("Bet365", "ou", "over", 1.5, 1.30), _row("Pinnacle", "ou", "under", 1.5, 3.60)]
    out, st = _pairs(rows)
    qs = out[KEY15]
    assert all((q.odds_a is None) != (q.odds_b is None) for q in qs)       # two one-sided quotes
    e = ME.evaluate(ME.Candidate(family="OU15", track="standard", p_model=0.76, kickoff=KO, now=NOW,
                                 quotes=qs, row_odds_a=1.30, row_odds_b=float("nan")))
    assert e["n_books"] == 0 and e["market_source"] == "ONE_SIDED"
    assert st["pairs_same_snapshot"] == 0


def test_same_book_different_snapshots_is_not_carried_forward():
    # Bet365's over moved at T2 but its under was not re-quoted at T2: the T1 pair is superseded
    rows = _ou15("Bet365") + [_row("Bet365", "ou", "over", 1.5, 1.33, ts=T2)]
    out, st = _pairs(rows)
    (q,) = out[KEY15]
    assert (q.odds_a, q.odds_b, q.ts) == (1.33, None, T2)
    assert st["one_sided_superseded"] == 1 and st["pairs_same_snapshot"] == 0


def test_post_kickoff_and_future_quotes_are_never_used():
    late = KO + pd.Timedelta(minutes=5)
    rows = _ou15("Bet365", ts=late) + _ou15("Pinnacle", ts=NOW + pd.Timedelta(minutes=1))
    out, st = _pairs(rows, now=KO + pd.Timedelta(minutes=10))      # board still lists it
    assert out == {}                                                # board kickoff passed -> not scored
    out, st = _pairs(rows)
    assert KEY15 not in out and st["matched_both_teams"] == 1
    assert st["rows_post_kickoff_or_future"] == 4
    # a quote whose own minutes_to_kickoff says the match had started is dropped too
    r = _ou15("Betano")
    for x in r:
        x["minutes_to_kickoff"] = -2.0
    out, st = _pairs(r)
    assert KEY15 not in out and st["rows_post_kickoff_or_future"] == 2


def test_line_mismatch_and_half_time_rows_are_rejected():
    rows = [_row("Bet365", "ou", "over", 1.5, 1.30), _row("Bet365", "ou", "under", 2.5, 1.90),
            _row("Pinnacle", "ht_ou", "over", 1.5, 3.1), _row("Pinnacle", "ht_ou", "under", 1.5, 1.35),
            _row("Betano", "ou", "over", 2.0, 1.6), _row("Betano", "ou", "under", 2.0, 2.3)]
    out, st = _pairs(rows)
    for fam in ("OU15", "OU25"):
        for q in out.get((KEY15[0], KEY15[1], fam), []):
            assert q.odds_a is None or q.odds_b is None                    # never a 1.5/2.5 pair
    assert st["pairs_same_snapshot"] == 0 and st["rows_other_line_or_market"] == 4
    assert (KEY15[0], KEY15[1], "OU35") not in out


def test_btts_pairs_yes_with_no():
    rows = [_row("Bet365", "btts", "yes", float("nan"), 1.80), _row("Bet365", "btts", "no", float("nan"), 1.95)]
    out, _ = _pairs(rows)
    (q,) = out[(KEY15[0], KEY15[1], "BTTS")]
    assert (q.odds_a, q.odds_b) == (1.80, 1.95)


# ── names ──────────────────────────────────────────────────────────────────────────────────────
def test_name_resolution_is_league_scoped_and_refuses_ambiguity():
    # "Manchester" is a token subset of BOTH API-Football clubs -> resolve() refuses it
    rows = _ou15("Bet365", home="Manchester City", away="Manchester United", fid=1) \
        + _ou15("Bet365", home="Manchester United", away="Manchester City", fid=2,
                ko=KO + pd.Timedelta(days=3))
    out, st = _pairs(rows, board=_board("Manchester", "Manchester"))
    assert out == {} and st["refused_ambiguous"] + st["unmatched_names"] == 1
    # the same clubs in ANOTHER league are never candidates
    out, st = _pairs(_ou15("Bet365", league="League One"))
    assert out == {} and st["unmatched_no_league_quotes"] == 1


def test_one_team_match_needs_exact_kickoff_and_same_side():
    rows = _ou15("Bet365", home="Sheffield Utd", away="Wrexham")
    b = _board("Sheffield Forgotten", "Wrexham AFC")
    out, st = _pairs(rows, board=b)
    assert st["matched_one_team_ko"] == 1 and len(out) == 1
    # kickoff an hour apart -> refused (only the both-teams rule tolerates a kickoff shift)
    out, st = _pairs(_ou15("Bet365", home="Sheffield Utd", away="Wrexham", ko=KO + pd.Timedelta(hours=1)),
                     board=b)
    assert out == {} and st["matched_one_team_ko"] == 0
    # resolved club on the other side -> refused
    out, _ = _pairs(rows, board=_board("Wrexham AFC", "Sheffield Forgotten"))
    assert out == {}


def test_one_af_fixture_never_prices_two_board_fixtures():
    b = pd.concat([_board("Lincoln City", "Birmingham City"),
                   _board("Lincoln City", "Birmingham City", date="2026-10-27")], ignore_index=True)
    out, st = _pairs(_ou15("Bet365"), board=b)
    assert out == {} and st["refused_ambiguous"] == 2


# ── consensus ──────────────────────────────────────────────────────────────────────────────────
def test_physical_book_votes_once_fresher_wins():
    oa = {KEY15: [ME.BookQuote("pinnacle", 1.29, 3.50, T1), ME.BookQuote("betsson", 1.27, 3.55, T1)]}
    af = {KEY15: [ME.BookQuote("af:pinnacle", 1.31, 3.40, T2), ME.BookQuote("af:bet365", 1.30, 3.40, T1)]}
    m = EAF.merge_pairs(oa, af)[KEY15]
    assert sorted(q.book for q in m) == ["af:bet365", "af:pinnacle", "betsson"]
    # an older API-Football quote does not displace a fresher OddsAPI one
    af_old = {KEY15: [ME.BookQuote("af:pinnacle", 1.31, 3.40, T1 - pd.Timedelta(hours=1))]}
    assert [q.book for q in EAF.merge_pairs(oa, af_old)[KEY15]] == ["pinnacle", "betsson"]


# ── the shadow run ─────────────────────────────────────────────────────────────────────────────
def _write_af(tmp_path, rows):
    d = tmp_path / "book_quotes"
    d.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{NOW:%Y-%m-%d}.csv", index=False)
    return d


def _af_rows():
    return (_ou15("Bet365") + _ou15("Pinnacle", 1.28, 3.60) + _ou15("Betano", 1.31, 3.30)
            + [_row("Bet365", "ou", "over", 3.5, 3.30), _row("Bet365", "ou", "under", 3.5, 1.33)])


def _run(tmp_path, board, af_dir, use=True, shadow="shadow.csv"):
    side = pipeline._generate_side_bets(board, config.SIDE_MARKETS)
    return S.run(board, None, side, historical=None, now=NOW, shadow_path=tmp_path / shadow,
                 health_path=tmp_path / "health.json", snapshots_path=tmp_path / "none.csv",
                 live_rows=[], af_quotes_dir=af_dir, use_af_quotes=use)


def test_shadow_gets_two_sided_ou15_and_health_counts_sources(tmp_path):
    _run(tmp_path, _board(), _write_af(tmp_path, _af_rows()))
    d = pd.read_csv(tmp_path / "shadow.csv")
    r15 = d[d["market"] == "OU15"].iloc[0]
    assert r15["market_source"] == "MEDIAN_TWO_SIDED_BOOKS" and int(r15["n_books"]) == 3
    assert r15["books_used"] == "af:bet365;af:betano;af:pinnacle"
    assert r15["price_timestamp"] == _z(T1) and r15["price_age_minutes"] == 40.0
    r35 = d[d["market"] == "OU35"].iloc[0]
    assert r35["market_source"] == "SINGLE_TWO_SIDED_BOOK" and r35["new_state"] == "NO_BET"   # O3.5 blocked
    h = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    af = h["apifootball_book_quotes"]
    assert af["enabled"] and af["pairs_same_snapshot"] == 4 and af["matched_both_teams"] == 1
    c = h["coverage"]["OU15|standard"]
    assert c["two_sided"] == 1 and c["two_sided_by_source"] == {"oddsapi": 0, "apifootball": 1, "both": 0}
    assert c["median_books_two_sided"]["apifootball"] == 3.0 and c["median_price_age_minutes"] == 40.0
    assert h["coverage"]["OU25|standard"]["two_sided_by_source"]["apifootball"] == 0


def test_disabled_source_reproduces_the_previous_engine(tmp_path):
    _run(tmp_path, _board(), _write_af(tmp_path, _af_rows()), use=False, shadow="off.csv")
    d = pd.read_csv(tmp_path / "off.csv")
    assert (d.loc[d["market"] == "OU15", "new_reason"] == "ONE_SIDED_PRICE").all()
    h = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    assert h["apifootball_book_quotes"] == {"enabled": False}


def test_existing_first_sight_rows_are_never_rewritten_by_new_coverage(tmp_path):
    af = _write_af(tmp_path, _af_rows())
    _run(tmp_path, _board(), af, use=False)                    # logged before the change
    first = (tmp_path / "shadow.csv").read_bytes()
    _run(tmp_path, _board(), af, use=True)                     # same fixtures, new source
    assert (tmp_path / "shadow.csv").read_bytes() == first
    b = pd.concat([_board(), _board("Norwich City", "Watford", _oa_event_id="evB")], ignore_index=True)
    rows = _af_rows() + _ou15("Bet365", fid=9002, home="Norwich", away="Watford")
    _write_af(tmp_path, rows)
    _run(tmp_path, b, af, use=True)
    after = (tmp_path / "shadow.csv").read_bytes()
    assert after.startswith(first) and len(after) > len(first)
    d = pd.read_csv(tmp_path / "shadow.csv")
    new = d[(d["fixture_id"] == "evB") & (d["market"] == "OU15")].iloc[0]
    old = d[(d["fixture_id"] == "evA") & (d["market"] == "OU15")].iloc[0]
    assert new["market_source"] == "SINGLE_TWO_SIDED_BOOK" and old["market_source"] == "ONE_SIDED"


def test_league_weekly_marks_new_two_sided_rows_as_priced(tmp_path):
    _run(tmp_path, _board(), _write_af(tmp_path, _af_rows()))
    d = pd.read_csv(tmp_path / "shadow.csv", dtype=str, keep_default_na=False)
    r = d[d["market"] == "OU15"].iloc[0].to_dict()
    v = LW.side_view(r, "OVER", ME.load_params(), (1.30, float("nan")))
    assert v["price_quality_state"] in LW.PRICED_STATES
    assert v["price_quality_state"] == "TWO_SIDED_MULTI_BOOK"
    # and the pre-change row for the same fixture stays INSUFFICIENT
    _run(tmp_path, _board(), tmp_path / "book_quotes", use=False, shadow="off.csv")
    o = pd.read_csv(tmp_path / "off.csv", dtype=str, keep_default_na=False)
    v0 = LW.side_view(o[o["market"] == "OU15"].iloc[0].to_dict(), "OVER", ME.load_params(),
                      (1.30, float("nan")))
    assert v0["price_quality_state"] == "INSUFFICIENT"


def test_unreadable_source_never_breaks_the_shadow(tmp_path):
    d = tmp_path / "book_quotes"
    d.mkdir()
    (d / f"{NOW:%Y-%m-%d}.csv").write_text("not,a\n\"broken", encoding="utf-8")
    per_side = _run(tmp_path, _board(), d)
    assert per_side is not None and (tmp_path / "shadow.csv").exists()


@pytest.mark.parametrize("name,expect", [("af:william_hill", "williamhill"), ("af:1xbet", "onexbet"),
                                         ("af:betfair", "af:betfair"), ("pinnacle", "pinnacle")])
def test_physical_book_map(name, expect):
    assert EAF.physical_book(name) == expect
