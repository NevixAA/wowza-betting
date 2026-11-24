"""src/tip_truth - the fail-closed tip truth gate (owner 2026-11-24: "no duplicates, no unreal data, not unreal
markets"). Synthetic API-Football fixtures and snapshots; no file read, no network."""
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import tip_truth as T

pytestmark = pytest.mark.real_tip_truth

LG = "Argentina Primera Division"
NOW = "2026-11-24T12:00:00Z"


def _af(status="NS", kickoff="2026-11-29T00:00:00Z", extra=()):
    fx = pd.DataFrame([{"fixture_id": 9001, "league_id": 128, "kickoff": kickoff, "status": status,
                        "home_id": 437, "away_id": 2432}, *extra])
    names = pd.DataFrame([{"league": LG, "name": "Rosario Central", "team_id": 437},
                          {"league": LG, "name": "Barracas Central", "team_id": 2432},
                          {"league": LG, "name": "Velez Sarsfield BA", "team_id": 435},
                          {"league": LG, "name": "Independiente", "team_id": 453}])
    return T.AFIndex(fx, names, {LG: 128})


def _snaps(book="pinnacle", market="BTTS", side="YES", ts="2026-11-23T01:00:00Z"):
    team = pd.DataFrame([{"snapshot_ts": ts, "league": LG, "match_date": "2026-11-29",
                          "match": "Rosario Central vs Barracas Central", "bookmaker": book, "market": market,
                          "side": side}])
    return T.Snapshots(team, None)


def _row(**k):
    r = {"league": LG, "kickoff_utc": "2026-11-28T15:00:00Z", "match_date": "2026-11-28",
         "home_team": "Rosario Central", "away_team": "Barracas Central", "market": "BTTS", "side": "YES",
         "odds": 2.2, "opposite_side_odds": 1.6, "entry_bookmaker": "pinnacle",
         "entry_captured_at": "2026-11-23T02:00:00Z", "fixture_id": "3c55"}
    r.update(k)
    return r


def _gate(af=None, snaps=None, bars=True):
    return T.TipTruth(af or _af(), snaps or _snaps(), (lambda *a: bars))


# ── 1. real fixture ──────────────────────────────────────────────────────────────────────────────
def test_real_fixture_within_one_day_resolves_to_the_af_fixture():
    why, info = _gate().check(_row(), NOW)
    assert why == "" and info["af_fixture_id"] == 9001
    assert info["af_kickoff_utc"] == pd.Timestamp("2026-11-29T00:00:00Z")


def test_oddsapi_only_event_is_blocked():
    assert _gate().check(_row(kickoff_utc="2026-11-26T15:00:00Z"), NOW)[0] == T.NO_AF_FIXTURE
    assert _gate().check(_row(home_team="Velez Sarsfield BA", away_team="Independiente"), NOW)[0] == T.NO_AF_FIXTURE
    assert _gate().check(_row(home_team="Nobody FC"), NOW)[0] == T.AF_TEAM_UNRESOLVED


def test_postponed_and_ambiguous_are_blocked():
    assert _gate(_af(status="PST")).check(_row(), NOW)[0] == T.AF_NOT_PLAYABLE
    two = _af(extra=[{"fixture_id": 9002, "league_id": 128, "kickoff": "2026-11-28T00:00:00Z", "status": "NS",
                      "home_id": 437, "away_id": 2432}])
    assert _gate(two).check(_row(), NOW)[0] == T.AF_AMBIGUOUS


# ── 3. real price ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kw,why", [
    ({"odds": ""}, T.NO_PRICE),
    ({"entry_bookmaker": ""}, T.NO_BOOK),
    ({"entry_bookmaker": "median:"}, T.MEDIAN_ONLY),
    ({"entry_bookmaker": "unibet_se"}, T.BOOK_NOT_APPROVED),
    ({"entry_captured_at": ""}, T.NO_CAPTURE),
    ({"entry_captured_at": "2026-11-24T13:00:00Z"}, T.CAPTURED_LATE),
    ({"opposite_side_odds": ""}, T.ONE_SIDED),
])
def test_price_must_be_a_named_approved_two_sided_capture_before_send(kw, why):
    assert _gate().check(_row(**kw), NOW)[0] == why


def test_median_of_named_books_is_a_real_price():
    """Today's engine median (2+ books) and PR #104's O/U 2.5 median (3+ books) both pass; each needs its books in the
    snapshot."""
    team = pd.DataFrame([{"snapshot_ts": "2026-11-23T01:00:00Z", "league": LG, "match_date": "2026-11-29",
                          "match": "Rosario Central vs Barracas Central", "bookmaker": b, "market": "BTTS",
                          "side": "YES"} for b in ("williamhill", "unibet_se", "betsson")])
    g = _gate(snaps=T.Snapshots(team, None))
    assert g.check(_row(entry_bookmaker="median:unibet_se;williamhill"), NOW)[0] == ""
    assert g.check(_row(entry_bookmaker="median:betsson;unibet_se;williamhill"), NOW)[0] == ""
    assert g.check(_row(entry_bookmaker="median:marathonbet;matchbook"), NOW)[0] == T.NOT_OFFERED
    assert T.median_books("median:b;a") == ["a", "b"] and T.median_books("bet365") is None


def test_book_aliases():
    assert T.book_key("1xBet") == "onexbet" and T.book_key("Bet365") == "bet365" and T.book_key("") == ""


# ── 4. real market ───────────────────────────────────────────────────────────────────────────────
def test_market_needs_an_applied_bar_and_the_book_quoting_it():
    assert _gate(bars=False).check(_row(), NOW)[0] == T.NOT_IN_BARS
    assert _gate(snaps=_snaps(book="bet365")).check(_row(), NOW)[0] == T.NOT_OFFERED
    assert _gate(snaps=_snaps(side="NO")).check(_row(), NOW)[0] == T.NOT_OFFERED
    late = _snaps(ts="2026-11-23T05:00:00Z")                     # quoted only AFTER the price's capture
    assert _gate(snaps=late).check(_row(), NOW)[0] == T.NOT_OFFERED


# ── 2. no duplicates ─────────────────────────────────────────────────────────────────────────────
def test_w48_rosario_two_event_ids_keep_only_the_event_closest_to_af():
    """The W48 bug: two OddsAPI event ids, one match. Only one survives; the stale id is dropped."""
    d = pd.DataFrame([_row(fixture_id="3c55", kickoff_utc="2026-11-28T15:00:00Z"),
                      _row(fixture_id="8799", kickoff_utc="2026-11-29T21:00:00Z", match_date="2026-11-29")])
    g = _gate().gate(d, NOW)
    assert list(g["truth_reason"]) == ["", T.STALE_EVENT]
    assert set(g["canonical_match"]) == {"af:9001"}


def test_one_pick_per_match_market_side_but_other_markets_allowed():
    team = pd.DataFrame([{"snapshot_ts": "2026-11-23T01:00:00Z", "league": LG, "match_date": "2026-11-29",
                          "match": "Rosario Central vs Barracas Central", "bookmaker": "pinnacle", "market": m,
                          "side": s} for m, s in (("BTTS", "YES"), ("OU25", "OVER"))])
    gt = _gate(snaps=T.Snapshots(team, None))
    d = pd.DataFrame([_row(), _row(), _row(market="OU25", side="OVER")])
    g = gt.gate(d, NOW)
    assert list(g["truth_reason"]) == ["", T.DUPLICATE, ""]
    # against keys already frozen / sent (another slot, week or sender)
    g2 = gt.gate(pd.DataFrame([_row()]), NOW, prior={("af:9001", "BTTS", "YES")})
    assert list(g2["truth_reason"]) == [T.DUPLICATE]


def test_pass_all_double_still_dedupes():
    d = pd.DataFrame([_row(), _row()])
    assert list(T.PassAll().gate(d, NOW)["truth_reason"]) == ["", T.DUPLICATE]


def test_default_fails_closed_when_files_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_factory", None)
    t = T.default(out_dir=tmp_path, root=tmp_path, bars_ok=lambda *a: True)
    assert t.check(_row(), NOW)[0] in (T.AF_TEAM_UNRESOLVED, T.NO_AF_FIXTURE)


# ── TOP-3 freeze uses the gate ───────────────────────────────────────────────────────────────────
def test_top3_freeze_never_fills_two_slots_with_one_match(monkeypatch):
    from src import top3_league as TL
    q = pd.DataFrame([_row(fixture_id="3c55"), _row(fixture_id="8799", kickoff_utc="2026-11-29T21:00:00Z")])
    g = _gate().gate(q, NOW)
    assert (g["truth_reason"] == "").sum() == 1
    assert TL.truth_bars_ok({"standard": {}, "side": {}, "applied": set()})(LG, "BTTS", "YES") is False
    assert TL.truth_bars_ok({"standard": {}, "side": {}, "applied": set()})(LG, "NOT_A_MARKET", "YES") is False


def test_props_have_no_approved_bar_so_fail_closed():
    from src import props_top3 as PT
    assert PT.truth_bars_ok("Premier League", "goals") is False


# ── the collector gap behind W48 (play-off fixtures after the stated season end) ─────────────────
def test_extended_season_is_reread():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import collect_competition_fixtures as F
    j = pd.DataFrame([{"league_id": 128, "season": "2026", "season_end": "2026-11-08",
                       "fetched_at": "2026-11-09T13:16:33+00:00"}])
    same = [{"season": "2026", "start": pd.Timestamp("2026-01-22"), "end": pd.Timestamp("2026-11-08")}]
    # inside the tail after the stated end: re-read (play-offs are added after the regular season)
    assert [s["season"] for s in F.plan_seasons(same, j, 128, pd.Timestamp("2020-01-01"),
                                                pd.Timestamp("2026-11-24"))] == ["2026"]
    j2 = j.assign(fetched_at="2026-12-15T00:00:00+00:00")
    assert F.plan_seasons(same, j2, 128, pd.Timestamp("2020-01-01"), pd.Timestamp("2026-12-15")) == []
    ext = [{"season": "2026", "start": pd.Timestamp("2026-01-22"), "end": pd.Timestamp("2026-12-20")}]
    assert [s["season"] for s in F.plan_seasons(ext, j2, 128, pd.Timestamp("2020-01-01"),
                                                pd.Timestamp("2026-12-15"))] == ["2026"]
