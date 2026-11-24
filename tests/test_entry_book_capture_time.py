"""Entry book + capture time on every tip / ledger row written from now on (owner 2026-11-23 "Yes, do all", D6).

Every writer records WHO quoted the entry price and WHEN the provider answer behind it was captured; a price
source that carries neither leaves both blank; a cross-book median is labelled as one ("median:<books>"),
never as an invented single book; rows written before stay exactly as they were (never back-filled).

    python -m pytest tests/test_entry_book_capture_time.py -q
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import pandas as pd
import pytest

import config

ENTRY = ["entry_bookmaker", "entry_captured_at"]
TS_RE = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$"


class _Resp:
    status_code = 200

    def __init__(self, payload, headers=None):
        self._p = payload
        self.headers = headers or {}

    def json(self):
        return self._p


def _ko(hours=20):
    return (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 1. board: predict._fetch_odds_api ────────────────────────────────────────────────────────────
def _totals(key, outs):
    return {"key": key, "markets": [{"key": "totals", "outcomes": [
        {"name": n, "point": pt, "price": p} for n, pt, p in outs]}]}


def test_board_records_book_and_capture_time_per_price(tmp_path, monkeypatch):
    import requests

    import src.btts_odds as B
    import src.predict as P
    import src.rolling_store as RS
    league = sorted(set(config.ODDS_API_SPORT_KEYS) & set(config.ENABLED_LEAGUES))[0]
    monkeypatch.setattr(config, "ODDS_API_SPORT_KEYS", {league: config.ODDS_API_SPORT_KEYS[league]})
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(B, "fetch", lambda rows: rows)
    monkeypatch.setattr(RS, "roll", lambda *a, **k: {})
    ev = [{"commence_time": _ko(), "home_team": "H", "away_team": "A", "bookmakers": [
        _totals("b1", [("Over", 2.5, 2.0), ("Under", 2.5, 1.8), ("Over", 1.5, 1.3)]),
        _totals("b2", [("Over", 2.5, 2.1), ("Under", 2.5, 1.7), ("Over", 1.5, 1.4)])]}]
    t0 = pd.Timestamp.now(tz="UTC").floor("s")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(ev))
    r = P._fetch_odds_api().iloc[0]
    t1 = pd.Timestamp.now(tz="UTC").ceil("s")
    assert r["odds_over25_book"] == "b1" and r["odds_under25_book"] == "b1" and r["odds_over15_book"] == "b1"
    for c in ("odds_over25_captured_at", "odds_under25_captured_at", "odds_over15_captured_at"):
        assert t0 <= pd.Timestamp(r[c]) <= t1, c
    # no book quoted Over 3.5 / BTTS: no price, no book, no time - never another price's source
    for c in ("odds_over35", "odds_over35_book", "odds_over35_captured_at", "odds_btts", "odds_btts_book",
              "odds_btts_captured_at"):
        assert pd.isna(r[c]), c


# ── 2. board: btts_odds cross-book medians ───────────────────────────────────────────────────────
def test_book_label_never_names_one_book_for_a_median():
    from src.btts_odds import book_label
    assert book_label(["pinnacle"]) == "pinnacle"
    assert book_label(["b", "a", "b"]) == "median:a;b"
    assert book_label([]) is None and book_label(["", None]) is None


@pytest.fixture
def btts_env(tmp_path, monkeypatch):
    import src.btts_odds as B
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "test", raising=False)
    monkeypatch.setattr(B, "append_book_rows", lambda rows: None)
    return B


def _event(*books):
    return {"bookmakers": [{"key": k, "markets": mk} for k, mk in books]}


def test_btts_and_alt_totals_carry_a_median_label_and_the_call_time(btts_env, monkeypatch):
    B = btts_env
    payload = _event(
        ("x", [{"key": "btts", "outcomes": [{"name": "Yes", "price": 1.8}, {"name": "No", "price": 2.0}]},
               {"key": "alternate_totals", "outcomes": [{"name": "Over", "point": 3.5, "price": 3.0},
                                                        {"name": "Under", "point": 3.5, "price": 1.4}]}]),
        ("y", [{"key": "btts", "outcomes": [{"name": "Yes", "price": 1.9}, {"name": "No", "price": 1.9}]}]))
    monkeypatch.setattr(B.requests, "get", lambda *a, **k: _Resp(payload, {"x-requests-last": "2"}))
    row = {"home_team": "H", "away_team": "A", "league": "L", "date": "2026-11-24", "kickoff_utc": _ko(),
           "_oa_event_id": "e1", "_oa_sport_key": "s", "odds_over15": 1.3, "odds_over15_book": "bulkbook",
           "odds_over15_captured_at": "2026-11-23T10:00:00Z", "odds_over35": None}
    t0 = pd.Timestamp.now(tz="UTC").floor("s")
    out = B.fetch([row])[0]
    t1 = pd.Timestamp.now(tz="UTC").ceil("s")
    assert out["odds_btts_book"] == "median:x;y" and out["odds_btts_no_book"] == "median:x;y"
    assert t0 <= pd.Timestamp(out["odds_btts_captured_at"]) <= t1
    assert out["odds_over35_book"] == "x" and out["odds_over35_captured_at"] == out["odds_btts_captured_at"]
    # a price the bulk call supplied is never overwritten - and neither are its book and time
    assert out["odds_over15_book"] == "bulkbook" and out["odds_over15_captured_at"] == "2026-11-23T10:00:00Z"


def test_btts_without_both_sides_leaves_no_book(btts_env, monkeypatch):
    B = btts_env
    payload = _event(("x", [{"key": "btts", "outcomes": [{"name": "Yes", "price": 1.8}]}]))
    monkeypatch.setattr(B.requests, "get", lambda *a, **k: _Resp(payload))
    out = B.fetch([{"home_team": "H", "away_team": "A", "league": "L", "date": "2026-11-24", "kickoff_utc": _ko(),
                    "_oa_event_id": "e1", "_oa_sport_key": "s", "odds_over15": 1.3, "odds_over35": 3.0}])[0]
    assert "odds_btts_book" not in out and "odds_btts_captured_at" not in out


# ── 3. bets_ledger ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def ledgers(tmp_path, monkeypatch):
    from src import ledger as L
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "SIDE_LEDGER_FILE", tmp_path / "side_bets_ledger.csv")
    monkeypatch.setattr(L, "HT_LEDGER_FILE", tmp_path / "ht_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    return L


def _bet(home, side, **kw):
    return {"date": "2026-11-24", "league": "Japan J-League", "home_team": home, "away_team": home + " B",
            "bet": side, "best_side": side, "odds_over25": 2.4, "odds_under25": 1.6, "best_edge": 0.05,
            "signal_tier": "QUALIFIED", "drift_signal": "", "model_type": "new_format", **kw}


def test_bets_ledger_records_the_sides_book_and_time_and_never_rewrites_old_rows(ledgers):
    L = ledgers
    old_cols = [c for c in L.LEDGER_COLS if c not in ENTRY]
    old = pd.DataFrame([{c: "" for c in old_cols} | {"source": "live", "generated_at": "2026-11-01 10:00",
                                                       "match_date": "2026-11-02", "home_team": "Old",
                                                       "away_team": "Old B", "side": "OVER", "odds": "2.05",
                                                       "signal_tier": "QUALIFIED"}])
    old.to_csv(L.LEDGER_FILE, index=False)
    L.append_tips(pd.DataFrame([
        _bet("U", "UNDER", odds_over25_book="bo", odds_under25_book="bu",
             odds_over25_captured_at="2026-11-23T10:00:00Z", odds_under25_captured_at="2026-11-23T10:00:01Z"),
        _bet("N", "OVER")]))                                       # a board row with no source recorded
    led = pd.read_csv(L.LEDGER_FILE, dtype=str, keep_default_na=False)
    assert list(led.columns[-2:]) == ENTRY
    by = led.set_index("home_team")
    assert by.loc["U", "entry_bookmaker"] == "bu" and by.loc["U", "entry_captured_at"] == "2026-11-23T10:00:01Z"
    assert by.loc["N", "entry_bookmaker"] == "" and by.loc["N", "entry_captured_at"] == ""
    # the row written before stays as it was: its values unchanged, the new columns blank (never back-filled)
    assert by.loc["Old", "odds"] == "2.05" and by.loc["Old", "generated_at"] == "2026-11-01 10:00"
    assert by.loc["Old", "entry_bookmaker"] == "" and by.loc["Old", "entry_captured_at"] == ""


# ── 4. side_bets_ledger (+ producer) ─────────────────────────────────────────────────────────────
def test_side_ledger_carries_the_source_of_market_odds(ledgers):
    L = ledgers
    base = {"date": "2026-11-24", "league": "Brazil Serie A", "away_team": "B", "market": "btts", "side": "YES",
            "signal_tier": "QUALIFIED", "edge": 0.05, "ev": 0.04, "model_prob": 0.56, "model_type": "new_format"}
    L.append_side_market_tips(pd.DataFrame([
        base | {"home_team": "P", "market_odds": 1.9, "entry_bookmaker": "median:a;b",
                "entry_captured_at": "2026-11-23T10:00:00Z"},
        base | {"home_team": "Q", "market_odds": 1.9},                  # priced, source unknown
        base | {"home_team": "R", "entry_bookmaker": "a", "entry_captured_at": "2026-11-23T10:00:00Z"}]))  # no price
    led = pd.read_csv(L.SIDE_LEDGER_FILE, dtype=str, keep_default_na=False).set_index("home_team")
    assert list(led.columns[-2:]) == ENTRY
    assert led.loc["P", "entry_bookmaker"] == "median:a;b" and led.loc["P", "entry_captured_at"] == "2026-11-23T10:00:00Z"
    assert led.loc["Q", "entry_bookmaker"] == "" and led.loc["Q", "entry_captured_at"] == ""
    assert led.loc["R", "odds"] == "" and led.loc["R", "entry_bookmaker"] == "" and led.loc["R", "entry_captured_at"] == ""


def test_side_producer_takes_the_book_of_the_column_that_became_market_odds(monkeypatch):
    import pipeline
    from tests.own_bars import install_historical
    install_historical(monkeypatch)
    # TEST-ONLY fair price for the synthetic row (as tests/test_side_stake_contract.py): the source columns are
    # what is under test here, not the de-vig
    from src.integrity import side_edge_devig as _SE
    monkeypatch.setattr(_SE, "load_quotes", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(_SE, "production_fair_probs", lambda df, target, quotes, asof: (1.0 / df["market_odds"]) / 1.08)
    lg = "Argentina Primera Division"
    preds = pd.DataFrame([{"date": "2026-10-23", "league": lg, "home_team": "A", "away_team": "A B",
                           "model_type": config.model_type_for_league(lg), "p_btts": 0.62, "odds_btts": 2.10,
                           "odds_btts_book": "median:a;b", "odds_btts_captured_at": "2026-10-23T08:00:00Z",
                           "p_over15": None, "odds_over15": None, "p_over35": None, "odds_over35": None,
                           # another column's source must never leak onto the BTTS tip
                           "odds_over25_book": "wrong", "odds_over25_captured_at": "2026-10-23T07:00:00Z"}])
    side = pipeline._generate_side_bets(preds, config.SIDE_MARKETS)
    assert len(side) == 1
    assert side.iloc[0]["entry_bookmaker"] == "median:a;b"
    assert side.iloc[0]["entry_captured_at"] == "2026-10-23T08:00:00Z"


# ── 5. ht_ledger ────────────────────────────────────────────────────────────────────────────────
def test_ht_price_book_is_bet365_by_construction_of_its_only_writer():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "capture_std_sidemarket_odds_forward.py").read_text(
        encoding="utf-8")
    assert "BET365 = 8" in src and 'if bk.get("id") != BET365:' in src


def test_ht_ledger_records_bet365_and_the_snapshot_ts_and_blank_without_price(ledgers, tmp_path):
    L = ledgers
    pd.DataFrame([
        {"snapshot_date": "2026-11-23", "snapshot_ts": "2026-11-23T08:00:00Z", "match_date": "2026-11-24",
         "kickoff_utc": "2026-11-24T15:00:00Z", "league": "Championship", "match": "Hull City vs Leeds",
         "market": "ht_over05", "odds": 1.4},
        {"snapshot_date": "2026-11-23", "snapshot_ts": "2026-11-23T09:30:00Z", "match_date": "2026-11-24",
         "kickoff_utc": "2026-11-24T15:00:00Z", "league": "Championship", "match": "Hull City vs Leeds",
         "market": "ht_over05", "odds": 1.35}]).to_csv(tmp_path / "standard_sidemarket_odds_history.csv",
                                                        index=False)
    preds = pd.DataFrame([
        {"date": "2026-11-24", "league": "Championship", "home_team": "Hull City", "away_team": "Leeds",
         "model_type": "standard", "p_ht_over05": 0.8, "p_ht_over15": 0.4},
        {"date": "2026-11-24", "league": "Championship", "home_team": "Nowhere", "away_team": "Else",
         "model_type": "standard", "p_ht_over05": 0.8, "p_ht_over15": 0.4}])
    L.append_ht_tips(preds)
    led = pd.read_csv(L.HT_LEDGER_FILE, dtype=str, keep_default_na=False).set_index("home_team")
    assert list(led.columns[-2:]) == ENTRY
    assert led.loc["Hull City", "entry_bookmaker"] == "bet365"
    assert led.loc["Hull City", "entry_captured_at"] == "2026-11-23T09:30:00Z"       # the row whose price is used
    assert led.loc["Nowhere", "odds"] == "" and led.loc["Nowhere", "entry_bookmaker"] == ""
    assert led.loc["Nowhere", "entry_captured_at"] == ""


# ── 6. v91_shadow ───────────────────────────────────────────────────────────────────────────────
def test_shadow_declares_the_entry_time_and_keeps_late_columns_out_of_the_cross_day_digest():
    from src import shadow_compare as sc
    assert sc.COLUMNS[-2:] == ENTRY
    assert set(ENTRY) <= set(sc.LATE_FIRST_SIGHT) and set(ENTRY) <= set(sc.FROZEN)


def test_shadow_row_takes_the_time_of_its_side(tmp_path, monkeypatch):
    import pipeline
    from src import shadow_compare as sc
    monkeypatch.setattr(sc, "SHADOW_FILE", tmp_path / "v91_shadow.csv")
    preds = pd.DataFrame([{"fixture_id": "f1", "date": "2026-11-24", "kickoff_utc": _ko(), "league": "L",
                           "model_type": "standard", "home_team": "H", "away_team": "A", "best_side": "UNDER",
                           "p_over25": 0.4, "odds_over25": 2.1, "odds_under25": 1.8,
                           "odds_over25_book": "bo", "odds_under25_book": "bu",
                           "odds_over25_captured_at": "2026-11-23T10:00:00Z",
                           "odds_under25_captured_at": "2026-11-23T10:00:05Z"}])
    pipeline._log_shadow(preds)
    d = pd.read_csv(sc.SHADOW_FILE, dtype=str)
    assert d.iloc[0]["entry_bookmaker"] == "bu" and d.iloc[0]["entry_captured_at"] == "2026-11-23T10:00:05Z"


# ── 7. TOP-3 candidates / ledger ─────────────────────────────────────────────────────────────────
def _sv(odds, side="OVER", src="ENGINE_SIDE"):
    return {"odds": odds, "side": side, "side_source": src}


def test_top3_entry_source_rules():
    from src.research import league_weekly as LW
    P = {"odds_haircut": 0.0}
    r = {"prediction_ts": "2026-11-23T10:05:00Z", "odds_source": "ROW", "books_used": "a;b",
         "price_timestamp": "2026-11-23T09:00:00Z"}
    b = {"odds_over25": 2.1, "odds_over25_book": "pinnacle", "odds_over25_captured_at": "2026-11-23T10:00:00Z"}
    # 1. the board price IS the pick price, captured before first sight -> the board's book and time
    assert LW.entry_source(r, _sv(2.1), b, "OU25", P) == ("pinnacle", "2026-11-23T10:00:00Z")
    # the board moved after first sight (captured later) -> blank, never a later capture's source
    late = b | {"odds_over25_captured_at": "2026-11-23T11:00:00Z"}
    assert LW.entry_source(r, _sv(2.1), late, "OU25", P) == ("", "")
    # a different price on the board -> blank
    assert LW.entry_source(r, _sv(2.2), b, "OU25", P) == ("", "")
    # 2. the engine's cross-book median -> median label + its price_timestamp (oldest quote)
    med = r | {"odds_source": "MEDIAN_VALID_BOOKS"}
    assert LW.entry_source(med, _sv(2.2), b, "OU25", P) == ("median:a;b", "2026-11-23T09:00:00Z")
    # ...but never under a haircut (no book quoted a haircut price), nor for a best-of-books price
    assert LW.entry_source(med, _sv(2.2), b, "OU25", {"odds_haircut": 0.02}) == ("", "")
    best = r | {"odds_source": "BEST_VALID_BOOK"}
    assert LW.entry_source(best, _sv(2.2), b, "OU25", P) == ("", "")
    # the other side reads its own column only
    assert LW.entry_source(r, _sv(2.1, side="UNDER"), b, "OU25", P) == ("", "")
    # no price -> blank
    assert LW.entry_source(r, _sv(float("nan")), b, "OU25", P) == ("", "")


def test_candidate_store_widens_an_old_file_without_touching_a_value(tmp_path):
    from src.research import league_weekly as LW
    from src.research.first_sight import frozen_digest
    old_cols = [c for c in LW.CAND_COLS if c not in LW.CAND_LATE]
    p = tmp_path / "league_weekly_candidates.csv"
    row = {c: "" for c in old_cols} | {"selection_engine_version": "v", "fixture_id": "f1", "market": "OU25",
                                         "first_seen_at": "2026-11-20T10:00:00Z", "kickoff_utc": "2026-11-21T15:00:00Z",
                                         "odds": "2.05"}
    pd.DataFrame([row]).to_csv(p, index=False, lineterminator="\n")
    before = pd.read_csv(p, dtype=str, keep_default_na=False)
    store = LW.candidate_store(p)
    # the system_health digest of the store ignores the appended columns, so it is unchanged by the widening
    d0 = store.integrity()["frozen_digest"]
    new = pd.DataFrame([row | {"fixture_id": "f2", "entry_bookmaker": "bk", "entry_captured_at": "2026-11-20T09:59:00Z"}])
    assert store.append(new) == 1
    after = store.read()
    assert list(after.columns) == LW.CAND_COLS
    pd.testing.assert_frame_equal(after.iloc[[0]][old_cols].reset_index(drop=True), before[old_cols])
    assert after.iloc[0]["entry_bookmaker"] == "" and after.iloc[1]["entry_bookmaker"] == "bk"
    assert frozen_digest(after, store.frozen_cols, 1) == d0
    with p.open(encoding="utf-8", newline="") as fh:
        assert next(csv.reader(fh)) == LW.CAND_COLS


def test_top3_migration_adds_the_entry_columns_blank_and_keeps_every_frozen_value(tmp_path):
    from src import top3_league as T
    from src.research.first_sight import FirstSightStore
    old_frozen = [c for c in T.PICK_FROZEN if c not in T.ENTRY_COLS]
    old = FirstSightStore(tmp_path / T.PICKS_NAME, key=T.PICK_KEY, columns=old_frozen + T.PICK_GRADING,
                          observed_col="issued_at", kickoff_col="kickoff_utc", frozen_cols=old_frozen, digest=True)
    row = {c: "" for c in old_frozen + T.PICK_GRADING} | {
        "selection_mode": "TOP3", "league_week_id": "L|2026-W47", "top3_rank": "1", "odds": "2.05",
        "issued_at": "2026-11-20T10:00:00Z", "kickoff_utc": "2026-11-21T15:00:00Z", "stake": "0.0",
        "stake_reason": "NO_STAKE_BELOW_BAR", "value_basis": "CALIBRATED"}
    assert old.append(pd.DataFrame([row])) == 1
    before = old.read()
    assert T.migrate_ledger(tmp_path)["migrated"] is True
    after = T.picks_store(tmp_path).read()
    assert list(after.columns) == T.PICK_COLS
    pd.testing.assert_frame_equal(after[old_frozen + T.PICK_GRADING], before[old_frozen + T.PICK_GRADING])
    assert (after[list(T.ENTRY_COLS)] == "").all().all()
    assert T.picks_store(tmp_path).verify()["ok"] is True
    assert T.migrate_ledger(tmp_path)["migrated"] is False


def test_top3_pick_carries_its_candidates_entry_source():
    from src import top3_league as T
    assert T.PICK_FROZEN[-4:-2] == ENTRY and tuple(T.PICK_FROZEN[-2:]) == T.TRUTH_COLS   # 2026-11-24 truth cols after
    r = {"entry_bookmaker": "bk", "entry_captured_at": "2026-11-20T09:59:00Z", "odds": 2.0}
    assert {k: r.get(k, "") for k in T.PICK_FROZEN if k in r}["entry_bookmaker"] == "bk"


# ── 8. props TOP-3 ──────────────────────────────────────────────────────────────────────────────
def test_props_candidates_carry_the_board_source_and_the_ledger_header_stays_a_prefix(tmp_path):
    from src import props_top3 as PT
    assert PT.PICK_COLS[-2:] == ENTRY and set(ENTRY) <= set(PT.PICK_FROZEN)
    now = pd.Timestamp("2026-11-20T10:00:00Z")
    board = pd.DataFrame([
        {"league": "Premier League", "date": "2026-11-22", "kickoff_utc": "2026-11-22T15:00:00Z",
         "match": "Arsenal vs Chelsea", "fixture_id": 11, "player_id": p, "player_name": n, "team": "Arsenal",
         "position": "F", "market": "goals", "model_prob": 0.4, "market_odds": 3.0, "fair_implied": 0.3,
         "fair_basis": "two_sided_devig", "entry_bookmaker": b, "entry_captured_at": t}
        for p, n, b, t in ((1, "A One", "Pinnacle", "2026-11-20T09:00:00Z"), (2, "B Two", None, None))])
    c = PT.build_candidates(board, now).set_index("player_id")
    assert c.loc["1", "entry_bookmaker"] == "Pinnacle" and c.loc["1", "entry_captured_at"] == "2026-11-20T09:00:00Z"
    assert c.loc["2", "entry_bookmaker"] == "" and c.loc["2", "entry_captured_at"] == ""


# ── 9. player props: odds_fetcher meta -> player_tips -> player_ledger ──────────────────────────
def test_prop_meta_tracks_the_books_at_the_best_price():
    from player_model import odds_fetcher as O
    meta = {}
    O._add_fixture_meta(meta, "2026-11-22", "X vs Y", "a one", "goals", 2.5, "B1", "2026-11-20T09:00:00Z")
    O._add_fixture_meta(meta, "2026-11-22", "X vs Y", "a one", "goals", 3.0, "B2", "2026-11-20T09:05:00Z")
    O._add_fixture_meta(meta, "2026-11-22", "X vs Y", "a one", "goals", 3.0, "B3", "2026-11-20T09:01:00Z")
    m = meta[("2026-11-22", "X vs Y")]["a one|goals"]
    assert m["odds"] == 3.0
    assert O.entry_of(m) == ("tie:B2;B3", "2026-11-20T09:01:00Z")
    O._add_fixture_meta(meta, "2026-11-22", "X vs Y", "a one", "goals", 3.5, "B4", "")
    assert O.entry_of(meta[("2026-11-22", "X vs Y")]["a one|goals"]) == ("B4", "")
    assert O.entry_of({"odds": 2.0, "src": {("", "2026-11-20T09:00:00Z")}}) == ("", "")


def test_prop_meta_reaches_the_tip_only_at_its_own_price(monkeypatch):
    from player_model import odds_fetcher as O
    from player_model import predict as PP
    tips = pd.DataFrame([
        {"player_name": "A One", "market": "goals", "date": "2026-11-22", "match": "X vs Y", "market_odds": 3.0},
        {"player_name": "B Two", "market": "goals", "date": "2026-11-22", "match": "X vs Y", "market_odds": None}])
    prices = {("2026-11-22", "X vs Y"): {f"{O._norm('A One')}|goals": 3.0, f"{O._norm('B Two')}|goals": 2.0}}
    meta = {}
    O._add_fixture_meta(meta, "2026-11-22", "X vs Y", O._norm("A One"), "goals", 3.0, "Pinnacle",
                        "2026-11-20T09:00:00Z")
    O._add_fixture_meta(meta, "2026-11-22", "X vs Y", O._norm("B Two"), "goals", 2.5, "Other", "2026-11-20T09:00:00Z")
    mapped = O.match_meta_to_tips(tips, fixture_meta=meta, fixture_prices=prices)
    key_a = O.tip_price_key("A One", "goals", "2026-11-22", "X vs Y")
    assert mapped == {key_a: ("Pinnacle", "2026-11-20T09:00:00Z")}        # B's meta is at another price: absent
    out = PP.attach_entry_source(tips, mapped)
    assert out.iloc[0]["entry_bookmaker"] == "Pinnacle" and out.iloc[0]["entry_captured_at"] == "2026-11-20T09:00:00Z"
    assert pd.isna(out.iloc[1]["entry_bookmaker"]) and pd.isna(out.iloc[1]["entry_captured_at"])


def test_player_ledger_writes_the_entry_source_and_leaves_old_rows_alone(tmp_path):
    from player_model import ledger as PL
    f = tmp_path / "player_ledger.csv"
    old_cols = [c for c in PL.PLAYER_LEDGER_COLS if c not in ENTRY]
    pd.DataFrame([{c: "" for c in old_cols} | {"signal_date": "2026-11-01", "match_date": "2026-11-02",
                                               "player_id": "7", "fixture_id": "70", "market": "goals",
                                               "market_odds": "2.75"}]).to_csv(f, index=False)
    tips = pd.DataFrame([
        {"tier": "RESEARCH", "match": "X vs Y", "date": "2026-11-22", "player_id": 1, "fixture_id": 11,
         "player_name": "A One", "market": "goals", "market_odds": 3.0, "entry_bookmaker": "Pinnacle",
         "entry_captured_at": "2026-11-20T09:00:00Z"},
        {"tier": "RESEARCH", "match": "X vs Y", "date": "2026-11-22", "player_id": 2, "fixture_id": 11,
         "player_name": "B Two", "market": "goals", "market_odds": None, "entry_bookmaker": "Stale",
         "entry_captured_at": "2026-11-20T09:00:00Z"}])
    assert PL.append_player_signals(tips, ledger_file=f) == 2
    led = pd.read_csv(f, dtype=str, keep_default_na=False).set_index("player_id")
    assert list(led.columns[-2:]) == ENTRY
    assert led.loc["1", "entry_bookmaker"] == "Pinnacle" and led.loc["1", "entry_captured_at"] == "2026-11-20T09:00:00Z"
    assert led.loc["2", "entry_bookmaker"] == "" and led.loc["2", "entry_captured_at"] == ""   # no price, no source
    assert led.loc["7", "market_odds"] == "2.75" and led.loc["7", "entry_bookmaker"] == ""


def test_api_football_capture_time_comes_from_the_cache_stamp(tmp_path, monkeypatch):
    from player_model import api_football as AF
    monkeypatch.setattr(AF, "CACHE_DIR", tmp_path)
    assert AF.cached_at_utc("/odds", {"fixture": 5}) is None                     # nothing fetched: unknown
    import json
    AF._save_cache("/odds_" + json.dumps({"fixture": 5}, sort_keys=True), {"response": []})
    got = AF.cached_at_utc("/odds", {"fixture": 5})
    now = datetime.now(timezone.utc)
    assert got is not None and abs((pd.Timestamp(got) - pd.Timestamp(now)).total_seconds()) < 5
