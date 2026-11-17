"""SF-PAPER-1 (owner 2026-11-15): paper-only guards, universe, signal, budget. No licensed data."""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src import sharp_paper as SP

ROOT = Path(__file__).resolve().parents[1]
P = SP.load_protocol()
REGISTERED_SHA = (ROOT / "config" / "sharp_paper" / "protocol.sha256").read_text().split()[0]


def _capture_mod():
    spec = importlib.util.spec_from_file_location("spc", ROOT / "scripts" / "sharp_paper_capture.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── the protocol ─────────────────────────────────────────────────────────────────────────────

def test_protocol_matches_its_registered_hash():
    assert SP.protocol_sha() == REGISTERED_SHA


def test_protocol_is_paper_flat_one_unit():
    assert P["track"] == "PAPER" and P["stake_units"] == 1 and SP.STAKE_UNITS == 1 and SP.TRACK == "PAPER"
    assert P["signal"]["primary"] == "CONS" and P["start_match_date"] == "2026-11-21"
    assert all(a["before_any_pick"] for a in P.get("amendments", []))


def test_every_threshold_is_per_league_and_none_below_min_history():
    for lg, t in P["signal"]["league_thresholds"].items():
        for k in ("pin", "cons"):
            assert (t[k]["q75"] is None) == (t[k]["n"] < P["signal"]["min_history"]), (lg, k)


# ── PAPER ONLY: never staking, exec or public tips ──────────────────────────────────────────

PRODUCTION_DIRS = ("src", "player_model", "telegram_bot", "scheduler", "pages", "agent", "utils", "worldcup")
PRODUCTION_FILES = ("pipeline.py", "app.py", "master_loop.py", "live_loop.py", "update_results.py", "retrain.py",
                    "dashboard_ui.py", "dashboard_data.py")


def test_no_production_module_imports_the_paper_track():
    pat = re.compile(r"(import|from)\s+[\w.]*sharp_paper|sharp_paper_capture")
    hits = []
    files = [ROOT / f for f in PRODUCTION_FILES if (ROOT / f).exists()]
    for d in PRODUCTION_DIRS:
        files += [p for p in (ROOT / d).rglob("*.py") if p.exists()]
    for p in files:
        if p.name == "sharp_paper.py":
            continue
        if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
            hits.append(str(p.relative_to(ROOT)))
    assert hits == []


def test_no_workflow_of_this_repo_runs_the_paper_track():
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        assert "sharp_paper" not in wf.read_text(encoding="utf-8"), wf.name


def test_paper_code_never_names_a_staking_exec_or_publishing_target():
    for f in (ROOT / "src" / "sharp_paper.py", ROOT / "scripts" / "sharp_paper_capture.py"):
        t = f.read_text(encoding="utf-8")
        for target in ("bets_ledger.csv\", \"w", "staking", "cloudbet", "telegram", "kelly", "exec_orders",
                       "allocation_registry", "notifier"):
            assert target not in t.lower(), (f.name, target)
        # the ledgers are only READ
        assert "to_csv(Path(args.ledgers)" not in t and "ledgers) / f).write" not in t


@pytest.mark.parametrize("path", ["output/x.parquet", "config/sharp_paper/p.json", "."])
def test_every_writer_refuses_the_public_repo(path):
    with pytest.raises(SP.NotPrivateError):
        SP.assert_private(ROOT / path)


def test_capture_cli_refuses_a_store_in_the_repo_and_a_low_floor(tmp_path):
    m = _capture_mod()
    with pytest.raises(SP.NotPrivateError):
        m.main(["--store", str(ROOT / "output" / "sp"), "--ledgers", str(tmp_path), "--mode", "evaluate"])
    with pytest.raises(SystemExit):
        m.main(["--store", str(tmp_path / "s"), "--ledgers", str(tmp_path), "--mode", "dry-run", "--floor", "1000"])


def test_paper_rows_are_paper_one_unit():
    picks, _ = SP.tip_universe(_bets(), None, P)
    paper = SP.build_paper(picks, _capture(picks.iloc[0]), P)
    assert set(paper["track_kind"]) == {"PAPER"} and set(paper["stake_units"]) == {1}
    assert set(paper["pnl"].dropna().abs()) <= {1.0, paper["bet_odds"].max() - 1.0}


# ── universe ────────────────────────────────────────────────────────────────────────────────

def _cell(mside):
    return next(c for c in P["universe"]["cells"] if c["mside"] == mside)


def _bets():
    c = _cell("over25_over")
    rows = [
        {"source": "live", "generated_at": "2026-11-15 10:00", "match_date": "2026-11-22", "league": c["league"],
         "home_team": "H1", "away_team": "A1", "side": "OVER", "signal_tier": "QUALIFIED", "result": "WIN"},
        {"source": "live", "generated_at": "2026-11-16 10:00", "match_date": "2026-11-22", "league": c["league"],
         "home_team": "H1", "away_team": "A1", "side": "OVER", "signal_tier": "QUALIFIED", "result": "WIN"},
        {"source": "live", "generated_at": "2026-11-15 10:00", "match_date": "2026-11-20", "league": c["league"],
         "home_team": "H2", "away_team": "A2", "side": "OVER", "signal_tier": "QUALIFIED", "result": "WIN"},
        {"source": "live", "generated_at": "2026-11-15 10:00", "match_date": "2026-11-23", "league": c["league"],
         "home_team": "H3", "away_team": "A3", "side": "OVER", "signal_tier": "AVOID", "result": ""},
        {"source": "backfill", "generated_at": "2026-11-15 10:00", "match_date": "2026-11-23", "league": c["league"],
         "home_team": "H4", "away_team": "A4", "side": "OVER", "signal_tier": "QUALIFIED", "result": ""},
        {"source": "live", "generated_at": "2026-11-15 10:00", "match_date": "2026-11-23", "league": "Not A Cell League",
         "home_team": "H5", "away_team": "A5", "side": "OVER", "signal_tier": "QUALIFIED", "result": ""},
    ]
    return pd.DataFrame(rows)


def test_universe_keeps_only_live_qualified_cell_tips_from_the_start_date_once():
    picks, n = SP.tip_universe(_bets(), None, P)
    assert list(picks["home"]) == ["H1"] and picks["generated_at"].iloc[0] == "2026-11-15 10:00"
    assert picks["mside"].iloc[0] == "over25_over" and picks["line"].iloc[0] == 2.5


def test_side_ledger_over15_over35_map_to_their_lines():
    c15 = _cell("over15_over")
    s = pd.DataFrame([{"source": "live", "generated_at": "2026-11-21 01:00", "match_date": "2026-11-24",
                       "league": c15["league"], "home_team": "H", "away_team": "A", "market": "over15",
                       "side": "YES", "signal_tier": "QUALIFIED", "result": ""}])
    picks, _ = SP.tip_universe(None, s, P)
    assert picks["line"].iloc[0] == 1.5 and picks["side_sign"].iloc[0] == 1


# ── signal and paper ledger ─────────────────────────────────────────────────────────────────

def _rows(slot, key, pin_over, cons_over, lead, n_books=4, line=2.5):
    rows = [{"pick_key": key, "slot": slot, "lead_h": lead, "bookmaker": "pinnacle", "line": line,
             "over": pin_over, "under": round(1 / (1.04 - 1 / pin_over), 3)}]
    for i in range(n_books):
        rows.append({"pick_key": key, "slot": slot, "lead_h": lead, "bookmaker": f"b{i}", "line": line,
                     "over": cons_over, "under": round(1 / (1.06 - 1 / cons_over), 3)})
    return rows


def _capture(p, pin6=1.70, cons6=1.70, close=1.65):
    key = f"{p.match_date}|{p.league}|{p.home}|{p.away}"
    return pd.DataFrame(_rows("T72", key, 2.00, 2.00, 72.0) + _rows("T6", key, pin6, cons6, 6.0)
                        + _rows("CLOSE", key, close, close, 0.1))


def test_a_large_move_toward_the_pick_is_agree_and_the_bet_price_is_t6_pinnacle():
    picks, _ = SP.tip_universe(_bets(), None, P)
    paper = SP.build_paper(picks, _capture(picks.iloc[0]), P)
    r = paper.iloc[0]
    assert r["cls_CONS"] == "AGREE" and r["cls_PIN"] == "AGREE" and r["cls_BOTH"] == "AGREE"
    assert r["bet_odds"] == 1.70 and r["close_odds"] == 1.65 and r["clv"] == pytest.approx(1.70 / 1.65 - 1)
    assert r["status"] == "SETTLED" and r["pnl"] == pytest.approx(0.70)


def test_under_pick_on_an_over_move_is_disagree():
    b = _bets()
    b.loc[0, "side"] = "UNDER"
    c = _cell("over25_under")
    b.loc[0, "league"] = c["league"]
    picks, _ = SP.tip_universe(b.iloc[[0]], None, P)
    paper = SP.build_paper(picks, _capture(picks.iloc[0]), P)
    assert paper["cls_CONS"].iloc[0] == "DISAGREE"


def test_too_few_books_or_no_threshold_is_no_data_never_a_stand_in():
    picks, _ = SP.tip_universe(_bets(), None, P)
    p = picks.iloc[0]
    key = f"{p.match_date}|{p.league}|{p.home}|{p.away}"
    cap = pd.DataFrame(_rows("T72", key, 2.0, 2.0, 72, n_books=2) + _rows("T6", key, 1.7, 1.7, 6, n_books=2))
    assert SP.build_paper(picks, cap, P)["cls_CONS"].iloc[0] == "NO_DATA"
    assert SP.classify({"pin": 0.5, "cons": 0.5}, "Unknown League", 1, P) == \
        {"PIN": "NO_DATA", "CONS": "NO_DATA", "BOTH": "NO_DATA"}


def test_no_pinnacle_line_at_t6_is_no_price_and_close_is_never_the_bet():
    picks, _ = SP.tip_universe(_bets(), None, P)
    cap = _capture(picks.iloc[0])
    cap = cap[~((cap["slot"] == "T6") & (cap["bookmaker"] == "pinnacle"))]
    r = SP.build_paper(picks, cap, P).iloc[0]
    assert r["status"] == "NO_PRICE" and pd.isna(r["bet_odds"]) and pd.isna(r["pnl"])


def test_summary_carries_no_price():
    picks, _ = SP.tip_universe(_bets(), None, P)
    s = SP.summarize(SP.build_paper(picks, _capture(picks.iloc[0]), P), P)
    txt = json.dumps(s)
    assert "bet_odds" not in txt and "close_odds" not in txt and "1.7" not in txt


# ── capture parsing, slots, budget ──────────────────────────────────────────────────────────

def test_parse_keeps_exact_two_sided_lines_before_kickoff_only():
    body = {"commence_time": "2026-11-22T15:00:00Z", "bookmakers": [
        {"key": "pinnacle", "markets": [
            {"key": "totals", "outcomes": [{"name": "Over", "point": 2.5, "price": 1.9},
                                           {"name": "Under", "point": 2.5, "price": 1.95}]},
            {"key": "alternate_totals", "outcomes": [{"name": "Over", "point": 2.75, "price": 2.1},
                                                     {"name": "Under", "point": 2.75, "price": 1.75},
                                                     {"name": "Over", "point": 3.5, "price": 3.0}]}]},
        {"key": "b1", "markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": 1.5, "price": 1.01},
                                                                 {"name": "Under", "point": 1.5, "price": 9.0}]}]}]}
    rows, st = SP.parse_event_odds(body, pd.Timestamp("2026-11-22T09:00:00Z"))
    assert [(r["bookmaker"], r["line"]) for r in rows] == [("pinnacle", 2.5)]
    assert st["one_sided"] == 2 and st["low_price"] == 1
    rows, st = SP.parse_event_odds(body, pd.Timestamp("2026-11-22T15:00:00Z"))
    assert rows == [] and st["post_kickoff"] == 1


def test_slots_are_due_only_inside_their_windows_and_once():
    ko = pd.Timestamp("2026-11-22T15:00:00Z")
    assert SP.due_slots(ko, ko - pd.Timedelta(hours=70), set(), P) == ["T72"]
    assert SP.due_slots(ko, ko - pd.Timedelta(hours=70), {"T72"}, P) == []
    assert SP.due_slots(ko, ko - pd.Timedelta(hours=5), set(), P) == ["T6"]
    assert SP.due_slots(ko, ko - pd.Timedelta(minutes=10), set(), P) == ["CLOSE"]
    assert SP.due_slots(ko, ko - pd.Timedelta(hours=30), set(), P) == []
    # short 2-hourly runs (amendment 2026-11-15): T6 is 3 h wide, CLOSE is the run inside the last 2 h
    assert SP.due_slots(ko, ko - pd.Timedelta(hours=6.5), set(), P) == ["T6"]
    assert SP.due_slots(ko, ko - pd.Timedelta(hours=1.5), set(), P) == ["CLOSE"]
    assert SP.due_slots(ko, ko - pd.Timedelta(minutes=2), set(), P) == []
    lo = min(w["lead_h"][1] - w["lead_h"][0] for w in P["capture"]["slots"].values())
    assert lo >= 1.95                                   # every window at least one 2-hourly cadence wide
    assert SP.due_slots(ko, ko + pd.Timedelta(minutes=1), set(), P) == []


def test_budget_never_spends_below_the_floor_or_over_the_caps(tmp_path):
    m = _capture_mod()
    b = m.Budget(tmp_path, floor=100_000, run_cap=10, day_cap=100, month_cap=1000)
    assert not b.can_spend(2)                                   # remaining unknown: no paid call
    b.remaining = 100_001
    assert not b.can_spend(2) and "floor" in b.stopped
    b.remaining = 300_000
    assert b.can_spend(2)
    b.book(10)
    assert not b.can_spend(2) and "run cap" in b.stopped
    b2 = m.Budget(tmp_path, floor=100_000, run_cap=10_000, day_cap=4, month_cap=1000)
    b2.remaining = 300_000
    b2.book(4)
    assert not b2.can_spend(2) and "day cap" in b2.stopped


def test_match_events_is_one_to_one_inside_the_day_window():
    fx = pd.DataFrame([{"match_date": "2026-11-22", "league": "L", "home": "Alpha FC", "away": "Beta"}])
    ev = [{"id": "e1", "home_team": "Alpha", "away_team": "Beta", "commence_time": "2026-11-22T15:00:00Z"},
          {"id": "e2", "home_team": "Alpha", "away_team": "Beta", "commence_time": "2026-11-26T15:00:00Z"}]
    m, st = SP.match_events(fx, ev)
    assert list(m.values())[0]["id"] == "e1" and st["matched"] == 1
    m, st = SP.match_events(fx, ev + [{**ev[0], "id": "e3"}])
    assert m == {} and st["ambiguous"] == 1


# ── owner 2026-11-15 "Capture stops at 100k" ─────────────────────────────────────────────────

def test_capture_floor_is_100k_refuses_below_and_accepts_at_or_above(tmp_path):
    m = _capture_mod()
    assert m.CAPTURE_MIN_FLOOR == 100_000
    with pytest.raises(SystemExit):
        m.main(["--store", str(tmp_path / "s"), "--ledgers", str(tmp_path), "--mode", "evaluate", "--floor", "99999"])
    for f in ("100000", "290000"):
        assert m.main(["--store", str(tmp_path / f), "--ledgers", str(tmp_path), "--mode", "evaluate",
                       "--floor", f]) == 0
    b = m.Budget(tmp_path, floor=m.CAPTURE_MIN_FLOOR, run_cap=100, day_cap=100, month_cap=100)
    b.remaining = 150_000
    assert b.can_spend(2)                                     # between 100k and the history's 290k: allowed
    b.remaining = 100_001
    assert not b.can_spend(2) and "floor" in b.stopped


def test_the_history_collectors_floor_is_not_the_captures():
    """The capture's floor is its own constant; the OddsAPI history collector keeps MIN_FLOOR 260,000 and
    refuses a lower floor (checked here when the collector is on this tree; its own suite
    tests/test_oddsapi_history.py proves the refusal on prep/oddsapi-history)."""
    m = _capture_mod()
    src = (ROOT / "scripts" / "sharp_paper_capture.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+[\w. ]*oddsapi_history", src, re.M)   # never imports its floor
    oh = ROOT / "src" / "oddsapi_history.py"
    if not oh.exists():
        pytest.skip("src/oddsapi_history.py is not on this branch (prep/oddsapi-history carries it)")
    from src import oddsapi_history as OH
    assert OH.MIN_FLOOR == 260_000 and m.CAPTURE_MIN_FLOOR < OH.MIN_FLOOR
    with pytest.raises(ValueError):
        OH.Meter("k", floor=OH.MIN_FLOOR - 1, run_cap=10, max_seconds=60)
