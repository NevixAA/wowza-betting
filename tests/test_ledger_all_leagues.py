"""Ledger-only predictions for the TRAINING_ONLY Standard leagues (src/ledger_all_leagues.py).

Owner 2026-11-19: predict every match of the ten TRAINING_ONLY leagues into the prediction ledger, in
every market the Standard models serve - NO tip, NO Telegram, NO stake, NO TOP-3 for them."""
from __future__ import annotations

import hashlib
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import ledger_all_leagues as LA
from src import prediction_ledger as PL

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 19, 21, 0, tzinfo=timezone.utc)
OWNER_TEN = {"Scottish Premiership", "Scottish Championship", "Scottish League One", "Scottish League Two",
             "Belgian First Division A", "Dutch Eredivisie", "Greek Super League", "Portuguese Primeira Liga",
             "Turkish Super Lig", "National League"}
P_COLS = ("p_over25", "p_over15", "p_over35", "p_btts", "p_ht_over05", "p_ht_over15")


# ── scope: exactly the owner's ten, and none of them can reach a bet path ─────────────────────────

def test_ledger_only_leagues_are_exactly_the_owner_ten():
    got = {e["name"] for e in LA.ledger_only_leagues()}
    assert got == OWNER_TEN
    for e in LA.ledger_only_leagues():
        assert e["production_state"] == "TRAINING_ONLY" and e["model_type"] == "standard"


def test_ledger_only_leagues_are_never_on_the_board_or_the_send_path():
    """The OddsAPI board (src.predict._fetch_odds_api) fetches config.ENABLED_LEAGUES only, and every
    tip / Telegram / stake / TOP-3 file is derived from that board - so a TRAINING_ONLY league can never
    produce a tip. Pinned here so a registry edit cannot silently send one."""
    import config
    for lg in OWNER_TEN:
        assert lg not in config.ENABLED_LEAGUES
        assert lg not in getattr(config, "PAPER_LEAGUES", set())
        assert lg in config.STANDARD_FORMAT_LEAGUES          # in-distribution for the Standard models
    src = (ROOT / "src" / "predict.py").read_text(encoding="utf-8")
    assert "if league not in config.ENABLED_LEAGUES:" in src


def test_no_bet_path_reads_the_ledger_only_outputs():
    """By construction: nothing but the writer itself (and its parity script / tests) refers to the
    ledger-only module or its board file. bets.csv, side_bets.csv, the TOP-3 ledger, the stake plan,
    Telegram and the dashboard tips are built from output/predictions.csv, never from these."""
    pat = re.compile(r"(import|from)\s+[\w.]*ledger_all_leagues|ledger_all_leagues\s+import|"
                     r"src\.ledger_all_leagues(?!\.py)|output/ledger_only|ledger_only/board")
    allowed = {Path("src/ledger_all_leagues.py"), Path("scripts/ledger_only_parity.py"),
               Path("tests/test_ledger_all_leagues.py")}
    hits = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in (".git", "output", "_v9", ".venv") or rel in allowed:
            continue
        try:
            if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
                hits.append(str(rel))
        except OSError:
            continue
    assert hits == []


def test_no_served_model_feature_needs_a_network_enrichment():
    """The writer blocks the network; that is safe only because no served payload reads a column the
    API enrichments of build_upcoming_features fill (they are display columns)."""
    import config
    from src.model import load_models
    enriched = {"key_attacker_absent_home", "key_attacker_absent_away", "season_stage_ratio", "is_late_season",
                "home_coach_tenure_days", "home_coach_is_caretaker", "away_coach_tenure_days",
                "away_coach_is_caretaker", "api_implied_btts", "api_implied_over35", "api_implied_over15",
                "api_implied_draw", "home_attack_formation", "away_attack_formation", "combined_attack_intent",
                "home_forward_count", "away_forward_count", "h2h_over25_rate", "h2h_avg_goals",
                "h2h_home_win_rate", "h2h_n", "api_implied_over25", "api_overround", "implied_prob_over",
                "implied_prob_under", "bookmaker_overround", "home_table_pos", "away_table_pos", "home_gd_pg",
                "away_gd_pg", "home_form_pts", "away_form_pts", "home_win_rate_h", "away_win_rate_a",
                "table_pos_gap", "referee_foul_avg"}
    files = [config.MODEL_FILE_STANDARD, config.HT_MODEL_FILE_05, config.HT_MODEL_FILE_15,
             *config.SIDE_MARKETS.values()]
    for f in files:
        if not Path(f).exists():
            continue
        p = load_models(model_file=f)
        cols = set(p.get("feature_cols") or [])
        for t in (p.get("tracks") or {}).values():
            if isinstance(t, dict):
                cols |= set(t.get("feature_cols") or [])
        assert not {c.replace("__missing", "") for c in cols} & enriched, Path(f).name


# ── identity: proven by results, never guessed ─────────────────────────────────────────────────────

def _leagues():
    return [{"name": "Scottish Premiership", "api_football_id": 179, "production_state": "TRAINING_ONLY",
             "model_type": "standard"}]


def _af(rows):
    return pd.DataFrame(rows, columns=["league_id", "date", "home_id", "home_name", "away_id", "away_name",
                                       "ft_home", "ft_away", "status"])


def _hist(rows):
    return pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals"])


def test_identity_needs_unique_key_evidence_and_agreement():
    days = ["2025-08-02", "2025-08-09", "2025-08-16", "2025-08-23"]
    af = _af([[179, d, 1, "Heart Of Midlothian", 2, "ST Mirren", i, 0, "FT"] for i, d in enumerate(days)])
    h = _hist([["Scottish Premiership", d, "Hearts", "St Mirren", i, 0] for i, d in enumerate(days)])
    ok, why = LA.identity_map(h, af, _leagues(), since="2025-01-01")
    assert ok == {("Scottish Premiership", 1): "Hearts", ("Scottish Premiership", 2): "St Mirren"}
    # a second same-score match the same day makes that day's key ambiguous: dropped, not guessed
    h2 = pd.concat([h, _hist([["Scottish Premiership", days[-1], "Celtic", "Rangers", 3, 0]])])
    ok2, why2 = LA.identity_map(h2, af, _leagues(), since="2025-01-01")
    assert ("Scottish Premiership", 1) in ok2                 # 3 unambiguous matches remain
    ok3, why3 = LA.identity_map(h2, af, _leagues(), since="2025-08-10")
    assert ("Scottish Premiership", 1) not in ok3 and "identity not proven" in why3[("Scottish Premiership", 1)]


def test_identity_refuses_disagreement_and_a_club_claimed_twice():
    days = ["2025-08-02", "2025-08-09", "2025-08-16"]
    af = _af([[179, d, 1, "A", 2, "B", i, 1, "FT"] for i, d in enumerate(days)]
             + [[179, "2025-08-30", 3, "C", 4, "D", 5, 5, "FT"], [179, "2025-09-06", 3, "C", 4, "D", 6, 5, "FT"],
                [179, "2025-09-13", 3, "C", 4, "D", 7, 5, "FT"]])
    h = _hist([["Scottish Premiership", d, "Alpha" if i < 2 else "Beta", "Bravo", i, 1] for i, d in enumerate(days)]
              + [["Scottish Premiership", d, "Bravo", "Delta", g, 5]
                 for d, g in (("2025-08-30", 5), ("2025-09-06", 6), ("2025-09-13", 7))])
    ok, why = LA.identity_map(h, af, _leagues(), since="2025-01-01")
    assert ("Scottish Premiership", 1) not in ok and "name" in why[("Scottish Premiership", 1)]
    # ids 2 and 3 both prove 'Bravo' -> refused for both
    assert ("Scottish Premiership", 2) not in ok and ("Scottish Premiership", 3) not in ok
    assert "claimed by more than one" in why[("Scottish Premiership", 3)]
    assert ok[("Scottish Premiership", 4)] == "Delta"


# ── the writer: ledger rows only, idempotent, flag-gated ─────────────────────────────────────────────

@pytest.fixture
def world(tmp_path, monkeypatch):
    up = pd.DataFrame({"fixture_id": [9001, 9002, 9003], "league_id": [179, 179, 40],
                       "season": [2026] * 3,
                       "kickoff": ["2026-11-21T15:00:00+00:00", "2026-11-21T20:00:00+00:00",
                                   "2026-11-21T15:00:00+00:00"],
                       "kickoff_ts": [0, 0, 0], "status": ["NS", "NS", "NS"], "home_id": [1, 3, 50],
                       "away_id": [2, 4, 51], "fetched_at": [""] * 3})
    upath = tmp_path / "upcoming.parquet"
    up.to_parquet(upath)
    fpath = tmp_path / "fixtures.parquet"
    pd.DataFrame({"fixture_id": [1], "league_id": [179], "date": ["2026-11-08"], "home_id": [1],
                  "home_name": ["Heart Of Midlothian"], "away_id": [2], "away_name": ["ST Mirren"],
                  "ft_home": [1], "ft_away": [0], "status": ["FT"]}).to_parquet(fpath)
    monkeypatch.setattr(LA, "identity_map", lambda h, f, lg: (
        {("Scottish Premiership", 1): "Hearts", ("Scottish Premiership", 2): "St Mirren"},
        {("Scottish Premiership", 3): "identity not proven: test"}))
    calls = []

    def fake_score(board, historical, payloads=None):
        calls.append(len(board))
        out = board.copy()
        out["model_type"] = "standard"
        ok = out["not_scored_reason"] == ""
        for i, c in enumerate(P_COLS):
            out[c] = np.where(ok, 0.3 + 0.05 * i, np.nan)
        for c in ("model_version", "model_version_over15", "model_version_over35", "model_version_btts",
                  "model_version_ht_over05", "model_version_ht_over15"):
            out[c] = "vtest"
        return out
    monkeypatch.setattr(LA, "score", fake_score)
    monkeypatch.setattr(LA, "_versions", lambda: {c: "vtest" for c in (
        "model_version", "model_version_over15", "model_version_over35", "model_version_btts",
        "model_version_ht_over05", "model_version_ht_over15")})
    return {"tmp": tmp_path, "up": upath, "fx": fpath, "calls": calls,
            "kw": dict(upcoming_path=upath, fixtures_path=fpath, root=tmp_path / "L",
                       board_path=tmp_path / "board.csv")}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def test_run_writes_ledger_rows_only_and_is_idempotent(world):
    protected = [ROOT / "output" / f for f in ("predictions.csv", "bets.csv", "side_bets.csv", "stake_plan.csv",
                                                "top3_league_ledger.csv", "ht_ledger.csv")]
    before = {p: _sha(p) for p in protected}
    hist = pd.DataFrame({"league": [], "date": []})
    s = LA.run(NOW, historical=hist, **world["kw"])
    assert s["fixtures_in_window"] == 2                       # league 40 (Championship) is not ledger-only
    assert s["append"]["written"] == 2 * 6 * 2 and s["append"]["refused"] == 0
    assert {p: _sha(p) for p in protected} == before
    root = world["kw"]["root"]
    parts = PL.partitions(root)
    assert parts and all(PL.parse_partition(root, p)["model_id"] == "v9_standard" for p in parts)
    assert {PL.parse_partition(root, p)["league_slug"] for p in parts} == {"scottish_premiership"}
    d = pd.concat([PL.read_partition(p) for p in parts])
    assert set(d["fixture_ref"]) == {"af:9001", "af:9002"} and set(d["fixture_id"]) == {"9001", "9002"}
    assert d["payload_version"].str.endswith("ledger_only:TRAINING_ONLY").all()
    assert set(d["market"]) == {"ou25", "ou15", "ou35", "btts", "ht_ou05", "ht_ou15"}
    a = d[d["fixture_ref"] == "af:9001"]
    assert (a["scored"].astype(str) == "1").all() and set(a["home_team"]) == {"Hearts"}
    b = d[d["fixture_ref"] == "af:9002"]                       # identity not proven -> unscored, with reason
    assert (b["scored"].astype(str) == "0").all()
    assert b["not_scored_reason"].str.contains("identity not proven").all()
    assert PL.validate(root) == []
    board = pd.read_csv(world["kw"]["board_path"])
    assert not [c for c in board.columns if PL.PRICE_LIKE.search(c)]
    assert set(board["league_state"]) == {"TRAINING_ONLY"}
    # second run: every due snapshot already stored -> skipped before any history load
    s2 = LA.run(NOW, historical=None, **world["kw"])
    assert s2.get("skipped") == "every due snapshot is already in the ledger"
    assert world["calls"] == [2]


def test_final_snapshot_is_added_near_kickoff(world):
    LA.run(NOW, historical=pd.DataFrame(), **world["kw"])
    late = datetime(2026, 11, 21, 13, 30, tzinfo=timezone.utc)   # 90 min before the 15:00 kickoff
    s = LA.run(late, historical=pd.DataFrame(), **world["kw"])
    assert s["append"]["written"] == 6 * 2                     # af:9001 final (af:9002 kicks off at 20:00)
    d = pd.concat([PL.read_partition(p) for p in PL.partitions(world["kw"]["root"])])
    assert set(d.loc[d["snapshot"] == "final", "fixture_ref"]) == {"af:9001"}


def test_flag_off_writes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(LA.FLAG_ENV, raising=False)
    monkeypatch.setenv(PL.FLAG_ENV, "true")
    monkeypatch.setattr(LA, "run", lambda *a, **k: pytest.fail("run() must not be called with the flag off"))
    assert LA.main(["board"]) == 0
    assert "nothing written" in capsys.readouterr().out
    monkeypatch.setenv(LA.FLAG_ENV, "true")
    monkeypatch.setenv(PL.FLAG_ENV, "false")
    assert LA.main(["board"]) == 0


def test_score_uses_the_board_serving_path_with_the_network_blocked(monkeypatch):
    seen = {}

    def fake_serve(upcoming, historical, payload, payload_newformat=None, side_payloads=None, *, record_health=True):
        seen["record_health"] = record_health
        with pytest.raises(OSError):
            socket.create_connection(("example.com", 443), timeout=1)
        out = upcoming.copy()
        out["model_type"] = "standard"
        out["p_over25"] = 0.5
        return out
    import src.predict as P
    monkeypatch.setattr(P, "serve_fixtures", fake_serve)
    board = pd.DataFrame({"league": ["Scottish Premiership"] * 2, "league_state": ["TRAINING_ONLY"] * 2,
                          "date": [pd.Timestamp("2026-11-21")] * 2,
                          "kickoff_utc": [pd.Timestamp("2026-11-21 15:00")] * 2,
                          "home_team": ["Hearts", "X"], "away_team": ["St Mirren", "Y"],
                          "odds_over25": [np.nan] * 2, "odds_under25": [np.nan] * 2, "fixture_id": [1, 2],
                          "not_scored_reason": ["", "home team: identity not proven"]})
    out = LA.score(board, pd.DataFrame(), payloads=({}, None, {}))
    assert seen == {"record_health": False}
    assert out.loc[out["fixture_id"] == 1, "p_over25"].iloc[0] == 0.5
    assert pd.isna(out.loc[out["fixture_id"] == 2, "p_over25"].iloc[0])
    # the network is back after the block
    assert socket.create_connection is not None and "_no_net" not in socket.create_connection.__name__
