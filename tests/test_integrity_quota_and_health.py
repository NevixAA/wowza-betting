"""R2-C: OddsAPI quota accounting (finding 23) and the model_integrity_health skeleton.

    python -m pytest tests/test_integrity_quota_and_health.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.integrity import oddsapi_quota as Q  # noqa: E402

LOG = """2026-11-01T09:26:40.1Z INFO [odds_fetcher] /v4/x OK | quota remaining: 50000
2026-11-01T09:26:42.1234567Z INFO:src.btts_odds:[btts] 10 call(s), 20 credit(s) spent, filled 9/30 fixtures (skipped 20 within 2h refresh, 0 no event id, 0 errors; near-kickoff window 6h)
2026-11-01T09:26:42.2Z INFO:src.btts_odds:[oddsapi_usage] {"calls": 10, "credits": 20, "credits_by_league": {"USA MLS": 12, "Serie B": 8}, "credits_by_market": {"btts": 20}, "fixtures_considered": 30, "suppressed_fresh": 20, "requests_remaining": 49980}
2026-11-01T21:26:40.1Z INFO [odds_fetcher] /v4/y OK | quota remaining: 48000
"""


def test_parse_and_health():
    r = Q.parse_log_text(LOG, "predict.yml", 1)
    assert r["usage"] and r["btts"] and len(r["remaining"]) == 3
    h = Q.build_health([r], 24, now=datetime(2026, 11, 2, tzinfo=timezone.utc))
    assert h["by_workflow"]["predict.yml"]["credits"] == 20          # structured line wins, no double count
    assert h["by_league"] == {"USA MLS": 12, "Serie B": 8}
    assert h["btts_duplicate_fetch_suppression_rate"] == pytest.approx(20 / 30, abs=1e-4)
    assert h["account_remaining_latest"] == 48000
    assert h["account_burn_per_day_from_remaining"] == 4000           # 2000 spent over 12h
    assert h["status"] == "PASS" and "OddsAPI quota PASS" in h["health_check_line"]


def test_old_btts_line_without_credits_is_estimated_and_flagged():
    old = "2026-10-20T10:00:00Z INFO [btts] 5 call(s), filled 5/9 fixtures (skipped 4 within 2h refresh, 0 no event id)"
    r = Q.parse_log_text(old, "predict.yml")
    h = Q.build_health([r], 24)
    assert h["by_workflow"]["predict.yml"]["runs_with_unknown_credits"] == 1
    assert h["by_workflow"]["predict.yml"]["credits"] == 10


def test_thresholds_warn_and_fail_before_exhaustion():
    mk = lambda rem: {"workflow": "w", "usage": [], "btts": [], "exchange_credits": 0,
                      "remaining": [("2026-11-01T00:00:00Z", rem + 3000), ("2026-11-01T12:00:00Z", rem)]}
    assert Q.build_health([mk(60000)], 24)["status"] == "PASS"
    assert Q.build_health([mk(20000)], 24)["status"] == "WARN"           # < 25% of 100k
    assert Q.build_health([mk(4000)], 24)["status"] == "FAIL"
    assert Q.build_health([{"workflow": "w", "usage": [], "btts": [], "exchange_credits": 0,
                            "remaining": []}], 24)["status"] == "UNKNOWN"   # never healthy unmeasured


def test_monthly_reset_jump_is_not_counted_as_spend():
    r = {"workflow": "w", "usage": [], "btts": [], "exchange_credits": 0,
         "remaining": [("2026-10-31T12:00:00Z", 500), ("2026-11-01T00:00:00Z", 100000),
                       ("2026-11-01T12:00:00Z", 99000)]}
    assert Q.build_health([r], 24)["account_burn_per_day_from_remaining"] == 1000


def test_btts_fetch_emits_parseable_usage_line(tmp_path, monkeypatch, caplog):
    import config
    from src import btts_odds

    class _R:
        status_code = 200
        headers = {"x-requests-last": "2", "x-requests-remaining": "777"}

        def json(self):
            return {"bookmakers": []}
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "ODDS_API_KEY", "SECRETKEYVALUE", raising=False)
    monkeypatch.setattr(btts_odds, "append_book_rows", lambda rows: None)
    monkeypatch.setattr(btts_odds.requests, "get", lambda *a, **k: _R())
    rows = [{"home_team": "H", "away_team": "A", "league": "USA MLS", "date": "2026-11-05",
             "kickoff_utc": "2026-11-05T20:00:00Z", "_oa_event_id": "e", "_oa_sport_key": "s"}]
    with caplog.at_level(logging.INFO, logger="src.btts_odds"):
        btts_odds.fetch(rows)
    line = next(r.getMessage() for r in caplog.records if "[oddsapi_usage]" in r.getMessage())
    rec = json.loads(line.split("[oddsapi_usage] ", 1)[1])
    assert rec["credits"] == 2 and rec["credits_by_league"] == {"USA MLS": 2}
    assert rec["requests_remaining"] == 777 and "SECRETKEYVALUE" not in line
    parsed = Q.parse_log_text("2026-11-01T00:00:00Z " + line, "predict.yml")
    assert parsed["usage"][0]["credits"] == 2
    assert (tmp_path / "btts_fetch_clock.json").exists()             # clock still written


def test_model_integrity_health_skeleton(tmp_path):
    from src.integrity import model_health as MH
    doc = MH.build(ROOT)
    need = {"model_type", "market", "recipe", "serving_version", "training_cutoff", "code_sha",
            "data_hash", "identity_version", "feature_recipe_version", "imputer_version",
            "season_resolver_version", "league_registry_version", "chronology", "parity",
            "batch_invariance", "weight_alignment", "holdout_independence", "price_provenance",
            "missing_feature_policy", "serving_fallback_policy", "status"}
    for k, t in doc["tracks"].items():
        if k.startswith("props|"):
            continue
        assert need <= set(t), (k, need - set(t))
        if t["unknown_checks"]:
            assert t["status"] != "PASS"                       # UNKNOWN never counts as PASS
    assert doc["tracks"]["standard|OU25"]["recipe"] in ("v9_repaired", "UNKNOWN")
