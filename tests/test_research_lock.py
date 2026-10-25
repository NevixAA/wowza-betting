"""Non-regression lock: research may never move production state, thresholds, stakes or eligibility
without a deliberate re-approval (owner rule 2026-10-25)."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src import research_lock as L

ROOT = Path(__file__).resolve().parents[1]
SNAP = json.loads(L.SNAPSHOT_PATH.read_text(encoding="utf-8"))
REG = json.loads(L.REGISTRY_PATH.read_text(encoding="utf-8"))
POLICY = json.loads(L.POLICY_PATH.read_text(encoding="utf-8"))

PRODUCTION_FILES = ["config/bankroll_policy.json", "config.py", ".github/workflows/predict.yml",
                    "models/best_params_standard.json", "models/best_params_side_markets.json",
                    "output/league_thresholds.json", "output/stake_plan.csv", "config/auto_leagues.json"]
RESEARCH_MODULES = ["scripts/build_league_regimes.py", "src/research_lock.py", "src/gameweek.py"]
FORBIDDEN_IMPORTS = ("pipeline", "src.edge_engine", "src.stake_plan", "src.betting", "src.predict", "src.ledger",
                     "src.cloudbet", "telegram_bot", "src.allocation", "src.portfolio", "retrain", "update_results")


def test_committed_snapshot_matches_the_repo():
    r = L.check()
    assert r["protected_tracks_modified"] is False, r["changes"]
    assert SNAP["approved_by"] and SNAP["approved_at"] and SNAP["fingerprint_sha256"] == L.digest(SNAP["fingerprint"])


def test_snapshot_covers_every_protected_track_and_the_global_protections():
    assert sorted(SNAP["fingerprint"]["protected_tracks"]) == L.protected_keys(REG)
    g = SNAP["fingerprint"]["global"]
    assert g["bankroll_policy"]["tracks"]["standard|OU25"]["state"] == "PAPER"
    assert g["bankroll_policy"]["tracks"]["player_props"]["state"] == "BLOCKED_FROM_REAL"
    assert g["execution"]["workflows_setting_CLOUDBET_MODE"] == []
    assert g["deployed_env"]["MARKSMAN_THRESHOLD"] == "0.08"


def test_changed_deployed_threshold_is_detected_and_named():
    env = dict(SNAP["fingerprint"]["global"]["deployed_env"], MARKSMAN_THRESHOLD="0.09")
    r = L.check(env=env)
    assert r["protected_tracks_modified"] is True
    assert any("deployed_env.MARKSMAN_THRESHOLD" in c for c in r["changes"])
    assert any("thresholds.MARKSMAN_THRESHOLD: 0.08 -> 0.09" in c for c in r["changes"])


def test_changed_state_or_stake_is_detected():
    p = copy.deepcopy(POLICY)
    p["tracks"]["standard|OU25"]["state"] = "ACTIVE"
    r = L.check(policy=p)
    assert r["protected_tracks_modified"] and any("tracks.standard|OU25.state" in c for c in r["changes"])
    p = copy.deepcopy(POLICY)
    p["tracks"]["new_format|BTTS"]["base_fraction"] = 0.015
    r = L.check(policy=p)
    assert any("new_format|BTTS.base_fraction" in c for c in r["changes"])
    if SNAP["fingerprint"]["protected_tracks"]:
        assert any(c.startswith("protected_tracks.") and c.endswith("0.0125 -> 0.015") for c in r["changes"])


def test_missing_snapshot_fails_closed():
    assert L.check(snapshot={})["protected_tracks_modified"] is True


def test_newly_protected_track_is_reported_not_counted_as_a_production_change():
    reg = copy.deepcopy(REG)
    e = next(x for x in reg["entries"] if x["research_regime"] != "PROTECTED")
    e["research_regime"] = "PROTECTED"
    r = L.check(registry=reg)
    assert r["protected_tracks_modified"] is False and r["unapproved_new_protected_tracks"] == [e["key"]]


def test_deployed_env_ignores_secrets_and_unknown_keys():
    yml = 'env:\n  ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}\n  VALUABLE_THRESHOLD: "0.03"  # c\n  FOO: "1"\n'
    assert L.deployed_env(yml, ["ODDS_API_KEY", "VALUABLE_THRESHOLD"]) == {"VALUABLE_THRESHOLD": "0.03"}


def test_research_code_never_imports_production_decision_modules():
    for rel in RESEARCH_MODULES:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        names = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                names += [n.module] + [f"{n.module}.{a.name}" for a in n.names]
        bad = [m for m in names if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_IMPORTS)]
        assert not bad, (rel, bad)


def _hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in PRODUCTION_FILES if (ROOT / f).exists()}


def test_research_code_path_never_writes_production_files(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("blr", ROOT / "scripts" / "build_league_regimes.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    writes = []
    real_write = Path.write_text
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **k: (writes.append(str(self)), real_write(self, *a, **k))[1])
    before = _hashes()
    af = pd.DataFrame({"fx": ["a"], "league": ["Championship"], "model_type": ["standard"], "market": ["ou25"],
                       "side": ["OVER"], "kick": [pd.Timestamp("2026-10-01T15:00Z")], "p_model": [0.6],
                       "odds": [1.9], "opp_odds": [1.9], "p_market_fair": [0.5], "prod_tier": ["SNIPER"],
                       "close_status": ["VALID_PRE_KICKOFF_CLOSE"], "gap_h": [0.2], "clv": [1.0],
                       "outcome_yes": [1.0], "won": [1.0], "pnl": [0.9], "md": [pd.Timestamp("2026-10-01")]})
    B.build_registry(af, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), B.json.loads(L.POLICY_PATH.read_text(
        encoding="utf-8")), "2026-10-25")
    L.check()
    L.approve("test", path=tmp_path / "snap.json")
    assert _hashes() == before
    assert writes == [str(tmp_path / "snap.json")]
    with pytest.raises(SystemExit):        # the builder refuses to write any other config/ file
        B.main(["--pro", str(tmp_path), "--out", str(ROOT / "config" / "bankroll_policy.json")])
    assert _hashes() == before
