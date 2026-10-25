"""Data contracts (src/contracts, 2026-10-25): the validator never raises, catches each kind of
violation, and the shadow hooks observe without changing what writers write."""
import json
import os

import pandas as pd

from src.contracts import SCHEMAS, committed_violations, validate
from src.contracts.schema import Col, Rule, Schema
from src.contracts.shadow import observe

S = Schema("T", "9", "test", columns=(
    Col("a", nullable=False), Col("p", "float", min=0, max=1), Col("k", allowed=("X", "Y")),
    Col("t", "datetime"), Col("n", "int", min=0), Col("h", pattern=r"[0-9a-f]{4}"),
    Col("opt", required=False)),
    key=("a",), unique_key=True,
    rules=(Rule("p_gt_half_needs_k", lambda d: (pd.to_numeric(d["p"], errors="coerce") > 0.5) & (d["k"] == ""),
                "doc", ("p", "k")),))


def _kinds(v):
    return sorted((x["kind"], x["column"] if isinstance(x["column"], str) else str(x["column"])) for x in v)


def test_clean_frame_has_no_violations():
    d = pd.DataFrame({"a": ["1", "2"], "p": ["0.1", "0.9"], "k": ["X", "Y"], "t": ["2026-10-25T10:00:00Z", ""],
                      "n": ["3", "0"], "h": ["00af", ""]})
    assert validate(d, S) == []


def test_each_kind_is_reported():
    d = pd.DataFrame({"a": ["1", "1", ""], "p": ["1.2", "x", "0.7"], "k": ["Z", "X", ""],
                      "t": ["not a date", "", ""], "n": ["1.5", "-1", "2"], "h": ["zz", "00af", ""]})
    k = _kinds(validate(d, S))
    for want in [("null", "a"), ("range", "p"), ("dtype", "p"), ("allowed", "k"), ("dtype", "t"),
                 ("dtype", "n"), ("range", "n"), ("pattern", "h"), ("duplicate_key", "['a']"),
                 ("rule", "p_gt_half_needs_k")]:
        assert want in k, (want, k)


def test_missing_column_and_non_frame_never_raise():
    assert _kinds(validate(pd.DataFrame({"a": ["1"]}), S))[0][0] == "missing_column"
    assert validate(None, S)[0]["kind"] == "not_a_frame"
    boom = Schema("B", "1", "", columns=(Col("a"),), rules=(Rule("r", lambda d: 1 / 0, "", ("a",)),))
    assert validate(pd.DataFrame({"a": ["1"]}), boom)[0]["kind"] == "internal_error"


def test_typed_and_string_frames_agree_on_nulls():
    typed = pd.DataFrame({"a": ["1"], "p": [float("nan")], "k": [None], "t": [pd.NaT], "n": [pd.NA], "h": [None]})
    assert validate(typed, S) == []


def test_research_observation_after_kickoff_is_flagged():
    d = pd.DataFrame({"fixture_id": ["f1", "f2"], "observed_at": ["2026-10-25T12:00:00Z", "2026-10-25T15:00:00Z"],
                      "kickoff_utc": ["2026-10-25T14:00:00Z", "2026-10-25T14:00:00Z"], "league": ["L", "L"],
                      "market": ["ou25", "ou25"]})
    v = validate(d, SCHEMAS["RESEARCH_OBSERVATION"])
    assert [x["kind"] for x in v] == ["rule"] and v[0]["n_rows"] == 1


def test_stake_plan_rules_catch_a_stake_on_a_paper_track():
    row = {"date": "2026-10-25", "fixture": "2026-10-25|A|B", "league": "L", "model_type": "standard",
           "market": "OU25", "side": "OVER", "signal_tier": "SNIPER", "odds": "2.0",
           "policy_version": "p", "track": "standard|OU25", "deployment_state": "PAPER",
           "final_fraction": "0.01", "reason": "OK"}
    v = validate(pd.DataFrame([row]), SCHEMAS["STAKE_PLAN_ROW"])
    assert ("rule", "stake_only_on_real_state") in _kinds(v)


def test_every_schema_describes_itself():
    for s in SCHEMAS.values():
        d = s.describe()
        assert d["version"] and d["columns"] and json.dumps(d)


def test_committed_files_validate_without_raising():
    r = committed_violations()
    assert set(r) and all("n_violations" in x for x in r.values())


def test_observe_records_and_never_mutates(tmp_path):
    p = tmp_path / "cv.json"
    d = pd.DataFrame({"a": ["1", "1"]})
    before = d.copy()
    v = observe("frame.csv", d, "STAKE_PLAN_ROW", path=p)
    pd.testing.assert_frame_equal(d, before)
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["generated_at"] and doc["frames"]["frame.csv"]["n_violations"] == len(v) > 0
    assert doc["frames"]["frame.csv"]["validated_at"]


def test_observe_uses_env_path_and_can_be_switched_off(tmp_path, monkeypatch):
    assert os.environ["WOWZA_CONTRACT_VIOLATIONS"].endswith("contract_violations.json")   # conftest isolation
    monkeypatch.setenv("WOWZA_CONTRACTS_SHADOW", "0")
    assert observe("x", pd.DataFrame(), "STAKE_PLAN_ROW") == []


def test_stake_plan_written_bytes_identical_with_and_without_observation(tmp_path, monkeypatch):
    """The shadow hook may not change a byte of stake_plan.csv or stake_audit.csv."""
    from src import stake_plan as SP
    bets = pd.DataFrame([{"date": "2026-10-25", "league": "Championship", "home_team": "A", "away_team": "B",
                          "best_side": "OVER", "odds_over25": 2.0, "odds_under25": 1.8, "signal_tier": "SNIPER",
                          "model_type": "standard"}])
    monkeypatch.setattr(SP, "load_league_ratings", lambda *a, **k: {})
    outs = []
    for flag in ("0", "1"):
        monkeypatch.setenv("WOWZA_CONTRACTS_SHADOW", flag)
        monkeypatch.setattr(SP, "PLAN", tmp_path / f"plan{flag}.csv")
        monkeypatch.setattr(SP, "AUDIT", tmp_path / f"audit{flag}.csv")
        SP.write(bets, None)
        outs.append([(tmp_path / f"{n}{flag}.csv").read_bytes() for n in ("plan", "audit")])
    import io
    for a, b in zip(*outs):                       # identical except the wall-clock generated_at
        fa, fb = (pd.read_csv(io.BytesIO(x), dtype=str, keep_default_na=False) for x in (a, b))
        assert list(fa.columns) == list(fb.columns) and len(fa) == 1
        pd.testing.assert_frame_equal(fa.drop(columns="generated_at"), fb.drop(columns="generated_at"))
    cv = json.loads(open(os.environ["WOWZA_CONTRACT_VIOLATIONS"], encoding="utf-8").read())
    assert "stake_plan.csv" in cv["frames"]
