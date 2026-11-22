"""P0 item 3 (2026-11-22): active serving joins, the legacy replay and the training lineage are separate
verdicts. The legacy replay may stay FAIL as historical evidence; a real active failure never turns PASS."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.integrity import joins as IJ
from src.integrity import training_lineage as TL

ROOT = Path(__file__).resolve().parents[1]


def _left():
    return pd.DataFrame({"league": ["A", "A", "B", "A"],
                         "_h": ["x", "y", "x", "z"], "_a": ["y", "x", "y", "w"],
                         "_d": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-02"],
                         "ht": [np.nan, np.nan, np.nan, 1.0]})


def _right_with_duplicates():
    # (A, x, y) twice with the same value, (A, y, x) twice with DIFFERENT values, (B, x, y) once
    return pd.DataFrame({"league": ["A", "A", "A", "A", "B"],
                         "_h": ["x", "x", "y", "y", "x"], "_a": ["y", "y", "x", "x", "y"],
                         "_d": ["2026-01-01"] * 5, "ht": [2.0, 2.0, 0.0, 1.0, 5.0]})


# ── active serving joins ─────────────────────────────────────────────────────────────────────
def test_active_safe_joins():
    left, right = _left(), _right_with_duplicates()
    out, rep = IJ.safe_left_enrich(left, right, keys=["league", "_h", "_a", "_d"], value_cols=["ht"],
                                   name="t_active", register=False)
    assert len(out) == len(left) and rep.output_rows == rep.left_rows and not rep.duplicate_output_rows
    assert list(out.index) == list(left.index)
    assert out["ht"].iloc[0] == 2.0                                     # duplicates that agree: one value
    assert np.isnan(out["ht"].iloc[1])                                  # duplicates that disagree: left unfilled
    assert out["ht"].iloc[3] == 1.0                                     # nan_only: never overwritten
    assert rep.right_duplicate_key_groups == 2 and rep.ambiguous_key_groups == 1 and rep.status == "WARN"
    # league B's row got league B's value: the same clubs and date in league A never leak across
    assert out.loc[2, "ht"] == 5.0
    with pytest.raises(ValueError):                                     # no identity -> refused
        IJ.safe_left_enrich(left, right, keys=["_h", "_a", "_d"], value_cols=["ht"], name="t_noid", register=False)
    # scope verdicts: a safe join that changed the row count is an ACTIVE failure
    doc = {"joins": {"s": {"mode": "safe", "status": "PASS", "left_rows": 4, "output_rows": 4},
                     "s.legacy": {"mode": "legacy_audit", "status": "FAIL", "left_rows": 4, "output_rows": 6}}}
    sc = TL.join_scopes(doc)
    assert sc["active_serving_join_status"] == "PASS" and sc["legacy_replay_status"] == "FAIL"
    doc["joins"]["s"]["output_rows"] = 5
    assert TL.join_scopes(doc)["active_serving_join_status"] == "FAIL"
    assert TL.join_scopes({})["active_serving_join_status"] == "UNKNOWN"


# ── legacy replay isolation ──────────────────────────────────────────────────────────────────
def test_legacy_join_replay_isolation():
    left, right = _left(), _right_with_duplicates()
    before = left.copy()
    rep = IJ.audit_legacy_merge(left, right, ["_h", "_a", "_d"], name="t_legacy", register=False)
    assert rep.status == "FAIL" and rep.legacy_expansion_rows > 0     # the replay still proves the old defect
    pd.testing.assert_frame_equal(left, before)                        # ... and writes nothing anywhere
    # the active code carries no contaminated join: the replay is audit-only, no positional copy-back,
    # no switch back to the league-free path, no identity opt-out
    src = {f: (ROOT / f).read_text(encoding="utf-8") for f in TL.TEAM_CODE}
    a = TL.code_audit(src)
    assert a["status"] == "PASS", a["findings"]
    assert a["safe_left_enrich_calls_in_loader"] >= 7
    # every replay call in the loader is a bare statement (its report is never a frame)
    for ln in src["src/data_loader.py"].splitlines():
        if "audit_legacy_merge(" in ln:
            assert ln.strip().startswith(("_IJ.audit_legacy_merge(", "#")), ln
    # the replay's FAIL reaches legacy_replay_status only, never a track's join verdict
    doc = {"joins": {"fd_history_ht_fill": {"mode": "safe", "status": "PASS", "left_rows": 9, "output_rows": 9},
                     "fd_history_ht_fill.legacy": {"mode": "legacy_audit", "status": "FAIL", "left_rows": 9,
                                                   "output_rows": 11}}}
    sc = TL.join_scopes(doc)
    assert (sc["active_serving_join_status"], sc["legacy_replay_status"]) == ("PASS", "FAIL")


def test_code_audit_flags_contaminated_lineage():
    bad = {"src/data_loader.py": (
        "def f(df, hist):\n"
        "    merged = df.merge(hist, on=['_h', '_a', '_d'], how='left')\n"
        "    df['x'] = [a if a == a else b for a, b in zip(df['x'].values, merged['x'].values)]\n"
        "    if os.getenv('WOWZA_SAFE_ENRICH_JOINS') == '0':\n        pass\n"
        "    df, r = _IJ.safe_left_enrich(df, hist, keys=['_h'], value_cols=['x'], name='n', require_identity=False)\n")}
    kinds = {f["kind"] for f in TL.code_audit(bad)["findings"]}
    assert kinds == {"positional_copy_back", "legacy_join_switch", "identity_free_safe_join"}
    doc_only = {"src/data_loader.py": '"""zip(df[c].values, merged[c].values) was the defect"""\nx = 1  # zip(a[b].values, c[d].values)\n'}
    assert TL.code_audit(doc_only)["status"] == "PASS"
    joins_without_guard = {"src/integrity/joins.py": "def safe_left_enrich():\n    pass\n"}
    assert {f["kind"] for f in TL.code_audit(joins_without_guard)["findings"]} == {"missing_guard"}


# ── training lineage: row counts, duplicate keys, league mismatches ──────────────────────────
def _keys(n=6):
    d = pd.date_range("2026-01-01", periods=n, freq="7D")
    return pd.DataFrame({"league": ["L1"] * n, "date": d,
                         "home_team": [f"h{i}" for i in range(n)], "away_team": [f"a{i}" for i in range(n)]})


def _track(lg):
    return {"L1": "standard", "N1": "new_format"}.get(lg)


def test_training_lineage():
    reg = {"L1", "N1"}
    k = _keys()
    ok = TL.frame_audit(k, "standard", _track, reg, train_end="2026-02-01", expected_rows=5)
    assert ok["status"] == "PASS" and ok["rows_to_train_end"] == 5 and ok["rows_after_train_end"] == 1
    # row count against the payload
    assert TL.frame_audit(k, "standard", _track, reg, train_end="2026-02-01", expected_rows=6)["status"] == "FAIL"
    # a duplicate fixture key (an expanded join)
    dup = pd.concat([k, k.iloc[[2]]], ignore_index=True)
    r = TL.frame_audit(dup, "standard", _track, reg)
    assert r["duplicate_fixture_keys"] == 1 and r["status"] == "FAIL"
    # a club twice on one day in one league (a row attached to the wrong fixture)
    tw = k.copy()
    tw.loc[1, ["date", "home_team"]] = [tw.loc[0, "date"], "h0"]
    r = TL.frame_audit(tw, "standard", _track, reg)
    assert r["club_twice_same_day"] == 1 and r["status"] == "FAIL"
    # league mismatches: another track's league, and a league outside the registry
    other = k.copy()
    other.loc[0, "league"] = "N1"
    other.loc[1, "league"] = "Mystery League"
    r = TL.frame_audit(other, "standard", _track, reg)
    assert r["leagues_on_other_track"] == ["Mystery League", "N1"] and r["leagues_not_in_registry"] == ["Mystery League"]
    assert r["status"] == "FAIL"


def test_point_in_time_audit():
    rng = np.random.default_rng(0)
    rows = []
    teams = ["t1", "t2", "t3", "t4"]
    for i in range(40):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({"league": "L", "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), "home_team": h,
                     "away_team": a, "home_goals": int(rng.integers(0, 4)), "away_goals": int(rng.integers(0, 4))})
    f = pd.DataFrame(rows)
    tc = pd.concat([f.reset_index().rename(columns={"home_team": "team", "home_goals": "gs"})[["index", "team", "gs"]]
                    .assign(side="home"),
                    f.reset_index().rename(columns={"away_team": "team", "away_goals": "gs"})[["index", "team", "gs"]]
                    .assign(side="away")]).sort_values("index", kind="mergesort")
    g = tc.groupby("team")["gs"]
    tc["prev"] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    tc["incl"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
    for side in ("home", "away"):
        s = tc[tc["side"] == side].set_index("index").sort_index()
        f[f"{side}_scored_last5"] = s["prev"]
    assert TL.pit_audit(f)["status"] == "PASS"
    leak = f.copy()
    for side in ("home", "away"):
        leak[f"{side}_scored_last5"] = tc[tc["side"] == side].set_index("index").sort_index()["incl"]
    assert TL.pit_audit(leak)["status"] == "FAIL"


def _root(tmp_path: Path, payload: bytes, evidence: dict | None) -> Path:
    (tmp_path / "models" / "registry").mkdir(parents=True)
    (tmp_path / "models" / "model_v9_standard.pkl").write_bytes(payload)
    (tmp_path / "models" / "registry" / "current_champions.json").write_text(json.dumps(
        {"standard": {"artifact_sha256": hashlib.sha256(payload).hexdigest()},
         "standard_repaired": {"status": "RETIRED"}}))
    if evidence is not None:
        (tmp_path / "reports" / "lineage").mkdir(parents=True)
        (tmp_path / "reports" / "lineage" / "training_lineage_evidence.json").write_text(json.dumps(evidence))
    return tmp_path


def test_lineage_never_pass_without_matching_evidence(tmp_path):
    pay = b"payload-v1"
    sha = hashlib.sha256(pay).hexdigest()
    r = TL.evaluate(_root(tmp_path / "a", pay, None))
    assert r["models"]["standard"]["status"] == "UNVERIFIED" and r["status"] == "UNVERIFIED"
    ev = {"payloads": {"model_v9_standard.pkl": {"artifact_sha256": "0" * 64, "status": "PASS"}}}
    assert TL.evaluate(_root(tmp_path / "b", pay, ev))["status"] == "UNVERIFIED"   # payload changed since
    ev = {"payloads": {"model_v9_standard.pkl": {"artifact_sha256": sha, "status": "FAIL"}}}
    assert TL.evaluate(_root(tmp_path / "c", pay, ev))["status"] == "FAIL"
    ev = {"payloads": {"model_v9_standard.pkl": {"artifact_sha256": sha, "status": "PASS"}}}
    assert TL.evaluate(_root(tmp_path / "d", pay, ev))["status"] == "PASS"


def test_committed_lineage_evidence_is_consistent():
    """The committed evidence's PASS follows from the numbers it records (no hand-set verdict)."""
    ev = json.loads(TL.EVIDENCE.read_text(encoding="utf-8"))
    assert ev["payloads"], "no payload evidence"
    for pfile, p in ev["payloads"].items():
        for part in p["parts"]:
            fa = part["frame_audit"]
            clean = (fa["duplicate_fixture_keys"] == 0 and fa["club_twice_same_day"] == 0
                     and not fa["leagues_not_in_registry"] and not fa["leagues_on_other_track"]
                     and fa["rows_to_train_end"] == part["training_rows"])
            assert (fa["status"] == "PASS") == clean, (pfile, part["track"])
            code_ok = ev["code"][part["code_sha"]]["status"] == "PASS"
            expect = "PASS" if clean and code_ok and part["link_ok"] else "FAIL"
            assert part["status"] == expect, (pfile, part["track"])
        assert p["status"] == TL.worst(x["status"] for x in p["parts"])
