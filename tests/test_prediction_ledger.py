"""Prediction ledger (src/prediction_ledger.py + src/prediction_ledger_v9.py) - owner rule 2026-11-17.

Pins: the stable league slug, the shared config, first-write-wins idempotency, the partition layout,
the Data truth 12 validator (duplicate key, partition mismatch, unknown league, post-kickoff snapshot,
price column, pending model with rows), the settled view (1:1, log loss / Brier / RPS / log score), the
flag (default OFF writes nothing) and that no price ever reaches a row.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import prediction_ledger as PL
from src import prediction_ledger_v9 as V

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
KO = "2026-11-21T15:00:00Z"


def _row(**kw):
    base = dict(repo="v9", model_id="v9_standard", model_version="standard_x_1", market="ou25", line=2.5,
                selection="OVER", fixture_ref="oa:abc", oddsapi_event_id="abc", league="Championship",
                season="2026", kickoff_utc=KO, home_team="Norwich City", away_team="Cardiff City",
                probability=0.55, snapshot="first_sight", predicted_at="2026-11-17T12:00:00Z",
                minutes_to_kickoff=6000)
    base.update(kw)
    return PL.make_row(**base)


# ── slug / registry ──────────────────────────────────────────────────────────────────────────────

def test_slug_is_the_registry_id_and_derivable_from_the_name():
    leagues = PL.load_leagues()
    ids = [e["id"] for e in leagues]
    assert len(ids) == len(set(ids)), "registry ids must be unique (they are partition directories)"
    for e in leagues:
        assert PL.slugify(e["name"]) == e["id"], f"{e['name']!r} -> {PL.slugify(e['name'])!r} != {e['id']!r}"
    s = PL.LeagueSlugs()
    assert s.slug("Championship") == "championship"
    assert s.slug(" Championship ") == "championship"
    assert s.slug("Danish Superliga") == "denmark_superliga"            # a registered alias
    assert s.slug("Championship (England)") is None                     # never fuzzy
    assert s.slug(None) is None


def test_league_scope_names_every_excluded_competition_with_a_reason():
    scope = PL.league_scope()
    assert {x["league_slug"] for x in scope} == {e["id"] for e in PL.load_leagues()}
    for x in scope:
        assert (x["scope"] == "tracked") == (x["not_scored_reason"] == "")
    by = {x["league_slug"]: x for x in scope}
    assert by["championship"]["scope"] == "tracked"
    assert by["romanian_superliga"]["scope"] == "tracked_not_on_board"


# ── config ───────────────────────────────────────────────────────────────────────────────────────

def test_markets_config_is_unique_and_side_markets_are_their_own_entries():
    m = PL.load_markets()
    keys = [(e["repo"], e["model_id"], e["market"]) for e in m["entries"]]
    assert len(keys) == len(set(keys))
    assert m["schema_version"] == PL.SCHEMA_VERSION
    for e in m["entries"]:
        assert e["status"] in ("active", "pending_model", "pending_writer")
        assert PL._ID.match(e["model_id"]) and PL._ID.match(e["market"])
    v9 = {(e["model_id"], e["market"]) for e in m["entries"] if e["repo"] == "v9" and e["status"] == "active"}
    for mid in ("v9_standard", "v9_new_format"):
        for mk in ("ou15", "ou25", "ou35", "btts"):
            assert (mid, mk) in v9
    assert ("v9_standard", "ht_ou05") in v9 and ("v9_new_format", "ht_ou05") not in v9
    for mk in ("goals", "goals2", "goals3", "assists", "sot", "sot2", "sot3", "sot4", "cards"):
        assert ("v9_props", mk) in v9
    # owner 2026-11-18: the rare markets serve the clean player base rate; their parked clean ML
    # challengers predict into the ledger beside it
    for mk in ("goals2", "goals3", "sot3", "sot4"):
        assert (V.CHALLENGER_MODEL_ID, mk) in v9
    pending = {e["market"] for e in m["entries"] if e["status"] == "pending_model"}
    assert {"corners_total", "corners_team", "cards_total", "cards_team"} <= pending
    pro = {e["market"] for e in m["entries"] if e["repo"] == "pro"}
    assert {"1x2", "correct_score", "goals_exact_0", "goals_le1", "goals_ge6", "combo"} <= pro


def test_schema_has_no_price_column():
    assert not [c for c in PL.COLUMNS if PL.PRICE_LIKE.search(c)]
    with pytest.raises(PL.LedgerError):
        PL.make_row(odds=1.9)
    with pytest.raises(PL.LedgerError):
        PL.make_row(model_id="x", probability=None, snapshot="final")       # unscored without a reason
    with pytest.raises(PL.LedgerError):
        _row(probability=1.2)


# ── append / idempotency ─────────────────────────────────────────────────────────────────────────

def test_first_write_wins_and_repeats_are_counted(tmp_path):
    st = PL.append([_row()], repo="v9", writer="t", root=tmp_path, now=NOW)
    assert st["written"] == 1 and st["violations"] == 0
    st = PL.append([_row(probability=0.9)], repo="v9", writer="t", root=tmp_path, now=NOW)
    assert st["written"] == 0 and st["duplicate_skipped"] == 1
    p = tmp_path / "v9" / "v9_standard" / "ou25" / "championship" / "2026.csv"
    d = pd.read_csv(p, dtype=str)
    assert len(d) == 1 and d["probability"].iloc[0] == "0.55"            # the first opinion is kept
    log = pd.read_csv(tmp_path / "v9" / "_runs" / "t" / "2026-11.csv")
    assert log["duplicate_skipped"].tolist() == [0, 1]
    # a new model version is new rows, never an overwrite
    st = PL.append([_row(model_version="standard_x_2", probability=0.4)], repo="v9", writer="t",
                   root=tmp_path, now=NOW)
    assert st["written"] == 1
    assert pd.read_csv(p, dtype=str)["model_version"].tolist() == ["standard_x_1", "standard_x_2"]
    assert PL.validate(tmp_path) == []


def test_refusals(tmp_path):
    rows = [_row(league="Atlantis League"),                               # unknown league
            _row(model_id="v9_corners", market="corners_total", line=9.5),  # pending model: no rows
            _row(fixture_ref=""),                                          # no fixture identity
            _row(predicted_at="2026-11-21T15:00:00Z"),                     # at kickoff: not pre-kickoff
            _row(season="")]
    st = PL.append(rows, repo="v9", writer="t", root=tmp_path, now=NOW)
    assert st["written"] == 0 and st["late_skipped"] == 1 and st["refused"] == 4
    assert not PL.partitions(tmp_path)


def test_snapshots_due():
    assert PL.snapshots_due(6000) == ["first_sight"]
    assert PL.snapshots_due(90) == ["first_sight", "final"]
    assert PL.snapshots_due(0) == [] and PL.snapshots_due(-5) == []


# ── validator (Data truth 12) ────────────────────────────────────────────────────────────────────

def _write(tmp_path, rows, rel="v9/v9_standard/ou25/championship/2026.csv"):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in PL.COLUMNS:
        if c not in df:
            df[c] = ""
    df[PL.COLUMNS].to_csv(p, index=False)
    return p


def test_validator_fails_on_duplicate_key(tmp_path, capsys):
    r = dict(_row(), league_slug="championship")
    p = _write(tmp_path, [r, r])
    assert any("duplicate key" in i for i in PL.validate(tmp_path))
    assert PL.validate(tmp_path, only=set()) == []                      # scoped check skips it
    assert any("duplicate key" in i for i in PL.validate(tmp_path, only={p}))


def test_validator_fails_on_partition_mismatch(tmp_path):
    r = dict(_row(), league_slug="championship")
    _write(tmp_path, [r], rel="v9/v9_standard/ou15/championship/2026.csv")       # market != partition
    assert any("market != partition" in i for i in PL.validate(tmp_path))
    shutil.rmtree(tmp_path / "v9")
    _write(tmp_path, [r], rel="v9/v9_standard/ou25/league_one/2026.csv")         # league != partition
    issues = PL.validate(tmp_path)
    assert any("league_slug != partition" in i for i in issues)
    assert any("league is not the partition league" in i for i in issues)


def test_validator_fails_on_unknown_league(tmp_path):
    r = dict(_row(), league="Atlantis League", league_slug="atlantis_league")
    _write(tmp_path, [r], rel="v9/v9_standard/ou25/atlantis_league/2026.csv")
    issues = PL.validate(tmp_path)
    assert any("unknown league slug" in i for i in issues)
    assert any("registry does not know" in i for i in issues)


def test_validator_fails_on_post_kickoff_snapshot(tmp_path):
    r = dict(_row(), league_slug="championship")
    r["predicted_at"] = "2026-11-21T15:05:00Z"
    _write(tmp_path, [r])
    assert any("not strictly before kickoff" in i for i in PL.validate(tmp_path))


def test_validator_fails_on_price_column_and_pending_rows(tmp_path):
    r = dict(_row(), league_slug="championship")
    p = _write(tmp_path, [r])
    d = pd.read_csv(p, dtype=str)
    d["odds"] = "1.90"
    d.to_csv(p, index=False)
    assert any("price-like column" in i for i in PL.validate(tmp_path))
    shutil.rmtree(tmp_path / "v9")
    r2 = dict(_row(model_id="v9_corners", market="corners_total", selection="OVER", line=9.5),
              league_slug="championship")
    _write(tmp_path, [r2], rel="v9/v9_corners/corners_total/championship/2026.csv")
    assert any("pending_model" in i for i in PL.validate(tmp_path))


def test_data_truth_rule_12_is_gated_and_red_on_a_violation(tmp_path, monkeypatch):
    from src.integrity import data_truth as DT
    assert any("Data truth 12".startswith(g) for g in DT.GATED)
    r = dict(_row(), league_slug="championship")
    _write(tmp_path, [r, r])
    monkeypatch.setattr(PL, "LEDGER_DIR", tmp_path)
    c = DT.prediction_ledger()
    assert c["status"] == DT.RED and "duplicate key" in c["detail"]
    assert DT.gate(checks=[c]) and "Data truth 12" in DT.gate(checks=[c])[0]
    monkeypatch.setattr(PL, "LEDGER_DIR", tmp_path / "empty")
    assert DT.prediction_ledger()["status"] == DT.GREEN


# ── v9 team writer ───────────────────────────────────────────────────────────────────────────────

def _board(**over):
    r = {"league": "Championship", "date": "2026-11-21", "kickoff_utc": "2026-11-21 15:00:00",
         "home_team": "Norwich City", "away_team": "Cardiff City", "_oa_event_id": "8e123",
         "odds_over25": 1.9, "home_scored_last5": 1.2, "table_pos_gap": 3, "serving_recipe": "v9_repaired",
         "p_over25": 0.6, "model_type": "standard", "p_btts": 0.5, "p_over15": 0.8, "p_over35": 0.3,
         "p_ht_over05": float("nan"), "p_ht_over15": 0.2, "generated_at": "2026-11-17T12:00:00Z",
         "git_sha": "abc", "model_version": "std_1", "model_version_btts": "btts_1",
         "model_version_over15": "o15_1", "model_version_over35": "o35_1",
         "model_version_ht_over05": "ht05_1", "model_version_ht_over15": "ht15_1",
         "edge_over": 0.1, "kelly_over": 0.02, "stake_over": 1, "tier_over": "VALUABLE"}
    r.update(over)
    return pd.DataFrame([r])


def test_team_rows_every_market_both_selections_no_prices(tmp_path):
    p = tmp_path / "predictions.csv"
    _board().to_csv(p, index=False)
    b = V.read_board(p)
    assert b["_oa_event_id"].iloc[0] == "8e123"                         # never parsed as 8e123 -> 8e+125
    rows = V.team_rows(b, now=NOW)
    d = pd.DataFrame(rows)
    assert set(d["market"]) == {"ou25", "ou15", "ou35", "btts", "ht_ou05", "ht_ou15"}
    assert set(d["snapshot"]) == {"first_sight"}
    assert d["fixture_ref"].unique().tolist() == ["oa:8e123"]
    assert list(d.columns) == PL.COLUMNS
    ou = d[d["market"] == "ou25"].set_index("selection")["probability"]
    assert math.isclose(ou["OVER"] + ou["UNDER"], 1.0)
    ht = d[d["market"] == "ht_ou05"]
    assert (ht["scored"] == 0).all() and ht["not_scored_reason"].str.contains("p_ht_over05").all()
    assert ht["probability"].eq("").all()                               # missing stays missing
    assert d["model_version"][d["market"] == "btts"].unique().tolist() == ["btts_1"]
    st = PL.append(rows, repo="v9", writer="t", root=tmp_path / "L", now=NOW)
    assert st["written"] == len(rows) and PL.validate(tmp_path / "L") == []


def test_team_rows_new_format_has_no_ht_and_final_snapshot_near_kickoff():
    b = _board(model_type="new_format", serving_recipe="legacy_v9", generated_at="2026-11-21T14:00:00Z")
    d = pd.DataFrame(V.team_rows(b, now=NOW))
    assert set(d["model_id"]) == {"v9_new_format"}
    assert not d["market"].str.startswith("ht_").any()
    assert set(d["snapshot"]) == {"first_sight", "final"}
    assert V.team_rows(_board(generated_at="2026-11-21T15:00:00Z"), now=NOW) == []   # at kickoff: nothing


def test_flag_off_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(PL.FLAG_ENV, raising=False)
    p = tmp_path / "predictions.csv"
    _board().to_csv(p, index=False)
    assert "skipped" in V.record_board(p, root=tmp_path / "L")
    assert not (tmp_path / "L").exists()
    assert V.props_recorder() is None
    for v in ("false", "1", "TRUE ", "yes"):
        monkeypatch.setenv(PL.FLAG_ENV, v)
        assert PL.enabled() == (v.strip().lower() == "true")


# ── v9 props recorder ────────────────────────────────────────────────────────────────────────────

def test_props_recorder_records_every_player(tmp_path, monkeypatch):
    monkeypatch.setenv(PL.FLAG_ENV, "true")
    rec = V.props_recorder()
    rec.now = NOW
    feat = pd.DataFrame({"player_id": [1, 2, 3], "player_name": ["A", "B", "C"], "team": ["X", "X", "Y"],
                         "f1": [0.1, 0.2, 0.3]})
    rec.add("sot", feat, np.array([0.4, 0.05, np.nan]), league="Championship", kickoff_utc=KO,
            fixture_id=1563270.0, home="X", away="Y", payload={"feature_cols": ["f1"]})
    rec.add("sot", feat, np.array([0.4, 0.05, 0.1]), league="Championship", kickoff_utc="", fixture_id=1)
    st = rec.flush(root=tmp_path)
    assert st["written"] == 3
    d = pd.read_csv(tmp_path / "v9" / "v9_props" / "sot" / "championship" / "2026.csv", dtype=str,
                    keep_default_na=False)
    assert d["fixture_ref"].unique().tolist() == ["af:1563270"]
    assert d["player_id"].tolist() == ["1", "2", "3"]
    assert d["scored"].tolist() == ["1", "1", "0"] and d["not_scored_reason"].iloc[2]
    assert PL.validate(tmp_path) == []


def test_props_recorder_writes_the_parked_challenger_beside_the_champion(tmp_path, monkeypatch):
    """Owner 2026-11-18: the clean ML challenger of a rare market is recorded on the same player-fixtures
    as the served base rate, under its own model_id / version / sha; a market without a challenger entry
    (sot) writes the champion only."""
    import player_model.props_gate as PPG

    class _M:
        def predict(self, X, **_):
            return np.full(len(X), 0.03)
    payload = {"recipe": "props_clean_harness", "feature_cols": ["f1"], "model": _M()}
    meta = {"model_version": "props_sot4_clean_recipe_retrained_8ed4b805", "artifact_sha256": "8ed4b805" + "0" * 56}
    monkeypatch.setattr(PPG, "load_parked", lambda m: (payload, meta) if m == "sot4" else None)
    monkeypatch.setenv(PL.FLAG_ENV, "true")
    rec = V.props_recorder()
    rec.now = NOW
    feat = pd.DataFrame({"player_id": [1, 2], "player_name": ["A", "B"], "team": ["X", "Y"], "f1": [0.1, 0.2]})
    for mk in ("sot4", "sot"):
        rec.add(mk, feat, np.array([0.004, 0.002]), league="Championship", kickoff_utc=KO, fixture_id=1563270,
                home="X", away="Y", payload={"feature_cols": []})
    assert rec.flush(root=tmp_path)["written"] == 6
    ch = pd.read_csv(tmp_path / "v9" / V.CHALLENGER_MODEL_ID / "sot4" / "championship" / "2026.csv", dtype=str,
                     keep_default_na=False)
    assert ch["model_version"].unique().tolist() == [meta["model_version"]]
    assert ch["model_sha"].unique().tolist() == [meta["artifact_sha256"]]
    assert ch["probability"].astype(float).tolist() == [0.03, 0.03]
    assert not (tmp_path / "v9" / V.CHALLENGER_MODEL_ID / "sot").exists()
    assert PL.validate(tmp_path) == []


def test_props_results_settle_goals3_from_the_goals_count(tmp_path, monkeypatch):
    d = pd.DataFrame({"fixture_id": [7, 7, 7], "player_id": [1, 2, 3], "minutes": [90, 90, 90],
                      "goals": [3.0, 2.0, np.nan]})
    for c in ("target_goals", "target_goals2", "target_assists", "target_sot", "target_sot2", "target_sot3",
              "target_sot4", "target_cards"):
        d[c] = 0.0
    hist = tmp_path / "ph"
    hist.mkdir()
    d.to_parquet(hist / "month_2026-11.parquet")
    monkeypatch.setattr(V, "PLAYER_HISTORY", hist)
    r = V.PropsResults({7})
    assert r.rows[(7, 1)]["target_goals3"] == 1.0 and r.rows[(7, 2)]["target_goals3"] == 0.0
    assert pd.isna(r.rows[(7, 3)]["target_goals3"])                     # missing count stays missing


# ── settled view ─────────────────────────────────────────────────────────────────────────────────

def test_match_outcome_rules():
    ft = {"rule": "total_goals_over", "period": "ft"}
    assert PL.match_outcome(ft, "OVER", 2.5, 2, 1)[0] == 1
    assert PL.match_outcome(ft, "UNDER", 2.5, 2, 1)[0] == 0
    assert PL.match_outcome({"rule": "total_goals_over", "period": "ht"}, "OVER", 0.5, 2, 1, None, None)[0] is None
    assert PL.match_outcome({"rule": "btts"}, "NO", None, 2, 0)[0] == 1
    assert PL.match_outcome({"rule": "result_1x2"}, "D", None, 1, 1) == (1, "D")
    assert PL.match_outcome({"rule": "correct_score"}, "other", None, 7, 0) == (1, "other")
    assert PL.match_outcome({"rule": "total_goals_ge6"}, "YES", 5.5, 4, 2)[0] == 1
    assert PL.match_outcome({"rule": "total_goals_le1"}, "YES", 1.5, 1, 1)[0] == 0
    assert PL.match_outcome({"rule": "team_goals_ge4"}, "AWAY", 3.5, 0, 4)[0] == 1


def test_settled_view_is_one_to_one_with_scores():
    raw = pd.DataFrame([dict(_row(), league_slug="championship"),
                        dict(_row(selection="UNDER", probability=0.45), league_slug="championship"),
                        dict(_row(fixture_ref="oa:zzz", probability=None, not_scored_reason="no p"),
                             league_slug="championship")]).astype(str)
    raw.loc[raw["probability"] == "", "probability"] = ""
    out = PL.settled_view(raw, lambda r: {"outcome_status": "SETTLED", "y": 1 if r["selection"] == "OVER" else 0})
    assert len(out) == len(raw) and out[PL.KEY_COLS].equals(raw[PL.KEY_COLS])
    assert math.isclose(float(out["log_loss"].iloc[0]), -math.log(0.55), rel_tol=1e-5)
    assert math.isclose(float(out["brier"].iloc[1]), 0.45 ** 2, rel_tol=1e-5)
    assert out["log_loss"].iloc[2] == ""                                  # unscored: no score


def test_rps_and_log_score():
    base = dict(model_id="pro_bb_dc", repo="pro", league_slug="championship")
    rows = [dict(_row(**base, market="1x2", selection=s, probability=p, line=None), league_slug="championship")
            for s, p in (("H", 0.5), ("D", 0.3), ("A", 0.2))]
    rows += [dict(_row(**base, market="correct_score", selection=s, probability=p, line=None),
                  league_slug="championship") for s, p in (("1-0", 0.1), ("0-0", 0.08))]
    raw = pd.DataFrame(rows).astype(str)
    real = {"1x2": "H", "correct_score": "1-0"}
    out = PL.settled_view(raw, lambda r: {"outcome_status": "SETTLED", "y": int(r["selection"] == real[r["market"]])})
    rps = float(out["rps"].iloc[0])
    assert math.isclose(rps, ((0.5 - 1) ** 2 + (0.8 - 1) ** 2) / 2, rel_tol=1e-5)
    assert (out["rps"].iloc[:3] == out["rps"].iloc[0]).all()
    assert math.isclose(float(out["log_score"].iloc[3]), -math.log(0.1), rel_tol=1e-5)
    assert out["log_score"].iloc[4] == ""


def test_write_settled_never_touches_raw_and_reschedule_voids(tmp_path, monkeypatch):
    root, sroot = tmp_path / "L", tmp_path / "S"
    rows = [_row(), _row(snapshot="final", kickoff_utc="2026-11-25T15:00:00Z",
                         predicted_at="2026-11-25T14:00:00Z", minutes_to_kickoff=60)]
    PL.append(rows, repo="v9", writer="t", root=root, now=NOW)
    raw_p = PL.partitions(root)[0]
    before = raw_p.read_bytes()

    class _T:
        def find(self, *a):
            return {"home_goals": 3, "away_goals": 1, "src": "football-data"}
    fn = V.outcome_fn(now=datetime(2026, 11, 26, tzinfo=timezone.utc), team=_T(),
                      latest_ko=V.latest_kickoffs(root))
    st = PL.write_settled(fn, repo="v9", root=root, settled_root=sroot)
    assert raw_p.read_bytes() == before
    s = pd.read_csv(PL.settled_path(sroot, root, raw_p), dtype=str, keep_default_na=False)
    assert s["outcome_status"].tolist() == ["VOID_RESCHEDULED", "SETTLED"] and st["settled"] == 1
    assert s["y"].tolist() == ["", "1"]
    assert PL.write_settled(fn, repo="v9", root=root, settled_root=sroot)["written"] == 0   # unchanged


def test_registry_lists_pending_and_counts(tmp_path):
    PL.append([_row()], repo="v9", writer="t", root=tmp_path, now=NOW)
    reg = PL.build_registry(root=tmp_path, settled_root=tmp_path / "none", now=NOW)
    by = {(m["repo"], m["model_id"], m["market"]): m for m in reg["models"]}
    assert by[("v9", "v9_standard", "ou25")]["rows"] == 1
    assert by[("v9", "v9_standard", "ou25")]["leagues"]["championship"]["predicted"] == 1
    assert by[("v9", "v9_corners", "corners_total")]["status"] == "pending_model"
    assert by[("v9", "v9_corners", "corners_total")]["rows"] == 0
    json.dumps(reg)
