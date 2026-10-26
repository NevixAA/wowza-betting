"""Owner rule 2026-10-26: the monthly backtest PROPOSES per-league thresholds, the owner approves, and
only src/threshold_apply.py (approved entries, matched by id + content hash) writes the files production
reads. These tests pin every half of that."""
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import research_lock as L
from src import threshold_apply as A
from src import threshold_proposals as TP

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
PROD = [ROOT / TP.PROD_STANDARD, ROOT / TP.PROD_SIDE]


def _bytes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROD}


def _models_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "models").glob("*.json")}


# ── synthetic backtest output ─────────────────────────────────────────────────────────────────────

def _std_results(seed=1, n=600):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-08-01", periods=n, freq="D")
    season = np.where(dates < "2023-07-01", "2022/23", np.where(dates < "2024-07-01", "2023/24", "2024/25"))
    return pd.DataFrame({"league": rng.choice(["Serie B", "League Two"], n), "season": season, "date": dates,
                         "best_side": rng.choice(["OVER", "UNDER"], n), "best_edge": rng.uniform(0.02, 0.25, n),
                         "over25": rng.integers(0, 2, n), "odds_over25": rng.uniform(1.75, 2.3, n),
                         "odds_under25": rng.uniform(1.75, 2.3, n)})


def _side_results(seed=2, n=500):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-08-01", periods=n, freq="D")
    season = np.where(dates < "2023-07-01", "2022/23", np.where(dates < "2024-07-01", "2023/24", "2024/25"))
    return pd.DataFrame({"league": rng.choice(["Serie B", "Championship"], n), "season": season, "date": dates,
                         "edge": rng.uniform(0.02, 0.2, n), "bet_stake": 1.0, "pnl": rng.choice([0.9, -1.0], n)})


def test_optimizers_emit_proposal_evidence_without_changing_the_approval_rule():
    from src.backtest import optimize_side_market_thresholds, optimize_standard_thresholds
    std = optimize_standard_thresholds(_std_results())
    side = optimize_side_market_thresholds(_side_results(), target="btts")
    for v in list(std.values()) + list(side.values()):
        assert "oos_matchdays" in v and "oos_roi_ci95" in v
        assert v["approved"] == bool(v["roi_oos"] is not None and v["roi_oos"] > 0 and v["bets_oos"] >= 20)
        if v["oos_roi_ci95"]:
            lo, hi = v["oos_roi_ci95"]
            assert lo <= v["roi_oos"] <= hi


# ── 1. the monthly job cannot modify production threshold files ──────────────────────────────────

def test_static_no_optimizer_path_writes_a_production_threshold_file():
    for rel in ("pipeline.py", "retrain.py"):
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines):
            if "MODELS_DIR" in ln and "best_params" in ln:
                window = "\n".join(lines[i:i + 4])
                assert not re.search(r"write_text|json\.dump\(|open\([^)]*['\"]w['\"]", window), (rel, i + 1, window)
    # the only remaining MODELS_DIR/best_params references are READS
    pl = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert len(re.findall(r'MODELS_DIR\s*/\s*f?"best_params', pl)) == 1          # _load_side_market_thresholds
    import ast
    tree = ast.parse((ROOT / "src" / "threshold_proposals.py").read_text(encoding="utf-8"))
    mods = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
           [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in mods if m and "threshold_apply" in m], "the proposal module must not import the applier"


def test_dynamic_backtest_writes_candidates_never_production(tmp_path, monkeypatch):
    import config
    import pipeline
    monkeypatch.setattr(TP, "CANDIDATE_DIR", tmp_path / "cand")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    std = _std_results()
    std_summary = {"bets": len(std)}
    monkeypatch.setattr(pipeline, "run_backtest",
                        lambda feat, enabled_leagues=None: (std, std_summary, pd.DataFrame({"league": ["Serie B"]})))
    monkeypatch.setattr(pipeline, "run_side_market_backtest",
                        lambda feat, target, enabled_leagues=None: (_side_results(), {"bets": 1},
                                                                     pd.DataFrame({"league": ["Serie B"], "roi_%": [1.0]})))
    before, before_models = _bytes(), _models_hashes()
    pipeline.mode_backtest(feat=pd.DataFrame(), only="standard")
    pipeline.mode_backtest(feat=pd.DataFrame(), only=None)
    pipeline.mode_backtest_side(feat=pd.DataFrame(), market="btts")
    assert _bytes() == before and _models_hashes() == before_models
    names = sorted(p.name for p in (tmp_path / "cand").iterdir())
    assert names == ["best_params_btts.json", "best_params_side_markets.json", "best_params_standard.json"]
    cands = TP.load_candidates(tmp_path / "cand")
    assert set(cands) == {"ou25", "btts", "over15", "over35"}


def test_write_candidate_and_proposal_writer_refuse_production_paths(tmp_path):
    with pytest.raises(PermissionError):
        TP.write_candidate("standard", {}, directory=ROOT / "models")
    with pytest.raises(PermissionError):
        TP._safe_write(ROOT / TP.PROD_STANDARD, "{}")
    with pytest.raises(PermissionError):
        TP._safe_write(ROOT / "config" / "threshold_approvals.json", "{}")


# ── fixtures: a sandbox copy of the production threshold files + lock files ──────────────────────

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "config").mkdir()
    for p in PROD:
        shutil.copy(p, tmp_path / "models" / p.name)
    shutil.copy(L.SNAPSHOT_PATH, tmp_path / "config" / "research_lock_snapshot.json")
    monkeypatch.setattr(L, "THRESHOLD_ROOT", tmp_path)
    monkeypatch.setattr(L, "THRESHOLD_LEDGER_PATH", tmp_path / "config" / "threshold_applied.json")
    monkeypatch.setattr(L, "SNAPSHOT_PATH", tmp_path / "config" / "research_lock_snapshot.json")
    (tmp_path / "config" / "threshold_applied.json").write_text(
        json.dumps({"files": L.threshold_file_digests(tmp_path), "history": []}), encoding="utf-8")
    return tmp_path


def _candidates(root):
    """Two cells moved: Serie B ou25 (approved, new threshold) and Championship btts (approval revoked)."""
    prod = TP.production_params(root)
    ou = copy.deepcopy(prod["ou25"])
    ou["Serie B"] = dict(ou["Serie B"], sniper_th=0.15, marksman_th=0.13, bets_oos=88, roi_oos=6.1,
                         oos_matchdays=61, oos_roi_ci95=[-12.0, 24.2])
    bt = copy.deepcopy(prod["btts"])
    bt["Championship"] = dict(bt["Championship"], approved=False, roi_oos=-3.0)
    return {"ou25": ou, "btts": bt}


def _proposal(root, tmp):
    p = TP.build_proposal(_candidates(root), root=root, created_at="2026-11-01T03:30:00Z")
    path = tmp / "2026-11.json"
    path.write_text(json.dumps(p), encoding="utf-8")
    return p, path


def _approve(tmp, p, ids, **over):
    by = {e["entry_id"]: e for e in p["entries"]}
    a = {"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"], "approved_by": "Nevo (owner) - test",
         "approved_at": "2026-11-02T08:00:00Z",
         "entries": [{"entry_id": i, "entry_hash": by[i]["entry_hash"]} for i in ids]}
    a.update(over)
    path = tmp / "approvals.json"
    path.write_text(json.dumps({"approvals": [a]}), encoding="utf-8")
    return path


def test_proposal_compares_with_production_and_resolves_served_gates(sandbox):
    p, _ = _proposal(sandbox, sandbox)
    by = {e["entry_id"]: e for e in p["entries"]}
    assert set(by) == {"ou25|Serie B", "btts|Championship"}
    sb = by["ou25|Serie B"]
    assert sb["change"] == "THRESHOLD_CHANGE" and sb["serves_change"]
    assert sb["served_now"]["sniper"] == 0.12 and sb["served_if_approved"]["sniper"] == 0.15
    assert sb["evidence"]["bets_oos"] == 88 and sb["evidence"]["oos_matchdays"] == 61
    ch = by["btts|Championship"]
    assert ch["change"] == "APPROVAL_REVOKED" and ch["served_if_approved"]["sniper"] == 0.10
    assert p["status"] == "AWAITING_OWNER" and p["proposal_id"].startswith("TP-2026-11-01-")
    md = TP.to_markdown(p)
    assert "ou25|Serie B" in md and "How to approve" in md
    # owner 2026-10-26: the report a person reads shows gap labels; the codes appear only in the one
    # explicit glossary line (the JSON proposal and its hashes keep the codes)
    from src.tier_labels import GLOSSARY
    assert "served now (Big gap / Medium gap)" in md
    assert not re.search(r"\b(SNIPER|MARKSMAN|VALUABLE)\b", md.replace(GLOSSARY, ""))


# ── 2. apply refuses anything not matched to an approved proposal hash ──────────────────────────

def test_apply_refuses_without_an_approval(sandbox):
    p, path = _proposal(sandbox, sandbox)
    empty = sandbox / "none.json"
    empty.write_text(json.dumps({"approvals": []}), encoding="utf-8")
    before = L.threshold_file_digests(sandbox)
    with pytest.raises(A.ApplyRefused, match="no owner approval"):
        A.apply(path, approvals_path=empty, lock=False)
    assert L.threshold_file_digests(sandbox) == before


@pytest.mark.parametrize("mutate,msg", [
    (lambda a, p: a.update(proposal_hash="0" * 64), "proposal_hash"),
    (lambda a, p: a["entries"][0].update(entry_hash="f" * 64), "does not match"),
    (lambda a, p: a["entries"][0].update(sniper_th=0.05), "thresholds are never typed"),
    (lambda a, p: a["entries"].append({"entry_id": "ou25|Nowhere", "entry_hash": "0" * 64}), "not in proposal"),
    (lambda a, p: a.update(approved_by=""), "approved_by"),
])
def test_apply_refuses_mismatched_or_hand_typed_approvals(sandbox, mutate, msg):
    p, path = _proposal(sandbox, sandbox)
    ap = _approve(sandbox, p, ["ou25|Serie B"])
    blob = json.loads(ap.read_text(encoding="utf-8"))
    mutate(blob["approvals"][0], p)
    ap.write_text(json.dumps(blob), encoding="utf-8")
    before = L.threshold_file_digests(sandbox)
    with pytest.raises(A.ApplyRefused, match=msg):
        A.apply(path, approvals_path=ap, lock=False)
    assert L.threshold_file_digests(sandbox) == before


def test_apply_refuses_a_hand_edited_proposal_and_a_stale_one(sandbox):
    p, path = _proposal(sandbox, sandbox)
    ap = _approve(sandbox, p, ["ou25|Serie B"])
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["entries"][0]["proposed"]["sniper_th"] = 0.04          # someone "tweaks" the value
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(A.ApplyRefused, match="edited"):
        A.apply(path, approvals_path=ap, lock=False)
    p, path = _proposal(sandbox, sandbox)
    ap = _approve(sandbox, p, ["ou25|Serie B"])
    std = json.loads((sandbox / TP.PROD_STANDARD).read_text(encoding="utf-8"))
    std["Serie B"]["sniper_th"] = 0.11                               # production moved after the proposal
    (sandbox / TP.PROD_STANDARD).write_text(json.dumps(std, indent=2), encoding="utf-8")
    with pytest.raises(A.ApplyRefused, match="stale"):
        A.apply(path, approvals_path=ap, lock=False)


# ── 3. apply updates exactly the approved thresholds + the lock snapshot ────────────────────────

def test_apply_updates_only_approved_entries_and_reapproves_the_lock(sandbox):
    assert L.check()["protected_tracks_modified"] is False
    p, path = _proposal(sandbox, sandbox)
    side_before = json.loads((sandbox / TP.PROD_SIDE).read_text(encoding="utf-8"))
    ap = _approve(sandbox, p, ["ou25|Serie B"])                      # Championship btts NOT approved
    r = A.apply(path, approvals_path=ap)
    assert r["applied"] == ["ou25|Serie B"]
    std = json.loads((sandbox / TP.PROD_STANDARD).read_text(encoding="utf-8"))
    assert std["Serie B"]["sniper_th"] == 0.15 and std["Serie B"]["marksman_th"] == 0.13
    assert json.loads((sandbox / TP.PROD_SIDE).read_text(encoding="utf-8")) == side_before
    led = json.loads(L.THRESHOLD_LEDGER_PATH.read_text(encoding="utf-8"))
    assert led["last_applied"]["proposal_id"] == p["proposal_id"]
    assert led["last_applied"]["approved_by"] == "Nevo (owner) - test"
    snap = json.loads(L.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snap["approved_by"] == "Nevo (owner) - test" and p["proposal_id"] in snap["approval_note"]
    assert L.check()["protected_tracks_modified"] is False
    # idempotent: a second apply of the same approval changes nothing
    assert A.apply(path, approvals_path=ap)["already_applied"] == ["ou25|Serie B"]
    # and any change that did NOT come through apply turns the lock RED, even for a non-protected league
    std["League One"]["sniper_th"] = 0.05
    (sandbox / TP.PROD_STANDARD).write_text(json.dumps(std, indent=2), encoding="utf-8")
    r = L.check()
    assert r["protected_tracks_modified"] is True
    assert any(c.startswith("threshold_files.models/best_params_standard.json") for c in r["changes"])
    # --approve cannot bless it: the ledger is untouched by research_lock.approve
    L.approve("someone", path=L.SNAPSHOT_PATH)
    assert L.check()["protected_tracks_modified"] is True


def test_apply_refuses_when_the_lock_is_already_red(sandbox):
    p, path = _proposal(sandbox, sandbox)
    ap = _approve(sandbox, p, ["ou25|Serie B"])
    led = json.loads(L.THRESHOLD_LEDGER_PATH.read_text(encoding="utf-8"))
    led["files"][TP.PROD_SIDE] = "0" * 64                            # an unrelated unapproved change is pending
    L.THRESHOLD_LEDGER_PATH.write_text(json.dumps(led), encoding="utf-8")
    before = L.threshold_file_digests(sandbox)
    with pytest.raises(A.ApplyRefused, match="not clean BEFORE"):
        A.apply(path, approvals_path=ap)
    assert L.threshold_file_digests(sandbox) == before


def test_missing_ledger_fails_closed(tmp_path):
    assert L.threshold_files_changes(ledger={}) and "no applied-threshold ledger" in L.threshold_files_changes({})[0]


# ── 4. served thresholds unchanged when nothing is approved ──────────────────────────────────────

def test_full_propose_run_on_the_real_repo_leaves_every_served_gate_unchanged(tmp_path, monkeypatch):
    from src import gate_resolver as G
    before_files = _bytes()
    before = [g.as_dict() for g in G.all_gates()]
    cand = tmp_path / "cand"
    cand.mkdir()
    for k, v in _candidates(ROOT).items():
        name = "standard" if k == "ou25" else k
        (cand / f"best_params_{name}.json").write_text(json.dumps(v), encoding="utf-8")
    monkeypatch.setattr(TP, "PROPOSAL_DIR", tmp_path / "proposals")
    assert TP.main(["propose", "--candidates", str(cand), "--evidence-dir", str(tmp_path), "--month", "2026-11"]) == 0
    latest = json.loads((tmp_path / "proposals" / "latest.json").read_text(encoding="utf-8"))
    assert latest["n_serving_changes"] == 2 and latest["status"] == "AWAITING_OWNER"
    assert (tmp_path / "proposals" / "2026-11.md").exists()
    # nothing approved -> the committed approvals file has no entry -> apply refuses, nothing served moves
    with pytest.raises(A.ApplyRefused):
        A.apply(tmp_path / "proposals" / "2026-11.json", lock=False)
    assert _bytes() == before_files
    assert [g.as_dict() for g in G.all_gates()] == before
    assert L.check()["protected_tracks_modified"] is False


def test_committed_ledger_and_approvals_are_consistent():
    led = json.loads((ROOT / "config" / "threshold_applied.json").read_text(encoding="utf-8"))
    assert led["files"] == L.threshold_file_digests(ROOT)
    ap = json.loads((ROOT / "config" / "threshold_approvals.json").read_text(encoding="utf-8"))
    assert isinstance(ap["approvals"], list)


# ── 5. workflows parse and stage only the proposal artifacts ─────────────────────────────────────

def _run_blocks(doc):
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            if "run" in step:
                yield step.get("name", ""), step["run"]


def _staged_paths(run: str) -> list[str]:
    """Every path a run block hands to `git add`, including the per-file `for f in ...; do git add` loops."""
    out = []
    flat = run.replace("\\\n", " ")
    for m in re.finditer(r"for [a-z]+ in (.*?); do\s*(.*?)done", flat, flags=re.S):
        if "git add" in m.group(2):
            out += m.group(1).split()
    for ln in flat.splitlines():
        if "git add" in ln:
            toks = ln.split("git add", 1)[1].split("&&")[0].split("||")[0].split()
            out += [t.strip('"') for t in toks if "/" in t and not t.startswith(("-", '"$', "$"))]
    return out


@pytest.mark.parametrize("name", ["backtest_matrix.yml", "backtest.yml", "retrain.yml", "preseason_retrain.yml"])
def test_workflows_parse_and_never_stage_production_threshold_files(name):
    doc = yaml.safe_load((WF / name).read_text(encoding="utf-8"))
    staged = [p for _, run in _run_blocks(doc) for p in _staged_paths(run)]
    assert staged, f"{name}: no staged paths found - the parser missed the commit step"
    bad = [p for p in staged if p in (TP.PROD_STANDARD, TP.PROD_SIDE) or p.startswith("models/best_params_side")
           or p == "models/best_params_standard.json" or p == "output/league_roi_config.json"
           or p == "models/best_params_*.json"]
    assert not bad, (name, bad)


def test_monthly_matrix_stages_only_the_proposal_and_its_evidence():
    doc = yaml.safe_load((WF / "backtest_matrix.yml").read_text(encoding="utf-8"))
    agg = dict(_run_blocks({"jobs": {"aggregate": doc["jobs"]["aggregate"]}}))
    staged = [p for run in agg.values() for p in _staged_paths(run)]
    assert staged and all(p.startswith(("output/threshold_proposals/", "output/backtest_results_",
                                        "output/backtest_by_league_")) for p in staged), staged
    assert not any(p.startswith("output/threshold_proposals/candidates") for p in staged)
    run = "\n".join(agg.values())
    assert "python -m src.threshold_proposals propose" in run
    assert "threshold_apply" not in run                                # the job can never apply
    assert "git diff --exit-code -- models/ config/" in run           # and fails red if it touched them
    up = [s for s in doc["jobs"]["backtest"]["steps"] if s.get("uses", "").startswith("actions/upload-artifact")][0]
    assert "models/" not in up["with"]["path"]


def test_no_workflow_runs_the_applier():
    for p in WF.glob("*.yml"):
        assert "threshold_apply" not in "\n".join(
            ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.strip().startswith("#")), p.name


# ── 6. dashboard notice (display only) ──────────────────────────────────────────────────────────

def test_dashboard_reads_the_proposal_state(tmp_path):
    from dashboard_data import production_view as V
    assert V.threshold_proposal(latest=tmp_path / "none.json") == {}
    lt = tmp_path / "latest.json"
    lt.write_text(json.dumps({"proposal_id": "TP-x", "status": "AWAITING_OWNER", "n_serving_changes": 1,
                              "serving_changes": [{"entry_id": "ou25|Serie B", "change": "THRESHOLD_CHANGE",
                                                   "served_now": {"sniper": 0.12, "marksman": 0.10},
                                                   "served_if_approved": {"sniper": 0.15, "marksman": 0.13},
                                                   "bets_oos": 88, "oos_matchdays": 61, "roi_oos": 6.1,
                                                   "oos_roi_ci95": [-12.0, 24.2]}]}), encoding="utf-8")
    none = tmp_path / "a.json"
    none.write_text(json.dumps({"approvals": []}), encoding="utf-8")
    r = V.threshold_proposal(latest=lt, approvals=none, applied=tmp_path / "x.json")
    from src.tier_labels import TIER_LABEL                                  # column shown as "Big gap ..."
    assert r["state"] == "AWAITING_OWNER" and r["rows"].iloc[0][f"{TIER_LABEL['SNIPER']} if approved"] == 0.15
    none.write_text(json.dumps({"approvals": [{"proposal_id": "TP-x", "approved_by": "Nevo"}]}), encoding="utf-8")
    assert V.threshold_proposal(latest=lt, approvals=none, applied=tmp_path / "x.json")["state"] == "APPROVED_NOT_APPLIED"
    led = tmp_path / "led.json"
    led.write_text(json.dumps({"history": [{"proposal_id": "TP-x", "applied_at": "2026-11-02T09:00:00Z"}]}),
                   encoding="utf-8")
    assert V.threshold_proposal(latest=lt, approvals=none, applied=led)["state"] == "APPLIED"
