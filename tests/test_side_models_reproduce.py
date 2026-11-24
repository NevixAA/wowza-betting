"""Side models (corners / cards): reproducible training workflow, recipe record, lineage, track registry."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integrity import training_lineage as TL  # noqa: E402

ART = ROOT / "models" / "side_models" / "side_models.json"
RECIPE = ROOT / "config" / "side_models_recipe.json"
EVID = ROOT / "reports" / "lineage" / "side_models_lineage_evidence.json"
WF = ROOT / ".github" / "workflows" / "train_side_models.yml"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_recipe_records_the_served_artifact():
    rec = json.loads(RECIPE.read_text(encoding="utf-8"))
    art = json.loads(ART.read_text(encoding="utf-8"))
    assert rec["model_version"] == art["model_version"]
    assert rec["trained_at"] == art["trained_at"]
    assert re.fullmatch(r"[0-9a-f]{40}", rec["data_ref"])
    assert rec["inputs"] and rec["inputs_digest"]
    assert "never automatic" in rec["scheduled_refit"]
    for k in ("data_end", "api_football_fullstats_fill", "competition_fixtures", "football_data_raw"):
        assert rec["data_cutoffs"][k]


def test_evidence_names_the_committed_artifact_and_passed():
    ev = json.loads(EVID.read_text(encoding="utf-8"))
    assert ev["artifact_sha256"] == _sha(ART)
    assert ev["status"] == "PASS"
    assert ev["reproduction"]["byte_parity_eol_normalised"] is True
    assert ev["reproduction"]["max_abs_prob_diff"] == 0.0
    assert ev["reproduction"]["probabilities_compared"] > 0
    assert TL.side_lineage(ROOT)["status"] == "PASS"


def _root(tmp: Path, art: dict | None, ev: dict | None, recipe: dict | None) -> Path:
    if art is not None:
        (tmp / "models" / "side_models").mkdir(parents=True)
        (tmp / "models" / "side_models" / "side_models.json").write_text(json.dumps(art), encoding="utf-8")
    if ev is not None:
        (tmp / "reports" / "lineage").mkdir(parents=True)
        (tmp / "reports" / "lineage" / "side_models_lineage_evidence.json").write_text(json.dumps(ev), encoding="utf-8")
    if recipe is not None:
        (tmp / "config").mkdir(parents=True)
        (tmp / "config" / "side_models_recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    return tmp


def test_side_lineage_states(tmp_path):
    art = {"model_version": "side_nb_v1_x"}
    assert TL.side_lineage(_root(tmp_path / "none", None, None, None)) is None
    r = _root(tmp_path / "a", art, None, {"model_version": "side_nb_v1_x"})
    assert TL.side_lineage(r)["status"] == "UNVERIFIED"
    sha = hashlib.sha256(json.dumps(art).encode()).hexdigest()
    ok_ev = {"artifact_sha256": sha, "model_version": "side_nb_v1_x", "status": "PASS", "summary": "s"}
    assert TL.side_lineage(_root(tmp_path / "b", art, ok_ev, {"model_version": "side_nb_v1_x"}))["status"] == "PASS"
    # the artifact changed since the evidence was written
    assert TL.side_lineage(_root(tmp_path / "c", art, {**ok_ev, "artifact_sha256": "0" * 64},
                                 {"model_version": "side_nb_v1_x"}))["status"] == "UNVERIFIED"
    # the recipe does not record the served model
    assert TL.side_lineage(_root(tmp_path / "d", art, ok_ev, {"model_version": "other"}))["status"] == "UNVERIFIED"
    # a failed reproduction stays FAIL
    assert TL.side_lineage(_root(tmp_path / "e", art, {**ok_ev, "status": "FAIL"},
                                 {"model_version": "side_nb_v1_x"}))["status"] == "FAIL"


def test_reproduce_never_writes_into_models():
    from src.side_models import reproduce as R
    with pytest.raises(SystemExit):
        R.run(ROOT / "models" / "side_models")


def test_workflow_never_writes_main_and_never_installs():
    """Monthly schedule + dispatch (owner 2026-11-24); read-only by default; the candidate job pushes ONLY
    side-models/ branches and opens a PR awaiting owner approval - no push to main, no merge."""
    t = WF.read_text(encoding="utf-8")
    on = t.split("\non:", 1)[1].split("\npermissions:", 1)[0]
    assert "workflow_dispatch" in on and re.search(r"cron: '\d+ \d+ \d+ \* \*'", on)          # monthly
    assert "push:" not in on and "pull_request" not in on
    assert re.search(r"^permissions:\s*\n\s+contents: read", t, re.M)
    pushes = [ln.strip() for ln in t.splitlines() if "git push" in ln]
    assert pushes and all("$br" in ln or "refs/heads/side-models/" in ln for ln in pushes), pushes
    assert 'br="side-models/candidate-' in t
    for bad in ("HEAD:main", "origin main", "refs/heads/main", "gh pr merge", "--admin"):
        assert bad not in t, bad
    assert "awaiting owner approval" in t and "upload-artifact" in t
    train = t.split("\n  train:", 1)[1].split("\n  candidate:", 1)[0]
    assert "contents: write" not in train and "git push" not in train


def test_track_registry_lists_side_models():
    reg = json.loads((ROOT / "config" / "track_registry.json").read_text(encoding="utf-8"))
    s = reg["side_models"]
    art = json.loads(ART.read_text(encoding="utf-8"))
    assert s["model_version"] == art["model_version"]
    assert s["lineage"]["status"] == "PASS"
    assert s["scheduled_refit"].startswith("MONTHLY CANDIDATE") and "never automatic" in s["scheduled_refit"]
    for fam, cov in art["coverage"].items():
        served = {lg for lg, c in cov.items() if c.get("covered")}
        assert {lg for lg, v in s["families"][fam]["leagues"].items() if v["served"]} == served


def test_build_until_cuts_rows(tmp_path):
    """A data cut-off leaves out every fixture after it (and changes nothing when absent)."""
    import pandas as pd
    from src.side_models import data as D
    fx = pd.DataFrame({"fixture_id": [1, 2], "league": ["L", "L"], "season": [2026, 2026],
                       "date": ["2026-01-01", "2026-02-01"], "status": ["FT", "FT"], "home_id": [10, 11],
                       "away_id": [11, 10], "home_name": ["a", "b"], "away_name": ["b", "a"],
                       "ft_home": [1, 0], "ft_away": [0, 2]})
    fill = pd.DataFrame({"fixture_id": [1, 2], "home_id": [10, 11], "away_id": [11, 10], "home_corners": [5, 6],
                         "away_corners": [4, 3], "home_yellow": [1, 2], "away_yellow": [2, 1],
                         "home_red": [0, 0], "away_red": [0, 0]})
    fx.to_parquet(tmp_path / "fx.parquet")
    fill.to_parquet(tmp_path / "fill.parquet")
    kw = dict(single_source={"L": "2020-01-01"}, fixtures_path=tmp_path / "fx.parquet",
              fill_path=tmp_path / "fill.parquet", sidecar_path=tmp_path / "absent.parquet")
    full, _ = D.build([], **kw)
    cut, audit = D.build([], until="2026-01-15", **kw)
    assert list(full["fixture_id"]) == [1, 2]
    assert list(cut["fixture_id"]) == [1] and audit["L"]["af_fixtures_in_window"] == 1


def test_cross_platform_tolerance_is_a_float_noise_bound():
    from src.side_models import reproduce as R
    assert 0 < R.PROB_TOL <= 1e-9
