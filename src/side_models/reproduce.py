"""Reproducible side-model training (corners / cards): re-create the served artifact from a RECORDED data ref.
=========================================================================================================

    python -m src.side_models.reproduce --out DIR
        the recorded recipe (config/side_models_recipe.json): inputs read from the git tree of its `data_ref`,
        its `until` cut-off and `trained_at` stamp -> DIR/side_models.json + holdout_metrics.csv, compared with
        the committed models/side_models/ (exit 1 when they differ).
    python -m src.side_models.reproduce --out DIR --evidence reports/lineage/side_models_lineage_evidence.json
        the same, and writes the lineage evidence src.integrity.training_lineage reads.
    python -m src.side_models.reproduce --out DIR --data-ref HEAD --until YYYY-MM-DD
        a REFIT CANDIDATE on newer committed data (artifact only; the comparison is reported, never enforced).

WHY. src/side_models/train.py was run once, offline, on 2026-11-18 (commit 156374b87) and its output committed.
The inputs it read are live files that the daily collectors rewrite (af fill, competition fixtures, fd_raw,
the side-stats sidecar), so "run train.py again" does NOT give the served model. The exact inputs are kept by
git: the tree of the commit that installed the artifact. This module reads them from there.

NEVER INSTALLS. --out inside models/ is refused; nothing here writes the serving artifact, the registry or
any ledger. No price is read. No API call is made.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.side_models import data as D  # noqa: E402
from src.side_models import model as M  # noqa: E402
from src.side_models import train as T  # noqa: E402

RECIPE = ROOT / "config" / "side_models_recipe.json"
ARTIFACT_DIR = ROOT / "models" / "side_models"
EVIDENCE = ROOT / "reports" / "lineage" / "side_models_lineage_evidence.json"
#: every file train.run reads (a path absent at the data ref is simply not there - e.g. the sidecar before 11-19)
INPUTS = ("output/collect/af_history_fullstats_fill.parquet", "output/strength/competition_fixtures.parquet",
          "output/collect/fd_raw", "output/collect/af_side_stats.parquet", "config/leagues.json",
          "config/side_models.json")
CODE = ("src/side_models/data.py", "src/side_models/features.py", "src/side_models/model.py",
        "src/side_models/train.py")
OUTPUT_FILES = ("side_models.json", "holdout_metrics.csv")


def _git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text).stdout


def resolve(ref: str) -> str:
    return _git("rev-parse", f"{ref}^{{commit}}").strip()


def tree_blobs(ref: str, paths=INPUTS) -> dict:
    """{path: git blob id} of every file under `paths` in the tree of `ref` (platform independent)."""
    out = {}
    for line in _git("ls-tree", "-r", ref, "--", *paths).splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return dict(sorted(out.items()))


def digest(blobs: dict) -> str:
    return hashlib.sha256("".join(f"{p} {b}\n" for p, b in sorted(blobs.items())).encode()).hexdigest()


def materialise(ref: str, dest: Path) -> dict:
    """Extract the INPUTS of `ref`'s tree into `dest`; returns their blob ids."""
    blobs = tree_blobs(ref)
    present = sorted({p if not p.startswith("output/collect/fd_raw/") else "output/collect/fd_raw" for p in blobs})
    raw = subprocess.run(["git", "archive", "--format=tar", ref, "--", *present], cwd=ROOT, check=True,
                         capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        tf.extractall(dest, filter="data")
    return blobs


@contextmanager
def inputs_at(root: Path):
    """Point the training modules at the input tree `root` (restored afterwards)."""
    saved = (D.ROOT, D.FILL, D.FIXTURES, D.FD_RAW, D.SIDECAR, T.CONFIG, T.LEAGUES)
    D.ROOT = root
    D.FILL = root / "output" / "collect" / "af_history_fullstats_fill.parquet"
    D.FIXTURES = root / "output" / "strength" / "competition_fixtures.parquet"
    D.FD_RAW = root / "output" / "collect" / "fd_raw"
    D.SIDECAR = root / "output" / "collect" / "af_side_stats.parquet"
    T.CONFIG = root / "config" / "side_models.json"
    T.LEAGUES = root / "config" / "leagues.json"
    try:
        yield
    finally:
        D.ROOT, D.FILL, D.FIXTURES, D.FD_RAW, D.SIDECAR, T.CONFIG, T.LEAGUES = saved


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _eol(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def prob_parity(table, new_art: dict, ref_art: dict, cfg: dict) -> dict:
    """Max |p_new - p_ref| per market x side x line over every training fixture of every league x family both
    artifacts serve, with the features of the reproduced table (the serving probability path, M.p_over)."""
    res = {"probabilities": 0, "max_abs_prob_diff": 0.0, "only_in_new": [], "only_in_ref": []}
    for fam, fc in cfg["families"].items():
        new_m, ref_m = new_art["models"].get(fam, {}), ref_art["models"].get(fam, {})
        res["only_in_new"] += [f"{fam}/{lg}" for lg in sorted(set(new_m) - set(ref_m))]
        res["only_in_ref"] += [f"{fam}/{lg}" for lg in sorted(set(ref_m) - set(new_m))]
        for lg in sorted(set(new_m) & set(ref_m)):
            fr = T.training_frame(table, lg, fam, cfg["features"], new_art["coverage"][fam][lg]["window_from"])
            if fr.empty:
                continue
            for market, mc in fc["markets"].items():
                for side in (["total"] if mc["part"] == "total" else ["home", "away"]):
                    pn, pr = new_m[lg]["params"][side], ref_m[lg]["params"][side]
                    mu_n, mu_r = M.mu_of(pn, fr, side), M.mu_of(pr, fr, side)
                    for line in mc["lines"]:
                        d = np.abs(M.p_over(mu_n, pn["alpha"], line) - M.p_over(mu_r, pr["alpha"], line))
                        res["max_abs_prob_diff"] = max(res["max_abs_prob_diff"], float(np.nanmax(d)))
                        res["probabilities"] += int(len(d))
    return res


def compare(out_dir: Path, ref_dir: Path, table, cfg: dict) -> dict:
    files = {}
    for f in OUTPUT_FILES:
        a, b = (out_dir / f).read_bytes(), (ref_dir / f).read_bytes()
        files[f] = {"raw_bytes_equal": a == b, "bytes_equal_eol_normalised": _eol(a) == _eol(b),
                    "sha256_new": _sha256(a), "sha256_ref": _sha256(b)}
    new_art = json.loads((out_dir / "side_models.json").read_text(encoding="utf-8"))
    ref_art = json.loads((ref_dir / "side_models.json").read_text(encoding="utf-8"))
    strip = lambda d: {k: v for k, v in d.items() if k != "trained_at"}                        # noqa: E731
    probs = prob_parity(table, new_art, ref_art, cfg)
    byte_ok = all(v["bytes_equal_eol_normalised"] for v in files.values())
    return {"files": files, "model_version_new": new_art["model_version"],
            "model_version_ref": ref_art["model_version"],
            "json_equal_except_trained_at": strip(new_art) == strip(ref_art),
            "probability": probs,
            "byte_parity": byte_ok,
            "probability_parity": probs["max_abs_prob_diff"] == 0.0 and not probs["only_in_new"]
            and not probs["only_in_ref"]}


def libraries() -> dict:
    import pandas
    import pyarrow
    import scipy
    return {"python": platform.python_version(), "pandas": pandas.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "pyarrow": pyarrow.__version__, "platform": platform.system()}


def run(out_dir: Path, *, data_ref: str | None = None, until: str | None = None, trained_at: str | None = None,
        ref_dir: Path | None = None, recipe_path: Path | None = None) -> dict:
    out_dir = Path(out_dir).resolve()
    try:
        out_dir.relative_to((ROOT / "models").resolve())
        raise SystemExit(f"{out_dir} is inside models/: refused (this never installs a model)")
    except ValueError:
        pass
    recipe = json.loads(Path(recipe_path or RECIPE).read_text(encoding="utf-8"))
    recorded = data_ref is None and until is None
    ref = resolve(data_ref or recipe["data_ref"])
    until = until if until is not None else (recipe.get("until") if recorded else None)
    stamp = trained_at or (recipe["trained_at"] if recorded else None)
    with tempfile.TemporaryDirectory(prefix="side_inputs_") as tmp:
        blobs = materialise(ref, Path(tmp))
        with inputs_at(Path(tmp)):
            res = T.run(write=True, out_dir=out_dir, until=until, trained_at=stamp)
            cfg = T.load_config()
            cmp = compare(out_dir, Path(ref_dir or ARTIFACT_DIR), res["table"], cfg)
    report = {
        "mode": "recorded_recipe" if recorded else "refit_candidate",
        "data_ref": ref, "until": until, "trained_at": stamp,
        "inputs": blobs, "inputs_digest": digest(blobs),
        "inputs_match_recipe": digest(blobs) == recipe.get("inputs_digest"),
        "code_blobs_head": tree_blobs("HEAD", CODE) if _has_head() else None,
        "model_version": res["artifact"]["model_version"], "data_end": res["artifact"]["data_end"],
        "libraries": libraries(), "comparison": cmp,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    report["parity"] = bool(cmp["byte_parity"] and cmp["probability_parity"])
    (out_dir / "reproduction.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return report


def _has_head() -> bool:
    try:
        _git("rev-parse", "HEAD")
        return True
    except subprocess.CalledProcessError:
        return False


def evidence(report: dict, artifact: Path | None = None) -> dict:
    """The lineage record src.integrity.training_lineage checks against the committed artifact's sha256."""
    art = Path(artifact or (ARTIFACT_DIR / "side_models.json"))
    ok = report["mode"] == "recorded_recipe" and report["parity"] and report["inputs_match_recipe"]
    c = report["comparison"]
    return {
        "artifact": "side_models_lineage_evidence", "version": "side-lineage-1.0",
        "generated_by": "python -m src.side_models.reproduce --evidence", "generated_at": report["generated_at"],
        "payload": "side_models/side_models.json", "artifact_sha256": _sha256(art.read_bytes()),
        "model_version": c["model_version_ref"], "recipe": "config/side_models_recipe.json",
        "data_ref": report["data_ref"], "inputs_digest": report["inputs_digest"], "until": report["until"],
        "data_end": report["data_end"], "libraries": report["libraries"],
        "reproduction": {"byte_parity_eol_normalised": c["byte_parity"],
                         "raw_bytes_equal": {f: v["raw_bytes_equal"] for f, v in c["files"].items()},
                         "json_equal_except_trained_at": c["json_equal_except_trained_at"],
                         "max_abs_prob_diff": c["probability"]["max_abs_prob_diff"],
                         "probabilities_compared": c["probability"]["probabilities"]},
        "status": "PASS" if ok else "FAIL",
        "summary": (f"re-trained from the git tree of {report['data_ref'][:9]} with the recorded recipe: "
                    f"{'byte-identical (line endings normalised)' if c['byte_parity'] else 'NOT byte-identical'}, "
                    f"max |p diff| {c['probability']['max_abs_prob_diff']} over "
                    f"{c['probability']['probabilities']} probabilities"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output directory (never inside models/)")
    ap.add_argument("--data-ref", default=None, help="git ref whose tree holds the inputs (default: the recipe's)")
    ap.add_argument("--until", default=None, help="data cut-off YYYY-MM-DD (refit candidates)")
    ap.add_argument("--trained-at", default=None, help="pin the trained_at stamp")
    ap.add_argument("--evidence", default=None, help="also write the lineage evidence file here")
    a = ap.parse_args(argv)
    rep = run(Path(a.out), data_ref=a.data_ref, until=a.until, trained_at=a.trained_at)
    c = rep["comparison"]
    print(f"mode {rep['mode']}  data_ref {rep['data_ref'][:9]}  until {rep['until']}  model_version {rep['model_version']}")
    print(f"inputs_match_recipe {rep['inputs_match_recipe']}  byte_parity {c['byte_parity']}  "
          f"json_equal_except_trained_at {c['json_equal_except_trained_at']}  "
          f"max_abs_prob_diff {c['probability']['max_abs_prob_diff']} over {c['probability']['probabilities']}")
    if a.evidence:
        ev = evidence(rep)
        p = Path(a.evidence)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ev, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"evidence {p}: {ev['status']}")
    if rep["mode"] == "recorded_recipe":
        return 0 if rep["parity"] and rep["inputs_match_recipe"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
