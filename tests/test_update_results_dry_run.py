"""`update_results.py --dry-run` writes no ledger (go-live rehearsal 2026-11-17).

The rehearsal's offline run found update_ht_results and update_ht_observations ignoring --dry-run: they wrote
output/ht_ledger.csv (2 rows graded) and output/ht_observations.csv (8 filled). Every grader now takes dry_run,
main() passes it to each, and no grader writes a CSV outside a `not dry_run` guard."""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import update_results as U  # noqa: E402

GRADERS = ("update_player_results", "update_sharp_results", "update_side_market_results", "update_ht_results",
           "update_ht_observations", "update_live_signal_results")


def test_every_grader_takes_dry_run():
    for name in GRADERS:
        assert "dry_run" in inspect.signature(getattr(U, name)).parameters, name


def test_main_passes_dry_run_to_every_grader():
    src = (ROOT / "update_results.py").read_text(encoding="utf-8")
    for name in GRADERS:
        calls = re.findall(rf"\b{name}\(([^)]*)\)", src.split("def main", 1)[-1] if "def main" in src else src)
        calls = [c for c in calls if c.strip()]        # the call sites, not the definitions
        assert calls and all("dry_run=args.dry_run" in c for c in calls), (name, calls)


def test_the_ht_writers_are_guarded():
    for name in ("update_ht_results", "update_ht_observations"):
        tree = ast.parse(inspect.getsource(getattr(U, name)).replace("\r\n", "\n"))
        guarded = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "dry_run" in ast.unparse(node.test):
                guarded += sum(1 for n in ast.walk(node) if isinstance(n, ast.Call)
                               and getattr(n.func, "attr", "") == "to_csv")
        total = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "to_csv")
        assert total >= 1 and guarded == total, (name, total, guarded)
