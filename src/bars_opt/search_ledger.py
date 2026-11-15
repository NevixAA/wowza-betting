"""Everything the bars studies have SEARCHED or TESTED so far, counted from their committed outputs in git
history (owner's strict PROVEN rule: "a CI above 0 after Holm, counting every configuration searched").

Each distinct committed version (git blob) of a study output is one run. Counted per run:
  bars_opt SELECT          output/bars_opt/study/choices.json            sum of configs_searched
  bars_opt FINAL           output/bars_opt/study/cells.json              cells tested (the run's Holm family_size)
  bars_opt monthly         output/bars_opt/study/monthly.json            simulated months x configs per family,
                                                                         + the FINAL monthly-rolling cycle
                                                                         (3 months x configs per family)
  bars_opt serving cells   output/threshold_proposals/bars_opt_serving/<month>.json   configs per family
  bars_opt monthly job     output/threshold_proposals/bars_opt/<month>.json           configs per family
  league_bars              output/threshold_proposals/league_bars/<track>/league_bars_evidence.json
                           grid bars per league x market given bars + every tested row
Configurations per family = len(study.configs()) (54 since amendment 3; a choices.json records its own
configs_searched, which is used where present). Nothing is estimated: a file that cannot be read stops the
count (LedgerError), it is never skipped.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "select": ["output/bars_opt/study/choices.json"],
    "final": ["output/bars_opt/study/cells.json"],
    "monthly": ["output/bars_opt/study/monthly.json"],
    "serving": "output/threshold_proposals/bars_opt_serving",
    "monthly_job": "output/threshold_proposals/bars_opt",
    "league_bars": ["output/threshold_proposals/league_bars/standard/league_bars_evidence.json",
                    "output/threshold_proposals/league_bars/new_format/league_bars_evidence.json"],
}
#: the FINAL monthly-rolling cycle re-selects in these 3 months (study.monthly_final_family)
ROLLING_MONTHS = 3
MONTH_FILE = re.compile(r"\d{4}-\d{2}\.json")


class LedgerError(RuntimeError):
    pass


def _git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise LedgerError(f"git {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def versions(path: str, ref: str = "HEAD") -> list[tuple[str, str]]:
    """[(commit, blob)] of every distinct committed version of `path` reachable from ref, oldest first."""
    out, seen = [], set()
    for c in reversed(_git("log", "--format=%H", ref, "--", path).split()):
        try:
            blob = _git("rev-parse", f"{c}:{path}").strip()
        except LedgerError:
            continue                                    # the commit deleted the file
        if blob not in seen:
            seen.add(blob)
            out.append((c, blob))
    return out


def _load(blob: str) -> object:
    try:
        return json.loads(_git("cat-file", "-p", blob))
    except json.JSONDecodeError as e:
        raise LedgerError(f"blob {blob} is not JSON: {e}") from e


def _history_paths(prefix: str, ref: str = "HEAD") -> list[str]:
    """Every <prefix>/<YYYY-MM>.json ever committed (also ones later removed); replays are not searches."""
    names = set(_git("log", "--format=", "--name-only", ref, "--", prefix).split())
    return sorted(n for n in names if n.count("/") == prefix.count("/") + 1 and MONTH_FILE.fullmatch(Path(n).name))


def build(ref: str = "HEAD") -> dict:
    from src.bars_opt import study as S
    from src import league_bars as LB
    per_family = len(S.configs())
    runs = []
    for path in PATHS["select"]:
        for c, b in versions(path, ref):
            d = _load(b)
            runs.append({"kind": "bars_opt_select", "commit": c[:9], "path": path,
                         "families": len(d), "configs": int(sum(int(x["configs_searched"]) for x in d)),
                         "tested": 0})
    for path in PATHS["final"]:
        for c, b in versions(path, ref):
            d = _load(b)
            # the run's own Holm family size (every cell it tested); else the cells that carry a p-value
            sizes = {x.get("family_size") for x in d if x.get("family_size") is not None}
            tested = max(sizes) if sizes else sum(1 for x in d if (x.get("final") or {}).get("p_one_sided") is not None)
            runs.append({"kind": "bars_opt_final", "commit": c[:9], "path": path, "families": None,
                         "configs": 0, "tested": int(tested)})
    for path in PATHS["monthly"]:
        for c, b in versions(path, ref):
            fams = _load(b)["sim_select"]["families"]
            months = sum(len(f.get("months") or []) for f in fams)
            runs.append({"kind": "bars_opt_monthly", "commit": c[:9], "path": path, "families": len(fams),
                         "configs": int((months + ROLLING_MONTHS * len(fams)) * per_family), "tested": 0})
    for kind, prefix in (("bars_opt_serving", PATHS["serving"]), ("bars_opt_monthly_job", PATHS["monthly_job"])):
        for path in _history_paths(prefix, ref):
            for c, b in versions(path, ref):
                d = _load(b)
                # serving evidence {"families": [...]}; monthly job {"entries": [...]}; older lists
                fams = d.get("families", d.get("entries")) if isinstance(d, dict) else d
                if fams is None:
                    raise LedgerError(f"{path}@{c[:9]}: neither families nor entries")
                n = len({(f["track"], f["mside"]) for f in fams})
                runs.append({"kind": kind, "commit": c[:9], "path": path, "families": n,
                             "configs": int(n * per_family), "tested": 0})
    for path in PATHS["league_bars"]:
        for c, b in versions(path, ref):
            rows = _load(b)["rows"]
            with_bars = sum(1 for r in rows if r.get("valuable_th") is not None)
            tested = sum(1 for r in rows if r.get("p_holm") is not None)
            runs.append({"kind": "league_bars", "commit": c[:9], "path": path, "families": None,
                         "configs": int(with_bars * len(LB.GRID)), "tested": int(tested)})
    return {"ref": _git("rev-parse", ref).strip()[:9], "configs_per_family": per_family, "runs": runs,
            "configs_searched": int(sum(r["configs"] for r in runs)),
            "cells_tested": int(sum(r["tested"] for r in runs)),
            "hypotheses": int(sum(r["configs"] + r["tested"] for r in runs))}
