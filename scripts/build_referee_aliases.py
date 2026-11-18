"""Build player_model/referee_aliases.json from EVIDENCE: every spelling each fixture's referee
was ever stored with, across every committed version of player_history.parquet on the given
refs, plus the referee differences a collect recorded in the corrections sidecar.

Rules (player_model/referee_names.py): two spellings are one referee only when the same fixture
carried both AND the names are compatible (initial/first name + surname tokens); per league; a
group bridged to two different full names is ambiguous and left unmapped.

    python scripts/build_referee_aliases.py                       # origin/main + origin/data-truth-100
    python scripts/build_referee_aliases.py --refs origin/main --check   # exit 1 if the file is stale

Reads git objects only (no API call).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import history_integrity as HI  # noqa: E402
from player_model import referee_names as RN  # noqa: E402

HISTORY = "player_history.parquet"


def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], capture_output=True, check=True, cwd=ROOT)
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def history_blobs(refs: list[str]) -> list[str]:
    blobs: set[str] = set()
    for ref in refs:
        for c in _git("log", "--format=%H", ref, "--", HISTORY).split():
            for line in _git("ls-tree", "-r", c, "--", HISTORY).splitlines():
                meta, _path = line.split("\t", 1)
                blobs.add(meta.split()[2])
    return sorted(blobs)


def observations(refs: list[str], corrections: Path | None = None) -> dict:
    obs: dict = defaultdict(set)

    def add(frame: pd.DataFrame):
        for f, lg, r in frame[["fixture_id", "league", "referee"]].drop_duplicates().itertuples(index=False):
            if isinstance(r, str) and r.strip():
                obs[(int(f), str(lg))].add(r.strip())

    for b in history_blobs(refs):
        d = pd.read_parquet(io.BytesIO(_git("cat-file", "blob", b, binary=True)))
        if {"fixture_id", "league", "referee"}.issubset(d.columns):
            add(d)
    if corrections is not None and corrections.exists():
        c = pd.read_parquet(corrections)
        c = c[c["column"] == "referee"]
        for col in ("stored_value", "fresh_value"):
            add(c.rename(columns={col: "referee"}))
    return obs


def build(refs: list[str]) -> dict:
    obs = observations(refs, HI.CORRECTIONS_PATH)
    amap, report = RN.derive_aliases(obs)
    return {"version": RN.ALIASES_VERSION,
            "built_from": {"refs": refs, "history_path": HISTORY,
                           "corrections": str(HI.CORRECTIONS_PATH.relative_to(ROOT)).replace("\\", "/")},
            "rule": "same fixture stored with both spellings AND compatible names (initial/first "
                    "name + surname tokens), per league; ambiguous groups unmapped",
            "report": report,
            "map": amap}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", default=["origin/main", "origin/data-truth-100"])
    ap.add_argument("--out", default=str(RN.ALIASES_PATH))
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from a rebuild")
    a = ap.parse_args(argv)
    d = build(a.refs)
    text = json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    out = Path(a.out)
    r = d["report"]
    print(f"[referee-aliases] {r['fixtures_observed']} fixtures, {r['fixtures_with_several_spellings']} "
          f"with several spellings, {r['aliases']} aliases, {len(r['ambiguous_groups'])} ambiguous "
          f"group(s), {r['incompatible_pairs_not_aliased']} incompatible pair(s) not aliased")
    if a.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print("[referee-aliases] up to date" if same else "[referee-aliases] STALE")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"[referee-aliases] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
