"""python -m src.contracts [--json] — validate every committed contract file (read-only)."""
from __future__ import annotations

import json
import sys

from src.contracts import CONTRACTS_VERSION, committed_violations


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    r = committed_violations()
    if "--json" in argv:
        print(json.dumps({"contracts_version": CONTRACTS_VERSION, "files": r}, indent=1, default=str))
        return 0
    for f, x in r.items():
        print(f"{x.get('n_violations', 0):3d} violation(s)  {x['schema']:30s} {f} ({x.get('rows', 0)} rows"
              + ("" if x.get("present", True) else ", ABSENT") + ")")
        for v in x.get("violations", []):
            print(f"      - {v.get('kind')} {v.get('column')}: {v.get('n_rows')} row(s) {v.get('detail', '')} "
                  f"e.g. {v.get('examples', [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
