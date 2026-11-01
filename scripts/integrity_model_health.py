"""output/model_integrity_health.json aggregator (R2 shared artifact; skeleton, R2-C).

    python scripts/integrity_model_health.py

Run AFTER the other integrity_* scripts: it reads their health files plus the committed payloads and
board. See src/integrity/model_health.py for what is filled and what stays UNKNOWN.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> int:
    from src.integrity import model_health as MH
    doc = MH.write(ROOT)
    for k, t in doc["tracks"].items():
        print(f"{k:18s} {t['status']:7s} unknown={t.get('unknown_checks', [])}")
    print("overall", doc["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
