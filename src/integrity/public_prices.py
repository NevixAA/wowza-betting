"""No bookmaker price in a public file (owner 2026-11-24: "so fix it! especially the public odds").

API-Football and OddsAPI prices live only in the private store (docs/AF_PRICES_PRIVATE_MIGRATION.md,
config/price_store_paths.json, scripts/price_store_live.py). This check reads the COMMITTED tree
(git ls-files, header / schema only - never a value) and reports:

  retired   a committed file on a path whose class has been retired from the public repo
            ("public_retired" in config/price_store_paths.json): must be 0
  unclassified
            a committed CSV / Parquet under output/, data/, reports/, docs/ or experiments/ with a
            per-row price column that no class rule covers and config/public_price_allowlist.json
            does not explain: must be 0. A new price file must be classified (A / B / O: it then
            moves with its class) or allowlisted with a reason (class C statistics, class F
            football-data.co.uk, model fair odds, FPL prices)
  pending   price files of classes that are still public (not retired yet): counted, not failed

Counts and paths only: no value of any file is read past its header.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATHS = ROOT / "config" / "price_store_paths.json"
ALLOW = ROOT / "config" / "public_price_allowlist.json"
SCAN = re.compile(r"^(output|data|reports|docs|experiments)/.+\.(csv|parquet)$")
#: a per-row price column (doc section 1.1): odds / price / implied / overround / p_market / fair odds, or a
#: football-data.co.uk bookmaker column (B365H, PSCH, MaxC>2.5, ...)
PRICE_COL = re.compile(
    r"(^|_)(odds|price|prices|implied|overround|p_market|market_prob|devig)(_|$)"
    r"|^(b365|bw|iw|ps|wh|vc|max|avg|bb|bbmx|bbav|lb|sj|gb|sb|so|sy)c?(h|d|a|>2\.5|<2\.5|ahh|aha)$", re.I)
#: aggregates and metadata that merely mention a price word (class C / not a price)
NOT_PRICE = re.compile(
    r"^(avg|mean|median|min|max|n|count|has|is|num|share|pct|real)_"
    r"|_(min|max|band|bands|source|sources|ts|at|time|utc|status|reason|flag|n|count|book|books|bookmaker|"
    r"key|id|ok|missing|kind|type|rule|class|provenance|capture|coverage|method|derived|seen|age_min|age_h)$", re.I)


def _cfg(path: Path = PATHS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _allow(path: Path = ALLOW) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for e in json.loads(Path(path).read_text(encoding="utf-8")).get("entries", []):
        out.append({**e, "_rx": re.compile(e["path"])})
    return out


def committed(root: Path = ROOT) -> list[str] | None:
    p = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True)
    if p.returncode != 0:
        return None
    return [f for f in p.stdout.decode("utf-8", "replace").split("\0") if f]


def header(root: Path, rel: str) -> list[str]:
    p = root / rel
    if rel.endswith(".csv"):
        with open(p, encoding="utf-8", errors="replace", newline="") as fh:
            import csv
            row = next(csv.reader([fh.readline()]), [])
        return [c.strip() for c in row]
    import pyarrow.parquet as pq
    return list(pq.read_schema(p).names)


def price_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if PRICE_COL.search(c) and not NOT_PRICE.search(c)]


def scan(repo: str = "wowza-betting", root: Path = ROOT, files: list[str] | None = None,
         cfg: dict | None = None, allow: list[dict] | None = None) -> dict:
    cfg = cfg if cfg is not None else _cfg()
    allow = allow if allow is not None else _allow()
    rules = cfg["rules"].get(repo) or {}
    retired = (cfg.get("public_retired") or {}).get(repo) or []
    rx_ret = [re.compile(p) for c in retired for p in rules.get(c, [])]
    rx_all = [re.compile(p) for c in rules.values() for p in c]
    files = files if files is not None else committed(root)
    res = {"repo": repo, "retired_classes": retired, "files_scanned": 0, "retired": [], "unclassified": [],
           "pending": 0, "allowlisted": 0, "unreadable": []}
    if files is None:
        res["error"] = "not a git tree"
        return res
    for rel in files:
        if any(r.search(rel) for r in rx_ret):
            res["retired"].append(rel)
            continue
        if not SCAN.match(rel):
            continue
        res["files_scanned"] += 1
        try:
            cols = price_columns(header(root, rel))
        except Exception:                                              # noqa: BLE001 - LFS stub, bad file
            res["unreadable"].append(rel)
            continue
        if not cols:
            continue
        if any(r.search(rel) for r in rx_all):
            res["pending"] += 1                                        # classified; moves with its class
            continue
        left = [c for c in cols if not any(e["_rx"].search(rel) and (e.get("columns") == "*" or c in e.get("columns", []))
                                             for e in allow)]
        if left:
            res["unclassified"].append({"path": rel, "columns": left[:8]})
        else:
            res["allowlisted"] += 1
    return res


def check(repo: str = "wowza-betting", **kw) -> tuple[bool, str, str]:
    """(ok, value, detail) for the data truth stoplight."""
    r = scan(repo, **kw)
    if r.get("error"):
        return False, r["error"], ""
    ok = not r["retired"] and not r["unclassified"]
    value = (f"{len(r['retired'])} retired price file(s) committed, {len(r['unclassified'])} unclassified price "
             f"file(s); {r['pending']} classified price file(s) still public (classes not yet retired: their "
             f"move is pending), {r['allowlisted']} allowlisted (not bookmaker prices / class F), "
             f"{r['files_scanned']} scanned")
    detail = "; ".join([f"retired: {p}" for p in r["retired"][:10]] +
                       [f"unclassified: {u['path']} {u['columns']}" for u in r["unclassified"][:10]])
    return ok, value, detail


if __name__ == "__main__":
    import sys
    print(json.dumps(scan(sys.argv[1] if len(sys.argv) > 1 else "wowza-betting"), indent=1))
