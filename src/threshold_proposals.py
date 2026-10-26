"""Per-league threshold PROPOSALS (owner rule 2026-10-26: the monthly backtest PROPOSES, the owner approves).

    python -m src.threshold_proposals propose --candidates <dir> [--evidence-dir output] [--month YYYY-MM]
    python -m src.threshold_proposals status

WHY. Until 2026-10-26 the monthly backtest_matrix (and retrain.py, weekly) wrote the per-league optimiser
output straight into models/best_params_standard.json / models/best_params_side_markets.json, and
production served any league the optimiser itself marked `approved` - a production threshold changed
with no human in the loop. Now:

  1. the optimisers write CANDIDATES only, into output/threshold_proposals/candidates/ (write_candidate);
  2. this module compares each candidate with what production serves today and writes a PROPOSAL:
       output/threshold_proposals/<YYYY-MM>.json   machine-readable, every entry content-hashed
       output/threshold_proposals/<YYYY-MM>.md     the owner's reading copy
       output/threshold_proposals/latest.json      pointer for the dashboard card
  3. the owner approves entries by id + hash in config/threshold_approvals.json, and
     src/threshold_apply.py applies exactly those entries (and nothing else) to the production files.

THIS MODULE NEVER WRITES A PRODUCTION FILE. Every write goes through _safe_write, which refuses any path
outside output/threshold_proposals/ - so a proposal run cannot change what predict serves even by mistake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_DIR = ROOT / "output" / "threshold_proposals"
CANDIDATE_DIR = PROPOSAL_DIR / "candidates"
APPROVALS_PATH = ROOT / "config" / "threshold_approvals.json"

#: production file per market group (relative to ROOT). Mirrors research_lock.THRESHOLD_FILES.
PROD_STANDARD = "models/best_params_standard.json"
PROD_SIDE = "models/best_params_side_markets.json"
SIDE_MARKETS = ("btts", "over15", "over35")
MARKETS = ("ou25",) + SIDE_MARKETS
SCHEMA = "wowza.threshold_proposal/1"
#: the fields that decide what production serves (src/betting.py::_base_tier, pipeline._generate_side_bets)
SERVING_FIELDS = ("approved", "drop", "sniper_th", "marksman_th")
EVIDENCE_FIELDS = ("bets_oos", "roi_oos", "oos_matchdays", "oos_roi_ci95", "bets_insample", "roi_insample",
                   "bets", "roi")
APPROVAL_RULE = ("optimizer rule (src/backtest.py): approved = walk-forward OOS ROI > 0 on >= 20 OOS bets; "
                 "this is a PROPOSAL only - nothing serves until the owner approves the entry")


# ── hashing ───────────────────────────────────────────────────────────────────────────────────────

def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def entry_hash(e: dict) -> str:
    """What an approval signs: the cell, the value production has now and the exact value proposed."""
    return sha({k: e.get(k) for k in ("entry_id", "league", "market", "current", "proposed")})


def proposal_hash(p: dict) -> str:
    return sha({"base_digests": p.get("base_digests"), "entries": [e["entry_hash"] for e in p.get("entries", [])]})


# ── safe IO ───────────────────────────────────────────────────────────────────────────────────────

def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    except Exception:                                                 # noqa: BLE001
        return {}


def _safe_write(path: Path, text: str) -> Path:
    """Refuse every path outside output/threshold_proposals/ - this module must never touch production."""
    path = Path(path).resolve()
    allowed = PROPOSAL_DIR.resolve()
    if allowed != path.parent and allowed not in path.parents:
        raise PermissionError(f"threshold_proposals may only write under {allowed}, refused {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_candidate(name: str, data: dict, directory: Path | None = None) -> Path:
    """Where the optimisers now write (pipeline.py --mode backtest / backtest-side, retrain.py).

    `name` is standard | side_markets | btts | over15 | over35. The file is a CANDIDATE: no production
    code reads this directory."""
    d = (Path(directory) if directory is not None else CANDIDATE_DIR).resolve()
    if (ROOT / "models").resolve() in (d, *d.parents):
        raise PermissionError(f"candidates may never be written into models/ ({d})")
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"best_params_{name}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def load_candidates(directory: Path) -> dict:
    """{market: {league: params}} from every best_params_*.json under `directory` (artifact downloads
    nest one folder per job, so the search is recursive). A combined side file fills markets the
    per-market files did not."""
    directory = Path(directory)
    out: dict = {}
    if not directory.exists():
        return out
    files = sorted(directory.rglob("best_params_*.json"))
    combined = [f for f in files if f.name == "best_params_side_markets.json"]
    for f in files:
        key = f.stem.replace("best_params_", "")
        if key == "standard":
            out["ou25"] = _read(f)
        elif key in SIDE_MARKETS:
            out[key] = _read(f)
    for f in combined:
        for m, v in _read(f).items():
            if m in SIDE_MARKETS and m not in out and isinstance(v, dict):
                out[m] = v
    return {m: v for m, v in out.items() if isinstance(v, dict) and v}


def production_params(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else ROOT
    std = _read(root / PROD_STANDARD)
    side = _read(root / PROD_SIDE)
    out = {"ou25": std}
    for m in SIDE_MARKETS:
        out[m] = side.get(m) or {}
    return out


# ── served-gate resolution (the deployed env, same mechanism as src/research_lock) ───────────────

_RESOLVE = r"""
import json, sys
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
import config
from src import gate_resolver as G
spec = json.loads(sys.stdin.read())
out = {}
for tag in ("current", "proposed"):
    overlay = spec[tag]
    G._standard_params = lambda o=overlay: o["standard"]
    G._side_params = lambda o=overlay: o["side"]
    for lg, mk in spec["cells"]:
        g = G.resolve_effective_tier_gates(lg, mk)
        out[f"{tag}|{lg}|{mk}"] = {"sniper": g.sniper, "marksman": g.marksman, "valuable": g.valuable,
                                    "sniper_source": g.sniper_source, "marksman_source": g.marksman_source}
print("@@JSON@@" + json.dumps(out, default=str))
"""


def resolve_served(cells: list, current: dict, proposed: dict) -> dict:
    """Gates production serves for `cells` with the `current` threshold files and with `proposed` (the
    proposal applied), each {"standard": ..., "side": ...}, through src/gate_resolver under the env
    predict.yml deploys - never this process's env or a local .env."""
    from src import research_lock as L
    env = L.deployed_env()
    keys = L.config_env_keys()
    e = {k: v for k, v in os.environ.items() if k not in keys}
    e.update({k: (v if isinstance(v, str) else v[0]) for k, v in env.items()})
    e["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-c", _RESOLVE], cwd=str(ROOT), env=e, capture_output=True, text=True,
                       encoding="utf-8", timeout=180,
                       input=json.dumps({"cells": cells, "current": current, "proposed": proposed}, default=str))
    if r.returncode != 0 or "@@JSON@@" not in r.stdout:
        raise RuntimeError(f"served-gate resolution failed: {r.stderr[-800:]}")
    return json.loads(r.stdout.split("@@JSON@@", 1)[1])


# ── proposal ──────────────────────────────────────────────────────────────────────────────────────

def _serving(lp) -> dict | None:
    return None if lp is None else {k: lp.get(k) for k in SERVING_FIELDS}


def _change_kind(cur, new) -> str:
    a = bool((cur or {}).get("approved"))
    b = bool((new or {}).get("approved"))
    if new is None:
        return "REMOVE"
    if not a and b:
        return "NEW_APPROVAL"
    if a and not b:
        return "APPROVAL_REVOKED"
    if a and b and (_serving(cur) != _serving(new)):
        return "THRESHOLD_CHANGE"
    if _serving(cur) != _serving(new):
        return "UNAPPROVED_VALUE_CHANGE"
    return "EVIDENCE_ONLY"


def _roi_by_league(evidence_dir: Path | None) -> dict:
    """Side-market per-league backtest ROI (what output/league_roi_config.json used to hold) - evidence."""
    if evidence_dir is None:
        return {}
    import pandas as pd
    roi: dict = {}
    for m in SIDE_MARKETS:
        hits = sorted(Path(evidence_dir).rglob(f"backtest_by_league_{m}.csv"))
        if not hits or hits[0].stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(hits[0])
        except Exception:                                             # noqa: BLE001
            continue
        if "league" not in df.columns or "roi_%" not in df.columns:
            continue
        for _, row in df.iterrows():
            roi.setdefault(str(row["league"]), {})[m] = round(float(row["roi_%"]), 2)
    return roi


def build_proposal(candidates: dict, *, month: str | None = None, created_at: str | None = None,
                   source: dict | None = None, evidence_dir: Path | None = None, root: Path | None = None,
                   resolve: bool = True) -> dict:
    """Compare every candidate cell with production and return the proposal (nothing is written)."""
    from src import research_lock as L
    root = Path(root) if root is not None else ROOT
    now = datetime.now(timezone.utc)
    created_at = created_at or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    month = month or created_at[:7]
    prod = production_params(root)

    entries = []
    for m in MARKETS:
        cand = candidates.get(m)
        if not cand:
            continue
        cur_all = prod.get(m) or {}
        for lg in sorted(set(cand) | set(cur_all)):
            cur, new = cur_all.get(lg), cand.get(lg)
            if cur == new:
                continue
            e = {"entry_id": f"{m}|{lg}", "league": lg, "market": m,
                 "production_file": PROD_STANDARD if m == "ou25" else PROD_SIDE,
                 "current": cur, "proposed": new, "change": _change_kind(cur, new),
                 "evidence": {k: (new or {}).get(k) for k in EVIDENCE_FIELDS if (new or {}).get(k) is not None}}
            e["entry_hash"] = entry_hash(e)
            entries.append(e)

    if resolve and entries:
        base_std, base_side = _read(root / PROD_STANDARD), _read(root / PROD_SIDE)
        overlay_std = json.loads(json.dumps(base_std))
        overlay_side = json.loads(json.dumps(base_side))
        for e in entries:
            if e["market"] == "ou25":
                tgt = overlay_std
            else:
                tgt = overlay_side.setdefault(e["market"], {})
            if e["proposed"] is None:
                tgt.pop(e["league"], None)
            else:
                tgt[e["league"]] = e["proposed"]
        served = resolve_served([[e["league"], e["market"]] for e in entries],
                                {"standard": base_std, "side": base_side},
                                {"standard": overlay_std, "side": overlay_side})
        for e in entries:
            c = served[f"current|{e['league']}|{e['market']}"]
            p = served[f"proposed|{e['league']}|{e['market']}"]
            e["served_now"], e["served_if_approved"] = c, p
            e["serves_change"] = (c["sniper"], c["marksman"]) != (p["sniper"], p["marksman"])

    p = {"schema": SCHEMA, "month": month, "created_at": created_at,
         "status": "AWAITING_OWNER" if any(e.get("serves_change") for e in entries) else "NO_SERVING_CHANGE",
         "source": source or {},
         "approval_rule": APPROVAL_RULE,
         "deployed_env": L.deployed_env(),
         "markets_with_candidates": sorted(candidates),
         "markets_missing": [m for m in MARKETS if m not in candidates],
         "base_digests": L.threshold_file_digests(root),
         "entries": entries,
         "roi_by_league": _roi_by_league(evidence_dir)}
    p["proposal_hash"] = proposal_hash(p)
    p["proposal_id"] = f"TP-{created_at[:10]}-{p['proposal_hash'][:8]}"
    return p


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def to_markdown(p: dict) -> str:
    # Tier words are the owner's display labels (src/tier_labels.py, 2026-10-26); the JSON proposal,
    # its hashes and the approval entries keep the internal codes (sniper / marksman).
    from src.tier_labels import GAP_EXPLAINER, GLOSSARY, TIER_LABEL
    lines = [f"# Threshold proposal {p['proposal_id']}", "",
             f"Created {p['created_at']} for {p['month']} · status **{p['status']}** · "
             f"proposal_hash `{p['proposal_hash']}`", "",
             "Nothing here is live. Production keeps serving the current thresholds until the owner "
             "approves entries in `config/threshold_approvals.json` and `python -m src.threshold_apply` "
             "applies them.", "",
             f"Deployed env: `{p.get('deployed_env')}` · candidates for: {', '.join(p['markets_with_candidates']) or 'none'}"
             + (f" · **missing (job failed or timed out): {', '.join(p['markets_missing'])}**"
                if p.get("markets_missing") else ""), "",
             f"Rule: {p['approval_rule']}", "",
             f"{GAP_EXPLAINER} {GLOSSARY}", ""]
    serving = [e for e in p["entries"] if e.get("serves_change")]
    other = [e for e in p["entries"] if not e.get("serves_change")]
    hdr = (f"| entry | change | served now ({TIER_LABEL['SNIPER']} / {TIER_LABEL['MARKSMAN']}) | if approved "
           "| OOS n | OOS match-days | OOS ROI % "
           "| OOS ROI 95% CI | in-sample ROI % | entry_hash |")
    sep = "|" + "---|" * 10
    for title, rows in (("Entries that change what production serves", serving),
                        ("Entries that change only the stored optimiser output (served gate unchanged)", other)):
        lines += [f"## {title} ({len(rows)})", ""]
        if not rows:
            lines += ["none", ""]
            continue
        lines += [hdr, sep]
        for e in rows:
            ev = e.get("evidence") or {}
            sn, sp = e.get("served_now") or {}, e.get("served_if_approved") or {}
            ci = ev.get("oos_roi_ci95")
            lines.append(
                f"| `{e['entry_id']}` | {e['change']} | {_fmt(sn.get('sniper'))} / {_fmt(sn.get('marksman'))} "
                f"| {_fmt(sp.get('sniper'))} / {_fmt(sp.get('marksman'))} | {_fmt(ev.get('bets_oos'))} "
                f"| {_fmt(ev.get('oos_matchdays'))} | {_fmt(ev.get('roi_oos'))} "
                f"| {('[' + ', '.join(_fmt(x) for x in ci) + ']') if ci else '-'} "
                f"| {_fmt(ev.get('roi_insample', ev.get('roi')))} | `{e['entry_hash'][:16]}` |")
        lines.append("")
    lines += ["## How to approve", "",
              "Add an object to `approvals` in `config/threshold_approvals.json` (ids and hashes copied from "
              f"`output/threshold_proposals/{Path(p.get('_path', p['month'] + '.json')).name}`; never type a "
              "threshold):", "", "```json",
              json.dumps({"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"],
                          "approved_by": "<owner name>", "approved_at": "<UTC timestamp>",
                          "entries": [{"entry_id": e["entry_id"], "entry_hash": e["entry_hash"]}
                                      for e in serving[:2]]}, indent=2),
              "```", "",
              f"then `python -m src.threshold_apply --proposal {p.get('_path') or 'output/threshold_proposals/' + p['month'] + '.json'}` "
              "(add `--dry-run` first).", ""]
    return "\n".join(lines)


def write_proposal(p: dict, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir is not None else PROPOSAL_DIR
    path = out_dir / f"{p['month']}.json"
    old = _read(path)
    if old and old.get("proposal_id") != p["proposal_id"]:
        # a re-run in the same month never overwrites a proposal someone may already be approving
        path = out_dir / f"{p['month']}_{p['proposal_hash'][:8]}.json"
    try:
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    p = dict(p, _path=rel)
    _safe_write(path, json.dumps({k: v for k, v in p.items() if k != "_path"}, indent=1, ensure_ascii=False,
                                 default=str) + "\n")
    _safe_write(path.with_suffix(".md"), to_markdown(p))
    serving = [e for e in p["entries"] if e.get("serves_change")]
    _safe_write(out_dir / "latest.json", json.dumps({
        "proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"], "path": p["_path"],
        "summary_md": str(Path(p["_path"]).with_suffix(".md")).replace("\\", "/"),
        "created_at": p["created_at"], "month": p["month"], "status": p["status"],
        "n_entries": len(p["entries"]), "n_serving_changes": len(serving),
        "serving_changes": [{"entry_id": e["entry_id"], "change": e["change"],
                             "served_now": e.get("served_now"), "served_if_approved": e.get("served_if_approved"),
                             "bets_oos": (e.get("evidence") or {}).get("bets_oos"),
                             "oos_matchdays": (e.get("evidence") or {}).get("oos_matchdays"),
                             "roi_oos": (e.get("evidence") or {}).get("roi_oos"),
                             "oos_roi_ci95": (e.get("evidence") or {}).get("oos_roi_ci95")} for e in serving],
        "markets_missing": p.get("markets_missing")}, indent=1, ensure_ascii=False, default=str) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("propose")
    pr.add_argument("--candidates", default=str(CANDIDATE_DIR))
    pr.add_argument("--evidence-dir", default=str(ROOT / "output"))
    pr.add_argument("--month", default=None)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(_read(PROPOSAL_DIR / "latest.json"), indent=1, ensure_ascii=False))
        return 0
    cands = load_candidates(Path(a.candidates))
    if not cands:
        print(f"no candidates under {a.candidates} - every backtest job failed; no proposal written")
        return 1
    src = {"workflow": os.getenv("GITHUB_WORKFLOW"), "run_id": os.getenv("GITHUB_RUN_ID"),
           "sha": os.getenv("GITHUB_SHA"), "candidates_dir": a.candidates}
    p = build_proposal(cands, month=a.month, source=src, evidence_dir=Path(a.evidence_dir))
    path = write_proposal(p)
    n = sum(1 for e in p["entries"] if e.get("serves_change"))
    print(f"{p['proposal_id']}: {len(p['entries'])} entries, {n} would change a served gate -> {path} "
          f"(status {p['status']}; markets missing: {p['markets_missing'] or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
