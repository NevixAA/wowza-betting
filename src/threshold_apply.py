"""APPLY owner-approved per-league threshold proposals - the ONLY writer of the production threshold files.

    python -m src.threshold_apply --proposal output/threshold_proposals/2026-11.json --dry-run
    python -m src.threshold_apply --proposal output/threshold_proposals/2026-11.json

Owner rule 2026-10-26: the monthly backtest PROPOSES (src/threshold_proposals.py), the owner approves,
this script applies. Production reads only models/best_params_standard.json and
models/best_params_side_markets.json, and only this script writes them.

WHAT AN APPROVAL IS. An object in config/threshold_approvals.json:

    {"proposal_id": "TP-2026-11-01-1a2b3c4d", "proposal_hash": "<64 hex>",
     "approved_by": "Nevo (owner)", "approved_at": "2026-11-02T08:00:00Z",
     "entries": [{"entry_id": "ou25|Serie B", "entry_hash": "<64 hex>"}]}

Entries are named by id + content hash only. A threshold value typed into the approval is refused - the
value applied is always the one inside the hashed proposal entry, so what the owner read is what serves.

WHAT APPLY DOES, all or nothing:
  1. re-hashes the proposal and every entry (a hand-edited proposal is refused);
  2. finds the approval whose proposal_id AND proposal_hash match, and requires every approved entry to
     match a proposal entry by id AND entry_hash;
  3. requires production's current value for each entry to equal the entry's `current` (a stale proposal,
     built on thresholds that have since changed, is refused; an entry already applied is skipped);
  4. requires the research lock to be clean BEFORE (so apply can never bless an unrelated change);
  5. writes only the approved league cells, records the files' digests in config/threshold_applied.json,
     re-approves config/research_lock_snapshot.json as `approved_by`, and re-checks the lock.
A failure after the first write restores every file it touched.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src import research_lock as L
from src import threshold_proposals as TP

APPROVALS_PATH = TP.APPROVALS_PATH
_ENTRY_KEYS = {"entry_id", "entry_hash", "note"}


class ApplyRefused(RuntimeError):
    pass


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def verify_proposal(p: dict) -> None:
    if p.get("schema") != TP.SCHEMA:
        raise ApplyRefused(f"not a threshold proposal (schema {p.get('schema')!r})")
    for e in p.get("entries", []):
        if TP.entry_hash(e) != e.get("entry_hash"):
            raise ApplyRefused(f"proposal entry {e.get('entry_id')} does not match its own hash - the proposal "
                               f"file was edited after it was generated")
    if TP.proposal_hash(p) != p.get("proposal_hash"):
        raise ApplyRefused("proposal_hash does not match the proposal content - the file was edited")


def find_approval(p: dict, approvals: dict) -> dict:
    hits = [a for a in approvals.get("approvals", []) if a.get("proposal_id") == p.get("proposal_id")]
    if not hits:
        raise ApplyRefused(f"no owner approval for {p.get('proposal_id')} in config/threshold_approvals.json")
    if len(hits) > 1:
        raise ApplyRefused(f"{len(hits)} approvals name {p['proposal_id']} - keep exactly one")
    a = hits[0]
    if a.get("proposal_hash") != p.get("proposal_hash"):
        raise ApplyRefused(f"approval for {p['proposal_id']} names proposal_hash "
                           f"{str(a.get('proposal_hash'))[:12]}, the proposal is {p['proposal_hash'][:12]}")
    if not str(a.get("approved_by") or "").strip() or not str(a.get("approved_at") or "").strip():
        raise ApplyRefused("an approval must carry approved_by and approved_at")
    if not a.get("entries"):
        raise ApplyRefused("the approval lists no entries")
    by_id = {e["entry_id"]: e for e in p.get("entries", [])}
    for ae in a["entries"]:
        extra = set(ae) - _ENTRY_KEYS
        if extra:
            raise ApplyRefused(f"approval entry {ae.get('entry_id')} carries {sorted(extra)} - approvals name "
                               f"entries by entry_id + entry_hash only; thresholds are never typed")
        pe = by_id.get(ae.get("entry_id"))
        if pe is None:
            raise ApplyRefused(f"approved entry {ae.get('entry_id')} is not in proposal {p['proposal_id']}")
        if pe["entry_hash"] != ae.get("entry_hash"):
            raise ApplyRefused(f"approved entry {ae['entry_id']} hash {str(ae.get('entry_hash'))[:12]} does not "
                               f"match the proposal's {pe['entry_hash'][:12]}")
    return a


def plan(p: dict, approval: dict, root: Path) -> tuple[dict, dict, list, list]:
    """New (standard, side) file contents, the entries to apply and the entries already applied."""
    std = _read(root / TP.PROD_STANDARD)
    side = _read(root / TP.PROD_SIDE)
    by_id = {e["entry_id"]: e for e in p["entries"]}
    todo, done = [], []
    for ae in approval["entries"]:
        e = by_id[ae["entry_id"]]
        tgt = std if e["market"] == "ou25" else side.setdefault(e["market"], {})
        now = tgt.get(e["league"])
        if now == e["proposed"]:
            done.append(e["entry_id"])
            continue
        if now != e["current"]:
            raise ApplyRefused(f"{e['entry_id']}: production changed since the proposal was built (stale) - "
                               f"generate a new proposal")
        if e["proposed"] is None:
            tgt.pop(e["league"], None)
        else:
            tgt[e["league"]] = e["proposed"]
        todo.append(e["entry_id"])
    return std, side, todo, done


def apply(proposal_path: Path, *, approvals_path: Path | None = None, dry_run: bool = False,
          lock: bool = True) -> dict:
    root = Path(L.THRESHOLD_ROOT)
    p = _read(Path(proposal_path))
    if not p:
        raise ApplyRefused(f"proposal {proposal_path} missing or empty")
    verify_proposal(p)
    approval = find_approval(p, _read(Path(approvals_path) if approvals_path else APPROVALS_PATH))
    std, side, todo, done = plan(p, approval, root)
    result = {"proposal_id": p["proposal_id"], "approved_by": approval["approved_by"], "applied": todo,
              "already_applied": done, "dry_run": dry_run}
    if dry_run or not todo:
        return result

    if lock:
        pre = L.check()
        if pre["protected_tracks_modified"]:
            raise ApplyRefused("the research lock is not clean BEFORE apply, so apply would bless an unrelated "
                               "change: " + "; ".join(pre["changes"][:5]))

    ledger_path = Path(L.THRESHOLD_LEDGER_PATH)
    touched = {root / TP.PROD_STANDARD: std, root / TP.PROD_SIDE: side}
    backup = {f: (f.read_bytes() if f.exists() else None) for f in [*touched, ledger_path, Path(L.SNAPSHOT_PATH)]}
    try:
        for f, data in touched.items():
            f.write_text(json.dumps(data, indent=2), encoding="utf-8")
        led = _read(ledger_path)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        led.setdefault("about", "Digests of the per-league threshold files as last written by src/threshold_apply.py. "
                                "src/research_lock.py turns RED when the live files differ. Do not hand-edit.")
        led["files"] = L.threshold_file_digests(root)
        led["last_applied"] = {"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"],
                               "approved_by": approval["approved_by"], "approved_at": approval["approved_at"],
                               "applied_at": now, "entries": todo}
        led.setdefault("history", []).append(led["last_applied"])
        ledger_path.write_text(json.dumps(led, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        if lock:
            L.approve(approval["approved_by"],
                      note=f"threshold_apply {p['proposal_id']} ({len(todo)} entries: {', '.join(todo)}); "
                           f"approved_at {approval['approved_at']}", path=Path(L.SNAPSHOT_PATH))
            post = L.check()
            if post["protected_tracks_modified"]:
                raise ApplyRefused("lock still reports changes after apply: " + "; ".join(post["changes"][:5]))
    except BaseException:
        for f, b in backup.items():
            if b is None:
                f.unlink(missing_ok=True)
            else:
                f.write_bytes(b)
        raise
    result["files"] = L.threshold_file_digests(root)
    return result


def init_ledger(by: str, note: str = "") -> dict:
    """One-time baseline: record the digests of the threshold files production serves TODAY. Refuses when
    a ledger already exists - after that, only apply() moves it."""
    path = Path(L.THRESHOLD_LEDGER_PATH)
    if path.exists():
        raise ApplyRefused(f"{path} already exists - thresholds change only through an approved apply")
    if not by.strip():
        raise ApplyRefused("--by is required")
    led = {"about": "Digests of the per-league threshold files as last written by src/threshold_apply.py. "
                    "src/research_lock.py turns RED when the live files differ. Do not hand-edit.",
           "files": L.threshold_file_digests(),
           "baseline": {"recorded_by": by.strip(), "note": note,
                        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
           "history": []}
    path.write_text(json.dumps(led, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return led


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal")
    ap.add_argument("--approvals", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--init-ledger", action="store_true")
    ap.add_argument("--by", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    try:
        if a.init_ledger:
            led = init_ledger(a.by, a.note)
            print(f"ledger written: {led['files']}")
            return 0
        if not a.proposal:
            ap.error("--proposal is required")
        r = apply(Path(a.proposal), approvals_path=Path(a.approvals) if a.approvals else None, dry_run=a.dry_run)
    except ApplyRefused as e:
        print(f"REFUSED: {e}")
        return 2
    verb = "would apply" if r["dry_run"] else "applied"
    print(f"{r['proposal_id']} (approved by {r['approved_by']}): {verb} {len(r['applied'])} entries "
          f"{r['applied']}; already applied {r['already_applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
