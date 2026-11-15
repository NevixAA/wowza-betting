"""Rebuild config/bankroll_policy.json -> cell_staking.cells from a bars_opt serving proposal.

    python scripts/cell_staking_from_proposal.py --proposal output/threshold_proposals/2026-11_<id>.json [--dry-run]
        [--set-fraction "<track>|<league>=<fraction>" ... --fraction-note "<owner words>"] [--record-approval]

--set-fraction (repeatable) sets a STAKED cell's base_fraction to the owner's value and records his words on the
cell as base_fraction_source (--fraction-note, required with it). It never creates a stake: a key that is not a
staked cell of the rebuilt policy is refused. --record-approval writes cell_staking.approved = {proposal_id,
proposal_hash} of --proposal (refused when the hash does not match the proposal's content); with --dry-run
nothing is written. Without these options the behaviour is unchanged.

Owner rule 2026-11-09 "Paper + PROMISING small": only the PROMISING cells of the proposal stake, at the policy's
existing ACTIVE_LOW level (base_fraction 0.005), flat at every tier (their evidence is flat stakes at or above
the VALUABLE bar). Every other cell is PAPER (no entry). The PROTECTED real-money track (Argentina Primera
Division BTTS, new_format) is carried over VERBATIM from the current policy, never rebuilt. A cell's max_odds is
its own odds_max; a cell without an odds range carries no max_odds (the reader's HARD ceiling applies).
Nothing else in the policy changes. Staking follows the proposal's CURRENT verdicts: a cell that is no longer
PROMISING loses its entry, a new PROMISING cell gains one - EXCEPT on a track the owner holds PAPER by an explicit
rule (tracks[...].state == PAPER with a paper_reason, e.g. owner 2026-11-02 "new-format O/U Unders are paper-only
until the clean retrain passes per-league validation"): a cell never lifts an owner hold, the same way it never
lifts a BLOCKED track. Such cells are listed as held_paper for the owner to decide.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "bankroll_policy.json"
#: bars_opt market-side -> the stake-plan track market (src/bankroll_policy.track_key)
TRACK_MARKET = {"over25_over": "OU25_OVER", "over25_under": "OU25_UNDER", "btts_yes": "BTTS",
                "over15_over": "OVER15", "over35_over": "OVER35"}
PROTECTED = ("new_format|BTTS|Argentina Primera Division",)
ACTIVE_LOW_BASE = 0.005           # the policy's existing ACTIVE_LOW level (owner 2026-11-09 "small")


APPROVALS = ROOT / "config" / "threshold_approvals.json"
#: owner 2026-11-11 "Strict, keep Argentina": why the PROTECTED entry keeps staking although its entry's
#: selection evidence is mostly backfill - recorded on the cell
ARGENTINA_BASIS = ("owner 2026-11-11 \"Strict, keep Argentina\": a LEAGUE_WIDE rule (every match up to odds 2.50, "
                   "no bar chosen from backfill); its PROMISING verdict rests on its own live-captured FINAL rows "
                   "(86 bets, snapshot before kickoff); ACTIVE 1.25% flat")


def withdrawn_entries(proposal: dict, path=APPROVALS) -> set:
    """Entry ids the owner withdrew from this proposal's approval (config/threshold_approvals.json)."""
    try:
        ap = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    out = set()
    for a in ap.get("approvals") or []:
        if a.get("proposal_id") == proposal.get("proposal_id"):
            out |= {w["entry_id"] for w in a.get("withdrawals") or []}
    return out


def promising_cells(proposal: dict, withdrawn: set | None = None) -> list[dict]:
    """PROMISING cells of the proposal, minus the entries the owner withdrew (they stake nothing, PAPER)."""
    withdrawn = withdrawn or set()
    out = []
    for e in proposal["entries"]:
        if e.get("entry_id") in withdrawn:
            continue
        for cell in ((e.get("proposed") or {}).get("bars_opt") or {}).values():
            if cell.get("verdict") == "PROMISING":
                out.append(cell)
    return out


def held_by_owner(policy: dict, track_key: str) -> str | None:
    """The owner's paper hold on a track (state PAPER with a paper_reason), else None."""
    t = (policy.get("tracks") or {}).get(track_key) or {}
    return t.get("paper_reason") if t.get("state") == "PAPER" and t.get("paper_reason") else None


def strip_tiers(entry: dict) -> dict:
    """A cell entry without tier multipliers (owner 2026-11-11: flat per cell, tiers retired)."""
    e = {k: v for k, v in entry.items() if k not in ("tier_multipliers", "tier_multipliers_note")}
    e.setdefault("stake_note", "flat per cell (owner 2026-11-11 'Clean rebuild'): the base fraction on every tip "
                               "that clears the cell's one approved bar; the old 1.1 / 0.9 / 1.0 tier "
                               "multipliers are retired")
    return e


def build_cells(proposal: dict, current: dict, policy: dict | None = None, held: list | None = None,
                withdrawn: set | None = None) -> dict:
    cells = {k: strip_tiers(current[k]) for k in PROTECTED if k in current}
    for k in cells:
        cells[k]["staking_basis"] = ARGENTINA_BASIS
    for c in promising_cells(proposal, withdrawn):
        tkey = f"{c['track']}|{TRACK_MARKET[c['mside']]}"
        key = f"{tkey}|{c['league']}"
        if key in cells:              # the protected entry is never overwritten
            continue
        if policy is not None and held_by_owner(policy, tkey):
            if held is not None:
                held.append(key)
            continue
        entry = {"state": "ACTIVE_LOW", "base_fraction": ACTIVE_LOW_BASE,
                 "source": (f"PROMISING bars_opt cell (proposal {proposal['proposal_id']}, study FINAL "
                            f"2026-09-01..11-08); owner 2026-11-09 'Paper + PROMISING small': the policy's "
                            f"ACTIVE_LOW level")}
        if c.get("odds_max") is not None:
            entry["max_odds"] = float(c["odds_max"])
            entry["max_odds_source"] = "the cell's own odds_max"
        else:
            entry["max_odds_source"] = "the cell has no odds range: the reader's HARD max_odds ceiling applies"
        cells[key] = strip_tiers(entry)
    return cells


def parse_fraction(spec: str) -> tuple[str, float]:
    """'<track>|<league>=<fraction>' -> (cell key, fraction); the fraction must be finite and in (0, 1)."""
    key, sep, val = str(spec).rpartition("=")
    if not sep or not key.strip():
        raise ValueError(f"--set-fraction {spec!r}: expected '<track>|<league>=<fraction>'")
    try:
        f = float(val)
    except ValueError:
        raise ValueError(f"--set-fraction {spec!r}: {val!r} is not a number") from None
    if not (0.0 < f < 1.0):
        raise ValueError(f"--set-fraction {spec!r}: the fraction must be > 0 and < 1")
    return key.strip(), f


def set_fractions(cells: dict, specs: list, note: str | None) -> list[dict]:
    """Apply the owner's explicit base fractions to STAKED cells (in place); returns what changed. Never
    creates an entry (a key not staked is refused) and never runs without the owner's words."""
    if not specs:
        return []
    if not (note and str(note).strip()):
        raise ValueError("--set-fraction needs --fraction-note with the owner's words")
    changes = []
    for spec in specs:
        key, f = parse_fraction(spec)
        if key not in cells:
            raise ValueError(f"--set-fraction: {key!r} is not a staked cell of this policy (staked: {sorted(cells)})")
        changes.append({"cell": key, "from": cells[key].get("base_fraction"), "to": f})
        cells[key]["base_fraction"] = f
        cells[key]["base_fraction_source"] = str(note).strip()
    return changes


def approval_record(proposal: dict) -> dict:
    """{proposal_id, proposal_hash} of the proposal, refused when the hash does not match its content."""
    sys.path.insert(0, str(ROOT))
    from src import threshold_proposals as TP
    if TP.proposal_hash(proposal) != proposal.get("proposal_hash"):
        raise ValueError("proposal_hash does not match the proposal content - the file was edited")
    return {"proposal_id": proposal["proposal_id"], "proposal_hash": proposal["proposal_hash"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-fraction", action="append", default=[], metavar="TRACK|LEAGUE=FRACTION",
                    help="set a staked cell's base_fraction (repeatable); needs --fraction-note")
    ap.add_argument("--fraction-note", default=None, help="the owner's words, recorded as base_fraction_source")
    ap.add_argument("--record-approval", action="store_true",
                    help="write cell_staking.approved = {proposal_id, proposal_hash} of --proposal")
    a = ap.parse_args(argv)
    proposal = json.loads(Path(a.proposal).read_text(encoding="utf-8"))
    raw = POLICY.read_text(encoding="utf-8")
    pol = json.loads(raw)
    cur = pol["cell_staking"]["cells"]
    held: list = []
    new = build_cells(proposal, cur, pol, held, withdrawn_entries(proposal))
    try:
        fractions = set_fractions(new, a.set_fraction, a.fraction_note)
        approval = approval_record(proposal) if a.record_approval else None
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    added, removed = sorted(set(new) - set(cur)), sorted(set(cur) - set(new))
    out = {"proposal": proposal["proposal_id"], "staked": sorted(new), "added": added,
           "removed": removed, "held_paper_by_owner_rule": sorted(held)}
    if fractions:
        out["set_fraction"] = fractions
    if approval:
        out["approval"] = approval
    print(json.dumps(out, indent=1))
    if not a.dry_run:
        pol["cell_staking"]["cells"] = new
        if approval:
            pol["cell_staking"]["approved"] = approval
        POLICY.write_text(json.dumps(pol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
