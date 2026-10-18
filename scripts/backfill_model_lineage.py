"""One-off: reconstruct model lineage from git history and retrain_log.json (2026-10-18).

    python scripts/backfill_model_lineage.py            # writes registry + lineage files
    python scripts/backfill_model_lineage.py --dry-run  # prints what it would write

WHAT IS RECONSTRUCTED, AND HOW FAR IT CAN BE TRUSTED
  * Every committed version of every serving .pkl is a real artifact. Its identity is the
    SHA-256 of the committed bytes and its deploy time is the commit time. That is evidence, not
    inference, so each gets a manifest with `reconstructed: true`.
  * `trained_at` is UNKNOWN for these — the retrain ran minutes to hours before the commit. The
    version id uses the COMMIT time and says so in the manifest.
  * Rejected challengers before 2026-10-18 were never kept; their metrics survive in
    retrain_log.json, their bytes do not. Those rows carry challenger_version=UNKNOWN_LEGACY.
    Nothing here invents a version for an artifact that cannot be produced.
  * Ledger rows are NOT rewritten. Pro attributes legacy bets to eras by timestamp in its own,
    labelled output (attribution_method=RECONSTRUCTED_BY_TIME).

Idempotent: re-running writes the same manifests (immutable — an identical rewrite is a no-op)
and rebuilds the two legacy CSVs only when they do not yet contain non-legacy rows.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import model_registry as reg  # noqa: E402
from src.champion_gate import MATERIAL_LL  # noqa: E402

GIT = os.getenv("WOWZA_GIT", "git")


def git(*args: str, binary: bool = False):
    out = subprocess.run([GIT, *args], cwd=ROOT, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode("utf-8", "replace")


def commits_for(path: Path) -> list[tuple[str, datetime]]:
    rel = path.relative_to(ROOT).as_posix()
    rows = []
    for line in git("log", "--format=%H %cI", "--", rel).splitlines():
        h, t = line.split(" ", 1)
        rows.append((h, datetime.fromisoformat(t).astimezone(timezone.utc)))
    return list(reversed(rows))                        # oldest first


def replay(delta, ci) -> str:
    """What the new gate would have said from the numbers retrain_log kept. CI-only: the later-half,
    calibration and league checks were not recorded, so this can never say more than it knows."""
    if delta is None:
        return "UNKNOWN"
    if ci is None:
        return "KEEP_CHAMPION" if delta >= 0 else "UNKNOWN_NO_CI"
    lo, hi = ci
    if delta >= 0:
        return "KEEP_CHAMPION"
    if delta <= -MATERIAL_LL and hi < 0:
        return "PROMOTE_CANDIDATE"     # would still need the unrecorded checks
    if lo <= -MATERIAL_LL:
        return "SHADOW_MORE"
    return "KEEP_CHAMPION"


def main(dry: bool) -> None:
    log_blob = json.loads((ROOT / "output" / "retrain_log.json").read_text(encoding="utf-8"))
    runs = log_blob.get("runs", {})

    manifests, history = [], []
    pointer = {}
    for mt, path in reg.MODEL_FILES.items():
        cs = commits_for(path)
        rel = path.relative_to(ROOT).as_posix()
        prior = ""
        for i, (h, t) in enumerate(cs):
            raw = git("show", f"{h}:{rel}", binary=True)
            sha = reg.sha256_bytes(raw)
            v = reg.make_version(mt, t, sha)
            day = t.strftime("%Y-%m-%d")
            ent = (runs.get(day) or {}).get(mt) or {}
            end = cs[i + 1][1] if i + 1 < len(cs) else None
            last = i == len(cs) - 1
            manifests.append({
                "model_type": mt, "model_version": v, "retrain_batch_id": f"rb_{day}_legacy",
                "trained_at": None, "deployed_at": reg.iso(t),
                "training_start": None, "training_end": None, "training_cutoff": None,
                "training_rows": ent.get("rows"), "feature_schema_version": None,
                "code_commit_sha": h[:12], "git_commit": h,
                "artifact_sha256": sha, "parent_champion_version": prior,
                "validation_metrics": {"logloss_old": ent.get("logloss_old"),
                                       "logloss_new": ent.get("logloss_new"),
                                       "basis": ent.get("comparison_basis"),
                                       "v91_gate": ent.get("v91_gate")} if ent else None,
                "promotion_status": "LEGACY_PROMOTED",
                "model_role": "CHAMPION" if last else "ARCHIVED",
                "reconstructed": True,
                "note": "identity = SHA-256 of the committed bytes; version timestamp is the "
                        "COMMIT (deploy) time, training time unknown",
            })
            history.append({
                "model_type": mt, "champion_version": v, "start_at": reg.iso(t),
                "end_at": reg.iso(end) if end else "", "promotion_retrain_batch": f"rb_{day}_legacy",
                "prior_champion": prior, "holdout_n": "", "holdout_logloss": ent.get("logloss_new", ""),
                "holdout_brier": "", "holdout_ece": "", "fresh_oos_logloss": "",
                "promotion_source": "LEGACY_GIT", "promotion_reason": (ent.get("why") or "")[:300],
                "artifact_sha256": sha, "git_commit": h[:12], "reconstructed": True,
            })
            prior = v
            if last:
                on_disk = reg.sha256_file(path)
                if on_disk != sha:
                    print(f"WARNING {mt}: working-tree file differs from HEAD — pointer uses HEAD")
                pointer[mt] = {"model_version": v, "champion_since": reg.iso(t),
                               "retrain_batch_id": f"rb_{day}_legacy", "training_cutoff": "",
                               "artifact_sha256": sha, "promotion_source": "LEGACY_GIT",
                               "promotion_reason": "reconstructed from git history 2026-10-18"}
                pointer[mt]["_raw"] = raw

    # ── retrain decisions (all of retrain_log) ─────────────────────────────────────────────
    by_sha_day = {(m["model_type"], m["deployed_at"][:10]): m["model_version"] for m in manifests}
    decisions, replays = [], []
    for day in sorted(runs):
        for mt, e in runs[day].items():
            g = e.get("v91_gate") or {}
            promoted = bool(e.get("promoted"))
            old, new = e.get("logloss_old"), e.get("logloss_new")
            delta = None if old is None or new is None else round(new - old, 6)
            ci = g.get("ci_log_loss")
            gd = g.get("delta_log_loss")
            chal = by_sha_day.get((mt, day), "") if promoted else ""
            decisions.append({
                "retrain_batch_id": f"rb_{day}_legacy", "decided_at": e.get("at", day),
                "model_type": mt, "challenger_version": chal or reg.UNKNOWN_LEGACY,
                "champion_at_training_time": "", "verdict":
                    "LEGACY_PROMOTED" if promoted else "LEGACY_BLOCKED",
                "promoted": promoted, "champion_after": "", "training_rows": e.get("rows"),
                "holdout_n": g.get("n", ""), "ll_champion": old, "ll_challenger": new,
                "ll_delta": delta, "ll_ci_lo": ci[0] if ci else "", "ll_ci_hi": ci[1] if ci else "",
                "comparison_basis": e.get("comparison_basis") or "stored_metrics",
                "reason": (e.get("why") or "")[:400], "legacy_rule_would_promote": promoted,
                "source": "LEGACY_RETRAIN_LOG",
            })
            # Replay on the served-ensemble delta when the v9.1 gate recorded one (it scores the
            # same thing the new gate scores); else the base-model mean, flagged.
            rd = gd if gd is not None else delta
            replays.append({"date": day, "model_type": mt, "legacy_promoted": promoted,
                            "delta_basis": "served_ensemble" if gd is not None else "mean_base_models",
                            "delta": rd, "ci_lo": ci[0] if ci else None,
                            "ci_hi": ci[1] if ci else None, "new_gate_replay": replay(rd, ci),
                            "artifact_kept": bool(chal)})

    print(f"manifests {len(manifests)}  history {len(history)}  decisions {len(decisions)}")
    rp = pd.DataFrame(replays)
    print(rp.groupby(["legacy_promoted", "new_gate_replay"]).size().to_string())
    if dry:
        return

    for m in manifests:
        reg.write_manifest(m)
    reg.LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
    for f, rows, cols in ((reg.CHAMPION_HISTORY, history, reg.CHAMPION_HISTORY_COLS),
                          (reg.RETRAIN_BATCHES, decisions, reg.RETRAIN_BATCH_COLS)):
        if f.exists():
            cur = pd.read_csv(f, dtype=str)
            if (cur.get("source", cur.get("promotion_source", pd.Series(dtype=str)))
                    .astype(str).str.startswith("LEGACY").all()):
                f.unlink()          # only legacy rows so far — safe to rebuild
            else:
                print(f"{f.name} already holds live rows — legacy rows NOT rewritten")
                continue
        for r in rows:
            reg._append_csv(f, cols, r)
    rp.to_csv(reg.LINEAGE_DIR / "legacy_gate_replay.csv", index=False)

    for mt, p in pointer.items():
        raw = p.pop("_raw")
        reg.store_champion_artifact(mt, p["model_version"], raw)
    if not reg.POINTER_FILE.exists():
        reg._atomic_write_json(reg.POINTER_FILE, pointer)
    hw = {mt: {"champion_version": p["model_version"],
               "historical": {"source": "LEGACY_GIT", "note": "no same-holdout validation was "
                              "recorded at promotion; the first new-gate retrain sets the "
                              "current-regime benchmark"},
               "current_regime": None} for mt, p in pointer.items()}
    if not reg.HIGH_WATER.exists():
        reg._atomic_write_json(reg.HIGH_WATER, hw)
    print("written:", reg.POINTER_FILE, reg.CHAMPION_HISTORY, reg.RETRAIN_BATCHES, reg.HIGH_WATER)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
