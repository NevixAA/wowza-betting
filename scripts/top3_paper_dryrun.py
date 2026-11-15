"""DRY RUN of the TOP-3 paper tracks (src/research/top3_paper.py) on the latest gameweek pool - writes ONLY to --out.

    python scripts/top3_paper_dryrun.py --out DIR --stats DIR/live_stats_2026-11-20.json

The go-live state does not exist yet, so the harness SIMULATES it, in memory only:
  * bars: the approved TP-2026-11-10-d999073a cells as if src/threshold_apply had applied them (U3b);
  * statistics: a DRY-RUN freeze record (scripts/top3_paper_freeze.py --dry-run);
  * the protocol start is moved to the pool's first week (the real start is 2026-11-21).
Pass STRICT: the owner's guards exactly as in production -> today's pool comes from pre-clean models, so every
candidate must be refused MODEL_NOT_CLEAN (the guard working).
Pass REHEARSAL: the model guard is told the pool's own model hashes are "clean" (in this harness only, labelled)
so the ranking, the ledger, the settlement and the summary run end to end. Nothing is committed or published.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import top3_league as TL  # noqa: E402
from src.research import league_weekly as LW  # noqa: E402
from src.research import top3_paper as P  # noqa: E402
from src.research import top3_ranking as R  # noqa: E402


def simulated_ctx(cells: dict) -> tuple[dict, dict]:
    std, side, applied = {}, {}, {}
    for (mk, lg, sd), (cell, pid) in cells.items():
        if mk == "ou25":
            std.setdefault(lg, {"bars_opt": {}})["bars_opt"][sd] = cell
        else:
            side.setdefault(mk, {}).setdefault(lg, {"bars_opt": {}})["bars_opt"][sd] = cell
        applied[f"{mk}|{lg}"] = pid
    return {"standard": std, "side": side, "applied": set(applied)}, applied


def _set(picks: pd.DataFrame, track: str) -> set:
    g = picks[picks["track"] == track]
    return set(zip(g["fixture_id"].astype(str), g["market"], g["side"]))


def one_pass(name, cands, ctx, guards, stats, cfg, sha, od: Path, now, v91: pd.DataFrame) -> dict:
    od.mkdir(parents=True, exist_ok=True)
    from src.direction_guard import committed
    # each league-week is decided at its own FREEZE (+5 min, one predict.yml run later), as live
    com, pk, wk = committed(), [], []
    for w, lock in sorted(LW.week_locks(LW._plain(cands)).items(), key=lambda kv: kv[1]):
        at = lock - pd.Timedelta(minutes=120) + pd.Timedelta(minutes=5)
        p_, w_ = P.freeze_due(cands[cands["league_week_id"] == w], set(), ctx, at, 120.0, com, stats, guards, cfg, sha)
        pk.append(p_)
        wk.append(w_)
    picks = pd.concat([x for x in pk if len(x)], ignore_index=True) if any(len(x) for x in pk) else pk[0]
    weeks = pd.concat([x for x in wk if len(x)], ignore_index=True) if any(len(x) for x in wk) else wk[0]
    P.picks_store(od).append(picks)
    P.weeks_store(od).append(weeks)
    ex = Counter()
    for e in weeks["excluded"] if len(weeks) else []:
        ex.update(json.loads(e))
    settled, n = P.settle_picks(P.picks_store(od).read(), v91, now + pd.Timedelta(days=30))
    summ = P.summary(settled, now, cfg, f"DRY_RUN {name}")
    (od / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    return {"pass": name, "league_weeks": int(weeks["league_week_id"].nunique()) if len(weeks) else 0,
            "track_weeks": int(len(weeks)), "picks": int(len(picks)),
            "picks_by_track": picks.groupby("track").size().to_dict() if len(picks) else {},
            "pool_sizes": int(weeks["n_pool"].sum()) if len(weeks) else 0, "excluded_total": dict(ex),
            "settled": n, "no_stat_picks": int((picks["no_stat"] == 1).sum()) if len(picks) else 0,
            "summary_tracks": summ["tracks"],
            "tracks_recorded": sorted(set(picks["track"])) if len(picks) else []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--candidates", required=True, help="candidate store (live copy from origin/main)")
    ap.add_argument("--v91", required=True, help="results source v91_shadow.csv (live copy from origin/main)")
    ap.add_argument("--until", required=True, help="end of the gameweek window (UTC); the pool = the 7 days before")
    a = ap.parse_args()
    cfg = P.load_config()
    rec = json.loads(Path(a.stats).read_text(encoding="utf-8"))
    stats = R.stats_from_freeze(rec)
    sha = P.sha256_lf(Path(a.stats))
    cands = LW._read(Path(a.candidates))
    until = pd.Timestamp(a.until)
    ko = pd.to_datetime(cands["kickoff_utc"], utc=True)
    cands = cands[(ko > until - pd.Timedelta(days=7)) & (ko <= until)]
    v91 = LW._read(Path(a.v91))
    cells, why = P.approved_cells(cfg)
    assert not why, why
    ctx, applied = simulated_ctx(cells)
    cfg_dry = cfg | {"start_utc": "2000-01-01T00:00:00Z"}
    now = pd.to_datetime(cands["kickoff_utc"], utc=True).max() + pd.Timedelta(hours=1)
    strict, _ = P.guards_from(cfg)
    strict["applied"] = applied
    rehearsal = dict(strict)
    rehearsal["clean"] = {}
    for _, r in cands.iterrows():
        fam = str(r.get("engine_market") or "")
        slot = LW.CHAMP_SLOT.get(fam) or ("standard" if str(r.get("model_type")) == "standard" else "newformat")
        short = str(r.get("prediction_engine_version")).rsplit("_", 1)[-1]
        rehearsal["clean"].setdefault(slot, set()).add(short + "0" * 56)
    rehearsal["champions"] = {}
    out = {"candidates": int(len(cands)), "league_weeks_in_pool": int(cands["league_week_id"].nunique()),
           "kickoff_window": [str(pd.to_datetime(cands["kickoff_utc"]).min()), str(pd.to_datetime(cands["kickoff_utc"]).max())],
           "engine_versions": cands["prediction_engine_version"].value_counts().to_dict(),
           "stats_record": {"kind": rec.get("kind"), "frame_hash": rec.get("frame_hash"), "sha256": sha,
                            "cells": len(stats)},
           "passes": [one_pass("STRICT", cands, ctx, strict, stats, cfg_dry, sha, Path(a.out) / "strict", now, v91),
                      one_pass("REHEARSAL_model_guard_relaxed_in_harness_only", cands, ctx, rehearsal, stats, cfg_dry,
                               sha, Path(a.out) / "rehearsal", now, v91)]}
    (Path(a.out) / "dryrun_report.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
