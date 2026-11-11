"""FREEZE the per-cell ranking statistics of the TOP-3 protocol (config/top3_ranking_paper_protocol_v4.json) ONCE.

    python scripts/top3_paper_freeze.py                       # the one real freeze (runbook, 2026-11-20)
    python scripts/top3_paper_freeze.py --dry-run --out DIR   # rehearsal: writes to DIR only, ledger untouched

Refuses (exit 2, nothing written) when:
  * the protocol file or the ranking code (src/research/top3_ranking.py) is not exactly the pinned one;
  * the study universes are not the OWN-PRICE universes of a CLEAN retrained frame listed in
    config/top3_paper_tracks.json -> clean_frames (frame 58f10476f4c82d7b = the clean champion_recipe re-run of
    2026-11-10; the Nov 19-20 final-refit frame is added there, with its universes rebuilt, before the freeze);
  * an approved proposal file does not verify against its pinned id and hash;
  * the freeze ledger already names a record (the freeze happens once, before the first pick) - real run only;
  * a real run outside its window [freeze_window_from_utc, start_utc) = 2026-11-20T00:00Z .. 2026-11-21T00:00Z.

WHAT IS COMPUTED, per approved MODEL cell (a LEAGUE_WIDE cell is the owner's own list, outside the pool) (league x market-side of TP-2026-11-10-d999073a, exactly as approved):
  * R2 / R3 calibration rows: every priced row of the cell's league x market-side in the own-price universe of
    the clean frame, dated before the as-of day, inside the cell's own odds range, scored by the cell's SERVED
    source (src.bars_opt.serve.score with the approved cell's source + source_params);
  * R1 bets: the cell's served tips (serve.score tier) on those rows dated from bets_from (2026-09-01) to the
    as-of day - out of every SELECT used for a served config; a FRESH_HOLDOUT_SELECTION cell only when its
    family's fresh holdout ended before bets_from, else it gets no R1 statistic.
  -> src.research.top3_ranking.cell_stats. Statistics only (no price, no row) go into the record.
The record's sha256 (LF) goes into the freeze ledger; commit both before the first paper pick.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bars_opt import serve as BS  # noqa: E402
from src.bars_opt import study as S  # noqa: E402
from src.research import top3_paper as TP3  # noqa: E402
from src.research import top3_ranking as R  # noqa: E402


def sha_raw(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean_frame(cfg: dict, root: Path) -> tuple[str | None, dict, list[str]]:
    """The listed clean frame whose universe files are byte-identical to the committed ones."""
    got = {t: sha_raw(root / f"output/bars_opt/universe_{t}.parquet") for t in ("standard", "new_format")}
    for fh, meta in (cfg.get("clean_frames") or {}).items():
        if meta.get("universe_sha256") == got:
            return fh, meta, []
    return None, {}, [f"the universes {({k: v[:12] for k, v in got.items()})} are not a listed clean frame"]


def fresh_ends(cfg: dict, root: Path) -> tuple[dict, list[str]]:
    p = root / str(cfg.get("fresh_holdout_record") or "")
    if not p.is_file() or sha_raw(p) != cfg.get("fresh_holdout_record_sha256"):
        return {}, ["fresh-holdout record missing or not the pinned file"]
    d = json.loads(p.read_text(encoding="utf-8"))
    return {(f["track"], f["mside"]): f.get("holdout_to") for f in d.get("families") or []}, []


def compute(cfg: dict, root: Path) -> tuple[pd.DataFrame, dict]:
    as_of_day = pd.Timestamp(cfg["stats_as_of_utc"]).tz_convert(None).normalize() + pd.Timedelta(days=1)
    bets_from = pd.Timestamp(cfg["bets_from"])
    cells, _ = TP3.approved_cells(cfg, root)
    ends, _ = fresh_ends(cfg, root)
    unis = {t: pd.read_parquet(root / f"output/bars_opt/universe_{t}.parquet") for t in ("standard", "new_format")}
    fams: dict = {}
    rows, bets, notes = [], [], []
    for (_mk, league, _side), (cell, _pid) in sorted(cells.items(), key=lambda kv: str(kv[0])):
        tr, ms = cell["track"], cell["mside"]
        if not R.in_pool(cell):
            notes.append(f"{tr}|{league}|{ms}: LEAGUE_WIDE cell - own list, outside the TOP-3 pool: no statistic")
            continue
        if (tr, ms) not in fams:
            fams[(tr, ms)] = S.family_frame(unis[tr], ms)
        f = fams[(tr, ms)]
        f = f[(f["league"] == league) & (f["date"] < as_of_day)]
        if f.empty:
            notes.append(f"{tr}|{league}|{ms}: no priced row before {as_of_day.date()}")
            continue
        sc = BS.score(f[["league", "p_side", "fair", "odds"]], cell)
        ok = sc["in_range"].astype(bool) & pd.to_numeric(sc["p_used"], errors="coerce").notna()
        rows.append(pd.DataFrame({"track": tr, "league": league, "mside": ms, "fair": f.loc[ok, "fair"],
                                  "p_used": pd.to_numeric(sc.loc[ok, "p_used"]), "won": f.loc[ok, "won"]}))
        r1_ok = True
        if cell.get("config_basis") == "FRESH_HOLDOUT_SELECTION":
            end = ends.get((tr, ms))
            r1_ok = bool(end) and pd.Timestamp(end) < bets_from
            if not r1_ok:
                notes.append(f"{tr}|{league}|{ms}: fresh-holdout selection reaches {end}: no R1 statistic")
        tipped = (sc["tier"].astype(str) != "") & (f["date"] >= bets_from)
        if r1_ok and tipped.any():
            bets.append(pd.DataFrame({"track": tr, "league": league, "mside": ms, "date": f.loc[tipped, "date"],
                                      "pnl": f.loc[tipped, "pnl"]}))
    rows_df = pd.concat(rows, ignore_index=True) if rows else None
    bets_df = pd.concat(bets, ignore_index=True) if bets else None
    st = R.cell_stats(bets_df, rows_df)
    keep = {(c["track"], c["league"], c["mside"]) for c, _ in cells.values() if R.in_pool(c)}
    st = st[[k in keep for k in zip(st["track"], st["league"], st["mside"])]].reset_index(drop=True)
    return st, {"approved_cells": len(cells), "cells_with_rows": int(len(rows)), "notes": notes,
                "as_of_day_exclusive": str(as_of_day.date()), "bets_from": str(bets_from.date())}


def _plain(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", help="dry run: folder for the record and a rehearsal ledger")
    ap.add_argument("--now", help=argparse.SUPPRESS)          # tests only: the clock for the window check
    a = ap.parse_args(argv)
    cfg = TP3.load_config(ROOT)
    why = TP3.hash_problems(cfg, ROOT)
    fh, fmeta, w = clean_frame(cfg, ROOT)
    why += w
    _, w = TP3.approved_cells(cfg, ROOT)
    why += w
    _, w = fresh_ends(cfg, ROOT)
    why += w
    ledger_p = ROOT / cfg["freeze_ledger"]
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC")
    lo, hi = pd.Timestamp(cfg["freeze_window_from_utc"]), pd.Timestamp(cfg["start_utc"])
    if not a.dry_run and not (lo <= now < hi):
        why.append(f"the real freeze runs only in its window {lo} .. {hi} (now {now}); use --dry-run to rehearse")
    if not a.dry_run and TP3._json(ledger_p).get("record"):
        why.append(f"{cfg['freeze_ledger']} already names a frozen record - the freeze happens once")
    if a.dry_run and not a.out:
        why.append("--dry-run needs --out (a folder outside the frozen record's place)")
    if why:
        print("REFUSED:\n  " + "\n  ".join(why))
        return 2
    st, meta = compute(cfg, ROOT)
    proto = json.loads((ROOT / cfg["protocol_file"]).read_text(encoding="utf-8"))
    rec = {"label": TP3.LABEL, "kind": "DRY_RUN" if a.dry_run else "FROZEN",
           "protocol_file": cfg["protocol_file"], "protocol_sha256": cfg["protocol_sha256"],
           "ranking_code_sha256": proto["rules_common"]["code"]["sha256_lf"],
           "stats_as_of_utc": cfg["stats_as_of_utc"], "frame_hash": fh, "frame": fmeta,
           "frame_note": ("statistics from the walk-forward predictions of the CLEAN retrained recipe (never "
                          "v9_repaired / pre-repair); if the final refit changes the frame before the freeze the "
                          "statistics are computed on the final frame - allowed because the freeze happens once, "
                          "before the first pick (protocol v2 live_statistics)"),
           "prices": "none in this record (own-price universes read locally; statistics only)",
           "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")} | meta | \
          {"cells": [{k: _plain(v) for k, v in r.items()} for r in st.to_dict("records")]}
    text = json.dumps(rec, indent=1, sort_keys=True) + "\n"
    if a.dry_run:
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        rp, lp = od / Path(cfg["freeze_record"]).name, od / Path(cfg["freeze_ledger"]).name
    else:
        rp, lp = ROOT / cfg["freeze_record"], ledger_p
        rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_bytes(text.encode("utf-8"))
    sha = TP3.sha256_lf(rp)
    led = {"record": cfg["freeze_record"] if not a.dry_run else str(rp), "sha256": sha, "kind": rec["kind"],
           "protocol_sha256": cfg["protocol_sha256"], "ranking_code_sha256": rec["ranking_code_sha256"],
           "frame_hash": fh, "frozen_at": rec["written_at"]}
    lp.write_bytes((json.dumps(led, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    n_l = int(st["lcb"].notna().sum())
    n_lam = int(st["lam"].notna().sum())
    print(json.dumps({"kind": rec["kind"], "record": str(rp), "sha256": sha, "frame_hash": fh, "cells": len(st),
                      "cells_with_R1": n_l, "cells_with_R2_R3": n_lam, "notes": len(meta["notes"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
