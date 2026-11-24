"""Monthly side-model (corners / cards) CANDIDATE refit - owner approval 2026-11-24 ("yes").
==========================================================================================

    python -m src.side_models.candidate gate  --dt-rc <data_truth --gate exit code> --out DIR
    python -m src.side_models.candidate build --out DIR [--run-date YYYY-MM-DD]
    python -m src.side_models.candidate notify --out DIR          # Telegram (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID)

Run by .github/workflows/train_side_models.yml (monthly schedule + dispatch, mode=monthly_candidate).

1. GATE. Runs only when `python -m src.integrity.data_truth --gate` is GREEN (exit 0) AND the served side models'
   lineage PASSes (src.integrity.training_lineage.side_lineage). Otherwise it SKIPS with the reason (gate.json,
   the run summary and the Telegram note).
2. REFIT. The INSTALL candidate is the recorded recipe (src.side_models.train.run, config/side_models.json) on every
   verified game up to the day before the run date, read from the git tree of the run's commit (src.side_models.
   reproduce.materialise). Its recipe record (new data_ref, until, data cut-offs, input blob ids) and its lineage
   evidence (a second, independent re-train from that recipe, which must match) are written beside it.
3. COMPARE ON HELD-OUT GAMES ONLY. Held-out window = [start, until] with start = the later of (serving data_end
   + 1 day) and (until - HOLDOUT_DAYS + 1). Neither model trained on a held-out game: the serving model's training
   data ends at its data_end, and the comparison is made with a candidate fitted by the SAME recipe on games before
   `start` only (`compare_fit`); every league's fit_to is checked to lie before `start` (else the run refuses).
   Log loss and Brier per league x market x side x line (corners/cards, total and team), with counts.
4. NEVER INSTALLS. Nothing here writes models/, config/ or reports/ of the checkout: everything goes to --out. The
   workflow pushes a candidate BRANCH and opens a PR titled "... awaiting owner approval"; merging it is the owner's
   one-click approval. No price is read or written.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.side_models import model as M  # noqa: E402
from src.side_models import reproduce as R  # noqa: E402
from src.side_models import train as T  # noqa: E402

HOLDOUT_DAYS = 42
EPS = 1e-15
SERVING = ROOT / "models" / "side_models" / "side_models.json"


# ── 1. gate ─────────────────────────────────────────────────────────────────────────────────────────────
def gate_decision(dt_rc: int | None, lineage: dict | None) -> tuple[bool, str]:
    """(run?, reason). Runs only on a GREEN data-truth gate AND a PASS side-model lineage."""
    reasons = []
    if dt_rc != 0:
        reasons.append("data_truth --gate is not GREEN" + (f" (exit {dt_rc})" if dt_rc is not None else " (not run)"))
    st = (lineage or {}).get("status")
    if st != "PASS":
        reasons.append(f"side-model lineage is {st or 'missing'}" + (f": {lineage.get('why')}" if lineage and
                                                                       lineage.get("why") else ""))
    return (not reasons), ("; ".join(reasons) if reasons else "data_truth GREEN and side-model lineage PASS")


# ── 3. held-out comparison ──────────────────────────────────────────────────────────────────────────────
def holdout_start(serving_data_end: str, until: str, days: int = HOLDOUT_DAYS) -> str:
    a = (date.fromisoformat(serving_data_end) + timedelta(days=1))
    b = (date.fromisoformat(until) - timedelta(days=days - 1))
    return max(a, b).isoformat()


def _ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(np.sum(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def assert_held_out(start: str, serving: dict, compare_fit: dict) -> None:
    """Refuse a comparison on any game a compared model trained on."""
    if str(serving["data_end"]) >= start:
        raise ValueError(f"serving data_end {serving['data_end']} is not before the held-out start {start}")
    for fam, mods in compare_fit["models"].items():
        for lg, m in mods.items():
            if str(m["fit_to"]) >= start:
                raise ValueError(f"compare_fit {fam}/{lg} fit_to {m['fit_to']} is not before the held-out start {start}")


def heldout_metrics(table: pd.DataFrame, serving: dict, compare_fit: dict, start: str, until: str,
                    cfg: dict) -> pd.DataFrame:
    """One row per family x league x market x side x line: n held-out games, log loss / Brier of the serving model
    and of the candidate recipe (fitted before `start`). Leagues served by only one model are listed with n = 0."""
    assert_held_out(start, serving, compare_fit)
    rows = []
    for fam, fc in cfg["families"].items():
        sv, cf = serving["models"].get(fam, {}), compare_fit["models"].get(fam, {})
        for lg in sorted(set(sv) | set(cf)):
            win = (compare_fit["coverage"].get(fam, {}).get(lg) or serving["coverage"].get(fam, {}).get(lg) or {})
            fr = T.training_frame(table, lg, fam, cfg["features"], win.get("window_from") or "1900-01-01") \
                if lg in sv and lg in cf else pd.DataFrame()
            if len(fr):
                fr = fr[(fr["date"] >= start) & (fr["date"] <= until)].reset_index(drop=True)
            for market, mc in fc["markets"].items():
                for side in (["total"] if mc["part"] == "total" else ["home", "away"]):
                    for line in mc["lines"]:
                        r = {"family": fam, "league": lg, "market": market, "side": side, "line": line,
                             "n": int(len(fr)), "in_serving": lg in sv, "in_candidate": lg in cf}
                        if len(fr):
                            y = T._targets(fr, fam)[side]
                            yy = (y > line).astype(float)
                            for tag, prm in (("serving", sv[lg]["params"][side]), ("candidate", cf[lg]["params"][side])):
                                p = M.p_over(M.mu_of(prm, fr, side), prm["alpha"], line)
                                r[f"logloss_{tag}"] = _ll(p, yy) / len(yy)
                                r[f"brier_{tag}"] = float(np.mean((p - yy) ** 2))
                        rows.append(r)
    return pd.DataFrame(rows)


def league_market_summary(m: pd.DataFrame) -> pd.DataFrame:
    """Per family x league x market: held-out games and the mean (over sides x lines) log loss / Brier, both models."""
    if m.empty:
        return m
    d = m[m["n"] > 0]
    if d.empty:
        return d
    g = d.groupby(["family", "league", "market"])
    out = g.agg(n_games=("n", "max"), logloss_serving=("logloss_serving", "mean"),
                logloss_candidate=("logloss_candidate", "mean"), brier_serving=("brier_serving", "mean"),
                brier_candidate=("brier_candidate", "mean")).reset_index()
    out["logloss_delta"] = out["logloss_candidate"] - out["logloss_serving"]
    out["brier_delta"] = out["brier_candidate"] - out["brier_serving"]
    return out.round(5)


# ── 2. refit + recipe + lineage ─────────────────────────────────────────────────────────────────────────
def _last_commit(ref: str, path: str) -> str:
    try:
        return R._git("log", "-1", "--format=%h %cI", ref, "--", path).strip()
    except subprocess.CalledProcessError:
        return ""


def candidate_recipe(base: dict, art: dict, ref: str, until: str, blobs: dict) -> dict:
    rec = dict(base)
    rec.update({
        "model_version": art["model_version"], "trained_at": art["trained_at"], "data_ref": ref,
        "data_ref_note": f"monthly candidate refit (src/side_models/candidate.py) on the tree of {ref[:9]}",
        "until": until, "inputs": blobs, "inputs_digest": R.digest(blobs),
        "data_cutoffs": {
            "data_end": f"{art['data_end']} (latest fixture with a verified corners or cards count)",
            "until": f"{until} (the day before the run date: every verified game up to it)",
            "api_football_fullstats_fill": _last_commit(ref, "output/collect/af_history_fullstats_fill.parquet"),
            "competition_fixtures": _last_commit(ref, "output/strength/competition_fixtures.parquet"),
            "football_data_raw": _last_commit(ref, "output/collect/fd_raw"),
            "side_stats_sidecar": _last_commit(ref, "output/collect/af_side_stats.parquet") or "absent at data_ref",
            "leagues_registry": _last_commit(ref, "config/leagues.json"),
            "per_league_fit_window": "models/side_models/side_models.json models.<family>.<league>.fit_from / fit_to / n_fit",
        },
    })
    return rec


def build(out_dir: Path, run_date: str | None = None, data_ref: str = "HEAD", days: int = HOLDOUT_DAYS) -> dict:
    out_dir = Path(out_dir).resolve()
    try:
        out_dir.relative_to(ROOT.resolve())
        raise SystemExit(f"{out_dir} is inside the repo: refused (the candidate never writes the checkout)")
    except ValueError:
        pass
    run_date = run_date or datetime.now(timezone.utc).date().isoformat()
    until = (date.fromisoformat(run_date) - timedelta(days=1)).isoformat()
    ref = R.resolve(data_ref)
    serving = json.loads(SERVING.read_text(encoding="utf-8"))
    start = holdout_start(serving["data_end"], until, days)
    cut = (date.fromisoformat(start) - timedelta(days=1)).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    inst_dir = out_dir / "install"
    art_dir = inst_dir / "models" / "side_models"
    with tempfile.TemporaryDirectory(prefix="side_cand_") as tmp:
        blobs = R.materialise(ref, Path(tmp))
        with R.inputs_at(Path(tmp)):
            inst = T.run(write=True, out_dir=art_dir, until=until, trained_at=stamp)
            cmpf = T.run(write=True, out_dir=out_dir / "compare_fit", until=cut, trained_at=stamp)
            cfg = T.load_config()
    m = heldout_metrics(inst["table"], serving, cmpf["artifact"], start, until, cfg)
    s = league_market_summary(m)
    m.to_csv(out_dir / "heldout_metrics.csv", index=False, lineterminator="\n")
    s.to_csv(out_dir / "heldout_summary.csv", index=False, lineterminator="\n")
    # recipe + independent re-train (lineage evidence) for the install candidate
    base = json.loads(R.RECIPE.read_text(encoding="utf-8"))
    rec = candidate_recipe(base, inst["artifact"], ref, until, blobs)
    (inst_dir / "config").mkdir(parents=True, exist_ok=True)
    rpath = inst_dir / "config" / "side_models_recipe.json"
    rpath.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    rep = R.run(out_dir / "verify", ref_dir=art_dir, recipe_path=rpath)
    ev = R.evidence(rep, artifact=art_dir / "side_models.json")
    (inst_dir / "reports" / "lineage").mkdir(parents=True, exist_ok=True)
    (inst_dir / "reports" / "lineage" / "side_models_lineage_evidence.json").write_text(
        json.dumps(ev, indent=1) + "\n", encoding="utf-8", newline="\n")
    d = s if len(s) else pd.DataFrame(columns=["logloss_delta", "brier_delta", "n_games"])
    summary = {
        "artifact": "side_models_candidate", "run_date": run_date, "until": until, "data_ref": ref,
        "serving_model_version": serving["model_version"], "serving_data_end": serving["data_end"],
        "candidate_model_version": inst["artifact"]["model_version"], "candidate_data_end": inst["artifact"]["data_end"],
        "heldout": {"start": start, "end": until, "days": days, "compare_fit_until": cut,
                    "rule": "neither model trained on a held-out game (serving data_end and every compare_fit fit_to "
                            "are before start; checked)"},
        "leagues": {fam: {"serving": sorted(serving["models"].get(fam, {})),
                          "candidate": sorted(inst["artifact"]["models"].get(fam, {}))} for fam in cfg["families"]},
        "cells": int(len(d)), "heldout_games_max": int(d["n_games"].max()) if len(d) else 0,
        "candidate_better_logloss": int((d["logloss_delta"] < 0).sum()),
        "candidate_worse_logloss": int((d["logloss_delta"] > 0).sum()),
        "candidate_better_brier": int((d["brier_delta"] < 0).sum()),
        "candidate_worse_brier": int((d["brier_delta"] > 0).sum()),
        "mean_logloss_delta": round(float(d["logloss_delta"].mean()), 5) if len(d) else None,
        "mean_brier_delta": round(float(d["brier_delta"].mean()), 5) if len(d) else None,
        "lineage": {"status": ev["status"], "parity_basis": rep.get("parity_basis")},
        "install": "NOT installed. A PR titled 'awaiting owner approval' carries it; merging it is the approval.",
        "generated_at": stamp,
    }
    (out_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8",
                                                    newline="\n")
    (inst_dir / "reports" / "side_models").mkdir(parents=True, exist_ok=True)
    (inst_dir / "reports" / "side_models" / "candidate_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    s.to_csv(inst_dir / "reports" / "side_models" / "candidate_heldout_summary.csv", index=False, lineterminator="\n")
    return summary


# ── 5. report ───────────────────────────────────────────────────────────────────────────────────────────
def message(summary: dict | None, gate: dict | None) -> str:
    """Short Telegram text: no prices (none exist here), counts and deltas only."""
    if gate and not gate.get("run"):
        return f"Side models (corners/cards): monthly candidate SKIPPED - {gate.get('reason')}."
    if not summary:
        return "Side models (corners/cards): monthly candidate FAILED - no summary written (see the Actions run)."
    s, h = summary, summary["heldout"]
    return (f"Side models (corners/cards): monthly candidate {s['candidate_model_version']} "
            f"(data to {s['candidate_data_end']}) vs serving {s['serving_model_version']}.\n"
            f"Held-out {h['start']}..{h['end']} (neither model trained on it), up to {s['heldout_games_max']} games per "
            f"league: log loss better in {s['candidate_better_logloss']} / worse in {s['candidate_worse_logloss']} of "
            f"{s['cells']} league x market cells (mean delta {s['mean_logloss_delta']}); Brier better "
            f"{s['candidate_better_brier']} / worse {s['candidate_worse_brier']}.\n"
            f"Lineage {s['lineage']['status']}. NOT installed: merge the 'awaiting owner approval' PR to install.")


def _send(text: str) -> bool:
    tok, chat = os.getenv("TELEGRAM_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not tok or not chat:
        print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set: message not sent")
        return False
    from telegram_bot.notifier import _send as send
    return bool(send(tok, chat, text))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--dt-rc", type=int, default=None)
    g.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--run-date", default=None)
    n = sub.add_parser("notify")
    n.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "gate":
        from src.integrity import training_lineage as TL
        lin = TL.side_lineage(ROOT)
        ok, why = gate_decision(a.dt_rc, lin)
        (out / "gate.json").write_text(json.dumps({"run": ok, "reason": why, "lineage": lin}, indent=1, default=str)
                                       + "\n", encoding="utf-8")
        print(f"gate: {'RUN' if ok else 'SKIP'} - {why}")
        go = os.getenv("GITHUB_OUTPUT")
        if go:
            with open(go, "a", encoding="utf-8") as f:
                f.write(f"go={'true' if ok else 'false'}\n")
        return 0
    if a.cmd == "build":
        s = build(out, a.run_date)
        print(json.dumps({k: v for k, v in s.items() if k != "leagues"}, indent=1))
        return 0
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8")) if (out / "gate.json").exists() else None
    summ = json.loads((out / "candidate_summary.json").read_text(encoding="utf-8")) \
        if (out / "candidate_summary.json").exists() else None
    text = message(summ, gate)
    print(text)
    _send(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
