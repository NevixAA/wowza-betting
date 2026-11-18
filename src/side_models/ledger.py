"""Side models -> prediction ledger: every upcoming fixture of a covered league x corners/cards market.
====================================================================================================

    python -m src.side_models.ledger board       # append due snapshots (predict.yml; flag-gated)
    python -m src.side_models.ledger preview     # print counts only, write nothing

LEARNING ONLY (owner rule 2026-11-17/18: every model predicts every match of every tracked league for
every market it serves). Not a tip, not a stake: nothing here reads or writes a tip, tier, gate, stake,
threshold, board or Telegram file, and no bookmaker price is read or stored.

FIXTURES: output/collect/upcoming_fixtures.parquet (API-Football ids; NS fixtures kicking off within
config/side_models.json `horizon_hours`). A fixture of a covered league that the model cannot score
(a side with too few verified matches in its window) is stored unscored with the reason. A tracked
league NOT covered for a family (single source / agreement below the rule) gets no rows; its reason is
in models/side_models/side_models.json `coverage`.

SWITCH: writes only when the repo variable SIDE_MODELS_LEDGER_ENABLED is 'true' (default OFF) AND the
ledger itself is on (PREDICTION_LEDGER_ENABLED, src/prediction_ledger.enabled). Never raises into the
caller: a failure is printed as a warning and the exit code stays 0 (predict must not stop).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import prediction_ledger as PL  # noqa: E402
from src.side_models import data as D  # noqa: E402
from src.side_models import features as F  # noqa: E402
from src.side_models import model as M  # noqa: E402
from src.side_models import train as T  # noqa: E402

REPO = "v9"
WRITER = "side_models"
UPCOMING = ROOT / "output" / "collect" / "upcoming_fixtures.parquet"
ARTIFACT = ROOT / "models" / "side_models" / "side_models.json"
FLAG_ENV = "SIDE_MODELS_LEDGER_ENABLED"
PAYLOAD_VERSION = "side/1.0"
UPCOMING_STATUSES = ("NS", "TBD")


def enabled() -> bool:
    return os.getenv(FLAG_ENV, "").strip().lower() == "true" and PL.enabled()


def load_artifact(path: Path | None = None) -> dict:
    return json.loads(Path(path or ARTIFACT).read_text(encoding="utf-8"))


def artifact_sha(path: Path | None = None) -> str:
    p = Path(path or ARTIFACT)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else ""


def _fhash(vals) -> str:
    h = hashlib.sha1()
    for v in vals:
        f = float(v)
        h.update(b"nan|" if f != f else f"{f:.10g}|".encode())
    return h.hexdigest()[:12]


def upcoming(now: datetime, horizon_h: float, path: Path | None = None, slugs: PL.LeagueSlugs | None = None
             ) -> pd.DataFrame:
    """NS fixtures of tracked registry leagues kicking off in (now, now + horizon]."""
    slugs = slugs or PL.LeagueSlugs()
    u = pd.read_parquet(path or UPCOMING)
    u = u[u["status"].isin(UPCOMING_STATUSES)].drop_duplicates("fixture_id")
    ko = pd.to_datetime(u["kickoff"], utc=True, errors="coerce")
    t0 = pd.Timestamp(now)
    u = u[(ko > t0) & (ko <= t0 + pd.Timedelta(hours=horizon_h))].copy()
    by_af = {int(e["api_football_id"]): e for e in slugs.leagues
             if e.get("api_football_id") is not None and PL.tracked(e)}
    u = u[u["league_id"].astype(int).isin(by_af)].copy()
    u["league"] = [by_af[int(x)]["name"] for x in u["league_id"]]
    u["kickoff_utc"] = pd.to_datetime(u["kickoff"], utc=True)
    u["date"] = u["kickoff_utc"].dt.strftime("%Y-%m-%d")
    return u.reset_index(drop=True)


def _team_names(fixtures_path: Path | None = None) -> dict:
    fx = pd.read_parquet(fixtures_path or D.FIXTURES, columns=["date", "home_id", "home_name", "away_id",
                                                               "away_name"])
    long = pd.concat([fx[["date", "home_id", "home_name"]].set_axis(["date", "id", "name"], axis=1),
                      fx[["date", "away_id", "away_name"]].set_axis(["date", "id", "name"], axis=1)])
    long = long.sort_values("date").drop_duplicates("id", keep="last")
    return dict(zip(long["id"].astype(int), long["name"]))


def score(table: pd.DataFrame, targets: pd.DataFrame, art: dict, cfg: dict) -> dict:
    """{family: features frame with p_<market>_<side>_<line> columns} for the covered leagues."""
    out = {}
    for fam, fc in cfg["families"].items():
        cov = {lg for lg, c in art["coverage"][fam].items() if c.get("covered")}
        tg = targets[targets["league"].isin(cov)]
        if tg.empty:
            out[fam] = tg
            continue
        parts = []
        for lg, g in tg.groupby("league"):
            f = F.build(table[table["league"] == lg], g, fam, **cfg["features"])
            prm = art["models"][fam][lg]["params"]
            ok = (f["not_scored_reason"] == "").to_numpy()
            for market, mc in fc["markets"].items():
                sides = ["total"] if mc["part"] == "total" else ["home", "away"]
                for side in sides:
                    mu = np.full(len(f), np.nan)
                    if ok.any():
                        mu[ok] = M.mu_of(prm[side], f[ok], side)
                    for line in mc["lines"]:
                        p = np.full(len(f), np.nan)
                        if ok.any():
                            p[ok] = M.p_over(mu[ok], prm[side]["alpha"], line)
                        f[f"p|{market}|{side}|{line:g}"] = p
            parts.append(f)
        out[fam] = pd.concat(parts, ignore_index=True)
    return out


def rows_for(scored: dict, art: dict, cfg: dict, *, now: datetime, slugs: PL.LeagueSlugs,
             names: dict | None = None) -> list:
    names = names or {}
    sha = artifact_sha()
    rows = []
    for fam, f in scored.items():
        if f is None or f.empty:
            continue
        fc = cfg["families"][fam]
        for r in f.to_dict("records"):
            ko = PL.utc_ts(r["kickoff_utc"])
            mtk = (ko - pd.Timestamp(now)).total_seconds() / 60.0
            due = PL.snapshots_due(mtk)
            if not due:
                continue
            entry = slugs.entry(r["league"])
            season = PL.season_of(entry, ko)
            fh = _fhash([r["home_for"], r["home_against"], r["away_for"], r["away_against"]])
            home, away = names.get(int(r["home_id"]), ""), names.get(int(r["away_id"]), "")
            reason = r["not_scored_reason"] or ""
            for market, mc in fc["markets"].items():
                sides = ["total"] if mc["part"] == "total" else ["home", "away"]
                for side in sides:
                    for line in mc["lines"]:
                        p = r[f"p|{market}|{side}|{line:g}"]
                        for over in (True, False):
                            if side == "total":
                                sel, team = ("OVER" if over else "UNDER"), ""
                            else:
                                sel = f"{side.upper()}_{'OVER' if over else 'UNDER'}"
                                team = home if side == "home" else away
                            prob = None if (p is None or p != p) else (float(p) if over else 1.0 - float(p))
                            for snap in due:
                                rows.append(PL.make_row(
                                    repo=REPO, model_id=fc["model_id"], model_version=art["model_version"],
                                    model_sha=sha, payload_version=PAYLOAD_VERSION, feature_hash=fh,
                                    track=(entry or {}).get("model_type", ""), market=market, line=line,
                                    selection=sel, fixture_ref=PL.fixture_ref(r["fixture_id"]),
                                    fixture_id=r["fixture_id"], league=r["league"], season=season,
                                    kickoff_utc=PL.iso(ko), home_team=home, away_team=away, team=team,
                                    probability=prob, snapshot=snap,
                                    predicted_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    minutes_to_kickoff=round(mtk, 1),
                                    not_scored_reason="" if prob is not None else (
                                        reason or "no probability"),
                                    git_sha=(os.getenv("GITHUB_SHA") or "")[:12],
                                    run_id=os.getenv("GITHUB_RUN_ID", "")))
    return rows


def build_rows(now: datetime | None = None, *, upcoming_path=None, slugs=None) -> tuple[list, dict]:
    """Every ledger row due now + a summary (no write)."""
    now = now or datetime.now(timezone.utc)
    cfg = T.load_config()
    art = load_artifact()
    slugs = slugs or PL.LeagueSlugs()
    tg = upcoming(now, cfg["horizon_hours"], upcoming_path, slugs)
    covered_any = set()
    for fam in cfg["families"]:
        covered_any |= {lg for lg, c in art["coverage"][fam].items() if c.get("covered")}
    table, _ = D.build(sorted(covered_any))
    scored = score(table, tg, art, cfg)
    rows = rows_for(scored, art, cfg, now=now, slugs=slugs, names=_team_names())
    summ = {"fixtures_in_horizon": int(len(tg)), "rows": len(rows),
            "scored_rows": int(sum(1 for r in rows if r["scored"] == 1)),
            "by_family": {fam: {"fixtures": int(len(f)),
                                "scored_fixtures": int((f["not_scored_reason"] == "").sum()) if len(f) else 0}
                          for fam, f in scored.items()},
            "not_covered_leagues_in_horizon": {
                fam: sorted(set(tg["league"]) - {lg for lg, c in art["coverage"][fam].items() if c.get("covered")})
                for fam in cfg["families"]}}
    return rows, summ


def board(now: datetime | None = None) -> dict:
    if not enabled():
        print(f"[side_models] {FLAG_ENV} / {PL.FLAG_ENV} not both 'true': nothing written")
        return {"written": 0, "disabled": True}
    rows, summ = build_rows(now)
    st = PL.append(rows, repo=REPO, writer=WRITER, run_id=os.getenv("GITHUB_RUN_ID", ""), now=now)
    print(f"[side_models] {json.dumps(summ)}")
    print(f"[side_models] append: offered {st['offered']} written {st['written']} duplicate "
          f"{st['duplicate_skipped']} late {st['late_skipped']} refused {st['refused']} "
          f"violations {st['violations']}")
    return st


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "preview"
    try:
        if cmd == "board":
            board()
        elif cmd == "preview":
            rows, summ = build_rows()
            print(json.dumps(summ, indent=1))
        else:
            print(f"unknown command {cmd!r} (board | preview)")
            return 2
    except Exception as e:                                             # noqa: BLE001
        print(f"::warning::side models ledger failed (learning only - tips unaffected): "
              f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
