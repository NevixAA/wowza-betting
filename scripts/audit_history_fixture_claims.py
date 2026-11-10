"""
audit_history_fixture_claims.py - one provider fixture, one history row: the 2026-11-10 audit
(fix/history-dupes, from the OddsAPI history matcher's findings). Stored provider data only, 0 calls.

    python scripts/audit_history_fixture_claims.py            # write the audit
    python scripts/audit_history_fixture_claims.py --dry-run  # print it; write nothing

Every row of each frame below claims at most ONE API-Football fixture (src.integrity.fixture_claims:
its own fixture id; else both clubs proven home->home / away->away on the same day; else one club
proven, the only fixture of that club that day, and the same full-time score; else both clubs on the
adjacent day with the same score). A fixture claimed by two or more rows of one frame is a duplicate.

  af_history_raw          output/af_history.parquet as committed
  af_history_corrected    after src.af_history_corrections (the owner's duplicate lists)
  af_history_shadows      the corrected file with each written-back row (no fixture id, no FT; never
                          served - src.af_history_provenance) given the score of the fd_history row
                          of the SAME key, so the shadow copies the name rule cannot see are counted
  af_history_served       the corrected file's provider rows (what the loader serves)
  fd_history_raw          output/fd_history.parquet as committed
  fd_history_corrected    after src.history_corrections (removal lists + provider dates)
  served                  fd_history + the backtest archive through src.data_loader.one_club_one_match
                          (the training frame; data truth rule 5's input)

Also recorded, each from the stored provider fixture lists:
  * VERIFIED rows: history rows a second source disputed that the provider's fixture record confirms
    (VERIFY below; the history is left as it is);
  * PROVIDER FIXTURES NOT SERVED: the provider's finished fixtures of a (league, API season) in CHECK_UNSERVED
    that no served row claims, with whether any history bank holds them.

OUTPUT output/corrections/history_fixture_claims_audit.json (no history file is changed).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integrity import fixture_claims as FC  # noqa: E402

OUT = ROOT / "output" / "corrections" / "history_fixture_claims_audit.json"
#: disputed history rows (league, date, home, away) - 2026-11-10: The Odds API listed 'Independiente'
#: where the history holds Independiente Rivadavia
VERIFY = [("Argentina Primera Division", "2024-05-12", "Lanus", "Ind. Rivadavia"),
          ("Argentina Primera Division", "2024-05-18", "Ind. Rivadavia", "Godoy Cruz")]
#: (league, API season) whose unserved provider fixtures are listed - 2026-11-10: the matcher found
#: Austrian Bundesliga May 2026 games in the raw files but not in served history
CHECK_UNSERVED = [("Austrian Bundesliga", "2025")]


def _canon(d: pd.DataFrame) -> pd.DataFrame:
    from src import league_registry as LR
    d = d.copy()
    d["league"] = d["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    return d


def _summary(rows: pd.DataFrame, pool: pd.DataFrame, hg: str, ag: str, fid: str | None = None,
             wb: pd.Series | None = None) -> tuple[dict, pd.DataFrame]:
    c = FC.claims(_canon(rows), pool, home_goals=hg, away_goals=ag, fixture_col=fid)
    g = FC.duplicate_claims(c)
    per = {}
    for lg, gg in g.groupby("league"):
        e = {"fixtures_claimed_twice": int(gg["fixture_id"].nunique()),
             "extra_rows": int(len(gg) - gg["fixture_id"].nunique())}
        if wb is not None:
            w = wb.reindex(gg["idx"]).to_numpy().astype(bool)
            kinds = pd.Series(w, index=gg.index).groupby(gg["fixture_id"]).agg(["any", "all"])
            e["written_back_and_provider"] = int((kinds["any"] & ~kinds["all"]).sum())
            e["written_back_only"] = int(kinds["all"].sum())
            e["provider_only"] = int((~kinds["any"]).sum())
        per[lg] = e
    return ({"rows": int(len(rows)), "claimed": int(len(c)),
             "fixtures_claimed_twice": int(g["fixture_id"].nunique()),
             "extra_rows": int(len(g) - g["fixture_id"].nunique()), "per_league": per}, c)


def verify(fd: pd.DataFrame, pool: pd.DataFrame) -> list[dict]:
    out = []
    d = pd.to_datetime(fd["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for lg, day, h, a in VERIFY:
        row = fd[(fd["league"] == lg) & (d == day) & (fd["home_team"] == h) & (fd["away_team"] == a)]
        c = FC.claims(_canon(row), pool) if len(row) else pd.DataFrame()
        e = {"league": lg, "date": day, "home_team": h, "away_team": a,
             "history_score": (None if row.empty else f"{row['home_goals'].iloc[0]:.0f}-{row['away_goals'].iloc[0]:.0f}")}
        if c.empty:
            e["verdict"] = "UNPROVEN: no provider fixture claimed"
        else:
            f = pool[pool["fixture_id"] == c["fixture_id"].iloc[0]].iloc[0]
            e.update({"provider_fixture_id": int(f["fixture_id"]), "claim_rule": c["rule"].iloc[0],
                      "provider_home": f["home_name"], "provider_away": f["away_name"],
                      "provider_home_id": None if pd.isna(f.get("home_id")) else int(f["home_id"]),
                      "provider_away_id": None if pd.isna(f.get("away_id")) else int(f["away_id"]),
                      "provider_score": f"{f['ft_home']:.0f}-{f['ft_away']:.0f}", "provider_status": f["status"]})
            same = e["history_score"] == e["provider_score"]
            e["verdict"] = ("HISTORY_CORRECT: the provider's fixture record names the history's clubs and "
                            "score; left as it is" if same else "SCORE_DIFFERS: listed, not changed")
        out.append(e)
    return out


def unserved(served_claims: pd.DataFrame, pool: pd.DataFrame, banks: dict) -> list[dict]:
    out = []
    for lg, season in CHECK_UNSERVED:
        p = pool[(pool["league"] == lg) & (pool["season"].astype(str) == season)]
        miss = p[~p["fixture_id"].isin(set(served_claims["fixture_id"]))]
        for _, f in miss.iterrows():
            where = {n: int(len(c[c["fixture_id"] == f["fixture_id"]])) for n, c in banks.items()}
            out.append({"league": lg, "season": season, "fixture_id": int(f["fixture_id"]), "date": f["_d"],
                        "home": f["home_name"], "away": f["away_name"],
                        "score": f"{f['ft_home']:.0f}-{f['ft_away']:.0f}", "rows_claiming_it": where})
    return out


def run(dry_run: bool = False) -> dict:
    logging.disable(logging.WARNING)
    from src import data_loader as DL
    from src.af_history_corrections import apply_af_history_corrections
    from src.af_history_provenance import written_back_mask
    from src.history_corrections import apply_date_corrections, apply_history_corrections
    from src.integrity import data_truth as DT

    pool = FC.provider_pool()
    af = pd.read_parquet(ROOT / "output" / "af_history.parquet")
    fd = pd.read_parquet(ROOT / "output" / "fd_history.parquet")
    doc = {"artifact": "history_fixture_claims_audit",
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "rule": FC.__doc__.split("THE CLAIM")[1].split("Provider pool")[0].strip(),
           "frames": {}}
    afc = apply_af_history_corrections(af, tag="audit")
    wbc = written_back_mask(afc)
    doc["frames"]["af_history_raw"], c_af = _summary(af, pool, "FTHG", "FTAG", "fixture_id", written_back_mask(af))
    doc["frames"]["af_history_corrected"], _ = _summary(afc, pool, "FTHG", "FTAG", "fixture_id", wbc)
    # shadows: a written-back row takes the score of the fd_history row of the same key
    k = ["league", "date", "home_team", "away_team"]
    sc = fd.dropna(subset=["home_goals", "away_goals"]).drop_duplicates(k)[k + ["home_goals", "away_goals"]]
    sh = afc.reset_index().merge(sc, on=k, how="left").set_index("index")
    sh["FTHG"] = sh["FTHG"].where(sh["FTHG"].notna(), sh["home_goals"])
    sh["FTAG"] = sh["FTAG"].where(sh["FTAG"].notna(), sh["away_goals"])
    doc["frames"]["af_history_shadows"], _ = _summary(sh, pool, "FTHG", "FTAG", "fixture_id", wbc)
    doc["frames"]["af_history_served"], _ = _summary(afc[~wbc.to_numpy()], pool, "FTHG", "FTAG", "fixture_id")
    doc["frames"]["fd_history_raw"], c_fd = _summary(fd, pool, "home_goals", "away_goals")
    fdc = apply_date_corrections(apply_history_corrections(fd, tag="audit"), tag="audit")
    doc["frames"]["fd_history_corrected"], _ = _summary(fdc, pool, "home_goals", "away_goals")
    parts = [DT._bank_matches("fd_history.parquet").assign(row_source="fd_cache"),
             DL._drop_misfiled_league_rows(DT._bank_matches("backtest_all_leagues.csv"), "audit")
             .assign(row_source="backtest_archive")]
    merged = pd.concat(parts, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)
    served = DL.one_club_one_match(merged)
    doc["frames"]["served"], c_sv = _summary(served, pool, "home_goals", "away_goals")
    doc["verified_rows"] = verify(fd, pool)
    doc["provider_fixtures_not_served"] = unserved(c_sv, pool, {"af_history": c_af, "fd_history": c_fd,
                                                                "served": c_sv})
    if not dry_run:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    doc = run(a.dry_run)
    for n, f in doc["frames"].items():
        print(f"{n:22} rows {f['rows']:>6}  claimed {f['claimed']:>6}  provider fixtures claimed twice "
              f"{f['fixtures_claimed_twice']:>5}  extra rows {f['extra_rows']:>5}")
    for v in doc["verified_rows"]:
        print(f"verified: {v['league']} {v['date']} {v['home_team']} v {v['away_team']}: {v['verdict']}")
    for u in doc["provider_fixtures_not_served"]:
        print(f"not served: {u['league']} {u['date']} {u['home']} v {u['away']} {u['score']} "
              f"(fixture {u['fixture_id']}) rows claiming it {u['rows_claiming_it']}")
    if not a.dry_run:
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
