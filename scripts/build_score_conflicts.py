"""
build_score_conflicts.py - history rows whose 90-minute score disagrees with API-Football's, settled by a
THIRD record: API-Football's own goal events of that fixture (owner 2026-11-15, go-live rehearsal R3).

    python scripts/build_score_conflicts.py [--fd-history PATH]      # write the list
    python scripts/build_score_conflicts.py --dry-run

INPUT. The raw rows (fd_history + the backtest archive, the history lists applied) of every provider
fixture whose SERVED row (data truth rule 5's frame: the loader's identity pipeline, this list not
applied) is claimed (src.integrity.fixture_claims; the stored fixture lists plus the probed fixtures)
with an FT that differs from the provider's score.fulltime, and for each the stored answer of
scripts/probe_fixture_record.py (output/collect/fixture_probe_<id>.json: /fixtures?id + /fixtures/
events). A conflict without a probe answer is listed UNRESOLVED (data truth rule 5 stays RED on it).

THE EVENTS SCORE. Goal events (type 'Goal', not 'Missed Penalty') up to the 90th minute (elapsed <= 90:
stoppage time is elapsed 90 + extra; elapsed 91-120 is extra time); an own goal counts for the team the
event names (API-Football names the side credited - Sligo v Dundalk 2022-08-29: 'Sligo Rovers Own Goal
A. Boyle', a Dundalk player). The decision, per row:
  serve_provider_90min  events == provider score.fulltime != history: the history row is served at the
                        provider's 90-minute score (wrong score, or a score that includes extra time);
  keep_history          events == history != provider: the history holds the score played on the pitch
                        (the provider's FT is an awarded score, or includes extra time);
  exclude_awarded       events == provider != history AND the history score is a forfeit score (3-0 /
                        0-3): the history holds an AWARDED result of a match not completed on the pitch -
                        excluded like every awarded result (owner rule 2026-11-04);
  UNRESOLVED            anything else (listed; never applied).
Never rewrites a bank: output/corrections/history_score_conflicts_resolved.csv, applied by
src.history_corrections.apply_score_conflicts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import history_corrections as HC  # noqa: E402
from src.integrity import fixture_claims as FC  # noqa: E402

PROBES = ROOT / "output" / "collect"
FORFEIT = {(3, 0), (0, 3)}
COLS = HC.KEY + ["row_source", "history_ft_home", "history_ft_away", "decision", "served_ft_home", "served_ft_away",
                 "fixture_id", "provider_ft_home", "provider_ft_away", "events_ft_home", "events_ft_away",
                 "provider_home", "provider_away", "round", "evidence"]


def events_score(probe: dict) -> tuple[int, int, list] | None:
    calls = {c["name"]: c for c in probe.get("calls", [])}
    fx = (calls.get("fixture") or {}).get("response") or []
    ev = (calls.get("events") or {}).get("response")
    if not fx or ev is None:
        return None
    home = fx[0]["teams"]["home"]["id"]
    goals = [e for e in ev if e.get("type") == "Goal" and e.get("detail") != "Missed Penalty"]
    g90 = [e for e in goals if (e.get("time") or {}).get("elapsed") is not None and e["time"]["elapsed"] <= 90]
    h = sum(1 for e in g90 if (e.get("team") or {}).get("id") == home)
    return h, len(g90) - h, goals


def decide(hist: tuple, prov: tuple, evs: tuple | None) -> str:
    if evs is None:
        return "UNRESOLVED"
    if evs == prov and evs != hist:
        return "exclude_awarded" if hist in FORFEIT else "serve_provider_90min"
    if evs == hist and evs != prov:
        return "keep_history"
    return "UNRESOLVED"


def conflicts(frame: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Rows of `frame` (league canonical) claimed to a provider fixture with a different FT."""
    c = FC.claims(frame, pool)
    if c.empty:
        return c
    p = pool.set_index("fixture_id")
    c = c.join(p[["ft_home", "ft_away", "home_name", "away_name"]], on="fixture_id")
    c = c.join(frame[["home_team", "away_team", "home_goals", "away_goals", "row_source"]], on="idx")
    bad = c["ft_home"].notna() & ((c["home_goals"] != c["ft_home"]) | (c["away_goals"] != c["ft_away"]))
    return c[bad]


def raw_frame(fd_path: Path) -> pd.DataFrame:
    """Data truth rule 5's banks, raw keys kept (row_source per bank), before the identity pipeline."""
    from src import data_loader as DL
    from src.integrity import data_truth as DT
    fd = pd.read_parquet(fd_path, columns=DT._MATCH_COLS)
    fd["date"] = pd.to_datetime(fd["date"], errors="coerce")
    parts = [fd.dropna(subset=["date", "home_goals", "away_goals"]).assign(row_source="fd_cache"),
             DL._drop_misfiled_league_rows(DT._bank_matches("backtest_all_leagues.csv"), "conflicts")
             .assign(row_source="backtest_archive")]
    return pd.concat(parts, ignore_index=True)


def probe_pool() -> pd.DataFrame:
    """The probed fixtures in the pool's layout (a fixture the stored lists lack, e.g. 1648307)."""
    import config
    by_id = {int(v): k for k, v in getattr(config, "API_FOOTBALL_IDS", {}).items() if v is not None}
    rows = []
    for f in sorted(PROBES.glob("fixture_probe_*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        fx = (next((c for c in doc.get("calls", []) if c["name"] == "fixture"), {}) or {}).get("response") or []
        if not fx:
            continue
        x = fx[0]
        lg = by_id.get(int(x["league"]["id"]))
        if lg is None:
            continue
        rows.append({"fixture_id": int(x["fixture"]["id"]), "league": lg, "season": str(x["league"]["season"]),
                     "date": x["fixture"]["date"][:10], "home_id": x["teams"]["home"]["id"],
                     "home_name": x["teams"]["home"]["name"], "away_id": x["teams"]["away"]["id"],
                     "away_name": x["teams"]["away"]["name"], "ft_home": x["score"]["fulltime"]["home"],
                     "ft_away": x["score"]["fulltime"]["away"], "status": x["fixture"]["status"]["short"]})
    return pd.DataFrame(rows)


def pool_with_probes() -> pd.DataFrame:
    stored = FC.provider_pool()
    extra = probe_pool()
    if extra.empty:
        return stored
    ht = pd.concat([extra, stored.drop(columns=["_d"])], ignore_index=True)
    return FC.provider_pool(ht=ht, comp=pd.DataFrame())


def served_conflict_fixtures(fd_path: Path, pool: pd.DataFrame) -> set:
    """Fixture ids whose SERVED row (data truth rule 5's frame, the conflict list NOT applied)
    disagrees with the provider - a raw copy the loader drops anyway is not a served conflict."""
    from unittest import mock
    from src import data_loader as DL
    from src import league_registry as LR
    with mock.patch.object(HC, "read_score_conflicts", lambda path=None: pd.DataFrame()):
        served = DL.one_club_one_match(raw_frame(fd_path).sort_values("date", kind="mergesort")
                                       .reset_index(drop=True))
    served = served.copy()
    served["league"] = served["league"].map(lambda s: LR.canonical_name(s) if isinstance(s, str) else None)
    return set(conflicts(served, pool)["fixture_id"].astype(int))


def build(fd_path: Path) -> pd.DataFrame:
    from src import league_registry as LR
    pool = pool_with_probes()
    served_ids = served_conflict_fixtures(fd_path, pool)
    raw = raw_frame(fd_path)
    # the history lists first (a removed copy or a 90-minute score is not a conflict); the identity
    # pipeline's one-match step is not applied: every raw row that disagrees is listed by its own key
    raw = HC.apply_history_corrections(raw, tag="conflicts", conflicts=pd.DataFrame())
    raw = raw.reset_index(drop=True)
    x = raw.copy()
    x["league"] = x["league"].map(lambda s: LR.canonical_name(s) if isinstance(s, str) else None)
    c = conflicts(x, pool)
    c = c[c["fixture_id"].astype(int).isin(served_ids)]
    rows = []
    for r in c.itertuples(index=False):
        fid = int(r.fixture_id)
        pf = PROBES / f"fixture_probe_{fid}.json"
        probe = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
        evs = events_score(probe) if probe else None
        hist, prov = (int(r.home_goals), int(r.away_goals)), (int(r.ft_home), int(r.ft_away))
        dec = decide(hist, prov, evs[:2] if evs else None)
        served = prov if dec == "serve_provider_90min" else hist if dec == "keep_history" else (None, None)
        fx = ((next((k for k in probe.get("calls", []) if k["name"] == "fixture"), {}) or {}).get("response")
              or [{}])[0]
        goals = "; ".join(f"{g['time']['elapsed']}' {g['team']['name']} {g['detail']}" for g in (evs[2] if evs else []))
        raw_row = raw.loc[r.idx]
        rows.append({"league": raw_row["league"], "date": pd.Timestamp(raw_row["date"]).strftime("%Y-%m-%d"),
                     "home_team": raw_row["home_team"], "away_team": raw_row["away_team"],
                     "row_source": r.row_source, "history_ft_home": hist[0], "history_ft_away": hist[1],
                     "decision": dec, "served_ft_home": served[0], "served_ft_away": served[1], "fixture_id": fid,
                     "provider_ft_home": prov[0], "provider_ft_away": prov[1],
                     "events_ft_home": evs[0] if evs else None, "events_ft_away": evs[1] if evs else None,
                     "provider_home": r.home_name, "provider_away": r.away_name,
                     "round": (fx.get("league") or {}).get("round"),
                     "evidence": (f"API-Football /fixtures/events ({probe.get('generated_at')}): goals {goals or 'none'}"
                                  if evs else "no probe answer stored")})
    out = pd.DataFrame(rows, columns=COLS)
    # one row per raw key (an fd copy and an archive copy of one match share league, date and clubs)
    out = out.drop_duplicates(HC.KEY + ["history_ft_home", "history_ft_away"])
    return out.sort_values(["league", "date"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-history", default=str(ROOT / "output" / "fd_history.parquet"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    out = build(Path(a.fd_history))
    print(out[HC.KEY + ["history_ft_home", "history_ft_away", "provider_ft_home", "provider_ft_away",
                        "events_ft_home", "events_ft_away", "decision"]].to_string())
    if not a.dry_run:
        out.to_csv(HC.SCORE_CONFLICTS, index=False, lineterminator="\n")
        print(f"wrote {HC.SCORE_CONFLICTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
