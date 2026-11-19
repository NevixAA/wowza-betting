"""
verify_fd_raw_seasons.py - provenance + API-Football cross-check of every banked football-data season row.
==========================================================================================================
fix/fd-ht-and-history-backfill (2026-11-18). src/data_loader.read_fd_raw_seasons serves the banked
football-data season files (output/collect/fd_raw/ and fd_raw_history/) as a history source. Before a
row may reach training its score is checked against the OTHER paid source, API-Football:

  * every row of every banked file -> one provenance row: the file, its download URL and sha256 (from
    the directory's manifest.json), the row's league / season / date / clubs / FT and HT score;
  * the row is paired with ONE API-Football fixture of output/collect/af_ht_scores.parquet by the HT
    collector's own rule (scripts/collect_ht_scores.claims: same league, clubs resolved home->home and
    away->away, same day else adjacent day, ambiguous -> refused). No call, no credit;
  * ft_check: `equal` (FT agrees), `mismatch` (FT differs -> the loader does NOT serve the row),
    `no_af_fixture` (API-Football holds no fixture for it - e.g. seasons before 2018/19, which the
    stored fixture lists do not cover; the row is served on football-data's word alone, labelled),
    `provider_ft_missing`, `refused_claimedclaimed_twice` (one fixture claimed by rows of different dates);
  * `provider_not_played` - no finished fixture, but the row IS a provider fixture marked awarded
    (AWD / WO) or abandoned (ABD): not a played score (owner, fix9), so the loader does NOT serve it;
  * ht_check: the same for the half-time score (`equal` / `mismatch` / `n/a`). An HT mismatch is
    REPORTED, never resolved by picking a side here.

OUTPUT (beside the files; nothing else is rewritten):
  output/collect/fd_raw_provenance.parquet   one row per banked football-data row (see above)
  output/collect/fd_raw_verify_summary.json  per league x season: rows, ft_check / ht_check counts

    python scripts/verify_fd_raw_seasons.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = ROOT / "output" / "collect" / "fd_raw_provenance.parquet"
SUMMARY = ROOT / "output" / "collect" / "fd_raw_verify_summary.json"
SCORES = ROOT / "output" / "collect" / "af_ht_scores.parquet"


def raw_rows(root: Path = ROOT) -> pd.DataFrame:
    """Every banked row with its file provenance (same reader and file precedence as the loader)."""
    import re
    from io import StringIO
    from src import data_loader as DL
    std, _new = DL._fd_download_tables()
    frames, seen = [], set()
    for rel in DL.FD_RAW_DIRS:
        d = root / rel
        if not d.is_dir():
            continue
        man_p = d / "manifest.json"
        man = json.loads(man_p.read_text(encoding="utf-8")).get("files", {}) if man_p.exists() else {}
        for league, code in sorted(std.items()):
            for p in sorted(d.glob(f"{code}_*.csv")):
                m = re.match(rf"^{re.escape(code)}_(\d{{2}})(\d{{2}})\.csv$", p.name)
                if not m or p.name in seen:
                    continue
                seen.add(p.name)
                raw = pd.read_csv(StringIO(DL._decode_csv(p.read_bytes())), on_bad_lines="skip",
                                  low_memory=False)
                f = DL.parse_fd_season_file(raw, league, f"20{m.group(1)}/{m.group(2)}")
                info = man.get(p.name, {})
                frames.append(f.assign(fd_file=p.name, fd_dir=rel.as_posix(), url=info.get("url"),
                                       sha256=info.get("sha256"), fetched_at=info.get("fetched_at")))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pair(a, b):
    return None if pd.isna(a) or pd.isna(b) else (int(a), int(b))


def verify(rows: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    import collect_ht_scores as HT
    lack = rows.assign(canon=rows["league"])[["canon", "league", "date", "home_team", "away_team",
                                               "home_goals", "away_goals"]].drop_duplicates()
    cl = HT.claims(lack, scores, require_ht=False) if len(scores) else pd.DataFrame()
    if len(cl):
        n_dates = cl.groupby("fixture_id")["date"].transform("nunique")
        cl = cl.assign(claimed_twice=n_dates > 1)
        cl = cl.drop_duplicates(["league", "date", "home_team", "away_team"])
        cl = cl[["league", "date", "home_team", "away_team", "fixture_id", "af_date", "match_rule",
                 "provider_ft_home", "provider_ft_away", "HTHG", "HTAG", "claimed_twice"]] \
            .rename(columns={"fixture_id": "af_fixture_id", "HTHG": "provider_ht_home",
                             "HTAG": "provider_ht_away"})
        out = rows.merge(cl, on=["league", "date", "home_team", "away_team"], how="left")
    else:
        out = rows.copy()
        for c in ["af_fixture_id", "af_date", "match_rule", "provider_ft_home", "provider_ft_away",
                  "provider_ht_home", "provider_ht_away", "claimed_twice"]:
            out[c] = None
    # NOT PLAYED: a row with no finished fixture that IS a provider fixture the provider marks awarded
    # (AWD / WO) or abandoned (ABD) - "awarded results are not real played scores" (owner, fix9).
    np_ids: dict = {}
    if len(scores) and "status" in scores.columns:
        rest = out[out["af_fixture_id"].isna()]
        lack2 = rest.assign(canon=rest["league"])[["canon", "league", "date", "home_team", "away_team",
                                                   "home_goals", "away_goals"]].drop_duplicates()
        cl2 = HT.claims(lack2, scores, require_ht=False, statuses=HT.NOT_PLAYED) if len(lack2) else pd.DataFrame()
        for r in cl2.itertuples(index=False):
            np_ids[(r.league, pd.Timestamp(r.date), r.home_team, r.away_team)] = (int(r.fixture_id), r.status)
    ft, ht = [], []
    for i, r in enumerate(out.itertuples(index=False)):
        hit = np_ids.get((r.league, pd.Timestamp(r.date), r.home_team, r.away_team))
        if pd.isna(r.af_fixture_id) and hit:
            out.iat[i, out.columns.get_loc("af_fixture_id")] = hit[0]
            ft.append("provider_not_played")
            ht.append("n/a")
            continue
        if pd.isna(r.af_fixture_id):
            ft.append("no_af_fixture")
            ht.append("n/a")
            continue
        if bool(r.claimed_twice):
            ft.append("refused_claimed_twice")
            ht.append("n/a")
            continue
        p, h = _pair(r.provider_ft_home, r.provider_ft_away), _pair(r.home_goals, r.away_goals)
        ft.append("provider_ft_missing" if p is None else "equal" if p == h else "mismatch")
        ph, hh = _pair(r.provider_ht_home, r.provider_ht_away), _pair(r.ht_home_goals, r.ht_away_goals)
        ht.append("n/a" if ph is None or hh is None else "equal" if ph == hh else "mismatch")
    out["ft_check"], out["ht_check"] = ft, ht
    out["verified_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["af_source"] = "api-football /fixtures (output/collect/af_ht_scores.parquet)"
    keep = ["fd_file", "fd_dir", "url", "sha256", "fetched_at", "league", "season", "date", "home_team",
            "away_team", "home_goals", "away_goals", "ht_home_goals", "ht_away_goals", "af_fixture_id",
            "af_date", "match_rule", "provider_ft_home", "provider_ft_away", "provider_ht_home",
            "provider_ht_away", "ft_check", "ht_check", "af_source", "verified_at"]
    out = out[keep]
    for c in ("af_fixture_id", "provider_ft_home", "provider_ft_away", "provider_ht_home",
              "provider_ht_away"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out["af_date"] = out["af_date"].astype(str).where(out["af_date"].notna(), None)
    return out


def summary(prov: pd.DataFrame) -> dict:
    s: dict = defaultdict(dict)
    for (lg, se), g in prov.groupby(["league", "season"]):
        s[lg][se] = {"rows": int(len(g)), "ft_check": {k: int(v) for k, v in g["ft_check"].value_counts().items()},
                     "ht_check": {k: int(v) for k, v in g["ht_check"].value_counts().items()},
                     "ht_present": int(g["ht_home_goals"].notna().sum())}
    tot = {"rows": int(len(prov)),
           "ft_check": {k: int(v) for k, v in prov["ft_check"].value_counts().items()},
           "ht_check": {k: int(v) for k, v in prov["ht_check"].value_counts().items()}}
    return {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rule": "football-data row paired with ONE API-Football fixture (collect_ht_scores.claims); "
                    "ft_check mismatch -> not served by src/data_loader.read_fd_raw_seasons",
            "total": tot, "per_league_season": dict(s)}


def run() -> dict:
    rows = raw_rows()
    scores = pd.read_parquet(SCORES) if SCORES.exists() else pd.DataFrame()
    prov = verify(rows, scores)
    prov.to_parquet(OUT, index=False)
    summ = summary(prov)
    SUMMARY.write_text(json.dumps(summ, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[fd_raw_verify] {summ['total']}")
    return summ


if __name__ == "__main__":
    run()
