"""
build_history_supplement.py - the provider fixtures of FINISHED seasons that the served history lacks
(owner 2026-11-10), written beside the banks for src/history_supplement.py. Stored provider data, 0 calls.

    python scripts/build_history_supplement.py            # write the supplement, the refusals, the audit
    python scripts/build_history_supplement.py --dry-run  # print the counts; write nothing

THE SERVED HISTORY: fd_history + the backtest archive through src.data_loader.one_club_one_match (data
truth rule 5's frame), WITHOUT the supplement. Every served row claims at most one API-Football fixture
(src.integrity.fixture_claims). THE PROVIDER LIST: the stored finished fixtures (FT / AET / PEN) of
output/collect/af_ht_scores.parquet + output/strength/competition_fixtures.parquet.

A provider fixture is ADDED when all hold:
  * its (league, provider season) is FINISHED (a later season of the league is stored) and the served
    history holds that season (at least one served row claims a fixture of it);
  * no served row claims it;
  * it is not dated before the first served fixture of that league-season (the served WINDOW starts
    there - extending it backwards is a different decision; counted as 'window_edge', not added);
  * no owner correction list names its fixture id (e.g. Queretaro v Atlas 816403: ruled not played);
  * its provider round label (output/collect/supplement_fixture_rounds.json, scripts/
    probe_supplement_rounds.py) is a LEAGUE round - regular season or a round-robin phase (owner
    2026-11-11: cup and knockout ties are not league matches; Argentina's cup rows were excluded on
    2026-11-04); a fixture without a stored label is refused;
  * no served row of the league within +-1 day is UNCLAIMED (it could be this match under names the
    resolver cannot place - never risk a second copy), and no served row within +-1 day claims a fixture
    of either of its clubs (a club does not play twice in a day: the served row would be this match
    filed under another date);
  * the provider's record holds a half-time score (a finished fixture without one is how API-Football
    files an AWARDED result as 'FT': Turkey 2022/23 Hatayspor / Gaziantep, Serie B Salernitana v
    Reggiana 2020-10-31);
  * both clubs have ONE served spelling: the spelling the served rows use for that provider team id in
    that league (by the fixtures they claim; the same season first, else the league's history), else the
    club's one served spelling in another league (a promoted club in a play-off) when no other club of
    this league uses it, or, for a club the served history never had, the provider's name when no served
    club already uses it.
Values: the provider's date, score.fulltime (the 90-minute score, also for AET / PEN), score.halftime
where the HT store holds it; nothing else (statistics come from the provider's af_history rows through
the loader's join, or stay missing). The season label is the served label of that league-season.

OUTPUT (no bank is rewritten):
  output/corrections/history_provider_supplement.csv          the added rows + the provider evidence
  output/corrections/history_provider_supplement_refused.csv  every candidate NOT added + reason
  output/corrections/history_provider_supplement_audit.json   counts per league / reason
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import history_supplement as HS  # noqa: E402
from src.integrity import fixture_claims as FC  # noqa: E402

OUT_DIR = ROOT / "output" / "corrections"
REFUSED = OUT_DIR / "history_provider_supplement_refused.csv"
AUDIT = OUT_DIR / "history_provider_supplement_audit.json"
REFUSED_COLS = ["league", "provider_season", "provider_date", "provider_fixture_id", "provider_home",
                "provider_away", "provider_status", "ft_home", "ft_away", "reason"]


def served_frame() -> pd.DataFrame:
    """Data truth rule 5's served frame, without the supplement."""
    from src import data_loader as DL
    from src.integrity import data_truth as DT
    parts = [DT._bank_matches("fd_history.parquet").assign(row_source="fd_cache"),
             DL._drop_misfiled_league_rows(DT._bank_matches("backtest_all_leagues.csv"), "supplement")
             .assign(row_source="backtest_archive")]
    merged = pd.concat(parts, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)
    return DL.one_club_one_match(merged)


def corrected_fixture_ids() -> set:
    """Every provider fixture id an owner correction list names (any '*fixture_id*' column)."""
    ids = set()
    for f in sorted(glob.glob(str(OUT_DIR / "*.csv"))):
        if Path(f).name.startswith("history_provider_supplement"):
            continue
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:                                              # noqa: BLE001
            continue
        for c in d.columns:
            if "fixture_id" in c:
                ids |= set(pd.to_numeric(d[c], errors="coerce").dropna().astype(int))
    return ids


#: the provider's round label of every fixture the supplement considered (scripts/probe_supplement_rounds.py)
ROUNDS = ROOT / "output" / "collect" / "supplement_fixture_rounds.json"
#: LEAGUE rounds (owner 2026-11-11): the regular season and the round-robin phases that follow it. Every
#: other label is a cup or knockout tie (finals, semi-finals, play-off ties, play-ins, placement games,
#: 'Relegation Round' / 'Relegation/Promotion' two-legged ties, the Argentine Copa de la Liga phases)
LEAGUE_ROUND = re.compile(r"^(Regular Season|Championship Round|Relegation Group|Conference League Play-off Group)\b")


def read_rounds(path: Path = ROUNDS) -> dict:
    """{fixture id: provider round label} ({} when the probe answer is absent)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {int(k): v.get("round") for k, v in (doc.get("fixtures") or {}).items()}


def round_reason(label) -> str | None:
    """None for a league round; else why the fixture is not a league match."""
    if not isinstance(label, str) or not label.strip():
        return "round label unknown (no provider answer): not proven a league match"
    if LEAGUE_ROUND.match(label.strip()):
        return None
    return f"cup / knockout round '{label.strip()}' (owner 2026-11-11: league rounds only)"


def build(served: pd.DataFrame, pool: pd.DataFrame, ht: pd.DataFrame, corrected: set,
          rounds: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`rounds` {fixture id: provider round label}: None skips the round rule (tests of the other rules)."""
    from src import league_registry as LR
    s = served.copy()
    s["league"] = s["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    c = FC.claims(s, pool)
    pool = pool.copy()
    pool["season"] = pool["season"].astype(str)
    pool["d"] = pd.to_datetime(pool["_d"])
    pool["pdate"] = pool["_d"]
    pinfo = pool.set_index("fixture_id")
    c = c.join(pinfo[["season", "home_id", "away_id", "d"]], on="fixture_id")
    c = c.join(s[["home_team", "away_team", "season"]].rename(columns={"season": "served_season"}), on="idx")
    claimed = set(c["fixture_id"])
    newest = pool.groupby("league")["season"].max()
    first = c.groupby(["league", "season"])["d"].min()
    # served spelling of a provider team id: per (league, season) and per league
    sp_season, sp_league, sp_any = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for r in c.itertuples(index=False):
        for tid, name in ((r.home_id, r.home_team), (r.away_id, r.away_team)):
            if pd.notna(tid):
                sp_season[(r.league, r.season, int(tid))][name] += 1
                sp_league[(r.league, int(tid))][name] += 1
                sp_any[int(tid)][name] += 1
    served_names = defaultdict(set)
    for (lg, tid), cnt in sp_league.items():
        for n in cnt:
            served_names[(lg, n)].add(tid)
    season_label = {k: g.mode().iloc[0] for k, g in c.groupby(["league", "season"])["served_season"]}
    # served rows per league-day: claimed clubs and unclaimed rows
    s["_d"] = pd.to_datetime(s["date"]).dt.normalize()
    unclaimed_days = defaultdict(int)
    for lg, d in s.loc[~s.index.isin(set(c["idx"])), ["league", "_d"]].itertuples(index=False):
        unclaimed_days[(lg, d)] += 1
    clubs_days = defaultdict(set)
    for r in c.itertuples(index=False):
        for tid in (r.home_id, r.away_id):
            if pd.notna(tid):
                clubs_days[(r.league, r.d)].add(int(tid))
    hts = ht.dropna(subset=["fixture_id"]).drop_duplicates("fixture_id").set_index("fixture_id")
    added, refused = [], []
    stats = defaultdict(Counter)
    for f in pool[~pool["fixture_id"].isin(claimed)].itertuples(index=False):
        lg, season, fid = f.league, f.season, int(f.fixture_id)
        if (lg, season) not in first.index:
            continue                                   # a season the served history does not hold
        if season == newest.get(lg):
            continue                                   # not finished
        ev = {"league": lg, "provider_season": season, "provider_date": f.pdate,
              "provider_fixture_id": fid, "provider_home": f.home_name, "provider_away": f.away_name,
              "provider_status": f.status, "ft_home": f.ft_home, "ft_away": f.ft_away}

        def refuse(why):
            refused.append({**ev, "reason": why})
            stats[lg][why] += 1
        if f.d < first[(lg, season)]:
            refuse("window_edge: before the first served fixture of this league-season")
            continue
        if fid in corrected:
            refuse("named by an owner correction list")
            continue
        if pd.isna(f.ft_home) or pd.isna(f.ft_away):
            refuse("no provider score.fulltime")
            continue
        near = [f.d + pd.Timedelta(days=k) for k in (-1, 0, 1)]
        if any(unclaimed_days.get((lg, d), 0) for d in near):
            refuse("an unclaimed served row of the league within 1 day (could be this match)")
            continue
        ids = {int(x) for x in (f.home_id, f.away_id) if pd.notna(x)}
        if len(ids) < 2:
            refuse("provider team id missing")
            continue
        if any(ids & clubs_days.get((lg, d), set()) for d in near):
            refuse("a club of this fixture already has a served match within 1 day")
            continue
        names, rules = [], []
        for tid, pname in ((int(f.home_id), f.home_name), (int(f.away_id), f.away_name)):
            cnt = sp_season.get((lg, season, tid)) or sp_league.get((lg, tid))
            if cnt and len(cnt) == 1:
                names.append(next(iter(cnt)))
                rules.append("served_spelling_same_season" if sp_season.get((lg, season, tid)) else
                             "served_spelling_league")
            elif cnt:
                names.append(None)
                rules.append("served_spellings_ambiguous:" + "|".join(sorted(cnt)))
            elif sp_any.get(tid) and len(sp_any[tid]) == 1 and not served_names.get((lg, next(iter(sp_any[tid])))):
                names.append(next(iter(sp_any[tid])))        # the club's served spelling in another league
                rules.append("served_spelling_other_league")
            elif sp_any.get(tid):
                names.append(None)
                rules.append("served_spellings_ambiguous_across_leagues:" + "|".join(sorted(sp_any[tid])))
            elif served_names.get((lg, pname)):
                names.append(None)
                rules.append("provider_name_is_another_served_club")
            else:
                names.append(pname)
                rules.append("provider_name_new_club")
        if None in names:
            refuse("club spelling not proven (" + "; ".join(r for r in rules if r.startswith(("served_spellings_",
                                                                                          "provider_name_is_"))) + ")")
            continue
        h = hts.loc[fid] if fid in hts.index else None
        hth = h["ht_home"] if h is not None and pd.notna(h.get("ht_home")) else None
        hta = h["ht_away"] if h is not None and pd.notna(h.get("ht_away")) else None
        if h is not None and (hth is None or hta is None):
            # the provider's record of a finished fixture with NO half-time score is how it files an
            # AWARDED result as 'FT' (Turkey 2022/23 Hatayspor / Gaziantep after the earthquake, Serie B
            # Salernitana v Reggiana 2020-10-31: all 3-0 / 0-3): not proven played - never added
            refuse("provider record holds no half-time score (an awarded result filed FT)")
            continue
        if rounds is not None:
            why = round_reason(rounds.get(fid))
            if why:
                refuse(why)
                continue
        gap = "after_last_served" if f.d > c.loc[(c["league"] == lg) & (c["season"] == season), "d"].max() else "inside"
        added.append({"date": ev["provider_date"], "league": lg,
                      "season": season_label.get((lg, season), season),
                      "home_team": names[0], "away_team": names[1],
                      "home_goals": float(f.ft_home), "away_goals": float(f.ft_away),
                      "ht_home_goals": hth, "ht_away_goals": hta,
                      "provider_fixture_id": fid, "provider_date": ev["provider_date"],
                      "provider_home": f.home_name, "provider_away": f.away_name,
                      "provider_home_id": int(f.home_id), "provider_away_id": int(f.away_id),
                      "provider_status": f.status, "provider_season": season,
                      "home_name_rule": rules[0], "away_name_rule": rules[1], "gap": gap})
        stats[lg]["added:" + gap] += 1
    add = pd.DataFrame(added, columns=HS.MATCH_COLS + HS.EVIDENCE_COLS)
    ref = pd.DataFrame(refused, columns=REFUSED_COLS)
    add = add.sort_values(["league", "date", "provider_fixture_id"]).reset_index(drop=True)
    ref = ref.sort_values(["league", "provider_date", "provider_fixture_id"]).reset_index(drop=True)
    return add, ref, {k: dict(v) for k, v in sorted(stats.items())}


def run(dry_run: bool = False) -> dict:
    logging.disable(logging.WARNING)
    served = served_frame()
    pool = FC.provider_pool()
    ht = pd.read_parquet(FC.HT_SCORES, columns=["fixture_id", "ht_home", "ht_away"])
    add, ref, per = build(served, pool, ht, corrected_fixture_ids(), read_rounds())
    audit = {"artifact": "history_provider_supplement_audit",
             "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "served_rows_without_supplement": int(len(served)), "added": int(len(add)),
             "refused": int(len(ref)),
             "added_by_status": {k: int(v) for k, v in add["provider_status"].value_counts().items()},
             "added_with_ht": int(add["ht_home_goals"].notna().sum()),
             "refused_by_reason": {k: int(v) for k, v in
                                   ref["reason"].str.split(r" \(| '", regex=True).str[0].value_counts().items()},
             "per_league": per}
    if not dry_run:
        add.to_csv(HS.SUPPLEMENT, index=False, lineterminator="\n")
        ref.to_csv(REFUSED, index=False, lineterminator="\n")
        AUDIT.write_text(json.dumps(audit, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return audit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.dry_run), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
