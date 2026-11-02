"""
Real results for the club-strength rating, one row per match.

SOURCES (only results we hold; nothing typed):
  output/strength/competition_fixtures.parquet  API-Football /fixtures?league&season, collected by
      scripts/collect_competition_fixtures.py for every registry club league AND every registry
      club cup (UEFA Champions / Europa / Conference League), plus the continental cups listed in
      src/strength/competitions.py (Libertadores, Sudamericana, AFC / CONCACAF Champions League,
      Club World Cup; competition key 'af_cup:<id>'). Rows carry API-Football team ids and the
      90-minute score.
  output/af_history.parquet   API-Football results of the new-format + standard leagues (names).
  output/fd_history.parquet   football-data.co.uk results (football-data spellings).

PRECEDENCE. Inside the date span of a collected (league, season) the league is read ONLY from the
collected file (ids, full season; spans, not season labels, because the banks label seasons
differently); af_history then fd_history fill the rest. The same pairing on dates
within DEDUP_DAYS of each other is one match reported by two sources (their kickoff dates differ by
time zone), and the higher-precedence row is kept.

IDENTITY. A club is 'af:<API-Football team id>' whenever a collected fixture names it, so a club is
the same entity in its domestic league and in a European / continental cup. A name-only row (af /
fd history) is resolved with src.team_names.resolve against the names of the SAME identity scope
(the league's country, from the registry); a football-data spelling that does not resolve to an
API-Football name of its scope is dropped and counted (never guessed), because two keys for one club
would split its rating.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COLLECTED = ROOT / "output" / "strength" / "competition_fixtures.parquet"
AF_HISTORY = ROOT / "output" / "af_history.parquet"
FD_HISTORY = ROOT / "output" / "fd_history.parquet"

# Two rows of the same ordered pairing this close together are one match seen by two sources.
DEDUP_DAYS = 2

COLUMNS = ["date", "competition", "season", "source", "home_key", "away_key",
           "home_goals", "away_goals", "home_scope", "away_scope"]


def _lr():
    from src import league_registry
    return league_registry


def competition_type(comp) -> str | None:
    """'club_league' / 'club_cup' / 'national_teams' for a registry league or a collected
    non-registry continental cup (src/strength/competitions.py); None when unknown."""
    if not isinstance(comp, str):
        return None
    from src.strength.competitions import CUP_KEYS
    if comp in CUP_KEYS:
        return "club_cup"
    return _lr().competition_type(comp)


def scope_of(league) -> str | None:
    """Identity scope of a CLUB LEAGUE: 'club_league:<country>' from the registry; None for a cup,
    national-team competition or a league the registry does not know."""
    lr = _lr()
    if not isinstance(league, str) or competition_type(league) != "club_league":
        return None
    country = (lr.entry(league) or {}).get("country")
    return f"club_league:{country}" if country else None


class ClubIndex:
    """name -> club key per identity scope. Built from the collected id-named fixtures first, then
    from name-only API-Football spellings that no id name resolves to."""

    def __init__(self):
        self.names: dict[str, dict[str, str]] = {}      # scope -> {name: key}
        self.home_scope: dict[str, str] = {}             # key -> scope (domestic league country)
        self._cache: dict = {}

    def add(self, scope: str, name: str, key: str) -> None:
        self.names.setdefault(scope, {}).setdefault(name, key)
        self.home_scope.setdefault(key, scope)
        self._cache.clear()

    def resolve(self, name, scope) -> str | None:
        """The club key of `name` in `scope` (src.team_names.resolve, ambiguity refused), or None."""
        if not isinstance(name, str) or not name or scope is None:
            return None
        k = (scope, name)
        if k not in self._cache:
            pool = self.names.get(scope) or {}
            if name in pool:
                self._cache[k] = pool[name]
            else:
                from src.team_names import resolve
                hit = resolve(name, list(pool)) if pool else None
                self._cache[k] = pool[hit] if hit else None
        return self._cache[k]

    def resolve_any_scope(self, name) -> str | None:
        """A club named outside any domestic scope (a cup row with no id): accepted only when
        exactly ONE scope resolves it."""
        hits = {self.resolve(name, s) for s in self.names}
        hits.discard(None)
        return hits.pop() if len(hits) == 1 else None


def _read(path: Path, columns=None) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as e:                                               # noqa: BLE001
        print(f"[strength] {Path(path).name} unreadable ({e}): not used")
        return pd.DataFrame()


def _canon(league):
    return _lr().canonical_name(league) if isinstance(league, str) else None


def load_results(collected: Path | None = None, af: Path | None = None, fd: Path | None = None,
                 verbose: bool = True) -> tuple[pd.DataFrame, ClubIndex, dict]:
    """(matches sorted by date, club index, report). Each match: date (day), competition (registry
    canonical name), season, source, home_key, away_key, home_goals, away_goals, home_scope /
    away_scope (the club's DOMESTIC scope when known, else None)."""
    collected = COLLECTED if collected is None else collected
    af = AF_HISTORY if af is None else af
    fd = FD_HISTORY if fd is None else fd
    report = {"dropped_unresolved": {}, "dropped_unknown_league": 0, "dropped_no_score": 0,
              "duplicates": 0}
    idx = ClubIndex()
    parts = []

    # 1. collected fixtures (ids). Domestic rows teach the index which id plays in which scope.
    c = _read(collected)
    covered: dict = {}                     # competition -> [(first day, last day)] collected
    if not c.empty:
        c = c.copy()
        from src.strength.competitions import cup_key
        reg = c["league"].map(_canon) if "league" in c.columns else pd.Series(None, index=c.index)
        cup = c["league_id"].map(cup_key) if "league_id" in c.columns else pd.Series(None, index=c.index)
        c["competition"] = reg.where(reg.notna(), cup)
        report["dropped_unknown_league"] += int(c["competition"].isna().sum())
        c = c[c["competition"].notna()]
        c["scope"] = c["competition"].map(scope_of)
        dom = c[c["scope"].notna()].sort_values("date")
        for side in ("home", "away"):
            for sc, tid, nm in zip(dom["scope"], dom[f"{side}_id"], dom[f"{side}_name"]):
                if tid == tid and nm == nm:
                    idx.add(sc, str(nm), f"af:{int(tid)}")
        ok = c["ft_home"].notna() & c["ft_away"].notna() & c["home_id"].notna() & c["away_id"].notna()
        report["dropped_no_score"] += int((~ok).sum())
        c = c[ok]
        # Coverage by DATE SPAN per competition-season: the history banks label seasons
        # differently ('2023/24' vs API-Football '2023'), so a season label cannot be compared.
        _d = pd.to_datetime(c["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        for (cp, _s), span in _d.groupby([c["competition"], c["season"].astype(str)]):
            if span.notna().any():
                covered.setdefault(cp, []).append((span.min(), span.max()))
        parts.append(pd.DataFrame({
            "date": pd.to_datetime(c["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize(),
            "competition": c["competition"], "season": c["season"].astype(str), "source": "collected",
            "home_key": "af:" + c["home_id"].astype("int64").astype(str),
            "away_key": "af:" + c["away_id"].astype("int64").astype(str),
            "home_goals": pd.to_numeric(c["ft_home"], errors="coerce"),
            "away_goals": pd.to_numeric(c["ft_away"], errors="coerce")}))

    # 2. af_history: API-Football spellings. A name the id index does not know becomes its own key.
    a = _read(af)
    rows = []
    if not a.empty:
        a = a.assign(competition=a["league"].map(_canon))
        report["dropped_unknown_league"] += int(a["competition"].isna().sum())
        a = a[a["competition"].notna()].copy()
        a["scope"] = a["competition"].map(scope_of)
        a = a[a["scope"].notna()]
        # Rows written since 2026-11-01 carry API-Football team ids: they teach the index too.
        for side in ("home", "away"):
            if f"{side}_team_id" in a.columns:
                for sc, tid, nm in zip(a["scope"], a[f"{side}_team_id"], a[f"{side}_team"]):
                    if tid == tid and tid is not None and nm == nm:
                        idx.add(sc, str(nm), f"af:{int(tid)}")
        for sc, sub in a.groupby("scope"):
            for nm in pd.unique(pd.concat([sub["home_team"], sub["away_team"]]).dropna()):
                if idx.resolve(str(nm), sc) is None:
                    idx.add(sc, str(nm), f"{sc}|{nm}")
        rows.append(("af", a, "FTHG", "FTAG"))

    # 3. fd_history: football-data spellings, resolved onto the API names of the same scope only.
    f = _read(fd)
    if not f.empty:
        f = f.assign(competition=f["league"].map(_canon))
        report["dropped_unknown_league"] += int(f["competition"].isna().sum())
        f = f[f["competition"].notna()].copy()
        f["scope"] = f["competition"].map(scope_of)
        f = f[f["scope"].notna()]
        rows.append(("fd", f, "home_goals", "away_goals"))

    for src_name, d, hg, ag in rows:
        _dd = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        inside = pd.Series(False, index=d.index)
        for cp, spans in covered.items():
            m_cp = d["competition"] == cp          # registry canonical names on both sides
            for lo, hi in spans:
                inside |= m_cp & (_dd >= lo) & (_dd <= hi)
        report["covered_by_collected"] = report.get("covered_by_collected", 0) + int(inside.sum())
        d = d[~inside]
        ok = d[hg].notna() & d[ag].notna()
        report["dropped_no_score"] += int((~ok).sum())
        d = d[ok]
        hk = [idx.resolve(str(n), s) for n, s in zip(d["home_team"], d["scope"])]
        ak = [idx.resolve(str(n), s) for n, s in zip(d["away_team"], d["scope"])]
        out = pd.DataFrame({
            "date": pd.to_datetime(d["date"], errors="coerce").dt.normalize(),
            "competition": d["competition"], "season": d["season"].astype(str), "source": src_name,
            "home_key": hk, "away_key": ak,
            "home_goals": pd.to_numeric(d[hg], errors="coerce"),
            "away_goals": pd.to_numeric(d[ag], errors="coerce")}, index=d.index)
        bad = out["home_key"].isna() | out["away_key"].isna()
        if bad.any():
            un = pd.concat([d.loc[out["home_key"].isna(), "home_team"],
                            d.loc[out["away_key"].isna(), "away_team"]])
            report["dropped_unresolved"][src_name] = {
                "matches": int(bad.sum()), "names": sorted(un.astype(str).unique())[:40]}
        parts.append(out[~bad])

    if not parts:
        return pd.DataFrame(columns=COLUMNS), idx, report
    m = pd.concat(parts, ignore_index=True)
    m["date"] = pd.to_datetime(m["date"], errors="coerce").astype("datetime64[ns]")
    m = m[m["date"].notna()]
    m = m[m["home_key"] != m["away_key"]]
    # Cup clubs named only by id with no domestic row keep the id key and an unknown home scope.
    m["home_scope"] = m["home_key"].map(idx.home_scope)
    m["away_scope"] = m["away_key"].map(idx.home_scope)

    # De-duplicate the same match reported by several sources (precedence collected > af > fd).
    prio = m["source"].map({"collected": 0, "af": 1, "fd": 2})
    m = m.assign(_p=prio).sort_values(["home_key", "away_key", "date", "_p"]).reset_index(drop=True)
    keep = np.ones(len(m), dtype=bool)
    hk, ak, dt, pr = m["home_key"].to_numpy(), m["away_key"].to_numpy(), m["date"].to_numpy(), m["_p"].to_numpy()
    last_kept = {}
    for i in range(len(m)):
        pair = (hk[i], ak[i])
        j = last_kept.get(pair)
        if j is not None and (dt[i] - dt[j]) <= np.timedelta64(DEDUP_DAYS, "D"):
            keep[i] = False
            if pr[i] < pr[j]:                     # a better source for the same match: swap
                keep[i], keep[j] = True, False
                last_kept[pair] = i
            continue
        last_kept[pair] = i
    report["duplicates"] = int((~keep).sum())
    m = m[keep].drop(columns="_p")
    m = m.sort_values(["date", "competition", "home_key"]).reset_index(drop=True)
    report["matches"] = int(len(m))
    report["by_source"] = {k: int(v) for k, v in m["source"].value_counts().items()}
    if verbose:
        print(f"[strength] {len(m):,} matches ({report['by_source']}); duplicates dropped "
              f"{report['duplicates']:,}; unresolved names dropped "
              f"{ {k: v['matches'] for k, v in report['dropped_unresolved'].items()} }")
    return m[COLUMNS], idx, report
