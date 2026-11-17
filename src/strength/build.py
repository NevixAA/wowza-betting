"""
Build the measured club / league strength tables.

    python -m src.strength.build            # fit K/H, run, write output/strength/*
    python -m src.strength.build --no-fit   # reuse K/H from output/strength/elo_fit.json

Writes:
  output/strength/elo_fit.json          fitted K, H, held-out loss, every evaluated grid point,
                                        the results report (sources, duplicates, unresolved names)
  output/strength/club_ratings.parquet  one row per (club, match): rating AFTER it, matches played
  output/strength/club_names.parquet    (scope, name, club_key): how a club name maps to its key
  output/strength/league_strength.parquet
        one row per (league, match day): the mean rating BEFORE that day of the clubs that played
        in the league in the previous LEAGUE_WINDOW_DAYS with >= MIN_CLUB_MATCHES matches, the
        number of such clubs, whether the league's country is in the reference component (linked by
        >= MIN_LINK_MATCHES cross-league matches per country pair), and its rank among the linked
        leagues on that day. strength / rank are NaN unless linked and >= MIN_LEAGUE_CLUBS rated.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strength import elo, results  # noqa: E402

OUT_DIR = ROOT / "output" / "strength"


class _UF:
    def __init__(self):
        self.p: dict = {}
        self.size: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        self.size.setdefault(x, 1)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.size[ra] += self.size[rb]

    def reference(self, universe) -> object | None:
        """Root of the largest component over `universe`; None on a tie (no single reference)."""
        sizes: dict = {}
        for s in universe:
            r = self.find(s)
            sizes[r] = sizes.get(r, 0) + 1
        if not sizes:
            return None
        top = sorted(sizes.values(), reverse=True)
        if len(top) > 1 and top[0] == top[1]:
            return None
        return max(sizes, key=sizes.get)


def league_strength_table(m: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """See module docstring. `m` from results.load_results, `ratings` from elo.club_rating_table."""
    cols = ["league", "date", "strength", "n_rated", "n_members", "linked", "rank", "n_ranked"]
    m = m.assign(_scope=m["competition"].map(results.scope_of))
    dom = m[m["_scope"].notna()]
    if dom.empty:
        return pd.DataFrame(columns=cols)
    win = np.timedelta64(elo.LEAGUE_WINDOW_DAYS, "D")
    pairs = []
    for lg, sub in dom.groupby("competition", sort=False):
        long = pd.concat([sub[["date", "home_key"]].rename(columns={"home_key": "club"}),
                          sub[["date", "away_key"]].rename(columns={"away_key": "club"})]).sort_values("date")
        days = np.sort(long["date"].unique())
        ld, lc = long["date"].to_numpy(), long["club"].to_numpy()
        q: deque = deque()
        count: dict = {}
        j = 0
        for d in days:
            while j < len(ld) and ld[j] < d:            # matches strictly before the snapshot day
                q.append((ld[j], lc[j])); count[lc[j]] = count.get(lc[j], 0) + 1; j += 1
            while q and q[0][0] < d - win:
                _, c0 = q.popleft(); count[c0] -= 1
                if not count[c0]:
                    del count[c0]
            for club in count:
                pairs.append((lg, d, club))
    if not pairs:
        return pd.DataFrame(columns=cols)
    p = pd.DataFrame(pairs, columns=["league", "date", "club_key"]).sort_values("date")
    r = ratings[["club_key", "date", "rating", "n_matches"]].sort_values("date")
    p = pd.merge_asof(p, r, on="date", by="club_key", allow_exact_matches=False)
    p["rated"] = p["n_matches"] >= elo.MIN_CLUB_MATCHES
    g = p.groupby(["league", "date"])
    snap = pd.DataFrame({"n_members": g.size(),
                         "n_rated": g["rated"].sum(),
                         "mean": p[p["rated"]].groupby(["league", "date"])["rating"].mean()}).reset_index()
    snap["n_rated"] = snap["n_rated"].astype(int)

    # Linking: country pairs joined by enough real cross-league matches BEFORE the snapshot day.
    scope = {lg: results.scope_of(lg) for lg in snap["league"].unique()}
    universe = sorted({s for s in scope.values() if s})
    cross = m[m["home_scope"].notna() & m["away_scope"].notna() & (m["home_scope"] != m["away_scope"])]
    cross = cross.sort_values("date")
    cd = cross["date"].to_numpy()
    ch, ca = cross["home_scope"].to_numpy(), cross["away_scope"].to_numpy()
    uf, links, j = _UF(), {}, 0
    for s in universe:
        uf.find(s)
    snap = snap.sort_values("date").reset_index(drop=True)
    linked = np.zeros(len(snap), dtype=bool)
    ref, ref_day = None, None
    for i, (lg, d) in enumerate(zip(snap["league"], snap["date"])):
        if d != ref_day:
            while j < len(cd) and cd[j] < d:
                key = tuple(sorted((ch[j], ca[j])))
                links[key] = links.get(key, 0) + 1
                if links[key] == elo.MIN_LINK_MATCHES:
                    uf.find(key[0]); uf.find(key[1]); uf.union(*key)
                j += 1
            ref, ref_day = uf.reference(universe), d
            # one country alone is not a cross-league scale unless it is the whole universe
            if ref is not None and len(universe) > 1 and sum(uf.find(s) == ref for s in universe) < 2:
                ref = None
        s = scope.get(lg)
        linked[i] = bool(s) and ref is not None and uf.find(s) == ref
    snap["linked"] = linked
    snap["strength"] = snap["mean"].where(snap["linked"] & (snap["n_rated"] >= elo.MIN_LEAGUE_CLUBS))

    # Rank among the linked leagues on each snapshot day (each league's latest snapshot within
    # the window; 1 = strongest).
    piv = snap.pivot_table(index="date", columns="league", values="strength", aggfunc="last")
    piv = piv.reindex(sorted(snap["date"].unique()))
    last_seen = (snap.dropna(subset=["strength"]).assign(_seen=lambda x: x["date"])
                 .pivot_table(index="date", columns="league", values="_seen", aggfunc="last"))
    piv = piv.ffill()
    seen = last_seen.reindex(piv.index).ffill()
    stale = (pd.DataFrame(np.repeat(piv.index.to_numpy()[:, None], piv.shape[1], axis=1),
                          index=piv.index, columns=piv.columns)
             - seen.reindex(columns=piv.columns)) > pd.Timedelta(days=elo.LEAGUE_WINDOW_DAYS)
    piv = piv.mask(stale)
    rk = piv.rank(axis=1, ascending=False, method="min")
    nr = piv.notna().sum(axis=1)
    rank_long = rk.stack().rename("rank").reset_index()
    snap = snap.merge(rank_long, on=["date", "league"], how="left")
    snap["n_ranked"] = snap["date"].map(nr)
    snap["rank"] = snap["rank"].where(snap["strength"].notna())
    snap["n_ranked"] = snap["n_ranked"].where(snap["strength"].notna())
    return snap[cols].sort_values(["league", "date"]).reset_index(drop=True)


def build(fit_params: bool = True, out_dir: Path | None = None, verbose: bool = True,
          **paths) -> dict:
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    m, idx, report = results.load_results(verbose=verbose, **paths)
    fit_file = out_dir / "elo_fit.json"
    if fit_params:
        fitted = elo.fit(m, verbose=verbose)
    else:
        fitted = json.loads(fit_file.read_text(encoding="utf-8"))
    ratings = elo.club_rating_table(m, fitted["k"], fitted["h"])
    leagues = league_strength_table(m, ratings)
    names = pd.DataFrame([(s, n, k) for s, d in idx.names.items() for n, k in d.items()],
                         columns=["scope", "name", "club_key"])
    ratings.to_parquet(out_dir / "club_ratings.parquet", index=False)
    leagues.to_parquet(out_dir / "league_strength.parquet", index=False)
    names.to_parquet(out_dir / "club_names.parquet", index=False)
    fitted = {**fitted, "results_report": report,
              "min_league_clubs": elo.MIN_LEAGUE_CLUBS, "min_link_matches": elo.MIN_LINK_MATCHES,
              "league_window_days": elo.LEAGUE_WINDOW_DAYS,
              "leagues_with_strength": sorted(leagues.dropna(subset=["strength"])["league"].unique().tolist())}
    fit_file.write_text(json.dumps(fitted, indent=1, default=str), encoding="utf-8")
    if verbose:
        print(f"[strength] wrote {len(ratings):,} club-match ratings, {len(leagues):,} league snapshots "
              f"({len(fitted['leagues_with_strength'])} league(s) with a linked strength) -> {out_dir}")
    return fitted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-fit", action="store_true", help="reuse K/H from elo_fit.json")
    a = ap.parse_args(argv)
    build(fit_params=not a.no_fit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
