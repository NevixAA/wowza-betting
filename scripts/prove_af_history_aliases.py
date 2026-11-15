"""
Prove history-spelling -> API-Football spelling aliases from same-fixture evidence (P7 item 1).
=============================================================================================
6,464 af_history rows (and 2,684 rows in the HT fill) could not be matched to the API-Football
fixture they are, although the provider's fixture lists for every one of those league-seasons were
collected and the dates agree: the club names differ ("Dag and Red" / "Dagenham & Redbridge",
"Boston Utd" / "Boston United"). team_names.resolve_result cannot spell-match them.

THE PROOF (never fuzzy; the same standard as src.identity_repair.derive_same_day_aliases):
  * a history row (league L, date D, home h, away a) and the provider's finished fixtures of L's
    API-Football league on D;
  * FORCED-OPPONENT evidence: exactly one provider fixture on D has one side that resolve_result
    proves is the row's club on that side; that fixture's other side is then the row's other club ->
    evidence (row name -> provider name) for the unresolved side, on date D;
  * SCORE evidence: where fd_history holds the score of the row (same league, date, names), exactly
    one provider fixture on D has that full-time score -> evidence for both unresolved sides;
  * a pair (history name -> provider name) is PROVEN in league L only if it has evidence on at least
    MIN_DATES distinct dates, is one-to-one within L (no history name to two provider names, no
    provider name claimed by two history names), and never contradicts itself: on every date the
    history name plays in L and the provider has fixtures, the provider name plays too.
MIN_DATES = 3 is the identity_repair rule (derive_same_day_aliases(min_dates=3)), not a new number.

OUTPUT config/af_history_proven_aliases.json: LEAGUE-SCOPED (a global alias could rename a club in
another league, and the predict-time store _ALIASES is a model-input change, so neither is touched).
Every pair carries its evidence (dates, first/last, kinds). Read by the collectors' matchers only
(scripts/collect_af_history_gap.match_fixtures and scripts/collect_ht_scores), before resolve_result.

    python scripts/prove_af_history_aliases.py            # writes the file + prints the summary
    python scripts/prove_af_history_aliases.py --dry-run  # prints only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "config" / "af_history_proven_aliases.json"
FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
AF_HISTORY = ROOT / "output" / "af_history.parquet"
FD_HISTORY = ROOT / "output" / "fd_history.parquet"
AWARDED = ROOT / "output" / "corrections" / "history_awarded_results.csv"
#: the identity_repair evidence rule (derive_same_day_aliases min_dates=3)
MIN_DATES = 3
RULE = "af-alias-proof/1.0"


def _af_ids() -> dict:
    import config
    return {k: int(v) for k, v in getattr(config, "API_FOOTBALL_IDS", {}).items() if v is not None}


def _history() -> pd.DataFrame:
    """af_history and fd_history rows (league canonical name, date, names, score where known)."""
    from src import league_registry as LR
    parts = []
    af = pd.read_parquet(AF_HISTORY, columns=["league", "date", "home_team", "away_team"])
    parts.append(af.assign(hg=pd.NA, ag=pd.NA, src="af_history"))
    if FD_HISTORY.exists():
        fd = pd.read_parquet(FD_HISTORY, columns=["league", "date", "home_team", "away_team",
                                                  "home_goals", "away_goals"])
        parts.append(fd.rename(columns={"home_goals": "hg", "away_goals": "ag"}).assign(src="fd_history"))
    h = pd.concat(parts, ignore_index=True)
    h["d"] = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # A result the owner ruled NOT PLAYED (output/corrections/history_awarded_results.csv: awarded,
    # walkover, cancelled - e.g. China 2022-12-23, provider fixtures 927036 / 926930 CANC) is no match
    # of either club: neither bank's copy of that key is evidence, nor a date the club "played"
    # (fix/history-supplement, owner 2026-11-10). The rule itself is unchanged.
    if AWARDED.exists():
        aw = pd.read_csv(AWARDED)
        if {"league", "date", "home_team", "away_team"} <= set(aw.columns) and len(aw):
            gone = set(zip(aw["league"].astype(str), pd.to_datetime(aw["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
                           aw["home_team"].astype(str), aw["away_team"].astype(str)))
            k = list(zip(h["league"].astype(str), h["d"], h["home_team"].astype(str), h["away_team"].astype(str)))
            h = h[[x not in gone for x in k]]
    h["canon"] = h["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    h = h[h["canon"].notna() & h["d"].notna() & h["home_team"].notna() & h["away_team"].notna()]
    # one row per (league, date, home, away); keep a known score
    h = h.sort_values("hg", na_position="first").drop_duplicates(
        ["canon", "d", "home_team", "away_team"], keep="last")
    return h


def _fixtures() -> pd.DataFrame:
    fx = pd.read_parquet(FIXTURES, columns=["fixture_id", "league_id", "date", "home_name", "away_name",
                                             "ft_home", "ft_away", "status"])
    fx = fx[fx["status"].isin(["FT", "AET", "PEN"])].copy()
    fx["d"] = pd.to_datetime(fx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fx["league_id"] = pd.to_numeric(fx["league_id"], errors="coerce")
    return fx.dropna(subset=["d", "league_id"])


def prove(h: pd.DataFrame, fx: pd.DataFrame, ids: dict, min_dates: int = MIN_DATES,
          max_rounds: int = 6) -> dict:
    """Iterate to a fixpoint: pairs proven in round k become anchors (forced-opponent sides) in round
    k+1. Each round applies the full rule set; the result of the last round is returned."""
    known: dict = {}
    res = None
    for rnd in range(1, max_rounds + 1):
        res = _prove_round(h, fx, ids, min_dates, known)
        n_new = sum(len(v) for v in res["map"].values())
        n_old = sum(len(v) for v in known.values())
        res["rounds"] = rnd
        if n_new == n_old and res["map"] == known:
            break
        known = res["map"]
    return res


def _prove_round(h: pd.DataFrame, fx: pd.DataFrame, ids: dict, min_dates: int, known: dict) -> dict:
    from src.team_names import resolve_result
    by_day = {k: g for k, g in fx.groupby(["league_id", "d"])}
    days_by_league = defaultdict(set)
    for (lid, d) in by_day:
        days_by_league[lid].add(d)
    memo: dict = {}
    canon_of_lid = {v: k for k, v in ids.items()}

    def base(name, pool, key):
        """resolve_result alone: decides WHICH names need a proof (stable across rounds)."""
        k = key + ("b", name)
        if k not in memo:
            memo[k] = resolve_result(str(name), [str(c) for c in pool])
        return memo[k]

    def anchor(name, pool, key):
        """resolve_result, else an alias proven in an EARLIER round: used only as the anchor side."""
        hit = base(name, pool, key)
        if hit is None and known:
            p = lookup(known, canon_of_lid.get(key[0]), str(name), key[1])
            hit = p if p is not None and p in set(map(str, pool)) else None
        return hit

    ev = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))   # canon -> hist -> prov -> {(d, kind)}
    for r in h.itertuples(index=False):
        lid = ids.get(r.canon)
        if lid is None:
            continue
        pool = by_day.get((lid, r.d))
        if pool is None:
            continue
        names = pd.unique(pd.concat([pool["home_name"], pool["away_name"]]))
        key = (lid, r.d)
        hb, ab = base(r.home_team, names, key), base(r.away_team, names, key)
        if hb and ab:
            continue                                           # matchable by the resolver: nothing to prove
        # FORCED OPPONENT: one side anchored (the resolver, or an alias proven in an earlier round), its
        # fixture names the other side, which the resolver alone cannot. A score known for the row must
        # AGREE with that fixture, else the date is not evidence (counted). Score alone with NO anchor is
        # never evidence: on a busy day another fixture can share the score (measured: 67 false
        # second candidates).
        for unres_name, unres_side, anc_name, anc_side, unres_base in (
                (r.away_team, "away_name", r.home_team, "home_name", ab),
                (r.home_team, "home_name", r.away_team, "away_name", hb)):
            if unres_base:
                continue
            anc = anchor(anc_name, names, key)
            if not anc:
                continue
            f = pool[pool[anc_side] == anc]
            if len(f) != 1:
                continue
            kind = "forced_opponent"
            if pd.notna(r.hg) and pd.notna(r.ag):
                same = (pd.to_numeric(f["ft_home"], errors="coerce").iloc[0] == float(r.hg)
                        and pd.to_numeric(f["ft_away"], errors="coerce").iloc[0] == float(r.ag))
                if not same:
                    ev[r.canon]["__score_conflict__"]["__"].add((r.d, "score_conflict"))
                    continue
                kind = "forced_opponent+score"
            ev[r.canon][unres_name][f.iloc[0][unres_side]].add((r.d, kind))

    # history dates per (league, club) for the contradiction check
    plays = defaultdict(set)
    for r in h.itertuples(index=False):
        plays[(r.canon, r.home_team)].add(r.d)
        plays[(r.canon, r.away_team)].add(r.d)
    prov_plays = defaultdict(set)
    for (lid, d), g in by_day.items():
        for n in pd.unique(pd.concat([g["home_name"], g["away_name"]])):
            prov_plays[(lid, n)].add(d)

    out_map: dict = {}
    evidence: dict = {}
    refused = defaultdict(int)
    for canon, hist in ev.items():
        lid = ids[canon]
        conflicts = hist.pop("__score_conflict__", {}).get("__", set())
        if conflicts:
            refused["dates_with_score_conflict"] += len(conflicts)
        cands = {}
        for hn, provs in hist.items():
            # One provider name, or a provider-side RENAME over time: several names whose evidence
            # date ranges do not overlap (e.g. 'Jahn Regensburg' to 2023-05, 'SSV Jahn Regensburg'
            # from 2024-08). Each segment needs its own >= min_dates; overlapping ranges = refused.
            segs = sorted(((pn, {d for d, _ in s}, s) for pn, s in provs.items()),
                          key=lambda x: min(x[1]))
            if any(len(s[1]) < min_dates for s in segs):
                refused["fewer_than_min_dates"] += 1
                continue
            if any(max(segs[i][1]) >= min(segs[i + 1][1]) for i in range(len(segs) - 1)):
                refused["two_provider_names_overlapping"] += 1
                continue
            bad = False
            for i, (pn, dates, _) in enumerate(segs):
                lo = min(dates) if i else "0000-00-00"
                hi = max(dates) if i < len(segs) - 1 else "9999-99-99"
                hist_days = {d for d in plays[(canon, hn)] & days_by_league[lid] if lo <= d <= hi}
                if hist_days - prov_plays[(lid, pn)]:
                    bad = True
                    break
            if bad:
                refused["contradicted_by_a_date"] += 1
                continue
            cands[hn] = segs
        claimed = defaultdict(list)
        for hn, segs in cands.items():
            for pn, _, _ in segs:
                claimed[pn].append(hn)
        for hn, segs in cands.items():
            if any(len(claimed[pn]) > 1 for pn, _, _ in segs):
                refused["provider_name_claimed_twice"] += 1
                continue
            entries = []
            for i, (pn, dates, kinds) in enumerate(segs):
                kc = defaultdict(int)
                for _, k in kinds:
                    kc[k] += 1
                entries.append({"provider": pn, "dates": len(dates), "first": min(dates),
                                "last": max(dates), "by_kind": dict(kc),
                                # the segment applies up to the next segment's first evidence date
                                "valid_from": min(dates) if i else None,
                                "valid_to": (min(segs[i + 1][1]) if i < len(segs) - 1 else None)})
            out_map.setdefault(canon, {})[hn] = (segs[0][0] if len(segs) == 1 else
                                                 [{"provider": e["provider"], "valid_from": e["valid_from"],
                                                   "valid_to": e["valid_to"]} for e in entries])
            evidence.setdefault(canon, {})[hn] = entries
    return {"map": out_map, "evidence": evidence, "refused": dict(refused)}


def lookup(aliases: dict, canon: str, name: str, date: str) -> str | None:
    """The proven API-Football spelling of history `name` in league `canon` on `date` (YYYY-MM-DD),
    or None. A renamed club's segments are chosen by date (valid_from inclusive, valid_to exclusive)."""
    v = (aliases.get(canon) or {}).get(name)
    if v is None or isinstance(v, str):
        return v
    for seg in v:
        if (seg["valid_from"] is None or date >= seg["valid_from"]) and \
                (seg["valid_to"] is None or date < seg["valid_to"]):
            return seg["provider"]
    return None


_LOADED: dict | None = None


def resolve_with_proven(name: str, pool, canon: str, date: str, base_fn) -> str | None:
    """`base_fn(name, pool)` (the collectors' resolver), else the league-scoped PROVEN alias of
    `name` on `date` when that provider spelling is in `pool`. Never anything else."""
    hit = base_fn(name, pool)
    if hit is not None:
        return hit
    global _LOADED
    if _LOADED is None:
        _LOADED = load_map()
    p = lookup(_LOADED, canon, str(name), str(date)[:10])
    return p if p is not None and p in {str(x) for x in pool} else None


def load_map(path: Path = OUT) -> dict:
    """config/af_history_proven_aliases.json 'map' ({} when the file does not exist)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("map", {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ids = _af_ids()
    res = prove(_history(), _fixtures(), ids)
    n = sum(len(v) for v in res["map"].values())
    print(f"[aliases] proven pairs: {n} in {len(res['map'])} league(s); refused: {res['refused']}")
    for canon, m in sorted(res["map"].items()):
        print(f"  {canon}: {len(m)}  e.g. {list(m.items())[:4]}")
    if not a.dry_run:
        doc = {"version": RULE, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "rule": (f"history name -> API-Football name in ONE league, proven by forced-opponent or "
                        f"same-score same-day evidence on >= {MIN_DATES} distinct dates "
                        f"(identity_repair.derive_same_day_aliases standard), one-to-one, never "
                        f"contradicted by a date the history club played. Read only by the collectors' "
                        f"matchers; never by predict (team_names._ALIASES is untouched)."),
               "map": res["map"], "evidence": res["evidence"], "refused": res["refused"]}
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[aliases] wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
