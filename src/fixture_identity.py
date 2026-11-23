"""ONE fixture, ONE identity across every prediction-ledger model (owner bug report 2026-11-23).

    from src.fixture_identity import resolve_identities, split_report
    ident = resolve_identities(fixtures)      # one row per ledger fixture_ref -> its canonical identity

WHY. The ledgers key a fixture by the provider id their writer's source carries (src.prediction_ledger.
fixture_ref): 'af:<API-Football fixture id>' (props, corners, cards) or 'oa:<OddsAPI event id>' (v9 team board,
every Pro row). The two sources spell clubs differently and may disagree on the kickoff minute, so the Predictions
page split one match into two (Puebla v Atletico San Luis = af:1551039, Puebla v Atlético San Luis =
oa:40846f0c...; Club Tijuana v Atlante FC at 03:00 and 03:10). Three kinds of split are resolved here:

    accent         the same spelling once accents / punctuation are stripped (src.team_names.norm)
    name_variant   two spellings of one club ('Club Tijuana' / 'Tijuana'): src.team_names.identity_key, else the
                   league-scoped src.team_names.resolve_result against the league's other spellings (the settlement
                   resolver; an ambiguous answer is None and merges nothing)
    kickoff_drift  the sources disagree on the kickoff: the identity NEVER depends on the kickoff minute

RULE (reused, not invented). Two ledger fixtures are one match when they are in the same league (league_slug), their
UTC kickoff DATES are at most one day apart (the +-1 day window of the settlement finder,
src.prediction_ledger_v9.ResultFinder._find_in) and BOTH clubs are the same club home->home and away->away by the
club rule above. Merging is transitive (union-find). A group holding two DIFFERENT API-Football fixture ids is
refused (AF_ID_CONFLICT): its members keep their own refs, and the conflict is counted, never resolved by a guess.

CANONICAL IDENTITY of a group:
    canonical_fixture_ref   the API-Football ref 'af:<id>' when a member carries one (the provider fixture), else the
                            smallest member ref (deterministic)
    canonical_kickoff_utc   the API-Football member's kickoff (AUTHORITATIVE, kickoff_source 'api-football'); without
                            one, the members' kickoff when they all agree ('unanimous'); when they disagree it stays
                            MISSING (kickoff_source 'conflict') - never silently picked. kickoff_drift_min records the
                            spread; any drift is a data-truth finding (counted by split_report).
    canonical_home/away     the API-Football member's spelling, else the canonical-ref member's (name_source)
The original fixture_ref / kickoff / names stay on every ledger row; nothing in the ledgers is changed.
Pure and deterministic: same input, same output. No file read, no clock.
"""
from __future__ import annotations

import pandas as pd

from src import team_names as TN

IDENTITY_VERSION = "fid/1.0"
DAY_WINDOW = 1                       # the settlement finder's +-1 day (src.prediction_ledger_v9.ResultFinder)
FIX_COLS = ["fixture_ref", "league_slug", "league", "kickoff_utc", "home_team", "away_team"]
IDENT_COLS = ["fixture_ref", "canonical_fixture_ref", "canonical_kickoff_utc", "kickoff_source", "kickoff_drift_min",
              "canonical_home", "canonical_away", "name_source", "identity_members", "identity_note"]


def _af_id(ref: str) -> str | None:
    return ref[3:] if str(ref).startswith("af:") else None


class _Clubs:
    """Same-club test inside ONE league (memoised)."""

    def __init__(self, spellings):
        self.pool = sorted({str(s) for s in spellings if str(s)})
        self._memo: dict = {}

    def same(self, a: str, b: str) -> bool:
        if a == b:
            return True
        k = (a, b) if a < b else (b, a)
        if k in self._memo:
            return self._memo[k]
        ok = TN.norm(a) == TN.norm(b)
        if not ok:
            ka, kb = TN.identity_key(a), TN.identity_key(b)
            ok = ka is not None and ka == kb
        if not ok:
            ok = self._resolves(a, b) or self._resolves(b, a)
        self._memo[k] = ok
        return ok

    def _resolves(self, a: str, b: str) -> bool:
        na = TN.norm(a)
        cands = [c for c in self.pool if TN.norm(c) != na]
        return bool(cands) and TN.resolve_result(a, cands) == b


def _fixtures(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture_ref (its earliest-predicted row: what the ledger first recorded)."""
    d = rows[FIX_COLS + (["predicted_at"] if "predicted_at" in rows.columns else [])].copy()
    d["fixture_ref"] = d["fixture_ref"].astype(str)
    for c in ("league_slug", "league", "home_team", "away_team"):
        d[c] = d[c].astype(str)
    d["kickoff_utc"] = pd.to_datetime(d["kickoff_utc"], utc=True, errors="coerce")
    sort = ["fixture_ref"] + (["predicted_at"] if "predicted_at" in d.columns else [])
    return d.sort_values(sort, kind="mergesort").drop_duplicates("fixture_ref")[FIX_COLS].reset_index(drop=True)


def resolve_identities(rows: pd.DataFrame) -> pd.DataFrame:
    """IDENT_COLS, one row per fixture_ref of `rows` (ledger rows or one row per fixture)."""
    fx = _fixtures(rows)
    parent = {r: r for r in fx["fixture_ref"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for slug, g in fx.groupby("league_slug", sort=True):
        clubs = _Clubs(pd.concat([g["home_team"], g["away_team"]]))
        g = g.dropna(subset=["kickoff_utc"]).sort_values(["kickoff_utc", "fixture_ref"], kind="mergesort")
        recs = list(g[["fixture_ref", "kickoff_utc", "home_team", "away_team"]].itertuples(index=False))
        days = [r.kickoff_utc.normalize() for r in recs]
        for i, a in enumerate(recs):
            for j in range(i + 1, len(recs)):
                if (days[j] - days[i]).days > DAY_WINDOW:
                    break
                b = recs[j]
                if clubs.same(a.home_team, b.home_team) and clubs.same(a.away_team, b.away_team):
                    ra, rb = find(a.fixture_ref), find(b.fixture_ref)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    fx["_grp"] = [find(r) for r in fx["fixture_ref"]]

    out = []
    for _, g in fx.groupby("_grp", sort=True):
        afs = sorted({a for a in (_af_id(r) for r in g["fixture_ref"]) if a})
        if len(afs) > 1:                                   # two provider fixtures: refuse, never guess
            for r in g.itertuples(index=False):
                out.append(_one(pd.DataFrame([r._asdict()]), note="AF_ID_CONFLICT"))
            continue
        out.append(_one(g, note=""))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=IDENT_COLS)
    return res[IDENT_COLS].sort_values("fixture_ref", kind="mergesort").reset_index(drop=True)


def _one(g: pd.DataFrame, note: str) -> pd.DataFrame:
    g = g.sort_values("fixture_ref", kind="mergesort")
    af = g[g["fixture_ref"].str.startswith("af:")]
    head = af.iloc[0] if len(af) else g.iloc[0]
    ko = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
    kos = ko.dropna().unique()
    if len(af) and pd.notna(af.iloc[0]["kickoff_utc"]):
        cko, src = pd.Timestamp(af.iloc[0]["kickoff_utc"]), "api-football"
    elif len(kos) == 1 and ko.notna().all():
        cko, src = pd.Timestamp(kos[0]), "unanimous"
    else:
        cko, src = pd.NaT, "conflict"
    drift = (ko.max() - ko.min()).total_seconds() / 60 if ko.notna().any() else float("nan")
    return pd.DataFrame({
        "fixture_ref": g["fixture_ref"].values,
        "canonical_fixture_ref": head["fixture_ref"],
        "canonical_kickoff_utc": cko,
        "kickoff_source": src,
        "kickoff_drift_min": drift,
        "canonical_home": head["home_team"],
        "canonical_away": head["away_team"],
        "name_source": "api-football" if len(af) else "canonical_ref",
        "identity_members": len(g),
        "identity_note": note,
    })


def page_key(df: pd.DataFrame, kickoff="kickoff_utc", home="home_team", away="away_team") -> pd.Series:
    """The pre-fix page match key (league | kickoff minute | home | away) - what split the matches."""
    return (df["league_slug"].astype(str) + "|" + pd.to_datetime(df[kickoff], utc=True).dt.strftime("%Y-%m-%dT%H:%MZ")
            + "|" + df[home].astype(str) + "|" + df[away].astype(str))


def split_report(rows: pd.DataFrame, ident: pd.DataFrame | None = None) -> dict:
    """Counts of matches the old page key split, per kind and per league, before / after canonical identity.
    'before' = distinct page keys, 'after' = distinct canonical fixtures. A merged group is counted under every
    kind it shows (accent, name_variant, kickoff_drift; ref_only = only the provider refs differ)."""
    fx = _fixtures(rows)
    ident = resolve_identities(fx) if ident is None else ident
    fx = fx.merge(ident[["fixture_ref", "canonical_fixture_ref", "kickoff_source", "identity_note"]],
                  on="fixture_ref", how="left")
    fx["_pk"] = page_key(fx)
    kinds: dict = {"accent": 0, "name_variant": 0, "kickoff_drift": 0, "ref_only": 0}
    per_league: dict = {}
    split = 0
    for cref, g in fx.groupby("canonical_fixture_ref", sort=True):
        if g["_pk"].nunique() < 2 and g["fixture_ref"].nunique() < 2:
            continue
        ks = set()
        for side in ("home_team", "away_team"):
            names = g[side].unique()
            if len(names) > 1:
                ks.add("accent" if len({TN.norm(n) for n in names}) == 1 else "name_variant")
        if g["kickoff_utc"].nunique() > 1:
            ks.add("kickoff_drift")
        if not ks:
            ks.add("ref_only")
        if g["_pk"].nunique() > 1:
            split += 1
        lg = str(g["league"].iloc[0])
        pl = per_league.setdefault(lg, {k: 0 for k in kinds} | {"matches_split": 0})
        for k in ks:
            kinds[k] += 1
            pl[k] += 1
        pl["matches_split"] += int(g["_pk"].nunique() > 1)
    return {"identity_version": IDENTITY_VERSION, "fixture_refs": int(fx["fixture_ref"].nunique()),
            "page_keys_before": int(fx["_pk"].nunique()),
            "canonical_fixtures_after": int(fx["canonical_fixture_ref"].nunique()),
            "matches_split_before": split, "kinds": kinds,
            "kickoff_conflicts_unresolved": int((ident["kickoff_source"] == "conflict").sum()),
            "kickoff_drift_groups": kinds["kickoff_drift"],
            "af_id_conflicts": int((ident["identity_note"] == "AF_ID_CONFLICT").sum()),
            "per_league": dict(sorted(per_league.items()))}
