"""FPL element id <-> API-Football player_id crosswalk (Data Truth round 2, fantasy finding a).

WHY. The two sources share no key. Fantasy V1 bridged them by NAME alone (FPL full name, then FPL
web_name, refusing a name two FPL elements share): measured 2026-11-02 on the committed history and
FPL bootstrap, 168 of the 734 Premier League players the projection scores were matched, and of the
players who appeared in the CURRENT season only a minority. API-Football writes "Gabriel Magalhães"
where FPL writes "Gabriel dos Santos Magalhães", "Rodrigo Hernández Cascante" vs "Rodri" etc.

HOW. One FPL element is paired with one API-Football player only when the pair is PROVEN by
  * CLUB  - the FPL club resolved onto the API-Football club names of the same league through
            src.team_names.resolve (league-scoped, ambiguity-refusing), and
  * NAME  - compatible names: equal after normalisation, or every token of the shorter spelling is
            a token of the longer one (an "X." initial must match the first letter of a token), and
  * DATE OF BIRTH - FPL `birth_date` equal to the API-Football birth date when the provider gave
            one (output/collect/player_profiles.parquet) - method 'name+club+dob'. When it did not,
            the API-Football squad AGE must be an age the FPL birth date gave within AGE_WINDOW_DAYS
            of the squad file's date (weaker, method 'name+club+age'). A birth date or an age that
            CONTRADICTS the FPL birth date always rejects the candidate.
  * Neither birth date nor age known: only an EXACT normalised name (FPL full / known / web name
    equal to the API-Football name) in the same club pairs, method 'name+club' (birth date
    unverified; upgraded on the next rebuild once the profile collector has stored the birth date).
  * Off-club fallback (a transfer the API squad file has not caught): exact normalised full name
    AND equal birth date, unique among all API-Football players. Without a birth date: refused.
AMBIGUITY REFUSAL. More than one candidate passing any tier -> no pair (logged); an API-Football
player claimed by two FPL elements -> neither pair is kept. A missing pair stays missing.

STORED. output/fpl_af_crosswalk.csv (one row per pair, with the evidence used) is the record;
`python -m player_model.fantasy_crosswalk` rebuilds it from the committed bootstrap, squads, history
and collected profiles. player_model.fantasy joins on these ids only.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
CROSSWALK_CSV = OUT / "fpl_af_crosswalk.csv"
REFUSALS_CSV = OUT / "fpl_af_crosswalk_refusals.csv"
SQUADS_CSV = OUT / "pl_squads_official.csv"
PROFILES = OUT / "collect" / "player_profiles.parquet"
CROSSWALK_COLS = ["fpl_id", "af_player_id", "method", "fpl_name", "af_name", "fpl_team", "af_team",
                  "fpl_birth_date", "af_birth_date", "af_age", "built_at"]
#: API-Football squad ages LAG the birth date: measured 2026-11-02 on 6 of 6 sampled Premier League
#: players (Richarlison, Gusto, Barcola, Collins, Rodon, Juanlu) the squad age was the FPL birth date's
#: age one year earlier. An age is therefore compared with every age the player had over this many days
#: up to the squad file's date (a one-year-wide check; the birth date check is exact).
AGE_WINDOW_DAYS = 400


def _norm_tokens(s) -> list[str]:
    from player_model.fpl_api import _norm
    import html
    import re
    import unicodedata
    if s is None or (isinstance(s, float) and s != s):
        return []
    # html.unescape: the API-Football history carries "O&apos;Shea" / "O&apos;Nien" literally.
    t = "".join(c for c in unicodedata.normalize("NFKD", html.unescape(str(s)).translate(_TRANSLIT))
                if not unicodedata.combining(c)).lower()
    out = []
    for w in re.split(r"[\s\-]+", t):
        initial = w.endswith(".") and len(_norm(w)) == 1
        k = _norm(w)
        if k:
            out.append(k + "." if initial else k)
    return out


_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "ı": "i",
                           "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "þ": "th", "œ": "oe"})


def names_compatible(a, b) -> bool:
    """True when two spellings can name the same person: equal token lists, or every token of the
    shorter one appears in the longer one ('K.' matches a token starting with k). Order-free."""
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return False
    if "".join(ta) == "".join(tb):
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    full = [t for t in short if not t.endswith(".")]
    if not full:
        return False                                  # initials alone prove nothing
    pool = list(long_)
    for t in short:
        if t.endswith("."):
            hit = next((p for p in pool if not p.endswith(".") and p[0] == t[0]), None)
        else:
            hit = next((p for p in pool if p == t), None)
        if hit is None:
            return False
        pool.remove(hit)
    return True


def _exact(a, b) -> bool:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    return bool(ta) and "".join(ta) == "".join(tb)


def age_on(dob, on: date) -> int | None:
    d = pd.to_datetime(dob, errors="coerce")
    if d is pd.NaT or d != d:
        return None
    d = d.date()
    return on.year - d.year - ((on.month, on.day) < (d.month, d.day))


def _age_ok(af_age, fpl_dob, ref: date | None) -> bool | None:
    """None = cannot be checked (no age, no FPL birth date or no reference date)."""
    a = pd.to_numeric(af_age, errors="coerce")
    if ref is None or a != a or a is None:
        return None
    ages = {age_on(fpl_dob, ref - timedelta(days=k)) for k in range(AGE_WINDOW_DAYS + 1)}
    ages.discard(None)
    if not ages:
        return None
    return int(a) in ages


def _dob(x) -> str | None:
    d = pd.to_datetime(x, errors="coerce")
    return None if d is pd.NaT or d != d else d.strftime("%Y-%m-%d")


def fpl_elements(bootstrap: dict) -> pd.DataFrame:
    """FPL elements with the crosswalk evidence: id, names, club, birth date."""
    teams = {t["id"]: t.get("name", "") for t in (bootstrap or {}).get("teams", [])}
    rows = []
    for e in (bootstrap or {}).get("elements", []):
        rows.append({"fpl_id": e.get("id"),
                     "fpl_name": f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
                     "web_name": e.get("web_name") or "",
                     "known_name": e.get("known_name") or "",
                     "fpl_team": teams.get(e.get("team"), ""),
                     "fpl_birth_date": _dob(e.get("birth_date"))})
    return pd.DataFrame(rows, columns=["fpl_id", "fpl_name", "web_name", "known_name", "fpl_team",
                                       "fpl_birth_date"])


def af_players(history: pd.DataFrame, squads: pd.DataFrame | None = None,
               profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per API-Football player: af_player_id, af_name (history full name), af_alt_name
    (squad-feed spelling), af_team (current squad club, else the latest history club), af_age,
    af_birth_date (provider birth date from the collected profiles; None when not collected)."""
    h = history.dropna(subset=["player_id"]).copy()
    h["player_id"] = pd.to_numeric(h["player_id"], errors="coerce")
    h = h[h["player_id"] > 0]
    latest = h.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    base = pd.DataFrame({"af_player_id": latest.index.astype("int64"),
                         "af_name": latest["player_name"].to_numpy(),
                         "af_team": latest["team"].to_numpy() if "team" in latest else None})
    base["af_alt_name"] = None
    base["af_age"] = np.nan
    if squads is not None and not squads.empty:
        sq = squads.copy()
        sq["player_id"] = pd.to_numeric(sq["player_id"], errors="coerce")
        sq = sq[sq["player_id"] > 0].drop_duplicates("player_id").set_index("player_id")
        extra = sq.index.difference(base["af_player_id"])
        if len(extra):
            base = pd.concat([base, pd.DataFrame({"af_player_id": extra.astype("int64"),
                                                  "af_name": sq.loc[extra, "player_name"].to_numpy(),
                                                  "af_team": sq.loc[extra, "team"].to_numpy(),
                                                  "af_alt_name": None, "af_age": np.nan})],
                             ignore_index=True)
        on = base["af_player_id"].isin(sq.index)
        ids = base.loc[on, "af_player_id"]
        base.loc[on, "af_team"] = sq.loc[ids, "team"].to_numpy()        # transfer-accurate club
        base.loc[on, "af_alt_name"] = sq.loc[ids, "player_name"].to_numpy()
        if "age" in sq.columns:
            base.loc[on, "af_age"] = pd.to_numeric(sq.loc[ids, "age"], errors="coerce").to_numpy()
    base["af_birth_date"] = None
    if profiles is not None and not profiles.empty and "birth_date" in profiles.columns:
        p = profiles.dropna(subset=["birth_date"]).copy()
        p["player_id"] = pd.to_numeric(p["player_id"], errors="coerce")
        dob = p.groupby("player_id")["birth_date"].agg(lambda s: set(map(_dob, s)) - {None})
        # A player whose profiles disagree on his birth date has no proven birth date.
        dob = dob[dob.map(len) == 1].map(lambda s: next(iter(s)))
        base["af_birth_date"] = base["af_player_id"].map(dob).astype(object)
        base.loc[base["af_birth_date"].isna(), "af_birth_date"] = None
    return base


def build(fpl: pd.DataFrame, af: pd.DataFrame, squads_date: date | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pairs, refusals). See the module docstring for the rules."""
    from src.team_names import resolve
    af_clubs = sorted({str(t) for t in af["af_team"].dropna()})
    club_map = {t: resolve(str(t), af_clubs) for t in fpl["fpl_team"].dropna().unique()}
    by_club = {c: g for c, g in af.groupby("af_team")}
    pairs, refused = [], []

    def names_of(r):
        return [n for n in (r["af_name"], r.get("af_alt_name")) if n is not None and n == n and str(n)]

    def fpl_names(e, web: bool = True):
        # web_name (often one surname) only ever counts as an EXACT match: as a token subset it
        # paired 'Bradley' (Conor Bradley) with 'Bradley Barcola'.
        return [n for n in (e["fpl_name"], e["known_name"], e["web_name"] if web else "") if n]

    def _has(x) -> bool:
        # a frame turns a None birth date into NaN, which is truthy: test for a real string
        return isinstance(x, str) and bool(x)

    def dob_verdict(e, r):
        """True / False when proven, None when it cannot be checked."""
        if _has(r["af_birth_date"]) and _has(e["fpl_birth_date"]):
            return r["af_birth_date"] == e["fpl_birth_date"]
        if _has(e["fpl_birth_date"]):
            return _age_ok(r["af_age"], e["fpl_birth_date"], squads_date)
        return None

    for _, e in fpl.iterrows():
        club = club_map.get(e["fpl_team"])
        exact_hits, compat_hits = [], []
        if club is not None:
            for _, r in by_club.get(club, pd.DataFrame()).iterrows():
                exact = any(_exact(a, b) for a in names_of(r) for b in fpl_names(e))
                if not exact and not any(names_compatible(a, b) for a in names_of(r)
                                         for b in fpl_names(e, web=False)):
                    continue
                v = dob_verdict(e, r)
                if v is False:
                    continue                    # contradicting birth date / age: not this player
                if v is None:
                    # neither birth date nor age: only an exact name proves the pair
                    if exact:
                        exact_hits.append((r, "name+club"))
                    continue
                method = ("name+club+dob" if _has(r["af_birth_date"]) and _has(e["fpl_birth_date"])
                          else "name+club+age")
                (exact_hits if exact else compat_hits).append((r, method))
        # An exact name outranks a merely compatible one; ambiguity is judged within the tier used.
        hits = exact_hits or compat_hits
        if not hits and _has(e["fpl_birth_date"]):
            # off-club: exact full name AND equal provider birth date, unique league-wide
            cand = af[af["af_birth_date"] == e["fpl_birth_date"]]
            for _, r in cand.iterrows():
                if any(_exact(a, b) for a in names_of(r) for b in (e["fpl_name"], e["known_name"]) if b):
                    hits.append((r, "name+dob"))
        if len(hits) == 1:
            r, method = hits[0]
            pairs.append({"fpl_id": int(e["fpl_id"]), "af_player_id": int(r["af_player_id"]),
                          "method": method, "fpl_name": e["fpl_name"], "af_name": r["af_name"],
                          "fpl_team": e["fpl_team"], "af_team": r["af_team"],
                          "fpl_birth_date": e["fpl_birth_date"], "af_birth_date": r["af_birth_date"],
                          "af_age": r["af_age"]})
        elif len(hits) > 1:
            refused.append({"fpl_id": int(e["fpl_id"]), "fpl_name": e["fpl_name"],
                            "fpl_team": e["fpl_team"], "reason": "ambiguous",
                            "candidates": ";".join(str(int(r["af_player_id"])) for r, _ in hits)})
        else:
            refused.append({"fpl_id": int(e["fpl_id"]), "fpl_name": e["fpl_name"],
                            "fpl_team": e["fpl_team"],
                            "reason": "club unresolved" if club is None else "no proven candidate",
                            "candidates": ""})
    p = pd.DataFrame(pairs, columns=[c for c in CROSSWALK_COLS if c != "built_at"])
    # one-to-one: an API-Football id claimed by two FPL elements proves neither
    dup = p["af_player_id"].duplicated(keep=False)
    for _, r in p[dup].iterrows():
        refused.append({"fpl_id": int(r["fpl_id"]), "fpl_name": r["fpl_name"], "fpl_team": r["fpl_team"],
                        "reason": "api-football id claimed twice", "candidates": str(r["af_player_id"])})
    p = p[~dup].reset_index(drop=True)
    p["built_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return p, pd.DataFrame(refused, columns=["fpl_id", "fpl_name", "fpl_team", "reason", "candidates"])


def squads_file_date(path: Path = SQUADS_CSV) -> date | None:
    """Date of the squad file's last commit (CI fetches and commits it in one run); None without git."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%cI", "--", str(path)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return datetime.fromisoformat(out.replace("Z", "+00:00")).date() if out else None
    except Exception:                                                     # noqa: BLE001
        return None


def load(path: Path = CROSSWALK_CSV) -> pd.DataFrame:
    """The stored crosswalk (empty frame when absent)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=CROSSWALK_COLS)
    return pd.read_csv(path)


def af_to_fpl(af_ids: pd.Series, live_fpl_ids, crosswalk: pd.DataFrame | None = None) -> pd.Series:
    """FPL element id per API-Football player id via the stored crosswalk; NaN when the pair is not
    established or its FPL element is not in the live FPL set."""
    cw = load() if crosswalk is None else crosswalk
    if cw.empty:
        return pd.Series(np.nan, index=af_ids.index, dtype="float64")
    live = set(pd.to_numeric(pd.Series(list(live_fpl_ids)), errors="coerce").dropna().astype(int))
    cw = cw[pd.to_numeric(cw["fpl_id"], errors="coerce").isin(live)]
    m = dict(zip(pd.to_numeric(cw["af_player_id"], errors="coerce").astype(int),
                 pd.to_numeric(cw["fpl_id"], errors="coerce").astype(float)))
    return pd.to_numeric(af_ids, errors="coerce").map(m).astype("float64")


def rebuild(history: pd.DataFrame | None = None, write: bool = True,
            squads_date: date | None = None) -> dict:
    """Rebuild output/fpl_af_crosswalk.csv from the committed inputs. Returns counts."""
    from player_model import fantasy as F
    boot = json.loads((OUT / "fpl_bootstrap.json").read_text(encoding="utf-8"))
    if history is None:
        import pyarrow.dataset as ds
        d = ds.dataset(str(ROOT / "player_history.parquet"), format="parquet")
        history = d.to_table(columns=["player_id", "player_name", "date", "league", "team"],
                             filter=ds.field("league").isin(F._fantasy_league_spellings())).to_pandas()
    history = F._fantasy_league_rows(history)
    squads = pd.read_csv(SQUADS_CSV) if SQUADS_CSV.exists() else None
    profiles = pd.read_parquet(PROFILES, columns=["player_id", "birth_date"]) if PROFILES.exists() else None
    fpl = fpl_elements(boot)
    af = af_players(history, squads, profiles)
    pairs, refused = build(fpl, af, squads_date or squads_file_date())
    if write:
        pairs.to_csv(CROSSWALK_CSV, index=False)
        refused.to_csv(REFUSALS_CSV, index=False)
    out = {"fpl_elements": len(fpl), "af_players": len(af), "pairs": len(pairs),
           "by_method": pairs["method"].value_counts().to_dict(),
           "refused": refused["reason"].value_counts().to_dict()}
    print(f"[crosswalk] {out}")
    return out


if __name__ == "__main__":
    rebuild()
