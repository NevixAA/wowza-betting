"""
Historical match data loader.

Reads every raw match sheet from the summary XLSX workbook and returns
a clean, consolidated DataFrame sorted by date.

Sheet naming convention: {League}_{YYMM}  e.g. "Championship_2526"

Output columns
──────────────
date, league, season, home_team, away_team
home_goals, away_goals, total_goals, ftr
over25, btts
home_corners, away_corners, home_fouls, away_fouls
home_shots, away_shots, home_sot, away_sot
home_sot_ratio, away_sot_ratio
odds_over25, odds_under25
referee
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import logging

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

log = logging.getLogger(__name__)

from src.integrity import joins as _IJ  # noqa: E402  (R2-C join integrity)


def _price_mapping_filter() -> bool:
    """WOWZA_PRICE_MAPPING_FILTER=1 drops capture rows whose market mapping is proven invalid
    (src/integrity/price_provenance.label_market_mapping: newformat O/U captured before the
    2026-08-10 parser fix, first-half BTTS > 3.20) from the training-price reads. Default ON since
    2026-11-01 (owner decision, Data Truth program): a price for the wrong market is not data.
    It changes training prices for ~360 summer-2026 NF fixtures. Rollback: set it to 0."""
    return os.getenv("WOWZA_PRICE_MAPPING_FILTER", "1").strip() == "1"


def _drop_mismapped(o: pd.DataFrame, source: str, tag: str) -> pd.DataFrame:
    """Rows with market_mapping_valid=False removed from THIS READ ONLY (raw file untouched).
    Always counts them in the log; removes them only with WOWZA_PRICE_MAPPING_FILTER=1."""
    try:
        if not {"snapshot_ts", "market", "odds"}.issubset(o.columns) or o.empty:
            return o
        from src.integrity.price_provenance import label_market_mapping
        ok = label_market_mapping(o[["snapshot_ts", "market", "odds"]], source)["market_mapping_valid"]
        n_bad = int((~ok.to_numpy()).sum())
        if n_bad:
            act = "dropped" if _price_mapping_filter() else "KEPT (WOWZA_PRICE_MAPPING_FILTER=0)"
            log.warning(f"[{tag}] {n_bad} row(s) of {source} carry an invalid market mapping — {act}")
            if _price_mapping_filter():
                return o[ok.to_numpy()]
    except Exception as _e:                                       # noqa: BLE001
        log.warning(f"[{tag}] price-mapping check skipped: {_e}")
    return o


def _safe_joins() -> bool:
    """Enrichment joins are ALWAYS league-keyed, collapse-then-merge. The legacy (home, away, date)
    merge without league put 85,454 shot/xG/inside-box values on the wrong fixtures
    (join_integrity_health.json); its positional copy-back and the WOWZA_SAFE_ENRICH_JOINS=0
    rollback were removed 2026-11-01 (Data Truth). A source without a league column is skipped
    with an error log, never joined without one. Kept as a function for src/integrity readers."""
    return True


# Preferred odds columns in order (first found with data wins)
_OVER_COLS  = ["AvgC>2.5", "MaxC>2.5", "B365C>2.5", "PC>2.5",
               "Avg>2.5",  "Max>2.5",  "B365>2.5",  "P>2.5"]
_UNDER_COLS = ["AvgC<2.5", "MaxC<2.5", "B365C<2.5", "PC<2.5",
               "Avg<2.5",  "Max<2.5",  "B365<2.5",  "P<2.5"]

# Side-market odds (not available in all leagues/seasons — NaN where absent)
_BTTS_COLS  = ["AvgBTSH", "MaxBTSH", "BbAvBTSH", "BBTSH"]
_OVER15_COLS = ["Avg>1.5", "Max>1.5", "BbAv>1.5", "B365>1.5"]
_OVER35_COLS = ["Avg>3.5", "Max>3.5", "BbAv>3.5", "B365>3.5"]

# Real historical O/U + BTTS odds for new-format leagues, backfilled from The Odds API.
# Replaces the bug where new-format odds_over25/under25 were mapped to 1X2 home/away odds.
from pathlib import Path as _Path
_NF_ODDS_FILE = _Path(__file__).resolve().parents[1] / "output" / "newformat_odds_history.csv"
_nf_odds_cache = None
def _nf_real_odds() -> dict:
    """Lookup {(YYYY-MM-DD, league, 'Home vs Away'): {market: odds}} from the backfill."""
    global _nf_odds_cache
    if _nf_odds_cache is None:
        _nf_odds_cache = {}
        if _NF_ODDS_FILE.exists():
            try:
                _o = pd.read_csv(_NF_ODDS_FILE)
                # ── DROP FIRST-HALF BTTS MISLABELLED AS FULL-MATCH (found 2026-08-23) ──────
                #
                # capture_nf_odds_forward.py carried the identical substring-matching defect as
                # the standard capture: `"Both Teams Score" in name` also matches bet 34, "Both
                # Teams Score - First Half" (Yes ~5.50), and the parse loop assigned on every
                # match so the last one won. Both parsers are fixed, so no NEW contamination
                # arrives.
                #
                # THE NEW-FORMAT DAMAGE IS FAR WORSE THAN THE STANDARD TRACK'S and was missed on
                # the first pass because only the standard file was measured:
                #     standard_sidemarket_odds_history.csv    271 / 6,358 btts_yes  =  4.3%
                #     newformat_odds_history.csv              931 / 3,778 btts_yes  = 24.6%
                #
                # A quarter of new-format BTTS training prices were first-half quotes. Left in,
                # they teach the model that BTTS pays ~6.0 in leagues where it pays ~1.9 — which
                # is not noise, it is a systematic bias toward a fictional longshot edge.
                #
                # Threshold and evidence: see _enrich_with_standard_sidemarket_odds. The two
                # populations do not overlap, so this cannot discard a genuine price.
                # DROPPED FROM THIS READ ONLY — the raw capture file is left untouched so the
                # contamination stays auditable.
                _bad = ((_o["market"].astype(str) == "btts_yes")
                        & (pd.to_numeric(_o["odds"], errors="coerce") > 3.20))
                if _bad.any():
                    log.warning(f"[nf_odds] dropped {int(_bad.sum())} btts_yes row(s) with odds "
                                f"> 3.20 — first-half BTTS mislabelled as full-match "
                                f"(2026-08-23 parser fix). Raw file unchanged.")
                    _o = _o[~_bad]
                _o = _drop_mismapped(_o, "newformat_odds_history", "nf_odds")
                # Keyed by the MATCH date (snapshot date only when absent), last PRE-KICKOFF
                # snapshot per market. Keying by snapshot date filed every price captured before
                # match day under the wrong day, so the training lookup (by match date) missed it
                # (Data Truth audit finding 39).
                _md = _o["match_date"].where(_o["match_date"].notna(), _o["snapshot_date"]) \
                    if "match_date" in _o.columns else _o["snapshot_date"]
                _o = _o.assign(md_key=_md.astype(str).str[:10])
                if "snapshot_ts" in _o.columns:
                    _ts = pd.to_datetime(_o["snapshot_ts"], errors="coerce", utc=True)
                    _ko = pd.to_datetime(_o.get("kickoff_utc"), errors="coerce", utc=True)
                    _o = _o[~(_ko.notna() & _ts.notna() & (_ts >= _ko))]
                    _o = _o.assign(ts_key=pd.to_datetime(_o["snapshot_ts"], errors="coerce", utc=True)
                                   ).sort_values("ts_key", kind="mergesort")
                # (itertuples renames columns that start with "_", so no underscore names here)
                for _r in _o.itertuples(index=False):
                    _nf_odds_cache.setdefault(
                        (str(_r.md_key), str(_r.league), str(_r.match)), {}
                    )[str(_r.market)] = float(_r.odds)          # later snapshot overwrites
            except Exception as e:                                     # noqa: BLE001
                # Never silent: an empty price table looks exactly like "no captures".
                log.error(f"[nf_odds] could not build the NF price table ({type(e).__name__}: {e})")
    return _nf_odds_cache

_nf_by_day: dict | None = None


def _nf_odds_lookup(nfo: dict, day: str, league: str, home: str, away: str) -> dict:
    """Prices for one fixture: the exact 'Home vs Away' key, else both names resolved within that
    day and league (src.team_names.resolve - the capture uses OddsAPI spellings, the history uses
    football-data's). Exactly one fixture or nothing (Data Truth audit finding 39)."""
    hit = nfo.get((day, league, f"{home} vs {away}"))
    if hit:
        return hit
    global _nf_by_day
    if _nf_by_day is None or _nf_by_day.get("_src") is not nfo:
        _nf_by_day = {"_src": nfo}
        for (d, lg, m) in nfo:
            if " vs " in m:
                h, a = m.split(" vs ", 1)
                _nf_by_day.setdefault((d, lg), []).append((h, a, m))
    cands = _nf_by_day.get((day, league), [])
    if not cands:
        return {}
    from src.team_names import resolve as _rs
    rh = _rs(home, [c[0] for c in cands])
    ra = _rs(away, [c[1] for c in cands])
    if not rh or not ra:
        return {}
    ms = [m for h, a, m in cands if h == rh and a == ra]
    return nfo.get((day, league, ms[0]), {}) if len(ms) == 1 else {}


# Module-level cache so we only parse the XLSX once per process
_CACHE: Optional[pd.DataFrame] = None


def _decode_csv(content) -> str:
    """Downloaded CSV bytes -> text: UTF-8 (BOM stripped) when valid, else Latin-1. `r.text` let
    requests guess ISO-8859-1 for UTF-8 files with no charset header, which banked
    'PreuÃ\\x9fen MÃ¼nster' beside 'Preußen Münster' (same club, two teams, every match twice)."""
    from src.integrity.duplicate_matches import decode_csv_bytes
    return decode_csv_bytes(content if isinstance(content, (bytes, bytearray)) else
                            str(content).encode("utf-8"))


def _pick_odds(raw: pd.DataFrame, cols: list[str]) -> pd.Series:
    for col in cols:
        if col in raw.columns:
            s = pd.to_numeric(raw[col], errors="coerce")
            if s.notna().sum() > len(raw) * 0.1:   # at least 10% coverage
                return s
    return pd.Series(np.nan, index=raw.index)


_AFFIX_RE = re.compile(
    r"^(1\.?\s*)?(fc|cf|sc|ac|as|ss|ssc|us|usc|rc|cd|ud|sv|tsv|vfl|vfb|fk|nk|bk|if|ik|afc|cfc)\b"
    r"|\b(fc|cf|sc|ac|afc|cfc|sk|sv|bk|if|ik|kv|vv)$")


def _norm_club(s: str) -> str:
    """Accent-, punctuation- and affix-insensitive club key.

    NO LONGER USED IN THIS MODULE (2026-11-02): _canonicalise_clubs keys clubs by
    src.team_names.identity_key. Kept ONLY because src/standard_repair.attach_entry_prices
    (serve-owned) still imports it as a join key; delete it when that file moves to
    team_names.resolve."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    prev = None
    while prev != t:
        prev = t
        t = _AFFIX_RE.sub("", t).strip()
    return re.sub(r"\s+", " ", t).strip()


def _resolve_onto_history(right: pd.DataFrame, df: pd.DataFrame, home_col: str, away_col: str,
                          name: str, aliases: dict | None = None) -> pd.DataFrame:
    """`right` restricted to the rows whose two clubs are PROVABLY clubs of `df`, with `_h`/`_a`
    holding df's own spelling of each (2026-11-02, owner rule).

    Every enrichment join below used to key both sides on api_football_ou._norm_name, a
    suffix-stripping key with its own alias dict -- an identity guess. Now each source name is
    resolved by src.team_names.resolve among df's clubs of the row's league (the canonical,
    league-scoped resolver); a name no rule proves is skipped, counted and logged, never guessed.
    df's side of the join is then simply its own home_team / away_team.

    `aliases` ({league: {spelling: canonical}}, Phase 2 HT repair 2026-11-03): the identity map
    one_club_one_match APPLIED to df (IDENTITY_ALIASES). A source keyed on the history's OWN
    pre-identity spelling (the HT sidecar) names a club the pipeline renamed by same-day evidence -
    'Atletico Paranaense' -> 'Athletico-PR' - which resolve() cannot spell-match. Only when resolve()
    finds nothing: the spelling (or one resolve() proves equal to it) must be an alias of exactly ONE
    canonical club that is in THIS league's pool; otherwise the row stays unplaced."""
    from src.team_names import resolve
    if right.empty or "league" not in right.columns or "league" not in df.columns:
        return right.iloc[0:0].assign(_h=pd.Series(dtype=object), _a=pd.Series(dtype=object))
    from src.integrity.duplicate_matches import fix_mojibake
    pools = {str(lg): sorted(set(g["home_team"].dropna().astype(str))
                             | set(g["away_team"].dropna().astype(str)))
             for lg, g in df.groupby("league")}
    r = right[right["league"].astype(str).isin(pools)].copy()
    memo: dict = {}
    pool_sets = {lg: set(p) for lg, p in pools.items()}

    def _by_identity_alias(lg, txt):
        amap = (aliases or {}).get(lg) or {}
        hits = {c for a, c in amap.items()
                if c in pool_sets[lg] and (a == txt or resolve(txt, [a]) == a)}
        return hits.pop() if len(hits) == 1 else None

    def _one(lg, nm):
        if nm is None or (isinstance(nm, float) and np.isnan(nm)) or not str(nm).strip():
            return None
        k = (lg, str(nm))
        if k not in memo:
            # a source name banked as mojibake ('PreuÃ\x9fen MÃ¼nster') resolves by its real text
            txt = fix_mojibake(str(nm))
            memo[k] = resolve(txt, pools[lg])
            if memo[k] is None and aliases:
                memo[k] = _by_identity_alias(lg, txt)
        return memo[k]

    lgs = r["league"].astype(str).tolist()
    r["_h"] = [_one(lg, n) for lg, n in zip(lgs, r[home_col].tolist())]
    r["_a"] = [_one(lg, n) for lg, n in zip(lgs, r[away_col].tolist())]
    bad = r["_h"].isna() | r["_a"].isna()
    if bad.any():
        miss = sorted({(lg, str(n)) for lg, n, ok in
                       zip(lgs, r[home_col].tolist(), r["_h"].notna()) if not ok}
                      | {(lg, str(n)) for lg, n, ok in
                         zip(lgs, r[away_col].tolist(), r["_a"].notna()) if not ok})
        log.info(f"  [{name}] {int(bad.sum())} source row(s) skipped: {len(miss)} club name(s) not "
                 f"provably a club of the league (team_names.resolve), e.g. {miss[:5]}")
    return r[~bad]


def _canonicalise_clubs(df: pd.DataFrame) -> pd.DataFrame:
    """One spelling per club per league, then a real de-duplication.

    WHY THIS EXISTS, AND IT IS A BUG I CAUSED. The cache-skip fix (2026-09-23) made the loader
    re-download finished seasons it had been serving from a stale cache. That was right -- it
    recovered four seasons of corners, half-time goals and prices. What it also did was bring
    football-data's club names in ALONGSIDE the API-Football names already cached, so the same
    match now appears twice:

        2023-07-29  FC Kaiserslautern  v  FC St. Pauli   1-2   season "2023"
        2023-07-29     Kaiserslautern  v      St Pauli   1-2   season "2023/24"

    Measured on the committed cache: 2,040 fixtures duplicated this way, 4,080 rows, and the
    scores AGREE on all 2,040 -- so they are unambiguously the same matches. The existing
    de-duplication keys on (date, league, home_team, away_team) and therefore never saw them,
    because the team strings differ.

    The damage is not the duplicate rows. It is that ONE CLUB BECOMES TWO TEAMS, so its rolling
    form is computed over half its matches -- invariant 11, the failure that once left 46% of
    standard fixtures with no form data at all. 50 clubs across 30 leagues, ~10,500 team-match
    rows, including Kaiserslautern, Koln, Magdeburg, Nurnberg and Schalke in Bundesliga 2 and
    Granada, Cartagena and Castellon in La Liga 2.

    THE MERGE IS EVIDENCE-BASED, NOT FUZZY. Two spellings are merged only when they have the same
    src.team_names.identity_key (same identity tokens once club-form words are removed; reserve
    sides apart) WITHIN THE SAME LEAGUE. The winner is the spelling seen most recently, because
    that is what future downloads will keep producing, so the vocabulary converges rather than
    oscillating. Ambiguity is impossible here by construction: the merge is within one league and
    one normalised key, and the duplicated fixtures it resolves agree on every score.
    """
    if not {"league", "home_team", "away_team"}.issubset(df.columns):
        return df
    d = df.copy()
    for c in ("home_team", "away_team"):
        d[c] = d[c].astype(str).str.strip()
    long = pd.concat([d[["league", "home_team", "date"]].rename(columns={"home_team": "t"}),
                      d[["league", "away_team", "date"]].rename(columns={"away_team": "t"})])
    # Club identity through the canonical resolver's token rule (src.team_names.identity_key); a
    # name with no identity token keys on itself, so it is never merged with anything.
    from src.team_names import identity_key as _ikey

    def _key(t):
        k = _ikey(t)
        return k if k is not None else f"raw|{t}"

    long["k"] = long["t"].map(_key)
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    # Most recently seen spelling wins; frequency breaks a tie.
    agg = (long.groupby(["league", "k", "t"])
               .agg(last=("date", "max"), n=("t", "size")).reset_index()
               .sort_values(["league", "k", "last", "n"], ascending=[True, True, False, False]))
    winner = agg.drop_duplicates(["league", "k"], keep="first")
    mapping = {(lg, k): t for lg, k, t in zip(winner["league"], winner["k"], winner["t"])}
    n_multi = int((agg.groupby(["league", "k"])["t"].transform("size") > 1).sum())

    for c in ("home_team", "away_team"):
        keys = list(zip(d["league"].astype(str), d[c].map(_key)))
        d[c] = [mapping.get(kk, orig) for kk, orig in zip(keys, d[c])]

    # CONFLICTING SCORES ARE DROPPED, NOT GUESSED.
    #
    # Unifying the spellings exposes duplicates that were previously invisible, and almost all of
    # them agree: 2,039 of 2,040. One does not -- Orlando City v Columbus Crew, 2023-11-25, is
    # 0-2 in one source and 0-0 in the other. Keeping whichever row happens to sort last would
    # ship a coin-flip scoreline into the training targets. One fixture out of 60,281 is not
    # worth guessing on, so the whole fixture is dropped and named in the log.
    #
    # (My first check on this compared only home_goals and reported "2,040 of 2,040 agree".
    # It missed this one, which differs on the away side. Hence both columns here.)
    key = (d["date"].astype(str).str.slice(0, 10) + "|" + d["league"].astype(str) + "|"
           + d["home_team"] + "|" + d["away_team"])
    conflict = key.isin(
        key[d.groupby(key)[["home_goals", "away_goals"]].transform("nunique").max(axis=1) > 1])
    n_conflict = int(key[conflict].nunique())
    if n_conflict:
        for k in sorted(key[conflict].unique())[:5]:
            log.warning(f"[clubs] sources disagree on the score for {k} — fixture dropped")
        d = d[~conflict]

    # KEEP THE RICHEST ROW, NOT THE LAST ONE.
    #
    # `keep="last"` after a date sort is arbitrary among duplicates, and arbitrary is not free:
    # one copy of a fixture may carry corners, half-time goals and a price while the other has
    # only the score, and the conflict check above cannot see the difference because
    # `nunique()` ignores NaN -- a row with 2.0 and a row with NaN look like agreement. Ranking
    # by how many fields are populated keeps the informative copy every time and makes the
    # choice deterministic rather than incidental.
    before = len(d)
    d = d.assign(_completeness=d.notna().sum(axis=1))
    d = (d.sort_values(["date", "_completeness"], kind="mergesort")
           .drop_duplicates(["date", "league", "home_team", "away_team"], keep="last")
           .drop(columns=["_completeness"]))
    if n_multi or before != len(d) or n_conflict:
        log.info(f"[clubs] unified {n_multi} alternate club spelling(s); removed "
                 f"{before - len(d):,} duplicate fixture row(s) the old key missed; "
                 f"quarantined {n_conflict} fixture(s) whose sources disagree on the score")
    return d


def _canonical_season_labels(d: pd.DataFrame) -> pd.Series:
    """Season label DERIVED from (league, date) by the registry's season rules (src.season_resolver):
    split-season leagues 'YYYY/YY', calendar-year leagues 'YYYY', J1 by its eras. A league the
    registry does not know keeps its stored label."""
    from src.season_resolver import resolve
    dt = pd.to_datetime(d["date"], errors="coerce")
    lg = d["league"].astype(str)
    ym = dt.dt.strftime("%Y-%m-01")
    lab = {}
    for league, first in set(zip(lg, ym)):
        if first == "NaT" or not isinstance(first, str):
            continue
        r = resolve(league, first)
        if not str(r.season).isdigit():
            continue
        s = int(r.season)
        if "calendar_year" in r.rule or "fixed" in r.rule:
            lab[(league, first)] = str(s)
        else:
            start = s - 1 if "end_year" in r.rule else s
            lab[(league, first)] = f"{start}/{(start + 1) % 100:02d}"
    new = pd.Series([lab.get(k) for k in zip(lg, ym)], index=d.index, dtype=object)
    return new.where(new.notna(), d.get("season"))


# One club, two spellings, from sources that never overlap in time (2026-11-01 audit of the merged
# history): no same-day evidence can exist, so the pairs are named - in config/leagues.json
# (same_club_spellings, read through src.league_registry), never in code.


#: one-side same-day aliases accepted on EVIDENCE alone (no team_names proof), logged each run so the
#: owner can move them into the alias store (2026-11-02, team_frame finding 1)
_EVIDENCE_ONLY_ALIASES: dict = {}


def _one_side_aliases(df: pd.DataFrame, min_dates: int = 3) -> dict:
    """{league: {alias: canonical}} from same-day, same-score fixtures where ONE side's name is
    identical and the other differs - 'Stoke City v Hull' next to 'Stoke v Hull'. A club cannot
    play twice on one day, so the differing names are one club. identity_repair (frozen
    v9r-identity-1.0, used by the repaired Standard recipe) only counts pairs where BOTH names
    differ, so these duplicates survived. Safeguards: >= min_dates distinct dates, the two names
    never met, one-to-one; newest spelling is canonical.

    PROOF IS THE EVIDENCE, NOT THE SPELLING (2026-11-02). The pair used to need src.team_names to
    prove it as well, so every club the resolver cannot spell-match kept both names and its matches
    twice: 1,907+ same-day pairs at data-truth-100 HEAD ('A. Klagenfurt'/'Austria Klagenfurt' 122
    dates, 'U. Cluj'/'Universitatea Cluj' 93, 'Athletico-PR'/'Atletico Paranaense' 120, ...). With
    one side the SAME club on the SAME day at the SAME score, the other side cannot be a different
    club - a club does not play twice in a day - so >= min_dates such dates are proof. Pairs the
    resolver does NOT also prove are recorded in _EVIDENCE_ONLY_ALIASES and logged."""
    from collections import Counter, defaultdict
    from src.identity_repair import _same_club
    _EVIDENCE_ONLY_ALIASES.clear()
    d = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    d["_d"] = pd.to_datetime(d["date"]).dt.normalize()
    ev: dict = defaultdict(Counter)
    _gk = ["league", "_d", "home_goals", "away_goals"]
    # only groups of >= 2 rows can hold evidence; iterating the ~100k singletons cost ~15 s
    multi = d[d.groupby(_gk, sort=False)["home_team"].transform("size") >= 2]
    multi = multi.assign(home_team=multi["home_team"].astype(str), away_team=multi["away_team"].astype(str))
    for (lg, dt, _hg, _ag), g in multi.groupby(_gk, sort=False):
        rows = list(zip(g["home_team"], g["away_team"]))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (h1, a1), (h2, a2) = rows[i], rows[j]
                if (h1 == h2) != (a1 == a2):                 # exactly one side identical
                    x, y = (a1, a2) if h1 == h2 else (h1, h2)
                    ev[lg][(tuple(sorted((x, y))), dt)] += 1
    last = pd.concat([d[["league", "home_team", "_d"]].rename(columns={"home_team": "t"}),
                      d[["league", "away_team", "_d"]].rename(columns={"away_team": "t"})]
                     ).groupby(["league", "t"])["_d"].max().to_dict()
    met = defaultdict(set)
    for lg, h, a in zip(d["league"], d["home_team"].astype(str), d["away_team"].astype(str)):
        met[lg].add(tuple(sorted((h, a))))
    out: dict = {}
    for lg, cnt in ev.items():
        dates = Counter(k for (k, _dt) in cnt)            # distinct dates per name pair
        strong = [(k, n) for k, n in dates.items() if n >= min_dates and k not in met[lg]]
        partners: dict = defaultdict(list)
        for (x, y), n in strong:
            partners[x].append((n, y))
            partners[y].append((n, x))
        m = {}
        for (x, y), n in strong:
            if max(partners[x])[1] != y or max(partners[y])[1] != x:
                continue
            canon, alias = sorted((x, y), key=lambda t: (last.get((lg, t), pd.Timestamp(0)), t),
                                  reverse=True)
            m[alias] = canon
            if not _same_club(x, y):
                _EVIDENCE_ONLY_ALIASES.setdefault(lg, {})[alias] = (canon, n)
        if m:
            out[lg] = m
    return out


#: last run's alias-store merges {league: {alias: canonical}} (logged; read by tests)
ALIAS_STORE_MERGES: dict = {}

#: the full club identity map {league: {spelling: canonical}} the last one_club_one_match applied
#: (src.identity_repair.canonicalise_and_dedupe report 'aliases'). The HT step uses it to place
#: sidecar rows keyed on the history's own pre-identity spelling (see _resolve_onto_history).
IDENTITY_ALIASES: dict = {}


def _alias_store_merges(out: pd.DataFrame, amap: dict, last: dict) -> dict:
    """Add to `amap` ({league: {alias: canonical}}, edited in place) every club the ALIAS STORE
    (src.team_names, fixture-level evidence per entry) names as one club but the league's history
    still holds under two spellings - the splits same-day evidence cannot see (< 3 shared dates:
    'Verona'/'Hellas Verona' in Serie B) or that one-to-one refused (Romanian Superliga 'CS
    Universitatea Craiova'/'Univ. Craiova', 2026-11-02). A group whose spellings ever meet in one
    row is contradicted by the data and skipped (logged). Newest spelling is canonical; every entry
    maps straight to it (canonicalise_and_dedupe applies one step, no chains)."""
    from src.team_names import alias_groups_in
    ALIAS_STORE_MERGES.clear()
    for lg, g in out.groupby(out["league"].astype(str), sort=True):
        h, a = g["home_team"].astype(str), g["away_team"].astype(str)
        m = dict(amap.get(lg, {}))
        added = {}
        for grp in alias_groups_in(set(h) | set(a)):
            if (h.isin(grp) & a.isin(grp)).any():
                log.warning(f"[identity] {lg}: alias-store spellings {sorted(grp)} meet in one row - "
                            f"not merged")
                continue
            names = set(grp) | {m.get(n, n) for n in grp}
            canon = max(names, key=lambda t: (last.get((lg, t), pd.Timestamp(0)), t))
            if m.get(canon) in names:
                del m[canon]
            for n in grp:
                if n != canon and m.get(n) != canon:
                    m[n] = added[n] = canon
            for k, v in list(m.items()):                       # nothing may point at a non-canonical
                if v in grp and v != canon:
                    m[k] = canon
        if added:
            amap[lg] = m
            ALIAS_STORE_MERGES[lg] = added
    if ALIAS_STORE_MERGES:
        log.info("[identity] alias-store merges: " + "; ".join(
            f"{lg}: {a} -> {c}" for lg, mm in ALIAS_STORE_MERGES.items() for a, c in mm.items()))
    return ALIAS_STORE_MERGES


def _canonical_identity_and_seasons(out: pd.DataFrame) -> pd.DataFrame:
    """ONE club, ONE fixture, ONE season label - for every model track, not only Standard repaired.

    The merged history carried the same match twice under two spellings and two labels
    ('Sheffield Wednesday' 2023 next to 'Sheffield Weds' 2023/24, 'Stoke City' 2026 next to 'Stoke'
    2026/27): 3,158 duplicate fixtures in 8 split leagues, calendar labels whose group spans two
    real campaigns (Championship '2025' = half of 2024/25 + half of 2025/26), and clubs split in
    two (Belgrano / Belgrano Cordoba, Kyoto / Kyoto Sanga, Urawa / Urawa Reds, UNAM). Season-to-date
    features - the NF model's top feature - averaged across campaigns and half-histories
    (Data Truth audit finding 37).

    Season labels come from the registry rules; clubs and duplicates go through
    src.identity_repair.canonicalise_and_dedupe (same-day evidence, >= 3 dates; score conflicts
    quarantined, never guessed) - the step the repaired Standard recipe already serves with.
    Rollback: WOWZA_IDENTITY_DEDUPE=0."""
    if os.getenv("WOWZA_IDENTITY_DEDUPE", "1").strip() == "0" or out.empty:
        return out
    try:
        before = len(out)
        out = out.copy()
        out["season"] = _canonical_season_labels(out)
        from src.identity_repair import canonicalise_and_dedupe
        one_side = _one_side_aliases(out)
        # Spellings of one club that never share a day (so no same-day evidence exists); the
        # most recently seen spelling is canonical.
        _last = pd.concat([out[["league", "home_team", "date"]].rename(columns={"home_team": "t"}),
                           out[["league", "away_team", "date"]].rename(columns={"away_team": "t"})]
                          ).groupby(["league", "t"])["date"].max().to_dict()
        from src import league_registry as _LR
        for lg in sorted(set(out["league"].astype(str))):
            for x, y in _LR.same_club_spellings(lg):
                if (lg, x) in _last and (lg, y) in _last:
                    canon, alias = sorted((x, y), key=lambda t: _last[(lg, t)], reverse=True)
                    one_side.setdefault(lg, {})[alias] = canon
        _alias_store_merges(out, one_side, _last)
        out, rep = canonicalise_and_dedupe(out, alias_map=one_side)
        # the map actually applied (explicit + derived), for sources keyed on the pre-identity
        # spelling (the HT sidecar) - Phase 2 HT repair, 2026-11-03
        IDENTITY_ALIASES.clear()
        IDENTITY_ALIASES.update({lg: dict(m) for lg, m in rep.get("aliases", {}).items() if m})
        log.info(f"[identity] one-side same-day aliases: "
                 f"{sum(len(m) for m in one_side.values())} in {len(one_side)} league(s)")
        _ev = [(lg, a, c, n) for lg, m in sorted(_EVIDENCE_ONLY_ALIASES.items())
               for a, (c, n) in sorted(m.items())]
        if _ev:
            log.warning(f"[identity] {len(_ev)} alias(es) proven by same-day evidence only "
                        f"(team_names cannot spell-match them; add to the alias store): "
                        + "; ".join(f"{lg}: {a} -> {c} ({n} dates)" for lg, a, c, n in _ev[:12]))
        log.info(f"[identity] {rep['n_aliases']} club aliases, {rep['team_cells_renamed']:,} names "
                 f"unified, {rep['duplicate_rows_removed']:,} duplicate fixtures removed, "
                 f"{rep['quarantined_fixtures']} score-conflict fixture(s) quarantined "
                 f"({before:,} -> {len(out):,} rows)")
        return out
    except Exception as e:                                             # noqa: BLE001
        log.error(f"[identity] canonical identity/season step FAILED ({type(e).__name__}: {e}) — "
                  f"history left as merged")
        return out


def _repair_club_text(df: pd.DataFrame) -> pd.DataFrame:
    """Club names whose UTF-8 bytes were decoded as Latin-1 ('PreuÃ\\x9fen MÃ¼nster'), repaired to
    the real text ('Preußen Münster') - in this READ only; the banks keep what they hold. The download
    now decodes the bytes itself (src.integrity.duplicate_matches.decode_csv_bytes), so this only
    cleans rows banked before that fix (68 fd_history + 78 af_history rows at data-truth-100 HEAD)."""
    from src.integrity.duplicate_matches import fix_mojibake
    if df is None or df.empty:
        return df
    df = df.copy()
    n = 0
    for c in ("home_team", "away_team"):
        if c in df.columns:
            new = df[c].map(fix_mojibake)
            n += int((new.astype(str) != df[c].astype(str)).sum())
            df[c] = new
    if n:
        log.warning(f"[clubs] {n} club name(s) were mojibake (UTF-8 read as Latin-1) - repaired")
    return df


def _one_match_one_row(df: pd.DataFrame) -> pd.DataFrame:
    """The last de-duplication: one row per match across sources and dates (see
    src.integrity.duplicate_matches). The two steps above key on the EXACT date, so the same MLS /
    Argentina / Brazil / Mexico fixture filed under its local date by one source and the UTC date by
    another (1-3 days apart) stayed twice, and the earlier copy sat inside the later copy's own
    last-5 form window. Survivor by the stated source rule (SOURCE_PRIORITY, then completeness, then
    earliest); counts logged."""
    from src.integrity.duplicate_matches import dedupe_matches
    if df is None or df.empty or not {"home_goals", "away_goals"}.issubset(df.columns):
        return df
    try:
        out, rep = dedupe_matches(df, source_col="row_source")
    except Exception as e:                                             # noqa: BLE001
        log.error(f"[one-match] de-duplication FAILED ({type(e).__name__}: {e}) - frame left as is")
        return df
    if rep["rows_removed"]:
        log.warning(f"[one-match] removed {rep['rows_removed']:,} duplicate match row(s): "
                    f"{rep['same_day_pairs']} same-day (of which {rep['one_side_same_day_pairs']} "
                    f"with one side spelled differently) + {rep['near_date_pairs']} 1-3 day "
                    f"pair(s); {rep['cells_filled_from_copy']} empty cell(s) filled from the "
                    f"dropped copy; e.g. {rep['examples'][:3]}")
    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    out.attrs["one_match_report"] = {k: v for k, v in rep.items() if k != "examples"}
    return out


def _apply_history_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """src.history_corrections.apply_history_corrections, never fatal: an unreadable list logs an
    error and the history is served uncorrected (the data-truth checks still see the duplicates)."""
    try:
        from src.history_corrections import apply_history_corrections
        return apply_history_corrections(df, tag="history-corrections")
    except Exception as e:                                             # noqa: BLE001
        log.error(f"[history-corrections] NOT applied ({type(e).__name__}: {e})")
        return df


def one_club_one_match(out: pd.DataFrame) -> pd.DataFrame:
    """THE identity pipeline of the merged history, in ONE place - load_all_matches (train and serve)
    and the data-truth check (src/integrity/data_truth.one_match) both call it: mojibake repaired,
    one spelling per club (_canonicalise_clubs, _canonical_identity_and_seasons), one row per match
    (_one_match_one_row: +-3 days, canonical clubs, same score).

    FIRST the owner's history corrections (2026-11-04, src/history_corrections.py): proven domestic-cup
    rows filed under a league are excluded ("cups are not in Wowza tips"), and the proven second copy
    of a match filed one day apart is removed (the provider's fixture date decides which copy stays).
    They run on the history's OWN key and spelling, so before any club is renamed."""
    out = _apply_history_corrections(out)
    out = _repair_club_text(out)
    out = _canonicalise_clubs(out)
    out = _canonical_identity_and_seasons(out)
    return _one_match_one_row(out)


def _impossible_stats_to_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Shot counts that cannot be true, set MISSING (never corrected, never a default), counted and
    logged (CLAUDE_PROMPT_2 P2 'v9 training stats'):
      * shots < shots on target (that side's shots and SOT -> missing),
      * shots < goals           (that side's shots and SOT -> missing; an own goal needs no shot,
                                 but then the SOT count is still unproven, so both go),
      * both sides 0 shots and 0 SOT on a row that is otherwise a played match (a placeholder, not a
        count) -> all four missing.
    Shot ratios are recomputed from what remains."""
    need = {"home_shots", "away_shots", "home_sot", "away_sot", "home_goals", "away_goals"}
    if df is None or df.empty or not need.issubset(df.columns):
        return df
    counts = {}
    zero = ((pd.to_numeric(df["home_shots"], errors="coerce") == 0)
            & (pd.to_numeric(df["away_shots"], errors="coerce") == 0)
            & ~(pd.to_numeric(df["home_sot"], errors="coerce") > 0)
            & ~(pd.to_numeric(df["away_sot"], errors="coerce") > 0))
    for side in ("home", "away"):
        sh = pd.to_numeric(df[f"{side}_shots"], errors="coerce")
        so = pd.to_numeric(df[f"{side}_sot"], errors="coerce")
        g = pd.to_numeric(df[f"{side}_goals"], errors="coerce")
        lt_sot = (sh < so) & ~zero
        lt_goals = (sh < g) & ~lt_sot & ~zero
        counts[f"{side}_shots_lt_sot"] = int(lt_sot.sum())
        counts[f"{side}_shots_lt_goals"] = int(lt_goals.sum())
        df.loc[lt_sot | lt_goals, [f"{side}_shots", f"{side}_sot"]] = np.nan
    counts["all_zero_shot_rows"] = int(zero.sum())
    df.loc[zero, ["home_shots", "away_shots", "home_sot", "away_sot"]] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        for side in ("home", "away"):
            sh, so = df[f"{side}_shots"], df[f"{side}_sot"]
            df[f"{side}_sot_ratio"] = np.where(sh > 0, so / sh, np.nan)
    if any(counts.values()):
        log.warning(f"[stats] impossible shot counts set MISSING: {counts}")
    df.attrs["impossible_stats"] = counts
    return df


def _flag_ou25_pair(df: pd.DataFrame) -> pd.DataFrame:
    """`ou25_pair_is_price`: False where the O/U 2.5 pair has a NEGATIVE margin (1/over + 1/under < 1).
    No single book quotes that - it is two best prices from different books (427 of the 432 such
    pairs are new-format, built per side from newformat_odds_history). The prices themselves stay
    (each is a real quote); the pair is NOT a price for the overround feature, which must read it as
    missing, never clip it to an invented 0 (src.integrity.price_provenance.overround_or_missing).
    NaN where either side is unpriced."""
    if df is None or df.empty or not {"odds_over25", "odds_under25"}.issubset(df.columns):
        return df
    from src.integrity.price_provenance import overround_or_missing
    o = pd.to_numeric(df["odds_over25"], errors="coerce")
    u = pd.to_numeric(df["odds_under25"], errors="coerce")
    priced = (o > 0) & (u > 0)
    ok = overround_or_missing(o, u).notna()
    df["ou25_pair_is_price"] = ok.astype("boolean").where(priced)
    n = int((priced & ~ok).sum())
    if n:
        log.warning(f"[odds] {n} O/U 2.5 pair(s) of {int(priced.sum()):,} have a negative margin - "
                    f"flagged ou25_pair_is_price=False (missing for the overround feature)")
    return df


def _fd_download_tables() -> tuple[dict, dict]:
    """({league: code} for per-season mmz4281 files, {league: code} for /new/ files), from the
    registry: every MODEL league with a football_data_code, standard -> mmz4281, new_format -> /new/.
    A league with model_type null is skipped here (it is not a model league and was being filed in
    the /new/ bucket by elimination); the HT-extra download reads those by football_data_format."""
    from src import league_registry as _LR
    std, new = {}, {}
    for e in _LR.leagues():
        code = e.get("football_data_code")
        if not code:
            continue
        if e.get("model_type") == "standard":
            std[e["name"]] = code
        elif e.get("model_type") == "new_format":
            new[e["name"]] = code
    return std, new


def _drop_misfiled_league_rows(d: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rows filed under the WRONG league, removed before they reach training, serving or backtests.

    Two proven defects (Data Truth audit 2026-11-01, findings 4/5/31):
      * a league that is not in config/leagues.json at all - 'Denmark 1st Div' holds 2,041
        Bundesliga 2 fixtures (Bielefeld, Bochum, Dresden...) a second time;
      * a split-season label ('2023/24') in a CALENDAR-YEAR league - every such 'Norway
        Eliteserien' row is a Dutch Eredivisie fixture (Ajax, PSV, AZ...), 1,800-2,106 of them,
        which made Dutch football 45% of the Norway history behind serving, backtests and gates.
    Registry aliases are fine (they resolve). Every drop is logged with its count."""
    if d is None or d.empty or "league" not in d.columns:
        return d
    from src import league_registry as _LR
    canon = {lg: _LR.canonical_name(lg) for lg in d["league"].astype(str).unique()}
    cal = {e["name"] for e in _LR.leagues()
           if (e.get("season_rule") or {}).get("type") == "calendar_year"}
    lg = d["league"].astype(str)
    unknown = lg.map(canon).isna()
    split_in_cal = (lg.map(canon).isin(cal)
                    & d.get("season", pd.Series("", index=d.index)).astype(str).str.contains("/"))
    bad = unknown | split_in_cal
    if bad.any():
        what = (d.loc[bad, "league"].astype(str)
                + np.where(unknown[bad], " (not in registry)", " (split label in calendar-year league)"))
        log.warning(f"[{source}] dropped {int(bad.sum()):,} misfiled row(s): "
                    f"{what.value_counts().to_dict()}")
    return d[~bad]


def _load_backtest_archive() -> pd.DataFrame:
    """Three extra seasons we already own, in a file nothing has ever read.

    `output/backtest_all_leagues.csv` is written by the monthly backtest and, until now, read by
    no line of code in any of the three repos -- a plain text search returns zero matches. It
    holds 36,796 completed fixtures back to 2020 with corners, fouls, shots and O/U prices at
    100%, against a loader whose standard-format leagues only reach 2023-08. Roughly 20,600 of
    those fixtures are in no other source.

    Measured before adding it (Pro's shadow-learning study, identical test fixtures): the extra
    history significantly improves Over 2.5 and Over 3.5 and significantly HURTS BTTS, and every
    attempt to shrink or down-weight history lost. The net is small and positive. It is added
    because it is free, already paid for, and makes the history complete -- not because it will
    transform prediction.

    WHY THIS IS SAFE NOW AND WAS NOT AN HOUR AGO. The file names clubs differently ("FC Koln"
    where football-data says "1. FC Köln"), so merging it used to mean either a fuzzy match in
    production -- forbidden -- or a hand-maintained mapping table. `_canonicalise_clubs` now runs
    over the merged frame and resolves those within-league spelling variants on evidence, so the
    archive can simply be appended and the unifier absorbs the vocabulary. Anything it cannot
    resolve stays a separate team, which is the old behaviour, not a regression.

    Set WOWZA_USE_BACKTEST_ARCHIVE=0 to turn this off; it is one variable to revert.
    """
    if os.getenv("WOWZA_USE_BACKTEST_ARCHIVE", "1") != "1":
        log.info("[archive] disabled by WOWZA_USE_BACKTEST_ARCHIVE=0")
        return pd.DataFrame()
    p = Path(__file__).resolve().parents[1] / "output" / "backtest_all_leagues.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(p, low_memory=False)
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"[archive] could not read {p.name}: {e}")
        return pd.DataFrame()
    keep = ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals",
            "home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
            "home_fouls", "away_fouls", "ftr", "odds_over25", "odds_under25"]
    cols = [c for c in keep if c in raw.columns]
    d = raw[cols].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "home_goals", "away_goals", "home_team", "away_team"])
    for c in cols:
        if c not in ("date", "league", "season", "home_team", "away_team", "ftr"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = _drop_misfiled_league_rows(d, "archive")
    log.info(f"[archive] {len(d):,} rows from backtest_all_leagues.csv "
             f"({d['date'].min().date()}..{d['date'].max().date()}, "
             f"{d['league'].nunique()} leagues)")
    return d


def _report_training_coverage(out: pd.DataFrame) -> None:
    """Write output/training_coverage.json — what the models are actually about to train on.

    THIS EXISTS BECAUSE THE GAP WAS INVISIBLE FOR FOUR SEASONS. Corners, half-time goals and the
    O/U 2.5 price were all missing from every finished season of the seven leagues we bet, and
    nothing failed: the download succeeded, the retrain succeeded, the gate promoted models, and
    the coverage was simply never looked at. A silent absence is worse than an error because
    there is nothing to notice.

    So: one small committed file per run, per league, per column. A regression shows up as a
    number that fell, and the half-time markets in particular have somewhere to point when they
    behave oddly -- ht_over05 was rejected by the promotion gate on 2026-09-22 with log loss
    rising 0.0886, on a track whose training data was missing half-time scores for four seasons
    of seven leagues.
    """
    try:
        cols = [c for c in ("home_corners", "ht_home_goals", "odds_over25", "odds_btts",
                            "odds_over15", "odds_over35", "home_shots", "home_sot",
                            "home_fouls") if c in out.columns]
        rows = {}
        for lg, g in out.groupby("league"):
            rows[str(lg)] = {"rows": int(len(g)),
                             "first": str(pd.to_datetime(g["date"]).min())[:10],
                             "last": str(pd.to_datetime(g["date"]).max())[:10],
                             **{c: round(float(g[c].notna().mean()), 4) for c in cols}}
        thin = sorted(lg for lg, r in rows.items()
                      if r.get("home_corners", 1.0) < 0.5 or r.get("ht_home_goals", 1.0) < 0.5)
        payload = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                   "total_rows": int(len(out)), "leagues": rows,
                   "thin_on_corners_or_halftime": thin}
        p = Path(__file__).resolve().parents[1] / "output" / "training_coverage.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if thin:
            log.warning(f"[coverage] {len(thin)} league(s) below 50% on corners or half-time "
                        f"goals: {thin[:8]}")
        log.info(f"[coverage] wrote training_coverage.json ({len(rows)} leagues, "
                 f"{len(out):,} rows)")
    except Exception as e:                                           # noqa: BLE001
        # Never fatal: a missing report is a missing report, not a failed training run.
        log.warning(f"[coverage] could not write training_coverage.json: {e}")


def _ci_download_all(have: set | None = None) -> list[pd.DataFrame]:
    """
    CI / GitHub Actions fallback: download current + recent seasons directly
    from football-data.co.uk when local Excel/CSV files are not available.
    Returns list of DataFrames (one per league/season).

    `have` is a set of (league, season_label) pairs ALREADY in the committed cache. Those are
    skipped outright rather than re-downloaded.

    WHY: a finished season is immutable. The 2023/24 Championship has a final table, final
    scores and closing prices that will never change again — yet this function re-downloaded
    four seasons of ~27 leagues on EVERY CI run, several hundred requests to a free third-party
    server every few minutes, to rebuild a frame identical to the previous one. On 2026-09-05
    that server returned 503 and predict failed for 12+ hours with no tips, because the pipeline
    depended on re-fetching data it already had.

    Fetch once, keep it, and only ask for what is genuinely new. The CURRENT season is still
    downloaded every time (it gains fixtures weekly); everything older comes from the cache.
    """
    import requests
    from io import StringIO
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    STD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"
    # DERIVED, NOT HARDCODED. This list was ["2526","2425","2324","2223"], so its newest season
    # was 2025/26 — while the live season is 2026/27. retrain.py already auto-detects the season
    # from the date (_auto_season, "month >= 7 -> new season"); this did not, so the loader could
    # never fetch the season currently being played, and its "current season" branch was really
    # refreshing last year's finished data.
    #
    # Same failure shape as COLLECT_SEASONS freezing at "2025" while PROP_SEASONS rolled to
    # "2026" (root CLAUDE.md): a season-keyed constant that someone must remember to bump, and
    # nobody does. Derive it and the July roll happens by itself.
    _now = __import__("datetime").datetime.now()
    _start = _now.year if _now.month >= 7 else _now.year - 1
    SEASONS = [f"{str(y)[2:]}{str(y + 1)[2:]}" for y in range(_start, _start - 4, -1)]

    # DERIVED from config/leagues.json (football_data_code + model_type), never a hand copy: the
    # copies here had drifted from the registry and left out Romanian Superliga (ENABLED and tipped)
    # plus Scottish League One/Two, so their football-data history was never downloaded (Data Truth
    # audit finding 45). Standard leagues use the per-season mmz4281 files, new-format the one file.
    std_leagues, new_leagues = _fd_download_tables()

    frames = []

    # Standard format (multiple seasons)
    have = have or set()
    skipped = 0
    # CURRENT season only is ever re-fetched; SEASONS[0] is the live one.
    for league, code in std_leagues.items():
        for season in SEASONS:
            season_label = f"20{season[:2]}/{season[2:]}"
            # Already cached AND finished -> never ask again. The live season is always fetched.
            if season != SEASONS[0] and (league, season_label) in have:
                skipped += 1
                continue
            url = STD_URL.format(season=season, code=code)
            try:
                r = requests.get(url, timeout=15, headers=HEADERS)
                if r.status_code != 200:
                    # Named, never silent: a 404/503 used to drop this league-season from
                    # training and serving history with no trace (Data Truth audit finding 43).
                    log.warning(f"[data] {league} {season_label}: HTTP {r.status_code} — not in history this run")
                    continue
                # The BYTES, decoded here: r.text guessed Latin-1 for UTF-8 files (mojibake names)
                raw = pd.read_csv(StringIO(_decode_csv(r.content)),
                                  on_bad_lines="skip", low_memory=False)
                df = pd.DataFrame()
                df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
                df = df[df["date"].notna()].copy()
                if df.empty:
                    continue
                df["league"] = league
                df["season"] = season_label
                df["home_team"] = raw.get("HomeTeam", np.nan)
                df["away_team"] = raw.get("AwayTeam", np.nan)
                for out_col, src_col in [
                    ("home_goals","FTHG"),("away_goals","FTAG"),
                    ("ht_home_goals","HTHG"),("ht_away_goals","HTAG"),
                    ("home_corners","HC"),("away_corners","AC"),
                    ("home_fouls","HF"),("away_fouls","AF"),
                    ("home_shots","HS"),("away_shots","AS"),
                    ("home_sot","HST"),("away_sot","AST"),
                ]:
                    df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
                df["odds_over25"]  = _pick_odds(raw, _OVER_COLS)
                df["odds_under25"] = _pick_odds(raw, _UNDER_COLS)
                df["odds_btts"]    = _pick_odds(raw, _BTTS_COLS)
                df["odds_over15"]  = _pick_odds(raw, _OVER15_COLS)
                df["odds_over35"]  = _pick_odds(raw, _OVER35_COLS)
                df["ftr"] = raw.get("FTR", np.nan)
                df = df[df["home_team"].notna() & df["away_team"].notna()]
                frames.append(df)
            except Exception as e:
                # Silently skipping here means a league-season quietly vanishes from TRAINING
                # data and the model is refit on less history with nothing to show for it.
                log.warning(f"[data] {league} {season_label} skipped ({type(e).__name__}: {e})")
                continue

    # New format (single file, all seasons)
    for league, code in new_leagues.items():
        url = NEW_URL.format(code=code)
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            if r.status_code != 200:
                log.warning(f"[data] {league}: HTTP {r.status_code} — not in history this run")
                continue
            raw = pd.read_csv(StringIO(_decode_csv(r.content)),
                              on_bad_lines="skip", low_memory=False)
            df = pd.DataFrame()
            df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
            # The same window as the standard leagues (the last 4 campaigns, rolling by itself),
            # by DATE. This was the literal Season filter '2022|2023|2024|2025|2026', which drops
            # every row from the 2027 calendar seasons on (Data Truth audit findings 32, 36, 46).
            df = df[df["date"] >= pd.Timestamp(f"{_start - 4}-01-01")].copy()
            if df.empty:
                continue
            df["league"] = league
            # By INDEX, never by position: `.values[:len(df)]` after dropping rows shifted every
            # label onto the wrong fixture, and a file with no Season column was told '2025/26'.
            if "Season" in raw.columns:
                df["season"] = raw.loc[df.index, "Season"].astype(str).to_numpy()
            else:
                log.warning(f"[data] {league}: no Season column — season left empty, not invented")
                df["season"] = np.nan
            df["home_team"] = raw.get("Home", np.nan)
            df["away_team"] = raw.get("Away", np.nan)
            for out_col, src_col in [
                ("home_goals","HG"),("away_goals","AG"),
            ]:
                df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
            # Real O/U + BTTS odds from the historical backfill (NOT 1X2 — that was the bug).
            _nfo   = _nf_real_odds()
            _days  = df["date"].dt.strftime("%Y-%m-%d")
            _mkeys = df["home_team"].astype(str) + " vs " + df["away_team"].astype(str)
            _rws   = [_nf_odds_lookup(_nfo, d, league, h, a) for d, h, a in
                      zip(_days, df["home_team"].astype(str), df["away_team"].astype(str))]
            df["odds_over25"]  = [r.get("over25",   np.nan) for r in _rws]
            df["odds_under25"] = [r.get("under25",  np.nan) for r in _rws]
            df["odds_over15"]  = [r.get("over15",   np.nan) for r in _rws]
            df["odds_over35"]  = [r.get("over35",   np.nan) for r in _rws]
            df["odds_btts"]    = [r.get("btts_yes", np.nan) for r in _rws]
            df = df[df["home_team"].notna() & df["away_team"].notna()]
            frames.append(df)
        except Exception as e:
            # Same trap as the standard loop above: a new-format league dropping out of
            # training used to be completely invisible.
            log.warning(f"[data] {league} (new-format) skipped ({type(e).__name__}: {e})")
            continue

    log.info(f"CI download: {len(frames)} league/season file(s) loaded, "
             f"{skipped} finished season(s) served from cache instead of re-fetched")
    return frames


def _load_api_football_only_leagues() -> list[pd.DataFrame]:
    """
    Load historical fixture results for leagues not on football-data.co.uk.
    Uses API-Football `/fixtures` endpoint, permanently cached.
    Returns DataFrames with the same columns as the FD loader output.
    """
    import os as _os
    api_key = _os.getenv("APIFOOTBALL_KEY", "") or config.API_KEY
    if not api_key:
        return []

    if not hasattr(config, "API_FOOTBALL_ONLY_LEAGUES"):
        return []

    try:
        from src.api_football_ou import fetch_league_fixture_results
    except ImportError:
        try:
            from api_football_ou import fetch_league_fixture_results
        except ImportError:
            return []

    frames = []
    for league in config.API_FOOTBALL_ONLY_LEAGUES:
        league_id = config.API_FOOTBALL_IDS.get(league)
        if not league_id:
            continue
        seasons = config.API_FOOTBALL_EXTRA_SEASONS.get(league, [])
        for season_year in seasons:
            try:
                df = fetch_league_fixture_results(league_id, season_year, league)
                if df.empty:
                    continue
                df["league"] = league
                df["season"] = season_year
                df["ftr"] = np.where(
                    df["home_goals"] > df["away_goals"], "H",
                    np.where(df["home_goals"] < df["away_goals"], "A", "D"),
                )
                for col in ["home_shots", "away_shots", "home_sot", "away_sot",
                            "home_corners", "away_corners", "home_fouls", "away_fouls",
                            "odds_over25", "odds_under25",
                            "odds_btts", "odds_over15", "odds_over35"]:
                    if col not in df.columns:
                        df[col] = np.nan
                frames.append(df)
            except Exception as e:
                log.warning(f"[api_football] {league} {season_year}: {e}")

    if frames:
        log.info(f"[api_football] Loaded {sum(len(f) for f in frames)} rows "
                 f"from {len(frames)} API-Football-only league/seasons")
        # Merge historical BTTS/O1.5/O3.5 odds backfill into new-format rows
        combined = pd.concat(frames, ignore_index=True)
        combined = _enrich_with_af_odds(combined)
        return [combined]
    return frames


def _enrich_with_af_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill odds_btts / odds_over15 / odds_over35 for new-format leagues using
    the API-Football historical odds backfill (output/af_odds_history.parquet).
    Only fills NaN cells — never overwrites existing odds.
    """
    af_odds_path = Path(__file__).resolve().parents[1] / "output" / "af_odds_history.parquet"
    if not af_odds_path.exists():
        return df

    try:
        odds_hist = pd.read_parquet(af_odds_path)
        # League-keyed or nothing: the old (home, away, date) lookup had no league and reindexed
        # positionally (Data Truth 2026-11-01).
        if "league" not in odds_hist.columns or "league" not in df.columns:
            log.error("[af_odds] af_odds_history has no league column — enrichment SKIPPED "
                      "(a join without league can put prices on another league's fixture)")
            return df
        odds_hist["date"] = pd.to_datetime(odds_hist["date"], errors="coerce")
        odds_hist = _resolve_onto_history(odds_hist, df, "home_team", "away_team", "af_odds")
        odds_hist["_d"] = odds_hist["date"].dt.strftime("%Y-%m-%d")

        df["_h"] = df["home_team"].astype(str)
        df["_a"] = df["away_team"].astype(str)
        df["_d"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        _vc = [c for c in ("odds_btts", "odds_over15", "odds_over35")
               if c in odds_hist.columns and c in df.columns]
        df, _rep = _IJ.safe_left_enrich(df, odds_hist, keys=["league", "_h", "_a", "_d"],
                                        value_cols=_vc, name="af_odds_history")
        log.info(f"  [af_odds] SAFE join filled {_rep.filled_cells}")
        df = df.drop(columns=["_h", "_a", "_d"])
    except Exception as e:
        log.warning(f"[af_odds] Could not merge odds history: {e}")

    return df


def _enrich_with_standard_sidemarket_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Fill odds_btts / odds_over15 / odds_over35 for STANDARD 2nd-division leagues from the
    real backfill output/standard_sidemarket_odds_history.csv. football-data.co.uk carries no
    BTTS/O1.5/O3.5 columns for the 2nd-divs, so those were synthetic (a flat ~1.85) — this plugs
    in REAL prices so the BTTS / side-market backtests reflect what a book would actually pay.
    NaN-only fill; NEVER touches odds_over25 — the standard O/U money-market stays 100%
    football-data, so the O/U backtest is provably unchanged. [v10 2026-07-13]"""
    path = Path(__file__).resolve().parents[1] / "output" / "standard_sidemarket_odds_history.csv"
    if not path.exists():
        return df
    try:
        _MKT = {"btts_yes": "odds_btts", "over15": "odds_over15", "over35": "odds_over35"}
        oh = pd.read_csv(path)
        oh = oh[oh["market"].isin(_MKT)].copy()
        if oh.empty:
            return df

        # ── DROP FIRST-HALF BTTS PRICES MISLABELLED AS FULL-MATCH (found 2026-08-23) ──────
        #
        # The capture parser matched the bet NAME by substring, so "Both Teams Score - First
        # Half" (bet 34, Yes ~5.50) overwrote "Both Teams Score" (bet 8, Yes ~1.91) whenever
        # Bet365 offered both. The parser is fixed, so no NEW contamination arrives — but 271 of
        # 6,358 historical btts_yes rows (4.3%, 2026-08-06 to 08-23) are first-half prices, and
        # `aggfunc="last"` below would feed them straight into odds_btts for training and the
        # BTTS backtest. A 6.0 price where the truth is 1.9 does not look like an outlier to any
        # downstream check; it looks like a longshot with a huge edge.
        #
        # THE TWO POPULATIONS DO NOT OVERLAP, so this filter cannot catch a real price:
        #     clean full-match btts_yes   1.34 - 2.97
        #     contaminated (first half)   3.40 - 7.00
        #     observations in 0.31-0.33 implied (odds 3.03-3.22):  ZERO
        # 3.20 sits inside that empty gap. A full-match BTTS-YES at 3.20 implies a 31% chance
        # both teams score across 90 minutes, which does not occur in professional football.
        #
        # Rows are DROPPED FROM THIS READ ONLY. Nothing is deleted from the CSV — the raw
        # capture stays intact so the contamination remains auditable.
        _BTTS_YES_MAX = 3.20
        _bad = (oh["market"] == "btts_yes") & (pd.to_numeric(oh["odds"], errors="coerce")
                                               > _BTTS_YES_MAX)
        if _bad.any():
            log.warning(f"  [std_sidemarket] dropped {int(_bad.sum())} btts_yes row(s) with "
                        f"odds > {_BTTS_YES_MAX} — first-half BTTS mislabelled as full-match "
                        f"(see 2026-08-23 parser fix). Raw file unchanged.")
            oh = oh[~_bad].copy()
        if oh.empty:
            return df
        if "league" not in oh.columns:
            log.error("[std_sidemarket] capture file has no league column — enrichment SKIPPED")
            return df
        parts = oh["match"].astype(str).str.split(" vs ", n=1, expand=True)
        oh["_hraw"] = parts[0]
        oh["_araw"] = parts[1] if parts.shape[1] > 1 else None
        oh = _resolve_onto_history(oh, df, "_hraw", "_araw", "std_sidemarket")
        oh["_d"] = oh["match_date"].astype(str).str[:10]
        oh["_col"] = oh["market"].map(_MKT)

        df["_h"] = df["home_team"].astype(str)
        df["_a"] = df["away_team"].astype(str)
        df["_d"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        try:   # R2-C: league-free key — audited (output/join_integrity_health.json)
            _IJ.audit_league_free_lookup(df, oh, ["_h", "_a", "_d"],
                                         name="std_sidemarket_odds.legacy")
        except Exception as _e:                                  # noqa: BLE001
            log.warning(f"[std_sidemarket] join audit skipped: {_e}")
        # NOTE: over25 deliberately excluded (the standard O/U money-market stays football-data)
        _w = oh.sort_values("snapshot_ts") if "snapshot_ts" in oh.columns else oh
        _w = (_w.drop_duplicates(["league", "_h", "_a", "_d", "_col"], keep="last")
                .pivot(index=["league", "_h", "_a", "_d"], columns="_col", values="odds")
                .reset_index())
        _vc = [c for c in ("odds_btts", "odds_over15", "odds_over35")
               if c in _w.columns and c in df.columns]
        df, _ = _IJ.safe_left_enrich(df, _w, keys=["league", "_h", "_a", "_d"],
                                     value_cols=_vc, name="std_sidemarket_odds")
        df = df.drop(columns=["_h", "_a", "_d"])
    except Exception as e:
        log.warning(f"[std_sidemarket] merge failed: {e}")
    return df


def _enrich_ou25_from_captures(df: pd.DataFrame) -> pd.DataFrame:
    """Fill odds_over25 / odds_under25 from OUR OWN forward odds captures, NaN-only.

    WHY THIS EXISTS. odds_over25 was deliberately excluded from every other enrichment so the
    O/U money-market stayed 100% football-data and its backtest was provably unchanged. That was
    the right call while football-data worked. On 2026-09-05 it returned 503 for many hours, the
    loader fell back to a cache with no prices, and retrain found 0 of 31,068 rows priced — it
    could not backtest at all, so it could not safely retrain.

    Meanwhile we hold the prices ourselves. The three forward captures carry over25 AND under25
    for 4,280 distinct fixtures, from 2024-08 up to the current board:

        newformat_odds_history.csv             3,610 fixtures
        standard_sidemarket_odds_history.csv     391
        standard_odds_history.csv                365

    These are the LATEST fixtures — the ones retrain most needs to learn from. football-data's
    unique contribution is old prices for old matches: volume, not signal.

    NaN-ONLY, deliberately. Where football-data has a price it still wins, so when the site
    returns the O/U backtest reverts to exactly what it was and remains comparable across runs.
    This only ever fills gaps that would otherwise drop the row entirely.

    The LAST snapshot per fixture is taken — the closest thing to a closing price we captured.
    """
    paths = [Path(__file__).resolve().parents[1] / "output" / n for n in
             ("standard_odds_history.csv", "standard_sidemarket_odds_history.csv",
              "newformat_odds_history.csv")]
    frames = []
    for p in paths:
        if not p.exists():
            continue
        try:
            o = pd.read_csv(p, low_memory=False)
        except Exception:                                        # noqa: BLE001
            continue
        if not {"market", "odds", "match", "match_date"}.issubset(o.columns):
            continue
        o = o[o["market"].isin(("over25", "under25"))]
        o = _drop_mismapped(o, p.stem, "ou25_captures")
        if len(o):
            frames.append(o)
    if not frames:
        return df
    try:
        oh = pd.concat(frames, ignore_index=True)
        if "league" not in oh.columns:
            log.error("[ou25_captures] captures have no league column — enrichment SKIPPED")
            return df
        # Sort so keep="last" really means the latest snapshot we hold for that fixture.
        if "snapshot_ts" in oh.columns:
            oh = oh.sort_values("snapshot_ts")
        # OddsAPI-vs-history is exactly the boundary team_names.resolve exists for: each capture
        # club is resolved among the history clubs of the capture's own league ("Colchester
        # United" -> "Colchester"); a club no rule proves is skipped and logged. This replaces
        # the old _norm_name exact key plus a resolver second pass (2026-11-02).
        parts = oh["match"].astype(str).str.split(" vs ", n=1, expand=True)
        oh["_hraw"] = parts[0]
        oh["_araw"] = parts[1] if parts.shape[1] > 1 else None
        oh = _resolve_onto_history(oh, df, "_hraw", "_araw", "ou25_captures")
        oh["_d"] = oh["match_date"].astype(str).str[:10]
        oh["_col"] = oh["market"].map({"over25": "odds_over25", "under25": "odds_under25"})

        df["_h"] = df["home_team"].astype(str)
        df["_a"] = df["away_team"].astype(str)
        df["_d"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        try:   # R2-C: league-free key — audited (output/join_integrity_health.json)
            _IJ.audit_league_free_lookup(df, oh, ["_h", "_a", "_d"],
                                         name="ou25_captures.legacy")
        except Exception as _e:                                  # noqa: BLE001
            log.warning(f"[ou25_captures] join audit skipped: {_e}")
        for col in ("odds_over25", "odds_under25"):
            if col not in df.columns:
                df[col] = np.nan
        _w = (oh.drop_duplicates(["league", "_h", "_a", "_d", "_col"], keep="last")
                .pivot(index=["league", "_h", "_a", "_d"], columns="_col", values="odds")
                .reset_index())
        _vc = [c for c in ("odds_over25", "odds_under25") if c in _w.columns]
        df, _rep = _IJ.safe_left_enrich(df, _w, keys=["league", "_h", "_a", "_d"],
                                        value_cols=_vc, name="ou25_captures")
        log.info(f"  [ou25_captures] SAFE join filled {_rep.filled_cells}")

        df = df.drop(columns=["_h", "_a", "_d"])
    except Exception as e:                                       # noqa: BLE001
        log.warning(f"[ou25_captures] merge failed: {e}")
    return df


#: HT score sources, in fill order. Same key (league, date, home_team, away_team) and HTHG/HTAG
#: layout. fd_history_ht_fill.parquet is written beside the history by scripts/collect_ht_scores.py
#: (data_truth_collect.yml step 'htscores'; API-Football score.halftime by fixture id, a null HT
#: stored as missing, never 0) - fd_history itself is never rewritten. Either may be absent.
HT_SOURCES = (("af_ht_history", Path("output") / "af_ht_history.parquet"),
              ("fd_history_ht_fill", Path("output") / "collect" / "fd_history_ht_fill.parquet"))


def ht_coverage(df: pd.DataFrame, leagues=None) -> dict:
    """{league: share of PLAYED rows (full-time score present) carrying both HT goals}, for the
    registry's new-format leagues (or `leagues`)."""
    if df is None or df.empty or "league" not in df.columns:
        return {}
    if leagues is None:
        leagues = [lg for lg in sorted(set(df["league"].astype(str)))
                   if config.model_type_for_league(lg) == "new_format"]
    played = df["home_goals"].notna() & df["away_goals"].notna() if "home_goals" in df.columns \
        else pd.Series(True, index=df.index)
    has = (df["ht_home_goals"].notna() & df["ht_away_goals"].notna()) \
        if {"ht_home_goals", "ht_away_goals"} <= set(df.columns) else pd.Series(False, index=df.index)
    out = {}
    for lg in leagues:
        m = played & (df["league"].astype(str) == lg)
        if m.any():
            out[lg] = round(float(has[m].mean()), 4)
    return out


def _ht_labels(df: pd.DataFrame) -> pd.DataFrame:
    """ht_total_goals / ht_over05 / ht_over15 from the HT goals; NaN where either is missing.
    Same guard as the loader's construction sites: only when some row carries an HT score."""
    if not {"ht_home_goals", "ht_away_goals"} <= set(df.columns) or not df["ht_home_goals"].notna().any():
        return df
    df["ht_total_goals"] = df["ht_home_goals"] + df["ht_away_goals"]
    _htn = df["ht_total_goals"].notna()
    df["ht_over05"] = (df["ht_total_goals"] >= 1).astype(float).where(_htn)
    df["ht_over15"] = (df["ht_total_goals"] >= 2).astype(float).where(_htn)
    return df


def _enrich_ht_scores(df: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Fill MISSING ht_home_goals / ht_away_goals from HT_SOURCES (nan_only: a stored HT score is
    never overwritten), league-keyed, clubs resolved onto the history's own spellings
    (team_names.resolve; an unprovable name is skipped and logged). Then the HT labels are
    recomputed - they were built before this step, so a filled score never reached ht_over05.
    HT coverage per new-format league before/after is logged and kept in df.attrs['ht_coverage']."""
    root = root or Path(__file__).resolve().parents[1]
    before = ht_coverage(df)
    filled = {}
    for name, rel in HT_SOURCES:
        p = root / rel
        if not p.exists():
            log.info(f"[{name}] {rel.as_posix()} not present - skipped")
            continue
        try:
            ht = pd.read_parquet(p)
            if "league" not in ht.columns or not {"HTHG", "HTAG"} <= set(ht.columns):
                log.error(f"[{name}] no league / HTHG / HTAG column - HT enrichment SKIPPED "
                          "(a join without league can put scores on another league's fixture)")
                continue
            ht = ht[ht["HTHG"].notna() & ht["HTAG"].notna()].copy()      # a null HT is missing
            ht["date"] = pd.to_datetime(ht["date"], errors="coerce")
            # keyed on the history's own spelling: let the identity map the loader applied to df
            # place a club it renamed (Phase 2 HT repair: 2,303 of 33,259 sidecar rows were lost)
            ht = _resolve_onto_history(ht, df, "home_team", "away_team", name,
                                       aliases=IDENTITY_ALIASES)
            ht["_date_str"] = ht["date"].dt.strftime("%Y-%m-%d")
            df["_h"] = df["home_team"].astype(str)
            df["_a"] = df["away_team"].astype(str)
            df["_date_str"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            _k3 = ["_h", "_a", "_date_str"]
            try:
                _IJ.audit_legacy_merge(df, ht, _k3, name=f"{name}.legacy")
            except Exception as _e:                              # noqa: BLE001
                log.warning(f"[{name}] join audit skipped: {_e}")
            df, rep = _IJ.safe_left_enrich(
                df, ht, keys=["league"] + _k3, value_cols=["HTHG", "HTAG"],
                target_cols={"HTHG": "ht_home_goals", "HTAG": "ht_away_goals"}, name=name)
            filled[name] = int(rep.filled_cells.get("ht_home_goals", 0))
        except Exception as e:                                   # noqa: BLE001
            log.warning(f"[{name}] failed to merge {rel.as_posix()}: {e}")
        finally:
            df = df.drop(columns=["_h", "_a", "_date_str"], errors="ignore")
    df = _ht_labels(df)
    after = ht_coverage(df)
    df.attrs["ht_coverage"] = {"before": before, "after": after, "filled_rows": filled}
    moved = {lg: (before.get(lg, 0.0), v) for lg, v in after.items() if v != before.get(lg, 0.0)}
    log.info(f"[ht] HT scores filled {filled}; new-format coverage before -> after: "
             + ("; ".join(f"{lg} {a:.1%} -> {b:.1%}" for lg, (a, b) in moved.items()) or "unchanged"))
    return df


#: Fixture statistics collected BESIDE af_history.parquet (Data Truth round 6) by
#: scripts/collect_af_history_gap.py --insidebox: API-Football 'Shots insidebox' (HIB/AIB) and
#: expected_goals (HXG/AXG) for af_history matches that lack them, keyed by af_history's own
#: (league, date, home_team, away_team). af_history.parquet itself is never rewritten.
AF_STATS_FILL = Path("output") / "collect" / "af_history_stats_fill.parquet"
_AF_STATS_FILL_COLS = {"HIB": "home_insidebox", "AIB": "away_insidebox",
                       "HXG": "home_xg", "AXG": "away_xg"}


def _enrich_af_stats_fill(df: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Fill MISSING home/away_insidebox and home/away_xg from AF_STATS_FILL (nan_only, a stored value
    is never overwritten), league-keyed, clubs resolved onto df's spellings (team_names.resolve),
    ambiguous cells refused by safe_left_enrich. Part of _enrich_with_api_shots, so training and
    serving read the same history. Absent sidecar -> df unchanged."""
    p = (root or Path(__file__).resolve().parents[1]) / AF_STATS_FILL
    if not p.exists():
        log.info(f"[af_stats_fill] {AF_STATS_FILL.as_posix()} not present - skipped")
        return df
    try:
        fill = pd.read_parquet(p)
        vals = [c for c in _AF_STATS_FILL_COLS if c in fill.columns]
        if "league" not in fill.columns or not vals:
            log.error("[af_stats_fill] no league / stat column - fill SKIPPED")
            return df
        fill = fill[fill[vals].notna().any(axis=1)].copy()
        fill["date"] = pd.to_datetime(fill["date"], errors="coerce")
        fill = _resolve_onto_history(fill, df, "home_team", "away_team", "af_stats_fill")
        fill["_date_str"] = fill["date"].dt.strftime("%Y-%m-%d")
        df["_h"] = df["home_team"].astype(str)
        df["_a"] = df["away_team"].astype(str)
        df["_date_str"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df, rep = _IJ.safe_left_enrich(
            df, fill, keys=["league", "_h", "_a", "_date_str"], value_cols=vals,
            target_cols={c: _AF_STATS_FILL_COLS[c] for c in vals}, name="af_stats_fill")
        log.info(f"[af_stats_fill] filled {rep.filled_cells}; ambiguous key groups "
                 f"{rep.ambiguous_key_groups}")
    except Exception as e:                                       # noqa: BLE001
        log.warning(f"[af_stats_fill] failed to merge {AF_STATS_FILL.as_posix()}: {e}")
    finally:
        df = df.drop(columns=["_h", "_a", "_date_str"], errors="ignore")
    return df


def _enrich_with_api_shots(df: pd.DataFrame) -> pd.DataFrame:
    """
    For new-format leagues that have no shot data from football-data.co.uk,
    fill home_shots / away_shots / home_sot / away_sot from API-Football.
    Skips silently when API_KEY is not set or API returns nothing.
    """
    import os as _os
    api_key = _os.getenv("APIFOOTBALL_KEY", "") or config.API_KEY

    # Pre-initialize xG/insidebox so merge suffixes work correctly for new columns
    for _col in ["home_xg", "away_xg", "home_insidebox", "away_insidebox"]:
        if _col not in df.columns:
            df[_col] = np.nan

    # ── Step 1: merge from af_history.parquet (historical backfill, one-time run) ──
    af_history_path = Path(__file__).resolve().parents[1] / "output" / "af_history.parquet"
    if af_history_path.exists():
        try:
            hist = pd.read_parquet(af_history_path)
            hist = _drop_misfiled_league_rows(hist, "af_history")
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist = _resolve_onto_history(hist, df, "home_team", "away_team", "af_history")
            hist["_date_str"]  = hist["date"].dt.strftime("%Y-%m-%d")

            # Map af_history column names → internal names
            col_map = {
                "HS": "home_shots", "AS": "away_shots",
                "HST": "home_sot",  "AST": "away_sot",
                "HC": "home_corners", "AC": "away_corners",
                "HF": "home_fouls",   "AF": "away_fouls",
                "HY": "home_yellows", "AY": "away_yellows",
                # THE MISSING LINK for the four xG features. They were pre-initialised as NaN a
                # few lines above and this map had no entry to fill them, so every merge left
                # them empty and the model median-imputed a constant — 0 of 27,610 rows non-null
                # in the live store. backfill_af_history.py now writes HXG/AXG/HIB/AIB from the
                # same API response it already fetches; this is what carries them across.
                "HXG": "home_xg", "AXG": "away_xg",
                "HIB": "home_insidebox", "AIB": "away_insidebox",
            }
            hist = hist.rename(columns=col_map)
            hist_cols = [c for c in col_map.values() if c in hist.columns]

            df["_h"] = df["home_team"].astype(str)
            df["_a"] = df["away_team"].astype(str)
            df["_date_str"]  = df["date"].dt.strftime("%Y-%m-%d")

            # R2-C finding 12. The legacy join keyed on (home, away, date) WITHOUT league and
            # copied back POSITIONALLY; af_history holds duplicate keys, so the zip shifted values
            # onto the wrong fixture. Only the league-keyed safe join exists now (legacy path
            # removed 2026-11-01); the legacy shape is still AUDITED.
            _k3 = ["_h", "_a", "_date_str"]
            try:
                _IJ.audit_legacy_merge(df, hist, _k3, name="af_history_shots.legacy")
            except Exception as _e:                              # noqa: BLE001
                log.warning(f"[af_history] join audit skipped: {_e}")
            df, _rep = _IJ.safe_left_enrich(
                df, hist, keys=["league"] + _k3, value_cols=hist_cols,
                name="af_history_shots")
            log.info(f"[af_history] SAFE join: matched {_rep.matched_left_rows:,}/"
                     f"{_rep.left_rows:,}; ambiguous key groups {_rep.ambiguous_key_groups}; "
                     f"right duplicate rows {_rep.right_duplicate_rows}")

            df.drop(columns=["_h", "_a", "_date_str"], errors="ignore", inplace=True)
            n_with_shots = int(df["home_shots"].notna().sum()) if "home_shots" in df.columns else 0
            log.info(f"[af_history] merged backfill: {n_with_shots} rows now have shot data")
        except Exception as e:
            log.warning(f"[af_history] failed to merge af_history.parquet: {e}")

    # ── Step 1b: merge HT scores where they are missing ──────────────────────
    df = _enrich_ht_scores(df)

    if not api_key:
        return _enrich_af_stats_fill(df)

    from src.api_football_ou import fetch_shots_for_league   # import failure must surface

    # ALL LEAGUES PRESENT, NOT ONLY NEW-FORMAT. This loop was `for league in
    # config.NEW_FORMAT_LEAGUES`, which is why home_xg_last5 / away_xg_last5 /
    # home_insidebox_last5 / away_insidebox_last5 are 100% NULL in the live store — 0 of 27,610
    # rows. The standard-format second divisions are the market that actually takes money, and
    # they were the ones being skipped. FEATURE_COLS has declared those four features all along
    # and model.py median-imputes them, so the model has four constant inputs.
    #
    # Nothing else about the enrichment needed changing: api_football_ou already reads
    # expected_goals and 'Shots insidebox' correctly, and af_history supplies the historical half.
    # History and live were already combined; the live half just never looked at these leagues.
    #
    # RUNTIME RISK, STATED RATHER THAN DISCOVERED. On 2026-08-15 switching on dormant
    # API-Football enrichments took predict from 2-3 min to 10+ against its timeout. Two things
    # bound it here: the skip-if-populated guard below, and apifootball_ou_cache/ (44,453 files),
    # so a league-season is fetched once and then read from disk. Watch predict's runtime after
    # this ships; if it climbs, gate this loop to the train/backtest path rather than reverting
    # the fix.
    # GATED TO TRAIN/BACKTEST — NOT PREDICT.
    #
    # Expanding this loop to every league is right for the MODEL and wrong for PREDICT. It fetches
    # /fixtures/statistics per fixture, so covering ~10 extra standard leagues adds thousands of
    # calls to a job that runs every 15 minutes against a 25-minute timeout. Observed directly: a
    # predict run walking Championship, League One, League Two and the rest, hundreds of calls
    # deep, exactly the failure root CLAUDE.md records from 2026-08-15 when enrichment took
    # predict from 2-3 minutes to over 10.
    #
    # Predict does not need it: it scores TODAY's fixtures with an already-trained model, and its
    # xG features come from whatever the training data carried. Training and backtesting DO need
    # it, and they run weekly with no tip deadline.
    #
    # Opt in with WOWZA_FULL_ENRICH=1 (set by retrain/backtest workflows). Default is the old
    # new-format-only behaviour, which is what predict has always done.
    import os as _os
    if _os.getenv("WOWZA_FULL_ENRICH", "").strip() == "1":
        _leagues = [l for l in df["league"].dropna().unique()
                    if config.API_FOOTBALL_IDS.get(l)]
        log.info(f"[api_football_ou] FULL enrichment: {len(_leagues)} league(s)")
    else:
        _leagues = [l for l in config.NEW_FORMAT_LEAGUES if config.API_FOOTBALL_IDS.get(l)]
    for league in _leagues:
        league_id = config.API_FOOTBALL_IDS.get(league)
        if not league_id:
            continue

        mask = df["league"] == league
        if not mask.any():
            continue

        # Skip only if xG is already populated (shots from CSV is fine, but xG always comes from API)
        if "home_xg" in df.columns and df.loc[mask, "home_xg"].notna().mean() > 0.5:
            continue

        # League-aware current season (R2 finding 6). Was API_FOOTBALL_SEASONS.get(league, "2025"):
        # a league missing from that table (the training-only Standard leagues under
        # WOWZA_FULL_ENRICH) was fetched on the finished 2025 season. SEASON_UNKNOWN -> skip.
        from src.season_resolver import resolve_api_season, is_known, record_degradation
        season = resolve_api_season(league)
        if not is_known(season):
            record_degradation("data_loader._enrich_with_api_shots", league)
            continue
        # Also try the previous calendar year — many leagues span two seasons
        # (e.g. Brazil 2026 is in-progress but 2025 is fully completed in FD)
        prev_season = str(int(season) - 1)
        all_shots = []
        for s in [season, prev_season]:
            try:
                s_df = fetch_shots_for_league(league_id, s, league)
                if not s_df.empty:
                    s_df["_date_str"] = s_df["date"].dt.strftime("%Y-%m-%d")
                    all_shots.append(s_df)
            except Exception as e:
                log.warning(f"[api_football_ou] {league} season {s} shots fetch failed: {e}")
        if not all_shots:
            continue
        shots_df = (
            pd.concat(all_shots, ignore_index=True)
            .drop_duplicates(subset=["home_team", "away_team", "_date_str"])
            if len(all_shots) > 1 else all_shots[0]
        )

        if shots_df.empty:
            continue

        # API-Football's clubs resolved onto this league's history spellings (team_names.resolve)
        league_rows = df[mask].copy()
        league_rows["_h"] = league_rows["home_team"].astype(str)
        league_rows["_a"] = league_rows["away_team"].astype(str)
        league_rows["_date_str"]  = league_rows["date"].dt.strftime("%Y-%m-%d")
        shots_df = _resolve_onto_history(shots_df.assign(league=league), league_rows,
                                         "home_team", "away_team",
                                         f"api_football_live_shots[{league}]")

        api_cols = ["home_shots", "away_shots", "home_sot", "away_sot",
                    "home_corners", "away_corners", "home_fouls", "away_fouls",
                    "home_xg", "away_xg", "home_insidebox", "away_insidebox"]
        avail_cols = [c for c in api_cols if c in shots_df.columns]
        # League-scoped already (one league id per fetch), but shots_df is de-duplicated only when
        # two seasons were fetched, so a duplicate key could still expand the merge and shift the
        # positional copy-back. Audited always; safe join with WOWZA_SAFE_ENRICH_JOINS=1.
        _k3 = ["_h", "_a", "_date_str"]
        try:
            _IJ.audit_legacy_merge(league_rows, shots_df, _k3,
                                   name=f"api_football_live_shots.legacy[{league}]")
        except Exception as _e:                                  # noqa: BLE001
            log.warning(f"[api_football_ou] join audit skipped: {_e}")
        # One league id per fetch, so the league key is that league; safe join only
        _sd = shots_df.copy()
        _sd["league"] = league
        _lr, _ = _IJ.safe_left_enrich(league_rows, _sd, keys=["league"] + _k3,
                                      value_cols=avail_cols,
                                      name=f"api_football_live_shots[{league}]")
        for col in avail_cols:
            df.loc[_lr.index, col] = _lr[col].to_numpy()

        n_filled = int(df.loc[mask, "home_shots"].notna().sum())
        log.info(f"[api_football_ou] {league}: {n_filled}/{int(mask.sum())} rows now have shot data")

    # ── BANK THE ENRICHMENT, or pay for it again tomorrow ───────────────────────────────
    # Everything above this line fetched shot and xG data for FINISHED matches — a game played
    # in March 2024 has a final shot count that will never change. Until 2026-09-22 that result
    # was used for training and then THROWN AWAY: nothing wrote it back to af_history.parquet,
    # which is the committed bank the merge at the top of this function reads from. So every
    # retrain re-derived the same immutable numbers from scratch, and on 2026-09-22 that cost
    # TWO HOURS of a single run — 30 leagues at ~4 minutes each.
    #
    # apifootball_ou_cache/ (44,453 files) was the only thing standing between this and a full
    # API re-fetch, and retrain.yml had no actions/cache step until the same day, so even that
    # started empty on every run.
    #
    # Writing the result back means the next run's merge already has these rows and the
    # skip-if-populated guard above (xG > 50% for a league) short-circuits the whole fetch. The
    # cost goes from "every run, forever" to "once per genuinely new match".
    #
    # ONLY rows that actually gained data are written, and only the columns af_history owns, so
    # this cannot widen the schema or overwrite a real value with a null.
    try:
        _hist_p = Path(__file__).resolve().parents[1] / "output" / "af_history.parquet"
        _COLMAP = {"home_shots": "HS", "away_shots": "AS", "home_sot": "HST", "away_sot": "AST",
                   "home_xg": "HXG", "away_xg": "AXG",
                   "home_insidebox": "HIB", "away_insidebox": "AIB"}
        _have = [c for c in _COLMAP if c in df.columns]
        if _have:
            _new = df[df[_have].notna().any(axis=1)][
                ["league", "season", "date", "home_team", "away_team"] + _have].copy()
            _new = _new.rename(columns={k: v for k, v in _COLMAP.items() if k in _have})
            _prev = pd.read_parquet(_hist_p) if _hist_p.exists() else None
            _merged = pd.concat([_prev, _new], ignore_index=True) if _prev is not None else _new
            # Never bank a misfiled row: this write-back is how the archive's Dutch-as-Norway and
            # Bundesliga-2-as-'Denmark 1st Div' copies got into af_history in the first place.
            _merged = _drop_misfiled_league_rows(_merged, "af_history write-back")
            # Newest row wins: a later fetch may fill a column an earlier one left null.
            _merged = _merged.drop_duplicates(
                subset=[c for c in ("date", "league", "home_team", "away_team")
                        if c in _merged.columns], keep="last")
            for _c in ("league", "season", "home_team", "away_team"):
                if _c in _merged.columns:
                    _merged[_c] = _merged[_c].astype(str)
            _hist_p.parent.mkdir(exist_ok=True)
            _merged.to_parquet(_hist_p, index=False)
            log.info(f"[af_history] BANKED enrichment: {len(_merged):,} rows "
                     f"({len(_merged) - (len(_prev) if _prev is not None else 0):+,} this run). "
                     f"Next run merges these instead of re-fetching them.")
    except Exception as e:                                       # noqa: BLE001
        # Never fail a training run over the cache. Losing the bank costs time, not correctness.
        log.warning(f"[af_history] could not bank the enrichment: {e}")

    # The collected fixture statistics, AFTER the bank step: they live beside af_history.parquet
    # and are never written into it.
    df = _enrich_af_stats_fill(df)

    # Recompute SOT ratios after enrichment
    with np.errstate(divide="ignore", invalid="ignore"):
        df["home_sot_ratio"] = np.where(
            df["home_shots"] > 0, df["home_sot"] / df["home_shots"], np.nan
        )
        df["away_sot_ratio"] = np.where(
            df["away_shots"] > 0, df["away_sot"] / df["away_shots"], np.nan
        )

    return df


def load_all_matches(xlsx_path: Optional[Path] = None, force: bool = False) -> pd.DataFrame:
    """Load every match sheet from the summary workbook into one clean DataFrame."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    path = xlsx_path or config.SUMMARY_XLSX

    # ── CI / GitHub Actions fallback ─────────────────────────────────────────
    if not path.exists():
        log.info("Local Excel not found — downloading data from football-data.co.uk (CI mode)...")
        # What is already banked? Finished seasons in the cache are never re-requested.
        _cache_p = Path(__file__).resolve().parents[1] / "output" / "fd_history.parquet"
        _cached = None
        _have: set = set()
        if _cache_p.exists():
            try:
                _cached = pd.read_parquet(_cache_p)
                if {"league", "season"}.issubset(_cached.columns):
                    _have = set(zip(_cached["league"].astype(str),
                                    _cached["season"].astype(str)))
                # ALSO KEY BY THE DATE-DERIVED LABEL, because the stored `season` is not in one
                # convention and the lookup below is.
                #
                # The downloader probes (league, "2024/25") for standard-format leagues. The
                # cache stores standard rows under a calendar YEAR — '2023'..'2026' — because
                # they were seeded from the new-format path, where a calendar year is correct
                # (Brazil, Japan, MLS run Feb-Nov). So the probe could never hit: the cache held
                # 138 league-seasons and served ZERO of them, and every CI run re-downloaded all
                # 72 files from football-data.co.uk. Observed 2026-09-22:
                #   "[fd_cache] 37,623 cached rows covering 138 league-season(s)"
                #   "CI download: 72 file(s) loaded, 0 finished season(s) served from cache"
                #
                # A standard season runs Aug-May, so its label is recoverable from any match
                # date in it — the same rule retrain._auto_season uses. Adding the derived key
                # ALONGSIDE the stored one means the lookup works whichever convention a row was
                # written under, and cannot regress a cache that was already correct.
                if {"league", "date"}.issubset(_cached.columns):
                    _dt = pd.to_datetime(_cached["date"], errors="coerce")
                    _ok = _dt.notna()
                    if _ok.any():
                        _start = _dt[_ok].dt.year.where(_dt[_ok].dt.month >= 7,
                                                        _dt[_ok].dt.year - 1)
                        _lbl = _start.astype(str) + "/" + ((_start + 1) % 100).map("{:02d}".format)
                        _have |= set(zip(_cached.loc[_ok, "league"].astype(str), _lbl))

                # A SEASON COUNTS AS BANKED ONLY IF ITS CACHED ROWS CARRY WHAT THIS PARSER
                # EXTRACTS. Otherwise "already cached" freezes a thin old copy forever.
                #
                # This is the other half of the fix above, and without it that fix made things
                # worse. Before it, the (league, "2024/25") probe never matched the cache's
                # (league, "2024") rows, so every finished season was re-downloaded on every run
                # — wasteful, but the re-download carried corners, half-time goals and O/U
                # prices, and `keep="last"` let the fresh copy win. Adding the derived key made
                # the probe hit, so the download started SKIPPING those seasons — and the rows
                # left standing were the old corner-less ones. Measured on 2026-09-23: every
                # finished season of the Championship, League One, League Two, La Liga 2,
                # Serie B, Bundesliga 2 and the Greek Super League sat at 0% corners, 0%
                # half-time goals and 0% O/U 2.5 price, while 2026/27 — the one season always
                # re-fetched — sat at 100% on all three. Four seasons of three columns, on
                # exactly the leagues we bet, invisible because nothing ever failed.
                #
                # So skip a season only when the cache proves it was parsed by a version that
                # understood these columns. A league that genuinely has none of them (the
                # National League publishes no corners) still banks on the strength of the
                # others, which is why this is an OR and not an AND.
                _PROOF = ("home_corners", "ht_home_goals", "odds_over25")
                _present = [c for c in _PROOF if c in _cached.columns]
                if _present and "league" in _cached.columns:
                    _rich = _cached[_present].notna().any(axis=1)
                    _thin = set()
                    for _key_season in ("season", "_derived"):
                        if _key_season == "season" and "season" not in _cached.columns:
                            continue
                        if _key_season == "_derived":
                            if not {"league", "date"}.issubset(_cached.columns):
                                continue
                            _dt2 = pd.to_datetime(_cached["date"], errors="coerce")
                            _m = _dt2.notna()
                            if not _m.any():
                                continue
                            _s2 = _dt2[_m].dt.year.where(_dt2[_m].dt.month >= 7,
                                                         _dt2[_m].dt.year - 1)
                            _keys = pd.Series(
                                list(zip(_cached.loc[_m, "league"].astype(str),
                                         _s2.astype(str) + "/"
                                         + ((_s2 + 1) % 100).map("{:02d}".format))),
                                index=_cached.index[_m])
                        else:
                            _keys = pd.Series(
                                list(zip(_cached["league"].astype(str),
                                         _cached["season"].astype(str))),
                                index=_cached.index)
                        _any_rich = _rich.reindex(_keys.index).groupby(_keys).any()
                        _thin |= set(_any_rich[~_any_rich].index)
                    if _thin:
                        _have -= _thin
                        log.warning(
                            f"[fd_cache] {len(_thin)} cached league-season(s) are missing "
                            f"corners/half-time goals/O-U price — re-downloading them rather "
                            f"than serving a thin copy: "
                            f"{sorted(f'{a} {b}' for a, b in _thin)[:6]}")
                log.info(f"[fd_cache] {len(_cached):,} cached rows covering "
                         f"{len(_have)} banked league-season key(s)")
            except Exception as e:                                   # noqa: BLE001
                log.warning(f"[fd_cache] cache unreadable, will re-download: {e}")
        # Kept SEPARATE from the cached rows: only genuinely-downloaded frames may update the
        # cache, or the refresh would merge the cache into itself and report rows it never
        # fetched.
        _downloaded = _ci_download_all(_have)
        # Every row carries its source (`row_source`) so the one-match de-duplication can apply its
        # stated source rule. Tagged on COPIES: the cache refresh below writes `_downloaded` itself.
        ci_frames = [f.assign(row_source="football_data") for f in _downloaded]
        # The cached rows ARE part of the answer, not just a fallback — they are the seasons the
        # download deliberately skipped.
        if _cached is not None and len(_cached):
            ci_frames.append(_cached.assign(row_source="fd_cache"))

        # ── FALL BACK TO THE COMMITTED CACHE ──────────────────────────────────────────
        # On 2026-09-05 football-data.co.uk returned HTTP 503 on every URL including its own
        # homepage. This branch downloaded 0 files, raised, and predict failed every run for
        # 12+ hours — no tips at all, because a third-party server was unavailable.
        #
        # It should never have been able to do that. Almost everything fetched here is
        # IMMUTABLE: a match played in 2023 has a final score and a closing price that will
        # never change again. We were re-downloading four seasons of every league on every CI
        # run — several hundred requests to someone else's free server, every few minutes — to
        # rebuild a frame identical to the last one.
        #
        # So the download result is now CACHED and committed, and the cache is the fallback when
        # the site is unreachable. The current season still refreshes normally whenever the site
        # is up; only the dependency on it being up disappears.
        cache_p = Path(__file__).resolve().parents[1] / "output" / "fd_history.parquet"
        if _downloaded:
            try:
                fresh = pd.concat(_downloaded, ignore_index=True)
                prev = pd.read_parquet(cache_p) if cache_p.exists() else None
                merged = (pd.concat([prev, fresh], ignore_index=True)
                          if prev is not None else fresh)
                # A re-download of the same fixture supersedes the cached copy (scores get
                # corrected, odds columns get filled in later), so the NEWEST row wins.
                merged = merged.drop_duplicates(
                    subset=[c for c in ("date", "league", "home_team", "away_team")
                            if c in merged.columns], keep="last")
                # ONE DTYPE PER COLUMN BEFORE PARQUET, or the write dies and the cache is never
                # refreshed. `season` is a calendar YEAR for new-format leagues and a "2024/25"
                # LABEL for standard ones, and concatenating the two conventions produced an
                # object column holding both ints and strings. pyarrow cannot infer a type for
                # that and threw:
                #   "Expected bytes, got a 'int' object / Conversion failed for column season"
                # The exception was swallowed into a warning, so every run logged
                # "could not update the cache" and carried on — the cache could never grow, and
                # the next run re-downloaded everything again. Observed 2026-09-22.
                #
                # Cast rather than unify the conventions: a calendar year is genuinely right for
                # Brazil/Japan/MLS and a split label is genuinely right for the Championship.
                # The lookup above now keys on both, so they no longer need to agree.
                for _c in ("season", "league", "home_team", "away_team"):
                    if _c in merged.columns:
                        merged[_c] = merged[_c].astype(str)
                cache_p.parent.mkdir(exist_ok=True)
                merged.to_parquet(cache_p, index=False)
                log.info(f"[fd_cache] refreshed {cache_p.name}: {len(merged):,} rows "
                         f"({len(fresh):,} downloaded this run)")
            except Exception as e:                                   # noqa: BLE001
                log.warning(f"[fd_cache] could not update the cache: {e}")
        elif not ci_frames:
            if cache_p.exists():
                try:
                    cached = pd.read_parquet(cache_p)
                    if len(cached):
                        log.warning(
                            f"[fd_cache] football-data.co.uk returned NOTHING — serving "
                            f"{len(cached):,} cached rows instead. Historical data does not "
                            f"change, so this is complete except for fixtures played since the "
                            f"last successful download.")
                        ci_frames = [cached.assign(row_source="fd_cache")]
                except Exception as e:                               # noqa: BLE001
                    log.warning(f"[fd_cache] cache unreadable: {e}")
        if not ci_frames:
            raise RuntimeError(
                "CI download failed and no cached history is available "
                f"({cache_p.name} missing or empty) — no data loaded")
        ci_frames.extend(f.assign(row_source="api_football")
                         for f in _load_api_football_only_leagues())
        # The archive goes in BEFORE club canonicalisation, which is the whole reason it can go
        # in at all: the unifier resolves its club vocabulary against football-data's.
        _arch = _load_backtest_archive()
        if len(_arch):
            ci_frames.append(_arch.assign(row_source="backtest_archive"))
        out = pd.concat(ci_frames, ignore_index=True)
        out = out[out["home_team"].notna() & out["away_team"].notna()]
        out = out.sort_values("date").reset_index(drop=True)
        out["total_goals"] = out["home_goals"] + out["away_goals"]
        # No score -> label NaN, never 0 (NaN > 2.5 is False, which read as an Under / no BTTS)
        _ftn = out["total_goals"].notna()
        out["over25"] = (out["total_goals"] > 2.5).astype(float).where(_ftn)
        out["over15"] = (out["total_goals"] > 1.5).astype(float).where(_ftn)
        out["over35"] = (out["total_goals"] > 3.5).astype(float).where(_ftn)
        out["btts"] = ((out["home_goals"] > 0) & (out["away_goals"] > 0)).astype(float).where(_ftn)
        if "ht_home_goals" in out.columns and out["ht_home_goals"].notna().any():
            out["ht_total_goals"] = out["ht_home_goals"] + out["ht_away_goals"]
            # MASK THE MISSING ONES. `(NaN >= 1).astype(float)` is 0.0, not NaN, so every
            # fixture with no half-time score was silently labelled "no half-time goal" — a
            # fabricated negative, which invariant 9 exists to forbid. The guard above only
            # asks whether ANY row has HT data, then applies the label to ALL rows.
            #
            # Measured cost: the HT training frame showed Brazil Serie A at an 8.6% half-time
            # scoring rate, USA MLS 11.4%, Japan 11.7%. Both source stores are correct
            # (af_ht_history 68.8%, fd_history 70.2%) — the zeros were manufactured here. About
            # 37% of the HT training rows were invented negatives, and dropna() could not
            # remove them because the column was never null.
            #
            # src/feature_engineering.py:387 has always done this correctly; this line did not.
            _htn = out["ht_total_goals"].notna()
            out["ht_over05"] = (out["ht_total_goals"] >= 1).astype(float).where(_htn)
            out["ht_over15"] = (out["ht_total_goals"] >= 2).astype(float).where(_htn)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["home_sot_ratio"] = np.where(out.get("home_shots", pd.Series(dtype=float)) > 0,
                out.get("home_sot", pd.Series(dtype=float)) / out.get("home_shots", pd.Series(dtype=float)), np.nan)
            out["away_sot_ratio"] = np.where(out.get("away_shots", pd.Series(dtype=float)) > 0,
                out.get("away_sot", pd.Series(dtype=float)) / out.get("away_shots", pd.Series(dtype=float)), np.nan)
        # Club identity BEFORE any enrichment or feature work: a club split across two
        # spellings is two teams to everything downstream.
        out = one_club_one_match(out)
        out = _enrich_with_api_shots(out)
        out = _enrich_with_standard_sidemarket_odds(out)   # real BTTS/O1.5/O3.5 for 2nd-divs
        out = _enrich_ou25_from_captures(out)             # our own O/U 2.5, NaN-only
        out = _impossible_stats_to_missing(out)
        out = _flag_ou25_pair(out)
        _report_training_coverage(out)
        _CACHE = out
        return out

    xl   = pd.ExcelFile(path)

    frames = []
    for sheet in xl.sheet_names:
        m = re.match(r"^(.+)_(\d{4})$", sheet)
        if not m:
            continue
        league = m.group(1)

        try:
            raw = xl.parse(sheet)
        except Exception as e:
            log.warning(f"[local] sheet {sheet} unreadable, skipped: {e}")
            continue

        df = pd.DataFrame()
        df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
        df = df[df["date"].notna()].copy()

        df["league"]    = league
        # Season DERIVED from the match dates by the registry's rules, not a hand-typed code map
        # (which stopped at 2025/26 and silently skipped newer sheets)
        df["season"]    = _canonical_season_labels(df.assign(season=None))
        df["home_team"] = raw.get("HomeTeam", np.nan)
        df["away_team"] = raw.get("AwayTeam", np.nan)

        for out_col, src_col in [
            ("home_goals",      "FTHG"),
            ("away_goals",      "FTAG"),
            ("ht_home_goals",   "HTHG"),
            ("ht_away_goals",   "HTAG"),
            ("home_corners",    "HC"),
            ("away_corners",    "AC"),
            ("home_fouls",      "HF"),
            ("away_fouls",      "AF"),
            ("home_shots",      "HS"),
            ("away_shots",      "AS"),
            ("home_sot",        "HST"),
            ("away_sot",        "AST"),
        ]:
            df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")

        df["ftr"]      = raw.get("FTR", np.nan)
        df["referee"]  = raw.get("Referee", np.nan)
        df["odds_over25"]  = _pick_odds(raw, _OVER_COLS)
        df["odds_under25"] = _pick_odds(raw, _UNDER_COLS)
        df["odds_btts"]    = _pick_odds(raw, _BTTS_COLS)
        df["odds_over15"]  = _pick_odds(raw, _OVER15_COLS)
        df["odds_over35"]  = _pick_odds(raw, _OVER35_COLS)

        frames.append(df)

    # ── CSV overrides: prefer downloaded CSVs over Excel sheets ─────────────────
    csv_dir = path.parent / "data" / "football_data"
    if csv_dir.exists():
        # League -> football-data code from the REGISTRY (football_data_code), the same table the
        # CI download path uses - not a hand copy. A league without a registry code is not read.
        _std_codes, _new_codes = _fd_download_tables()
        for league_name, fd_code in {**_std_codes, **_new_codes}.items():
            code_dir = csv_dir / fd_code
            if not code_dir.exists():
                continue
            for csv_path in sorted(code_dir.glob(f"{fd_code}_*.csv")):
                try:
                    raw = pd.read_csv(csv_path, low_memory=False)
                    df = pd.DataFrame()
                    df["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
                    df = df[df["date"].notna()].copy()
                    df["league"]    = league_name
                    # season derived from the file's match dates (registry rules)
                    df["season"]    = _canonical_season_labels(df.assign(season=None))
                    df["home_team"] = raw.get("HomeTeam", np.nan)
                    df["away_team"] = raw.get("AwayTeam", np.nan)
                    for out_col, src_col in [
                        ("home_goals","FTHG"),("away_goals","FTAG"),
                        ("ht_home_goals","HTHG"),("ht_away_goals","HTAG"),
                        ("home_corners","HC"),("away_corners","AC"),
                        ("home_fouls","HF"),("away_fouls","AF"),
                        ("home_shots","HS"),("away_shots","AS"),
                        ("home_sot","HST"),("away_sot","AST"),
                    ]:
                        df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
                    df["ftr"]           = raw.get("FTR", np.nan)
                    df["referee"]       = raw.get("Referee", np.nan)
                    df["odds_over25"]   = _pick_odds(raw, _OVER_COLS)
                    df["odds_under25"]  = _pick_odds(raw, _UNDER_COLS)
                    df["odds_btts"]     = _pick_odds(raw, _BTTS_COLS)
                    df["odds_over15"]   = _pick_odds(raw, _OVER15_COLS)
                    df["odds_over35"]   = _pick_odds(raw, _OVER35_COLS)
                    df = df[df["home_team"].notna() & df["away_team"].notna()]
                    # Replace the matching Excel frame with this clean CSV data
                    _lbls = set(df["season"].dropna())
                    frames = [f for f in frames if not (
                        (f["league"] == league_name).all() and _lbls and
                        f["season"].isin(_lbls).all()
                    )]
                    frames.append(df)
                    log.debug(f"CSV override: {league_name} {sorted(_lbls)} ({len(df)} rows)")
                except Exception as e:
                    log.warning(f"Failed to load CSV override {csv_path}: {e}")

    # ── Direct HT download: registry standard leagues not in local files ─────
    # Was a hand table of big-5 names/codes ('Bundesliga 1' is not even a registry name). Now every
    # registry league whose football-data file family is the per-season mmz4281 one
    # (football_data_format, LR-1.2): the standard model leagues plus the big-5, which have no model
    # but whose codes (E0/D1/SP1/I1/F1) the registry now holds. Skipped when a local file supplied it.
    from src import league_registry as _LRx
    _HT_EXTRA = {e["name"]: e["football_data_code"] for e in _LRx.leagues()
                 if e.get("football_data_format") == "mmz4281" and e.get("football_data_code")}
    # The CURRENT and PREVIOUS campaigns, derived from the date and labelled to match. This was the
    # literal 2526 folder labelled '2025/26', so HT history never got the season being played
    # (Data Truth audit finding 44).
    _STD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    _n = __import__("datetime").datetime.now()
    _y0 = _n.year if _n.month >= 7 else _n.year - 1
    _ht_seasons = [(f"{y % 100:02d}{(y + 1) % 100:02d}", f"{y}/{(y + 1) % 100:02d}") for y in (_y0, _y0 - 1)]
    # Leagues already loaded via local CSV, decided ONCE - checked inside the loop, the current
    # season's own download would make the previous season look "already loaded".
    from src.league_registry import canonical_name as _canon
    _local = {_canon(str(f["league"].iloc[0])) for f in frames if not f.empty}
    for (lg_name, code), (s_code, s_label) in [(lc, s) for lc in _HT_EXTRA.items() for s in _ht_seasons]:
        if lg_name in _local:                 # exact registry identity, never a substring
            continue
        try:
            import requests as _req
            from io import StringIO as _SIO
            r = _req.get(_STD_URL.format(season=s_code, code=code), timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                log.warning(f"[data] HT extra {lg_name} {s_label}: HTTP {r.status_code}")
                continue
            raw = pd.read_csv(_SIO(_decode_csv(r.content)), on_bad_lines="skip", low_memory=False)
            if "FTHG" not in raw.columns:
                continue
            df = pd.DataFrame()
            df["date"]      = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
            df = df[df["date"].notna()].copy()
            df["league"]    = lg_name
            df["season"]    = s_label
            df["home_team"] = raw.get("HomeTeam", np.nan)
            df["away_team"] = raw.get("AwayTeam", np.nan)
            for out_col, src_col in [
                ("home_goals","FTHG"),("away_goals","FTAG"),
                ("ht_home_goals","HTHG"),("ht_away_goals","HTAG"),
                ("home_shots","HS"),("away_shots","AS"),
                ("home_sot","HST"),("away_sot","AST"),
                ("home_corners","HC"),("away_corners","AC"),
                ("home_fouls","HF"),("away_fouls","AF"),
            ]:
                df[out_col] = pd.to_numeric(raw.get(src_col, np.nan), errors="coerce")
            df["ftr"] = raw.get("FTR", np.nan)
            df["odds_over25"]  = _pick_odds(raw, _OVER_COLS)
            df["odds_under25"] = _pick_odds(raw, _UNDER_COLS)
            df["odds_btts"]    = _pick_odds(raw, _BTTS_COLS)
            df["odds_over15"]  = _pick_odds(raw, _OVER15_COLS)
            df["odds_over35"]  = _pick_odds(raw, _OVER35_COLS)
            df = df[df["home_team"].notna() & df["away_team"].notna()]
            frames.append(df)
            log.debug(f"HT extra: {lg_name} {s_label} {len(df)} rows")
        except Exception as e:
            log.warning(f"[data] HT extra {lg_name} {s_label} skipped ({type(e).__name__}: {e})")

    if not frames:
        raise RuntimeError(f"No match sheets found in {path}")

    frames = [f.assign(row_source="football_data") for f in frames]
    frames.extend(f.assign(row_source="api_football") for f in _load_api_football_only_leagues())
    out = pd.concat(frames, ignore_index=True)
    out = out[out["home_team"].notna() & out["away_team"].notna()]
    out = _apply_history_corrections(out)                 # owner 2026-11-04, as one_club_one_match
    out = _repair_club_text(out)
    out = _one_match_one_row(out)                         # same rule as the CI path
    out = out.sort_values("date").reset_index(drop=True)

    # Derived columns. No score -> label NaN, never 0 (NaN > 2.5 is False = an invented Under).
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    _ftn = out["total_goals"].notna()
    out["over25"]      = (out["total_goals"] > 2.5).astype(float).where(_ftn)
    out["over15"]      = (out["total_goals"] > 1.5).astype(float).where(_ftn)
    out["over35"]      = (out["total_goals"] > 3.5).astype(float).where(_ftn)
    out["btts"]        = (
        (out["home_goals"] > 0) & (out["away_goals"] > 0)
    ).astype(float).where(_ftn)

    # HT targets (only populated when HTHG/HTAG available)
    if "ht_home_goals" in out.columns and out["ht_home_goals"].notna().any():
        out["ht_total_goals"] = out["ht_home_goals"] + out["ht_away_goals"]
        # Same masking as the other construction site — see the note there. A missing half-time
        # score must stay NaN so dropna() can remove it, never become a fabricated 0.
        _htn = out["ht_total_goals"].notna()
        out["ht_over05"]      = (out["ht_total_goals"] >= 1).astype(float).where(_htn)
        out["ht_over15"]      = (out["ht_total_goals"] >= 2).astype(float).where(_htn)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["home_sot_ratio"] = np.where(
            out["home_shots"] > 0,
            out["home_sot"] / out["home_shots"],
            np.nan,
        )
        out["away_sot_ratio"] = np.where(
            out["away_shots"] > 0,
            out["away_sot"] / out["away_shots"],
            np.nan,
        )

    # ── API-Football shot enrichment for new-format leagues ──────────────────────
    out = _enrich_with_api_shots(out)
    # ── Real BTTS/O1.5/O3.5 odds for standard 2nd-divs (was synthetic 1.85) ───────
    out = _enrich_with_standard_sidemarket_odds(out)
    out = _enrich_ou25_from_captures(out)

    # ── Misfiled rows: the registry-based check the CI path uses (was a hand list of German clubs
    # and typed Danish league names) ─────────────────────────────────────────────
    out = _drop_misfiled_league_rows(out, "local")
    out = _impossible_stats_to_missing(out)
    out = _flag_ou25_pair(out)

    _CACHE = out
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_all_matches()
    print(f"Loaded {len(df):,} matches across {df['league'].nunique()} leagues")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Over 2.5 rate: {df['over25'].mean():.1%}")
    print(df[["date","league","home_team","away_team","home_goals","away_goals","over25"]].tail(8).to_string())
