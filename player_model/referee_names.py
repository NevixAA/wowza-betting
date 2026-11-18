"""One canonical referee name per referee (data truth, 2026-11-18).

THE DEFECT. API-Football's /fixtures `fixture.referee` is a free-text string with no referee id,
and the provider changes the spelling of the SAME fixture between fetches: "Ramon Abatti Abel,
Brazil" one day, "R. Abatti" the next, "Ramon Abatti" on a third. Measured over every committed
version of player_history.parquet (163 blobs on main + data-truth-100, 2026-11-18): 1,110
fixtures were stored with two or three different spellings. Every spelling is a separate
"referee" to build_features' `groupby("_ref")`, so referee_yellows_pg / referee_strictness /
card_clash_index split one referee's history in two, and a nightly re-fetch that flipped the
spelling recomputed those features on ~3,000 rows.

THE RULE (evidence only, nothing invented). There is no referee id, so the evidence that two
spellings are one person is the SAME FIXTURE observed with both -- and that alone is not enough:
the provider also corrects a wrongly assigned referee (fixture 1490598 went "S. Boiko" ->
"Ricardo Montero Araya"), so a pair is an alias only when the names are also COMPATIBLE:

  * after dropping the ", Country" suffix, accents, dots and case, the shorter name's first
    token is the longer name's first token or its initial, and every other token of the shorter
    name appears in the longer one ("R. Abatti" ~ "Ramon Abatti" ~ "Ramon Abatti Abel, Brazil").

Pairs are scoped per league. Within a league the compatible pairs form groups; a group maps to
the most complete spelling in it (country suffix, then most tokens, then longest) ONLY if every
other spelling in the group is compatible with that one. A group where an abbreviation bridges
two different full names ("R. Pereira" seen against two different people) is AMBIGUOUS and is
left unmapped (reported, never guessed).

The map is built by scripts/build_referee_aliases.py and committed as referee_aliases.json next
to this module. It is applied to NEW rows at collect time (pipeline.mode_collect) so the stored
name and the referee features use the canonical spelling. Committed rows are never rewritten;
readers that group by referee over stored rows call canonicalise() first.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ALIASES_PATH = Path(__file__).with_name("referee_aliases.json")
ALIASES_VERSION = "referee-aliases/1.0"


def _tokens(name: str) -> list[str]:
    s = str(name or "").split(",")[0]
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[.\-']", " ", s.lower())
    return [t for t in s.split() if t]


def _has_country(name: str) -> bool:
    return "," in str(name or "")


def completeness(name: str) -> tuple:
    """Sort key: the most complete spelling has the largest key."""
    return (_has_country(name), len(_tokens(name)), len(str(name)), str(name))


def compatible(a: str, b: str) -> bool:
    """True when the shorter spelling can be an abbreviation of the longer one."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    first, rest = short[0], short[1:]
    if not (first == long_[0] or (len(first) == 1 and long_[0].startswith(first))
            or (len(long_[0]) == 1 and first.startswith(long_[0]))):
        return False
    if not rest:
        return False                       # a lone initial / first name proves nothing
    pool = list(long_[1:])
    for t in rest:
        if t in pool:
            pool.remove(t)
        else:
            return False
    return True


def derive_aliases(observations: dict) -> tuple[dict, dict]:
    """observations: {(fixture_id, league): set(spellings)} from every stored version.

    Returns ({league: {alias: canonical}}, report)."""
    pairs: dict = defaultdict(lambda: defaultdict(set))   # league -> name -> {names}
    evidence: dict = defaultdict(int)
    incompatible = 0
    for (_fid, league), names in observations.items():
        names = sorted({str(n).strip() for n in names if isinstance(n, str) and n.strip()})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if compatible(a, b):
                    pairs[league][a].add(b)
                    pairs[league][b].add(a)
                    evidence[(league, a, b)] += 1
                else:
                    incompatible += 1
    out: dict = {}
    ambiguous: list = []
    for league in sorted(pairs, key=str):
        graph = pairs[league]
        seen: set = set()
        for start in sorted(graph):
            if start in seen:
                continue
            group, stack = set(), [start]
            while stack:
                n = stack.pop()
                if n in group:
                    continue
                group.add(n)
                stack.extend(graph[n] - group)
            seen |= group
            canon = max(group, key=completeness)
            if all(n == canon or compatible(n, canon) for n in group):
                for n in sorted(group):
                    if n != canon:
                        out.setdefault(str(league), {})[n] = canon
            else:
                ambiguous.append({"league": str(league), "names": sorted(group)})
    report = {
        "fixtures_observed": len(observations),
        "fixtures_with_several_spellings": sum(1 for v in observations.values() if len(set(v)) > 1),
        "incompatible_pairs_not_aliased": incompatible,
        "aliases": sum(len(m) for m in out.values()),
        "ambiguous_groups": ambiguous,
    }
    return out, report


def load_aliases(path=None) -> dict:
    p = Path(path) if path is not None else ALIASES_PATH
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {str(lg): {str(k): str(v) for k, v in (m or {}).items()}
            for lg, m in (d.get("map", {}) or {}).items()}


def canonical(name, league, aliases: dict) -> str:
    if not isinstance(name, str):
        return name
    return aliases.get(str(league), {}).get(name.strip(), name)


def canonicalise(referee: pd.Series, league: pd.Series, aliases: dict | None = None) -> pd.Series:
    """Vectorised canonical() over a frame's referee/league columns."""
    aliases = load_aliases() if aliases is None else aliases
    if not aliases:
        return referee
    out = referee.copy()
    for lg, m in aliases.items():
        mask = league.astype(str).eq(lg) & referee.isin(list(m))
        if mask.any():
            out.loc[mask] = referee.loc[mask].map(m)
    return out


def canonicalise_rows(rows: list[dict], aliases: dict | None = None) -> int:
    """In place on collect rows ({"referee", "league", ...}). Returns rows changed."""
    aliases = load_aliases() if aliases is None else aliases
    n = 0
    for r in rows:
        ref = r.get("referee")
        c = canonical(ref, r.get("league"), aliases)
        if isinstance(ref, str) and c != ref:
            r["referee"] = c
            n += 1
    return n
