"""
Club-name resolution between data sources.
==========================================
OddsAPI (used at PREDICT time) and football-data.co.uk (the HISTORY we train on) spell the
same club very differently:

    OddsAPI                     football-data.co.uk
    1. FC Kaiserslautern        Kaiserslautern      ordinal + generic prefix
    VfL Wolfsburg               Wolfsburg           generic prefix
    SV Darmstadt 98             Darmstadt           generic prefix + year suffix
    Arminia Bielefeld           Bielefeld           extra descriptor token
    Lincoln City                Lincoln             extra descriptor token
    Cadiz CF / Almeria          Cadiz / Almeria     accents + generic suffix
    Karlsruher SC               Karlsruhe           inflected stem
    Sporting Gijon              Sp Gijon            abbreviated token
    Queens Park Rangers         QPR                 initialism (alias table)

`build_features` (training) reads football-data names; `build_upcoming_features` (predict)
receives OddsAPI names. When they fail to reconcile, the fixture gets NO rolling-form
features and every one of them is median-imputed — the model is effectively blind on that
match. Measured 2026-08-15: 46% of standard fixtures, 27 clubs, on the core features
(scored/conceded/over25 rate/attack & defense strength), not the 0%-importance extras.

SAFETY: the old fallback was `history_name.lower().startswith(oddsapi_name.split()[0])`,
which would happily map "Real Valladolid CF" onto any club beginning with "Real". Here,
resolution is always scoped to ONE league and an ambiguous match — more than one candidate
at the same confidence level — is REJECTED rather than guessed. Precision over recall: a
missing club costs one fixture's features, a wrong club poisons them with another team's form.
"""
from __future__ import annotations

import re
import unicodedata

# Club-form words that carry no identity on their own.
_GENERIC = {
    "fc", "cf", "sc", "afc", "cd", "sd", "ad", "ac", "ss", "as", "us", "sv", "vfl", "vfb",
    "fsv", "tsg", "spvgg", "msv", "bsc", "if", "ifk", "fk", "sk", "bk", "ca", "cp", "club",
    "calcio", "sporting",  # "Sporting Gijon" -> Gijon; the alias table covers Sporting CP
    "real",  # dropped only as a LAST resort, see _tokens(keep_real)
}
# Descriptors that distinguish English clubs from each other and must NEVER be stripped
# globally ("Manchester City" vs "Manchester United"). Subset matching handles them instead.
_KEEP = {"city", "united", "utd", "town", "rovers", "rvs", "county", "athletic", "wanderers"}

_ALIASES = {
    # initialisms and heavy abbreviations no token rule can derive
    "queens park rangers": "qpr",
    # football-data contracts by dropping vowels, which no stem rule can derive
    "bristol rovers": "bristol rvs",
    "blackburn rovers": "blackburn",
    "tranmere rovers": "tranmere",
    "doncaster rovers": "doncaster",
    "forest green rovers": "forest green",
    "peterborough united": "peterboro",
    "sporting gijon": "sp gijon",
    "sheffield wednesday": "sheffield weds",
    "west bromwich albion": "west brom",
    "wolverhampton wanderers": "wolves",
    "brighton and hove albion": "brighton",
    "milton keynes dons": "mk dons",
    "deportivo la coruna": "la coruna",
    "racing santander": "santander",
    "real sporting de gijon": "sp gijon",
}


def norm(s: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", str(s))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = ascii_str.lower().replace("&", " and ")
    ascii_str = re.sub(r"[^a-z0-9 ]", " ", ascii_str)
    return re.sub(r"\s+", " ", ascii_str).strip()


def _tokens(s: str, keep_real: bool = True) -> set[str]:
    """Identity tokens: generic club words and bare numbers (98, 04, 1.) removed."""
    generic = _GENERIC - ({"real"} if keep_real else set())
    out = set()
    for t in norm(s).split():
        if t in generic or t.isdigit():
            continue
        out.add(t)
    return out


def _prefix_eq(a: str, b: str, minlen: int = 3) -> bool:
    """One token is the other's stem: karlsruher/karlsruhe, peterboro/peterborough.

    minlen is 3 so football-data's short forms work — man/manchester, utd/united,
    nott/nottingham. Ambiguity is still caught by the caller: "Man City" pairs only with
    "Manchester City" because `city` cannot pair with `united`.
    """
    if a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    return len(lo) >= minlen and hi.startswith(lo)


def _pairs_up(small: set[str], big: set[str]) -> bool:
    """Every token in `small` pairs with a distinct prefix-compatible token in `big`."""
    remaining = set(big)
    for t in small:
        hit = next((o for o in remaining if _prefix_eq(t, o)), None)
        if hit is None:
            return False
        remaining.discard(hit)
    return True


def resolve(name: str, candidates: list[str]) -> str | None:
    """Map an OddsAPI club name onto one of `candidates` (history names for the SAME league).

    Returns the matching candidate, or None when there is no match or the match is ambiguous.
    Tiers are tried in order and the first tier with EXACTLY ONE hit wins.
    """
    if not name or not candidates:
        return None
    n = norm(name)

    by_norm = {}
    for c in candidates:
        by_norm.setdefault(norm(c), c)

    # 1. identical once normalised
    if n in by_norm:
        return by_norm[n]

    # 2. explicit alias, in either direction
    alias = _ALIASES.get(n)
    if alias and alias in by_norm:
        return by_norm[alias]
    for long_form, short in _ALIASES.items():
        if n == short and long_form in by_norm:
            return by_norm[long_form]

    nt = _tokens(name)
    if not nt:
        return None
    cand_tokens = [(c, _tokens(c)) for c in candidates]

    def _only(hits):
        return hits[0] if len(hits) == 1 else None

    # 3. identical identity-token sets  ("Cadiz CF" == "Cadiz", "Wimbledon" == "AFC Wimbledon")
    hit = _only([c for c, t in cand_tokens if t == nt])
    if hit:
        return hit

    # 4. same size, tokens pair by stem  ("Karlsruher SC" -> "Karlsruhe", "Sporting Gijon")
    hit = _only([c for c, t in cand_tokens
                 if len(t) == len(nt) and _pairs_up(nt, t)])
    if hit:
        return hit

    # 5. strict subset either way, tokens pairing by stem
    #    ("Lincoln City" -> "Lincoln", "Arminia Bielefeld" -> "Bielefeld",
    #     "Peterborough United" -> "Peterboro"). "Manchester City" and "Manchester United"
    #    are NOT subsets of each other, so they stay apart.
    hit = _only([c for c, t in cand_tokens
                 if t and (
                     (len(t) < len(nt) and _pairs_up(t, nt)) or
                     (len(nt) < len(t) and _pairs_up(nt, t))
                 )])
    if hit:
        return hit

    # 6. last resort: allow "Real" to be dropped ("Real Valladolid CF" -> "Valladolid").
    #    Still ambiguity-checked, so "Real Madrid"/"Real Sociedad" cannot collapse together.
    nt2 = _tokens(name, keep_real=False)
    if nt2 and nt2 != nt:
        hit = _only([c for c, t in cand_tokens if t == nt2])
        if hit:
            return hit
        hit = _only([c for c, t in cand_tokens
                     if t and len(t) < len(nt2) and _pairs_up(t, nt2)])
        if hit:
            return hit

    return None
