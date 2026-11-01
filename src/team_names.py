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
    # German endonym/exonym pairs no token or stem rule can bridge. Added 2026-08-17 when
    # prop-odds matching showed Bayern and Koln failing against OddsAPI's English spellings.
    "bayern munich": "bayern munchen",
    "cologne": "koln",
    # OddsAPI spellings that resolved to NOTHING, so the fixture got no form data at all
    # (invariant 8). Owner-approved 2026-11-01 ("we must fix those bugs"). MLS: one history spelling
    # each, so this only adds data. China: the history holds each club under TWO names with
    # overlapping dates (a duplicate identity, the Standard bug the NF repair still has to fix);
    # until then the CURRENT name is used.
    "d c united": "dc united",
    "la galaxy": "los angeles galaxy",
    "shandong luneng taishan fc": "shandong taishan",
    "henan fc": "henan songshan longmen",
    # Data Truth audit 2026-11-01 (finding 6): each club has ONE spelling in its league's history
    # (checked on fd_history + af_history since 2024), so these only add data - the clubs had no
    # form at all and every fixture of theirs was AVOID as blind. Clubs whose history holds TWO
    # spellings (Belgrano, Kyoto, Urawa, UNAM, Celta B) are NOT aliased here: that needs the
    # duplicate identity unified first (finding 37), or an alias would drop half their history.
    "saint etienne": "st etienne",
    "austria wien": "austria vienna",
    "atletico mineiro": "atletico mg",
    "argentinos juniors": "argentinos jrs",
    # Data Truth repair 2026-11-01 (data profile: unresolved_club_name). Same rule as above: ONE
    # spelling of the club in its league's history bank (fd_history + af_history), and the same
    # club proven at fixture level - on every ledger/quote fixture of the predict spelling, the
    # bank row for that league, date (+-1 day) and the resolved opponent carries exactly this
    # history spelling in the club's slot (counts in brackets).
    "estudiantes": "estudiantes l p",           # Argentina; Estudiantes de Rio Cuarto is another club (27/27)
    "gimnasia la plata": "gimnasia l p",        # Argentina; Gimnasia Mendoza is another club (26/29)
    "cercle brugge ksv": "cercle brugge",       # Belgium; tokens also hit Club Brugge (2/2)
    "oh leuven": "oud heverlee leuven",         # Belgium (4/4)
    "beijing fc": "beijing guoan",              # China; Beijing Renhe last played 2019 (24/24)
    "boston united": "boston utd",              # National League (6/6)
    "inverness ct": "inverness c",              # Scottish Championship (6/6)
    "raith rovers": "raith rvs",                # Scottish Championship (4/4)
    "airdrie united": "airdrie utd",            # Scottish League One (4/4)
    "queen of the south": "queen of sth",       # Scottish League One (4/4)
    "sporting cp": "sp lisbon",                 # Portugal (4/4)
    "vitoria sc": "guimaraes",                  # Portugal; Vitoria Setubal is "Setubal" (4/4)
    "basaksehir": "buyuksehyr",                 # Turkey (6/6)
}


# RESULT-SETTLEMENT ALIASES ONLY (owner-approved 2026-10-22) - used by resolve_result(), NEVER by resolve().
# OddsAPI / predict names -> API-Football RESULT spellings, found by resolving every club in 689 committed
# predictions snapshots against API-Football's own names; without them those clubs' results could not be
# settled (v91 shadow). Kept OUT of _ALIASES on purpose: the training history holds both spellings of some
# clubs (e.g. "SK Rapid" and "Rapid Vienna"), and an alias there would switch which history rows feed the
# model's form features at predict time - a model-input change nobody approved.
_RESULT_ALIASES = {
    # England / Spain
    "sheffield united": "sheffield utd",
    "accrington stanley": "accrington st",
    "celta fortuna": "celta de vigo ii",
    "real sociedad b": "real sociedad ii",
    # Brazil / USA / Finland
    "bragantino sp": "rb bragantino",
    "atletico mineiro": "atletico mg",
    "d c united": "dc united",
    "la galaxy": "los angeles galaxy",
    "tps turku": "turku ps",
    # Argentina - "Estudiantes" is explicit because Estudiantes de Rio Cuarto is a different club
    "argentinos juniors": "argentinos jrs",
    "estudiantes": "estudiantes l p",
    "gimnasia la plata": "gimnasia l p",
    "gimnasia mendoza": "gimnasia m",
    # Austria - German endonyms vs API-Football's English/older forms
    "austria wien": "austria vienna",
    "rapid wien": "rapid vienna",
    "rb salzburg": "red bull salzburg",
    "rheindorf altach": "scr altach",
    "wsg tirol": "wsg wattens",
    # China - API-Football keeps the clubs' FORMER names (each pairing is one-to-one across the
    # 16-club league: renamed club -> its previous name)
    "zhejiang": "hangzhou greentown",
    "chengdu rongcheng fc": "chengdu better city",
    "dalian yingbo": "dalian zhixing",
    "liaoning tieren fc": "shenyang urban",
    "qingdao hainiu fc": "qingdao jonoon",
    "qingdao west coast fc": "qingdao youth island",
    "shenzhen peng city fc": "sichuan jiuniu",
    "tianjin jinmen tiger fc": "tianjin teda",
    # France / Japan (Data Truth audit 2026-11-01, finding 20: never resolved, never graded).
    # Urawa is spelled both "Urawa" and "Urawa Reds" in the result sources, so the token tiers
    # refuse it as ambiguous; the alias names the one spelling to use.
    "saint etienne": "st etienne",
    "urawa red diamonds": "urawa reds",
    # Data Truth repair 2026-11-01: clubs whose history bank holds TWO spellings of the one club
    # (a duplicate identity, finding 37), so the token tiers refuse them as ambiguous. Proven the
    # same club at fixture level: both spellings sit in the club's slot on the same league + date
    # + opponent as the predict spelling. Settlement only - in _ALIASES they would pick one half
    # of the club's form history, a model-input change that waits for the identity merge.
    "belgrano de cordoba": "belgrano",          # Argentina: 'Belgrano' 18 / 'Belgrano Cordoba' 20
    "instituto de cordoba": "instituto",        # Argentina: 'Instituto' 19 / 'Instituto Cordoba' 18
    "chapecoense": "chapecoense sc",            # Brazil: 'Chapecoense-SC' 30 / 'Chapecoense-sc' 24
    "sc paderborn": "paderborn",                # Bundesliga 2: 'Paderborn' 2 / 'SC Paderborn 07' 3
    "shanghai sipg fc": "shanghai port",        # China (renamed 2021): 'Shanghai Port' 24 / 'SHANGHAI SIPG' 7
    "st patricks athletic": "st patricks",      # Ireland: 'St. Patricks' 24 / "St Patrick's Athl." 25
    "kyoto purple sanga": "kyoto sanga",        # Japan: 'Kyoto Sanga' 13 / 'Kyoto' 20
    "andorra cf": "andorra",                    # La Liga 2: 'Andorra' 19 / 'FC Andorra' 10
    "cd castellon": "castellon",                # La Liga 2: 'Castellon' 17 / 'Castellón' 8
    "pumas": "unam pumas",                      # Liga MX: 'UNAM Pumas' 21 / 'U.N.A.M. - Pumas' 18
}


_RESERVE_SUFFIX = {"ii", "b", "reserves"}
_RESERVE_ANY = {"u19", "u21", "u23"}


def _is_reserve(s: str) -> bool:
    """A reserve side: "... II" / "... B" / "... Reserves" as the LAST word, or an age group anywhere.
    Last-word only, so first teams such as "B. Monchengladbach" or "Fortuna Dusseldorf" are never
    mistaken for reserves ("Celta Fortuna" is covered by its explicit alias instead)."""
    w = norm(s).split()
    return bool(w) and (w[-1] in _RESERVE_SUFFIX or bool(set(w) & _RESERVE_ANY))


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
    # RESERVE SIDES ONLY MATCH RESERVE SIDES in the token tiers (2026-10-22): "Celta Vigo" (first
    # team) is a token subset of "Celta de Vigo II" and resolved to it, so a first-team fixture could
    # be settled with the reserve team's result. Explicit aliases above are unaffected.
    res = _is_reserve(name)
    cand_tokens = [(c, _tokens(c)) for c in candidates if _is_reserve(c) == res]

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


def resolve_result(name: str, candidates: list[str]) -> str | None:
    """resolve() for RESULT settlement: the normal league-scoped tiers first; only if they find
    nothing, the API-Football result-spelling aliases (either direction). Predict-time matching
    must keep using resolve()."""
    hit = resolve(name, candidates)
    if hit or not name or not candidates:
        return hit
    n = norm(name)
    by_norm = {}
    for c in candidates:
        by_norm.setdefault(norm(c), c)
    alias = _RESULT_ALIASES.get(n)
    if alias and alias in by_norm:
        return by_norm[alias]
    for long_form, short in _RESULT_ALIASES.items():
        if n == short and long_form in by_norm:
            return by_norm[long_form]
    return None
