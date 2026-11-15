"""
Canonical league registry (R2 finding 7) - config/leagues.json is the ONE machine-readable source.
=================================================================================================

    python -m src.league_registry --write-health     # output/league_registry_health.json
    python -m src.league_registry --check            # exit 1 on any validation error

WHY. config.py held ten hand-maintained per-league tables (format sets, enabled/paper sets, API-
Football ids, seasons, OddsAPI sport keys, football-data codes, timezones, Sofascore ids...). They
drifted: "Danish Superliga" in the season table while every other table said "Denmark Superliga"
(Denmark ran on a finished season for months), "Austria Bundesliga" next to "Austrian Bundesliga"
as two keys for one league, Ligue 2 on the wrong API id in one copy. Each table is now DERIVED from
config/leagues.json by `derive()`, and config.py keeps exposing the same names, so no consumer
changes.

ALIASES are alternative spellings of ONE league. They resolve to the canonical name through
`canonical_name()` and the alias-aware `LeagueMap`, but they are never separate keys, never
iterated, never counted twice. Resolution is EXACT (after trimming whitespace): there is no fuzzy
or case-folding match, because a near-name that silently resolves is the same failure shape as a
near-name that silently routes to the Standard model (R2 finding 8).

PER-LEAGUE FACTS (LR-1.1). Facts the code used to type next to a league name - prop coverage, the FPL
league, competition type (club league / club cup / national teams), timezones, OddsAPI keys - live
in the registry too and are read through prop_leagues(), prop_sport_keys(), odds_api_sport_key(),
odds_api_sport_keys_for(), sport_key_to_league(), timezone(), competition_type(), leagues_of_type(),
fpl_league(), provider_name() and same_club_spellings(). An unknown league or a fact the registry
does not hold returns None (or an empty list): never a default.

LR-1.2 (2026-11-02): football_data_format (the football-data file family of a code, so a no-model
league's code is never filed under /new/ by elimination), odds_api_props (OddsAPI has priced player
props for the league; read through odds_api_prop_sport_keys()), historical league spellings as
aliases, and the national-team competitions of player_history (season_rule null = SEASON_UNKNOWN).

THIS MODULE MUST NOT IMPORT config: config.py imports it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone as _tz
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "config" / "leagues.json"
HEALTH_FILE = ROOT / "output" / "league_registry_health.json"

MODEL_TYPES = ("standard", "new_format")
PRODUCTION_STATES = ("ENABLED", "ENABLED_PROVIDER_UNSUPPORTED", "PAPER", "TRAINING_ONLY", "COLLECTION_ONLY")
SEASON_RULE_TYPES = ("start_year", "calendar_year", "end_year", "fixed")
COMPETITION_TYPES = ("club_league", "club_cup", "national_teams")
FOOTBALL_DATA_FORMATS = ("mmz4281", "new")
_MODEL_FD_FORMAT = {"standard": "mmz4281", "new_format": "new"}

_IANA: set | None = None


def _iana_zones() -> set:
    """Every IANA zone this interpreter knows; empty when it has no tz database at all (Windows
    without the tzdata package), in which case the zone-name check is skipped rather than failing
    every league - config.py refuses to start on a validation error."""
    global _IANA
    if _IANA is None:
        try:
            import zoneinfo
            _IANA = set(zoneinfo.available_timezones())
        except Exception:                                                  # noqa: BLE001
            _IANA = set()
    return _IANA


class RegistryError(ValueError):
    """config/leagues.json is internally inconsistent. config.py refuses to start on it."""


class LeagueMap(dict):
    """A plain dict keyed by CANONICAL league names whose lookups also accept a registered alias.

    `m["Austria Bundesliga"]`, `m.get("Austria Bundesliga")` and `"Austria Bundesliga" in m` all
    resolve to the canonical "Austrian Bundesliga" entry. Iteration, len(), keys(), items() see
    canonical names only, so an alias can never be fetched, priced or counted twice.
    """

    def __init__(self, data=(), aliases: dict | None = None):
        super().__init__(data)
        self._aliases = dict(aliases or {})

    def _canon(self, key):
        if isinstance(key, str):
            k = key.strip()
            if dict.__contains__(self, k):
                return k
            return self._aliases.get(k, k)
        return key

    def __missing__(self, key):
        k = self._canon(key)
        if k != key and dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        raise KeyError(key)

    def get(self, key, default=None):
        k = self._canon(key)
        return dict.get(self, k, default)

    def __contains__(self, key):
        return dict.__contains__(self, self._canon(key))

    def copy(self):
        return LeagueMap(self, self._aliases)

    def __reduce__(self):                      # pickles / deep-copies as itself, aliases included
        return (LeagueMap, (dict(self), self._aliases))


@dataclass
class Derived:
    """Every table config.py derives from the registry, in registry order."""
    version: str
    standard: set = field(default_factory=set)
    new_format: set = field(default_factory=set)
    enabled: set = field(default_factory=set)
    paper: set = field(default_factory=set)
    api_football_only: set = field(default_factory=set)
    api_football_extra_seasons: dict = field(default_factory=dict)
    football_data: dict = field(default_factory=dict)
    timezones: dict = field(default_factory=dict)
    api_football_ids: dict = field(default_factory=dict)
    odds_api_sport_keys: dict = field(default_factory=dict)
    sofascore_ids: dict = field(default_factory=dict)
    provider_unsupported: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)          # alias -> canonical


_CACHE: dict = {}


def load_registry(path: Path | None = None) -> dict:
    """Parsed + validated registry. Cached per path. Raises RegistryError on any inconsistency."""
    p = Path(path or REGISTRY_FILE)
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    try:
        reg = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                              # noqa: BLE001
        raise RegistryError(f"cannot read league registry {p}: {e}") from e
    errors = validate(reg)
    if errors:
        raise RegistryError(f"league registry {p} is invalid: " + "; ".join(errors))
    _CACHE[key] = reg
    return reg


def validate(reg: dict) -> list[str]:
    """Every structural rule the registry must satisfy. Empty list = valid."""
    errs: list[str] = []
    leagues = reg.get("leagues")
    if not isinstance(leagues, list) or not leagues:
        return ["'leagues' must be a non-empty list"]
    names: dict[str, int] = {}
    ids: dict[str, str] = {}
    for i, e in enumerate(leagues):
        n = str(e.get("name", "")).strip()
        if not n or n != e.get("name"):
            errs.append(f"league #{i}: name missing or has surrounding whitespace")
            continue
        if n in names:
            errs.append(f"duplicate league name {n!r}")
        names[n] = i
        lid = str(e.get("id", "")).strip()
        if not lid:
            errs.append(f"{n}: missing id")
        elif lid in ids:
            errs.append(f"duplicate id {lid!r} ({ids[lid]} / {n})")
        else:
            ids[lid] = n
        if e.get("model_type") not in (None,) + MODEL_TYPES:
            errs.append(f"{n}: model_type {e.get('model_type')!r} not in {MODEL_TYPES} or null")
        if e.get("production_state") not in PRODUCTION_STATES:
            errs.append(f"{n}: production_state {e.get('production_state')!r} not in {PRODUCTION_STATES}")
        if e.get("enabled") and e.get("model_type") is None:
            errs.append(f"{n}: enabled league must have a model_type (an enabled league with no model "
                        f"would be routed nowhere)")
        if e.get("paper") and not e.get("enabled"):
            errs.append(f"{n}: a paper league must stay enabled (config.PAPER_LEAGUES rule)")
        if e.get("season_rule") is None and e.get("model_type") is None \
                and e.get("api_football_id") is None and not e.get("enabled"):
            pass    # LR-1.2: season convention unknown -> season_resolver gives SEASON_UNKNOWN
        else:
            errs += [f"{n}: {m}" for m in _validate_season_rule(e.get("season_rule"))]
        errs += [f"{n}: {m}" for m in _validate_facts(e)]
    fpl = [e.get("name") for e in leagues if e.get("fpl") is True]
    if len(fpl) > 1:
        errs.append(f"fpl is true for {len(fpl)} leagues {fpl}: FPL scores exactly one league")
    # aliases: never a canonical name, never two targets, never chained
    seen: dict[str, str] = {}
    for e in leagues:
        for a in e.get("aliases", []) or []:
            a2 = str(a).strip()
            if a2 != a or not a2:
                errs.append(f"{e.get('name')}: alias {a!r} empty or has surrounding whitespace")
            if a2 in names:
                errs.append(f"alias {a2!r} of {e.get('name')!r} is itself a canonical league name (alias cycle)")
            if a2 in seen and seen[a2] != e.get("name"):
                errs.append(f"alias {a2!r} points to two leagues ({seen[a2]} / {e.get('name')})")
            seen[a2] = e.get("name")
    # provider ids must be unique: two canonical leagues on one API id / sport key is a collapse
    for fld in ("api_football_id", "odds_api_sport_key", "football_data_code"):
        owner: dict = {}
        for e in leagues:
            v = e.get(fld)
            if v in (None, ""):
                continue
            if v in owner:
                errs.append(f"{fld} {v!r} used by two leagues ({owner[v]} / {e['name']})")
            owner[v] = e["name"]
    # an extra OddsAPI key is still one league's key: never another league's primary or extra key
    key_owner = {e["odds_api_sport_key"]: e["name"] for e in leagues if e.get("odds_api_sport_key")}
    for e in leagues:
        for k in e.get("odds_api_extra_sport_keys") or []:
            if k in key_owner and key_owner[k] != e.get("name"):
                errs.append(f"odds_api_sport_key {k!r} used by two leagues ({key_owner[k]} / {e.get('name')})")
            elif k in key_owner:
                errs.append(f"{e.get('name')}: extra OddsAPI key {k!r} repeats its own primary key")
            key_owner[k] = e.get("name")
    return errs


def _validate_facts(e: dict) -> list[str]:
    """The per-league facts added in LR-1.1 (competition type, props, FPL, timezone, extra OddsAPI
    keys, same-club spellings). Each is REQUIRED to be well-formed when present; competition_type,
    props and fpl are required on every entry so a consumer never has to assume one."""
    out = []
    if e.get("competition_type") not in COMPETITION_TYPES:
        out.append(f"competition_type {e.get('competition_type')!r} not in {COMPETITION_TYPES}")
    for flag in ("props", "fpl"):
        if not isinstance(e.get(flag), bool):
            out.append(f"{flag} must be true or false, got {e.get(flag)!r}")
    if e.get("props") is True and not e.get("api_football_id"):
        out.append("props league must have an api_football_id (the prop pipeline fetches by id)")
    # LR-1.2
    if not isinstance(e.get("odds_api_props"), bool):
        out.append(f"odds_api_props must be true or false, got {e.get('odds_api_props')!r}")
    elif e["odds_api_props"] and not e.get("odds_api_sport_key"):
        out.append("odds_api_props league must have an odds_api_sport_key")
    fmt, code = e.get("football_data_format"), e.get("football_data_code")
    if fmt not in (None,) + FOOTBALL_DATA_FORMATS:
        out.append(f"football_data_format {fmt!r} not in {FOOTBALL_DATA_FORMATS} or null")
    elif bool(fmt) != bool(code):
        out.append("football_data_format and football_data_code must be set together")
    elif fmt and e.get("model_type") and fmt != _MODEL_FD_FORMAT[e["model_type"]]:
        out.append(f"football_data_format {fmt!r} contradicts model_type {e['model_type']!r}")
    if e.get("fpl") is True and e.get("competition_type") != "club_league":
        out.append("the fpl league must be a club_league")
    tz = e.get("timezone")
    if tz is not None:
        if not isinstance(tz, str) or not tz.strip() or tz != tz.strip():
            out.append(f"timezone {tz!r} must be a non-empty IANA name or null")
        elif _iana_zones() and tz not in _iana_zones():
            out.append(f"timezone {tz!r} is not an IANA zone")
    if e.get("competition_type") == "club_league" and e.get("enabled") and not tz:
        out.append("enabled club league must have a timezone (kickoff times would be shown in a guessed zone)")
    extra = e.get("odds_api_extra_sport_keys")
    if extra is not None and (not isinstance(extra, list)
                              or not all(isinstance(k, str) and k.strip() == k and k for k in extra)
                              or len(set(extra)) != len(extra)):
        out.append(f"odds_api_extra_sport_keys {extra!r} must be a list of distinct non-empty keys")
    if extra and not e.get("odds_api_sport_key"):
        out.append("odds_api_extra_sport_keys without a primary odds_api_sport_key")
    pairs = e.get("same_club_spellings")
    if pairs is not None:
        if not isinstance(pairs, list) or not all(
                isinstance(p, list) and len(p) == 2 and all(isinstance(s, str) and s.strip() == s and s for s in p)
                and p[0] != p[1] for p in pairs):
            out.append(f"same_club_spellings {pairs!r} must be a list of [spelling, other spelling] pairs")
    return out


def _validate_season_rule(rule) -> list[str]:
    if not isinstance(rule, dict):
        return ["season_rule missing"]
    if "eras" in rule:
        eras = rule["eras"]
        if not isinstance(eras, list) or not eras:
            return ["season_rule.eras must be a non-empty list"]
        out = []
        for er in eras:
            out += _validate_season_rule({k: v for k, v in er.items() if k not in ("from", "until")})
        return out
    t = rule.get("type")
    if t not in SEASON_RULE_TYPES:
        return [f"season_rule.type {t!r} not in {SEASON_RULE_TYPES}"]
    if t in ("start_year", "end_year") and not (1 <= int(rule.get("start_month", 0) or 0) <= 12):
        return [f"season_rule.type {t} needs start_month 1..12"]
    if t == "fixed" and not str(rule.get("season", "")).isdigit():
        return ["season_rule.type fixed needs a numeric season"]
    return []


def leagues(path: Path | None = None) -> list[dict]:
    return list(load_registry(path)["leagues"])


def entry(name: str, path: Path | None = None) -> dict | None:
    """The registry entry for a canonical name OR a registered alias; None when unknown."""
    c = canonical_name(name, path)
    if c is None:
        return None
    for e in load_registry(path)["leagues"]:
        if e["name"] == c:
            return e
    return None


def alias_map(path: Path | None = None) -> dict:
    return {a: e["name"] for e in load_registry(path)["leagues"] for a in e.get("aliases", []) or []}


def canonical_name(name, path: Path | None = None) -> str | None:
    """Canonical league name for an exact canonical name or registered alias, else None.

    Exact match after trimming surrounding whitespace only - no case folding, no fuzzy matching.
    """
    if name is None:
        return None
    n = str(name).strip()
    if not n:
        return None
    reg = load_registry(path)
    for e in reg["leagues"]:
        if e["name"] == n:
            return n
    return alias_map(path).get(n)


def derive(path: Path | None = None) -> Derived:
    """Build every config.py league table from the registry, preserving registry order."""
    reg = load_registry(path)
    d = Derived(version=str(reg.get("registry_version", "")), aliases=alias_map(path))
    for e in reg["leagues"]:
        n = e["name"]
        if e.get("model_type") == "standard":
            d.standard.add(n)
        elif e.get("model_type") == "new_format":
            d.new_format.add(n)
        if e.get("enabled"):
            d.enabled.add(n)
        if e.get("paper"):
            d.paper.add(n)
        if e.get("api_football_only"):
            d.api_football_only.add(n)
        if e.get("api_football_extra_seasons"):
            d.api_football_extra_seasons[n] = list(e["api_football_extra_seasons"])
        for fld, tgt in (("football_data_code", d.football_data), ("timezone", d.timezones),
                         ("api_football_id", d.api_football_ids), ("odds_api_sport_key", d.odds_api_sport_keys),
                         ("sofascore_tournament_id", d.sofascore_ids)):
            if e.get(fld) not in (None, ""):
                tgt[n] = e[fld]
        if e.get("provider_unsupported"):
            d.provider_unsupported[n] = dict(e["provider_unsupported"])
    for attr in ("api_football_extra_seasons", "football_data", "timezones", "api_football_ids",
                 "odds_api_sport_keys", "sofascore_ids", "provider_unsupported"):
        setattr(d, attr, LeagueMap(getattr(d, attr), d.aliases))
    return d


# ── per-league facts (LR-1.1) ───────────────────────────────────────────────────────────────────
# Each accessor takes a canonical name OR a registered alias and returns None for an unknown league
# or a fact the registry does not hold: the caller must skip and log, never substitute a value.
def _fact(league, fld, path: Path | None = None):
    e = entry(league, path) if league is not None else None
    if e is None:
        return None
    v = e.get(fld)
    return None if v in (None, "") else v


def prop_leagues(path: Path | None = None) -> LeagueMap:
    """{canonical name: API-Football id} for every league the player-prop pipeline covers, in
    registry order. Replaces player_model/config.py PROP_LEAGUES (aliases such as "World Cup" still
    resolve on lookup)."""
    return LeagueMap({e["name"]: e["api_football_id"] for e in leagues(path) if e.get("props") is True},
                     alias_map(path))


def prop_sport_keys(path: Path | None = None) -> LeagueMap:
    """{canonical name: OddsAPI sport key} for prop leagues that have one. Replaces
    player_model/odds_fetcher.py PROP_SPORT_KEYS; extra keys (UCL qualification) come from
    `odds_api_sport_keys_for` / `sport_key_to_league`."""
    return LeagueMap({e["name"]: e["odds_api_sport_key"] for e in leagues(path)
                      if e.get("props") is True and e.get("odds_api_sport_key")}, alias_map(path))


def odds_api_prop_sport_keys(path: Path | None = None) -> LeagueMap:
    """{canonical name: OddsAPI sport key} for the leagues OddsAPI has actually priced player props
    for (`odds_api_props: true`, evidence output/prop_odds_coverage.json). The props fetcher can ask
    only these instead of probing every prop league."""
    return LeagueMap({e["name"]: e["odds_api_sport_key"] for e in leagues(path)
                      if e.get("odds_api_props") is True and e.get("odds_api_sport_key")}, alias_map(path))


def football_data_format(league, path: Path | None = None) -> str | None:
    """'mmz4281' | 'new' - the football-data.co.uk file family of the league's code, or None."""
    return _fact(league, "football_data_format", path)


def odds_api_sport_key(league, path: Path | None = None) -> str | None:
    """The league's primary OddsAPI sport key, or None (unknown league / no key registered)."""
    return _fact(league, "odds_api_sport_key", path)


def odds_api_sport_keys_for(league, path: Path | None = None) -> list[str]:
    """Primary + extra OddsAPI keys of one league (empty list when it has none)."""
    e = entry(league, path) if league is not None else None
    if e is None or not e.get("odds_api_sport_key"):
        return []
    return [e["odds_api_sport_key"], *(e.get("odds_api_extra_sport_keys") or [])]


def sport_key_to_league(path: Path | None = None) -> dict:
    """{OddsAPI sport key: canonical league} over every primary and extra key."""
    return {k: e["name"] for e in leagues(path) for k in odds_api_sport_keys_for(e["name"], path)}


def timezone(league, path: Path | None = None) -> str | None:
    """IANA timezone of the league, or None when the registry holds none (multi-country competitions,
    and leagues no source in the repos gives a zone for). Never 'UTC' by default."""
    return _fact(league, "timezone", path)


def competition_type(league, path: Path | None = None) -> str | None:
    """'club_league' | 'club_cup' | 'national_teams', or None for an unknown league."""
    return _fact(league, "competition_type", path)


def leagues_of_type(ctype: str, path: Path | None = None) -> list[str]:
    """Canonical names of every registry league of one competition type, in registry order."""
    if ctype not in COMPETITION_TYPES:
        raise ValueError(f"competition type {ctype!r} not in {COMPETITION_TYPES}")
    return [e["name"] for e in leagues(path) if e.get("competition_type") == ctype]


def fpl_league(path: Path | None = None) -> str | None:
    """Canonical name of the one league Fantasy Premier League scores, or None if none is flagged."""
    hit = [e["name"] for e in leagues(path) if e.get("fpl") is True]
    return hit[0] if hit else None


def provider_name(league, provider: str, path: Path | None = None):
    """A provider's own id/name record for the league (e.g. provider='fbref' -> {'id', 'slug'}),
    or None."""
    names = _fact(league, "provider_names", path) or {}
    return names.get(provider)


def same_club_spellings(league, path: Path | None = None) -> list[tuple[str, str]]:
    """Registered pairs of history spellings of ONE club in this league (sources that never overlap
    in time, so no same-day evidence can link them). Empty list when none are registered."""
    return [tuple(p) for p in (_fact(league, "same_club_spellings", path) or [])]


# ── health ──────────────────────────────────────────────────────────────────────────────────────
def consistency_with_config(cfg=None) -> dict:
    """Does the LIVE config (after the auto-league overlay) still agree with the registry?

    Every registry-derived table must equal its derivation for registry leagues; leagues present
    in config but not in the registry are listed (expected only for config/auto_leagues.json)."""
    if cfg is None:
        import config as cfg                                                   # noqa: WPS433
    d = derive()
    reg_names = {e["name"] for e in leagues()}
    auto = set(getattr(cfg, "AUTO_LEAGUES", []) or [])
    checks = {}
    pairs = {"STANDARD_FORMAT_LEAGUES": d.standard, "NEW_FORMAT_LEAGUES": d.new_format,
             "ENABLED_LEAGUES": d.enabled, "PAPER_LEAGUES": d.paper,
             "API_FOOTBALL_ONLY_LEAGUES": d.api_football_only,
             "API_FOOTBALL_EXTRA_SEASONS": d.api_football_extra_seasons,
             "FOOTBALL_DATA_LEAGUES": d.football_data, "LEAGUE_TIMEZONES": d.timezones,
             "API_FOOTBALL_IDS": d.api_football_ids, "ODDS_API_SPORT_KEYS": d.odds_api_sport_keys,
             "SOFASCORE_TOURNAMENT_IDS": d.sofascore_ids}
    for name, want in pairs.items():
        have = getattr(cfg, name, None)
        if have is None:
            checks[name] = {"status": "FAIL", "detail": "missing from config"}
            continue
        if isinstance(want, set):
            h = {x for x in have if x in reg_names}
            extra = sorted(set(have) - reg_names)
            ok = h == want
            detail = {"missing": sorted(want - h), "unexpected": sorted(h - want)}
        else:
            h = {k: v for k, v in dict(have).items() if k in reg_names}
            extra = sorted(set(dict(have)) - reg_names)
            ok = h == dict(want)
            detail = {"differs": sorted(k for k in set(h) | set(want) if h.get(k) != dict(want).get(k))}
        unexplained = sorted(set(extra) - auto)
        checks[name] = {"status": "PASS" if ok and not unexplained else "FAIL",
                        "non_registry_keys": extra, "unexplained_non_registry_keys": unexplained, **detail}
    return checks


def health(routing: dict | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(_tz.utc)
    out = {"generated_at": now.isoformat(timespec="seconds"), "registry_file": "config/leagues.json"}
    try:
        reg = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        errs = validate(reg)
    except Exception as e:                                                     # noqa: BLE001
        return {**out, "status": "FAIL", "errors": [f"unreadable registry: {e}"]}
    out["registry_version"] = reg.get("registry_version")
    out["errors"] = errs
    ls = reg.get("leagues", [])
    out["counts"] = {"leagues": len(ls), "aliases": sum(len(e.get("aliases", []) or []) for e in ls),
                     "standard": sum(e.get("model_type") == "standard" for e in ls),
                     "new_format": sum(e.get("model_type") == "new_format" for e in ls),
                     "enabled": sum(bool(e.get("enabled")) for e in ls),
                     "paper": sum(bool(e.get("paper")) for e in ls),
                     "props": sum(e.get("props") is True for e in ls)}
    out["aliases"] = {a: e["name"] for e in ls for a in e.get("aliases", []) or []}
    try:
        cons = consistency_with_config()
    except Exception as e:                                                     # noqa: BLE001
        cons = {"_error": {"status": "FAIL", "detail": f"{type(e).__name__}: {e}"}}
    out["config_consistency"] = cons
    gaps = []
    for e in ls:
        if e.get("enabled") and not e.get("odds_api_sport_key") and not e.get("provider_unsupported"):
            gaps.append(f"{e['name']}: enabled with no OddsAPI sport key and no PROVIDER_UNSUPPORTED evidence")
        if e.get("model_type") and not e.get("api_football_id"):
            gaps.append(f"{e['name']}: model league with no API-Football id")
        if e.get("competition_type") == "club_league" and not e.get("timezone"):
            gaps.append(f"{e['name']}: club league with no timezone in any repo source")
    out["coverage_gaps"] = gaps
    if routing is not None:
        out["last_predict_routing"] = routing
    bad = bool(errs) or any(v.get("status") == "FAIL" for v in cons.values())
    warn = bool(gaps) or bool(routing and routing.get("unknown_league_fixtures"))
    out["status"] = "FAIL" if bad else "WARN" if warn else "PASS"
    return out


def write_health(routing: dict | None = None, path: Path | None = None) -> dict:
    h = health(routing)
    p = Path(path or HEALTH_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(h, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return h


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-health", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    h = write_health() if a.write_health else health()
    print(json.dumps({k: h[k] for k in ("status", "registry_version", "counts", "errors", "coverage_gaps")
                      if k in h}, indent=1))
    return 1 if (a.check and h["status"] == "FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
