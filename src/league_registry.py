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

THIS MODULE MUST NOT IMPORT config: config.py imports it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "config" / "leagues.json"
HEALTH_FILE = ROOT / "output" / "league_registry_health.json"

MODEL_TYPES = ("standard", "new_format")
PRODUCTION_STATES = ("ENABLED", "ENABLED_PROVIDER_UNSUPPORTED", "PAPER", "TRAINING_ONLY", "COLLECTION_ONLY")
SEASON_RULE_TYPES = ("start_year", "calendar_year", "end_year", "fixed")


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
        errs += [f"{n}: {m}" for m in _validate_season_rule(e.get("season_rule"))]
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
    return errs


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
    now = now or datetime.now(timezone.utc)
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
                     "paper": sum(bool(e.get("paper")) for e in ls)}
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
