"""
League-aware API season resolver (R2 finding 6). Replaces the generic `config.API_SEASON = "2025"`.
=================================================================================================

    from src.season_resolver import resolve_api_season, SEASON_UNKNOWN, is_known
    season = resolve_api_season("Japan J-League", kickoff_date="2026-11-08")    # -> "2027"
    if not is_known(season):
        ...  # skip the enrichment and say so - NEVER substitute a guessed year

    python -m src.season_resolver --write-health      # output/season_resolution_health.json

WHY. `config.API_SEASON` defaulted to "2025" and four API-Football call sites in
src/api_football_ou.py used it for EVERY league whenever no season was passed - which was always.
Six more sites fell back to a literal "2025" or to `str(now.year)` for any league missing from
`API_FOOTBALL_SEASONS`. A finished season returns plausible-looking data (last season's team stats,
last season's standings), so the defect never looks like a failure. And no single year is right for
every league at once: European leagues key a season by its START year, the spring-autumn leagues by
the calendar year, and Japan's J1 moved to an autumn-spring calendar in 2026 and keys the live
league by its END year (season 2027 = 2026-08-07..2027-06-06; verified on the API 2026-10-22).

RULES come from config/leagues.json (`season_rule` per league; see `season_rule_types` there):
    start_year(m)   month >= m -> year,     else year - 1
    calendar_year   year
    end_year(m)     month >= m -> year + 1, else year
    fixed           the one season of a tournament
    eras            a dated list of the above, for a league that changed calendar
A league that is not in the registry (or the auto-onboarding overlay), a provider with no registered
convention, or an unparseable date resolves to SEASON_UNKNOWN. Callers must treat that as
"enrichment unavailable" and record the degradation (`record_degradation`), never fall back to a year.

Does not import config (config.py derives API_FOOTBALL_SEASONS from this module).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import league_registry as LR  # noqa: E402

SEASON_UNKNOWN = "SEASON_UNKNOWN"
RESOLVER_VERSION = "SR-1.0-2026-11-01"
PROVIDERS = ("api_football",)
AUTO_LEAGUES_FILE = ROOT / "config" / "auto_leagues.json"
HEALTH_FILE = ROOT / "output" / "season_resolution_health.json"

#: (call_site, league, reason) recorded in-process by callers that skipped work on SEASON_UNKNOWN
_DEGRADATIONS: list[dict] = []


@dataclass
class Resolution:
    league: str
    canonical: str | None
    provider: str
    kickoff_date: str | None
    season: str
    rule: str
    source: str
    reason: str = ""


def is_known(season) -> bool:
    return isinstance(season, str) and season.isdigit()


def _to_date(kickoff_date) -> date | None:
    if kickoff_date is None:
        return datetime.now(timezone.utc).date()
    try:
        import pandas as pd
        if kickoff_date is pd.NaT or (not isinstance(kickoff_date, str) and pd.isna(kickoff_date)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(kickoff_date, datetime):
        return kickoff_date.date()
    if isinstance(kickoff_date, date):
        return kickoff_date
    try:
        import pandas as pd
        t = pd.Timestamp(kickoff_date)
        if pd.isna(t):
            return None
        return t.date()
    except Exception:                                                     # noqa: BLE001
        return None


def _apply_rule(rule: dict, d: date) -> tuple[str, str]:
    """(season, rule description) or (SEASON_UNKNOWN, why)."""
    if "eras" in rule:
        for er in rule["eras"]:
            lo = date.fromisoformat(er["from"]) if er.get("from") else date.min
            hi = date.fromisoformat(er["until"]) if er.get("until") else date.max
            if lo <= d < hi:
                s, desc = _apply_rule({k: v for k, v in er.items() if k not in ("from", "until")}, d)
                return s, f"era[{er.get('from', '')}..{er.get('until', '')}) {desc}"
        return SEASON_UNKNOWN, "no season era covers this date"
    t = rule.get("type")
    if t == "start_year":
        m = int(rule["start_month"])
        return str(d.year if d.month >= m else d.year - 1), f"start_year(month>={m})"
    if t == "calendar_year":
        return str(d.year), "calendar_year"
    if t == "end_year":
        m = int(rule["start_month"])
        return str(d.year + 1 if d.month >= m else d.year), f"end_year(month>={m})"
    if t == "fixed":
        return str(rule["season"]), "fixed"
    return SEASON_UNKNOWN, f"unknown rule type {t!r}"


def _auto_entry(name: str) -> dict | None:
    try:
        data = json.loads(AUTO_LEAGUES_FILE.read_text(encoding="utf-8")) if AUTO_LEAGUES_FILE.exists() else {}
    except Exception:                                                     # noqa: BLE001
        return None
    for e in data.get("leagues", []) or []:
        if str(e.get("name", "")).strip() == name:
            return e
    return None


def resolve(league, kickoff_date=None, provider: str = "api_football") -> Resolution:
    """Full resolution record. Never raises."""
    lg = "" if league is None else str(league).strip()
    d = _to_date(kickoff_date)
    ds = d.isoformat() if d else None
    if provider not in PROVIDERS:
        return Resolution(lg, None, provider, ds, SEASON_UNKNOWN, "", "none",
                          f"no season convention registered for provider {provider!r}")
    if d is None:
        return Resolution(lg, None, provider, None, SEASON_UNKNOWN, "", "none",
                          f"unparseable kickoff_date {kickoff_date!r}")
    try:
        canon = LR.canonical_name(lg)
        ent = LR.entry(lg) if canon else None
    except Exception as e:                                                # noqa: BLE001
        return Resolution(lg, None, provider, ds, SEASON_UNKNOWN, "", "none", f"registry unreadable: {e}")
    if ent is not None:
        try:
            s, desc = _apply_rule(ent["season_rule"], d)
        except Exception as e:                                            # noqa: BLE001
            s, desc = SEASON_UNKNOWN, f"bad season_rule: {e}"
        return Resolution(lg, canon, provider, ds, s, desc, "config/leagues.json",
                          "" if is_known(s) else desc)
    auto = _auto_entry(lg)
    if auto is not None:
        if isinstance(auto.get("season_rule"), dict):
            s, desc = _apply_rule(auto["season_rule"], d)
            return Resolution(lg, lg, provider, ds, s, desc, "config/auto_leagues.json",
                              "" if is_known(s) else desc)
        cur = str(auto.get("current_season", "")).strip()
        if cur.isdigit():
            # Provider-verified at onboarding (Pro League Scout reads the API's `current` flag), but
            # STATIC: it does not roll. Reported as such in the health file.
            return Resolution(lg, lg, provider, ds, cur, "static current_season (does not roll)",
                              "config/auto_leagues.json")
    return Resolution(lg, None, provider, ds, SEASON_UNKNOWN, "", "none",
                      "league not in config/leagues.json (canonical or alias) nor config/auto_leagues.json")


def resolve_api_season(league, kickoff_date=None, provider: str = "api_football") -> str:
    """The provider's season id for `league` at `kickoff_date` (default: today UTC), or
    SEASON_UNKNOWN. Never a guessed year."""
    return resolve(league, kickoff_date, provider).season


def record_degradation(call_site: str, league, reason: str = "SEASON_UNKNOWN") -> None:
    """Callers that SKIP an enrichment because the season is unknown record it here; the list is
    written into season_resolution_health.json by the process that wrote the health file."""
    _DEGRADATIONS.append({"call_site": call_site, "league": str(league), "reason": reason,
                          "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def degradations() -> list[dict]:
    return list(_DEGRADATIONS)


def health(today=None, extra_leagues=()) -> dict:
    """Resolved season per league (registry + auto overlay + any `extra_leagues`)."""
    d = _to_date(today)
    rows = []
    names = [e["name"] for e in LR.leagues()]
    try:
        auto = json.loads(AUTO_LEAGUES_FILE.read_text(encoding="utf-8")).get("leagues", []) \
            if AUTO_LEAGUES_FILE.exists() else []
    except Exception:                                                     # noqa: BLE001
        auto = []
    names += [str(e.get("name", "")).strip() for e in auto if str(e.get("name", "")).strip() not in names]
    names += [n for n in extra_leagues if n not in names]
    for n in names:
        r = resolve(n, d)
        ent = LR.entry(n)
        rows.append({**asdict(r), "active": bool(ent and ent.get("enabled")),
                     "model_type": ent.get("model_type") if ent else None,
                     "api_football_id": ent.get("api_football_id") if ent else None,
                     "status": ("PASS" if is_known(r.season) and r.source == "config/leagues.json"
                                else "WARN" if is_known(r.season) else
                                "FAIL" if (ent and ent.get("enabled")) else "WARN")})
    deg = degradations()
    st = "FAIL" if any(r["status"] == "FAIL" for r in rows) else \
         "WARN" if (deg or any(r["status"] == "WARN" for r in rows)) else "PASS"
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resolver_version": RESOLVER_VERSION, "as_of_date": d.isoformat() if d else None,
            "status": st,
            "rule": "FAIL if an ACTIVE league resolves to SEASON_UNKNOWN; WARN for an inactive unknown, a "
                    "static auto-league season, or any call site that skipped work on SEASON_UNKNOWN",
            "deprecated": {"config.API_SEASON": "removed from every call site (R2-A); value is SEASON_UNKNOWN"},
            "leagues": rows, "degradations_this_process": deg}


def write_health(path: Path | None = None, today=None) -> dict:
    h = health(today)
    p = Path(path or HEALTH_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(h, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return h


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-health", action="store_true")
    ap.add_argument("--date", default=None)
    a = ap.parse_args(argv)
    h = write_health(today=a.date) if a.write_health else health(a.date)
    print(json.dumps({"status": h["status"], "as_of_date": h["as_of_date"],
                      "seasons": {r["league"]: r["season"] for r in h["leagues"]}}, indent=1))
    return 1 if h["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
