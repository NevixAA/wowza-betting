"""Data Truth checks - the owner's four rules as RED/AMBER/GREEN health lines (2026-11-01).

    python -m src.integrity.data_truth            # -> output/data_truth_health.json (+ printed)
    python -m src.integrity.data_truth --baseline # rewrite config/data_truth_baseline.json (owner review)

The 56-finding audit of 2026-11-01 fixed the violations it found. These checks exist so the next one
turns system health RED instead of waiting for the next audit:

    1 NAMING     every league value in the committed boards, ledgers and history banks is a
                 registry league (canonical or alias); blind board fixtures are counted.
    2 DEFAULTS   no NEW invented default in production code: fillna(<non-zero literal>), literal
                 season fallbacks, literal season-year lists, pinned football-data folders. Known
                 remaining ones live in config/data_truth_baseline.json, each with a reason.
    3 MISSING    the live board carries its prices and its serving features (rest days, HT form),
                 and the history banks are fresh.
    4 LEAGUES    no misfiled league rows in the banks, the registry validates, joins serve
                 league-keyed.

Every check reads committed files only, so it runs offline and in CI. Nothing here changes data.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
BASELINE = ROOT / "config" / "data_truth_baseline.json"
HEALTH = OUT / "data_truth_health.json"
VERSION = "DATA-TRUTH-1"

GREEN, AMBER, RED, GREY = "GREEN", "AMBER", "RED", "NO_DATA"

# Production code a default can hide in (tips, stakes, settlement, features, loaders).
SCAN_FILES = ["pipeline.py", "update_results.py", "config.py", "src/predict.py", "src/feature_engineering.py",
              "src/ht_features.py", "src/betting.py", "src/data_loader.py", "src/stake_plan.py",
              "src/ledger.py", "src/api_football_ou.py", "src/btts_odds.py", "player_model/predict.py",
              "player_model/pipeline.py", "player_model/config.py", "player_model/feature_engineering.py",
              "player_model/data_fetcher.py", "player_model/fantasy.py", "player_model/league_quality.py",
              "telegram_bot/notifier.py"]
DEFAULT_PATTERNS = {
    "fillna_literal": re.compile(r"\.fillna\(\s*-?(?:0*[1-9]\d*(?:\.\d+)?|0?\.\d*[1-9]\d*)\s*\)"),
    "season_literal_fallback": re.compile(r"""\.get\([^()]*,\s*["']20\d\d["']\s*\)"""),
    "season_year_list": re.compile(r"20\d\d\|20\d\d"),
    "pinned_fd_folder": re.compile(r"mmz4281/\d{4}/"),
}

BOARD_FILES = ["predictions.csv", "bets_ledger.csv", "side_bets_ledger.csv", "player_tips.csv"]
BANK_FILES = ["fd_history.parquet", "af_history.parquet"]


def _c(check, status, value, rule, detail=""):
    return {"check": check, "status": status, "value": value, "rule": rule, "detail": detail}


def _read(p: Path, cols=None) -> pd.DataFrame:
    try:
        if p.suffix == ".parquet":
            return pd.read_parquet(p, columns=cols)
        return pd.read_csv(p, usecols=lambda c: cols is None or c in cols, low_memory=False)
    except Exception:                                                 # noqa: BLE001
        return pd.DataFrame()


# ── 1 NAMING ─────────────────────────────────────────────────────────────────────────────────────
def naming() -> dict:
    rule = ("every league value in predictions / ledgers / player tips / history banks is a registry league "
            "(canonical or alias): RED if any is not; AMBER if > 10% of board fixtures are blind (no form)")
    from src.league_registry import canonical_name
    unknown: dict[str, set] = {}
    for f in BOARD_FILES + BANK_FILES:
        d = _read(OUT / f, ["league"])
        if d.empty or "league" not in d.columns:
            continue
        bad = {lg for lg in d["league"].dropna().astype(str).unique() if not canonical_name(lg)}
        if bad:
            unknown[f] = bad
    p = _read(OUT / "predictions.csv", ["no_form_data", "home_scored_last5", "away_scored_last5"])
    blind = None
    if not p.empty and {"home_scored_last5", "away_scored_last5"} <= set(p.columns):
        blind = float((p["home_scored_last5"].isna() | p["away_scored_last5"].isna()).mean() * 100)
    n_bad = sum(len(v) for v in unknown.values())
    st = RED if n_bad else AMBER if (blind is not None and blind > 10) else GREEN
    detail = "; ".join(f"{f}: {sorted(v)[:5]}" for f, v in unknown.items())
    return _c("Data truth 1: one canonical name", st,
              f"{n_bad} unknown league name(s); blind board fixtures "
              f"{'n/a' if blind is None else f'{blind:.1f}%'}", rule, detail)


# ── 2 DEFAULTS ───────────────────────────────────────────────────────────────────────────────────
def scan_defaults() -> list[dict]:
    """Every invented-default site in SCAN_FILES: {file, kind, code}. Comments are ignored."""
    hits = []
    for rel in SCAN_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            code = ln.split("#", 1)[0]
            s = code.strip()
            if not s or s.startswith(('"', "'")):
                continue
            for kind, pat in DEFAULT_PATTERNS.items():
                if pat.search(code):
                    hits.append({"file": rel, "kind": kind, "code": s})
    return hits


def _baseline() -> dict:
    try:
        return json.loads(BASELINE.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {"entries": []}


def defaults() -> dict:
    rule = ("no NEW invented default in production code (fillna(<non-zero literal>), literal season "
            "fallback/list, pinned football-data folder): RED if any site is not in "
            "config/data_truth_baseline.json; GREEN otherwise (baseline entries each carry a reason)")
    known = {(e["file"], e["code"]) for e in _baseline().get("entries", [])}
    hits = scan_defaults()
    new = [h for h in hits if (h["file"], h["code"]) not in known]
    gone = len(known - {(h["file"], h["code"]) for h in hits})
    return _c("Data truth 2: no silent defaults", RED if new else GREEN,
              f"{len(new)} new default(s); {len(hits) - len(new)} known (baseline); {gone} baseline entr(ies) "
              f"now gone", rule, "; ".join(f"{h['file']}: {h['code'][:80]}" for h in new[:5]))


# ── 3 MISSING ────────────────────────────────────────────────────────────────────────────────────
def missing(now: datetime | None = None) -> dict:
    rule = ("live board: every fixture priced (odds_over25); Standard fixtures' rest days and HT form "
            "present (RED if > 50% missing, AMBER if > 20%); history banks fresh (AMBER if the newest "
            "match is > 10 days old)")
    now = now or datetime.now(timezone.utc)
    p = _read(OUT / "predictions.csv")
    if p.empty:
        return _c("Data truth 3: no missing data", GREY, "no board", rule)
    issues, worst = [], GREEN
    unpriced = int(p["odds_over25"].isna().sum()) if "odds_over25" in p.columns else len(p)
    if unpriced:
        issues.append(f"{unpriced} unpriced fixture(s)")
        worst = RED
    std = p[p.get("model_type", pd.Series("", index=p.index)).astype(str) == "standard"]
    for col in ("home_rest_days", "home_ht_over05_rate"):
        if len(std) and col in std.columns:
            share = float(std[col].isna().mean() * 100)
            if share > 50:
                worst = RED
            elif share > 20 and worst != RED:
                worst = AMBER
            if share > 20:
                issues.append(f"{col} missing on {share:.0f}% of Standard fixtures")
    for f in BANK_FILES:
        d = _read(OUT / f, ["date"])
        if not d.empty:
            age = (pd.Timestamp(now) - pd.to_datetime(d["date"], errors="coerce", utc=True).max()).days
            if age > 10:
                issues.append(f"{f} newest match {age} days old")
                worst = AMBER if worst == GREEN else worst
    return _c("Data truth 3: no missing data", worst, "; ".join(issues) or "board priced, features present",
              rule)


# ── 4 LEAGUES ────────────────────────────────────────────────────────────────────────────────────
def leagues() -> dict:
    rule = ("no misfiled league rows in af_history / backtest archive (unknown league or split label in a "
            "calendar-year league), config/leagues.json validates, and the enrichment joins serve "
            "league-keyed (WOWZA_SAFE_ENRICH_JOINS on): RED on any violation")
    from src import data_loader as DL
    from src.league_registry import load_registry, validate
    issues = []
    for f in ("af_history.parquet", "backtest_all_leagues.csv"):
        d = _read(OUT / f, ["league", "season"])
        if not d.empty:
            n = len(d) - len(DL._drop_misfiled_league_rows(d, f"data_truth:{f}"))
            if n:
                issues.append(f"{f}: {n} misfiled row(s)")
    errs = validate(load_registry())
    if errs:
        issues.append(f"registry: {len(errs)} error(s) e.g. {errs[0]}")
    if not DL._safe_joins():
        issues.append("enrichment joins are NOT league-keyed (WOWZA_SAFE_ENRICH_JOINS=0)")
    return _c("Data truth 4: right league relations", RED if issues else GREEN,
              "; ".join(issues) or "banks clean, registry valid, joins league-keyed", rule)


def evaluate(now: datetime | None = None) -> list[dict]:
    """The four rule checks. Never raises: a check that cannot run is itself RED."""
    out = []
    for fn in (naming, defaults, lambda: missing(now), leagues):
        try:
            out.append(fn())
        except Exception as e:                                         # noqa: BLE001
            out.append(_c(getattr(fn, "__name__", "data truth check"), RED, "check failed", "must run",
                          f"{type(e).__name__}: {e}"[:200]))
    return out


def write_baseline() -> int:
    """Record today's remaining default sites for owner review (each needs a reason)."""
    prev = {(e["file"], e["code"]): e.get("reason", "") for e in _baseline().get("entries", [])}
    hits = scan_defaults()
    entries = [{**h, "reason": prev.get((h["file"], h["code"]), "TODO: reason")} for h in hits]
    BASELINE.write_text(json.dumps({"version": VERSION, "entries": entries}, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    return len(entries)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--baseline" in argv:
        print(f"baseline: {write_baseline()} entr(ies) -> {BASELINE.relative_to(ROOT)}")
        return 0
    checks = evaluate()
    doc = {"artifact": "data_truth_health", "version": VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "checks": checks}
    HEALTH.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    for c in checks:
        print(f"{c['status']:7} {c['check']}: {c['value']}" + (f"  [{c['detail']}]" if c["detail"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
