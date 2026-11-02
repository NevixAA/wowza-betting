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
    5 ONE MATCH  the banks, through the loader's identity pipeline, hold no match twice (same day
                 under two spellings, or 1-3 days apart) and no row whose form window holds its own
                 match (2026-11-02).

A check whose input is missing is NO_DATA (or RED where the input is a production bank), never GREEN.

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
    read = []
    for f in BOARD_FILES + BANK_FILES:
        d = _read(OUT / f, ["league"])
        if d.empty or "league" not in d.columns:
            continue
        read.append(f)
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
    if not read:
        # nothing was read, so nothing was checked (it read GREEN before 2026-11-02)
        return _c("Data truth 1: one canonical name", GREY, "no board, ledger or bank to check", rule)
    unread = [f for f in BOARD_FILES + BANK_FILES if f not in read]
    if unread:
        detail = (detail + "; " if detail else "") + f"not checked (missing/unreadable): {unread}"
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
        if d.empty or "date" not in d.columns:
            # a missing history bank is missing data, not a fresh one (it read GREEN before)
            issues.append(f"{f} missing or unreadable")
            worst = RED
            continue
        newest = pd.to_datetime(d["date"], errors="coerce", utc=True).max()
        if pd.isna(newest):
            issues.append(f"{f} has no readable match date")
            worst = RED
            continue
        age = (pd.Timestamp(now) - newest).days
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
    issues, unread = [], []
    for f in ("af_history.parquet", "backtest_all_leagues.csv"):
        d = _read(OUT / f, ["league", "season"])
        if d.empty:
            unread.append(f)
            continue
        n = len(d) - len(DL._drop_misfiled_league_rows(d, f"data_truth:{f}"))
        if n:
            issues.append(f"{f}: {n} misfiled row(s)")
    errs = validate(load_registry())
    if errs:
        issues.append(f"registry: {len(errs)} error(s) e.g. {errs[0]}")
    if not DL._safe_joins():
        issues.append("enrichment joins are NOT league-keyed (WOWZA_SAFE_ENRICH_JOINS=0)")
    if issues:
        return _c("Data truth 4: right league relations", RED, "; ".join(issues), rule)
    if unread:
        # an unread bank was not checked: NO_DATA, never "banks clean" (GREEN before 2026-11-02)
        return _c("Data truth 4: right league relations", GREY,
                  f"not checked (missing/unreadable): {unread}", rule)
    return _c("Data truth 4: right league relations", GREEN,
              "banks clean, registry valid, joins league-keyed", rule)


# ── 5 ONE MATCH, ONE ROW ─────────────────────────────────────────────────────────────────────────
_MATCH_COLS = ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals"]


def _bank_matches(f: str) -> pd.DataFrame:
    d = _read(OUT / f, _MATCH_COLS)
    if d.empty or not set(_MATCH_COLS) - {"season"} <= set(d.columns):
        return pd.DataFrame()
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("home_goals", "away_goals"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])


def one_match(banks: dict | None = None) -> dict:
    """The banks the training frame is built from, merged and put through the loader's own identity
    pipeline (src/data_loader.one_club_one_match - the same code train and serve run), must hold
    no match twice. `banks` {name: frame} overrides the committed files (tests)."""
    rule = ("one match, one row: fd_history + the backtest archive, through data_loader.one_club_one_match, "
            "hold 0 duplicate-match pairs (same league, canonical clubs, same score, dates <= 3 d apart; or "
            "same day, same score, one side the same club) and 0 rows whose last-5 window holds their own "
            "match: RED on any; NO_DATA if fd_history is missing")
    from src import data_loader as DL
    from src.integrity.duplicate_matches import duplicate_pairs, self_containing_rows
    if banks is None:
        banks = {"fd_cache": _bank_matches("fd_history.parquet"),
                 "backtest_archive": _bank_matches("backtest_all_leagues.csv")}
    fd = banks.get("fd_cache", pd.DataFrame())
    if fd is None or fd.empty:
        return _c("Data truth 5: one match, one row", GREY, "no fd_history to check", rule)
    parts = []
    for src, d in banks.items():
        if d is None or d.empty:
            continue
        if src == "backtest_archive":
            d = DL._drop_misfiled_league_rows(d, "data_truth:archive")
        parts.append(d.assign(row_source=src))
    merged = pd.concat(parts, ignore_index=True).sort_values("date", kind="mergesort").reset_index(drop=True)
    clean = DL.one_club_one_match(merged)
    pairs = duplicate_pairs(clean)
    sc = self_containing_rows(clean, pairs)
    removed = len(merged) - len(clean)
    st = RED if (len(pairs) or int(sc.sum())) else GREEN
    ex = "; ".join(f"{p.league}: rows {p.i}/{p.j} {p.gap_days} d ({p.kind})" for p in pairs.head(3).itertuples())
    return _c("Data truth 5: one match, one row", st,
              f"{len(pairs)} duplicate pair(s), {int(sc.sum())} self-containing row(s) after the loader's "
              f"identity pipeline ({len(merged):,} bank rows -> {len(clean):,}; {removed:,} copies removed)",
              rule, ex)


def evaluate(now: datetime | None = None) -> list[dict]:
    """The rule checks. Never raises: a check that cannot run is itself RED."""
    out = []
    # A crashed check carries its rule's label, so a gated rule that cannot run blocks the push (it
    # was named after the function - 'one_match' - which the GATED prefixes never matched).
    for label, fn in (("Data truth 1: one canonical name", naming),
                      ("Data truth 2: no silent defaults", defaults),
                      ("Data truth 3: no missing data", lambda: missing(now)),
                      ("Data truth 4: right league relations", leagues),
                      ("Data truth 5: one match, one row", lambda: one_match())):
        try:
            out.append(fn())
        except Exception as e:                                         # noqa: BLE001
            out.append(_c(label, RED, "check failed", "must run", f"{type(e).__name__}: {e}"[:200]))
    return out


def write_baseline() -> int:
    """Record today's remaining default sites for owner review (each needs a reason)."""
    prev = {(e["file"], e["code"]): e.get("reason", "") for e in _baseline().get("entries", [])}
    hits = scan_defaults()
    entries = [{**h, "reason": prev.get((h["file"], h["code"]), "TODO: reason")} for h in hits]
    BASELINE.write_text(json.dumps({"version": VERSION, "entries": entries}, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    return len(entries)


# ── STOPLIGHT GATE (owner 2026-11-01: "we can't go back from 100% data") ─────────────────────────
RATCHET = ROOT / "config" / "data_truth_ratchet.json"
#: rules whose RED blocks a push/PR. Rule 3 (missing data) is operational - an API outage must not
#: block code - and already turns system health RED daily via evidence_refresh. Rule 5 (one match,
#: one row) is gated since 2026-11-02 (owner): a code change that lets a match into the banks twice
#: must not merge. It runs the loader's identity pipeline on the committed banks (~15 s; it was
#: ~19-60 s before team_names memoised norm/identity_key and duplicate_pairs stopped slicing
#: pandas groups).
GATED = ("Data truth 1", "Data truth 2", "Data truth 4", "Data truth 5")


def _ratchet() -> dict:
    return json.loads(RATCHET.read_text(encoding="utf-8"))


def gate(previous_ceiling: int | None = None, checks: list[dict] | None = None) -> list[str]:
    """Reasons the stoplight is RED for this commit ([] = green). The known-defaults ceiling in
    config/data_truth_ratchet.json may only go DOWN: the baseline may not exceed it, and the ceiling
    may not rise above the previous commit's."""
    fails = []
    for c in checks if checks is not None else evaluate():
        if c["status"] == RED and c["check"].startswith(GATED):
            fails.append(f"{c['check']}: {c['value']} {c['detail']}".strip())
    ceiling = int(_ratchet()["max_known_defaults"])
    n = len(_baseline().get("entries", []))
    if n > ceiling:
        fails.append(f"known-defaults baseline has {n} entries > ratchet ceiling {ceiling} "
                     "(remove the default instead of baselining it)")
    if previous_ceiling is not None and ceiling > previous_ceiling:
        fails.append(f"ratchet ceiling raised {previous_ceiling} -> {ceiling}: it may only go down")
    return fails


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--baseline" in argv:
        print(f"baseline: {write_baseline()} entr(ies) -> {BASELINE.relative_to(ROOT)}")
        return 0
    if "--gate" in argv:
        prev = None
        if "--previous-ceiling" in argv:
            v = argv[argv.index("--previous-ceiling") + 1]
            prev = int(v) if v.strip().isdigit() else None
        checks = evaluate()
        for c in checks:
            print(f"{c['status']:7} {c['check']}: {c['value']}")
        fails = gate(prev, checks)
        for f in fails:
            print(f"::error title=Data truth stoplight RED::{f}")
        print("STOPLIGHT", "RED" if fails else "GREEN")
        return 1 if fails else 0
    checks = evaluate()
    doc = {"artifact": "data_truth_health", "version": VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "checks": checks}
    HEALTH.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    for c in checks:
        print(f"{c['status']:7} {c['check']}: {c['value']}" + (f"  [{c['detail']}]" if c["detail"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
