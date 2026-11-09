"""Data Truth checks - the owner's four rules as RED/AMBER/GREEN health lines (2026-11-01).

    python -m src.integrity.data_truth            # -> output/data_truth_health.json (+ printed)
    python -m src.integrity.data_truth --baseline # rewrite config/data_truth_baseline.json (owner review)

The 56-finding audit of 2026-11-01 fixed the violations it found. These checks exist so the next one
turns system health RED instead of waiting for the next audit:

    1 NAMING     every league value in the committed boards, ledgers and history banks is a
                 registry league (canonical or alias); blind board fixtures are counted.
    2 DEFAULTS   no NEW invented default in production code: fillna(<non-zero literal>), literal
                 season fallbacks, literal season-year lists, pinned football-data folders, and
                 (fix9/defaults 2026-11-04) numeric `.get(<key>, <number>)` defaults, `or 0` on a
                 parsed value and fillna(0) on a price / odds. Scanned: ALL production code (not tests,
                 not scripts/audit). Known remaining ones live in config/data_truth_baseline.json,
                 each with a reason; genuine counters and display code are named, with a reason each,
                 in config/data_truth_allowlist.json.
    3 MISSING    the live board carries its prices and, on EVERY served model track (standard and
                 new_format x O/U 2.5 / BTTS / Over 1.5 / Over 3.5, HT O/U on standard), every input
                 the scoring payload needs plus the HT form and rest days (2026-11-04: it checked two
                 Standard columns only, so the NF HT gap of 2026-11-02 stayed GREEN); banks fresh.
    4 LEAGUES    no misfiled league rows in the banks, the registry validates, joins serve
                 league-keyed.
    5 ONE MATCH  the banks, through the loader's identity pipeline, hold no match twice (same day
                 under two spellings, or 1-3 days apart) and no row whose form window holds its own
                 match (2026-11-02).
    6 NO GLOBAL  (owner 2026-11-09: "no global vars") no serving module reads a global tier bar:
                 config.HISTORICAL_SIDE_GLOBAL_BARS / SIDE_GLOBAL_BARS, VALUABLE_THRESHOLD /
                 VALUE_THRESHOLD, MARKSMAN_THRESHOLD, SNIPER_THRESHOLD(_OVER/_UNDER), EDGE_CEILING,
                 LEAGUE_EDGE_CEILING, the hand-set LEAGUE_SNIPER/MARKSMAN tables, or writes the GLOBAL
                 threshold source. Tiers come only from each league's own owner-applied bars.

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

ALLOWLIST = ROOT / "config" / "data_truth_allowlist.json"

# ALL production code (fix9/defaults 2026-11-04). Until then rule 2 scanned a hand-picked list of 20
# files, so a wide scan of data-truth-100 on 2026-11-04 still found typed defaults in the side-market
# tier bars, the FPL minutes baseline, prop prices, Kelly and the dashboard's tier P/L. Now every .py
# file of the repository is scanned except tests, the audit scripts and non-code folders.
SCAN_EXCLUDE_DIRS = ("tests/", "scripts/audit/", "_archive/", "output/", "docs/", "reports/", "deploy/")


def production_files(root: Path | None = None) -> list[str]:
    """Every production .py file (repo-relative, '/'-separated, sorted)."""
    root = root or ROOT
    out = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(x.startswith(".") or x in ("__pycache__", "node_modules", "site-packages") for x in parts):
            continue
        if rel.startswith(SCAN_EXCLUDE_DIRS):
            continue
        out.append(rel)
    return sorted(out)


SCAN_FILES = production_files()
DEFAULT_PATTERNS = {
    "fillna_literal": re.compile(r"\.fillna\(\s*-?(?:0*[1-9]\d*(?:\.\d+)?|0?\.\d*[1-9]\d*)\s*\)"),
    "season_literal_fallback": re.compile(r"""\.get\([^()]*,\s*["']20\d\d["']\s*\)"""),
    "season_year_list": re.compile(r"20\d\d\|20\d\d"),
    "pinned_fd_folder": re.compile(r"mmz4281/\d{4}/"),
    # fix9/defaults 2026-11-04 - the classes the 2026-11-04 wide scan found and the 4 patterns missed:
    #   d.get("price", 0) / lp.get("sniper_th", 0.10) - a number typed in for a field that was not sent
    "numeric_get_default": re.compile(r"""\.get\(\s*[^(),]+?,\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*\)"""),
    #   int(x or 0) / float(v.get("n") or 0.0) - "no value" read as a real zero
    "or_zero": re.compile(r"""\bor\s+0(?:\.0*)?(?![\d.eE])"""),
    #   odds.fillna(0) - a missing price turned into a price of 0 (Kelly, EV, staking)
    "fillna_zero_price": re.compile(r"""(?i)(?:odds|price)[^#\n]*\.fillna\(\s*0(?:\.0*)?\s*\)"""),
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
def _docstring_lines(text: str) -> set[int]:
    """1-based line numbers inside a string literal that is a statement of its own (a docstring or
    a bare string block). Prose there describes old code ("it was `int(x or 0)`") and is not code;
    the line-based scan used to skip only lines that START with a quote."""
    import io
    import tokenize
    skip: set[int] = set()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return skip
    prev = None
    for t in toks:
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING):
            continue
        if t.type == tokenize.STRING and prev in (None, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            skip.update(range(t.start[0], t.end[0] + 1))
        prev = t.type
    return skip


def scan_file(rel: str, text: str) -> list[dict]:
    """Every invented-default site of one file: {file, kind, code}. Comments and docstrings are ignored."""
    hits = []
    skip = _docstring_lines(text)
    for i, ln in enumerate(text.splitlines(), start=1):
        if i in skip:
            continue
        code = ln.split("#", 1)[0]
        s = code.strip()
        # A line that merely STARTS with a quote is scanned since fix9/defaults: a continuation line
        # such as  "RETROSPECTIVE_ONLY" if clv.get("n", 0) else "NONE"  is code (docstrings are skipped
        # above by the tokenizer, which the old start-with-a-quote rule only approximated).
        if not s:
            continue
        for kind, pat in DEFAULT_PATTERNS.items():
            if pat.search(code):
                hits.append({"file": rel, "kind": kind, "code": s})
    return hits


def scan_defaults(files: list[str] | None = None) -> list[dict]:
    """Every invented-default site in SCAN_FILES (all production code): {file, kind, code}."""
    hits = []
    for rel in (SCAN_FILES if files is None else files):
        p = ROOT / rel
        if not p.exists():
            continue
        hits += scan_file(rel, p.read_text(encoding="utf-8", errors="replace"))
    return hits


def _baseline() -> dict:
    try:
        return json.loads(BASELINE.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {"entries": []}


def _allowlist() -> dict:
    """config/data_truth_allowlist.json: sites that are NOT invented data - genuine counters (a tally
    that starts at zero), lookup tables with a fail-closed fallback, config switches and display
    code - each named exactly ({file, kind, code}) with its reason. NOT a baseline: a real default
    is fixed, never listed here; an entry without a reason is refused (RED)."""
    try:
        return json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {"entries": []}


def defaults() -> dict:
    rule = ("no NEW invented default in ALL production code (fillna(<non-zero literal>), literal season "
            "fallback/list, pinned football-data folder, numeric .get(<key>, <number>), an or-zero on a "
            "parsed value, a zero fillna on a price/odds): RED if any site is in neither config/data_truth_baseline.json "
            "nor config/data_truth_allowlist.json, or an allow-list entry has no reason; GREEN otherwise")
    known = {(e["file"], e["code"]) for e in _baseline().get("entries", [])}
    allow_entries = _allowlist().get("entries", [])
    allowed = {(e["file"], e["code"]) for e in allow_entries}
    unreasoned = [e for e in allow_entries if not str(e.get("reason") or "").strip()]
    hits = scan_defaults()
    new = [h for h in hits if (h["file"], h["code"]) not in known | allowed]
    n_known = sum((h["file"], h["code"]) in known for h in hits)
    n_allowed = sum((h["file"], h["code"]) in allowed and (h["file"], h["code"]) not in known for h in hits)
    seen = {(h["file"], h["code"]) for h in hits}
    gone = len(known - seen)
    stale = len(allowed - seen)
    detail = "; ".join(f"{h['file']}: {h['code'][:80]}" for h in new[:5])
    if unreasoned:
        detail = (detail + "; " if detail else "") + "allow-list entries without a reason: " + "; ".join(
            f"{e.get('file')}: {str(e.get('code'))[:60]}" for e in unreasoned[:5])
    return _c("Data truth 2: no silent defaults", RED if new or unreasoned else GREEN,
              f"{len(new)} new default(s); {n_known} known (baseline); {n_allowed} allow-listed (counters/"
              f"display); {gone} baseline entr(ies) now gone; {stale} allow-list entr(ies) now gone; "
              f"{len(SCAN_FILES)} production files scanned", rule, detail)


# ── 3 MISSING ────────────────────────────────────────────────────────────────────────────────────
#: Rule 3 covers EVERY model track since 2026-11-04 (owner). Until then it checked two columns on
#: Standard fixtures only, so the new-format half-time gap of 2026-11-02 (NF fixtures served with
#: their HT form missing) stayed GREEN. Each served (model_type, market) is checked on the live
#: board against the feature list of the payload that actually scores it, plus the HT form inputs
#: and rest days explicitly (they are the inputs the 2026-11-02 gap lost).
MODEL_TYPES = ("standard", "new_format")
#: market -> (config attribute of the payload file, model types it serves). O/U 2.5 is per model type
#: (standard / newformat payloads); the side markets are one payload serving both model types
#: (or a track-split payload since 2026-11-02); HT O/U serves Standard fixtures only (src/predict.py 5b).
MARKET_PAYLOADS = {"BTTS": ("MODEL_FILE_BTTS", MODEL_TYPES), "Over 1.5": ("MODEL_FILE_OVER15", MODEL_TYPES),
                   "Over 3.5": ("MODEL_FILE_OVER35", MODEL_TYPES),
                   "HT Over 0.5": ("HT_MODEL_FILE_05", ("standard",)),
                   "HT Over 1.5": ("HT_MODEL_FILE_15", ("standard",))}
#: the HT form + rest inputs checked on every track whatever its payload lists (2026-11-02 gap)
HT_FORM_INPUTS = ("home_ht_over05_rate", "away_ht_over05_rate", "home_ht_over15_rate", "away_ht_over15_rate",
                  "home_ht_scored_last5", "away_ht_scored_last5", "home_ht_conceded_last5",
                  "away_ht_conceded_last5")
REST_INPUTS = ("home_rest_days", "away_rest_days")
MISSING_RED_PCT, MISSING_AMBER_PCT = 50.0, 20.0


def _payload_cols(payload: dict, model_type: str):
    """Feature list that scores `model_type` rows in `payload`; None = the payload serves no model
    for that track (a track-split payload whose routed track is None: no probability, not a gap)."""
    if payload.get("tracks") is not None and model_type != "standard":
        sub = (payload.get("tracks") or {}).get(model_type)
        return None if sub is None else list(sub.get("feature_cols") or [])
    return list(payload.get("feature_cols") or [])


def serving_features() -> dict:
    """{(model_type, market): [feature columns] | None | "UNREADABLE: why"} for every served track,
    read from the committed model payloads (the same files src/predict.py loads)."""
    import config
    from src.model import load_models
    out, cache = {}, {}

    def _load(attr):
        if attr not in cache:
            try:
                cache[attr] = load_models(model_file=getattr(config, attr))
            except Exception as e:                                     # noqa: BLE001
                cache[attr] = f"UNREADABLE: {type(e).__name__}"
        return cache[attr]
    recipe = str(getattr(config, "STANDARD_OU25_SERVING_RECIPE", "legacy_v9"))
    std_attr = "MODEL_FILE_STANDARD" if recipe == "legacy_v9" else "MODEL_FILE_STANDARD_REPAIRED"
    for mt, attr in (("standard", std_attr), ("new_format", "MODEL_FILE_NEWFORMAT")):
        pl = _load(attr)
        out[(mt, "O/U 2.5")] = pl if isinstance(pl, str) else _payload_cols(pl, mt)
    for market, (attr, mts) in MARKET_PAYLOADS.items():
        pl = _load(attr)
        for mt in mts:
            out[(mt, market)] = pl if isinstance(pl, str) else _payload_cols(pl, mt)
    return out


def track_gaps(p: pd.DataFrame, features: dict) -> dict:
    """Per served track: fixtures, worst missing share, the columns over the AMBER line, status.
    A column the payload needs but the board does not carry counts as 100% missing."""
    out = {}
    mt_col = p.get("model_type", pd.Series("", index=p.index)).astype(str)
    for (mt, market), cols in features.items():
        key = f"{mt}|{market}"
        sub = p[mt_col == mt]
        if isinstance(cols, str):
            out[key] = {"status": AMBER, "fixtures": int(len(sub)), "worst_pct": None,
                        "missing": {}, "note": f"payload {cols} - not checked"}
            continue
        if cols is None:
            out[key] = {"status": GREEN, "fixtures": int(len(sub)), "worst_pct": None, "missing": {},
                        "note": "payload serves no model for this track (no probability, no tip)"}
            continue
        if sub.empty:
            out[key] = {"status": GREY, "fixtures": 0, "worst_pct": None, "missing": {}, "note": "no fixtures"}
            continue
        need = list(dict.fromkeys(list(cols) + list(HT_FORM_INPUTS) + list(REST_INPUTS)))
        if market.startswith("HT"):
            need = list(dict.fromkeys(list(cols) + list(HT_FORM_INPUTS)))
        share = {c: (float(pd.to_numeric(sub[c], errors="coerce").isna().mean() * 100) if c in sub.columns
                     else 100.0) for c in need}
        gaps = {c: round(v, 1) for c, v in sorted(share.items(), key=lambda kv: -kv[1]) if v > MISSING_AMBER_PCT}
        worst = max(share.values()) if share else 0.0
        st = RED if worst > MISSING_RED_PCT else AMBER if worst > MISSING_AMBER_PCT else GREEN
        out[key] = {"status": st, "fixtures": int(len(sub)), "worst_pct": round(worst, 1), "missing": gaps,
                    "n_inputs": len(need)}
    return out


def missing(now: datetime | None = None, features: dict | None = None) -> dict:
    rule = ("live board: every fixture priced (odds_over25); on EVERY served model track (standard and "
            "new_format x O/U 2.5, BTTS, Over 1.5, Over 3.5; HT O/U on standard) each input the scoring "
            "payload needs, plus the HT form and rest-day inputs, is present (RED if > 50% missing on a "
            "track, AMBER if > 20%); history banks fresh (AMBER if the newest match is > 10 days old)")
    now = now or datetime.now(timezone.utc)
    p = _read(OUT / "predictions.csv")
    if p.empty:
        return _c("Data truth 3: no missing data", GREY, "no board", rule)
    issues, worst = [], GREEN
    unpriced = int(p["odds_over25"].isna().sum()) if "odds_over25" in p.columns else len(p)
    if unpriced:
        issues.append(f"{unpriced} unpriced fixture(s)")
        worst = RED
    tracks = track_gaps(p, serving_features() if features is None else features)
    rank = {GREEN: 0, GREY: 0, AMBER: 1, RED: 2}
    for key, t in tracks.items():
        if rank[t["status"]] > rank[worst]:
            worst = t["status"]
        if t["status"] in (AMBER, RED):
            top = ", ".join(f"{c} {v:.0f}%" for c, v in list(t["missing"].items())[:4])
            issues.append(f"{key}: {t.get('note') or top} ({t['fixtures']} fixtures)")
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
    out = _c("Data truth 3: no missing data", worst,
             "; ".join(issues) or f"board priced, features present on {len(tracks)} served tracks", rule)
    out["tracks"] = tracks
    return out


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


# ── 6 NO GLOBAL BAR IN SERVING ───────────────────────────────────────────────────────────────────
#: the modules that tier, gate or stake a tip (and the gate mirror they share). A file listed here that
#: does not exist on this branch is skipped (the TOP-3 selection lives on its own branch until merged).
SERVING_FILES = ("pipeline.py", "src/betting.py", "src/side_thresholds.py", "src/gate_resolver.py",
                 "src/integrity/side_edge_devig.py", "src/stake_plan.py", "src/bankroll_policy.py",
                 "src/cloudbet/bot.py", "src/top3_league.py", "src/bars_opt/serve.py", "src/bars_opt/cells.py")
#: global / pooled / hand-set bars that served before 2026-11-09 (now historical values in config.py)
GLOBAL_BAR_NAMES = ("HISTORICAL_SIDE_GLOBAL_BARS", "SIDE_GLOBAL_BARS", "VALUABLE_THRESHOLD", "VALUE_THRESHOLD",
                    "MARKSMAN_THRESHOLD", "SNIPER_THRESHOLD", "SNIPER_THRESHOLD_OVER", "SNIPER_THRESHOLD_UNDER",
                    "EDGE_CEILING", "LEAGUE_EDGE_CEILING", "LEAGUE_SNIPER_THRESHOLDS",
                    "LEAGUE_MARKSMAN_THRESHOLDS", "_LEAGUE_SNIPER_CALIBRATED", "_LEAGUE_MARKSMAN_CALIBRATED",
                    "THRESHOLD_SOURCE_GLOBAL",
                    # 2026-11-09 (bars_opt serving): the global odds limits and the assumed-margin fair-price
                    # switch - a served tip uses its league's own cell odds range and a real two-sided de-vig
                    "MIN_OVER_ODDS", "MIN_UNDER_ODDS", "NF_MAX_OU_ODDS", "SIDE_EDGE_FAIR_PRICE")
_GLOBAL_BAR_RE = re.compile(
    r"\b(?:app_)?config\s*\.\s*(" + "|".join(GLOBAL_BAR_NAMES) + r")\b"
    r"|getattr\(\s*(?:app_)?config\s*,\s*[\"'](" + "|".join(GLOBAL_BAR_NAMES) + r")[\"']"
    r"|\bglobal_bars\s*\(")


def scan_global_bars(files=None, root: Path | None = None) -> list[dict]:
    """Every code line (not a comment or docstring) of a serving module that reads a global bar."""
    r = Path(root) if root is not None else ROOT
    hits = []
    for rel in (SERVING_FILES if files is None else files):
        p = r / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        skip = _docstring_lines(text)
        for i, ln in enumerate(text.splitlines(), start=1):
            code = ln.split("#", 1)[0]
            if i in skip or not code.strip():
                continue
            if _GLOBAL_BAR_RE.search(code):
                hits.append({"file": rel, "line": i, "code": code.strip()})
    return hits


def no_global_bars() -> dict:
    rule = ("no serving module (tiering, gates, staking) reads a global / pooled / hand-set tier bar or "
            "writes the GLOBAL threshold source: every tier is the league's own owner-applied bars "
            "(src/side_thresholds.own_bars) or none (NO_LEAGUE_BAR); RED on any hit")
    hits = scan_global_bars()
    scanned = sum((ROOT / f).exists() for f in SERVING_FILES)
    return _c("Data truth 6: no global bars in serving", RED if hits else GREEN,
              f"{len(hits)} global-bar read(s) in {scanned} serving module(s)", rule,
              "; ".join(f"{h['file']}:{h['line']}: {h['code'][:70]}" for h in hits[:5]))


def evaluate(now: datetime | None = None) -> list[dict]:
    """The rule checks. Never raises: a check that cannot run is itself RED."""
    out = []
    # A crashed check carries its rule's label, so a gated rule that cannot run blocks the push (it
    # was named after the function - 'one_match' - which the GATED prefixes never matched).
    for label, fn in (("Data truth 1: one canonical name", naming),
                      ("Data truth 2: no silent defaults", defaults),
                      ("Data truth 3: no missing data", lambda: missing(now)),
                      ("Data truth 4: right league relations", leagues),
                      ("Data truth 5: one match, one row", lambda: one_match()),
                      ("Data truth 6: no global bars in serving", no_global_bars)):
        try:
            out.append(fn())
        except Exception as e:                                         # noqa: BLE001
            out.append(_c(label, RED, "check failed", "must run", f"{type(e).__name__}: {e}"[:200]))
    return out


def write_baseline() -> int:
    """Record today's remaining default sites for owner review (each needs a reason)."""
    prev = {(e["file"], e["code"]): e.get("reason", "") for e in _baseline().get("entries", [])}
    allowed = {(e["file"], e["code"]) for e in _allowlist().get("entries", [])}
    hits = [h for h in scan_defaults() if (h["file"], h["code"]) not in allowed]
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
GATED = ("Data truth 1", "Data truth 2", "Data truth 4", "Data truth 5", "Data truth 6")


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
    # the gate verdict travels with the checks (dashboard data truth panel, 2026-11-05), so a reader
    # never re-derives it from the statuses
    fails = gate(checks=checks)
    doc = {"artifact": "data_truth_health", "version": VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "checks": checks,
           "stoplight": "RED" if fails else "GREEN", "gate_reasons": fails}
    HEALTH.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    for c in checks:
        print(f"{c['status']:7} {c['check']}: {c['value']}" + (f"  [{c['detail']}]" if c["detail"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
