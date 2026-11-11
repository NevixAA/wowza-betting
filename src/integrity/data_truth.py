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
                 match (2026-11-02); and (2026-11-10) no PROVIDER fixture claimed by two served rows,
                 nor by two served af_history provider rows (src.integrity.fixture_claims: a match
                 filed under both clubs' old AND new names has no side in common by name).
    6 NO GLOBAL  (owner 2026-11-09: "no global vars") no serving module reads a global tier bar:
                 config.HISTORICAL_SIDE_GLOBAL_BARS / SIDE_GLOBAL_BARS, VALUABLE_THRESHOLD /
                 VALUE_THRESHOLD, MARKSMAN_THRESHOLD, SNIPER_THRESHOLD(_OVER/_UNDER), EDGE_CEILING,
                 LEAGUE_EDGE_CEILING, the hand-set LEAGUE_SNIPER/MARKSMAN tables, or writes the GLOBAL
                 threshold source. Tiers come only from each league's own owner-applied bars.
    9 LEDGERS    (2026-11-11) from the Clean Record start the live ledgers hold no multi-tip / opposite-tip
                 fixture, post-kickoff tip, close <= 1.0, |CLV| > 50 new-format row, duplicate player tip
                 or twice-logged v91 fixture (src/ledger_corrections corrects the rows before it). Not
                 gated (operational data, like rule 3): RED in system health.

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
from src.imputer import MISSING_SUFFIX  # noqa: E402  the one definition of the indicator suffix


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
        # "<f>__missing" is the imputer's 0/1 indicator, built from <f> at scoring time (src/imputer.py),
        # so no board ever carries it: its base column <f> is the input checked. An indicator whose base
        # is not in the payload stays in `need` and counts as missing.
        cols = [c for c in cols if not (c.endswith(MISSING_SUFFIX) and c[:-len(MISSING_SUFFIX)] in cols)]
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


# ── 3 (2026-11-10) SERVED FEATURES CARRY DATA ────────────────────────────────────────────────────
# A served model feature whose training rows hold ONE value (a placeholder written on every row, or
# NaN on every row) is not data, whatever its name. Found 2026-11-10: the 2026-11-09 props payloads
# list six league-quality columns that are NaN on every training row (league_quality.LEGACY_COLUMNS),
# and player_history stores age 25 / 180 cm / 75 kg on 100% of rows (read as missing by the feature
# code; this check proves no served model learned them). Evidence, from the payload itself:
#   * training_profile_ (props harness since 2026-11-10): the most common value and its share over
#     ALL training rows -> RED above SINGLE_VALUE_MAX_SHARE;
#   * otherwise what the fitted estimators prove: a column the HGB usable-columns step dropped (at most
#     one distinct real value on every fit row), a scaler variance of 0 (one value after the fill),
#     a team imputer feature whose training min == max or that is missing on > SINGLE_VALUE_MAX_SHARE.
SINGLE_VALUE_MAX_SHARE = 0.999
#: a props market hidden from tips and the board: its payload is listed, never counted
NOT_SERVED = "NOT_SERVED"
#: values the old build_features wrote into every player_history row (and the arithmetic on them)
KNOWN_PLACEHOLDERS = {"age": 25.0, "height_cm": 180.0, "weight_kg": 75.0, "age_peak_delta": 0.0,
                      "opp_def_mean_height": 180.0, "height_diff_vs_opp_def": 0.0,
                      "height_advantage_score": 0.0}


def _num_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _single_value(feature: str, value, share, how: str) -> dict:
    v = _num_or_none(value)
    ph = v is not None and KNOWN_PLACEHOLDERS.get(feature) == v
    return {"value": "NaN" if v is None else v, "share": None if share is None else round(float(share), 6),
            "how": how, "placeholder": bool(ph)}


def _estimator_constants(cols: list[str], calibrated) -> dict:
    """{feature: finding} the fitted pipeline proves constant over its fit rows."""
    out = {}
    pipe = getattr(calibrated, "_pipe", None)
    for _, step in (getattr(pipe, "steps", None) or []):
        keep = getattr(step, "keep_", None)
        if keep is not None:
            kept = set(keep)
            for j, c in enumerate(cols):
                if j not in kept:
                    out[c] = _single_value(c, None, None, "dropped at fit: at most one distinct value on every fit row")
        var = getattr(step, "var_", None)
        if var is not None and len(var) == len(cols):
            mean = getattr(step, "mean_", None)
            for j, c in enumerate(cols):
                if float(var[j]) == 0.0:
                    out.setdefault(c, _single_value(c, None if mean is None else mean[j], 1.0,
                                                    "scaler variance 0 on every fit row"))
    return out


def props_payload_spread(payload: dict) -> dict:
    """{feature: finding} for the served features of one props payload that hold a single value."""
    cols = list(payload.get("feature_cols") or [])
    m = payload.get("model")
    prof = getattr(m, "training_profile_", None)
    if isinstance(prof, dict) and prof:
        return {c: _single_value(c, prof[c].get("top_value"), prof[c].get("top_share"), "training_profile")
                for c in cols if c in prof and prof[c].get("top_share") is not None
                and prof[c]["top_share"] > SINGLE_VALUE_MAX_SHARE}
    out = {}
    fcols = list(getattr(m, "feature_cols_", None) or cols)
    if getattr(m, "model_", None) is not None:                    # hgb_nan_native
        out.update(_estimator_constants(fcols, m.model_))
    for sub in (getattr(m, "models_", None) or {}).values():      # recipe_retrained
        out.update({c: f for c, f in _estimator_constants(fcols, sub).items() if c not in out})
    if m is None:                                                 # legacy payload: {"models": {...}}
        for sub in (payload.get("models") or {}).values():
            out.update({c: f for c, f in _estimator_constants(cols, sub).items() if c not in out})
    return {c: f for c, f in out.items() if c in cols}


def team_payload_spread(payload: dict) -> dict:
    """{feature: finding} for a team payload, from its train-only imputer's per-feature record."""
    out = {}
    feats = ((payload.get("imputer") or {}).get("features") or {})
    for c in payload.get("feature_cols") or []:
        e = feats.get(c)
        if not isinstance(e, dict):
            continue
        mr = _num_or_none(e.get("missing_rate"))
        lo, hi = _num_or_none(e.get("min")), _num_or_none(e.get("max"))
        if mr is not None and mr > SINGLE_VALUE_MAX_SHARE:
            out[c] = _single_value(c, None, mr, "imputer: missing on the training rows")
        elif lo is not None and lo == hi:
            out[c] = _single_value(c, lo, None if mr is None else 1.0 - mr, "imputer: training min == max")
    return out


def _load_props_payload(path: Path) -> dict:
    import pickle

    class _U(pickle.Unpickler):                 # harness classes pickled under __main__
        def find_class(self, module, name):
            return super().find_class("player_model.props_harness" if module == "__main__" else module, name)
    with open(path, "rb") as f:
        return _U(f).load()


def served_feature_spread() -> dict:
    """{payload label: {"status", "single_value": {feature: finding}, "note"}} over every served
    payload: the team payloads src/predict.py loads and the props payloads of the markets tips use
    (config.HIDDEN_PENDING_CLEAN_RETRAIN markets are not served)."""
    import config
    from src.model import load_models
    out = {}
    recipe = str(getattr(config, "STANDARD_OU25_SERVING_RECIPE", "legacy_v9"))
    attrs = ["MODEL_FILE_STANDARD" if recipe == "legacy_v9" else "MODEL_FILE_STANDARD_REPAIRED",
             "MODEL_FILE_NEWFORMAT"] + [a for a, _ in MARKET_PAYLOADS.values()]
    for attr in dict.fromkeys(attrs):
        label = f"team:{Path(getattr(config, attr)).name}"
        try:
            pl = load_models(model_file=getattr(config, attr))
        except Exception as e:                                         # noqa: BLE001
            out[label] = {"status": RED, "single_value": {}, "note": f"unreadable: {type(e).__name__}"}
            continue
        parts = {label: pl}
        for mt, sub in (pl.get("tracks") or {}).items():
            if isinstance(sub, dict) and sub.get("feature_cols"):
                parts[f"{label}[{mt}]"] = {**pl, **sub}
        for lab, part in parts.items():
            sv = team_payload_spread(part)
            out[lab] = {"status": RED if sv else GREEN, "single_value": sv,
                        "note": "" if (part.get("imputer") or {}).get("features") else "no imputer record"}
    from player_model import config as pconfig
    for market, path in pconfig.MODEL_FILES.items():
        hidden = market in pconfig.HIDDEN_PENDING_CLEAN_RETRAIN
        label = f"props:{Path(path).name}"
        if not Path(path).exists():
            out[label] = {"status": RED, "single_value": {}, "note": "served payload missing"}
            continue
        try:
            sv = props_payload_spread(_load_props_payload(Path(path)))
        except Exception as e:                                         # noqa: BLE001
            out[label] = {"status": RED, "single_value": {}, "note": f"unreadable: {type(e).__name__}"}
            continue
        if hidden:                     # listed for the record; a hidden market tips nothing
            out[label] = {"status": NOT_SERVED, "single_value": sv,
                          "note": "hidden pending clean retrain (config.HIDDEN_PENDING_CLEAN_RETRAIN)"}
        else:
            out[label] = {"status": RED if sv else GREEN, "single_value": sv, "note": ""}
    return out


def _spread_issues(spread: dict) -> list[str]:
    msgs = []
    for lab, s in spread.items():
        if s["status"] != RED:
            continue
        if not s["single_value"]:
            msgs.append(f"{lab}: {s['note']}")
            continue
        fs = s["single_value"]
        ph = [c for c, f in fs.items() if f["placeholder"]]
        names = ", ".join(f"{c}={f['value']}" for c, f in list(fs.items())[:6])
        msgs.append(f"{lab}: {len(fs)} served feature(s) hold one value on the training rows "
                    f"({names}{', ...' if len(fs) > 6 else ''})"
                    + (f"; PLACEHOLDER: {', '.join(ph)}" if ph else ""))
    return msgs


def missing(now: datetime | None = None, features: dict | None = None,
            spread: dict | None = None) -> dict:
    rule = ("live board: every fixture priced (odds_over25); on EVERY served model track (standard and "
            "new_format x O/U 2.5, BTTS, Over 1.5, Over 3.5; HT O/U on standard) each input the scoring "
            "payload needs, plus the HT form and rest-day inputs, is present (RED if > 50% missing on a "
            "track, AMBER if > 20%); history banks fresh (AMBER if the newest match is > 10 days old); "
            "every served model feature (team and props payloads) carries data on its training rows "
            f"(RED if one value - a placeholder, a constant or NaN - covers > {SINGLE_VALUE_MAX_SHARE:.1%} "
            "of them)")
    now = now or datetime.now(timezone.utc)
    spread = served_feature_spread() if spread is None else spread
    sp_issues = _spread_issues(spread)
    p = _read(OUT / "predictions.csv")
    if p.empty:
        out = _c("Data truth 3: no missing data", RED if sp_issues else GREY,
                 "; ".join(["no board"] + sp_issues), rule)
        out["served_feature_spread"] = spread
        return out
    issues, worst = list(sp_issues), RED if sp_issues else GREEN
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
    out["served_feature_spread"] = spread
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


def _af_provider_rows() -> pd.DataFrame | None:
    """af_history as every reader serves it: the owner's duplicate lists applied
    (src.af_history_corrections), the rows the old loader wrote back dropped (never served,
    src.af_history_provenance). None when af_history is unreadable."""
    d = _read(OUT / "af_history.parquet")
    if d.empty:
        return None
    from src.af_history_corrections import apply_af_history_corrections
    from src.af_history_provenance import written_back_mask
    d = apply_af_history_corrections(d, tag="data_truth")
    return d[~written_back_mask(d).to_numpy()]


def _fixture_dupes(frame: pd.DataFrame, pool: pd.DataFrame, home_goals: str, away_goals: str,
                   fixture_col: str | None = None) -> pd.DataFrame:
    """Claims (src.integrity.fixture_claims) of every provider fixture two or more rows claim."""
    from src import league_registry as LR
    from src.integrity.fixture_claims import claims, duplicate_claims
    f = frame.copy()
    f["league"] = f["league"].map(lambda x: LR.canonical_name(x) if isinstance(x, str) else None)
    return duplicate_claims(claims(f, pool, home_goals=home_goals, away_goals=away_goals,
                                   fixture_col=fixture_col))


def one_match(banks: dict | None = None, pool: pd.DataFrame | None = None,
              af: pd.DataFrame | None = None) -> dict:
    """The banks the training frame is built from, merged and put through the loader's own identity
    pipeline (src/data_loader.one_club_one_match - the same code train and serve run), must hold
    no match twice. `banks` {name: frame} overrides the committed files (tests).

    2026-11-10 (fix/history-dupes): also by the PROVIDER's fixture id. A match filed under BOTH
    clubs' old and new names ('Tianjin Teda v Meizhou Kejia' / 'Tianjin Jinmen Tiger v Meizhou
    Hakka') has no side in common by name, so duplicate_pairs cannot see it; both rows claim one
    API-Football fixture (src.integrity.fixture_claims). RED when any provider fixture is claimed by
    two served rows, or by two served af_history provider rows. `pool` (the provider fixture lists)
    and `af` (af_history provider rows) override the committed files (tests); a missing provider
    pool is RED (the check could not run). With `banks` given and `af` not, af_history is not read."""
    rule = ("one match, one row: fd_history + the backtest archive + the provider supplement, through "
            "data_loader.one_club_one_match, "
            "hold 0 duplicate-match pairs (same league, canonical clubs, same score, dates <= 3 d apart; or "
            "same day, same score, one side the same club), 0 rows whose last-5 window holds their own "
            "match, and 0 provider fixtures (API-Football fixture id) claimed by two served rows or by two "
            "served af_history provider rows: RED on any, or when the provider fixture lists are missing; "
            "NO_DATA if fd_history is missing")
    from src import data_loader as DL
    from src.integrity.duplicate_matches import duplicate_pairs, self_containing_rows
    from src.integrity.fixture_claims import provider_pool
    read_af = banks is None and af is None
    if banks is None:
        from src.history_supplement import read_supplement
        banks = {"fd_cache": _bank_matches("fd_history.parquet"),
                 "backtest_archive": _bank_matches("backtest_all_leagues.csv"),
                 # provider fixtures no bank holds, as the loader adds them (owner 2026-11-10)
                 "provider_supplement": read_supplement().drop(columns=["provider_fixture_id"])}
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
    pool = provider_pool() if pool is None else pool
    if read_af:
        af = _af_provider_rows()
    no_pool = pool is None or pool.empty
    fx = (pd.DataFrame(columns=["fixture_id", "league", "date", "idx"]) if no_pool
          else _fixture_dupes(clean, pool, "home_goals", "away_goals"))
    afx = (pd.DataFrame(columns=["fixture_id", "league", "date", "idx"]) if no_pool or af is None or af.empty
           else _fixture_dupes(af, pool, "FTHG", "FTAG", fixture_col="fixture_id"))
    n_fx, n_afx = fx["fixture_id"].nunique(), afx["fixture_id"].nunique()
    st = RED if (len(pairs) or int(sc.sum()) or n_fx or n_afx or no_pool) else GREEN
    ex = "; ".join(f"{p.league}: rows {p.i}/{p.j} {p.gap_days} d ({p.kind})" for p in pairs.head(3).itertuples())
    for name, g in (("served", fx), ("af_history", afx)):
        for fid, gg in list(g.groupby("fixture_id", sort=False))[:3]:
            ex += (("; " if ex else "") + f"{name} {gg['league'].iloc[0]} {gg['date'].iloc[0]}: provider fixture "
                   f"{int(fid)} claimed by rows {list(gg['idx'])}")
    fx_txt = ("provider fixture check NOT RUN (no provider fixture lists)" if no_pool else
              f"{n_fx} provider fixture(s) claimed by 2+ served rows"
              + ("" if af is None else f", {n_afx} by 2+ af_history provider rows"))
    return _c("Data truth 5: one match, one row", st,
              f"{len(pairs)} duplicate pair(s), {int(sc.sum())} self-containing row(s) after the loader's "
              f"identity pipeline ({len(merged):,} bank rows -> {len(clean):,}; {removed:,} copies removed); "
              f"{fx_txt}",
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


# ── 7 NO RETIRED TIERS ON A FUTURE TIP ──────────────────────────────────────────────────────────
#: owner 2026-11-11 ("Clean rebuild": "the old sniper/mm/valuable dont important anymore for the future tips ...
#: do not use them"): from go-live (the Clean Record starts 2026-11-21) no tip row carries SNIPER / MARKSMAN /
#: VALUABLE or a tier multiplier other than 1. Historical rows keep their old columns (study only).
TIERS_RETIRED_FROM = "2026-11-21"
#: (file, date column, tier column, multiplier column) of every tip / stake record a future tip lands in
TIP_RECORDS = (("bets_ledger.csv", "match_date", "signal_tier", None),
               ("side_bets_ledger.csv", "match_date", "signal_tier", None),
               ("bets.csv", "date", "signal_tier", None), ("side_bets.csv", "date", "signal_tier", None),
               ("ht_ledger.csv", "match_date", "signal_tier", None),
               ("player_ledger.csv", "match_date", "tier", None),
               ("stake_audit.csv", "date", "signal_tier", "tier_multiplier"),
               ("top3_league_ledger.csv", "match_date", "tier", None))


def retired_tier_rows(out_dir: Path | None = None, since: str = TIERS_RETIRED_FROM) -> list[dict]:
    """Every record dated on/after `since` that carries a retired tier code or a tier multiplier != 1."""
    from src.tier_labels import LEGACY_TIERS
    od = Path(out_dir) if out_dir is not None else OUT
    cut = pd.Timestamp(since)
    hits = []
    for name, dcol, tcol, mcol in TIP_RECORDS:
        d = _read(od / name)
        if d.empty or dcol not in d.columns:
            continue
        # a record WRITTEN on/after the cut (generated_at where the record has it) - a row written before go-live
        # for a later match is history; without a write time the match date decides
        wcol = "generated_at" if "generated_at" in d.columns else dcol
        dt = pd.to_datetime(d[wcol].astype(str).str[:10], errors="coerce")
        fut = d[dt >= cut]
        if fut.empty:
            continue
        if tcol in fut.columns:
            n = int(fut[tcol].astype(str).str.strip().str.upper().isin(LEGACY_TIERS).sum())
            if n:
                hits.append({"file": name, "kind": "retired tier label", "rows": n})
        if mcol and mcol in fut.columns:
            m = pd.to_numeric(fut[mcol], errors="coerce")
            n = int((m.notna() & (m != 1.0)).sum())
            if n:
                hits.append({"file": name, "kind": "tier multiplier != 1", "rows": n})
    return hits


def no_retired_tiers() -> dict:
    rule = (f"owner 2026-11-11: no tip / stake record dated on or after {TIERS_RETIRED_FROM} carries a retired tier "
            "label (SNIPER / MARKSMAN / VALUABLE) or a tier multiplier other than 1; a tip qualifies only by its "
            "cell's one approved bar; RED on any hit")
    hits = retired_tier_rows()
    return _c("Data truth 7: no retired tiers on future tips", RED if hits else GREEN,
              f"{sum(h['rows'] for h in hits)} future row(s) with a retired tier / multiplier", rule,
              "; ".join(f"{h['file']}: {h['rows']} {h['kind']}" for h in hits[:6]))


# ── 8 BET-TIME EVIDENCE ONLY ──────────────────────────────────────────────────────────────────────
#: owner 2026-11-11 ("Bet-time snapshot + re-run", "Strict, keep Argentina"): a bar or verdict may never rest on a
#: price whose capture time is not proven before kickoff. For every APPROVED, NOT WITHDRAWN entry of
#: config/threshold_approvals.json: its proposal's price-provenance record must exist and list each of the entry's
#: cells with no OddsAPI-history-priced row and an unproven-capture share <= UNPROVEN_MAX - unless the owner
#: recorded an explicit exemption with its basis on the approval.
UNPROVEN_MAX = 0.5                      # owner 2026-11-11: "mainly" = more than 50% of the evidence rows
APPROVALS = ROOT / "config" / "threshold_approvals.json"


def unproven_evidence(approvals_path: Path | None = None, root: Path | None = None) -> list[dict]:
    r = Path(root) if root is not None else ROOT
    ap = json.loads(Path(approvals_path or APPROVALS).read_text(encoding="utf-8"))
    hits = []
    for a in ap.get("approvals") or []:
        rel = a.get("price_provenance")
        gone = {w["entry_id"] for w in a.get("withdrawals") or []}
        exempt = {x["entry_id"] for x in a.get("exemptions") or [] if str(x.get("basis") or "").strip()}
        live = [e["entry_id"] for e in a.get("entries") or [] if e["entry_id"] not in gone]
        if not rel:
            if live:
                hits.append({"proposal": a.get("proposal_id"), "entry": "*", "why": "no price-provenance record"})
            continue
        prov = json.loads((r / rel).read_text(encoding="utf-8"))
        by = {}
        for c in prov.get("cells") or []:
            by.setdefault(c["entry_id"], []).append(c)
        for eid in live:
            if eid in exempt:
                continue
            for c in by.get(eid, []):
                share = c.get("unproven_share")
                if c.get("oddsapi_hist"):
                    hits.append({"proposal": a["proposal_id"], "entry": eid, "why": "OddsAPI-history-priced evidence"})
                elif share is not None and share > UNPROVEN_MAX:
                    hits.append({"proposal": a["proposal_id"], "entry": eid,
                                 "why": f"{share:.0%} of evidence rows without a proven pre-kickoff capture time"})
    return hits


#: a CLOSING price column / tier name (football-data "C" columns, our *_close columns, the OddsAPI close tiers)
_CLOSING_RE = re.compile(r"(AvgC|MaxC|B365C|BbAvC|^PC[<>]|_close$|^close$)")


def closing_as_bet_price(root: Path | None = None) -> list[dict]:
    """Every place a CLOSING price would be read as a BET price (owner 2026-11-11: never - bars, backtests, Phase 2
    market benchmarks). Closing prices are for CLV only."""
    import inspect
    r = Path(root) if root is not None else ROOT
    hits = []
    from src import data_loader as DL                          # the served pre-match pair
    for name, oc, uc in DL._OU25_PREMATCH:
        if _CLOSING_RE.search(oc) or _CLOSING_RE.search(uc):
            hits.append({"where": "src/data_loader._OU25_PREMATCH", "what": f"{oc}/{uc}"})
    from src import league_bars as LB                          # the study's football-data bet pair
    src = inspect.getsource(LB.fd_ou25_pairs)
    for m in re.findall(r'"([^"]*[<>]2\.5)"', src):
        if _CLOSING_RE.search(m):
            hits.append({"where": "src/league_bars.fd_ou25_pairs", "what": m})
    from src import fd_prematch as F
    for name, oc, uc in F.PREMATCH:
        if _CLOSING_RE.search(oc) or _CLOSING_RE.search(uc):
            hits.append({"where": "src/fd_prematch.PREMATCH", "what": f"{oc}/{uc}"})
    cfg = json.loads((r / "config" / "phase2_retrain.json").read_text(encoding="utf-8"))   # Phase 2 benchmarks
    for mk, m in (cfg.get("markets") or {}).items():
        pr = (m or {}).get("price") or {}
        for k in ("yes", "no"):
            v = pr.get(k)
            if v and _CLOSING_RE.search(str(v)):
                hits.append({"where": f"config/phase2_retrain.json markets.{mk}.price.{k}", "what": v})
    proto = r / "config" / "bars_fresh_holdout_protocol.json"     # history tiers: bet role never a close tier
    if proto.exists():
        d = json.loads(proto.read_text(encoding="utf-8")).get("data") or {}
        both = set(d.get("price_tiers") or []) & set(d.get("close_tiers") or [])
        if both:
            hits.append({"where": "config/bars_fresh_holdout_protocol.json", "what": f"tier(s) {sorted(both)} both bet and close"})
    return hits


def bet_time_evidence() -> dict:
    rule = ("owner 2026-11-11: no approved, non-withdrawn bar or verdict rests on a price whose capture time is not "
            "proven before kickoff (no OddsAPI-history close price, unproven-capture share <= 50%), unless the owner "
            "recorded an exemption with its basis; RED on any hit")
    rule += ("; and no CLOSING price column / tier is ever read as a bet price (served pre-match pair, the "
             "study's football-data pair, Phase 2 market benchmarks, the history bet tiers) - closing is CLV only")
    hits = unproven_evidence()
    closing = closing_as_bet_price()
    return _c("Data truth 8: bars rest on bet-time prices only", RED if (hits or closing) else GREEN,
              f"{len(hits)} approved entr(ies) on unproven capture times; {len(closing)} closing-as-bet read(s)", rule,
              "; ".join([f"{h['entry']}: {h['why']}" for h in hits[:4]]
                        + [f"{h['where']}: {h['what']}" for h in closing[:4]]))


# ── 9 CLEAN LEDGERS (2026-11-11 certification) ─────────────────────────────────────────────────
def _clean_record_start():
    try:
        cfg = json.loads((ROOT / "config" / "trustworthy_performance.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    v = cfg.get("clean_record_start_utc")
    t = pd.to_datetime(v, errors="coerce", utc=True) if v else pd.NaT
    return None if pd.isna(t) else t


def ledgers(start=None, bets: pd.DataFrame | None = None, players: pd.DataFrame | None = None,
            v91: pd.DataFrame | None = None, pool: pd.DataFrame | None = None) -> dict:
    """The live ledgers hold, from the Clean Record start on, no fixture with two live O/U tips (opposite
    or not), no tip generated after the provider kick-off, no close <= 1.0, no |CLV| > 50 on a new-format
    row, no second copy of a player tip and no provider fixture logged twice in v91_shadow - the
    findings of the 2026-11-11 certification, corrected beside the ledgers (src/ledger_corrections).
    Rows before the start are the corrected 'before the repairs' record. NO_DATA while the start is not
    set (config/trustworthy_performance.json clean_record_start_utc, set by the owner at go-live)."""
    rule = ("from the Clean Record start: 0 multi-tip / opposite-tip fixtures, 0 post-kickoff tips, 0 closes "
            "<= 1.0, 0 |CLV| > 50 new-format rows, 0 duplicate player tips, 0 v91_shadow fixtures logged "
            "twice: RED on any; NO_DATA before the start is set")
    from src import ledger_corrections as LC
    start = _clean_record_start() if start is None else pd.to_datetime(start, utc=True)
    if start is None:
        return _c("Data truth 9: clean ledgers", GREY, "Clean Record start not set (go-live sets it)", rule)
    def since(d, col):
        if d is None or d.empty or col not in d.columns:
            return pd.DataFrame()
        t = pd.to_datetime(d[col], errors="coerce", utc=True, format="mixed")
        return d[(t >= start).to_numpy()]
    bets = _read(OUT / "bets_ledger.csv") if bets is None else bets
    players = _read(OUT / "player_ledger.csv") if players is None else players
    v91 = _read(OUT / "v91_shadow.csv") if v91 is None else v91
    b = since(bets, "generated_at")
    if "source" in b.columns:
        b = b[b["source"].astype(str) == "live"]
    issues = []
    if len(b):
        m = LC.multi_tip_rows(b)
        if len(m):
            fx = m.drop_duplicates(["match_date", "league", "home_team", "away_team"])
            issues.append(f"{len(fx)} fixture(s) with more than one live tip "
                          f"({int(fx['opposite_tip_pair'].sum())} opposite)")
        pk = LC.post_kickoff_rows(b, pool)
        if len(pk):
            issues.append(f"{len(pk)} tip(s) generated after kick-off")
        bad = LC.invalid_close_rows(b)
        if len(bad):
            issues.append(f"{len(bad)} close(s) <= 1.0")
        clv = pd.to_numeric(b.get("clv_pct"), errors="coerce")
        nf = b.get("model_type", pd.Series("", index=b.index)).astype(str).eq("new_format")
        n50 = int(((clv.abs() > 50) & nf).sum())
        if n50:
            issues.append(f"{n50} new-format row(s) with |CLV| > 50")
    pl = since(players, "signal_date")
    if len(pl):
        pd_ = LC.player_duplicate_rows(pl, pool)
        if len(pd_):
            issues.append(f"{len(pd_)} duplicate player tip(s)")
    v = since(v91, "logged_at")
    if len(v):
        vd = LC.v91_duplicate_rows(v)
        if len(vd):
            issues.append(f"{len(vd)} v91_shadow fixture(s) logged twice")
    st = RED if issues else GREEN
    return _c("Data truth 9: clean ledgers", st,
              "; ".join(issues) if issues else
              f"clean since {start:%Y-%m-%d} ({len(b)} live tip(s), {len(pl)} player row(s), {len(v)} v91 row(s))",
              rule)


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
                      ("Data truth 6: no global bars in serving", no_global_bars),
                      ("Data truth 7: no retired tiers on future tips", no_retired_tiers),
                      ("Data truth 8: bars rest on bet-time prices only", bet_time_evidence),
                      ("Data truth 9: clean ledgers", lambda: ledgers())):
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
GATED = ("Data truth 1", "Data truth 2", "Data truth 4", "Data truth 5", "Data truth 6", "Data truth 7",
         "Data truth 8")


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
