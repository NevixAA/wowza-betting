"""REAL DATA ONLY - feature provenance registry and health (owner P0, 2026-11-15).

    python -m src.integrity.real_data_provenance            # print the health (A / B / C)
    python -m src.integrity.real_data_provenance --write    # rewrite config/feature_provenance.json
                                                            # (+ output/real_data_provenance.json)
    python -m src.integrity.real_data_provenance --check    # exit 1 on a NEW violation (C)

OWNER RULE. No football observation may be invented, silently approximated or presented as real
when the data is unavailable. A derived feature is fine when every input is valid, point-in-time and
traceable. Removing a feature from config is not enough: the SERIALIZED payload decides what a model
reads, so every list here is read from the payload files src/predict.py loads.

WHAT IT RECORDS, per served artifact (team payloads + the fantasy projection inputs):
    feature, source_type, provider, raw_columns, transformation (file:line, located at run time
    from an anchor so a line number never goes stale), point_in_time_verified (+ the test that
    proves it), training_coverage (MEASURED: the share of base-training rows with a value, from the
    payload's own train-only imputer record), serving_coverage (MEASURED: the share of the board's
    fixtures of that track with a value, output/predictions.csv), missing_policy, classification,
    active_model_eligible.

CLASSIFICATION (one per artifact x feature; the first rule that applies wins):
    TRAIN_SERVE_MISMATCH  the payload has no persisted train-only imputer (batch-median at serving),
                          or the feature is in KNOWN_MISMATCHES (unfixed)
    FABRICATED_VALUE      a 1.0 imputer filled a feature with NO training values with a constant
                          (it learned a typed number as if it were an observation)
    UNVERIFIED            the feature is not in TEAM_FEATURES (nobody traced it)
    MISSING_SOURCE        training coverage below MISSING_SOURCE_MAX (the provider barely has it)
    VALID_MODEL_ESTIMATE  training coverage below LOW_COVERAGE: most training rows are the explicit
                          train-median fill + a learned `__missing` indicator, not observations
    SOURCE_VERIFIED / DERIVED_VERIFIED  the TEAM_FEATURES class (inputs real, PIT, traceable)
TIME_LEAKAGE is a class the PIT tests (tests/test_real_data_pipeline_trace.py) would assign; none
is registered today, and a feature whose PIT test is absent is UNVERIFIED.

HEALTH - three numbers, never one combined green:
    (A) REGISTERED historical exceptions: violations in artifacts that serve nothing (standby /
        rollback / challenger) and the documented exceptions of HISTORICAL_EXCEPTIONS;
    (B) ACTIVE production violations: a non-eligible feature in a payload that scores live fixtures
        today, or an active payload the serving guard refuses;
    (C) NEW violations: a violation (artifact, feature, class) that config/feature_provenance.json
        (the committed registry) does not hold, or an active feature it does not list.
GREEN only when unsupported active = 0, fabricated active = 0 and critical mismatches = 0. Props
come from the props agent's outputs (fix/props-data-audit); until they land, props are PENDING.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config" / "feature_provenance.json"
HEALTH_OUT = ROOT / "output" / "real_data_provenance.json"
VERSION = "REAL-DATA-PROVENANCE-1"

SOURCE_VERIFIED, DERIVED_VERIFIED = "SOURCE_VERIFIED", "DERIVED_VERIFIED"
VALID_MODEL_ESTIMATE, MISSING_SOURCE = "VALID_MODEL_ESTIMATE", "MISSING_SOURCE"
FABRICATED_VALUE, TIME_LEAKAGE = "FABRICATED_VALUE", "TIME_LEAKAGE"
TRAIN_SERVE_MISMATCH, UNVERIFIED = "TRAIN_SERVE_MISMATCH", "UNVERIFIED"
CLASSES = (SOURCE_VERIFIED, DERIVED_VERIFIED, VALID_MODEL_ESTIMATE, MISSING_SOURCE, FABRICATED_VALUE,
           TIME_LEAKAGE, TRAIN_SERVE_MISMATCH, UNVERIFIED)
ELIGIBLE = (SOURCE_VERIFIED, DERIVED_VERIFIED, VALID_MODEL_ESTIMATE)
#: classes that count as "unsupported" in the per-model table (fabricated / mismatch counted apart)
UNSUPPORTED = (MISSING_SOURCE, UNVERIFIED)
MISSING_SOURCE_MAX = 0.05
LOW_COVERAGE = 0.50
#: serving coverage this far under training coverage is a collection gap (AMBER, never GREEN-silent)
SERVING_GAP_PP = 0.15

FE = "src/feature_engineering.py"
PIT_TEST = "tests/test_real_data_pipeline_trace.py"

# ── what each team feature IS ─────────────────────────────────────────────────────────────────
# anchor = (file, regex): the transformation's line, located when the registry is built.
_FD_AF = "football-data.co.uk CSV (Standard leagues) / API-Football fixtures (af_history, New Format)"
_AF_STATS = "API-Football /fixtures/statistics (api_football cache + output/collect/af_history_stats_fill.parquet)"
_HT = "football-data HTHG/HTAG / API-Football score.halftime (+ ht fills, src/data_loader)"


def _form(stat, raw, provider, cls=DERIVED_VERIFIED):
    return {"source_type": "rolling team form", "provider": provider, "raw_columns": raw,
            "anchor": (FE, rf'"{stat}":\s*\("home_'), "cls": cls,
            "transformation": f"team-centric `{stat}`, shift(1), mean of the last ROLLING_N matches "
                              "per (league, team) - _rolling / _latest_form (one function, train = serve)"}


def _spec():
    s = {}
    for side in ("home", "away"):
        s[f"{side}_scored_last5"] = _form("scored", ["home_goals", "away_goals"], _FD_AF)
        s[f"{side}_conceded_last5"] = _form("conceded", ["home_goals", "away_goals"], _FD_AF)
        s[f"{side}_over25_last5"] = _form("over25", ["home_goals", "away_goals"], _FD_AF)
        s[f"{side}_corners_pg_roll"] = _form("corners", ["home_corners", "away_corners"], _FD_AF)
        s[f"{side}_fouls_pg_roll"] = _form("fouls", ["home_fouls", "away_fouls"], _FD_AF)
        s[f"{side}_xg_last5"] = _form("xg", ["home_xg", "away_xg"], _AF_STATS)
        s[f"{side}_insidebox_last5"] = _form("insidebox", ["home_insidebox", "away_insidebox"], _AF_STATS)
        s[f"{side}_ht_scored_last5"] = _form("ht_scored", ["ht_home_goals", "ht_away_goals"], _HT)
        s[f"{side}_ht_conceded_last5"] = _form("ht_conceded", ["ht_home_goals", "ht_away_goals"], _HT)
        s[f"{side}_ht_over05_rate"] = _form("ht_over05_flag", ["ht_home_goals", "ht_away_goals"], _HT)
        s[f"{side}_ht_over15_rate"] = _form("ht_over15_flag", ["ht_home_goals", "ht_away_goals"], _HT)
        s[f"{side}_has_ht_history"] = {
            "source_type": "data-availability indicator", "provider": _HT,
            "raw_columns": ["ht_home_goals", "ht_away_goals"], "anchor": (FE, r"def _ht_history_flag"),
            "cls": DERIVED_VERIFIED,
            "transformation": "1.0 when the form window holds >= 1 real half-time score, else 0.0 (a fact "
                              "about the window, not a stand-in value)"}
        s[f"{side}_attack_str"] = {
            "source_type": "derived ratio", "provider": _FD_AF, "raw_columns": ["home_goals", "away_goals"],
            "anchor": (FE, rf'df\["{side}_attack_str"\]\s*=\s*df\["{side}_scored_last5"\]'),
            "cls": DERIVED_VERIFIED,
            "transformation": f"{side}_scored_last5 / (league average goals on strictly earlier dates / 2)"}
        s[f"{side}_defense_str"] = {
            "source_type": "derived ratio", "provider": _FD_AF, "raw_columns": ["home_goals", "away_goals"],
            "anchor": (FE, rf'df\["{side}_defense_str"\]\s*=\s*df\["{side}_conceded_last5"\]'),
            "cls": DERIVED_VERIFIED,
            "transformation": f"{side}_conceded_last5 / (league average goals on strictly earlier dates / 2)"}
        s[f"{side}_ht_attack_str"] = {
            "source_type": "derived ratio", "provider": _HT, "raw_columns": ["ht_home_goals", "ht_away_goals"],
            "anchor": (FE, rf'df\["{side}_ht_attack_str"\]\s*=\s*df\["{side}_ht_scored_last5"\]'),
            "cls": DERIVED_VERIFIED,
            "transformation": f"{side}_ht_scored_last5 / (league average goals on strictly earlier dates / 2)"}
        s[f"{side}_rest_days"] = {
            "source_type": "derived calendar", "provider": _FD_AF, "raw_columns": ["date"],
            "anchor": (FE, r"def _add_rest_days"), "cls": DERIVED_VERIFIED,
            "transformation": "days since the team's previous match in the same league (no previous -> NaN)"}
    for col, (v, g, c) in {"home_season_goals_h": ("home", "home_goals", "scored"),
                           "home_season_conceded_h": ("home", "away_goals", "conceded"),
                           "home_cs_rate_h": ("home", "away_goals", "clean sheets"),
                           "away_season_goals_a": ("away", "away_goals", "scored"),
                           "away_season_conceded_a": ("away", "home_goals", "conceded"),
                           "away_cs_rate_a": ("away", "home_goals", "clean sheets")}.items():
        s[col] = {"source_type": "season-to-date venue stat", "provider": _FD_AF,
                  "raw_columns": ["date", "league", "home_goals", "away_goals"],
                  "anchor": (FE, rf'out\.loc\[idx, "{col}"\]'), "cls": DERIVED_VERIFIED,
                  "transformation": f"{v} matches of the fixture's (league, derived season): expanding mean "
                                    f"of {c} ({g}), shift(1). Serving: the same history rows; API-Football "
                                    "/teams/statistics is display-only (af_* columns) since 2026-11-15"}
    s["combined_sot_ratio"] = {
        "source_type": "derived ratio", "provider": _FD_AF,
        "raw_columns": ["home_shots", "away_shots", "home_sot", "away_sot"],
        "anchor": (FE, r'df\["combined_sot_ratio"\]\s*='), "cls": DERIVED_VERIFIED,
        "transformation": "mean of each side's rolling SoT / rolling shots (no shot history -> NaN)"}
    s["combined_ht_goals_avg"] = {
        "source_type": "derived sum", "provider": _HT, "raw_columns": ["ht_home_goals", "ht_away_goals"],
        "anchor": (FE, r'df\["combined_ht_goals_avg"\]\s*='), "cls": DERIVED_VERIFIED,
        "transformation": "home_ht_scored_last5 + away_ht_scored_last5 (either unknown -> NaN)"}
    s["p_over25_poisson_dc"] = {
        "source_type": "derived model estimate", "provider": _FD_AF, "raw_columns": ["home_goals", "away_goals"],
        "anchor": (FE, r'df\["p_over25_poisson_dc"\]\s*=\s*\[_dc_p_over25'), "cls": DERIVED_VERIFIED,
        "transformation": "Dixon-Coles P(over 2.5) from the two rolling goals-for rates as lambdas, each "
                          "clipped to [0.3, 4.0] (no form -> NaN, never the market price)"}
    for col, who in (("home_advantage", "home team"), ("away_home_adv_factor", "away team")):
        s[col] = {"source_type": "derived rate", "provider": _FD_AF,
                  "raw_columns": ["date", "league", "home_team", "home_goals", "away_goals"],
                  "anchor": (FE, r"def home_advantage_flags"), "cls": DERIVED_VERIFIED,
                  "transformation": f"share of the {who}'s last ROLLING_N home matches in which it scored more "
                                    "than the league's average away goals on dates strictly before THAT "
                                    "match (training's rule; serving reads the same flags since 2026-11-15)"}
    return s


TEAM_FEATURES = _spec()
#: feature -> reason: a known, UNFIXED train/serve mismatch. Empty since fix/real-data-provenance
#: (season venue stats: API-Football override removed; home advantage: point-in-time reference).
KNOWN_MISMATCHES: dict = {}

#: the served team artifacts: (label, config attribute, track(s) it scores). Status from config.
TEAM_ARTIFACTS = (
    ("Standard O/U 2.5", "MODEL_FILE_STANDARD", ("standard",)),
    ("Standard O/U 2.5 (retired v9_repaired)", "MODEL_FILE_STANDARD_REPAIRED", ("standard",)),
    ("New Format O/U 2.5", "MODEL_FILE_NEWFORMAT", ("new_format",)),
    ("BTTS", "MODEL_FILE_BTTS", ("standard", "new_format")),
    ("Over 1.5", "MODEL_FILE_OVER15", ("standard", "new_format")),
    ("Over 3.5", "MODEL_FILE_OVER35", ("standard", "new_format")),
    ("HT Over 0.5", "HT_MODEL_FILE_05", ("standard",)),
    ("HT Over 1.5", "HT_MODEL_FILE_15", ("standard",)),
)

#: the FANTASY projection inputs (player_model/fantasy*.py). No fantasy model is serialized: the V2
#: models are fitted at run time on pre-deadline FPL snapshots, the V1 board is formula-based on the
#: props payloads (owned by the props agent). Classes after fix/real-data-provenance.
FANTASY_FEATURES = {
    "fpl_ep_next / price / form / ownership": ("FPL bootstrap-static", "player_model/fpl_api.py",
                                               r"def _f_or_nan", SOURCE_VERIFIED, "NaN"),
    "xg90 / xa90 / bps90 (V2)": ("FPL snapshot, shrunk to the same-snapshot position rate",
                                 "player_model/fantasy_panel.py", r"xg90", VALID_MODEL_ESTIMATE,
                                 "position with no minutes -> NaN"),
    "team gf / ga / xg / xga (V2)": ("FPL snapshot team totals, 4-game prior to the league mean",
                                     "player_model/fantasy_panel.py", r"def _team_strength|prior",
                                     VALID_MODEL_ESTIMATE,
                                     "a team with games but no xG stays NaN (no league-mean fill, 2026-11-15)"),
    "P(0) / P(1-59) / P(60) (V2 MinutesModel)": ("fitted at run time on FINAL pre-deadline gameweeks",
                                                 "player_model/fantasy_residual.py", r"class MinutesModel",
                                                 VALID_MODEL_ESTIMATE, "NaN-aware boosting"),
    "residual training rows (V2)": ("FINAL gameweeks only, as the walk-forward validates (2026-11-15)",
                                    "player_model/fantasy_v2.py", r'prep\["status"\]', DERIVED_VERIFIED,
                                    "-"),
    "xpts_3gw / xpts_5gw (V2)": ("FPL fixtures + ep_next", "player_model/fantasy_v2.py",
                                 r"def _multi_gw|xpts_5gw", VALID_MODEL_ESTIMATE,
                                 "no fixture list for the club, or negative ep_next -> NaN (2026-11-15)"),
    "p_goal / p_assist / p_sot2 (V1)": ("props payloads models/model_player_{goals,assists,sot2}.pkl",
                                        "player_model/fantasy.py", r"predict_proba", UNVERIFIED,
                                        "props provenance PENDING (fix/props-data-audit)"),
    "def_actions_pg (V1 bonus input)": ("API-Football tackles + interceptions + duels won",
                                        "player_model/fantasy.py", r"min_count=", DERIVED_VERIFIED,
                                        "any missing component -> NaN (2026-11-15; was read as 0)"),
    "injured / doubtful (V1)": ("FPL bootstrap status via output/fpl_af_crosswalk.csv",
                                "player_model/fantasy.py", r'pl\["injured"\]', SOURCE_VERIFIED,
                                "unmatched player -> unknown (NA), no match at all -> no board (2026-11-15)"),
    "fixture_adj_pts (V1)": ("FPL FDR + home flag", "player_model/fantasy.py", r"def _fdr_mult|_no_fdr",
                             VALID_MODEL_ESTIMATE, "unknown FDR -> NaN (2026-11-15; was x1.0)"),
}

#: registered, documented historical exceptions (A). Each is NOT reaching a served model today.
HISTORICAL_EXCEPTIONS = (
    {"id": "standard-repaired-xg-zero", "artifact": "Standard O/U 2.5 (retired v9_repaired)",
     "what": "the 2026-10-28 repaired Standard payload's 1.0 imputer filled xG and inside-box shots "
             "(0% training coverage) with 0.0 on every training row",
     "status": "RETIRED by the owner 2026-11-15 (\"fix what you need\"): v9_repaired is no longer a "
               "serving recipe, owner_install refuses the type, the registry pointer says RETIRED; the "
               "file stays for audit and the serving guard (src.model.serving_refusal) refuses it too"},
    {"id": "legacy-batch-median", "artifact": "any payload without a persisted imputer",
     "what": "legacy src.model._prep fills from the scoring batch's medians (all-missing column -> 0.0)",
     "status": "no served payload uses it; predict_proba(serving=True) refuses such a payload"},
    {"id": "nf-hist-af-override", "artifact": "every team payload (served 2026-08-16 .. 2026-11-15)",
     "what": "API-Football /teams/statistics overwrote the six season-venue model features at serving; "
             "home advantage compared against today's league average",
     "status": "fixed at serving (fix/real-data-provenance); predictions logged before the fix keep "
               "their values (historical evidence, unchanged)"},
)


# ── helpers ───────────────────────────────────────────────────────────────────────────────────
_LINE_CACHE: dict = {}


def locate(path: str, pattern: str) -> str:
    """'file:line' of the first line matching `pattern` (regex), or 'file:?' when it is gone."""
    key = (path, pattern)
    if key not in _LINE_CACHE:
        ln = "?"
        p = ROOT / path
        if p.exists():
            rx = re.compile(pattern)
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    ln = str(i)
                    break
        _LINE_CACHE[key] = f"{path}:{ln}"
    return _LINE_CACHE[key]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def artifact_status(attr: str) -> str:
    """active / standby for a team payload attribute, from the serving configuration."""
    import config
    recipe = str(getattr(config, "STANDARD_OU25_SERVING_RECIPE", "legacy_v9"))
    if attr == "MODEL_FILE_STANDARD":
        return "active" if recipe == "legacy_v9" else "standby"
    if attr == "MODEL_FILE_STANDARD_REPAIRED":
        # RETIRED by the owner 2026-11-15 ("fix what you need"): no serving recipe selects it
        return "retired (kept for audit)"
    return "active"


def _parts(payload: dict, tracks: tuple) -> dict:
    """{track: flat payload} - a track-split payload scores new_format rows with its sub-payload."""
    out = {}
    for t in tracks:
        if t != "standard" and payload.get("tracks") is not None:
            sub = (payload.get("tracks") or {}).get(t)
            if sub is not None:
                out[t] = {**payload, **sub}
        else:
            out[t] = payload
    return out


def serving_coverage(board: pd.DataFrame | None, track: str, feature: str):
    """Share of the board's fixtures of `track` carrying `feature`; None when there is no board row."""
    if board is None or board.empty or "model_type" not in board.columns:
        return None
    sub = board[board["model_type"].astype(str) == track]
    if sub.empty:
        return None
    if feature not in sub.columns:
        return None             # not written to the board (computed at scoring): not measurable here
    return round(float(pd.to_numeric(sub[feature], errors="coerce").notna().mean()), 4)


def classify(payload: dict, feature: str, train_cov) -> tuple[str, str]:
    """(class, reason) of one feature of one flat payload - the rules in the module docstring."""
    imp = payload.get("imputer") or {}
    st = (imp.get("features") or {}).get(feature)
    base = feature[:-len("__missing")] if feature.endswith("__missing") else feature
    if not imp:
        return TRAIN_SERVE_MISMATCH, "no persisted imputer: batch-median imputation at serving"
    if base in KNOWN_MISMATCHES:
        return TRAIN_SERVE_MISMATCH, KNOWN_MISMATCHES[base]
    if (imp.get("version") == "v9-train-only-imputer-1.0" and isinstance(st, dict)
            and st.get("median") is None and not feature.endswith("__missing")):
        return FABRICATED_VALUE, (f"no training values: filled with the constant {imp.get('fill_all_missing')} "
                                  "on every training row and at serving")
    if isinstance(st, dict) and st.get("median") is None and not feature.endswith("__missing"):
        return FABRICATED_VALUE, "no training values"
    if base not in TEAM_FEATURES:
        return UNVERIFIED, "not traced in TEAM_FEATURES"
    if feature.endswith("__missing"):
        return DERIVED_VERIFIED, f"0/1 indicator: {base} missing on this row (built at scoring time)"
    if train_cov is not None and train_cov < MISSING_SOURCE_MAX:
        return MISSING_SOURCE, f"training coverage {train_cov:.1%} < {MISSING_SOURCE_MAX:.0%}"
    if train_cov is not None and train_cov < LOW_COVERAGE:
        return VALID_MODEL_ESTIMATE, (f"training coverage {train_cov:.1%}: the other rows are the explicit "
                                      "train-median fill + a learned missing indicator")
    return TEAM_FEATURES[base]["cls"], ""


def _missing_policy(payload: dict, feature: str) -> str:
    imp = payload.get("imputer") or {}
    if not imp:
        return "batch median of the scoring frame (legacy _prep; refused at serving)"
    st = (imp.get("features") or {}).get(feature) or {}
    if feature.endswith("__missing"):
        return "computed from the base column at scoring time"
    if st.get("median") is None:
        return f"constant {imp.get('fill_all_missing')} (1.0 imputer)" if imp.get(
            "version") == "v9-train-only-imputer-1.0" else "refused (1.1 imputer fails closed)"
    req = " REQUIRED: a fixture missing it gets no probability at serving" if st.get("required") else ""
    clip = ", clipped to the training range" if imp.get("clip_to_training_range") else ""
    return f"train-only median {round(float(st['median']), 6)} + `__missing` indicator{clip}.{req}"


# ── registry ──────────────────────────────────────────────────────────────────────────────────
def team_registry(board: pd.DataFrame | None = None) -> list[dict]:
    """One row per (served team artifact, track, payload feature)."""
    import config
    from src.model import load_models, serving_refusal
    if board is None:
        board = _board()
    rows = []
    for label, attr, tracks in TEAM_ARTIFACTS:
        path = Path(getattr(config, attr))
        status = artifact_status(attr)
        try:
            pl = load_models(model_file=path)
        except Exception as e:                                         # noqa: BLE001
            rows.append({"artifact": label, "track": None, "feature": None, "status": status,
                         "classification": UNVERIFIED, "reason": f"unreadable: {type(e).__name__}",
                         "active_model_eligible": False})
            continue
        for track, part in _parts(pl, tracks).items():
            refusal = serving_refusal(part)
            feats = (part.get("imputer") or {}).get("features") or {}
            for f in part.get("feature_cols") or []:
                st = feats.get(f) or {}
                mr = st.get("missing_rate") if st else None
                tc = None if mr is None else round(1.0 - float(mr), 4)
                if f.endswith("__missing"):
                    tc = 1.0
                cls, why = classify(part, f, tc)
                base = f[:-len("__missing")] if f.endswith("__missing") else f
                spec = TEAM_FEATURES.get(base, {})
                sc = serving_coverage(board, track, base)
                if f.endswith("__missing") and sc is not None:
                    sc = 1.0
                gap = (tc is not None and sc is not None and not f.endswith("__missing")
                       and tc - sc > SERVING_GAP_PP)
                rows.append({
                    "artifact": label, "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT)
                    else str(path), "status": status, "track": track, "feature": f,
                    "source_type": spec.get("source_type", "missing-data indicator" if f.endswith("__missing")
                                            else "UNTRACED"),
                    "provider": spec.get("provider", ""), "raw_columns": spec.get("raw_columns", []),
                    "transformation": spec.get("transformation", ""),
                    "transformation_at": locate(*spec["anchor"]) if spec.get("anchor") else
                    locate("src/imputer.py", r"if name\.endswith\(MISSING_SUFFIX\)"),
                    "point_in_time_verified": bool(spec), "point_in_time_test": PIT_TEST if spec else None,
                    "training_coverage": tc, "serving_coverage": sc,
                    "serving_coverage_gap": bool(gap),
                    "missing_policy": _missing_policy(part, f),
                    "classification": cls, "reason": why,
                    "active_model_eligible": cls in ELIGIBLE and not refusal,
                    "serving_refusal": refusal,
                })
    return rows


def fantasy_registry() -> list[dict]:
    rows = []
    for feat, (src, path, anchor, cls, missing) in FANTASY_FEATURES.items():
        rows.append({"artifact": "Fantasy", "path": None, "status": "active (fitted at run time)",
                     "track": "fpl", "feature": feat, "source_type": "fantasy input", "provider": src,
                     "raw_columns": [], "transformation": src, "transformation_at": locate(path, anchor),
                     "point_in_time_verified": cls != UNVERIFIED,
                     "point_in_time_test": "tests/test_fantasy_v2.py; tests/test_real_data_fantasy.py",
                     "training_coverage": None, "serving_coverage": None, "serving_coverage_gap": False,
                     "missing_policy": missing, "classification": cls,
                     "reason": "props provenance pending" if cls == UNVERIFIED else "",
                     "active_model_eligible": cls in ELIGIBLE, "serving_refusal": None,
                     "depends_on_props": cls == UNVERIFIED})
    return rows


def _board() -> pd.DataFrame | None:
    p = ROOT / "output" / "predictions.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:                                                  # noqa: BLE001
        return None


def props_status() -> dict:
    """The props agent's verdict, when its outputs have landed (fix/props-data-audit); else PENDING."""
    for name in ("props_real_data_provenance.json", "props_feature_provenance.json"):
        p = ROOT / "output" / name
        if p.exists():
            try:
                return {"status": "REPORTED", "source": f"output/{name}",
                        **json.loads(p.read_text(encoding="utf-8"))}
            except Exception as e:                                     # noqa: BLE001
                return {"status": "UNREADABLE", "source": f"output/{name}", "detail": str(e)[:120]}
    return {"status": "PENDING", "source": None,
            "detail": "props provenance not yet delivered (fix/props-data-audit): props are NOT verified"}


def _violation(r: dict) -> bool:
    """A feature-level violation (an artifact the serving guard refuses is counted once, apart)."""
    return r["classification"] not in ELIGIBLE


def _key(r: dict) -> str:
    return f"{r['artifact']}|{r['track']}|{r['feature']}|{r['classification']}"


def per_model(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        k = f"{r['artifact']} [{r['track']}]"
        m = out.setdefault(k, {"status": r["status"], "features": 0, "real_data_verified": 0,
                               "eligible_learned_or_imputed": 0, "unsupported_active": 0,
                               "fabricated_active": 0, "pit_violations": 0, "train_serve_mismatches": 0,
                               "serving_refusal": r.get("serving_refusal"), "serving_coverage_gaps": 0})
        m["features"] += 1
        c = r["classification"]
        if c in (SOURCE_VERIFIED, DERIVED_VERIFIED):
            m["real_data_verified"] += 1
        elif c == VALID_MODEL_ESTIMATE:
            m["eligible_learned_or_imputed"] += 1
        elif c in UNSUPPORTED:
            m["unsupported_active"] += 1
        elif c == FABRICATED_VALUE:
            m["fabricated_active"] += 1
        elif c == TIME_LEAKAGE:
            m["pit_violations"] += 1
        elif c == TRAIN_SERVE_MISMATCH:
            m["train_serve_mismatches"] += 1
        m["serving_coverage_gaps"] += int(bool(r.get("serving_coverage_gap")))
    return out


def health(rows: list[dict] | None = None, committed: dict | None = None) -> dict:
    """(A) registered historical exceptions, (B) active production violations, (C) new violations."""
    rows = rows if rows is not None else team_registry() + fantasy_registry()
    committed = committed if committed is not None else (
        json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {"rows": []})
    known = {_key(r) for r in committed.get("rows", [])}
    known_feats = {(r["artifact"], r["track"], r["feature"]) for r in committed.get("rows", [])}
    active = [r for r in rows if str(r["status"]).startswith("active")]
    inactive = [r for r in rows if not str(r["status"]).startswith("active")]
    # props-dependent fantasy inputs are PENDING (the props agent's verdict), not a team violation
    b = [r for r in active if _violation(r) and not r.get("depends_on_props")]
    a = [r for r in inactive if _violation(r)]
    c = [r for r in rows if _violation(r) and not r.get("depends_on_props") and _key(r) not in known]
    c += [r for r in active if (r["artifact"], r["track"], r["feature"]) not in known_feats
          and r not in c and committed.get("rows")]
    models = per_model(active)
    unsupported = sum(m["unsupported_active"] for m in models.values())
    fabricated = sum(m["fabricated_active"] for m in models.values())
    mismatch = sum(m["train_serve_mismatches"] for m in models.values())
    refused = sorted({f"{r['artifact']} [{r['track']}]" for r in active if r.get("serving_refusal")})
    gaps = sorted({f"{r['artifact']} [{r['track']}] {r['feature']}" for r in active if r.get("serving_coverage_gap")})
    green = not b and not c and not refused and unsupported == 0 and fabricated == 0 and mismatch == 0
    return {"artifact": "real_data_provenance", "version": VERSION,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "A_registered_historical_exceptions": {
                "count": len(HISTORICAL_EXCEPTIONS) + len(a), "documented": list(HISTORICAL_EXCEPTIONS),
                "inactive_artifact_violations": [_key(r) for r in a]},
            "B_active_production_violations": {"count": len(b) + len(refused), "rows": [_key(r) for r in b],
                                               "refused_active_payloads": refused},
            "C_new_violations": {"count": len(c), "rows": [_key(r) for r in c]},
            "per_model": models, "serving_coverage_gaps": gaps, "props": props_status(),
            "team_fantasy_status": "GREEN" if green else "RED",
            "note": "GREEN only when unsupported active = 0, fabricated active = 0, critical "
                    "mismatches = 0, no active payload refused and no new violation. Props are "
                    "reported apart and are not verified until the props agent's outputs land."}


def checks(h: dict | None = None) -> list[dict]:
    """The three data-truth lines (never one combined green)."""
    h = h or health()
    a, b, c = (h["A_registered_historical_exceptions"], h["B_active_production_violations"],
               h["C_new_violations"])
    props = h["props"]["status"]
    gaps = h["serving_coverage_gaps"]
    rule = "REAL DATA ONLY (owner 2026-11-15): src/integrity/real_data_provenance"
    out = [
        {"check": "Data truth 8a: active production violations",
         "status": "RED" if b["count"] else ("AMBER" if (gaps or props != "REPORTED") else "GREEN"),
         "value": f"{b['count']} active violation(s); props {props}; {len(gaps)} serving coverage gap(s)",
         "rule": rule + " - fabricated, unsupported, mismatched or refused inputs in a payload that scores live "
                        "fixtures (props: not yet verified until the props agent reports)",
         "detail": "; ".join((b["rows"] + b["refused_active_payloads"])[:8] + gaps[:4])},
        {"check": "Data truth 8b: new provenance violations",
         "status": "RED" if c["count"] else "GREEN", "value": f"{c['count']} new violation(s)",
         "rule": rule + " - a violation or an active feature config/feature_provenance.json does not hold",
         "detail": "; ".join(c["rows"][:8])},
        {"check": "Data truth 8c: registered historical exceptions",
         "status": "AMBER" if a["count"] else "GREEN", "value": f"{a['count']} registered exception(s)",
         "rule": rule + " - documented, not reaching a served model",
         "detail": "; ".join([e["id"] for e in a["documented"]] + a["inactive_artifact_violations"][:6])},
    ]
    return out


def write(rows: list[dict] | None = None) -> dict:
    """Rewrite config/feature_provenance.json: each feature's description ONCE (`features`), then one
    compact row per (artifact, track, feature) with the measured coverage and the classification."""
    rows = rows if rows is not None else team_registry() + fantasy_registry()
    desc = ("source_type", "provider", "raw_columns", "transformation", "transformation_at",
            "point_in_time_verified", "point_in_time_test")
    keep = ("artifact", "path", "status", "track", "feature", "transformation_at", "training_coverage",
            "serving_coverage", "missing_policy", "classification", "reason", "active_model_eligible")
    feats = {}
    for r in rows:
        base = r["feature"][:-len("__missing")] if str(r["feature"]).endswith("__missing") else r["feature"]
        feats.setdefault(base if not str(r["feature"]).endswith("__missing") else r["feature"],
                         {k: r.get(k) for k in desc})
    doc = {"artifact": "feature_provenance", "version": VERSION,
           "generated_by": "python -m src.integrity.real_data_provenance --write",
           "classes": list(CLASSES), "eligible": list(ELIGIBLE),
           "serving_board": "output/predictions.csv (serving_coverage, measured when written)",
           "features": feats, "rows": [{k: r.get(k) for k in keep} for r in rows]}
    REGISTRY.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    h = health(rows, doc)
    HEALTH_OUT.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_OUT.write_text(json.dumps(h, indent=1, default=str) + "\n", encoding="utf-8")
    return h


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--write" in argv:
        h = write()
    else:
        h = health()
    for c in checks(h):
        print(f"{c['status']:7} {c['check']}: {c['value']}")
    for k, m in h["per_model"].items():
        print(f"  {k}: verified {m['real_data_verified']}, estimates {m['eligible_learned_or_imputed']}, "
              f"unsupported {m['unsupported_active']}, fabricated {m['fabricated_active']}, "
              f"PIT {m['pit_violations']}, mismatch {m['train_serve_mismatches']}")
    if "--check" in argv:
        return 1 if h["C_new_violations"]["count"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
