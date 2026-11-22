"""Join lineage of the ACTIVE models (owner P0 deep-repair item 3, 2026-11-22).

output/join_integrity_health.json mixes two different things under one `status`:
  * the joins AS SERVED (mode "safe", src.integrity.joins.safe_left_enrich, league / fixture keyed,
    row count asserted) - these build every training frame and every live board, and
  * the LEGACY REPLAY (mode "legacy_audit", audit_legacy_merge): a counterfactual measurement of what
    the league-free merge + positional copy-back, removed from src/data_loader on 2026-11-01, WOULD
    do on today's data. It never writes a value into any frame.
Its aggregate FAIL (fd_history_ht_fill.legacy) was propagated to every team model. This module
keeps the scopes apart and proves the lineage of each active payload separately:

  active_serving_join_status     worst status of the mode-"safe" join reports
  legacy_replay_status           worst status of the mode-"legacy_audit" reports (historical evidence)
  active_training_lineage_status per active payload: the committed evidence file
                                 (reports/lineage/training_lineage_evidence.json, written by
                                 scripts/verify_training_lineage.py from the training code at the
                                 payload's own commit and the saved training frame / training keys)
                                 must name the payload's CURRENT sha256 and record PASS. A payload
                                 without matching evidence is UNVERIFIED, never PASS.

Pure functions (code_audit, frame_audit, pit_audit) are what the evidence script runs; tests run
them on synthetic frames and sources.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports" / "lineage" / "training_lineage_evidence.json"
CHAMPIONS = ROOT / "models" / "registry" / "current_champions.json"
LINEAGE_VERSION = "training-lineage-1.0"
PASS, WARN, FAIL, UNVERIFIED = "PASS", "WARN", "FAIL", "UNVERIFIED"
_RANK = {PASS: 0, WARN: 1, UNVERIFIED: 1, "UNKNOWN": 1, FAIL: 2}

#: registry model type -> serving payload file (config.MODEL_FILE_* / HT_MODEL_FILE_*)
PAYLOAD_OF = {"standard": "model_v9_standard.pkl", "newformat": "model_v9_newformat.pkl",
              "btts": "model_v9_btts.pkl", "over15": "model_v9_over15.pkl", "over35": "model_v9_over35.pkl",
              "ht_over05": "model_ht_over05.pkl", "ht_over15": "model_ht_over15.pkl",
              "ht_over05_nf": "model_v9_ht_over05_nf.pkl", "ht_over15_nf": "model_v9_ht_over15_nf.pkl"}

#: training-code files whose joins build the team training frame
TEAM_CODE = ("src/data_loader.py", "src/integrity/joins.py", "src/feature_engineering.py",
             "src/retrain_harness/protocol.py")

# A source is CONTAMINATED when it can put a value on the wrong fixture through a join.
_CONTAMINATION = (
    ("positional_copy_back",
     re.compile(r"zip\(\s*\w+\[[^\]]+\]\.values\s*,\s*\w+\[[^\]]+\]\.values"),
     "positional zip copy-back of merged values (shifts every row after the first duplicate key)"),
    ("legacy_join_switch",
     re.compile(r"""(getenv|environ\.get|environ\[)\(?\s*["']WOWZA_SAFE_ENRICH_JOINS"""),
     "an env switch that can re-enable the league-free legacy join path"),
    ("identity_free_safe_join",
     re.compile(r"require_identity\s*=\s*False"),
     "a safe_left_enrich call that opts out of league / fixture identity"),
    ("legacy_audit_result_used_as_frame",
     re.compile(r"^\s*(df|out|hist|frame|d)\s*=\s*_?\w*\.?audit_legacy_merge\(", re.M),
     "the legacy replay's result assigned to a frame (the replay must stay audit-only)"),
)
#: guards src/integrity/joins.py must carry at the training commit
_JOIN_GUARDS = (
    ('validate="many_to_one"', "merge validated many-to-one after the right side is collapsed"),
    ("merge changed row count", "row-count change raises JoinIntegrityError"),
    ("do not map 1:1 onto left rows", "write-back checked against the left row number"),
    ("carries no league or fixture_id", "a key without league / fixture_id is refused"),
)


def worst(statuses) -> str:
    s = [x for x in statuses if x is not None]
    if not s:
        return "UNKNOWN"
    # an unknown status word ranks with UNKNOWN (never as PASS)
    w = max(s, key=lambda x: _RANK[x] if x in _RANK else _RANK["UNKNOWN"])
    return w if w in (PASS, FAIL) else (UNVERIFIED if w == UNVERIFIED else WARN)


# ── join scopes ────────────────────────────────────────────────────────────────────────────────
def join_scopes(join_doc: dict | None) -> dict:
    """Active serving joins vs the legacy replay, from a join_integrity_health document."""
    reps = (join_doc or {}).get("joins") or {}
    active = {k: r.get("status") for k, r in reps.items() if r.get("mode") == "safe"}
    legacy = {k: r.get("status") for k, r in reps.items() if r.get("mode") == "legacy_audit"}
    expanded = sorted(k for k, r in reps.items() if r.get("mode") == "safe"
                      and (r.get("output_rows") != r.get("left_rows") or r.get("duplicate_output_rows")))
    act = worst(active.values()) if active else "UNKNOWN"
    if expanded:
        act = FAIL
    return {"active_serving_join_status": act,
            "active_serving_joins": active,
            "active_joins_that_changed_row_count": expanded,
            "legacy_replay_status": worst(legacy.values()) if legacy else "UNKNOWN",
            "legacy_replay_joins": legacy,
            "legacy_replay_note": "counterfactual replay of the league-free merge + positional copy-back "
                                  "removed 2026-11-01; audit-only, writes no value into any frame"}


# ── pure audits (run by scripts/verify_training_lineage.py and the tests) ──────────────────────
def _code_only(text: str) -> str:
    """The source with comments and triple-quoted strings (docstrings) blanked, line numbers kept:
    a docstring that DESCRIBES the removed defect is not the defect."""
    import io
    import tokenize
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text
    rows = [list(x) for x in text.splitlines(keepends=True)]
    for t in toks:
        is_doc = t.type == tokenize.STRING and t.string.lstrip("rbuRBUfF")[:3] in ('"""', "'''")
        if t.type != tokenize.COMMENT and not is_doc:
            continue
        (r0, c0), (r1, c1) = t.start, t.end
        for r in range(r0, r1 + 1):
            row = rows[r - 1]
            a, b = (c0 if r == r0 else 0), (c1 if r == r1 else len(row))
            for i in range(a, min(b, len(row))):
                if row[i] != "\n":
                    row[i] = " "
    return "".join("".join(x) for x in rows)


def code_audit(sources: dict) -> dict:
    """sources: {path: text} of the training code at the payload's commit. FAIL on any contaminated
    join pattern, or when src/integrity/joins.py lacks one of its guards."""
    findings = []
    for path, text in sources.items():
        if text is None:
            findings.append({"file": path, "kind": "missing", "detail": "file not found at the commit"})
            continue
        code = _code_only(text)
        for kind, rx, why in _CONTAMINATION:
            for m in rx.finditer(code):
                line = code.count("\n", 0, m.start()) + 1
                findings.append({"file": path, "kind": kind, "line": line, "detail": why})
        if path.endswith("integrity/joins.py"):
            for needle, why in _JOIN_GUARDS:
                if needle not in text:
                    findings.append({"file": path, "kind": "missing_guard", "detail": why})
    n_safe = sum(t.count("safe_left_enrich(") for p, t in sources.items() if t and p.endswith("data_loader.py"))
    return {"status": FAIL if findings else PASS, "findings": findings,
            "safe_left_enrich_calls_in_loader": n_safe, "files": sorted(sources)}


def frame_audit(keys: pd.DataFrame, track: str, track_of, registry: set, train_end=None,
                expected_rows: int | None = None) -> dict:
    """Training rows of ONE track x market: duplicate fixture keys, a club twice on one day in one
    league, leagues outside the registry or routed to the other track, rows after the training end
    (only the untouched holdout tail may be there) and the row count against the payload."""
    k = keys.copy()
    k["date"] = pd.to_datetime(k["date"]).dt.normalize()
    key = ["league", "date", "home_team", "away_team"]
    dup = int(k.duplicated(key).sum())
    side = pd.concat([k[["league", "date", "home_team"]].rename(columns={"home_team": "team"}),
                      k[["league", "date", "away_team"]].rename(columns={"away_team": "team"})])
    twice = int(side.duplicated(["league", "date", "team"]).sum())
    lg = pd.Series(pd.unique(k["league"].astype(str)))
    unreg = sorted(x for x in lg if x not in registry)
    routed = {x: track_of(x) for x in lg}
    wrong = sorted(x for x, t in routed.items() if t != track)
    out = {"rows": int(len(k)), "duplicate_fixture_keys": dup, "club_twice_same_day": twice,
           "leagues": int(len(lg)), "leagues_not_in_registry": unreg, "leagues_on_other_track": wrong,
           "date_min": str(k["date"].min().date()) if len(k) else None,
           "date_max": str(k["date"].max().date()) if len(k) else None}
    if train_end is not None:
        out["rows_after_train_end"] = int((k["date"] > pd.Timestamp(train_end)).sum())
        out["rows_to_train_end"] = int((k["date"] <= pd.Timestamp(train_end)).sum())
    if expected_rows is not None:
        out["payload_training_rows"] = int(expected_rows)
        out["row_count_matches_payload"] = out.get("rows_to_train_end", out["rows"]) == int(expected_rows)
    bad = dup or twice or unreg or wrong or (expected_rows is not None and not out["row_count_matches_payload"])
    out["status"] = FAIL if bad else PASS
    return out


def pit_audit(frame: pd.DataFrame, n: int = 5) -> dict:
    """Point in time: home/away_scored_last5 must equal the (league, club) mean of the PREVIOUS <= n
    matches in the loaded frame and must NOT equal the window that includes the match itself.
    Rows near a gap in the frame (COVID seasons are removed AFTER the features are built) can
    differ from both; the verdict compares the two agreement rates."""
    need = {"league", "date", "home_team", "away_team", "home_goals", "away_goals", "home_scored_last5"}
    if not need <= set(frame.columns):
        return {"status": "UNKNOWN", "note": "frame lacks the columns for the check"}
    f = frame[list(need | {"away_scored_last5"})].copy().reset_index(drop=True)
    f["date"] = pd.to_datetime(f["date"])
    f["_row"] = range(len(f))
    tc = pd.concat([
        f[["_row", "league", "date", "home_team", "home_goals"]].rename(
            columns={"home_team": "team", "home_goals": "gs"}).assign(side="home"),
        f[["_row", "league", "date", "away_team", "away_goals"]].rename(
            columns={"away_team": "team", "away_goals": "gs"}).assign(side="away")])
    tc = tc.sort_values(["date", "_row"], kind="mergesort")
    g = tc.groupby(["league", "team"], sort=False)["gs"]
    tc["prev"] = g.transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    tc["incl"] = g.transform(lambda x: x.rolling(n, min_periods=1).mean())
    res = {}
    for side in ("home", "away"):
        s = tc[tc["side"] == side].set_index("_row").sort_index()
        feat = f[f"{side}_scored_last5"]
        ok = feat.notna() & s["prev"].notna()
        res[side] = {"rows": int(ok.sum()),
                     "matches_previous_window": round(float(((feat - s["prev"]).abs() < 1e-9)[ok].mean()), 4),
                     "matches_window_with_current_match": round(float(((feat - s["incl"]).abs() < 1e-9)[ok].mean()), 4)}
    good = all(v["matches_previous_window"] > v["matches_window_with_current_match"] for v in res.values())
    return {"status": PASS if good else FAIL, "feature": "home/away_scored_last5", **res}


# ── evaluation against the committed evidence ─────────────────────────────────────────────────
def _sha256(p: Path) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def active_models(root: Path | None = None) -> dict:
    """{registry model type: champion record} for every non-retired team champion."""
    root = root or ROOT
    try:
        ch = json.loads((root / "models" / "registry" / "current_champions.json").read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return {}
    return {k: v for k, v in ch.items() if str(v.get("status", "")).upper() != "RETIRED" and k in PAYLOAD_OF}


def evaluate(root: Path | None = None, evidence: dict | None = None) -> dict:
    """Per active payload: PASS only when the evidence names its current sha256 and every audit
    in it passed. Props payloads are listed from the evidence's own props block."""
    root = root or ROOT
    if evidence is None:
        try:
            evidence = json.loads((root / "reports" / "lineage" / "training_lineage_evidence.json")
                                  .read_text(encoding="utf-8"))
        except Exception:                                              # noqa: BLE001
            evidence = None
    ev = (evidence or {}).get("payloads") or {}
    models = {}
    for mt, rec in active_models(root).items():
        pfile = PAYLOAD_OF[mt]
        sha = _sha256(root / "models" / pfile)
        e = ev.get(pfile)
        if sha is None:
            st, why = FAIL, "serving payload file missing"
        elif e is None:
            st, why = UNVERIFIED, "no lineage evidence for this payload"
        elif e.get("artifact_sha256") != sha:
            st, why = UNVERIFIED, "evidence is for another file version (payload changed since it was verified)"
        else:
            st, why = e.get("status", UNVERIFIED), e.get("summary", "")
        models[mt] = {"payload": pfile, "artifact_sha256_16": (sha or "")[:16], "status": st, "why": why,
                      "registry_sha256_16": str(rec.get("artifact_sha256", ""))[:16]}
    for pfile, e in ((evidence or {}).get("props") or {}).items():
        sha = _sha256(root / "models" / pfile)
        st = (e.get("status", UNVERIFIED) if sha and e.get("artifact_sha256") == sha
              else (FAIL if sha is None else UNVERIFIED))
        models["props:" + pfile] = {"payload": pfile, "artifact_sha256_16": (sha or "")[:16], "status": st,
                                    "why": e.get("summary", "") if st != UNVERIFIED else
                                    "evidence is for another file version"}
    status = worst(m["status"] for m in models.values()) if models else UNVERIFIED
    return {"version": LINEAGE_VERSION, "status": status,
            "evidence_file": "reports/lineage/training_lineage_evidence.json",
            "evidence_generated_at": (evidence or {}).get("generated_at"), "models": models}


#: serving track key (src.integrity.model_health.TRACKS) -> registry model type
TRACK_MODEL = {"standard|OU25": "standard", "new_format|OU25": "newformat",
               "standard|BTTS": "btts", "new_format|BTTS": "btts", "standard|OU15": "over15",
               "new_format|OU15": "over15", "standard|OU35": "over35", "new_format|OU35": "over35",
               "standard|HT_O05": "ht_over05", "standard|HT_O15": "ht_over15",
               "new_format|HT_O05": "ht_over05_nf", "new_format|HT_O15": "ht_over15_nf"}
