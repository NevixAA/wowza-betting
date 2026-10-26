"""NON-REGRESSION LOCK for the research system (owner rule 2026-10-25).

    python -m src.research_lock --check                 # exit 1 + exact diff if production behaviour moved
    python -m src.research_lock --approve --by "<name>" --note "<why>"   # deliberate re-approval

    from src.research_lock import check
    check()["protected_tracks_modified"]                # bool, read by src/system_health.py

WHAT IT GUARDS. Research (scripts/build_league_regimes.py, the weekly panel that follows) must never
change a threshold, a stake, a market's eligibility, an ACTIVE/PAPER state or a selection rule. This
module fingerprints the PRODUCTION-FACING behaviour into config/research_lock_snapshot.json and the
check fails when the live repo no longer matches it. A legitimate change (an owner-approved policy PR)
is accepted by re-running --approve, which records approved_by / approved_at; nothing is ever accepted
silently.

PER-LEAGUE THRESHOLD FILES (owner rule 2026-10-26: the monthly backtest PROPOSES, the owner approves).
models/best_params_standard.json and models/best_params_side_markets.json are the files production
reads its auto-optimised per-league gates from. They may change ONLY through src/threshold_apply.py,
which applies owner-approved proposal entries, records the files' content digests in
config/threshold_applied.json and re-approves this snapshot with the approver's name. check() compares
the live files against that ledger, so ANY other change to them (a workflow, a hand edit, a re-run of
the optimizer) is reported under `threshold_files.*` and sets protected_tracks_modified - for every
league, not only PROTECTED ones. `--approve` cannot bless such a change: it does not touch the ledger.

FINGERPRINT CONTENTS
  global.bankroll_policy   config/bankroll_policy.json: policy_version, portfolio caps, tier and
                           per-track tier multipliers, convergence, paper_leagues, drawdown throttle,
                           league_rating_sizing, and EVERY track's state/base_fraction/min/max odds/
                           evidence_score (so standard|OU25 PAPER, *|OVER35 BLOCKED, player_props
                           BLOCKED_FROM_REAL are all locked).
  global.deployed_env      every env var config.py reads that .github/workflows/predict.yml sets (the
                           deployed thresholds are NOT the config.py defaults — CLAUDE.md trap 1).
  global.thresholds        config tables + global thresholds RESOLVED UNDER THAT ENV (subprocess, so the
                           local .env and this process's environment cannot leak in).
  global.eligibility       ENABLED / PAPER / STANDARD / NEW_FORMAT league sets, PROVIDER_UNSUPPORTED,
                           auto-onboarded leagues, MIN_OVER/UNDER_ODDS, NF_MAX_OU_ODDS.
  global.execution         Cloudbet default mode line (src/cloudbet/client.py) and any workflow that sets
                           CLOUDBET_MODE (must be none: Cloudbet stays PAPER_ONLY).
  protected_tracks[key]    for every PROTECTED entry of config/league_market_regimes.json: production
                           state, track state, base fraction, effective tier multipliers, track odds caps
                           and the gates src/gate_resolver resolves for that league x market under the
                           deployed env (with their sources).

protected_tracks_modified = any difference in `global` or in a track protected in the APPROVED snapshot.
A track newly classified PROTECTED by research is reported in `unapproved_new_protected_tracks` (it is
not a production change) and is folded in by the next --approve.

READ-ONLY w.r.t. production: it reads policy/config/workflow files and imports config + gate_resolver
(both read-only by contract). The only file it ever writes is config/research_lock_snapshot.json, and
only on --approve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.bankroll_policy import track_key

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "config" / "research_lock_snapshot.json"
REGISTRY_PATH = ROOT / "config" / "league_market_regimes.json"
POLICY_PATH = ROOT / "config" / "bankroll_policy.json"
PREDICT_YML = ROOT / ".github" / "workflows" / "predict.yml"
CONFIG_PY = ROOT / "config.py"
CLOUDBET_CLIENT = ROOT / "src" / "cloudbet" / "client.py"
WORKFLOWS = ROOT / ".github" / "workflows"
#: The per-league threshold files production reads (src/betting.py, pipeline._generate_side_bets,
#: src/gate_resolver.py). Written only by src/threshold_apply.py.
THRESHOLD_FILES = ("models/best_params_standard.json", "models/best_params_side_markets.json")
THRESHOLD_ROOT = ROOT
THRESHOLD_LEDGER_PATH = ROOT / "config" / "threshold_applied.json"
#: Registry market -> gate_resolver market key
GATE_MARKET = {"OU25": "ou25", "BTTS": "btts", "OVER15": "over15", "OVER35": "over35"}

_RESOLVE_CODE = r"""
import json, sys
import dotenv
dotenv.load_dotenv = lambda *a, **k: False          # the deployed env only - never a local .env
import config
from src import gate_resolver as G
cells = json.loads(sys.argv[1])
g = {}
for lg, mk, side in cells:
    x = G.resolve_effective_tier_gates(lg, mk, None, side).as_dict()
    g[f"{lg}|{mk}|{side}"] = {k: x[k] for k in ("valuable", "marksman", "sniper", "valuable_source",
        "marksman_source", "sniper_source", "approved_optimizer_applied", "league_cap_applied",
        "edge_ceiling", "edge_ceiling_active")}
out = {"gates": g, "thresholds": {
    "VALUABLE_THRESHOLD": config.VALUABLE_THRESHOLD, "MARKSMAN_THRESHOLD": config.MARKSMAN_THRESHOLD,
    "SNIPER_THRESHOLD": config.SNIPER_THRESHOLD, "SNIPER_THRESHOLD_OVER": config.SNIPER_THRESHOLD_OVER,
    "SNIPER_THRESHOLD_UNDER": config.SNIPER_THRESHOLD_UNDER, "EDGE_CEILING": config.EDGE_CEILING,
    "LEAGUE_SNIPER_THRESHOLDS": config.LEAGUE_SNIPER_THRESHOLDS,
    "LEAGUE_SNIPER_CALIBRATED": config._LEAGUE_SNIPER_CALIBRATED,
    "LEAGUE_MARKSMAN_THRESHOLDS": config.LEAGUE_MARKSMAN_THRESHOLDS,
    "LEAGUE_EDGE_CEILING": config.LEAGUE_EDGE_CEILING,
    "SIDE_GLOBAL": [G.SIDE_GLOBAL_SNIPER, G.SIDE_GLOBAL_MARKSMAN, G.SIDE_VALUABLE_HARDCODED]},
  "eligibility": {
    "ENABLED_LEAGUES": sorted(config.ENABLED_LEAGUES), "PAPER_LEAGUES": sorted(config.PAPER_LEAGUES),
    "STANDARD_FORMAT_LEAGUES": sorted(config.STANDARD_FORMAT_LEAGUES),
    "NEW_FORMAT_LEAGUES": sorted(config.NEW_FORMAT_LEAGUES),
    "PROVIDER_UNSUPPORTED": sorted(config.PROVIDER_UNSUPPORTED), "AUTO_LEAGUES": sorted(config.AUTO_LEAGUES),
    "MIN_OVER_ODDS": config.MIN_OVER_ODDS, "MIN_UNDER_ODDS": config.MIN_UNDER_ODDS,
    "NF_MAX_OU_ODDS": config.NF_MAX_OU_ODDS}}
print("@@JSON@@" + json.dumps(out, sort_keys=True, default=str))
"""


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def config_env_keys(config_src: str | None = None) -> list[str]:
    """Every environment variable config.py reads."""
    src = config_src if config_src is not None else CONFIG_PY.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'os\.getenv\(\s*"([A-Z0-9_]+)"', src)))


def deployed_env(yml_text: str | None = None, keys: list[str] | None = None) -> dict:
    """Non-secret values predict.yml sets for config-read env vars. A key set to different values in
    different steps is recorded as the sorted list of values (and is itself a red flag)."""
    text = yml_text if yml_text is not None else PREDICT_YML.read_text(encoding="utf-8")
    keys = keys if keys is not None else config_env_keys()
    found: dict[str, set] = {}
    for m in re.finditer(r'^\s+([A-Z0-9_]+):\s*(.*?)\s*$', text, flags=re.M):
        k, v = m.group(1), m.group(2)
        if k not in keys or "${{" in v:
            continue
        v = v.split(" #")[0].strip().strip('"').strip("'")
        found.setdefault(k, set()).add(v)
    return {k: (sorted(v)[0] if len(v) == 1 else sorted(v)) for k, v in sorted(found.items())}


def _resolve(env: dict, cells: list) -> dict:
    keys = config_env_keys()
    e = {k: v for k, v in os.environ.items() if k not in keys}
    e.update({k: (v if isinstance(v, str) else v[0]) for k, v in env.items()})
    e["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-c", _RESOLVE_CODE, json.dumps(cells)], cwd=str(ROOT), env=e,
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    if r.returncode != 0 or "@@JSON@@" not in r.stdout:
        raise RuntimeError(f"gate resolution failed: {r.stderr[-800:]}")
    return json.loads(r.stdout.split("@@JSON@@", 1)[1])


def _policy_block(policy: dict) -> dict:
    keep = ("policy_version", "portfolio", "tier_multipliers", "track_tier_multipliers", "convergence",
            "paper_leagues", "drawdown_throttle")
    out = {k: policy.get(k) for k in keep}
    lrs = policy.get("league_rating_sizing") or {}
    out["league_rating_sizing"] = {"enabled": lrs.get("enabled"), "min_scale": lrs.get("min_scale")}
    out["tracks"] = {t: {k: v for k, v in sorted(d.items())} for t, d in sorted((policy.get("tracks") or {}).items())}
    return out


def _execution_block() -> dict:
    src = CLOUDBET_CLIENT.read_text(encoding="utf-8") if CLOUDBET_CLIENT.exists() else ""
    m = re.search(r'^MODE\s*=.*$', src, flags=re.M)
    setters = sorted(p.name for p in WORKFLOWS.glob("*.yml")
                     if "CLOUDBET_MODE" in p.read_text(encoding="utf-8", errors="replace"))
    return {"cloudbet_mode_line": m.group(0).strip() if m else "MISSING", "workflows_setting_CLOUDBET_MODE": setters}


def protected_keys(registry: dict) -> list[str]:
    return sorted(e["key"] for e in registry.get("entries", []) if e.get("research_regime") == "PROTECTED")


def fingerprint(track_keys: list[str] | None = None, registry: dict | None = None,
                policy: dict | None = None, env: dict | None = None) -> dict:
    """Production-facing fingerprint. `track_keys` defaults to the registry's PROTECTED entries."""
    registry = registry if registry is not None else _read_json(REGISTRY_PATH)
    policy = policy if policy is not None else _read_json(POLICY_PATH)
    env = env if env is not None else deployed_env()
    track_keys = sorted(track_keys if track_keys is not None else protected_keys(registry))
    cells = []
    for k in track_keys:
        lg, market, side, _mt = k.split("|")
        cells.append([lg, GATE_MARKET[market], side if market == "OU25" else ""])
    res = _resolve(env, cells)
    paper = set(policy.get("paper_leagues", [])) | set(res["eligibility"]["PAPER_LEAGUES"])
    tracks = {}
    for k, (lg, gm, side) in zip(track_keys, cells):
        _lg, market, _side, mt = k.split("|")
        tkey = track_key(mt, market, _side)              # O/U 2.5 per side since 2026-10-26
        t = (policy.get("tracks") or {}).get(tkey) or {}
        tt = (policy.get("track_tier_multipliers") or {}).get(tkey)
        tracks[k] = {"track": tkey, "track_state": t.get("state", "UNSUPPORTED"),
                     "production_state": "PAPER_LEAGUE" if lg in paper else t.get("state", "UNSUPPORTED"),
                     "base_fraction": t.get("base_fraction"), "min_odds": t.get("min_odds"),
                     "max_odds": t.get("max_odds"), "evidence_score": t.get("evidence_score"),
                     "tier_multipliers": tt if tt is not None else policy.get("tier_multipliers"),
                     "gates": res["gates"][f"{lg}|{gm}|{side}"]}
    return {"global": {"bankroll_policy": _policy_block(policy), "deployed_env": env,
                       "thresholds": res["thresholds"], "eligibility": res["eligibility"],
                       "execution": _execution_block()},
            "protected_tracks": tracks}


def digest(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def diff(old, new, path: str = "") -> list[str]:
    """Every leaf that differs, as 'path: old -> new'."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in old:
                out.append(f"{p}: <absent> -> {json.dumps(new[k], sort_keys=True, default=str)}")
            elif k not in new:
                out.append(f"{p}: {json.dumps(old[k], sort_keys=True, default=str)} -> <absent>")
            else:
                out += diff(old[k], new[k], p)
        return out
    if old != new:
        return [f"{path}: {json.dumps(old, sort_keys=True, default=str)} -> {json.dumps(new, sort_keys=True, default=str)}"]
    return []


def canonical_digest(path: Path) -> str:
    """Content digest of a JSON file that ignores formatting and line endings (CRLF checkouts on
    Windows vs LF on CI must not read as a change). ABSENT / UNREADABLE are digests too."""
    p = Path(path)
    if not p.exists():
        return "ABSENT"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return "UNREADABLE:" + hashlib.sha256(p.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def threshold_file_digests(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else THRESHOLD_ROOT
    return {rel: canonical_digest(root / rel) for rel in THRESHOLD_FILES}


def threshold_files_changes(ledger: dict | None = None, root: Path | None = None) -> list[str]:
    """Differences between the live per-league threshold files and the digests the last APPROVED
    apply recorded. A missing ledger fails closed."""
    led = ledger if ledger is not None else _read_json(THRESHOLD_LEDGER_PATH)
    if not led.get("files"):
        return ["threshold_files: no applied-threshold ledger (config/threshold_applied.json) - "
                "per-league thresholds cannot be verified"]
    out = []
    for rel, cur in threshold_file_digests(root).items():
        exp = led["files"].get(rel)
        if cur != exp:
            out.append(f"threshold_files.{rel}: {str(exp)[:12]} -> {str(cur)[:12]} (changed without an "
                       f"owner-approved `python -m src.threshold_apply`)")
    return out


def check(snapshot: dict | None = None, registry: dict | None = None, policy: dict | None = None,
          env: dict | None = None, ledger: dict | None = None) -> dict:
    snap = snapshot if snapshot is not None else _read_json(SNAPSHOT_PATH)
    registry = registry if registry is not None else _read_json(REGISTRY_PATH)
    if not snap.get("fingerprint"):
        return {"protected_tracks_modified": True, "changes": ["no approved snapshot (config/research_lock_snapshot.json)"],
                "unapproved_new_protected_tracks": [], "approved_by": None, "approved_at": None}
    approved_keys = sorted(snap["fingerprint"].get("protected_tracks", {}))
    now = fingerprint(approved_keys, registry=registry, policy=policy, env=env)
    tf_changes = threshold_files_changes(ledger)
    changes = diff(snap["fingerprint"], now) + tf_changes
    new_keys = sorted(set(protected_keys(registry)) - set(approved_keys))
    return {"protected_tracks_modified": bool(changes), "changes": changes,
            "threshold_files_unapproved": tf_changes,
            "unapproved_new_protected_tracks": new_keys, "n_protected_tracks": len(approved_keys),
            "approved_by": snap.get("approved_by"), "approved_at": snap.get("approved_at"),
            "approved_digest": snap.get("fingerprint_sha256"), "current_digest": digest(now)}


def approve(by: str, note: str = "", path: Path = SNAPSHOT_PATH) -> dict:
    if not by.strip():
        raise SystemExit("--by is required: an approval must name who approved it")
    fp = fingerprint()
    snap = {"about": "Approved production-facing fingerprint for src/research_lock.py. Re-approve ONLY for a "
                     "deliberate, owner-approved production change. Do not hand-edit.",
            "approved_by": by.strip(), "approved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "approval_note": note, "fingerprint_sha256": digest(fp), "fingerprint": fp}
    path.write_text(json.dumps(snap, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    return snap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--approve", action="store_true")
    ap.add_argument("--by", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.approve:
        s = approve(a.by, a.note)
        print(f"approved by {s['approved_by']} at {s['approved_at']}: {s['fingerprint_sha256'][:12]}, "
              f"{len(s['fingerprint']['protected_tracks'])} protected tracks")
        return 0
    r = check()
    print(f"protected_tracks_modified={r['protected_tracks_modified']} (approved by {r['approved_by']} "
          f"at {r['approved_at']})")
    for c in r["changes"]:
        print("  CHANGED", c)
    for k in r["unapproved_new_protected_tracks"]:
        print("  NEW PROTECTED (not yet approved)", k)
    return 1 if r["protected_tracks_modified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
