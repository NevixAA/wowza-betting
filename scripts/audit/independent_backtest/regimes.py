"""Threshold-regime timeline reconstructed from git history (read-only, no production imports).

For every commit that touched a regime file, record the EFFECTIVE values:
  * config.py literal defaults, read with `ast` (never executed);
  * overridden by the `env:` block of .github/workflows/predict.yml at that commit
    (the deployed thresholds are the env values, not the config defaults - CLAUDE.md trap 1);
  * per-league approved gates from models/best_params_standard.json and
    models/best_params_side_markets.json at that commit.
Code-regime switches (logic, not numbers) are dated from the git log of src/betting.py /
src/ledger.py / pipeline.py and listed in CODE_REGIMES.
"""
from __future__ import annotations

import ast
import json
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime

from .common import REF, file_commits, show

REGIME_FILES = ["config.py", ".github/workflows/predict.yml", "models/best_params_standard.json",
                "models/best_params_side_markets.json"]

SCALARS = ["VALUABLE_THRESHOLD", "MARKSMAN_THRESHOLD", "SNIPER_THRESHOLD", "SNIPER_THRESHOLD_OVER",
           "SNIPER_THRESHOLD_UNDER", "EDGE_CEILING", "MIN_OVER_ODDS", "MIN_UNDER_ODDS", "NF_MAX_OU_ODDS",
           "LEAGUE_SNIPER_CAP"]
ENV_ONLY = ["VALUE_THRESHOLD", "REQUIRE_FORM_DATA", "REQUIRE_FORM_DATA_UNTIL", "STANDARD_OU25_SERVING_RECIPE"]

# Dated logic switches (commit evidence in parentheses). UTC-ish ISO; compared to decision time.
CODE_REGIMES = {
    "blind_guard_code": "2026-08-15T16:34:18Z",      # d4bd8ba5e
    "no_bettable_side_avoid": "2026-08-16T18:20:42Z",  # 8033f8f1f (122 side=AVOID rows before)
    "flip_guard": "2026-10-11T07:05:20Z",             # 467588bd3
    "direction_guard": "2026-10-25T00:00:00Z",        # src/direction_guard.py (owner 2026-10-25)
    "nf_price_cap": "2026-10-22T06:10:54Z",           # 581a07c1b
    "drift_tier_removed": "2026-10-26T09:43:49Z",     # 53a10e9f6
    "first_tier_stands": "2026-10-26T12:00:00Z",      # dfcdf289d
    "standard_repaired_serving": "2026-10-28T12:45:00Z",
    # both_losing guard read v7 best_params.json from DATA_DIR (outside the repo, never committed);
    # removed 2026-06-25 -> before that a league-wide mute is NOT reconstructable from git.
    "both_losing_guard_removed": "2026-06-25T00:00:00Z",
}


def _literal_default(node):
    """float(os.getenv("X", "0.04")) -> 0.04 ; plain number -> number ; else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "float" and node.args:
        inner = node.args[0]
        if isinstance(inner, ast.Call) and len(inner.args) >= 2 and isinstance(inner.args[1], ast.Constant):
            try:
                return float(inner.args[1].value)
            except (TypeError, ValueError):
                return None
        if isinstance(inner, ast.Constant):
            try:
                return float(inner.value)
            except (TypeError, ValueError):
                return None
    return None


def parse_config(src: str) -> dict:
    out: dict = {"league_sniper": {}, "league_marksman": {}, "league_edge_ceiling": {}}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        for t in targets:
            name = getattr(t, "id", None)
            if not name or value is None:
                continue
            if name == "VALUE_THRESHOLD" and "VALUABLE_THRESHOLD" not in out:
                v = _literal_default(value)          # pre-June name of the minimum-edge bar
                if v is not None:
                    out["VALUABLE_THRESHOLD"] = v
            if name in SCALARS or name == "_SNIPER_CAP":
                v = _literal_default(value)
                if v is not None:
                    out["LEAGUE_SNIPER_CAP" if name == "_SNIPER_CAP" else name] = v
            dict_target = {"_LEAGUE_SNIPER_CALIBRATED": "league_sniper", "LEAGUE_SNIPER_THRESHOLDS": "league_sniper",
                           "_LEAGUE_MARKSMAN_CALIBRATED": "league_marksman",
                           "LEAGUE_MARKSMAN_THRESHOLDS": "league_marksman",
                           "LEAGUE_EDGE_CEILING": "league_edge_ceiling"}.get(name)
            if dict_target and isinstance(value, ast.Dict):
                try:
                    out[dict_target] = {k: float(v) for k, v in ast.literal_eval(value).items()}
                except Exception:
                    pass
    return out


_ENV_RE = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)\s*:\s*"?([^"#\n]*?)"?\s*(?:#.*)?$')


def parse_predict_env(yml: str) -> dict:
    """Env values set for the predict job (simple line parse; secrets lines ignored)."""
    env = {}
    for line in yml.splitlines():
        m = _ENV_RE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k in SCALARS or k in ENV_ONLY:
            if "${{" in v:
                continue
            env[k] = v
    return env


@dataclass
class Regime:
    since_sha: str
    since_ts: str
    values: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    std_params: dict = field(default_factory=dict)
    side_params: dict = field(default_factory=dict)

    def get(self, name: str, default=None):
        if name == "VALUABLE_THRESHOLD" and name not in self.env and "VALUE_THRESHOLD" in self.env:
            name_env = "VALUE_THRESHOLD"
            try:
                return float(self.env[name_env])
            except ValueError:
                pass
        if name in self.env:
            try:
                return float(self.env[name])
            except ValueError:
                return self.env[name]
        return self.values.get(name, default)

    def fingerprint(self) -> str:
        keys = ["VALUABLE_THRESHOLD", "MARKSMAN_THRESHOLD", "LEAGUE_SNIPER_CAP", "MIN_OVER_ODDS", "NF_MAX_OU_ODDS"]
        approved = sorted(k for k, v in self.std_params.items() if isinstance(v, dict) and v.get("approved"))
        return "|".join(f"{k}={self.get(k)}" for k in keys) + f"|std_approved={','.join(approved)}"


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


class RegimeTimeline:
    def __init__(self, ref: str = REF):
        events = {}
        for f in REGIME_FILES:
            for sha, ts in file_commits(f, ref):
                events[sha] = ts
        ordered = sorted(events.items(), key=lambda kv: _ts(kv[1]))
        self.regimes: list[Regime] = []
        for sha, ts in ordered:
            def _txt(p):
                b = show(sha, p)
                return b.decode("utf-8", errors="replace") if b else ""
            cfg = parse_config(_txt("config.py"))
            env = parse_predict_env(_txt(".github/workflows/predict.yml"))
            try:
                std = json.loads(_txt("models/best_params_standard.json") or "{}")
            except json.JSONDecodeError:
                std = {}
            try:
                side = json.loads(_txt("models/best_params_side_markets.json") or "{}")
            except json.JSONDecodeError:
                side = {}
            self.regimes.append(Regime(sha, ts, cfg, env, std, side))
        self._keys = [_ts(r.since_ts) for r in self.regimes]

    def at(self, when) -> Regime | None:
        t = _ts(when) if not isinstance(when, datetime) else when
        i = bisect_right(self._keys, t) - 1
        return self.regimes[i] if i >= 0 else None

    def summary(self) -> list[dict]:
        rows, last = [], None
        for r in self.regimes:
            fp = r.fingerprint()
            if fp != last:
                rows.append({"since": r.since_ts, "sha": r.since_sha[:9], "fingerprint": fp})
                last = fp
        return rows


def code_regime_active(name: str, when) -> bool:
    t = _ts(when) if not isinstance(when, datetime) else when
    return t >= _ts(CODE_REGIMES[name])
