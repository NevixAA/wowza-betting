"""CANONICAL TRACK REGISTRY — one generated read-model per league x market x side x model_type.

    python scripts/build_track_registry.py            # writes config/track_registry.json
    python scripts/build_track_registry.py --check    # exit 1 if the committed file is out of date

WHAT IT JOINS (2026-10-25 platform brief, step a3). Each row answers "what is true of this track
right now, and who says so":

    research_regime, owner_locked, evidence_*   config/league_market_regimes.json (research labels)
    production_state, paper_league, enabled,    config/bankroll_policy.json + config.py eligibility,
      sent, real_money, blocked_market, odds     resolved under the DEPLOYED predict.yml env (the same
      caps, gates                                subprocess src/research_lock.py uses), i.e. what
                                                 production actually applies
    prediction_engine                           models/registry/current_champions.json (champion that
                                                 scores this market for this model_type)
    selection_engine                            production "legacy_raw_edge"; shadow "market_edge_v2"
    stake_policy                                bankroll policy version, base fraction, tier multipliers
    last_reviewed                               regime registry date, policy date, lock approval date

READ-MODEL ONLY. No production code reads this file in this step: switching a reader to it would be a
production change and needs the ADD -> SHADOW -> COMPARE -> VALIDATE -> OWNER REVIEW -> CUTOVER path.
tests/test_track_registry.py proves it agrees with what production applies, so drift is caught.

DETERMINISTIC: no wall clock. The same inputs give byte-identical output; source digests are taken
over canonical JSON (line-ending independent, so Windows CRLF checkouts and CI agree).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "config" / "track_registry.json"
REGIMES_PATH = ROOT / "config" / "league_market_regimes.json"
POLICY_PATH = ROOT / "config" / "bankroll_policy.json"
CHAMPIONS_PATH = ROOT / "models" / "registry" / "current_champions.json"
LOCK_PATH = ROOT / "config" / "research_lock_snapshot.json"
EDGE_CFG_PATH = ROOT / "config" / "market_edge_v2.json"

REGISTRY_VERSION = "TRACK-REGISTRY-1"
REAL_STATES = ("PROBATION", "ACTIVE_LOW", "ACTIVE", "ACTIVE_STRONG", "DEGRADED")
BLOCKED_STATES = ("BLOCKED", "BLOCKED_FROM_REAL")
#: registry market -> (gate_resolver market, champion slot by model_type)
MARKET_MODEL = {"OU25": {"standard": "standard", "new_format": "newformat"},
                "BTTS": "btts", "OVER15": "over15", "OVER35": "over35"}
GATE_MARKET = {"OU25": "ou25", "BTTS": "btts", "OVER15": "over15", "OVER35": "over35"}


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def canonical_sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def champion_slot(market: str, model_type: str) -> str | None:
    m = MARKET_MODEL.get(market)
    if isinstance(m, dict):
        return m.get(model_type)
    return m


def production_view(league: str, market: str, side: str, model_type: str, policy: dict,
                    elig: dict) -> dict:
    """What production applies to this cell. Mirrors src/bankroll_policy.size_day (track state, base
    fraction, paper-league override, odds caps) and config (ENABLED/PAPER leagues, provider support,
    NF O/U cap, min O/U odds), with eligibility resolved under the deployed env."""
    track = f"{model_type}|{market}"
    t = (policy.get("tracks") or {}).get(track) or {}
    track_state = t.get("state", "UNSUPPORTED")
    base = float(t.get("base_fraction", 0.0) or 0.0)
    enabled = league in set(elig.get("ENABLED_LEAGUES", []))
    paper = league in set(policy.get("paper_leagues", [])) or league in set(elig.get("PAPER_LEAGUES", []))
    if not enabled:
        state = "NOT_ENABLED"
    elif paper:
        state = "PAPER_LEAGUE"
    else:
        state = track_state
    hard = float((policy.get("portfolio") or {}).get("hard_max_odds", 3.0))
    caps = {"hard_max_odds": hard, "track_min_odds": t.get("min_odds"), "track_max_odds": t.get("max_odds")}
    if market == "OU25":
        caps["min_side_odds"] = elig.get("MIN_OVER_ODDS") if side == "OVER" else elig.get("MIN_UNDER_ODDS")
        if model_type == "new_format":
            caps["nf_max_ou_odds"] = elig.get("NF_MAX_OU_ODDS")
    tt = (policy.get("track_tier_multipliers") or {}).get(track)
    return {
        "track": track, "track_state": track_state, "production_state": state,
        "enabled": enabled, "paper_league": paper,
        "provider_supported": league not in set(elig.get("PROVIDER_UNSUPPORTED", [])),
        "sent": bool(enabled and not paper),
        "real_money": bool(state in REAL_STATES and base > 0),
        "blocked_market": track_state in BLOCKED_STATES,
        "odds_caps": caps,
        "stake_policy": {"policy_version": policy.get("policy_version"), "base_fraction": base,
                         "tier_multipliers": tt if tt is not None else policy.get("tier_multipliers"),
                         "tier_multiplier_source": "track_tier_multipliers" if tt is not None else "tier_multipliers",
                         "evidence_score": t.get("evidence_score")},
    }


def evidence_status(e: dict) -> str:
    n, md = int(e.get("evidence_n") or 0), int(e.get("independent_matchdays") or 0)
    if n == 0:
        return "NONE"
    return "ADEQUATE" if (n >= 30 and md >= 8) else "INSUFFICIENT"


def build(regimes: dict, policy: dict, champions: dict, elig: dict, gates: dict, lock: dict,
          edge_cfg: dict) -> dict:
    """Pure: same inputs -> byte-identical output."""
    entries = []
    for e in sorted(regimes.get("entries", []), key=lambda x: x["key"]):
        lg, mk, sd, mt = e["league"], e["market"], e["side"], e["model_type"]
        pv = production_view(lg, mk, sd, mt, policy, elig)
        slot = champion_slot(mk, mt)
        ch = champions.get(slot or "", {}) if slot else {}
        g = gates.get(f"{lg}|{GATE_MARKET[mk]}|{sd if mk == 'OU25' else ''}")
        entries.append({
            "key": e["key"], "league": lg, "market": mk, "side": sd, "model_type": mt,
            "research_regime": e.get("research_regime"), "owner_locked": bool(e.get("owner_locked")),
            **{k: pv[k] for k in ("production_state", "track", "track_state", "enabled", "paper_league",
                                  "provider_supported", "sent", "real_money", "blocked_market", "odds_caps")},
            "gates": g,
            "prediction_engine": {"model_slot": slot, "model_version": ch.get("model_version"),
                                  "champion_since": ch.get("champion_since"),
                                  "artifact_sha256_16": (ch.get("artifact_sha256") or "")[:16] or None,
                                  "promotion_source": ch.get("promotion_source")},
            "selection_engine": {"production": "legacy_raw_edge",
                                 "shadow": edge_cfg.get("engine_version", "market_edge_v2"),
                                 "shadow_config_status": edge_cfg.get("status")},
            "stake_policy": pv["stake_policy"],
            "evidence_status": evidence_status(e), "evidence_n": e.get("evidence_n"),
            "independent_matchdays": e.get("independent_matchdays"), "evidence_reason": e.get("reason"),
            "evidence_cutoff": e.get("evidence_cutoff"),
            "last_reviewed": {"research_regime": e.get("last_updated") or regimes.get("generated_for"),
                              "bankroll_policy": policy.get("generated_at"),
                              "owner_lock_approved_at": lock.get("approved_at") if e.get("owner_locked") else None},
        })
    counts = {}
    for x in entries:
        counts[x["production_state"]] = counts.get(x["production_state"], 0) + 1
    return {
        "about": "CANONICAL TRACK REGISTRY (read-model). Generated by scripts/build_track_registry.py from "
                 "the regime registry, the bankroll policy, config eligibility under the deployed predict.yml "
                 "env, the champion pointer and the edge-engine config. Production does NOT read this file "
                 "yet; tests/test_track_registry.py checks it agrees with what production applies. Do not "
                 "hand-edit.",
        "registry_version": REGISTRY_VERSION,
        "generated_for": regimes.get("generated_for"),
        "policy_version": policy.get("policy_version"),
        "selection_engine_version": {"production": "legacy_raw_edge",
                                     "shadow": edge_cfg.get("engine_version", "market_edge_v2")},
        "sources": {
            "league_market_regimes": {"path": "config/league_market_regimes.json", "sha256_16": canonical_sha16(regimes)},
            "bankroll_policy": {"path": "config/bankroll_policy.json", "sha256_16": canonical_sha16(policy)},
            "current_champions": {"path": "models/registry/current_champions.json", "sha256_16": canonical_sha16(champions)},
            "deployed_eligibility": {"path": "config.py under .github/workflows/predict.yml env",
                                     "sha256_16": canonical_sha16(elig)},
            "market_edge_v2": {"path": "config/market_edge_v2.json", "sha256_16": canonical_sha16(edge_cfg)},
            "research_lock_snapshot": {"path": "config/research_lock_snapshot.json",
                                       "approved_at": lock.get("approved_at"), "approved_by": lock.get("approved_by")},
        },
        "global": {
            "paper_leagues": sorted(set(policy.get("paper_leagues", [])) | set(elig.get("PAPER_LEAGUES", []))),
            "blocked_tracks": sorted(k for k, v in (policy.get("tracks") or {}).items()
                                     if v.get("state") in BLOCKED_STATES),
            "real_money_tracks": sorted(k for k, v in (policy.get("tracks") or {}).items()
                                        if v.get("state") in REAL_STATES and float(v.get("base_fraction", 0) or 0) > 0),
            "player_props": (policy.get("tracks") or {}).get("player_props", {}).get("state"),
            "cloudbet": "PAPER_ONLY",
        },
        "n_entries": len(entries), "production_state_counts": dict(sorted(counts.items())),
        "entries": entries,
    }


def resolve_deployed(cells: list) -> tuple[dict, dict]:
    """Eligibility + gates under the deployed predict.yml env (src/research_lock's subprocess)."""
    from src import research_lock as RL
    res = RL._resolve(RL.deployed_env(), cells)
    return res["eligibility"], res["gates"]


def generate() -> dict:
    regimes = _read(REGIMES_PATH)
    cells = sorted({(e["league"], GATE_MARKET[e["market"]], e["side"] if e["market"] == "OU25" else "")
                    for e in regimes.get("entries", [])})
    elig, gates = resolve_deployed([list(c) for c in cells])
    return build(regimes, _read(POLICY_PATH), _read(CHAMPIONS_PATH), elig, gates, _read(LOCK_PATH),
                 _read(EDGE_CFG_PATH))


def render(reg: dict) -> str:
    return json.dumps(reg, indent=1, ensure_ascii=False, default=str) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if config/track_registry.json is stale")
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    reg = generate()
    text = render(reg)
    out = Path(a.out)
    if a.check:
        cur = _read(out)
        stale = cur != json.loads(text)
        print(f"track registry {'STALE' if stale else 'in sync'} ({reg['n_entries']} entries)")
        return 1 if stale else 0
    if out.name != "track_registry.json":
        raise SystemExit("refusing to write anything but a track_registry.json")
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{out}: {reg['n_entries']} entries, states {reg['production_state_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
