"""Contract violations detail + policy/CLV/registry/champion summaries for one worktree (read only).
usage: python run_extra.py <worktree> <out.json>"""
import json, os, sys, glob
from collections import Counter
wt, out = sys.argv[1], sys.argv[2]
os.chdir(wt); sys.path.insert(0, wt)
from src import system_health as SH
res = {}
chk, con = SH.contracts_check()
res["contracts_check"] = {k: chk.get(k) for k in ("status", "value", "detail")}
res["contracts"] = {k: v for k, v in con.items() if k != "files"}
res["contract_files"] = {f: v for f, v in (con.get("files") or {}).items()
                         if isinstance(v, dict) and (v.get("new") or v.get("violations") or v.get("n_new"))}
try:
    res["contract_baseline_n"] = len(json.load(open("config/contract_baseline.json", encoding="utf-8")).get("entries", json.load(open("config/contract_baseline.json", encoding="utf-8"))))
except Exception as e:
    res["contract_baseline_n"] = str(e)
for fn in ("edge_shadow_health", "promotion_gate_health"):
    try:
        from datetime import datetime, timezone
        c = getattr(SH, fn)(datetime(2026, 11, 3, 12, tzinfo=timezone.utc))
        res[fn] = {k: c.get(k) for k in ("status", "value", "detail", "rule")}
    except Exception as e:
        res[fn] = str(e)
c = SH.track_registry_check(); res["track_registry_check"] = {k: c.get(k) for k in ("status", "value", "detail")}
# bankroll policy
pol = json.load(open("config/bankroll_policy.json", encoding="utf-8"))
res["policy_version"] = pol.get("policy_version")
tracks = pol.get("tracks") or {}
res["policy_states"] = {k: (v.get("state") if isinstance(v, dict) else v) for k, v in tracks.items()} if isinstance(tracks, dict) else str(type(tracks))
res["policy_state_counts"] = dict(Counter(res["policy_states"].values())) if isinstance(res["policy_states"], dict) else None
res["policy_top_keys"] = list(pol.keys())
# pooled league rating flag
res["policy_flags"] = {k: v for k, v in pol.items() if not isinstance(v, (dict, list))}
# clv health
clv = SH._json(SH.OUT / "clv_collection_health.json")
res["clv_collection_health"] = {k: v for k, v in clv.items() if not isinstance(v, (list,)) } if clv else "missing"
# track registry
reg = None
for p in ("registry/track_registry.json", "config/track_registry.json"):
    if os.path.exists(p):
        reg = json.load(open(p, encoding="utf-8")); res["track_registry_path"] = p; break
if reg is not None:
    ents = reg.get("tracks", reg.get("entries", reg)) if isinstance(reg, dict) else reg
    if isinstance(ents, dict): ents = list(ents.values())
    res["track_registry_n"] = len(ents)
    for key in ("state", "status", "lifecycle", "stage", "money_state", "evidence_class"):
        vals = [e.get(key) for e in ents if isinstance(e, dict) and key in e]
        if vals: res[f"registry_{key}"] = dict(Counter(map(str, vals)))
res["registry_dir"] = sorted(os.listdir("registry")) if os.path.isdir("registry") else None
res["models_champion_files"] = sorted(glob.glob("models/**/*champion*", recursive=True))[:30]
json.dump(res, open(out, "w", encoding="utf-8"), indent=1, default=str)
