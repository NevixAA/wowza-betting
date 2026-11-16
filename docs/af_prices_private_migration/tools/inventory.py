"""Build the inventory tables (markdown) from the profiles. Counts, sizes, dates, names only."""
import json, os, re, sys, subprocess, collections
sys.path.insert(0, os.path.dirname(__file__)); from classify import classify
S = sys.argv[1]; G = r"C:\Program Files\Git\cmd\git.exe"
SRC = [("wowza-betting","origin/main"),("wowzaV9-Pro","origin/main"),("wowza_v11","origin/main"),("wowza-dashboard","origin/main"),
       ("wowza-betting","origin/data-truth-100"),("wowza-betting","origin/prep/bars-oddsapi-hist")]
WRITER = {
 "book_quotes":"nf_odds_capture.yml + std_odds_capture.yml -> scripts/capture_{nf_odds,std_sidemarket_odds}_forward.py via src/book_quotes.py",
 "specials_quotes":"nf_odds_capture.yml + std_odds_capture.yml -> capture scripts via src/specials_quotes.py",
 "prop_quotes":"player_props.yml + prop_odds_snapshot.yml -> player_model/odds_fetcher.py (+ prop_odds_af.py for API-Football)",
 "player_prop_odds_history":"player_props.yml + prop_odds_snapshot.yml -> player_model/odds_fetcher.py, scripts/snapshot_prop_odds.py; src/rolling_store.py rolls to _archive/",
 "newformat_odds_history":"nf_odds_capture.yml -> scripts/capture_nf_odds_forward.py",
 "standard_sidemarket_odds_history":"std_odds_capture.yml -> scripts/capture_std_sidemarket_odds_forward.py",
 "odds_heartbeat":"nf_odds_capture.yml + std_odds_capture.yml -> src/odds_heartbeat.py",
 "book_odds_snapshots":"predict.yml -> src/predict.py + src/btts_odds.py (OddsAPI); src/rolling_store.py rolls to _archive/",
 "exchange_quotes":"exchange_odds_capture.yml -> scripts/capture_exchange_odds.py (OddsAPI)",
 "standard_odds_history":"predict.yml -> src/drift.py (OddsAPI payload)",
 "newformat_odds_dense":"predict.yml -> src/drift.py (OddsAPI payload)",
 "odds_history_v9.json":"predict.yml -> src/predict.py / src/drift.py (OddsAPI)",
 "sharp_history":"sharp_tracker.yml -> src/sharp_tracker.py (OddsAPI)",
 "worldcup_history":"worldcup.yml -> worldcup/tracker.py (OddsAPI)",
 "fd_history":"predict.yml / retrain.yml -> src/data_loader.py (football-data.co.uk download)",
 "scout_odds":"pro_scout.yml + pro_scout_near.yml -> src/scout/collect.py",
 "scout_specials_quotes":"pro_scout.yml + pro_scout_near.yml -> src/scout/specials.py",
 "live_odds_snapshots":"pro_live_odds.yml -> src/pipelines/live_odds.py",
 "market_snapshots":"pro_collect.yml -> src/importers/current_wowza.py (imports v9 predictions + capture histories)",
 "pro_book_odds_snapshots":"pro_collect.yml -> src/importers/current_wowza.py (imports v9 book_odds_snapshots / book_quotes)",
 "league_scout_af_odds_probe":"one-off probe 2026-10-08 (no workflow)",
}
TOKEN = {"pro_book_odds_snapshots":"book_odds_snapshots","odds_history_v9.json":"odds_history_v9","league_scout_af_odds_probe":"league_scout_af_odds_probe"}
def refs(repo, token):
    d = rf"C:\Users\nevo\dev\af-inv\{repo}"
    out = subprocess.run([G,"-C",d,"-c","core.quotepath=off","grep","-l","-F",token,"--","*.py","*.yml","*.sh"],capture_output=True,text=True,encoding="utf-8").stdout.split()
    wf = sorted(os.path.basename(p) for p in out if p.startswith(".github/workflows/"))
    tests = [p for p in out if p.startswith("tests/") or "/tests/" in p or p.startswith("src/pro_tests")]
    pages = [p for p in out if p.startswith(("pages/","dashboard_data/"))]
    code = [p for p in out if p not in tests and p not in pages and not p.startswith(".github/workflows/") and not p.startswith("_archive/")]
    return wf, code, pages, tests
groups = collections.OrderedDict()
for repo, ref in SRC:
    f = os.path.join(S, f"prof_{repo}_{ref.replace('/','_')}.json")
    if not os.path.exists(f): continue
    for r in json.load(open(f)):
        if r.get("error"): continue
        c, prov, ds = classify(repo, r["path"])
        k = (repo, ref.replace("origin/",""), c, ds)
        x = groups.setdefault(k, dict(prov=prov, n=0, bytes=0, rows=0, dmin="", dmax="", src=collections.Counter(), blobs=set(), paths=[]))
        x["n"] += 1; x["bytes"] += r["bytes"]; x["rows"] += int(r.get("rows") or 0); x["blobs"].add((r["path"], r["blob"])); x["paths"].append(r["path"])
        if r.get("date_min"):
            x["dmin"] = min(x["dmin"] or "9", r["date_min"]); x["dmax"] = max(x["dmax"], r["date_max"])
        for col, vc in (r.get("src") or {}).items():
            if col in ("source","odds_source","price_source"):
                for kk, vv in vc.items(): x["src"][f"{kk or '(blank)'}"] += vv
json.dump({"|".join(k): {"n":v["n"],"bytes":v["bytes"],"rows":v["rows"]} for k,v in groups.items()}, open(os.path.join(S,"groups.json"),"w"))
mainblobs = collections.defaultdict(set)
for (repo, ref, c, ds), x in groups.items():
    if ref == "main": mainblobs[repo] |= x["blobs"]
def mb(b): return f"{b/1e6:,.1f}"
def path_of(x):
    p = x["paths"][0]
    if x["n"] == 1: return p
    common = os.path.commonprefix(x["paths"])
    return common if common.endswith("/") else common + "*"
lines = {}
for cls in ("A","B","O","F"):
    L = ["| Repo @ branch | Path | Dataset | Provider | Files | MB | Rows | Date range | Source mix (rows) | Writer | Readers (workflows / code / dashboard / tests) |",
         "|---|---|---|---|---:|---:|---:|---|---|---|---|"]
    for (repo, ref, c, ds), x in groups.items():
        if c != cls or ref != "main": continue
        tok = TOKEN.get(ds, ds if c != "B" else os.path.basename(ds.rstrip("/")).replace("*","").replace(".csv","").replace(".parquet","").replace(".json","").strip("_-/"))
        wf, code, pages, tests = refs(repo, tok) if tok else ([],[],[],[])
        rd = f"wf: {', '.join(wf) or '-'}; code ({len(code)}): {', '.join(code[:8])}{' ...' if len(code)>8 else ''}; dashboard ({len(pages)}): {', '.join(p.split('/')[-1] for p in pages[:5])}{' ...' if len(pages)>5 else ''}; tests: {len(tests)}"
        src = ", ".join(f"{k}: {v:,}" for k, v in x["src"].most_common(4)) or "-"
        dr = f"{x['dmin']} .. {x['dmax']}" if x["dmin"] else "-"
        L.append(f"| {repo} @ {ref} | `{path_of(x)}` | {ds if c!='B' else '-'} | {x['prov'] if c!='B' else 'own record'} | {x['n']:,} | {mb(x['bytes'])} | {x['rows']:,} | {dr} | {src} | {WRITER.get(ds,'see workflows in readers column') } | {rd} |")
    lines[cls] = "\n".join(L)
# branch deltas
B = ["| Repo @ branch | Class | Files with price columns | MB | Rows | Files not byte-identical to main |", "|---|---|---:|---:|---:|---:|"]
for (repo, ref, c, ds), x in groups.items(): pass
agg = collections.defaultdict(lambda: dict(n=0,bytes=0,rows=0,diff=0))
for (repo, ref, c, ds), x in groups.items():
    a = agg[(repo, ref, c)]; a["n"] += x["n"]; a["bytes"] += x["bytes"]; a["rows"] += x["rows"]
    a["diff"] += len([1 for pb in x["blobs"] if pb not in mainblobs[repo]]) if ref != "main" else 0
for (repo, ref, c), a in sorted(agg.items()):
    B.append(f"| {repo} @ {ref} | {c} | {a['n']:,} | {mb(a['bytes'])} | {a['rows']:,} | {a['diff'] if ref!='main' else '-'} |")
lines["BR"] = "\n".join(B)
json.dump(lines, open(os.path.join(S,"tables.json"),"w"))
print({k: v.count("\n")-1 for k, v in lines.items()})
