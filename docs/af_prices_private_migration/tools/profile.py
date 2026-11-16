"""Inventory API-Football / provider price files in a git ref. Prints NO prices: only counts,
column NAMES, date ranges and source-label counts."""
import subprocess, sys, re, io, json, os, csv
import pandas as pd, pyarrow.parquet as pq
G = r"C:\Program Files\Git\cmd\git.exe"
repo_dir, repo, ref, out_path = sys.argv[1:5]
sha_ref = subprocess.run([G,"-C",repo_dir,"rev-parse",ref],capture_output=True,text=True).stdout.strip()
ls = subprocess.run([G,"-C",repo_dir,"-c","core.quotepath=off","ls-tree","-r","-l",ref],capture_output=True,text=True,encoding="utf-8").stdout
PRICE = re.compile(r"(^|_)(odds|odd|price)(_|$)|^odds|_odds$|best_back|exch_back|median_back|entry_back|best_opposite|odds_(over|under|btts|home|away|draw)|implied_prob|api_implied|^best_(over|under)|^close_(first|other)$|^(odds|close)_(first|other)$", re.I)
NOTPRICE = re.compile(r"^(avg|mean|median|fair|min|max|hard|track|cov|n|nf|std)_|band|blocked|snapshots|source|clv|_ts$|_at$|age|stale|updated|^is_|status|quality|changes|provenance|^price_(ts|timestamp|delta|change|band|derived|is_real|mode)|out_price|in_price|haircut|cap|fallback|independence|chance|sell", re.I)
FANTASY = re.compile(r"fantasy|fpl_")
CANDIDATE = re.compile(r"\.(csv|parquet|json)$")
DATECOLS = ["snapshot_ts","captured_at","snapshot_date","match_date","date","kickoff_utc","kickoff","commence_time","ts","generated_at","created_at","settled_at","tip_date"]
SRCCOLS = ["source","odds_source","price_source","writer","bookmaker_key","capture"]
JSON_PRICE = {"odds_history_v9.json","output/worldcup_history.json","output/weekly_league_report.json"}
def blob(sha):
    return subprocess.run([G,"-C",repo_dir,"cat-file","-p",sha],capture_output=True).stdout
res=[]
for line in ls.splitlines():
    meta, path = line.split("\t",1)
    _,typ,sha,size = meta.split()
    if typ!="blob" or not CANDIDATE.search(path) or FANTASY.search(path): continue
    if path.startswith((".github/","tests/","src/","scripts/","pages/","dashboard_data/","agent/","config/","registry/","models/","player_model/props_feature")): continue
    is_json = path.endswith(".json")
    if is_json and not (path in JSON_PRICE or path.startswith("output/sharp_history/")): continue
    rec={"repo":repo,"ref":ref,"commit":sha_ref,"path":path,"blob":sha,"bytes":int(size)}
    b=blob(sha)
    try:
        if path.endswith(".csv"):
            df=pd.read_csv(io.BytesIO(b),dtype=str,keep_default_na=False,low_memory=False,on_bad_lines="warn")
            cols=list(df.columns)
        elif path.endswith(".parquet"):
            df=pq.read_table(io.BytesIO(b)).to_pandas(); cols=list(df.columns)
        else:
            j=json.loads(b.decode("utf-8")); cols=["<json>"]
            rec.update(price_cols=["<json values>"], rows=len(j) if hasattr(j,"__len__") else None)
            res.append(rec); continue
    except Exception as e:
        rec["error"]=type(e).__name__; rec["price_cols"]=[]; res.append(rec); continue
    pc=[c for c in cols if PRICE.search(str(c)) and not NOTPRICE.search(str(c))]
    if not pc: continue
    rec["price_cols"]=pc; rec["rows"]=int(len(df)); rec["ncols"]=len(cols)
    for c in DATECOLS:
        if c in df.columns:
            v=df[c].astype(str).str.slice(0,10); v=v[v.str.match(r"^\d{4}-\d{2}-\d{2}$")]
            if len(v): rec["date_col"]=c; rec["date_min"]=v.min(); rec["date_max"]=v.max(); break
    for c in SRCCOLS:
        if c in df.columns:
            rec.setdefault("src",{})[c]={str(k):int(n) for k,n in df[c].astype(str).value_counts().head(12).items()}
    res.append(rec)
json.dump(res,open(out_path,"w"),indent=0)
print(repo,ref,sha_ref[:10],"files",len(res),"rows",sum(r.get("rows") or 0 for r in res),"bytes",sum(r["bytes"] for r in res))
