"""Build orphan branch af-prices-archive in wowza-exec by plumbing: byte-exact copies of the class A
and class B files, no transformation. Prints counts only - never file contents."""
import json, sys, os, subprocess, hashlib, csv, io, tempfile, pathlib
sys.path.insert(0, os.path.dirname(__file__)); from classify import classify
G = r"C:\Program Files\Git\cmd\git.exe"
EXEC = r"C:\Users\nevo\dev\wowza-exec"
S = sys.argv[1]; extra_files = sys.argv[2]   # dir holding README/.gitattributes to add
SOURCES = [("wowza-betting","origin/main","main"),("wowza-betting","origin/data-truth-100","data-truth-100"),
           ("wowza-betting","origin/prep/bars-oddsapi-hist","prep__bars-oddsapi-hist"),
           ("wowzaV9-Pro","origin/main","main"),("wowza_v11","origin/main","main"),("wowza-dashboard","origin/main","main")]
def cat(repo, sha): return subprocess.run([G,"-C",rf"C:\Users\nevo\dev\{repo}","cat-file","blob",sha],capture_output=True,check=True).stdout
rows=[]; idx=[]
for repo, ref, label in SOURCES:
    f = os.path.join(S, f"prof_{repo}_{ref.replace('/','_')}.json")
    for r in json.load(open(f)):
        if r.get("error"): continue
        c, prov, ds = classify(repo, r["path"])
        if c not in ("A","B"): continue
        data = cat(repo, r["blob"])
        h = hashlib.sha256(data).hexdigest()
        oid = subprocess.run([G,"-C",EXEC,"hash-object","-w","--no-filters","--stdin"],input=data,capture_output=True,check=True).stdout.decode().strip()
        if oid != r["blob"] or len(data) != r["bytes"]:
            sys.exit(f"MISMATCH {repo} {ref} {r['path']}")
        ap = f"{repo}/{label}/{r['path']}"
        idx.append(f"100644 {oid}\t{ap}")
        rows.append(dict(archive_path=ap, source_repo=f"NevixAA/{repo}", source_ref=ref.replace("origin/",""), source_commit=r["commit"],
                         source_path=r["path"], cls=c, dataset=ds, provider=prov, bytes=len(data), rows=r.get("rows",""),
                         date_min=r.get("date_min",""), date_max=r.get("date_max",""), sha256=h, git_blob=oid))
    print(f"[archive] {repo}@{label}: {sum(1 for x in rows if x['archive_path'].startswith(repo+'/'+label+'/'))} file(s)", flush=True)
buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
man = buf.getvalue().encode()
moid = subprocess.run([G,"-C",EXEC,"hash-object","-w","--no-filters","--stdin"],input=man,capture_output=True,check=True).stdout.decode().strip()
idx.append(f"100644 {moid}\tMANIFEST.csv")
for name in os.listdir(extra_files):
    b = open(os.path.join(extra_files,name),"rb").read()
    o = subprocess.run([G,"-C",EXEC,"hash-object","-w","--no-filters","--stdin"],input=b,capture_output=True,check=True).stdout.decode().strip()
    idx.append(f"100644 {o}\t{name}")
env = {**os.environ, "GIT_INDEX_FILE": os.path.join(S,"archive.index")}
if os.path.exists(env["GIT_INDEX_FILE"]): os.remove(env["GIT_INDEX_FILE"])
subprocess.run([G,"-C",EXEC,"update-index","--add","--index-info"],input="\n".join(idx)+"\n",text=True,env=env,check=True)
tree = subprocess.run([G,"-C",EXEC,"write-tree"],capture_output=True,text=True,env=env,check=True).stdout.strip()
open(os.path.join(S,"archive_tree.txt"),"w").write(tree)
open(os.path.join(S,"MANIFEST.csv"),"wb").write(man)
print(f"[archive] files={len(rows)} bytes={sum(x['bytes'] for x in rows)} rows={sum(int(x['rows'] or 0) for x in rows)} tree={tree}")
