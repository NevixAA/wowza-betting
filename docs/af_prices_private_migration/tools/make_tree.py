import csv, subprocess, os, sys
G = r"C:\Program Files\Git\cmd\git.exe"; EXEC = r"C:\Users\nevo\dev\wowza-exec"; S = sys.argv[1]
def ho(b): return subprocess.run([G,"-C",EXEC,"hash-object","-w","--no-filters","--stdin"],input=b,capture_output=True,check=True).stdout.decode().strip()
lines = []
man = open(os.path.join(S,"MANIFEST.csv"),"rb").read()
for r in csv.DictReader(man.decode().splitlines()):
    lines.append(f"100644 {r['git_blob']}\t{r['archive_path']}")
lines.append(f"100644 {ho(man)}\tMANIFEST.csv")
for n in os.listdir(os.path.join(S,"extra")):
    lines.append(f"100644 {ho(open(os.path.join(S,'extra',n),'rb').read())}\t{n}")
idx = os.path.join(S,"archive2.index")
if os.path.exists(idx): os.remove(idx)
env = {**os.environ, "GIT_INDEX_FILE": idx}
p = subprocess.run([G,"-C",EXEC,"update-index","--add","--index-info"],input=("\n".join(lines)+"\n").encode(),env=env,capture_output=True)
print("update-index rc", p.returncode, p.stderr.decode()[:300])
n = subprocess.run([G,"-C",EXEC,"ls-files"],env=env,capture_output=True,text=True).stdout.count("\n")
tree = subprocess.run([G,"-C",EXEC,"write-tree"],env=env,capture_output=True,text=True,check=True).stdout.strip()
print("entries", n, "expected", len(lines), "tree", tree)
