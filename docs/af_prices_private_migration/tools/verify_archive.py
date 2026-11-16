"""Independent check: for every MANIFEST row, re-read the file from the PUSHED exec commit and from the
public source repo at the recorded commit by PATH (not by blob id), sha256 both, compare with the
manifest. Prints counts only."""
import csv, io, sys, subprocess, hashlib
G = r"C:\Program Files\Git\cmd\git.exe"; EXEC = r"C:\Users\nevo\dev\wowza-exec"
commit = sys.argv[1]
def cat(repo_dir, spec):
    return subprocess.run([G,"-C",repo_dir,"cat-file","blob",spec],capture_output=True,check=True).stdout
man = cat(EXEC, f"{commit}:MANIFEST.csv").decode()
ok = bad = 0; nbytes = 0
for r in csv.DictReader(io.StringIO(man)):
    a = hashlib.sha256(cat(EXEC, f"{commit}:{r['archive_path']}")).hexdigest()
    src_dir = rf"C:\Users\nevo\dev\{r['source_repo'].split('/')[1]}"
    s = hashlib.sha256(cat(src_dir, f"{r['source_commit']}:{r['source_path']}")).hexdigest()
    if a == s == r["sha256"]: ok += 1; nbytes += int(r["bytes"])
    else: bad += 1; print("MISMATCH", r["archive_path"])
print(f"[verify] commit={commit} ok={ok} mismatched={bad} bytes_ok={nbytes}")
sys.exit(1 if bad else 0)
