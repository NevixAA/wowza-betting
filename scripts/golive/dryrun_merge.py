r"""Go-live DRY-RUN of the data-truth-100 -> main merge, for all five Wowza repos. NOTHING IS PUSHED.

    python scripts/golive/dryrun_merge.py --root C:\Users\nevo\dev\golive-dry [--repos v9,pro,...]
    python scripts/golive/dryrun_merge.py --root C:\Users\nevo\dev\golive-dry --cleanup

Per repo it makes a THROWAWAY clone under --root (objects borrowed from the local clone in
C:\Users\nevo\dev\<repo> through --reference, refs fetched fresh from GitHub), checks out
origin/main, runs `git merge --no-ff --no-commit origin/data-truth-100`, and records:

  * the SHAs of origin/main and origin/data-truth-100 at fetch time and their merge base,
  * the commits each side has that the other lacks (git rev-list --left-right --count),
  * every conflicted path, its conflict type (git ls-files -u stages), and the commits on each side
    that touched it since the merge base,
  * the files main-only commits changed (by top-level directory) - the auto-commit footprint.

The merge is left in the clone UNCOMMITTED (resolve + test there by hand). The clone has no push URL
(`git remote set-url --push origin DISABLED`), so a push from it fails. --cleanup deletes the clones.
Writes <root>/dryrun_<repo>.json. Exit 0 always (conflicts are a finding, not an error).
"""
from __future__ import annotations

import argparse
import json
import shutil
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path

GIT = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
DEV = Path(r"C:\Users\nevo\dev")
REPOS = {
    "v9": "wowza-betting",
    "pro": "wowzaV9-Pro",
    "v11": "wowza_v11",
    "dashboard": "wowza-dashboard",
    "exec": "wowza-exec",
}


def git(cwd: Path | None, *args: str, check: bool = True) -> str:
    cmd = [GIT, "-c", "core.autocrlf=false", "-c", "core.longpaths=true"]
    if cwd is not None:
        cmd += ["-C", str(cwd)]
    r = subprocess.run(cmd + list(args), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({r.returncode}): {r.stderr.strip()[:800]}")
    return r.stdout


def dryrun(root: Path, key: str) -> dict:
    name = REPOS[key]
    dest = root / name
    if dest.exists():
        raise SystemExit(f"{dest} exists - run --cleanup first")
    url = f"https://github.com/NevixAA/{name}.git"
    git(None, "clone", "-q", "--no-checkout", "--reference", str(DEV / name), url, str(dest))
    git(dest, "remote", "set-url", "--push", "origin", "DISABLED")
    git(dest, "fetch", "-q", "origin", "main", "data-truth-100")
    main = git(dest, "rev-parse", "origin/main").strip()
    dt = git(dest, "rev-parse", "origin/data-truth-100").strip()
    base = git(dest, "merge-base", main, dt).strip()
    left, right = git(dest, "rev-list", "--left-right", "--count", f"{main}...{dt}").split()
    main_only_files = [f for f in git(dest, "log", "--format=", "--name-only", f"{dt}..{main}").splitlines() if f]
    authors = Counter(git(dest, "log", "--format=%an", f"{dt}..{main}").splitlines())
    git(dest, "checkout", "-q", "-B", "golive-dryrun", main)
    m = subprocess.run([GIT, "-c", "core.autocrlf=false", "-c", "user.name=dryrun", "-c", "user.email=dryrun@localhost",
                        "-C", str(dest), "merge", "--no-ff", "--no-commit", dt],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    unmerged: dict[str, set] = {}
    for line in git(dest, "ls-files", "-u").splitlines():
        meta, path = line.split("\t", 1)
        unmerged.setdefault(path, set()).add(int(meta.split()[2]))
    kinds = {frozenset({1, 2, 3}): "both modified", frozenset({2, 3}): "both added",
             frozenset({1, 2}): "deleted by data-truth-100", frozenset({1, 3}): "deleted by main"}
    conflicts = []
    for path, stages in sorted(unmerged.items()):
        conflicts.append({
            "path": path,
            "type": kinds.get(frozenset(stages), str(sorted(stages))),
            "main_commits": git(dest, "log", "--format=%h %an: %s", f"{base}..{main}", "--", path).splitlines()[:6],
            "dt_commits": git(dest, "log", "--format=%h %an: %s", f"{base}..{dt}", "--", path).splitlines()[:6],
        })
    staged = git(dest, "diff", "--cached", "--name-only", "--diff-filter=ACMRD").splitlines()
    out = {
        "repo": name, "clone": str(dest), "origin_main": main, "origin_data_truth_100": dt, "merge_base": base,
        "main_only_commits": int(left), "dt_only_commits": int(right),
        "main_only_authors": dict(authors),
        "main_only_files_by_dir": dict(Counter("/".join(f.split("/")[:2]) for f in set(main_only_files)).most_common(25)),
        "merge_exit": m.returncode, "merge_stdout_tail": m.stdout.strip().splitlines()[-5:],
        "conflicts": conflicts, "n_conflicts": len(conflicts), "n_files_changed_vs_main": len(staged),
    }
    (root / f"dryrun_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _rm_ro(func, path, _exc):
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repos", default=",".join(REPOS))
    ap.add_argument("--cleanup", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cleanup:
        for key in a.repos.split(","):
            d = root / REPOS[key]
            if d.exists():
                shutil.rmtree(d, onerror=_rm_ro)
                print("removed", d)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    for key in a.repos.split(","):
        r = dryrun(root, key)
        print(f"{r['repo']}: main-only {r['main_only_commits']}, dt-only {r['dt_only_commits']}, "
              f"conflicts {r['n_conflicts']}, merge exit {r['merge_exit']}")
        for c in r["conflicts"]:
            print(f"   {c['type']:<26} {c['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
