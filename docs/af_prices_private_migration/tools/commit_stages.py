import subprocess, os, sys
G = r"C:\Program Files\Git\cmd\git.exe"; EXEC = r"C:\Users\nevo\dev\wowza-exec"; S = sys.argv[1]; FULL = sys.argv[2]
ent = subprocess.run([G,"-C",EXEC,"ls-tree","-r",FULL],capture_output=True,check=True).stdout.decode().splitlines()
stages = ["wowza-betting/main/","wowza-betting/data-truth-100/","wowza-betting/prep__bars-oddsapi-hist/","wowzaV9-Pro/main/data/season_2026_27/scout_specials_quotes/",
          "wowzaV9-Pro/main/data/season_2026_27/movement_observations/","wowzaV9-Pro/main/","wowza_v11/main/",""]
env = {**os.environ, "GIT_AUTHOR_NAME":"Nevo Hershkovitz","GIT_AUTHOR_EMAIL":"89246276+NevixAA@users.noreply.github.com",
       "GIT_COMMITTER_NAME":"Nevo Hershkovitz","GIT_COMMITTER_EMAIL":"89246276+NevixAA@users.noreply.github.com"}
parent = None; included = []; commits = []
for i, pre in enumerate(stages):
    included.append(pre)
    sel = [l for l in ent if any(l.split("\t",1)[1].startswith(p) for p in included if p)] if pre else ent
    idx = os.path.join(S, f"stage{i}.index")
    if os.path.exists(idx): os.remove(idx)
    e2 = {**env, "GIT_INDEX_FILE": idx}
    subprocess.run([G,"-C",EXEC,"update-index","--add","--index-info"],input=("\n".join(sel)+"\n").encode(),env=e2,check=True)
    tree = subprocess.run([G,"-C",EXEC,"write-tree"],env=e2,capture_output=True,text=True,check=True).stdout.strip()
    if pre:
        msg = f"af-prices-archive: stage {i+1}/{len(stages)} - byte-exact copy of {pre.rstrip('/')} ({len(sel)} files cumulative)\n\nOwner decision 2026-11-16: API-Football prices move from the public repos to this private store.\nStraight copy, no transformation; MANIFEST.csv arrives in the final commit.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    else:
        msg = f"af-prices-archive: MANIFEST.csv + README (sha256 per file, {len(sel)-3} data files)\n\nByte-exact copy of every class A (raw API-Football / mixed price archives) and class B\n(per-row prices in our own ledgers, tips and research frames) file in the public repos\nwowza-betting (main, data-truth-100, prep/bars-oddsapi-hist), wowzaV9-Pro (main) and\nwowza_v11 (main) at the commits recorded per row. Every git blob id equals its source\nblob id. Inventory: wowza-betting docs/AF_PRICES_PRIVATE_MIGRATION.md (branch chore/af-prices-private).\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    if tree == FULL or not pre: assert tree == FULL, "final tree mismatch"
    args = [G,"-C",EXEC,"commit-tree",tree] + (["-p",parent] if parent else [])
    parent = subprocess.run(args,input=msg.encode(),env=env,capture_output=True,check=True).stdout.decode().strip()
    commits.append(parent); print(f"stage {i+1}: files={len(sel)} commit={parent}", flush=True)
open(os.path.join(S,"stage_commits.txt"),"w").write("\n".join(commits)+"\n")
