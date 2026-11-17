"""NF HT-input coverage across the last N main commits of output/predictions.csv. Read only (git show)."""
import io, subprocess, sys
import pandas as pd
G = r"C:\Program Files\Git\cmd\git.exe"; R = r"C:\Users\nevo\dev\wowza-betting"
log = subprocess.run([G, "-C", R, "log", "-12", "--format=%h %cI", "origin/main", "--", "output/predictions.csv"],
                     capture_output=True, text=True).stdout.split("\n")
for line in filter(None, log):
    h, t = line.split()
    raw = subprocess.run([G, "-C", R, "show", f"{h}:output/predictions.csv"], capture_output=True).stdout
    p = pd.read_csv(io.BytesIO(raw), low_memory=False, usecols=lambda c: c in ("model_type", "home_ht_over05_rate", "home_rest_days"))
    out = []
    for mt, g in p.groupby(p["model_type"].astype(str)):
        out.append(f"{mt}: n={len(g)} ht_miss={g['home_ht_over05_rate'].isna().mean()*100:.0f}% rest_miss={g['home_rest_days'].isna().mean()*100:.0f}%")
    print(h, t, " | ".join(out))
