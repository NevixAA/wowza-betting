import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

wc = pd.read_csv('output/worldcup_tips.csv')
wc['date'] = pd.to_datetime(wc['date'], errors='coerce')
wc = wc.sort_values(['signal', 'drift_pct'])

print(f"=== WORLD CUP 2026 — {len(wc)} drift signals ===")
print(f"Started: June 11, 2026\n")

for sig, emoji in [('STRONG','🔴'), ('SHARP','🟡'), ('FADING','⬆️')]:
    t = wc[wc['signal']==sig]
    if t.empty: continue
    print(f"{emoji} {sig} ({len(t)}):")
    for _, r in t.iterrows():
        d = str(r['date'])[:10]
        print(f"  {d} | {r['match']:<40} | {r['market']:<25} | {r['drift_pct']:+.1f}% | {r['snapshots']} snaps")
    print()
