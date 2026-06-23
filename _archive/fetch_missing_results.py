"""Fetch missing June 6 results from OddsAPI and update ledger."""
import requests, sys, unicodedata
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
import config

LEDGER_FILE = config.OUTPUT_DIR / "bets_ledger.csv"

def norm(name):
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

def names_match(a, b):
    na, nb = norm(a), norm(b)
    if na == nb: return True
    if na in nb or nb in na: return True
    if na.split()[0] == nb.split()[0] and len(na.split()[0]) >= 4: return True
    return False

# ── Fetch completed scores from OddsAPI ──────────────────────────
completed = []
for sport_key, label in [
    ('soccer_japan_j_league',           'Japan J-League'),
    ('soccer_spain_segunda_division',   'La Liga 2'),
]:
    r = requests.get(
        f'https://api.the-odds-api.com/v4/sports/{sport_key}/scores',
        params={'apiKey': config.ODDS_API_KEY, 'daysFrom': 3},
        timeout=15
    )
    remaining = r.headers.get('x-requests-remaining', '?')
    print(f'{label}: HTTP {r.status_code} | credits remaining: {remaining}')
    if r.status_code == 200:
        for ev in r.json():
            if not ev.get('completed'):
                continue
            scores = {s['name']: int(s['score']) for s in (ev.get('scores') or [])
                      if s.get('score') is not None}
            h = ev['home_team']; a = ev['away_team']
            if h not in scores or a not in scores:
                continue
            date_str = ev['commence_time'][:10]
            print(f'  {date_str}  {h} {scores[h]}-{scores[a]} {a}')
            completed.append({
                'home_team':  h,
                'away_team':  a,
                'home_score': scores[h],
                'away_score': scores[a],
                'date_str':   date_str,
            })

if not completed:
    print('\nNo completed results found.')
    sys.exit(0)

# ── Match to ledger ───────────────────────────────────────────────
ledger = pd.read_csv(LEDGER_FILE, dtype=str)
if 'ht_score' not in ledger.columns:
    ledger['ht_score'] = ''

today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
pending_mask = (
    (ledger['result'].isna() | (ledger['result'].str.strip() == '')) &
    (ledger['match_date'] < today_str)
)
pending = ledger[pending_mask].copy()
print(f'\nPending in ledger: {len(pending)}')

filled = 0
for idx, row in pending.iterrows():
    for ev in completed:
        if ev['date_str'] != str(row['match_date'])[:10]:
            continue
        if not (names_match(row['home_team'], ev['home_team']) and
                names_match(row['away_team'], ev['away_team'])):
            continue

        total = ev['home_score'] + ev['away_score']
        side  = row['side']
        if side == 'OVER':
            won = total > 2.5
        elif side == 'UNDER':
            won = total <= 2.5
        else:
            break

        result = 'WIN' if won else 'LOSS'
        odds   = float(row['odds'])
        pnl    = round(odds - 1.0, 4) if won else -1.0

        ledger.at[idx, 'result'] = result
        ledger.at[idx, 'pnl']    = str(pnl)
        print(f'  [OK] {row["home_team"]} vs {row["away_team"]} '
              f'({row["match_date"]}) {side} @ {odds} '
              f'Goals={total} → {result} PnL={pnl:+.3f}u')
        filled += 1
        break

if filled > 0:
    ledger.to_csv(LEDGER_FILE, index=False)
    print(f'\nLedger updated: {filled} result(s) filled.')
    wins   = sum(1 for u in range(filled))  # simplified
    total_pnl = ledger.loc[ledger['result'].notna() & ledger['pnl'].notna(),
                            'pnl'].astype(float).tail(filled).sum()
else:
    print('\nNo matches found in OddsAPI results to update.')
