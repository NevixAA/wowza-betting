# Wowza Roadmap — Future Tasks

## Option 1: More Markets (same leagues, zero extra API cost) ← NEXT
Add 1X2, Asian Handicap, BTTS predictions alongside O/U 2.5.
Same 5 standard leagues, 3-4x more signals per game.
- Train 1X2 model (home/draw/away) using existing features
- Add AH (-0.5, -1.0, +0.5) from Poisson matrix (already computed in agent page)
- Add BTTS model (already computed in agent page)
- Update bets.csv / notifier / dashboard for new market types

## Option 2: More Leagues (needs OddsAPI upgrade)
Scottish League 1 (7.2% bookmaker margin) and League 2 (7.5%) — highest margins found.
Also: Brazil Serie B, Superettan (Sweden Div 2).
- Requires OddsAPI plan upgrade to cover Scottish leagues
- Historical data available on football-data.co.uk (SC1, SC2 codes)
- Full data: shots, corners, HT scores — standard format

## Option 3: Cross-Market Confirmation SNIPER
Instead of lowering threshold, require 3-way agreement:
  O/U signal (ML model) + Sharp Money (odds drift) + Formula model (Poisson/Dixon-Coles)
Fewer bets but maximum confidence.

## Option 4: Better Features via API-Football (~$15/month)
- First-half shot data (not available free) → better HT Poisson λ
- Live shot data for live scanner PRESSURE_COOKER signal
- xG data → more accurate goal expectation model
- Player availability / lineup data
