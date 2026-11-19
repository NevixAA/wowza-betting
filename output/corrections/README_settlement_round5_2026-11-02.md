# Settlement round 5 (settle5), 2026-11-02

Owner rules applied: props and goal markets settle on 90 minutes (regular time plus stoppage; extra
time ignored). A rescheduled match voids if it is not played within 48h of the original kick-off.
Corrections go beside the original, and no committed ledger row is rewritten.

## 1. Extra time in the team ledgers (round-4 P1-2e)
- `src/api_football_ou.fetch_league_fixture_results(..., statuses="FT-AET-PEN")`: AET and PEN
  fixtures carry their **90-minute score** (`score.fulltime`). If that score is missing, the
  fixture is skipped. It is never filled from the 120-minute `goals`. The training loader's
  default call (`statuses="FT"`) returns the same frame as before.
- Knockout guard (`update_results._knockout_guard`): before a football-data or OddsAPI score
  settles an O/U, side-market, sharp, live-signal or shadow row, the league-season's AET/PEN list
  is read from API-Football. If the fixture is on that list, the row settles on the 90-minute
  score. If the list cannot be fetched while a key is set, the row stays pending
  (`knockout_guard_unavailable`). This costs 1 call per (league, season) per run.
- Already-settled rows are not rewritten. Run `python update_results.py --restate-extra-time`
  in CI to write `team_ledgers_extra_time_restated.csv` (original and restated result/P&L side
  by side).

## 2. Rescheduled more than 48h, then played
API-Football moves a rescheduled fixture to its new date. Once the 48h window (D-1..D+2) has
closed, a same-pair lookup runs over the league-season fixtures (clubs resolved through
`src.team_names`, same home/away):
- exactly one fixture, after D+2: **VOID `RESCHEDULED`**;
- exactly one, inside the window: its status decides;
- two or more: `reschedule_ambiguous`, stays pending;
- only an earlier meeting: `reschedule_not_this_match`, stays pending.

## 3. The 111 restored player rows
`python update_results.py --grade-restored` grades
`player_ledger_restored_2026-11.csv` into new `graded_*` columns, using the same per-row rules as
the ledger (`_grade_player_rows`). The first-sight `result`/`pnl`/`notes` columns are unchanged,
and the rows are never appended to `player_ledger.csv`.

## 4. One card rule
`player_model/clv_tracker.py` grades `cards` with `update_results.card_market_result`. A yellow
card with an unknown red now settles WIN, as it does in the ledger. A row with no yellow and an
unknown red stays pending.
