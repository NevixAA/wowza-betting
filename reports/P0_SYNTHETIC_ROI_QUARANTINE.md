# P0 item 6 - Synthetic props ROI quarantine (2026-11-22)

Repo: NevixAA/wowza-betting. Branch `p0/props-cert-synthetic-roi`, based on origin/main `2a86ab87a`.
The consumer trace covered origin/main of wowza-betting `2a86ab87a`, wowzaV9-Pro `eacaeec28`, wowza_v11
`c0d2961ea` and wowza-dashboard `e2e96f0a8`.

## Root cause

`output/player_props_calibration.json` (commit 9e4251705, 2026-06-24) holds per-market ROI computed against
synthetic implied odds of about 1 / base rate, not against a bookmaker quote: Goals +140.36%, Assists
+133.06%, SOT2+ +128.91%, SOT3+ +263.51%, Goals2 +9656% (n = 1). Finding M9 in
`reports/MODEL_DATA_AUDIT_2026-08-10.md`. The file had no marker, so any future reader could have shown
these figures as performance.

## Consumer trace (all 4 public repos, origin/main)

| Consumer class | Reads the file? |
|---|---|
| Dashboard pages / dashboard_data (v9 and wowza-dashboard) | no |
| Telegram (telegram_bot, test_telegram.py) | no (test_telegram copies only root-level `*.json`, not output/) |
| Promotion gates (props_gate, promotion_gate, retrain.py) | no |
| Staking / bankroll (stake_plan, staked, bankroll_policy) | no |
| Edge validation (market_edge, calibrated_edge, edge_engine) | no |
| predict.enrich_no_odds_markets (the old synthetic assists price) | no longer: removed 2026-11-01/04; tests/test_props_no_synthetic_price.py |
| wowzaV9-Pro | no. `src/studies/props_calibration.py` is a different study on the real settled player_ledger and real market odds. |
| wowza_v11 | no |
| Workflows (.yml), all repos | no writer, no reader |
| wowza-dashboard tree | the file is not in that repo |

No live consumer exists today. The quarantine prevents a future one.

## Fix

- **In-file marker.** The file keeps all its audit content and gains
  `"_quarantine": {"status": "INVALID_FOR_REAL_BETTING_ROI", "usable_for": ["audit"], "refused_for": [...]}`.
- **Loader refusal.** `src/integrity/synthetic_roi.py` is the only reader. `load_for_audit` requires the
  marker. `load_quarantined` / `assert_not_quarantined` raise `SyntheticROIError` for betting_performance,
  roi, staking, bankroll, promotion, edge_validation, dashboard_profitability, dashboard_ranking, telegram
  and pro_research_roi.
- **Real ROI contract.** `assert_real_roi_rows` / `real_roi_problems` refuse any row that:
  - misses real entry odds (> 1), the bookmaker, the capture time, the market, the line, or a settlement;
  - was captured at or after kickoff;
  - carries a synthetic label (`SYNTHETIC_PRICE_DIAGNOSTIC_ONLY`, `INVALID_FOR_PROFITABILITY`,
    `INVALID_FOR_REAL_BETTING_ROI`, `price_synthetic`).

  The quarantined calibration rows can never pass it.

## Tests

- `tests/test_no_synthetic_roi.py`:
  - the marker is present and the audit content is kept;
  - every refused purpose raises;
  - the audit read needs the marker;
  - no code file (py / yml / sh, outside tests / reports / docs / _archive) names the file;
  - no dashboard or Telegram module reads props-calibration ROI keys.
- `tests/test_no_synthetic_prices_in_real_roi.py`:
  - a complete real row passes;
  - each of the 6 fields missing or empty is refused;
  - unsettled rows, post-kickoff captures and odds <= 1 are refused;
  - every synthetic label is refused, including a `label_side_prices` fill constant;
  - the calibration rows are refused;
  - one bad row refuses the whole frame.
- 30 passed locally.

## Files changed

`output/player_props_calibration.json` (marker added, values untouched), `src/integrity/synthetic_roi.py` (new),
the two tests.

## Before / after

| | Before | After |
|---|---|---|
| Synthetic ROI file | unlabeled, any code could read it | INVALID_FOR_REAL_BETTING_ROI marker; the loader refuses every real purpose |
| Real-ROI price requirements | only `assert_profitability_inputs` (synthetic or mis-mapped label) | plus required entry odds, bookmaker, capture time, market, line and settlement |

## Blocker / owner decision

`output/player_ledger.csv` (the props results ledger) has no bookmaker, capture-time or line column. Under the
new contract, none of its 5,603 priced settled rows qualifies as a real ROI row today. The contract is not
wired into the existing props P/L views yet: `dashboard_data/props_view`, `trustworthy_performance` view B
(already labelled research/paper) and the Telegram props results. Wiring it would hide props P/L until the
ledger records those fields. That is a visible change that overlaps PR #70 (props TOP-3 / stake by proof),
so it is left for the owner. Recommended next step: record bookmaker, quote time and line on each props tip
(the TOP-3 work), then gate the props P/L views with `assert_real_roi_rows`.

Production impact: none. No stake, bankroll, promotion, ledger row or historical record changed.
Owner approval: not needed for this PR. It is needed to wire the contract into the live props P/L views.
