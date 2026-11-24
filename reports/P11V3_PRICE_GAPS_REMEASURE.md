# P11 v3: bars re-measured on the enlarged priced universes (PROPOSAL, awaiting owner approval)

Owner 2026-11-23: HT prices = "Wire in + re-measure (Recommended)"; fallback books = "Extend to all markets (Recommended)" (owner 2026-11-23). Serving is unchanged: TP-2026-11-18-1ddfc9f1 (P11 v2) stays served and `config/top3_tiers_p11.json` is untouched. Counts and statistics only; no prices, keys or club names.

## What changed (price inputs only; the P11 v2 bars protocol is unchanged)

- Protocol `p11-bars-tiers-clean/3` (`config/p11_bars_tiers_protocol_v3.json`, sha256 LF `3578814439cbece7`), committed before the re-measure.
- BET tiers BT, BTU, OUU **+ HTB** (first-half totals at the bet time). `ht_over05` / `ht_over15` take only the first-half 0.5 / 1.5 line.
- The owner's O1 fallback order (Bet365 > Pinnacle > William Hill > 1xBet > Ladbrokes) on BTTS, O1.5, O3.5, HT 0.5 / 1.5 (both tracks) and New-Format O/U 2.5. Standard O/U 2.5 keeps football-data / Bet365 > Pinnacle.
- History pairs stamped exactly 00:00:00 UTC are dropped (counted). Own 00:00 backfill stays stripped.
- Predictions (models served now): Standard: Standard predictions = D1 serving models: harness on main (train_start standard 2016-07-01; HT market_exclude home_has_ht_history) on the SAVED D1 frame 57382c902c57328b (golive/d1_1123/feat_new.parquet; hash recomputed and must match) - the frame the merged D1 payloads were refitted on. D1 payloads (sha256/16): standard 72b38fd71e3e485e, btts e6a51f2e80f65647, over15 44d014a13a79ba60, over35 540aa360b2310b5c, ht_over05 8ee194e4827eb8a0, ht_over15 b886cd05ffec43b6 - verified against main after PR #96 merged.
- New-Format: unchanged from v2/v3: harness code 694995f90 on frame 9ed237ebc04aee09 (serving New-Format payload 303c9a05 and its byte-identical sub-models).
- Addendum `config/p11_bars_tiers_protocol_v3_addendum_d1.json` (sha256 LF `7876526da7ee4709`), committed before this re-run's fit.
- **Not used:** the pending Standard refit D1 (another agent's draft PR, awaiting owner approval).
- Store: wowza-exec `odds-history` @ `2ea84459`. No credits spent.

## Result in brief

- Old (P11 v2) cell statuses: {'PROMISING': 8, 'UNPROVEN': 40, 'OFF': 146}.
- New (v3) cell statuses: {'PROMISING': 9, 'UNPROVEN': 48, 'OFF': 143}.
- New proposal: `TP-2026-11-24-7e5f3889` (hash `7e5f38899ab1`), 57 entries, 55 would change serving IF approved.
- PROVEN still needs >= 150 validation bets + Holm + CI lower bound > 0. The validation window was scored before (P9, P11 v1, v2): these are re-validation numbers, never proof (only live tips prove a cell).
- PROVEN cells: old 0, new 0.
- **Read the Standard differences with care.** Old vs new on the Standard track moves two things at once: the price inputs AND the predictions (P11 v2 used the refit_1118b frame `9ed237eb`, 2021/22 start; the models served now train from 2016/17 (see Predictions), so their out-of-sample universe has 244,988 rows). New-Format predictions are identical, so New-Format differences come from the prices only.
- Tier parameters (T1/T2/T3) were re-fitted on the fit window only (`tier_fit.json`); the tier validation was NOT re-run. The proposed tier config is `output/threshold_proposals/bars_p11v3/top3_tiers_p11v3_PROPOSED.json`; the served `config/top3_tiers_p11.json` is unchanged.

- EFL status changes (old -> new): Championship ht_over15_over: OFF -> UNPROVEN; Championship over15_over: OFF -> UNPROVEN; Championship over25_over: OFF -> UNPROVEN; League One ht_over15_over: OFF -> UNPROVEN; League Two btts_yes: UNPROVEN -> OFF; League Two over15_over: OFF -> UNPROVEN; League Two over25_over: OFF -> UNPROVEN; National League btts_yes: OFF -> UNPROVEN.

- standard: price source before {'oddsapi_hist': 33287, 'own': 29512}, after {'oddsapi_hist': 40462, 'own': 29512}; integrity failures after: 0; 00:00-stamped history pairs dropped: 0.
- new_format: price source before {'oddsapi_hist': 24465, 'own': 3061}, after {'oddsapi_hist': 31598, 'own': 3061}; integrity failures after: 0; 00:00-stamped history pairs dropped: 0.

## Old (P11 v2) vs new (v3) per cell - EFL first

Bar = VALUABLE bar. WF = walk-forward bets at the bar. Study = validation (2026-09-01..11-08) bets, ROI [matchday-block 95% CI], CLV.

| Track | League | Cell | Old status | Old bar | Old WF | Old study bets | Old ROI [CI] | Old CLV | New status | New bar | New WF | New study bets | New ROI [CI] | New CLV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard | Championship | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Championship | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Championship | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 265 | 79 | -12.5% [-41.2%, +16.0%] | -0.39% (n 54) |
| standard | Championship | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.08 | 42 | 0 | - | - |
| standard | Championship | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.16 | 144 | 0 | - | - |
| standard | Championship | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Championship | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | League One | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | League One | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | League One | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 469 | 82 | -35.0% [-59.3%, -10.8%] | -0.32% (n 45) |
| standard | League One | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | League One | over25_over | UNPROVEN | 0.11 | 97 | 1 | -100.0% - | +2.38% (n 1) | UNPROVEN | 0.16 | 84 | 0 | - | - |
| standard | League One | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | League One | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | League Two | btts_yes | UNPROVEN | 0.02 | 4 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | League Two | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | League Two | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | League Two | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.03 | 48 | 1 | +40.0% - | +0.00% (n 1) |
| standard | League Two | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.17 | 66 | 1 | -100.0% - | -4.55% (n 1) |
| standard | League Two | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | League Two | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | National League | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 2 | 0 | - | - |
| standard | National League | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | National League | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | National League | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | National League | over25_over | OFF (OFF_NO_VALUE) | - | 45 | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | National League | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | National League | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Argentina Primera Division | btts_yes | PROMISING | 0.11 | 456 | 20 | +29.8% [-23.2%, +82.7%] | +0.46% (n 20) | PROMISING | 0.11 | 456 | 20 | +29.8% [-23.2%, +82.7%] | +0.46% (n 20) |
| new_format | Argentina Primera Division | over15_over | PROMISING | 0.05 | 375 | 37 | +1.9% [-25.4%, +29.2%] | +0.49% (n 37) | OFF (OFF_NO_OWN_ESTIMATE) | - | 185 | 11 | +16.8% [-34.8%, +68.5%] | +1.20% (n 11) |
| new_format | Argentina Primera Division | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Argentina Primera Division | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 83 | 68 | -21.3% [-44.8%, +1.6%] | -1.42% (n 68) |
| new_format | Argentina Primera Division | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Austrian Bundesliga | btts_yes | PROMISING | 0.02 | 53 | 3 | +18.3% [-100.0%, +134.3%] | +2.20% (n 3) | PROMISING | 0.02 | 53 | 3 | +18.3% [-100.0%, +134.3%] | +2.20% (n 3) |
| new_format | Austrian Bundesliga | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Austrian Bundesliga | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Austrian Bundesliga | over25_under | UNPROVEN | 0.02 | 87 | 29 | -33.0% [-83.9%, +17.9%] | +0.88% (n 29) | UNPROVEN | 0.02 | 364 | 36 | -29.5% [-66.5%, +7.5%] | +0.19% (n 36) |
| new_format | Austrian Bundesliga | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Brazil Serie A | btts_yes | UNPROVEN | 0.02 | 501 | 21 | +26.5% [-17.1%, +70.1%] | -0.45% (n 21) | UNPROVEN | 0.02 | 501 | 21 | +26.5% [-17.1%, +70.1%] | -0.45% (n 21) |
| new_format | Brazil Serie A | ht_over05_over | - | - | - | - | - | - | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - |
| new_format | Brazil Serie A | ht_over15_over | - | - | - | - | - | - | PROMISING | 0.02 | 118 | 21 | +13.6% [-50.6%, +77.8%] | - |
| new_format | Brazil Serie A | over15_over | UNPROVEN | 0.02 | 193 | 6 | +17.3% [-28.7%, +63.3%] | -0.80% (n 6) | UNPROVEN | 0.02 | 136 | 5 | -43.2% [-111.4%, +25.0%] | -1.48% (n 5) |
| new_format | Brazil Serie A | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.10 | 93 | 3 | -17.3% [-179.4%, +148.0%] | -5.51% (n 3) |
| new_format | Brazil Serie A | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Brazil Serie A | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | China Super League | btts_yes | UNPROVEN | 0.11 | 7 | 0 | - | - | UNPROVEN | 0.11 | 7 | 0 | - | - |
| new_format | China Super League | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | China Super League | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | China Super League | over25_under | UNPROVEN | 0.02 | 13 | 0 | - | - | UNPROVEN | 0.02 | 214 | 20 | +22.5% [-33.4%, +78.4%] | -2.71% (n 20) |
| new_format | China Super League | over35_over | UNPROVEN | 0.02 | 215 | 24 | -16.9% [-59.5%, +25.8%] | +1.38% (n 24) | UNPROVEN | 0.03 | 57 | 0 | - | - |
| new_format | Denmark Superliga | btts_yes | UNPROVEN | 0.02 | 28 | 0 | - | - | UNPROVEN | 0.02 | 28 | 0 | - | - |
| new_format | Denmark Superliga | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 221 | 0 | - | - |
| new_format | Denmark Superliga | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Denmark Superliga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Denmark Superliga | over35_over | UNPROVEN | 0.02 | 366 | 33 | -3.8% [-57.2%, +49.7%] | +0.04% (n 33) | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Finland Veikkausliiga | btts_yes | UNPROVEN | 0.02 | 43 | 0 | - | - | UNPROVEN | 0.02 | 43 | 0 | - | - |
| new_format | Finland Veikkausliiga | over15_over | UNPROVEN | 0.02 | 7 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Finland Veikkausliiga | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 91 | 3 | -36.7% [-160.8%, +90.0%] | +0.90% (n 3) |
| new_format | Finland Veikkausliiga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Finland Veikkausliiga | over35_over | UNPROVEN | 0.02 | 144 | 37 | -37.0% [-77.5%, +3.5%] | +0.19% (n 37) | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Ireland Premier Division | btts_yes | UNPROVEN | 0.02 | 80 | 4 | -1.2% [-146.4%, +143.9%] | +2.08% (n 4) | UNPROVEN | 0.02 | 80 | 4 | -1.2% [-146.4%, +143.9%] | +2.08% (n 4) |
| new_format | Ireland Premier Division | over15_over | UNPROVEN | 0.02 | 27 | 0 | - | - | UNPROVEN | 0.02 | 257 | 0 | - | - |
| new_format | Ireland Premier Division | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Ireland Premier Division | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Ireland Premier Division | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Japan J-League | btts_yes | UNPROVEN | 0.02 | 125 | 0 | - | - | UNPROVEN | 0.02 | 125 | 0 | - | - |
| new_format | Japan J-League | over15_over | UNPROVEN | 0.02 | 84 | 2 | +40.5% [+39.5%, +41.5%] | -2.72% (n 2) | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Japan J-League | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NEGATIVE) | - | 61 | 2 | -100.0% [-100.0%, -100.0%] | -1.51% (n 2) |
| new_format | Japan J-League | over25_under | PROMISING | 0.02 | 146 | 21 | +1.9% [-49.9%, +52.8%] | +2.09% (n 21) | UNPROVEN | 0.02 | 363 | 28 | -3.8% [-43.0%, +35.5%] | +2.60% (n 28) |
| new_format | Japan J-League | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | K-League 1 | btts_yes | UNPROVEN | 0.02 | 31 | 0 | - | - | UNPROVEN | 0.02 | 31 | 0 | - | - |
| new_format | K-League 1 | over15_over | UNPROVEN | 0.10 | 5 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | K-League 1 | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | K-League 1 | over25_under | UNPROVEN | 0.02 | 36 | 0 | - | - | UNPROVEN | 0.02 | 328 | 0 | - | - |
| new_format | K-League 1 | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Mexico Liga MX | btts_yes | UNPROVEN | 0.11 | 4 | 0 | - | - | UNPROVEN | 0.11 | 4 | 0 | - | - |
| new_format | Mexico Liga MX | ht_over05_over | - | - | - | - | - | - | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - |
| new_format | Mexico Liga MX | ht_over15_over | - | - | - | - | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| new_format | Mexico Liga MX | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Mexico Liga MX | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Mexico Liga MX | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 11 | 0 | - | - |
| new_format | Mexico Liga MX | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 21 | 0 | - | - |
| new_format | Norway Eliteserien | btts_yes | UNPROVEN | 0.02 | 40 | 0 | - | - | UNPROVEN | 0.02 | 40 | 0 | - | - |
| new_format | Norway Eliteserien | over15_over | UNPROVEN | 0.02 | 3 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Norway Eliteserien | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 121 | 8 | +22.8% [-30.3%, +75.8%] | -1.69% (n 8) |
| new_format | Norway Eliteserien | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Norway Eliteserien | over35_over | PROMISING | 0.02 | 465 | 52 | +0.2% [-29.6%, +29.6%] | +0.18% (n 52) | UNPROVEN | 0.03 | 232 | 18 | +22.0% [-41.2%, +81.3%] | -0.57% (n 18) |
| new_format | Romanian Superliga | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Romanian Superliga | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Romanian Superliga | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Romanian Superliga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Romanian Superliga | over35_over | UNPROVEN | 0.02 | 35 | 0 | - | - | UNPROVEN | 0.02 | 35 | 0 | - | - |
| new_format | Saudi Pro League | btts_yes | UNPROVEN | 0.11 | 4 | 0 | - | - | UNPROVEN | 0.11 | 4 | 0 | - | - |
| new_format | Saudi Pro League | over15_over | OFF (OFF_THIN) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Saudi Pro League | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.10 | 2 | 0 | - | - |
| new_format | Saudi Pro League | over25_under | UNPROVEN | 0.02 | 47 | 6 | +28.3% [-100.0%, +143.3%] | -2.52% (n 6) | UNPROVEN | 0.02 | 55 | 6 | +28.3% [-100.0%, +143.3%] | -2.52% (n 6) |
| new_format | Saudi Pro League | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Sweden Allsvenskan | btts_yes | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Sweden Allsvenskan | over15_over | UNPROVEN | 0.02 | 9 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | Sweden Allsvenskan | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| new_format | Sweden Allsvenskan | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 42 | 0 | - | - |
| new_format | Sweden Allsvenskan | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | USA MLS | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | USA MLS | ht_over05_over | - | - | - | - | - | - | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - |
| new_format | USA MLS | ht_over15_over | - | - | - | - | - | - | UNPROVEN | 0.02 | 490 | 48 | -17.7% [-51.7%, +16.6%] | - |
| new_format | USA MLS | over15_over | UNPROVEN | 0.06 | 2 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | USA MLS | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | PROMISING | 0.02 | 227 | 6 | +7.3% [-80.7%, +95.3%] | +0.89% (n 6) |
| new_format | USA MLS | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| new_format | USA MLS | over35_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Belgian First Division A | btts_yes | UNPROVEN | 0.02 | 5 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Belgian First Division A | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Belgian First Division A | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Belgian First Division A | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Belgian First Division A | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Belgian First Division A | over25_under | UNPROVEN | 0.02 | 109 | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Belgian First Division A | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Bundesliga 2 | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Bundesliga 2 | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Bundesliga 2 | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Bundesliga 2 | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Bundesliga 2 | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Bundesliga 2 | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Bundesliga 2 | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Dutch Eredivisie | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Dutch Eredivisie | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_POSITIVE_BAR) | - | - | 0 | - | - |
| standard | Dutch Eredivisie | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Dutch Eredivisie | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Dutch Eredivisie | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | PROMISING | 0.11 | 46 | 1 | +155.0% - | +62.42% (n 1) |
| standard | Dutch Eredivisie | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Dutch Eredivisie | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.02 | 386 | 0 | - | - |
| standard | Greek Super League | btts_yes | UNPROVEN | 0.02 | 47 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Greek Super League | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Greek Super League | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Greek Super League | over15_over | UNPROVEN | 0.02 | 5 | 0 | - | - | UNPROVEN | 0.02 | 101 | 0 | - | - |
| standard | Greek Super League | over25_over | UNPROVEN | 0.10 | 124 | 1 | +125.0% - | -6.25% (n 1) | UNPROVEN | 0.11 | 130 | 2 | +12.5% [-208.0%, +233.0%] | -0.95% (n 2) |
| standard | Greek Super League | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Greek Super League | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | La Liga 2 | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | La Liga 2 | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | La Liga 2 | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | La Liga 2 | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.12 | 92 | 2 | +55.5% [+52.6%, +58.4%] | +0.00% (n 2) |
| standard | La Liga 2 | over25_over | PROMISING | 0.13 | 384 | 5 | +60.0% [-68.1%, +188.1%] | +4.24% (n 5) | PROMISING | 0.17 | 416 | 6 | +33.3% [-83.6%, +150.2%] | +4.37% (n 6) |
| standard | La Liga 2 | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | La Liga 2 | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Ligue 2 | btts_yes | UNPROVEN | 0.02 | 4 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Ligue 2 | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Ligue 2 | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Ligue 2 | over15_over | UNPROVEN | 0.02 | 32 | 0 | - | - | UNPROVEN | 0.04 | 135 | 2 | -28.0% [-169.1%, +113.1%] | +4.38% (n 2) |
| standard | Ligue 2 | over25_over | UNPROVEN | 0.14 | 69 | 0 | - | - | UNPROVEN | 0.16 | 124 | 1 | -100.0% - | -3.70% (n 1) |
| standard | Ligue 2 | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Ligue 2 | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Portuguese Primeira Liga | btts_yes | UNPROVEN | 0.02 | 6 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Portuguese Primeira Liga | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Portuguese Primeira Liga | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Portuguese Primeira Liga | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Portuguese Primeira Liga | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | PROMISING | 0.16 | 88 | 4 | +3.2% [-114.5%, +125.0%] | +6.04% (n 4) |
| standard | Portuguese Primeira Liga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Portuguese Primeira Liga | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Scottish Championship | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish Championship | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish Championship | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish Championship | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish Championship | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | UNPROVEN | 0.12 | 66 | 1 | -100.0% - | -4.26% (n 1) |
| standard | Scottish Championship | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Scottish Championship | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish League One | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish League One | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish League One | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish League One | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Scottish League One | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Scottish League One | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Scottish League One | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish League Two | btts_yes | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish League Two | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish League Two | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish League Two | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Scottish League Two | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Scottish League Two | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Scottish League Two | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish Premiership | btts_yes | UNPROVEN | 0.02 | 7 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Scottish Premiership | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish Premiership | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Scottish Premiership | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.09 | 9 | 0 | - | - |
| standard | Scottish Premiership | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Scottish Premiership | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Scottish Premiership | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Serie B | btts_yes | PROMISING | 0.02 | 305 | 8 | +23.8% [-58.8%, +106.3%] | +0.83% (n 8) | UNPROVEN | 0.02 | 846 | 56 | -8.5% [-42.1%, +25.1%] | +1.16% (n 56) |
| standard | Serie B | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Serie B | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (OWN_DATA_TOO_THIN) | - | - | 0 | - | - |
| standard | Serie B | over15_over | PROMISING | 0.02 | 176 | 3 | +39.3% [+35.0%, +43.7%] | +3.81% (n 3) | PROMISING | 0.02 | 297 | 3 | +42.0% [+39.4%, +44.6%] | +1.05% (n 3) |
| standard | Serie B | over25_over | UNPROVEN | 0.09 | 377 | 7 | -3.6% [-103.4%, +96.2%] | +8.71% (n 7) | PROMISING | 0.11 | 316 | 7 | +29.3% [-65.7%, +119.2%] | +7.80% (n 7) |
| standard | Serie B | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Serie B | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Turkish Super Lig | btts_yes | UNPROVEN | 0.02 | 24 | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |
| standard | Turkish Super Lig | ht_over05_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Turkish Super Lig | ht_over15_over | OFF (INSUFFICIENT_SELECTION_EVIDENCE) | - | - | 0 | - | - | OFF (NO_SELECT_ROWS) | - | - | 0 | - | - |
| standard | Turkish Super Lig | over15_over | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | UNPROVEN | 0.08 | 2 | 0 | - | - |
| standard | Turkish Super Lig | over25_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_VALUE) | - | - | 0 | - | - |
| standard | Turkish Super Lig | over25_under | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - | OFF (OFF_THIN) | - | - | 0 | - | - |
| standard | Turkish Super Lig | over35_over | OFF (OFF_NO_VALUE) | - | - | 0 | - | - | OFF (OFF_NO_OWN_ESTIMATE) | - | - | 0 | - | - |

## Coverage before / after per league x market (fit + validation rows with a result)

Before = v2 price rules (what the served bars were fitted on, on the same prediction rows). After = v3 rules.

| Track | League | Market | Rows | Priced before (v2 rules) | Priced after (v3 rules) | Added | After: Pinnacle / WH / 1xBet / Ladbrokes (history rows) |
|---|---|---|---|---|---|---|---|
| standard | Championship | btts | 3,787 | 1,602 | 1,602 | +0 | 0 / 1113 / 394 / 1 |
| standard | Championship | ht_over05 | 3,773 | 73 | 537 | +464 | 464 / 0 / 0 / 0 |
| standard | Championship | ht_over15 | 3,773 | 73 | 564 | +491 | 491 / 0 / 0 / 0 |
| standard | Championship | over15 | 3,787 | 1,098 | 1,551 | +453 | 998 / 269 / 0 / 184 |
| standard | Championship | over25 | 3,787 | 2,384 | 2,384 | +0 | 7 / 0 / 0 / 0 |
| standard | Championship | over35 | 3,787 | 1,104 | 1,462 | +358 | 990 / 0 / 0 / 358 |
| standard | League One | btts | 3,760 | 1,413 | 1,413 | +0 | 0 / 1065 / 273 / 0 |
| standard | League One | ht_over05 | 3,746 | 61 | 497 | +436 | 436 / 0 / 0 / 0 |
| standard | League One | ht_over15 | 3,746 | 58 | 551 | +493 | 493 / 0 / 0 / 0 |
| standard | League One | over15 | 3,760 | 838 | 1,230 | +392 | 747 / 342 / 0 / 50 |
| standard | League One | over25 | 3,760 | 2,359 | 2,359 | +0 | 9 / 0 / 0 / 0 |
| standard | League One | over35 | 3,760 | 839 | 1,047 | +208 | 744 / 0 / 0 / 208 |
| standard | League Two | btts | 3,769 | 1,534 | 1,534 | +0 | 0 / 1150 / 304 / 0 |
| standard | League Two | ht_over05 | 3,756 | 59 | 599 | +540 | 540 / 0 / 0 / 0 |
| standard | League Two | ht_over15 | 3,756 | 57 | 655 | +598 | 598 / 0 / 0 / 0 |
| standard | League Two | over15 | 3,769 | 968 | 1,354 | +386 | 879 / 318 / 0 / 68 |
| standard | League Two | over25 | 3,769 | 2,373 | 2,373 | +0 | 10 / 0 / 0 / 0 |
| standard | League Two | over35 | 3,769 | 970 | 1,171 | +201 | 875 / 0 / 0 / 201 |
| standard | National League | btts | 3,723 | 70 | 70 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | ht_over05 | 3,723 | 42 | 42 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | ht_over15 | 3,723 | 43 | 43 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | over15 | 3,723 | 59 | 59 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | over25 | 3,723 | 2,370 | 2,370 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | over35 | 3,723 | 69 | 69 | +0 | 0 / 0 / 0 / 0 |
| new_format | Argentina Primera Division | btts | 2,642 | 1,064 | 1,064 | +0 | 0 / 0 / 919 / 42 |
| new_format | Argentina Primera Division | ht_over05 | 2,626 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Argentina Primera Division | ht_over15 | 2,626 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Argentina Primera Division | over15 | 2,642 | 642 | 970 | +328 | 532 / 0 / 0 / 328 |
| new_format | Argentina Primera Division | over25 | 2,642 | 644 | 967 | +323 | 520 / 0 / 0 / 323 |
| new_format | Argentina Primera Division | over35 | 2,642 | 607 | 962 | +355 | 502 / 0 / 0 / 355 |
| new_format | Austrian Bundesliga | btts | 1,038 | 571 | 571 | +0 | 0 / 397 / 151 / 0 |
| new_format | Austrian Bundesliga | ht_over05 | 1,032 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Austrian Bundesliga | ht_over15 | 1,032 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Austrian Bundesliga | over15 | 1,038 | 449 | 492 | +43 | 420 / 43 / 0 / 0 |
| new_format | Austrian Bundesliga | over25 | 1,038 | 450 | 492 | +42 | 416 / 42 / 0 / 0 |
| new_format | Austrian Bundesliga | over35 | 1,038 | 454 | 454 | +0 | 420 / 0 / 0 / 0 |
| new_format | Brazil Serie A | btts | 2,115 | 1,113 | 1,113 | +0 | 0 / 705 / 290 / 40 |
| new_format | Brazil Serie A | ht_over05 | 2,114 | 0 | 370 | +370 | 370 / 0 / 0 / 0 |
| new_format | Brazil Serie A | ht_over15 | 2,114 | 0 | 278 | +278 | 278 / 0 / 0 / 0 |
| new_format | Brazil Serie A | over15 | 2,115 | 588 | 1,071 | +483 | 515 / 257 / 0 / 226 |
| new_format | Brazil Serie A | over25 | 2,115 | 562 | 1,069 | +507 | 480 / 282 / 0 / 225 |
| new_format | Brazil Serie A | over35 | 2,115 | 586 | 956 | +370 | 515 / 0 / 0 / 370 |
| new_format | China Super League | btts | 1,372 | 333 | 333 | +0 | 0 / 94 / 213 / 0 |
| new_format | China Super League | ht_over05 | 1,291 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | China Super League | ht_over15 | 1,291 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | China Super League | over15 | 1,372 | 296 | 299 | +3 | 263 / 3 / 0 / 0 |
| new_format | China Super League | over25 | 1,372 | 309 | 309 | +0 | 277 / 0 / 0 / 0 |
| new_format | China Super League | over35 | 1,372 | 316 | 316 | +0 | 279 / 0 / 0 / 0 |
| new_format | Denmark Superliga | btts | 1,038 | 535 | 535 | +0 | 0 / 360 / 145 / 0 |
| new_format | Denmark Superliga | ht_over05 | 1,031 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Denmark Superliga | ht_over15 | 1,031 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Denmark Superliga | over15 | 1,038 | 386 | 441 | +55 | 357 / 53 / 0 / 2 |
| new_format | Denmark Superliga | over25 | 1,038 | 371 | 442 | +71 | 340 / 69 / 0 / 2 |
| new_format | Denmark Superliga | over35 | 1,038 | 387 | 406 | +19 | 362 / 0 / 0 / 19 |
| new_format | Finland Veikkausliiga | btts | 951 | 362 | 362 | +0 | 0 / 198 / 134 / 0 |
| new_format | Finland Veikkausliiga | ht_over05 | 945 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Finland Veikkausliiga | ht_over15 | 945 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Finland Veikkausliiga | over15 | 951 | 295 | 335 | +40 | 260 / 39 / 0 / 1 |
| new_format | Finland Veikkausliiga | over25 | 951 | 276 | 341 | +65 | 236 / 64 / 0 / 1 |
| new_format | Finland Veikkausliiga | over35 | 951 | 300 | 316 | +16 | 262 / 0 / 0 / 16 |
| new_format | Ireland Premier Division | btts | 980 | 295 | 295 | +0 | 0 / 101 / 159 / 0 |
| new_format | Ireland Premier Division | ht_over05 | 978 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Ireland Premier Division | ht_over15 | 978 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Ireland Premier Division | over15 | 980 | 261 | 270 | +9 | 224 / 9 / 0 / 0 |
| new_format | Ireland Premier Division | over25 | 980 | 255 | 268 | +13 | 217 / 13 / 0 / 0 |
| new_format | Ireland Premier Division | over35 | 980 | 261 | 263 | +2 | 222 / 0 / 0 / 2 |
| new_format | Japan J-League | btts | 1,870 | 620 | 620 | +0 | 0 / 431 / 186 / 0 |
| new_format | Japan J-League | ht_over05 | 1,858 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Japan J-League | ht_over15 | 1,858 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Japan J-League | over15 | 1,870 | 550 | 593 | +43 | 543 / 42 / 0 / 1 |
| new_format | Japan J-League | over25 | 1,870 | 512 | 594 | +82 | 502 / 81 / 0 / 1 |
| new_format | Japan J-League | over35 | 1,870 | 550 | 564 | +14 | 543 / 0 / 0 / 14 |
| new_format | K-League 1 | btts | 1,188 | 382 | 382 | +0 | 0 / 322 / 51 / 0 |
| new_format | K-League 1 | ht_over05 | 1,188 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | K-League 1 | ht_over15 | 1,188 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | K-League 1 | over15 | 1,188 | 337 | 370 | +33 | 325 / 33 / 0 / 0 |
| new_format | K-League 1 | over25 | 1,188 | 314 | 368 | +54 | 300 / 54 / 0 / 0 |
| new_format | K-League 1 | over35 | 1,188 | 340 | 340 | +0 | 326 / 0 / 0 / 0 |
| new_format | Mexico Liga MX | btts | 1,830 | 597 | 597 | +0 | 0 / 420 / 111 / 3 |
| new_format | Mexico Liga MX | ht_over05 | 1,823 | 0 | 95 | +95 | 4 / 0 / 91 / 0 |
| new_format | Mexico Liga MX | ht_over15 | 1,823 | 0 | 96 | +96 | 15 / 0 / 81 / 0 |
| new_format | Mexico Liga MX | over15 | 1,830 | 447 | 657 | +210 | 370 / 137 / 71 / 2 |
| new_format | Mexico Liga MX | over25 | 1,830 | 453 | 663 | +210 | 368 / 137 / 71 / 2 |
| new_format | Mexico Liga MX | over35 | 1,830 | 451 | 604 | +153 | 372 / 0 / 145 / 8 |
| new_format | Norway Eliteserien | btts | 1,303 | 606 | 606 | +0 | 0 / 428 / 139 / 1 |
| new_format | Norway Eliteserien | ht_over05 | 1,295 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Norway Eliteserien | ht_over15 | 1,295 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Norway Eliteserien | over15 | 1,303 | 493 | 564 | +71 | 453 / 71 / 0 / 0 |
| new_format | Norway Eliteserien | over25 | 1,303 | 498 | 564 | +66 | 454 / 66 / 0 / 0 |
| new_format | Norway Eliteserien | over35 | 1,303 | 505 | 530 | +25 | 463 / 0 / 0 / 25 |
| new_format | Romanian Superliga | btts | 1,698 | 44 | 44 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | ht_over05 | 1,689 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | ht_over15 | 1,689 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | over15 | 1,698 | 53 | 53 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | over25 | 1,698 | 53 | 53 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | over35 | 1,698 | 50 | 50 | +0 | 0 / 0 / 0 / 0 |
| new_format | Saudi Pro League | btts | 1,420 | 46 | 46 | +0 | 0 / 0 / 27 / 0 |
| new_format | Saudi Pro League | ht_over05 | 1,420 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Saudi Pro League | ht_over15 | 1,420 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Saudi Pro League | over15 | 1,420 | 57 | 57 | +0 | 24 / 0 / 0 / 0 |
| new_format | Saudi Pro League | over25 | 1,420 | 59 | 59 | +0 | 26 / 0 / 0 / 0 |
| new_format | Saudi Pro League | over35 | 1,420 | 60 | 60 | +0 | 26 / 0 / 0 / 0 |
| new_format | Sweden Allsvenskan | btts | 1,335 | 610 | 610 | +0 | 0 / 382 / 183 / 0 |
| new_format | Sweden Allsvenskan | ht_over05 | 1,327 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Sweden Allsvenskan | ht_over15 | 1,327 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Sweden Allsvenskan | over15 | 1,335 | 418 | 515 | +97 | 369 / 92 / 0 / 5 |
| new_format | Sweden Allsvenskan | over25 | 1,335 | 403 | 520 | +117 | 350 / 112 / 0 / 5 |
| new_format | Sweden Allsvenskan | over35 | 1,335 | 420 | 456 | +36 | 373 / 0 / 0 / 36 |
| new_format | USA MLS | btts | 2,859 | 1,560 | 1,560 | +0 | 0 / 731 / 689 / 3 |
| new_format | USA MLS | ht_over05 | 2,844 | 0 | 390 | +390 | 141 / 0 / 249 / 0 |
| new_format | USA MLS | ht_over15 | 2,844 | 0 | 834 | +834 | 717 / 0 / 117 / 0 |
| new_format | USA MLS | over15 | 2,859 | 1,027 | 1,419 | +392 | 878 / 212 / 4 / 176 |
| new_format | USA MLS | over25 | 2,859 | 1,000 | 1,418 | +418 | 850 / 236 / 4 / 178 |
| new_format | USA MLS | over35 | 2,859 | 1,043 | 1,348 | +305 | 876 / 0 / 86 / 219 |
| standard | Belgian First Division A | btts | 2,025 | 878 | 878 | +0 | 0 / 670 / 162 / 5 |
| standard | Belgian First Division A | ht_over05 | 2,022 | 22 | 22 | +0 | 0 / 0 / 0 / 0 |
| standard | Belgian First Division A | ht_over15 | 2,022 | 27 | 27 | +0 | 0 / 0 / 0 / 0 |
| standard | Belgian First Division A | over15 | 2,025 | 827 | 879 | +52 | 783 / 26 / 0 / 26 |
| standard | Belgian First Division A | over25 | 2,025 | 1,331 | 1,331 | +0 | 0 / 0 / 0 / 0 |
| standard | Belgian First Division A | over35 | 2,025 | 830 | 873 | +43 | 783 / 0 / 0 / 43 |
| standard | Bundesliga 2 | btts | 2,072 | 951 | 951 | +0 | 0 / 827 / 83 / 0 |
| standard | Bundesliga 2 | ht_over05 | 2,069 | 21 | 21 | +0 | 0 / 0 / 0 / 0 |
| standard | Bundesliga 2 | ht_over15 | 2,069 | 30 | 30 | +0 | 0 / 0 / 0 / 0 |
| standard | Bundesliga 2 | over15 | 2,072 | 872 | 948 | +76 | 817 / 51 / 0 / 25 |
| standard | Bundesliga 2 | over25 | 2,072 | 1,316 | 1,316 | +0 | 0 / 0 / 0 / 0 |
| standard | Bundesliga 2 | over35 | 2,072 | 876 | 940 | +64 | 822 / 0 / 0 / 64 |
| standard | Dutch Eredivisie | btts | 2,156 | 1,017 | 1,017 | +0 | 0 / 883 / 76 / 4 |
| standard | Dutch Eredivisie | ht_over05 | 2,156 | 31 | 134 | +103 | 0 / 0 / 103 / 0 |
| standard | Dutch Eredivisie | ht_over15 | 2,156 | 35 | 134 | +99 | 0 / 0 / 99 / 0 |
| standard | Dutch Eredivisie | over15 | 2,156 | 858 | 1,008 | +150 | 804 / 107 / 6 / 37 |
| standard | Dutch Eredivisie | over25 | 2,156 | 1,330 | 1,330 | +0 | 0 / 0 / 0 / 0 |
| standard | Dutch Eredivisie | over35 | 2,156 | 912 | 996 | +84 | 855 / 0 / 23 / 61 |
| standard | Greek Super League | btts | 1,613 | 621 | 621 | +0 | 0 / 490 / 84 / 6 |
| standard | Greek Super League | ht_over05 | 1,608 | 32 | 32 | +0 | 0 / 0 / 0 / 0 |
| standard | Greek Super League | ht_over15 | 1,608 | 34 | 34 | +0 | 0 / 0 / 0 / 0 |
| standard | Greek Super League | over15 | 1,613 | 523 | 604 | +81 | 468 / 50 / 0 / 31 |
| standard | Greek Super League | over25 | 1,613 | 1,005 | 1,005 | +0 | 0 / 0 / 0 / 0 |
| standard | Greek Super League | over35 | 1,613 | 521 | 572 | +51 | 478 / 0 / 0 / 51 |
| standard | La Liga 2 | btts | 3,165 | 1,378 | 1,378 | +0 | 0 / 1092 / 183 / 19 |
| standard | La Liga 2 | ht_over05 | 3,127 | 59 | 59 | +0 | 0 / 0 / 0 / 0 |
| standard | La Liga 2 | ht_over15 | 3,127 | 63 | 63 | +0 | 0 / 0 / 0 / 0 |
| standard | La Liga 2 | over15 | 3,165 | 1,228 | 1,357 | +129 | 1136 / 68 / 0 / 61 |
| standard | La Liga 2 | over25 | 3,165 | 1,980 | 1,980 | +0 | 6 / 0 / 0 / 0 |
| standard | La Liga 2 | over35 | 3,165 | 1,228 | 1,346 | +118 | 1127 / 0 / 0 / 118 |
| standard | Ligue 2 | btts | 2,422 | 819 | 819 | +0 | 0 / 620 / 179 / 0 |
| standard | Ligue 2 | ht_over05 | 2,421 | 8 | 8 | +0 | 0 / 0 / 0 / 0 |
| standard | Ligue 2 | ht_over15 | 2,421 | 13 | 13 | +0 | 0 / 0 / 0 / 0 |
| standard | Ligue 2 | over15 | 2,422 | 795 | 850 | +55 | 768 / 55 / 0 / 0 |
| standard | Ligue 2 | over25 | 2,422 | 1,466 | 1,466 | +0 | 1 / 0 / 0 / 0 |
| standard | Ligue 2 | over35 | 2,422 | 795 | 828 | +33 | 768 / 0 / 0 / 33 |
| standard | Portuguese Primeira Liga | btts | 2,104 | 846 | 846 | +0 | 0 / 711 / 65 / 19 |
| standard | Portuguese Primeira Liga | ht_over05 | 2,104 | 29 | 29 | +0 | 0 / 0 / 0 / 0 |
| standard | Portuguese Primeira Liga | ht_over15 | 2,104 | 37 | 37 | +0 | 0 / 0 / 0 / 0 |
| standard | Portuguese Primeira Liga | over15 | 2,104 | 757 | 849 | +92 | 708 / 61 / 0 / 31 |
| standard | Portuguese Primeira Liga | over25 | 2,104 | 1,318 | 1,318 | +0 | 0 / 0 / 0 / 0 |
| standard | Portuguese Primeira Liga | over35 | 2,104 | 766 | 840 | +74 | 716 / 0 / 0 / 74 |
| standard | Scottish Championship | btts | 1,268 | 22 | 22 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | ht_over05 | 1,268 | 11 | 11 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | ht_over15 | 1,268 | 12 | 12 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | over15 | 1,268 | 22 | 22 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | over25 | 1,268 | 786 | 786 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | over35 | 1,268 | 28 | 28 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | btts | 1,262 | 12 | 12 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | ht_over05 | 1,262 | 10 | 10 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | ht_over15 | 1,262 | 14 | 14 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | over15 | 1,262 | 15 | 15 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | over25 | 1,262 | 779 | 779 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | over35 | 1,262 | 12 | 12 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | btts | 1,248 | 12 | 12 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | ht_over05 | 1,248 | 10 | 10 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | ht_over15 | 1,248 | 11 | 11 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | over15 | 1,248 | 10 | 10 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | over25 | 1,248 | 783 | 783 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | over35 | 1,248 | 13 | 13 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | btts | 1,578 | 698 | 698 | +0 | 0 / 563 / 100 / 0 |
| standard | Scottish Premiership | ht_over05 | 1,578 | 27 | 27 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | ht_over15 | 1,578 | 21 | 21 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | over15 | 1,578 | 486 | 682 | +196 | 461 / 131 / 0 / 65 |
| standard | Scottish Premiership | over25 | 1,578 | 986 | 986 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | over35 | 1,578 | 497 | 664 | +167 | 462 / 0 / 0 / 167 |
| standard | Serie B | btts | 2,630 | 1,011 | 1,011 | +0 | 0 / 759 / 177 / 7 |
| standard | Serie B | ht_over05 | 2,601 | 53 | 53 | +0 | 0 / 0 / 0 / 0 |
| standard | Serie B | ht_over15 | 2,601 | 61 | 61 | +0 | 0 / 0 / 0 / 0 |
| standard | Serie B | over15 | 2,630 | 856 | 1,001 | +145 | 774 / 60 / 0 / 85 |
| standard | Serie B | over25 | 2,630 | 1,621 | 1,621 | +0 | 5 / 0 / 0 / 0 |
| standard | Serie B | over35 | 2,630 | 859 | 998 | +139 | 779 / 0 / 0 / 139 |
| standard | Turkish Super Lig | btts | 2,299 | 948 | 948 | +0 | 0 / 0 / 864 / 38 |
| standard | Turkish Super Lig | ht_over05 | 2,270 | 26 | 26 | +0 | 0 / 0 / 0 / 0 |
| standard | Turkish Super Lig | ht_over15 | 2,270 | 27 | 27 | +0 | 0 / 0 / 0 / 0 |
| standard | Turkish Super Lig | over15 | 2,299 | 806 | 916 | +110 | 757 / 0 / 0 / 110 |
| standard | Turkish Super Lig | over25 | 2,299 | 1,429 | 1,429 | +0 | 0 / 0 / 0 / 0 |
| standard | Turkish Super Lig | over35 | 2,299 | 824 | 918 | +94 | 764 / 0 / 0 / 94 |

## Reproduce

1. Standard predictions: `src.retrain_harness.runner.run` (champion_recipe, walk_forward + holdout, track standard) on main with the saved serving frame named under Predictions (hash re-checked). New-Format: the P11 v2 harness run (code 694995f90, frame 9ed237ebc04aee09).
2. `scripts/bars_opt.py universe` (Standard with `--certified-fd`); New-Format universe of P11 v2.
3. `scripts/p11v3_bars_prices.py prepare`, `bars`, `tiers-fit`, `propose`, `compare`.
4. Install = owner approval only (`config/threshold_approvals.json`); nothing here applies it.

## O/U 2.5 on two bases: fixed-order (D1) vs median real price (addenda 2 + 3)

Median basis (owner 2026-11-24, option (a)): per side the median of the real (vigged) prices of every eu + uk book with a bet-time two-sided 2.5 pair whose overround is inside config/market_edge_v2.json and that de-vigs; >= 3 books, else unpriced. No single-book own / football-data price on this basis, and nothing before 2023-05-03 (no per-book history), so the fit window is much shorter.
Median-basis proposal: `TP-2026-11-24-100cf25c` (hash `100cf25c821e`), 51 entries, namespace `bars_p11v3med` - a separate PROPOSAL, not installed. Statuses (all cells of that run): {'PROMISING': 5, 'UNPROVEN': 45, 'OFF': 140}. Only its O/U 2.5 cells differ from the fixed run.
The validation window (2026-09-01..11-08) was seen before: re-validation numbers, never proof.

| Track | League | Cell | Old v2 status / bar / study bets / ROI [CI] | Fixed D1 status / bar / WF / study bets / ROI [CI] / CLV | Median status / bar / WF / study bets / ROI [CI] / CLV |
|---|---|---|---|---|---|
| standard | Championship | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | UNPROVEN / 0.16 / 144 / 0 / - / - | UNPROVEN / 0.09 / 119 / 0 / - / - |
| standard | Championship | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | League One | over25_over | UNPROVEN / 0.11 / 1 / -100.0% - | UNPROVEN / 0.16 / 84 / 0 / - / - | UNPROVEN / 0.09 / 76 / 0 / - / - |
| standard | League One | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | League Two | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | UNPROVEN / 0.17 / 66 / 1 / -100.0% - / -4.55% (n 1) | UNPROVEN / 0.08 / 83 / 0 / - / - |
| standard | League Two | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | UNPROVEN / 0.02 / 2 / 0 / - / - |
| standard | National League | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | - |
| standard | National League | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | - |
| new_format | Argentina Primera Division | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | UNPROVEN / 0.14 / 64 / 0 / - / - |
| new_format | Argentina Primera Division | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | UNPROVEN / 0.02 / 83 / 68 / -21.3% [-44.8%, +1.6%] / -1.42% (n 68) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Austrian Bundesliga | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Austrian Bundesliga | over25_under | UNPROVEN / 0.02 / 29 / -33.0% [-83.9%, +17.9%] | UNPROVEN / 0.02 / 364 / 36 / -29.5% [-66.5%, +7.5%] / +0.19% (n 36) | UNPROVEN / 0.03 / 169 / 9 / -46.5% [-117.2%, +24.2%] / -1.55% (n 9) |
| new_format | Brazil Serie A | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | UNPROVEN / 0.10 / 93 / 3 / -17.3% [-179.4%, +148.0%] / -5.51% (n 3) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Brazil Serie A | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | China Super League | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| new_format | China Super League | over25_under | UNPROVEN / 0.02 / 0 / - | UNPROVEN / 0.02 / 214 / 20 / +22.5% [-33.4%, +78.4%] / -2.71% (n 20) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Denmark Superliga | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / 205 / 3 / +12.7% [-100.0%, +123.2%] / +0.99% (n 3) |
| new_format | Denmark Superliga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Finland Veikkausliiga | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | UNPROVEN / 0.02 / 91 / 3 / -36.7% [-160.8%, +90.0%] / +0.90% (n 3) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Finland Veikkausliiga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Ireland Premier Division | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Ireland Premier Division | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Japan J-League | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NEGATIVE) / - / 61 / 2 / -100.0% [-100.0%, -100.0%] / -1.51% (n 2) | UNPROVEN / 0.10 / 46 / 0 / - / - |
| new_format | Japan J-League | over25_under | PROMISING / 0.02 / 21 / +1.9% [-49.9%, +52.8%] | UNPROVEN / 0.02 / 363 / 28 / -3.8% [-43.0%, +35.5%] / +2.60% (n 28) | PROMISING / 0.03 / 150 / 14 / +23.6% [-34.8%, +82.0%] / +5.11% (n 14) |
| new_format | K-League 1 | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | K-League 1 | over25_under | UNPROVEN / 0.02 / 0 / - | UNPROVEN / 0.02 / 328 / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Mexico Liga MX | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Mexico Liga MX | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | UNPROVEN / 0.02 / 11 / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Norway Eliteserien | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | UNPROVEN / 0.02 / 121 / 8 / +22.8% [-30.3%, +75.8%] / -1.69% (n 8) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Norway Eliteserien | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Romanian Superliga | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | - |
| new_format | Romanian Superliga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - | - |
| new_format | Saudi Pro League | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | UNPROVEN / 0.10 / 2 / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| new_format | Saudi Pro League | over25_under | UNPROVEN / 0.02 / 6 / +28.3% [-100.0%, +143.3%] | UNPROVEN / 0.02 / 55 / 6 / +28.3% [-100.0%, +143.3%] / -2.52% (n 6) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Sweden Allsvenskan | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | Sweden Allsvenskan | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | UNPROVEN / 0.02 / 42 / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | USA MLS | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | PROMISING / 0.02 / 227 / 6 / +7.3% [-80.7%, +95.3%] / +0.89% (n 6) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| new_format | USA MLS | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| standard | Belgian First Division A | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| standard | Belgian First Division A | over25_under | UNPROVEN / 0.02 / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | UNPROVEN / 0.02 / 40 / 9 / -51.1% [-119.7%, +28.6%] / +0.05% (n 7) |
| standard | Bundesliga 2 | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| standard | Bundesliga 2 | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | Dutch Eredivisie | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | PROMISING / 0.11 / 46 / 1 / +155.0% - / +62.42% (n 1) | OFF (OFF_NO_VALUE) / - / - / 0 / - / - |
| standard | Dutch Eredivisie | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | Greek Super League | over25_over | UNPROVEN / 0.10 / 1 / +125.0% - | UNPROVEN / 0.11 / 130 / 2 / +12.5% [-208.0%, +233.0%] / -0.95% (n 2) | UNPROVEN / 0.08 / 72 / 0 / - / - |
| standard | Greek Super League | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | La Liga 2 | over25_over | PROMISING / 0.13 / 5 / +60.0% [-68.1%, +188.1%] | PROMISING / 0.17 / 416 / 6 / +33.3% [-83.6%, +150.2%] / +4.37% (n 6) | UNPROVEN / 0.09 / 336 / 0 / - / - |
| standard | La Liga 2 | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | Ligue 2 | over25_over | UNPROVEN / 0.14 / 0 / - | UNPROVEN / 0.16 / 124 / 1 / -100.0% - / -3.70% (n 1) | UNPROVEN / 0.09 / 81 / 0 / - / - |
| standard | Ligue 2 | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | Portuguese Primeira Liga | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | PROMISING / 0.16 / 88 / 4 / +3.2% [-114.5%, +125.0%] / +6.04% (n 4) | UNPROVEN / 0.09 / 94 / 0 / - / - |
| standard | Portuguese Primeira Liga | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | Scottish Championship | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | UNPROVEN / 0.12 / 66 / 1 / -100.0% - / -4.26% (n 1) | - |
| standard | Scottish Championship | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | - |
| standard | Scottish League One | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | - |
| standard | Scottish League One | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | - |
| standard | Scottish League Two | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | - |
| standard | Scottish League Two | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | - |
| standard | Scottish Premiership | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | UNPROVEN / 0.09 / 40 / 0 / - / - |
| standard | Scottish Premiership | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | Serie B | over25_over | UNPROVEN / 0.09 / 7 / -3.6% [-103.4%, +96.2%] | PROMISING / 0.11 / 316 / 7 / +29.3% [-65.7%, +119.2%] / +7.80% (n 7) | UNPROVEN / 0.08 / 243 / 0 / - / - |
| standard | Serie B | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
| standard | Turkish Super Lig | over25_over | OFF (OFF_NO_VALUE) / - / 0 / - | OFF (OFF_NO_VALUE) / - / - / 0 / - / - | UNPROVEN / 0.08 / 41 / 0 / - / - |
| standard | Turkish Super Lig | over25_under | OFF (OFF_NO_OWN_ESTIMATE) / - / 0 / - | OFF (OFF_THIN) / - / - / 0 / - / - | OFF (OFF_NO_OWN_ESTIMATE) / - / - / 0 / - / - |
