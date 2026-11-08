# P3 - Per-league tip bars from the clean retrain (PROPOSALS ONLY)

Owner decisions 2026-11-08: every league has its OWN bars for every market and tier (no global bar); small leagues are shrunk toward the pooled estimate by an empirical-Bayes weight learned from the selection data; PROVEN only under the strict rule (>= 150 out-of-sample bets and ROI above 0 with Holm-adjusted one-sided p < 0.025 across every league x market tested). Owner, same day: "no way with 100% real data we don't have true bars" - a league x market priced only from 2026-04-01 gets IN_PERIOD_SPLIT bars; NO_PRICE_DATA only where no real two-sided bet-time price exists at any time (no number is invented there).

Nothing here serves. The candidates are in `output/threshold_proposals/league_bars/<track>/`; production changes only through `config/threshold_approvals.json` after the owner approves an entry. Serving today uses a league's own bars only when `approved` is true and has no per-league VALUABLE bar: using these bars for every league (and VALUABLE per league) needs a serving change the owner decides at go-live.

Context: on the Standard holdout the de-vigged CLOSING O/U 2.5 price beat every model (champion_recipe log loss minus market_only = +0.0146; market_only is a non-deployable benchmark). Bars cannot add information the model does not have; expect few or no PROVEN cells.

## How the bars are built

* Model: `champion_recipe` (the design that won the retrain comparison), deployable rows.
* **WALK_FORWARD_HOLDOUT** evidence: bars chosen on walk-forward rows (< 2026-04-01) only; the holdout (2026-04-01..2026-11-08) is scored once.
* **IN_PERIOD_SPLIT** evidence (no priced walk-forward row): the holdout period is split in time order at the month boundary that leaves about 40% of the LEAGUE's priced matchdays in the CHECK part (its own split; shrinkage pools every league's priced rows in the same earlier window); bars are chosen on the earlier part, scored once on the later part, same strict rule. One short window: weaker evidence than WALK_FORWARD_HOLDOUT. Split dates per league are in the tables.
* Edge = p_model - power de-vigged fair probability of a two-sided pair captured before kickoff. Never the closing price; a fixture without a pair is not bet.
* VALUABLE = lowest bar with shrunk walk-forward ROI > 0; MARKSMAN = bar maximising shrunk total profit; SNIPER = bar maximising shrunk ROI per bet; only bars with >= 40 own selection bets (basis OWN_BETS), else where the pooled bets reach it (basis SHRUNK_TO_POOL: mostly the pooled curve; the league's own weight is the 'Shrink w' column).
* Out-of-sample bets = every tip at or above VALUABLE, 1 unit flat; 95% CI by block bootstrap over match dates.

## standard

Predictions: `C:/Users/nevo/AppData/Local/Temp/claude/1ee106cff9d5ca11/84b5f38e-4e7e-4236-9e42-6479e0aff4a9/scratchpad/retrain_std/run_20261108T190758Z/predictions.parquet`; built 2026-11-08T20:24:02Z.

Price sources:

* **btts**: output/standard_sidemarket_odds_history.csv: earliest provable pre-kickoff same-snapshot pair from match_date-1d ({'BACKFILL_HISTORICAL': 6087, 'FORWARD_CAPTURE': 769})
* **ht_over05**: no price exists for this market
* **ht_over15**: no price exists for this market
* **over15**: output/standard_sidemarket_odds_history.csv: earliest provable pre-kickoff same-snapshot pair from match_date-1d ({'FORWARD_CAPTURE': 830, 'BACKFILL_HISTORICAL': 1})
* **over25**: football-data PRE-MATCH same-book pair ({'B365': 24547, 'Pinnacle': 13})
* **over35**: output/standard_sidemarket_odds_history.csv: earliest provable pre-kickoff same-snapshot pair from match_date-1d ({'FORWARD_CAPTURE': 888, 'BACKFILL_HISTORICAL': 18})

Status count per market:

| Market | INSUFFICIENT_BETS (IN_PERIOD_SPLIT) | NO_PRICE_DATA | UNPROVEN (IN_PERIOD_SPLIT) | UNPROVEN (WALK_FORWARD_HOLDOUT) |
|---|---|---|---|---|
| btts | 1 | 0 | 9 | 7 |
| ht_over05 | 0 | 17 | 0 | 0 |
| ht_over15 | 0 | 17 | 0 | 0 |
| over15 | 1 | 0 | 16 | 0 |
| over25 | 0 | 0 | 0 | 17 |
| over35 | 1 | 0 | 16 | 0 |

### standard / btts

| League | Status | Evidence | Split: select / check (check share) | Basis | VALUABLE | MARKSMAN | SNIPER | Shrink w | Selection bets | Selection ROI | OOS bets | OOS ROI | 95% CI | p Holm | Price match (rows priced / rows) | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Belgian First Division A | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-25 (36%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.0258 | 2 | -2.5% | 2 | +80.0% | [-, -] | - | 28/1163 | 2 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| Bundesliga 2 | UNPROVEN | WALK_FORWARD_HOLDOUT | - | SHRUNK_TO_POOL | 0.05 | 0.13 | 0.13 | 0 | 0 | - | 0 | - | [-, -] | - | 593/1145 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Championship | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.04 | 0.07 | 0.09 | 0.6114 | 266 | +1.5% | 18 | -23.4% | [-64.7%, +19.4%] | 1 | 1077/2102 | 18 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Dutch Eredivisie | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-25 (47%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 1 | +80.0% | 0 | - | [-, -] | - | 42/1184 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Greek Super League | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-23..2026-09-20 / 2026-10-10..2026-10-25 (31%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.3605 | 23 | +3.2% | 7 | -40.7% | [-100.0%, +66.0%] | 1 | 32/909 | 7 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| La Liga 2 | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.02 | 0.12 | 0.13 | 0.4999 | 508 | +3.6% | 81 | +13.0% | [-8.8%, +34.9%] | 1 | 965/1770 | 81 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| League One | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.07 | 0.09 | 0.09 | 0.1897 | 93 | -3.6% | 9 | -76.7% | [-100.0%, -10.0%] | 1 | 1159/2113 | 9 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| League Two | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.07 | 0.07 | 0.08 | 0.1697 | 77 | +12.4% | 10 | +6.4% | [-62.0%, +84.8%] | 1 | 1166/2120 | 10 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| Ligue 2 | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.08 | 0.09 | 0.09 | 0 | 59 | -14.0% | 3 | -31.7% | [-100.0%, +105.0%] | 1 | 615/1299 | 3 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| National League | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-11..2026-09-30 / 2026-10-03..2026-10-27 (36%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 1 | -100.0% | 2 | +91.0% | [-, -] | - | 62/2113 | 2 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| Portuguese Primeira Liga | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-26 (53%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.2496 | 14 | +13.6% | 12 | -13.8% | [-57.6%, +38.6%] | 1 | 36/1183 | 12 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Scottish Championship | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-11..2026-09-19 / 2026-10-03..2026-10-27 (67%) | SHRUNK_TO_POOL | 0.04 | 0.04 | 0.04 | 0 | 4 | -100.0% | 7 | +13.4% | [-34.5%, +61.0%] | 1 | 20/687 | 7 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| Scottish League One | UNPROVEN | IN_PERIOD_SPLIT | 2026-10-17..2026-10-24 / 2026-10-27..2026-10-27 (33%) | SHRUNK_TO_POOL | 0.1 | 0.1 | 0.1 | 0 | 0 | - | 0 | - | [-, -] | - | 7/687 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Scottish League Two | INSUFFICIENT_BETS | IN_PERIOD_SPLIT | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 4/691 | real prices on fewer than 2 matchdays: no earlier part to pick bars on |
| Scottish Premiership | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-05..2026-09-20 / 2026-10-10..2026-10-28 (50%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 1 | +91.0% | 3 | +26.7% | [-100.0%, +100.0%] | 1 | 29/884 | 3 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| Serie B | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.02 | 0.04 | 0.04 | 0.492 | 441 | +8.0% | 54 | -16.9% | [-43.5%, +9.5%] | 1 | 807/1470 | 54 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Turkish Super Lig | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-05..2026-09-20 / 2026-10-09..2026-10-26 (58%) | SHRUNK_TO_POOL | 0.05 | 0.05 | 0.05 | 0 | 2 | -100.0% | 2 | +5.0% | [-, -] | - | 35/1289 | 2 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |

### standard / ht_over05

| League | Status | Evidence | Split: select / check (check share) | Basis | VALUABLE | MARKSMAN | SNIPER | Shrink w | Selection bets | Selection ROI | OOS bets | OOS ROI | 95% CI | p Holm | Price match (rows priced / rows) | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Belgian First Division A | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/564 | no real two-sided bet-time price at any time |
| Bundesliga 2 | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/555 | no real two-sided bet-time price at any time |
| Championship | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1006 | no real two-sided bet-time price at any time |
| Dutch Eredivisie | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/565 | no real two-sided bet-time price at any time |
| Greek Super League | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/423 | no real two-sided bet-time price at any time |
| La Liga 2 | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/843 | no real two-sided bet-time price at any time |
| League One | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1026 | no real two-sided bet-time price at any time |
| League Two | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1031 | no real two-sided bet-time price at any time |
| Ligue 2 | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/562 | no real two-sided bet-time price at any time |
| National League | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1026 | no real two-sided bet-time price at any time |
| Portuguese Primeira Liga | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/575 | no real two-sided bet-time price at any time |
| Scottish Championship | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/339 | no real two-sided bet-time price at any time |
| Scottish League One | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/339 | no real two-sided bet-time price at any time |
| Scottish League Two | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/343 | no real two-sided bet-time price at any time |
| Scottish Premiership | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/427 | no real two-sided bet-time price at any time |
| Serie B | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/684 | no real two-sided bet-time price at any time |
| Turkish Super Lig | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/589 | no real two-sided bet-time price at any time |

### standard / ht_over15

| League | Status | Evidence | Split: select / check (check share) | Basis | VALUABLE | MARKSMAN | SNIPER | Shrink w | Selection bets | Selection ROI | OOS bets | OOS ROI | 95% CI | p Holm | Price match (rows priced / rows) | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Belgian First Division A | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/564 | no real two-sided bet-time price at any time |
| Bundesliga 2 | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/555 | no real two-sided bet-time price at any time |
| Championship | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1006 | no real two-sided bet-time price at any time |
| Dutch Eredivisie | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/565 | no real two-sided bet-time price at any time |
| Greek Super League | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/423 | no real two-sided bet-time price at any time |
| La Liga 2 | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/843 | no real two-sided bet-time price at any time |
| League One | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1026 | no real two-sided bet-time price at any time |
| League Two | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1031 | no real two-sided bet-time price at any time |
| Ligue 2 | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/562 | no real two-sided bet-time price at any time |
| National League | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/1026 | no real two-sided bet-time price at any time |
| Portuguese Primeira Liga | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/575 | no real two-sided bet-time price at any time |
| Scottish Championship | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/339 | no real two-sided bet-time price at any time |
| Scottish League One | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/339 | no real two-sided bet-time price at any time |
| Scottish League Two | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/343 | no real two-sided bet-time price at any time |
| Scottish Premiership | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/427 | no real two-sided bet-time price at any time |
| Serie B | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/684 | no real two-sided bet-time price at any time |
| Turkish Super Lig | NO_PRICE_DATA | - | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 0/589 | no real two-sided bet-time price at any time |

### standard / over15

| League | Status | Evidence | Split: select / check (check share) | Basis | VALUABLE | MARKSMAN | SNIPER | Shrink w | Selection bets | Selection ROI | OOS bets | OOS ROI | 95% CI | p Holm | Price match (rows priced / rows) | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Belgian First Division A | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-25 (42%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 0 | - | 1 | +36.0% | [-, -] | - | 30/1163 | 1 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| Bundesliga 2 | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-07..2026-09-20 / 2026-10-09..2026-10-25 (38%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 0 | - | 0 | - | [-, -] | - | 41/1145 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Championship | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-17..2026-09-20 / 2026-10-10..2026-10-24 (30%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.9594 | 3 | +34.0% | 2 | -33.5% | [-, -] | - | 73/2102 | 2 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Dutch Eredivisie | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-25 (47%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 0 | - | 0 | - | [-, -] | - | 40/1184 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Greek Super League | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-22..2026-09-20 / 2026-10-10..2026-10-25 (37%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.8655 | 4 | +39.0% | 3 | -53.3% | [-, -] | - | 43/909 | 3 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| La Liga 2 | UNPROVEN | IN_PERIOD_SPLIT | 2026-05-16..2026-09-28 / 2026-10-02..2026-10-26 (37%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.0621 | 15 | -13.8% | 8 | -14.5% | [-51.3%, +22.2%] | 1 | 79/1770 | 8 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| League One | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-15..2026-09-26 / 2026-10-03..2026-10-27 (37%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 2 | +36.0% | 2 | -30.0% | [-, -] | - | 78/2113 | 2 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| League Two | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-15..2026-09-26 / 2026-10-03..2026-10-27 (47%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.6583 | 2 | +36.5% | 4 | -33.5% | [-100.0%, +33.0%] | 1 | 81/2120 | 4 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Ligue 2 | UNPROVEN | IN_PERIOD_SPLIT | 2026-10-09..2026-10-17 / 2026-10-19..2026-10-26 (44%) | SHRUNK_TO_POOL | 0.02 | 0.04 | 0.04 | 0.8847 | 2 | +42.0% | 0 | - | [-, -] | - | 18/1299 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| National League | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-11..2026-09-30 / 2026-10-03..2026-10-28 (40%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 0 | - | 0 | - | [-, -] | - | 52/2113 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Portuguese Primeira Liga | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-26 (47%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.0204 | 5 | -17.6% | 2 | -100.0% | [-, -] | - | 34/1183 | 2 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Scottish Championship | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-12..2026-09-25 / 2026-10-03..2026-10-27 (64%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.0053 | 2 | -33.5% | 2 | +43.0% | [-, -] | - | 19/687 | 2 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| Scottish League One | UNPROVEN | IN_PERIOD_SPLIT | 2026-10-17..2026-10-24 / 2026-10-27..2026-10-27 (33%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 0 | - | 0 | - | [-, -] | - | 8/687 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Scottish League Two | INSUFFICIENT_BETS | IN_PERIOD_SPLIT | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 2/691 | real prices on fewer than 2 matchdays: no earlier part to pick bars on |
| Scottish Premiership | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-05..2026-09-19 / 2026-10-10..2026-10-28 (50%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 0 | - | 0 | - | [-, -] | - | 21/884 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Serie B | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-21..2026-09-20 / 2026-10-10..2026-10-28 (36%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0.1074 | 18 | +7.9% | 6 | -33.0% | [-67.5%, +33.0%] | 1 | 67/1470 | 6 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Turkish Super Lig | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-05..2026-09-20 / 2026-10-09..2026-10-25 (50%) | SHRUNK_TO_POOL | 0.02 | 0.02 | 0.02 | 0 | 2 | -100.0% | 3 | -54.7% | [-100.0%, +36.0%] | 1 | 37/1289 | 3 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |

### standard / over25

| League | Status | Evidence | Split: select / check (check share) | Basis | VALUABLE | MARKSMAN | SNIPER | Shrink w | Selection bets | Selection ROI | OOS bets | OOS ROI | 95% CI | p Holm | Price match (rows priced / rows) | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Belgian First Division A | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.02 | 0.02 | 0.02 | 0.5285 | 772 | +1.5% | 101 | +9.2% | [-12.4%, +32.2%] | 1 | 1151/1163 | 101 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| Bundesliga 2 | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.12 | 0.12 | 0.12 | 0 | 112 | -11.5% | 21 | -47.2% | [-100.0%, -7.0%] | 1 | 1124/1145 | 21 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Championship | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.07 | 0.07 | 0.07 | 0.344 | 402 | +1.8% | 29 | -8.5% | [-44.9%, +32.2%] | 1 | 2064/2102 | 29 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Dutch Eredivisie | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.18 | 0.18 | 0.18 | 0 | 76 | -7.5% | 10 | +34.3% | [-65.0%, +140.6%] | 1 | 1166/1184 | 10 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| Greek Super League | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.02 | 0.02 | 0.02 | 0.4791 | 659 | -3.3% | 74 | -5.2% | [-24.6%, +13.6%] | 1 | 885/909 | 74 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| La Liga 2 | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.18 | 0.18 | 0.18 | 0 | 57 | -2.2% | 2 | +170.0% | [-, -] | - | 1730/1770 | 2 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| League One | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.12 | 0.12 | 0.12 | 0 | 55 | +10.1% | 5 | -52.0% | [-100.0%, +44.0%] | 1 | 1511/2113 | 5 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| League Two | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.12 | 0.12 | 0.12 | 0 | 44 | -7.0% | 1 | -100.0% | [-, -] | - | 2070/2120 | 1 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Ligue 2 | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.02 | 0.02 | 0.02 | 0.5594 | 903 | -1.8% | 92 | -20.8% | [-36.7%, -3.5%] | 1 | 1278/1299 | 92 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| National League | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.09 | 0.09 | 0.09 | 0.2062 | 318 | +4.1% | 40 | -32.7% | [-65.7%, +3.5%] | 1 | 2078/2113 | 40 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Portuguese Primeira Liga | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.12 | 0.12 | 0.12 | 0 | 153 | -8.4% | 13 | -7.7% | [-77.5%, +65.0%] | 1 | 1173/1183 | 13 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Scottish Championship | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.1 | 0.1 | 0.1 | 0.0026 | 57 | -7.9% | 6 | -20.0% | [-100.0%, +80.0%] | 1 | 676/687 | 6 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Scottish League One | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.1 | 0.1 | 0.1 | 0.0027 | 64 | -16.9% | 4 | -100.0% | [-100.0%, -100.0%] | 1 | 675/687 | 4 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Scottish League Two | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.02 | 0.02 | 0.02 | 0.421 | 461 | +8.5% | 47 | -26.3% | [-49.1%, -4.9%] | 1 | 686/691 | 47 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Scottish Premiership | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.18 | 0.18 | 0.18 | 0 | 40 | -6.3% | 1 | +220.0% | [-, -] | - | 866/884 | 1 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| Serie B | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.02 | 0.02 | 0.02 | 0.592 | 1059 | +1.3% | 86 | -18.1% | [-40.4%, +7.0%] | 1 | 1412/1470 | 86 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Turkish Super Lig | UNPROVEN | WALK_FORWARD_HOLDOUT | - | OWN_BETS | 0.18 | 0.18 | 0.18 | 0 | 70 | -21.1% | 0 | - | [-, -] | - | 1271/1289 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |

### standard / over35

| League | Status | Evidence | Split: select / check (check share) | Basis | VALUABLE | MARKSMAN | SNIPER | Shrink w | Selection bets | Selection ROI | OOS bets | OOS ROI | 95% CI | p Holm | Price match (rows priced / rows) | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Belgian First Division A | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-25 (42%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 1 | -100.0% | 0 | - | [-, -] | - | 33/1163 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Bundesliga 2 | UNPROVEN | IN_PERIOD_SPLIT | 2026-05-10..2026-09-19 / 2026-10-09..2026-10-25 (38%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 0 | - | 0 | - | [-, -] | - | 50/1145 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Championship | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-17..2026-09-20 / 2026-10-10..2026-10-24 (36%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 10 | -32.5% | 3 | +25.0% | [-, -] | - | 86/2102 | 3 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| Dutch Eredivisie | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-25 (47%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 0 | - | 0 | - | [-, -] | - | 43/1184 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Greek Super League | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-22..2026-09-20 / 2026-10-10..2026-10-25 (32%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 8 | -50.0% | 2 | +100.0% | [-, -] | - | 34/909 | 2 out-of-sample bets < 150; Holm-adjusted p None not < 0.025 |
| La Liga 2 | UNPROVEN | IN_PERIOD_SPLIT | 2026-05-10..2026-09-27 / 2026-10-02..2026-10-26 (35%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 24 | +25.3% | 12 | +33.8% | [-51.9%, +133.7%] | 1 | 92/1770 | 12 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| League One | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-15..2026-09-26 / 2026-10-03..2026-10-27 (33%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 13 | +5.0% | 3 | +25.0% | [-100.0%, +275.0%] | 1 | 84/2113 | 3 out-of-sample bets < 150; Holm-adjusted p 1.0 not < 0.025 |
| League Two | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-15..2026-09-26 / 2026-10-03..2026-10-27 (47%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 7 | +45.0% | 7 | -50.0% | [-100.0%, +5.0%] | 1 | 88/2120 | 7 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Ligue 2 | UNPROVEN | IN_PERIOD_SPLIT | 2026-10-09..2026-10-17 / 2026-10-19..2026-10-26 (44%) | SHRUNK_TO_POOL | 0.03 | 0.03 | 0.03 | 0 | 3 | -100.0% | 1 | -100.0% | [-, -] | - | 14/1299 | 1 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| National League | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-11..2026-09-30 / 2026-10-03..2026-10-28 (40%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 0 | - | 0 | - | [-, -] | - | 61/2113 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Portuguese Primeira Liga | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-06..2026-09-20 / 2026-10-09..2026-10-26 (47%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 11 | +34.1% | 2 | -100.0% | [-, -] | - | 35/1183 | 2 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Scottish Championship | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-12..2026-09-25 / 2026-10-03..2026-10-27 (58%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 6 | +16.7% | 1 | -100.0% | [-, -] | - | 22/687 | 1 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Scottish League One | UNPROVEN | IN_PERIOD_SPLIT | 2026-10-17..2026-10-24 / 2026-10-27..2026-10-27 (33%) | SHRUNK_TO_POOL | 0.07 | 0.07 | 0.07 | 0 | 0 | - | 0 | - | [-, -] | - | 6/687 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Scottish League Two | INSUFFICIENT_BETS | IN_PERIOD_SPLIT | - | - | - | - | - | - | - | - | - | - | [-, -] | - | 4/691 | real prices on fewer than 2 matchdays: no earlier part to pick bars on |
| Scottish Premiership | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-05..2026-09-19 / 2026-10-10..2026-10-28 (55%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 1 | -100.0% | 0 | - | [-, -] | - | 27/884 | 0 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p None not < 0.025 |
| Serie B | UNPROVEN | IN_PERIOD_SPLIT | 2026-08-22..2026-09-20 / 2026-10-10..2026-10-28 (44%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 23 | -33.0% | 6 | -45.8% | [-100.0%, +225.0%] | 1 | 61/1470 | 6 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |
| Turkish Super Lig | UNPROVEN | IN_PERIOD_SPLIT | 2026-09-05..2026-09-20 / 2026-10-09..2026-10-26 (52%) | SHRUNK_TO_POOL | 0.03 | 0.04 | 0.04 | 0 | 8 | +31.2% | 4 | -100.0% | [-100.0%, -100.0%] | 1 | 45/1289 | 4 out-of-sample bets < 150; out-of-sample ROI not above 0; Holm-adjusted p 1.0 not < 0.025 |

## new_format

NOT BUILT: no `new_format` retrain predictions are available (the job did not finish with an artifact). Run `scripts/league_bars.py --track new_format` on its predictions.parquet, then `--report`.

## What the owner decides

1. Approve or reject each PROVEN entry (only those can be real money under the strict rule).
2. Whether UNPROVEN leagues tip on their own bars as paper, and the serving change for per-league VALUABLE.
3. NO_PRICE_DATA cells (no real two-sided bet-time price at any time, e.g. half-time markets): no bars exist; they need price capture first.
