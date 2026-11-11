# P8 - Per-league bars, APPROVED state (2026-11)

**Approved** by owner (Nevo) at 2026-11-11T07:01:22Z (owner 2026-11-11: "Approve, paper + small", ALL entries): proposal **TP-2026-11-10-d999073a**, proposal_hash `d999073a5f1744c512011fe7a487a957ced743c6d8b467ec88dc6302d2e100d6`, file `output/threshold_proposals/bars_oddsapi_hist/2026-11.json`, 74 of 74 entries by id + hash in `config/threshold_approvals.json`. Supersedes TP-2026-11-09-abfb5dc4 (reports/P6_BARS_FOR_APPROVAL.md).

**Not yet applied.** `python -m src.threshold_apply --proposal output/threshold_proposals/bars_oddsapi_hist/2026-11.json` runs on go-live day after the research-lock re-approval (runbook U3a, U3b). Dry run 2026-11-11: would apply 74, already applied 0.

**Nothing is PROVEN.** Statuses below are per league x market; there is no global bar anywhere.

## How these bars were made

* Predictions rebuilt on the corrected history (owner "Rebuild first"): data-truth-100 `2e6e6e745`, frame `58f10476f4c82d7b` (was `8d973b3f2f1f04f7`); standard run_20261110T203611Z, new_format run_20261110T203319Z; champion_recipe unchanged on both tracks.
* Fresh holdout (owner "Fresh holdout", "Fix the rule"), protocol bars-fresh-holdout/3 sha256 `711c6a71df36a5be9dd7e228cd4a55b8e43530fa1cdfd94f8296c8417ea84f3e`: OddsAPI history prices from tiers ['B1', 'B2'] only, holdout = rows whose price no earlier run ever had; tested ONCE (2026-11-10T21:50:45Z), Holm family 21 counted before the test; verdict record TP-FH-2026-11-10-bc7459cc.
* Cells without a fresh stretch keep selection-based bars and verdicts (study re-run on the rebuilt universes, output/bars_opt/study_58f10476), as in TP-2026-11-09-abfb5dc4; thin leagues on the learned blend.

## Staking this season (owner 2026-11-11)

| Staked cell | State | Base fraction | Max odds | Tier multipliers S / M / V |
|---|---|---|---|---|
| `new_format|BTTS|Argentina Primera Division` | ACTIVE | 0.0125 | 2.6 | 1.1 / 0.9 / 1.0 |
| `standard|OU25_OVER|La Liga 2` | ACTIVE_LOW | 0.005 | HARD ceiling (no cell range) | 1.0 / 1.0 / 1.0 |
| `standard|BTTS|La Liga 2` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |

Every other cell is PAPER. **Sweden Allsvenskan NF Under 2.5** is PROMISING but stays PAPER (owner rule 2026-11-02, track paper_reason). Brazil NF BTTS, Serie B BTTS and Serie B Over 2.5 dropped to PAPER (no longer PROMISING). Cloudbet stays PAPER_ONLY; wowza-exec cells are all PAPER (dry run 68 cells, 3 USA MLS held BLOCKED).

**TOP-3** (owner "Paper track"): a named PAPER track in the Clean Record from 2026-11-21; stakes follow the per-cell status only, never the TOP-3 label.

## Fresh-holdout results (tested once)

| Track | Market-side | League | Holdout bets | ROI | ROI 95% CI | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| standard | over15_over | Greek Super League | 98 | -9.3% | [-26.3%, +7.8%] | 1.0 | UNPROVEN |
| standard | over15_over | Scottish Premiership | 2 | -27.5% | [-169.6%, +114.6%] | None | UNPROVEN |
| standard | over15_over | Serie B | 8 | +13.9% | [-41.5%, +64.8%] | 1.0 | UNPROVEN |
| standard | over35_over | Dutch Eredivisie | 191 | -6.0% | [-21.1%, +9.2%] | 1.0 | UNPROVEN |
| new_format | over25_over | Argentina Primera Division | 55 | -17.7% | [-46.8%, +11.5%] | 1.0 | UNPROVEN |
| new_format | over25_over | Brazil Serie A | 80 | -0.1% | [-23.1%, +23.0%] | 1.0 | UNPROVEN |
| new_format | over25_over | China Super League | 114 | -2.8% | [-17.7%, +12.2%] | 1.0 | UNPROVEN |
| new_format | over25_over | Norway Eliteserien | 192 | -5.8% | [-20.8%, +9.2%] | 1.0 | UNPROVEN |
| new_format | over25_over | Sweden Allsvenskan | 48 | -9.6% | [-38.5%, +19.4%] | 1.0 | UNPROVEN |
| new_format | over25_under | Argentina Primera Division | 159 | +8.5% | [-1.8%, +18.6%] | 1.0 | UNPROVEN |
| new_format | over25_under | Austrian Bundesliga | 268 | -1.8% | [-16.2%, +12.5%] | 1.0 | UNPROVEN |
| new_format | over25_under | Denmark Superliga | 74 | -5.3% | [-30.2%, +20.2%] | 1.0 | UNPROVEN |
| new_format | over25_under | Finland Veikkausliiga | 88 | -13.4% | [-36.0%, +9.2%] | 1.0 | UNPROVEN |
| new_format | btts_yes | Brazil Serie A | 178 | -2.3% | [-20.0%, +15.3%] | 1.0 | UNPROVEN |
| new_format | btts_yes | USA MLS | 252 | -11.0% | [-22.9%, +1.2%] | 1.0 | UNPROVEN |
| new_format | over15_over | Argentina Primera Division | 90 | -16.3% | [-37.3%, +4.6%] | 1.0 | UNPROVEN |
| new_format | over15_over | Brazil Serie A | 108 | +2.7% | [-11.3%, +16.6%] | 1.0 | UNPROVEN |
| new_format | over15_over | K-League 1 | 25 | -17.8% | [-50.2%, +14.5%] | 1.0 | UNPROVEN |
| new_format | over35_over | China Super League | 82 | +4.5% | [-22.1%, +31.1%] | 1.0 | UNPROVEN |
| new_format | over35_over | Norway Eliteserien | 78 | +10.3% | [-14.6%, +35.2%] | 1.0 | UNPROVEN |
| new_format | over35_over | Sweden Allsvenskan | 37 | -41.2% | [-76.8%, -2.7%] | 1.0 | UNPROVEN |

Known defect of that run (fixed afterwards, not re-run): the single history snapshot also served as its own closing price, so CLV read 0 and positive-ROI fresh cells (Serie B O1.5, Argentina NF Under, Brazil NF O1.5, China NF O3.5, Norway NF O3.5) could not be labelled PROMISING. PROVEN does not use CLV.

Families without a fresh stretch (no historical PROVEN test; they wait for the live Clean Record): standard over25_over, standard over25_under, standard btts_yes, standard ht_over05_over, standard ht_over15_over, new_format ht_over05_over, new_format ht_over15_over.

## Status counts

| Track | Market-side | PROVEN | PROMISING | UNPROVEN | OFF |
|---|---|---|---|---|---|
| new_format | btts_yes | 0 | 1 | 10 | 4 |
| new_format | over15_over | 0 | 0 | 8 | 7 |
| new_format | over25_over | 0 | 0 | 14 | 1 |
| new_format | over25_under | 0 | 1 | 11 | 3 |
| new_format | over35_over | 0 | 0 | 3 | 12 |
| standard | btts_yes | 0 | 1 | 6 | 10 |
| standard | ht_over05_over | 0 | 0 | 0 | 17 |
| standard | ht_over15_over | 0 | 0 | 0 | 17 |
| standard | over15_over | 0 | 0 | 6 | 11 |
| standard | over25_over | 0 | 1 | 4 | 12 |
| standard | over25_under | 0 | 0 | 1 | 16 |
| standard | over35_over | 0 | 0 | 4 | 13 |

Total: 194 league x market cells: OFF 123, PROMISING 4, UNPROVEN 67.

## Every league x market

| Track | Market-side | League | Status | Bars V / M / S | Configuration | Basis / reason |
|---|---|---|---|---|---|---|
| new_format | btts_yes | Argentina Primera Division | PROMISING | 0.02 / 0.02 / 0.08 | BLEND|EV|max-|min- | SELECTED (OWN_BETS) |
| new_format | btts_yes | Austrian Bundesliga | UNPROVEN | 0.02 / 0.03 / 0.05 | BLEND|EV|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | btts_yes | Brazil Serie A | UNPROVEN | 0.02 / 0.02 / 0.02 | BLEND|EDGE|max2.5|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | btts_yes | China Super League | UNPROVEN | 0.02 / 0.03 / 0.03 | BLEND|EV|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | btts_yes | Denmark Superliga | UNPROVEN | 0.02 / 0.02 / 0.1 | BLEND|EV|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | btts_yes | Finland Veikkausliiga | UNPROVEN | 0.02 / 0.05 / 0.05 | BLEND|EV|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | btts_yes | Ireland Premier Division | UNPROVEN | 0.02 / 0.03 / 0.03 | BLEND|EV|max-|min- | SELECTED (OWN_BETS) |
| new_format | btts_yes | Japan J-League | OFF | - | - | OFF_NO_VALUE |
| new_format | btts_yes | K-League 1 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | btts_yes | Mexico Liga MX | UNPROVEN | 0.02 / 0.02 / 0.1 | BLEND|EV|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | btts_yes | Norway Eliteserien | UNPROVEN | 0.02 / 0.02 / 0.1 | BLEND|EV|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | btts_yes | Romanian Superliga | UNPROVEN | 0.02 / 0.02 / 0.1 | BLEND|EV|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | btts_yes | Saudi Pro League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | btts_yes | Sweden Allsvenskan | OFF | - | - | OFF_NEGATIVE |
| new_format | btts_yes | USA MLS | UNPROVEN | 0.02 / 0.02 / 0.02 | BLEND|EDGE|max2.5|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over15_over | Argentina Primera Division | UNPROVEN | 0.21 / 0.24 / 0.26 | RAW|EV|max-|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over15_over | Austrian Bundesliga | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over15_over | Brazil Serie A | UNPROVEN | 0.02 / 0.05 / 0.05 | RAW|EV|max-|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over15_over | China Super League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over15_over | Denmark Superliga | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over15_over | Finland Veikkausliiga | UNPROVEN | 0.02 / 0.06 / 0.06 | BLEND|EDGE|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | over15_over | Ireland Premier Division | UNPROVEN | 0.02 / 0.02 / 0.03 | BLEND|EDGE|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | over15_over | Japan J-League | UNPROVEN | 0.04 / 0.04 / 0.05 | BLEND|EDGE|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | over15_over | K-League 1 | UNPROVEN | 0.02 / 0.03 / 0.04 | RAW|EV|max-|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over15_over | Mexico Liga MX | UNPROVEN | 0.02 / 0.02 / 0.02 | BLEND|EDGE|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | over15_over | Norway Eliteserien | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over15_over | Romanian Superliga | UNPROVEN | 0.02 / 0.02 / 0.06 | BLEND|EDGE|max-|min- | SELECTED (LEARNED_BLEND) |
| new_format | over15_over | Saudi Pro League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over15_over | Sweden Allsvenskan | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over15_over | USA MLS | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over25_over | Argentina Primera Division | UNPROVEN | 0.02 / 0.04 / 0.05 | LEARNED|EV|max2.5|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_over | Austrian Bundesliga | UNPROVEN | 0.02 / 0.03 / 0.03 | BLEND|EDGE|max2.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_over | Brazil Serie A | UNPROVEN | 0.02 / 0.03 / 0.03 | LEARNED|EV|max2.5|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_over | China Super League | UNPROVEN | 0.02 / 0.03 / 0.03 | LEARNED|EV|max2.5|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_over | Denmark Superliga | UNPROVEN | 0.02 / 0.02 / 0.03 | BLEND|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_over | Finland Veikkausliiga | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over25_over | Ireland Premier Division | UNPROVEN | 0.02 / 0.02 / 0.02 | BLEND|EDGE|max2.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_over | Japan J-League | UNPROVEN | 0.02 / 0.02 / 0.02 | BLEND|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_over | K-League 1 | UNPROVEN | 0.02 / 0.03 / 0.03 | BLEND|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_over | Mexico Liga MX | UNPROVEN | 0.02 / 0.02 / 0.02 | BLEND|EDGE|max2.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_over | Norway Eliteserien | UNPROVEN | 0.02 / 0.03 / 0.03 | LEARNED|EV|max2.5|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_over | Romanian Superliga | UNPROVEN | 0.02 / 0.02 / 0.03 | BLEND|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_over | Saudi Pro League | UNPROVEN | 0.02 / 0.03 / 0.03 | BLEND|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_over | Sweden Allsvenskan | UNPROVEN | 0.02 / 0.02 / 0.02 | LEARNED|EV|max2.5|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_over | USA MLS | UNPROVEN | 0.02 / 0.03 / 0.03 | BLEND|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_under | Argentina Primera Division | UNPROVEN | 0.02 / 0.02 / 0.02 | LEARNED|EV|max-|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_under | Austrian Bundesliga | UNPROVEN | 0.02 / 0.07 / 0.07 | LEARNED|EV|max-|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_under | Brazil Serie A | UNPROVEN | 0.13 / 0.14 / 0.14 | RAW|EV|max3.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_under | China Super League | UNPROVEN | 0.13 / 0.13 / 0.13 | RAW|EV|max3.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_under | Denmark Superliga | UNPROVEN | 0.02 / 0.02 / 0.02 | LEARNED|EV|max-|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_under | Finland Veikkausliiga | UNPROVEN | 0.02 / 0.02 / 0.02 | LEARNED|EV|max-|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over25_under | Ireland Premier Division | OFF | - | - | OFF_NO_VALUE |
| new_format | over25_under | Japan J-League | OFF | - | - | OFF_NO_VALUE |
| new_format | over25_under | K-League 1 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| new_format | over25_under | Mexico Liga MX | UNPROVEN | 0.13 / 0.13 / 0.13 | RAW|EV|max3.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_under | Norway Eliteserien | UNPROVEN | 0.13 / 0.14 / 0.14 | RAW|EV|max3.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_under | Romanian Superliga | UNPROVEN | 0.13 / 0.21 / 0.26 | RAW|EV|max3.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_under | Saudi Pro League | UNPROVEN | 0.13 / 0.3 / 0.3 | RAW|EV|max3.5|min- | SELECTED (LEARNED_BLEND) |
| new_format | over25_under | Sweden Allsvenskan | PROMISING | 0.13 / 0.15 / 0.15 | RAW|EV|max3.5|min- | SELECTED (OWN_BETS) |
| new_format | over25_under | USA MLS | UNPROVEN | 0.13 / 0.18 / 0.18 | RAW|EV|max3.5|min- | SELECTED (OWN_BETS) |
| new_format | over35_over | Argentina Primera Division | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Austrian Bundesliga | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Brazil Serie A | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | China Super League | UNPROVEN | 0.02 / 0.02 / 0.3 | NONE|ALL|max2.5|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over35_over | Denmark Superliga | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Finland Veikkausliiga | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Ireland Premier Division | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Japan J-League | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | K-League 1 | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Mexico Liga MX | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Norway Eliteserien | UNPROVEN | 0.02 / 0.02 / 0.3 | NONE|ALL|max2.5|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over35_over | Romanian Superliga | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Saudi Pro League | OFF | - | - | OFF_NO_VALUE |
| new_format | over35_over | Sweden Allsvenskan | UNPROVEN | 0.02 / 0.02 / 0.3 | NONE|ALL|max2.5|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| new_format | over35_over | USA MLS | OFF | - | - | OFF_NO_VALUE |
| standard | btts_yes | Belgian First Division A | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Bundesliga 2 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Championship | UNPROVEN | 0.02 / 0.02 / 0.02 | SHRINK|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| standard | btts_yes | Dutch Eredivisie | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Greek Super League | UNPROVEN | 0.02 / 0.02 / 0.02 | SHRINK|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| standard | btts_yes | La Liga 2 | PROMISING | 0.02 / 0.02 / 0.3 | NONE|ALL|max2.5|min- | SELECTED (OWN_BETS) |
| standard | btts_yes | League One | UNPROVEN | 0.02 / 0.02 / 0.02 | SHRINK|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| standard | btts_yes | League Two | UNPROVEN | 0.02 / 0.02 / 0.02 | SHRINK|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| standard | btts_yes | Ligue 2 | UNPROVEN | 0.02 / 0.02 / 0.02 | SHRINK|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| standard | btts_yes | National League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Portuguese Primeira Liga | UNPROVEN | 0.02 / 0.02 / 0.02 | SHRINK|EDGE|max2.5|min- | SELECTED (LEARNED_BLEND) |
| standard | btts_yes | Scottish Championship | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Scottish League One | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Scottish League Two | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Scottish Premiership | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Serie B | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | btts_yes | Turkish Super Lig | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | ht_over05_over | Belgian First Division A | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Bundesliga 2 | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Championship | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Dutch Eredivisie | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Greek Super League | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | La Liga 2 | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | League One | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | League Two | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Ligue 2 | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | National League | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Portuguese Primeira Liga | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Scottish Championship | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Scottish League One | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Scottish League Two | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Scottish Premiership | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Serie B | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over05_over | Turkish Super Lig | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Belgian First Division A | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Bundesliga 2 | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Championship | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Dutch Eredivisie | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Greek Super League | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | La Liga 2 | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | League One | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | League Two | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Ligue 2 | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | National League | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Portuguese Primeira Liga | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Scottish Championship | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Scottish League One | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Scottish League Two | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Scottish Premiership | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Serie B | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | ht_over15_over | Turkish Super Lig | OFF | - | - | INSUFFICIENT_SELECTION_EVIDENCE |
| standard | over15_over | Belgian First Division A | UNPROVEN | 0.02 / 0.02 / 0.02 | RAW|EDGE|max-|min- | NO_SELECTION_EVIDENCE (LEARNED_BLEND) |
| standard | over15_over | Bundesliga 2 | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Championship | UNPROVEN | 0.03 / 0.03 / 0.03 | RAW|EDGE|max-|min- | NO_SELECTION_EVIDENCE (LEARNED_BLEND) |
| standard | over15_over | Dutch Eredivisie | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Greek Super League | UNPROVEN | 0.02 / 0.03 / 0.03 | LEARNED|EV|max-|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| standard | over15_over | La Liga 2 | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | League One | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | League Two | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Ligue 2 | UNPROVEN | 0.02 / 0.04 / 0.04 | RAW|EDGE|max-|min- | NO_SELECTION_EVIDENCE (LEARNED_BLEND) |
| standard | over15_over | National League | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Portuguese Primeira Liga | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Scottish Championship | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Scottish League One | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Scottish League Two | OFF | - | - | OFF_NO_VALUE |
| standard | over15_over | Scottish Premiership | UNPROVEN | 0.02 / 0.02 / 0.02 | LEARNED|EV|max-|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| standard | over15_over | Serie B | UNPROVEN | 0.02 / 0.02 / 0.03 | LEARNED|EV|max-|min1.4 | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| standard | over15_over | Turkish Super Lig | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Belgian First Division A | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Bundesliga 2 | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Championship | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Dutch Eredivisie | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Greek Super League | UNPROVEN | 0.11 / 0.13 / 0.13 | RAW|EDGE|max-|min- | SELECTED (OWN_BETS) |
| standard | over25_over | La Liga 2 | PROMISING | 0.13 / 0.17 / 0.19 | RAW|EDGE|max-|min- | SELECTED (OWN_BETS) |
| standard | over25_over | League One | UNPROVEN | 0.12 / 0.13 / 0.13 | RAW|EDGE|max-|min- | SELECTED (OWN_BETS) |
| standard | over25_over | League Two | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Ligue 2 | UNPROVEN | 0.14 / 0.14 / 0.14 | RAW|EDGE|max-|min- | SELECTED (OWN_BETS) |
| standard | over25_over | National League | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Portuguese Primeira Liga | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Scottish Championship | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Scottish League One | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Scottish League Two | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Scottish Premiership | OFF | - | - | OFF_NO_VALUE |
| standard | over25_over | Serie B | UNPROVEN | 0.09 / 0.09 / 0.16 | RAW|EDGE|max-|min- | SELECTED (OWN_BETS) |
| standard | over25_over | Turkish Super Lig | OFF | - | - | OFF_NO_VALUE |
| standard | over25_under | Belgian First Division A | UNPROVEN | 0.02 / 0.02 / 0.02 | LEARNED|EV|max2.5|min- | SELECTED (OWN_BETS) |
| standard | over25_under | Bundesliga 2 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Championship | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Dutch Eredivisie | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Greek Super League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | La Liga 2 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | League One | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | League Two | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Ligue 2 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | National League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Portuguese Primeira Liga | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Scottish Championship | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Scottish League One | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Scottish League Two | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Scottish Premiership | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Serie B | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over25_under | Turkish Super Lig | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Belgian First Division A | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Bundesliga 2 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Championship | UNPROVEN | 0.02 / 0.06 / 0.06 | LEARNED|EDGE|max-|min- | SELECTED (OWN_BETS) |
| standard | over35_over | Dutch Eredivisie | UNPROVEN | 0.02 / 0.02 / 0.3 | NONE|ALL|max2.5|min- | FRESH_HOLDOUT_SELECTION (OWN_BETS) |
| standard | over35_over | Greek Super League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | La Liga 2 | UNPROVEN | 0.02 / 0.04 / 0.06 | LEARNED|EDGE|max-|min- | SELECTED (OWN_BETS) |
| standard | over35_over | League One | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | League Two | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Ligue 2 | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | National League | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Portuguese Primeira Liga | UNPROVEN | 0.02 / 0.06 / 0.06 | LEARNED|EDGE|max-|min- | SELECTED (LEARNED_BLEND) |
| standard | over35_over | Scottish Championship | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Scottish League One | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Scottish League Two | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Scottish Premiership | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Serie B | OFF | - | - | OFF_NO_OWN_ESTIMATE |
| standard | over35_over | Turkish Super Lig | OFF | - | - | OFF_NO_OWN_ESTIMATE |

## After the final refit (runbook R7)

After the 2026-11-19/20 refit, `python scripts/bars_opt_oddsapi.py recheck` rebuilds this table on the refitted predictions; any status change goes back to the owner before the Clean Record starts.
