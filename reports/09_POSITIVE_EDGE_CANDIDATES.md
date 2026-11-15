# 09 — Positive-edge candidates (AU3)

Audited: v9 `origin/main` **fb0e9697ec8556d4dd3df57f31f8ed5415a5012d**; Pro `origin/main` **e0ccc27b98f5** (frozen
registrations read with `git show origin/main:registry/...`; the local Pro clone is 8 commits ahead and was not used).
Scripts: `scripts/audit/au3_phase9.py`, `au3_baselines.py`, `au3_pred_panel.py` (+ `au3_first_sight.py`,
`au3_settle.py`). Numbers: `reports/au3_data/phase9_numbers.json`, `phase9_concentration.csv`,
`phase9_league_cells_since0901.csv`, `phase9_nf_btts_universe.csv`, `baselines.json`.
Windows: **since 2026-09-01** (counts) and **all live** (side ledger starts 2026-08-09), labelled throughout.
Everything below is a DISCOVERY-period number; nothing is proposed as a filter.

## 1. The claim, reproduced exactly

"NF BTTS 142 bets, ~+18.9%, CI ~+1.8..+35.0 (as of 10-31)" is the committed `side_bets_ledger.csv`, market `btts`,
`model_type=new_format`, `source=live`, **ledger `signal_tier` in SNIPER/MARKSMAN**, match_date ≤ 2026-10-31:
**n 142, +26.88u, ROI +18.9%**, day-block 95% CI [+1.3, +35.5], i.i.d. [+1.5, +37.2]. Reproduced.

That population has two defects (Phase 6): 73 of the 299 NF BTTS tips carry an **upgraded** tier, and 53 were settled at
a **later run's price** (both stopped 2026-10-26; rows were not rewritten).

## 2. Corrected versions (independent settlement, first-sight price unless stated)

| population | all live | since 2026-09-01 |
|---|---|---|
| claimed: ledger-tier S+M, ledger price | n 139, +27.52u, +19.8% [+2.6, +35.5] | n 119, +17.66u, +14.8% [−4.6, +32.6] |
| ledger-tier S+M, **first-sight price** | n 139, +25.78u, +18.5% [+1.5, +34.1] | n 119, +16.42u, +13.8% [−5.5, +31.6] |
| **first-sight tier S+M** (what was actually sent as staked), first-sight price | n 83, +24.16u, +29.1% [+3.4, +53.0] | n 70, +16.12u, **+23.0% [−5.3, +49.7]** |
| all tiers, first-sight price | n 267, +35.03u, +13.1% [+0.9, +26.2] | n 234, +18.64u, **+8.0% [−5.2, +21.5]** |
| first-sight VALUABLE only | n 184, +10.87u, +5.9% | n 164, +2.53u, +1.5% |

Since 2026-09-01 **no NF BTTS population has a CI excluding zero.** The all-live positives depend on August (n 33,
+49.7%).

## 3. Forensics

**Argentina concentration.** Argentina is 47% of the bets and **105% of the profit** (all live: Argentina n 125
+36.85u, +29.5% [+11.3, +48.5]; everything else n 142, −1.82u, −1.3% [−14.4, +11.9]). Since 09-01: Argentina n 103
+25.6u +24.9% [+3.9, +47.1]; ex-Argentina n 131 −6.96u −5.3%; ex-Argentina-ex-Brazil n 92 **−15.0% [−28.8, −0.5]**.

**League regime, not selection.** On the all-fixture first-sight panel (every NF fixture with a BTTS price, settled
independently), **blind "BTTS Yes on every Argentina fixture" returns +28.6% (n 132, [+9.6, +48.1]) all live and
+24.1% (n 110) since 09-01 — the same as the model's picks** — and the model tips **94.7%** of priced Argentina
fixtures. League-stratified tipped-minus-not-tipped difference: **+2.2 pp all live, −0.8 pp since 09-01** (10 leagues
with ≥5 in each arm). Overall blind NF BTTS Yes: +0.2% all live, −1.7% since 09-01. The model's apparent edge is the
mix of leagues it tips, chiefly Argentina. Same conclusion as Pro's `src/studies/argentina_btts.py` and
`registry/btts_league_regime`.

**Brazil.** NF BTTS tipped n 46, +25.7% [+4.4, +43.9] all live; +17.7% (n 39, [−5.5, +35.7]) since 09-01. Blind Brazil
Yes +12.1% (n 101) / +7.0% (n 83); not-tipped Brazil +1.1% (n 55) / −2.2% (n 44). The tipped-vs-not gap (≈+19 pp) is the
only league where selection looks additive, on n ≈ 40 per arm with overlapping CIs, chosen after looking — not
evidence. Brazil O/U 2.5: UNDER +5.2% (n 44), OVER +31.7% (n 10) since 09-01, CIs span zero.

**Other "positive" NF side market — Over 1.5:** +11.6% (n 76) since 09-01, CI [−4.1, +27.7]; **71 of 76 are Argentina**.
Same regime.

**Odds bands** (since 09-01): <1.8 +3.4% (n 58); 1.8–2.0 **−21.1%** (n 65); 2.0–2.2 +23.8% (n 69); 2.2–2.5 +37.1% (n 37).
Non-monotone; no band CI excludes zero except 1.8–2.0 negative.

**Date clustering / decay.** By month: Aug +49.7% (n 33), **Sep +21.9% (n 110, [+6.7, +38.5]), Oct −4.4% (n 124)**. The
top 5 match days carry +24.25u of the +18.64u since 09-01 (16% of bets); without them the remainder is −2.9%.

**Model version.** Only one BTTS version is stamped on ledger rows (`btts_20261005T101905Z_238e3419`, rows first logged
from 2026-10-18): n 31, **−23.8%** [−53.1, +7.3]. Earlier rows are `UNKNOWN_LEGACY` (n 203, +12.8% since 09-01) and cannot
be attributed to a model (statistical limitation, never back-filled).

**Threshold.** NF leagues are absent from `models/best_params_side_markets.json`, so every NF BTTS tip used the fixed
pipeline gates (VALUABLE edge ≥ 0.04, MARKSMAN 0.08, SNIPER 0.10 on `p − (1/odds)/1.08`) throughout. The 2026-10-06
automated backtest rewrite of that file changed only **Standard** BTTS gates (Championship approved 0.09/0.07; La Liga 2,
League One, League Two, Ligue 2, Serie B became un-approved → fixed globals). Edge bands since 09-01: 4–6% −4.9% (n 118),
6–8% +18.2%, 8–10% −11.0%, 10–15% +38.4% (n 40, [+0.1, +75.4]) — non-monotone.

**Books / price provenance.** All 299 NF BTTS ledger odds equal `odds_btts` in `predictions.csv` of the first-sight
commit (`src/btts_odds.py`): the **median** Yes price across OddsAPI books that quoted *both* sides; median 5 books; 6 tips
rest on a single book, 27 on two. The median is achievable at ≥ half the quoting books at that snapshot (conservative vs
best price) but is a consensus figure, not a named-book fill. Median two-sided margin 8.0%. 11 tips are priced exactly
1.85 (the historical synthetic BTTS fallback value) but each carries a two-sided multi-book quote — genuine, not the
fallback. No tip was first logged after kickoff (min 0.4 h, median 160 h before kickoff — entry is ~6–7 days out).

**Duplicates / settlement.** 0 duplicate keys; 267 of 270 ledger-settled NF BTTS rows independently settled, **0 result
mismatches**; 11 ledger-settled rows could not be verified (score not yet in af/fd or ambiguous names). 95 NF BTTS tips
share the fixture with an OVER 2.5 tip and 73 with an UNDER 2.5 tip — correlated positions; the day-block bootstrap
covers same-day correlation, not same-fixture hedging.

**Availability bias.** Since 09-01 every NF fixture on the board had a two-sided BTTS price (100% in all 12 leagues), so
the tipped set is not filtered by price availability in the counting window. Before 09-01 only 1,105 of 1,815 NF
fixtures were priced (the whole-board per-event BTTS fetch began 2026-08-22, commit 079d9bebc).

**Multiple testing.** Since 09-01, **69** league × market × track cells (O/U 2.5 by side, BTTS, O1.5, O3.5) were scored;
the best one-sided bootstrap p is 0.012 (NF BTTS Argentina), and **no cell survives Benjamini-Hochberg at q ≤ 0.10**
(minimum q 0.72). The candidates themselves came out of a search the registry records as 1,682 hypotheses.

## 4. Pre-registered forward tests (frozen; status from AU3 recompute, first-sight data to 2026-11-01)

| id (registry) | definition | forward start | n_min | AU3 recompute | status |
|---|---|---|---|---|---|
| H-BTTS-01 (v9 `registry/preregistered_hypotheses.json`, frozen family BTTS_ARGENTINA 2026-10-04) | staked (S+M) BTTS, Argentina | 2026-10-02 | 150 | ledger-tier S+M n 31, **−7.0%**; first-sight-tier S+M n 19, +5.4% (15 of the 31 entered "staked" by a tier upgrade) | ACCRUING (13–21%) |
| H-BTTS-02 (same) | staked BTTS ex-Argentina | 2026-10-02 | 150 | ledger-tier n 24 −3.8%; first-sight n 7 | ACCRUING |
| H-BTTS-03 | Argentina consensus CLV (≥3 two-sided books) | 2026-10-05 | 150 | not evaluable from ledgers (single-book CLV only; Pro reports −0.85% as context) | ACCRUING / NOT EVALUABLE HERE |
| H-BTTS-04 | Argentina model vs de-vigged market Brier | 2026-10-05 | 150 | Pro 10-20: model 0.240 vs market 0.216 (n 8) | ACCRUING |
| H-VAL-02 (Pro `registry/valuable/hypotheses_v1.json`) | NF BTTS VALUABLE, edge 4–8% | 2026-10-21 | 150 | n 37, **−28.8%** [−47.4, −3.1] | ACCRUING, trending to REJECTED |
| H-VAL-03 (same) | Std BTTS VALUABLE 4–8% | 2026-10-21 | 100 | n 18, −9.3% | ACCRUING |
| H-EDGE2-NF-BTTS-01 (Pro `registry/edge2_hypotheses_v1.json`) | market_edge_v2 p_decision LL < de-vigged market, NF BTTS | 2026-10-26 | 400 | not recomputed (Pro shadow) | ACCRUING |
| H-GATE-13 + H-BTTS-REGIME-SB / H-BTTS-MODEL-SB (Pro `registry/btts_league_regime/registration_v1.json`) | Serie B BTTS Yes: league regime vs model | 2026-10-30 | 150 / 60 per arm | 2 days in | REGISTERED |

**Gaps (tests that are needed and do not exist):**
1. **Argentina BTTS Yes — regime control.** H-BTTS-01 tests the model's picks; nothing pre-registers *blind* Argentina
   BTTS Yes (or tipped vs not tipped inside Argentina, which cannot accrue: the model tips ~95%). A frozen
   "H-BTTS-REGIME-ARG" like the Serie B one is required before any Argentina result can be read as model skill.
2. **Brazil NF BTTS tipped vs not tipped** (the one league where selection looked additive) — not registered.
3. **NF O1.5 in Argentina** — 93% of the NF O1.5 evidence; not registered as its own cell (H-GATE-01 is DISCOVERY with
   no gate).
4. **First-sight tier definition.** H-BTTS-01/02 say "staked (SNIPER+MARKSMAN)" without fixing *which* tier; Pro's
   evaluator (`src/studies/edge_reassessment.py`) reads the ledger tier, which until 2026-10-26 could be upgraded
   after the bet — 15 of 31 Argentina "staked" rows since 10-02 were upgraded in. An amendment (new versioned entry,
   per the registry's own policy) should fix the first-logged tier.

## 5. Verdict inputs

| track | claimed | independently verified (claim population) | corrected (first-sight tier & price, since 09-01) | valid bets | CI | verdict input |
|---|---|---|---|---|---|---|
| NF BTTS | +18.9% (n 142) | +19.8% (n 139, ledger price) | +23.0% (n 70, S+M) / +8.0% (n 234, all tiers) | 234 since 09-01 | [−5.2, +21.5] all tiers | **PROMISING-UNCONFIRMED at best; the profit is an Argentina league regime (blind Yes earns the same), decaying (Oct −4.4%), the only stamped model version is −23.8%** |
| NF BTTS ex-Argentina | — | — | −5.3% (n 131) | 131 | [−18.2, +7.5] | NO DEMONSTRATED EDGE |
| Brazil (BTTS) | — | — | +17.7% (n 39) | 39 | [−5.5, +35.7] | INSUFFICIENT DATA |
| NF O1.5 | — | +11.6% (n 76) | same | 76 | [−4.1, +27.7] | INSUFFICIENT DATA (Argentina regime) |

## 6. Bugs (Phase-11 fields)

| Bug ID | Sev | Repo | File/line | Root cause | Dates | Rows | Prod impact | Prob | Selection | ROI impact | Status | Smallest safe fix | Regression test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU3-09-A | P2 | Pro | `src/studies/edge_reassessment.py` (~l.402–430) | H-BTTS-01/02 evaluator selects "staked" by ledger tier, which was upgradable until 10-26 | 2026-10-02 → 10-26 entries | 15/31 ARG, 19/24 ex-ARG ledger-staked rows upgraded-in | forward-test sample contaminated by post-bet information | none | changes which bets count | ARG −7.0% (ledger tier) vs +5.4% (first sight), same window | ACTIVE (research) | evaluate on first-sight tier (`ledger_tier_seen.csv` / git first-sight); amend registry with a new version entry | test: an upgraded tier never moves a row into the staked arm |
| AU3-09-B | P3 | v9 | `models/best_params_side_markets.json` written by the monthly backtest 2026-10-06 (d311bb2f8) | automated threshold rewrite during the BTTS_ARGENTINA freeze | 2026-10-06 | Standard BTTS leagues (6) | changed selection inside H-BTTS-02's window (ex-Argentina includes Standard BTTS) | none | Standard BTTS gates changed | NOT IDENTIFIABLE | HISTORICAL (thresholds now owner-approved only, 8b46b37f8 10-26) | record 10-06 as a regime break for H-BTTS-02 | research_lock already fingerprints the file |
| AU3-09-C | P3 | Pro | `output/edge_reassessment/edge_reassessment.json` | H-BTTS progress last generated 2026-10-20 | 10-20 → now | — | stale forward-test status | — | — | — | ACTIVE (staleness) | schedule or note the as-of date | — |
