# 04 — UNDER 2.5 forensic investigation (AU3)

Audited: v9 `origin/main` **fb0e9697ec8556d4dd3df57f31f8ed5415a5012d** (worktree `v9_audit_au3`, branch `audit/au3`).
Data: the committed git history of `output/bets_ledger.csv` (798 versions), `output/predictions.csv` (1,347 versions),
`output/af_history.parquet` + `output/fd_history.parquet` (results), `output/v91_shadow.csv`. Nothing in production was
modified. Live evidence is reported in two clearly labelled windows: **since 2026-09-01** (the window that counts, after
the August 2026 bug) and **all live** (2026-04-26 onward) for context.

Commands (from the worktree, `PY=..\v9\.venv\Scripts\python.exe`):
```
PY scripts/audit/au3_first_sight.py      # first-sight rows from every committed ledger version
PY scripts/audit/au3_settle.py           # independent settlement (af+fd results, no update_results import)
PY scripts/audit/au3_trace.py            # join each first-sight tip to predictions.csv of the SAME commit
PY scripts/audit/au3_phase4.py           # splits, calibration, cause checks -> reports/au3_data/phase4_*
PY scripts/audit/au3_pred_panel.py       # all-fixture first-sight panel (availability universe)
PY scripts/audit/au3_baselines.py        # blind OVER/UNDER baselines -> reports/au3_data/baselines.json
PY -m pytest tests/audit/test_au3_settlement.py
```

## 1. What the "OVER ~+6.4% vs UNDER ~−12.5%" claim is

No artifact in v9 or Pro states those exact figures, so the population was reconstructed. The only populations that
produce an OVER-positive / UNDER-about-−12.5% split are **staked tiers (SNIPER+MARKSMAN) read from the committed ledger's
`signal_tier`**:

| committed ledger, SNIPER+MARKSMAN (ledger tier) | OVER | UNDER |
|---|---|---|
| all live | n 183, **+4.3%** | n 618, **−12.0%** |
| match_date ≥ 2026-08-01 | n 146, +5.9% | n 518, −12.9% |
| match_date ≥ 2026-09-01 | n 108, +8.3% | n 363, −13.0% |
| all tiers, ≥ 2026-09-01 | n 264, −4.9% | n 613, −5.2% |

The claim therefore rests on the **ledger tier**, which until 2026-10-26 was upgraded in place ("best tier wins",
`src/ledger.py` docstring). 403 of the 1,694 first-sight O/U tips carry a tier that is not the tier first logged (301
VALUABLE→MARKSMAN, 52 VALUABLE→SNIPER, 50 MARKSMAN→SNIPER; the 80 VALUE→VALUABLE renames are excluded). Read at the
**first-sight** tier and settled independently at the first-sight price, the split changes:

| since 2026-09-01, independent settlement, first-sight price | OVER | UNDER |
|---|---|---|
| first-sight SNIPER+MARKSMAN | n 55, +1.47u, **+2.7%** | n 190, −35.87u, **−18.9%** |
| ledger (upgraded) SNIPER+MARKSMAN | n 106, +8.99u, +8.5% | n 352, −43.64u, −12.4% |
| all tiers | n 258, −11.78u, **−4.6%** [day-block 95% −16.9, +8.7] | n 587, −29.37u, **−5.0%** [−16.1, +3.7] |

**Conclusion of §1:** across all tiers OVER and UNDER lose the same ~5% since 2026-09-01. The large OVER/UNDER gap exists
only inside the staked tiers, and its size depends on which tier label is used (first-sight vs upgraded). Neither side's
confidence interval excludes zero on the all-tier population since 2026-09-01.

## 2. Row traces (match → features → probability → price → side → ledger → result → P&L)

Every first-sight O/U tip was joined to the `predictions.csv` row **in the same commit** that first logged it (1,649 of
1,694 rows; 45 untraceable because the predictions row was not in that commit). Four representative rows
(full set: `reports/au3_data/phase4_rows.csv`):

| | Inter Miami v Atlanta Utd (MLS) | Sheffield Utd v Bolton (Champ.) | Dunkerque v Clermont (Ligue 2) | Avispa Fukuoka v Urawa (J1) |
|---|---|---|---|---|
| first-sight commit / time | c5bff765 / 08-30 01:40 | f443b5e3 / 08-29 16:46 | 1be9dbc5 / 08-31 13:40 | c15ac532 / 08-28 16:09 |
| form features (h scored/conc, a scored/conc last5) | 2.6/2.2, **NaN/NaN** (blind; guard owner-disabled 08-21→09-15) | 1.2/1.0, NaN/NaN | 2.0/2.6, 1.4/1.2 | 0.4/1.0, 1.0/2.8 |
| p_over25 (model) | 0.533 | 0.473 | 0.447 | 0.519 |
| odds Over / Under (same snapshot) | 1.25 / 3.50 (overround 1.086) | 1.74 / 2.06 | 1.88 / 1.93 | 2.35 / 1.50 |
| side, p_side, edge = p_side − 1/odds | UNDER, 0.467, +18.2% | UNDER, 0.527, +4.2% | UNDER, 0.553, +3.5% | OVER, 0.519, +9.4% |
| market de-vigged p_side | 0.263 | 0.458 | 0.493 | 0.390 |
| ledger: side, odds, tier | UNDER @3.50 SNIPER | UNDER @2.06 VALUABLE | UNDER @1.93 VALUABLE | OVER @2.35 MARKSMAN |
| independent score (source) | 2-2 (af+fd) | 3-2 (fd) | 1-1 (fd) | 2-3 (af+fd) |
| result / P&L (independent = ledger) | LOSS −1 | LOSS −1 | WIN +0.93 | WIN +1.35 |

The chain is arithmetically correct on every traced row (§3). The first row shows the UNDER failure shape: a SNIPER
UNDER at 3.50 where the model says 47% and the market says 26%.

## 3. The 13 candidate causes

| # | Cause | Test | Result | Verdict |
|---|---|---|---|---|
| 1 | Under probability not 1−p | `edge_under == (1−p_over) − 1/odds_under` in predictions.csv; ledger edge recomputed | 1,649/1,649 predictions rows match to 1e-3; ledger first-sight edge matches recomputation on 1,647/1,649 (2 differ by rounding) | **REJECTED** |
| 2 | Under odds wrong | ledger odds vs predictions `odds_under25` of the same commit | 1,649/1,649 equal the chosen side's price; 0 carry the other side's price | **REJECTED** |
| 3 | Side inversion | chosen side = larger-edge side; ledger side = predictions `bet` | 1,649/1,649 larger-edge; 1,647/1,649 equal `bet` (2 rows: `bet` blank at that commit, side still the larger-edge side) | **REJECTED** |
| 4 | Settlement error | independent settlement from af_history + fd_history, league-scoped name resolution, no import of update_results | 1,475 rows settled independently; **0** WIN/LOSS mismatches vs ledger; af and fd never disagree on a score | **REJECTED** |
| 5 | Wrong line | UNDER wins iff total ≤ 2 independently | identical row-by-row with ledger (cause 4) | **REJECTED** |
| 6 | Goal target wrong | full-time goals, not HT; score table FT columns | FT used in both; 0 mismatches | **REJECTED** |
| 7 | Overconfidence | claimed p_side vs realised on bets | since 09-01 UNDER: claimed **0.504**, realised 0.414, market 0.402 (z 4.4, n 587); OVER: claimed 0.534, realised 0.450, market 0.440 (z 2.7, n 258). Brier model 0.251/0.257 vs de-vigged market 0.237/0.247 | **CONFIRMED, both sides** |
| 8 | League / base-rate bias | all-fixture first-sight panel: model mean p_over vs market vs realised | **new_format: model 0.498, market fair 0.551, realised 0.571** (n 798 since 09-01); standard: 0.527 / 0.520 / 0.508 (n 598). v91_shadow (n 447 NF since 10-04): model 0.478 vs realised 0.577; per-league bias −14 to −22 pp in MLS, J1, Ireland, Denmark, Austria, China, Norway | **CONFIRMED for new_format**: the NF model under-prices goals by ~7 pp vs the market and ~7–10 pp vs reality, so it routes volume to UNDER. 74% of UNDER tips since 09-01 are NF (432/587). Root cause NOT IDENTIFIABLE from AU3 data (candidate: the NF train/serve parity defects fixed today — see Phase 2) |
| 9 | Odds-band effect | side × odds band | UNDER ≥ 3.0: since 09-01 n 40, **−29.9%**, hit 0.200 vs model 0.430 vs market 0.286; all live n 78, −47.6% [−76.3, −12.3]. UNDER < 2.0 is +10.8% (n 86) | **CONFIRMED (long-odds UNDER)** — same band the CLAUDE.md "odds > 3.0" finding names |
| 10 | Legacy/repaired mixing | serving recipe stamped on predictions rows; first-sight time vs the 2026-10-28 12:38 UTC switch | 23 Standard tips logged after the switch, all stamped `v9_repaired` + `standard_repaired_20261028…`; none before. 4 settled (ledger), 0 independently settled (results not yet in af/fd) | **No mixing found**; repaired population too small to evaluate (see Phase 3) |
| 11 | Direction thresholds | per threshold regime (git history of predict.yml env + best_params) | UNDER loses in T0, T1, T3, T4, T6; positive in T2 (+2.3%) and T5 (+11.2%, n 81). No regime has a CI excluding zero both ways | **Not a cause in isolation**; thresholds identical for both sides |
| 12 | Larger-edge selection bias | ROI by first-sight edge band; Spearman(edge, P&L) | since 09-01 UNDER: 3–5% **+8.3%** (n 252), 5–8% −2.8%, **8–12% −30.5%** [−49.5, −9.7] (n 113), 12–20% −17.2%; Spearman −0.07 (UNDER), −0.04 (OVER). Model−market gap ≥ 8 pp → UNDER −11.7% to −17.7% | **CONFIRMED**: a larger claimed edge is a larger model error, not a larger advantage; the tier ladder concentrates exactly those rows |
| 13 | Longshot unders | UNDER odds ≥ 2.5 | covered by 9 and 12: market fair p_under median 0.405 on UNDER tips; ≥ 2.5 rows hit 0.20–0.35 | **CONFIRMED** (same mechanism as 8+12) |

### Blind baselines (does the model's UNDER selection add or subtract?)
All-fixture panel, back every fixture at its first two-sided O/U 2.5 price, independent settlement
(`reports/au3_data/baselines.json`):

| since 2026-09-01 | blind (every fixture) | model-tipped | not tipped |
|---|---|---|---|
| NF UNDER | n 798, **−10.9%** [−17.5, −5.2] | n 432, −5.0% | n 366, −17.9% |
| NF OVER | n 798, −2.6% | n 87, +9.4% | n 711, −4.0% |
| Standard UNDER | n 598, −5.1% | n 155, −9.3% | n 443, −3.6% |
| Standard OVER | n 598, −9.7% | n 171, −12.9% | n 427, −8.5% |

The UNDER market itself lost ~11% blind on new-format this season (realised overs 0.571 vs fair 0.551). On NF the
model's UNDER selection is *better* than blind UNDER, but its downward bias (cause 8) sends most of its volume to the
losing side. On Standard the model's selection is worse than blind on both sides.

## 4. Required splits (since 2026-09-01, first-sight price, independent settlement; all-live in `phase4_splits.csv`)

**By side / track:** NF OVER n 87 +10.9% [−17.3, +39.6]; NF UNDER n 432 −4.3% [−17.0, +6.8]; Std OVER n 171 −12.4%
[−26.9, +4.6]; Std UNDER n 155 −6.9% [−20.7, +8.8].

**By odds band** (UNDER / OVER): <2.0: +10.8% n 86 / −10.0% n 109; 2.0–2.5: −5.8% n 325 / +3.1% n 106; 2.5–3.0: −5.7%
n 136 / −13.9% n 38; ≥3.0: −29.9% n 40 / +22.0% n 5.

**By model version** (stamped since 2026-10-18 only; earlier rows are `UNKNOWN_LEGACY` and cannot be attributed):
UNDER `newformat_20261018…` n 76 −21.9% [−42.9, −8.0]; UNDER `UNKNOWN_LEGACY` n 497 −2.1%; `standard_20261018…` n 14
−17.9%; `standard_repaired_20261028…` 0 independently settled.

**By threshold regime** (first-sight commit time): T3 env-widened 08-21: UNDER +18.0% n 41; T4 backtest 09-01: UNDER
−14.1% n 211; T5 10-01: +11.2% n 81; T6 10-06: −6.3% n 254. OVER: −5.2% / −14.7% / −4.0% in T4/T5/T6.

**By league (UNDER, since 09-01):** worst MLS n 114 −20.4%, Austria n 21 −52.2% [−88.6, −4.5], Ireland n 26 −37.7%
[−69.0, −3.0], League One n 33 −30.2%; best Finland n 27 +37.9% [9.7, 74.4], Sweden n 42 +22.0%, Mexico n 47 +17.6%.
69 league×market cells were examined in Phase 9; none survives Benjamini-Hochberg at q ≤ 0.10 (Phase 9 §6), so no
league cell here is evidence.

## 5. Findings and verdict inputs

* **No implementation defect explains the UNDER losses.** Probability, price, side, ledger write, settlement, line and
  goal target were each verified row by row (causes 1–6: 0 defects on 1,649 traced / 1,475 independently settled rows).
* **The losses are model error, not plumbing:** (a) a new-format downward goal bias of ~7 pp against the market
  (cause 8), (b) overconfidence of ~9 pp on both sides (cause 7), (c) edge-ranked tiers that load the most-wrong rows
  (cause 12), concentrated in long-odds UNDER (causes 9/13). The market's de-vigged price beats the model on Brier on
  the model's own bets, on both sides.
* **The OVER/UNDER asymmetry is mostly a reporting artifact** of upgraded ledger tiers and of slicing to staked tiers;
  on all tiers both sides lose ~5% since 2026-09-01 with CIs spanning zero.
* **Repaired Standard model:** 23 tips, 4 ledger-settled — INSUFFICIENT FORWARD EVIDENCE — CANNOT DECLARE REPAIRED MODEL
  UNPROFITABLE.
* Blind (missing-form) tips: 50 since 09-01 were tipped while the owner-approved guard override was active
  (2026-08-21 → 2026-09-15); 0 after the override expired. Their ROI does not differ from form-backed tips (UNDER blind
  −4.3% n 27 vs −5.0% n 560). Intentional configuration, not a bug.
* No new filter is proposed. Any UNDER-odds cap or NF bias correction would have been chosen on this data and needs an
  untouched forward period (Phase 9 lists the registered tests).

## 6. Bugs (Phase-11 fields)

| Bug ID | Sev | Repo | File/line | Root cause | Dates | Rows | Prod impact | Prob impact | Selection impact | ROI impact | Status | Smallest safe fix | Regression test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU3-04-A | P2 | v9 | `src/ledger.py` append_tips (pre-2026-10-26 "best tier wins") | later runs overwrote `signal_tier`/`edge_pct`/`kelly_pct` in place | 2026-05 → 2026-10-26 | 403 tips upgraded (O/U); 135 side-market tips | tier-level ROI reported against tiers never sent (staked-tier UNDER −12.4% ledger vs −18.9% first sight; OVER +8.5% vs +2.7%) | none | none on the bet itself | tier-level only; flat all-tier ROI unchanged (odds never rewritten on O/U) | HISTORICAL (fixed 2026-10-26; rows not rewritten by design) | read first-sight tiers from git history / `ledger_tier_seen.csv` for any tier analysis | `scripts/audit/au3_first_sight.py` diff (0 tier mutations after 10-26) |
| AU3-04-B | P3 | v9 | ledger model attribution | `model_version` stamped only from 2026-10-18 | 2026-04-26 → 10-18 | 1,478 of 1,694 O/U tips unattributed (`UNKNOWN_LEGACY`/blank) | per-version ROI not identifiable before 10-18 | none | none | NOT IDENTIFIABLE | HISTORICAL (statistical limitation) | none (never back-fill) | — |
| AU3-04-C | — | v9 | model (new_format OU25) | systematic downward goal bias (p_over 0.498 vs fair 0.551 vs realised 0.571) | season 2026/27 | 798 NF fixtures since 09-01 | UNDER-heavy selection on NF | ~−5 to −7 pp on p_over | adds UNDER, removes OVER | NOT IDENTIFIABLE without a corrected model on untouched data | SUSPICIOUS — NEEDS EVIDENCE (model limitation, possibly linked to NF parity defects; Phase 2) | none here; evaluate repaired NF challenger forward | — |
