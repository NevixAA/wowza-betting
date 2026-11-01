# 07 — Configuration Regime Audit (AU1, v9 @ fb0e9697e)

Evidence: `scripts/audit/au1_config_timeline.py` (git walk of `predict.yml`, `config.py`, every model and
threshold artifact → `reports/au1_data/config_timeline.csv`, `artifact_epochs.csv`), `scripts/audit/au1_effective_thresholds.py`
(executes `betting._base_tier` in a child process under each regime's env → `effective_thresholds.csv`),
`scripts/audit/au1_regime_performance.py` (stamps every live ledger row with the regime in force → `regime_ledger_rows.csv`,
`regime_performance.csv`). No threshold below is read from `config.py` text.

## 1. Documented vs deployed thresholds (O/U 2.5)

`predict.yml` env since commit 69b1a5da5 (2026-08-21 16:06Z): `LEAGUE_SNIPER_CAP=0.12`, `MARKSMAN_THRESHOLD=0.08`,
`VALUABLE_THRESHOLD=0.03`. `config.py` defaults: MARKSMAN 0.14, VALUABLE 0.04, cap 1.0, SNIPER_OVER/UNDER 0.12,
EDGE_CEILING 0.19. Approved entries in `models/best_params_standard.json` override both.

Effective floors today (production env, current best_params):

| League group | VALUABLE | MARKSMAN | SNIPER | Source of the binding number |
|---|---|---|---|---|
| 13 New Format leagues | 0.03 | 0.08 | 0.12 | env + `SNIPER_THRESHOLD_OVER/UNDER` default (cap does not apply: no per-league entry) |
| Championship, La Liga 2, League One, Ligue 2 | 0.03 | 0.08 | 0.12 | config per-league value capped to 0.12 |
| Bundesliga 2 | 0.03 | **none** | 0.12 | per-league MARKSMAN 0.20 capped to 0.12 = SNIPER, so MARKSMAN cannot fire |
| League Two | 0.03 | **0.11** | **0.13** | approved best_params (10-06) — **above the documented 0.12 cap** (AU1-B02) |
| Serie B | 0.03 | **0.10** | 0.12 | approved best_params (10-06) |

Before 2026-08-21 (defaults): NF leagues had **no MARKSMAN tier at all** (MARKSMAN 0.14 > SNIPER 0.12, so the ladder
went VALUABLE 0.04 → SNIPER 0.12); Standard per-league SNIPER 0.13-0.25.

**EDGE_CEILING / LEAGUE_EDGE_CEILING never fire (AU1-B01).** `_base_tier` returns SNIPER at `edge >= sniper_thresh`
before it reaches the ceiling branch, and every ceiling (0.15-0.19) is above every non-per-league SNIPER floor
(0.12). The probe over edges 0-0.40 finds no non-monotone ladder in any league under either regime. Config text says
"above 19% … downgraded to MARKSMAN" and lists per-league NF ceilings; deployed behaviour is the opposite. Live NF
O/U rows with edge above their league ceiling: 158 (157 tiered SNIPER), 139 settled; since 2026-08-10 41 settled,
ROI -3.1% (-1.25u). Because the 2026-10-22 policy stakes O/U tiers equally and staking is PAPER, the defect changes
labels and Telegram wording, not a staked amount — **selection impact: relabel only; ROI impact: none**.

## 2. Timeline (git, origin/main; UTC unless stated)

| When | Commit | Change | Regime id |
|---|---|---|---|
| 2026-05-21 | ab32e4886 | repo initialised: SNIPER 0.10, MIN odds 1.75, drift rule ON (inert: history not committed) | ENV_PRE0617 |
| 06-04 / 06-08 / 06-10 / 06-15 | 643a63a1c / 3d3463bdd / 78fac6cbc / 49fa85ce5 | SNIPER 0.15 → 0.10 → 0.12; 3-tier system (MARKSMAN 0.08, VALUABLE 0.04); EDGE_CEILING 0.19 | |
| 2026-06-17 11:05 | d0b6ce10c | MARKSMAN 0.08 → 0.14 | ENV_DEFAULT |
| 2026-06-22 | 075eeb313 | side markets + `best_params_side_markets.json` (per-league side thresholds used unless `drop`) | |
| 2026-07-26 11:24 | 3fd830da2 | `best_params_standard.json` v1 (Serie B approved 0.05/0.04) | BP1 |
| 2026-08-02 12:12 | d3ed30983 | monthly backtest: BP2 (Ligue 2 0.12/0.10, Serie B 0.13/0.11); side O1.5 per-league 0.07-0.10 certified on the 1.40 constant (see report 01) | BP2 |
| 2026-08-09/10 | 76a0fbc92..ee219d06f | PERFORMANCE_CUTOFF 2026-08-09 → 08-10 (NF calibration fix) | |
| 2026-08-15 13:54 | 164eaf5bb | `odds_history_v9.json` committed → drift rule becomes ACTIVE | drift ACTIVE |
| **2026-08-21 16:06** | **69b1a5da5** | **env widening: cap 0.12, MARKSMAN 0.08, VALUABLE 0.03** | ENV_WIDE |
| 2026-08-21 16:11 → 09-15 | 9f25e0421 | blind-fixture guard OFF until 2026-09-15 (self-expiring) | form_guard OFF |
| 2026-09-01 09:43 | d57a3e1bf | BP3 (Championship approved 0.07/0.05, Serie B 0.12/0.10) | BP3 |
| 2026-09-22 08:25 | bddae8a2a | side per-league thresholds used only when `approved` (OOS) | |
| 2026-10-06 11:26 | d311bb2f8 | BP4 (League Two 0.13/0.11, Serie B 0.12/0.10); side BTTS approved Bundesliga 2 0.06, Championship 0.09 | BP4 |
| 2026-10-07 | 2e1431932 | USA MLS → paper league (collected, never sent/counted) | |
| 2026-10-13 → 10-26 | d77c48b1b … ab23726d5 | bankroll policy BALANCED: file 10-13, wired to stake_plan 10-14, league rating 10-18, equal OU25 tier stakes + VALUABLE staked 10-22, side split 10-26. Status WIRED_PAPER (no real bet placed) | |
| 2026-10-22 06:10 | 581a07c1b | NF O/U 2.5 price cap 3.00 (`NF_MAX_OU_ODDS`) | nf_cap 3.00 |
| 2026-10-25 12:17 | 61462e712 | one goal direction per fixture (direction guard) | |
| 2026-10-25 08:46 | 9187ca697 | NF model rolled back to the 10-18 blob; PG-1.0 governs promotion | |
| 2026-10-26 08:12 | 8b46b37f8 | thresholds: backtest only PROPOSES; `threshold_apply` with owner approval is the only writer | |
| 2026-10-26 09:43 | 53a10e9f6 | drift tier adjustment removed (info only; `drift_shadow_tier`) | drift OFF |
| 2026-10-26 12:00 | dfcdf289d | ledgers: first logged tier stands (no in-place upgrades) | |
| **2026-10-28 12:38** | **cc46a26bf / f2761eba7** | **Standard O/U 2.5 serves `v9_repaired`** | std REPAIRED |
| 2026-11-01 10:25 | 246b2cc4e | league registry; recipe default (`legacy_v9` when env unset) made explicit; fail-closed routing | |

Model artifacts change far more often than config: `model_v9_standard.pkl` 32 commits, `model_v9_newformat.pkl` 35,
side models 22-24, HT 31-32 (daily retrains 09-24 → 10-05, then weekly). Full list with blob ids:
`reports/au1_data/artifact_epochs.csv`.

## 3. Regime-scoped live performance (never pooled across cells)

Settled live ledger rows only (WIN → odds-1, LOSS → -1, VOID/blank excluded), flat 1u, matchday-block bootstrap 95% CI,
rows within ±30 min of any regime boundary excluded (21). Brief's evidence start 2026-09-01 shown; 08-10 cut in the CSV.
Caveat: before 2026-10-26 the ledger upgraded tiers in place, so a pre-10-26 row's tier may not be its first-sight tier;
side and odds are first-sight.

### By threshold epoch (env × best_params version) and side, from 2026-09-01

| Family | Epoch | Side | n | ROI | 95% CI |
|---|---|---|---|---|---|
| O/U 2.5 Standard | ENV_WIDE \| BP3 (09-01) | OVER | 107 | -15.4% | [-33.8, +9.1] |
| O/U 2.5 Standard | ENV_WIDE \| BP3 | UNDER | 79 | -6.3% | [-25.4, +15.8] |
| O/U 2.5 Standard | ENV_WIDE \| BP4 (10-06) | OVER | 58 | -9.3% | [-33.9, +17.0] |
| O/U 2.5 Standard | ENV_WIDE \| BP4 | UNDER | 69 | -4.8% | [-26.0, +13.6] |
| O/U 2.5 NF | ENV_WIDE \| BP3 | OVER | 55 | +11.3% | [-28.9, +47.9] |
| O/U 2.5 NF | ENV_WIDE \| BP3 | UNDER | 210 | -7.1% | [-29.1, +9.9] |
| O/U 2.5 NF | ENV_WIDE \| BP4 | OVER | 18 | +23.8% | [-28.9, +86.1] |
| O/U 2.5 NF | ENV_WIDE \| BP4 | UNDER | 208 | -6.9% | [-22.5, +5.7] |
| BTTS NF | ENV_WIDE \| BP3 | YES | 131 | +23.3% | [+5.1, +40.8] |
| BTTS NF | ENV_WIDE \| BP4 | YES | 80 | -11.7% | [-30.8, +13.0] |
| BTTS Standard | ENV_WIDE \| BP3 / BP4 | YES | 88 / 55 | +2.1% / -4.4% | both span 0 |
| O1.5 NF | ENV_WIDE \| BP3 / BP4 | OVER | 40 / 26 | +16.5% / +23.7% | [-1.1, +36.2] / [-2.8, +47.5] |

The only cell from 09-01 whose CI excludes zero, BTTS NF in BP3, is followed by a negative BP4 cell of 80 bets; the
two are not comparable by construction (different side thresholds for Standard leagues only — NF BTTS uses the global
bars in both — and different model artifacts). Neither is a threshold verdict.

### Other regime cuts, from 2026-09-01

* **Standard serving recipe.** Standard O/U rows generated after 2026-10-28 12:38Z (REPAIRED): 23 ledger rows,
  4 settled (-1.65u). INSUFFICIENT FORWARD EVIDENCE — CANNOT DECLARE REPAIRED MODEL UNPROFITABLE. Report 03 owns this.
* **NF price cap 3.00** (from 10-22): NF UNDER after the cap n=41, ROI -26.1% [-49.3, -12.5] over 6 matchdays —
  clustered in one week, not a cap verdict.
* **Drift**: from 09-01 almost every row is ACTIVE; OFF has 11 NF rows. From 08-10, Standard O/U ACTIVE n=404
  ROI -13.3% [-23.1, -2.5]. Drift-on vs drift-off is not identifiable on live data (no overlapping period); the
  removed rule is evaluated only through `drift_shadow.csv` going forward.
* **Blind-fixture guard OFF (08-21 → 09-15)**: BTTS NF OFF n=89 ROI +25.2% [+4.3, +47.3] vs ON n=141 +1.3%
  [-14.4, +18.5]. The OFF window is also the early-season window; season stage and guard state are confounded.
* **Model epoch** (artifact blob in force at generation): NF O/U cells range -13% to +45% on 14-186 rows; no blob has
  ≥150 settled rows; every CI with n ≥ 80 spans zero.

## 4. Separating skill, selection, price, stake, profit (from 2026-09-01, selected bets only)

`p_model = edge_pct/100 + 1/odds` (edge is computed on the raw price), `p_price = 1/odds` (raw, vig included).

| Family / side | n | Hit | mean p_model | mean p_price | Brier model | Brier price | ROI [95% CI] |
|---|---|---|---|---|---|---|---|
| O/U 2.5 Std OVER | 172 | 0.442 | 0.568 | 0.502 | 0.263 | **0.250** | -11.8% [-26.5, +5.7] |
| O/U 2.5 Std UNDER | 168 | 0.411 | 0.503 | 0.441 | 0.252 | **0.241** | -6.7% [-19.9, +8.7] |
| O/U 2.5 NF OVER | 88 | 0.466 | 0.473 | 0.421 | **0.239** | 0.245 | +10.4% [-18.4, +39.9] |
| O/U 2.5 NF UNDER | 439 | 0.415 | 0.516 | 0.430 | 0.252 | **0.236** | -4.4% [-17.2, +6.7] |
| BTTS NF YES | 230 | 0.557 | 0.546 | 0.511 | **0.247** | 0.252 | +10.5% [-3.4, +24.4] |
| BTTS Std YES | 153 | 0.536 | 0.561 | 0.545 | 0.249 | **0.247** | -1.8% [-17.1, +14.2] |
| O1.5 NF OVER | 79 | 0.722 | 0.704 | 0.644 | **0.199** | 0.203 | +11.7% [-4.6, +27.9] |

* **Model skill vs price**: on O/U 2.5 the raw vigged price has lower Brier than the model on its own selected bets
  for Standard both sides and NF UNDER; the model is overconfident by 6-13 pp there. NF OVER, NF BTTS and NF O1.5 are
  the only cells where the model's Brier beats the raw price; all have CIs spanning zero.
* **Selection (tier ladder)**: NF O/U SNIPER hit 34.5% vs VALUABLE 49.2% (n 113 / 191); Standard SNIPER 38.1% vs
  MARKSMAN 45.1%. Higher tiers do not select better outcomes in either track (tier caveat above applies).
* **Price**: first-book OddsAPI price; best-of-books is not used for selection (report 05).
* **Stake**: bankroll policy is WIRED_PAPER since 10-14; no real stake exists, so there is no stake-allocation P&L to
  separate. All ROI here is flat 1u paper.
* **Profit**: flat-stake paper P&L only.

## 5. Rules respected / not identifiable

* Thresholds were never evaluated on data used to select them: the BP versions were chosen on backtest data before
  their live periods; every live cell is post-selection.
* Not identifiable from committed data: the effect of any single knob change across a boundary (each boundary changes
  several things at once, e.g. 08-21 changed cap, MARKSMAN, VALUABLE and, 5 minutes later, the form guard).
* Pre-10-26 tier labels may be upgraded labels; tier-split results before that date are not first-sight.

## 6. Bug table (Phase 11 fields)

| ID | Sev | Repo | File/line | Root cause | Affected dates | Affected rows | Production impact | Probability impact | Selection impact | ROI impact | Status | Classification | Smallest safe fix | Regression test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU1-B01 | P3 | v9 | src/betting.py:180-186 | ceiling branch after SNIPER return; ceilings > SNIPER floor | since 06-10 (ceiling introduced) | 158 live NF rows edge > ceiling (157 SNIPER) | documented downgrade never happens | none | relabel SNIPER→MARKSMAN would apply to 157 rows | none on equal-stake paper | ACTIVE | true bug vs documented intent (owner decides whether to delete or fix) | check ceiling before SNIPER, or delete ceiling config | tests/audit test_edge_above_league_ceiling_is_downgraded |
| AU1-B02 | P3 | v9 | src/betting.py:155-158 vs predict.yml env | approved best_params bypass `LEAGUE_SNIPER_CAP` | 2026-10-06 → now (League Two); 09-01→10-06 Championship 0.07 below cap | League Two SNIPER/MARKSMAN floors 0.13/0.11, Serie B MARKSMAN 0.10 | deployed bars differ from predict.yml comment | none | fewer League Two SNIPER/MARKSMAN than documented | not measured | ACTIVE | suspicious, needs owner (may be intended) | document or apply cap uniformly | test_league_sniper_cap_applies_to_every_league |
| AU1-B11 | P3 | v9 | output/bets_ledger.csv | `model_version` added ~10-18 | 2026-04 → 10-18 | 5,705 / 5,921 rows | model attribution only via git reconstruction | none | none | evaluation hazard | HISTORICAL | provenance gap | keep; use `regime_ledger_rows.csv` | — |
| AU1-B12 | P3 | v9 | output/bets_ledger.csv | backtest and live rows share one file | all | 4,115 backtest + 1,806 live | readers that do not filter `source` mix backtest into live | none | none | potential double counting | ACTIVE | evaluation hazard | separate file or mandatory filter in readers | — |
| AU1-B14 | P3 | v9 | config.py MARKSMAN 0.14 vs SNIPER_OVER/UNDER 0.12 | ladder inversion | 06-17 → 08-21 | NF leagues had no MARKSMAN tier | tier mix differs pre/post 08-21 beyond the documented widening | none | MARKSMAN band absent for NF | n/a | HISTORICAL | intentional config consequence, undocumented | n/a | effective_thresholds.csv |
| AU1-B15 | P3 | v9 | src/ledger.py before dfcdf289d | in-place tier upgrade | ≤ 2026-10-26 | pre-10-26 ledger tiers | tier-split history not first-sight | none | none | tier-level ROI before 10-26 biased toward upgraded tiers | HISTORICAL (fixed 10-26) | evaluation limitation | use `ledger_tier_seen.csv` going forward | — |

## Reproduce

```
python scripts/audit/au1_config_timeline.py
python scripts/audit/au1_effective_thresholds.py
python scripts/audit/au1_regime_performance.py
```
