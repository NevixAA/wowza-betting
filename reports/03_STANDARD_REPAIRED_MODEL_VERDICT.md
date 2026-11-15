# 03 — Standard O/U 2.5 repaired model: switch verification and forward verdict

Audit team AU2, 2026-11-01. Read-only with respect to production.

| item | value |
|---|---|
| v9 commit audited | `fb0e9697ec8556d4dd3df57f31f8ed5415a5012d` (origin/main, 2026-11-01 11:16 UTC) |
| Pro commit audited | `e0ccc27b98f57c60ee28eedba81812e9b5c5c25b` (origin/main, 2026-11-01 09:49 UTC) |
| Branch | `audit/au2` (worktree `C:\Users\nevo.APOLLO\dev\wowza\v9_audit_au2`) |
| Scripts | `scripts/audit/au2_switch_verify.py`, `au2_phase3.py`, `au2_live_board.py`, `au2_history_lag.py`, `au2_ledger_immutability.py` |
| Machine outputs | `reports/au2_data/switch_boards.csv`, `phase3_populations.json`, `phase3_paired_dll.json`, `live_board_forensics.json`, `history_lag.json`, `ledger_immutability.json` |

## Verdict

**INSUFFICIENT FORWARD EVIDENCE — CANNOT DECLARE REPAIRED MODEL UNPROFITABLE.**

The repaired model has 23 live Standard O/U 2.5 tips in the ledger. Only 4 had settled on the audited
commit (1 win, 3 losses, −1.65 u). The decision threshold used here is 150 settled tips. That is the
same `MIN_CLV_N = 150` bar the estate already uses for a CLV or ROI decision. I stated it before
looking at the results and did not tune it. The full-board probability populations are also too small
to separate the arms (see §4). No result in this report is evidence for or against the repaired model's
profitability.

## 1. Was the switch real? Yes, and the evidence is consistent

| evidence | finding |
|---|---|
| Workflow config | `.github/workflows/predict.yml` commit `f2761eba7` (2026-10-28 12:38 UTC) sets `STANDARD_OU25_SERVING_RECIPE: "v9_repaired"` in the env of the `python pipeline.py --mode predict` step. |
| Other callers | No other workflow runs team predict. `player_props.yml` runs `player_model.pipeline`, and `backtest*.yml` / `retrain.yml` run `--mode train` / `--mode backtest`. predict.yml has no workflow-level or job-level `env:` and no `vars.` reference, so no Actions repository or organisation variable can override the step env. |
| Artifact | `models/model_v9_standard_repaired.pkl`, sha256 `1f5805e63c…`. It was installed by `cc46a26bf` (2026-10-28 12:37:39 UTC, "owner install") and has not changed since. It is recipe `v9_repaired`: 41 features with no price feature, a persisted train-only imputer and a frozen identity alias map (142 aliases). It was trained on 31,086 rows, cutoff 2026-10-28, with blocks base 2021-07-30..2024-11-22, calibration ..2025-03-14, meta ..2025-09-27 and holdout 2025-09-28..2026-10-26. |
| Legacy rollback artifact | `models/model_v9_standard.pkl` (`standard_20261018T080423Z_64313878`) has been unchanged since 2026-10-18. |
| Prediction boards (36 committed boards, 2026-10-26 .. 2026-11-01) | The last legacy board is `995f07d33` (generated 2026-10-28 05:35:48 UTC). The first repaired board is `31a7bdb2c` (12:48:14 UTC). All 20 boards from that point have every Standard row stamped `serving_recipe = v9_repaired` and `model_version = standard_repaired_20261028T123040Z_1f5805e6`. There are **0** `legacy_v9_fallback`, **0** `v9_repaired_failed` and **0** NaN `p_over25` on Standard rows. |
| Serving health | `output/standard_serving_health.json` at the audited board: PASS, 70/70 Standard fixtures served by the repaired model, 0 unscored. |
| Ledger attribution | 23 live Standard rows carry the repaired version, all generated at or after 2026-10-28 12:48. There are 0 legacy-version rows after the switch, 0 repaired rows before it (compared at the ledger's minute precision) and 0 unattributed Standard rows after 2026-10-18 08:05. |
| Independent recompute | I rebuilt the repaired features for the 70 Standard rows on the audited board from the committed history cache. Scoring them with the payload reproduces the served `p_over25` to within **0.03 pp** (median 0.00 pp). The served number is what the artifact computes. |

Board availability gap, recorded but not a model issue: there is no committed board between 2026-10-30 19:57
and 2026-11-01 01:00 UTC, a gap of 29 hours. On 2026-11-01 commit `2a02d05a4` says the OddsAPI BTTS budget ran dry on
10-30/31.

## 2. Effective configuration actually deployed (CLAUDE.md trap 1)

| variable | deployed value | source | used by serving? |
|---|---|---|---|
| `STANDARD_OU25_SERVING_RECIPE` | `v9_repaired` | predict.yml step env | **yes**. Since R2-A (`746ba0f87`, 2026-11-01) it fails closed. Before that, a repaired failure fell back to legacy, stamped `legacy_v9_fallback`. That fallback never fired on any committed board. |
| `WOWZA_MODEL_RECIPE` | unset → `legacy_v9` | config.py default | **no**. It is research only (`python -m src.standard_repair`), and predict never reads it. |
| `WOWZA_MARKET_FEATURE_MODE` | unset → `none` | config.py default | **no**. It is research only. The repaired serving payload is `price_mode = none` (validated in `standard_serving.validate_payload`). |
| `MARKSMAN_THRESHOLD` / `VALUABLE_THRESHOLD` / `LEAGUE_SNIPER_CAP` | 0.08 / 0.03 / 0.12 | predict.yml | yes, unchanged by the switch |
| `REQUIRE_FORM_DATA` / `REQUIRE_FORM_DATA_UNTIL` | `0` / `2026-09-15` | predict.yml | The override has expired, so the no-form guard is armed. Repaired rows use the repaired form (`serving_no_form`). 1–2 Standard rows per board are AVOID for this reason. |
| `NF_FEATURE_PARITY` | unset → `1` (on) | feature_engineering default | It applies to the legacy frame only. It came in today (2026-11-01) and does not touch repaired Standard OU2.5. |
| `WOWZA_SAFE_ENRICH_JOINS` | unset → off | data_loader default | It affects legacy history enrichment (af_history join). The repaired model reads the same loaded history. |

## 3. Populations, never pooled

The populations are:

- **L1**: legacy, unversioned ledger rows, 2026-09-01 .. 2026-10-18.
- **L2**: legacy champion `standard_20261018T080423Z_64313878`.
- **R**: repaired `standard_repaired_20261028T123040Z_1f5805e6`.

All are live Standard O/U 2.5 tips. ROI uses flat 1u stakes recomputed from the ledger odds: a win pays odds − 1 and a loss pays −1. My recomputed P/L equals the ledger `pnl` on every settled row. CIs are a cluster bootstrap with match date as the cluster, 4,000 resamples. The side probability is p_side = edge_pct/100 + 1/odds.

| population | tips | settled | W | hit | flat P/L | flat ROI | ROI 95% CI | mean p_side | calib. gap | side LL | side Brier | side AUC | mean CLV % (n) | CLV > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 legacy unversioned | 299 | 283 | 123 | 43.5% | −23.35 u | −8.3% | [−18.8%, +5.0%] | 0.544 | +10.9 pp | 0.714 | 0.260 | 0.523 | +0.34 (283), CI [−0.30, +1.01] | 41.0% |
| L2 legacy 20261018 | 43 | 38 | 14 | 36.8% | −6.39 u | −16.8% | [−52.1%, +19.0%] | 0.491 | +12.3 pp | 0.673 | 0.240 | 0.601 | +1.32 (38), CI [−0.23, +3.44] | 52.6% |
| **R repaired** | 23 | **4** | 1 | 25% | −1.65 u | −41% | not estimable (1 matchday) | 0.513 | n/a | n/a | n/a | n/a | +2.15 (4) | 25% |

Direction and odds-band splits are in `phase3_populations.json`, including R, where every cell has n ≤ 3. L1:

- OVER: n 160, ROI −12.6%.
- UNDER: n 123, ROI −2.6%.
- Odds band <2: n 110, ROI −8.3%.
- Odds band 2–2.5: n 141, ROI −8.8%.
- Odds band 2.5–3: n 30, ROI −20.2%.
- Odds band ≥3: n 2.

L2:

- OVER: n 10, ROI −23.7%.
- UNDER: n 28, ROI −14.4%.

None of these CIs excludes zero.

**Full-board probabilities, first sight per fixture.** Source: `output/model_lineage/shadow_h2h.csv`. Only fixtures settled from committed sources are included. The market is the de-vigged consensus at the edge engine's first sight, `output/edge_engine_shadow.csv`, which starts 2026-10-25.

| population | fixtures (settled) | LL | Brier | AUC | mean p vs base rate | same rows: market LL / AUC |
|---|---|---|---|---|---|---|
| B_leg: legacy first sight, logged 10-18 .. 10-27 | 212 (175) | 0.6965 | 0.2517 | 0.462 | 0.531 vs 0.526 | on 86 rows: model 0.6867 vs market **0.6654**, AUC 0.489 vs 0.603 |
| B_rep: repaired first sight, logged 10-28 .. 11-01 | 131 (60) | 0.6955 | 0.2512 | 0.426 | 0.517 vs 0.600 | on 60 rows: model 0.6955 vs market **0.6804**, AUC 0.426 vs 0.534 |

**Paired, same fixture, same scoring instant.** Source: Pro `output/standard_repair/forward_shadow.csv`, OU25, first
sight per fixture. 125 fixtures, scored 2026-10-28 11:54 .. 2026-11-01 06:54. 58 are settled, over 3 match dates.

| arm | LL | Brier | AUC | mean p (base 0.586) |
|---|---|---|---|---|
| p_CURRENT_V9_SERVED (legacy on 87 fixtures first seen before the switch, repaired on 38 after) | 0.6794 | 0.2432 | 0.547 | 0.534 |
| p_V9_REPAIRED | 0.6961 | 0.2514 | 0.429 | 0.517 |
| p_MARKET (genuine two-sided de-vig) | 0.6853 | 0.2458 | 0.529 | 0.553 |

The paired dLL for repaired minus served is +0.0167, and for repaired minus market +0.0108. Both use a matchday-block
bootstrap, but **with only 3 match dates the bootstrap cannot produce a meaningful interval** (the intervals in
`phase3_paired_dll.json` are shown for completeness only). On post-switch rows the served probability equals the
repaired probability exactly, with a maximum difference of 0.0. That is a second, independent confirmation of the switch. On
pre-switch rows the two differ by 3.17 pp on average.

Reading: on every forward sample the repaired probabilities lose to the de-vigged market on LL and AUC. So did the
legacy probabilities. That is the long-standing estate result, and these few fixtures neither overturn nor confirm it.
Pro's own pre-registered hypotheses H1–H4 (`STD-V9R-SHADOW-2026Q4`) are **COLLECTING**. This audit takes no decision
on them, because they are frozen to their own minimum N and matchday counts.

## 4. Why nothing can be concluded yet (sample size, stated plainly)

* R has 4 settled bets. At a typical flat-stake ROI SD of about 1.0 per bet, a 95% CI of ±10 pp on ROI needs roughly
  (1.96/0.10)² ≈ 384 settled bets. Even ±20 pp needs about 96. Four bets cannot distinguish +10% from −40%.
* Board-level LL differences between arms here are 0.01–0.02 nats. The estate's own measured model-vs-market gap is
  about 0.014. Resolving a gap that size takes hundreds of settled fixtures over many matchdays, not 58–60 over 3–4.
* **Break-even that would change the conclusion:** with the repaired tips' mean odds of 2.20, the break-even hit rate
  is 1/2.20 = 45.5%. After 150 settled repaired tips, a hit rate whose 95% interval lies entirely below 45.5% would
  make "repaired is unprofitable on its tips" demonstrable. With n = 150 that means an observed hit rate ≲ 37%. An
  interval entirely above 45.5% would need an observed hit rate ≳ 53.5%.

## 5. Caveats that bound every number above

1. **Lineage attribution defect (AU2-B06).** After the switch, the H-MODEL-01 experiment rows in
   `output/model_lineage/shadow_experiments.csv` are labelled `champion_version = standard_20261018…` (legacy), but
   their `p_served` is the repaired served probability. All 42 post-switch rows differ from `p_champion_raw`. I checked
   34/34 rows of the 11-01 01:00 board against `predictions.csv`. `_log_h2h` copies these rows into
   `shadow_h2h.csv` under model_type `standard`. I excluded them from B_leg. Any evaluation that groups
   `shadow_h2h`/`shadow_experiments` by champion_version mixes two models under one version.
2. **Ledger rows before 2026-10-26 12:00 UTC are not all first-sight.** 44 live rows between 2026-10-18 and 2026-10-26
   had `edge_pct` and `signal_tier` rewritten in place (`ledger_immutability.json`, 89 committed ledger versions
   walked). No rewrite has occurred since the owner fix `dfcdf289d`. 0 rows disappeared and 0 settlement fields were overwritten.
   L1/L2 p_side values may therefore come from a later price or edge than the first sight. R is entirely post-fix.
3. **Serving history lags real results (AU2-L01).** On the audited board, 48 of 70 Standard fixtures have at least
   one team whose most recent league match had already kicked off but was not yet in the history used to score it.
   That covers 24/24 Championship fixtures, whose committed history ends 2026-10-24. The repaired
   model's median `home_rest_days` on that board is 11 days. Training rows never have this gap. This affects both
   recipes equally and is a data-availability limit (football-data publication delay), not a recipe defect. It is
   still a train/serve difference the repaired recipe does not guard against.
4. **Outcomes** come only from committed sources: ledger W/L, v9 lineage settlement, the Pro settled file and v9's
   committed history cache. 0 conflicts were found between sources. Missing outcomes are UNSETTLED, never losses. Live
   football-data was unreachable from this machine (TLS chain not trusted). I did not bypass it, so fixtures
   played 10-31/11-01 are mostly unsettled here.
5. No look-ahead backfill is labelled OOS anywhere in this report. The repaired payload's own holdout metrics
   (LL 0.6878, AUC 0.555 on 6,239 rows from 2025-09-28 to 2026-10-26) are **backtest** numbers on data that informed
   the switch decision, not forward evidence. They are quoted only for reference.

## 6. Reproduce

```
git -C <v9> worktree add <wt> -b audit/au2 origin/main          # fb0e9697e
git -C <pro> show origin/main:output/standard_repair/forward_shadow.csv > <dir>/forward_shadow.csv   # + forward_shadow_settled.csv
python scripts/audit/au2_switch_verify.py
python scripts/audit/au2_phase3.py --pro-dir <dir>
python scripts/audit/au2_phase3.py --paired <dir>
python scripts/audit/au2_live_board.py        # under HEAVY.lock (taken by the script)
python scripts/audit/au2_history_lag.py
python scripts/audit/au2_ledger_immutability.py
```
