# 01 — Defaults and Fallback Bugs (AU1, v9 @ fb0e9697e)

Evidence: `scripts/audit/au1_fallback_evidence.py` → `reports/au1_data/fallback_evidence.json`,
`side_ledger_genuine_price_performance.csv`, `side_ledger_price_classes.csv`, `board_feature_imputation.csv`,
`board_rest_days_consistency.csv`. Tests: `tests/audit/test_au1_findings.py` (4 pass, 7 xfail-strict that encode
the correct behaviour; every xfail is required to fail with an `AssertionError`, not an import or data error).

## 1. Synthetic side-market prices (BTTS 1.85 / O1.5 1.40 / O3.5 2.60)

Defined: `src/backtest.py:233 _DEFAULT_ODDS`; mirrored in `src/market_edge.py:82`, `src/integrity/price_provenance.py:34`,
`dashboard_data/evidence_panels.py:419`. Applied: `run_side_market_backtest` fills NaN odds, flags `price_synthetic`,
forces `signal_tier=NO_PRICE`, `edge=NaN`. Not applied on the live path: predict never fills a side price; a fixture
with no quote drops out at `dropna(subset=[prob_col, odds_col])` in `pipeline._generate_side_bets`.

### Reproduced on the committed artifacts (backtest of 2026-10-06, commit d311bb2f8)

| Market | Rows | `price_synthetic` | Price == constant | Real-priced rows | Real-price window | Synthetic rows with BET or P&L |
|---|---|---|---|---|---|---|
| Over 1.5 | 22,516 | **22,441** | 22,447 | 75 | 2024-09-21 .. 2024-10-18 | 0 |
| BTTS | 22,516 | **21,335** | 21,377 | 1,181 | 2023-08-26 .. 2024-10-18 | 0 |
| Over 3.5 | 22,516 | **22,439** | 22,439 | 77 | 2024-08-31 .. 2024-10-18 | 0 |

The brief's 22,441 / 22,516 reproduces exactly. The guard holds: no synthetic row is a bet or carries P&L
(`test_synthetic_side_prices_never_bet`, 3 passes). Six O1.5 and 42 BTTS rows are real quotes that happen to equal
the constant; the explicit flag keeps them, the fail-closed label in `price_provenance` would drop them.

### NEW: the committed side backtests are stale against current code + committed data (AU1-B04)

Re-running only the loader join (`data_loader._enrich_with_standard_sidemarket_odds`, legacy path, join-audit write
disabled) on the backtest's own fixture keys and the committed `standard_sidemarket_odds_history.csv`:

| | Artifact real rows | Current loader would price |
|---|---|---|
| BTTS | 1,181 | **6,352** (5,487 of them in 2024-11 .. 2026-10) |
| Over 1.5 | 75 | 364 |
| Over 3.5 | 77 | 381 |

Months 2024-11 .. 2026-10 are 100% synthetic in the artifact while ~72% of the same fixtures join to a price today.
The history file already held those months at the backtest's commit (2,744 lines for 2025-01..04 at d311bb2f8),
so the gap is not explained by data arriving later. Root cause in the CI run: **NOT IDENTIFIED** (a full rebuild of
`load_all_matches` was not run). Consequence: per-league side thresholds (`best_params_side_markets.json`, e.g. BTTS
Bundesliga 2 approved on 36 OOS bets, Championship on 68) were certified on a much smaller real-price sample than
the data supports. Encoded as `test_side_backtest_real_price_coverage_matches_loader` (xfail-strict).

### What the "real" backtest prices are (rule 6)

`standard_sidemarket_odds_history.csv` (66,158 rows): 12,272 rows carry a `T00:00:00Z` placeholder timestamp and no
kickoff (2024-08..2026-05 backfill), 2,495 timed rows without kickoff, 51,391 timed rows with kickoff. The loader takes
`aggfunc="last"` per fixture, i.e. the latest snapshot, which is the closest thing to a closing price. **No side-market
backtest ROI is a decision-time ROI**; it is closing-price research at best. Real-price backtest results, for
information only: BTTS all tiers 514 bets, +18.8% per stake; O1.5 25 bets +2.3%; O3.5 67 bets +3.3% — none is
executable evidence.

### Performance on genuinely observed, timestamp-valid live prices only

Source: `output/side_bets_ledger.csv` (652 rows, all `source=live`, first-sight since the ledger rule). Each row was
matched to an observed kickoff from `book_odds_snapshots*`, `standard_odds_history`, `newformat_odds_history`,
`standard_sidemarket_odds_history`. Classes (fail-closed order):

| Class | Rows | Meaning |
|---|---|---|
| GENUINE_TIMESTAMP_VALID | 612 | real OddsAPI quote, generated before observed kickoff |
| KICKOFF_UNOBSERVED | 21 | all NF O1.5, match dates 2026-08-09..08-18, before per-book capture existed |
| EQUALS_FILL_CONSTANT | 19 | 18 BTTS at 1.85, 1 O1.5 at 1.40 — live quotes (predict has no fill), excluded fail-closed |
| POST_KICKOFF / NO_PRICE / SETTLED_AT_FAIR_ODDS | 0 | |

Median lead time 8,686 min (~6 days). Ledger P&L equals recomputed `odds-1 / -1` on all 602 settled rows (0 mismatches);
50 rows are unsettled, excluded (never 0-filled). Flat 1u, matchday-block bootstrap 95% CI:

| From | Family | n | Matchdays | ROI | 95% CI | Mean odds |
|---|---|---|---|---|---|---|
| 2026-09-01 | BTTS new_format | 223 | 50 | +12.4% | [-1.9%, +28.1%] | 2.00 |
| 2026-09-01 | BTTS standard | 146 | 35 | +1.7% | [-13.3%, +18.5%] | 1.85 |
| 2026-09-01 | O1.5 new_format | 79 | 37 | +11.7% | [-5.2%, +28.1%] | 1.56 |
| 2026-09-01 | O1.5 standard | 8 | 7 | -10.9% | [-64%, +33%] | 1.45 |
| 2026-09-01 | O3.5 new_format | 19 | 13 | -18.6% | [-59.6%, +24.4%] | 2.08 |
| 2026-08-10 | BTTS new_format | 260 | 58 | +16.4% | [+3.2%, +30.1%] | 2.02 |
| 2026-08-10 | O1.5 new_format | 100 | 45 | +8.7% | [-7.1%, +23.5%] | 1.56 |

Every cell from 2026-09-01 has a CI spanning zero. These pool all model versions and threshold epochs inside a family;
the regime-split view is in report 07. Removing the fail-closed constant rows raises BTTS NF slightly (the 10 removed
rows lost 4.45u) — it does not create the result.

## 2. Model-input defaults on the live board

### 2a. Batch-median imputation in every legacy serving payload (AU1-B05, known as P0 bug 1, still ACTIVE)

`src/model.py:124 _prep(imputer=None)`: NaN → median **of the scoring batch**, all-NaN column → `0.0`. Serving
payloads without a persisted imputer: New Format O/U 2.5, BTTS, O1.5, O3.5, HT O0.5, HT O1.5 (only Standard O/U 2.5
is repaired). On the committed board:

| Payload | Batch | Cells imputed | Notable columns (NaN share in batch) |
|---|---|---|---|
| newformat | 88 NF rows | 8.3% | rest days 32% / 43%, corners+fouls 15-23%, form 4.5% |
| btts / over15 / over35 | 158 rows, **std + NF in one batch** | 13.3% (NF 13.7%, std 12.7%) | HT over-rates 54-57%, rest days 47-48% |
| ht_over05 / ht_over15 | 70 std rows | ~13% | same as standard |

No column is all-NaN on this board, so the `0.0` branch is not triggered here; it is triggered whenever a fixture is
scored alone or the whole batch lacks a feature. Measured batch dependence: re-scoring each of the 55 NF fixtures that
have any NaN feature **alone** changes P(over 2.5) by median **0.017**, max **0.179**; fixtures with complete features
change by 0.000. Re-scoring the board reproduces the stored `p_over25` to 5e-4. Side models see a mixed std+NF batch,
so an NF fixture's missing value is filled with a median that includes Standard fixtures (and vice versa).

### 2b. Serving-path rest days are misaligned and mostly missing (AU1-B03, NEW, ACTIVE)

`src/feature_engineering.py:788-793` (build_upcoming_features):

```python
combined = pd.concat([hist[["date","home_team","away_team"]].tail(1000), df[[...]]]).sort_values("date").reset_index(drop=True)
combined_rd = _add_rest_days(combined)
tail = combined_rd.tail(len(df)).reset_index(drop=True)
df["home_rest_days"] = tail["home_rest_days"].values      # positional
```

Three defects: (1) the upcoming rows are re-ordered by date, then written back **by position** onto a board ordered
by league; (2) only the last 1,000 history rows across all leagues are used; (3) raw OddsAPI club names are used, not
the league-scoped resolved names the form features use. Evidence on the committed board: rest days are NaN on **47.5%**
of team-slots (training frame: ~0.2%). Of 57 cases where a club appears twice on the board, **15 report more rest
days before the later fixture than the gap since its earlier board fixture — physically impossible**, 34 are NaN, only
4 equal the gap. Unit replica `test_serving_rest_days_follow_their_fixture` and data test
`test_committed_board_rest_days_physically_possible` (both xfail-strict).

Features using `home/away_rest_days`: every legacy payload above and the repaired Standard payload (the repaired one
builds its own features through `features_repaired`, which AU1 did not audit). Sensitivity of the served NF payload:
setting both rest days to 7 moves P(over) by median 0.003, max 0.021. Probability impact is therefore small per
fixture; selection/ROI impact is **NOT IDENTIFIABLE** without re-serving historical boards with corrected rest days.

### 2c. Other model-input fills

| Default | Where | Train | Serve | Note |
|---|---|---|---|---|
| `combined_ht_goals_avg = ht_scored.fillna(0)+...` | feature_engineering.py:468, 778; features_repaired.py:285 | **48.4% of Standard training rows are 0** (2020-22 ~100%: HT form absent) | 7.1% std / 14.8% NF on board | invented 0 = "missing"; used by std legacy, std repaired, side, HT payloads (AU1-B08) |
| `p_over25_poisson_dc = implied_prob_over.fillna(0.5)` | feature_engineering.py:535, 1004 | only if DC helper unavailable; 0% NaN in training | same | path effectively dormant |
| referee foul default 22.0 / league mean | feature_engineering.py:250-256 | yes | yes | not in served feature lists checked |
| meta-model failure → simple mean | model.py:733-738 `except Exception` | n/a | silent, unlogged | changes probability with no trace (AU1-B10) |
| missing declared column | model.py `_prep` legacy: column silently dropped → sklearn raises → side model caught → `p_side` NaN (no tip) | | | fails closed in effect, logged as a warning |
| `key_attacker_absent_*` = 0.0 on 100% of board rows | predictions.csv | | | not a model input (display only) |
| Props: missing feature columns → 0.0 with a warning; NaN → batch median, all-NaN → 0.0; constant fills 1.3 / 4.5 / 7.0 / 6.0 / 0.75 / 0.50 | player_model/model.py:28-40; player_model/feature_engineering.py:102, 331, 407, 473-475, 496 | | | trigger frequency NOT MEASURED by AU1 (props are another team's phase) |

## 3. Routing, provenance and environment fallbacks

| Fallback | Where | Status | Evidence |
|---|---|---|---|
| NF fixture → Standard model when NF artifact missing; unknown league → Standard model | predict.py (pre R2-A) | FIXED today (R2-A fail-closed) | 3 historical ledger rows `model_type=unknown` (pre-2026-06-17); routing health on the board: 0 unknown |
| Standard repaired failure → legacy re-score | standard_serving.py (pre R2-A) | FIXED (fail closed, `v9_repaired_failed`) | board 70/70 repaired |
| Recipe default `legacy_v9` when env unset | config.py:260 | INTENTIONAL (documented rollback) | predict.yml sets `v9_repaired`; no other workflow predicts |
| `feature_health.json` "standard" uses the legacy payload's 42 columns | predict.py:451 `payload.get("feature_cols")` | ACTIVE, monitoring only (AU1-B06) | served repaired payload has 41 columns and its own feature frame |
| v7 `best_params.json` read from `DATA_DIR = repo parent` | betting.py:63-79, config.py:27 | ACTIVE config divergence (AU1-B09) | absent on CI runners → `both_losing` and per-league min odds inert in production; present in the OneDrive `mixed/` folder → active in a local run. Board: `both_losing` False on 158/158 |
| `WOWZA_USE_BACKTEST_ARCHIVE` train `0`, predict default `1` | data_loader.py:294, retrain.yml:134 | ACTIVE train/serve history difference (AU1-B07) | archive = 44,207 rows 2019-09..2026-04-23; affects as-of features near season start; impact NOT MEASURED |
| `WOWZA_FULL_ENRICH` train `1`, predict unset | data_loader.py:1065 | INTENTIONAL (runtime), but a train/serve coverage difference | Standard xG at serve: 1-4% NaN on board |
| `odds = float(row.get("market_odds", 0.0) or 0.0)` | ledger.py:434 | latent (side frame drops NaN odds first) | 0 missing / non-positive odds in side ledger and in bets ledger |
| `model_version` missing on ledger rows | ledger | HISTORICAL (column added ~10-18) | 5,705 / 5,921 bets_ledger rows blank |
| Side markets use the FIRST book's price; O1.5/O3.5 exist only where that book quotes the line on the list endpoint | predict.py `_fetch_odds_api` | INTENTIONAL, but a price-availability selection effect (NF O1.5 tips exist mainly on low-total fixtures; cf. price_provenance MAIN_LINE_ONLY) | for report 05/09 |
| EDGE_CEILING / LEAGUE_EDGE_CEILING downgrade | betting.py:180-186 | DEAD CODE (AU1-B01) | see report 07: 157 live NF SNIPER rows had edge above their league ceiling |

## 4. Bug table (Phase 11 fields)

| ID | Sev | Repo | File/line | Root cause | Affected dates | Affected rows | Production impact | Probability impact | Selection impact | ROI impact (same sample) | Status | Classification | Smallest safe fix | Regression test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU1-B03 | P2 | v9 | src/feature_engineering.py:788-793 | positional write-back of date-sorted rest days; tail(1000); unresolved names | since rest days entered upcoming features (pre-git 2026-05) | 47.5% of board team-slots NaN; 15/57 repeat-club slots impossible | NF O/U, side, HT inputs wrong or median-imputed | median 0.003, max 0.021 (NF, rest→7 test) | NOT IDENTIFIABLE | NOT IDENTIFIABLE | ACTIVE | true production bug | merge rest days back on (date, home, away) keys, league-scoped resolved names, full history | tests/audit test_serving_rest_days_follow_their_fixture, test_committed_board_rest_days_physically_possible |
| AU1-B05 | P1 | v9 | src/model.py:124-142 | legacy `_prep` imputes from the scoring batch; mixed std+NF batch for side models | all dates, all legacy payloads | 55/88 NF board rows have ≥1 NaN feature | probability depends on neighbours | median 0.017, max 0.179 (solo vs batch) | NOT IDENTIFIABLE | NOT IDENTIFIABLE | ACTIVE (fixed only for Standard O/U 2.5) | true production bug (known, R2 P0 bug 1) | persisted train-only imputer per payload (repaired recipe) | test_newformat_probability_independent_of_batch |
| AU1-B04 | P2 | v9 | output/backtest_results_{btts,over15,over35}.csv vs data_loader.py:627-713 | committed side backtest priced far fewer rows than current loader joins | backtest 2026-10-06; thresholds in force 10-06 → now | BTTS 1,181 vs 6,352 real rows | side thresholds certified on understated real sample | none | thresholds may differ after a correct rerun | NOT IDENTIFIABLE | ACTIVE (artifact) | research/certification flaw; root cause UNVERIFIED | rerun side backtest with join audit; compare | test_side_backtest_real_price_coverage_matches_loader |
| AU1-B13 | P2 | v9 | data_loader.py `pivot_table(aggfunc="last")` | side backtest "real" price = last snapshot (closing-like); 12,272 rows midnight placeholder ts | all side backtests | all real-priced side rows | backtest ROI treated as entry-price ROI | none | none live | invalidates side backtest ROI as decision evidence | ACTIVE | research-only flaw | take first pre-decision snapshot; label CLOSE otherwise | — (documented; report 05 owner) |
| AU1-B08 | P3 | v9 | feature_engineering.py:468, 778; features_repaired.py:285 | `fillna(0)` builds an invented 0 | all | 48.4% std training rows; 7-15% board | model reads "missing HT form" as "0 HT goals" | not measured | not measured | not measured | ACTIVE | true bug (invariant 9) | keep NaN, let the imputer handle it, add a missing flag | — |
| AU1-B06 | P3 | v9 | src/predict.py:451 `record_feature_health` | health handed the legacy payload | since 2026-10-28 | every predict run | serving health for Standard monitors the wrong model | none | none | none | ACTIVE | monitoring flaw | pass repaired feature frame/cols | test_feature_health_standard_matches_served_payload |
| AU1-B07 | P3 | v9 | data_loader.py:294 vs retrain.yml:134 | archive held out of training, loaded at predict | 2026-09-24 → now | 44,207 history rows (≤2026-04-23) | serving history ≠ training history | not measured | not measured | not measured | ACTIVE | suspicious, needs evidence | set the same flag in predict.yml or drop the default | — |
| AU1-B09 | P3 | v9 | betting.py:63-79, config.py:27 | v7 params outside the repo | all CI runs | all | both_losing/min-odds per league differ local vs CI | none | local backtests ≠ production | not measured | ACTIVE | config divergence | commit the file or remove the read | — |
| AU1-B10 | P3 | v9 | model.py:733-738 | bare `except` → mean of base models | any meta failure | unknown (unlogged) | silent probability change | unknown | unknown | unknown | ACTIVE latent | suspicious | log + fail closed | — |

## Reproduce

```
python scripts/audit/au1_fallback_evidence.py
python -m pytest tests/audit/test_au1_findings.py -q -rxX
```
